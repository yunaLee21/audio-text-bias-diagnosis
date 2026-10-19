"""Metrics over a predictions.jsonl file.

Per condition: accuracy, mean confidence, P(expected), text-follow rate, readout diagnostics.
MCR-BENCH (Sec. 3, 2508.15407), with audio_only as the neutral condition:
    Acc_t, Norm_t = C_t / C_neu, Macro = mean(Acc_fth, Acc_adv, Acc_irr),
    TIR_fth = Δic / N, TIR_adv = Δci / N, TIR_irr = (Δic + Δci) / N,
    MRS = α·Acc_adv/Acc_neu + (1−α)·Acc_irr/Acc_neu, α = 0.8.
Conflict: text-follow rate over both conflict directions (0.5 = balanced, >0.5 = text bias)
and Paper 2's Δ(T−A) = text-side accuracy − audio-side accuracy.

Two readouts are scored with the same metrics:
  ab      A/B forced choice. Conflict items are run in both orders; collapse_orders() averages the
          two per option value. order-flip = share of items whose answer changed with the order alone.
  yes_no  one Yes/No question per value (no option list); value score = log-odds(Yes).
Order-free conflict decomposition (per item, signed so that > 0 means the AUDIO value wins):
  A/B     m_A, m_B = logit(audio value) - logit(text value) with the audio value in slot A / slot B;
          content = (m_A + m_B) / 2, position = (m_B - m_A) / 2 (> 0 = slot B favoured).
  Yes/No  content = log-odds(Yes | audio value) - log-odds(Yes | text value);
          calibrated = the same minus its value in the question-only (no_input) item of that fact.
          Averaged over both conflict directions the value prior cancels by design (each fact appears
          with the audio/text values swapped), so calibration only changes per-item results.

    python -m src.eval.metrics --results results/pilot_run_1
"""

import argparse
import math
from collections import defaultdict
from statistics import mean

from src.common import read_jsonl, resolve, write_json

MCR_MAP = {"neu": "audio_only", "fth": "faithful", "adv": "conflict_audio_correct", "irr": "irrelevant"}
CONFLICT = ("conflict_audio_correct", "conflict_text_correct")


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return mean(xs) if xs else None


def _div(a, b):
    return a / b if a is not None and b else None


def collapse_orders(rows):
    """Merge the A/B-order replicates of an item (same fact + condition) into one scored row.

    Probabilities are averaged per option *value*, so a preference for a letter position cancels.
    Items run in a single order pass through unchanged (apart from dropping the letter fields).
    """
    groups = defaultdict(list)
    for r in rows:
        groups[(r["fact_id"], r["condition"])].append(r)
    out = []
    for reps in groups.values():
        yn = next((r["yes_no_out"] for r in reps if r.get("yes_no_out")), None)
        p_value = defaultdict(list)
        for r in reps:
            for letter in "AB":
                p_value[r["options"][letter]].append(r[f"p_{letter}"])
        p = {v: mean(ps) for v, ps in p_value.items()}
        chosen = max(p, key=p.get)
        row = {k: v for k, v in reps[0].items()
               if k not in ("stim_id", "options", "p_A", "p_B", "logit_A", "logit_B", "chosen_letter",
                            "greedy_raw", "greedy_letter", "top_token", "samples")}
        row.update(
            item_id=f"{reps[0]['fact_id']}_{reps[0]['condition']}",
            n_orders=len(reps),
            p_value=p,
            chosen_value=chosen,
            correct=chosen == reps[0]["expected"],
            confidence=p[chosen],
            p_expected=p[reps[0]["expected"]],
            margin=_mean([r.get("margin") for r in reps]),
            mass_ab=_mean([r.get("mass_ab") for r in reps]),
            # did the chosen value change when only the A/B order changed?
            order_flip=len({r["chosen_value"] for r in reps}) > 1 if len(reps) > 1 else None,
            yes_no_out=yn,
            **_ab_content_position(reps),
        )
        out.append(row)
    return out


def _ab_content_position(reps):
    """Split a counterbalanced conflict pair into content and position effects (logit units)."""
    r0 = reps[0]
    audio, text = r0["audio_value"], r0["text_value"]
    if len(reps) != 2 or not audio or not text or audio == text or "logit_A" not in r0:
        return {}
    m = {}
    for r in reps:
        slot = "A" if r["options"]["A"] == audio else "B"
        other = "B" if slot == "A" else "A"
        m[slot] = r[f"logit_{slot}"] - r[f"logit_{other}"]
    if set(m) != {"A", "B"}:
        return {}
    return {"ab_m_audio_in_A": m["A"], "ab_m_audio_in_B": m["B"],
            "ab_content": (m["A"] + m["B"]) / 2, "ab_position": (m["B"] - m["A"]) / 2}


def _sigmoid(x):
    return 1 / (1 + math.exp(-x))


def yes_no_items(items):
    """Re-score A/B-collapsed items with the Yes/No readout (same field names, so all metrics apply)."""
    prior = {r["fact_id"]: r["yes_no_out"] for r in items
             if r["condition"] == "no_input" and r.get("yes_no_out")}
    out = []
    for r in items:
        yn = r.get("yes_no_out")
        if not yn:
            continue
        lo = {v: o["log_odds"] for v, o in yn.items()}
        exp = r["expected"]
        other = next(v for v in lo if v != exp)
        margin = lo[exp] - lo[other]
        row = {k: v for k, v in r.items() if not k.startswith("ab_") and k not in ("p_value", "order_flip", "mass_ab")}
        row.update(
            yn_log_odds=lo,
            margin=margin,
            chosen_value=max(lo, key=lo.get),
            correct=lo[exp] >= lo[other],
            p_expected=_sigmoid(margin),
            confidence=_sigmoid(abs(margin)),
            mass_yn=_mean([o.get("mass_yn") for o in yn.values()]),
            order_flip=None,
        )
        audio, text = r["audio_value"], r["text_value"]
        if audio and text and audio != text:
            row["yn_content"] = lo[audio] - lo[text]
            p0 = prior.get(r["fact_id"])
            if p0:
                row["yn_content_cal"] = row["yn_content"] - (p0[audio]["log_odds"] - p0[text]["log_odds"])
        out.append(row)
    return out


def readout_items(rows, readout="yes_no"):
    """One scored row per item for the chosen readout (falls back to ab for runs without Yes/No)."""
    items = collapse_orders(rows)
    if readout == "yes_no":
        yn = yes_no_items(items)
        if yn:
            return yn
    return items


def per_condition(rows, items):
    """rows: raw predictions (letter-level diagnostics); items: one scored row per item."""
    raw = defaultdict(list)
    for r in rows:
        raw[(r["kind"], r["condition"])].append(r)
    groups = defaultdict(list)
    for r in items:
        groups[(r["kind"], r["condition"])].append(r)
    out = {}
    for (kind, cond), rs in sorted(groups.items()):
        reps = raw[(kind, cond)]
        conflict = [r for r in rs if r["audio_value"] and r["text_value"] and r["audio_value"] != r["text_value"]]
        out[f"{kind}/{cond}"] = {
            "n": len(rs),
            "accuracy": _mean([r["correct"] for r in rs]),
            "confidence": _mean([r["confidence"] for r in rs]),
            "p_expected": _mean([r["p_expected"] for r in rs]),
            "text_follow_rate": _mean([r["chosen_value"] == r["text_value"] for r in conflict]) if conflict else None,
            "order_flip_rate": _mean([r["order_flip"] for r in rs]),
            "share_B": _mean([r["chosen_letter"] == "B" for r in reps]),
            "mass_ab": _mean([r.get("mass_ab") for r in reps]),
            "mass_yn": _mean([r.get("mass_yn") for r in rs]),
            "greedy_agrees": _mean([r["greedy_letter"] == r["chosen_letter"] for r in reps if "greedy_letter" in r]),
            "greedy_format_ok": _mean([r["greedy_letter"] is not None for r in reps if "greedy_letter" in r]),
        }
    return out


def mcr_metrics(rows, kind, alpha=0.8):
    by = defaultdict(dict)
    for r in rows:
        if r["kind"] == kind:
            by[r["condition"]][r["fact_id"]] = r["correct"]
    neu = by.get(MCR_MAP["neu"], {})
    if not neu:
        return None
    acc = {k: _mean(list(by[c].values())) if by.get(c) else None for k, c in MCR_MAP.items()}
    res = {"acc": acc, "norm": {k: _div(acc[k], acc["neu"]) for k in ("fth", "adv", "irr")}}
    parts = [acc[k] for k in ("fth", "adv", "irr")]
    res["macro"] = mean(parts) if None not in parts else None

    tir = {}
    for k in ("fth", "adv", "irr"):
        cond = by.get(MCR_MAP[k], {})
        pairs = [(neu[f], cond[f]) for f in cond if f in neu]
        if not pairs:
            continue
        ic = sum(1 for a, b in pairs if not a and b)
        ci = sum(1 for a, b in pairs if a and not b)
        tir[k] = {"fth": ic, "adv": ci, "irr": ic + ci}[k] / len(pairs)
    res["tir"] = tir
    if res["norm"]["adv"] is not None and res["norm"]["irr"] is not None:
        res["mrs"] = alpha * res["norm"]["adv"] + (1 - alpha) * res["norm"]["irr"]
    return res


def conflict_metrics(rows, kind):
    rs = [r for r in rows if r["kind"] == kind and r["condition"] in CONFLICT]
    if not rs:
        return None
    acc = {c: _mean([r["correct"] for r in rs if r["condition"] == c]) for c in CONFLICT}
    return {
        "text_follow_rate": _mean([r["chosen_value"] == r["text_value"] for r in rs]),
        "delta_text_minus_audio": (acc["conflict_text_correct"] - acc["conflict_audio_correct"])
        if None not in acc.values() else None,
        "confidence_when_following_text": _mean([r["confidence"] for r in rs if r["chosen_value"] == r["text_value"]]),
        "confidence_when_following_audio": _mean([r["confidence"] for r in rs if r["chosen_value"] == r["audio_value"]]),
    }


def order_free(ab_items, yn_items, kind):
    """Conflict decomposition for one kind; all values signed so that > 0 = audio wins."""
    ab = [r for r in ab_items if r["kind"] == kind and "ab_content" in r]
    yn = [r for r in yn_items if r["kind"] == kind and "yn_content" in r]
    if not ab and not yn:
        return None
    return {
        "n_ab": len(ab),
        "ab_content": _mean([r["ab_content"] for r in ab]),
        "ab_abs_position": _mean([abs(r["ab_position"]) for r in ab]),
        "ab_position": _mean([r["ab_position"] for r in ab]),
        "ab_position_dominated": _mean([abs(r["ab_position"]) > abs(r["ab_content"]) for r in ab]),
        "n_yn": len(yn),
        "yn_content": _mean([r["yn_content"] for r in yn]),
        "yn_audio_wins": _mean([r["yn_content"] > 0 for r in yn]),
        "yn_audio_wins_cal": _mean([r["yn_content_cal"] > 0 for r in yn if "yn_content_cal" in r]),
    }


def _readout_metrics(rows, items, kinds):
    return {
        "per_condition": per_condition(rows, items),
        "mcr": {k: mcr_metrics(items, k) for k in kinds},
        "conflict": {k: conflict_metrics(items, k) for k in kinds},
    }


def compute_all(rows):
    ab_items = collapse_orders(rows)
    yn_items = yes_no_items(ab_items)
    kinds = sorted({r["kind"] for r in rows})
    res = {"ab": _readout_metrics(rows, ab_items, kinds)}
    if yn_items:
        res["yes_no"] = _readout_metrics(rows, yn_items, kinds)
    res["order_free"] = {k: order_free(ab_items, yn_items, k) for k in kinds}
    return res


def _fmt(x):
    return "–" if x is None else f"{x:.2f}"


def _mcr_table(m):
    lines = ["| kind | text-follow (conflict) | Δ(T−A) | Norm adv | Norm irr | TIR fth | TIR adv | TIR irr | MRS |",
             "|---|---|---|---|---|---|---|---|---|"]
    for kind in m["mcr"]:
        c, r = m["conflict"][kind] or {}, m["mcr"][kind] or {}
        norm, tir = r.get("norm", {}), r.get("tir", {})
        lines.append(f"| {kind} | {_fmt(c.get('text_follow_rate'))} | {_fmt(c.get('delta_text_minus_audio'))} | "
                     f"{_fmt(norm.get('adv'))} | {_fmt(norm.get('irr'))} | {_fmt(tir.get('fth'))} | "
                     f"{_fmt(tir.get('adv'))} | {_fmt(tir.get('irr'))} | {_fmt(r.get('mrs'))} |")
    return lines


def summary_markdown(metrics):
    lines = []
    if "yes_no" in metrics:
        yn = metrics["yes_no"]
        lines += ["## Yes/No readout (primary: no option list, no order effect)", "",
                  "| kind / condition | n | acc | conf | P(expected) | text-follow | mass(Yes,No) |",
                  "|---|---|---|---|---|---|---|"]
        for key, m in yn["per_condition"].items():
            lines.append(f"| {key} | {m['n']} | {_fmt(m['accuracy'])} | {_fmt(m['confidence'])} | "
                         f"{_fmt(m['p_expected'])} | {_fmt(m['text_follow_rate'])} | {_fmt(m['mass_yn'])} |")
        lines += [""] + _mcr_table(yn) + [""]

    ab = metrics["ab"]
    lines += ["## A/B readout (comparable to Paper 2; conflict orders averaged)", "",
              "| kind / condition | n | acc | conf | P(expected) | text-follow | order-flip | share B | mass(A,B) | greedy agrees |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for key, m in ab["per_condition"].items():
        lines.append(f"| {key} | {m['n']} | {_fmt(m['accuracy'])} | {_fmt(m['confidence'])} | "
                     f"{_fmt(m['p_expected'])} | {_fmt(m['text_follow_rate'])} | {_fmt(m['order_flip_rate'])} | "
                     f"{_fmt(m['share_B'])} | {_fmt(m['mass_ab'])} | {_fmt(m['greedy_agrees'])} |")
    lines += [""] + _mcr_table(ab)

    lines += ["", "## Conflict: content vs. position (log-odds / logit units; > 0 = audio wins)", "",
              "| kind | A/B content | A/B position (+ = slot B) | A/B mean abs. position | A/B position-dominated | "
              "Yes/No content | Yes/No audio wins | Yes/No audio wins (calibrated) |",
              "|---|---|---|---|---|---|---|---|"]
    for kind, d in metrics["order_free"].items():
        if d:
            lines.append(f"| {kind} | {_fmt(d['ab_content'])} | {_fmt(d['ab_position'])} | {_fmt(d['ab_abs_position'])} | "
                         f"{_fmt(d['ab_position_dominated'])} | {_fmt(d['yn_content'])} | {_fmt(d['yn_audio_wins'])} | "
                         f"{_fmt(d['yn_audio_wins_cal'])} |")
    return "\n".join(lines) + "\n"


def write_metrics(results_dir):
    rows = read_jsonl(resolve(results_dir) / "predictions.jsonl")
    metrics = compute_all(rows)
    write_json(metrics, resolve(results_dir) / "metrics.json")
    md = summary_markdown(metrics)
    (resolve(results_dir) / "summary.md").write_text(md, encoding="utf-8")
    return md


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", required=True, help="run directory containing predictions.jsonl")
    print(write_metrics(ap.parse_args().results))


if __name__ == "__main__":
    main()
