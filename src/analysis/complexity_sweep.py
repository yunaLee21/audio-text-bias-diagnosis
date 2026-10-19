"""Level sweep for the complexity dataset: does text-following rise as audio confidence falls?

Per level L0-L3: audio-only / text-only accuracy and P(expected), audio input-tracking, conflict
text-follow rate (both directions), Δ(T−A), and the order-flip rate.
Across all (fact, level) items: Spearman ρ between audio-only P(planted) and, under
conflict_audio_correct, P(audio value).
  H2 predicts ρ > 0 (weaker audio -> text wins more), with text-follow climbing over levels.
  H1 predicts text-follow already high at L0, where audio-only is confident, and a flat relation.
Compare text-follow only against the baselines of the same level: a harder sentence is harder to
hear than to read, so the levels are not equally difficult for the two modalities.

    python -m src.analysis.complexity_sweep --results results/complexity_run_1             # Yes/No readout
    python -m src.analysis.complexity_sweep --results results/complexity_run_1 --readout ab
"""

import argparse
from collections import defaultdict
from statistics import mean

from src.common import read_jsonl, resolve, write_json
from src.eval.metrics import readout_items

CONFLICT = ("conflict_audio_correct", "conflict_text_correct")


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return mean(xs) if xs else None


def spearman(xs, ys):
    try:
        from scipy.stats import spearmanr
        res = spearmanr(xs, ys)
        return float(res.statistic), float(res.pvalue)
    except ImportError:
        return None, None


def sweep(rows, readout="yes_no"):
    items = readout_items([r for r in rows if str(r["kind"]).startswith("complexity_")], readout)
    by_level = defaultdict(lambda: defaultdict(list))
    by_item = defaultdict(dict)
    for r in items:
        by_level[r["level"]][r["condition"]].append(r)
        by_item[(r["fact_id"], r["level"])][r["condition"]] = r

    levels = {}
    for lvl in sorted(by_level):
        c = by_level[lvl]
        conflict = c["conflict_audio_correct"] + c["conflict_text_correct"]
        facts = [f for (fid, l), f in by_item.items() if l == lvl]
        tracked = [f["audio_only"]["correct"] and f["audio_only_flip"]["correct"]
                   for f in facts if "audio_only" in f and "audio_only_flip" in f]
        acc = {k: _mean([r["correct"] for r in c[k]]) for k in CONFLICT}
        levels[f"L{lvl}"] = {
            "n_facts": len(facts),
            "audio_only_acc": _mean([r["correct"] for r in c["audio_only"]]),
            "audio_only_p": _mean([r["p_expected"] for r in c["audio_only"]]),
            "text_only_acc": _mean([r["correct"] for r in c["text_only"]]),
            "text_only_p": _mean([r["p_expected"] for r in c["text_only"]]),
            "audio_tracking": _mean(tracked),
            "text_follow": _mean([r["chosen_value"] == r["text_value"] for r in conflict]),
            "delta_text_minus_audio": acc[CONFLICT[1]] - acc[CONFLICT[0]] if None not in acc.values() else None,
            "order_flip": _mean([r["order_flip"] for r in conflict]),
        }

    pairs = [(f["audio_only"]["p_expected"], f["conflict_audio_correct"]["p_expected"])
             for f in by_item.values() if "audio_only" in f and "conflict_audio_correct" in f]
    rho, p = spearman(*zip(*pairs)) if len(pairs) > 2 else (None, None)
    return {"readout": readout, "levels": levels, "audio_conf_vs_conflict": {"n": len(pairs), "spearman_rho": rho, "p": p}}


def _fmt(x):
    return "–" if x is None else f"{x:.2f}"


def markdown(res):
    cols = ["audio_only_acc", "audio_only_p", "text_only_acc", "text_only_p", "audio_tracking",
            "text_follow", "delta_text_minus_audio", "order_flip"]
    lines = ["| level | n | " + " | ".join(cols) + " |", "|---|---|" + "---|" * len(cols)]
    for lvl, m in res["levels"].items():
        lines.append(f"| {lvl} | {m['n_facts']} | " + " | ".join(_fmt(m[c]) for c in cols) + " |")
    s = res["audio_conf_vs_conflict"]
    lines += ["", f"Spearman ρ(audio-only P(planted), conflict P(audio value)) = {_fmt(s['spearman_rho'])} "
                  f"(p = {_fmt(s['p'])}, n = {s['n']})"]
    return "\n".join(lines) + "\n"


def plot(res, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    names = list(res["levels"])
    fig, ax = plt.subplots(figsize=(5.5, 4))
    for key, label in [("audio_only_p", "audio-only P(expected)"), ("text_only_p", "text-only P(expected)"),
                       ("text_follow", "conflict text-follow rate")]:
        ax.plot(names, [res["levels"][n][key] for n in names], marker="o", label=label)
    ax.axhline(0.5, color="grey", lw=0.8, ls="--")
    ax.set(ylim=(0, 1.02), xlabel="complexity level", title=f"Complexity sweep ({res['readout']})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", required=True)
    ap.add_argument("--readout", default="yes_no", choices=["yes_no", "ab"])
    args = ap.parse_args()

    run = resolve(args.results)
    res = sweep(read_jsonl(run / "predictions.jsonl"), args.readout)
    tag = f"complexity_sweep_{args.readout}"
    write_json(res, run / f"{tag}.json")
    md = markdown(res)
    (run / f"{tag}.md").write_text(md, encoding="utf-8")
    if plot(res, run / f"{tag}.png"):
        print(f"figure: {run / f'{tag}.png'}")
    print(md)


if __name__ == "__main__":
    main()
