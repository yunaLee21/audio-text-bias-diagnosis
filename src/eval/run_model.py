"""Run stimuli through the model and write results/<run>/predictions.jsonl + metrics.

    python -m src.eval.run_model --out results/pilot_run_1
    python -m src.eval.run_model --out results/smoke --limit 4 --conditions conflict_audio_correct
    python -m src.eval.run_model --stimuli data/complexity/stimuli.jsonl --out results/complexity_run_1

Predictions are appended one row at a time and already-finished stim_ids are skipped, so a
Colab/Kaggle session that dies mid-run can simply be restarted with the same command.
--mock replaces the model with random answers to test the pipeline without a GPU.
"""

import argparse
import json
import math
import random

from src.common import load_config, read_jsonl, resolve
from src.eval import metrics
from src.eval.prompts import build_conversation


def predict(stim, backend, cfg):
    """Return the model-side fields of one prediction row."""
    conv = build_conversation(stim, cfg["prompt"]["variant"], cfg["prompt"]["order"])
    ev = cfg["eval"]
    # Yes/No polling does not depend on the A/B order: run it once per item, on the first order only
    want_yn = "yes_no" in ev.get("readouts", ["ab"]) and not stim["stim_id"].endswith("_o2")

    if backend == "mock":
        la, lb = random.gauss(0, 2), random.gauss(0, 2)
        p_a = math.exp(la) / (math.exp(la) + math.exp(lb))
        out = {"logit_A": la, "logit_B": lb, "p_A": p_a, "p_B": 1 - p_a}
        if want_yn:
            out["yes_no_out"] = {}
            for v in stim["yes_no"]:
                lo = random.gauss(0, 2)
                out["yes_no_out"][v] = {"log_odds": lo, "p_yes": 1 / (1 + math.exp(-lo)), "mass_yn": 1.0}
        return out

    from src.eval import logits as L
    model, processor = backend
    audio = L.load_audio(str(resolve(stim["audio"])), processor) if stim["audio"] else None
    if ev["uq"] == "logit":
        out = L.logit_readout(model, processor, conv, audio)
    else:
        out = L.self_consistency(model, processor, conv, audio, ev["n_samples"], ev["temperature"])
    if ev.get("greedy_check"):
        raw = L.generate(model, processor, conv, audio)[0]
        out.update(greedy_raw=raw, greedy_letter=L.parse_letter(raw))
    if want_yn:
        out["yes_no_out"] = {
            v: L.yes_no_readout(model, processor,
                                build_conversation(stim, cfg["prompt"]["variant"], cfg["prompt"]["order"], v), audio)
            for v in stim["yes_no"]
        }
    return out


def score(stim, out):
    letter = "A" if out["p_A"] >= out["p_B"] else "B"
    exp_letter = "A" if stim["options"]["A"] == stim["expected"] else "B"
    row = dict(stim, **out)
    row.update(
        chosen_letter=letter,
        chosen_value=stim["options"][letter],
        correct=stim["options"][letter] == stim["expected"],
        confidence=max(out["p_A"], out["p_B"]),
        p_expected=out[f"p_{exp_letter}"],
    )
    if "logit_A" in out:
        other = "B" if exp_letter == "A" else "A"
        row["margin"] = out[f"logit_{exp_letter}"] - out[f"logit_{other}"]
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/pilot.yaml")
    ap.add_argument("--stimuli", default="data/generated/stimuli.jsonl")
    ap.add_argument("--out", default="results/pilot_run_1")
    ap.add_argument("--conditions", nargs="+", help="subset of conditions (default: config)")
    ap.add_argument("--kinds", nargs="+", help="only these dataset kinds (default: all in the file)")
    ap.add_argument("--limit", type=int, help="only the first N facts (smoke tests)")
    ap.add_argument("--uq", choices=["logit", "self_consistency"])
    ap.add_argument("--mock", action="store_true", help="random answers, no model (pipeline test)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    random.seed(cfg["seed"])
    if args.uq:
        cfg["eval"]["uq"] = args.uq
    conditions = set(args.conditions or cfg["eval"]["conditions"])

    stimuli = [s for s in read_jsonl(args.stimuli) if s["condition"] in conditions and (not args.kinds or s["kind"] in args.kinds)]
    if args.limit:
        keep = []
        for kind in sorted({s["kind"] for s in stimuli}):
            ids = sorted({s["fact_id"] for s in stimuli if s["kind"] == kind})[: args.limit]
            keep += ids
        stimuli = [s for s in stimuli if s["fact_id"] in keep]

    out_dir = resolve(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / "predictions.jsonl"
    done = {r["stim_id"] for r in read_jsonl(pred_path)} if pred_path.exists() else set()
    todo = [s for s in stimuli if s["stim_id"] not in done]
    (out_dir / "config_used.json").write_text(json.dumps(dict(cfg, args=vars(args)), indent=2), encoding="utf-8")
    print(f"{len(stimuli)} stimuli selected, {len(done)} already done, {len(todo)} to run")

    if args.mock:
        backend = "mock"
    else:
        from src.eval.logits import load_model
        backend = load_model(cfg["model"])

    with open(pred_path, "a", encoding="utf-8") as f:
        for i, stim in enumerate(todo, 1):
            row = score(stim, predict(stim, backend, cfg))
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            print(f"[{i}/{len(todo)}] {stim['stim_id']}: {row['chosen_value']!r} "
                  f"(expected {stim['expected']!r}, conf {row['confidence']:.2f})")

    print(metrics.write_metrics(args.out))


if __name__ == "__main__":
    main()
