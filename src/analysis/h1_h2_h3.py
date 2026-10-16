"""Sort fictitious facts into H1 / H2 / H3 evidence buckets (report/Report Summary.md, Sec. 1).

Per fact, using P(planted) in audio_only as the audio-side confidence (threshold --tau):
  H1 suppression    audio_only confident & correct, but conflict_audio_correct confidently picks the text
  flip_unsure       audio_only confident & correct, conflict picks the text with low confidence
  held              audio_only confident & correct, conflict still follows the audio
  H2 weak_tracking  audio_only weak, but the answer tracks the input (audio_only vs audio_only_flip differ correctly)
  H3 not_tracking   audio_only answer does not follow the input when the stated value is flipped
Group level (H3): audio_only / text_only accuracy vs chance (binomial test), input-tracking rate,
and the share of answers that are "A" (position bias).

    python -m src.analysis.h1_h2_h3 --results results/pilot_run_1
"""

import argparse
from collections import Counter, defaultdict

from src.common import read_jsonl, resolve, write_json


def bucket(fact_rows, tau):
    a = fact_rows.get("audio_only")
    a_flip = fact_rows.get("audio_only_flip")
    conflict = fact_rows.get("conflict_audio_correct")
    if a is None:
        return None
    tracks = a_flip is not None and a["correct"] and a_flip["correct"]
    if a_flip is not None and not tracks and a["chosen_value"] == a_flip["chosen_value"]:
        return "H3_not_tracking"
    if a["p_expected"] < tau:
        return "H2_weak_tracking" if tracks else "weak_unclear"
    if not a["correct"]:
        return "weak_unclear"
    if conflict is None:
        return "confident_no_conflict_run"
    if conflict["correct"]:
        return "held"
    return "H1_suppression" if conflict["confidence"] >= tau else "flip_unsure"


def binom_p(k, n):
    try:
        from scipy.stats import binomtest
        return binomtest(k, n, 0.5, alternative="greater").pvalue
    except ImportError:
        return None


def tracking(by_fact, base, flip):
    pairs = [(f[base], f[flip]) for f in by_fact.values() if base in f and flip in f]
    if not pairs:
        return None
    return sum(b["correct"] and fl["correct"] for b, fl in pairs) / len(pairs)


def analyze(rows, tau):
    rows = [r for r in rows if r["kind"] == "fictitious"]
    by_fact = defaultdict(dict)
    for r in rows:
        by_fact[r["fact_id"]][r["condition"]] = r

    buckets = {fid: bucket(conds, tau) for fid, conds in sorted(by_fact.items())}
    group = {}
    for cond in ("audio_only", "text_only"):
        rs = [r for r in rows if r["condition"] == cond]
        if rs:
            k = sum(r["correct"] for r in rs)
            group[cond] = {"n": len(rs), "accuracy": k / len(rs), "p_vs_chance": binom_p(k, len(rs))}
    group["audio_tracking_rate"] = tracking(by_fact, "audio_only", "audio_only_flip")
    group["text_tracking_rate"] = tracking(by_fact, "text_only", "text_only_flip")
    group["share_answer_A"] = sum(r["chosen_letter"] == "A" for r in rows) / len(rows) if rows else None
    return {"tau": tau, "counts": dict(Counter(b for b in buckets.values() if b)), "group": group,
            "per_fact": buckets}


def plot(rows, path):
    """audio-only confidence vs P(audio value) under conflict, one dot per fact."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    by_fact = defaultdict(dict)
    for r in rows:
        if r["kind"] == "fictitious":
            by_fact[r["fact_id"]][r["condition"]] = r
    pts = [(f["audio_only"]["p_expected"], f["conflict_audio_correct"]["p_expected"])
           for f in by_fact.values() if "audio_only" in f and "conflict_audio_correct" in f]
    if not pts:
        return False
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.scatter(*zip(*pts), alpha=0.8)
    ax.axhline(0.5, color="grey", lw=0.8, ls="--")
    ax.axvline(0.5, color="grey", lw=0.8, ls="--")
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="audio-only P(planted)",
           ylabel="conflict P(audio value)", title="Audio confidence vs. conflict outcome")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", required=True)
    ap.add_argument("--tau", type=float, default=0.8, help="confidence threshold for 'confident'")
    args = ap.parse_args()

    run = resolve(args.results)
    rows = read_jsonl(run / "predictions.jsonl")
    res = analyze(rows, args.tau)
    write_json(res, run / "h_buckets.json")
    if plot(rows, run / "audio_conf_vs_conflict.png"):
        print(f"figure: {run / 'audio_conf_vs_conflict.png'}")
    print(f"bucket counts (tau={args.tau}): {res['counts']}")
    for k, v in res["group"].items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
