"""Turn facts into evaluation stimuli (one JSONL row per fact x condition).

Adapts MCR-BENCH's faithful / adversarial / irrelevant text conditions to invented facts:
  faithful    -> text states the same value as the audio
  adversarial -> text states the other value (conflict_audio_correct)
  irrelevant  -> text is a different fact from the same pool (no real corpus to retrieve from)
plus single-modality baselines, a question-only baseline and the reverse conflict direction. Conflict items are emitted in
both A/B orders (stim_id suffix _o1 / _o2); metrics.collapse_orders() averages them.

    python -m src.data_gen.text_variants \
        --facts data/fictitious_facts/facts.json data/positive_control/facts.json \
        --out data/generated/stimuli.jsonl

"expected" is the value the model should answer if it reads the input; for the conflict
conditions it is the planted value, carried by audio in one direction and by text in the other.
"""

import argparse
import random

from src.common import audio_path, read_json, write_jsonl
from src.data_gen.yes_no import questions

# condition -> (audio clip, text statement, expected answer)
#   'planted' / 'flipped' name which value that modality states; None = modality absent.
CONDITIONS = {
    "audio_only":             ("planted", None,      "planted"),
    "audio_only_flip":        ("flipped", None,      "flipped"),
    "text_only":              (None,      "planted", "planted"),
    "text_only_flip":         (None,      "flipped", "flipped"),
    "faithful":               ("planted", "planted", "planted"),
    "conflict_audio_correct": ("planted", "flipped", "planted"),  # MCR-BENCH "adversarial"
    "conflict_text_correct":  ("flipped", "planted", "planted"),
    "irrelevant":             ("planted", "other",   "planted"),
    # question only: baseline for each value's prior likelihood (calibrates the Yes/No readout)
    "no_input":               (None,      None,      "planted"),
}
COUNTERBALANCED = {"conflict_audio_correct", "conflict_text_correct"}


def build_stimuli(facts, audio_dir, seed):
    rng = random.Random(seed)
    rows = []
    for fact in facts:
        # irrelevant distractor: a fact of the same kind about a different attribute
        others = [f for f in facts if f["kind"] == fact["kind"] and f["attribute"] != fact["attribute"]]
        distractor = rng.choice(others) if others else None

        for cond, (audio_key, text_key, expected_key) in CONDITIONS.items():
            if text_key == "other" and distractor is None:
                continue
            options = list(fact["values"])
            rng.shuffle(options)  # randomize A/B position per stimulus (as in Paper 2)
            # conflict items are run in both A/B orders; metrics average the two so position bias cancels
            orders = [options, options[::-1]] if cond in COUNTERBALANCED else [options]

            if text_key == "other":
                text, text_value = distractor["statements"][distractor["planted"]], None
            elif text_key:
                text_value = fact[text_key]
                text = fact["statements"][text_value]
            else:
                text, text_value = None, None

            for k, opts in enumerate(orders, 1):
                rows.append({
                    "stim_id": f"{fact['fact_id']}_{cond}" + (f"_o{k}" if len(orders) > 1 else ""),
                    "fact_id": fact["fact_id"],
                    "kind": fact["kind"],
                    "level": fact.get("level"),
                    "attribute": fact["attribute"],
                    "condition": cond,
                    "audio": audio_path(audio_dir, fact["fact_id"], audio_key) if audio_key else None,
                    "audio_value": fact[audio_key] if audio_key else None,
                    "text": text,
                    "text_value": text_value,
                    "question": fact["question"],
                    "yes_no": questions(fact),
                    "options": {"A": opts[0], "B": opts[1]},
                    "expected": fact[expected_key],
                    "planted": fact["planted"],
                })
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--facts", nargs="+",
                    default=["data/fictitious_facts/facts.json", "data/positive_control/facts.json"])
    ap.add_argument("--audio-dir", default="data/audio")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data/generated/stimuli.jsonl")
    args = ap.parse_args()

    facts = [f for path in args.facts for f in read_json(path)]
    rows = build_stimuli(facts, args.audio_dir, args.seed)
    write_jsonl(rows, args.out)
    print(f"wrote {len(rows)} stimuli from {len(facts)} facts to {args.out}")


if __name__ == "__main__":
    main()
