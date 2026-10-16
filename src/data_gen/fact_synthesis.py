"""Invent fictitious entity + binary-attribute facts (the benchmark's source of truth).

Each fact names a made-up creature and one attribute with two possible values. Which value is
the "true" (planted) one is randomized per fact, so a model that ignores the input cannot score
above chance — this is what separates H2 (weak but input-tracking) from H3 (memory fallback).

    python -m src.data_gen.fact_synthesis --n 15 --out data/fictitious_facts/facts.json

Generated names are random syllable strings: skim the output once to drop anything that
accidentally resembles a real word or name.
"""

import argparse
import random

from src.common import write_json

ONSETS = ["b", "d", "f", "g", "k", "l", "m", "n", "p", "r", "s", "t", "v", "z",
          "br", "dr", "gl", "kr", "pl", "sk", "tr", "vr", "zh"]
VOWELS = ["a", "e", "i", "o", "u", "ai", "ou"]
CODAS = ["", "", "n", "l", "r", "x", "th", "m"]

# key: (question, (label_0, statement_0), (label_1, statement_1)); "{e}" is the entity name.
ATTRIBUTES = {
    "activity": ("When is the {e} active?",
                 ("Day", "The {e} sleeps at night and is active during the day."),
                 ("Night", "The {e} sleeps during the day and is active at night.")),
    "call_pitch": ("What does the {e}'s call sound like?",
                   ("High-pitched", "The {e} has a high-pitched call."),
                   ("Low-pitched", "The {e} has a low-pitched call.")),
    "color": ("What color is the {e}'s fur?",
              ("Blue", "The {e} has blue fur."),
              ("Orange", "The {e} has orange fur.")),
    "habitat": ("Where does the {e} live?",
                ("Mountains", "The {e} lives high in the mountains."),
                ("Swamps", "The {e} lives deep in the swamps.")),
    "diet": ("What does the {e} eat?",
             ("Fruit", "The {e} eats only fruit."),
             ("Insects", "The {e} eats only insects.")),
    "climate": ("What climate does the {e} live in?",
                ("Hot", "The {e} can only survive in hot climates."),
                ("Cold", "The {e} can only survive in cold climates.")),
    "social": ("How does the {e} live?",
               ("Alone", "The {e} always lives alone."),
               ("In groups", "The {e} always lives in large groups.")),
    "legs": ("How many legs does the {e} have?",
             ("Four", "The {e} has four legs."),
             ("Six", "The {e} has six legs.")),
    "size": ("How big is the {e} compared to a cat?",
             ("Smaller", "The {e} is smaller than a cat."),
             ("Larger", "The {e} is larger than a cat.")),
    "season": ("When does the {e} lay its eggs?",
               ("Spring", "The {e} lays its eggs in spring."),
               ("Autumn", "The {e} lays its eggs in autumn.")),
    "covering": ("What covers the {e}'s body?",
                 ("Scales", "The {e}'s body is covered in scales."),
                 ("Feathers", "The {e}'s body is covered in feathers.")),
    "hearts": ("How many hearts does the {e} have?",
               ("Two", "The {e} has two hearts."),
               ("Three", "The {e} has three hearts.")),
}


def make_name(rng, taken):
    while True:
        n_syll = rng.choice([2, 2, 3])
        name = "".join(rng.choice(ONSETS) + rng.choice(VOWELS) for _ in range(n_syll))
        name = (name + rng.choice(CODAS)).capitalize()
        if 6 <= len(name) <= 10 and name not in taken:
            taken.add(name)
            return name


def make_facts(n, seed):
    rng = random.Random(seed)
    keys = list(ATTRIBUTES)
    rng.shuffle(keys)
    # balanced coin: half the facts plant the first value, half the second (chance stays at 50%)
    plant_first = [i < n // 2 for i in range(n)]
    rng.shuffle(plant_first)
    taken, facts = set(), []
    for i in range(n):
        attr = keys[i % len(keys)]  # cycle so attributes are balanced across facts
        question, (l0, s0), (l1, s1) = ATTRIBUTES[attr]
        entity = make_name(rng, taken)
        planted, flipped = (l0, l1) if plant_first[i] else (l1, l0)
        facts.append({
            "fact_id": f"f{i + 1:03d}",
            "kind": "fictitious",
            "entity": entity,
            "attribute": attr,
            "question": question.format(e=entity),
            "values": [l0, l1],
            "statements": {l0: s0.format(e=entity), l1: s1.format(e=entity)},
            "planted": planted,
            "flipped": flipped,
        })
    return facts


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data/fictitious_facts/facts.json")
    args = ap.parse_args()

    facts = make_facts(args.n, args.seed)
    write_json(facts, args.out)
    n_first = sum(f["planted"] == f["values"][0] for f in facts)
    print(f"wrote {len(facts)} facts to {args.out} (planted = first value in {n_first}/{len(facts)})")


if __name__ == "__main__":
    main()
