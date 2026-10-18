"""Build the complexity dataset: the same fictitious facts at four sentence-difficulty levels.

    L0  baseline         The Zufovum has six legs.
    L1  extra info       The Zufovum, which has blue fur, has six legs and lays its eggs in spring.
    L2  deep embedding   Researchers who spent several years ... have reported that the Zufovum, ..., has six legs.
    L3  lure             Unlike its close relative the Trovaim, which has four legs, the Zufovum has six legs.

Every level is a minimal pair: the planted and flipped sentences differ only in the tested value
(L3 swaps the two values between the entity and its relative, so both values appear in both).
Filler, frame and lure choices are fixed per fact, so a fact's levels differ only in structure.
Each (fact, level) becomes its own fact row with kind "complexity_L<n>", so text_variants.py and
tts.py work on it unchanged.

    python -m src.data_gen.complexity --n 12 --out data/complexity/facts.json
    python -m src.data_gen.text_variants --facts data/complexity/facts.json \
        --audio-dir data/complexity/audio --out data/complexity/stimuli.jsonl
    python -m src.data_gen.tts --facts data/complexity/facts.json --out data/complexity/audio
"""

import argparse
import random

from src.common import write_json
from src.data_gen.fact_synthesis import make_name

# key: (question, (label_0, predicate_0), (label_1, predicate_1)); predicates follow "The {e} ..."
PREDICATES = {
    "activity": ("When is the {e} active?",
                 ("Day", "is active during the day"), ("Night", "is active at night")),
    "call_pitch": ("What does the {e}'s call sound like?",
                   ("High-pitched", "has a high-pitched call"), ("Low-pitched", "has a low-pitched call")),
    "color": ("What color is the {e}'s fur?",
              ("Blue", "has blue fur"), ("Orange", "has orange fur")),
    "habitat": ("Where does the {e} live?",
                ("Mountains", "lives high in the mountains"), ("Swamps", "lives deep in the swamps")),
    "diet": ("What does the {e} eat?",
             ("Fruit", "eats only fruit"), ("Insects", "eats only insects")),
    "climate": ("What climate does the {e} live in?",
                ("Hot", "can only survive in hot climates"), ("Cold", "can only survive in cold climates")),
    "social": ("How does the {e} live?",
               ("Alone", "always lives alone"), ("In groups", "always lives in large groups")),
    "legs": ("How many legs does the {e} have?",
             ("Four", "has four legs"), ("Six", "has six legs")),
    "size": ("How big is the {e} compared to a cat?",
             ("Smaller", "is smaller than a cat"), ("Larger", "is larger than a cat")),
    "season": ("When does the {e} lay its eggs?",
               ("Spring", "lays its eggs in spring"), ("Autumn", "lays its eggs in autumn")),
    "covering": ("What covers the {e}'s body?",
                 ("Scales", "has a body covered in scales"), ("Feathers", "has a body covered in feathers")),
    "hearts": ("How many hearts does the {e} have?",
               ("Two", "has two hearts"), ("Three", "has three hearts")),
}

LEVELS = {
    0: ["The {e} {p}."],
    1: ["The {e}, which {f1}, {p} and {f2}."],
    2: [
        "Researchers who spent several years observing animals in the remote valleys have reported "
        "that the {e}, which few people have ever seen, {p}.",
        "According to an old field guide that was rediscovered in a library last year, the {e}, "
        "a creature rarely mentioned by travellers, {p}.",
        "Villagers living near the forest, who have told stories about it for generations, insist "
        "that the {e}, despite its shy nature, {p}.",
    ],
    3: [
        "Unlike its close relative the {e2}, which {lure}, the {e} {p}.",
        "The {e} {p}, unlike its close relative the {e2}, which {lure}.",
    ],
}


def make_facts(n, seed):
    rng = random.Random(seed)
    keys = list(PREDICATES)
    rng.shuffle(keys)
    plant_first = [i < n // 2 for i in range(n)]
    rng.shuffle(plant_first)
    taken, facts = set(), []
    for i in range(n):
        attr = keys[i % len(keys)]
        question, (l0, p0), (l1, p1) = PREDICATES[attr]
        pred = {l0: p0, l1: p1}
        planted, flipped = (l0, l1) if plant_first[i] else (l1, l0)
        entity, relative = make_name(rng, taken), make_name(rng, taken)

        # per-fact choices shared by the planted and flipped sentences (keeps every level a minimal pair)
        filler_attrs = rng.sample([k for k in keys if k != attr], 2)
        fillers = [PREDICATES[k][rng.randint(1, 2)][1] for k in filler_attrs]
        frames = {lvl: rng.choice(templates) for lvl, templates in LEVELS.items()}

        for lvl, frame in frames.items():
            statements = {}
            for value in (l0, l1):
                other = l1 if value == l0 else l0
                statements[value] = frame.format(e=entity, e2=relative, p=pred[value], lure=pred[other],
                                                 f1=fillers[0], f2=fillers[1])
            facts.append({
                "fact_id": f"c{i + 1:03d}_L{lvl}",
                "base_fact_id": f"c{i + 1:03d}",
                "kind": f"complexity_L{lvl}",
                "level": lvl,
                "entity": entity,
                "lure_entity": relative if lvl == 3 else None,
                "attribute": attr,
                "question": question.format(e=entity),
                "values": [l0, l1],
                "statements": statements,
                "planted": planted,
                "flipped": flipped,
            })
    return facts


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=12, help="number of base facts (each appears at every level)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="data/complexity/facts.json")
    args = ap.parse_args()

    facts = make_facts(args.n, args.seed)
    write_json(facts, args.out)
    print(f"wrote {args.n} base facts x {len(LEVELS)} levels = {len(facts)} rows to {args.out}")


if __name__ == "__main__":
    main()
