"""Yes/No polling questions: one question per answer value, so no option list and no option order.

Each fact gets {value: question}; e.g. legs -> {"Four": "Does the Pemoux have four legs?",
"Six": "Does the Pemoux have six legs?"}. The readout compares P(Yes) between the two questions,
so a general yes-bias applies to both and cancels (as in POPE-style polling).

Fictitious and complexity facts share these attribute templates; the control facts carry their
own hand-written "yes_no" field in data/positive_control/facts.json.
"""

TEMPLATES = {
    "activity":   {"Day": "Is the {e} active during the day?", "Night": "Is the {e} active at night?"},
    "call_pitch": {"High-pitched": "Does the {e} have a high-pitched call?",
                   "Low-pitched": "Does the {e} have a low-pitched call?"},
    "color":      {"Blue": "Does the {e} have blue fur?", "Orange": "Does the {e} have orange fur?"},
    "habitat":    {"Mountains": "Does the {e} live in the mountains?", "Swamps": "Does the {e} live in the swamps?"},
    "diet":       {"Fruit": "Does the {e} eat only fruit?", "Insects": "Does the {e} eat only insects?"},
    "climate":    {"Hot": "Does the {e} live in a hot climate?", "Cold": "Does the {e} live in a cold climate?"},
    "social":     {"Alone": "Does the {e} live alone?", "In groups": "Does the {e} live in groups?"},
    "legs":       {"Four": "Does the {e} have four legs?", "Six": "Does the {e} have six legs?"},
    "size":       {"Smaller": "Is the {e} smaller than a cat?", "Larger": "Is the {e} larger than a cat?"},
    "season":     {"Spring": "Does the {e} lay its eggs in spring?", "Autumn": "Does the {e} lay its eggs in autumn?"},
    "covering":   {"Scales": "Is the {e}'s body covered in scales?", "Feathers": "Is the {e}'s body covered in feathers?"},
    "hearts":     {"Two": "Does the {e} have two hearts?", "Three": "Does the {e} have three hearts?"},
}


def questions(fact):
    """Return {value: yes/no question} for a fact."""
    if "yes_no" in fact:
        return fact["yes_no"]
    return {value: q.format(e=fact["entity"]) for value, q in TEMPLATES[fact["attribute"]].items()}
