"""Prompts, following Paper 2 (Who Wins the Conflict?, Appendix, Fig. 6).

Two answer formats share the same inputs and system-prompt wording:
  ab      forced choice between two listed options, answer "A" | "B"
  yes_no  one question per answer value, answer "Yes" | "No" (no option list, so no option order)
The assistant turn is pre-filled with ANSWER_PREFIX so the very next token is the answer;
its logits give P(A) vs P(B), or P(Yes) vs P(No).
"""

_FORMATS = {
    "ab": ('{"answer": "A" | "B", "rationale": "brief", "confidence": 0.0-1.0}',
           '"A" or "B" corresponding to the correct choice'),
    "yes_no": ('{"answer": "Yes" | "No", "rationale": "brief", "confidence": 0.0-1.0}',
               '"Yes" or "No"'),
}


def system_prompt(answer_format="ab"):
    schema, allowed = _FORMATS[answer_format]
    single = "a single capital letter" if answer_format == "ab" else "a single word"
    return (
        "You are an expert speech and language analyst.\n"
        "Answer the question based on the provided inputs.\n"
        "RESPONSE FORMAT:\n"
        f"- Output ONLY valid JSON: {schema}\n"
        f'- Your "answer" MUST be ONLY {single}: either {allowed}.\n'
        "- Do NOT translate, paraphrase, or reformat the choice text.\n"
        "- Do not include any extra text or markdown formatting."
    )


SYSTEM_PROMPT = system_prompt("ab")

# MCR-BENCH's evaluation wrapper (Sec. 3), used by the "mcr" prompt variant.
MCR_DISCLAIMER = (
    "The following text provides descriptions derived from audio content. "
    "This information may be accurate, inaccurate, or irrelevant. "
    "Answer the question with discretion regarding how you use these details."
)

ANSWER_PREFIX = '{"answer":"'


def build_conversation(stim, variant="paper2", order="text_first", yes_no_value=None):
    """Return a Qwen2-Audio chat conversation for one stimulus.

    yes_no_value=None -> A/B question; otherwise the Yes/No question about that answer value.
    """
    text_parts = []
    if stim["text"] is not None:
        if variant == "mcr":
            text_parts.append({"type": "text", "text": f"{MCR_DISCLAIMER}\nDescription: {stim['text']}\n"})
        else:
            text_parts.append({"type": "text", "text": f"Transcript: {stim['text']}\n"})
    audio_parts = [{"type": "audio", "audio_url": stim["audio"]}] if stim["audio"] else []

    if yes_no_value is None:
        opts = stim["options"]
        question_text = f"Question: {stim['question']}\nChoices: (A) {opts['A']} (B) {opts['B']}\n"
    else:
        question_text = f"Question: {stim['yes_no'][yes_no_value]}\n"
    question = {"type": "text", "text": question_text + "Respond with JSON only."}

    inputs = text_parts + audio_parts if order == "text_first" else audio_parts + text_parts
    return [
        {"role": "system", "content": system_prompt("ab" if yes_no_value is None else "yes_no")},
        {"role": "user", "content": inputs + [question]},
    ]
