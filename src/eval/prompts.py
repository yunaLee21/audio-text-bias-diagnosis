"""Forced-choice A/B prompt, following Paper 2 (Who Wins the Conflict?, Appendix, Fig. 6).

The assistant turn is pre-filled with ANSWER_PREFIX so the very next token is the answer
letter; its logits give a clean P(A) vs P(B).
"""

SYSTEM_PROMPT = (
    "You are an expert speech and language analyst.\n"
    "Answer the question based on the provided inputs.\n"
    "RESPONSE FORMAT:\n"
    '- Output ONLY valid JSON: {"answer": "A" | "B", "rationale": "brief", "confidence": 0.0-1.0}\n'
    '- Your "answer" MUST be ONLY a single capital letter: either "A" or "B" corresponding to the correct choice.\n'
    "- Do NOT translate, paraphrase, or reformat the choice text.\n"
    "- Do not include any extra text or markdown formatting."
)

# MCR-BENCH's evaluation wrapper (Sec. 3), used by the "mcr" prompt variant.
MCR_DISCLAIMER = (
    "The following text provides descriptions derived from audio content. "
    "This information may be accurate, inaccurate, or irrelevant. "
    "Answer the question with discretion regarding how you use these details."
)

ANSWER_PREFIX = '{"answer":"'


def build_conversation(stim, variant="paper2", order="text_first"):
    """Return a Qwen2-Audio chat conversation for one stimulus."""
    text_parts = []
    if stim["text"] is not None:
        if variant == "mcr":
            text_parts.append({"type": "text", "text": f"{MCR_DISCLAIMER}\nDescription: {stim['text']}\n"})
        else:
            text_parts.append({"type": "text", "text": f"Transcript: {stim['text']}\n"})
    audio_parts = [{"type": "audio", "audio_url": stim["audio"]}] if stim["audio"] else []

    opts = stim["options"]
    question = {
        "type": "text",
        "text": f"Question: {stim['question']}\nChoices: (A) {opts['A']} (B) {opts['B']}\nRespond with JSON only.",
    }
    inputs = text_parts + audio_parts if order == "text_first" else audio_parts + text_parts
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": inputs + [question]},
    ]
