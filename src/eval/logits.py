"""Model loading and the two uncertainty readouts.

logit_readout     one forward pass with the assistant turn pre-filled up to the answer letter;
                  P(A), P(B) come from the next-token distribution. `mass_ab` (the share of the
                  full-vocabulary probability that lands on "A" or "B") is the sanity check:
                  if it is low, the model wants to emit something else and the readout is unreliable.
self_consistency  fallback when logits are unusable: sample n free-form answers, count letters.
"""

import inspect
import re

import torch

from src.eval.prompts import ANSWER_PREFIX

# tolerates the formats seen in free-form output: "answer", 'answer', \"Answer\"
LETTER_RE = re.compile(r'''answer['"\\]*\s*:\s*['"\\]*\s*([AB])\b''', re.IGNORECASE)


def load_model(model_cfg):
    import transformers
    from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

    dtype = getattr(torch, model_cfg["dtype"])
    # torch_dtype was renamed to dtype in transformers 4.56
    major, minor = (int(x) for x in transformers.__version__.split(".")[:2])
    dtype_key = "dtype" if (major, minor) >= (4, 56) else "torch_dtype"
    kwargs = {"device_map": "auto", dtype_key: dtype}
    if model_cfg.get("quantization") in ("4bit", "8bit"):
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = (
            BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=dtype)
            if model_cfg["quantization"] == "4bit" else BitsAndBytesConfig(load_in_8bit=True)
        )
    processor = AutoProcessor.from_pretrained(model_cfg["hf_id"])
    model = Qwen2AudioForConditionalGeneration.from_pretrained(model_cfg["hf_id"], **kwargs)
    model.eval()
    return model, processor


def load_audio(path, processor):
    import librosa
    sr = processor.feature_extractor.sampling_rate
    return librosa.load(path, sr=sr)[0]


def prepare_inputs(model, processor, conversation, audio, prefill):
    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    if prefill:
        text += ANSWER_PREFIX
    kwargs = {"text": text, "return_tensors": "pt"}
    if audio is not None:
        # the audio kwarg was renamed audios -> audio in newer transformers
        params = inspect.signature(processor.__call__).parameters
        kwargs["audio" if "audio" in params else "audios"] = [audio]
        kwargs["sampling_rate"] = processor.feature_extractor.sampling_rate
    inputs = processor(**kwargs).to(model.device)
    for k, v in inputs.items():
        if torch.is_floating_point(v):
            inputs[k] = v.to(model.dtype)
    return inputs


def answer_token_ids(tokenizer, words):
    ids = {}
    for word in words:
        toks = tokenizer.encode(word, add_special_tokens=False)
        if len(toks) != 1:
            raise RuntimeError(f"'{word}' is not a single token ({toks}); logit readout would be wrong")
        ids[word] = toks[0]
    return ids


@torch.no_grad()
def _next_token_pair(model, processor, conversation, audio, words):
    """Logits of two candidate answer tokens right after the pre-filled answer prefix."""
    inputs = prepare_inputs(model, processor, conversation, audio, prefill=True)
    logits = model(**inputs).logits[0, -1].float()
    ids = answer_token_ids(processor.tokenizer, words)
    probs = logits.softmax(-1)
    a, b = words
    pair = torch.stack([logits[ids[a]], logits[ids[b]]]).softmax(-1)
    return logits, ids, probs, pair


def logit_readout(model, processor, conversation, audio):
    logits, ids, probs, pair = _next_token_pair(model, processor, conversation, audio, ("A", "B"))
    return {
        "logit_A": float(logits[ids["A"]]),
        "logit_B": float(logits[ids["B"]]),
        "p_A": float(pair[0]),
        "p_B": float(pair[1]),
        "mass_ab": float(probs[ids["A"]] + probs[ids["B"]]),
        "top_token": processor.tokenizer.decode([int(logits.argmax())]),
    }


def yes_no_readout(model, processor, conversation, audio):
    """P(Yes) for one Yes/No question; log_odds = logit(Yes) - logit(No)."""
    logits, ids, probs, pair = _next_token_pair(model, processor, conversation, audio, ("Yes", "No"))
    return {
        "log_odds": float(logits[ids["Yes"]] - logits[ids["No"]]),
        "p_yes": float(pair[0]),
        "mass_yn": float(probs[ids["Yes"]] + probs[ids["No"]]),
        "top_token": processor.tokenizer.decode([int(logits.argmax())]),
    }


def parse_letter(text):
    m = LETTER_RE.search(text)
    if m:
        return m.group(1)
    m = re.search(r"\b([AB])\b", text)
    return m.group(1) if m else None


@torch.no_grad()
def generate(model, processor, conversation, audio, n=1, temperature=None, max_new_tokens=48):
    """Free-form generation without the pre-filled prefix (checks real format compliance)."""
    inputs = prepare_inputs(model, processor, conversation, audio, prefill=False)
    gen_kwargs = {"max_new_tokens": max_new_tokens, "num_return_sequences": n}
    if temperature:
        gen_kwargs.update(do_sample=True, temperature=temperature)
    else:
        gen_kwargs.update(do_sample=False)
    out = model.generate(**inputs, **gen_kwargs)
    new = out[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(new, skip_special_tokens=True)


def self_consistency(model, processor, conversation, audio, n, temperature):
    texts = generate(model, processor, conversation, audio, n=n, temperature=temperature)
    letters = [parse_letter(t) for t in texts]
    valid = [l for l in letters if l]
    n_a, n_b = valid.count("A"), valid.count("B")
    total = max(len(valid), 1)
    return {"p_A": n_a / total, "p_B": n_b / total, "n_valid": len(valid), "samples": letters}
