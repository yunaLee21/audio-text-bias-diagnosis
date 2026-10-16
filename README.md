# Diagnosing Text Bias in Audio-Language Models with a Contamination-Free Benchmark

> Phân tích nguyên nhân thiên vị văn bản trong mô hình Audio-Ngôn ngữ qua benchmark không nhiễu dữ liệu huấn luyện

Undergraduate thesis project. This repo will hold the data-generation pipeline, model-evaluation harness, and analysis code for a benchmark that tests **why** Audio-Language Models (Audio-LLMs) default to text over audio when the two disagree — and specifically whether that "text bias" is a genuine modality-level preference, or an artifact of two confounds no prior benchmark controls for.

Status: **pilot code written, not yet run on the model**. This README captures the full background, the design that came out of two advisor meetings, and the plan for the pilot implementation about to start.

---

## 1. Background — why this project exists

### 1.1 The phenomenon (already established)

When an Audio-LLM receives audio together with a text description, and the description is wrong, the model's answer tends to follow the **text**, not the audio — even though the audio is correct. This is not a hunch; it's been benchmarked independently at least twice in the last ~12 months:

- **MCR-BENCH** (EMNLP 2025, [`2508.15407`](Paper/2508.15407%20-%20When%20Audio%20and%20Text%20Disagree%20(MCR-BENCH).pdf)) — 3,000 samples across 3 tasks (audio QA, speech emotion recognition, vocal sound classification). Under adversarial (contradicting) text, accuracy collapses: e.g. Qwen-Audio-Chat 87.8% → 1.7%. Text Influence Rate (fraction of predictions that flip when text is added) exceeds 95% for most models. Critically, a probe on the model's own hidden states can tell faithful vs. adversarial audio-text pairs apart with 98% accuracy — **the model internally "knows" there's a conflict, but that awareness never reaches the output.**
- **ALME** (2026, [`2602.11488`](Paper/2602.11488%20-%20When%20Audio-LLMs%20Dont%20Listen%20(ALME).pdf)) — extends this to 8 languages, 57,600 stimuli. Models follow misleading text 10–26× more often than a text-vs-text control, *even when explicitly instructed to trust the audio* (GPT-4o: 23.2% text-follow rate; Gemini 2.0 Flash: 16.6%).

So "does text dominate audio?" is a closed question — **yes, strongly**. The open question is *why*.

### 1.2 Why it happens: perception failure or arbitration failure?

Three independent mechanistic lines of evidence all point the same direction — **arbitration/suppression, not perception**:

| Evidence | Paper | Finding |
|---|---|---|
| Activation patching | *Who Wins the Conflict?* ([`2606.18924`](Paper/2606.18924%20-%20Who%20Wins%20the%20Conflict%20(Mechanistic%20Interpretability).pdf)) | Localizes a "text circuit" (specific attention heads/MLPs). Ablating it makes audio accuracy jump (Qwen: 33%→82%). Audio was correctly represented all along — the circuit was actively *suppressing* it. |
| Counterfactual repair | *Beyond Text Following* / GACL ([`2606.05161`](Paper/2606.05161%20-%20Beyond%20Text%20Following%20(Arbitration%20Reversals).pdf)) | Simply deleting the misleading text and rerunning fixes 64.1% of wrong answers — the correct (audio-grounded) answer was already present in the model, just overridden at the answer position. |
| Attention-head localization | *Are Audio-Language Models Listening?* ([`2603.06854`](Paper/2603.06854%20-%20Are%20Audio-Language%20Models%20Listening%20(Audio-Specialist%20Heads).pdf)) | A small set of "audio-specialist heads" predicts correctness; steering them up recovers +8.0 pts on MMAU with no retraining. |

**Conclusion so far:** text dominance is diagnosable and (partially) fixable at inference time via circuit-level interventions — it is not a hard ceiling on the model's audio-understanding capacity. All three fixes are inference-time only; none involve retraining.

A tempting architectural explanation is that most Audio-LLMs are a text-pretrained LLM backbone (trillions of text tokens) with an audio encoder/adapter bolted on and fine-tuned on much less audio-text data — so a strong "text circuit" is inherited from the backbone. That's plausible as a *root cause* for why the circuit exists, but it does **not** by itself predict whether the failure mode is suppression (H1 below) or weak representation (H2) — and the inference-time-only recovery evidence above leans toward suppression. This is exactly the empirical question this thesis is designed to test, not something that can be settled by architectural reasoning alone.

### 1.3 The two confounds no existing benchmark controls for

This is the actual gap the thesis targets. Every benchmark above uses **real, naturally-occurring** audio (Common Voice, AudioCaps, MELD, …). That introduces two confounds that get silently baked into the headline "text bias" numbers:

1. **Confound #8 — concept typicality.** A dog bark is extremely common in pretraining data; the model has a very strong prior for it, so text saying "this is a cat" can't override it. A rare/ambiguous sound has a weak prior and gets overridden easily. None of the existing benchmarks stratify by how typical/familiar the audio concept is — so an aggregate "TIR > 95%" может hide a distribution that's ~0% dominance for familiar sounds and ~100% for unfamiliar ones. (Indirect support from the vision side: *Mechanisms of Prompt-Induced Hallucination in VLMs*, [`2601.05201`](Paper/2601.05201%20-%20Mechanisms%20of%20Prompt-Induced%20Hallucination%20in%20VLMs.pdf), finds base-probability confidence correlates with resistance to prompt override at ρ = 0.37–0.46.)
2. **Confound #9 — training-data contamination.** Stimuli from Common Voice / AudioCaps were very likely seen during pretraining. A model might be "remembering" a clip it already knows rather than actually reasoning over new audio. ALME flags this itself but argues it affects all models equally so relative comparisons still hold — absolute numbers do not.

These are **different** confounds: #8 is about how common an audio *concept* is in general; #9 is about whether this *specific sample* leaked into pretraining. A benchmark could be clean on #9 (novel clips) and still fail on #8 (a novel clip of an everyday sound like a dog bark).

**Additionally**, no prior work separates:
- Whether the model is actually reading the current input at all, versus **falling back on parametric memory** (a pattern/answer it "remembers" from pretraining, unrelated to what's actually in front of it) — a distinction well studied in text-only NLP (context-faithfulness vs. parametric-knowledge-reliance / knowledge-conflict literature) but never applied to multimodal (audio-text) conflict.
- *How confident* the model is when it picks a side. Every existing benchmark only measures which modality "won," never how close the call was. Two models with identical accuracy/TIR could be either confidently wrong or barely-50/50 wrong — indistinguishable today.

---

## 2. Research questions / hypotheses

The thesis benchmark is designed to pull apart three hypotheses that current behavioral-only measurements cannot distinguish:

| | Audio-only baseline | Conflicting condition | What it means |
|---|---|---|---|
| **H1 — Suppression (real bias)** | High confidence, correct | Still confidently flips to text | Genuine modality-level bias at the decision stage |
| **H2 — Representation / confidence confound** | Low confidence to begin with | Flips to text | Audio-side weakness because the concept is novel/unfamiliar, not true modality bias |
| **H3 — Memory fallback** | Answer doesn't track the (fictitious) ground truth at all, even alone | — | Model isn't reading the input, it's defaulting to something it "remembers" |

An earlier version of the plan ("make up 100-200 brand-new facts; if the model is still text-biased on them, that proves real bias") has a logical gap the advisor caught: brand-new audio concepts are *by definition* low-typicality, so H1 and H2 predict the **same observable outcome** on behavioral data alone. The fix (built into the design below) is to measure **audio-only confidence** as a baseline *before* introducing conflict, so H1 and H2 make different, separable predictions.

---

## 3. Experimental design

### 3.1 Data

- **100–200 fully fictitious entity/fact samples** — invented entities with invented binary-attribute facts (e.g. day/night, high-pitched/low-pitched, color A/color B), none of which exist in the real world. This eliminates confound #9 (contamination) by construction and sets confound #8 (typicality) to a known, controlled baseline of zero for every sample.
- **Generation pipeline** (adapted from MCR-BENCH's faithful/adversarial/irrelevant synthesis, since MCR-BENCH is the only one of the three core papers with a fully explicit, reusable data-synthesis process):

| MCR-BENCH step | Original input | Adaptation for fictitious-knowledge data |
| -------------- | -------------- | ---------------------------------------- |
| Faithful       |      GPT-4o converts a real caption into a true statement          |         No real caption exists - invent the fact first, then synthesize audio (TTS) that matches it                                 |
|      Adversarial          |   GPT-4o generates a contradicting statement alongside the true one    |    Same mechanism, applied to the invented fact (flip one attribute)              |
|      Irrelevant          | Retrieve a low-similarity real caption from AudioCaps  |  No real corpus to retrieve from - reuse a different fictitious fact from the same pool as the "irrelevant" distractor |


- **Positive control**: a small set (a few dozen) of real, familiar audio-text pairs run through the identical pipeline/metrics — not used to test H1/H2/H3, only to confirm the measurement pipeline reproduces the known bias pattern (high TIR) before trusting results on fictitious data.
- Primary scope is **speech** (TTS-read factual sentences) — this matches Paper 2's forced-choice paradigm directly and allows tight control over the counterfactual. **Non-speech/paralinguistic** fictitious sounds (e.g. a synthesized "creature call," high vs. low pitch) are a high-priority extension if time allows, but carry more technical risk (need a text-to-audio pipeline plus acoustic-feature consistency checks).

### 3.2 Conditions

Three conditions over the same underlying sample, forced-choice A/B format (mirrors Paper 2, keeps the logit comparison clean):

1. **Audio-only** — just the fictitious audio, no text.
2. **Text-only** — just the fictitious text, no audio.
3. **Conflicting** — both, contradicting each other — tested in **both directions** (audio-correct/text-wrong and audio-wrong/text-correct) to avoid one-directional bias in the design itself.

### 3.3 Metrics

- **Accuracy** per condition (standard).
- **Uncertainty quantification (UQ)** — logit margin (`logit(A) − logit(B)`) and/or entropy over repeated self-consistency samples. This is the key addition over MCR-BENCH/ALME/Paper 2, which only record *which* answer the model gave, never *how confident* it was.
- Reusable from MCR-BENCH: Normalized Accuracy, Text Influence Rate (TIR), Modal Robustness Score (MRS) — computed on the new data for comparability with prior literature.

### 3.4 How this answers H1 / H2 / H3

- **H1 vs. H2**: if audio-only confidence is *high* but the model still confidently flips to text under conflict → H1 (real suppression bias). If audio-only confidence is already *low* before any conflict is introduced → H2 (representation/confidence confound, not true modality bias).
- **H3**: because facts are fully invented, there is no pre-existing "remembered" correct answer. If the model's answer in the *unconflicted* audio-only or text-only condition doesn't track the (randomized) planted ground truth at all, that's direct evidence of memory fallback rather than context-reading.

### 3.5 Models

Multiple open-weight Audio-LLMs, access permitting — primarily **Qwen2-Audio-7B-Instruct** (most used in the literature, easiest to compare against), with Ultravox as a secondary target if feasible.

### 3.6 Explicitly out of scope

- Full activation-patching / circuit-level causal analysis (stretch goal only — the 3-condition + UQ design above is intended to answer H1/H2/H3 without needing to open up model internals).
- Testing whether the "text circuit" found in prior work is architecture/language-universal — noted as an open question in the literature, not something this thesis has the resources to resolve (would need multi-model, multi-language activation patching).
- Paralinguistic/non-speech conflict — logged as a high-priority future extension, not core scope.

---

## 4. Current status (as of 2nd advisor meeting, 2026-10-02)

Literature review and experimental design are settled (see `report/` for the full back-and-forth with the advisor). **The pilot code is written (Section 5) but has not been run on the model yet.** The immediate next step is a small pilot, not the full 100–200 sample run — the goal is to have one real pilot result for the 2nd report, before the topic is finalized.

**Goal of the pilot:** answer, cheaply, "does the text-bias effect still show up on fully clean (contamination-free) data?" — either answer is a reportable, thesis-justifying result:
- **Yes** → strong enough evidence to finalize the topic and ask the advisor to approve the full-scale run.
- **No** → also a reportable finding (suggests the previously reported bias comes from familiarity/contamination, not a true modality preference).

Planned pilot stages (see `report/Task.md` for the detailed Vietnamese checklist):

1. **Infra** — pick one model (Qwen2-Audio-7B-Instruct), set up a GPU environment (Colab/Kaggle if no local GPU — check the required VRAM first), pick a free TTS tool (`edge-tts` or `gTTS`), and — highest technical risk — confirm whether the model exposes per-choice logits/logprobs at all. If not, fall back to self-consistency sampling (run N times, count A/B picks) instead of logit-margin UQ. Confirm this *before* building out the data generation, so the UQ method is locked in early.
2. **Data (10–20 samples, not 100–200)** — hand-write or LLM-assist 10–20 fictitious entity+fact pairs, each a simple binary attribute; randomize which modality carries the correct answer; synthesize audio via TTS; fix a forced-choice A/B JSON prompt template (borrowing Paper 2's structure).
3. **Run & measure** — all 3 conditions over the pilot set; record accuracy + confidence per condition; run the positive control (5–10 real, familiar audio-text pairs) through the same pipeline to confirm it reproduces the known bias pattern before trusting pilot results on invented data.
4. **Read results & report** — answer the main question (does text bias appear on the fictitious data, and how strongly compared with the positive control?); if there is enough data, do a rough H1-vs-H2 split by audio-only confidence (no firm conclusion needed — only show that the signal exists and is measurable); write a short pilot summary + one results table, used to get advisor sign-off to scale to the full 100–200 sample benchmark.

If time-constrained, the floor is: finish infra + data generation, and get **at least the conflicting condition** running on a handful of samples — proving the pipeline works end-to-end matters more than full coverage at this stage.

---

## 5. Repository structure

```
Thesis/
├── README.md                  # this file
├── requirements.txt
├── configs/
│   └── pilot.yaml              # model, dtype/quantization, TTS voice, prompt variant, conditions, UQ method
├── report/                     # advisor-meeting notes
│   ├── TOPIC.md                #   topic statement + per-paper notes
│   ├── Literature Review.md    #   literature scan (threads 1–6)
│   ├── Report 1st Meeting.md   #   gap analysis, proposed ideas, advisor feedback, design
│   ├── Report Summary.md       #   condensed final design (H1/H2/H3, data, metrics)
│   └── Task.md                 #   pilot task checklist (current work)
├── Paper/                      # reference PDFs
│
├── data/
│   ├── fictitious_facts/facts.json   # invented entity + binary-attribute facts (source of truth)
│   ├── positive_control/facts.json   # 8 real, familiar facts in the same schema (hand-written)
│   ├── audio/                        # <fact_id>_planted.mp3 / <fact_id>_flipped.mp3 (TTS)
│   └── generated/stimuli.jsonl       # one row per fact x condition: audio, text, question, A/B options
│
├── src/
│   ├── common.py               # config + JSON/JSONL helpers
│   ├── data_gen/
│   │   ├── fact_synthesis.py   # invent entities/facts, balanced random planted value
│   │   ├── text_variants.py    # faithful/adversarial/irrelevant + baselines -> stimuli.jsonl
│   │   └── tts.py              # statements -> audio via edge-tts (gTTS fallback)
│   ├── eval/
│   │   ├── prompts.py          # Paper 2 strict-JSON A/B prompt (+ MCR-BENCH disclaimer variant)
│   │   ├── logits.py           # Qwen2-Audio loading, P(A)/P(B) readout, self-consistency fallback
│   │   ├── run_model.py        # runs stimuli, writes predictions.jsonl (resumable), --mock for testing
│   │   └── metrics.py          # accuracy, confidence, text-follow, Δ(T−A), MCR-BENCH Norm/TIR/MRS
│   └── analysis/
│       └── h1_h2_h3.py         # per-fact H1/H2/H3 buckets, chance test, input-tracking, scatter plot
│
├── notebooks/                  # exploratory analysis, pilot result write-ups
└── results/<run>/              # predictions.jsonl, metrics.json, summary.md, h_buckets.json, figure
```

### Conditions (one stimulus per fact per condition)

Every fact has a randomly **planted** value and its **flipped** counterpart; each is recorded as its own audio clip.

| Condition | Audio says | Text says | Expected answer | Used for |
|---|---|---|---|---|
| `audio_only` | planted | – | planted | audio-side confidence (H1 vs H2); MCR-BENCH neutral |
| `audio_only_flip` | flipped | – | flipped | does the answer follow the audio when it changes? (H3) |
| `text_only` | – | planted | planted | text-side confidence |
| `text_only_flip` | – | flipped | flipped | does the answer follow the text when it changes? (H3) |
| `faithful` | planted | planted | planted | MCR-BENCH faithful |
| `conflict_audio_correct` | planted | flipped | planted | MCR-BENCH adversarial; H1 test |
| `conflict_text_correct` | flipped | planted | planted | reverse conflict direction |
| `irrelevant` | planted | another fact | planted | MCR-BENCH irrelevant |

The A/B position of the two options is randomized per stimulus. The prompt is Paper 2's (Appendix, Fig. 6): the assistant turn is pre-filled with `{"answer":"`, so the next token is the letter and its logits give P(A) vs P(B). In Qwen2-Audio's tokenizer, `A` and `B` are single tokens (32, 33). Each prediction row also records `mass_ab`, the share of the full-vocabulary probability on A+B, and the free-form greedy answer. These two values show whether the logit readout matches what the model would actually say.

---

## 6. Quick start

Steps 1–3 run on a CPU (step 3 needs internet). Step 4 needs a GPU: use Kaggle with **GPU T4 x2** and Internet on. `notebooks/kaggle_smoke_test.ipynb` clones this repo from GitHub, picks fp16 or 4-bit for the GPU it gets, and runs the smoke test. Run everything from the repo root.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. invent facts (skim the names: drop any that resemble a real word)
python -m src.data_gen.fact_synthesis --n 15 --out data/fictitious_facts/facts.json

# 2. build stimuli for the fictitious facts + positive control
python -m src.data_gen.text_variants

# 3. synthesize audio (skips clips that already exist)
python -m src.data_gen.tts

# 4. run the model (GPU). Start with a smoke test, then the full pilot.
python -m src.eval.run_model --out results/smoke --limit 2
python -m src.eval.run_model --out results/pilot_run_1
#    --uq self_consistency   fallback if mass_ab is low / logits unusable
#    --conditions conflict_audio_correct conflict_text_correct   time-constrained minimum
#    --mock                  random answers, tests the pipeline without a GPU

# 5. H1/H2/H3 analysis (metrics.json + summary.md are written automatically by step 4)
python -m src.analysis.h1_h2_h3 --results results/pilot_run_1
```

---

## 7. Reference papers (in `Paper/`)

**Core (directly extended by this thesis):**
- `2508.15407` — MCR-BENCH: When Audio and Text Disagree
- `2602.11488` — ALME: When Audio-LLMs Don't Listen (cross-linguistic)
- `2606.18924` — Who Wins the Conflict? (mechanistic interpretability, activation patching)
- `2606.05161` — Beyond Text Following (GACL, counterfactual repair)
- `2603.06854` — Are Audio-Language Models Listening? (audio-specialist heads)

**Related / contextual:**
- `2604.16902` — Beyond Text-Dominance (omni-modal: audio loses to both text *and* vision)
- `2506.07233` — Audio-Aware Decoding (inference-time mitigation via contrastive decoding)
- `2605.20266` — Survey of Large Audio Language Models
- `2407.10759`, `2310.13289` — Qwen2-Audio / SALMONN technical reports (target models)
- `2402.07729`, `2406.16020`, `2508.21376`, `2025.emnlp-main.1303`, `2510.04584` — general audio benchmark/robustness background
- `2601.05201`, `2604.21911`, `2505.00557`, `2604.18803`, `2607.02853`, `2207.11850` — analogous prompt-override/hallucination literature from the vision-language side (methodological template; the audio field is ~5–8 years behind vision here)
- `2411.04077`, `1612.00837`, `1712.00377` — VQA language-prior-bias foundations (VQA-CP, GVQA, H-POPE) — origin of the probing-based design pattern being adapted here

Full paper-by-paper notes, limitations, and the complete gap analysis are in `report/Report 1st Meeting.md` and `report/Literature Review.md`; the condensed final design is in `report/Report Summary.md`.