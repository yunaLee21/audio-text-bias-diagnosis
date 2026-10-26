# Diagnosing Text Bias in Audio-Language Models with a Contamination-Free Benchmark

> *Phân tích nguyên nhân thiên vị văn bản trong mô hình Audio-Ngôn ngữ qua benchmark không nhiễu dữ liệu huấn luyện*

When an Audio-Language Model (Audio-LLM) receives audio together with text that contradicts it, it usually answers according to the **text**. This project asks **why**. The text bias could be a genuine preference for one modality, or it could come from two confounds that existing benchmarks don't control: how *familiar* the audio concept is, and whether the test sample *leaked into pretraining*.

To remove both confounds by construction, the benchmark uses **fully fictitious facts**: invented creatures with invented attributes, read aloud with text-to-speech. It then uses each sample's **single-modality confidence** to tell genuine suppression apart from weak audio understanding and from answering out of memory.

**Status:** the pilot pipeline is implemented (data generation → evaluation → analysis). Smoke tests on Qwen2-Audio-7B-Instruct confirmed that the logit readout is reliable, and showed that under conflict the A/B answer is driven mostly by option position. An order-free Yes/No readout is now the primary measure. Next: the full pilot and the complexity dataset.

---

## Contents

- [Background](#background)
- [Hypotheses](#hypotheses)
- [Benchmark design](#benchmark-design)
- [Repository structure](#repository-structure)
- [Getting started](#getting-started)
- [Scope](#scope)
- [References](#references)

---

## Background

### Text dominates audio — this is established

- **MCR-BENCH** [1] pairs audio with faithful, adversarial and irrelevant text descriptions. Under adversarial text, accuracy collapses: on audio QA, Qwen-Audio-Chat drops from 87.8% to 1.7%. The Text Influence Rate exceeds 95% for most models. A probe on the hidden states still separates faithful from adversarial pairs with ~98% accuracy, so the model registers the conflict internally, but that awareness never reaches its answer.
- **ALME** [2] extends this to 57,602 stimuli in 8 languages. Models follow misleading text 10–26× more often than in a text-vs-text control, even when told to trust the audio (GPT-4o 23.2%, Gemini 2.0 Flash 16.6%).

### It looks like arbitration, not perception

| Evidence | Paper | Finding |
|---|---|---|
| Activation patching | *Who Wins the Conflict?* [3] | A localized "text circuit" suppresses the audio. Ablating it raises audio accuracy (Qwen: 33% → 82%). |
| Counterfactual repair | *Beyond Text Following* [4] | Removing the misleading text fixes 64.1% of wrong answers, so the audio-grounded answer was already present. |
| Attention-head steering | *Are Audio-Language Models Listening?* [5] | Amplifying a few "audio-specialist" heads gives +8.0 points on MMAU without retraining. |

All three interventions work at inference time only. This suggests the audio is represented correctly and then overridden when the answer is chosen.

### What existing benchmarks don't control

1. **Concept typicality.** A dog bark carries a strong pretraining prior; a rare sound does not. No benchmark stratifies by typicality, so an aggregate "TIR > 95%" can hide very different behaviour for familiar and unfamiliar concepts. On the vision side, the model's base confidence correlates with resistance to prompt override (ρ = 0.37–0.46) [6].
2. **Training-data contamination.** Stimuli from Common Voice or AudioCaps were probably seen during pretraining. ALME acknowledges this and argues that relative comparisons between models remain fair, but absolute numbers do not.

These confounds are distinct. A benchmark can use brand-new clips (no contamination) and still use a very familiar concept such as a dog bark.

Two further questions are open:
- **Context vs. memory:** is the model reading the current input at all, or answering from its parametric memory? This is well studied for text-only LLMs, with entity-substitution benchmarks such as NQ-Swap, ConFiQA and Faithfulness-QA. It has not been studied for audio-text conflict.
- **Confidence as a baseline:** benchmarks report *which* modality wins. MCR-BENCH also reports aggregate confidence, finding that models stay confident under adversarial text. No benchmark uses a sample's **single-modality confidence** as a baseline for interpreting what happens under conflict.

---

## Hypotheses

| | Audio-only baseline | Under conflict | Interpretation |
|---|---|---|---|
| **H1: Suppression** | confident and correct | still confidently follows the text | genuine modality-level bias at the decision stage |
| **H2: Weak representation** | low confidence, but the answer still tracks the input | follows the text | the audio is under-represented because it is unfamiliar; not a modality bias |
| **H3: Memory fallback** | the answer does not track the input at all | — | the model is not reading the input |

Novel audio concepts are by definition unfamiliar, so H1 and H2 predict the same *behaviour* under conflict. Measuring audio-only confidence first is what separates them. Samples are classified **per sample, on the same dataset**. The share of samples in each group shows how much each mechanism contributes.

---

## Benchmark design

### Data

- **Fictitious facts** (`data/fictitious_facts/facts.json`): an invented entity name (e.g. *Pemoux*) plus one of 12 binary attributes (activity, call pitch, fur color, habitat, diet, climate, social life, legs, size, breeding season, body covering, hearts).
  - Which value is "true" (the **planted** value) is randomized, balanced 50/50 across facts. A model that ignores the input therefore cannot score above chance, and that is what separates H2 from H3.
- **Positive control** (`data/positive_control/facts.json`): 8 everyday statements whose answer world knowledge does not decide, in the style of *Who Wins the Conflict?* [3]. They cover its four swap types (adjective, negation, number, time), e.g. *"Closed / Open stations are shown in normal font."* Because the same model and prompt are used, the control checks that the setup reproduces that paper's known text bias before the results on fictitious data are trusted.
  - Real-world facts (e.g. *"The capital of France is Paris"*) are deliberately avoided: there the model follows world knowledge, so they measure input-vs-memory rather than audio-vs-text.
- **Audio:** one clip per fact per value (`<fact_id>_planted.mp3`, `<fact_id>_flipped.mp3`), synthesized with `edge-tts` (`gTTS` as fallback).
- **Text variants:** adapted from MCR-BENCH's faithful / adversarial / irrelevant synthesis. Facts are invented rather than taken from real captions. The adversarial text flips the attribute, and the irrelevant text is a different fact from the same pool.

### Complexity dataset

A separate dataset (`data/complexity/`) presents 12 further fictitious facts (one per attribute) at four sentence-difficulty levels. Its purpose is to lower audio-side confidence in controlled steps. With simple sentences, audio-only confidence sits at the ceiling, which leaves H1 vs. H2 untestable.

| Level | Change | Example |
|---|---|---|
| L0 | baseline | *The Planai has six legs.* |
| L1 | extra information about the entity | *The Planai, which eats only fruit, has six legs and has a high-pitched call.* |
| L2 | answer deep inside the sentence | *Researchers who spent several years observing animals in the remote valleys have reported that the Planai, which few people have ever seen, has six legs.* |
| L3 | lure: both values appear | *Unlike its close relative the Plodedun, which has four legs, the Planai has six legs.* |

- **Minimal pairs:** at every level, the planted and flipped sentences differ only in the tested value. At L3 the two values swap between the entity and its relative.
- **Fixed per fact:** fillers, frames and lures are chosen once per fact, so a fact's levels differ only in structure.
- **Predictions:** H2 predicts the text-follow rate rises as audio-only confidence falls across levels. H1 predicts it is already high at L0.
- **Comparing levels:** a hard sentence is harder to hear than to read, so text-follow is compared with the audio-only and text-only baselines *of the same level*.
- **Clip length:** clips are 3–11 s, well under the ~30 s that Qwen2-Audio’s Whisper-based audio encoder takes in one window.

### Conditions

One item per fact per condition. The two conflict conditions are run in **both A/B orders**, and their probabilities are averaged per option value, so a preference for a letter position cancels out (`order-flip` reports how often the order alone changed the answer):

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
| `no_input` | – | – | – | question only: each value's prior likelihood (calibrates the Yes/No readout) |

### Prompts and readouts

Both readouts use the strict-JSON prompt from *Who Wins the Conflict?* [3], with the assistant turn pre-filled with `{"answer":"` so that the next token is the answer. Its logits give the answer probabilities.

- **Yes/No (primary):** one question per answer value, e.g. *"Does the Pemoux have six legs?"* and *"Does the Pemoux have four legs?"*.
  - There is no option list, so option order cannot affect the answer.
  - A value's score is log-odds(Yes) = logit(Yes) − logit(No). The model's answer is the value with the higher score, so a general yes-bias applies to both values and cancels (as in POPE-style polling).
  - `Yes` and `No` are single tokens in Qwen2-Audio's tokenizer.
- **A/B (secondary):** forced choice between two listed options, as in [3], so results stay comparable with that paper.
  - The A/B position is randomized per item, and conflict items are run in both orders.
  - `A` and `B` are single tokens.
- **Why both:** the smoke tests showed that under conflict the A/B answer is driven mostly by option position (the model picks the second option), not by modality.
- **Sanity checks:** each prediction records `mass_ab` / `mass_yn` (the share of full-vocabulary probability on A+B or Yes+No) and, for A/B, the model's free-form greedy answer.
- **Self-consistency (fallback):** sample *n* free-form answers and use the share of the majority letter as the confidence.
- **Prompt variant:** `mcr` adds MCR-BENCH's "may be accurate, inaccurate, or irrelevant" disclaimer. The input order (`text_first` / `audio_first`) is configurable.

### Metrics

- **Per condition:**
  - accuracy
  - mean confidence
  - P(expected answer)
  - text-follow rate
  - order-flip rate and share of "B" answers (A/B position-bias diagnostics)
- **Conflict:**
  - text-follow rate over both conflict directions (0.5 = balanced)
  - Δ(T−A) = text-side accuracy − audio-side accuracy
- **Content vs. position** (per conflict item; > 0 means the audio value wins):
  - A/B: with m_A, m_B = logit(audio value) − logit(text value) when the audio value sits in slot A / slot B,
    content = (m_A + m_B) / 2 and position = (m_B − m_A) / 2 (> 0 = slot B favoured)
  - Yes/No: content = log-odds(Yes | audio value) − log-odds(Yes | text value), optionally calibrated by subtracting the same difference in the `no_input` item. Averaged over both conflict directions the value prior cancels by design, so calibration only changes per-item results.
- **MCR-BENCH** [1], with `audio_only` as the neutral condition:
  - Normalized accuracy Norm_t = C_t / C_neu
  - Macro accuracy
  - TIR_fth = Δic / N, TIR_adv = Δci / N, TIR_irr = (Δic + Δci) / N
  - MRS = 0.8 · Acc_adv/Acc_neu + 0.2 · Acc_irr/Acc_neu
- **H1/H2/H3 analysis:**
  - per-fact buckets, using audio-only P(planted) against a threshold τ (default 0.8)
  - audio-only / text-only accuracy vs. chance (binomial test)
  - input-tracking rate (answer flips when the stated value flips)
  - A/B position bias
  - a scatter plot of audio-only confidence vs. the conflict outcome
  - `--readout yes_no` (default) or `--readout ab`
- **Complexity sweep:** per level, audio-only / text-only confidence, input tracking and conflict text-follow; Spearman ρ between audio-only confidence and the conflict outcome across all (fact, level) items

---

## Repository structure

```
.
├── configs/
│   └── pilot.yaml              # model, dtype/quantization, TTS, prompt variant, conditions, UQ method
├── data/
│   ├── fictitious_facts/       # facts.json: invented facts (source of truth)
│   ├── positive_control/       # facts.json: knowledge-neutral control statements
│   ├── audio/                  # TTS clips (main pilot)
│   ├── generated/              # stimuli.jsonl: one row per fact x condition (x A/B order)
│   └── complexity/             # facts.json, audio/, stimuli.jsonl: the complexity dataset
├── notebooks/
│   ├── smoke_test.ipynb        # GPU smoke test (local Jupyter)
│   └── pilot.ipynb             # full pilot + complexity run (local Jupyter)
├── results/                    # one folder per run (created by run_model)
├── src/
│   ├── common.py               # config + JSON/JSONL helpers
│   ├── data_gen/
│   │   ├── fact_synthesis.py   # invent entities/facts, balanced planted values
│   │   ├── complexity.py       # complexity dataset (levels L0-L3)
│   │   ├── yes_no.py           # Yes/No question per answer value
│   │   ├── text_variants.py    # facts -> stimuli.jsonl
│   │   └── tts.py              # statements -> audio
│   ├── eval/
│   │   ├── prompts.py          # A/B and Yes/No prompts
│   │   ├── logits.py           # model loading, A/B and Yes/No readouts, self-consistency
│   │   ├── run_model.py        # run stimuli -> predictions.jsonl (resumable)
│   │   └── metrics.py          # both readouts, content/position split, metrics.json + summary.md
│   └── analysis/
│       ├── h1_h2_h3.py         # H1/H2/H3 buckets, chance tests, plot
│       └── complexity_sweep.py # per-level sweep, confidence-vs-conflict correlation
└── requirements.txt
```

---

## Getting started

### 1. Local setup and data generation (CPU)

Run all commands from the repository root. The repo already contains both datasets, so this step is only needed to change or regenerate them:
- main pilot: 15 fictitious + 8 control facts, 46 clips, 253 items
- complexity dataset: 12 facts × 4 levels, 96 clips, 528 items

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python -m src.data_gen.fact_synthesis --n 15   # invent facts (skim names; drop any that resemble real words)
python -m src.data_gen.text_variants           # build stimuli for fictitious + positive-control facts
python -m src.data_gen.tts                     # synthesize audio (needs internet; skips existing clips)

# complexity dataset
python -m src.data_gen.complexity --n 12
python -m src.data_gen.text_variants --facts data/complexity/facts.json \
    --audio-dir data/complexity/audio --out data/complexity/stimuli.jsonl
python -m src.data_gen.tts --facts data/complexity/facts.json --out data/complexity/audio
```

To test the pipeline without a GPU, `--mock` replaces the model with random answers:

```bash
python -m src.eval.run_model --mock --out results/mock
```

### 2. Run in Jupyter (GPU)

1. On a machine with a CUDA GPU, install the requirements and start Jupyter from the repo: `jupyter lab`.
2. Open `notebooks/smoke_test.ipynb` and **Run All**. The first run downloads the model (~15 GB) into the Hugging Face cache (`~/.cache/huggingface`, or `HF_HOME`).

What the notebook does:
- moves the kernel to the repo root
- picks fp16 (≥ 20 GB total VRAM) or 4-bit (a single 16 GB GPU) and writes `configs/local.yaml` (git-ignored)
- runs 2 fictitious + 2 control facts × 9 conditions (44 items)
- reports whether both readouts are reliable (`mass_ab`, `greedy_agrees` and `mass_yn` all ≥ 0.9)

After the smoke test passes, run `notebooks/pilot.ipynb` the same way. It runs both datasets and their analyses:

```bash
python -m src.eval.run_model --config configs/local.yaml --out results/pilot_run_1
python -m src.analysis.h1_h2_h3 --results results/pilot_run_1 --kind fictitious   # add --readout ab for A/B

python -m src.eval.run_model --config configs/local.yaml \
    --stimuli data/complexity/stimuli.jsonl --out results/complexity_run_1
python -m src.analysis.complexity_sweep --results results/complexity_run_1        # add --readout ab for A/B
```

Useful flags for `run_model`:

| Flag | Effect |
|---|---|
| `--limit N` | only the first N facts of each kind |
| `--conditions ...` | run a subset of conditions |
| `--uq self_consistency` | sampling-based confidence instead of logits |
| `--kinds fictitious` | only these dataset kinds (e.g. skip the control) |
| `--stimuli PATH` | which stimuli file to run (default: main pilot) |
| `--mock` | random answers, no model |

Runs are resumable: finished stimuli are skipped when the same command is re-run.

### Outputs (`results/<run>/`)

| File | Content |
|---|---|
| `predictions.jsonl` | one row per stimulus (and A/B order): choice, P(A)/P(B), confidence, margin, `mass_ab`, greedy answer |
| `metrics.json`, `summary.md` | per-condition and MCR-BENCH metrics for both readouts, plus the content/position table |
| `h_buckets_<kind>_<readout>.json` | per-fact H1/H2/H3 buckets and group-level tests |
| `audio_conf_vs_conflict_<kind>_<readout>.png` | audio-only confidence vs. conflict outcome |
| `complexity_sweep_<readout>.{json,md,png}` | per-level sweep (complexity runs) |
| `config_used.json` | exact configuration of the run |

---

## Scope

- **Core scope:** speech, i.e. spoken factual sentences. This matches the forced-choice paradigm of [3] and gives tight control over the counterfactual.
- **Planned extension:** non-speech fictitious sounds, such as a synthesized creature call that is high- vs. low-pitched. This needs a text-to-audio pipeline and acoustic consistency checks.
- **Models:** Qwen2-Audio-7B-Instruct first. Ultravox is a secondary target.
- **Out of scope:**
  - full activation-patching / circuit-level analysis (stretch goal only)
  - testing whether the text circuit is universal across architectures and languages

---

## References

**Core**

1. *When Audio and Text Disagree: Revealing Text Bias in Large Audio-Language Models* (MCR-BENCH), EMNLP 2025 — [arXiv:2508.15407](https://arxiv.org/abs/2508.15407)
2. *When Audio-LLMs Don't Listen: A Cross-Linguistic Study of Modality Arbitration* (ALME) — [arXiv:2602.11488](https://arxiv.org/abs/2602.11488)
3. *Who Wins the Conflict? Mechanistic Interpretability of Text Bias in Audio LLMs* — [arXiv:2606.18924](https://arxiv.org/abs/2606.18924)
4. *Beyond Text Following: Repairable Arbitration Reversals in Audio-Language Models* — [arXiv:2606.05161](https://arxiv.org/abs/2606.05161)
5. *Are Audio-Language Models Listening? Audio-Specialist Heads for Adaptive Audio Steering* — [arXiv:2603.06854](https://arxiv.org/abs/2603.06854)
6. *Mechanisms of Prompt-Induced Hallucination in Vision-Language Models* — [arXiv:2601.05201](https://arxiv.org/abs/2601.05201)

**Related**

- Omni-modal preference: *Beyond Text-Dominance* — [arXiv:2604.16902](https://arxiv.org/abs/2604.16902)
- Mitigation: *Reducing Object Hallucination via Audio-Aware Decoding* — [arXiv:2506.07233](https://arxiv.org/abs/2506.07233)
- Survey: *A Survey of Large Audio Language Models* — [arXiv:2605.20266](https://arxiv.org/abs/2605.20266)
- Target models: Qwen2-Audio — [arXiv:2407.10759](https://arxiv.org/abs/2407.10759); SALMONN — [arXiv:2310.13289](https://arxiv.org/abs/2310.13289)
- Audio benchmarks and robustness:
  - AIR-Bench — [arXiv:2402.07729](https://arxiv.org/abs/2402.07729)
  - AudioBench — [arXiv:2406.16020](https://arxiv.org/abs/2406.16020)
  - AHELM — [arXiv:2508.21376](https://arxiv.org/abs/2508.21376)
  - HalluAudio — [arXiv:2604.19300](https://arxiv.org/abs/2604.19300)
  - *Can Large Audio-Language Models Truly Hear?* — [arXiv:2410.16130](https://arxiv.org/abs/2410.16130)
  - RSA-Bench — [arXiv:2601.10384](https://arxiv.org/abs/2601.10384)
  - *Evaluating Robustness of Large Audio Language Models* — [EMNLP 2025](https://aclanthology.org/2025.emnlp-main.1303.pdf)
  - *Robustness Assessment of Large Audio Language Models* — [arXiv:2510.04584](https://arxiv.org/abs/2510.04584)
- Vision-language analogues (prompt override and language priors):
  - HalluScope — [arXiv:2604.21911](https://arxiv.org/abs/2604.21911)
  - [arXiv:2505.00557](https://arxiv.org/abs/2505.00557)
  - [arXiv:2604.18803](https://arxiv.org/abs/2604.18803)
  - [arXiv:2607.02853](https://arxiv.org/abs/2607.02853)
  - [arXiv:2207.11850](https://arxiv.org/abs/2207.11850)
  - H-POPE — [arXiv:2411.04077](https://arxiv.org/abs/2411.04077)
  - VQA v2 — [arXiv:1612.00837](https://arxiv.org/abs/1612.00837)
  - GVQA — [arXiv:1712.00377](https://arxiv.org/abs/1712.00377)
