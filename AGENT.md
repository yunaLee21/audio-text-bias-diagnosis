# AGENT.md — working on this project

Guide for coding agents. Read `README.md` first for the research design (hypotheses H1/H2/H3, conditions, readouts, metrics); this file covers **how to work** in the repo.

## Environments

| Machine | Role | Notes |
|---|---|---|
| Dev machine (WSL, no GPU) | edit code, generate data, commit | test the pipeline with `--mock`; never claim a GPU result from here |
| Jupyter server (`~/audio-text-bias-diagnosis`, conda env `tlenv-thesis`) | runs the model on GPU | no `nano`, no write access to `~/.zshrc`; edit files with JupyterLab (**File → Open from Path…**), `%%writefile`, `vi` or a heredoc |

- Code moves between machines through GitHub only: commit on the dev machine → user pushes → `git pull` on the server.
- Kaggle/Colab are no longer used. Do not reintroduce Kaggle-specific code (`/kaggle/working`, `kaggle_secrets`, repo cloning, output zips).

## Rule 1 — main experiments run as `.py` scripts, notebooks only test the workflow

- **Main experiments** (full pilot, complexity run, any new full-scale run) are launched from a terminal as `python -m src.<module> ...`, not from a notebook cell. Progress is tracked by the script itself:
  - stdout prints `[i/N] stim_id: ...` per stimulus — send it to a log file inside the run folder;
  - `results/<run>/predictions.jsonl` is appended and flushed one row at a time, so it is the ground truth for what has finished;
  - re-running the **same command** resumes: finished `stim_id`s are skipped.

  ```bash
  mkdir -p results/pilot_run_1
  nohup python -m src.eval.run_model --config configs/local.yaml --out results/pilot_run_1 \
      > results/pilot_run_1/run.log 2>&1 &
  tail -f results/pilot_run_1/run.log          # follow progress
  wc -l results/pilot_run_1/predictions.jsonl  # rows finished
  ```
- **Notebooks** (`notebooks/`) are only for checking that the workflow runs: setup/GPU check, small runs with `--limit`, `--mock`, listening to a clip, inspecting a few predictions. Keep them short; a notebook cell must not be the only way to reproduce a result.
- New experiment logic goes into a module under `src/` with an `argparse` CLI and a usage docstring (see `src/eval/run_model.py`), never only into a notebook.
- Analyses are scripts too (`src/analysis/*.py`) that read a finished `results/<run>/` and write their JSON/MD/PNG into the same folder.

## Rule 2 — settings live in config files, overridden with `--flags`

- All experiment settings (model, dtype/quantization, TTS, prompt variant/order, conditions, UQ method, readouts, seed) live in `configs/*.yaml`. `configs/pilot.yaml` is the canonical, committed config.
- Choose what to run from the command line: `--config`, `--out`, `--stimuli`, `--conditions`, `--kinds`, `--limit`, `--uq`, `--mock`. CLI flags override the YAML.
- Do not hard-code settings in code or edit them inside notebooks. A new setting gets a key in `configs/pilot.yaml` (with a short comment) and, if it is varied per run, a matching `--flag`.
- Machine-specific values go in `configs/local.yaml` (git-ignored), a copy of `pilot.yaml` with `model.dtype` / `model.quantization` for that GPU (the notebook setup cell writes it). ≥ 20 GB total VRAM → fp16, a single 16 GB GPU → `4bit`; bf16 only on Ampere or newer.
- Every run saves its exact settings to `results/<run>/config_used.json`; keep it that way for any new runner.

## Running

- Always run from the repo root as modules: `python -m src.eval.run_model ...` (not `python src/eval/run_model.py`). Paths in configs and flags are relative to the repo root (`src.common.resolve`).
- In notebooks, call scripts with `!{sys.executable} -m src....` so they use the kernel's environment; the setup cell moves the kernel to the repo root.
- One folder per run in `results/`; give a new experiment a new `--out` name instead of overwriting. Never delete a results folder without asking.
- Order of work for a new model/config: `--mock` on the dev machine → smoke test (`notebooks/smoke_test.ipynb`, `--limit 2`; `mass_ab`, `greedy_agrees`, `mass_yn` ≥ 0.9) → full run as a script.
- If the logit readout fails the smoke test, use `--uq self_consistency`.

## Secrets

- `HF_TOKEN` lives in `.env` in the repo root (git-ignored). The notebook setup cell loads it; for scripts, export it in the shell (`set -a; source .env; set +a`) or rely on `hf auth login`.
- Never print, log or commit the token. Never `git add .` / `git commit -a` on the server.

## Data

- Source of truth: `data/fictitious_facts/facts.json`, `data/positive_control/facts.json`, `data/complexity/facts.json`. Stimuli (`stimuli.jsonl`) and audio are generated from them with `src/data_gen/*` — do not hand-edit generated files; change the generator and regenerate.
- Keep the design invariants: planted values balanced 50/50, minimal pairs (planted vs flipped differ only in the tested value), conflict items in both A/B orders, `no_input` baseline, control facts knowledge-neutral (no real-world facts).
- Yes/No is the primary readout; A/B is secondary (comparability with Paper 2 and position-bias diagnostics).

## Code conventions

- Match the existing style: small modules, short docstrings with usage examples, comments only where the reason is not obvious, helpers from `src/common.py` for config and JSON/JSONL I/O.
- Keep compatibility with both older and newer `transformers` (the server has 5.x): e.g. `dtype` vs `torch_dtype`, `audio` vs `audios` in `src/eval/logits.py`.
- After editing a notebook, check it still validates (`nbformat.validate`) and its code compiles. Do not commit notebook outputs.
- When the pipeline, flags, outputs or file layout change, update `README.md` in the same commit.

## Git

- Branch `main`. Commit only when the user asks; the user pushes.
- Git-ignored, never commit: `Paper/`, `report/`, `results/`, `.env`, `configs/local.yaml`, `.ipynb_checkpoints/`, `.venv/`.
- On the server, a run notebook is modified by its outputs; before `git pull`, clear outputs or `git restore notebooks/<name>.ipynb`.
