"""Shared helpers: config loading, JSON / JSONL I/O, repo paths."""

import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def load_config(path="configs/pilot.yaml"):
    with open(ROOT / path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve(path):
    """Paths in configs and CLI flags are relative to the repo root."""
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def read_json(path):
    with open(resolve(path), encoding="utf-8") as f:
        return json.load(f)


def write_json(obj, path):
    path = resolve(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def read_jsonl(path):
    with open(resolve(path), encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(rows, path):
    path = resolve(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def audio_path(audio_dir, fact_id, value_key):
    """One clip per (fact, stated value). value_key is 'planted' or 'flipped'."""
    return str(Path(audio_dir) / f"{fact_id}_{value_key}.mp3")
