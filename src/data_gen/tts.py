"""Synthesize one clip per (fact, value): <fact_id>_planted.mp3 and <fact_id>_flipped.mp3.

    python -m src.data_gen.tts \
        --facts data/fictitious_facts/facts.json data/positive_control/facts.json \
        --out data/audio

edge-tts (default) needs internet access; gtts is the fallback. Existing files are skipped
unless --overwrite is given, so re-running is cheap.
"""

import argparse
import asyncio
from pathlib import Path

from src.common import audio_path, load_config, read_json, resolve


async def _edge_tts(text, path, voice):
    import edge_tts
    await edge_tts.Communicate(text, voice).save(str(path))


def synthesize(text, path, engine, voice, lang):
    if engine == "edge-tts":
        asyncio.run(_edge_tts(text, path, voice))
    elif engine == "gtts":
        from gtts import gTTS
        gTTS(text, lang=lang).save(str(path))
    else:
        raise ValueError(f"unknown TTS engine: {engine}")


def main():
    cfg = load_config()["tts"]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--facts", nargs="+",
                    default=["data/fictitious_facts/facts.json", "data/positive_control/facts.json"])
    ap.add_argument("--out", default="data/audio")
    ap.add_argument("--engine", default=cfg["engine"], choices=["edge-tts", "gtts"])
    ap.add_argument("--voice", default=cfg["voice"])
    ap.add_argument("--lang", default=cfg["lang"])
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    resolve(args.out).mkdir(parents=True, exist_ok=True)
    made = skipped = 0
    for path in args.facts:
        for fact in read_json(path):
            for key in ("planted", "flipped"):
                out = resolve(audio_path(args.out, fact["fact_id"], key))
                if out.exists() and not args.overwrite:
                    skipped += 1
                    continue
                synthesize(fact["statements"][fact[key]], out, args.engine, args.voice, args.lang)
                made += 1
    print(f"synthesized {made} clips, skipped {skipped} existing, in {args.out}")


if __name__ == "__main__":
    main()
