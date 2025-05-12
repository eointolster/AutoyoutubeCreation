

#!/usr/bin/env python3
"""
2JustAudio.py  –  generate one narration WAV per commentary line
================================================================
* Reads **pregenerated_content.json** in the current folder (expects a list of
  dicts that contain at least `id` and `commentary`).
* Uses **Dia 1.6 B** with its **default voice** – no cloning, no prompt prep.
  If the commentary string starts with speaker tags like `[S1]`, Dia will speak
  them as intended.
* Writes each file to  `example/sound_outputs/individual_narrations/`  with the
  simple name pattern  `clip_<ID‑4digits>_narration.wav` so they line up with
  your existing silent MP4 clips.

Usage
-----
```bash
python 2JustAudio.py        # assumes virtual‑env has  dia-tts  +  torch
```

If you run on CPU change `COMPUTE_DTYPE` to "auto" or "float32"; on GPU the
current `float16` setting is fastest.
"""

import json
import sys
import traceback
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parent
JSON_FILE   = BASE_DIR / "pregenerated_content.json"
AUDIO_DIR   = BASE_DIR / "sound_outputs" / "individual_narrations"

# ── Dia config ─────────────────────────────────────────────────────────────
DIA_MODEL       = "nari-labs/Dia-1.6B"
COMPUTE_DTYPE   = "float16"      # "auto" for CPU‑only boxes
USE_TORCH_COMPILE = False        # True ↦ Triton compile; leave False for stability

# ---------------------------------------------------------------------------
#  helper
# ---------------------------------------------------------------------------

def load_json(p: Path):
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        sys.exit(f"❌  Cannot read JSON '{p}': {e}")
    if not isinstance(data, list):
        sys.exit("❌  JSON should be a list of objects with 'commentary'.")
    return data

# ---------------------------------------------------------------------------
#  main
# ---------------------------------------------------------------------------

def main() -> None:
    print("\n=== 2JustAudio.py  – Dia batch narration ===")

    if not JSON_FILE.exists():
        sys.exit(f"❌  {JSON_FILE} not found – create it first")

    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    print(f"📂  Output DIR: {AUDIO_DIR}")

    items = load_json(JSON_FILE)
    print(f"📑  Loaded {len(items)} commentary rows")

    # Dia import (torch optional)
    try:
        from dia.model import Dia
    except ImportError:
        sys.exit("❌  dia-tts not installed →  pip install dia-tts")

    print(f"🔊  Loading Dia model '{DIA_MODEL}' …")
    dia = Dia.from_pretrained(DIA_MODEL, compute_dtype=COMPUTE_DTYPE)
    print("✅  Dia ready\n")

    ok = 0
    for idx, itm in enumerate(items, start=1):
        cid  = itm.get("id", idx)
        text = itm.get("commentary", "").strip()
        if not text:
            print(f"• skip (id {cid}) – empty text")
            continue

        file_name = f"clip_{cid:04d}_narration.wav"
        out_path  = AUDIO_DIR / file_name
        print(f"  → id {cid:04d}  → {file_name}")

        try:
            audio = dia.generate(
                text,
                use_torch_compile=USE_TORCH_COMPILE,
                verbose=False,
            )
            dia.save_audio(str(out_path), audio)
            ok += 1
        except Exception as e:
            print(f"    ERROR id {cid}: {e}")
            traceback.print_exc()

    # cleanup GPU mem if any
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass

    print(f"\n✔️  Done – {ok}/{len(items)} WAVs generated to {AUDIO_DIR}")


if __name__ == "__main__":
    main()
