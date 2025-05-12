#!/usr/bin/env python3
"""
clone_batch_pad.py – Dia 1.6 B voice-cloning with tail-padding
--------------------------------------------------------------

* clones voice from  myexample.wav
* reads pregenerated_content.json
* forces newline after every line so EOS is explicit
* appends 300 ms of digital silence (24 kHz) to stop cutoff
* writes clip_XXXX_narration.wav under sound_outputs/individual_narrations
"""

import json, random, sys, traceback
from pathlib import Path

import numpy as np, torch
from dia.model import Dia                     # pip install dia-tts

# ── CONFIG ───────────────────────────────────────────────────────────────
DIA_MODEL      = "nari-labs/Dia-1.6B"
DTYPE          = "float16"
REF_WAV        = "myexample.wav"          # reference voice
REF_TRANSCRIPT = (
    "[S1] We pry open a rusty manhole, and billowing steam invites us downward like a theatrical curtain rising for adventure."
)
PAD_S   = 0.3                             # 300 ms of silence
SEED    = 42                              # None → non-deterministic

SAMPLE_RATE = 24_000                      # Dia’s fixed SR

# ── PATHS ────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parent
JSON_FILE = ROOT / "pregenerated_content.json"
OUT_DIR   = ROOT / "sound_outputs" / "individual_narrations"
REF_PATH  = ROOT / REF_WAV

# ── HELPERS ──────────────────────────────────────────────────────────────
def set_seed(s: int):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

def load_json(p: Path):
    try:
        dat = json.loads(p.read_text("utf-8")); assert isinstance(dat, list)
        return dat
    except Exception as e:
        sys.exit(f"❌  JSON load error: {e}")

# ── MAIN ─────────────────────────────────────────────────────────────────
def main():
    if not REF_PATH.exists(): sys.exit(f"❌  {REF_WAV} not found")
    if not JSON_FILE.exists(): sys.exit("❌  pregenerated_content.json missing")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    items = load_json(JSON_FILE)
    if SEED is not None: set_seed(SEED)

    dia = Dia.from_pretrained(DIA_MODEL, compute_dtype=DTYPE)
    pad = np.zeros(int(PAD_S * SAMPLE_RATE), dtype=np.float32)

    ok = 0
    for idx, itm in enumerate(items, 1):
        cid  = itm.get("id", idx)
        line = itm.get("commentary", "").rstrip()
        if not line: continue

        prompt = f"{REF_TRANSCRIPT}\n{line}\n"   # ensure newline EOS
        try:
            wav = dia.generate(
                prompt,
                audio_prompt=str(REF_PATH),
                verbose=False,
            )
            wav_out = np.concatenate([wav, pad])
            out_path = OUT_DIR / f"clip_{cid:04d}_narration.wav"
            dia.save_audio(str(out_path), wav_out)   # Dia defaults to 24 kHz
            ok += 1
            print(f"✓ {cid:04d}  → {out_path.name}")
        except Exception as e:
            print(f"⚠️  id {cid}: {e}"); traceback.print_exc()

    if torch.cuda.is_available(): torch.cuda.empty_cache()
    print(f"\n✔️  {ok}/{len(items)} clips generated with cloned voice")

if __name__ == "__main__":
    main()














