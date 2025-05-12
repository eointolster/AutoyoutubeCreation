#!/usr/bin/env python3
"""
combine_narrations.py  (v2 – with stretch/freeze rules)

See docstring in v1 for folder layout.  New behaviour:

• diff ≤ 4 s  → slow video with setpts
• diff >  4 s → pad last frame with tpad
"""
import os
import re
import subprocess
from pathlib import Path
from shlex import quote as shquote

# --------------------------------------------------------------------------- #
sound_dir   = Path("sound_outputs/individual_narrations")
video_dir   = Path("video_outputs/mp4_clips")
merged_dir  = Path("video_outputs/merged_clips")
final_file  = Path("video_outputs/final_compilation.mp4")
ffmpeg_exe  = "ffmpeg"
ffprobe_exe = "ffprobe"
audio_codec = "aac"      # leave as "aac" unless you _know_ WAVs are MP4-compatible
video_codec = "libx264"  # re-encode because we touch the video track with filters
                         # (use hevc_nvenc / h264_nvenc, etc. if you prefer GPU encode)

wav_pat  = re.compile(r"clip_(\d{4})_narration\.wav$")
mp4_pat  = re.compile(r"narrativegen_clip_(\d{4})__\d+\.mp4$")

# --------------------------------------------------------------------------- #
def duration(path: Path) -> float:
    """Return media duration in seconds using ffprobe."""
    cmd = [
        ffprobe_exe,
        "-v", "error",
        "-select_streams", "v:a",     # any stream is fine
        "-show_entries", "format=duration",
        "-of", "csv=p=0",
        str(path)
    ]
    out = subprocess.check_output(cmd, text=True).strip()
    return float(out)

def index_from(fname: Path, pattern: re.Pattern) -> str|None:
    m = pattern.search(fname.name)
    return m.group(1) if m else None

def run(cmd: list[str]):
    """Run a subprocess and surface ffmpeg output if it fails."""
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as exc:
        print("FFmpeg failed!\n", exc.stdout.decode(errors="ignore"))
        raise

# --------------------------------------------------------------------------- #
def main():
    merged_dir.mkdir(parents=True, exist_ok=True)

    wavs = {index_from(p, wav_pat): p for p in sound_dir.glob("*.wav")}
    mp4s = {index_from(p, mp4_pat): p for p in video_dir.glob("*.mp4")}

    common = sorted(set(wavs) & set(mp4s))
    if not common:
        raise RuntimeError("No matching indices found!")

    concat_list_file = merged_dir / "concat.txt"
    merged_paths: list[Path] = []

    for idx in common:
        wav_path = wavs[idx]
        mp4_path = mp4s[idx]
        out_path = merged_dir / f"merged_{idx}.mp4"

        a_dur = duration(wav_path)
        v_dur = duration(mp4_path)
        diff  = a_dur - v_dur         # positive means audio is longer

        print(f"\n[{idx}] audio={a_dur:6.2f}s  video={v_dur:6.2f}s  diff={diff:+.2f}s")

        if diff > 0 and diff <= 4:
            # ---- Rule 1: tiny over-hang – slow down the video ----
            factor = a_dur / v_dur     # >1 → longer
            vf = f"setpts={factor}*PTS"
            print(f"→ Slowing video by factor {factor:.4f}")
        elif diff > 4:
            # ---- Rule 2: big over-hang – freeze last frame ----
            vf = f"tpad=stop_mode=clone:stop_duration={diff}"
            print(f"→ Freezing last frame for {diff:.2f}s")
        else:
            # audio fits or is shorter; no filter
            vf = "null"
            if diff < 0:
                print("→ Audio shorter than video – leaving gap (you can add silence if needed)")
            else:
                print("→ Perfect match – no adjustment")

        cmd = [
            ffmpeg_exe, "-y",
            "-i", str(mp4_path),
            "-i", str(wav_path),
            "-filter_complex", f"[0:v]{vf}[v]",
            "-map", "[v]", "-map", "1:a",
            "-c:v", video_codec,
            "-c:a", audio_codec,
            "-shortest",                 # trims if we made video _longer_ than audio
            str(out_path)
        ]
        run(cmd)
        merged_paths.append(out_path)

    # --------------------------------------------------------------------- #
    # Concatenate
    with concat_list_file.open("w", encoding="utf-8") as f:
        for p in merged_paths:
            # concat demuxer needs paths quoted or escaped
            f.write(f"file {shquote(str(p.resolve()))}\n")

    print(f"\n→ Concatenating {len(merged_paths)} clips")
    cmd_concat = [
        ffmpeg_exe, "-y",
        "-f", "concat", "-safe", "0",
        "-i", str(concat_list_file),
        "-c", "copy",
        str(final_file)
    ]
    run(cmd_concat)
    print(f"\n✅  Done – final movie: {final_file.resolve()}")

if __name__ == "__main__":
    main()
