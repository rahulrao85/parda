"""The baseline Parda is measured against: a regex sweep over OCR, written fresh for the benchmark.

This is the shape of what I used before the hackathon (that code is not in this repo): OCR a frame twice a second,
match each line against a handful of patterns (PAN, AWS key, sk- key, email, IFSC, a run of ten or more digits),
black out the whole line, hold the boxes until the next OCR'd frame. No model decides anything, nothing follows
motion between samples, speech is not touched. It uses Parda's own OCR (RapidOCR) so the comparison is about the
method, not the OCR engine.

    python scripts/baseline_regex.py clip.mp4 out/clip/baseline.mp4
"""
import re, subprocess, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from parda import sample  # noqa: E402
from parda.locate import Ocr  # noqa: E402

PATTERNS = [re.compile(p) for p in (
    r"\b[A-Z]{5}[0-9]{4}[A-Z]\b",          # PAN
    r"AKIA[0-9A-Z]{16}",                   # AWS access key id
    r"sk-[A-Za-z0-9_-]{16,}",              # OpenAI-style key
    r"[\w.+-]+@[\w-]+\.[\w.]+",            # email
    r"\b[A-Z]{4}0[A-Z0-9]{6}\b",           # IFSC
    r"(?:\d[ -]?){10,}",                   # phone / account / Aadhaar / card: ten or more digits
)]
RATE = 2.0  # OCR'd frames per second


def sweep(clip: Path, out: Path) -> dict:
    info = sample.probe(clip)
    fps, W, H = info["fps"], info["w"], info["h"]
    ocr = Ocr(Path("cache/ocr"))
    step = max(1, round(fps / RATE))
    dec = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-i", str(clip), "-fps_mode", "cfr", "-r", f"{fps}",
                            "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
    enc = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                            "-r", f"{fps}", "-i", "-", "-i", str(clip), "-map", "0:v", "-map", "1:a?", "-c:v", "libx264",
                            "-crf", "18", "-pix_fmt", "yuv420p", "-c:a", "copy", "-shortest", str(out)], stdin=subprocess.PIPE)
    boxes, n, covered, hits = [], 0, 0, 0
    size = W * H * 3
    while True:
        buf = dec.stdout.read(size)
        if len(buf) < size:
            break
        rgb = np.frombuffer(buf, np.uint8).reshape(H, W, 3)
        if n % step == 0:
            boxes = [ln["box"] for ln in ocr.frame(rgb) if any(p.search(ln["text"]) for p in PATTERNS)]
            hits += len(boxes)
        if boxes:
            rgb = rgb.copy()
            for x1, y1, x2, y2 in boxes:
                rgb[int(y1):int(y2) + 1, int(x1):int(x2) + 1] = 0
            covered += 1
        enc.stdin.write(rgb.tobytes())
        n += 1
    dec.stdout.close()
    enc.stdin.close()
    enc.wait()
    return {"frames": n, "frames_covered": covered, "ocr_frames": (n + step - 1) // step, "line_hits": hits,
            "ocr_seconds": round(ocr.seconds, 1)}


if __name__ == "__main__":
    clip, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.parent.mkdir(parents=True, exist_ok=True)
    print(sweep(clip, out))
