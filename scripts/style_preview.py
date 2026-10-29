"""Show every cover style on the same frames, side by side, without rendering a whole video.

    python scripts/style_preview.py <clip.mp4> <out dir with flags.json + tracks.json> <seconds> [...] -o preview.png
"""
import argparse, json, subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from parda import render


def frame_at(clip: Path, t: float, w: int, h: int) -> np.ndarray:
    raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-ss", str(t), "-i", str(clip), "-frames:v", "1", "-f",
                          "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(h, w, 3)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("clip", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("times", type=float, nargs="+")
    p.add_argument("-o", "--output", type=Path, default=Path("style_preview.png"))
    p.add_argument("--crop", type=int, nargs=4, metavar=("X", "Y", "W", "H"))
    a = p.parse_args()
    flags = json.loads((a.out / "flags.json").read_text(encoding="utf-8"))
    tracks = {k: {int(f): b for f, b in v.items()} for k, v in json.loads((a.out / "tracks.json").read_text()).items()}
    info = flags["video"]
    boxes = render.frame_boxes(flags["items"], tracks, info)
    pad = max(6, info["w"] // 160)
    rows = []
    for t in a.times:
        src = frame_at(a.clip, t, info["w"], info["h"])
        cells = []
        for style in ("original",) + render.STYLES:
            img = src.copy()
            if style != "original":
                render._cover(img, boxes.get(round(t * info["fps"]), []), pad, style)
            if a.crop:
                x, y, w, h = a.crop
                img = img[y:y + h, x:x + w]
            pil = Image.fromarray(img)
            pil.thumbnail((640, 640))
            d = ImageDraw.Draw(pil)
            d.rectangle([0, 0, 140, 26], fill=(15, 23, 42))
            d.text((8, 5), style, fill=(255, 255, 255), font=render._font(15))
            cells.append(pil)
        row = Image.new("RGB", (sum(c.width for c in cells) + 8 * (len(cells) - 1), cells[0].height), "white")
        x = 0
        for c in cells:
            row.paste(c, (x, 0))
            x += c.width + 8
        rows.append(row)
    sheet = Image.new("RGB", (max(r.width for r in rows), sum(r.height for r in rows) + 8 * (len(rows) - 1)), "white")
    y = 0
    for r in rows:
        sheet.paste(r, (0, y))
        y += r.height + 8
    sheet.save(a.output)
    print(a.output)


if __name__ == "__main__":
    main()
