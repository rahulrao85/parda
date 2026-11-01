"""Source vs covered frames side by side, using the boxes in tracks.json, without rendering a whole video.

    python scripts/style_preview.py <clip.mp4> <out dir with flags.json + tracks.json> t1 t2 ... -o preview.png
"""
import argparse, json, subprocess, sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from parda import render  # noqa: E402


def frame_at(clip: Path, t: float, w: int, h: int) -> np.ndarray:
    raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-ss", str(t), "-i", str(clip), "-frames:v", "1", "-f",
                          "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(h, w, 3).copy()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("clip", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("times", type=float, nargs="+")
    p.add_argument("-o", "--output", type=Path, default=Path("preview.png"))
    p.add_argument("--crop", type=int, nargs=4, metavar=("X", "Y", "W", "H"))
    p.add_argument("--width", type=int, default=900, help="width of each cell")
    a = p.parse_args()
    flags = json.loads((a.out / "flags.json").read_text(encoding="utf-8"))
    tracks = json.loads((a.out / "tracks.json").read_text(encoding="utf-8"))
    info = flags["video"]
    boxes = render.frame_boxes(flags["items"], tracks)
    rows = []
    for t in a.times:
        src = frame_at(a.clip, t, info["w"], info["h"])
        dst = src.copy()
        fb = boxes.get(round(t * info["fps"]), [])
        render.cover(dst, fb)
        cells = []
        for label, img in ((f"source {t}s", src), (f"covered ({len(fb)} boxes)", dst)):
            if a.crop:
                x, y, w, h = a.crop
                img = img[y:y + h, x:x + w]
            pil = Image.fromarray(img)
            pil.thumbnail((a.width, 10_000))
            ImageDraw.Draw(pil).rectangle([0, 0, 190, 20], fill=(15, 23, 42))
            ImageDraw.Draw(pil).text((6, 4), label, fill=(255, 255, 255))
            cells.append(pil)
        row = Image.new("RGB", (cells[0].width + cells[1].width + 8, max(c.height for c in cells)), "white")
        row.paste(cells[0], (0, 0))
        row.paste(cells[1], (cells[0].width + 8, 0))
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
