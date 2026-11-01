"""Render the clean video: every tracked secret typed over as asterisks, every spoken one bleeped.

A text value's pixels are replaced by its line's own paper colour, then one row of asterisks is drawn in the line's
own ink colour at the line's own size, so the file looks as if it was typed that way: no tint, box, edge or label.
A photo or other non-text item becomes a flat patch in the surrounding colour. Colours are read only from pixels
OUTSIDE the box (paper) or as a single median (ink), never blurred or pixelated, so nothing can be recovered.
"""
import subprocess
from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw, ImageFont

NON_TEXT = {"photo_of_person", "face", "image"}
PAD = 1            # px added around every exact box
RING = 2           # px of surrounding colour sampled for the paper
INSET = 3          # px a non-text patch stays inside its box, so a photo's frame line survives
MIN_CONTRAST = 25  # luminance levels between ink and paper below which the ink guess is not trusted


def _is_text(kind: str, w: int, h: int) -> bool:
    if kind in NON_TEXT:
        return False
    return not (kind == "other" and h > 70 and w > 150)


@lru_cache(maxsize=8)
def _font(size: int):
    for name in ("consola.ttf", "DejaVuSansMono.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


@lru_cache(maxsize=256)
def _stars(n: int, size: int) -> np.ndarray:
    """n asterisks as a 0..1 alpha mask cropped tight to their ink (asterisks sit high in a font; this centres them)."""
    img = Image.new("L", (size * (n + 2), size * 2), 0)
    ImageDraw.Draw(img).text((size // 2, size // 2), "*" * n, font=_font(size), fill=255)
    return np.asarray(img.crop(img.getbbox() or (0, 0, 1, 1)), np.float32)[..., None] / 255


def _strip(width: int, height: int) -> np.ndarray | None:
    """The widest row of asterisks (at least 3) that fits `width`, glyphs ~0.9 x the line height."""
    size = int(np.clip(round(height * 0.9), 8, 48))
    while True:
        one = _stars(1, size).shape[1]
        adv = _stars(2, size).shape[1] - one or max(1, size // 2)
        n = max(3, (width - one) // adv + 1)
        while n > 3 and _stars(n, size).shape[1] > width:
            n -= 1
        s = _stars(n, size)
        if s.shape[1] <= width or size <= 8:
            return s[:, :width] if s.shape[1] > width else s
        size -= 1


def _lum(px: np.ndarray) -> np.ndarray:
    return px[..., 0] * 0.299 + px[..., 1] * 0.587 + px[..., 2] * 0.114


def _ink_and_paper(src: np.ndarray, x1: int, y1: int, x2: int, y2: int):
    """paper = median of the ring just outside the box; ink = median of the 10% of box pixels farthest from it."""
    H, W, _ = src.shape
    rx1, ry1, rx2, ry2 = max(0, x1 - RING), max(0, y1 - RING), min(W, x2 + RING), min(H, y2 + RING)
    outer = src[ry1:ry2, rx1:rx2].reshape(-1, 3)
    keep = np.ones((ry2 - ry1, rx2 - rx1), bool)
    keep[y1 - ry1:y2 - ry1, x1 - rx1:x2 - rx1] = False
    ring = outer[keep.ravel()]
    inside = src[y1:y2, x1:x2].reshape(-1, 3).astype(np.float32)
    paper = np.median(ring if len(ring) else inside, axis=0)
    pl = float(_lum(paper))
    d = np.abs(_lum(inside) - pl)
    k = max(1, len(d) // 10)
    ink = np.median(inside[np.argpartition(d, -k)[-k:]], axis=0)
    if abs(float(_lum(ink)) - pl) < MIN_CONTRAST:
        ink = np.array([200, 200, 200] if pl < 128 else [60, 60, 60], np.float32)
    return paper.astype(np.float32), ink.astype(np.float32)


def _merge_boxes(boxes: list) -> list:
    """One cover per secret: overlapping boxes of the same class (text / non-text) become their union."""
    out = [list(b) for b in boxes]
    changed = True
    while changed:
        changed = False
        for i in range(len(out)):
            for j in range(i + 1, len(out)):
                a, b = out[i], out[j]
                if (a[4] in NON_TEXT) != (b[4] in NON_TEXT):
                    continue
                ix = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
                small = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
                if small > 0 and ix / small > 0.3:
                    out[i] = [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]), a[4]]
                    del out[j]
                    changed = True
                    break
            if changed:
                break
    return out


def frame_boxes(items: list[dict], tracks: dict) -> dict[int, list]:
    """{frame: [[x1, y1, x2, y2, kind], ...]} from the exact tracked boxes only."""
    kinds = {it["id"]: it.get("kind", "other") for it in items
             if it.get("where") == "screen" and it.get("decision") != "rejected"}
    out: dict[int, list] = {}
    for iid, per in tracks.items():
        if iid not in kinds:
            continue
        for n, box in per.items():
            out.setdefault(int(n), []).append([*(int(round(v)) for v in box[:4]), kinds[iid]])
    return out


def cover(rgb: np.ndarray, boxes: list) -> None:
    """Paint every box in place. Colours come from the untouched frame, not from a cover drawn a moment ago."""
    H, W, _ = rgb.shape
    src = rgb.copy()
    for b in _merge_boxes(boxes):
        x1, y1 = max(0, b[0] - PAD), max(0, b[1] - PAD)
        x2, y2 = min(W, b[2] + PAD), min(H, b[3] + PAD)
        bw, bh = x2 - x1, y2 - y1
        if bw < 2 or bh < 2:
            continue
        if not _is_text(b[4], bw, bh):  # a photo: SAM's box includes the frame around it; keep that frame
            x1, y1, x2, y2 = x1 + INSET, y1 + INSET, x2 - INSET, y2 - INSET
            if x2 - x1 < 2 or y2 - y1 < 2:
                continue
            paper, _ = _ink_and_paper(src, x1, y1, x2, y2)
            rgb[y1:y2, x1:x2] = paper.astype(np.uint8)
            continue
        paper, ink = _ink_and_paper(src, x1, y1, x2, y2)
        rgb[y1:y2, x1:x2] = paper.astype(np.uint8)
        if bw < 6:
            continue
        a = _strip(bw - 2, bh)
        ah, aw = a.shape[:2]
        oy, ox = max(0, y1 + (bh - ah) // 2), x1 + 2
        h, w = min(ah, H - oy), min(aw, W - ox)
        reg = rgb[oy:oy + h, ox:ox + w].astype(np.float32)
        rgb[oy:oy + h, ox:ox + w] = (reg * (1 - a[:h, :w]) + ink * a[:h, :w]).astype(np.uint8)


def _bleep_filter(spans: list[tuple[float, float]]) -> str:
    on = "+".join(f"between(t,{a:.2f},{b:.2f})" for a, b in spans)
    return (f"[1:a]volume=0:enable='{on}'[m];"
            f"sine=frequency=1000:sample_rate=48000,volume=0.2,volume=0:enable='not({on})'[s];"
            "[m][s]amix=inputs=2:duration=first:normalize=0[a]")


def render(clip, out, items: list[dict], tracks: dict, info: dict) -> dict:
    """tracks = {item_id: {frame: [x1, y1, x2, y2]}}: exact value boxes in source pixels. Only those are drawn."""
    W, H, fps = info["w"], info["h"], info["fps"]
    boxes = frame_boxes(items, tracks)
    spans = [(it["start_s"], it["end_s"]) for it in items
             if it["where"] == "speech" and it.get("decision") != "rejected"]
    has_audio = bool(subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries",
                                     "stream=index", "-of", "csv=p=0", str(clip)], capture_output=True,
                                    text=True).stdout.strip())
    cmd = ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
           "-r", f"{fps}", "-i", "-", "-i", str(clip)]
    if has_audio and spans:
        cmd += ["-filter_complex", _bleep_filter(spans), "-map", "0:v", "-map", "[a]"]
    elif has_audio:
        cmd += ["-map", "0:v", "-map", "1:a"]
    cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest",
            str(out)]
    enc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    dec = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-i", str(clip), "-fps_mode", "cfr", "-r", f"{fps}",
                            "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)  # same clock as locate.py
    size, n, covered = W * H * 3, 0, 0
    while (chunk := dec.stdout.read(size)) and len(chunk) == size:
        rgb = np.frombuffer(chunk, np.uint8).reshape(H, W, 3)
        if boxes.get(n):
            rgb = rgb.copy()
            cover(rgb, boxes[n])
            covered += 1
        enc.stdin.write(rgb.tobytes())
        n += 1
    dec.wait()
    enc.stdin.close()
    enc.wait()
    return {"frames": n, "frames_covered": covered, "bleeps": [[round(a, 2), round(b, 2)] for a, b in spans]}
