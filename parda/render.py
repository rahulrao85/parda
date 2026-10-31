"""Render the clean video: an opaque cover over every tracked secret, a bleep over every spoken one.

Covers are painted only from the pixels AROUND a secret, never from the secret itself, so nothing can be recovered
from them (blur and pixelation can be partly reversed). Styles:
  mask   only the value is replaced by ****** with a small "hidden by Parda"; the document stays readable (default)
  smart  text becomes password dots, one row per line; ID documents become a card with a lock
  erase  the secret is painted out in the surrounding colour, as if it was never there
  pill   a rounded "private" chip with a lock
  box    a flat patch with a blue edge and "hidden by Parda"
"""
import subprocess
from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw, ImageFont

STYLES = ("mask", "smart", "erase", "pill", "box")
EDGE = np.array([37, 99, 235], np.uint8)
DOC_MIN_H = 70  # taller than a line of text: treat as a document or panel, not a single value


def _font(size: int):
    try:
        return ImageFont.truetype("arialbd.ttf" if size >= 18 else "arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


@lru_cache(maxsize=16)
def _stamp(text: str, size: int, lock: bool = True) -> np.ndarray:
    """Text (with a small padlock in front) as a 0..1 alpha mask, rendered once per text and size."""
    font = _font(size)
    tw = int(ImageDraw.Draw(Image.new("L", (1, 1))).textlength(text, font=font))
    lw = int(size * 0.9) if lock else 0
    img = Image.new("L", (tw + lw + (size // 3 if lock else 0) + 2, int(size * 1.4)), 0)
    d = ImageDraw.Draw(img)
    if lock:  # shackle arc over a rounded body
        bx, by, bw, bh = 1, int(size * 0.55), lw - 2, int(size * 0.7)
        d.arc([bx + bw * 0.18, by - bh * 0.75, bx + bw * 0.82, by + bh * 0.35], 180, 360, fill=255,
              width=max(2, size // 8))
        d.rounded_rectangle([bx, by, bx + bw, by + bh], radius=max(2, size // 6), fill=255)
    d.text((lw + (size // 3 if lock else 0), img.height / 2), text, font=font, fill=255, anchor="lm")
    return np.asarray(img, np.float32)[..., None] / 255


@lru_cache(maxsize=64)
def _ink(text: str, size: int, mono: bool = False) -> np.ndarray:
    """Text as a 0..1 alpha mask cropped tight to its ink, so asterisks (which sit high in most fonts) centre."""
    try:
        font = ImageFont.truetype("consolab.ttf" if mono else "arial.ttf", size)
    except OSError:
        font = ImageFont.load_default()
    img = Image.new("L", (size * (len(text) + 2), size * 2), 0)
    ImageDraw.Draw(img).text((size // 2, size // 2), text, font=font, fill=255)
    box = img.getbbox() or (0, 0, 1, 1)
    return np.asarray(img.crop(box), np.float32)[..., None] / 255


def _mask_cover(rgb, x1, y1, x2, y2, base, dark) -> None:
    """****** where the value was, then 'hidden by Parda' in the space left over, both centred on the line."""
    bw, bh = x2 - x1, y2 - y1
    tint = np.array([37, 99, 235], np.float32)
    rgb[y1:y2, x1:x2] = (base * 0.9 + tint * 0.1).astype(np.uint8)  # a faint blue wash marks the field
    size = int(np.clip(bh * 0.75, 10, 44))
    stars_ink = (226, 232, 240) if dark else (51, 65, 85)
    label = _ink("hidden by Parda", int(np.clip(size * 0.45, 9, 16)))
    n = int(np.clip(round(bw * 0.5 / (size * 0.55)), 3, 14))
    while n >= 3:
        stars = _ink("*" * n, size, mono=True)
        room = bw - 12 - stars.shape[1]
        if room >= 0:
            break
        n -= 1
    else:
        return
    x = x1 + 6
    _blend(rgb, y1 + (bh - stars.shape[0]) // 2, x, stars, stars_ink)
    if room >= label.shape[1] + 10 and bh >= label.shape[0] + 2:
        _blend(rgb, y1 + (bh - label.shape[0]) // 2, x + stars.shape[1] + 10, label,
               (147, 197, 253) if dark else (37, 99, 235))


@lru_cache(maxsize=64)
def _disc(r: int) -> np.ndarray:
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return (x * x + y * y <= r * r)


@lru_cache(maxsize=256)
def _rounded(w: int, h: int, r: int) -> np.ndarray:
    img = Image.new("L", (w, h), 0)
    ImageDraw.Draw(img).rounded_rectangle([0, 0, w - 1, h - 1], radius=r, fill=255)
    return np.asarray(img) > 127


def _blend(rgb, y, x, alpha, ink) -> None:
    h, w = alpha.shape[:2]
    h, w = min(h, rgb.shape[0] - y), min(w, rgb.shape[1] - x)
    if h <= 0 or w <= 0:
        return
    region = rgb[y:y + h, x:x + w].astype(np.float32)
    rgb[y:y + h, x:x + w] = (region * (1 - alpha[:h, :w]) + np.asarray(ink, np.float32) * alpha[:h, :w]).astype(np.uint8)


def frame_boxes(items: list[dict], tracks: dict, info: dict) -> dict[int, list[list[int]]]:
    """Every box to cover, by source frame: SAM's track AND Spark's own sightings. During a zoom, scroll or window
    jump SAM can sit on the wrong line for a moment, so a sighting is always covered too; a few extra pixels of
    patch cost nothing, a leak costs everything."""
    fps, hold = info["fps"], 0.5 / info.get("rate", 2)
    out: dict[int, list] = {}
    for it in items:
        if it["where"] != "screen" or it.get("decision") == "rejected":
            continue
        kind = it.get("kind", "other")
        for n, box in tracks.get(it["id"], {}).items():
            out.setdefault(n, []).append([*box[:4], kind])
        for s in it.get("seen", []):
            for n in range(max(0, round((s["t"] - hold) * fps)), round((s["t"] + hold) * fps) + 1):
                out.setdefault(n, []).append([*s["box"][:4], kind])
    return out


def _merge_boxes(boxes: list) -> list:
    """One cover per secret: SAM's box and Spark's box for the same thing overlap, so draw their union once."""
    out = [list(b) for b in boxes]
    changed = True
    while changed:
        changed = False
        for i in range(len(out)):
            for j in range(i + 1, len(out)):
                a, b = out[i], out[j]
                ix = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
                small = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
                if small > 0 and ix / small > 0.3:
                    kinds = {a[4] if len(a) > 4 else "other", b[4] if len(b) > 4 else "other"}
                    out[i] = [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]),
                              "id_document" if "id_document" in kinds else kinds.pop()]
                    del out[j]
                    changed = True
                    break
            if changed:
                break
    return out


def _cover(rgb: np.ndarray, boxes: list, pad: int, style: str = "mask") -> None:
    h, w, _ = rgb.shape
    src = rgb.copy()  # colours come from the untouched frame, not from a cover drawn a moment ago
    for box in _merge_boxes(boxes):
        x1, y1 = max(0, box[0] - pad), max(0, box[1] - pad)
        x2, y2 = min(w, box[2] + pad), min(h, box[3] + pad)
        bw, bh = x2 - x1, y2 - y1
        if bw < 4 or bh < 4:
            continue
        ring = np.concatenate([src[y1, x1:x2], src[y2 - 1, x1:x2], src[y1:y2, x1], src[y1:y2, x2 - 1]])
        base = np.median(ring, axis=0)
        dark = base.mean() < 128
        ink = (203, 213, 225) if dark else (71, 85, 105)
        rgb[y1:y2, x1:x2] = base.astype(np.uint8)  # every style starts by painting the secret out
        if style == "erase":
            continue
        if style == "mask":
            _mask_cover(rgb, x1, y1, x2, y2, base, dark)
            continue
        if style == "box":
            rgb[y1:y2, x1:x2] = np.clip(base + (22 if dark else -22), 0, 255).astype(np.uint8)
            rgb[y1:y1 + 2, x1:x2] = rgb[y2 - 2:y2, x1:x2] = EDGE
            rgb[y1:y2, x1:x1 + 2] = rgb[y1:y2, x2 - 2:x2] = EDGE
            lab = _stamp("hidden by Parda", 15, lock=False)
            if bw >= lab.shape[1] + 16 and bh >= lab.shape[0] + 2:
                _blend(rgb, (y1 + y2 - lab.shape[0]) // 2, x1 + 8, lab, ink)
            continue
        kind = box[4] if len(box) > 4 else "other"
        if style == "pill" or (style == "smart" and kind == "id_document" and bh >= DOC_MIN_H):
            tint = np.array([238, 242, 255] if not dark else [30, 41, 59], np.float32)
            card = (base * 0.35 + tint * 0.65) if style == "smart" else tint
            mask = _rounded(bw, bh, min(bh // 2 if style == "pill" else 14, 40))
            reg = rgb[y1:y2, x1:x2]
            reg[mask] = card.astype(np.uint8)
            edge = mask & ~np.pad(mask[2:-2, 2:-2], 2) if bw > 4 and bh > 4 else mask
            reg[edge] = (EDGE * 0.7 + base * 0.3).astype(np.uint8)
            label = "Private document" if style == "smart" else "private"
            size = int(np.clip(bh * (0.12 if style == "smart" else 0.55), 11, 28))
            lab = _stamp(label, size)
            if bw >= lab.shape[1] + 12 and bh >= lab.shape[0] + 2:
                _blend(rgb, (y1 + y2 - lab.shape[0]) // 2, (x1 + x2 - lab.shape[1]) // 2, lab, EDGE if not dark
                       else (147, 197, 253))
            continue
        # smart, text: password dots, one row per line of text in the box
        line = max(18, round(30 * w / 1920))
        rows = max(1, round(bh / line)) if bh > 1.6 * line else 1
        r = int(np.clip(bh / rows * 0.17, 2, 9))
        disc, step = _disc(r), r * 3
        for k in range(rows):
            cy = y1 + round((k + 0.5) * bh / rows)
            for cx in range(x1 + 2 * r, x2 - r, step):
                reg = rgb[cy - r:cy + r + 1, cx - r:cx + r + 1]
                if reg.shape[:2] == disc.shape:
                    reg[disc] = ink


def _bleep_filter(spans: list[tuple[float, float]]) -> str:
    on = "+".join(f"between(t,{a:.2f},{b:.2f})" for a, b in spans)
    return (f"[1:a]volume=0:enable='{on}'[m];"
            f"sine=frequency=1000:sample_rate=48000,volume=0.2,volume=0:enable='not({on})'[s];"
            "[m][s]amix=inputs=2:duration=first:normalize=0[a]")


def render(clip, out, items: list[dict], tracks: dict, info: dict, style: str = "mask") -> dict:
    W, H, fps = info["w"], info["h"], info["fps"]
    boxes = frame_boxes(items, tracks, info)
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
    cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(out)]
    enc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    dec = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-i", str(clip), "-f", "rawvideo", "-pix_fmt", "rgb24",
                            "-"], stdout=subprocess.PIPE)
    pad, size, n, covered = max(6, W // 160), W * H * 3, 0, 0
    while (chunk := dec.stdout.read(size)) and len(chunk) == size:
        rgb = np.frombuffer(chunk, np.uint8).reshape(H, W, 3)
        if boxes.get(n):
            rgb = rgb.copy()
            _cover(rgb, boxes[n], pad, style)
        covered += bool(boxes.get(n))
        enc.stdin.write(rgb.tobytes())
        n += 1
    dec.wait()
    enc.stdin.close()
    enc.wait()
    return {"style": style, "frames": n, "frames_covered": covered, "bleeps": [[round(a, 2), round(b, 2)] for a, b in spans]}
