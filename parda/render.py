"""Render the clean video: an opaque patch over every tracked secret, a bleep over every spoken one.

Patches are solid, never blur or pixelation: pixelated text can be reversed with known tools, a solid patch cannot.
They stay visible (a shade off the surrounding colour, a thin edge, a label) so a viewer knows something was hidden.
"""
import subprocess

import numpy as np
from PIL import Image, ImageDraw, ImageFont

EDGE = np.array([37, 99, 235], np.uint8)


def _label() -> np.ndarray:
    """'hidden by Parda' as a 0..1 alpha mask, rendered once and stamped into every patch big enough to hold it."""
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except OSError:
        font = ImageFont.load_default()
    img = Image.new("L", (124, 20), 0)
    ImageDraw.Draw(img).text((0, 1), "hidden by Parda", font=font, fill=255)
    return np.asarray(img, np.float32)[..., None] / 255


LABEL = _label()


def frame_boxes(items: list[dict], tracks: dict, info: dict) -> dict[int, list[list[int]]]:
    """Every box to cover, by source frame. SAM's track first; Spark's own sightings fill any frame SAM missed."""
    fps, hold = info["fps"], 0.5 / info.get("rate", 2)
    out: dict[int, list] = {}
    for it in items:
        if it["where"] != "screen" or it.get("decision") == "rejected":
            continue
        tr = tracks.get(it["id"], {})
        for n, box in tr.items():
            out.setdefault(n, []).append(box)
        for s in it.get("seen", []):
            for n in range(round((s["t"] - hold) * fps), round((s["t"] + hold) * fps) + 1):
                if n >= 0 and n not in tr:
                    out.setdefault(n, []).append(s["box"])
    return out


def _cover(rgb: np.ndarray, boxes: list, pad: int) -> None:
    h, w, _ = rgb.shape
    src = rgb.copy()  # colours come from the untouched frame, not from a patch drawn a moment ago
    for box in boxes:
        x1, y1 = max(0, box[0] - pad), max(0, box[1] - pad)
        x2, y2 = min(w, box[2] + pad), min(h, box[3] + pad)
        if x2 - x1 < 4 or y2 - y1 < 4:
            continue
        ring = np.concatenate([src[y1, x1:x2], src[y2 - 1, x1:x2], src[y1:y2, x1], src[y1:y2, x2 - 1]])
        base = np.median(ring, axis=0)
        dark = base.mean() < 128
        rgb[y1:y2, x1:x2] = np.clip(base + (22 if dark else -22), 0, 255).astype(np.uint8)
        rgb[y1:y1 + 2, x1:x2] = rgb[y2 - 2:y2, x1:x2] = EDGE
        rgb[y1:y2, x1:x1 + 2] = rgb[y1:y2, x2 - 2:x2] = EDGE
        lh, lw, _ = LABEL.shape
        if x2 - x1 >= lw + 16 and y2 - y1 >= lh + 4:
            ly, lx = (y1 + y2 - lh) // 2, x1 + 8
            ink = np.array([226, 232, 240] if dark else [51, 65, 85], np.float32)
            region = rgb[ly:ly + lh, lx:lx + lw].astype(np.float32)
            rgb[ly:ly + lh, lx:lx + lw] = (region * (1 - LABEL) + ink * LABEL).astype(np.uint8)


def _bleep_filter(spans: list[tuple[float, float]]) -> str:
    on = "+".join(f"between(t,{a:.2f},{b:.2f})" for a, b in spans)
    return (f"[1:a]volume=0:enable='{on}'[m];"
            f"sine=frequency=1000:sample_rate=48000,volume=0.2,volume=0:enable='not({on})'[s];"
            "[m][s]amix=inputs=2:duration=first:normalize=0[a]")


def render(clip, out, items: list[dict], tracks: dict, info: dict) -> dict:
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
            _cover(rgb, boxes[n], pad)
        covered += bool(boxes.get(n))
        enc.stdin.write(rgb.tobytes())
        n += 1
    dec.wait()
    enc.stdin.close()
    enc.wait()
    return {"frames": n, "frames_covered": covered, "bleeps": [[round(a, 2), round(b, 2)] for a, b in spans]}
