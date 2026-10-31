"""Pick the frames worth showing a model: sample at a fixed rate, keep a frame only when something on screen changed.

A screen recording is mostly identical frames. Comparing small grayscale thumbnails block by block (not one global
hash) means a single new line of text in a terminal still counts as a change, while a static screen costs nothing.
"""
import json, subprocess
from dataclasses import dataclass

import numpy as np

THUMB_W, THUMB_H, BLOCK = 160, 90, 10


@dataclass
class Frame:
    t: float            # seconds from the start of the clip
    rgb: np.ndarray     # H x W x 3, at analysis size (see max_w)


def probe(clip) -> dict:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=width,height,avg_frame_rate:format=duration", "-of", "json", str(clip)],
                         capture_output=True, check=True, text=True).stdout
    j = json.loads(out)
    s, (num, den) = j["streams"][0], j["streams"][0]["avg_frame_rate"].split("/")
    return {"w": s["width"], "h": s["height"], "fps": float(num) / float(den or 1),
            "duration": float(j["format"]["duration"])}


def _thumb(rgb: np.ndarray) -> np.ndarray:
    h, w, _ = rgb.shape
    gray = rgb.mean(axis=2)
    ys = (np.arange(THUMB_H) * h // THUMB_H)
    xs = (np.arange(THUMB_W) * w // THUMB_W)
    return gray[np.ix_(ys, xs)]


def changed(a: np.ndarray, b: np.ndarray, thresh: float = 4.0) -> bool:
    """True if any BLOCK x BLOCK cell of the thumbnails differs by more than thresh grey levels on average."""
    d = np.abs(a - b).reshape(THUMB_H // BLOCK, BLOCK, THUMB_W // BLOCK, BLOCK).mean(axis=(1, 3))
    return bool(d.max() > thresh)


def sample(clip, rate: float = 2.0, max_w: int = 1280) -> tuple[list[Frame], dict]:
    """Frames at `rate` per second, downscaled to max_w, dropping any frame that matches the last kept one."""
    info = probe(clip)
    scale = min(1.0, max_w / info["w"])
    w, h = int(info["w"] * scale) // 2 * 2, int(info["h"] * scale) // 2 * 2
    # Every step-th decoded frame, by number, so frame k is exactly source frame k*step and its time is exact.
    # (ffmpeg's fps filter keeps the LAST frame of each slot, a quarter second late at 2/s: covers then landed
    # where moving text was about to be, not where it was.)
    step = max(1, round(info["fps"] / rate))
    proc = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-i", str(clip), "-vf",
                             f"select='not(mod(n\,{step}))',scale={w}:{h}", "-fps_mode", "passthrough",
                             "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
    kept, last, i, size = [], None, 0, w * h * 3
    while chunk := proc.stdout.read(size):
        if len(chunk) < size:
            break
        rgb = np.frombuffer(chunk, np.uint8).reshape(h, w, 3)
        th = _thumb(rgb)
        if last is None or changed(th, last):
            kept.append(Frame(round(i * step / info["fps"], 3), rgb))
            last = th
        i += 1
    proc.wait()
    info.update(analysis_w=w, analysis_h=h, sampled=i, kept=len(kept), rate=info["fps"] / step)
    return kept, info
