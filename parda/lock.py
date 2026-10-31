"""Keep every cover on its secret in EVERY frame, and guard the moments when that is not possible.

Muse Spark sees a secret twice a second and SAM 3.1 follows it ten times a second; a fast scroll or drag moves
text further than that between looks. So between those keyframes Parda locks on with template matching: the
secret plus a little of its surroundings (its label, its line) is found again in each frame, which follows a
scroll or drag pixel for pixel. When the lock breaks (a zoom, a window snapping to full screen) the motion guard
frosts the whole corridor the secret can be in, from where it was last locked to where Spark next sees it.
"""
import bisect, subprocess
from collections import deque

import cv2
import numpy as np

SCALE = 0.5          # matching runs on half-size grey frames
LOCK_MIN = 0.72      # normalised correlation needed to trust a match
KEEP_MIN = 0.86      # stricter, to keep following after Spark's last sighting
GAP_S = 3.0          # sightings closer than this are one appearance; a lost lock between them is guarded
TAIL_S = 0.6         # guard this long after the last sighting when the lock is lost
PREROLL_S = 0.6      # also lock backwards: a secret can appear up to half a sampling step before it is seen
JUMP_FRAC = 0.06     # share of the screen that must change in one frame to count as a layout jump
SETTLE_FRAC = 0.004  # below this the screen has settled (a blinking cursor stays under it)


class _Lock:
    def __init__(self, gray: np.ndarray, box, kind: str):
        x1, y1, x2, y2 = box
        bh = max(4, y2 - y1)
        self.cx, self.cy = max(6, bh // 2), max(4, bh // 2)  # context around the value: its label, its line
        h, w = gray.shape
        tx1, ty1 = max(0, x1 - self.cx), max(0, y1 - self.cy)
        tx2, ty2 = min(w, x2 + self.cx), min(h, y2 + self.cy)
        self.tmpl = gray[ty1:ty2, tx1:tx2].copy()
        self.dx, self.dy = x1 - tx1, y1 - ty1  # where the box sits inside the template
        self.size = (x2 - x1, y2 - y1)
        self.box, self.kind = [x1, y1, x2, y2], kind
        self.flat = self.tmpl.size == 0 or float(self.tmpl.std()) < 3.0  # nothing to match on

    def match(self, gray: np.ndarray) -> float:
        """Find the template near its last position; move the box there. Returns the match score."""
        if self.flat:
            return 0.0
        th, tw = self.tmpl.shape
        m = max(24, 3 * self.size[1])  # how far it can move in one frame
        h, w = gray.shape
        x1, y1 = max(0, self.box[0] - self.dx - m), max(0, self.box[1] - self.dy - m)
        x2, y2 = min(w, self.box[0] - self.dx + tw + m), min(h, self.box[1] - self.dy + th + m)
        win = gray[y1:y2, x1:x2]
        if win.shape[0] < th or win.shape[1] < tw:
            return 0.0
        res = cv2.matchTemplate(win, self.tmpl, cv2.TM_CCOEFF_NORMED)
        _, score, _, (lx, ly) = cv2.minMaxLoc(res)
        if np.isfinite(score) and score >= LOCK_MIN:
            nx, ny = x1 + lx + self.dx, y1 + ly + self.dy
            self.box = [nx, ny, nx + self.size[0], ny + self.size[1]]
        return float(score) if np.isfinite(score) else 0.0


def _union(boxes, pad: int, w: int, h: int) -> list[int]:
    return [max(0, min(b[0] for b in boxes) - pad), max(0, min(b[1] for b in boxes) - pad),
            min(w, max(b[2] for b in boxes) + pad), min(h, max(b[3] for b in boxes) + pad)]


def lock(clip, items: list[dict], tracks: dict, info: dict) -> tuple[dict, dict]:
    """Per source frame: covers [x1, y1, x2, y2, kind], and guard regions to frost while the screen moves.

    Covers = SAM's track + Spark's sightings (held half a sampling step) + the template lock when it is near
    certain. Guards come from real motion: if the screen changes inside a secret's corridor (everywhere it can be
    between two sightings), the corridor is frosted until the motion stops."""
    fps, W, H = info["fps"], info["w"], info["h"]
    w, h = int(W * SCALE) // 2 * 2, int(H * SCALE) // 2 * 2
    sx, sy = w / W, h / H
    hold = 0.5 / info.get("rate", 2)
    live = [it for it in items if it["where"] == "screen" and it.get("seen") and it.get("decision") != "rejected"]
    anchors: dict[int, list] = {}
    sight = []
    for k, it in enumerate(live):
        pts = sorted((round(s["t"] * fps), s["box"]) for s in it["seen"])
        for n, b in pts:
            anchors.setdefault(n, []).append((k, b))
        sight.append(pts)
    covers: dict[int, list] = {}
    for k, it in enumerate(live):
        for n, box in tracks.get(it["id"], {}).items():
            covers.setdefault(n, []).append([*box[:4], it["kind"]])
        for n0, box in sight[k]:
            for n in range(max(0, n0 - round(hold * fps)), n0 + round(hold * fps) + 1):
                covers.setdefault(n, []).append([*box[:4], it["kind"]])

    def corridor(k: int, n: int) -> list[int] | None:
        """Everywhere secret k can be at frame n: from its last sighting to its next one (plus SAM's box)."""
        frames = [f for f, _ in sight[k]]
        i = bisect.bisect_right(frames, n)
        prv, nxt = (sight[k][i - 1] if i else None), (sight[k][i] if i < len(frames) else None)
        boxes = []
        if prv and n - prv[0] <= GAP_S * fps:
            boxes.append(prv[1])
        if nxt and nxt[0] - n <= GAP_S * fps:
            boxes.append(nxt[1])
        if not boxes:
            return None
        if n in tracks.get(live[k]["id"], {}):
            boxes.append(tracks[live[k]["id"]][n])
        pad = max(10, boxes[0][3] - boxes[0][1])
        return _union(boxes, pad, W, H)

    guards: dict[int, list] = {}
    anchor_frames = sorted(anchors)
    jump: list | None = None  # region frosted after a layout jump, until Spark looks again
    jump_until = -1
    state: list = [None] * len(live)
    calm = [0] * len(live)  # frames since the corridor last moved, for a little hysteresis
    small = lambda b: [round(b[0] * sx), round(b[1] * sy), round(b[2] * sx), round(b[3] * sy)]
    big = lambda b: [round(b[0] / sx), round(b[1] / sy), round(b[2] / sx), round(b[3] / sy)]
    proc = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-i", str(clip), "-vf", f"scale={w}:{h}", "-f",
                             "rawvideo", "-pix_fmt", "gray", "-"], stdout=subprocess.PIPE)
    n, size, prev = 0, w * h, None
    while (chunk := proc.stdout.read(size)) and len(chunk) == size:
        g = np.frombuffer(chunk, np.uint8).reshape(h, w)
        for k, box in anchors.get(n, []):
            state[k] = _Lock(g, small(box), live[k]["kind"])
        if prev is not None:  # a layout jump (maximise, zoom, new window): content lands where nothing predicts it
            changed = cv2.absdiff(g, prev) > 25
            frac = float(changed.mean())
            active = jump is not None and n <= jump_until
            # start on a big change; while active, any further motion (an animation still running) extends it
            if (frac > JUMP_FRAC or (active and frac > SETTLE_FRAC)) and any(corridor(k, n) for k in range(len(live))):
                ys, xs = np.nonzero(changed)
                box = big([int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1])
                jump = _union([jump, box], 0, W, H) if active else box
                i = bisect.bisect_right(anchor_frames, n)  # released only at Spark's first look after it settles
                jump_until = anchor_frames[i] if i < len(anchor_frames) else n + round(0.5 * fps)
        if jump is not None and n <= jump_until:
            guards.setdefault(n, []).append(_union([jump], 16, W, H))
        for k in range(len(live)):
            c = corridor(k, n)
            if c is None:
                state[k] = None
                continue
            if prev is not None:
                x1, y1, x2, y2 = small(c)
                diff = cv2.absdiff(g[y1:y2, x1:x2], prev[y1:y2, x1:x2])
                moving = diff.size and float((diff > 25).mean()) > 0.02
                calm[k] = 0 if moving else calm[k] + 1
                if calm[k] <= 3:  # moving now or a moment ago: frost the whole corridor
                    guards.setdefault(n, []).append(c)
            st = state[k]
            if st is not None and st.match(g) >= 0.9:
                covers.setdefault(n, []).append([*big(st.box), st.kind])
        prev = g
        n += 1
    proc.wait()
    return covers, guards
