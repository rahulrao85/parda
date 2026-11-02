"""Find the exact pixels of every on-screen text secret, on every frame: Spark decides WHAT, OCR decides WHERE.

Muse Spark names each secret ("AKIAIOSFODNN7EXAMPLE") but its boxes are 10-20 px off, and SAM is an object tracker
that does not know where a word ends. So the box drawn on a frame comes from OCR on that frame's own pixels: the
line that contains the value, cut down to the value alone (labels and the `=` of `KEY=value` stay visible), with
both edges snapped to the blank columns between glyphs.

Full-frame OCR is slow, so it runs only on keyframes (the first frame, a frame where the screen changed, a layout
jump). In between, each located value is carried while its pixels stay the same, and moved by phase correlation
while they move (a scroll or a drag), with crop OCR to confirm it now and then. Values first found together move
together, so one that scrolls in from off-screen is predicted from its neighbours. A value that cannot be confirmed
is still covered for 0.3 s and then dropped: never a big frosted region instead.

Detection runs in full-width strips with overlap (a line is never cut sideways, and an unchanged strip is a cache
hit). The angle classifier is off: on a 1080p .env in Notepad++ it is what dropped `AWS_ACCESS_KEY_ID=...`.
"""
import hashlib, json, re, subprocess, time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path

import cv2
import numpy as np

from . import sample

OCR_VERSION = "locate-v3"  # bump when the OCR call changes (keys the cache)
STRIP_H, STRIP_OVERLAP = 360, 96  # overlap > the tallest line, so every line is whole in at least one strip
GRACE_S = 0.3        # keep covering an unconfirmed value this long, then drop it
MOTION_CAP_S = 0.5   # full-frame OCR at most this often while the screen keeps changing and every value is confirmed
UNSURE_CAP_S = 0.1   # ... and at most this often while some value is not confirmed
# (A value that appears on a still screen between two keyframes is backfilled: see Locator._backfill.)
OCR_EVERY = 10       # frames between confirming crop OCRs for a moving value whose pixels still line up
JUMP_FRAC = 0.3      # share of thumbnail blocks changed in one frame that counts as a layout jump
POST_JUMP_S = 0.6    # full-frame OCR on every frame this long after a jump, so nothing waits for the next keyframe
EDGE_BAND = 160      # px beyond the frame's shift that the entry-edge scan covers while the screen scrolls
SAME = 3.0           # mean grey difference under which two crops show the same pixels (codec noise passes)
BACKFILL_S = 0.7     # look back this far for a value that was already there before the keyframe that found it
MIN_SCORE = 0.5      # OCR line confidence to keep a line


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9@]", "", s.lower())


def value_of(text: str) -> str:
    """Spark sometimes returns the label with the value ("AWS_KEY=AKIA...", "Name: Priya"): keep only the value."""
    m = re.match(r"^\s*[A-Za-z][\w .-]{0,40}?\s*[=:]\s*(\S.*)$", text)
    return m.group(1) if m and len(norm(m.group(1))) >= 4 else text


# ---------------------------------------------------------------- OCR

class Ocr:
    """RapidOCR with no resizing (det limit 4096, max side 4096), no angle classifier, and two caches: whole strips
    on disk by their pixels, and recognised lines in memory by their pixels."""

    def __init__(self, cache: Path):
        from rapidocr_onnxruntime import RapidOCR
        self.eng = RapidOCR(det_limit_type="max", det_limit_side_len=4096, max_side_len=4096)
        self.cache = cache
        cache.mkdir(parents=True, exist_ok=True)
        self.recs: dict[str, tuple] = {}
        self.n = Counter()
        self.seconds = 0.0
        self.min_chars = 3  # shortest value being looked for, in characters

    def _lines(self, rgb: np.ndarray) -> list[dict]:
        """Detect + recognise. Each line: box [x1,y1,x2,y2], text, score, chars [[ch, centre x], ...]."""
        rgb = _detint(rgb)
        t = time.time()
        quads, _ = self.eng.text_det(rgb)
        self.n["det_ms"] += int(1000 * (time.time() - t))
        if quads is None or not len(quads):
            return []
        H, W = rgb.shape[:2]
        boxes, crops, keys = [], [], []
        for q in quads:
            x1, y1 = max(0, int(np.floor(q[:, 0].min()))), max(0, int(np.floor(q[:, 1].min())))
            x2, y2 = min(W, int(np.ceil(q[:, 0].max()))), min(H, int(np.ceil(q[:, 1].max())))
            if x2 - x1 < 4 or y2 - y1 < 4 or x2 - x1 < 0.3 * (y2 - y1) * self.min_chars:
                continue  # too short to hold the shortest value: not worth recognising
            x1, y1, x2, y2 = _tight(rgb, x1, y1, x2, y2)
            crop = np.ascontiguousarray(rgb[y1:y2, x1:x2])
            boxes.append([x1, y1, x2, y2])
            crops.append(crop)
            keys.append(hashlib.sha1(crop.tobytes() + str(crop.shape).encode()).hexdigest())
        todo = [i for i, k in enumerate(keys) if k not in self.recs]
        if todo:
            t = time.time()
            res, _ = self.eng.text_rec([crops[i] for i in todo], return_word_box=True)
            self.n["rec_ms"] += int(1000 * (time.time() - t))
            for i, r in zip(todo, res):
                self.recs[keys[i]] = r
        self.n["rec_lines"] += len(todo)
        self.n["rec_cached"] += len(keys) - len(todo)
        out = []
        for box, key in zip(boxes, keys):
            text, score, info = (list(self.recs[key]) + [None, None, None])[:3]
            if not text or score < MIN_SCORE:
                continue
            chars = []
            if info and info[0]:  # CTC columns -> x centre of every recognised character
                step = (box[2] - box[0]) / info[0]
                chars = [[c, round(box[0] + (col + 0.5) * step, 1)]
                         for word, cols in zip(info[1], info[2]) for c, col in zip(word, cols)]
            out.append({"box": box, "text": text, "score": round(float(score), 3), "chars": chars})
        return out

    def _cached(self, key: str, fn) -> list[dict]:
        path = self.cache / f"{key}.json"
        if path.exists():
            self.n["cache_hits"] += 1
            return json.loads(path.read_text(encoding="utf-8"))
        t = time.time()
        lines = fn()
        self.seconds += time.time() - t
        path.write_text(json.dumps(lines, ensure_ascii=False), encoding="utf-8")
        return lines

    def frame(self, rgb: np.ndarray) -> list[dict]:
        """Every line on the frame: full-width strips with overlap, lines cut by a strip edge dropped (the next
        strip has them whole), duplicates from the overlap merged."""
        self.n["full"] += 1
        H, W = rgb.shape[:2]
        found = []
        starts = list(range(0, max(1, H - STRIP_OVERLAP), STRIP_H - STRIP_OVERLAP))
        for y0 in starts:
            y1 = min(H, y0 + STRIP_H)
            strip = np.ascontiguousarray(rgb[y0:y1])
            key = hashlib.sha1(OCR_VERSION.encode() + strip.tobytes() + str(strip.shape).encode()).hexdigest()
            self.n["strips"] += 1
            for ln in self._cached(key, lambda: self._lines(strip)):
                bx = ln["box"]
                if (bx[1] <= 1 and y0 > 0) or (bx[3] >= y1 - y0 - 1 and y1 < H):
                    continue
                found.append(dict(ln, box=[bx[0], bx[1] + y0, bx[2], bx[3] + y0]))
            if y1 >= H:
                break
        out: list[dict] = []
        for ln in sorted(found, key=lambda l: -len(l["text"])):
            if not any(_iou(ln["box"], o["box"]) > 0.5 for o in out):
                out.append(ln)
        return out

    def crop(self, rgb: np.ndarray, region: list[int]) -> list[dict]:
        """Lines inside a region, in frame coordinates; upscaled 2x when its text is small."""
        self.n["crop"] += 1
        x1, y1, x2, y2 = region
        img = np.ascontiguousarray(rgb[y1:y2, x1:x2])
        if img.shape[0] < 8 or img.shape[1] < 8:
            return []
        up = 2 if img.shape[0] < 3 * 24 else 1  # region is ~3 line heights tall
        if up > 1:
            img = cv2.resize(img, None, fx=up, fy=up, interpolation=cv2.INTER_CUBIC)
        key = hashlib.sha1(OCR_VERSION.encode() + b"crop" + img.tobytes() + str(img.shape).encode()).hexdigest()
        lines = self._cached(key, lambda: self._lines(img))
        return [dict(ln, box=[x1 + ln["box"][0] / up, y1 + ln["box"][1] / up, x1 + ln["box"][2] / up,
                              y1 + ln["box"][3] / up],
                     chars=[[c, x1 + x / up] for c, x in ln["chars"]], clipped=_clipped(ln["box"], img.shape))
                for ln in lines]


def _detint(rgb: np.ndarray) -> np.ndarray:
    """Pale coloured pixels (a SPECIMEN watermark, a selection tint) become paper before OCR: the recogniser
    could not read an amount with a red watermark across it, so the amount stayed uncovered for seconds. Text
    is dark and survives; a dark coloured heading is dark too."""
    c = rgb.astype(np.int16)
    lum = (c[..., 0] * 30 + c[..., 1] * 59 + c[..., 2] * 11) // 100
    tint = (c.max(axis=2) - c.min(axis=2) > 25) & (lum > 150)
    if not tint.any():
        return rgb
    out = rgb.copy()
    out[tint] = 255
    return out


def _tight(rgb: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> tuple[int, int, int, int]:
    """The detection box cut to its ink plus 3 px. Detection boxes jitter by a pixel as text moves; the ink does
    not, so a line that only scrolled or was dragged hits the recognition cache."""
    c = rgb[y1:y2, x1:x2].astype(np.int16)
    paper = np.median(np.concatenate([c[0], c[-1], c[:, 0], c[:, -1]]), axis=0)
    ink = np.abs(c - paper).sum(axis=2) > 90
    ys, xs = np.nonzero(ink.any(axis=1))[0], np.nonzero(ink.any(axis=0))[0]
    if not len(ys) or not len(xs):
        return x1, y1, x2, y2
    H, W = rgb.shape[:2]
    return (max(0, x1 + int(xs[0]) - 3), max(0, y1 + int(ys[0]) - 3), min(W, x1 + int(xs[-1]) + 4),
            min(H, y1 + int(ys[-1]) + 4))


def _clipped(box, shape) -> bool:
    return box[0] <= 2 or box[2] >= shape[1] - 2


def _inside(a, b) -> float:
    """Share of box a's area that lies inside box b."""
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    return ix / max(1, (a[2] - a[0]) * (a[3] - a[1]))


def _iou(a, b) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix
    return ix / u if u > 0 else 0.0


# ---------------------------------------------------------------- value span inside a line

CURSOR = set("|lI1!")  # what OCR makes of a text cursor at the end of a form field
TOKEN_GLUE = "@._-+/#"  # characters that keep a token together (e-mails, keys, dotted names)
PARTIAL_NEVER = {"financial_amount", "amount", "date_of_birth", "date"}  # a piece of these is a different value


def _match(line: dict, value: str, kind: str | None = None) -> tuple[int, int] | None:
    """[i, j) of the value among the line's normalised characters: exact first, fuzzy only for long values.
    A match must sit at a token boundary in the raw text ("500000" inside "-25,000.00" is not "-5,000.00"), so
    a short value never claims the tail of a longer one."""
    raw = [c for c, _ in line["chars"]] if line["chars"] else list(line["text"])
    kept = [k for k, c in enumerate(raw) if norm(c)]  # raw index of every normalised character
    t = "".join(norm(raw[k]) for k in kept)
    v = norm(value_of(value))
    if len(v) < 3:
        # A 1-2 character value ("25" in a date-of-birth day box) only when it is the whole line.
        return (0, len(t)) if len(v) >= 2 and t == v else None

    one_class = v.isdigit() or v.isalpha()

    def edge(outside: str, inside: str) -> bool:
        # A boundary is a non-alphanumeric character or, for an all-digit or all-letter value, a change of class:
        # OCR drops the space in "number 000123456789", and the letter-to-digit step is still the value's edge.
        # (Not for mixed values: "K7" followed by "M" inside a key is the middle of the key.)
        return not outside.isalnum() or (one_class and outside.isalpha() != inside.isalpha())

    def bounded(i: int, j: int) -> bool:  # letters and digits on either side would make it part of a longer token
        a, b = kept[i] - 1, kept[j - 1] + 1
        after_ok = b >= len(raw) or edge(raw[b], raw[kept[j - 1]]) or (b == len(raw) - 1 and raw[b] in CURSOR)
        return (a < 0 or edge(raw[a], raw[kept[i]])) and after_ok

    i = t.find(v)
    while i >= 0:
        if bounded(i, i + len(v)):
            return i, i + len(v)
        i = t.find(v, i + 1)
    L = len(v)
    if L >= 5 and kind not in PARTIAL_NEVER:
        # Spark saw the value while it was being typed ("anita.verm" of "anita.verma@example.com") or verify read a
        # piece of it ("erma@exa"): a secret is a token, so the whole token the piece sits in is the value, as long
        # as the token is not far longer than the piece (an amount inside a longer amount is a different value).
        def glue(lo: int, hi: int) -> bool:  # raw[lo:hi] is all token characters (no space, comma, bracket ...)
            return all(c.isalnum() or c in TOKEN_GLUE for c in raw[lo:hi])

        i = t.find(v)
        while i >= 0:
            a, b = i, i + L
            while a > 0 and glue(kept[a - 1], kept[a]):
                a -= 1
            while b < len(t) and glue(kept[b - 1] + 1, kept[b] + 1):
                b += 1
            if b - a <= 3 * L and bounded(a, b):
                return a, b
            i = t.find(v, i + 1)
    if L >= 8 and len(t) >= L - 1:
        # Spark drops or doubles a repeated character ("SMPL000123" for "SMPL0000123"): a token-bounded window one
        # character shorter or longer that agrees with the value almost everywhere is the value.
        from difflib import SequenceMatcher
        best, bij = 0.0, None
        for wl in (L - 1, L + 1):
            for s in range(len(t) - wl + 1):
                if not bounded(s, s + wl):
                    continue
                r = SequenceMatcher(None, t[s:s + wl], v).ratio()
                if r > best:
                    best, bij = r, (s, s + wl)
        if best >= 0.88:
            return bij
    if kind == "password" and len(line["text"]) >= 6 and " " not in value.strip() and L >= 6:
        # (a Spark description such as "masked password dots" is not a string to look for)
        # Spark and OCR each read a random string a little differently ("QAVbxx*P6Kk" vs "vQAVbxx^P6k6"): a
        # token of the line that is mostly the same string is the password.
        from difflib import SequenceMatcher
        best, bij = 0.0, None
        pos = 0
        for tok in line["text"].split():
            start = line["text"].index(tok, pos)
            pos = start + len(tok)
            ks = [k for k in range(start, pos) if k in set(kept)]
            if len(ks) < 6:
                continue
            i, j = kept.index(ks[0]), kept.index(ks[-1]) + 1
            if abs((j - i) - L) > 3 or not any(ch.isdigit() for ch in t[i:j]):
                continue  # a label ("Mobile numbe", cut by a popup) is never a password: no digits, wrong length
            r = SequenceMatcher(None, t[i:j], v).ratio()
            if r > best:
                best, bij = r, (i, j)
        if best >= 0.6:
            return bij
    # A value being typed: the line ends with the first 4+ characters of it (a form field mid-entry). Covered
    # from the fourth character, so a name or an email is not readable letter by letter while it is entered.
    for n in range(min(L - 1, len(t)), 3, -1):
        if t.endswith(v[:n]) and bounded(len(t) - n, len(t)):
            return len(t) - n, len(t)
    run = 6 if L >= 10 else 5
    if L < (10 if kind in PARTIAL_NEVER else 8) or len(t) < L or not any(v[k:k + run] in t for k in range(0, L - run + 1)):
        return None  # fuzzy only for long values (8+, amounts and dates 10+), on a line that shares a run of them
    # OCR misreads are mostly substitutions (0/O, l/I: "Pooja lyer"): the best same-length window must agree on 85%
    best, bi = 0, -1
    for s in range(len(t) - L + 1):
        eq = sum(a == b for a, b in zip(t[s:s + L], v))
        if eq > best:
            best, bi = eq, s
    if best < 0.85 * L or not bounded(bi, bi + L):
        return None
    return bi, bi + L


def _ink_cols(rgb: np.ndarray, y1: int, y2: int, x1: int, x2: int):
    """Per column of a line band: does it hold any ink? Paper = median colour of the band's top and bottom rows;
    ink = at least half as far from the paper as the text's own strongest pixels, so a faint watermark or a
    selection tint running through the line does not count as glyphs."""
    band = rgb[y1:y2, x1:x2].astype(np.int16)
    paper = np.median(np.concatenate([band[0], band[-1]]), axis=0)
    dist = np.abs(band - paper).sum(axis=2)
    strong = np.percentile(dist, 98)
    ink = dist > max(90, 0.5 * strong)
    return ink.any(axis=0), ink


def span(rgb: np.ndarray, line: dict, value: str, kind: str | None = None) -> list[int] | None:
    """Box of `value` inside an OCR line, or None. Character positions come from the recogniser's CTC columns
    (OCR drops `_` and spaces, so a plain character count drifts by half a glyph); each edge then moves to the
    nearest blank column, never more than a third of a glyph into the value. A value at either end of the line
    runs to the last ink, even past a detection box that clipped the final glyph."""
    ij = _match(line, value, kind)
    if ij is None:
        return None
    i, j = ij
    H, W = rgb.shape[:2]
    lx1, ly1, lx2, ly2 = (int(round(v)) for v in line["box"])
    raw = value_of(value)
    xs = [x for c, x in line["chars"] if norm(c)]
    if len(xs) >= 2:
        cw = (xs[-1] - xs[0]) / (len(xs) - 1)
    else:  # no CTC columns: proportional by character index
        n = max(1, len(norm(line["text"])))
        cw = (lx2 - lx1) / n
        xs = [lx1 + (k + 0.5) * cw for k in range(n)]
    lead = len(raw) - len(raw.lstrip("+-$₹€£#(")) if raw else 0  # a sign or currency in front is part of the value
    trail = len(raw) - len(raw.rstrip(")%")) if raw else 0
    left, right = xs[i] - cw / 2 - lead * cw, xs[j - 1] + cw / 2 + trail * cw
    bx1, bx2 = max(0, int(lx1 - 3 * cw)), min(W, int(lx2 + 3 * cw))
    by1, by2 = max(0, ly1 - 2), min(H, ly2 + 2)
    if by2 - by1 < 4 or bx2 - bx1 < 4:
        return None
    cols, ink = _ink_cols(rgb, by1, by2, bx1, bx2)
    gap = ~cols

    def ink_end(x0: float, step: int) -> int:  # last ink column of the line from x0 outward
        x, last, run = int(x0) - bx1, int(x0) - bx1, 0
        while 0 <= x < len(cols) and run <= 0.8 * cw:
            if cols[x]:
                last, run = x, 0
            else:
                run += 1
            x += step
        return last + bx1

    def snap(x: float, outward: int) -> int:
        lo, hi = (x - 0.6 * cw, x + 0.35 * cw) if outward < 0 else (x - 0.35 * cw, x + 0.6 * cw)
        cand = [c for c in range(max(0, int(lo) - bx1), min(len(gap), int(hi) - bx1 + 1)) if gap[c]]
        if not cand:
            return int(x) + outward  # no blank column nearby: one pixel outward
        return min(cand, key=lambda c: (abs(c + bx1 - x), -outward * c)) + bx1

    # A value at an end of its line runs exactly to the line's ink there (a sign in front included), at most one
    # glyph past the detection box, so a box that clipped the last "0" of "+600.00" still covers it.
    if i == 0 and not line.get("clipped"):
        left = max(lx1 - cw, ink_end(lx1 + cw / 2, -1))
    if j >= len(xs) and not line.get("clipped"):
        right = min(lx2 + cw, ink_end(lx2 - cw / 2, +1) + 1)
    x1, x2 = snap(left, -1), snap(right, +1) + 1
    if x2 - x1 < 2:
        return None
    y1, y2 = _rows(ink[:, max(0, x1 - bx1):max(0, x2 - bx1)], (ly1 + ly2) / 2 - by1)
    y1, y2 = by1 + y1, by1 + y2
    lh = ly2 - ly1
    if y2 - y1 < 0.6 * lh:  # low ink (dots, dashes): keep a line-sized box around the line's middle
        y1, y2 = min(y1, ly1 + lh // 5), max(y2, ly2 - lh // 5)
    pad = 2
    if i == 0 and j >= len(xs):  # the whole line
        x1, x2 = x1 - pad, x2 + pad
    return [int(max(0, x1)), int(max(0, y1 - pad)), int(min(W, x2)), int(min(H, y2 + pad))]


def _rows(ink: np.ndarray, mid: float) -> tuple[int, int]:
    """The run of inked rows through the line's middle: stops at the blank rows that separate it from the lines
    above and below (a detection box often reaches into its neighbours)."""
    rows = ink.any(axis=1)
    if not rows.any():
        return 0, len(rows)
    m = int(np.clip(round(mid), 0, len(rows) - 1))
    if not rows[m]:  # middle row blank (between two words' x-heights is rare): nearest inked row
        m = int(np.nonzero(rows)[0][np.argmin(np.abs(np.nonzero(rows)[0] - m))])
    top, bot = m, m
    while top > 0 and rows[top - 1]:
        top -= 1
    while bot < len(rows) - 1 and rows[bot + 1]:
        bot += 1
    return top, bot + 1


# ---------------------------------------------------------------- per-value state

@dataclass
class Track:
    key: str            # tracks.json key: the item id, or "id#2" when the same value shows in two places at once
    item: dict
    box: list[int]
    cluster: int
    confirmed: int      # last frame on which OCR or unchanged pixels proved the value is under the box
    ocr_at: int         # last frame a crop OCR ran for it
    partial: bool = False  # (partly) off-screen since it was last confirmed: its cluster vouches for it
    lh: int = field(init=False)
    ocr_ok: int = field(init=False)  # last frame OCR itself read the value here (not a carried box)
    followed: int = -1  # last frame its box was moved by the members of its cluster that OCR found again
    wrong: int = -1     # last frame crop OCR read a different text under the box (it sits on a label, not the value)

    def __post_init__(self):
        self.lh = max(8, self.box[3] - self.box[1])
        self.ocr_ok = self.confirmed


def _clip(box, W, H):
    return [max(0, box[0]), max(0, box[1]), min(W, box[2]), min(H, box[3])]


def _visible(box, W, H) -> float:
    b = _clip(box, W, H)
    area = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    return area / max(1, (box[2] - box[0]) * (box[3] - box[1]))


def _same(a: np.ndarray, b: np.ndarray, ba, bb) -> bool:
    """Do box ba in frame a and box bb in frame b show the same pixels (compared over their on-screen parts)?"""
    H, W = a.shape
    dx, dy = bb[0] - ba[0], bb[1] - ba[1]
    x1, y1 = max(0, ba[0], -dx), max(0, ba[1], -dy)
    x2, y2 = min(W, ba[2], W - dx), min(H, ba[3], H - dy)
    if x2 - x1 < 3 or y2 - y1 < 3:
        return False
    pa = a[y1:y2, x1:x2]
    pb = b[y1 + dy:y2 + dy, x1 + dx:x2 + dx]
    return float(cv2.absdiff(pa, pb).mean()) < SAME and float(pa.std()) > 2  # a blank patch proves nothing


def _has_ink(gray: np.ndarray, box) -> bool:
    """Is there text-grade contrast under the box (glyphs, a toolbar edge), or only paper and a faint watermark?"""
    b = _clip([int(v) for v in box], gray.shape[1], gray.shape[0])
    reg = gray[b[1]:b[3], b[0]:b[2]].astype(np.int16)
    if reg.size == 0:
        return False
    paper = np.median(np.concatenate([reg[0], reg[-1]]))
    return float((np.abs(reg - paper) > 80).mean()) > 0.02


def _shift(prev: np.ndarray, cur: np.ndarray, box, lh: int) -> tuple[int, int, float]:
    """How far the content around a box moved between two frames: phase correlation, then a template match if the
    moved pixels do not line up (a shift larger than the window)."""
    H, W = prev.shape
    m = max(96, 4 * lh)
    x1, y1, x2, y2 = _clip([box[0] - m, box[1] - m, box[2] + m, box[3] + m], W, H)
    if x2 - x1 >= 16 and y2 - y1 >= 16:
        a, b = prev[y1:y2, x1:x2].astype(np.float32), cur[y1:y2, x1:x2].astype(np.float32)
        (dx, dy), resp = cv2.phaseCorrelate(a, b, cv2.createHanningWindow((x2 - x1, y2 - y1), cv2.CV_32F))
        dx, dy = int(round(dx)), int(round(dy))
        if _same(prev, cur, box, [box[0] + dx, box[1] + dy, box[2] + dx, box[3] + dy]):
            return dx, dy, float(resp)
    tb = _clip([box[0] - 4, box[1] - 4, box[2] + 4, box[3] + 4], W, H)
    tmpl = prev[tb[1]:tb[3], tb[0]:tb[2]]
    if tmpl.size == 0 or tmpl.std() < 3:
        return 0, 0, 0.0
    m = max(240, 8 * lh)
    sx1, sy1, sx2, sy2 = _clip([tb[0] - m, tb[1] - m, tb[2] + m, tb[3] + m], W, H)
    win = cur[sy1:sy2, sx1:sx2]
    if win.shape[0] < tmpl.shape[0] or win.shape[1] < tmpl.shape[1]:
        return 0, 0, 0.0
    res = cv2.matchTemplate(win, tmpl, cv2.TM_CCOEFF_NORMED)
    _, score, _, (lx, ly) = cv2.minMaxLoc(res)
    return sx1 + lx - tb[0], sy1 + ly - tb[1], float(score) if np.isfinite(score) else 0.0


# ---------------------------------------------------------------- the pass over the clip

def _translation(a: np.ndarray, b: np.ndarray) -> tuple[bool, float, float]:
    """Is frame b just frame a shifted (a scroll, a drag)? Phase correlation of the thumbnails answers with a
    sharp peak; a zoom or a window maximise gives a weak, smeared one. Returns (moving, dx, dy) in thumbnail px."""
    (dx, dy), resp = cv2.phaseCorrelate(a, b)
    return resp > 0.35 and (abs(dx) >= 0.5 or abs(dy) >= 0.5), dx, dy


class Locator:
    def __init__(self, items: list[dict], info: dict, ocr: Ocr):
        self.items, self.info, self.ocr = items, info, ocr
        self.ids = {id(it) for it in items}
        self.fps, self.W, self.H = info["fps"], info["w"], info["h"]
        self.grace = round(GRACE_S * self.fps)
        self.live: list[Track] = []
        self.out: dict[str, dict[int, list[int]]] = defaultdict(dict)
        self.kind: dict[str, str] = {}
        self.hints: dict[int, list[tuple[dict, list]]] = defaultdict(list)  # Spark sightings, by frame
        for it in items:
            for s in it.get("seen", []):
                self.hints[round(s["t"] * self.fps)].append((it, s["box"]))
        self.ring: list[tuple[int, np.ndarray]] = []  # recent grey frames, for backfill
        self.why = Counter()  # what triggered each full-frame OCR
        self.jump_frame = -1  # the frame of the latest layout jump (zoom, maximise): only fresh OCR boxes count
        self.shift_now = (0, 0)  # how far content moved on the current frame, for the confirm crop

    # -- matching

    def _wanted(self, it: dict, n: int) -> bool:
        """Short values (a 4-digit amount) match anywhere, so they count only near a Spark sighting."""
        if len(norm(value_of(it["text"]))) >= 6:
            return True
        return any(abs(s["t"] * self.fps - n) <= 3 * self.fps for s in it.get("seen", []))

    def _find(self, rgb, lines, items, n) -> list[tuple[dict, list[int]]]:
        hits = []
        for it in items:
            if not self._wanted(it, n):
                continue
            for ln in lines:
                b = span(rgb, ln, it["text"], it.get("kind"))
                if b:
                    hits.append((it, b))
        return hits

    def _new_key(self, it: dict) -> str:
        used = {t.key for t in self.live}
        k, key = 1, it["id"]
        while key in used:
            k += 1
            key = f"{it['id']}#{k}"
        self.kind[key] = it.get("kind", "other")
        return key

    def _assign(self, hits, n: int, cluster: int, gray, rgb=None) -> None:
        """Pair each located box with the nearest live track of the same item; the rest start new tracks."""
        moved: dict[int, list] = defaultdict(list)  # old cluster -> corrections of the members found again
        located = set()
        for it, b in hits:
            cands = [t for t in self.live if t.item is it and id(t) not in located]
            near = min(cands, key=lambda t: abs(t.box[0] - b[0]) + abs(t.box[1] - b[1]), default=None)
            if near and near.ocr_ok == n and not _near(near.box, b, 8) and rgb is not None:
                # OCR read the value under the track's box on this very frame and this hit disagrees with it.
                # The value cannot be in two places unless it is shown twice, so the hit must read again on its
                # own pixels before it is believed.
                lh = max(8, b[3] - b[1])
                region = _clip([b[0] - lh, b[1] - lh // 2, b[2] + lh, b[3] + lh // 2], self.W, self.H)
                if not any(span(rgb, ln, it["text"], it.get("kind")) for ln in self.ocr.crop(rgb, region)):
                    self.ocr.n["stale_hits"] += 1
                    continue
                near = None  # a genuine second showing of the value: its own track
            if near and abs(near.box[0] - b[0]) + abs(near.box[1] - b[1]) <= max(60, 3 * near.lh):
                if near.confirmed >= n - 1 and not near.partial:
                    # Only a member whose position was known a frame ago says how far the cluster moved. The
                    # correction of a box that sat unconfirmed at the screen edge for seconds says nothing about
                    # the others: applying it once dragged a whole statement one line off.
                    moved[near.cluster].append((b[0] - near.box[0], b[1] - near.box[1]))
                near.box, near.confirmed, near.cluster, near.partial = b, n, cluster, False
                near.ocr_ok = n
                located.add(id(near))
            else:
                t = Track(self._new_key(it), it, b, cluster, n, n)
                self.live.append(t)
                located.add(id(t))
                self._backfill(t, gray, n)
        for t in self.live:  # members not seen this time (off-screen) follow the ones that were
            if id(t) not in located and t.cluster in moved:
                dx = int(np.median([d[0] for d in moved[t.cluster]]))
                dy = int(np.median([d[1] for d in moved[t.cluster]]))
                t.box = [t.box[0] + dx, t.box[1] + dy, t.box[2] + dx, t.box[3] + dy]
                t.cluster, t.followed = cluster, n

    def _backfill(self, t: Track, gray, n: int) -> None:
        """A value OCR finds now may have been readable a few frames earlier, before this keyframe: cover those
        frames too while the pixels under the box are the same."""
        for m, g in reversed(self.ring):
            if m >= n:
                continue
            if not _same(g, gray, t.box, t.box):
                break
            self.out[t.key].setdefault(m, list(t.box))

    # -- per frame

    def _follow(self, n, prev, gray, rgb, page_shift: tuple[int, int] | None) -> None:
        """Carry values whose pixels did not change; shift the rest with their cluster and confirm them.
        page_shift is how far the whole screen moved this frame (a scroll or drag), the fallback when the shift
        measured around a cluster's lead cannot be trusted: a box one frame behind a fast scroll is a leak."""
        moving: dict[int, list[Track]] = defaultdict(list)
        for t in self.live:
            vis = _visible(t.box, self.W, self.H)
            if vis > 0 and _same(prev, gray, t.box, t.box):
                if t.confirmed == n - 1:  # the same pixels as a confirmed frame: still the value
                    t.confirmed = n
                elif vis == 1.0 and n - t.ocr_at >= OCR_EVERY:
                    self._confirm(t, rgb, n)  # unconfirmed on static pixels: look again now and then
                continue
            if vis > 0:
                moving[t.cluster].append(t)
        for c, members in moving.items():
            lead = max(members, key=lambda t: _visible(t.box, self.W, self.H))
            dx, dy, resp = _shift(prev, gray, _clip(lead.box, self.W, self.H), lead.lh)
            if resp < 0.3 and page_shift is not None:
                dx, dy, resp = page_shift[0], page_shift[1], 0.3
            self.shift_now = (dx, dy)
            static = {id(t) for t in self.live if t.cluster == c and id(t) not in {id(m) for m in members}
                      and _visible(t.box, self.W, self.H) > 0}
            for t in self.live:
                if t.cluster != c or id(t) in static:
                    continue
                old = t.box
                t.box = [old[0] + dx, old[1] + dy, old[2] + dx, old[3] + dy]
                vis = _visible(t.box, self.W, self.H)
                if vis < 1.0:
                    t.partial = True
                if id(t) not in {id(m) for m in members}:
                    continue
                pixels_ok = _same(prev, gray, old, t.box)
                if pixels_ok and t.confirmed >= n - 1:
                    t.confirmed = n
                if vis == 1.0 and (not pixels_ok or resp < 0.3 or n - t.ocr_at >= OCR_EVERY):
                    self._confirm(t, rgb, n)

    def _confirm(self, t: Track, rgb, n: int) -> None:
        """Crop OCR around the predicted box: the value's line plus about a line above and below."""
        t.ocr_at = n
        w = t.box[2] - t.box[0]
        mx, my = (abs(v) for v in self.shift_now)  # a fast scroll can move a line further than a line height
        region = _clip([int(t.box[0] - 0.15 * w - t.lh - mx), t.box[1] - t.lh - my,
                        int(t.box[2] + 0.15 * w + t.lh + mx), t.box[3] + t.lh + my], self.W, self.H)
        best = None
        for ln in self.ocr.crop(rgb, region):
            b = span(rgb, ln, t.item["text"], t.item.get("kind"))
            if b and (best is None or abs(b[1] - t.box[1]) < abs(best[1] - t.box[1])):
                best = b
        if best:
            t.box, t.confirmed, t.partial, t.ocr_ok = best, n, False, n
        elif any(ln["score"] >= 0.9 and len(norm(ln["text"])) >= 3
                 and max(_inside(t.box, ln["box"]), _inside(ln["box"], t.box)) >= 0.4
                 for ln in self.ocr.crop(rgb, region)):
            t.wrong = n  # OCR reads something else there with confidence: the box has slid onto other text

    def _hint(self, n, rgb, gray) -> None:
        """At a Spark sighting, a value with no confirmed track gets a crop OCR around Spark's (rough) box."""
        for it, hb in self.hints.get(n, []):
            if id(it) not in self.ids or any(t.item is it and t.confirmed == n for t in self.live):
                continue
            h = max(20, hb[3] - hb[1])
            w = hb[2] - hb[0]
            region = _clip([int(hb[0] - 0.25 * w - 40), hb[1] - 2 * h, int(hb[2] + 0.25 * w + 40), hb[3] + 2 * h],
                           self.W, self.H)
            hits = [(it, b) for ln in self.ocr.crop(rgb, region) if (b := span(rgb, ln, it["text"], it.get("kind")))]
            if hits:
                self._assign(hits, n, n, gray, rgb)

    def _edge_scan(self, n: int, rgb, gray, dx: float, dy: float) -> None:
        """While the screen scrolls or is dragged, OCR the band at the edge the content enters from, every frame:
        a value sliding into view is caught on its first frame instead of at the next keyframe."""
        W, H = self.W, self.H
        band = int(min(H, abs(dy) + EDGE_BAND))
        side = int(min(W, abs(dx) + EDGE_BAND))
        regions = []
        if dy >= 0.5:
            regions.append([0, 0, W, band])
        elif dy <= -0.5:
            regions.append([0, H - band, W, H])
        if dx >= 0.5:
            regions.append([0, 0, side, H])
        elif dx <= -0.5:
            regions.append([W - side, 0, W, H])
        lines = [ln for r in regions for ln in self.ocr.crop(rgb, r)]
        if lines:
            self.ocr.n["edge"] += 1
            self._assign(self._find(rgb, lines, self.items, n), n, n, gray, rgb)

    def _record(self, n: int, gray: np.ndarray) -> None:
        conf = defaultdict(lambda: -10 ** 9)
        for t in self.live:
            if _visible(t.box, self.W, self.H) == 1.0:
                conf[t.cluster] = max(conf[t.cluster], t.confirmed)
        keep = []
        for t in self.live:
            if n == self.jump_frame and t.confirmed != n and not (t.followed == n and _has_ink(gray, t.box)):
                continue  # the layout just changed scale or shape: a box from before it covers nothing now,
                # unless the members OCR did find again moved it here and there is text under it (a line half
                # under the viewer's toolbar on the last frame before the window closes)
            own = n - t.confirmed <= self.grace
            # A member its cluster still vouches for stays covered even when its own crop OCR fails: a line half
            # hidden under the viewer's toolbar or bottom edge, or one that is partly off-screen. But only while
            # there is text-grade ink under the box: a box predicted onto blank page or a watermark is a phantom.
            vouched = n - conf[t.cluster] <= self.grace and t.wrong < t.confirmed and (
                (t.partial and _visible(t.box, self.W, self.H) < 1.0) or _has_ink(gray, t.box))
            if not (own or vouched):
                continue  # not confirmed for GRACE_S: drop it (a keyframe will find it again if it is there)
            keep.append(t)
            b = _clip(t.box, self.W, self.H)
            if b[2] - b[0] >= 2 and b[3] - b[1] >= 2:
                self.out[t.key][n] = b
        self.live = keep

    def run(self, clip) -> dict:
        W, H, fps = self.W, self.H, self.fps
        ys = np.arange(sample.THUMB_H) * H // sample.THUMB_H
        xs = np.arange(sample.THUMB_W) * W // sample.THUMB_W
        # Constant frame rate at the clip's own fps: frame n is always time n/fps, in here and in render.py, even if
        # the decoder drops a damaged frame (it is duplicated, not skipped, so later boxes do not land one frame late).
        proc = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-i", str(clip), "-fps_mode", "cfr", "-r", f"{fps}",
                                 "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
        n, size, prev, prev_th, key_th, last_full = 0, W * H * 3, None, None, None, -10 ** 9
        keyframes = []
        while (chunk := proc.stdout.read(size)) and len(chunk) == size:
            rgb = np.frombuffer(chunk, np.uint8).reshape(H, W, 3)
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            th = gray[np.ix_(ys, xs)].astype(np.float32)
            # A scroll or drag moves every block, but phase correlation of the thumbnails explains it as one shift.
            moving, mdx, mdy = (False, 0.0, 0.0) if prev_th is None else _translation(prev_th, th)
            mdx, mdy = mdx * W / sample.THUMB_W, mdy * H / sample.THUMB_H  # thumbnail -> frame pixels
            self.shift_now = (int(round(mdx)), int(round(mdy))) if moving else (0, 0)
            if prev is not None:
                self._follow(n, prev, gray, rgb, self.shift_now if moving else None)
            full = n == 0
            if not full and sample.changed(th, key_th):
                d = np.abs(th - prev_th).reshape(sample.THUMB_H // sample.BLOCK, sample.BLOCK,
                                                 sample.THUMB_W // sample.BLOCK, sample.BLOCK).mean(axis=(1, 3))
                # A jump is a big change that is NOT a plain shift: a zoom, a maximise, a new window.
                jump = float((d > 4).mean()) > JUMP_FRAC and not moving
                if moving:
                    self._edge_scan(n, rgb, gray, mdx, mdy)
                unsure = moving or any(t.confirmed < n and _visible(t.box, W, H) == 1.0 for t in self.live)
                cap = UNSURE_CAP_S if unsure else MOTION_CAP_S
                full = jump or n - last_full >= cap * fps
                if jump:
                    self.jump_frame = n  # every frame of a zoom animation is OCR'd; stale boxes are not drawn
                if full:
                    self.why["jump" if jump else "unsure" if unsure else "changed"] += 1
            if not full and 0 < n - self.jump_frame <= round(POST_JUMP_S * fps):
                full = True  # right after a jump the layout is new: look every frame until everything is found again
                self.why["after jump"] += 1
            if full:
                hits = self._find(rgb, self.ocr.frame(rgb), self.items, n)
                self._assign(hits, n, n, gray, rgb)
                key_th, last_full = th, n
                keyframes.append(n)
            self._hint(n, rgb, gray)
            self._record(n, gray)
            self.ring = (self.ring + [(n, gray)])[-round(BACKFILL_S * fps):]
            prev, prev_th = gray, th
            n += 1
        proc.wait()
        self.frames, self.keyframes = n, keyframes
        return self._dedupe()

    def _dedupe(self) -> dict:
        """A value inside another located value on the same frame (the name inside the email) is covered by it."""
        by_frame: dict[int, list] = defaultdict(list)
        for key, per in self.out.items():
            for n, b in per.items():
                by_frame[n].append((key, b))
        out: dict[str, dict[int, list[int]]] = defaultdict(dict)
        for n, boxes in by_frame.items():
            for key, b in boxes:
                area = max(1, (b[2] - b[0]) * (b[3] - b[1]))
                inside = any(k != key and k.split("#")[0] != key.split("#")[0] and o[0] <= b[0] + 2 and
                             o[1] <= b[1] + 4 and b[2] <= o[2] + 2 and b[3] <= o[3] + 4 and
                             (o[2] - o[0]) * (o[3] - o[1]) > area for k, o in boxes)
                if not inside:
                    out[key][n] = b
        return dict(out)


SMOOTH_GAP = 3      # a value missing for at most this many frames between two sightings is still there


def _near(a, b, tol: int) -> bool:
    return all(abs(x - y) <= tol for x, y in zip(a, b))


def _smooth(tracks: dict[str, dict[int, list[int]]]) -> dict[str, dict[int, list[int]]]:
    """A value seen on frame a and again on frame b a few frames later was there in between (a scroll step whose
    crop OCR failed under a watermark, say): interpolate the box over the gap. Nothing is dropped or moved."""
    out = {}
    for key, per in tracks.items():
        per = dict(sorted(per.items()))
        frames = list(per)
        for a, b in zip(frames, frames[1:]):
            if 1 < b - a <= SMOOTH_GAP + 1 and _near(per[a], per[b], 120):
                for m in range(a + 1, b):
                    w = (m - a) / (b - a)
                    per[m] = [int(round(x + (y - x) * w)) for x, y in zip(per[a], per[b])]
        out[key] = dict(sorted(per.items()))
    return out


def locate(clip, items: list[dict], info: dict, out_dir: Path) -> dict[str, dict[int, list[int]]]:
    """For every on-screen TEXT item: {source frame: [x1, y1, x2, y2]} of the value alone, in source pixels.
    A value shown in two places at once gets a second key, "<id>#2"."""
    t0 = time.time()
    text = [it for it in items if it.get("where") == "screen" and it.get("decision") != "rejected"
            and len(norm(value_of(it.get("text", "")))) >= 2]
    if not text:
        return {}
    ocr = Ocr(Path("cache/ocr"))
    ocr.min_chars = min(len(norm(value_of(it["text"]))) for it in text)
    loc = Locator(text, info, ocr)
    tracks = _smooth(loc.run(clip))
    per_item = Counter()
    for key, per in tracks.items():
        per_item[key.split("#")[0]] += len(per)
    stats = {"seconds": round(time.time() - t0, 1), "ocr_seconds": round(ocr.seconds, 1), "frames": loc.frames,
             "keyframes": len(loc.keyframes), "keyframe_reasons": dict(loc.why), "ocr_calls": dict(ocr.n),
             "items_located": sorted(per_item), "items_not_located": [it["id"] for it in text
                                                                     if it["id"] not in per_item],
             "frames_per_item": dict(per_item), "frames_per_key": {k: len(v) for k, v in tracks.items()}}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "locate_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return tracks
