"""Image level: PIL-drawn .env lines and a fake ID card go through OCR (the way locate.py reads a frame), span()
and render.cover(). Labels must stay pixel-identical, values must be gone, and the cover must use the line's own
paper and ink. Every value here is fake."""
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

from conftest import MONO, SANS, needs_font
from parda import locate, render

pytestmark = needs_font

ENV = [("AWS_ACCESS_KEY_ID=", "AKIAIOSFODNN7EXAMPLE"),
       ("AWS_SECRET_ACCESS_KEY=", "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"),
       ("SUPPORT_EMAIL=", "priya.sharma@example.com")]
PALETTES = {  # paper, ink
    "dark-on-light": ((255, 255, 255), (24, 24, 27)),
    "light-on-dark": ((17, 24, 39), (229, 231, 235)),
    "blue-ink-on-cream": ((250, 247, 238), (37, 99, 235)),
    "green-terminal": ((12, 12, 12), (34, 197, 94)),
}


def ink_box(size, xy, text, font):
    """The exact ink bounds of text drawn at xy (textbbox includes bearings, not just ink)."""
    m = Image.new("L", size, 0)
    ImageDraw.Draw(m).text(xy, text, font=font, fill=255)
    return m.getbbox()


@pytest.fixture(scope="module")
def ocr(tmp_path_factory):
    return locate.Ocr(tmp_path_factory.mktemp("ocr"))


def draw_env(paper, ink, size=22):
    """The .env lines, plus each line's label and value ink boxes as drawn."""
    font = ImageFont.truetype(str(MONO), size)
    img = Image.new("RGB", (1100, 40 + 44 * len(ENV)), paper)
    d = ImageDraw.Draw(img)
    boxes = []
    for k, (label, value) in enumerate(ENV):
        x, y = 30, 20 + 44 * k
        d.text((x, y), label + value, font=font, fill=ink)
        boxes.append((ink_box(img.size, (x, y), label, font),
                      ink_box(img.size, (x + font.getlength(label), y), value, font),
                      ink_box(img.size, (x, y), label.rstrip()[:-1], font)))
    return np.array(img), boxes


def label_kept(before, after, lbox, pbox, paper):
    """The label stays: every pixel up to its final separator ('=' or ':') identical, and the separator itself
    still there (at least 70% of its clear ink pixels unchanged; see the xfail below for the rest)."""
    x1, y1, x2, y2 = pbox
    if not np.array_equal(before[y1:y2, x1:x2], after[y1:y2, x1:x2]):
        return False
    x1, y1, x2, y2 = pbox[2], lbox[1], lbox[2], lbox[3]
    a, b = before[y1:y2, x1:x2].astype(int), after[y1:y2, x1:x2].astype(int)
    strong = np.abs(a - np.array(paper)).sum(axis=2) > 90
    return strong.any() and (np.all(a == b, axis=2) & strong).sum() >= 0.7 * strong.sum()


def locate_value(ocr, rgb, value, kind="api_key"):
    hits = [b for ln in ocr.frame(rgb) if (b := locate.span(rgb, ln, value, kind))]
    assert len(hits) == 1, f"{value!r} located {len(hits)} times"
    return hits[0]


def readable(ocr, rgb, value) -> bool:
    v = locate.norm(value)
    return any(v in locate.norm(ln["text"]) or locate._match(ln, value) for ln in ocr.frame(rgb))


def outside(shape, box, pad):
    m = np.ones(shape[:2], bool)
    m[max(0, box[1] - pad):box[3] + pad, max(0, box[0] - pad):box[2] + pad] = False
    return m


def check_cover_colours(before, after, box, paper, ink):
    x1, y1, x2, y2 = box
    patch = after[y1:y2, x1:x2].reshape(-1, 3).astype(int)
    # paper: the cover's background is exactly the line's paper (the most common colour under the cover)
    colours, counts = np.unique(patch, axis=0, return_counts=True)
    assert np.abs(colours[counts.argmax()] - np.array(paper)).max() <= 2
    # ink: the strongest asterisk pixels are the line's own ink colour, not grey and not a fixed black
    far = patch[np.abs(patch - np.array(paper)).sum(axis=1).argmax()]
    assert np.abs(far - np.array(ink)).max() <= 40, (far, ink)
    # and something was actually drawn: asterisks, not a blank patch
    assert (np.abs(patch - np.array(paper)).sum(axis=1) > 60).sum() > 20


@pytest.mark.parametrize("palette", PALETTES, ids=list(PALETTES))
def test_env_values_covered_labels_untouched(ocr, palette):
    paper, ink = PALETTES[palette]
    rgb, drawn = draw_env(paper, ink)
    out = rgb.copy()
    boxes = []
    for (label, value), (lbox, vbox, _) in zip(ENV, drawn):
        b = locate_value(ocr, rgb, value, "email" if "@" in value else "api_key")
        assert b[0] >= lbox[2] - 2, f"cover for {value!r} starts inside its label"
        assert b[0] <= vbox[0] + 1 and b[2] >= vbox[2] - 1, f"cover {b} misses part of {value!r} at {vbox}"
        assert b[1] <= vbox[1] + 1 and b[3] >= vbox[3] - 1
        boxes.append([*b, "api_key"])
    render.cover(out, boxes)
    for (label, value), (lbox, vbox, pbox), b in zip(ENV, drawn, boxes):
        assert label_kept(rgb, out, lbox, pbox, paper), f"label {label!r} changed"
        vx1, vy1, vx2, vy2 = vbox
        changed = np.any(out[vy1:vy2, vx1:vx2] != rgb[vy1:vy2, vx1:vx2], axis=2).mean()
        assert changed > 0.15, f"only {changed:.0%} of {value!r} changed"
        check_cover_colours(rgb, out, b[:4], paper, ink)
    keep = np.logical_and.reduce([outside(rgb.shape, b, render.PAD) for b in boxes])
    assert np.array_equal(out[keep], rgb[keep]), "pixels outside the covers changed"
    for label, value in ENV:
        assert not readable(ocr, out, value), f"{value!r} still readable after cover"


@pytest.mark.xfail(reason="the cover's snap plus 1 px pad clips the last 1-2 ink columns of the '=' in a monospace "
                          "KEY=value line", strict=True)
def test_env_separator_fully_untouched(ocr):
    paper, ink = PALETTES["dark-on-light"]
    rgb, drawn = draw_env(paper, ink)
    out = rgb.copy()
    render.cover(out, [[*locate_value(ocr, rgb, v), "api_key"] for _, v in ENV])
    for lbox, _, _ in drawn:
        x1, y1, x2, y2 = lbox
        assert np.array_equal(out[y1:y2, x1:x2], rgb[y1:y2, x1:x2])


def draw_card():
    """A fake ID card on a dark desktop: coloured header, pale body, a photo with a frame line."""
    img = Image.new("RGB", (900, 520), (30, 41, 59))
    d = ImageDraw.Draw(img)
    card, header, body = (60, 40, 760, 480), (37, 99, 235), (232, 240, 250)
    d.rectangle(card, fill=body)
    d.rectangle((60, 40, 760, 100), fill=header)
    big, f = ImageFont.truetype(str(SANS), 30), ImageFont.truetype(str(SANS), 26)
    d.text((90, 52), "SAMPLE ID CARD", font=big, fill=(255, 255, 255))
    fields = [("Name: ", "TEST USER", (20, 20, 20)), ("ID No: ", "ABCDE1234F", (20, 20, 20)),
              ("Date of birth: ", "01-Jan-1990", (120, 20, 20))]
    drawn = []
    for k, (label, value, ink) in enumerate(fields):
        x, y = 90, 150 + 70 * k
        d.text((x, y), label + value, font=f, fill=ink)
        drawn.append((label, value, ink, ink_box(img.size, (x, y), label, f),
                      ink_box(img.size, (x + f.getlength(label), y), value, f),
                      ink_box(img.size, (x, y), label.rstrip()[:-1], f)))
    photo = (560, 140, 720, 340)
    d.rectangle(photo, fill=(200, 205, 210), outline=(90, 90, 90), width=2)
    d.ellipse((600, 170, 680, 260), fill=(150, 120, 100))
    d.rectangle((585, 265, 695, 338), fill=(60, 80, 120))
    return np.array(img), drawn, body, photo


def test_fake_id_card(ocr):
    rgb, drawn, body, photo = draw_card()
    out = rgb.copy()
    boxes = []
    for label, value, ink, lbox, vbox, _ in drawn:
        b = locate_value(ocr, rgb, value, "id_number")
        assert b[0] >= lbox[2] - 2 and b[0] <= vbox[0] + 1 and b[2] >= vbox[2] - 1
        boxes.append([*b, "id_number"])
    boxes.append([*photo, "photo_of_person"])
    render.cover(out, boxes)
    for (label, value, ink, lbox, vbox, pbox), b in zip(drawn, boxes):
        assert label_kept(rgb, out, lbox, pbox, body), f"label {label!r} changed"
        check_cover_colours(rgb, out, b[:4], body, ink)
        assert not readable(ocr, out, value)
    # the photo becomes one flat patch in the colour just around it (here its frame line), never a blur; the
    # frame line itself, INSET px inside the box, survives
    x1, y1, x2, y2 = photo
    i = render.INSET - render.PAD
    inner = out[y1 + i:y2 - i, x1 + i:x2 - i].reshape(-1, 3)
    assert len(np.unique(inner, axis=0)) == 1
    assert np.abs(inner[0].astype(int) - 90).max() <= 2
    assert np.array_equal(out[y1:y1 + 2, x1:x2], rgb[y1:y1 + 2, x1:x2]), "photo frame line was painted over"
    assert np.array_equal(out[y2 - 1:y2 + 1, x1:x2], rgb[y2 - 1:y2 + 1, x1:x2])
    title = [ln for ln in ocr.frame(out) if "SAMPLE" in ln["text"]]
    assert title, "the document title must stay readable"


def test_ink_and_paper_reads_the_line():
    rgb = np.full((40, 200, 3), (250, 247, 238), np.uint8)
    rgb[15:25, 20:180:4] = (37, 99, 235)  # vertical ink strokes
    paper, ink = render._ink_and_paper(rgb, 10, 10, 190, 30)
    assert np.abs(paper - (250, 247, 238)).max() <= 1
    assert np.abs(ink - (37, 99, 235)).max() <= 1


def test_ink_and_paper_low_contrast_falls_back():
    rgb = np.full((40, 200, 3), 30, np.uint8)
    rgb[15:25, 20:180:4] = 40  # almost invisible text on a dark screen: ink is a light grey guess
    paper, ink = render._ink_and_paper(rgb, 10, 10, 190, 30)
    assert np.abs(paper - 30).max() <= 1 and list(ink) == [200, 200, 200]
    rgb = np.full((40, 200, 3), 230, np.uint8)
    paper, ink = render._ink_and_paper(rgb, 10, 10, 190, 30)
    assert list(ink) == [60, 60, 60]


def test_frame_boxes_only_screen_items_not_rejected():
    items = [{"id": "s1", "where": "screen", "kind": "api_key"},
             {"id": "s2", "where": "screen", "kind": "email", "decision": "rejected"},
             {"id": "a1", "where": "speech", "kind": "phone"}]
    tracks = {"s1": {0: [1.4, 2, 30, 12], 5: [3, 2, 33, 12]}, "s2": {0: [50, 50, 90, 60]}, "a1": {0: [0, 0, 5, 5]},
              "zz": {0: [0, 0, 9, 9]}}
    assert render.frame_boxes(items, tracks) == {0: [[1, 2, 30, 12, "api_key"]], 5: [[3, 2, 33, 12, "api_key"]]}


def test_cover_skips_tiny_and_empty():
    rgb = np.full((20, 20, 3), 128, np.uint8)
    before = rgb.copy()
    render.cover(rgb, [])
    render.cover(rgb, [[5, 5, 5, 5, "api_key"]])  # zero-size box, even after PAD, paints at most 2 px
    assert np.abs(rgb.astype(int) - before).sum(axis=2).astype(bool).sum() <= 4
