"""parda/screen.py pure helpers: how Spark's flags are normalised and merged into one item per secret."""
import pytest

from parda.screen import _iou, _merge, _norm, _same


@pytest.mark.parametrize("raw, want", [
    ("AKIAIOSFODNN7EXAMPLE", "akiaiosfodnn7example"),
    ("Priya.Sharma@Example.com", "priya.sharma@example.com"),
    ("ABCDE 1234 F", "abcde1234f"),
    ("+91 98765-43210", "919876543210"),
    ("wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY", "wjalrxutnfemik7mdengbpxrficyexamplekey"),
    ("", ""),
])
def test_norm(raw, want):
    assert _norm(raw) == want


@pytest.mark.parametrize("a, b, want", [
    ([0, 0, 10, 10], [0, 0, 10, 10], 1.0),
    ([0, 0, 10, 10], [5, 0, 15, 10], 50 / 150),
    ([0, 0, 10, 10], [10, 0, 20, 10], 0.0),       # touching edges share no area
    ([0, 0, 10, 10], [20, 20, 30, 30], 0.0),
    ([0, 0, 10, 10], [2, 2, 4, 4], 4 / 100),      # nested
    ([5, 5, 5, 5], [5, 5, 5, 5], 0.0),            # zero area: no division by zero
])
def test_iou(a, b, want):
    assert _iou(a, b) == pytest.approx(want)
    assert _iou(b, a) == pytest.approx(want)


def item(kind, text, *seen):
    return {"kind": kind, "text": text, "seen": [{"t": t, "box": box} for t, box in seen]}


def test_same_needs_same_kind():
    a, b = item("api_key", "AKIAIOSFODNN7EXAMPLE", (0, [0, 0, 9, 9])), item("password", "AKIAIOSFODNN7EXAMPLE", (0, [0, 0, 9, 9]))
    assert not _same(a, b, {_norm(a["text"])}, {_norm(b["text"])})


def test_same_by_containment_and_near_identical_wording():
    a = item("id_number", "PAN card", (0, [0, 0, 9, 9]))
    b = item("id_number", "Sample PAN card", (9, [500, 500, 600, 600]))
    assert _same(a, b, {"pancard"}, {"samplepancard"})
    # one character misread in a long key, far apart in time and space
    a = item("api_key", "AKIAIOSFODNN7EXAMPLE", (0, [0, 0, 9, 9]))
    b = item("api_key", "AKIAI0SFODNN7EXAMPLE", (9, [500, 500, 600, 600]))
    assert _same(a, b, {_norm(a["text"])}, {_norm(b["text"])})


def test_same_short_wordings_do_not_merge_by_text():
    a = item("person_name", "Raj", (0, [0, 0, 9, 9]))
    b = item("person_name", "Raja", (5, [500, 500, 600, 600]))
    assert not _same(a, b, {"raj"}, {"raja"})


def test_same_by_box_at_the_same_moment():
    a = item("other", "photo", (1.0, [100, 100, 200, 200]))
    b = item("other", "face", (1.5, [105, 100, 205, 200]))
    assert _same(a, b, {"photo"}, {"face"})
    late = item("other", "face", (3.0, [105, 100, 205, 200]))  # same box, but more than 1 s later
    assert not _same(a, late, {"photo"}, {"face"})
    moved = item("other", "face", (1.5, [300, 300, 400, 400]))
    assert not _same(a, moved, {"photo"}, {"face"})


def test_merge_groups_variants_and_sorts_sightings():
    items = [
        item("api_key", "AKIAIOSFODNN7EXAMPLE", (2.0, [0, 0, 9, 9])),
        item("id_number", "ABCDE1234F", (0.0, [50, 50, 90, 60])),
        item("api_key", "AKIAI0SFODNN7EXAMPLE", (0.5, [0, 0, 9, 9])),
        item("api_key", "aws key AKIAIOSFODNN7EXAMPLE", (1.0, [0, 0, 9, 9])),
    ]
    out = _merge(items)
    assert [it["text"] for it in out] == ["AKIAIOSFODNN7EXAMPLE", "ABCDE1234F"]
    assert [s["t"] for s in out[0]["seen"]] == [0.5, 1.0, 2.0]
    assert len(out[1]["seen"]) == 1


def test_merge_keeps_distinct_secrets_apart():
    items = [item("email", "priya.sharma@example.com", (0, [0, 0, 9, 9])),
             item("email", "rahul.verma@example.org", (0, [0, 50, 9, 59]))]
    assert len(_merge(items)) == 2
    assert _merge([]) == []
