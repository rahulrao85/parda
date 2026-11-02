"""Offline end to end: redact samples/fake_pan_12s.mp4 from a recorded flags.json, with the network blocked.

The fixture is one real `find` run on the sample clip (screen items s1-s3, speech items a1/a2 that already carry
word timing, nothing non-text, so no SAM call). redact() then only runs OCR locate and render, both local."""
import json, shutil, socket
from pathlib import Path

import numpy as np
import pytest

from conftest import FIXTURES, SAMPLES, NetworkBlocked, frame_count, frames_at, needs_ffmpeg

pytestmark = needs_ffmpeg

CLIP = SAMPLES / "fake_pan_12s.mp4"
W, H, FPS, N = 1280, 720, 30, 360
CHECK_FRAMES = [0, 75, 149, 150, 240, 359]


def _refuse(*a, **k):
    raise NetworkBlocked("network access in an offline test")


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    from parda import api, screen, speech, track
    from parda.__main__ import redact
    work = tmp_path_factory.mktemp("e2e")
    out = work / "out"
    out.mkdir()
    shutil.copy(FIXTURES / "fake_pan_12s.flags.json", out / "flags.json")
    with pytest.MonkeyPatch.context() as mp:
        mp.chdir(work)  # locate keeps its OCR cache in ./cache/ocr
        for name in ("connect", "connect_ex"):
            mp.setattr(socket.socket, name, _refuse)
        mp.setattr(socket, "create_connection", _refuse)
        mp.setattr(socket, "getaddrinfo", _refuse)
        # and no model at all, even one that would not need a socket to fail
        for mod, name in ((api, "spark_json"), (track, "track"), (screen, "find"), (speech, "find"),
                          (speech, "transcribe")):
            mp.setattr(mod, name, _refuse)
        mp.setattr(api, "COST_LOG", [])
        audit = redact(CLIP, out, verify_passes=0)
    tracks = {k: {int(f): b for f, b in v.items()} for k, v in json.loads((out / "tracks.json").read_text()).items()}
    return {"out": out, "audit": audit, "tracks": tracks,
            "src": frames_at(CLIP, set(CHECK_FRAMES), W, H), "clean": frames_at(out / "clean.mp4", set(CHECK_FRAMES), W, H)}


@pytest.fixture(scope="module")
def truth():
    flags = json.loads((FIXTURES / "fake_pan_12s.flags.json").read_text(encoding="utf-8"))
    by_text = {it["text"]: it["id"] for it in flags["items"] if it["where"] == "screen"}
    secrets = json.loads((SAMPLES / "fake_pan_12s.truth.json").read_text())["secrets"]
    screen = [dict(s, id=by_text[s["text"]]) for s in secrets if s["where"] == "screen"]
    screen.append({"text": "TEST USER", "id": by_text["TEST USER"], "start_s": 0.0, "end_s": 12})
    return screen


def test_audit_and_output(run):
    a = run["audit"]
    assert a["verdict"] == "unverified" and a["verify"] == [] and a["models"] == []
    assert a["render"]["frames"] == N and a["render"]["frames_covered"] == N
    assert a["render"]["bleeps"] == [[3.8, 7.3], [8.15, 10.59]]
    assert len(a["sha256"]) == 64
    assert {it["id"] for it in a["items"]} == {"s1", "s2", "s3", "a1", "a2"}
    assert frame_count(run["out"] / "clean.mp4") == N
    assert json.loads((run["out"] / "cost.json").read_text())["usd"] == 0


def test_every_truth_frame_has_a_box(run, truth):
    for s in truth:
        want = set(range(round(s["start_s"] * FPS), min(N, round(s["end_s"] * FPS))))
        got = set(run["tracks"].get(s["id"], {}))
        assert want <= got, f"{s['text']}: {len(want - got)} frames uncovered, first {min(want - got)}"
        assert not got - set(range(round(s["start_s"] * FPS) - 1, N)), f"{s['text']} covered before it appears"


def test_value_regions_changed(run, truth):
    for n in CHECK_FRAMES:
        src, clean = run["src"][n].astype(int), run["clean"][n].astype(int)
        mask = np.zeros((H, W), bool)
        for s in truth:
            box = run["tracks"].get(s["id"], {}).get(n)
            if box is None:
                continue
            x1, y1, x2, y2 = box
            diff = np.abs(src[y1:y2, x1:x2] - clean[y1:y2, x1:x2]).mean()
            assert diff > 15, f"frame {n}: {s['text']} region barely changed ({diff:.1f})"
            mask[max(0, y1 - 2):y2 + 2, max(0, x1 - 2):x2 + 2] = True
        rest = np.abs(src - clean)[~mask].mean()
        assert rest < 1.5, f"frame {n}: pixels outside the covers changed ({rest:.2f})"


def test_values_unreadable_labels_readable(run, truth, tmp_path):
    from parda import locate
    ocr = locate.Ocr(tmp_path / "ocr")
    for n in (0, 240):
        src_text = " ".join(ln["text"] for ln in ocr.frame(run["src"][n]))
        clean_lines = ocr.frame(run["clean"][n])
        clean_text = " ".join(ln["text"] for ln in clean_lines)
        for s in truth:
            if s["start_s"] * FPS > n:
                continue
            assert locate.norm(s["text"]) in locate.norm(src_text)  # the source really shows it
            assert locate.norm(s["text"]) not in locate.norm(clean_text), f"frame {n}: {s['text']} readable"
            assert not any(locate._match(ln, s["text"]) for ln in clean_lines)
        assert "name" in locate.norm(clean_text) and "permanentaccountnumber" in locate.norm(clean_text)
        if n >= 150:
            assert "awsaccesskeyid" in locate.norm(clean_text)
