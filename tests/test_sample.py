"""parda/sample.py on clips built with ffmpeg testsrc + drawtext: which frames are kept, at which exact times."""
import subprocess

import numpy as np
import pytest

from conftest import SANS, needs_ffmpeg
from parda import sample

pytestmark = needs_ffmpeg


def make_clip(path, *, seconds=3, fps=30, size="1280x720", counter=True, still=False):
    src = f"color=c=0x1e293b:s={size}:r={fps}:d={seconds}" if still else f"testsrc=s={size}:r={fps}:d={seconds}"
    vf = []
    if counter:  # a big frame number, so every frame differs from the last
        font = f"fontfile={SANS}:" if SANS.exists() else ""
        vf.append(f"drawtext={font}text='%{{frame_num}}':x=40:y=40:fontsize=120:fontcolor=white:box=1:boxcolor=black")
    cmd = ["ffmpeg", "-loglevel", "error", "-y", "-f", "lavfi", "-i", src]
    cmd += ["-vf", ",".join(vf)] if vf else []
    subprocess.run(cmd + ["-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", str(path)], check=True)
    return path


@pytest.fixture(scope="module")
def clip30(tmp_path_factory):
    return make_clip(tmp_path_factory.mktemp("clips") / "c30.mp4")


def test_probe(clip30):
    info = sample.probe(clip30)
    assert info["w"] == 1280 and info["h"] == 720 and info["fps"] == 30.0
    assert info["duration"] == pytest.approx(3.0, abs=0.05)


def test_sample_30fps_rate_2(clip30):
    frames, info = sample.sample(clip30, rate=2.0)
    assert [f.t for f in frames] == [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]
    assert all(f.rgb.shape == (720, 1280, 3) and f.rgb.dtype == np.uint8 for f in frames)
    assert {k: info[k] for k in ("w", "h", "fps", "analysis_w", "analysis_h", "sampled", "kept", "rate")} == {
        "w": 1280, "h": 720, "fps": 30.0, "analysis_w": 1280, "analysis_h": 720, "sampled": 6, "kept": 6, "rate": 2.0}


def test_kept_frames_are_the_exact_source_frames(clip30):
    """Frame k of the sample is source frame k*step (not ffmpeg's fps filter, which keeps the last of a slot)."""
    frames, _ = sample.sample(clip30, rate=2.0)
    src = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", str(clip30), "-f", "rawvideo", "-pix_fmt", "rgb24",
                          "-"], capture_output=True, check=True).stdout
    src = np.frombuffer(src, np.uint8).reshape(-1, 720, 1280, 3)
    assert len(src) == 90
    for k, f in enumerate(frames):
        n = k * 15
        diffs = [np.abs(f.rgb.astype(int) - src[m]).mean() for m in range(max(0, n - 2), min(90, n + 3))]
        assert int(np.argmin(diffs)) + max(0, n - 2) == n
        assert min(diffs) < 0.5


def test_sample_rate_4_and_downscale(clip30):
    frames, info = sample.sample(clip30, rate=4.0, max_w=640)
    # step = round(30 / 4) = round(7.5) = 8 (Python rounds half to even), so the real rate is 3.75/s
    assert [f.t for f in frames] == [round(k * 8 / 30, 3) for k in range(12)]
    assert frames[1].t == 0.267
    assert info["rate"] == 3.75 and info["sampled"] == 12
    assert (info["analysis_w"], info["analysis_h"]) == (640, 360)
    assert frames[0].rgb.shape == (360, 640, 3)


def test_sample_25fps_half_step(tmp_path):
    clip = make_clip(tmp_path / "c25.mp4", seconds=2, fps=25, size="1000x562")
    frames, info = sample.sample(clip, rate=2.0, max_w=640)
    # step = round(12.5) = 12; analysis size stays even: 1000x562 * 0.64 -> 640x359.68 -> 640x358
    assert [f.t for f in frames] == [0.0, 0.48, 0.96, 1.44, 1.92]
    assert info["rate"] == pytest.approx(25 / 12)
    assert (info["analysis_w"], info["analysis_h"]) == (640, 358)


def test_still_screen_keeps_one_frame(tmp_path):
    clip = make_clip(tmp_path / "still.mp4", counter=False, still=True)
    frames, info = sample.sample(clip, rate=2.0)
    assert [f.t for f in frames] == [0.0]
    assert info["sampled"] == 6 and info["kept"] == 1


def test_changed_blocks():
    a = np.full((sample.THUMB_H, sample.THUMB_W), 100.0)
    assert not sample.changed(a, a.copy())
    b = a.copy()
    b[:sample.BLOCK, :sample.BLOCK] += 5  # one cell, 5 grey levels: a new line of text in a terminal counts
    assert sample.changed(a, b)
    c = a + 3.0  # everything a little brighter (codec noise, a fade) does not
    assert not sample.changed(a, c)


def test_thumb_shape():
    rgb = np.zeros((720, 1280, 3), np.uint8)
    rgb[:, 640:] = 255
    th = sample._thumb(rgb)
    assert th.shape == (sample.THUMB_H, sample.THUMB_W)
    assert th[0, 0] == 0 and th[0, -1] == 255
