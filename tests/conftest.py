"""Shared helpers. The default suite (-m "not api and not slow") must never reach the network or a model."""
import os, shutil, socket, subprocess
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "samples"
FIXTURES = Path(__file__).resolve().parent / "fixtures"
FONT_DIR = Path("/usr/share/fonts/truetype/dejavu")
MONO = FONT_DIR / "DejaVuSansMono.ttf"
SANS = FONT_DIR / "DejaVuSans.ttf"

needs_ffmpeg = pytest.mark.skipif(not (shutil.which("ffmpeg") and shutil.which("ffprobe")), reason="needs ffmpeg")
needs_font = pytest.mark.skipif(not MONO.exists(), reason="needs DejaVu fonts (apt install fonts-dejavu-core)")


class NetworkBlocked(RuntimeError):
    pass


@pytest.fixture
def no_network(monkeypatch):
    """Any socket connection from this process raises (ffmpeg subprocesses only read local files)."""
    def refuse(*a, **k):
        raise NetworkBlocked("network access in an offline test")
    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket.socket, "connect_ex", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(socket, "getaddrinfo", refuse)


def frames_at(clip: Path, wanted: set[int], w: int, h: int) -> dict[int, np.ndarray]:
    """Decode a clip on the same constant-rate clock as locate/render and keep only the wanted frame numbers."""
    fps = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=avg_frame_rate", "-of", "csv=p=0", str(clip)],
                         capture_output=True, text=True, check=True).stdout.strip()
    proc = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-i", str(clip), "-fps_mode", "cfr", "-r", fps,
                             "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
    out, n, size = {}, 0, w * h * 3
    while (chunk := proc.stdout.read(size)) and len(chunk) == size:
        if n in wanted:
            out[n] = np.frombuffer(chunk, np.uint8).reshape(h, w, 3).copy()
        n += 1
    proc.stdout.close()
    proc.wait()
    return out


def frame_count(clip: Path) -> int:
    out = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries",
                          "stream=nb_read_frames", "-of", "csv=p=0", str(clip)],
                         capture_output=True, text=True, check=True).stdout
    return int(out.strip())
