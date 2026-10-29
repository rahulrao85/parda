"""Verify the rendered video the way a stranger would see it: scan the OUTPUT again, frames and audio.

Anything still visible or audible comes back as a new item. The caller tracks it on the original video, renders
again and re-verifies, until a pass finds nothing or the pass limit is reached.
"""
import hashlib
from pathlib import Path

from . import sample, screen, speech


def scan(clean: Path, work: Path) -> list[dict]:
    """Secrets still present in the rendered video, as items on the same timeline as the source."""
    frames, info = sample.sample(clean)
    on_screen = screen.find(frames, info, Path("cache/spark"))
    spoken, _ = speech.find(clean, info, work)
    return on_screen + spoken


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
