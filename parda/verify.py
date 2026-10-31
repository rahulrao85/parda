"""Verify the rendered video the way a stranger would see it: scan the OUTPUT again, frames and audio.

Anything still visible or audible comes back as a new item. The caller tracks it on the original video, renders
again and re-verifies, until a pass finds nothing or the pass limit is reached.
"""
import hashlib, re
from pathlib import Path

from . import sample, screen, speech

# Backstop for the verify prompt: a finding that describes a cover ("redacted value", "dots") is not a leak.
COVER_WORDS = re.compile(r"redact|masked|asterisk|\*{3,}|\bdots?\b|●|hidden|private document|\bcover", re.I)


def scan(clean: Path, work: Path) -> list[dict]:
    """Secrets still present in the rendered video, as items on the same timeline as the source."""
    frames, info = sample.sample(clean)
    on_screen = [it for it in screen.find(frames, info, Path("cache/spark"), verify=True)
                 if not COVER_WORDS.search(f"{it['text']} {it.get('reason', '')}")]
    spoken, _ = speech.find(clean, info, work)
    return on_screen + spoken


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
