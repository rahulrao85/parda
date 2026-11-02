"""Opt-in tests. `slow` needs the torchaudio MMS_FA model (a download on first use); `api` needs a real Meta key.
Neither runs in the default suite: pytest -m "not api and not slow"."""
import pytest

from conftest import SAMPLES, needs_ffmpeg

CLIP = SAMPLES / "fake_pan_12s.mp4"
# The one transcript turn of the sample clip (fake data), as Voice Transcribe returned it.
TURN = {"start_s": 0.06, "end_s": 10.22,
        "text": "Okay, quick note before I forget. Call me on nine eight seven six five four three two one zero "
                "or email priya.charma at example.com"}


@pytest.mark.slow
@needs_ffmpeg
def test_mms_alignment_times_the_spoken_number():
    pytest.importorskip("torchaudio")
    from parda import speech
    al = speech.Aligner()
    assert al.model is not None, "torchaudio MMS_FA did not load"
    words = speech._words(TURN["text"])
    times = al.words(CLIP, TURN, words)
    assert times is not None and len(times) == len(words)
    starts = [a for a, _ in times]
    assert starts == sorted(starts) and all(a <= b for a, b in times)
    k = words.index("nine")
    nine, zero = times[k], times[k + 9]
    # the fixture's word-timed bleep for this number is 3.8-7.3 s (0.15 s padding included)
    assert 3.0 < nine[0] < 4.8 and 6.2 < zero[1] < 8.0
    assert times[-1][1] <= TURN["end_s"] + 0.3


@pytest.mark.api
def test_spark_answers_json():
    from parda import api  # loads .env, like the CLI
    if not api.KEY:
        pytest.skip("needs META_API_KEY (environment or .env)")
    reply = api.spark_json([{"type": "text", "text": 'Reply with exactly this JSON and nothing else: {"ok": true}'}])
    assert reply.get("ok") is True
    assert api.COST_LOG and api.COST_LOG[-1]["model"] == api.SPARK

