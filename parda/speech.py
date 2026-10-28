"""Find private things said aloud: Muse Voice Transcribe, then Muse Spark picks the secret phrases, then word timing.

Voice Transcribe gives turn-level times only, so the exact span of a spoken secret comes from forced alignment of
that turn with Meta's open MMS model (torchaudio MMS_FA). Without torch installed, the whole turn is bleeped.
"""
import json, re, subprocess, tempfile, wave
from pathlib import Path

import numpy as np
import requests

from . import api

CHUNK_S = 570  # Voice Transcribe takes up to 10 minutes per request
DIGITS = dict(zip("0123456789", "zero one two three four five six seven eight nine".split()))
DIGIT_WORDS = set(DIGITS.values())
PROMPT = """This is the transcript of a video's audio, split into numbered turns. Find every phrase a stranger
must not hear: phone numbers, emails, government ID numbers, passwords, PINs or OTPs, card or bank numbers, home
addresses, API keys. Numbers are often spoken digit by digit. Do not flag ordinary words.
Reply with JSON only: {"spans": [{"turn": <turn number>, "quote": "<exact words copied from that turn, the full
secret and nothing else>", "kind": "phone|email|id_number|password|payment_card|bank_detail|address|api_key|other",
"reason": "<few words>"}]}
Turns:
"""


def _wav(clip, path: Path, rate: int, start: float = 0, dur: float | None = None) -> None:
    cmd = ["ffmpeg", "-loglevel", "error", "-y", "-ss", str(start), "-i", str(clip)]
    cmd += ["-t", str(dur)] if dur else []
    subprocess.run(cmd + ["-vn", "-ac", "1", "-ar", str(rate), "-c:a", "pcm_s16le", str(path)], check=True)


def transcribe(clip, duration: float, cache: Path) -> list[dict]:
    """Turns with absolute start/end seconds, transcribed in chunks that fit the API limit."""
    cache.mkdir(parents=True, exist_ok=True)
    hit = cache / "transcript.json"
    if hit.exists():
        return json.loads(hit.read_text(encoding="utf-8"))
    turns = []
    with tempfile.TemporaryDirectory() as tmp:
        for start in np.arange(0, duration, CHUNK_S):
            wav = Path(tmp) / "chunk.wav"
            _wav(clip, wav, 24000, start, CHUNK_S)
            r = requests.post(f"{api.BASE}/asr/transcribe", headers={"Authorization": f"Bearer {api.KEY}"},
                              timeout=300, files={
                                  "request": (None, json.dumps({"mode": "ENDPOINTING", "audioEncoding": "WAV",
                                                                "model": "muse-voice-transcribe-1.0"}), "application/json"),
                                  "audio": ("chunk.wav", wav.read_bytes(), "audio/wav")})
            r.raise_for_status()
            j = r.json()
            api.log_cost("muse-voice-transcribe-1.0", j.get("audioDurationMs", 0) / 3.6e6 * 0.18)
            for tr in j.get("turns", []):
                turns.append({"start_s": start + tr["startMs"] / 1000, "end_s": start + tr["endMs"] / 1000,
                              "text": tr["transcript"]})
    hit.write_text(json.dumps(turns, indent=1, ensure_ascii=False), encoding="utf-8")
    return turns


def _words(text: str) -> list[str]:
    """Alignment words: lowercase letters only, digits spelled out one by one (that is how they are spoken)."""
    out = []
    for m in re.finditer(r"[A-Za-z']+|\d", text):
        w = DIGITS.get(m.group(), m.group().lower().replace("'", ""))
        if w:
            out.append(w)
    return out


class Aligner:
    """MMS forced alignment, loaded once. Returns per-word (start_s, end_s) for a turn, or None without torch."""

    def __init__(self):
        try:
            import torch, torchaudio
            self.torch, self.bundle = torch, torchaudio.pipelines.MMS_FA
            self.model = self.bundle.get_model(with_star=False)
            self.tok, self.align = self.bundle.get_tokenizer(), self.bundle.get_aligner()
        except Exception:
            self.model = None

    def words(self, clip, turn: dict, words: list[str]) -> list[tuple[float, float]] | None:
        if self.model is None or not words:
            return None
        with tempfile.TemporaryDirectory() as tmp:
            wav = Path(tmp) / "turn.wav"
            _wav(clip, wav, 16000, turn["start_s"], turn["end_s"] - turn["start_s"] + 0.3)
            with wave.open(str(wav)) as w:
                pcm = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768
        wav_t = self.torch.from_numpy(pcm).unsqueeze(0)
        with self.torch.inference_mode():
            emission, _ = self.model(wav_t)
            spans = self.align(emission[0], self.tok(words))
        sec = wav_t.size(1) / emission.size(1) / 16000
        return [(turn["start_s"] + s[0].start * sec, turn["start_s"] + s[-1].end * sec) for s in spans]


def find(clip, info: dict, cache: Path) -> tuple[list[dict], list[dict]]:
    """Spoken secrets as items with start/end seconds, plus the transcript turns."""
    cache.mkdir(parents=True, exist_ok=True)
    turns = transcribe(clip, info["duration"], cache)
    if not turns:
        return [], turns
    listing = "\n".join(f"{i}: {t['text']}" for i, t in enumerate(turns))
    reply = api.spark_json([{"type": "text", "text": PROMPT + listing}])
    aligner, items = None, []
    for sp in reply.get("spans", []):
        i = sp.get("turn")
        if not (isinstance(i, int) and 0 <= i < len(turns)):
            continue
        turn, quote = turns[i], _words(sp.get("quote", ""))
        start, end, timing = turn["start_s"], turn["end_s"], "turn"
        aligner = aligner or Aligner()
        all_words = _words(turn["text"])
        at = next((k for k in range(len(all_words) - len(quote) + 1)
                   if quote and all_words[k:k + len(quote)] == quote), None)
        if at is not None:  # Spark sometimes stops a digit-by-digit number early; take the whole run of digits
            n = len(quote)
            while at + n < len(all_words) and all_words[at + n] in DIGIT_WORDS:
                n += 1
            while at > 0 and all_words[at - 1] in DIGIT_WORDS and all_words[at] in DIGIT_WORDS:
                at, n = at - 1, n + 1
            quote = all_words[at:at + n]
        times = aligner.words(clip, turn, all_words) if at is not None else None
        if times:
            start, end, timing = times[at][0] - 0.15, times[at + len(quote) - 1][1] + 0.15, "word"
        text = " ".join(quote) if at is not None else sp.get("quote", "")
        items.append({"id": f"a{len(items) + 1}", "kind": sp.get("kind", "other"), "text": text,
                      "reason": sp.get("reason", ""), "where": "speech", "start_s": round(max(0, start), 2),
                      "end_s": round(end, 2), "timing": timing})
    return items, turns
