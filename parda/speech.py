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
# Number words as Hindi transcripts write them: English words in Devanagari, and Hindi numerals (romanised as said).
HI_NUM = {"नाइन": "nine", "एट": "eight", "सेवन": "seven", "सिक्स": "six", "फाइव": "five", "फोर": "four",
          "थ्री": "three", "टू": "two", "वन": "one", "जीरो": "zero", "जेरो": "zero", "शून्य": "shunya", "एक": "ek",
          "दो": "do", "तीन": "teen", "चार": "char", "पांच": "paanch", "पाँच": "paanch", "छह": "chhah", "सात": "saat",
          "आठ": "aath", "नौ": "nau"}
DIGIT_WORDS = set(DIGITS.values()) | set(HI_NUM.values())
DEV_CONS = dict(zip("कखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसह",
                    "k kh g gh n ch chh j jh n t th d dh n t th d dh n p ph b bh m y r l v sh sh s h".split()))
DEV_SIGNS = dict(zip("ािीुूृेैोौ", "a i ee u oo ri e ai o au".split()))
DEV_VOWELS = dict(zip("अआइईउऊऋएऐओऔंँः", "a aa i ee u oo ri e ai o au n n h".split()))
PROMPT = """This is the transcript of a video's audio, split into numbered turns. Find every phrase a stranger
must not hear: phone numbers, emails, government ID numbers, passwords, PINs or OTPs, card or bank numbers, home
addresses, API keys. Numbers are often spoken digit by digit, and a Hindi or Hinglish transcript may write
them in Devanagari ("नाइन एट सेवन" or "एक दो तीन"). Do not flag ordinary words.
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


def _roman(word: str) -> str:
    """Rough Devanagari -> Latin, enough for the aligner to hear Hinglish ("नंबर" -> "nambar")."""
    out, i = [], 0
    while i < len(word):
        ch = word[i]
        if ch in DEV_CONS:
            out.append(DEV_CONS[ch])
            nxt = word[i + 1] if i + 1 < len(word) else ""
            if nxt == "़":  # nukta: same sound for our purpose
                i, nxt = i + 1, word[i + 2] if i + 2 < len(word) else ""
            if nxt in DEV_SIGNS:
                out.append(DEV_SIGNS[nxt]); i += 1
            elif nxt == "्":  # virama: no vowel
                i += 1
            elif i + 1 < len(word):
                out.append("a")  # inherent vowel, dropped at the end of a word as Hindi does
        else:
            out.append(DEV_VOWELS.get(ch, ""))
        i += 1
    return "".join(out)


def _words(text: str) -> list[str]:
    """Alignment words: lowercase Latin letters only, digits spelled out one by one (that is how they are spoken),
    Devanagari romanised. Hinglish transcripts often write English number words in Devanagari ("नाइन एट")."""
    out = []
    for m in re.finditer(r"[A-Za-z']+|\d|[ऀ-ॿ]+", text):
        g = m.group()
        if g[0] >= "ऀ":
            w = HI_NUM.get(g.replace("़", "")) or _roman(g)
        else:
            w = DIGITS.get(g, g.lower().replace("'", ""))
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


_DOT = r"\s*(?:\.|\bdot\b)\s*"
EMAIL = re.compile(rf"\w+(?:{_DOT}\w+)*\s*(?:@|\bat\b)\s*[\w-]+(?:{_DOT}[\w-]+)*{_DOT}(?:com|in|org|net|co|io|ai)\b", re.I)


def _rules(turns: list[dict]) -> list[tuple]:
    """Deterministic backstop: an email pattern or a run of 5+ spoken digits is always a secret, whatever Spark says.
    A privacy check cannot rest on one model call, which does not answer the same way every time."""
    out = []
    for i, t in enumerate(turns):
        for m in EMAIL.finditer(t["text"]):
            out.append((i, _words(m.group()), "email", "email pattern (rule)"))
        words, k = _words(t["text"]), 0
        while k < len(words):
            n = 0
            while k + n < len(words) and words[k + n] in DIGIT_WORDS:
                n += 1
            if n >= 5:
                out.append((i, words[k:k + n], "number", f"{n} spoken digits (rule)"))
            k += max(n, 1)
    return out


def find(clip, info: dict, cache: Path) -> tuple[list[dict], list[dict]]:
    """Spoken secrets as items with start/end seconds, plus the transcript turns."""
    cache.mkdir(parents=True, exist_ok=True)
    turns = transcribe(clip, info["duration"], cache)
    if not turns:
        return [], turns
    listing = "\n".join(f"{i}: {t['text']}" for i, t in enumerate(turns))
    reply = api.spark_json([{"type": "text", "text": PROMPT + listing}])
    found = [(sp.get("turn"), _words(sp.get("quote", "")), sp.get("kind", "other"), sp.get("reason", ""))
             for sp in reply.get("spans", [])]
    found += _rules(turns)
    aligner, items, taken = None, [], set()
    for i, quote, kind, reason in found:
        if not (isinstance(i, int) and 0 <= i < len(turns)):
            continue
        turn, all_words = turns[i], _words(turns[i]["text"])
        start, end, timing = turn["start_s"], turn["end_s"], "turn"
        at = next((k for k in range(len(all_words) - len(quote) + 1)
                   if quote and all_words[k:k + len(quote)] == quote), None)
        if at is not None:  # Spark sometimes stops a digit-by-digit number early; take the whole run of digits
            n = len(quote)
            while at + n < len(all_words) and all_words[at + n] in DIGIT_WORDS:
                n += 1
            while at > 0 and all_words[at - 1] in DIGIT_WORDS and all_words[at] in DIGIT_WORDS:
                at, n = at - 1, n + 1
            quote = all_words[at:at + n]
            span = {(i, k) for k in range(at, at + n)}
            if span & taken:  # Spark and a rule found the same words
                continue
            taken |= span
        aligner = aligner or Aligner()
        times = aligner.words(clip, turn, all_words) if at is not None else None
        if times:
            start, end, timing = times[at][0] - 0.15, times[at + len(quote) - 1][1] + 0.15, "word"
        items.append({"id": f"a{len(items) + 1}", "kind": kind, "text": " ".join(quote), "reason": reason,
                      "where": "speech", "start_s": round(max(0, start), 2), "end_s": round(end, 2),
                      "timing": timing})
    return items, turns
