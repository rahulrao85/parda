"""One small call to each Meta model: confirms access, shows the response shape, and logs what it cost.

    python scripts/probe_models.py samples/fake_pan_10s.mp4 [spark|transcribe|sam ...]

Raw responses land in cache/probe_<model>.txt for reading.
"""
import base64, json, os, subprocess, sys, tempfile, time
from pathlib import Path

import requests
from dotenv import find_dotenv, load_dotenv
from openai import OpenAI

load_dotenv(find_dotenv(usecwd=True))
KEY = os.getenv("META_API_KEY") or os.getenv("META_API")
BASE = "https://api.meta.ai/v1"
CACHE = Path("cache"); CACHE.mkdir(exist_ok=True)
client = OpenAI(api_key=KEY, base_url=BASE)
# Fake test data only, so the cheap Contributor tier is fine here. Real footage never goes to Contributor models.
SPARK = os.getenv("PARDA_SPARK_MODEL", "muse-spark-1.3-contributor")


def frame_png(clip: Path, at: float) -> bytes:
    return subprocess.run(["ffmpeg", "-loglevel", "error", "-ss", str(at), "-i", str(clip), "-frames:v", "1",
                           "-f", "image2pipe", "-vcodec", "png", "-"], capture_output=True, check=True).stdout


def probe_spark(clip: Path) -> None:
    img = base64.b64encode(frame_png(clip, 6)).decode()
    prompt = ("You review screen-recording frames before they are published. List every item a viewer should not "
              "see: ID numbers, API keys, passwords, emails, phone numbers, addresses, bank details, personal "
              "documents. Reply with JSON only: {\"flags\": [{\"kind\": str, \"text\": str, "
              "\"box\": [x1, y1, x2, y2] in pixels of this 1280x720 frame, \"reason\": str}]}")
    t = time.time()
    r = client.chat.completions.create(model=SPARK, messages=[{"role": "user", "content": [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img}"}}]}])
    out = r.choices[0].message.content
    (CACHE / "probe_spark.txt").write_text(out, encoding="utf-8")
    u = r.usage
    print(f"[spark] {SPARK} {time.time() - t:.1f}s in={u.prompt_tokens} out={u.completion_tokens}")
    print(out[:1200])


def probe_transcribe(clip: Path) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "a.wav"
        subprocess.run(["ffmpeg", "-loglevel", "error", "-i", str(clip), "-ac", "1", "-ar", "24000",
                        "-c:a", "pcm_s16le", str(wav)], check=True)
        t = time.time()
        r = requests.post(f"{BASE}/asr/transcribe", headers={"Authorization": f"Bearer {KEY}"}, timeout=120,
                          files={"request": (None, json.dumps({"mode": "ENDPOINTING",
                                 "model": "muse-voice-transcribe-1.0", "audioEncoding": "WAV"}), "application/json"),
                                 "audio": ("a.wav", wav.read_bytes(), "audio/wav")})
    (CACHE / "probe_transcribe.txt").write_text(r.text, encoding="utf-8")
    print(f"[transcribe] HTTP {r.status_code} {time.time() - t:.1f}s")
    print(r.text[:1200])


def probe_sam(clip: Path) -> None:
    vid = base64.b64encode(clip.read_bytes()).decode()
    t, lines = time.time(), []
    stream = client.responses.create(model="sam-3.1", stream=True, input=[{"role": "user", "content": [
        {"type": "input_text", "text": "document window"},
        {"type": "input_video", "video_url": f"data:video/mp4;base64,{vid}"}]}])
    for ev in stream:
        if ev.type == "response.output_text.delta":
            lines.append(ev.delta)
    raw = "".join(lines)
    (CACHE / "probe_sam.txt").write_text(raw, encoding="utf-8")
    frames = [l for l in raw.splitlines() if l.strip()]
    print(f"[sam] {time.time() - t:.1f}s frames_with_objects={len(frames)} chars={len(raw)}")
    for l in frames[:2] + frames[-1:]:
        print("  ", l[:160])


if __name__ == "__main__":
    clip = Path(sys.argv[1])
    for name in sys.argv[2:] or ["spark", "transcribe", "sam"]:
        try:
            globals()[f"probe_{name}"](clip)
        except Exception as e:  # keep probing the other models
            print(f"[{name}] FAILED: {type(e).__name__}: {str(e)[:600]}")
