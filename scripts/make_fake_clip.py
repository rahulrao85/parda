"""Generate a test clip with planted FAKE secrets, plus the ground truth Parda should find.

The clip imitates a screen recording: a dark desktop, a document window that drifts across the screen (so masks
have to follow it), a terminal line that appears halfway, and a voice that says a dummy phone number and email.
Every value is fake: ABCDE1234F is the sample PAN used on Indian tax forms, AKIAIOSFODNN7EXAMPLE is AWS's published
example key, and example.com is a reserved domain.

    python scripts/make_fake_clip.py samples/fake_pan_10s.mp4
"""
import json, subprocess, sys, tempfile
from pathlib import Path

W, H, SECONDS, FPS = 1280, 720, 10, 30
FONT = "C\\:/Windows/Fonts/consola.ttf"
SPEECH = ("Okay, quick note before I forget. Call me on nine eight seven six five, four three two one zero, "
          "or email priya dot sharma at example dot com.")
TRUTH = [
    {"kind": "id_document", "text": "ABCDE1234F", "where": "screen", "start_s": 0.0, "end_s": SECONDS},
    {"kind": "api_key", "text": "AKIAIOSFODNN7EXAMPLE", "where": "screen", "start_s": 5.0, "end_s": SECONDS},
    {"kind": "phone", "text": "98765 43210", "where": "speech"},
    {"kind": "email", "text": "priya.sharma@example.com", "where": "speech"},
]


def say(text: str, wav: Path) -> None:
    """Offline TTS through Windows SAPI, so test clips cost nothing to make."""
    ps = ("Add-Type -AssemblyName System.Speech; $s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
          f"$s.SetOutputToWaveFile('{wav}'); $s.Speak(\"{text}\"); $s.Dispose()")
    subprocess.run(["powershell", "-NoProfile", "-Command", ps], check=True)


def text(t: str, x: str, y: str, size: int, color: str, enable: str = "") -> str:
    t = t.replace(":", "\\:")
    f = f"drawtext=fontfile='{FONT}':text='{t}':x={x}:y={y}:fontsize={size}:fontcolor={color}"
    return f + (f":enable='{enable}'" if enable else "")


def main(out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    win_x = f"80+t*{(W - 560 - 160) / SECONDS:.1f}"  # document window drifts left to right
    window = ",".join([
        "drawbox=x=0:y=0:w=560:h=40:color=0x2563eb:t=fill",
        text("PAN_card_scan.pdf", "16", "10", 22, "white"),
        text("INCOME TAX DEPARTMENT  (SAMPLE)", "24", "70", 22, "black"),
        text("Name: TEST USER", "24", "120", 24, "black"),
        text("Permanent Account Number", "24", "180", 20, "0x555555"),
        text("ABCDE1234F", "24", "210", 44, "black"),
    ])
    desktop = ",".join([
        "drawbox=x=0:y=560:w=1280:h=160:color=0x111827:t=fill",
        text("$ cat .env", "24", "580", 24, "0x22c55e"),
        text("AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE", "24", "620", 24, "0xe5e7eb", "gte(t,5)"),
    ])
    graph = f"[0:v]{desktop}[bg];[1:v]{window}[win];[bg][win]overlay=x='{win_x}':y=120[v]"
    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "voice.wav"
        say(SPEECH, wav)
        subprocess.run([
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "lavfi", "-i", f"color=c=0x1e293b:s={W}x{H}:r={FPS}:d={SECONDS}",
            "-f", "lavfi", "-i", f"color=c=white:s=560x340:r={FPS}:d={SECONDS}",
            "-i", str(wav), "-filter_complex", graph, "-map", "[v]", "-map", "2:a",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-af", "apad", "-t", str(SECONDS),
            str(out),
        ], check=True)
    out.with_suffix(".truth.json").write_text(json.dumps({"clip": out.name, "secrets": TRUTH}, indent=2))
    print(out)


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "samples/fake_pan_10s.mp4"))
