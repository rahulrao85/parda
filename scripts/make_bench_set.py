"""Ten generated screen recordings with planted FAKE secrets, for the benchmark (scripts/bench.py).

Each clip is 12 s, 1280x720, 30 fps, drawn frame by frame so the motion is exact: a drifting document window,
values typed character by character with a cursor, a statement scrolling, step scrolls in a spreadsheet, a window
maximise, a smooth code scroll, a dragged ID card, a value scrolling out and back in. Four clips also carry a
voice (Windows SAPI, free) that says a secret.

Every value is fake: ABCDE1234F is the sample PAN on Indian tax forms, AKIAIOSFODNN7EXAMPLE is AWS's published
example key, example.com is reserved, every document says SPECIMEN.

    python scripts/make_bench_set.py samples/bench        # writes bench_01.mp4 ... bench_10.mp4 + .truth.json
"""
import json, subprocess, sys, tempfile
from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

W, H, FPS, SEC = 1280, 720, 30, 12
BG, BLUE, DARK, WHITE, INK, GREY = (30, 41, 59), (37, 99, 235), (17, 24, 39), (255, 255, 255), (20, 20, 20), (90, 100, 115)
FONTS = "C:/Windows/Fonts/"


@lru_cache(None)
def font(name: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONTS + name, size)


def S(kind: str, text: str, start: float = 0.0, end: float = SEC, where: str = "screen") -> dict:
    return {"kind": kind, "text": text, "where": where, "start_s": round(start, 2), "end_s": round(end, 2)}


def typed(text: str, t: float, t0: float, cps: float = 12.0) -> str:
    """What has been typed of `text` at time t, typing started at t0."""
    if t < t0:
        return ""
    return text[: min(len(text), int((t - t0) * cps))]


def done_at(text: str, t0: float, cps: float = 12.0) -> float:
    return t0 + len(text) / cps


def cursor(t: float) -> str:
    return "|" if int(t * 2) % 2 == 0 else ""


def window(img: Image.Image, x: int, y: int, w: int, h: int, title: str, body=WHITE, scale: float = 1.0) -> None:
    d = ImageDraw.Draw(img)
    bar = int(36 * scale)
    d.rectangle([x, y, x + w, y + h], fill=body)
    d.rectangle([x, y, x + w, y + bar], fill=BLUE)
    d.text((x + int(12 * scale), y + int(8 * scale)), title, font=font("arial.ttf", int(18 * scale)), fill=WHITE)


def specimen(img: Image.Image, x: int, y: int, size: int) -> None:
    ImageDraw.Draw(img).text((x, y), "SPECIMEN", font=font("arialbd.ttf", size), fill=(240, 160, 160))


def desktop() -> Image.Image:
    return Image.new("RGB", (W, H), BG)


# ---------------------------------------------------------------- scenes: each returns (frame(t), truth, speech)

def pan_drift():
    card = Image.new("RGB", (560, 340), WHITE)
    window(card, 0, 0, 560, 340, "PAN_card_scan.pdf")
    d = ImageDraw.Draw(card)
    d.text((24, 56), "INCOME TAX DEPARTMENT  (SAMPLE)", font=font("arialbd.ttf", 22), fill=INK)
    specimen(card, 300, 120, 30)
    d.text((24, 110), "Name: PRIYA SHARMA", font=font("arial.ttf", 24), fill=INK)
    d.text((24, 150), "Date of Birth: 01/01/1990", font=font("arial.ttf", 24), fill=INK)
    d.text((24, 200), "Permanent Account Number", font=font("arial.ttf", 20), fill=GREY)
    d.text((24, 232), "ABCDE1234F", font=font("arialbd.ttf", 44), fill=INK)

    def frame(t):
        img = desktop()
        d = ImageDraw.Draw(img)
        d.rectangle([0, 560, W, H], fill=DARK)
        d.text((24, 580), "$ cat .env", font=font("consola.ttf", 24), fill=(34, 197, 94))
        if t >= 5:
            d.text((24, 620), "AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE", font=font("consola.ttf", 24), fill=(229, 231, 235))
        img.paste(card, (int(80 + t * (W - 560 - 160) / SEC), 120))
        return img

    truth = [S("id_number", "ABCDE1234F"), S("name", "PRIYA SHARMA"), S("dob", "01/01/1990"),
             S("api_key", "AKIAIOSFODNN7EXAMPLE", 5.0), S("phone", "98765 43210", where="speech")]
    return frame, truth, "Okay, quick note. Call me on nine eight seven six five, four three two one zero."


def env_typed():
    lines = ["APP_ENV=production",
             "DATABASE_URL=postgres://admin:hunter2-not-real@db.example.com:5432/app",
             "AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE",
             "AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
             "OPENAI_API_KEY=sk-test-FAKE0000parda0000demo0000key",
             "SUPPORT_EMAIL=priya.sharma@example.com"]
    starts = [1.0 + 1.7 * i for i in range(len(lines))]
    cps = 40.0

    def frame(t):
        img = Image.new("RGB", (W, H), DARK)
        d = ImageDraw.Draw(img)
        d.text((24, 24), "$ cat .env", font=font("consola.ttf", 24), fill=(34, 197, 94))
        for i, (ln, t0) in enumerate(zip(lines, starts)):
            s = typed(ln, t, t0, cps)
            if s:
                if len(s) < len(ln):
                    s += cursor(t)
                d.text((24, 70 + 40 * i), s, font=font("consola.ttf", 24), fill=(229, 231, 235))
        return img

    def when(i, value):
        return done_at(lines[i][: lines[i].index(value)], starts[i], cps)

    truth = [S("password", "hunter2-not-real", when(1, "hunter2-not-real")),
             S("api_key", "AKIAIOSFODNN7EXAMPLE", when(2, "AKIA")),
             S("api_key", "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY", when(3, "wJal")),
             S("api_key", "sk-test-FAKE0000parda0000demo0000key", when(4, "sk-")),
             S("email", "priya.sharma@example.com", when(5, "priya"))]
    return frame, truth, None


def statement_scroll():
    page = Image.new("RGB", (W, 1700), WHITE)
    d = ImageDraw.Draw(page)
    d.text((60, 30), "SAMPLE BANK  -  Account Statement", font=font("arialbd.ttf", 30), fill=INK)
    specimen(page, 900, 30, 34)
    head = [("Name", "PRIYA SHARMA"), ("Account No", "000123456789"), ("IFSC", "SMPL0000123"),
            ("Email", "priya.sharma@example.com"), ("Period", "01-Sep-2026 to 30-Sep-2026")]
    for i, (k, v) in enumerate(head):
        d.text((60, 90 + 32 * i), f"{k}:", font=font("arial.ttf", 22), fill=GREY)
        d.text((240, 90 + 32 * i), v, font=font("arial.ttf", 22), fill=INK)
    y = 280
    d.text((60, y), "Date            Description                              Amount          Balance", font=font("consola.ttf", 20), fill=GREY)
    bal = 125000.0
    for r in range(40):
        amt = [-2500.0, 4200.0, -850.0, -12000.0, 85000.0, -399.0][r % 6] * (1 + r % 3)
        bal += amt
        row = f"{(r % 28) + 1:02d}-Sep-2026    {['UPI to grocer', 'Salary credit', 'Card payment', 'Rent', 'Transfer in', 'Subscription'][r % 6]:<38} {amt:>12,.2f}   {bal:>13,.2f}"
        d.text((60, y + 34 * (r + 1)), row, font=font("consola.ttf", 20), fill=INK)

    def frame(t):
        off = 0 if t < 3 else min(980, int((t - 3) * 70))
        return page.crop((0, off, W, off + H))

    truth = [S("name", "PRIYA SHARMA", 0, 6), S("bank_detail", "000123456789", 0, 6), S("bank_detail", "SMPL0000123", 0, 6),
             S("email", "priya.sharma@example.com", 0, 6)]
    return frame, truth, None


def signup_form():
    fields = [("Full name", "ANITA VERMA", 1.0), ("Email", "anita.verma@example.com", 3.0), ("Mobile", "9876543210", 5.5),
              ("Password", "Tq7#demo2026", 7.5), ("Date of birth", "14/03/1992", 9.5)]
    cps = 10.0

    def frame(t):
        img = desktop()
        window(img, 290, 60, 700, 600, "Create your account - SampleApp")
        d = ImageDraw.Draw(img)
        d.text((330, 120), "Sign up", font=font("arialbd.ttf", 30), fill=INK)
        for i, (label, value, t0) in enumerate(fields):
            y = 180 + 88 * i
            d.text((330, y), label, font=font("arial.ttf", 18), fill=GREY)
            d.rectangle([330, y + 26, 950, y + 66], outline=(180, 186, 196), width=2)
            s = typed(value, t, t0, cps)
            active = t0 <= t < done_at(value, t0, cps) + 0.6
            d.text((342, y + 33), s + (cursor(t) if active else ""), font=font("arial.ttf", 22), fill=INK)
        d.text((330, 630), "By signing up you agree to the SampleApp terms.", font=font("arial.ttf", 14), fill=GREY)
        return img

    truth = [S(k, v, t0) for k, (_, v, t0) in zip(["name", "email", "phone", "password", "dob"], fields)]
    truth.append(S("email", "anita.verma@example.com", where="speech"))
    return frame, truth, "My email is anita dot verma at example dot com, send it there."


def chat_otp():
    msgs = [(0.5, False, "Hey, did you get the code?"), (2.0, True, "Your OTP is 482916. Do not share it."),
            (3.5, False, "Got it. Also call me on 9876543210 when free"), (5.0, True, "Send the money to anita@examplebank"),
            (6.5, False, "Done. Thanks!"), (8.0, True, "Sure, see you tomorrow")]

    def frame(t):
        img = desktop()
        window(img, 340, 40, 600, 640, "Chat - Anita", body=(243, 244, 246))
        d = ImageDraw.Draw(img)
        y = 100
        for t0, mine, text in msgs:
            if t < t0:
                break
            f = font("arial.ttf", 20)
            w = d.textlength(text, font=f) + 28
            x = 920 - w if mine else 360
            d.rounded_rectangle([x, y, x + w, y + 44], radius=12, fill=(219, 234, 254) if mine else WHITE)
            d.text((x + 14, y + 10), text, font=f, fill=INK)
            y += 64
        return img

    truth = [S("otp", "482916", 2.0), S("phone", "9876543210", 3.5), S("bank_detail", "anita@examplebank", 5.0),
             S("otp", "482916", where="speech")]
    return frame, truth, "The OTP is four eight two nine one six."


def sheet_amounts():
    rows = [("E101", "Ravi Menon", "0000 1111 2222", "ABCDE1234F", "85,000"), ("E102", "Anita Verma", "3333 4444 5555", "FGHIJ5678K", "92,500"),
            ("E103", "Rajesh Kumar", "6666 7777 8888", "KLMNO9012P", "78,250")]
    filler = [(f"E1{10 + i:02d}", ["Sunil Rao", "Meera Nair", "Kiran Das", "Pooja Iyer"][i % 4], "", "", f"{60 + i * 3},000") for i in range(14)]
    table = filler[:4] + rows + filler[4:]
    cols = [(40, "ID"), (140, "Name"), (360, "Aadhaar"), (600, "PAN"), (800, "Salary")]
    grid = Image.new("RGB", (W, 40 + 36 * (len(table) + 1)), WHITE)
    d = ImageDraw.Draw(grid)
    for x, name in cols:
        d.text((x, 10), name, font=font("arialbd.ttf", 16), fill=INK)
    for r, row in enumerate(table):
        y = 40 + 36 * r
        d.line([0, y, W, y], fill=(220, 224, 230))
        for (x, _), v in zip(cols, row):
            d.text((x, y + 9), v, font=font("arial.ttf", 16), fill=INK)
    for x, _ in cols[1:]:
        d.line([x - 12, 0, x - 12, grid.height], fill=(220, 224, 230))

    def frame(t):
        img = desktop()
        window(img, 0, 0, W, H, "payroll_sep.xlsx - Sheet1")
        off = 0 if t < 5 else (72 if t < 8 else 144)  # two step scrolls of two rows
        img.paste(grid.crop((0, off, W, off + H - 36)), (0, 36))
        specimen(img, 1000, 60, 28)
        return img

    truth = [S("id_number", r[2]) for r in rows] + [S("id_number", r[3]) for r in rows]
    return frame, truth, None


def maximise_jump():
    def content(scale: float) -> Image.Image:
        w, h = int(640 * scale), int(360 * scale)
        img = Image.new("RGB", (w, h), WHITE)
        window(img, 0, 0, w, h, "Customer record", scale=scale)
        d = ImageDraw.Draw(img)
        f = font("arial.ttf", int(20 * scale))
        for i, (k, v) in enumerate([("Name", "RAJESH KUMAR"), ("PAN", "FGHIJ5678K"), ("Account", "111122223333"),
                                    ("Phone", "9123456780"), ("Plan", "Gold")]):
            d.text((int(24 * scale), int(60 * scale) + int(44 * scale) * i), f"{k}:", font=f, fill=GREY)
            d.text((int(150 * scale), int(60 * scale) + int(44 * scale) * i), v, font=f, fill=INK)
        specimen(img, int(400 * scale), int(300 * scale), int(22 * scale))
        return img

    small, big = content(1.0), content(2.0)

    def frame(t):
        img = desktop()
        if t < 6:
            img.paste(small, (320, 180))
        else:
            img.paste(big, (0, 0))
        return img

    truth = [S("name", "RAJESH KUMAR"), S("id_number", "FGHIJ5678K"), S("bank_detail", "111122223333"), S("phone", "9123456780")]
    return frame, truth, None


def code_editor():
    code = ["import os", "from pathlib import Path", "", "", "# settings for the demo deployment (all fake)",
            'DATABASE_URL = "postgres://admin:S3cretPwd-fake@db.example.com/app"', "", "",
            'API_KEY = "sk-test-FAKE0000parda0000demo0000key"', "TIMEOUT = 30", "", "",
            "def connect():", "    # call Anita on +91 98765 43210 if this breaks", "    return open_db(DATABASE_URL)", "", "",
            "def main():", "    db = connect()", "    for row in db.rows():", "        print(row)", "", "",
            "if __name__ == '__main__':", "    main()"] + [""] * 12
    page = Image.new("RGB", (W, 40 + 30 * len(code)), DARK)
    d = ImageDraw.Draw(page)
    for i, ln in enumerate(code):
        d.text((20, 20 + 30 * i), f"{i + 1:>3}", font=font("consola.ttf", 20), fill=GREY)
        d.text((80, 20 + 30 * i), ln, font=font("consola.ttf", 20), fill=(229, 231, 235))

    def frame(t):
        off = 0 if t < 4 else min(page.height - H, int((t - 4) * 40))
        return page.crop((0, off, W, off + H))

    truth = [S("password", "S3cretPwd-fake"), S("api_key", "sk-test-FAKE0000parda0000demo0000key"), S("phone", "+91 98765 43210")]
    return frame, truth, None


def id_drag():
    card = Image.new("RGB", (620, 380), WHITE)
    window(card, 0, 0, 620, 380, "licence_scan.jpg")
    d = ImageDraw.Draw(card)
    d.text((24, 52), "DRIVING LICENCE  (SAMPLE)", font=font("arialbd.ttf", 24), fill=INK)
    d.rectangle([24, 100, 184, 300], outline=GREY, width=3)
    d.text((70, 190), "PHOTO", font=font("arial.ttf", 22), fill=GREY)
    for i, (k, v) in enumerate([("Name", "ANITA VERMA"), ("DOB", "14/03/1992"), ("DL No", "DL-0420110012345"), ("Phone", "9876543210")]):
        d.text((210, 110 + 48 * i), f"{k}:", font=font("arial.ttf", 20), fill=GREY)
        d.text((300, 108 + 48 * i), v, font=font("arial.ttf", 24), fill=INK)
    specimen(card, 380, 320, 26)
    stops = [(0, (100, 120)), (3, (100, 120)), (5, (560, 260)), (8, (560, 260)), (10, (300, 60)), (SEC, (300, 60))]

    def pos(t):
        for (t0, p0), (t1, p1) in zip(stops, stops[1:]):
            if t0 <= t <= t1:
                a = 0 if t1 == t0 else (t - t0) / (t1 - t0)
                return int(p0[0] + a * (p1[0] - p0[0])), int(p0[1] + a * (p1[1] - p0[1]))
        return stops[-1][1]

    def frame(t):
        img = desktop()
        img.paste(card, pos(t))
        return img

    truth = [S("name", "ANITA VERMA"), S("dob", "14/03/1992"), S("id_number", "DL-0420110012345"), S("phone", "9876543210")]
    return frame, truth, None


def scroll_out_in():
    page = Image.new("RGB", (W, 1500), WHITE)
    d = ImageDraw.Draw(page)
    d.text((80, 40), "Vendor onboarding - notes", font=font("arialbd.ttf", 30), fill=INK)
    specimen(page, 950, 40, 30)
    paras = ["Spoke to the vendor today about the September invoice and the revised delivery schedule.",
             "Payment details: account number 000123456789, IFSC SMPL0000123 (Sample Bank, Mumbai).",
             "Contact: RAVI MENON, ravi.menon@example.com, available on weekdays after 11:00.",
             "Next step: legal reviews the amended contract, then finance releases the first tranche.",
             "", "Open questions", "Who owns the warranty claims after month twelve?", "Is the delivery penalty per day or per week?",
             "Does the renewal price track inflation or a fixed 5 percent?", "", "Decisions",
             "The pilot runs in two cities first. Expansion only after the Q3 review.",
             "All invoices go through the shared mailbox, not to individuals."]
    for i, p in enumerate(paras):
        d.text((80, 110 + 46 * i), p, font=font("arial.ttf", 22), fill=INK)

    def frame(t):
        if t < 3:
            off = 0
        elif t < 5:
            off = int((t - 3) / 2 * 600)
        elif t < 7:
            off = 600
        elif t < 9:
            off = int((9 - t) / 2 * 600)
        else:
            off = 0
        return page.crop((0, off, W, off + H))

    truth = [S("bank_detail", "000123456789"), S("bank_detail", "SMPL0000123"), S("name", "RAVI MENON"), S("email", "ravi.menon@example.com")]
    return frame, truth, None


SCENES = [pan_drift, env_typed, statement_scroll, signup_form, chat_otp, sheet_amounts, maximise_jump, code_editor, id_drag, scroll_out_in]


# ---------------------------------------------------------------- encode

def say(text: str, wav: Path) -> None:
    ps = ("Add-Type -AssemblyName System.Speech; $s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
          f"$s.SetOutputToWaveFile('{wav}'); $s.Speak(\"{text}\"); $s.Dispose()")
    subprocess.run(["powershell", "-NoProfile", "-Command", ps], check=True)


def encode(frame, out: Path, speech: str | None) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        if speech:
            wav = Path(tmp) / "voice.wav"
            say(speech, wav)
            audio = ["-i", str(wav)]
        else:
            audio = ["-f", "lavfi", "-i", "anullsrc=r=48000:cl=mono"]
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
               "-i", "-", *audio, "-map", "0:v", "-map", "1:a", "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
               "-c:a", "aac", "-af", "apad", "-t", str(SEC), str(out)]
        p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        for n in range(SEC * FPS):
            p.stdin.write(frame(n / FPS).tobytes())
        p.stdin.close()
        p.wait()
        if p.returncode:
            raise SystemExit(f"ffmpeg failed on {out}")


def main(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, scene in enumerate(SCENES, start=1):
        frame, truth, speech = scene()
        out = out_dir / f"bench_{i:02d}.mp4"
        encode(frame, out, speech)
        out.with_suffix(".truth.json").write_text(
            json.dumps({"clip": out.name, "scene": scene.__name__, "speech": speech, "secrets": truth}, indent=2), encoding="utf-8")
        print(f"{out}  {scene.__name__}  {len(truth)} secrets" + ("  + voice" if speech else ""))


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "samples/bench"))
