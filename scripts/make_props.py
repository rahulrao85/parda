"""Fake documents to show on screen when recording realistic test clips. Every value is fake and marked SPECIMEN.

    python scripts/make_props.py samples/props

Writes fake_pan_card.pdf, fake_bank_statement.pdf, fake.env and props.truth.json (what Parda should find).
"""
import json, sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

FONTS = "C:/Windows/Fonts/"
PAN, NAME, DOB = "ABCDE1234F", "PRIYA SHARMA", "01/01/1990"
ACCOUNT, IFSC = "000123456789", "SMPL0000123"
ENV = """# local settings - DO NOT COMMIT
APP_ENV=production
DATABASE_URL=postgres://admin:hunter2-not-real@db.example.com:5432/app
AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE
AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY
OPENAI_API_KEY=sk-test-FAKE0000parda0000demo0000key
SUPPORT_EMAIL=priya.sharma@example.com
"""
TRUTH = [
    {"kind": "id_number", "text": PAN, "where": "screen"},
    {"kind": "id_document", "text": "fake_pan_card.pdf", "where": "screen"},
    {"kind": "bank_detail", "text": ACCOUNT, "where": "screen"},
    {"kind": "password", "text": "hunter2-not-real", "where": "screen"},
    {"kind": "api_key", "text": "AKIAIOSFODNN7EXAMPLE", "where": "screen"},
    {"kind": "api_key", "text": "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY", "where": "screen"},
    {"kind": "api_key", "text": "sk-test-FAKE0000parda0000demo0000key", "where": "screen"},
    {"kind": "email", "text": "priya.sharma@example.com", "where": "screen"},
    {"kind": "phone", "text": "98765 43210", "where": "speech"},
    {"kind": "email", "text": "priya.sharma@example.com", "where": "speech"},
    {"kind": "password", "text": "482916", "where": "speech"},
]


def font(name: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONTS + name, size)


def specimen(img: Image.Image) -> None:
    """Diagonal SPECIMEN watermark, so nobody can mistake a prop for a real document."""
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    d.text((img.width * 0.18, img.height * 0.38), "SPECIMEN - NOT VALID", font=font("arialbd.ttf", img.width // 14),
           fill=(220, 38, 38, 70))
    img.alpha_composite(layer.rotate(18, center=(img.width / 2, img.height / 2)))


def pan_card(out: Path) -> None:
    card = Image.new("RGBA", (1012, 638), (236, 244, 255, 255))
    d = ImageDraw.Draw(card)
    d.rectangle([0, 0, 1012, 96], fill=(37, 99, 235))
    d.text((32, 22), "INCOME TAX DEPARTMENT", font=font("arialbd.ttf", 40), fill="white")
    d.text((640, 34), "SAMPLE ID CARD", font=font("arial.ttf", 28), fill="white")
    d.rectangle([40, 140, 250, 400], outline=(100, 116, 139), width=3)  # photo placeholder
    d.text((92, 255), "PHOTO", font=font("arial.ttf", 30), fill=(100, 116, 139))
    rows = [("Name", NAME), ("Father's Name", "RAJESH SHARMA"), ("Date of Birth", DOB)]
    for i, (k, v) in enumerate(rows):
        d.text((300, 140 + i * 86), k, font=font("arial.ttf", 24), fill=(71, 85, 105))
        d.text((300, 170 + i * 86), v, font=font("arialbd.ttf", 32), fill=(15, 23, 42))
    d.text((40, 450), "Permanent Account Number", font=font("arial.ttf", 26), fill=(71, 85, 105))
    d.text((40, 488), PAN, font=font("consolab.ttf", 72), fill=(15, 23, 42))
    specimen(card)
    page = Image.new("RGB", (1240, 1754), "white")  # A4 at 150 dpi
    page.paste(card.convert("RGB"), (114, 160))
    ImageDraw.Draw(page).text((114, 860), "Scanned copy for KYC (fake test document for Parda)",
                              font=font("arial.ttf", 26), fill=(71, 85, 105))
    page.save(out / "fake_pan_card.pdf", resolution=150)


def bank_statement(out: Path) -> None:
    page = Image.new("RGBA", (1240, 1754), (255, 255, 255, 255))
    d = ImageDraw.Draw(page)
    d.text((80, 80), "SAMPLE BANK LTD", font=font("arialbd.ttf", 48), fill=(37, 99, 235))
    d.text((80, 150), "Account Statement  |  01-Oct-2026 to 27-Oct-2026", font=font("arial.ttf", 26), fill="black")
    for i, (k, v) in enumerate([("Account holder", NAME), ("Account number", ACCOUNT), ("IFSC", IFSC),
                                ("Registered email", "priya.sharma@example.com")]):
        d.text((80, 230 + i * 48), f"{k}:", font=font("arial.ttf", 26), fill=(71, 85, 105))
        d.text((380, 230 + i * 48), v, font=font("consola.ttf", 28), fill="black")
    y = 470
    d.line([80, y - 10, 1160, y - 10], fill=(203, 213, 225), width=2)
    for date, desc, amt in [("02-Oct", "UPI/grocery store", "-1,240.00"), ("05-Oct", "Salary credit", "+85,000.00"),
                            ("09-Oct", "Electricity bill", "-2,315.00"), ("14-Oct", "UPI/friend refund", "+600.00"),
                            ("21-Oct", "Rent transfer", "-25,000.00"), ("26-Oct", "ATM withdrawal", "-5,000.00")]:
        d.text((80, y), date, font=font("arial.ttf", 26), fill="black")
        d.text((260, y), desc, font=font("arial.ttf", 26), fill="black")
        d.text((900, y), amt, font=font("consola.ttf", 26), fill="black")
        y += 52
    specimen(page)
    page.convert("RGB").save(out / "fake_bank_statement.pdf", resolution=150)


def main(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    pan_card(out)
    bank_statement(out)
    (out / "fake.env").write_text(ENV)
    (out / "props.truth.json").write_text(json.dumps({"secrets": TRUTH}, indent=2))
    print("\n".join(str(p) for p in sorted(out.iterdir())))


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "samples/props"))
