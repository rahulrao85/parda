"""Find private things on screen: send batches of sampled frames to Muse Spark, get flags back as JSON."""
import base64, hashlib, io, json, re
from difflib import SequenceMatcher
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

from . import api
from .sample import Frame

PROMPT_VERSION = "screen-v4"
KINDS = ["id_number", "person_name", "date_of_birth", "photo_of_person", "api_key", "password", "email", "phone",
         "address", "bank_detail", "payment_card", "financial_amount", "private_message", "other"]
PROMPT = f"""You check frames of a screen recording before it is published. For EACH frame, list every item a
viewer must not see: government ID numbers and ID documents, API keys, tokens, passwords, emails, phone numbers,
home addresses, bank or card details, private messages, a person's name shown next to their ID, account, contact
or document details, and money amounts on bank statements, payslips or invoices. Flag anything that LOOKS like one of these even if it might
be a sample, placeholder or test value: a person reviews every flag, so a missed secret is far worse than an extra
flag. Do not flag ordinary UI text or menu labels.
Never flag a whole document, window or page. The viewer should still see WHAT the document is (a PAN card, a bank
statement, a .env file); flag each private VALUE inside it separately: the name, the ID number, the date of birth,
the photo, the account number, each key or password value. Labels such as "Name:" or "AWS_SECRET_ACCESS_KEY=" and
document titles stay visible.
Boxes are [x1, y1, x2, y2] on a 0-1000 scale of that frame's width and height, tight around the private value only.
Reply with JSON only:
{{"frames": [{{"i": <frame number>, "flags": [{{"kind": one of {KINDS}, "text": "<exact text as shown, or a short
description for a document>", "box": [x1, y1, x2, y2], "reason": "<few words>"}}]}}]}}
Include every frame number, with an empty flags list when nothing is private."""
# Verify passes look at Parda's own output, where covers are already drawn. Without this, Spark reads a row of dots
# as "a redacted secret" and every pass covers more of the screen.
VERIFY_NOTE = """
These frames were already redacted. Asterisks (****) with "hidden by Parda", rows of dots, lock cards, flat
patches and "private" chips are covers that already hide something: never flag a cover. Documents, labels and titles
are meant to stay visible. Flag only private VALUES that are still READABLE, even partly (a few visible characters
of a key, number or name count)."""


def _jpeg(frame: Frame) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(frame.rgb).save(buf, "JPEG", quality=88)
    return buf.getvalue()


def _batch(frames: list[Frame], cache: Path, prompt: str = PROMPT) -> dict[float, list]:
    jpegs = [_jpeg(f) for f in frames]
    key = hashlib.sha1(PROMPT_VERSION.encode() + prompt.encode() + api.SPARK.encode() + b"".join(jpegs)).hexdigest()
    hit = cache / f"{key}.json"
    if hit.exists():
        return {float(k): v for k, v in json.loads(hit.read_text()).items()}
    content = [{"type": "text", "text": prompt}]
    for n, (f, jp) in enumerate(zip(frames, jpegs)):
        content += [{"type": "text", "text": f"Frame {n} (t={f.t:.1f}s):"},
                    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(jp).decode()}}]
    reply = api.spark_json(content)
    by_t = {f.t: [] for f in frames}
    for fr in reply.get("frames", []):
        i = fr.get("i")
        if isinstance(i, int) and 0 <= i < len(frames):
            by_t[frames[i].t] = [fl for fl in fr.get("flags", []) if len(fl.get("box", [])) == 4]
    hit.write_text(json.dumps(by_t))
    return by_t


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9@.]", "", text.lower())


def _iou(a, b) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix
    return ix / union if union > 0 else 0.0


def _same(a: dict, b: dict, words_a: set, words_b: set) -> bool:
    """Same secret? Spark words a document differently per batch ("PAN card" vs "Sample PAN ID card") and misreads
    long keys by a character, so: one wording contains the other, near-identical wording, or the same box at the
    same moment."""
    if a["kind"] != b["kind"]:
        return False
    for x in words_a:
        for y in words_b:
            if min(len(x), len(y)) >= 6 and (x in y or y in x or SequenceMatcher(None, x, y).ratio() >= 0.9):
                return True
    return any(abs(p["t"] - q["t"]) <= 1.0 and _iou(p["box"], q["box"]) >= 0.5 for p in a["seen"] for q in b["seen"])


def _merge(items: list[dict]) -> list[dict]:
    out: list[tuple[set, dict]] = []  # (every wording seen for this secret, item)
    for it in items:
        words = {_norm(it["text"])}
        match = next((o for o in out if _same(o[1], it, o[0], words)), None)
        if match:
            match[0].update(words)
            match[1]["seen"] = sorted(match[1]["seen"] + it["seen"], key=lambda s: s["t"])
        else:
            out.append((words, it))
    return [it for _, it in out]


def find(frames: list[Frame], info: dict, cache: Path, batch: int = 8, workers: int = 4,
         verify: bool = False) -> list[dict]:
    """Flags grouped into items: one item per distinct secret, with every frame it was seen on."""
    cache.mkdir(parents=True, exist_ok=True)
    groups = [frames[i:i + batch] for i in range(0, len(frames), batch)]
    per_frame: dict[float, list] = {}
    with ThreadPoolExecutor(workers) as pool:
        prompt = PROMPT + VERIFY_NOTE if verify else PROMPT
        for res in pool.map(lambda g: _batch(g, cache, prompt), groups):
            per_frame.update(res)
    W, H, items = info["w"], info["h"], {}
    for t in sorted(per_frame):
        for fl in per_frame[t]:
            k = (fl.get("kind", "other"), _norm(fl.get("text", "")))
            x1, y1, x2, y2 = (float(v) for v in fl["box"])
            box = [round(x1 * W / 1000), round(y1 * H / 1000), round(x2 * W / 1000), round(y2 * H / 1000)]
            it = items.setdefault(k, {"kind": k[0], "text": fl.get("text", ""), "reason": fl.get("reason", ""),
                                      "where": "screen", "seen": []})
            it["seen"].append({"t": t, "box": box})
    out = _merge(list(items.values()))
    for n, it in enumerate(out):
        it["id"] = f"s{n + 1}"
        it["start_s"], it["end_s"] = it["seen"][0]["t"], it["seen"][-1]["t"]
    return out
