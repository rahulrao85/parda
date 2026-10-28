"""Find private things on screen: send batches of sampled frames to Muse Spark, get flags back as JSON."""
import base64, hashlib, io, json, re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

from . import api
from .sample import Frame

PROMPT_VERSION = "screen-v2"
KINDS = ["id_number", "id_document", "api_key", "password", "email", "phone", "address", "bank_detail",
         "payment_card", "private_message", "other"]
PROMPT = f"""You check frames of a screen recording before it is published. For EACH frame, list every item a
viewer must not see: government ID numbers and ID documents, API keys, tokens, passwords, emails, phone numbers,
home addresses, bank or card details, private messages. Flag anything that LOOKS like one of these even if it might
be a sample, placeholder or test value: a person reviews every flag, so a missed secret is far worse than an extra
flag. Do not flag ordinary UI text or menu labels.
Boxes are [x1, y1, x2, y2] on a 0-1000 scale of that frame's width and height, tight around the private text or
document itself.
Reply with JSON only:
{{"frames": [{{"i": <frame number>, "flags": [{{"kind": one of {KINDS}, "text": "<exact text as shown, or a short
description for a document>", "box": [x1, y1, x2, y2], "reason": "<few words>"}}]}}]}}
Include every frame number, with an empty flags list when nothing is private."""


def _jpeg(frame: Frame) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(frame.rgb).save(buf, "JPEG", quality=88)
    return buf.getvalue()


def _batch(frames: list[Frame], cache: Path) -> dict[float, list]:
    jpegs = [_jpeg(f) for f in frames]
    key = hashlib.sha1(PROMPT_VERSION.encode() + api.SPARK.encode() + b"".join(jpegs)).hexdigest()
    hit = cache / f"{key}.json"
    if hit.exists():
        return {float(k): v for k, v in json.loads(hit.read_text()).items()}
    content = [{"type": "text", "text": PROMPT}]
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


def _merge(items: list[dict]) -> list[dict]:
    """Spark words the same document differently from batch to batch ("PAN_card_scan.pdf" vs "PAN_card_scan.pdf ID
    scan document"). Same kind and one text containing the other means the same secret."""
    out: list[tuple[set, dict]] = []  # (every wording seen for this secret, item)
    for it in items:
        a = _norm(it["text"])
        match = next((o for o in out if o[1]["kind"] == it["kind"] and len(a) >= 6
                      and any(len(b) >= 6 and (a in b or b in a) for b in o[0])), None)
        if match:
            match[0].add(a)
            match[1]["seen"] = sorted(match[1]["seen"] + it["seen"], key=lambda s: s["t"])
        else:
            out.append(({a}, it))
    out = [it for _, it in out]
    return out


def find(frames: list[Frame], info: dict, cache: Path, batch: int = 8, workers: int = 4) -> list[dict]:
    """Flags grouped into items: one item per distinct secret, with every frame it was seen on."""
    cache.mkdir(parents=True, exist_ok=True)
    groups = [frames[i:i + batch] for i in range(0, len(frames), batch)]
    per_frame: dict[float, list] = {}
    with ThreadPoolExecutor(workers) as pool:
        for res in pool.map(lambda g: _batch(g, cache), groups):
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
