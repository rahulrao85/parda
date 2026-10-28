"""Follow every on-screen flag through the video with SAM 3.1 box prompts.

SAM is billed per frame, so each request only sees the stretch of video around its items (plus a second either
side), not the whole clip. Items whose boxes nest (a document and the number on it) go in separate requests,
because SAM drops the inner object when both are prompted together.
"""
import base64, re, subprocess, tempfile
from pathlib import Path

from . import api

PAD_S, MAX_OBJECTS, MAX_PROMPTS, SEG_MAX_W = 1.0, 16, 6, 1280
BOX = re.compile(r"(?:^<\d+f>|,)(\d+)<\|box;x1=(-?\d+);y1=(-?\d+);x2=(-?\d+);y2=(-?\d+);w=(\d+);h=(\d+)")


def _inside(a, b) -> bool:
    """True if box a lies (mostly) inside box b, or the other way round."""
    def frac(p, q):
        ix = max(0, min(p[2], q[2]) - max(p[0], q[0])) * max(0, min(p[3], q[3]) - max(p[1], q[1]))
        return ix / max(1, (p[2] - p[0]) * (p[3] - p[1]))
    return frac(a, b) > 0.8 or frac(b, a) > 0.8


def _groups(items: list[dict]) -> list[list[dict]]:
    groups: list[list[dict]] = []
    for it in sorted(items, key=lambda i: i["start_s"]):
        box = it["seen"][0]["box"]
        g = next((g for g in groups if len(g) < MAX_OBJECTS
                  and not any(_inside(box, o["seen"][0]["box"]) for o in g)), None)
        (g.append(it) if g else groups.append([it]))
    return groups


def _segment(clip, start: float, end: float, info: dict, out: Path) -> tuple[int, int]:
    scale = min(1.0, SEG_MAX_W / info["w"])
    w, h = int(info["w"] * scale) // 2 * 2, int(info["h"] * scale) // 2 * 2
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-ss", f"{start:.3f}", "-i", str(clip), "-t",
                    f"{end - start:.3f}", "-an", "-vf", f"scale={w}:{h}", "-r", f"{info['fps']}", "-c:v", "libx264",
                    "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p", str(out)], check=True)
    return w, h


def _sam(video: Path, prompt: str) -> str:
    data = video.read_bytes()
    if len(data) < 45_000_000:
        part = {"type": "input_video", "video_url": "data:video/mp4;base64," + base64.b64encode(data).decode()}
    else:
        f = api.client.files.create(file=video.open("rb"), purpose="user_data")
        part = {"type": "input_video", "file_id": f.id}
    out, frames = [], 0
    stream = api.client.responses.create(model="sam-3.1", stream=True, metadata={"mask_encoding": "one_bit"},
                                         input=[{"role": "user", "content": [{"type": "input_text", "text": prompt},
                                                                             part]}])
    for ev in stream:
        if ev.type == "response.output_text.delta":
            out.append(ev.delta)
        elif ev.type == "response.completed":
            frames = getattr(ev.response.usage, "video_frames_processed", 0) or 0
    api.log_cost("sam-3.1", max(frames, 50) * 0.20 / 1000, frames=frames)
    return "".join(out)


def track(clip, items: list[dict], info: dict) -> dict[str, dict[int, list[int]]]:
    """For each screen item id: {source frame number: [x1, y1, x2, y2] in source pixels}."""
    fps, tracks = info["fps"], {}
    for group in _groups([i for i in items if i["where"] == "screen" and i.get("seen")]):
        start = max(0.0, min(i["start_s"] for i in group) - PAD_S)
        end = min(info["duration"], max(i["end_s"] for i in group) + PAD_S + 1 / info.get("rate", 2))
        with tempfile.TemporaryDirectory() as tmp:
            seg = Path(tmp) / "seg.mp4"
            w, h = _segment(clip, start, end, info, seg)
            sx, sy = w / info["w"], h / info["h"]
            prompts: dict[int, list[str]] = {}
            for oid, it in enumerate(group, start=1):
                last, n = -1e9, 0
                for s in it["seen"]:  # re-prompt every 1.5 s so a scroll or jump does not lose the object
                    if s["t"] - last < 1.5 or n >= MAX_PROMPTS:
                        continue
                    last, n = s["t"], n + 1
                    x1, y1, x2, y2 = s["box"]
                    f = max(0, round((s["t"] - start) * fps))
                    prompts.setdefault(f, []).append(
                        f"{oid}<|box;x1={round(x1 * sx)};y1={round(y1 * sy)};x2={round(x2 * sx)};y2={round(y2 * sy)};w={w};h={h}|>")
            prompt = "".join(f"<{f}f>" + ",".join(p) for f, p in sorted(prompts.items()))
            raw = _sam(seg, prompt)
        offset = round(start * fps)
        for line in raw.splitlines():
            fm = re.match(r"<(\d+)f>", line)
            if not fm:
                continue
            for oid, x1, y1, x2, y2, _, _ in BOX.findall(line):
                k = int(oid) - 1
                if 0 <= k < len(group):
                    box = [round(int(x1) / sx), round(int(y1) / sy), round(int(x2) / sx), round(int(y2) / sy)]
                    tracks.setdefault(group[k]["id"], {})[int(fm.group(1)) + offset] = box
    return tracks
