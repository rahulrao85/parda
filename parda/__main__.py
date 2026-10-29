"""Parda command line.

    python -m parda find clip.mp4 [-o out/clip]     # writes flags.json, transcript.json, cost.json
    python -m parda redact clip.mp4 [-o out/clip]   # find (or reuse flags.json), track, render clean.mp4 + audit.json
"""
import argparse, json, time
from pathlib import Path

from . import api, render, sample, screen, speech, track


def find(clip: Path, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    t = time.time()
    frames, info = sample.sample(clip)
    print(f"sampled {info['sampled']} frames, kept {info['kept']} that changed")
    on_screen = screen.find(frames, info, Path("cache/spark"))
    print(f"screen: {len(on_screen)} items")
    spoken, turns = speech.find(clip, info, out)
    print(f"speech: {len(spoken)} items from {len(turns)} turns")
    result = {"clip": clip.name, "video": info, "items": on_screen + spoken}
    (out / "flags.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    usd = api.write_cost(out)
    print(f"done in {time.time() - t:.0f}s, ${usd:.4f}, model {api.SPARK} -> {out / 'flags.json'}")
    return result


def redact(clip: Path, out: Path) -> dict:
    flags_path = out / "flags.json"
    result = json.loads(flags_path.read_text(encoding="utf-8")) if flags_path.exists() else find(clip, out)
    items, info = result["items"], result["video"]
    t = time.time()
    tracks_path = out / "tracks.json"
    if tracks_path.exists() and tracks_path.stat().st_mtime > flags_path.stat().st_mtime:
        tracks = {k: {int(f): b for f, b in v.items()} for k, v in json.loads(tracks_path.read_text()).items()}
        print("reusing tracks.json (flags unchanged since tracking)")
    else:
        tracks = track.track(clip, items, info)
        tracks_path.write_text(json.dumps(tracks))
        print(f"tracked {len(tracks)} screen items over {sum(len(v) for v in tracks.values())} frames "
              f"in {time.time() - t:.0f}s")
    t = time.time()
    stats = render.render(clip, out / "clean.mp4", items, tracks, info)
    audit = {"clip": clip.name, "output": "clean.mp4", "models": sorted({c["model"] for c in api.COST_LOG}),
             "render": stats, "items": [
                 {k: it.get(k) for k in ("id", "kind", "where", "text", "reason", "start_s", "end_s", "timing")}
                 | {"frames_tracked": len(tracks.get(it["id"], {})), "decision": it.get("decision", "auto")}
                 for it in items]}
    (out / "audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    usd = api.write_cost(out)
    print(f"rendered {stats['frames']} frames ({stats['frames_covered']} covered), {len(stats['bleeps'])} bleeps "
          f"in {time.time() - t:.0f}s, ${usd:.4f} -> {out / 'clean.mp4'}")
    return audit


def main() -> None:
    p = argparse.ArgumentParser(prog="parda")
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("find", help="flag private things on screen and in speech")
    f.add_argument("clip", type=Path)
    f.add_argument("-o", "--out", type=Path)
    r = sub.add_parser("redact", help="find, track and render the clean video")
    r.add_argument("clip", type=Path)
    r.add_argument("-o", "--out", type=Path)
    a = p.parse_args()
    {"find": find, "redact": redact}[a.cmd](a.clip, a.out or Path("out") / a.clip.stem)


if __name__ == "__main__":
    main()
