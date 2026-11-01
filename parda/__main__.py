"""Parda command line.

    python -m parda find clip.mp4 [-o out/clip]     # writes flags.json, transcript.json, cost.json
    python -m parda redact clip.mp4 [-o out/clip] [--verify 3] [--drop s3]
        # find (or reuse flags.json), locate/track, render clean.mp4, verify the output and fix leaks, audit.json

Text secrets are located by OCR (parda/locate.py: exact value boxes, every frame); photos and other non-text
items are followed by SAM 3.1 (parda/track.py).
"""
import argparse, json, time
from pathlib import Path

from . import api, locate, render, sample, screen, speech, track, verify

NON_TEXT = getattr(render, "NON_TEXT", {"photo_of_person"})


def follow(clip: Path, items: list[dict], info: dict, out: Path) -> dict:
    """Boxes for every screen item: OCR for text values, SAM for photos and other non-text things."""
    screen_items = [it for it in items if it["where"] == "screen" and it.get("seen")]
    text = [it for it in screen_items if it["kind"] not in NON_TEXT]
    other = [it for it in screen_items if it["kind"] in NON_TEXT]
    tracks = locate.locate(clip, text, info, out) if text else {}
    if other:
        tracks.update(track.track(clip, other, info))
    return tracks


def _aliases(items: list[dict], tracks: dict) -> list[dict]:
    """A value shown in two places at once has a second key ("s7#2"); give it an item so render draws it."""
    by_id = {it["id"]: it for it in items}
    return [dict(by_id[k.split("#")[0]], id=k) for k in tracks if "#" in k and k.split("#")[0] in by_id]


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


def redact(clip: Path, out: Path, verify_passes: int = 3, drop: tuple = ()) -> dict:
    """Find (or reuse flags.json), locate/track, then verify the output and fix what it still shows."""
    flags_path = out / "flags.json"
    result = json.loads(flags_path.read_text(encoding="utf-8")) if flags_path.exists() else find(clip, out)
    info = result["video"]
    items = [it for it in result["items"] if it["id"] not in drop]  # --drop: plant a miss to test the verify loop
    t = time.time()
    tracks_path = out / "tracks.json"
    if tracks_path.exists() and tracks_path.stat().st_mtime > flags_path.stat().st_mtime:
        tracks = {k: {int(f): b for f, b in v.items()} for k, v in json.loads(tracks_path.read_text()).items()}
        audit_path = out / "audit.json"
        if audit_path.exists():  # what earlier verify passes found is tracked too; keep covering it
            known = {it["id"] for it in items}
            items += [it for it in json.loads(audit_path.read_text(encoding="utf-8"))["items"]
                      if it["found_by"] != "find" and it["id"] not in known and it["id"] not in drop]
        print("reusing tracks.json (flags unchanged since tracking)")
    else:
        tracks = follow(clip, items, info, out)
        tracks_path.write_text(json.dumps(tracks))
        print(f"located/tracked {len(tracks)} screen items over {sum(len(v) for v in tracks.values())} frames "
              f"in {time.time() - t:.0f}s")
    clean, passes = out / "clean.mp4", []
    for n in range(verify_passes + 1):
        t = time.time()
        stats = render.render(clip, clean, items + _aliases(items, tracks), tracks, info)
        print(f"rendered {stats['frames']} frames ({stats['frames_covered']} covered), {len(stats['bleeps'])} bleeps "
              f"in {time.time() - t:.0f}s")
        if n == verify_passes:
            break
        t = time.time()
        leaks = verify.scan(clean, out / f"verify{n + 1}")
        passes.append({"pass": n + 1, "found": [{k: lk.get(k) for k in ("kind", "where", "text", "start_s", "end_s")}
                                                for lk in leaks]})
        print(f"verify pass {n + 1}: {len(leaks)} still visible or audible ({time.time() - t:.0f}s)")
        if not leaks:
            break
        for k, lk in enumerate(leaks, start=1):  # fix: track the leak on the source video, render again
            lk["id"], lk["found_by"] = f"v{n + 1}.{k}", f"verify pass {n + 1}"
        items += leaks
        tracks.update(follow(clip, leaks, info, out / f"verify{n + 1}"))
    tracks_path.write_text(json.dumps(tracks))  # now with the boxes of everything verify found, for a re-render
    verdict = "clean" if passes and not passes[-1]["found"] else ("unverified" if not passes else "needs review")
    audit = {"clip": clip.name, "output": clean.name, "sha256": verify.sha256(clean), "verdict": verdict,
             "models": sorted({c["model"] for c in api.COST_LOG}), "render": stats, "verify": passes,
             "dropped_for_test": list(drop), "items": [
                 {k: it.get(k) for k in ("id", "kind", "where", "text", "reason", "start_s", "end_s", "timing")}
                 | {"frames_tracked": sum(len(v) for k, v in tracks.items() if k.split("#")[0] == it["id"]), "decision": it.get("decision", "auto"),
                    "found_by": it.get("found_by", "find")}
                 for it in items]}
    (out / "audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    usd = api.write_cost(out)
    print(f"verdict: {verdict} after {len(passes)} verify pass(es), ${usd:.4f} -> {clean}")
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
    r.add_argument("--verify", type=int, default=3, help="max verify-and-fix passes (0 = skip)")
    r.add_argument("--drop", nargs="*", default=[], help="item ids to leave out, to test that verify catches them")
    a = p.parse_args()
    out = a.out or Path("out") / a.clip.stem
    if a.cmd == "find":
        find(a.clip, out)
    else:
        redact(a.clip, out, a.verify, tuple(a.drop))


if __name__ == "__main__":
    main()
