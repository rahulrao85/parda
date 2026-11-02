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


def _known(text: str, known: set[str]) -> bool:
    """Is this verify finding a value Parda already covers? Spark reads a half-covered or small value with a letter
    or two wrong ("priya.snama@example.com"), which must not become a second item for the same secret."""
    if text in known:
        return True
    if text.isdigit() or not any(c.isalpha() for c in text):
        return False  # a number one digit off is a different amount or account, not a misread
    return len(text) >= 8 and any(abs(len(k) - len(text)) <= 2 and _edits(k, text) <= 2 for k in known)


def _edits(a: str, b: str) -> int:
    """Levenshtein distance; the strings are short."""
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


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
    print(f"done in {time.time() - t:.0f}s, ${usd:.4f}, model {api.screen_model()} -> {out / 'flags.json'}")
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
        expected = round(info["duration"] * info["fps"])
        if abs(stats["frames"] - expected) > 1:
            print(f"WARNING: rendered {stats['frames']} frames but the clip has about {expected}: the decoder dropped "
                  f"frames (a damaged read?); covers after the drop may be a frame off")
        if n == verify_passes:
            break
        t = time.time()
        leaks = verify.scan(clean, out / f"verify{n + 1}")
        passes.append({"pass": n + 1, "found": [{k: lk.get(k) for k in ("kind", "where", "text", "start_s", "end_s")}
                                                for lk in leaks]})
        print(f"verify pass {n + 1}: {len(leaks)} still visible or audible ({time.time() - t:.0f}s)")
        if not leaks:
            break
        # A leak that is a value Parda already knows is a location miss, not a new secret: no new item, but the
        # whole set is located again below. Anything new becomes an item.
        known = {screen._norm(it["text"]) for it in items if it["where"] == "screen"}
        new = [lk for lk in leaks if lk["where"] != "screen" or not _known(screen._norm(lk["text"]), known)]
        for k, lk in enumerate(new, start=1):  # fix: track the leak on the source video, render again
            lk["id"], lk["found_by"] = f"v{n + 1}.{k}", f"verify pass {n + 1}"
        items += new
        leaks = new
        # Text leaks are located together with everything already known, so they share its clusters (a value
        # scrolling in is predicted from its neighbours); the OCR cache makes the repeat pass cheap.
        text = [it for it in items if it["where"] == "screen" and it.get("seen") and it["kind"] not in NON_TEXT]
        other = [lk for lk in leaks if lk["where"] == "screen" and lk.get("seen") and lk["kind"] in NON_TEXT]
        tracks = {k: v for k, v in tracks.items() if k.split("#")[0] not in {it["id"] for it in text}}
        tracks.update(locate.locate(clip, text, info, out / f"verify{n + 1}"))
        if other:
            tracks.update(track.track(clip, other, info))
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
    for sp in (f, r):
        sp.add_argument("--private", action="store_true", help="find on-screen secrets with Muse Glimmer on your own "
                        "GPU (PARDA_PRIVATE_URL) instead of Muse Spark; same as PARDA_BACKEND=private")
    a = p.parse_args()
    out = a.out or Path("out") / a.clip.stem
    if a.private:
        api.use_private()
    if a.cmd == "find":
        find(a.clip, out)
    else:
        redact(a.clip, out, a.verify, tuple(a.drop))


if __name__ == "__main__":
    main()
