"""Parda command line.

    python -m parda find clip.mp4 [-o out/clip]     # writes flags.json, transcript.json, cost.json
"""
import argparse, json, time
from pathlib import Path

from . import api, sample, screen, speech


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


def main() -> None:
    p = argparse.ArgumentParser(prog="parda")
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("find", help="flag private things on screen and in speech")
    f.add_argument("clip", type=Path)
    f.add_argument("-o", "--out", type=Path)
    a = p.parse_args()
    if a.cmd == "find":
        find(a.clip, a.out or Path("out") / a.clip.stem)


if __name__ == "__main__":
    main()
