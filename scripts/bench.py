"""Benchmark: Parda against the regex-over-OCR baseline on the ten generated clips (scripts/make_bench_set.py).

One measure, applied the same way to both outputs: a stranger's view. Every output is OCR'd at 4 frames per
second; a planted value counts as readable on a frame when a run of at least 60 % of its characters (six at least)
comes back. The score is readable output frames over readable source frames, per value: 0 % is clean. The same
pass measures how much of each frame was altered (pixels that differ from the source by more than 24 levels), so
"looks untouched" is a number too. Spoken values are scored by whether Parda flagged them and bleeped them.

    PARDA_TEST_DATA=1 python scripts/bench.py            # runs everything, writes out/bench/results.json + BENCHMARK.md
    python scripts/bench.py --no-run                     # re-score existing outputs
    python scripts/bench.py --only bench_03 bench_10     # re-run two clips, re-score all
"""
import json, math, os, re, shutil, subprocess, sys, time
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from parda import sample  # noqa: E402
from parda.locate import Ocr  # noqa: E402
from scripts.score_find import spoken  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CLIPS = sorted((ROOT / "samples" / "bench").glob("bench_*.mp4"))
OUT = ROOT / "out" / "bench"
PY = sys.executable
SCORE_FPS = 4


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def frames_at(clip: Path, rate: float) -> list[np.ndarray]:
    info = sample.probe(clip)
    W, H = info["w"], info["h"]
    raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", str(clip), "-vf", f"fps={rate}", "-f", "rawvideo",
                          "-pix_fmt", "rgb24", "-"], capture_output=True, check=True).stdout
    n = len(raw) // (W * H * 3)
    return [np.frombuffer(raw[i * W * H * 3:(i + 1) * W * H * 3], np.uint8).reshape(H, W, 3) for i in range(n)]


def readable(joined: str, value: str) -> bool:
    v = norm(value)
    k = max(6, math.ceil(0.6 * len(v)))
    if len(v) <= k:
        return v in joined
    return any(v[i:i + k] in joined for i in range(len(v) - k + 1))


def run_parda(clip: Path, out: Path) -> dict:
    # A fresh locate/verify run: flags.json (the Spark find) is kept if present, so a re-run measures the locating.
    for stale in ("tracks.json", "audit.json", "clean.mp4"):
        (out / stale).unlink(missing_ok=True)
    for d in out.glob("verify*"):
        shutil.rmtree(d, ignore_errors=True)
    t = time.time()
    env = dict(os.environ, PARDA_TEST_DATA="1")
    r = subprocess.run([PY, "-m", "parda", "redact", str(clip), "-o", str(out)], cwd=ROOT, env=env,
                       capture_output=True, text=True)
    (out / "bench_log.txt").write_text(r.stdout + r.stderr, encoding="utf-8")
    if r.returncode:
        raise SystemExit(f"parda failed on {clip.name}:\n{r.stderr[-2000:]}")
    return {"wall_s": round(time.time() - t, 1)}


def run_baseline(clip: Path, out: Path) -> dict:
    t = time.time()
    r = subprocess.run([PY, "scripts/baseline_regex.py", str(clip), str(out / "baseline.mp4")], cwd=ROOT,
                       capture_output=True, text=True)
    if r.returncode:
        raise SystemExit(f"baseline failed on {clip.name}:\n{r.stderr[-2000:]}")
    return {"wall_s": round(time.time() - t, 1)}


def score_clip(clip: Path, out: Path, truth: dict, ocr: Ocr) -> dict:
    src = frames_at(clip, SCORE_FPS)
    outs = {"parda": frames_at(out / "clean.mp4", SCORE_FPS), "baseline": frames_at(out / "baseline.mp4", SCORE_FPS)}
    n = min(len(src), *(len(v) for v in outs.values()))
    text = {"src": [], "parda": [], "baseline": []}
    altered = {"parda": [], "baseline": []}
    for i in range(n):
        text["src"].append(norm(" ".join(ln["text"] for ln in ocr.frame(src[i]))))
        for name, fr in outs.items():
            text[name].append(norm(" ".join(ln["text"] for ln in ocr.frame(fr[i]))))
            diff = np.abs(fr[i].astype(np.int16) - src[i].astype(np.int16)).max(axis=2) > 24
            altered[name].append(float(diff.mean()))
    audit = json.loads((out / "audit.json").read_text(encoding="utf-8"))
    cost = json.loads((out / "cost.json").read_text(encoding="utf-8"))
    rows = []
    for sec in truth["secrets"]:
        row = {"kind": sec["kind"], "text": sec["text"], "where": sec["where"]}
        if sec["where"] == "screen":
            vis = [i for i in range(n) if readable(text["src"][i], sec["text"])]
            row["source_frames"] = len(vis)
            for name in ("parda", "baseline"):
                leaked = [i for i in vis if readable(text[name][i], sec["text"])]
                row[f"{name}_leaked"] = len(leaked)
                row[f"{name}_leak_pct"] = round(100 * len(leaked) / len(vis), 1) if vis else None
        else:
            target = spoken(sec["text"])
            # fuzzy, as in score_find: speech-to-text mishears names ("varma") and drops ".com"
            flagged = [it for it in audit["items"] if it["where"] == "speech" and
                       (target in spoken(it["text"]) or spoken(it["text"]) in target or
                        SequenceMatcher(None, target, spoken(it["text"])).ratio() >= 0.75)]
            row["parda_flagged"] = bool(flagged)
            row["parda_bleeped"] = bool(flagged) and bool(audit["render"].get("bleeps"))
            row["baseline_flagged"] = False
        rows.append(row)
    return {"clip": clip.name, "scene": truth.get("scene"), "frames_scored": n, "secrets": rows,
            "altered_pct": {k: round(100 * float(np.mean(v)), 2) for k, v in altered.items()},
            "parda": {"verdict": audit["verdict"], "verify_passes": len(audit["verify"]), "usd": cost.get("usd"),
                      "items": len(audit["items"])}}


def table(results: list[dict]) -> str:
    lines = ["| Clip | Scene | Planted | Parda: readable frames | Baseline: readable frames | Altered area (Parda / baseline) | Parda $ | Parda s | Baseline s |",
             "|---|---|---|---|---|---|---|---|---|"]
    tot = {"src": 0, "parda": 0, "baseline": 0, "sp": 0, "sp_ok": 0}
    for r in results:
        scr = [s for s in r["secrets"] if s["where"] == "screen" and s["source_frames"]]
        sp = [s for s in r["secrets"] if s["where"] == "speech"]
        src = sum(s["source_frames"] for s in scr)
        pl, bl = sum(s["parda_leaked"] for s in scr), sum(s["baseline_leaked"] for s in scr)
        tot["src"] += src; tot["parda"] += pl; tot["baseline"] += bl
        tot["sp"] += len(sp); tot["sp_ok"] += sum(s["parda_bleeped"] for s in sp)
        spk = f" + {sum(s['parda_bleeped'] for s in sp)}/{len(sp)} spoken bleeped" if sp else ""
        lines.append(f"| {r['clip']} | {r['scene']} | {len(scr)} on screen{', ' + str(len(sp)) + ' spoken' if sp else ''} | "
                     f"{pl}/{src} ({100 * pl / src:.0f} %){spk} | {bl}/{src} ({100 * bl / src:.0f} %) | "
                     f"{r['altered_pct']['parda']} % / {r['altered_pct']['baseline']} % | {r['parda']['usd']:.3f} | "
                     f"{r['parda']['wall_s']} | {r['baseline']['wall_s']} |")
    lines.append(f"| **All** | | | **{tot['parda']}/{tot['src']} ({100 * tot['parda'] / tot['src']:.1f} %)**, "
                 f"{tot['sp_ok']}/{tot['sp']} spoken bleeped | **{tot['baseline']}/{tot['src']} "
                 f"({100 * tot['baseline'] / tot['src']:.1f} %)**, 0/{tot['sp']} spoken | | | | |")
    return "\n".join(lines)


def per_value(results: list[dict]) -> str:
    lines = ["| Clip | Kind | Value | Source frames | Parda leaked | Baseline leaked |", "|---|---|---|---|---|---|"]
    for r in results:
        for s in r["secrets"]:
            if s["where"] == "speech":
                lines.append(f"| {r['clip']} | {s['kind']} (spoken) | {s['text']} | | "
                             f"{'bleeped' if s['parda_bleeped'] else 'MISSED'} | not handled |")
            else:
                lines.append(f"| {r['clip']} | {s['kind']} | {s['text']} | {s['source_frames']} | "
                             f"{s['parda_leaked']} | {s['baseline_leaked']} |")
    return "\n".join(lines)


def main(run: bool, only: set[str] = frozenset()) -> None:
    """run: re-run Parda and the baseline (on every clip, or on `only`); otherwise score what is in out/bench."""
    OUT.mkdir(parents=True, exist_ok=True)
    ocr = Ocr(Path("cache/ocr"))
    results = []
    for clip in CLIPS:
        out = OUT / clip.stem
        truth = json.loads(clip.with_suffix(".truth.json").read_text(encoding="utf-8"))
        timing = json.loads((out / "bench_timing.json").read_text()) if (out / "bench_timing.json").exists() else {}
        if (run and (not only or clip.stem in only)) or not (out / "clean.mp4").exists():
            out.mkdir(parents=True, exist_ok=True)
            print(f"== {clip.name}: parda", flush=True)
            timing["parda"] = run_parda(clip, out)
            print(f"== {clip.name}: baseline", flush=True)
            timing["baseline"] = run_baseline(clip, out)
            (out / "bench_timing.json").write_text(json.dumps(timing))
        print(f"== {clip.name}: scoring", flush=True)
        r = score_clip(clip, out, truth, ocr)
        r["parda"].update(timing.get("parda", {}))
        r["baseline"] = timing.get("baseline", {})
        results.append(r)
        print(json.dumps({k: r[k] for k in ("clip", "altered_pct")}), [(s["text"], s.get("parda_leaked"), s.get("baseline_leaked"))
                                                                        for s in r["secrets"]], flush=True)
    (OUT / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    (OUT / "table.md").write_text(table(results) + "\n\n" + per_value(results) + "\n", encoding="utf-8")
    print(table(results))


if __name__ == "__main__":
    # python scripts/bench.py                      run + score all
    # python scripts/bench.py --only bench_03 bench_10   re-run those, re-score all
    # python scripts/bench.py --no-run             re-score only
    args = sys.argv[1:]
    only = set(args[args.index("--only") + 1:]) if "--only" in args else set()
    main(run="--no-run" not in args, only=only)
