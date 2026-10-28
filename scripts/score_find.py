"""Score a find run against the clip's planted ground truth: which secrets were found, which were missed.

    python scripts/score_find.py samples/fake_pan_12s.truth.json out/fake_pan_12s/flags.json
"""
import json, re, sys
from difflib import SequenceMatcher


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def spoken(s: str) -> str:
    """'nine eight seven' -> '987', 'priya dot sharma at example' -> 'priyasharmaexample' style, for speech matches."""
    words = "zero one two three four five six seven eight nine".split()
    s = s.lower()
    for i, w in enumerate(words):
        s = re.sub(rf"\b{w}\b", str(i), s)
    return norm(re.sub(r"\b(dot|at)\b", "", s))


def main(truth_path: str, flags_path: str) -> None:
    truth = json.load(open(truth_path))["secrets"]
    items = json.load(open(flags_path, encoding="utf-8"))["items"]
    found = 0
    for sec in truth:
        target = norm(sec["text"])
        cands = [it for it in items if it["where"] == sec["where"]]
        conv = spoken if sec["where"] == "speech" else norm
        # fuzzy, because speech-to-text mishears names ("sharma" -> "charma") and drops ".com"
        hit = next((it for it in cands if target in conv(it["text"]) or
                    SequenceMatcher(None, target, conv(it["text"])).ratio() >= 0.75), None)
        found += bool(hit)
        detail = f"-> {hit['id']} {hit['start_s']}-{hit['end_s']}s" if hit else ""
        print(f"{'FOUND' if hit else 'MISS '} {sec['where']:6} {sec['kind']:12} {sec['text']} {detail}")
    extra = len(items) - found
    print(f"recall {found}/{len(truth)}, other flags {max(0, extra)}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
