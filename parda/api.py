"""Meta Model API access shared by every Parda step, with a per-run cost log."""
import json, os, re, time
from pathlib import Path

from dotenv import find_dotenv, load_dotenv
from openai import OpenAI

load_dotenv(find_dotenv(usecwd=True))
KEY = os.getenv("META_API_KEY") or os.getenv("META_API")
BASE = "https://api.meta.ai/v1"
client = OpenAI(api_key=KEY, base_url=BASE, timeout=300, max_retries=3)

# Contributor models are 12x cheaper but Meta may train on what they see, so they are only for fake test clips.
# Real footage always goes to the standard model. Set PARDA_TEST_DATA=1 to opt a run into the cheap tier.
SPARK = "muse-spark-1.3-contributor" if os.getenv("PARDA_TEST_DATA") == "1" else "muse-spark-1.3"
PRICE = {  # USD per 1M tokens (input, output), from dev.meta.ai pricing, 28-Oct-2026
    "muse-spark-1.3": (1.25, 4.25), "muse-spark-1.3-contributor": (0.10, 0.20),
}
COST_LOG: list[dict] = []


def spark_json(content: list, effort: str = "low", attempts: int = 3) -> dict:
    """One Muse Spark call that must answer with a JSON object. Retries a reply that does not parse; raises
    ValueError if none does, so a caller never mistakes a broken reply for "nothing private here"."""
    for attempt in range(attempts):
        t = time.time()
        r = client.chat.completions.create(model=SPARK, reasoning_effort=effort,
                                           messages=[{"role": "user", "content": content}])
        u = r.usage
        pin, pout = PRICE[SPARK]
        COST_LOG.append({"model": SPARK, "secs": round(time.time() - t, 1), "in": u.prompt_tokens,
                         "out": u.completion_tokens, "usd": (u.prompt_tokens * pin + u.completion_tokens * pout) / 1e6})
        text = r.choices[0].message.content or ""
        m = re.search(r"\{.*\}", text, re.S)  # tolerate ```json fences or a stray sentence around the object
        try:
            return json.loads(m.group(0)) if m else {}
        except json.JSONDecodeError:
            continue
    raise ValueError(f"Muse Spark returned unparseable JSON {attempts} times")


def log_cost(model: str, usd: float, **extra) -> None:
    COST_LOG.append({"model": model, "usd": usd, **extra})


def write_cost(out_dir: Path) -> float:
    total = sum(c["usd"] for c in COST_LOG)
    (out_dir / "cost.json").write_text(json.dumps({"usd": round(total, 4), "calls": COST_LOG}, indent=2))
    return total
