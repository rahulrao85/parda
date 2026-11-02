"""Meta Model API access shared by every Parda step, with a per-run cost log.

Private mode (PARDA_BACKEND=private, or --private on the command line) sends the on-screen find step to Muse
Glimmer, Meta's open-weight vision model, on your own GPU (deploy/modal_glimmer.py) instead of Muse Spark. Speech
still uses Muse Voice Transcribe and Muse Spark. See docs/PRIVATE_MODE.md.
"""
import json, os, re, threading, time
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


# --- private backend: Muse Glimmer behind an OpenAI-compatible endpoint you run (vLLM on Modal) ---
BACKEND = os.getenv("PARDA_BACKEND", "spark")
PRIVATE_URL = os.getenv("PARDA_PRIVATE_URL", "")      # e.g. https://<workspace>--parda-glimmer-serve.modal.run
PRIVATE_MODEL = os.getenv("PARDA_PRIVATE_MODEL", "muse-glimmer")
PRIVATE_KEY = os.getenv("PARDA_PRIVATE_KEY", "")      # the bearer key the server was started with (VLLM_API_KEY)
PRIVATE_GPU_USD_S = float(os.getenv("PARDA_PRIVATE_GPU_USD_S", "0.000542"))  # Modal L40S list price, 02-Nov-2026
PRIVATE_BATCH = int(os.getenv("PARDA_PRIVATE_BATCH", "4"))  # frames per Glimmer call
# Glimmer reads its reasoning effort from the system prompt; the screen prompt itself is shared with Spark unchanged.
PRIVATE_SYSTEM = "Reasoning strength: low"
_private_client = None
_inflight, _inflight_lock = 0, threading.Lock()


def use_private(on: bool = True) -> None:
    global BACKEND
    BACKEND = "private" if on else "spark"


def private() -> bool:
    return BACKEND == "private"


def screen_model() -> str:
    """The model that judges sampled frames. Part of the screen cache key, so Spark and Glimmer never share hits."""
    return PRIVATE_MODEL if private() else SPARK


def screen_json(content: list, effort: str = "low") -> dict:
    """One screen-find call on the active backend."""
    return private_json(content) if private() else spark_json(content, effort)


def private_json(content: list, attempts: int = 3) -> dict:
    """spark_json for Muse Glimmer on your own endpoint. Logs GPU seconds: the request's wall time divided by the
    calls sharing the GPU when it started, priced at PARDA_PRIVATE_GPU_USD_S (idle and cold-start time is billed
    by Modal too; see docs/PRIVATE_MODE.md)."""
    global _private_client, _inflight
    if not PRIVATE_URL or not PRIVATE_KEY:
        raise RuntimeError("private mode needs PARDA_PRIVATE_URL and PARDA_PRIVATE_KEY (see docs/PRIVATE_MODE.md)")
    if _private_client is None:
        _private_client = OpenAI(api_key=PRIVATE_KEY, base_url=PRIVATE_URL.rstrip("/") + "/v1", timeout=1200,
                                 max_retries=2)
    for attempt in range(attempts):
        with _inflight_lock:
            _inflight += 1
            sharing = _inflight
        t = time.time()
        try:
            r = _private_client.chat.completions.create(
                model=PRIVATE_MODEL, max_tokens=4096,
                messages=[{"role": "system", "content": PRIVATE_SYSTEM}, {"role": "user", "content": content}])
        finally:
            with _inflight_lock:
                _inflight -= 1
        secs = time.time() - t
        u = r.usage
        COST_LOG.append({"model": PRIVATE_MODEL, "backend": "private", "secs": round(secs, 1),
                         "gpu_secs": round(secs / sharing, 1), "in": u.prompt_tokens, "out": u.completion_tokens,
                         "usd": secs / sharing * PRIVATE_GPU_USD_S})
        text = r.choices[0].message.content or ""
        m = re.search(r"\{.*\}", text, re.S)
        try:
            return json.loads(m.group(0)) if m else {}
        except json.JSONDecodeError:
            continue
    raise ValueError(f"Muse Glimmer returned unparseable JSON {attempts} times")


def log_cost(model: str, usd: float, **extra) -> None:
    COST_LOG.append({"model": model, "usd": usd, **extra})


def write_cost(out_dir: Path) -> float:
    total = sum(c["usd"] for c in COST_LOG)
    summary = {"usd": round(total, 4)}
    if any("gpu_secs" in c for c in COST_LOG):
        summary["gpu_secs"] = round(sum(c.get("gpu_secs", 0) for c in COST_LOG), 1)
    (out_dir / "cost.json").write_text(json.dumps({**summary, "calls": COST_LOG}, indent=2))
    return total
