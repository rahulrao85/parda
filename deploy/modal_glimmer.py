"""Muse Glimmer on Modal: an OpenAI-compatible chat-completions server for Parda's private mode.

Meta's open-weight vision model Muse Glimmer 30B (Apache 2.0), as Red Hat's FP8 build of meta-models/Muse-Glimmer-30B,
served by vLLM on one L40S. Weights live in a Modal Volume, the container scales to zero after 5 idle minutes, and
every /v1 request needs the bearer key stored in the Modal Secret "parda-glimmer-key" (vLLM --api-key).

    modal secret create parda-glimmer-key VLLM_API_KEY=<random key>   # once; keep the same key as PARDA_PRIVATE_KEY
    modal run deploy/modal_glimmer.py::download                       # once: ~34 GB into the Volume, CPU only
    modal deploy deploy/modal_glimmer.py                              # prints the URL to use as PARDA_PRIVATE_URL
    modal app stop parda-glimmer                                      # when done

See docs/PRIVATE_MODE.md for running Parda against it and for measured cost.
"""
import os
import subprocess

import modal

APP = "parda-glimmer"
MODEL_REPO = "RedHatAI/Muse-Glimmer-30B-FP8-block"  # FP8 W8A8 block of meta-models/Muse-Glimmer-30B; vision tower BF16
MODEL_REV = "1deb4641ff84f9a728dd11b27cac1f6a02a9ed14"
MODEL_DIR = f"/models/{MODEL_REPO.split('/')[-1]}"
SERVED_NAME = "muse-glimmer"
GPU = os.getenv("PARDA_GLIMMER_GPU", "L40S")  # 48 GB: 33 GB of FP8 weights + KV for one 8-frame batch; the cheapest Modal GPU that fits
IDLE_S = 300            # scale to zero after 5 idle minutes
PORT = 8000

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("vllm==0.32.0", "huggingface_hub[hf_xet]>=1.0")
    # FlashInfer's sampler JIT-compiles with nvcc, which the slim image lacks; vLLM's own sampler needs no compiler
    .env({"HF_XET_HIGH_PERFORMANCE": "1", "VLLM_USE_V1": "1", "VLLM_USE_FLASHINFER_SAMPLER": "0"})
)
weights = modal.Volume.from_name("parda-glimmer-weights", create_if_missing=True)
compile_cache = modal.Volume.from_name("parda-glimmer-vllm-cache", create_if_missing=True)  # shorter cold starts
app = modal.App(APP)
fetch = modal.App(APP + "-download")  # its own app, so fetching weights needs no endpoint Secret


@fetch.function(image=image, volumes={"/models": weights}, cpu=4, memory=8192, timeout=3600)
def download() -> None:
    """Fetch the pinned checkpoint into the Volume once, on a CPU container (no GPU billed)."""
    from huggingface_hub import snapshot_download

    snapshot_download(MODEL_REPO, revision=MODEL_REV, local_dir=MODEL_DIR)
    weights.commit()
    print(f"{MODEL_REPO}@{MODEL_REV[:8]} -> {MODEL_DIR}")


@app.function(image=image, gpu=GPU, volumes={"/models": weights, "/root/.cache/vllm": compile_cache},
              secrets=[modal.Secret.from_name("parda-glimmer-key")], scaledown_window=IDLE_S, timeout=3600,
              max_containers=1)
@modal.concurrent(max_inputs=8)
@modal.web_server(port=PORT, startup_timeout=1200)
def serve() -> None:
    if not os.environ.get("VLLM_API_KEY"):
        raise RuntimeError("Modal Secret parda-glimmer-key must set VLLM_API_KEY; the endpoint is never served open")
    cmd = ["vllm", "serve", MODEL_DIR, "--served-model-name", SERVED_NAME, "--host", "0.0.0.0", "--port", str(PORT),
           "--max-model-len", "32768", "--gpu-memory-utilization", "0.92", "--max-num-seqs", "8",
           "--limit-mm-per-prompt", '{"image": 8}', "--generation-config", "auto",
           "--reasoning-parser", "muse_glimmer"]  # vLLM reads VLLM_API_KEY from the environment: every /v1 call needs it
    subprocess.Popen(cmd)
