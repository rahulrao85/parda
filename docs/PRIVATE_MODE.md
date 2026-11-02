# Private mode: find on-screen secrets with Muse Glimmer on your own GPU

By default Parda sends sampled frames to Muse Spark on the Meta Model API. Private mode sends the on-screen find
step to **Muse Glimmer**, Meta's open-weight 30B vision model (Apache 2.0), served from your own Modal account. In
this mode, frames never leave infrastructure you control. OCR (`parda/locate.py`) and rendering (`parda/render.py`)
are local in both modes.

| | Default | `--private` |
|---|---|---|
| Which frames hold secrets | Muse Spark (Meta Model API) | Muse Glimmer 30B on your Modal GPU |
| Where each value is on every frame | local OCR | local OCR |
| Speech | Muse Voice Transcribe + Muse Spark | same (see Known limits) |
| Verify loop (re-scan the output) | Muse Spark | Muse Glimmer |

## Model

- Weights: [`RedHatAI/Muse-Glimmer-30B-FP8-block`](https://huggingface.co/RedHatAI/Muse-Glimmer-30B-FP8-block) at
  revision `1deb4641`. This is Red Hat's FP8 (W8A8 block) build of
  [`meta-models/Muse-Glimmer-30B`](https://huggingface.co/meta-models/Muse-Glimmer-30B), 34.4 GB on disk, with the
  vision tower and embeddings kept in BF16. Neither repo is gated. The BF16 original is 59.6 GB and does not fit
  on one 48 GB GPU.
- Server: vLLM 0.32.0 (`MuseGlimmerForConditionalGeneration`, image input), with `--reasoning-parser muse_glimmer`
  so the answer arrives as plain `content`.
- GPU: one L40S (48 GB, $0.000542/s = $1.95/h on Modal, 02-Nov-2026), the cheapest Modal GPU that holds the FP8
  weights plus KV cache for an 8-frame batch. To use another GPU, set `PARDA_GLIMMER_GPU` (e.g. `A100-80GB`) at
  deploy time.

## Deploy (once)

Needs a Modal account **with a payment method on file**. Modal refuses every GPU type, free credits included,
until one is added. Run these from the repo root:

```bash
pip install 'modal[api-proxy-support]'   # the extra is only needed behind an HTTP proxy
modal token set --token-id <ak-...> --token-secret <as-...>
```

```bash
modal secret create parda-glimmer-key VLLM_API_KEY="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
```

Keep the same key for the client (step 4 below). Then fetch the weights into a Modal Volume on a CPU container
(about 2.5 minutes, no GPU billed):

```bash
modal run deploy/modal_glimmer.py::download
```

```bash
modal deploy deploy/modal_glimmer.py
```

`modal deploy` prints the endpoint URL, `https://<workspace>--parda-glimmer-serve.modal.run`.

The endpoint:

- **Scales to zero** after 5 idle minutes (`scaledown_window=300`) and runs at most one container.
- **Is never open.** Every `/v1` request needs `Authorization: Bearer <VLLM_API_KEY>` (vLLM `--api-key`). Without
  the Secret, the container refuses to start. Only vLLM's `/health` and `/metrics` answer without the key; they
  expose no data.

## Run

Put these in `.env` (git-ignored) or the environment:

```
PARDA_PRIVATE_URL=https://<workspace>--parda-glimmer-serve.modal.run
PARDA_PRIVATE_KEY=<the VLLM_API_KEY above>
PARDA_PRIVATE_MODEL=muse-glimmer
```

```bash
python -m parda find samples/fake_pan_12s.mp4 -o out/fake_pan_12s_private --private
```

```bash
python scripts/score_find.py samples/fake_pan_12s.truth.json out/fake_pan_12s_private/flags.json
```

```bash
python -m parda redact samples/fake_pan_12s.mp4 -o out/fake_pan_12s_private --private --verify 1
```

`PARDA_BACKEND=private` does the same as `--private`. Stop the GPU when done (it also stops by itself after 5
idle minutes):

```bash
modal app stop parda-glimmer
```

### Settings that apply only in private mode (`parda/api.py`)

| Variable | Default | Meaning |
|---|---|---|
| `PARDA_PRIVATE_BATCH` | 8 | frames per Glimmer call; 4 also parses but is slower (138 s vs 107–120 s) |
| `PARDA_PRIVATE_GPU_USD_S` | 0.000542 | GPU price used for the cost log (Modal L40S) |

Glimmer gets one extra system message, `Reasoning strength: low`. The screen prompt itself (`screen-v4`) is the
same text Spark gets. Screen-cache keys include the model name, so Spark and Glimmer results never mix.

## Cost log

Each Glimmer call in `cost.json` records `secs` (request wall time), `start`/`end`, and `gpu_secs`: its share of
the GPU's busy time. While n calls overlap, each is charged 1/n of that stretch, so the shares add up to the
time the GPU served the run. `usd` is `gpu_secs` × `PARDA_PRIVATE_GPU_USD_S`, and the
top-level `gpu_secs` is the run total. This is the GPU time Parda's own requests used. Modal also bills cold
start and the 5-minute idle tail; those are listed separately below.

## Measured on `samples/fake_pan_12s.mp4` (12 s, 1280x720, fake data), 02-Nov-2026

Both runs saw the same 24 sampled frames, with no cache hits. Spark is `muse-spark-1.3-contributor`; Glimmer is the
FP8 build on one Modal L40S, already warm. Both use batches of 8 frames.

| | Muse Spark | Muse Glimmer (private) |
|---|---|---|
| On-screen recall (2 planted) | **2/2** | **2/2** |
| On-screen items flagged | 3 (name, PAN number, AWS key) | the same 3, tight boxes |
| Total recall vs truth (incl. speech) | 4/4 | 3/4 (see below) |
| Screen calls | 3 × 8 frames, slowest 15.6 s | 3 × 8 frames, slowest 76.0 s |
| `find` wall time | ≈ 45 s | 120 s |
| Cold start | none | 181 s the first time, 85 s after (compile cache in a Volume) |
| GPU time for the clip | — | 76.1 GPU-seconds |
| Screen cost for the clip | $0.0039 | $0.041 of GPU time |
| **Screen cost per minute of video** | **≈ $0.02** | **≈ $0.21** while warm |

The speech miss is not Glimmer's. Speech runs the same code in both modes (Muse Voice Transcribe, then Muse Spark).
In the private runs, Voice Transcribe split the sentence and dropped "nine eight seven six five" from the transcript,
so the phone number never reached the phrase-picking step. Each run transcribes again, and the Spark run happened
to get one clean turn.

`redact --private --verify 1` from that `find` wrote `clean.mp4` with 360 of 360 frames, all covered, and 2 bleeps.
Glimmer's verify pass re-scanned the output and found nothing still readable: verdict **clean**. The verify pass
took 36 s and 29.5 GPU-seconds.

A batch of 4 frames also parses (6 calls), but the run took 138 s instead of 107–120 s, so 8 is the default.

What one session costs beyond the per-minute rate:

- A cold start is 85 s of L40S, about $0.05 (181 s, about $0.10, the very first time).
- The idle tail before scale-to-zero is 300 s, about $0.16. Run `modal app stop parda-glimmer` to skip it.

All of this work used **$0.98 of Modal credit**: the weight download, image builds, three cold starts, five
private runs and the idle tails. Nothing was billed to the card.

## Known limits

- **Speech still uses Meta's API.** Muse Voice Transcribe turns the audio into text, and Muse Spark picks the
  spoken secrets from the transcript. Private mode keeps frames off the API, not audio. Moving speech to an
  open model (e.g. MMS ASR, which Parda already uses for word timing) is the next step.
- The FP8 build is Red Hat's quantization of Meta's weights, not Meta's own file.
- The first request after an idle period waits for a cold start: 85 s (container start, 15 s to load 34 GB of
  weights from the Volume, and vLLM warm-up).
- Glimmer is about 5x slower than Spark per call (76 s vs 15.6 s for 8 frames) and about 10x dearer per minute
  while warm. Private mode trades speed and cost for keeping frames on your own GPU.
- The Modal URL shows as "unauthenticated" in `modal deploy`, meaning Modal's own proxy login is off. vLLM's
  `--api-key` guards every `/v1` route; requests with no key or a wrong key get 401 (checked 02-Nov-2026).
