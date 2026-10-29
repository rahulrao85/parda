# Parda

**Record freely. Parda makes sure nothing private gets published.**

Parda ("curtain" in Hindi) is a privacy filter for videos. It finds private things in a screen recording or
talking-head clip, on screen and in speech, covers them (a mask that follows moving windows, a bleep for spoken
secrets), lets you approve each one, and renders a clean MP4 plus a privacy receipt (`audit.json`). A verify agent
then re-checks the rendered output until it is clean. A private mode runs the find step on Meta open weights so
footage never goes to a closed API.

Built for the Meta Global AI Developer Hackathon (Indie track, *Build for Your World*).

## Why
I publish screen recordings for my YouTube channel. One of them showed my tax-ID document for a few seconds and I
only noticed after it was live. Manual review of every frame does not scale, and regex-over-OCR misses anything
that is not a neat string. Parda is the step between "stop recording" and "publish".

## Meta models used
| Model | Job in Parda |
|---|---|
| Muse Spark (Meta Model API) | Reads sampled frames + OCR text and returns structured flags (what, where, why) |
| SAM 3.1 (Meta Model API) | Turns each flag box into a mask and follows it through the clip |
| Muse Voice Transcribe (Meta Model API) | Transcript of the audio, so spoken secrets get bleeped |
| Muse Spark Open (open weights) | Private mode: the same find step on open weights, on partner inference |

## Status
Work in progress. See *Built during the hackathon* below for what exists today.

## Built during the hackathon
Everything in this repository was written during the building period (27-Oct to 04-Nov-2026). The first commit is
inside that window.

- `scripts/make_fake_clip.py`: generates test clips with planted **fake** secrets (fake ID document, AWS's
  published example key, an `example.com` email, a spoken dummy phone number). No real personal data is used
  anywhere in this repo or its demo.
- `scripts/probe_models.py`: one small call to each Meta model to confirm access and measure cost.
- `python -m parda find clip.mp4`: the find step.
  - `parda/sample.py` samples 2 frames/s and keeps a frame only if a block of the screen changed, so static
    stretches cost nothing but a new terminal line still counts.
  - `parda/screen.py` sends frames to Muse Spark in batches of 8 and gets flags back as JSON (kind, text, box).
  - `parda/speech.py` runs Muse Voice Transcribe, has Muse Spark pick the secret phrases, then times them to the
    word with forced alignment on Meta's open MMS model.
  - Writes `flags.json`, `transcript.json` and `cost.json` (every model call and its price).
- `scripts/score_find.py`: recall against a clip's planted ground truth.
- `python -m parda redact clip.mp4`: find, then track, then render.
  - `parda/track.py` follows each flag with SAM 3.1 box prompts. Each request sees only the stretch of video
    around its items, and nested boxes (a document and the number on it) go in separate requests.
  - `parda/render.py` paints a solid, visible patch over every tracked box and bleeps every spoken secret.
    No blur or pixelation, because pixelated text can be reconstructed.
  - `parda/verify.py` then scans the **rendered output** the way a viewer would: fresh Muse Spark passes over
    its frames and a fresh Voice Transcribe of its audio. Anything still visible or audible becomes a new item,
    gets tracked on the source and re-rendered, and the output is checked again (up to 3 passes).
  - Writes `clean.mp4` and `audit.json`: every item, which pass found it, each verify pass, the verdict
    (`clean` / `needs review`) and the SHA-256 of the clean file.
  - `--drop s3 a2` leaves chosen flags out on purpose, to show the verify loop catching what the first pass
    missed.
- Spoken secrets: Muse Spark picks them from the transcript, and deterministic rules (email patterns, runs of 5+
  spoken digits, Hindi number words included) back it up, because one model call is not the same every time.
- `scripts/make_props.py`: fake documents (SPECIMEN PAN card, bank statement, `.env`) for recording realistic test
  clips on a real desktop.

Real footage always goes to the standard Muse Spark model. The cheaper Contributor tier, where Meta may train on
inputs, is used only for the fake test clips (`PARDA_TEST_DATA=1`).

**Baseline, not included:** before the hackathon I used a regex-over-EasyOCR sweep for my own videos. That code is
not in this repo; it is only the baseline Parda's recall is compared against.

## Setup
```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt   # Windows; use .venv/bin/pip elsewhere
```
Put your key in a `.env` file (this repo or any parent folder): `META_API_KEY=...`. ffmpeg must be on `PATH`.

## Licence
MIT
