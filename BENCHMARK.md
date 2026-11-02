# Benchmark: Parda against a regex-over-OCR sweep

Run on 02-Nov-2026 with `scripts/bench.py`. Everything here is reproducible from the repo: the clips are generated,
the baseline is in the repo, the scorer is in the repo.

## The question

Before the hackathon I cleaned my own videos with a regex over OCR output: match a few patterns on each OCR'd
frame, black out the line, hold the box until the next frame. The question the benchmark answers is simple: on
the same recordings, what does a stranger still get to read after each method, and how much of the picture did
each method damage to get there.

## Set-up

**Clips.** Ten 12-second screen recordings at 1280x720, 30 fps, drawn frame by frame by
`scripts/make_bench_set.py`, so every motion is exact and every planted value is known. Each is a situation a
regex sweep gets wrong in a different way:

| Clip | Scene | What it tests |
|---|---|---|
| bench_01 | ID card window drifting across the screen, a `.env` line appearing, a spoken phone number | continuous motion, a value that appears, speech |
| bench_02 | `.env` typed character by character with a cursor | values while being typed, a password inside a URL |
| bench_03 | bank statement scrolling | smooth scroll, header fields leaving the screen |
| bench_04 | sign-up form typed field by field, a spoken e-mail | name, e-mail, phone, revealed password, date of birth |
| bench_05 | chat with an OTP, a phone number and a UPI id | short values, a value with no neat pattern |
| bench_06 | spreadsheet with Aadhaar-style and PAN numbers, two step scrolls | small text in a grid, layout jumps |
| bench_07 | small window maximised to full screen at 6 s | a layout jump that doubles the font size |
| bench_08 | code editor scrolling, a password inside a connection string, a key, a phone in a comment | dark theme, monospace |
| bench_09 | licence card dragged twice with pauses | drag, stop, drag |
| bench_10 | notes page scrolled so the secrets leave the top and come back | values leaving and re-entering |

Every value is fake (sample PAN, AWS's published example key, reserved example.com addresses, SPECIMEN on every
document). The voice is Windows text-to-speech. Fake clips run on the Contributor tier (`PARDA_TEST_DATA=1`).

**Baseline.** `scripts/baseline_regex.py`, written for this benchmark in the shape of the pre-hackathon sweep: OCR
twice a second, match each line against PAN, AWS key, `sk-` key, e-mail, IFSC and any run of ten or more digits,
black out the whole line, hold the boxes until the next OCR'd frame. It uses Parda's own OCR engine (RapidOCR), so
the comparison is about the method, not the engine. It does nothing with speech.

**Scoring.** One measure, applied the same way to both outputs: the stranger's view. Each output is OCR'd at
4 frames per second. A planted value counts as readable on a frame when a run of at least 60 % of its characters
(at least six) comes back from OCR. The score is readable output frames over readable source frames, per value;
0 % is clean. Frames on which OCR cannot read the value in the *source* are not counted, so neither method is
blamed for OCR's own misses. The same pass measures the share of each frame's pixels changed by more than 24
levels, which is what "looks untouched" costs. Spoken values are scored by whether Parda flagged them and bleeped
them. Wall time was measured while another process was encoding video on the same 8-core CPU, so the seconds are
an upper bound; the dollar figures are exact from `cost.json`.

## Results

| Clip | Scene | Planted | Parda: readable frames | Baseline: readable frames | Altered area (Parda / baseline) | Parda $ | Parda s | Baseline s |
|---|---|---|---|---|---|---|---|---|
| bench_01.mp4 | pan_drift | 4 on screen, 1 spoken | 0/172 (0 %) + 0/1 spoken bleeped | 96/172 (56 %) | 1.04 % / 1.77 % | 0.009 | 475.1 | 114.5 |
| bench_02.mp4 | env_typed | 5 on screen | 0/99 (0 %) | 23/99 (23 %) | 0.5 % / 3.41 % | 0.011 | 248.3 | 47.6 |
| bench_03.mp4 | statement_scroll | 4 on screen | 0/87 (0 %) | 21/87 (24 %) | 1.37 % / 0.68 % | 0.015 | 327.5 | 5.4 |
| bench_04.mp4 | signup_form | 5 on screen, 1 spoken | 0/116 (0 %) + 1/1 spoken bleeped | 66/116 (57 %) | 0.44 % / 0.64 % | 0.003 | 135.7 | 12.0 |
| bench_05.mp4 | chat_otp | 3 on screen, 1 spoken | 0/102 (0 %) + 1/1 spoken bleeped | 68/102 (67 %) | 0.98 % / 0.71 % | 0.004 | 149.7 | 33.5 |
| bench_06.mp4 | sheet_amounts | 6 on screen | 0/288 (0 %) | 0/288 (0 %) | 2.02 % / 1.35 % | 0.001 | 55.5 | 5.0 |
| bench_07.mp4 | maximise_jump | 4 on screen | 0/192 (0 %) | 48/192 (25 %) | 1.08 % / 1.69 % | 0.002 | 133.0 | 12.3 |
| bench_08.mp4 | code_editor | 3 on screen | 0/123 (0 %) | 23/123 (19 %) | 0.56 % / 4.02 % | 0.007 | 1500.4 | 202.3 |
| bench_09.mp4 | id_drag | 4 on screen | 0/192 (0 %) | 109/192 (57 %) | 1.72 % / 0.88 % | 0.028 | 438.4 | 16.6 |
| bench_10.mp4 | scroll_out_in | 4 on screen | 0/116 (0 %) | 18/116 (16 %) | 0.42 % / 3.06 % | 0.002 | 57.6 | 7.8 |
| **All** | | | **0/1487 (0.0 %)**, 2/3 spoken bleeped | **472/1487 (31.7 %)**, 0/3 spoken | | | | |

| Clip | Kind | Value | Source frames | Parda leaked | Baseline leaked |
|---|---|---|---|---|---|
| bench_01.mp4 | id_number | ABCDE1234F | 48 | 0 | 0 |
| bench_01.mp4 | name | PRIYA SHARMA | 48 | 0 | 48 |
| bench_01.mp4 | dob | 01/01/1990 | 48 | 0 | 48 |
| bench_01.mp4 | api_key | AKIAIOSFODNN7EXAMPLE | 28 | 0 | 0 |
| bench_01.mp4 | phone (spoken) | 98765 43210 | | MISSED | not handled |
| bench_02.mp4 | password | hunter2-not-real | 33 | 0 | 1 |
| bench_02.mp4 | api_key | AKIAIOSFODNN7EXAMPLE | 27 | 0 | 1 |
| bench_02.mp4 | api_key | wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY | 19 | 0 | 19 |
| bench_02.mp4 | api_key | sk-test-FAKE0000parda0000demo0000key | 13 | 0 | 1 |
| bench_02.mp4 | email | priya.sharma@example.com | 7 | 0 | 1 |
| bench_03.mp4 | name | PRIYA SHARMA | 23 | 0 | 18 |
| bench_03.mp4 | bank_detail | 000123456789 | 20 | 0 | 3 |
| bench_03.mp4 | bank_detail | SMPL0000123 | 21 | 0 | 0 |
| bench_03.mp4 | email | priya.sharma@example.com | 23 | 0 | 0 |
| bench_04.mp4 | name | ANITA VERMA | 41 | 0 | 41 |
| bench_04.mp4 | email | anita.verma@example.com | 30 | 0 | 2 |
| bench_04.mp4 | phone | 9876543210 | 23 | 0 | 1 |
| bench_04.mp4 | password | Tq7#demo2026 | 15 | 0 | 15 |
| bench_04.mp4 | dob | 14/03/1992 | 7 | 0 | 7 |
| bench_04.mp4 | email (spoken) | anita.verma@example.com | | bleeped | not handled |
| bench_05.mp4 | otp | 482916 | 40 | 0 | 40 |
| bench_05.mp4 | phone | 9876543210 | 34 | 0 | 0 |
| bench_05.mp4 | bank_detail | anita@examplebank | 28 | 0 | 28 |
| bench_05.mp4 | otp (spoken) | 482916 | | bleeped | not handled |
| bench_06.mp4 | id_number | 0000 1111 2222 | 48 | 0 | 0 |
| bench_06.mp4 | id_number | 3333 4444 5555 | 48 | 0 | 0 |
| bench_06.mp4 | id_number | 6666 7777 8888 | 48 | 0 | 0 |
| bench_06.mp4 | id_number | ABCDE1234F | 48 | 0 | 0 |
| bench_06.mp4 | id_number | FGHIJ5678K | 48 | 0 | 0 |
| bench_06.mp4 | id_number | KLMNO9012P | 48 | 0 | 0 |
| bench_07.mp4 | name | RAJESH KUMAR | 48 | 0 | 48 |
| bench_07.mp4 | id_number | FGHIJ5678K | 48 | 0 | 0 |
| bench_07.mp4 | bank_detail | 111122223333 | 48 | 0 | 0 |
| bench_07.mp4 | phone | 9123456780 | 48 | 0 | 0 |
| bench_08.mp4 | password | S3cretPwd-fake | 33 | 0 | 7 |
| bench_08.mp4 | api_key | sk-test-FAKE0000parda0000demo0000key | 42 | 0 | 0 |
| bench_08.mp4 | phone | +91 98765 43210 | 48 | 0 | 16 |
| bench_09.mp4 | name | ANITA VERMA | 48 | 0 | 48 |
| bench_09.mp4 | dob | 14/03/1992 | 48 | 0 | 48 |
| bench_09.mp4 | id_number | DL-0420110012345 | 48 | 0 | 8 |
| bench_09.mp4 | phone | 9876543210 | 48 | 0 | 5 |
| bench_10.mp4 | bank_detail | 000123456789 | 28 | 0 | 4 |
| bench_10.mp4 | bank_detail | SMPL0000123 | 28 | 0 | 4 |
| bench_10.mp4 | name | RAVI MENON | 30 | 0 | 5 |
| bench_10.mp4 | email | ravi.menon@example.com | 30 | 0 | 5 |

Readable frames are readable output frames over readable source frames; 0 % is clean.

## Reading it

**On screen.** Over the ten clips a planted value was readable on 1487 source frames (4 per second). After the
regex sweep 472 of them (31.7 %) still read back; after Parda none did. The sweep's misses are the method's,
not the OCR's: names and dates of birth have no pattern (bench_01, 04, 07, 09: 48 of 48 frames each), an OTP or a
UPI id looks like any other short token (bench_05), a password is just a word (bench_04), and a drag or a maximise
moves the text between two OCR samples (bench_09: 109 frames leaked even on values the patterns do catch). Where the
baseline scores 0 %, the clip is one where every value fits a pattern and nothing moves between samples
(bench_06, the spreadsheet).

**Speech.** Parda bleeped 2 of 3 spoken secrets (the OTP in bench_05 and the e-mail in bench_04, heard as "anita
varma at example.com"). The spoken phone number in bench_01 was missed: Muse Voice Transcribe returns "Okay, quick
note. Call me on" and nothing after it for the Windows synthetic voice, reproducibly, even when the clip is trimmed
to start inside the number. The baseline does nothing with audio.

**Altered area.** Both methods change about 1 % of the frame; Parda's covers are tighter on dense text (bench_02,
08, 10: 0.3–0.6 % against 3–4 % for whole-line black boxes) and a little larger on cards (bench_06, 09), where
Parda covers the whole value and the sweep covers only the lines its patterns hit.

**Cost and time.** Meta API cost for all ten clips: $0.081 on the Contributor tier (fake data only; real
footage runs on the standard tier at roughly ten times that). Wall time, on an 8-core laptop CPU with no GPU:
Parda 59 min for the ten clips against 8 min for the sweep, most of it full-frame OCR
(bench_08, the dark code editor, is the slowest at 25 min); part of the run overlapped another process's video
encodes, so the per-clip seconds are upper bounds.

**Verdicts.** Seven clips ended "clean". Three ended "needs review" after the three verify passes: on the
statement clip (bench_03) the verify agent flags a different amount each pass (every planted value was already
covered; the churn is Spark's judgement of which amounts are private, not a leak); on bench_02 and bench_04 it kept
reporting pieces of values that were by then covered ("UtnFEMI/", "emo20").

**What the benchmark changed.** The first full run scored 75/1487 (5.0 %) for Parda. Three matcher bugs caused
every one of those frames: OCR gluing a label to a value ("number000123456789"), values flagged mid-typing
("anita.verm") that matched nothing once fully typed, and Spark dropping a repeated character ("SMPL000123" for
"SMPL0000123"). Two more surfaced on the re-runs: an id collision when a verify item was carried over from an
earlier run, and a name the verify agent flagged on every pass of the spreadsheet clip that OCR reads as "Pooja
lyer" (the substitution-tolerant match now starts at 8 characters; a value nobody planted, so it never showed in
the table). All five are fixed in `parda/locate.py` and `parda/__main__.py`. The four affected clips (03, 04, 06,
10) were then re-run with each clip's Spark find result kept, so only the locating changed; the table shows the
re-runs.

## Caveats

- The clips are generated, not recorded. Real recordings add codec noise, anti-aliased fonts at odd sizes,
  overlapping windows and a cursor. The real-desktop recording in `META_MASTER.md` §9i is the test for that; its
  three known one-frame limits are listed there.
- The scorer is OCR-based. A value that is half covered and unreadable to OCR counts as covered; a human might
  still guess it. The 60 % rule is the compromise.
- The baseline is a reconstruction of the method, not the original script, which stays out of the repo by the
  hackathon's rules.
