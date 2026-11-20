# Football Highlight Generator

Upload a full football match recording. The app:

1. Extracts the audio and transcribes the commentary locally with [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (no cloud API, no cost).
2. Sends the transcript to an LLM (`gpt-5.6-luna`) in 15-minute chunks, which reads the commentary semantically and reports the moments worth watching — each classified as a **goal**, **foul**, **half time**, **full time**, or other highlight.
3. Audits the goals and whistles in a second pass over the **whole** transcript, reconciling them against the running score, so replays and recaps don't get reported as goals and no goal goes missing.
4. **Finds the scoreboard automatically and reads goals off it**, since the commentary describes a goal up to a minute after the ball crosses the line.
5. Picks the highest-importance, non-overlapping moments and cuts them into clips with ffmpeg.
6. Concatenates the clips into a single highlight reel you can preview and download, alongside the full timestamped transcript.
7. **Indexes the finished reel** by event: a clickable chapter list in the UI, a `chapters.json` you can download, and real chapter markers written into the mp4 so VLC/QuickTime/mpv/YouTube expose them as seekable chapters.

Video and audio never leave your PC — only the text transcript is sent to the LLM for moment detection.

## Requirements

- Python 3.10+ (tested on 3.14)
- [ffmpeg](https://ffmpeg.org/) available on your `PATH` (already installed on this machine via `winget install Gyan.FFmpeg` — if you're setting this up elsewhere: `winget install Gyan.FFmpeg`, then **restart your terminal** so the updated `PATH` takes effect)

## Setup

```powershell
py -m venv venv
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

If you have an NVIDIA GPU, also run:

```powershell
.\venv\Scripts\python.exe -m pip install -r requirements-gpu.txt
```

This installs the CUDA 12 cuBLAS/cuDNN runtime DLLs faster-whisper's GPU backend needs on Windows (skip it on CPU-only machines — it's a ~1.3GB download). The app auto-detects a working CUDA device at startup and uses it; no flags needed. GPU transcription is roughly **10-20x faster** than CPU for the same model size.

The first transcription run downloads the chosen Whisper model from Hugging Face and caches it locally — about **3 GB for the default `large-v3`**, a few hundred MB for `small`. It needs roughly 3-4 GB of VRAM at `large-v3` in float16, versus ~1 GB at `small`.

## Running

```powershell
.\venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Then open **http://127.0.0.1:8000** in your browser.

## Using the app

1. Drag & drop (or click to choose) a match video file — `.mp4`, `.mkv`, `.mov`, `.avi`, `.ts`, `.webm`, `.flv`, `.m4v`.
2. Adjust settings if you like:
   - **Target highlight length** — roughly how long the final reel should be.
   - **Sensitivity** — higher picks up more/looser moments; lower keeps only the clearest ones.
   - **Transcription model** — `large-v3` is the default and recommended: measured on an RTX 5060 it is only ~1.9x slower than `small` but is the difference between hearing "Chantalexander-Ols" and "Trent Alexander-Arnold". Drop to `small`/`medium` only if you are on CPU.
   - **Teams & players** (optional) — a comma-separated list of the names in this match. These are passed to Whisper as hotwords, which bias decoding on every window. Football commentary is full of proper nouns Whisper has never heard, and this is the cheapest fix for them.
3. Click **Generate Highlights** and wait. Processing is dominated by transcription. Measured on a 109-minute match with an RTX 5060: ~14 min with `small`, ~27 min with `large-v3`, plus ~7 min for analysis, cutting and encoding. CPU-only machines are far slower — drop the model size accordingly.
4. When done, preview the highlight video in the browser, jump around using the **Index**, and download the video, the transcript, and/or the index.

## The reel index

Every clip in the finished reel is labelled with the event that earned it a place:

| Category | What it marks |
| --- | --- |
| **Goal** | The ball goes in and the goal stands |
| **Foul** | Foul, free kick, penalty award, booking, sending off |
| **Half time** | The referee ends the first half |
| **Full time** | The final whistle |
| **Highlight** | Anything else worth watching — saves, big chances, near misses, controversial decisions |

Half time and full time are structural rather than exciting, so they'd never win a slot on importance alone. They're treated specially: one of each is always kept when the commentary contains it, with tighter padding, and they don't count against the target length — indexing the reel never costs it a goal.

The index is available three ways: the clickable list under the video (click an entry to seek), `GET /api/chapters/{job_id}` (or the **Download index** button, which saves `chapters.json`), and chapter markers embedded in the mp4 itself. Each entry carries both timelines — where the event sits in the reel (`reel_start_sec`) and where it happened in the original match (`match_clock`).

To add more categories later, extend `CATEGORIES`/`CATEGORY_LABELS` in `backend/llm_moments.py` and describe the new category in `SYSTEM_PROMPT`.

## Goals come from the scoreboard, not the commentary

The commentary is a bad clock. Measured frame-by-frame on a real match, Haaland's goal went in at **31:47** and the words "in the back of the net" arrived at **32:37** — fifty seconds later — because the commentator goes quiet during the roar and only describes the goal over the replay. No amount of prompt engineering fixes that; the information simply is not in the transcript at the right time.

The score bug burned into the top-left corner has no such lag. `backend/scoreboard.py` samples that region and reports every persistent change:

1. **Sample keyframes only.** Decoding all 50 fps of a 109-minute match takes over ten minutes; keyframes take ~30 seconds and land every 5 seconds, finer than the graphic's own update latency. (`-vf fps=1` does *not* help — it still decodes every frame before discarding.)
2. **Require the change to persist.** A goal holds the new score for the rest of the half; replays and graphics revert. Comparing a median of the frames before against a median of the frames after ignores anything transient.
3. **Require the change to be local.** This is the part that makes it work. Magnitude is useless — every false positive (kickoff, half-time and full-time graphics) produced a *larger* pixel change than either real goal. But a goal repaints **one** digit and leaves the other untouched, while a graphic repaints the whole panel. The quiet digit reads ~1.5 on a real goal and 10–47 on noise.

On the test match that gives exactly two detections, both goals, no false positives, timed to within 1–6 seconds, and the final score (1-1) derived from vision alone. Which digit changed also tells you which team scored.

The two sources then compose by their strengths: the **transcript** supplies who scored and at what score, the **scoreboard** supplies when and how many. A transcript goal with no corresponding score change is dropped as a misread replay; a score change with no transcript goal is added anyway. If the scan finds nothing — no bug, unusual layout, wrong region — the transcript's goals are kept untouched, because a detector that sees nothing must never erase a real goal.

### Finding the scoreboard on an unseen broadcast

`locate_score_region()` finds it automatically, using three signals. Each exists because the one before it was not enough on its own:

1. **Where is the graphic?** A scoreboard is rendered, not filmed, so it comes out pixel-identical frame after frame; grass and crowd never do, because of sensor noise. Measured: **825 identical pixels** in the top band of the frame, **0** in the bottom band of pure pitch.
2. **Which rows are the score?** The clock shares the panel and ticks every second, so counting how often each *row* changes splits them cleanly — **13-16 changes** for the score rows against **32-91** for the clock. Only the longest contiguous quiet run is kept, or one stray quiet row past the clock stretches the band back over it.
3. **Which columns are the digits?** Counting per column works, but *only* among frames where the panel is actually on screen. Without that filter every column shows ~12 changes — the panel being covered by replays and coming back — and the digits are invisible. With it, the crests read **0** and the digits read **3-4**.

On the test match this recovers x=66 y=32 w=46 h=18 against a hand-measured x=66 y=30 w=48 h=14, and detection still finds exactly the two goals.

Regions are stored as **fractions** of frame width and height, not pixels, so one region survives the same broadcaster shipping 480p, 720p or 1080p. `DEFAULT_REGION` is the measured Sky Sports box and is used only as a fallback when automatic location fails. If that is wrong too, the scan finds nothing and the transcript's goals stand — a wrong region costs you timing precision, never the reel.

## Why there are two detection passes

The chunked pass reads the match 15 minutes at a time, and that isolation causes two specific errors, both observed on a real match:

- It reported a **goal that never happened** — the pre-half-time recap of a goal scored 12 minutes earlier ("first touch with his right… that's his 70th goal") reads exactly like a live goal when you cannot see the rest of the match.
- It credited the equaliser to **the wrong player**, because the scorer is named in the analysis afterwards and in the full-time summary, not in the garbled seconds around the goal itself.

The reconciliation pass fixes both by reading the whole transcript at once and using the running score as a consistency check: work out the final score, then require the goals to add up to it. A phantom goal makes the arithmetic wrong, and a missing one leaves it short. On the match above it dropped the phantom, corrected the scorer, and moved the goal timestamp 70 seconds later onto the actual strike.

Fouls and general highlights are left to the chunked pass — those are local judgements it makes well. Only goals and the whistles need the whole picture. If the reconciliation call fails, the job keeps the unreconciled results rather than failing.

## How highlight detection works, and how to tune it

The detector never analyzes video frames — it only reads what was said. That means detection quality depends entirely on how audible and expressive the commentary is:

- Works best with commentary that verbally reacts to what happens. Unlike keyword matching, the LLM understands context: it can tell "he smashes it into the back of the net" is a goal while "50 Premier League goals, quicker than Andy Cole" is a stats aside.
- Struggles with flat commentary, music/crowd noise drowning out speech, or recordings with no commentator at all. If half time or full time are never verbally marked, they won't appear in the index.
- Every reported moment must be backed by an exact quote from the transcript that appears near the claimed timestamp, otherwise it's rejected — this catches both fabricated quotes and real quotes pinned to the wrong point in the match.
- **Sensitivity** sets the importance floor for regular highlights (fouls and other action). Goals, half time and full time bypass it entirely, and goals are also exempt from the target-length budget — a reel that drops a goal to stay under time is simply wrong, so the target governs the discretionary highlights around them.
- Cost is about **2.9 cents per 109-minute match** at current `gpt-5.6-luna` rates (~47k input / ~16k output tokens across 9 calls). `REASONING_EFFORT` and `RECONCILE_EFFORT` in `backend/llm_moments.py` are the dials: dropping the chunked pass back to `low` saves roughly 0.45 cents and some latency, at some cost to foul/highlight quality.
- If nothing gets detected at all, the app reports an error rather than producing an empty video.

Detection needs an OpenAI API key in `.env` at the repo root:

```
OPENAI_API_KEY=sk-...
```

## Project layout

```
backend/
  main.py           FastAPI app: upload / status / chapters / download endpoints
  jobs.py           In-memory background job tracking (single worker thread)
  pipeline.py       Orchestrates the end-to-end process per job, builds the chapter index
  transcription.py  faster-whisper wrapper
  llm_moments.py    LLM moment detection, event categories, and clip selection
  gpu_setup.py      Registers the NVIDIA runtime DLL dirs on Windows
  ffmpeg_utils.py   ffmpeg/ffprobe subprocess wrappers, incl. mp4 chapter embedding
frontend/
  index.html, style.css, app.js   Upload UI, progress polling, result playback + index
data/
  jobs/{job_id}/    Per-job input video, working files, output video, transcript, chapters.json
```

## Notes and limitations

- This is a **local, single-user** app: jobs are tracked in memory (lost on restart) and processed one at a time by a single worker thread, since transcription and video encoding are both CPU-heavy and don't benefit from running in parallel on typical hardware.
- Uploads are streamed to disk in chunks, so multi-GB match recordings are fine; make sure you have enough free disk space (roughly: source video + a duplicate of each selected clip + the final reel).
- Clip boundaries are re-encoded (not just container-copied) so cuts land precisely on the detected moment rather than snapping to the nearest keyframe.
- Clips are joined with a crossfade, so consecutive chapters overlap slightly. The index accounts for this: each chapter runs until the next one starts, and the last one ends with the reel.
- Job output files aren't automatically deleted — periodically clear out `data/jobs/` if disk space matters, or use the `DELETE /api/jobs/{job_id}` endpoint.
