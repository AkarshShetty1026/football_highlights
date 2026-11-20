"""Local speech-to-text via faster-whisper (CTranslate2 build of OpenAI Whisper)."""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from faster_whisper import WhisperModel

_MODEL_CACHE: dict[str, tuple[WhisperModel, str]] = {}

# Whisper hears football commentary badly without priming: measured output on a
# real match included "Harlan" (Haaland), "Soverslite" (Szoboszlai) and
# "Chantalexander-Ols" (Trent Alexander-Arnold). Hotwords bias decoding toward
# the domain vocabulary, and unlike initial_prompt they apply to every window
# rather than just the first (which matters over a 90-minute match).
FOOTBALL_HOTWORDS = (
    "goal, penalty, offside, corner, free kick, yellow card, red card, booking, "
    "sending off, VAR, half-time, full-time, equaliser, header, cross, volley, "
    "goalkeeper, referee, substitution, injury time, back of the net"
)

# faster-whisper truncates the hotword prompt at half the decoder context, so
# keep the combined string well short of that.
MAX_HOTWORDS_CHARS = 600


def build_hotwords(extra: str = "") -> str:
    """Combine the standing football vocabulary with any match-specific names
    the user supplied (team and player names are what Whisper mangles most)."""
    combined = f"{extra.strip()}, {FOOTBALL_HOTWORDS}" if extra.strip() else FOOTBALL_HOTWORDS
    return combined[:MAX_HOTWORDS_CHARS]


@dataclass
class Segment:
    start: float
    end: float
    text: str


def _pick_device_and_compute_type() -> tuple[str, str]:
    try:
        import ctranslate2

        if ctranslate2.get_cuda_device_count() > 0:
            return "cuda", "float16"
    except Exception:
        pass
    return "cpu", "int8"


def get_model(model_size: str = "large-v3") -> tuple[WhisperModel, str]:
    if model_size not in _MODEL_CACHE:
        device, compute_type = _pick_device_and_compute_type()
        # Deliberately not overriding cpu_threads: benchmarking showed forcing
        # it to the full logical core count made CPU transcription *slower*
        # (measured ~8x vs ~16x realtime factor for the "small" model) than
        # ctranslate2's own default, likely due to thread oversubscription.
        model = WhisperModel(model_size, device=device, compute_type=compute_type)
        _MODEL_CACHE[model_size] = (model, device)
    return _MODEL_CACHE[model_size]


def transcribe(
    wav_path: str,
    model_size: str = "large-v3",
    progress_cb: Optional[Callable[[float, float], None]] = None,
    hotwords: str = "",
) -> tuple[list[Segment], float]:
    model, device = get_model(model_size)
    # Beam search is cheap on GPU but a real cost on CPU -- fall back to
    # greedy decoding there rather than paying 3-5x for marginal accuracy.
    beam_size = 5 if device == "cuda" else 1
    segments_iter, info = model.transcribe(
        wav_path,
        beam_size=beam_size,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500},
        condition_on_previous_text=False,
        hotwords=build_hotwords(hotwords),
    )
    segments: list[Segment] = []
    total_duration = info.duration or 0.0
    for seg in segments_iter:
        segments.append(Segment(seg.start, seg.end, seg.text.strip()))
        if progress_cb and total_duration > 0:
            progress_cb(seg.end, total_duration)
    return segments, total_duration


def save_transcript(segments: list[Segment], out_path: str) -> None:
    def fmt(t: float) -> str:
        h, rem = divmod(int(t), 3600)
        m, s = divmod(rem, 60)
        return f"{h:02d}:{m:02d}:{s:02d}"

    lines = [f"[{fmt(seg.start)} - {fmt(seg.end)}] {seg.text}" for seg in segments]
    Path(out_path).write_text("\n".join(lines), encoding="utf-8")


_LINE_RE = re.compile(r"\[(\d+):(\d+):(\d+)\s*-\s*(\d+):(\d+):(\d+)\]\s*(.*)")


def load_transcript(path: str) -> list[Segment]:
    """Read back a transcript written by save_transcript().

    Transcription is by far the most expensive stage, so a job whose transcript
    already exists can be re-run from here -- to recover an interrupted job, or
    to try different highlight settings without paying for Whisper again."""
    segments: list[Segment] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        m = _LINE_RE.match(line.strip())
        if not m:
            continue
        g = [int(v) for v in m.groups()[:6]]
        start = g[0] * 3600 + g[1] * 60 + g[2]
        end = g[3] * 3600 + g[4] * 60 + g[5]
        segments.append(Segment(float(start), float(end), m.group(7)))
    return segments
