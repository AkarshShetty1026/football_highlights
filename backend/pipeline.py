"""End-to-end orchestration: match video in, highlight reel + transcript out."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from . import ffmpeg_utils, llm_moments, scoreboard, transcription
from .llm_moments import HighlightWindow


@dataclass
class Settings:
    model_size: str = "large-v3"       # tiny | base | small | medium | large-v3
    target_duration_sec: float = 300.0  # desired total highlight reel length
    sensitivity: float = 0.5            # 0..1, higher = more/looser highlights
    clip_pre: float = 8.0               # seconds kept before a detected moment
    clip_post: float = 6.0              # seconds kept after a detected moment
    hotwords: str = ""                  # team/player names to prime transcription
    reconcile: bool = True              # audit goals against the whole match
    use_scoreboard: bool = True         # ground goal timing in the on-screen score
    score_region: object = None         # None = find the scoreboard automatically
    reuse_transcript: bool = False      # resume from an existing transcript


@dataclass
class PipelineResult:
    output_video_path: str
    transcript_path: str
    chapters_path: str
    clip_count: int
    highlight_duration_sec: float
    source_duration_sec: float
    windows: list[HighlightWindow]
    chapters: list[dict]
    reconciliation: dict
    scoreboard: dict


ProgressCB = Callable[[str, float], None]


def _format_clock(seconds: float) -> str:
    seconds = max(int(round(seconds)), 0)
    h, m, s = seconds // 3600, (seconds % 3600) // 60, seconds % 60
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def _build_chapters(
    windows: list[HighlightWindow],
    reel_spans: list[tuple[float, float]],
    reel_duration: float,
) -> list[dict]:
    """Map each selected window onto its position in the finished reel. Chapter
    spans are made contiguous (each runs until the next one starts) so players
    treat them as a continuous index rather than as gapped bookmarks."""
    chapters: list[dict] = []
    for i, (window, (reel_start, reel_end)) in enumerate(zip(windows, reel_spans)):
        next_start = reel_spans[i + 1][0] if i + 1 < len(reel_spans) else reel_duration
        chapters.append({
            "index": i + 1,
            "category": window.category,
            "label": window.label,
            "title": f"{window.label} · {_format_clock(window.start)}",
            "reel_start_sec": round(max(reel_start, 0.0), 3),
            "reel_end_sec": round(min(max(next_start, reel_start), reel_duration), 3),
            "clip_end_sec": round(min(reel_end, reel_duration), 3),
            "match_start_sec": round(window.start, 3),
            "match_end_sec": round(window.end, 3),
            "match_clock": _format_clock(window.start),
            "importance": round(window.importance, 3),
            "reason": window.reason,
        })
    return chapters


def process_video(
    video_path: str,
    output_video_path: str,
    transcript_path: str,
    work_dir: str,
    settings: Settings,
    progress_cb: ProgressCB = lambda stage, frac: None,
    chapters_path: str | None = None,
) -> PipelineResult:
    work_dir_p = Path(work_dir)
    work_dir_p.mkdir(parents=True, exist_ok=True)
    wav_path = work_dir_p / "audio.wav"
    chapters_file = Path(chapters_path) if chapters_path else Path(output_video_path).with_name("chapters.json")

    progress_cb("extracting_audio", 0.0)
    source_duration = ffmpeg_utils.get_duration(video_path)
    ffmpeg_utils.extract_audio(video_path, str(wav_path))
    progress_cb("extracting_audio", 1.0)

    progress_cb("transcribing", 0.0)

    def whisper_progress(current: float, total: float) -> None:
        progress_cb("transcribing", min(current / total, 1.0) if total else 0.0)

    existing = Path(transcript_path)
    if settings.reuse_transcript and existing.exists():
        # Transcription dominates the runtime, so recovering an interrupted job
        # (or retrying with different highlight settings) should never pay for
        # it twice.
        segments = transcription.load_transcript(transcript_path)
        audio_duration = segments[-1].end if segments else 0.0
        print(f"pipeline: reusing existing transcript ({len(segments)} segments)")
    else:
        segments, audio_duration = transcription.transcribe(
            str(wav_path), settings.model_size, whisper_progress, hotwords=settings.hotwords
        )
        transcription.save_transcript(segments, transcript_path)
    progress_cb("transcribing", 1.0)

    progress_cb("analyzing", 0.0)
    moments = llm_moments.detect_moments(segments, audio_duration)
    progress_cb("analyzing", 1.0)

    # The chunked pass above reads the match 15 minutes at a time, which is why
    # it mistakes replays for goals and misses ones it has no score context for.
    # One whole-match pass fixes both.
    reconciliation: dict = {"ran": False}
    if settings.reconcile:
        progress_cb("reconciling", 0.0)
        moments, reconciliation = llm_moments.reconcile_key_events(moments, segments)
        progress_cb("reconciling", 1.0)

    # The transcript says who scored and at what score; the scoreboard says
    # when, and how many. Vision wins on timing -- commentary describes a goal
    # up to a minute after the ball crosses the line.
    scoreboard_info: dict = {"used": False}
    if settings.use_scoreboard:
        progress_cb("scoreboard", 0.0)
        try:
            # Find the scoreboard on this particular broadcast. Falls back to the
            # known-good region rather than giving up, and if that is wrong too
            # the scan simply finds nothing and the transcript's goals stand.
            region, located = settings.score_region, bool(settings.score_region)
            if region is None:
                region = scoreboard.locate_score_region(video_path)
                located = region is not None
                if region is None:
                    print("scoreboard: could not locate a scoreboard, using the default region")
                    region = scoreboard.DEFAULT_REGION
            progress_cb("scoreboard", 0.3)
            changes = scoreboard.find_goals(video_path, region=region)
            progress_cb("scoreboard", 0.7)
            moments, scoreboard_info = llm_moments.apply_scoreboard_goals(
                moments, changes, final_score=reconciliation.get('final_score', '')
            )
            box = region.to_pixels(*scoreboard.video_dimensions(video_path))
            scoreboard_info["region"] = {"x": box.x, "y": box.y, "w": box.w, "h": box.h,
                                         "auto_located": located}
        except Exception as exc:  # noqa: BLE001 -- vision is an upgrade, not a dependency
            print(f"pipeline: scoreboard scan failed, keeping transcript goals: {exc}")
            scoreboard_info = {"used": False, "error": str(exc)}
        progress_cb("scoreboard", 1.0)

    windows = llm_moments.select_highlight_windows(
        moments,
        settings.target_duration_sec,
        sensitivity=settings.sensitivity,
        clip_pre=settings.clip_pre,
        clip_post=settings.clip_post,
    )

    if not windows:
        raise RuntimeError(
            "No highlight-worthy moments were detected. Try raising the sensitivity "
            "setting, or check that the uploaded video has an audible commentary track."
        )

    progress_cb("cutting_clips", 0.0)
    clips_dir = work_dir_p / "clips"
    clips_dir.mkdir(exist_ok=True)
    clip_paths = []
    for i, window in enumerate(windows):
        clip_path = clips_dir / f"clip_{i:03d}.mp4"
        ffmpeg_utils.cut_clip(video_path, window.start, window.end, str(clip_path))
        clip_paths.append(str(clip_path))
        progress_cb("cutting_clips", (i + 1) / len(windows))

    progress_cb("finalizing", 0.0)
    reel_spans = ffmpeg_utils.concat_clips_crossfade(clip_paths, output_video_path)
    progress_cb("finalizing", 0.6)

    reel_duration = ffmpeg_utils.get_duration(output_video_path)
    chapters = _build_chapters(windows, reel_spans, reel_duration)
    chapters_file.write_text(
        json.dumps(
            {"reel_duration_sec": round(reel_duration, 3), "chapters": chapters},
            indent=2,
        ),
        encoding="utf-8",
    )
    ffmpeg_utils.embed_chapters(
        output_video_path,
        [(c["reel_start_sec"], c["reel_end_sec"], c["title"]) for c in chapters],
        str(work_dir_p),
    )
    progress_cb("finalizing", 1.0)

    return PipelineResult(
        output_video_path=output_video_path,
        transcript_path=transcript_path,
        chapters_path=str(chapters_file),
        clip_count=len(windows),
        highlight_duration_sec=sum(w.end - w.start for w in windows),
        source_duration_sec=source_duration,
        windows=windows,
        chapters=chapters,
        reconciliation=reconciliation,
        scoreboard=scoreboard_info,
    )
