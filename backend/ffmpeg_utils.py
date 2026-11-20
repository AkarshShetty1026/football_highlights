"""Thin wrappers around the ffmpeg/ffprobe CLI for audio extraction and clip cutting."""

import shutil
import subprocess
from pathlib import Path

FFMPEG = "ffmpeg"
FFPROBE = "ffprobe"


class FFmpegError(RuntimeError):
    pass


def _run(args: list[str]) -> None:
    result = subprocess.run(args, capture_output=True, text=True)
    if result.returncode != 0:
        raise FFmpegError(f"Command failed: {' '.join(args)}\n{result.stderr[-4000:]}")


def get_duration(media_path: str) -> float:
    result = subprocess.run(
        [
            FFPROBE, "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            media_path,
        ],
        capture_output=True, text=True,
    )
    if result.returncode != 0 or not result.stdout.strip():
        raise FFmpegError(f"ffprobe failed on {media_path}\n{result.stderr[-2000:]}")
    return float(result.stdout.strip())


def extract_audio(video_path: str, wav_path: str, sample_rate: int = 16000) -> None:
    """Extract mono PCM16 WAV audio at the given sample rate (what faster-whisper expects)."""
    _run([
        FFMPEG, "-y", "-i", video_path,
        "-vn", "-ac", "1", "-ar", str(sample_rate),
        "-f", "wav", wav_path,
    ])


def cut_clip(video_path: str, start: float, end: float, out_path: str) -> None:
    """Cut [start, end) from video_path into out_path, re-encoding for frame-accurate boundaries."""
    duration = max(end - start, 0.1)
    _run([
        FFMPEG, "-y",
        "-ss", f"{start:.2f}", "-i", video_path, "-t", f"{duration:.2f}",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-c:a", "aac", "-b:a", "160k",
        "-avoid_negative_ts", "make_zero",
        out_path,
    ])


def concat_clips_crossfade(
    clip_paths: list[str], output_path: str, fade_duration: float = 0.6
) -> list[tuple[float, float]]:
    """Concatenate clips with a crossfade between each pair instead of a hard cut,
    via a chain of xfade (video) / acrossfade (audio) filters.

    Returns where each input clip lands on the output timeline, as (start, end)
    seconds -- that mapping is what the chapter index is built from. Consecutive
    spans overlap by the fade duration, since that is what a crossfade is."""
    if len(clip_paths) == 1:
        shutil.copyfile(clip_paths[0], output_path)
        return [(0.0, get_duration(output_path))]

    durations = [get_duration(p) for p in clip_paths]
    # Keep the fade well inside the shortest clip so xfade offsets never go negative.
    fade = max(min(fade_duration, min(durations) * 0.4), 0.05)

    inputs: list[str] = []
    for p in clip_paths:
        inputs += ["-i", p]

    video_parts = []
    audio_parts = []
    prev_v, prev_a = "0:v", "0:a"
    running_duration = durations[0]
    clip_offsets = [0.0]
    for i in range(1, len(clip_paths)):
        offset = max(running_duration - fade, 0.0)
        out_v, out_a = f"v{i}", f"a{i}"
        video_parts.append(
            f"[{prev_v}][{i}:v]xfade=transition=fade:duration={fade:.3f}:offset={offset:.3f}[{out_v}]"
        )
        audio_parts.append(f"[{prev_a}][{i}:a]acrossfade=d={fade:.3f}[{out_a}]")
        clip_offsets.append(offset)
        running_duration = running_duration + durations[i] - fade
        prev_v, prev_a = out_v, out_a

    filter_complex = ";".join(video_parts + audio_parts)
    _run([
        FFMPEG, "-y", *inputs,
        "-filter_complex", filter_complex,
        "-map", f"[{prev_v}]", "-map", f"[{prev_a}]",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-c:a", "aac", "-b:a", "160k",
        output_path,
    ])
    return [(o, o + d) for o, d in zip(clip_offsets, durations)]


def _escape_ffmetadata(value: str) -> str:
    for ch in ("\\", "=", ";", "#", "\n"):
        value = value.replace(ch, "\\" + ch)
    return value


def embed_chapters(video_path: str, chapters: list[tuple[float, float, str]], work_dir: str) -> None:
    """Write chapter markers into the mp4 container so players (VLC, QuickTime,
    mpv, YouTube) expose the index as seekable chapters. Remuxes with -c copy,
    so this is a metadata rewrite, not a re-encode."""
    if not chapters:
        return

    lines = [";FFMETADATA1"]
    for start, end, title in chapters:
        start_ms, end_ms = int(round(start * 1000)), int(round(end * 1000))
        if end_ms <= start_ms:
            end_ms = start_ms + 1
        lines += [
            "[CHAPTER]",
            "TIMEBASE=1/1000",
            f"START={start_ms}",
            f"END={end_ms}",
            f"title={_escape_ffmetadata(title)}",
        ]

    meta_path = Path(work_dir) / "chapters.ffmetadata"
    meta_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    video = Path(video_path)
    tagged = video.with_name(video.stem + ".chapters" + video.suffix)
    _run([
        FFMPEG, "-y", "-i", str(video), "-i", str(meta_path),
        "-map_metadata", "1", "-map_chapters", "1",
        "-c", "copy", str(tagged),
    ])
    tagged.replace(video)
