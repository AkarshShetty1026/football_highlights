"""In-memory background job tracking. This app runs single-user/local, so a
plain dict + one worker thread is enough -- no need for a database or queue."""

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Optional

from . import pipeline

# One worker: whisper transcription and ffmpeg encoding are both CPU-heavy,
# so running jobs concurrently would just make each one slower.
_executor = ThreadPoolExecutor(max_workers=1)
_jobs: dict[str, "Job"] = {}
_lock = threading.Lock()

STAGE_ORDER = [
    "extracting_audio", "transcribing", "analyzing", "reconciling", "scoreboard",
    "cutting_clips", "finalizing",
]
STAGE_LABELS = {
    "extracting_audio": "Extracting audio",
    "transcribing": "Transcribing commentary",
    "analyzing": "Detecting highlight moments",
    "reconciling": "Checking goals against the full match",
    "scoreboard": "Reading goals off the scoreboard",
    "cutting_clips": "Cutting clips",
    "finalizing": "Assembling and indexing highlight video",
}


@dataclass
class Job:
    id: str
    video_path: str
    output_video_path: str
    transcript_path: str
    chapters_path: str = ""
    status: str = "queued"  # queued | processing | done | error
    stage: str = ""
    stage_progress: float = 0.0
    error: Optional[str] = None
    result: Optional[dict] = None
    created_at: float = field(default_factory=time.time)

    @property
    def overall_progress(self) -> float:
        if self.status == "done":
            return 1.0
        if self.status in ("error", "queued"):
            return 0.0
        try:
            idx = STAGE_ORDER.index(self.stage)
        except ValueError:
            return 0.0
        stage_weight = 1.0 / len(STAGE_ORDER)
        return idx * stage_weight + self.stage_progress * stage_weight

    @property
    def stage_label(self) -> str:
        if self.status == "done":
            return "Done"
        if self.status == "error":
            return "Error"
        return STAGE_LABELS.get(self.stage, "Starting")


def create_job(
    job_id: str,
    video_path: str,
    output_video_path: str,
    transcript_path: str,
    chapters_path: str,
    work_dir: str,
    settings: pipeline.Settings,
) -> Job:
    job = Job(
        id=job_id,
        video_path=video_path,
        output_video_path=output_video_path,
        transcript_path=transcript_path,
        chapters_path=chapters_path,
    )
    with _lock:
        _jobs[job_id] = job
    _executor.submit(_run_job, job_id, work_dir, settings)
    return job


def _run_job(job_id: str, work_dir: str, settings: pipeline.Settings) -> None:
    job = _jobs[job_id]
    job.status = "processing"

    def progress_cb(stage: str, frac: float) -> None:
        job.stage = stage
        job.stage_progress = frac

    try:
        result = pipeline.process_video(
            job.video_path,
            job.output_video_path,
            job.transcript_path,
            work_dir,
            settings,
            progress_cb,
            chapters_path=job.chapters_path or None,
        )
        job.result = {
            "clip_count": result.clip_count,
            "highlight_duration_sec": result.highlight_duration_sec,
            "source_duration_sec": result.source_duration_sec,
            "chapters": result.chapters,
            "reconciliation": result.reconciliation,
            "scoreboard": result.scoreboard,
        }
        job.status = "done"
    except Exception as exc:  # noqa: BLE001 -- surface any failure to the UI
        job.status = "error"
        job.error = str(exc)


def get_job(job_id: str) -> Optional[Job]:
    with _lock:
        return _jobs.get(job_id)


def forget_job(job_id: str) -> None:
    with _lock:
        _jobs.pop(job_id, None)
