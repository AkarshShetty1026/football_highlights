"""FastAPI app: upload a match video, poll job status, download the highlight reel."""

import shutil
import uuid
from pathlib import Path

from .gpu_setup import register_nvidia_dll_dirs

register_nvidia_dll_dirs()  # must run before any faster_whisper/ctranslate2 import below

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from . import jobs
from .pipeline import Settings

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
JOBS_DIR = DATA_DIR / "jobs"
FRONTEND_DIR = BASE_DIR / "frontend"

JOBS_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_SUFFIXES = {".mp4", ".mkv", ".mov", ".avi", ".ts", ".webm", ".flv", ".m4v"}
UPLOAD_CHUNK_SIZE = 8 * 1024 * 1024  # 8MB, keeps multi-GB uploads off the heap

app = FastAPI(title="Football Highlight Generator")
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    return HTMLResponse((FRONTEND_DIR / "index.html").read_text(encoding="utf-8"))


@app.post("/api/upload")
async def upload(
    file: UploadFile = File(...),
    target_minutes: float = Form(5.0),
    sensitivity: float = Form(0.5),
    model_size: str = Form("large-v3"),
    hotwords: str = Form(""),
):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(
            400,
            f"Unsupported file type '{suffix}'. Allowed: {', '.join(sorted(ALLOWED_SUFFIXES))}",
        )
    if model_size not in {"tiny", "base", "small", "medium", "large-v3"}:
        raise HTTPException(400, f"Unknown model size '{model_size}'")

    job_id = uuid.uuid4().hex[:12]
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    video_path = job_dir / f"input{suffix}"
    with video_path.open("wb") as out:
        while chunk := await file.read(UPLOAD_CHUNK_SIZE):
            out.write(chunk)

    output_video_path = job_dir / "highlights.mp4"
    transcript_path = job_dir / "transcript.txt"
    chapters_path = job_dir / "chapters.json"
    work_dir = job_dir / "work"

    settings = Settings(
        model_size=model_size,
        target_duration_sec=max(target_minutes, 0.5) * 60.0,
        sensitivity=min(max(sensitivity, 0.0), 1.0),
        hotwords=hotwords.strip()[:500],
    )

    jobs.create_job(
        job_id,
        str(video_path),
        str(output_video_path),
        str(transcript_path),
        str(chapters_path),
        str(work_dir),
        settings,
    )
    return {"job_id": job_id}


@app.post("/api/reprocess/{job_id}")
def reprocess(job_id: str, target_minutes: float = Form(8.0), sensitivity: float = Form(0.5)):
    """Re-run a job from its existing files.

    Works off the job DIRECTORY, not the in-memory registry, so it recovers a
    job whose server restarted -- the transcript is the expensive artifact and
    it is already on disk."""
    job_dir = JOBS_DIR / job_id
    video = next((p for p in job_dir.glob("input.*") if p.suffix.lower() in ALLOWED_SUFFIXES), None)
    transcript_path = job_dir / "transcript.txt"
    if not job_dir.is_dir() or video is None:
        raise HTTPException(404, f"No job directory with an input video for '{job_id}'")
    if not transcript_path.exists():
        raise HTTPException(409, "That job has no transcript yet, so there is nothing to resume from")

    existing = jobs.get_job(job_id)
    if existing is not None and existing.status == "processing":
        raise HTTPException(409, "That job is already running")

    settings = Settings(
        target_duration_sec=max(target_minutes, 0.5) * 60.0,
        sensitivity=min(max(sensitivity, 0.0), 1.0),
        reuse_transcript=True,
    )
    jobs.forget_job(job_id)
    jobs.create_job(
        job_id,
        str(video),
        str(job_dir / "highlights.mp4"),
        str(transcript_path),
        str(job_dir / "chapters.json"),
        str(job_dir / "work"),
        settings,
    )
    return {"job_id": job_id, "resumed": True}


@app.get("/api/status/{job_id}")
def status(job_id: str):
    job = jobs.get_job(job_id)
    if job is None:
        raise HTTPException(404, "Unknown job id")
    return {
        "status": job.status,
        "stage": job.stage,
        "stage_label": job.stage_label,
        "progress": round(job.overall_progress, 4),
        "error": job.error,
        "result": job.result,
    }


@app.get("/api/download/{job_id}/video")
def download_video(job_id: str):
    job = jobs.get_job(job_id)
    if job is None:
        raise HTTPException(404, "Unknown job id")
    if job.status != "done":
        raise HTTPException(409, "Job is not finished yet")
    return FileResponse(job.output_video_path, media_type="video/mp4", filename="highlights.mp4")


@app.get("/api/chapters/{job_id}")
def chapters(job_id: str):
    """The reel's event index: goals, fouls, half time and full time, each with
    its position in the highlight video and in the original match."""
    job = jobs.get_job(job_id)
    if job is None:
        raise HTTPException(404, "Unknown job id")
    if job.status != "done":
        raise HTTPException(409, "Job is not finished yet")
    return {"chapters": (job.result or {}).get("chapters", [])}


@app.get("/api/download/{job_id}/chapters")
def download_chapters(job_id: str):
    job = jobs.get_job(job_id)
    if job is None:
        raise HTTPException(404, "Unknown job id")
    if job.status != "done":
        raise HTTPException(409, "Job is not finished yet")
    return FileResponse(job.chapters_path, media_type="application/json", filename="chapters.json")


@app.get("/api/download/{job_id}/transcript")
def download_transcript(job_id: str):
    job = jobs.get_job(job_id)
    if job is None:
        raise HTTPException(404, "Unknown job id")
    if job.status != "done":
        raise HTTPException(409, "Job is not finished yet")
    return FileResponse(job.transcript_path, media_type="text/plain", filename="transcript.txt")


@app.delete("/api/jobs/{job_id}")
def delete_job(job_id: str):
    job = jobs.get_job(job_id)
    if job is None:
        raise HTTPException(404, "Unknown job id")
    if job.status == "processing":
        raise HTTPException(409, "Cannot delete a job that is still processing")
    shutil.rmtree(JOBS_DIR / job_id, ignore_errors=True)
    jobs.forget_job(job_id)
    return {"deleted": True}
