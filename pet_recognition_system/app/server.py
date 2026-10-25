"""FastAPI backend for the mobile app: upload clips, follow their analysis, read results.

Run from pet_recognition_system/ with PET_DATA_ROOT pointing at the data folder:
    uvicorn app.server:app --port 8000
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import json
import os
from pathlib import Path
import re
import shutil
import sys
import uuid

import cv2
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, JSONResponse

PROJECT = Path(__file__).resolve().parents[1]
STAGES = ("detect", "keypoints", "behaviour", "interaction")
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}
MAX_UPLOAD_BYTES = 500 * 1024 * 1024
CHUNK = 1024 * 1024
CLIP_ID = re.compile(r"^[0-9a-f]{12}$")
METRICS = {
    "dog_behavior": PROJECT / "outputs" / "best_models" / "dog_behavior" / "test_metrics.json",
    "interaction": PROJECT / "outputs" / "best_models" / "interaction" / "ensemble_test_metrics.json",
}


def run_pipeline(video: Path, output_dir: Path, progress) -> None:
    """Analyse one stored video with the released models; writes result.json into output_dir."""
    if str(PROJECT / "tools") not in sys.path:
        sys.path.insert(0, str(PROJECT / "tools"))
    from predict_video import predict_video  # heavy imports, so only load them when a job runs

    predict_video(video, output_dir, task="both", data_root=os.environ.get("PET_DATA_ROOT"),
                  device=os.environ.get("PET_DEVICE", "cpu"), progress=progress)


def _parse_recorded_at(value: str) -> str:
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(:\d{2})?", value or ""):
        raise HTTPException(422, "recorded_at must look like YYYY-MM-DDTHH:MM")
    try:
        return datetime.fromisoformat(value).strftime("%Y-%m-%dT%H:%M:%S")
    except ValueError:
        raise HTTPException(422, "recorded_at is not a real date and time") from None


def _parse_day(value: str | None, name: str) -> str | None:
    if value is None:
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%d").strftime("%Y-%m-%d")
    except ValueError:
        raise HTTPException(422, f"{name} must look like YYYY-MM-DD") from None


def _probe_and_cover(video: Path, cover: Path) -> float:
    """Return the duration in seconds and save a frame from about one second in as the cover."""
    capture = cv2.VideoCapture(str(video))
    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        frames = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        if not capture.isOpened() or fps <= 0 or frames <= 0:
            raise ValueError("unreadable video")
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(min(fps, frames / 2)))
        ok, frame = capture.read()
        if ok:
            height, width = frame.shape[:2]
            if width > 720:
                frame = cv2.resize(frame, (720, round(height * 720 / width)), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(cover), frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return frames / fps
    finally:
        capture.release()


def create_app(store: Path | None = None, *, analyse=None, max_upload_bytes: int = MAX_UPLOAD_BYTES) -> FastAPI:
    store = Path(store or os.environ.get("PET_APP_STORE") or PROJECT / "outputs" / "app_store") / "clips"
    store.mkdir(parents=True, exist_ok=True)
    analyse = analyse or run_pipeline
    executor = ThreadPoolExecutor(max_workers=1)  # CPU inference is heavy: one clip at a time
    jobs: dict[str, dict] = {}
    app = FastAPI(title="Pet behaviour API")

    def folder(clip_id: str) -> Path:
        if not CLIP_ID.fullmatch(clip_id) or not (store / clip_id / "clip.json").exists():
            raise HTTPException(404, "Clip not found")
        return store / clip_id

    def read_meta(path: Path) -> dict:
        return json.loads((path / "clip.json").read_text(encoding="utf-8"))

    def write_meta(path: Path, meta: dict) -> None:
        temp = path / "clip.json.tmp"
        temp.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        temp.replace(path / "clip.json")

    def public(path: Path, meta: dict, *, windows: bool) -> dict:
        record = {key: meta.get(key) for key in ("id", "file_name", "recorded_at", "camera", "status", "error")}
        record["cover_url"] = f"/api/clips/{meta['id']}/cover" if (path / "cover.jpg").exists() else None
        result_path = path / "result.json"
        if result_path.exists():
            result = json.loads(result_path.read_text(encoding="utf-8"))
            result.pop("input", None)  # absolute server path, not for clients
            if not windows:
                result.pop("windows", None)
            record["result"] = result
        return record

    # a clip still "processing" at startup lost its worker when the server stopped
    for meta_path in store.glob("*/clip.json"):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("status") == "processing":
            meta.update(status="failed", error="The server restarted before the analysis finished")
            write_meta(meta_path.parent, meta)

    def work(clip_id: str) -> None:
        path, job = store / clip_id, jobs[clip_id]
        job["status"] = "running"

        def progress(stage: str, done: int, total: int) -> None:
            index = STAGES.index(stage)
            job["step"] = index
            job["progress"] = (index + min(done / max(total, 1), 1)) / len(STAGES)

        meta = read_meta(path)
        try:
            analyse(path / meta["video"], path, progress)
            if not (path / "result.json").exists():
                raise RuntimeError("the analysis produced no result")
            meta.update(status="done", error=None)
            job.update(status="done", step=len(STAGES), progress=1.0)
        except Exception as error:  # report any pipeline failure on the job instead of losing it
            meta.update(status="failed", error=str(error) or type(error).__name__)
            job.update(status="failed", error=meta["error"])
        write_meta(path, meta)

    @app.post("/api/clips", status_code=202)
    async def create_clip(video: UploadFile = File(...), recorded_at: str = Form(...), camera: str = Form(...)):
        recorded = _parse_recorded_at(recorded_at)
        camera = camera.strip()
        if not 1 <= len(camera) <= 40:
            raise HTTPException(422, "camera must be 1-40 characters")
        extension = Path(video.filename or "").suffix.lower()
        if not (video.content_type or "").startswith("video/") or extension not in VIDEO_EXTENSIONS:
            raise HTTPException(400, "Upload a video file (mp4, mov, avi, mkv or m4v)")

        clip_id = uuid.uuid4().hex[:12]
        path = store / clip_id
        path.mkdir()
        target = path / f"video{extension}"  # never reuse the client's file name on disk
        size = 0
        try:
            with target.open("wb") as handle:
                while chunk := await video.read(CHUNK):
                    size += len(chunk)
                    if size > max_upload_bytes:
                        raise HTTPException(413, f"Video is larger than {max_upload_bytes // (1024 * 1024)} MB")
                    handle.write(chunk)
            try:
                duration = _probe_and_cover(target, path / "cover.jpg")
            except ValueError:
                raise HTTPException(400, "Could not read this video") from None
        except BaseException:
            shutil.rmtree(path, ignore_errors=True)
            raise

        meta = {"id": clip_id, "file_name": Path(video.filename).name[:120], "video": target.name,
                "recorded_at": recorded, "camera": camera, "duration_sec": duration,
                "status": "processing", "error": None}
        write_meta(path, meta)
        jobs[clip_id] = {"status": "queued", "step": 0, "progress": 0.0, "error": None}
        executor.submit(work, clip_id)
        return {"job_id": clip_id, "clip_id": clip_id}

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str):
        path = folder(job_id)
        meta = read_meta(path)
        # jobs only live in memory; after a restart fall back to the stored clip status
        job = jobs.get(job_id) or {"status": meta["status"] if meta["status"] != "processing" else "failed",
                                   "step": len(STAGES) if meta["status"] == "done" else 0,
                                   "progress": 1.0 if meta["status"] == "done" else 0.0, "error": meta.get("error")}
        return {"job_id": job_id, **job, "clip": public(path, meta, windows=False) | {"result": None}}

    @app.get("/api/clips")
    def list_clips(from_: str | None = Query(None, alias="from"), to: str | None = None):
        start, end = _parse_day(from_, "from"), _parse_day(to, "to")
        records = []
        for meta_path in store.glob("*/clip.json"):
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            day = meta["recorded_at"][:10]
            if meta.get("status") == "done" and (not start or day >= start) and (not end or day <= end):
                records.append(public(meta_path.parent, meta, windows=False))
        return sorted(records, key=lambda record: record["recorded_at"])

    @app.get("/api/clips/{clip_id}")
    def get_clip(clip_id: str):
        path = folder(clip_id)
        return public(path, read_meta(path), windows=True)

    @app.get("/api/clips/{clip_id}/cover")
    def get_cover(clip_id: str):
        cover = folder(clip_id) / "cover.jpg"
        if not cover.exists():
            raise HTTPException(404, "No cover for this clip")
        return FileResponse(cover, media_type="image/jpeg")

    @app.get("/api/models/metrics")
    def model_metrics():
        out = {}
        for task, path in METRICS.items():
            if not path.exists():
                return JSONResponse({"detail": f"missing metrics file: {path.name}"}, status_code=503)
            metrics = json.loads(path.read_text(encoding="utf-8"))
            out[task] = {"accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"],
                         "test_clips": sum(value["support"] for value in metrics["per_class"].values())}
        return out

    return app


app = create_app()
