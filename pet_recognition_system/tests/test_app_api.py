import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from app.server import create_app  # noqa: E402


def fake_result():
    window = {"index": 0, "start_sec": 0.0, "end_sec": 4.0, "detection": {"person": 0.0, "dog": 0.9},
              "dog_behavior": {"status": "ok", "raw_probabilities": {"sitting": 0.8, "standing": 0.2},
                               "label": "sitting", "confidence": 0.8,
                               "probabilities": {"sitting": 0.8, "standing": 0.2}},
              "interaction": {"status": "insufficient_detection", "label": "insufficient_detection",
                              "confidence": 0.0}}
    return {"input": "C:/server/private/video.mp4", "task": "both",
            "metadata": {"duration_sec": 4.0, "width": 320, "height": 240},
            "windows": [window],
            "timeline": {"dog_behavior": [{"start_sec": 0.0, "end_sec": 4.0, "label": "sitting",
                                           "confidence": 0.8, "windows": 1}],
                         "interaction": [{"start_sec": 0.0, "end_sec": 4.0, "label": "insufficient_detection",
                                          "confidence": 0.0, "windows": 1}]}}


def fake_pipeline(video, output_dir, progress):
    assert Path(video).exists()
    for stage, done, total in (("detect", 0, 1), ("keypoints", 1, 1), ("behaviour", 0, 1), ("interaction", 0, 1)):
        progress(stage, done, total)
    (Path(output_dir) / "result.json").write_text(json.dumps(fake_result()), encoding="utf-8")


def failing_pipeline(video, output_dir, progress):
    progress("detect", 0, 1)
    raise FileNotFoundError("missing pose pipeline weights: rtmdet.pth")


@pytest.fixture(scope="module")
def video_bytes(tmp_path_factory):
    path = tmp_path_factory.mktemp("video") / "clip.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (320, 240))
    for index in range(30):
        frame = np.full((240, 320, 3), index * 8, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return path.read_bytes()


def upload(client, data, *, name="clip.mp4", kind="video/mp4", recorded_at="2026-10-25T09:08", camera="Kitchen"):
    return client.post("/api/clips", files={"video": (name, data, kind)},
                       data={"recorded_at": recorded_at, "camera": camera})


def wait_done(client, job_id):
    for _ in range(100):
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in ("done", "failed"):
            return job
        time.sleep(0.05)
    raise AssertionError("job did not finish")


def test_upload_analyse_list_and_read(tmp_path, video_bytes):
    client = TestClient(create_app(tmp_path, analyse=fake_pipeline))
    response = upload(client, video_bytes, name="../../living room.mp4")
    assert response.status_code == 202
    clip_id = response.json()["clip_id"]
    assert response.json()["job_id"] == clip_id

    job = wait_done(client, clip_id)
    assert (job["status"], job["step"], job["progress"]) == ("done", 4, 1.0)
    assert job["clip"]["file_name"] == "living room.mp4"  # directory parts of the client name are dropped
    assert not list(tmp_path.rglob("living room.mp4"))

    listed = client.get("/api/clips", params={"from": "2026-10-25", "to": "2026-10-25"}).json()
    assert [clip["id"] for clip in listed] == [clip_id]
    assert listed[0]["recorded_at"] == "2026-10-25T09:08:00"
    assert "windows" not in listed[0]["result"] and "input" not in listed[0]["result"]
    assert client.get("/api/clips", params={"from": "2026-10-26"}).json() == []

    clip = client.get(f"/api/clips/{clip_id}").json()
    assert clip["result"]["windows"][0]["dog_behavior"]["label"] == "sitting"
    assert clip["camera"] == "Kitchen" and clip["cover_url"] == f"/api/clips/{clip_id}/cover"

    cover = client.get(clip["cover_url"])
    assert cover.status_code == 200 and cover.headers["content-type"] == "image/jpeg"
    assert cover.content[:2] == b"\xff\xd8"


def test_model_metrics_match_released_reports(tmp_path):
    metrics = TestClient(create_app(tmp_path)).get("/api/models/metrics").json()
    assert metrics["dog_behavior"]["test_clips"] == 33
    assert round(metrics["dog_behavior"]["accuracy"], 3) == 0.788
    assert metrics["interaction"]["test_clips"] == 24
    assert round(metrics["interaction"]["accuracy"], 3) == 0.833


def test_rejects_bad_uploads(tmp_path, video_bytes):
    client = TestClient(create_app(tmp_path, analyse=fake_pipeline, max_upload_bytes=1024))
    assert upload(client, b"not a video", name="notes.txt", kind="text/plain").status_code == 400
    assert upload(client, b"x" * 100, name="fake.mp4").status_code == 400  # claims to be mp4 but cannot be read
    assert upload(client, video_bytes, recorded_at="25/10/2026 09:08").status_code == 422
    assert upload(client, video_bytes, recorded_at="2026-02-30T09:08").status_code == 422
    assert upload(client, video_bytes, camera="   ").status_code == 422
    assert upload(client, video_bytes).status_code == 413
    assert list((tmp_path / "clips").iterdir()) == []  # nothing left behind by rejected uploads


def test_unknown_ids_and_bad_queries(tmp_path):
    client = TestClient(create_app(tmp_path))
    assert client.get("/api/clips/0123456789ab").status_code == 404
    assert client.get("/api/clips/..%2F..%2Fetc").status_code == 404
    assert client.get("/api/jobs/missing").status_code == 404
    assert client.get("/api/clips/0123456789ab/cover").status_code == 404
    assert client.get("/api/clips", params={"from": "yesterday"}).status_code == 422


def test_failed_analysis_is_reported(tmp_path, video_bytes):
    client = TestClient(create_app(tmp_path, analyse=failing_pipeline))
    clip_id = upload(client, video_bytes).json()["clip_id"]
    job = wait_done(client, clip_id)
    assert job["status"] == "failed" and "rtmdet" in job["error"]
    assert client.get("/api/clips").json() == []  # failed clips stay out of the story
    assert client.get(f"/api/clips/{clip_id}").json()["status"] == "failed"


def test_restart_marks_unfinished_clips_failed(tmp_path, video_bytes):
    blocked = TestClient(create_app(tmp_path, analyse=lambda *args: time.sleep(5)))
    clip_id = upload(blocked, video_bytes).json()["clip_id"]
    restarted = TestClient(create_app(tmp_path, analyse=fake_pipeline))
    job = restarted.get(f"/api/jobs/{clip_id}").json()
    assert job["status"] == "failed" and "restarted" in job["error"]
