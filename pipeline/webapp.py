import os
import sys
import time
import threading
import uuid

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

from pipeline import run_pipeline
from tools.utils import usage_stats

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

app = FastAPI()


# In-memory job store — this is a single-user local tool, not a multi-tenant service.
JOBS: dict[str, dict] = {}


class GenerateRequest(BaseModel):
    topic: str
    research_mode: str = "web"  # "deep" | "web" | "none"
    use_internet_image_search: bool = True
    fast_mode: bool = False
    language: str = "english"
    enable_veo: bool = False
    veo_direction_by_director: bool = True
    target_duration_minutes: float | None = None
    background_music: bool = False
    voice_one: str | None = None
    voice_two: str | None = None
    tts_speed: float = 1.0


def _run_job(job_id: str, req: GenerateRequest):
    job = JOBS[job_id]

    def on_progress(stage, done, total):
        job["stage"] = stage
        job["done"] = done
        job["total"] = total
        # Mark when planning finished (done hits 1): ETA is based on the pace
        # of scene steps only, since research/planning time says nothing about
        # how fast the remaining scene work will go.
        if done >= 1 and job.get("plan_time") is None:
            job["plan_time"] = time.time()

    try:
        result = run_pipeline(
            req.topic,
            do_research=(req.research_mode == "deep"),
            do_web_search=(req.research_mode == "web"),
            use_internet_image_search=req.use_internet_image_search,
            fast_mode=req.fast_mode,
            language=req.language,
            enable_veo=req.enable_veo,
            veo_direction_by_director=req.veo_direction_by_director,
            target_duration_minutes=req.target_duration_minutes,
            background_music=req.background_music,
            voice_one=req.voice_one,
            voice_two=req.voice_two,
            tts_speed=req.tts_speed,
            on_progress=on_progress,
        )
        if result and os.path.exists(result):
            job["status"] = "done"
            job["result_path"] = result
        else:
            job["status"] = "error"
            job["error"] = "Pipeline finished but produced no video."
    except Exception as e:
        job["status"] = "error"
        job["error"] = str(e)


@app.post("/api/generate")
def generate(req: GenerateRequest):
    job_id = str(uuid.uuid4())
    JOBS[job_id] = {
        "status": "running",
        "stage": "Starting pipeline",
        "done": 0,
        "total": 1,
        "start_time": time.time(),
        "plan_time": None,
        "result_path": None,
        "error": None,
    }
    threading.Thread(target=_run_job, args=(job_id, req), daemon=True).start()
    return {"job_id": job_id}


@app.get("/api/status/{job_id}")
def status(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Unknown job_id")

    elapsed = time.time() - job["start_time"]
    done, total = job["done"], job["total"]
    percent = min(99 if job["status"] == "running" else 100, round(100 * done / total)) if total else 0
    eta_seconds = None
    plan_time = job.get("plan_time")
    if job["status"] == "running" and plan_time and done > 1:
        rate = (time.time() - plan_time) / (done - 1)
        eta_seconds = max(0, round(rate * (total - done)))

    return {
        "status": job["status"],
        "stage": job["stage"],
        "percent": percent,
        "elapsed_seconds": round(elapsed),
        "eta_seconds": eta_seconds,
        "error": job["error"],
        "video_url": f"/api/video/{job_id}" if job["status"] == "done" else None,
        "montage_url": f"/api/montage/{job_id}" if job["status"] == "done" and _montage_path(job) else None,
    }


def _montage_path(job) -> str | None:
    if not job.get("result_path"):
        return None
    path = os.path.join(os.path.dirname(job["result_path"]), "canvas_montage.png")
    return path if os.path.exists(path) else None


@app.get("/api/montage/{job_id}")
def montage(job_id: str):
    job = JOBS.get(job_id)
    path = _montage_path(job) if job else None
    if not path:
        raise HTTPException(status_code=404, detail="Montage not available")
    return FileResponse(path, media_type="image/png", filename="canvas_montage.png")


@app.get("/api/music-available")
def music_available():
    assets = os.path.join(os.path.dirname(__file__), "assets")
    found = any(
        os.path.exists(os.path.join(assets, f"background_music{ext}"))
        for ext in (".mp3", ".wav", ".m4a", ".ogg")
    )
    return {"available": found}


@app.get("/api/usage")
def usage():
    groq = usage_stats["groq"]
    groq_percent = None
    if groq["limit_requests"] and groq["remaining_requests"] is not None:
        limit = int(groq["limit_requests"])
        remaining = int(groq["remaining_requests"])
        groq_percent = round(100 * remaining / limit) if limit else None
    return {
        "groq": {**groq, "percent_remaining": groq_percent},
        "hf_calls": usage_stats["hf_calls"],
    }


@app.get("/api/video/{job_id}")
def video(job_id: str):
    job = JOBS.get(job_id)
    if not job or job["status"] != "done" or not job["result_path"]:
        raise HTTPException(status_code=404, detail="Video not ready")
    return FileResponse(job["result_path"], media_type="video/mp4", filename="storyboard.mp4")


@app.get("/", response_class=HTMLResponse)
def index():
    with open(os.path.join(os.path.dirname(__file__), "static", "index.html"), encoding="utf-8") as f:
        return f.read()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
