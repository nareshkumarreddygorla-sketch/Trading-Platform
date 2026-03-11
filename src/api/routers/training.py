"""Training management API: trigger training, check status, view results."""
import json
import logging
import os
import subprocess
import sys
import threading
from collections import deque
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request

from src.api.auth import get_current_user, require_roles
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(tags=["training"])

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
META_PATH = os.path.join(MODELS_DIR, "training_meta.json")

# Bounded log buffer drained by a background reader thread
_MAX_LOG_LINES = 500
_log_buffer: deque = deque(maxlen=_MAX_LOG_LINES)

# Track running training process
# NOTE: Concurrent read/write of _training_state is benign under CPython's GIL.
# dict key assignments are atomic under the GIL, so no lock is needed for
# the simple flag/value updates performed here.
_training_state = {
    "running": False,
    "process": None,
    "started_at": None,
    "mode": None,
    "logs": [],
    "result": None,
}


def _drain_pipe(pipe, buf: deque, state: dict):
    """Read lines from *pipe* until EOF, appending to *buf*.

    Runs in a daemon thread so the subprocess pipe never fills up (which
    would deadlock at ~64 KB on most OSes).  When the pipe hits EOF the
    thread marks the process as finished.
    """
    try:
        for line in pipe:
            stripped = line.rstrip("\n").rstrip("\r")
            if stripped:
                buf.append(stripped)
    except Exception:
        pass
    finally:
        try:
            pipe.close()
        except Exception:
            pass


class TrainRequest(BaseModel):
    mode: str = "quick"  # quick, standard, full
    models: Optional[str] = None  # comma-separated: lstm,transformer,rl
    skip_data: bool = False


@router.get("/status")
async def get_training_status(current_user: dict = Depends(get_current_user)):
    """Get current training status and last training metadata."""
    meta = None
    if os.path.exists(META_PATH):
        with open(META_PATH) as f:
            meta = json.load(f)

    # Check if process is still running
    proc = _training_state.get("process")
    if proc and proc.poll() is not None:
        # Process finished — the drain thread has already captured all output
        _training_state["running"] = False
        _training_state["result"] = {
            "returncode": proc.returncode,
            "success": proc.returncode == 0,
            "finished_at": datetime.now(timezone.utc).isoformat() + "Z",
        }
        _training_state["process"] = None

    # Model files status
    model_files = {}
    for name, fname in [
        ("xgboost", "alpha_xgb.joblib"),
        ("lstm", "lstm_predictor.pt"),
        ("transformer", "transformer_predictor.pt"),
        ("rl", "rl_agent.zip"),
    ]:
        path = os.path.join(MODELS_DIR, fname)
        if os.path.exists(path):
            stat = os.stat(path)
            model_files[name] = {
                "exists": True,
                "size_mb": round(stat.st_size / (1024 * 1024), 2),
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat() + "Z",
            }
        else:
            model_files[name] = {"exists": False}

    return {
        "is_training": _training_state["running"],
        "started_at": _training_state.get("started_at"),
        "mode": _training_state.get("mode"),
        "recent_logs": list(_training_state["logs"])[-20:],
        "last_result": _training_state.get("result"),
        "last_training": meta,
        "model_files": model_files,
    }


@router.post("/start")
async def start_training(req: TrainRequest, current_user: dict = Depends(require_roles(["admin"]))):
    """Trigger AI model training as a background process."""
    if _training_state["running"]:
        raise HTTPException(status_code=409, detail="Training already in progress")

    # Build command
    script = os.path.join(PROJECT_ROOT, "scripts", "auto_train_all.py")
    if not os.path.exists(script):
        raise HTTPException(status_code=500, detail="Training script not found")

    cmd = [sys.executable, script]
    if req.mode == "quick":
        cmd.append("--quick")
    elif req.mode == "full":
        cmd.append("--full")

    if req.models:
        cmd.extend(["--models", req.models])

    if req.skip_data:
        cmd.append("--skip-data")

    # Launch as background process
    env = os.environ.copy()
    env["PYTHONPATH"] = PROJECT_ROOT

    try:
        proc = subprocess.Popen(
            cmd, env=env, cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True,
        )
        _log_buffer.clear()
        _log_buffer.append(f"Training started: mode={req.mode}, models={req.models or 'all'}")

        # Start a daemon thread to continuously drain stdout so the pipe
        # buffer never fills up (which would deadlock at ~64 KB).
        reader = threading.Thread(
            target=_drain_pipe,
            args=(proc.stdout, _log_buffer, _training_state),
            daemon=True,
        )
        reader.start()

        _training_state["running"] = True
        _training_state["process"] = proc
        _training_state["started_at"] = datetime.now(timezone.utc).isoformat() + "Z"
        _training_state["mode"] = req.mode
        _training_state["logs"] = _log_buffer  # point to the shared deque
        _training_state["result"] = None

        logger.info("Training started: pid=%d, mode=%s", proc.pid, req.mode)

        return {
            "status": "started",
            "pid": proc.pid,
            "mode": req.mode,
            "models": req.models or "all",
        }
    except Exception as e:
        logger.exception("Failed to start training")
        raise HTTPException(status_code=500, detail="Training start failed")


@router.post("/stop")
async def stop_training(current_user: dict = Depends(require_roles(["admin"]))):
    """Stop a running training process."""
    proc = _training_state.get("process")
    if not proc or not _training_state["running"]:
        raise HTTPException(status_code=404, detail="No training in progress")

    import asyncio
    try:
        proc.terminate()
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, lambda: proc.wait(timeout=10))
    except subprocess.TimeoutExpired:
        proc.kill()

    _training_state["running"] = False
    _training_state["process"] = None
    _training_state["logs"].append("Training stopped by user")
    _training_state["result"] = {"returncode": -1, "success": False, "stopped_by_user": True}

    return {"status": "stopped"}


@router.get("/logs")
async def get_training_logs(lines: int = 50, current_user: dict = Depends(get_current_user)):
    """Get recent training output logs."""
    # The drain thread continuously populates _log_buffer / _training_state["logs"],
    # so we just return the tail — no pipe reads needed here.
    logs = _training_state.get("logs") or _log_buffer
    return {"logs": list(logs)[-lines:]}
