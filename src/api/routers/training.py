"""Training management API: trigger training, check status, view results."""
import json
import logging
import os
import subprocess
import sys
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request

from src.api.auth import get_current_user, require_roles
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/training", tags=["training"])

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
META_PATH = os.path.join(MODELS_DIR, "training_meta.json")

# Track running training process
_training_state = {
    "running": False,
    "process": None,
    "started_at": None,
    "mode": None,
    "logs": [],
    "result": None,
}


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
        # Process finished
        _training_state["running"] = False
        import asyncio as _aio
        _loop = _aio.get_event_loop()
        stdout = (await _loop.run_in_executor(None, proc.stdout.read)) if proc.stdout else ""
        stderr = (await _loop.run_in_executor(None, proc.stderr.read)) if proc.stderr else ""
        _training_state["result"] = {
            "returncode": proc.returncode,
            "success": proc.returncode == 0,
            "finished_at": datetime.utcnow().isoformat() + "Z",
        }
        if stdout:
            _training_state["logs"].extend(stdout.strip().split("\n")[-20:])
        if stderr:
            _training_state["logs"].extend(stderr.strip().split("\n")[-10:])
        # Close pipe file descriptors to prevent resource leak
        if proc.stdout:
            proc.stdout.close()
        if proc.stderr:
            proc.stderr.close()
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
        "recent_logs": _training_state["logs"][-20:],
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
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True,
        )
        _training_state["running"] = True
        _training_state["process"] = proc
        _training_state["started_at"] = datetime.utcnow().isoformat() + "Z"
        _training_state["mode"] = req.mode
        _training_state["logs"] = [f"Training started: mode={req.mode}, models={req.models or 'all'}"]
        _training_state["result"] = None

        logger.info("Training started: pid=%d, mode=%s", proc.pid, req.mode)

        return {
            "status": "started",
            "pid": proc.pid,
            "mode": req.mode,
            "models": req.models or "all",
        }
    except Exception as e:
        logger.error("Failed to start training: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


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
    # If process is running, try to read available output without blocking
    proc = _training_state.get("process")
    if proc and _training_state["running"] and proc.stdout:
        import asyncio, select
        loop = asyncio.get_running_loop()
        readable = await loop.run_in_executor(None, lambda: select.select([proc.stdout], [], [], 0)[0])
        if readable:
            new_line = await loop.run_in_executor(None, proc.stdout.readline)
            if new_line:
                _training_state["logs"].append(new_line.strip())

    return {"logs": _training_state["logs"][-lines:]}
