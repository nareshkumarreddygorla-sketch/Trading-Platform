"""Agent management API: status, start/stop agents, autonomous loop state."""
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, Request, HTTPException
from pydantic import BaseModel

from src.api.auth import get_current_user, require_roles


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------

class AutonomousLoopStatus(BaseModel):
    running: bool
    message: Optional[str] = None
    tick_count: Optional[int] = None
    open_trades: Optional[int] = None
    daily_pnl: Optional[float] = None
    last_tick_ts: Optional[str] = None
    paper_mode: Optional[bool] = None
    poll_interval_seconds: Optional[float] = None
    signals_generated: Optional[int] = None
    orders_submitted: Optional[int] = None


class AgentStatusResponse(BaseModel):
    agents: Dict[str, Any]
    autonomous_loop: AutonomousLoopStatus


class AgentActionResponse(BaseModel):
    status: str
    agent: str


router = APIRouter(tags=["agents"])


def _autonomous_loop_status(request: Request) -> dict:
    """Build status dict for the core autonomous execution loop."""
    al = getattr(request.app.state, "autonomous_loop", None)
    if al is None:
        return {"running": False, "message": "Autonomous loop not initialized"}
    return {
        "running": getattr(al, "_running", False),
        "tick_count": getattr(al, "_tick_count", 0),
        "open_trades": len(getattr(al, "_open_trades", {})),
        "daily_pnl": getattr(al, "_daily_pnl", 0.0),
        "last_tick_ts": getattr(al, "_last_tick_ts", None),
        "paper_mode": getattr(al, "_paper_mode", True),
        "poll_interval_seconds": getattr(al, "_poll_interval", 60.0),
        "signals_generated": getattr(al, "_signals_generated", 0),
        "orders_submitted": getattr(al, "_orders_submitted", 0),
    }


@router.get("/status", response_model=AgentStatusResponse)
async def get_all_agent_status(request: Request, current_user: dict = Depends(get_current_user)):
    """Get status of all running agents including autonomous loop."""
    orchestrator = getattr(request.app.state, "agent_orchestrator", None)
    agents = orchestrator.get_all_status() if orchestrator is not None else {}
    return {
        "agents": agents,
        "autonomous_loop": _autonomous_loop_status(request),
    }


@router.get("/{agent_name}/status")
async def get_agent_status(agent_name: str, request: Request, current_user: dict = Depends(get_current_user)):
    """Get status of a specific agent."""
    orchestrator = getattr(request.app.state, "agent_orchestrator", None)
    if orchestrator is None:
        raise HTTPException(status_code=503, detail="Agent orchestrator not initialized")
    agent = orchestrator.get_agent(agent_name)
    if agent is None:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_name}' not found")
    return agent.get_status()


@router.post("/{agent_name}/stop")
async def stop_agent(agent_name: str, request: Request, current_user: dict = Depends(require_roles(["admin"]))):
    """Stop a specific agent."""
    orchestrator = getattr(request.app.state, "agent_orchestrator", None)
    if orchestrator is None:
        raise HTTPException(status_code=503, detail="Agent orchestrator not initialized")
    agent = orchestrator.get_agent(agent_name)
    if agent is None:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_name}' not found")
    await agent.stop()
    return AgentActionResponse(status="stopped", agent=agent_name)


@router.post("/{agent_name}/start")
async def start_agent(agent_name: str, request: Request, current_user: dict = Depends(require_roles(["admin"]))):
    """Start a specific agent."""
    orchestrator = getattr(request.app.state, "agent_orchestrator", None)
    if orchestrator is None:
        raise HTTPException(status_code=503, detail="Agent orchestrator not initialized")
    agent = orchestrator.get_agent(agent_name)
    if agent is None:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_name}' not found")
    agent.start()
    return AgentActionResponse(status="started", agent=agent_name)
