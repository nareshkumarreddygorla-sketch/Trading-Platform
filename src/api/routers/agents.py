"""Agent management API: status, start/stop agents."""
from fastapi import APIRouter, Request, HTTPException

router = APIRouter(prefix="/api/v1/agents", tags=["agents"])


@router.get("/status")
async def get_all_agent_status(request: Request):
    """Get status of all running agents."""
    orchestrator = getattr(request.app.state, "agent_orchestrator", None)
    if orchestrator is None:
        return {"agents": {}, "message": "Agent orchestrator not initialized"}
    return {"agents": orchestrator.get_all_status()}


@router.get("/{agent_name}/status")
async def get_agent_status(agent_name: str, request: Request):
    """Get status of a specific agent."""
    orchestrator = getattr(request.app.state, "agent_orchestrator", None)
    if orchestrator is None:
        raise HTTPException(status_code=503, detail="Agent orchestrator not initialized")
    agent = orchestrator.get_agent(agent_name)
    if agent is None:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_name}' not found")
    return agent.get_status()


@router.post("/{agent_name}/stop")
async def stop_agent(agent_name: str, request: Request):
    """Stop a specific agent."""
    orchestrator = getattr(request.app.state, "agent_orchestrator", None)
    if orchestrator is None:
        raise HTTPException(status_code=503, detail="Agent orchestrator not initialized")
    agent = orchestrator.get_agent(agent_name)
    if agent is None:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_name}' not found")
    await agent.stop()
    return {"status": "stopped", "agent": agent_name}


@router.post("/{agent_name}/start")
async def start_agent(agent_name: str, request: Request):
    """Start a specific agent."""
    orchestrator = getattr(request.app.state, "agent_orchestrator", None)
    if orchestrator is None:
        raise HTTPException(status_code=503, detail="Agent orchestrator not initialized")
    agent = orchestrator.get_agent(agent_name)
    if agent is None:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_name}' not found")
    agent.start()
    return {"status": "started", "agent": agent_name}
