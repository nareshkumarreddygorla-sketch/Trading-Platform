"""
Capital deployment gate: GET /capital/validate.
If validate() fails, autonomous live mode must not be enabled; manual paper still allowed.
"""
from fastapi import APIRouter, Request

router = APIRouter()


@router.get("/capital/validate")
async def capital_validate(request: Request):
    """
    Run capital gate checks (Redis, broker, market data, stress/restart).
    If ok is false, do not enable autonomous live mode. Manual paper mode still allowed.
    """
    gate = getattr(request.app.state, "capital_gate", None)
    if gate is None:
        return {"ok": False, "checks": {}, "message": "Capital gate not configured"}
    return await gate.validate()
