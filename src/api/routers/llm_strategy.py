"""API router for LLM-powered strategy generation."""

import logging

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/strategy-builder")


class GenerateRequest(BaseModel):
    prompt: str
    timeframe: str | None = None


class GenerateResponse(BaseModel):
    name: str
    description: str
    code: str
    config: dict
    validated: bool
    errors: list


def _get_generator(request: Request):
    gen = getattr(request.app.state, "strategy_generator", None)
    if gen is None:
        from src.ai.llm.strategy_generator import StrategyGenerator

        gen = StrategyGenerator()
        request.app.state.strategy_generator = gen
    return gen


@router.post("/generate", response_model=GenerateResponse)
async def generate_strategy(body: GenerateRequest, request: Request):
    """Generate a trading strategy from natural language description."""
    gen = _get_generator(request)
    prompt = body.prompt
    if body.timeframe:
        prompt += f" (timeframe: {body.timeframe})"
    result = await gen.generate(prompt)
    return GenerateResponse(
        name=result.name,
        description=result.description,
        code=result.code,
        config=result.config,
        validated=result.validated,
        errors=result.errors,
    )


@router.get("/strategies")
async def list_strategies(request: Request):
    """List all generated strategies."""
    gen = _get_generator(request)
    return [
        {
            "name": s.name,
            "description": s.description,
            "validated": s.validated,
            "created_at": s.created_at,
            "prompt": s.prompt,
        }
        for s in gen.list_generated()
    ]


@router.get("/strategies/{name}")
async def get_strategy(name: str, request: Request):
    """Get a specific generated strategy with its code."""
    gen = _get_generator(request)
    s = gen.get_strategy(name)
    if not s:
        raise HTTPException(status_code=404, detail="Strategy not found")
    return {
        "name": s.name,
        "description": s.description,
        "code": s.code,
        "config": s.config,
        "validated": s.validated,
        "errors": s.errors,
        "created_at": s.created_at,
        "prompt": s.prompt,
    }


@router.delete("/strategies/{name}")
async def delete_strategy(name: str, request: Request):
    """Delete a generated strategy."""
    gen = _get_generator(request)
    if not gen.delete_strategy(name):
        raise HTTPException(status_code=404, detail="Strategy not found")
    return {"status": "deleted", "name": name}


@router.get("/templates")
async def get_templates():
    """Return example prompts users can start from."""
    return [
        {
            "name": "RSI Mean Reversion",
            "prompt": "Buy when RSI drops below 30 and close when RSI goes above 70. Use 2% stop loss.",
            "category": "mean_reversion",
        },
        {
            "name": "MACD Crossover",
            "prompt": "Enter long when MACD crosses above signal line, exit when it crosses below. 15m timeframe.",
            "category": "trend_following",
        },
        {
            "name": "Bollinger Bounce",
            "prompt": "Buy when price touches lower Bollinger Band, sell at upper band. 5m timeframe with 1.5% stop loss.",
            "category": "mean_reversion",
        },
        {
            "name": "VWAP Breakout",
            "prompt": "Go long when price breaks above VWAP with increasing volume. Take profit at 3%. 1m timeframe.",
            "category": "momentum",
        },
        {
            "name": "Supertrend Trend Follow",
            "prompt": "Follow Supertrend indicator. Long above, short below. 15m timeframe with ATR-based stop.",
            "category": "trend_following",
        },
        {
            "name": "Multi-Indicator Confluence",
            "prompt": "Buy when RSI < 40 AND price is above EMA 20 AND MACD is positive. Sell when RSI > 70. 5m timeframe.",
            "category": "confluence",
        },
    ]
