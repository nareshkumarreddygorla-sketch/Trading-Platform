"""
Lifespan orchestrator: calls each init function in order and handles shutdown.
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from .database import init_database, shutdown_database
from .risk import init_risk
from .execution import init_execution, shutdown_execution
from .ai import init_ai
from .market_data import init_market_data, shutdown_market_data
from .trading import init_trading, shutdown_trading

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_database(app)
    await init_execution(app)
    await init_risk(app)
    await init_market_data(app)
    await init_ai(app)
    await init_trading(app)
    yield
    await shutdown_trading(app)
    await shutdown_market_data(app)
    await shutdown_execution(app)
    await shutdown_database(app)
    logger.info("API shutdown complete")
