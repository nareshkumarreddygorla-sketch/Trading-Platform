"""FastAPI dependencies: OrderEntryService, etc. Resolve from app.state."""
from typing import Any, Optional

from fastapi import Request


def get_order_entry_service(request: Request) -> Optional[Any]:
    """Return OrderEntryService from app state. None if not configured."""
    return getattr(request.app.state, "order_entry_service", None)


def get_kill_switch(request: Request) -> Optional[Any]:
    """Return KillSwitch from app state. None if not configured."""
    return getattr(request.app.state, "kill_switch", None)
