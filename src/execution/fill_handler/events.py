"""Fill events from broker WebSocket or REST."""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class FillType(str, Enum):
    FILL = "fill"
    PARTIAL_FILL = "partial_fill"
    CANCEL = "cancel"
    REJECT = "reject"


@dataclass
class FillEvent:
    order_id: str
    broker_order_id: str | None
    symbol: str
    exchange: str
    side: str
    fill_type: FillType
    filled_qty: float
    remaining_qty: float
    avg_price: float | None
    ts: datetime
    strategy_id: str = ""
    metadata: dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
