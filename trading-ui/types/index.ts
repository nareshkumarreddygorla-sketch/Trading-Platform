export interface User {
  user_id: string;
  email?: string;
  roles?: string[];
}

export interface DashboardSnapshot {
  equity: number;
  daily_pnl: number;
  unrealized_pnl: number;
  realized_pnl: number;
  open_positions_count: number;
  risk_exposure_pct?: number;
  circuit_breaker_open: boolean;
  consecutive_losses: number;
  active_strategies_count?: number;
}

export interface Strategy {
  id: string;
  name: string;
  description?: string;
  status: "active" | "inactive" | "disabled" | "drifted";
  capital_allocated: number;
  historical_return_pct?: number;
  max_drawdown_pct?: number;
  sharpe_ratio?: number;
  risk_score?: string;
  win_rate?: number;
  total_pnl?: number;
  total_trades?: number;
}


export interface Position {
  symbol: string;
  exchange: string;
  side: "BUY" | "SELL";
  quantity: number;
  entry_price: number;
  avg_price?: number;
  current_price?: number;
  unrealized_pnl?: number;
  pct_change?: number;
  strategy_id?: string;
}

export interface BrokerStatus {
  connected: boolean;
  latency_ms?: number;
  session_expiry?: string;
  status: "connected" | "disconnected" | "degraded";
}

// ── WebSocket Event Types (shared constants) ──

export const WsEventType = {
  CONNECTED: "connected",
  // Real-time event types (new structured events via broadcast_event)
  SNAPSHOT: "snapshot",
  POSITION_UPDATE: "position_update",
  PNL_UPDATE: "pnl_update",
  ORDER_SUBMITTED: "order_submitted",
  RISK_ALERT: "risk_alert",
  CIRCUIT_STATE: "circuit_state",
  MARKET_FEED_STATUS: "market_feed_status",
  // Legacy event types (still supported)
  ORDER_CREATED: "order_created",
  ORDER_FILLED: "order_filled",
  POSITION_UPDATED: "position_updated",
  EQUITY_UPDATED: "equity_updated",
  RISK_UPDATED: "risk_updated",
  CIRCUIT_OPEN: "circuit_open",
  KILL_SWITCH_ARMED: "kill_switch_armed",
  MARKET_STATUS_UPDATED: "market_status_updated",
  // Agent events
  AGENT_RESEARCH_UPDATE: "agent_research_update",
  AGENT_RISK_ALERT: "agent_risk_alert",
  AGENT_EXPOSURE_ADJUSTED: "agent_exposure_adjusted",
  AGENT_REGIME_CHANGE: "agent_regime_change",
  // Signal & trade lifecycle
  SIGNAL_GENERATED: "signal_generated",
  TRADE_CLOSED: "trade_closed",
  PORTFOLIO_MARK_TO_MARKET: "portfolio_mark_to_market",
  STRATEGY_DISABLED: "strategy_disabled",
  PONG: "pong",
} as const;

// ── Agent & Signal Types ──

export interface AgentStatus {
  name: string;
  description: string;
  running: boolean;
  status: string;
  tick_count: number;
  last_run: string | null;
}

export interface SignalEvent {
  symbol: string;
  direction: string;
  confidence: number;
  source: string;
  timestamp: string;
}

export interface RiskAlert {
  type: string;
  severity: string;
  message: string;
  timestamp: string;
}

export interface MarketFeedStatus {
  connected: boolean;
  healthy: boolean;
  last_tick_ts: string | null;
}

/** Payload from the periodic `snapshot` WebSocket event (every 5s). */
export interface SnapshotData {
  equity: number;
  daily_pnl: number;
  open_positions_count: number;
  circuit_open: boolean;
  kill_switch_armed: boolean;
  market_feed_healthy: boolean;
  last_tick_ts: number | null;
  open_trades: number;
  tick_count: number;
  safe_mode: boolean;
}

