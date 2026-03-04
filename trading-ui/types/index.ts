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

export interface RiskState {
  max_daily_loss_pct: number;
  current_daily_loss_pct: number;
  var_pct?: number;
  sector_concentration_pct?: number;
  per_symbol_exposure_pct?: number;
  circuit_breaker_open: boolean;
  volatility_multiplier?: number;
  consecutive_losses: number;
}

export interface BrokerStatus {
  connected: boolean;
  latency_ms?: number;
  session_expiry?: string;
  status: "connected" | "disconnected" | "degraded";
}

export interface AuditEvent {
  id: string;
  ts: string;
  event_type: string;
  actor: string;
  payload?: Record<string, unknown>;
}

export interface ApiError {
  detail: string;
  status?: number;
}

// ── WebSocket Event Types (shared constants) ──

export const WsEventType = {
  CONNECTED: "connected",
  ORDER_CREATED: "order_created",
  ORDER_FILLED: "order_filled",
  POSITION_UPDATED: "position_updated",
  EQUITY_UPDATED: "equity_updated",
  RISK_UPDATED: "risk_updated",
  CIRCUIT_OPEN: "circuit_open",
  KILL_SWITCH_ARMED: "kill_switch_armed",
  MARKET_STATUS_UPDATED: "market_status_updated",
  AGENT_RESEARCH_UPDATE: "agent_research_update",
  AGENT_RISK_ALERT: "agent_risk_alert",
  AGENT_EXPOSURE_ADJUSTED: "agent_exposure_adjusted",
  AGENT_REGIME_CHANGE: "agent_regime_change",
  SIGNAL_GENERATED: "signal_generated",
  TRADE_CLOSED: "trade_closed",
  PORTFOLIO_MARK_TO_MARKET: "portfolio_mark_to_market",
  STRATEGY_DISABLED: "strategy_disabled",
  PONG: "pong",
} as const;

export type WsEventTypeValue = (typeof WsEventType)[keyof typeof WsEventType];

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

// ── Order Enums (matching backend) ──

export type OrderSide = "BUY" | "SELL";
export type OrderTypeValue = "LIMIT" | "MARKET" | "IOC" | "FOK";
export type ExchangeValue = "NSE" | "BSE" | "NYSE" | "NASDAQ" | "LSE" | "FX";
