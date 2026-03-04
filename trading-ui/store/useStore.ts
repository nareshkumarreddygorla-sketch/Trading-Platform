"use client";

import { create } from "zustand";
import type { User, DashboardSnapshot, BrokerStatus, Position, AgentStatus, SignalEvent, RiskAlert, MarketFeedStatus } from "@/types";
import { WsEventType } from "@/types";

interface AppState {
  user: User | null;
  setUser: (u: User | null) => void;
  equity: number;
  dailyPnl: number;
  setDashboard: (s: Partial<DashboardSnapshot>) => void;
  autonomyOn: boolean;
  setAutonomy: (on: boolean) => void;
  safeMode: boolean;
  setSafeMode: (v: boolean) => void;
  tradingMode: "paper" | "live";
  setTradingMode: (m: "paper" | "live") => void;
  broker: BrokerStatus;
  setBroker: (b: Partial<BrokerStatus>) => void;
  circuitOpen: boolean;
  killSwitchArmed: boolean;
  positions: Position[];
  /** Market data feed status: green = healthy, orange = reconnecting, red = disconnected */
  marketFeed: MarketFeedStatus;
  /** AI Agent statuses */
  agents: Record<string, AgentStatus>;
  /** Live signal feed */
  recentSignals: SignalEvent[];
  /** Risk alerts */
  riskAlerts: RiskAlert[];
  /** Current market regime */
  currentRegime: string;
  /** Research opportunities */
  topOpportunities: Array<{ symbol: string; confidence: number; direction: string }>;
  /** WebSocket connection status */
  wsStatus: "connecting" | "connected" | "disconnected" | "reconnecting";
  setWsStatus: (status: "connecting" | "connected" | "disconnected" | "reconnecting") => void;
  /** Apply a WebSocket event. */
  applyWsEvent: (msg: Record<string, unknown>) => void;
}

export const useStore = create<AppState>((set) => ({
  user: null,
  setUser: (user) => set({ user }),
  equity: 0,
  dailyPnl: 0,
  setDashboard: (s) =>
    set((state) => ({
      equity: s.equity ?? state.equity,
      dailyPnl: s.daily_pnl ?? state.dailyPnl,
    })),
  autonomyOn: false,
  setAutonomy: (autonomyOn) => set({ autonomyOn }),
  safeMode: false,
  setSafeMode: (safeMode) => set({ safeMode }),
  tradingMode: "paper",
  setTradingMode: (tradingMode) => set({ tradingMode }),
  broker: { connected: false, status: "disconnected" },
  setBroker: (b) => set((state) => ({ broker: { ...state.broker, ...b } })),
  circuitOpen: false,
  killSwitchArmed: false,
  positions: [],
  marketFeed: { connected: false, healthy: false, last_tick_ts: null },
  agents: {},
  recentSignals: [],
  riskAlerts: [],
  currentRegime: "unknown",
  topOpportunities: [],
  wsStatus: "disconnected",
  setWsStatus: (wsStatus) => set({ wsStatus }),
  applyWsEvent: (msg) => {
    const type = msg.type as string;
    if (type === WsEventType.POSITION_UPDATED && Array.isArray(msg.positions)) {
      set({
        positions: (msg.positions as Array<{ symbol: string; exchange: string; side: string; quantity: number; avg_price?: number }>).map(
          (p) => ({ symbol: p.symbol, exchange: p.exchange, side: p.side as "BUY" | "SELL", quantity: p.quantity, entry_price: typeof p.avg_price === "number" ? p.avg_price : 0 })
        ),
      });
    }
    if (type === WsEventType.EQUITY_UPDATED) {
      set({
        equity: typeof msg.equity === "number" ? msg.equity : 0,
        dailyPnl: typeof msg.daily_pnl === "number" ? msg.daily_pnl : 0,
      });
    }
    if (type === WsEventType.RISK_UPDATED) {
      set({
        equity: typeof msg.equity === "number" ? msg.equity : 0,
        dailyPnl: typeof msg.daily_pnl === "number" ? msg.daily_pnl : 0,
        circuitOpen: msg.circuit_open === true,
      });
    }
    if (type === WsEventType.CIRCUIT_OPEN) {
      set({ circuitOpen: true });
    }
    if (type === WsEventType.KILL_SWITCH_ARMED) {
      set({ killSwitchArmed: true });
    }
    if (type === WsEventType.MARKET_STATUS_UPDATED) {
      set({
        marketFeed: {
          connected: msg.connected === true,
          healthy: msg.healthy === true,
          last_tick_ts: typeof msg.last_tick_ts === "string" ? msg.last_tick_ts : null,
        },
      });
    }
    // ── Agent events ──
    if (type === WsEventType.AGENT_RESEARCH_UPDATE) {
      const payload = msg.payload as Record<string, unknown> | undefined;
      if (payload) {
        const opps = (payload.top_opportunities as Array<{ symbol: string; confidence: number; direction: string }>) || [];
        set({ topOpportunities: opps.slice(0, 10) });
      }
    }
    if (type === WsEventType.AGENT_RISK_ALERT || type === WsEventType.AGENT_EXPOSURE_ADJUSTED) {
      const payload = msg.payload as Record<string, unknown> | undefined;
      if (payload) {
        set((state) => ({
          riskAlerts: [
            {
              type: (payload.type as string) || type,
              severity: (payload.severity as string) || "info",
              message: JSON.stringify(payload),
              timestamp: (msg.timestamp as string) || new Date().toISOString(),
            },
            ...state.riskAlerts,
          ].slice(0, 20),
        }));
      }
    }
    if (type === WsEventType.AGENT_REGIME_CHANGE) {
      const payload = msg.payload as Record<string, unknown> | undefined;
      if (payload && typeof payload.new_regime === "string") {
        set({ currentRegime: payload.new_regime });
      }
    }
    if (type === WsEventType.SIGNAL_GENERATED) {
      set((state) => ({
        recentSignals: [
          {
            symbol: (msg.symbol as string) || "",
            direction: (msg.side as string) || (msg.direction as string) || "",
            confidence: (msg.score as number) || (msg.confidence as number) || 0,
            source: (msg.strategy_id as string) || (msg.source as string) || "",
            timestamp: (msg.timestamp as string) || new Date().toISOString(),
          },
          ...state.recentSignals,
        ].slice(0, 50),
      }));
    }
    if (type === WsEventType.TRADE_CLOSED) {
      set((state) => ({
        recentSignals: [
          {
            symbol: (msg.symbol as string) || "",
            direction: `${(msg.reason as string) || "CLOSED"} ${(msg.side as string) || ""}`,
            confidence: 1.0,
            source: (msg.strategy_id as string) || "",
            timestamp: (msg.timestamp as string) || new Date().toISOString(),
          },
          ...state.recentSignals,
        ].slice(0, 50),
      }));
    }
    if (type === WsEventType.PORTFOLIO_MARK_TO_MARKET) {
      const pnl = msg.total_pnl;
      if (typeof pnl === "number") {
        set({ dailyPnl: pnl });
      }
    }
    if (type === WsEventType.STRATEGY_DISABLED) {
      // Could show toast notification
    }
  },
}));
