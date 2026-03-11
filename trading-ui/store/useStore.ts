"use client";

import { create } from "zustand";
import type { User, DashboardSnapshot, BrokerStatus, Position, AgentStatus, SignalEvent, RiskAlert, MarketFeedStatus, SnapshotData } from "@/types";
import { WsEventType } from "@/types";

interface AppState {
  user: User | null;
  setUser: (u: User | null) => void;
  equity: number;
  dailyPnl: number;
  openPositionsCount: number;
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

// Helper to safely extract `data` from new structured events ({type, data, timestamp})
// while remaining backward-compatible with flat legacy events.
function eventData(msg: Record<string, unknown>): Record<string, unknown> {
  if (msg.data && typeof msg.data === "object" && !Array.isArray(msg.data)) {
    return msg.data as Record<string, unknown>;
  }
  return msg;
}

export const useStore = create<AppState>((set) => ({
  user: null,
  setUser: (user) => set({ user }),
  equity: 0,
  dailyPnl: 0,
  openPositionsCount: 0,
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
    const d = eventData(msg);

    // ── New structured event: snapshot (every 5s from backend) ──
    if (type === WsEventType.SNAPSHOT) {
      const snap = d as unknown as SnapshotData;
      set({
        equity: typeof snap.equity === "number" ? snap.equity : 0,
        dailyPnl: typeof snap.daily_pnl === "number" ? snap.daily_pnl : 0,
        openPositionsCount: typeof snap.open_positions_count === "number" ? snap.open_positions_count : 0,
        circuitOpen: snap.circuit_open === true,
        killSwitchArmed: snap.kill_switch_armed === true,
        safeMode: snap.safe_mode === true,
        marketFeed: {
          connected: snap.market_feed_healthy === true,
          healthy: snap.market_feed_healthy === true,
          last_tick_ts: snap.last_tick_ts != null ? String(snap.last_tick_ts) : null,
        },
      });
      return;
    }

    // ── New structured event: position_update ──
    if (type === WsEventType.POSITION_UPDATE) {
      const positions = d.positions;
      if (Array.isArray(positions)) {
        set({
          positions: (positions as Array<{ symbol: string; exchange: string; side: string; quantity: number; avg_price?: number; current_price?: number; unrealized_pnl?: number }>).map(
            (p) => ({
              symbol: p.symbol,
              exchange: p.exchange,
              side: p.side as "BUY" | "SELL",
              quantity: p.quantity,
              entry_price: typeof p.avg_price === "number" ? p.avg_price : 0,
              current_price: typeof p.current_price === "number" ? p.current_price : undefined,
              unrealized_pnl: typeof p.unrealized_pnl === "number" ? p.unrealized_pnl : undefined,
            })
          ),
        });
      }
      return;
    }

    // ── New structured event: pnl_update ──
    if (type === WsEventType.PNL_UPDATE) {
      set((state) => ({
        equity: typeof d.equity === "number" ? d.equity : state.equity,
        dailyPnl: typeof d.daily_pnl === "number" ? d.daily_pnl : state.dailyPnl,
      }));
      return;
    }

    // ── New structured event: order_submitted ──
    if (type === WsEventType.ORDER_SUBMITTED) {
      // Signals feed shows order submissions alongside strategy signals
      set((state) => ({
        recentSignals: [
          {
            symbol: (d.symbol as string) || "",
            direction: (d.side as string) || "BUY",
            confidence: 1.0,
            source: `order:${(d.strategy_id as string) || "manual"}`,
            timestamp: (msg.timestamp as string) || new Date().toISOString(),
          },
          ...state.recentSignals,
        ].slice(0, 50),
      }));
      return;
    }

    // ── New structured event: risk_alert ──
    if (type === WsEventType.RISK_ALERT) {
      set((state) => ({
        riskAlerts: [
          {
            type: (d.alert_type as string) || "risk_alert",
            severity: (d.severity as string) || "warning",
            message: (d.message as string) || JSON.stringify(d),
            timestamp: (msg.timestamp as string) || new Date().toISOString(),
          },
          ...state.riskAlerts,
        ].slice(0, 20),
      }));
      return;
    }

    // ── New structured event: circuit_state ──
    if (type === WsEventType.CIRCUIT_STATE) {
      set({
        circuitOpen: d.circuit_open === true,
      });
      return;
    }

    // ── New structured event: market_feed_status ──
    if (type === WsEventType.MARKET_FEED_STATUS) {
      set({
        marketFeed: {
          connected: d.connected === true || d.healthy === true,
          healthy: d.healthy === true,
          last_tick_ts: typeof d.last_tick_ts === "string" ? d.last_tick_ts : null,
        },
      });
      return;
    }

    // ── Legacy events (backward compatible) ──

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
        killSwitchArmed: msg.kill_switch_armed === true,
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
      // Handle both new structured ({data: {...}}) and legacy flat format
      const sd = d;
      set((state) => ({
        recentSignals: [
          {
            symbol: (sd.symbol as string) || "",
            direction: (sd.side as string) || (sd.direction as string) || "",
            confidence: (sd.score as number) || (sd.confidence as number) || 0,
            source: (sd.strategy_id as string) || (sd.source as string) || "",
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
    if (type === WsEventType.ORDER_FILLED) {
      // order_filled events from legacy broadcasts
      set((state) => ({
        recentSignals: [
          {
            symbol: (msg.symbol as string) || "",
            direction: `FILLED ${(msg.side as string) || ""}`,
            confidence: 1.0,
            source: (msg.strategy_id as string) || "broker",
            timestamp: (msg.timestamp as string) || new Date().toISOString(),
          },
          ...state.recentSignals,
        ].slice(0, 50),
      }));
    }
    if (type === WsEventType.STRATEGY_DISABLED) {
      // Could show toast notification
    }
  },
}));
