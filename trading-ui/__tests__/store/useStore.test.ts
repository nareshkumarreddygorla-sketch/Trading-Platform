/**
 * Tests for the Zustand store at store/useStore.ts
 *
 * Covers:
 *  - Initial state values
 *  - setUser updates user
 *  - setDashboard updates equity and dailyPnl
 *  - applyWsEvent handles position_updated
 *  - applyWsEvent handles equity_updated
 *  - applyWsEvent handles circuit_open
 *  - applyWsEvent handles signal_generated (with max 50 limit)
 */

import { useStore } from "@/store/useStore";
import { WsEventType } from "@/types";

// ---------------------------------------------------------------------------
// Zustand stores are singletons.  We reset the store data before each test
// using a merge (not replace) so the action functions remain intact.
// ---------------------------------------------------------------------------

const dataOnlyReset = {
  user: null,
  equity: 0,
  dailyPnl: 0,
  autonomyOn: false,
  safeMode: false,
  tradingMode: "paper" as const,
  broker: { connected: false, status: "disconnected" as const },
  circuitOpen: false,
  killSwitchArmed: false,
  positions: [] as unknown[],
  marketFeed: { connected: false, healthy: false, last_tick_ts: null },
  agents: {},
  recentSignals: [] as unknown[],
  riskAlerts: [] as unknown[],
  currentRegime: "unknown",
  topOpportunities: [] as unknown[],
};

beforeEach(() => {
  // Merge (default) – keeps the action functions created by create()
  useStore.setState(dataOnlyReset);
});

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe("useStore – initial state", () => {
  it("has correct initial state values", () => {
    const state = useStore.getState();

    expect(state.user).toBeNull();
    expect(state.equity).toBe(0);
    expect(state.dailyPnl).toBe(0);
    expect(state.autonomyOn).toBe(false);
    expect(state.safeMode).toBe(false);
    expect(state.tradingMode).toBe("paper");
    expect(state.broker).toEqual({ connected: false, status: "disconnected" });
    expect(state.circuitOpen).toBe(false);
    expect(state.killSwitchArmed).toBe(false);
    expect(state.positions).toEqual([]);
    expect(state.marketFeed).toEqual({ connected: false, healthy: false, last_tick_ts: null });
    expect(state.recentSignals).toEqual([]);
    expect(state.riskAlerts).toEqual([]);
    expect(state.currentRegime).toBe("unknown");
    expect(state.topOpportunities).toEqual([]);
  });
});

describe("useStore – setUser", () => {
  it("updates the user correctly", () => {
    const testUser = { user_id: "u-123", email: "trader@example.com", roles: ["admin"] };

    useStore.getState().setUser(testUser);

    expect(useStore.getState().user).toEqual(testUser);
  });

  it("can clear the user by setting null", () => {
    useStore.getState().setUser({ user_id: "u-123" });
    expect(useStore.getState().user).not.toBeNull();

    useStore.getState().setUser(null);
    expect(useStore.getState().user).toBeNull();
  });
});

describe("useStore – setDashboard", () => {
  it("updates equity when provided", () => {
    useStore.getState().setDashboard({ equity: 150000 });
    expect(useStore.getState().equity).toBe(150000);
  });

  it("updates daily_pnl when provided", () => {
    useStore.getState().setDashboard({ daily_pnl: 2500 } as Record<string, unknown> & { daily_pnl: number });
    expect(useStore.getState().dailyPnl).toBe(2500);
  });

  it("updates both equity and dailyPnl at once", () => {
    useStore.getState().setDashboard({ equity: 200000, daily_pnl: -500 } as Record<string, unknown> & { equity: number; daily_pnl: number });

    const state = useStore.getState();
    expect(state.equity).toBe(200000);
    expect(state.dailyPnl).toBe(-500);
  });

  it("preserves existing values when partial update is provided", () => {
    useStore.getState().setDashboard({ equity: 100000, daily_pnl: 1000 } as Record<string, unknown> & { equity: number; daily_pnl: number });
    useStore.getState().setDashboard({ equity: 110000 });

    const state = useStore.getState();
    expect(state.equity).toBe(110000);
    // dailyPnl should remain from previous update
    expect(state.dailyPnl).toBe(1000);
  });
});

describe("useStore – applyWsEvent: position_updated", () => {
  it("updates positions from a position_updated event", () => {
    const event = {
      type: WsEventType.POSITION_UPDATED,
      positions: [
        { symbol: "AAPL", exchange: "NASDAQ", side: "BUY", quantity: 100, avg_price: 175.5 },
        { symbol: "GOOGL", exchange: "NASDAQ", side: "SELL", quantity: 50 },
      ],
    };

    useStore.getState().applyWsEvent(event);

    const positions = useStore.getState().positions;
    expect(positions).toHaveLength(2);
    expect(positions[0]).toEqual({
      symbol: "AAPL",
      exchange: "NASDAQ",
      side: "BUY",
      quantity: 100,
      entry_price: 175.5,
    });
    expect(positions[1]).toEqual({
      symbol: "GOOGL",
      exchange: "NASDAQ",
      side: "SELL",
      quantity: 50,
      entry_price: 0, // no avg_price provided -> defaults to 0
    });
  });

  it("ignores event when positions is not an array", () => {
    const event = {
      type: WsEventType.POSITION_UPDATED,
      positions: "not-an-array",
    };

    useStore.getState().applyWsEvent(event);
    expect(useStore.getState().positions).toEqual([]);
  });
});

describe("useStore – applyWsEvent: equity_updated", () => {
  it("updates equity and dailyPnl from an equity_updated event", () => {
    const event = {
      type: WsEventType.EQUITY_UPDATED,
      equity: 250000,
      daily_pnl: 3200,
    };

    useStore.getState().applyWsEvent(event);

    const state = useStore.getState();
    expect(state.equity).toBe(250000);
    expect(state.dailyPnl).toBe(3200);
  });

  it("defaults to 0 when numeric fields are missing", () => {
    const event = {
      type: WsEventType.EQUITY_UPDATED,
    };

    // Set non-zero values first
    useStore.setState({ equity: 100, dailyPnl: 50 });

    useStore.getState().applyWsEvent(event);

    const state = useStore.getState();
    expect(state.equity).toBe(0);
    expect(state.dailyPnl).toBe(0);
  });
});

describe("useStore – applyWsEvent: circuit_open", () => {
  it("sets circuitOpen to true on circuit_open event", () => {
    expect(useStore.getState().circuitOpen).toBe(false);

    useStore.getState().applyWsEvent({ type: WsEventType.CIRCUIT_OPEN });

    expect(useStore.getState().circuitOpen).toBe(true);
  });
});

describe("useStore – applyWsEvent: signal_generated", () => {
  it("adds a signal to recentSignals", () => {
    const event = {
      type: WsEventType.SIGNAL_GENERATED,
      symbol: "TSLA",
      side: "BUY",
      score: 0.87,
      strategy_id: "momentum-v2",
      timestamp: "2024-01-15T10:30:00Z",
    };

    useStore.getState().applyWsEvent(event);

    const signals = useStore.getState().recentSignals;
    expect(signals).toHaveLength(1);
    expect(signals[0]).toEqual({
      symbol: "TSLA",
      direction: "BUY",
      confidence: 0.87,
      source: "momentum-v2",
      timestamp: "2024-01-15T10:30:00Z",
    });
  });

  it("prepends new signals (most recent first)", () => {
    useStore.getState().applyWsEvent({
      type: WsEventType.SIGNAL_GENERATED,
      symbol: "AAPL",
      side: "BUY",
      score: 0.5,
      strategy_id: "s1",
      timestamp: "2024-01-15T10:00:00Z",
    });

    useStore.getState().applyWsEvent({
      type: WsEventType.SIGNAL_GENERATED,
      symbol: "GOOGL",
      side: "SELL",
      score: 0.9,
      strategy_id: "s2",
      timestamp: "2024-01-15T10:01:00Z",
    });

    const signals = useStore.getState().recentSignals;
    expect(signals).toHaveLength(2);
    expect(signals[0].symbol).toBe("GOOGL"); // most recent first
    expect(signals[1].symbol).toBe("AAPL");
  });

  it("enforces a maximum of 50 signals", () => {
    // Pre-populate with 50 signals
    const existingSignals = Array.from({ length: 50 }, (_, i) => ({
      symbol: `SYM-${i}`,
      direction: "BUY",
      confidence: 0.5,
      source: "test",
      timestamp: `2024-01-15T${String(i).padStart(2, "0")}:00:00Z`,
    }));
    useStore.setState({ recentSignals: existingSignals });
    expect(useStore.getState().recentSignals).toHaveLength(50);

    // Add one more – this should cause the oldest to be dropped
    useStore.getState().applyWsEvent({
      type: WsEventType.SIGNAL_GENERATED,
      symbol: "NEW-SIGNAL",
      side: "SELL",
      score: 0.99,
      strategy_id: "overflow-test",
      timestamp: "2024-01-15T23:59:59Z",
    });

    const signals = useStore.getState().recentSignals;
    expect(signals).toHaveLength(50);
    expect(signals[0].symbol).toBe("NEW-SIGNAL"); // newest first
    // The very last of the original 50 should have been dropped
    expect(signals[49].symbol).toBe("SYM-48");
  });

  it("uses fallback values for missing signal fields", () => {
    useStore.getState().applyWsEvent({
      type: WsEventType.SIGNAL_GENERATED,
      // no symbol, side, score, strategy_id, timestamp provided
    });

    const signal = useStore.getState().recentSignals[0];
    expect(signal.symbol).toBe("");
    expect(signal.direction).toBe("");
    expect(signal.confidence).toBe(0);
    expect(signal.source).toBe("");
    // timestamp should default to something truthy (ISO string from Date)
    expect(signal.timestamp).toBeTruthy();
  });
});
