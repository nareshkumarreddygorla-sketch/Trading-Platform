/**
 * Tests for the WsEventType constants exported from types/index.ts
 *
 * Covers:
 *  - WsEventType has all expected keys
 *  - All WsEventType values are strings
 */

import { WsEventType } from "@/types";

describe("WsEventType – expected keys", () => {
  const expectedKeys = [
    "CONNECTED",
    "ORDER_CREATED",
    "ORDER_FILLED",
    "POSITION_UPDATED",
    "EQUITY_UPDATED",
    "RISK_UPDATED",
    "CIRCUIT_OPEN",
    "KILL_SWITCH_ARMED",
    "MARKET_STATUS_UPDATED",
    "AGENT_RESEARCH_UPDATE",
    "AGENT_RISK_ALERT",
    "AGENT_EXPOSURE_ADJUSTED",
    "AGENT_REGIME_CHANGE",
    "SIGNAL_GENERATED",
    "TRADE_CLOSED",
    "PORTFOLIO_MARK_TO_MARKET",
    "STRATEGY_DISABLED",
    "PONG",
  ];

  it("contains every expected key", () => {
    for (const key of expectedKeys) {
      expect(WsEventType).toHaveProperty(key);
    }
  });

  it("has exactly the expected number of keys (no extra, no missing)", () => {
    const actualKeys = Object.keys(WsEventType);
    expect(actualKeys.sort()).toEqual(expectedKeys.sort());
  });
});

describe("WsEventType – value types", () => {
  it("every value is a string", () => {
    for (const [key, value] of Object.entries(WsEventType)) {
      expect(typeof value).toBe("string");
      // Also verify the key and value have a consistent naming convention:
      // key is UPPER_SNAKE_CASE, value is lower_snake_case
      expect(key).toMatch(/^[A-Z][A-Z0-9_]*$/);
      expect(value).toMatch(/^[a-z][a-z0-9_]*$/);
    }
  });

  it("values are unique (no duplicates)", () => {
    const values = Object.values(WsEventType);
    const uniqueValues = new Set(values);
    expect(uniqueValues.size).toBe(values.length);
  });
});

describe("WsEventType – specific values", () => {
  it("maps keys to expected snake_case values", () => {
    expect(WsEventType.CONNECTED).toBe("connected");
    expect(WsEventType.ORDER_CREATED).toBe("order_created");
    expect(WsEventType.ORDER_FILLED).toBe("order_filled");
    expect(WsEventType.POSITION_UPDATED).toBe("position_updated");
    expect(WsEventType.EQUITY_UPDATED).toBe("equity_updated");
    expect(WsEventType.RISK_UPDATED).toBe("risk_updated");
    expect(WsEventType.CIRCUIT_OPEN).toBe("circuit_open");
    expect(WsEventType.KILL_SWITCH_ARMED).toBe("kill_switch_armed");
    expect(WsEventType.MARKET_STATUS_UPDATED).toBe("market_status_updated");
    expect(WsEventType.AGENT_RESEARCH_UPDATE).toBe("agent_research_update");
    expect(WsEventType.AGENT_RISK_ALERT).toBe("agent_risk_alert");
    expect(WsEventType.AGENT_EXPOSURE_ADJUSTED).toBe("agent_exposure_adjusted");
    expect(WsEventType.AGENT_REGIME_CHANGE).toBe("agent_regime_change");
    expect(WsEventType.SIGNAL_GENERATED).toBe("signal_generated");
    expect(WsEventType.TRADE_CLOSED).toBe("trade_closed");
    expect(WsEventType.PORTFOLIO_MARK_TO_MARKET).toBe("portfolio_mark_to_market");
    expect(WsEventType.STRATEGY_DISABLED).toBe("strategy_disabled");
    expect(WsEventType.PONG).toBe("pong");
  });
});
