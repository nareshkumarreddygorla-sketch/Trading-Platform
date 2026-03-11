"use client";

import { useEffect, useRef, useCallback } from "react";
import { useStore } from "@/store/useStore";
import { dispatchToast } from "@/components/Toaster";
import { clearAuthTokens } from "@/lib/api/client";

type WsStatus = "connecting" | "connected" | "disconnected" | "reconnecting";

const MAX_RETRIES = 20;
const HEARTBEAT_INTERVAL_MS = 30_000;

/**
 * Determine WebSocket backend URL from environment or current location.
 */
function getWsUrl(): string {
    const envUrl = process.env.NEXT_PUBLIC_WS_URL;
    if (envUrl) return `${envUrl}/ws`;
    const proto = window.location.protocol === "https:" ? "wss" : "ws";
    const port = process.env.NEXT_PUBLIC_API_PORT || "8000";
    return `${proto}://${window.location.hostname}:${port}/ws`;
}

/**
 * WebSocket hook: connects to the backend /ws endpoint,
 * dispatches events to the Zustand store, auto-reconnects,
 * and sends heartbeat pings.
 *
 * Auth: sends JWT via Sec-WebSocket-Protocol header (preferred)
 * to avoid token leaking in URL/logs/referrer headers.
 */
export function useWebSocket() {
    const applyWsEvent = useStore((s) => s.applyWsEvent);
    const setWsStatus = useStore((s) => s.setWsStatus);
    const wsRef = useRef<WebSocket | null>(null);
    const statusRef = useRef<WsStatus>("disconnected");
    const reconnectTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
    const heartbeatTimer = useRef<ReturnType<typeof setInterval> | null>(null);
    const retryCount = useRef(0);

    // Use a ref to break circular dependency between connect ↔ scheduleReconnect
    const scheduleReconnectRef = useRef<() => void>(() => {});

    const updateStatus = useCallback((status: WsStatus) => {
        statusRef.current = status;
        setWsStatus(status);
    }, [setWsStatus]);

    const stopHeartbeat = useCallback(() => {
        if (heartbeatTimer.current) {
            clearInterval(heartbeatTimer.current);
            heartbeatTimer.current = null;
        }
    }, []);

    const startHeartbeat = useCallback((ws: WebSocket) => {
        stopHeartbeat();
        heartbeatTimer.current = setInterval(() => {
            if (ws.readyState === WebSocket.OPEN) {
                ws.send("ping");
            }
        }, HEARTBEAT_INTERVAL_MS);
    }, [stopHeartbeat]);

    const connect = useCallback(() => {
        if (typeof window === "undefined") return;

        const token = localStorage.getItem("token");

        // Use subprotocol header only for token — never send token in URL query params
        // (tokens in URLs leak via logs, referrer headers, and browser history)
        const url = getWsUrl();

        try {
            // Send JWT via subprotocol header only (secure)
            const protocols = token ? [`access_token.${token}`] : undefined;
            const ws = new WebSocket(url, protocols);
            wsRef.current = ws;
            updateStatus("connecting");

            ws.onopen = () => {
                updateStatus("connected");
                retryCount.current = 0;
                startHeartbeat(ws);
            };

            ws.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    if (data && typeof data === "object" && data.type) {
                        // Respond to server heartbeat pings
                        if (data.type === "ping") {
                            if (ws.readyState === WebSocket.OPEN) {
                                ws.send("pong");
                            }
                            return;
                        }
                        if (data.type === "pong") return;
                        applyWsEvent(data);
                        // Dispatch custom event for Notifications component
                        window.dispatchEvent(new CustomEvent("ws-event", { detail: data }));

                        // Dispatch toast notifications for key trading events.
                        // Supports both new structured events ({data: {...}}) and legacy flat events.
                        const ed = data.data && typeof data.data === "object" ? data.data : data;

                        if (data.type === "order_filled") {
                            dispatchToast(
                                "success",
                                "Order Filled",
                                `${ed.side || "BUY"} ${ed.symbol || "?"} x ${ed.filled_qty ?? ed.quantity ?? 0}`
                            );
                        }
                        if (data.type === "order_submitted") {
                            dispatchToast(
                                "info",
                                "Order Submitted",
                                `${ed.side || "BUY"} ${ed.symbol || "?"} x ${ed.quantity ?? 0}`
                            );
                        }
                        if (data.type === "signal_generated") {
                            dispatchToast(
                                "info",
                                "Signal",
                                `${ed.side || ed.direction || "?"} ${ed.symbol || "?"} (${((ed.score ?? ed.confidence ?? 0) * 100).toFixed(0)}%)`
                            );
                        }
                        if (data.type === "risk_alert") {
                            const severity = ed.severity === "critical" ? "error" : "warning";
                            dispatchToast(
                                severity,
                                "Risk Alert",
                                (ed.message as string) || "Risk limit breached"
                            );
                        }
                        if (data.type === "circuit_state") {
                            if (ed.circuit_open === true) {
                                dispatchToast(
                                    "warning",
                                    "Circuit Breaker Open",
                                    "Trading halted — circuit breaker triggered"
                                );
                            } else {
                                dispatchToast(
                                    "success",
                                    "Circuit Breaker Reset",
                                    "Trading resumed — circuit breaker cleared"
                                );
                            }
                        }
                        if (data.type === "circuit_open") {
                            dispatchToast(
                                "warning",
                                "Circuit Breaker Triggered",
                                "Trading halted — daily loss limit reached"
                            );
                        }
                        if (data.type === "kill_switch_armed") {
                            dispatchToast(
                                "error",
                                "Kill Switch Armed",
                                (data.reason as string) || "Trading halted — reduce-only mode active"
                            );
                        }
                        if (data.type === "market_feed_status") {
                            if (ed.healthy === false) {
                                dispatchToast(
                                    "warning",
                                    "Market Feed",
                                    "Market data feed is unhealthy or disconnected"
                                );
                            }
                        }
                        if (data.type === "trade_closed") {
                            const pnl = typeof data.pnl === "number" ? data.pnl : null;
                            const pnlStr = pnl !== null ? `P&L: ${pnl >= 0 ? "+" : ""}${pnl.toFixed(2)}` : "";
                            dispatchToast(
                                "info",
                                "Trade Closed",
                                `${data.symbol || "?"} ${pnlStr}`
                            );
                        }
                    }
                } catch {
                    // ignore non-JSON messages
                }
            };

            ws.onclose = (event) => {
                updateStatus("disconnected");
                wsRef.current = null;
                stopHeartbeat();
                // Don't reconnect if server rejected auth (code 4001)
                if (event.code === 4001) {
                    // Clear stale tokens and auth cookie, then redirect to login
                    clearAuthTokens();
                    if (!window.location.pathname.includes("/login")) {
                        window.location.href = "/login";
                    }
                    return;
                }
                scheduleReconnectRef.current();
            };

            ws.onerror = () => {
                // onclose will fire after onerror
            };
        } catch {
            scheduleReconnectRef.current();
        }
    }, [applyWsEvent, startHeartbeat, stopHeartbeat, updateStatus]);

    const scheduleReconnect = useCallback(() => {
        if (retryCount.current >= MAX_RETRIES) return;
        retryCount.current += 1;
        const delay = Math.min(1000 * Math.pow(2, retryCount.current - 1), 30000);
        updateStatus("reconnecting");
        reconnectTimer.current = setTimeout(() => {
            connect();
        }, delay);
    }, [connect, updateStatus]);

    // Keep the ref in sync so connect always calls the latest scheduleReconnect
    scheduleReconnectRef.current = scheduleReconnect;

    useEffect(() => {
        connect();
        return () => {
            stopHeartbeat();
            if (reconnectTimer.current) clearTimeout(reconnectTimer.current);
            if (wsRef.current) {
                wsRef.current.onclose = null;
                wsRef.current.close();
            }
        };
    }, [connect, stopHeartbeat]);
}
