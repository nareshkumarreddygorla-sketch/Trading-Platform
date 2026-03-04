"use client";

import { useEffect, useRef, useCallback } from "react";
import { useStore } from "@/store/useStore";

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
    const wsRef = useRef<WebSocket | null>(null);
    const statusRef = useRef<WsStatus>("disconnected");
    const reconnectTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
    const heartbeatTimer = useRef<ReturnType<typeof setInterval> | null>(null);
    const retryCount = useRef(0);

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

        const url = getWsUrl();
        const token = localStorage.getItem("token");

        try {
            // Send JWT via subprotocol header instead of URL query param
            const protocols = token ? [`access_token.${token}`] : undefined;
            const ws = new WebSocket(url, protocols);
            wsRef.current = ws;
            statusRef.current = "connecting";

            ws.onopen = () => {
                statusRef.current = "connected";
                retryCount.current = 0;
                startHeartbeat(ws);
            };

            ws.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    if (data && typeof data === "object" && data.type) {
                        if (data.type === "pong") return;
                        applyWsEvent(data);
                        // Dispatch custom event for Notifications component
                        window.dispatchEvent(new CustomEvent("ws-event", { detail: data }));
                    }
                } catch {
                    // ignore non-JSON messages
                }
            };

            ws.onclose = () => {
                statusRef.current = "disconnected";
                wsRef.current = null;
                stopHeartbeat();
                scheduleReconnect();
            };

            ws.onerror = () => {
                // onclose will fire after onerror
            };
        } catch {
            scheduleReconnect();
        }
    }, [applyWsEvent, startHeartbeat, stopHeartbeat]);

    const scheduleReconnect = useCallback(() => {
        if (retryCount.current >= MAX_RETRIES) return;
        retryCount.current += 1;
        const delay = Math.min(1000 * Math.pow(2, retryCount.current - 1), 30000);
        statusRef.current = "reconnecting";
        reconnectTimer.current = setTimeout(() => {
            connect();
        }, delay);
    }, [connect]);

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
