"use client";

import { useState, useEffect, useCallback, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { cn } from "@/lib/utils";
import { X, TrendingUp, AlertTriangle, ShieldAlert, Bell } from "lucide-react";

interface Notification {
  id: string;
  type: "success" | "warning" | "error" | "info";
  title: string;
  message: string;
  ts: number;
}

export function Notifications() {
  const [notifications, setNotifications] = useState<Notification[]>([]);
  const timersRef = useRef<Set<ReturnType<typeof setTimeout>>>(new Set());

  const addNotification = useCallback((type: Notification["type"], title: string, message: string) => {
    const id = Math.random().toString(36).slice(2);
    setNotifications((prev) => [{ id, type, title, message, ts: Date.now() }, ...prev].slice(0, 5));
    const timer = setTimeout(() => {
      setNotifications((prev) => prev.filter((n) => n.id !== id));
      timersRef.current.delete(timer);
    }, 6000);
    timersRef.current.add(timer);
  }, []);

  useEffect(() => {
    return () => {
      timersRef.current.forEach(clearTimeout);
    };
  }, []);

  // Listen for WebSocket events dispatched from useWebSocket hook
  useEffect(() => {
    const handler = (ev: Event) => {
      const msg = (ev as CustomEvent).detail;
      if (!msg || !msg.type) return;
      const type = msg.type as string;

      if (type === "order_created") {
        addNotification("info", "Order Created",
          `${msg.side} ${msg.quantity} x ${msg.symbol}`);
      }
      if (type === "order_filled") {
        addNotification("success", "Order Filled",
          `${msg.side} ${msg.filled_qty ?? msg.quantity} x ${msg.symbol} @ ${Number(msg.avg_price ?? 0).toFixed(2)}`);
      }
      if (type === "trade_closed") {
        const reason = msg.reason || "closed";
        const pnl = typeof msg.pnl === "number" ? ` | P&L: ${msg.pnl >= 0 ? "+" : ""}${msg.pnl.toFixed(2)}` : "";
        addNotification(
          typeof msg.pnl === "number" && msg.pnl >= 0 ? "success" : "warning",
          "Trade Closed",
          `${msg.symbol} ${reason}${pnl}`
        );
      }
      if (type === "signal_generated") {
        const conf = msg.confidence || msg.score;
        addNotification("info", "Signal Generated",
          `${msg.side || msg.direction} ${msg.symbol}${conf ? ` (${(conf * 100).toFixed(0)}%)` : ""}`);
      }
      if (type === "circuit_open") {
        addNotification("error", "Circuit Breaker Triggered", "Trading halted — circuit breaker open");
      }
      if (type === "kill_switch_armed") {
        addNotification("warning", "Kill Switch Armed", msg.reason || "Reduce-only mode active");
      }
      if (type === "agent_regime_change") {
        const payload = msg.payload;
        if (payload) {
          addNotification("info", "Regime Change",
            `Market regime: ${payload.new_regime || "unknown"}`);
        }
      }
      if (type === "agent_risk_alert") {
        const payload = msg.payload;
        addNotification("warning", "Risk Alert",
          payload?.message || payload?.type || "Risk event detected");
      }
      if (type === "strategy_disabled") {
        addNotification("warning", "Strategy Disabled",
          `${msg.strategy_id || "Strategy"} was disabled${msg.reason ? `: ${msg.reason}` : ""}`);
      }
    };

    window.addEventListener("ws-event", handler);
    return () => window.removeEventListener("ws-event", handler);
  }, [addNotification]);

  const dismiss = useCallback((id: string) => {
    setNotifications((prev) => prev.filter((n) => n.id !== id));
  }, []);

  const iconMap = {
    success: TrendingUp,
    warning: AlertTriangle,
    error: ShieldAlert,
    info: Bell,
  };

  const colorMap = {
    success: "border-profit/40 bg-profit/5",
    warning: "border-yellow-500/40 bg-yellow-500/5",
    error: "border-loss/40 bg-loss/5",
    info: "border-primary/40 bg-primary/5",
  };

  const iconColorMap = {
    success: "text-profit",
    warning: "text-yellow-500",
    error: "text-loss",
    info: "text-primary",
  };

  return (
    <div className="fixed top-4 right-4 z-[100] space-y-2 pointer-events-none" style={{ maxWidth: 360 }} role="region" aria-label="System notifications" aria-live="polite">
      <AnimatePresence>
        {notifications.map((n) => {
          const Icon = iconMap[n.type];
          return (
            <motion.div
              key={n.id}
              initial={{ opacity: 0, x: 50, scale: 0.95 }}
              animate={{ opacity: 1, x: 0, scale: 1 }}
              exit={{ opacity: 0, x: 50, scale: 0.9 }}
              className={cn(
                "pointer-events-auto rounded-xl border p-3.5 shadow-2xl backdrop-blur-xl",
                colorMap[n.type]
              )}
            >
              <div className="flex items-start gap-3">
                <Icon className={cn("h-4 w-4 mt-0.5 shrink-0", iconColorMap[n.type])} />
                <div className="flex-1 min-w-0">
                  <p className="text-xs font-semibold">{n.title}</p>
                  <p className="text-[11px] text-muted-foreground mt-0.5 truncate">{n.message}</p>
                </div>
                <button
                  onClick={() => dismiss(n.id)}
                  className="shrink-0 rounded-lg p-1 hover:bg-muted/50 transition-colors"
                  aria-label="Dismiss notification"
                >
                  <X className="h-3 w-3 text-muted-foreground" />
                </button>
              </div>
              {/* Auto-dismiss progress bar */}
              <motion.div
                className={cn("h-0.5 rounded-full mt-2", {
                  "bg-profit/40": n.type === "success",
                  "bg-yellow-500/40": n.type === "warning",
                  "bg-loss/40": n.type === "error",
                  "bg-primary/40": n.type === "info",
                })}
                initial={{ width: "100%" }}
                animate={{ width: "0%" }}
                transition={{ duration: 6, ease: "linear" }}
              />
            </motion.div>
          );
        })}
      </AnimatePresence>
    </div>
  );
}
