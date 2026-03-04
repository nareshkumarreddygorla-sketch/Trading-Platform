"use client";

import { useEffect, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { useStore } from "@/store/useStore";
import { Switch } from "@/components/ui/switch";
import { cn, formatCurrency, formatPercent } from "@/lib/utils";
import { endpoints } from "@/lib/api/client";
import {
  ArrowUpRight,
  ArrowDownRight,
  Radio,
  Wifi,
  WifiOff,
  Shield,
  Bell,
  User,
  FileQuestion,
} from "lucide-react";
import { ConnectionStatus } from "@/components/ConnectionStatus";

export function Topbar() {
  const autonomyOn = useStore((s) => s.autonomyOn);
  const setAutonomy = useStore((s) => s.setAutonomy);
  const safeMode = useStore((s) => s.safeMode);
  const setSafeMode = useStore((s) => s.setSafeMode);
  const tradingMode = useStore((s) => s.tradingMode);
  const setTradingMode = useStore((s) => s.setTradingMode);
  const equity = useStore((s) => s.equity);
  const dailyPnl = useStore((s) => s.dailyPnl);
  const broker = useStore((s) => s.broker);
  const marketFeed = useStore((s) => s.marketFeed);

  const isProfit = dailyPnl >= 0;
  const isPaper = tradingMode === "paper";

  // Poll /trading/mode on mount + every 10 seconds (stable interval)
  const mountedRef = useRef(false);
  useEffect(() => {
    if (mountedRef.current) return;
    mountedRef.current = true;

    const fetchMode = async () => {
      try {
        const res = await endpoints.tradingMode();
        useStore.getState().setTradingMode(res.mode);
        useStore.getState().setSafeMode(res.safe_mode);
        useStore.getState().setAutonomy(res.autonomous);
      } catch {
        useStore.getState().setTradingMode("paper");
      }
    };

    const fetchEquity = async () => {
      try {
        const res = await endpoints.risk();
        if (typeof res.equity === "number") {
          useStore.getState().setDashboard({ equity: res.equity, daily_pnl: res.daily_pnl ?? 0 });
        }
      } catch { /* ignore */ }
    };

    fetchMode();
    fetchEquity();
    const interval = setInterval(fetchMode, 10_000);
    return () => clearInterval(interval);
  }, []);

  // Toggle autonomous via API
  const handleAutonomyToggle = async (on: boolean) => {
    setAutonomy(on); // optimistic
    try {
      await endpoints.toggleAutonomous(on);
    } catch {
      setAutonomy(!on); // revert
    }
  };

  return (
    <>
      {/* Safe mode banner */}
      <AnimatePresence>
        {safeMode && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            className="overflow-hidden"
          >
            <div className="flex items-center justify-center gap-2 bg-gradient-to-r from-safe-mode/10 via-safe-mode/20 to-safe-mode/10 px-4 py-2 border-b border-safe-mode/20">
              <Shield className="h-3.5 w-3.5 text-safe-mode" />
              <span className="text-xs font-semibold text-safe-mode tracking-wide">
                SAFE MODE ACTIVE — TRADING DISABLED
              </span>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Main topbar */}
      <header className="sticky top-0 z-20 flex h-16 items-center justify-between border-b border-border/40 bg-background/80 backdrop-blur-xl pl-16 pr-5 lg:pl-5"
        style={{ background: "linear-gradient(180deg, hsl(222 47% 7% / 0.95), hsl(222 47% 5% / 0.9))" }}
      >
        {/* Left section: Mode badge + Autonomy toggle */}
        <div className="flex items-center gap-6">
          {/* Trading mode badge */}
          <div className="flex items-center gap-2">
            <span className={cn(
              "inline-flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-[11px] font-bold tracking-wider transition-all",
              isPaper
                ? "bg-warning/10 text-warning border border-warning/30"
                : "bg-profit/10 text-profit border border-profit/30 shadow-[0_0_10px_hsl(152,69%,53%,0.15)]"
            )}>
              {isPaper ? (
                <>
                  <FileQuestion className="h-3 w-3" />
                  PAPER
                </>
              ) : (
                <>
                  <span className="h-1.5 w-1.5 rounded-full bg-profit animate-pulse" />
                  LIVE
                </>
              )}
            </span>
          </div>

          {/* Divider */}
          <div className="h-8 w-px bg-border/50" />

          {/* Autonomous trading toggle */}
          <div className="flex items-center gap-3">
            <div className="flex flex-col">
              <span className="text-[10px] font-semibold uppercase tracking-widest text-muted-foreground">
                Autonomous
              </span>
              <span className={cn(
                "text-xs font-bold",
                autonomyOn && !safeMode ? "text-profit" : "text-muted-foreground"
              )}>
                {autonomyOn && !safeMode ? "ACTIVE" : "INACTIVE"}
              </span>
            </div>
            <div className="relative">
              <Switch
                checked={autonomyOn && !safeMode}
                onCheckedChange={handleAutonomyToggle}
                disabled={safeMode}
                className={cn(
                  "transition-all duration-300",
                  autonomyOn && !safeMode && "shadow-[0_0_15px_hsl(152,69%,53%,0.4)] ring-2 ring-profit/20"
                )}
              />
              {autonomyOn && !safeMode && (
                <motion.div
                  className="absolute -right-1 -top-1 h-2.5 w-2.5 rounded-full bg-profit"
                  animate={{ scale: [1, 1.3, 1], opacity: [1, 0.7, 1] }}
                  transition={{ duration: 2, repeat: Infinity }}
                />
              )}
            </div>
          </div>

          {/* Divider */}
          <div className="h-8 w-px bg-border/50" />

          {/* Status indicators */}
          <div className="flex items-center gap-4">
            {/* Broker */}
            <div className="flex items-center gap-1.5" title={isPaper ? "Paper trading (no broker)" : "Broker connection"}>
              <span className={cn(
                "flex h-2 w-2 rounded-full",
                broker.status === "connected" && "bg-profit shadow-[0_0_6px_hsl(152,69%,53%,0.5)]",
                broker.status === "disconnected" && isPaper && "bg-warning",
                broker.status === "disconnected" && !isPaper && "bg-loss",
                broker.status === "degraded" && "bg-warning animate-pulse"
              )} />
              <span className="text-[11px] text-muted-foreground hidden xl:inline">
                {broker.status === "connected" ? "Broker" : isPaper ? "Paper Mode" : "Disconnected"}
              </span>
            </div>

            {/* Market feed */}
            <div className="flex items-center gap-1.5" title={marketFeed.last_tick_ts ?? "Market feed"}>
              {marketFeed.healthy ? (
                <Radio className="h-3.5 w-3.5 text-profit" />
              ) : marketFeed.connected ? (
                <Wifi className="h-3.5 w-3.5 text-warning animate-pulse" />
              ) : isPaper ? (
                <Radio className="h-3.5 w-3.5 text-warning" />
              ) : (
                <WifiOff className="h-3.5 w-3.5 text-loss" />
              )}
              <span className="text-[11px] text-muted-foreground hidden xl:inline">
                {marketFeed.healthy ? "Live Feed" : isPaper ? "YFinance" : "Reconnecting"}
              </span>
            </div>

            {/* WebSocket — status driven by broker connection or paper mode active */}
            {(broker.connected || (isPaper && autonomyOn)) && (
              <span className="status-badge-live">
                <span className={cn("h-1.5 w-1.5 rounded-full animate-pulse", isPaper ? "bg-warning" : "bg-profit")} />
                {isPaper ? "PAPER" : "LIVE"}
              </span>
            )}

            {/* WebSocket connection status indicator */}
            <ConnectionStatus />
          </div>
        </div>

        {/* Right section: P&L + user */}
        <div className="flex items-center gap-5">
          {/* Equity */}
          <div className="text-right hidden sm:block">
            <div className="kpi-label">Portfolio</div>
            <div className="font-mono text-sm font-bold tracking-tight">
              {formatCurrency(equity)}
            </div>
          </div>

          {/* Divider */}
          <div className="h-8 w-px bg-border/50 hidden sm:block" />

          {/* Daily P&L */}
          <motion.div
            className="text-right"
            key={dailyPnl}
            initial={{ opacity: 0, y: -5 }}
            animate={{ opacity: 1, y: 0 }}
          >
            <div className="kpi-label">Daily P&L</div>
            <div className="flex items-center gap-1 justify-end">
              {isProfit ? (
                <ArrowUpRight className="h-3.5 w-3.5 text-profit" />
              ) : (
                <ArrowDownRight className="h-3.5 w-3.5 text-loss" />
              )}
              <span className={cn(
                "font-mono text-sm font-bold",
                isProfit ? "text-profit" : "text-loss"
              )}>
                {formatPercent(dailyPnl)}
              </span>
            </div>
          </motion.div>

          {/* Divider */}
          <div className="h-8 w-px bg-border/50" />

          {/* Notifications */}
          <button className="relative rounded-lg p-2 text-muted-foreground transition-colors hover:bg-muted hover:text-foreground">
            <Bell className="h-4 w-4" />
            <span className="absolute right-1 top-1 h-2 w-2 rounded-full bg-primary animate-pulse" />
          </button>

          {/* User avatar */}
          <div className="flex h-8 w-8 items-center justify-center rounded-full bg-gradient-to-br from-primary/80 to-accent text-xs font-bold text-white shadow-lg">
            <User className="h-4 w-4" />
          </div>
        </div>
      </header>
    </>
  );
}
