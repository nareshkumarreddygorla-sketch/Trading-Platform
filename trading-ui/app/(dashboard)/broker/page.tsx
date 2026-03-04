"use client";

import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { motion } from "framer-motion";
import { cn } from "@/lib/utils";
import { useStore } from "@/store/useStore";
import { endpoints } from "@/lib/api/client";
import {
  Plug, Unplug, CheckCircle, XCircle, Radio, Wifi,
  WifiOff, RefreshCw, Clock, Key, Shield,
  ArrowUpRight, ArrowDownRight, Activity,
  Play, Square, Zap, Eye, EyeOff,
} from "lucide-react";
import Link from "next/link";

type RecentOrder = {
  order_id: string;
  symbol: string;
  side: string;
  quantity: number;
  status: string;
  avg_price?: number;
  filled_qty: number;
  ts?: string;
};

export default function BrokerPage() {
  const broker = useStore((s) => s.broker);
  const marketFeed = useStore((s) => s.marketFeed);
  const [reconnecting, setReconnecting] = useState(false);
  const [showCredForm, setShowCredForm] = useState(false);
  const [showPasswords, setShowPasswords] = useState(false);
  const [credForm, setCredForm] = useState({
    api_key: "",
    client_id: "",
    password: "",
    totp_secret: "",
  });
  const [credError, setCredError] = useState("");
  const [credSuccess, setCredSuccess] = useState("");
  const queryClient = useQueryClient();

  // Fetch broker status from API
  const { data: brokerStatus, refetch: refetchBroker } = useQuery({
    queryKey: ["broker-status"],
    queryFn: () => endpoints.brokerStatus(),
    refetchInterval: 5000,
  });

  // Fetch trading mode
  const { data: tradingModeData } = useQuery({
    queryKey: ["trading-mode"],
    queryFn: () => endpoints.tradingMode(),
    refetchInterval: 5000,
  });

  // Fetch risk snapshot
  const { data: risk } = useQuery({
    queryKey: ["risk-broker"],
    queryFn: () => endpoints.risk(),
    refetchInterval: 5000,
  });

  // Fetch recent orders
  const { data: ordersData } = useQuery({
    queryKey: ["orders-broker"],
    queryFn: () => endpoints.orders(10),
    refetchInterval: 5000,
  });

  // Configure broker credentials
  const configureMutation = useMutation({
    mutationFn: (creds: typeof credForm) => endpoints.brokerConfigure(creds),
    onSuccess: (data) => {
      setCredSuccess(data.message);
      setCredError("");
      setShowCredForm(false);
      queryClient.invalidateQueries({ queryKey: ["broker-status"] });
      queryClient.invalidateQueries({ queryKey: ["trading-mode"] });
      useStore.setState({
        broker: { connected: true, status: "connected" },
        tradingMode: "live",
      });
    },
    onError: (err: Error) => {
      setCredError(err.message);
      setCredSuccess("");
    },
  });

  // Disconnect broker
  const disconnectMutation = useMutation({
    mutationFn: () => endpoints.brokerDisconnect(),
    onSuccess: (data) => {
      setCredSuccess(data.message);
      setCredError("");
      setShowCredForm(false);
      setCredForm({ api_key: "", client_id: "", password: "", totp_secret: "" });
      queryClient.invalidateQueries({ queryKey: ["broker-status"] });
      queryClient.invalidateQueries({ queryKey: ["trading-mode"] });
      useStore.setState({
        broker: { connected: false, status: "disconnected" },
        tradingMode: "paper",
      });
    },
    onError: (err: Error) => {
      setCredError(err.message);
    },
  });

  // Toggle autonomous trading
  const autonomousMutation = useMutation({
    mutationFn: (enabled: boolean) => endpoints.toggleAutonomous(enabled),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["broker-status"] });
      queryClient.invalidateQueries({ queryKey: ["trading-mode"] });
    },
  });

  const isConnected = brokerStatus?.connected ?? broker.status === "connected";
  const isLive = brokerStatus?.mode === "live";
  const autonomousRunning = brokerStatus?.autonomous_running ?? tradingModeData?.autonomous ?? false;
  const openPositions = (risk?.positions as unknown[])?.length ?? 0;
  const recentOrders = (ordersData?.orders ?? []) as RecentOrder[];
  const tickCount = brokerStatus?.tick_count ?? 0;
  const openTrades = brokerStatus?.open_trades ?? 0;

  const handleReconnect = async () => {
    setReconnecting(true);
    try {
      await endpoints.health();
      await refetchBroker();
      useStore.setState({
        broker: { connected: true, status: "connected" },
        marketFeed: { connected: true, healthy: true, last_tick_ts: new Date().toISOString() },
      });
    } catch {
      useStore.setState({
        broker: { connected: false, status: "disconnected" },
      });
    } finally {
      setReconnecting(false);
    }
  };

  const handleConfigureSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    setCredError("");
    setCredSuccess("");
    configureMutation.mutate(credForm);
  };

  return (
    <div className="space-y-6 pb-8">
      <motion.div initial={{ opacity: 0, y: -10 }} animate={{ opacity: 1, y: 0 }}>
        <h1 className="text-2xl font-bold tracking-tight">Broker</h1>
        <p className="text-sm text-muted-foreground mt-0.5">Connection, credentials & autonomous trading control</p>
      </motion.div>

      {/* Connection + Autonomous Control cards */}
      <div className="grid gap-6 xl:grid-cols-2">
        {/* Broker connection */}
        <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.1 }}>
          <div className={cn(
            isConnected ? "glass-card-profit" : "glass-card-loss",
            "p-6"
          )}>
            <div className="flex items-start justify-between mb-4">
              <div className="flex items-center gap-3">
                <div className={cn(
                  "flex h-12 w-12 items-center justify-center rounded-xl",
                  isConnected ? "bg-profit/10" : "bg-loss/10"
                )}>
                  <Plug className={cn("h-6 w-6", isConnected ? "text-profit" : "text-loss")} />
                </div>
                <div>
                  <h3 className="text-lg font-semibold">Broker Connection</h3>
                  <p className="text-xs text-muted-foreground mt-0.5">Angel One Trading API</p>
                </div>
              </div>
              {isConnected ? (
                <span className="status-badge-live">
                  <CheckCircle className="h-3 w-3" />
                  {isLive ? "LIVE" : "PAPER"}
                </span>
              ) : (
                <span className="status-badge-danger">
                  <XCircle className="h-3 w-3" />
                  DISCONNECTED
                </span>
              )}
            </div>

            <div className="grid grid-cols-2 gap-3 mt-4">
              {[
                { label: "Mode", value: isLive ? "LIVE TRADING" : "PAPER MODE" },
                { label: "Status", value: brokerStatus?.healthy ? "HEALTHY" : brokerStatus?.safe_mode ? "SAFE MODE" : "OFFLINE" },
                { label: "Client ID", value: brokerStatus?.client_id ?? "---" },
                { label: "Last Connected", value: brokerStatus?.last_connected ? new Date(brokerStatus.last_connected).toLocaleString("en-IN", { hour: "2-digit", minute: "2-digit", day: "2-digit", month: "short" }) : "---" },
                { label: "Open Positions", value: String(openPositions) },
                { label: "Daily P&L", value: `${(risk?.daily_pnl ?? 0).toFixed(2)}%` },
              ].map((item) => (
                <div key={item.label} className="rounded-lg bg-muted/20 p-3">
                  <div className="text-[10px] text-muted-foreground">{item.label}</div>
                  <div className="font-mono text-sm font-semibold mt-0.5">{item.value}</div>
                </div>
              ))}
            </div>

            <div className="flex gap-2 mt-4">
              <button
                onClick={() => setShowCredForm(!showCredForm)}
                className="flex-1 flex items-center justify-center gap-2 rounded-xl border border-primary/30 py-2.5 text-xs font-medium text-primary transition-all hover:bg-primary/5"
              >
                <Key className="h-3.5 w-3.5" />
                {showCredForm ? "Cancel" : "Configure Credentials"}
              </button>
              {isConnected && (
                <button
                  onClick={() => disconnectMutation.mutate()}
                  disabled={disconnectMutation.isPending}
                  className={cn(
                    "flex-1 flex items-center justify-center gap-2 rounded-xl border border-loss/30 py-2.5 text-xs font-medium text-loss transition-all hover:bg-loss/5",
                    disconnectMutation.isPending && "opacity-50 cursor-not-allowed"
                  )}
                >
                  {disconnectMutation.isPending ? (
                    <RefreshCw className="h-3.5 w-3.5 animate-spin" />
                  ) : (
                    <Unplug className="h-3.5 w-3.5" />
                  )}
                  {disconnectMutation.isPending ? "Disconnecting..." : "Disconnect"}
                </button>
              )}
              <button
                onClick={handleReconnect}
                disabled={reconnecting}
                className={cn(
                  "flex-1 flex items-center justify-center gap-2 rounded-xl border border-border/50 py-2.5 text-xs font-medium transition-all",
                  reconnecting
                    ? "text-muted-foreground opacity-50 cursor-not-allowed"
                    : "text-muted-foreground hover:border-primary/40 hover:text-primary hover:bg-primary/5"
                )}
              >
                <RefreshCw className={cn("h-3.5 w-3.5", reconnecting && "animate-spin")} />
                Reconnect
              </button>
            </div>
          </div>
        </motion.div>

        {/* Autonomous Trading Control */}
        <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.15 }}>
          <div className={cn(
            autonomousRunning ? "glass-card-profit" : "glass-card",
            "p-6"
          )}>
            <div className="flex items-start justify-between mb-4">
              <div className="flex items-center gap-3">
                <div className={cn(
                  "flex h-12 w-12 items-center justify-center rounded-xl",
                  autonomousRunning ? "bg-profit/10" : "bg-muted/20"
                )}>
                  <Zap className={cn("h-6 w-6", autonomousRunning ? "text-profit" : "text-muted-foreground")} />
                </div>
                <div>
                  <h3 className="text-lg font-semibold">Autonomous Trading</h3>
                  <p className="text-xs text-muted-foreground mt-0.5">AI-powered fully autonomous execution</p>
                </div>
              </div>
              {autonomousRunning ? (
                <span className="status-badge-live">
                  <span className="h-1.5 w-1.5 rounded-full bg-profit animate-pulse" />
                  RUNNING
                </span>
              ) : (
                <span className="text-xs font-medium text-muted-foreground bg-muted/30 px-3 py-1 rounded-full">
                  PAUSED
                </span>
              )}
            </div>

            <div className="grid grid-cols-2 gap-3 mt-4">
              {[
                { label: "Ticks Processed", value: String(tickCount) },
                { label: "Active Trades", value: String(openTrades) },
                { label: "Circuit Breaker", value: tradingModeData?.circuit_open ? "OPEN" : "CLOSED" },
                { label: "Kill Switch", value: tradingModeData?.kill_switch_armed ? "ARMED" : "OFF" },
              ].map((item) => (
                <div key={item.label} className="rounded-lg bg-muted/20 p-3">
                  <div className="text-[10px] text-muted-foreground">{item.label}</div>
                  <div className="font-mono text-sm font-semibold mt-0.5">{item.value}</div>
                </div>
              ))}
            </div>

            <div className="mt-4">
              <button
                onClick={() => autonomousMutation.mutate(!autonomousRunning)}
                disabled={autonomousMutation.isPending || tradingModeData?.safe_mode}
                className={cn(
                  "w-full flex items-center justify-center gap-2 rounded-xl py-3 text-sm font-semibold transition-all",
                  autonomousRunning
                    ? "bg-loss/10 text-loss border border-loss/30 hover:bg-loss/20"
                    : "bg-profit/10 text-profit border border-profit/30 hover:bg-profit/20",
                  (autonomousMutation.isPending || tradingModeData?.safe_mode) && "opacity-50 cursor-not-allowed"
                )}
              >
                {autonomousRunning ? (
                  <>
                    <Square className="h-4 w-4" />
                    Stop Autonomous Trading
                  </>
                ) : (
                  <>
                    <Play className="h-4 w-4" />
                    Start Autonomous Trading
                  </>
                )}
              </button>
              {tradingModeData?.safe_mode && (
                <p className="text-xs text-loss mt-2 text-center">
                  Safe mode active — broker unreachable. Cannot enable autonomous trading.
                </p>
              )}
            </div>
          </div>
        </motion.div>
      </div>

      {/* Credential Configuration Form */}
      {showCredForm && (
        <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }}>
          <div className="glass-card p-6">
            <div className="flex items-center gap-3 mb-4">
              <Shield className="h-5 w-5 text-primary" />
              <div>
                <h3 className="text-sm font-semibold">Angel One Credentials</h3>
                <p className="text-xs text-muted-foreground mt-0.5">
                  Enter your Angel One API credentials to enable live trading
                </p>
              </div>
            </div>

            <form onSubmit={handleConfigureSubmit} className="space-y-4">
              <div className="grid gap-4 sm:grid-cols-2">
                <div>
                  <label className="text-xs font-medium text-muted-foreground mb-1.5 block">API Key</label>
                  <input
                    type={showPasswords ? "text" : "password"}
                    value={credForm.api_key}
                    onChange={(e) => setCredForm({ ...credForm, api_key: e.target.value })}
                    placeholder="Your Angel One API key"
                    required
                    className="w-full rounded-lg border border-border/50 bg-muted/10 px-3 py-2 text-sm font-mono focus:border-primary/50 focus:outline-none"
                  />
                </div>
                <div>
                  <label className="text-xs font-medium text-muted-foreground mb-1.5 block">Client ID</label>
                  <input
                    type="text"
                    value={credForm.client_id}
                    onChange={(e) => setCredForm({ ...credForm, client_id: e.target.value })}
                    placeholder="e.g. A12345"
                    required
                    className="w-full rounded-lg border border-border/50 bg-muted/10 px-3 py-2 text-sm font-mono focus:border-primary/50 focus:outline-none"
                  />
                </div>
                <div>
                  <label className="text-xs font-medium text-muted-foreground mb-1.5 block">Password</label>
                  <input
                    type={showPasswords ? "text" : "password"}
                    value={credForm.password}
                    onChange={(e) => setCredForm({ ...credForm, password: e.target.value })}
                    placeholder="Trading password"
                    required
                    className="w-full rounded-lg border border-border/50 bg-muted/10 px-3 py-2 text-sm font-mono focus:border-primary/50 focus:outline-none"
                  />
                </div>
                <div>
                  <label className="text-xs font-medium text-muted-foreground mb-1.5 block">TOTP Secret</label>
                  <input
                    type={showPasswords ? "text" : "password"}
                    value={credForm.totp_secret}
                    onChange={(e) => setCredForm({ ...credForm, totp_secret: e.target.value })}
                    placeholder="Base32 TOTP secret"
                    required
                    className="w-full rounded-lg border border-border/50 bg-muted/10 px-3 py-2 text-sm font-mono focus:border-primary/50 focus:outline-none"
                  />
                </div>
              </div>

              <div className="flex items-center gap-2">
                <button
                  type="button"
                  onClick={() => setShowPasswords(!showPasswords)}
                  className="flex items-center gap-1.5 text-xs text-muted-foreground hover:text-primary transition-colors"
                >
                  {showPasswords ? <EyeOff className="h-3.5 w-3.5" /> : <Eye className="h-3.5 w-3.5" />}
                  {showPasswords ? "Hide" : "Show"} credentials
                </button>
              </div>

              {credError && (
                <div className="rounded-lg bg-loss/10 border border-loss/30 p-3">
                  <p className="text-xs text-loss font-medium">{credError}</p>
                </div>
              )}
              {credSuccess && (
                <div className="rounded-lg bg-profit/10 border border-profit/30 p-3">
                  <p className="text-xs text-profit font-medium">{credSuccess}</p>
                </div>
              )}

              <div className="flex gap-3">
                <button
                  type="submit"
                  disabled={configureMutation.isPending}
                  className={cn(
                    "flex-1 flex items-center justify-center gap-2 rounded-xl bg-primary py-2.5 text-sm font-semibold text-primary-foreground transition-all hover:bg-primary/90",
                    configureMutation.isPending && "opacity-50 cursor-not-allowed"
                  )}
                >
                  {configureMutation.isPending ? (
                    <RefreshCw className="h-4 w-4 animate-spin" />
                  ) : (
                    <Plug className="h-4 w-4" />
                  )}
                  {configureMutation.isPending ? "Connecting..." : "Connect & Switch to Live"}
                </button>
              </div>

              <p className="text-[10px] text-muted-foreground text-center">
                Credentials are validated against Angel One API before switching to live mode.
                They are stored in memory only — not persisted to disk.
              </p>
            </form>
          </div>
        </motion.div>
      )}

      {/* Market Feed */}
      <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.2 }}>
        <div className={cn(
          marketFeed.healthy ? "glass-card-profit" : "glass-card-loss",
          "p-6"
        )}>
          <div className="flex items-start justify-between mb-4">
            <div className="flex items-center gap-3">
              <div className={cn(
                "flex h-12 w-12 items-center justify-center rounded-xl",
                marketFeed.healthy ? "bg-profit/10" : "bg-loss/10"
              )}>
                {marketFeed.healthy ? (
                  <Radio className="h-6 w-6 text-profit" />
                ) : marketFeed.connected ? (
                  <Wifi className="h-6 w-6 text-warning animate-pulse" />
                ) : (
                  <WifiOff className="h-6 w-6 text-loss" />
                )}
              </div>
              <div>
                <h3 className="text-lg font-semibold">Market Data Feed</h3>
                <p className="text-xs text-muted-foreground mt-0.5">Real-time price streaming</p>
              </div>
            </div>
            {marketFeed.healthy ? (
              <span className="status-badge-live">
                <span className="h-1.5 w-1.5 rounded-full bg-profit animate-pulse" />
                STREAMING
              </span>
            ) : (
              <span className="status-badge-danger">OFFLINE</span>
            )}
          </div>

          <div className="grid grid-cols-2 gap-3 mt-4">
            {[
              { label: "Status", value: marketFeed.healthy ? "HEALTHY" : marketFeed.connected ? "DEGRADED" : "OFFLINE" },
              { label: "Last Tick", value: marketFeed.last_tick_ts ? new Date(marketFeed.last_tick_ts).toLocaleTimeString() : "—" },
              { label: "Connected", value: marketFeed.connected ? "Yes" : "No" },
              { label: "Feed Health", value: marketFeed.healthy ? "OK" : "—" },
            ].map((item) => (
              <div key={item.label} className="rounded-lg bg-muted/20 p-3">
                <div className="text-[10px] text-muted-foreground">{item.label}</div>
                <div className="font-mono text-sm font-semibold mt-0.5">{item.value}</div>
              </div>
            ))}
          </div>
        </div>
      </motion.div>

      {/* Recent Orders */}
      <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.25 }}>
        <div className="glass-card overflow-hidden">
          <div className="flex items-center justify-between border-b border-border/30 p-5">
            <div>
              <h3 className="text-sm font-semibold">Recent Orders</h3>
              <p className="text-xs text-muted-foreground mt-0.5">Last 10 execution events</p>
            </div>
            <div className="flex items-center gap-3">
              <Link
                href="/trades"
                className="text-xs font-medium text-primary hover:text-primary/80 transition-colors"
              >
                View All
              </Link>
              <Clock className="h-4 w-4 text-primary" />
            </div>
          </div>

          {recentOrders.length === 0 ? (
            <div className="flex flex-col items-center justify-center py-12 text-muted-foreground">
              <Activity className="h-10 w-10 mb-3 opacity-30" />
              <p className="text-sm font-medium">No orders yet</p>
              <p className="text-xs mt-1">Orders will appear when strategies execute trades</p>
            </div>
          ) : (
            <div className="divide-y divide-border/10">
              {recentOrders.map((o, i) => (
                <motion.div
                  key={o.order_id || `${o.symbol}-${i}`}
                  initial={{ opacity: 0, x: -10 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: 0.05 + i * 0.03 }}
                  className="flex items-center gap-4 px-5 py-3 hover:bg-muted/10 transition-colors"
                >
                  <div className={cn(
                    "flex h-8 w-8 items-center justify-center rounded-lg",
                    o.side === "BUY" ? "bg-profit/10" : "bg-loss/10"
                  )}>
                    {o.side === "BUY" ? (
                      <ArrowUpRight className="h-4 w-4 text-profit" />
                    ) : (
                      <ArrowDownRight className="h-4 w-4 text-loss" />
                    )}
                  </div>
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2">
                      <span className="font-mono text-sm font-semibold">{o.symbol}</span>
                      <span className={cn(
                        "text-[10px] font-bold px-1.5 py-0.5 rounded",
                        o.side === "BUY" ? "bg-profit/10 text-profit" : "bg-loss/10 text-loss"
                      )}>
                        {o.side}
                      </span>
                    </div>
                    <span className="text-[10px] text-muted-foreground">
                      {o.filled_qty}/{o.quantity} filled {o.avg_price ? `@ ${o.avg_price.toFixed(2)}` : ""}
                    </span>
                  </div>
                  <div className="text-right">
                    <span className={cn(
                      "text-[10px] font-bold px-2 py-0.5 rounded",
                      o.status === "FILLED" ? "bg-profit/10 text-profit" :
                      o.status === "CANCELLED" || o.status === "REJECTED" ? "bg-loss/10 text-loss" :
                      "bg-warning/10 text-warning"
                    )}>
                      {o.status}
                    </span>
                    <div className="text-[10px] text-muted-foreground mt-0.5">
                      {o.ts ? new Date(o.ts).toLocaleTimeString("en-IN", { hour: "2-digit", minute: "2-digit" }) : ""}
                    </div>
                  </div>
                </motion.div>
              ))}
            </div>
          )}
        </div>
      </motion.div>
    </div>
  );
}
