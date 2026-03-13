"use client";

import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { motion } from "framer-motion";
import { cn } from "@/lib/utils";
import { useStore } from "@/store/useStore";
import { endpoints } from "@/lib/api/client";
import {
  Plug, Unplug, CheckCircle, XCircle, Radio, Wifi,
  WifiOff, RefreshCw, Clock, Key, Shield, ShieldCheck,
  ArrowUpRight, ArrowDownRight, Activity,
  Play, Square, Zap, Eye, EyeOff, AlertTriangle,
} from "lucide-react";
import Link from "next/link";
import { dispatchToast } from "@/components/Toaster";

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

type ConfigStep = "idle" | "form" | "validating" | "confirm" | "switching";
type TradingModeChoice = "paper" | "live";

export default function BrokerPage() {
  const broker = useStore((s) => s.broker);
  const marketFeed = useStore((s) => s.marketFeed);
  const [reconnecting, setReconnecting] = useState(false);
  const [configStep, setConfigStep] = useState<ConfigStep>("idle");
  const [showPasswords, setShowPasswords] = useState(false);
  const [selectedMode, setSelectedMode] = useState<TradingModeChoice>("paper");
  const [credForm, setCredForm] = useState({
    api_key: "",
    client_id: "",
    password: "",
    totp_secret: "",
  });
  const [credError, setCredError] = useState("");
  const [credSuccess, setCredSuccess] = useState("");
  const [confirmToken, setConfirmToken] = useState<string | null>(null);
  const [validatedClientId, setValidatedClientId] = useState("");
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

  // Step 1: Validate credentials (paper mode connects directly, live mode needs confirmation)
  const validateMutation = useMutation({
    mutationFn: (creds: typeof credForm & { mode: TradingModeChoice }) =>
      endpoints.brokerConfigure(creds),
    onSuccess: (data) => {
      if (data.confirm_token) {
        // Live mode: two-step flow — credentials validated, awaiting confirmation
        setConfirmToken(data.confirm_token);
        setValidatedClientId(credForm.client_id);
        setConfigStep("confirm");
        setCredError("");
        setCredSuccess("Credentials validated! Confirm to switch to LIVE mode.");
        dispatchToast("success", "Credentials Valid", "Angel One login successful. Confirm to go live.");
      } else {
        // Paper mode: connected directly (or direct connect)
        setCredSuccess(data.message || "Connected successfully!");
        setCredError("");
        resetForm();
        queryClient.invalidateQueries({ queryKey: ["broker-status"] });
        queryClient.invalidateQueries({ queryKey: ["trading-mode"] });
        useStore.setState({
          broker: { connected: data.connected, status: data.connected ? "connected" : "disconnected" },
          tradingMode: data.mode === "live" ? "live" : "paper",
        });
        dispatchToast(
          "success",
          "Paper Trading Active",
          "Connected with real market data. Orders are simulated."
        );
      }
    },
    onError: (err: Error) => {
      setCredError(err.message);
      setCredSuccess("");
      setConfigStep("form");
    },
  });

  // Step 2: Confirm live switch
  const confirmMutation = useMutation({
    mutationFn: (token: string) => endpoints.brokerConfirmLive(token),
    onSuccess: (data) => {
      setCredSuccess(data.message);
      setCredError("");
      resetForm();
      queryClient.invalidateQueries({ queryKey: ["broker-status"] });
      queryClient.invalidateQueries({ queryKey: ["trading-mode"] });
      useStore.setState({
        broker: { connected: data.connected, status: data.connected ? "connected" : "disconnected" },
        tradingMode: data.mode === "live" ? "live" : "paper",
      });
      dispatchToast(
        "success",
        "Live Mode Active",
        data.auto_started
          ? "Broker connected and autonomous trading started!"
          : "Broker connected. You can start autonomous trading."
      );
    },
    onError: (err: Error) => {
      setCredError(err.message);
      setCredSuccess("");
      setConfigStep("form");
      setConfirmToken(null);
    },
  });

  // Disconnect broker
  const disconnectMutation = useMutation({
    mutationFn: () => endpoints.brokerDisconnect(),
    onSuccess: (data) => {
      setCredSuccess(data.message);
      setCredError("");
      resetForm();
      queryClient.invalidateQueries({ queryKey: ["broker-status"] });
      queryClient.invalidateQueries({ queryKey: ["trading-mode"] });
      useStore.setState({
        broker: { connected: false, status: "disconnected" },
        tradingMode: "paper",
      });
      dispatchToast("success", "Disconnected", "Switched back to paper trading mode.");
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
  const isPaperWithCreds = isConnected && !isLive;
  const autonomousRunning = brokerStatus?.autonomous_running ?? tradingModeData?.autonomous ?? false;
  const openPositions = (risk?.positions as unknown[])?.length ?? 0;
  const recentOrders = (ordersData?.orders ?? []) as RecentOrder[];
  const tickCount = brokerStatus?.tick_count ?? 0;
  const openTrades = brokerStatus?.open_trades ?? 0;

  const resetForm = () => {
    setConfigStep("idle");
    setCredForm({ api_key: "", client_id: "", password: "", totp_secret: "" });
    setConfirmToken(null);
    setValidatedClientId("");
    setShowPasswords(false);
    setSelectedMode("paper");
  };

  const handleReconnect = async () => {
    setReconnecting(true);
    try {
      await endpoints.health();
      const status = await endpoints.brokerStatus();
      await refetchBroker();
      useStore.setState({
        broker: { connected: status.connected, status: status.connected ? "connected" : "disconnected" },
      });
    } catch (err) {
      useStore.setState({
        broker: { connected: false, status: "disconnected" },
      });
      dispatchToast(
        "error",
        "Reconnect Failed",
        err instanceof Error ? err.message : "Could not reach the trading server."
      );
    } finally {
      setReconnecting(false);
    }
  };

  const handleValidateSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    setCredError("");
    setCredSuccess("");
    setConfigStep("validating");
    validateMutation.mutate({ ...credForm, mode: selectedMode });
  };

  const handleConfirmLive = () => {
    if (!confirmToken) return;
    setConfigStep("switching");
    confirmMutation.mutate(confirmToken);
  };

  const handleCancelConfirm = () => {
    setConfirmToken(null);
    setConfigStep("form");
    setCredSuccess("");
  };

  const showForm = configStep === "form" || configStep === "validating";
  const showConfirm = configStep === "confirm" || configStep === "switching";

  return (
    <div className="space-y-6 pb-8">
      <motion.div initial={{ opacity: 0, y: -10 }} animate={{ opacity: 1, y: 0 }}>
        <h1 className="text-2xl font-bold tracking-tight">Broker Settings</h1>
        <p className="text-sm text-muted-foreground mt-0.5">Connect your broker, manage credentials & control autonomous trading</p>
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
                  <p className="text-xs text-muted-foreground mt-0.5">Angel One SmartAPI</p>
                </div>
              </div>
              {isConnected ? (
                isLive ? (
                  <span className="status-badge-live">
                    <CheckCircle className="h-3 w-3" />
                    LIVE
                  </span>
                ) : (
                  <span className="text-xs font-bold px-3 py-1 rounded-full bg-primary/10 text-primary border border-primary/30 flex items-center gap-1.5">
                    <CheckCircle className="h-3 w-3" />
                    PAPER
                  </span>
                )
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
                { label: "Daily P&L", value: `₹${(risk?.daily_pnl ?? 0).toLocaleString('en-IN', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}` },
              ].map((item) => (
                <div key={item.label} className="rounded-lg bg-muted/20 p-3">
                  <div className="text-[10px] text-muted-foreground">{item.label}</div>
                  <div className="font-mono text-sm font-semibold mt-0.5">{item.value}</div>
                </div>
              ))}
            </div>

            <div className="flex gap-2 mt-4">
              <button
                onClick={() => {
                  if (configStep === "idle") {
                    setConfigStep("form");
                    setCredError("");
                    setCredSuccess("");
                  } else {
                    resetForm();
                  }
                }}
                className="flex-1 flex items-center justify-center gap-2 rounded-xl border border-primary/30 py-2.5 text-xs font-medium text-primary transition-all hover:bg-primary/5"
                aria-label={configStep !== "idle" ? "Cancel credential configuration" : "Configure broker credentials"}
                aria-expanded={configStep !== "idle"}
              >
                <Key className="h-3.5 w-3.5" />
                {configStep !== "idle" ? "Cancel" : "Configure Credentials"}
              </button>
              {isConnected && (
                <button
                  onClick={() => disconnectMutation.mutate()}
                  disabled={disconnectMutation.isPending}
                  aria-label={disconnectMutation.isPending ? "Disconnecting from broker" : "Disconnect from broker"}
                  aria-busy={disconnectMutation.isPending}
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
                aria-label={reconnecting ? "Reconnecting to broker" : "Reconnect to broker"}
                aria-busy={reconnecting}
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
                aria-label={autonomousRunning ? "Stop autonomous trading" : "Start autonomous trading"}
                aria-busy={autonomousMutation.isPending}
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

      {/* Step 1: Credential Entry Form */}
      {showForm && (
        <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }}>
          <div className="glass-card p-6">
            <div className="flex items-center gap-3 mb-4">
              <Shield className="h-5 w-5 text-primary" />
              <div>
                <h3 className="text-sm font-semibold">Angel One Credentials</h3>
                <p className="text-xs text-muted-foreground mt-0.5">
                  Enter your Angel One SmartAPI credentials to enable live trading. Credentials are validated against Angel One servers and stored securely in memory only.
                </p>
              </div>
            </div>

            <form onSubmit={handleValidateSubmit} className="space-y-4" aria-label="Broker credentials configuration">
              {/* Paper / Live Mode Toggle */}
              <div className="flex items-center justify-center gap-1 p-1 rounded-xl bg-muted/20 border border-border/30">
                <button
                  type="button"
                  onClick={() => setSelectedMode("paper")}
                  className={cn(
                    "flex-1 flex items-center justify-center gap-2 rounded-lg py-2.5 text-sm font-semibold transition-all",
                    selectedMode === "paper"
                      ? "bg-primary text-primary-foreground shadow-sm"
                      : "text-muted-foreground hover:text-foreground hover:bg-muted/20"
                  )}
                >
                  <Shield className="h-4 w-4" />
                  Paper Trading
                </button>
                <button
                  type="button"
                  onClick={() => setSelectedMode("live")}
                  className={cn(
                    "flex-1 flex items-center justify-center gap-2 rounded-lg py-2.5 text-sm font-semibold transition-all",
                    selectedMode === "live"
                      ? "bg-warning text-black shadow-sm"
                      : "text-muted-foreground hover:text-foreground hover:bg-muted/20"
                  )}
                >
                  <Zap className="h-4 w-4" />
                  Live Trading
                </button>
              </div>

              {/* Mode description */}
              <div className={cn(
                "rounded-lg p-3 text-xs",
                selectedMode === "paper"
                  ? "bg-primary/5 border border-primary/20 text-primary"
                  : "bg-warning/5 border border-warning/20 text-warning"
              )}>
                {selectedMode === "paper" ? (
                  <p><span className="font-semibold">Paper Trading:</span> Real Angel One market data feed with simulated order execution. No real money at risk. Perfect for testing strategies.</p>
                ) : (
                  <p><span className="font-semibold">Live Trading:</span> Real orders with real money through Angel One. Requires additional confirmation step. Ensure risk limits are configured.</p>
                )}
              </div>

              <div className="grid gap-4 sm:grid-cols-2">
                <div>
                  <label htmlFor="broker-api-key" className="text-xs font-medium text-muted-foreground mb-1.5 block">API Key</label>
                  <input
                    id="broker-api-key"
                    type={showPasswords ? "text" : "password"}
                    value={credForm.api_key}
                    onChange={(e) => setCredForm({ ...credForm, api_key: e.target.value })}
                    placeholder="From SmartAPI developer portal"
                    required
                    autoComplete="off"
                    aria-required="true"
                    className="w-full rounded-lg border border-border/50 bg-muted/10 px-3 py-2 text-sm font-mono focus:border-primary/50 focus:outline-none"
                  />
                </div>
                <div>
                  <label htmlFor="broker-client-id" className="text-xs font-medium text-muted-foreground mb-1.5 block">Client Code</label>
                  <input
                    id="broker-client-id"
                    type="text"
                    value={credForm.client_id}
                    onChange={(e) => setCredForm({ ...credForm, client_id: e.target.value })}
                    placeholder="e.g. G58945700"
                    required
                    autoComplete="off"
                    aria-required="true"
                    className="w-full rounded-lg border border-border/50 bg-muted/10 px-3 py-2 text-sm font-mono focus:border-primary/50 focus:outline-none"
                  />
                </div>
                <div>
                  <label htmlFor="broker-password" className="text-xs font-medium text-muted-foreground mb-1.5 block">Password</label>
                  <input
                    id="broker-password"
                    type={showPasswords ? "text" : "password"}
                    value={credForm.password}
                    onChange={(e) => setCredForm({ ...credForm, password: e.target.value })}
                    placeholder="Trading password (PIN)"
                    required
                    autoComplete="off"
                    aria-required="true"
                    className="w-full rounded-lg border border-border/50 bg-muted/10 px-3 py-2 text-sm font-mono focus:border-primary/50 focus:outline-none"
                  />
                </div>
                <div>
                  <label htmlFor="broker-totp" className="text-xs font-medium text-muted-foreground mb-1.5 block">TOTP Secret</label>
                  <input
                    id="broker-totp"
                    type={showPasswords ? "text" : "password"}
                    value={credForm.totp_secret}
                    onChange={(e) => setCredForm({ ...credForm, totp_secret: e.target.value })}
                    placeholder="Base32 TOTP secret for auto-login"
                    required
                    autoComplete="off"
                    aria-required="true"
                    className="w-full rounded-lg border border-border/50 bg-muted/10 px-3 py-2 text-sm font-mono focus:border-primary/50 focus:outline-none"
                  />
                  <p className="text-[10px] text-muted-foreground mt-1">The Base32 secret from your authenticator app setup, NOT the 6-digit code</p>
                </div>
              </div>

              <div className="flex items-center gap-2">
                <button
                  type="button"
                  onClick={() => setShowPasswords(!showPasswords)}
                  aria-label={showPasswords ? "Hide credentials" : "Show credentials"}
                  aria-pressed={showPasswords}
                  className="flex items-center gap-1.5 text-xs text-muted-foreground hover:text-primary transition-colors"
                >
                  {showPasswords ? <EyeOff className="h-3.5 w-3.5" /> : <Eye className="h-3.5 w-3.5" />}
                  {showPasswords ? "Hide" : "Show"} credentials
                </button>
              </div>

              {credError && (
                <div className="rounded-lg bg-loss/10 border border-loss/30 p-3" role="alert">
                  <p className="text-xs text-loss font-medium">{credError}</p>
                </div>
              )}

              <div className="flex gap-3">
                <button
                  type="submit"
                  disabled={validateMutation.isPending || !credForm.api_key || !credForm.client_id || !credForm.password || !credForm.totp_secret}
                  aria-label={validateMutation.isPending ? "Validating credentials" : "Validate credentials with Angel One"}
                  aria-busy={validateMutation.isPending}
                  className={cn(
                    "flex-1 flex items-center justify-center gap-2 rounded-xl bg-primary py-2.5 text-sm font-semibold text-primary-foreground transition-all hover:bg-primary/90",
                    (validateMutation.isPending || !credForm.api_key || !credForm.client_id || !credForm.password || !credForm.totp_secret) && "opacity-50 cursor-not-allowed"
                  )}
                >
                  {validateMutation.isPending ? (
                    <RefreshCw className="h-4 w-4 animate-spin" />
                  ) : (
                    <ShieldCheck className="h-4 w-4" />
                  )}
                  {validateMutation.isPending
                    ? "Validating with Angel One..."
                    : selectedMode === "paper"
                      ? "Validate & Start Paper Trading"
                      : "Validate & Connect"
                  }
                </button>
              </div>

              <p className="text-[10px] text-muted-foreground text-center">
                Credentials are validated against Angel One API. A TOTP code is auto-generated from your secret.
                Nothing is stored on disk — credentials live in memory only for this session.
              </p>
            </form>
          </div>
        </motion.div>
      )}

      {/* Step 2: Live Mode Confirmation */}
      {showConfirm && (
        <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }}>
          <div className="glass-card p-6 border-2 border-warning/30">
            <div className="flex items-center gap-3 mb-4">
              <div className="flex h-12 w-12 items-center justify-center rounded-xl bg-warning/10">
                <AlertTriangle className="h-6 w-6 text-warning" />
              </div>
              <div>
                <h3 className="text-lg font-semibold">Confirm Live Trading</h3>
                <p className="text-xs text-muted-foreground mt-0.5">
                  You are about to switch from PAPER to LIVE mode
                </p>
              </div>
            </div>

            <div className="rounded-lg bg-warning/5 border border-warning/20 p-4 mb-4 space-y-2">
              <div className="flex items-center gap-2">
                <CheckCircle className="h-4 w-4 text-profit shrink-0" />
                <span className="text-sm">Credentials validated for client <span className="font-mono font-bold">{validatedClientId}</span></span>
              </div>
              <div className="flex items-center gap-2">
                <CheckCircle className="h-4 w-4 text-profit shrink-0" />
                <span className="text-sm">Angel One login successful — session token acquired</span>
              </div>
              <div className="flex items-center gap-2">
                <CheckCircle className="h-4 w-4 text-profit shrink-0" />
                <span className="text-sm">TOTP auto-generation working — token refresh enabled</span>
              </div>
            </div>

            <div className="rounded-lg bg-loss/5 border border-loss/20 p-4 mb-4">
              <p className="text-sm font-medium text-loss">
                Once confirmed, the system will place REAL orders with REAL money through Angel One.
                Make sure your risk limits are configured correctly.
              </p>
              <p className="text-xs text-muted-foreground mt-2">
                Confirmation expires in 5 minutes. You can disconnect at any time from the Broker panel.
              </p>
            </div>

            {credError && (
              <div className="rounded-lg bg-loss/10 border border-loss/30 p-3 mb-4" role="alert">
                <p className="text-xs text-loss font-medium">{credError}</p>
              </div>
            )}

            <div className="flex gap-3">
              <button
                onClick={handleCancelConfirm}
                className="flex-1 flex items-center justify-center gap-2 rounded-xl border border-border/50 py-3 text-sm font-medium text-muted-foreground transition-all hover:bg-muted/10"
              >
                Cancel — Stay on Paper
              </button>
              <button
                onClick={handleConfirmLive}
                disabled={confirmMutation.isPending}
                aria-label={confirmMutation.isPending ? "Switching to live mode" : "Confirm switch to live trading"}
                aria-busy={confirmMutation.isPending}
                className={cn(
                  "flex-1 flex items-center justify-center gap-2 rounded-xl bg-warning py-3 text-sm font-bold text-black transition-all hover:bg-warning/90",
                  confirmMutation.isPending && "opacity-50 cursor-not-allowed"
                )}
              >
                {confirmMutation.isPending ? (
                  <RefreshCw className="h-4 w-4 animate-spin" />
                ) : (
                  <Zap className="h-4 w-4" />
                )}
                {confirmMutation.isPending ? "Switching to Live..." : "Confirm — Switch to LIVE"}
              </button>
            </div>
          </div>
        </motion.div>
      )}

      {/* Success message */}
      {credSuccess && configStep === "idle" && (
        <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }}>
          <div className="rounded-lg bg-profit/10 border border-profit/30 p-4" role="status">
            <div className="flex items-center gap-2">
              <CheckCircle className="h-4 w-4 text-profit shrink-0" />
              <p className="text-sm text-profit font-medium">{credSuccess}</p>
            </div>
          </div>
        </motion.div>
      )}

      {/* ── Trading Controls: Two Independent Switches ── */}
      {isConnected && configStep === "idle" && (
        <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.18 }}>
          <div className="glass-card p-6">
            <div className="flex items-center gap-3 mb-5">
              <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-primary/10">
                <Activity className="h-5 w-5 text-primary" />
              </div>
              <div>
                <h3 className="text-lg font-semibold">Trading Controls</h3>
                <p className="text-xs text-muted-foreground mt-0.5">Control how your system trades</p>
              </div>
            </div>

            <div className="grid gap-4 sm:grid-cols-2">
              {/* Switch 1: Paper / Live */}
              <div className="rounded-xl border border-border/30 bg-muted/5 p-4">
                <div className="flex items-center justify-between mb-3">
                  <div>
                    <div className="text-sm font-semibold">Trading Mode</div>
                    <div className="text-[10px] text-muted-foreground mt-0.5">
                      {isLive ? "Real orders with real money" : "Simulated orders, real market data"}
                    </div>
                  </div>
                  <span className={cn(
                    "text-[10px] font-bold px-2.5 py-1 rounded-full",
                    isLive
                      ? "bg-warning/10 text-warning border border-warning/30"
                      : "bg-primary/10 text-primary border border-primary/30"
                  )}>
                    {isLive ? "LIVE" : "PAPER"}
                  </span>
                </div>
                <div className="flex items-center gap-1 p-1 rounded-xl bg-muted/20 border border-border/20">
                  <button
                    type="button"
                    onClick={() => {
                      if (!isLive) return;
                      endpoints.setTradingMode("paper").then(() => {
                        queryClient.invalidateQueries({ queryKey: ["broker-status"] });
                        queryClient.invalidateQueries({ queryKey: ["trading-mode"] });
                        dispatchToast("success", "Paper Mode", "Switched to paper trading — orders are simulated.");
                      }).catch((err: Error) => {
                        dispatchToast("error", "Switch Failed", err.message);
                      });
                    }}
                    className={cn(
                      "flex-1 flex items-center justify-center gap-1.5 rounded-lg py-2 text-xs font-semibold transition-all",
                      !isLive
                        ? "bg-primary text-primary-foreground shadow-sm"
                        : "text-muted-foreground hover:text-foreground hover:bg-muted/20"
                    )}
                  >
                    <Shield className="h-3.5 w-3.5" />
                    Paper
                  </button>
                  <button
                    type="button"
                    onClick={() => {
                      if (isLive) return;
                      if (!confirm("Switch to LIVE trading? Real orders will be placed with your broker using real money.")) return;
                      endpoints.setTradingMode("live").then((res) => {
                        queryClient.invalidateQueries({ queryKey: ["broker-status"] });
                        queryClient.invalidateQueries({ queryKey: ["trading-mode"] });
                        dispatchToast("success", "Live Mode", res.message);
                      }).catch((err: Error) => {
                        dispatchToast("error", "Switch Failed", err.message);
                      });
                    }}
                    className={cn(
                      "flex-1 flex items-center justify-center gap-1.5 rounded-lg py-2 text-xs font-semibold transition-all",
                      isLive
                        ? "bg-warning text-black shadow-sm"
                        : "text-muted-foreground hover:text-foreground hover:bg-muted/20"
                    )}
                  >
                    <Zap className="h-3.5 w-3.5" />
                    Live
                  </button>
                </div>
              </div>

              {/* Switch 2: Manual / Autonomous */}
              <div className="rounded-xl border border-border/30 bg-muted/5 p-4">
                <div className="flex items-center justify-between mb-3">
                  <div>
                    <div className="text-sm font-semibold">Execution Mode</div>
                    <div className="text-[10px] text-muted-foreground mt-0.5">
                      {autonomousRunning ? "AI generates signals & executes trades" : "Manual order placement only"}
                    </div>
                  </div>
                  <span className={cn(
                    "text-[10px] font-bold px-2.5 py-1 rounded-full",
                    autonomousRunning
                      ? "bg-profit/10 text-profit border border-profit/30"
                      : "bg-muted/20 text-muted-foreground border border-border/30"
                  )}>
                    {autonomousRunning ? "AUTO" : "MANUAL"}
                  </span>
                </div>
                <div className="flex items-center gap-1 p-1 rounded-xl bg-muted/20 border border-border/20">
                  <button
                    type="button"
                    onClick={() => {
                      if (!autonomousRunning) return;
                      autonomousMutation.mutate(false);
                      dispatchToast("success", "Manual Mode", "Autonomous trading stopped. Place orders manually.");
                    }}
                    disabled={autonomousMutation.isPending}
                    className={cn(
                      "flex-1 flex items-center justify-center gap-1.5 rounded-lg py-2 text-xs font-semibold transition-all",
                      !autonomousRunning
                        ? "bg-foreground/10 text-foreground shadow-sm"
                        : "text-muted-foreground hover:text-foreground hover:bg-muted/20",
                      autonomousMutation.isPending && "opacity-50 cursor-not-allowed"
                    )}
                  >
                    <Square className="h-3.5 w-3.5" />
                    Manual
                  </button>
                  <button
                    type="button"
                    onClick={() => {
                      if (autonomousRunning) return;
                      autonomousMutation.mutate(true);
                      dispatchToast("success", "Autonomous Mode", "AI trading started — signals will be generated automatically.");
                    }}
                    disabled={autonomousMutation.isPending || tradingModeData?.safe_mode}
                    className={cn(
                      "flex-1 flex items-center justify-center gap-1.5 rounded-lg py-2 text-xs font-semibold transition-all",
                      autonomousRunning
                        ? "bg-profit text-profit-foreground shadow-sm"
                        : "text-muted-foreground hover:text-foreground hover:bg-muted/20",
                      (autonomousMutation.isPending || tradingModeData?.safe_mode) && "opacity-50 cursor-not-allowed"
                    )}
                  >
                    <Play className="h-3.5 w-3.5" />
                    Autonomous
                  </button>
                </div>
                {tradingModeData?.safe_mode && (
                  <p className="text-[10px] text-loss mt-2">Safe mode active — cannot enable autonomous trading.</p>
                )}
              </div>
            </div>

            {/* Status summary */}
            <div className="mt-4 flex items-center gap-4 rounded-lg bg-muted/10 px-4 py-2.5 text-xs text-muted-foreground">
              <span className="flex items-center gap-1.5">
                <span className={cn("h-1.5 w-1.5 rounded-full", isLive ? "bg-warning" : "bg-primary")} />
                {isLive ? "Live" : "Paper"}
              </span>
              <span className="flex items-center gap-1.5">
                <span className={cn("h-1.5 w-1.5 rounded-full", autonomousRunning ? "bg-profit animate-pulse" : "bg-muted-foreground")} />
                {autonomousRunning ? "Autonomous" : "Manual"}
              </span>
              <span>{tickCount} ticks</span>
              <span>{openTrades} active trades</span>
              <span>Circuit: {tradingModeData?.circuit_open ? "OPEN" : "CLOSED"}</span>
            </div>
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
                <p className="text-xs text-muted-foreground mt-0.5">
                  {isLive ? "Angel One WebSocket (real-time)" : "YFinance fallback (1-min polling)"}
                </p>
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
              { label: "Last Tick", value: marketFeed.last_tick_ts ? new Date(marketFeed.last_tick_ts).toLocaleTimeString() : "---" },
              { label: "Source", value: isLive ? "Angel One WS" : "YFinance" },
              { label: "Feed Health", value: marketFeed.healthy ? "OK" : "---" },
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
