"use client";

import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { motion } from "framer-motion";
import { cn } from "@/lib/utils";
import { endpoints, clearAuthTokens } from "@/lib/api/client";
import { useStore } from "@/store/useStore";
import {
  User, Shield, Bell, Key,
  Save, LogOut, AlertTriangle,
} from "lucide-react";

export default function SettingsPage() {
  const queryClient = useQueryClient();
  const tradingMode = useStore((s) => s.tradingMode);
  const autonomyOn = useStore((s) => s.autonomyOn);

  // Risk limits
  const { data: limits } = useQuery({
    queryKey: ["risk-limits-settings"],
    queryFn: () => endpoints.riskLimits(),
  });

  const [maxDailyLoss, setMaxDailyLoss] = useState<number | null>(null);
  const [maxPositions, setMaxPositions] = useState<number | null>(null);
  const [maxPositionPct, setMaxPositionPct] = useState<number | null>(null);
  const [validationError, setValidationError] = useState<string | null>(null);
  const [saveSuccess, setSaveSuccess] = useState(false);

  const updateLimitsMutation = useMutation({
    mutationFn: (body: { max_daily_loss_pct?: number; max_open_positions?: number; max_position_pct?: number }) =>
      endpoints.updateRiskLimits(body),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["risk-limits-settings"] });
      setMaxDailyLoss(null);
      setMaxPositions(null);
      setMaxPositionPct(null);
      setValidationError(null);
      setSaveSuccess(true);
      setTimeout(() => setSaveSuccess(false), 3000);
    },
    onError: (err: Error) => {
      setValidationError(err.message);
      setSaveSuccess(false);
    },
  });

  const lim = limits ?? { max_position_pct: 10, max_daily_loss_pct: 2, max_open_positions: 10 };

  const handleLogout = () => {
    clearAuthTokens();
    window.location.href = "/login";
  };

  return (
    <div className="space-y-6 pb-8">
      <motion.div initial={{ opacity: 0, y: -10 }} animate={{ opacity: 1, y: 0 }}>
        <h1 className="text-2xl font-bold tracking-tight">Settings</h1>
        <p className="text-sm text-muted-foreground mt-0.5">Configure your trading platform</p>
      </motion.div>

      <div className="grid gap-6 max-w-3xl">
        {/* Profile */}
        <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.05 }}>
          <div className="glass-card p-6">
            <div className="flex items-center gap-3 mb-5">
              <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-primary/10 text-primary">
                <User className="h-5 w-5" />
              </div>
              <div>
                <h3 className="text-sm font-semibold">Profile</h3>
                <p className="text-xs text-muted-foreground">Account information</p>
              </div>
            </div>
            <div className="grid gap-4 sm:grid-cols-2">
              <div>
                <label className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground block mb-1">Username</label>
                <div className="h-9 rounded-lg border border-border/50 bg-muted/20 px-3 flex items-center text-sm font-mono text-muted-foreground">
                  admin
                </div>
              </div>
              <div>
                <label className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground block mb-1">Trading Mode</label>
                <div className={cn(
                  "h-9 rounded-lg border px-3 flex items-center text-sm font-bold",
                  tradingMode === "paper" ? "border-warning/30 bg-warning/10 text-warning" : "border-profit/30 bg-profit/10 text-profit"
                )}>
                  {tradingMode === "paper" ? "PAPER TRADING" : "LIVE TRADING"}
                </div>
              </div>
              <div>
                <label className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground block mb-1">Autonomous Mode</label>
                <div className={cn(
                  "h-9 rounded-lg border px-3 flex items-center text-sm font-bold",
                  autonomyOn ? "border-profit/30 bg-profit/10 text-profit" : "border-border/50 bg-muted/20 text-muted-foreground"
                )}>
                  {autonomyOn ? "ACTIVE" : "DISABLED"}
                </div>
              </div>
              <div>
                <label className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground block mb-1">Role</label>
                <div className="h-9 rounded-lg border border-border/50 bg-muted/20 px-3 flex items-center text-sm font-mono text-muted-foreground">
                  Administrator
                </div>
              </div>
            </div>
          </div>
        </motion.div>

        {/* Risk Configuration */}
        <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.1 }}>
          <div className="glass-card p-6">
            <div className="flex items-center justify-between mb-5">
              <div className="flex items-center gap-3">
                <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-warning/10 text-warning">
                  <Shield className="h-5 w-5" />
                </div>
                <div>
                  <h3 className="text-sm font-semibold">Risk Configuration</h3>
                  <p className="text-xs text-muted-foreground">Global risk limits</p>
                </div>
              </div>
              <button
                onClick={() => {
                  setValidationError(null);
                  setSaveSuccess(false);
                  const body: Record<string, number> = {};
                  if (maxDailyLoss !== null) body.max_daily_loss_pct = maxDailyLoss;
                  if (maxPositions !== null) body.max_open_positions = maxPositions;
                  if (maxPositionPct !== null) body.max_position_pct = maxPositionPct;
                  if (Object.keys(body).length === 0) return;
                  // Validate ranges before submitting
                  if (body.max_daily_loss_pct !== undefined && (body.max_daily_loss_pct < 0.5 || body.max_daily_loss_pct > 10)) {
                    setValidationError("Max Daily Loss must be between 0.5% and 10%");
                    return;
                  }
                  if (body.max_open_positions !== undefined && (body.max_open_positions < 1 || body.max_open_positions > 50)) {
                    setValidationError("Max Positions must be between 1 and 50");
                    return;
                  }
                  if (body.max_position_pct !== undefined && (body.max_position_pct < 1 || body.max_position_pct > 50)) {
                    setValidationError("Max Position % must be between 1% and 50%");
                    return;
                  }
                  updateLimitsMutation.mutate(body);
                }}
                disabled={updateLimitsMutation.isPending || (maxDailyLoss === null && maxPositions === null && maxPositionPct === null)}
                aria-label="Save risk configuration"
                aria-busy={updateLimitsMutation.isPending}
                className={cn(
                  "flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-xs font-medium transition-all",
                  (maxDailyLoss !== null || maxPositions !== null || maxPositionPct !== null)
                    ? "bg-primary/10 text-primary border border-primary/30 hover:bg-primary/20"
                    : "text-muted-foreground opacity-50 cursor-not-allowed"
                )}
              >
                <Save className="h-3 w-3" />
                Save
              </button>
            </div>
            <div className="grid gap-6 sm:grid-cols-3">
              <div>
                <div className="flex items-center justify-between mb-2">
                  <span className="text-xs font-medium text-muted-foreground">Max Daily Loss %</span>
                  <span className="font-mono text-sm font-bold text-warning">{(maxDailyLoss ?? lim.max_daily_loss_pct).toFixed(1)}%</span>
                </div>
                <input
                  type="range" min="0.5" max="10" step="0.5"
                  value={maxDailyLoss ?? lim.max_daily_loss_pct}
                  onChange={(e) => { setMaxDailyLoss(Number(e.target.value)); setValidationError(null); setSaveSuccess(false); }}
                  className="w-full h-2 rounded-full appearance-none cursor-pointer accent-warning bg-muted"
                />
              </div>
              <div>
                <div className="flex items-center justify-between mb-2">
                  <span className="text-xs font-medium text-muted-foreground">Max Positions</span>
                  <span className="font-mono text-sm font-bold">{maxPositions ?? lim.max_open_positions}</span>
                </div>
                <input
                  type="range" min="1" max="50" step="1"
                  value={maxPositions ?? lim.max_open_positions}
                  onChange={(e) => { setMaxPositions(Number(e.target.value)); setValidationError(null); setSaveSuccess(false); }}
                  className="w-full h-2 rounded-full appearance-none cursor-pointer accent-primary bg-muted"
                />
              </div>
              <div>
                <div className="flex items-center justify-between mb-2">
                  <span className="text-xs font-medium text-muted-foreground">Max Position %</span>
                  <span className="font-mono text-sm font-bold">{(maxPositionPct ?? lim.max_position_pct).toFixed(0)}%</span>
                </div>
                <input
                  type="range" min="1" max="50" step="1"
                  value={maxPositionPct ?? lim.max_position_pct}
                  onChange={(e) => { setMaxPositionPct(Number(e.target.value)); setValidationError(null); setSaveSuccess(false); }}
                  className="w-full h-2 rounded-full appearance-none cursor-pointer accent-primary bg-muted"
                />
              </div>
            </div>

            {/* Validation error */}
            {validationError && (
              <div className="rounded-lg bg-loss/10 border border-loss/30 p-3 mt-4" role="alert">
                <p className="text-xs text-loss font-medium">{validationError}</p>
              </div>
            )}

            {/* Mutation error */}
            {updateLimitsMutation.isError && !validationError && (
              <div className="rounded-lg bg-loss/10 border border-loss/30 p-3 mt-4" role="alert">
                <p className="text-xs text-loss font-medium">
                  Failed to save: {(updateLimitsMutation.error as Error)?.message || "Unknown error"}
                </p>
              </div>
            )}

            {/* Save success */}
            {saveSuccess && (
              <div className="rounded-lg bg-profit/10 border border-profit/30 p-3 mt-4 flex items-center gap-2" role="status">
                <svg className="h-4 w-4 text-profit shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                </svg>
                <p className="text-xs text-profit font-medium">Risk limits saved successfully</p>
              </div>
            )}
          </div>
        </motion.div>

        {/* Notifications */}
        <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.15 }}>
          <div className="glass-card p-6">
            <div className="flex items-center gap-3 mb-5">
              <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-primary/10 text-primary">
                <Bell className="h-5 w-5" />
              </div>
              <div>
                <h3 className="text-sm font-semibold">Notifications</h3>
                <p className="text-xs text-muted-foreground">Alert preferences</p>
              </div>
            </div>
            <div className="space-y-4">
              {[
                { label: "Trade Executions", description: "Notify when orders are filled", defaultOn: true },
                { label: "Risk Alerts", description: "Circuit breaker and exposure warnings", defaultOn: true },
                { label: "Strategy Updates", description: "Strategy enabled/disabled events", defaultOn: false },
                { label: "Market Regime Changes", description: "Regime shift notifications", defaultOn: true },
                { label: "Daily Summary", description: "End-of-day P&L summary", defaultOn: false },
              ].map((item) => (
                <div key={item.label} className="flex items-center justify-between rounded-lg bg-muted/20 p-3">
                  <div>
                    <span className="text-sm font-medium">{item.label}</span>
                    <p className="text-[10px] text-muted-foreground mt-0.5">{item.description}</p>
                  </div>
                  <div className="relative">
                    <input
                      type="checkbox"
                      defaultChecked={item.defaultOn}
                      className="sr-only peer"
                      id={`notif-${item.label}`}
                    />
                    <label
                      htmlFor={`notif-${item.label}`}
                      className="block h-6 w-11 cursor-pointer rounded-full bg-muted transition-colors peer-checked:bg-primary/80"
                    />
                    <span className="pointer-events-none absolute left-0.5 top-0.5 h-5 w-5 rounded-full bg-white shadow transition-transform peer-checked:translate-x-5" />
                  </div>
                </div>
              ))}
            </div>
          </div>
        </motion.div>

        {/* API Keys */}
        <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.2 }}>
          <div className="glass-card p-6">
            <div className="flex items-center gap-3 mb-5">
              <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-muted/50 text-muted-foreground">
                <Key className="h-5 w-5" />
              </div>
              <div>
                <h3 className="text-sm font-semibold">API Keys</h3>
                <p className="text-xs text-muted-foreground">Manage programmatic access keys</p>
              </div>
            </div>
            <div className="rounded-lg bg-muted/20 p-4 text-center">
              <Key className="h-8 w-8 mx-auto text-muted-foreground/30 mb-2" />
              <p className="text-xs text-muted-foreground">No API keys generated yet</p>
              <button className="mt-3 rounded-lg bg-primary/10 border border-primary/30 px-4 py-2 text-xs font-medium text-primary hover:bg-primary/20 transition-colors" aria-label="Generate a new API key">
                Generate API Key
              </button>
            </div>
          </div>
        </motion.div>

        {/* Danger Zone */}
        <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.25 }}>
          <div className="glass-card-loss p-6">
            <div className="flex items-center gap-3 mb-5">
              <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-loss/10 text-loss">
                <AlertTriangle className="h-5 w-5" />
              </div>
              <div>
                <h3 className="text-sm font-semibold text-loss">Danger Zone</h3>
                <p className="text-xs text-muted-foreground">Destructive actions</p>
              </div>
            </div>
            <button
              onClick={handleLogout}
              className="flex items-center gap-2 rounded-lg bg-loss/10 border border-loss/30 px-4 py-2.5 text-sm font-semibold text-loss hover:bg-loss/20 transition-colors"
              aria-label="Sign out of your account"
            >
              <LogOut className="h-4 w-4" />
              Sign Out
            </button>
          </div>
        </motion.div>
      </div>
    </div>
  );
}
