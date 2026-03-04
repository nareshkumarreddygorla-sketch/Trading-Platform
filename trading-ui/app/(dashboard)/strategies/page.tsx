"use client";

import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { motion, AnimatePresence } from "framer-motion";
import { cn, formatCurrency, formatPercent } from "@/lib/utils";
import { endpoints } from "@/lib/api/client";
import type { Strategy } from "@/types";
import {
  Bot, TrendingUp, TrendingDown, Target, DollarSign,
  ToggleLeft, ToggleRight, Pencil, X, Save,
  Activity, BarChart3, Zap, Shield,
} from "lucide-react";

export default function StrategiesPage() {
  const queryClient = useQueryClient();
  const [editId, setEditId] = useState<string | null>(null);
  const [capital, setCapital] = useState("");

  const { data, isLoading } = useQuery({
    queryKey: ["strategies"],
    queryFn: async () => {
      const res = await endpoints.strategies();
      return (res.strategies ?? []) as Strategy[];
    },
  });

  const toggleMutation = useMutation({
    mutationFn: ({ id, enabled }: { id: string; enabled: boolean }) =>
      endpoints.toggleStrategy(id, enabled),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["strategies"] }),
  });

  const capitalMutation = useMutation({
    mutationFn: ({ id, capital }: { id: string; capital: number }) =>
      endpoints.updateCapital(id, capital),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["strategies"] });
      setEditId(null);
    },
  });

  const strategies = data ?? [];
  const activeCount = strategies.filter((s) => s.status === "active").length;
  const totalPnl = strategies.reduce((a, s) => a + (s.total_pnl ?? 0), 0);

  return (
    <div className="space-y-6 pb-8">
      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: -10 }}
        animate={{ opacity: 1, y: 0 }}
        className="flex items-center justify-between"
      >
        <div>
          <h1 className="text-2xl font-bold tracking-tight">Strategy Engine</h1>
          <p className="text-sm text-muted-foreground mt-0.5">
            {activeCount} of {strategies.length} strategies active
          </p>
        </div>
        <div className="flex items-center gap-3">
          <div className="glass-card px-4 py-2 flex items-center gap-2">
            <Zap className="h-4 w-4 text-primary" />
            <span className="text-xs font-medium text-muted-foreground">Total P&L</span>
            <span className={cn(
              "font-mono text-sm font-bold",
              totalPnl >= 0 ? "text-profit" : "text-loss"
            )}>
              {totalPnl >= 0 ? "+" : ""}{formatCurrency(totalPnl)}
            </span>
          </div>
        </div>
      </motion.div>

      {/* KPI row */}
      <div className="grid gap-4 md:grid-cols-4">
        {[
          { label: "Active", value: String(activeCount), icon: Activity, color: "text-profit" },
          { label: "Total Trades", value: String(strategies.reduce((a, s) => a + (s.total_trades ?? 0), 0)), icon: BarChart3, color: "text-primary" },
          { label: "Avg Win Rate", value: `${(strategies.reduce((a, s) => a + (s.win_rate ?? 0), 0) / Math.max(strategies.length, 1)).toFixed(0)}%`, icon: Target, color: "text-primary" },
          { label: "Total P&L", value: `${totalPnl >= 0 ? "+" : ""}${formatCurrency(totalPnl)}`, icon: DollarSign, color: totalPnl >= 0 ? "text-profit" : "text-loss" },
        ].map((kpi, i) => (
          <motion.div
            key={kpi.label}
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: i * 0.05 }}
          >
            <div className="glass-card p-4 flex items-center gap-3">
              <div className={cn("flex h-9 w-9 items-center justify-center rounded-xl bg-muted/50", kpi.color)}>
                <kpi.icon className="h-4 w-4" />
              </div>
              <div>
                <div className="kpi-label">{kpi.label}</div>
                <div className={cn("font-mono text-lg font-bold", kpi.color)}>{kpi.value}</div>
              </div>
            </div>
          </motion.div>
        ))}
      </div>

      {/* Strategy cards grid */}
      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
        {isLoading
          ? Array.from({ length: 6 }).map((_, i) => (
            <div key={i} className="glass-card p-5">
              <div className="shimmer h-6 w-32 rounded-lg mb-3" />
              <div className="shimmer h-4 w-24 rounded mb-2" />
              <div className="shimmer h-20 w-full rounded-lg" />
            </div>
          ))
          : strategies.map((s, i) => {
            const isActive = s.status === "active";
            const isEditing = editId === s.id;
            const pnl = s.total_pnl ?? 0;
            const winRate = s.win_rate ?? 0;

            return (
              <motion.div
                key={s.id}
                initial={{ opacity: 0, y: 12 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.1 + i * 0.05 }}
              >
                <div className={cn(
                  "glass-card p-5 transition-all duration-300",
                  isActive && "border-primary/20",
                  !isActive && "opacity-60 grayscale-[30%]"
                )}>
                  {/* Header with toggle */}
                  <div className="flex items-start justify-between mb-3">
                    <div className="flex items-center gap-2.5">
                      <div className={cn(
                        "flex h-9 w-9 items-center justify-center rounded-xl",
                        isActive ? "bg-primary/10 text-primary" : "bg-muted/50 text-muted-foreground"
                      )}>
                        <Bot className="h-4 w-4" />
                      </div>
                      <div>
                        <h3 className="text-sm font-semibold">{s.name || s.id}</h3>
                        {s.description && (
                          <p className="text-[10px] text-muted-foreground line-clamp-2" title={s.description}>{s.description}</p>
                        )}
                      </div>
                    </div>
                    <button
                      onClick={() => toggleMutation.mutate({ id: s.id, enabled: !isActive })}
                      disabled={toggleMutation.isPending}
                      className="transition-all"
                      title={isActive ? "Disable strategy" : "Enable strategy"}
                      aria-label={isActive ? `Disable ${s.name || s.id} strategy` : `Enable ${s.name || s.id} strategy`}
                      aria-pressed={isActive}
                    >
                      {isActive ? (
                        <ToggleRight className="h-6 w-6 text-profit" />
                      ) : (
                        <ToggleLeft className="h-6 w-6 text-muted-foreground" />
                      )}
                    </button>
                  </div>

                  {/* Status badge */}
                  <div className="mb-3">
                    {isActive ? (
                      <span className="status-badge-live">
                        <span className="h-1.5 w-1.5 rounded-full bg-profit animate-pulse" />
                        LIVE
                      </span>
                    ) : (
                      <span className="status-badge-warning">PAUSED</span>
                    )}
                  </div>

                  {/* Stats row */}
                  <div className="grid grid-cols-3 gap-2 mb-3">
                    <div className="rounded-lg bg-muted/30 p-2 text-center">
                      <div className="text-[10px] text-muted-foreground">Win Rate</div>
                      <div className={cn(
                        "font-mono text-sm font-bold",
                        winRate >= 55 ? "text-profit" : winRate >= 45 ? "text-foreground" : "text-loss"
                      )}>
                        {winRate.toFixed(0)}%
                      </div>
                    </div>
                    <div className="rounded-lg bg-muted/30 p-2 text-center">
                      <div className="text-[10px] text-muted-foreground">Trades</div>
                      <div className="font-mono text-sm font-bold">{s.total_trades ?? 0}</div>
                    </div>
                    <div className="rounded-lg bg-muted/30 p-2 text-center">
                      <div className="text-[10px] text-muted-foreground">P&L</div>
                      <div className={cn(
                        "font-mono text-sm font-bold",
                        pnl >= 0 ? "text-profit" : "text-loss"
                      )}>
                        {pnl >= 0 ? "+" : ""}{formatCurrency(pnl)}
                      </div>
                    </div>
                  </div>

                  {/* Win rate bar */}
                  <div className="mb-3">
                    <div className="flex items-center justify-between mb-1">
                      <span className="text-[10px] text-muted-foreground">Win Rate</span>
                      <span className="text-[10px] font-mono text-muted-foreground">{winRate.toFixed(0)}%</span>
                    </div>
                    <div className="h-1.5 w-full rounded-full bg-muted overflow-hidden">
                      <motion.div
                        initial={{ width: 0 }}
                        animate={{ width: `${Math.min(winRate, 100)}%` }}
                        transition={{ delay: 0.3 + i * 0.05, duration: 0.8, ease: "easeOut" }}
                        className={cn(
                          "h-full rounded-full",
                          winRate >= 60 ? "bg-profit" : winRate >= 50 ? "bg-primary" : winRate >= 40 ? "bg-warning" : "bg-loss"
                        )}
                      />
                    </div>
                  </div>

                  {/* Capital row */}
                  <div className="flex items-center justify-between pt-2 border-t border-border/30">
                    {isEditing ? (
                      <div className="flex items-center gap-2 flex-1">
                        <input
                          type="number"
                          value={capital}
                          onChange={(e) => setCapital(e.target.value)}
                          className="h-8 w-full rounded-lg border border-border/50 bg-muted/30 px-3 text-sm font-mono focus:border-primary/50 focus:outline-none focus:ring-1 focus:ring-primary/20"
                          autoFocus
                        />
                        <button
                          onClick={() => capitalMutation.mutate({ id: s.id, capital: Number(capital) })}
                          className="flex h-8 w-8 items-center justify-center rounded-lg bg-profit/10 text-profit hover:bg-profit/20 transition-colors"
                          aria-label="Save capital allocation"
                        >
                          <Save className="h-3.5 w-3.5" />
                        </button>
                        <button
                          onClick={() => setEditId(null)}
                          className="flex h-8 w-8 items-center justify-center rounded-lg bg-muted/50 text-muted-foreground hover:bg-muted transition-colors"
                          aria-label="Cancel editing capital"
                        >
                          <X className="h-3.5 w-3.5" />
                        </button>
                      </div>
                    ) : (
                      <>
                        <div>
                          <span className="text-[10px] text-muted-foreground">Capital</span>
                          <div className="font-mono text-sm font-semibold">
                            {formatCurrency(s.capital_allocated ?? 0)}
                          </div>
                        </div>
                        <button
                          onClick={() => {
                            setEditId(s.id);
                            setCapital(String(s.capital_allocated ?? 0));
                          }}
                          className="flex h-8 w-8 items-center justify-center rounded-lg text-muted-foreground hover:bg-muted hover:text-foreground transition-all"
                          aria-label={`Edit capital allocation for ${s.name || s.id}`}
                        >
                          <Pencil className="h-3.5 w-3.5" />
                        </button>
                      </>
                    )}
                  </div>
                </div>
              </motion.div>
            );
          })}
      </div>
    </div>
  );
}
