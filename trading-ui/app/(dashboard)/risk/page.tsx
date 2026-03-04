"use client";

import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { motion } from "framer-motion";
import { cn } from "@/lib/utils";
import { endpoints } from "@/lib/api/client";
import { useStore } from "@/store/useStore";
import { useState, useMemo } from "react";
import {
  PieChart, Pie, Cell, ResponsiveContainer,
} from "recharts";
import {
  ShieldAlert, AlertTriangle, Activity, Gauge,
  TrendingDown, Lock, Unlock, Save, BarChart3,
} from "lucide-react";

const SECTOR_COLORS = [
  "hsl(217, 91%, 60%)", "hsl(152, 69%, 53%)", "hsl(258, 90%, 66%)",
  "hsl(38, 92%, 50%)", "hsl(215, 20%, 45%)", "hsl(340, 80%, 55%)",
];

export default function RiskPage() {
  const queryClient = useQueryClient();
  const storePositions = useStore((s) => s.positions);

  const { data: riskState } = useQuery({
    queryKey: ["risk-state"],
    queryFn: () => endpoints.riskState(),
    refetchInterval: 5000,
  });

  // Fetch positions from API (more reliable than WebSocket-only store)
  const { data: apiPositions } = useQuery({
    queryKey: ["risk-positions"],
    queryFn: async () => {
      const res = await endpoints.positions();
      return ((res.positions ?? []) as any[]).map((p: any) => ({
        symbol: p.symbol,
        exchange: p.exchange,
        side: p.side,
        quantity: p.quantity,
        entry_price: p.entry_price ?? p.avg_price ?? 0,
        current_price: p.current_price ?? p.entry_price ?? p.avg_price ?? 0,
        unrealized_pnl: p.unrealized_pnl ?? 0,
      }));
    },
    refetchInterval: 5000,
  });

  // Use API positions (with mark-to-market) if available, fall back to store
  const positions = (apiPositions && apiPositions.length > 0) ? apiPositions : storePositions;

  const { data: limits } = useQuery({
    queryKey: ["risk-limits"],
    queryFn: () => endpoints.riskLimits(),
  });

  const [maxDailyLoss, setMaxDailyLoss] = useState<number | null>(null);
  const [maxPositions, setMaxPositions] = useState<number | null>(null);

  const updateMutation = useMutation({
    mutationFn: (body: { max_daily_loss_pct?: number; max_open_positions?: number }) =>
      endpoints.updateRiskLimits(body),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["risk-limits"] }),
  });

  const state = riskState ?? { circuit_open: false, daily_pnl: 0, open_positions: 0, var_95: 0, max_drawdown_pct: 0 };
  const lim = limits ?? { max_position_pct: 10, max_daily_loss_pct: 2, max_open_positions: 10 };

  const dailyLossDisplay = maxDailyLoss ?? lim.max_daily_loss_pct;
  const maxPosDisplay = maxPositions ?? lim.max_open_positions;

  // Compute sector & per-symbol exposure from real positions
  const { sectorData, symbolExposure } = useMemo(() => {
    if (!positions || positions.length === 0) {
      return {
        sectorData: [{ name: "No Positions", value: 100, color: "hsl(215, 20%, 30%)" }],
        symbolExposure: [] as Array<{ symbol: string; pct: number }>,
      };
    }

    // Approximate sector mapping for common NSE stocks
    const sectorMap: Record<string, string> = {
      RELIANCE: "Energy", ONGC: "Energy", BPCL: "Energy", IOC: "Energy",
      TCS: "IT", INFY: "IT", WIPRO: "IT", HCLTECH: "IT", TECHM: "IT", LTI: "IT",
      HDFCBANK: "Banking", ICICIBANK: "Banking", SBIN: "Banking", KOTAKBANK: "Banking", AXISBANK: "Banking",
      SUNPHARMA: "Pharma", DRREDDY: "Pharma", CIPLA: "Pharma", DIVISLAB: "Pharma",
      MARUTI: "Auto", TATAMOTORS: "Auto", M_M: "Auto", BAJAJ_AUTO: "Auto",
      HINDUNILVR: "FMCG", ITC: "FMCG", NESTLEIND: "FMCG",
    };

    const totalQty = positions.reduce((sum, p) => sum + Math.abs(p.quantity), 0);
    if (totalQty === 0) {
      return {
        sectorData: [{ name: "No Exposure", value: 100, color: "hsl(215, 20%, 30%)" }],
        symbolExposure: [],
      };
    }

    // Per-symbol exposure
    const symbolExp = positions.map((p) => ({
      symbol: p.symbol,
      pct: Math.round((Math.abs(p.quantity) / totalQty) * 100),
    })).sort((a, b) => b.pct - a.pct).slice(0, 8);

    // Sector aggregation
    const sectorTotals: Record<string, number> = {};
    for (const p of positions) {
      const sym = p.symbol.replace(/[-_]/g, "_");
      const sector = sectorMap[sym] || "Other";
      sectorTotals[sector] = (sectorTotals[sector] || 0) + Math.abs(p.quantity);
    }
    const sectors = Object.entries(sectorTotals)
      .map(([name, qty], i) => ({
        name,
        value: Math.round((qty / totalQty) * 100),
        color: SECTOR_COLORS[i % SECTOR_COLORS.length],
      }))
      .sort((a, b) => b.value - a.value);

    return { sectorData: sectors, symbolExposure: symbolExp };
  }, [positions]);

  return (
    <div className="space-y-6 pb-8">
      <motion.div initial={{ opacity: 0, y: -10 }} animate={{ opacity: 1, y: 0 }}>
        <h1 className="text-2xl font-bold tracking-tight">Risk Engine</h1>
        <p className="text-sm text-muted-foreground mt-0.5">Real-time risk monitoring & limit controls</p>
      </motion.div>

      {/* Circuit breaker status */}
      {state.circuit_open && (
        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="rounded-xl bg-loss/10 border border-loss/20 p-4 flex items-center gap-3">
          <AlertTriangle className="h-5 w-5 text-loss" />
          <div>
            <p className="text-sm font-semibold text-loss">Circuit Breaker OPEN</p>
            <p className="text-xs text-loss/70">Trading halted — daily loss limit exceeded</p>
          </div>
        </motion.div>
      )}

      {/* Risk KPIs */}
      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
        {[
          { label: "Max Daily Loss", value: `${lim.max_daily_loss_pct.toFixed(1)}%`, icon: ShieldAlert, color: "text-warning" },
          { label: "Current Daily Loss", value: `${state.daily_pnl.toFixed(2)}%`, icon: TrendingDown, color: state.daily_pnl < -1 ? "text-loss" : "text-foreground" },
          { label: "VaR (95%)", value: `${state.var_95.toFixed(2)}%`, icon: Gauge, color: "text-primary" },
          { label: "Circuit Breaker", value: state.circuit_open ? "OPEN" : "CLOSED", icon: state.circuit_open ? Lock : Unlock, color: state.circuit_open ? "text-loss" : "text-profit" },
        ].map((kpi, i) => (
          <motion.div key={kpi.label} initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: i * 0.05 }}>
            <div className="glass-card p-5">
              <div className="flex items-start justify-between">
                <div>
                  <div className="kpi-label">{kpi.label}</div>
                  <div className={cn("kpi-value mt-1", kpi.color)}>{kpi.value}</div>
                </div>
                <div className={cn("flex h-10 w-10 items-center justify-center rounded-xl bg-muted/40", kpi.color)}>
                  <kpi.icon className="h-5 w-5" />
                </div>
              </div>
            </div>
          </motion.div>
        ))}
      </div>

      {/* Charts row */}
      <div className="grid gap-6 xl:grid-cols-2">
        {/* Sector concentration */}
        <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.2 }}>
          <div className="glass-card p-5">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-sm font-semibold">Sector Concentration</h3>
              <BarChart3 className="h-4 w-4 text-primary" />
            </div>
            <div className="flex items-center gap-6">
              <div className="h-40 w-40">
                <ResponsiveContainer width="100%" height="100%">
                  <PieChart>
                    <Pie data={sectorData} cx="50%" cy="50%" innerRadius={35} outerRadius={60} paddingAngle={3} dataKey="value" strokeWidth={0}>
                      {sectorData.map((e, i) => <Cell key={i} fill={e.color} />)}
                    </Pie>
                  </PieChart>
                </ResponsiveContainer>
              </div>
              <div className="space-y-2 flex-1">
                {sectorData.map((s) => (
                  <div key={s.name} className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <span className="h-2.5 w-2.5 rounded-full" style={{ background: s.color }} />
                      <span className="text-xs text-muted-foreground">{s.name}</span>
                    </div>
                    <span className="font-mono text-xs font-semibold">{s.value}%</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </motion.div>

        {/* Per-symbol exposure */}
        <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.25 }}>
          <div className="glass-card p-5">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-sm font-semibold">Per-Symbol Exposure</h3>
              <Activity className="h-4 w-4 text-primary" />
            </div>
            <div className="space-y-3">
              {symbolExposure.length === 0 ? (
                <p className="text-xs text-muted-foreground text-center py-6">No open positions</p>
              ) : symbolExposure.map((item) => (
                <div key={item.symbol}>
                  <div className="flex items-center justify-between mb-1">
                    <span className="font-mono text-xs font-medium">{item.symbol}</span>
                    <span className="text-[10px] text-muted-foreground">{item.pct}%</span>
                  </div>
                  <div className="h-1.5 rounded-full bg-muted overflow-hidden">
                    <div
                      className={cn(
                        "h-full rounded-full transition-all",
                        item.pct >= 20 ? "bg-warning" : "bg-primary"
                      )}
                      style={{ width: `${Math.min(item.pct, 100)}%` }}
                    />
                  </div>
                </div>
              ))}
            </div>
          </div>
        </motion.div>
      </div>

      {/* Risk limits controls */}
      <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.3 }}>
        <div className="glass-card p-5">
          <div className="flex items-center justify-between mb-5">
            <div>
              <h3 className="text-sm font-semibold">Risk Limits</h3>
              <p className="text-xs text-muted-foreground mt-0.5">Adjust thresholds — changes take effect immediately</p>
            </div>
            <button
              onClick={() => {
                const body: Record<string, number> = {};
                if (maxDailyLoss !== null) body.max_daily_loss_pct = maxDailyLoss;
                if (maxPositions !== null) body.max_open_positions = maxPositions;
                if (Object.keys(body).length > 0) updateMutation.mutate(body);
              }}
              disabled={updateMutation.isPending || (maxDailyLoss === null && maxPositions === null)}
              className={cn(
                "flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-xs font-medium transition-all",
                (maxDailyLoss !== null || maxPositions !== null)
                  ? "bg-primary/10 text-primary border border-primary/30 hover:bg-primary/20"
                  : "text-muted-foreground opacity-50 cursor-not-allowed"
              )}
            >
              <Save className="h-3 w-3" />
              Save
            </button>
          </div>
          <div className="grid gap-6 md:grid-cols-2">
            <div>
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs font-medium text-muted-foreground">Max Daily Loss %</span>
                <span className="font-mono text-sm font-bold text-warning">{dailyLossDisplay.toFixed(1)}%</span>
              </div>
              <input
                type="range"
                min="0.5"
                max="10"
                step="0.5"
                value={dailyLossDisplay}
                onChange={(e) => setMaxDailyLoss(Number(e.target.value))}
                className="w-full h-2 rounded-full appearance-none cursor-pointer accent-primary bg-muted"
              />
              <div className="flex justify-between mt-1">
                <span className="text-[9px] text-muted-foreground">0.5%</span>
                <span className="text-[9px] text-muted-foreground">10%</span>
              </div>
            </div>
            <div>
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs font-medium text-muted-foreground">Max Open Positions</span>
                <span className="font-mono text-sm font-bold">{maxPosDisplay}</span>
              </div>
              <input
                type="range"
                min="1"
                max="50"
                step="1"
                value={maxPosDisplay}
                onChange={(e) => setMaxPositions(Number(e.target.value))}
                className="w-full h-2 rounded-full appearance-none cursor-pointer accent-primary bg-muted"
              />
              <div className="flex justify-between mt-1">
                <span className="text-[9px] text-muted-foreground">1</span>
                <span className="text-[9px] text-muted-foreground">50</span>
              </div>
            </div>
          </div>
        </div>
      </motion.div>
    </div>
  );
}
