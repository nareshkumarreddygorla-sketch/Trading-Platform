"use client";

import { useQuery } from "@tanstack/react-query";
import { motion } from "framer-motion";
import { cn, formatCurrency, formatPercent } from "@/lib/utils";
import { endpoints } from "@/lib/api/client";
import {
  AreaChart, Area, BarChart, Bar, XAxis, YAxis,
  ResponsiveContainer, Tooltip, Cell,
} from "recharts";
import {
  TrendingUp, TrendingDown, Target, BarChart3,
  Activity, Shield, Award, Zap, DollarSign, Percent,
  ArrowUpRight,
} from "lucide-react";

function ChartTooltip({ active, payload, label }: any) {
  if (!active || !payload?.length) return null;
  const val = payload[0].value;
  return (
    <div className="chart-tooltip">
      <p className="text-[10px] text-muted-foreground font-medium">{label}</p>
      <p className="font-mono text-sm font-bold">
        {typeof val === "number" && val > 1000
          ? formatCurrency(val)
          : typeof val === "number"
            ? val.toFixed(2) + "%"
            : val}
      </p>
    </div>
  );
}

export default function PerformancePage() {
  const { data: summaryData, isLoading: summaryLoading } = useQuery({
    queryKey: ["performance-summary"],
    queryFn: () => endpoints.performanceSummary(26),
    refetchInterval: 60000,
  });

  const { data: equityData } = useQuery({
    queryKey: ["performance-equity"],
    queryFn: () => endpoints.performanceEquityCurve(26),
  });

  const { data: drawdownData } = useQuery({
    queryKey: ["performance-drawdown"],
    queryFn: () => endpoints.performanceDrawdown(26),
  });

  const { data: monthlyData } = useQuery({
    queryKey: ["performance-monthly"],
    queryFn: () => endpoints.performanceMonthlyReturns(26),
  });

  const { data: perfData } = useQuery({
    queryKey: ["strategies-performance"],
    queryFn: () => endpoints.strategiesPerformance(),
  });

  const summary = summaryData ?? {
    sharpe_ratio: 0, max_drawdown_pct: 0, total_pnl: 0, total_return_pct: 0,
    win_rate: 0, total_trades: 0, avg_trade_pnl: 0, profit_factor: 0,
    initial_equity: 100000, final_equity: 100000, weeks: 0,
  };

  const equityCurve = (equityData?.equity_curve ?? []).map((pt: any) => ({
    w: pt.label, v: pt.equity,
  }));

  const drawdown = (drawdownData?.drawdown ?? []).map((pt: any) => ({
    w: pt.label, v: pt.drawdown,
  }));

  const monthlyReturns = (monthlyData?.monthly_returns ?? []).map((pt: any) => ({
    m: pt.month, v: pt.return_pct,
  }));

  const kpiCards = [
    { label: "Win Rate", value: `${summary.win_rate.toFixed(1)}%`, icon: Target, color: "profit" as const, sub: `${summary.total_trades} trades` },
    { label: "Total P&L", value: `${summary.total_pnl >= 0 ? "+" : ""}${formatCurrency(summary.total_pnl)}`, icon: summary.total_pnl >= 0 ? TrendingUp : TrendingDown, color: (summary.total_pnl >= 0 ? "profit" : "loss") as "profit" | "loss", sub: `${summary.total_return_pct >= 0 ? "+" : ""}${summary.total_return_pct.toFixed(1)}% return` },
    { label: "Max Drawdown", value: `-${summary.max_drawdown_pct.toFixed(1)}%`, icon: Shield, color: "loss" as const, sub: "Peak-to-trough" },
    { label: "Sharpe Ratio", value: summary.sharpe_ratio.toFixed(2), icon: Award, color: (summary.sharpe_ratio >= 1 ? "profit" : "default") as "profit" | "default", sub: "Annualized (weekly)" },
  ];

  const secondaryKpis = [
    { label: "Avg Trade P&L", value: formatCurrency(summary.avg_trade_pnl), icon: DollarSign, color: summary.avg_trade_pnl >= 0 ? "text-profit" : "text-loss" },
    { label: "Profit Factor", value: summary.profit_factor.toFixed(2), icon: Zap, color: summary.profit_factor >= 1.5 ? "text-profit" : "text-warning" },
    { label: "Return", value: `${summary.total_return_pct >= 0 ? "+" : ""}${summary.total_return_pct.toFixed(2)}%`, icon: Percent, color: summary.total_return_pct >= 0 ? "text-profit" : "text-loss" },
  ];

  return (
    <div className="space-y-6 pb-8">
      {/* Header */}
      <motion.div initial={{ opacity: 0, y: -10 }} animate={{ opacity: 1, y: 0 }}>
        <div className="flex items-center gap-3">
          <h1 className="text-2xl font-bold tracking-tight">Performance</h1>
          <div className="flex items-center gap-1.5 glass-card px-3 py-1.5 rounded-xl">
            <ArrowUpRight className="h-3 w-3 text-profit" />
            <span className="text-xs font-mono font-semibold text-profit">+{summary.total_return_pct.toFixed(1)}%</span>
          </div>
        </div>
        <p className="text-sm text-muted-foreground mt-1">
          Portfolio analytics & strategy metrics
        </p>
      </motion.div>

      {/* Main KPI cards */}
      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
        {kpiCards.map((kpi, i) => {
          const cardClass = kpi.color === "profit" ? "glass-card-profit" :
            kpi.color === "loss" ? "glass-card-loss" : "glass-card";
          const textColor = kpi.color === "profit" ? "text-profit" :
            kpi.color === "loss" ? "text-loss" : "text-primary";
          return (
            <motion.div
              key={kpi.label}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: i * 0.06 }}
            >
              <div className={cn(cardClass, "p-5 group")}>
                <div className="flex items-start justify-between">
                  <div>
                    <div className="kpi-label">{kpi.label}</div>
                    <motion.div
                      key={kpi.value}
                      initial={{ opacity: 0, y: 8, filter: "blur(4px)" }}
                      animate={{ opacity: 1, y: 0, filter: "blur(0px)" }}
                      className={cn("kpi-value mt-1.5", textColor)}
                    >
                      {kpi.value}
                    </motion.div>
                    <div className="text-[10px] text-muted-foreground mt-1.5">{kpi.sub}</div>
                  </div>
                  <div className={cn(
                    "flex h-10 w-10 items-center justify-center rounded-xl transition-all duration-300 group-hover:scale-110",
                    kpi.color === "profit" ? "bg-profit/10 text-profit" :
                      kpi.color === "loss" ? "bg-loss/10 text-loss" :
                        "bg-primary/10 text-primary"
                  )}>
                    <kpi.icon className="h-5 w-5" />
                  </div>
                </div>
              </div>
            </motion.div>
          );
        })}
      </div>

      {/* Secondary metrics */}
      <div className="grid gap-4 md:grid-cols-3">
        {secondaryKpis.map((kpi, i) => (
          <motion.div
            key={kpi.label}
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.25 + i * 0.06 }}
          >
            <div className="glass-card p-4 group">
              <div className="flex items-center gap-3.5">
                <div className={cn("flex h-10 w-10 items-center justify-center rounded-xl bg-muted/30 transition-all duration-300 group-hover:scale-110", kpi.color)}>
                  <kpi.icon className="h-4.5 w-4.5" />
                </div>
                <div>
                  <div className="text-[10px] text-muted-foreground uppercase tracking-[0.1em] font-semibold">{kpi.label}</div>
                  <div className={cn("text-xl font-bold font-mono mt-0.5", kpi.color)}>{kpi.value}</div>
                </div>
              </div>
            </div>
          </motion.div>
        ))}
      </div>

      {/* Charts */}
      <div className="grid gap-6 xl:grid-cols-2">
        {/* Equity curve */}
        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.3 }}>
          <div className="glass-card p-6">
            <div className="flex items-center justify-between mb-5">
              <div>
                <h3 className="text-sm font-semibold">Equity Curve</h3>
                <p className="text-xs text-muted-foreground mt-0.5">Cumulative portfolio value</p>
              </div>
              <BarChart3 className="h-4 w-4 text-primary" />
            </div>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={equityCurve} margin={{ top: 5, right: 5, left: 5, bottom: 0 }}>
                  <defs>
                    <linearGradient id="perfEquityFill" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor="hsl(152, 69%, 53%)" stopOpacity={0.25} />
                      <stop offset="40%" stopColor="hsl(152, 69%, 53%)" stopOpacity={0.08} />
                      <stop offset="100%" stopColor="hsl(152, 69%, 53%)" stopOpacity={0} />
                    </linearGradient>
                    <linearGradient id="perfEquityStroke" x1="0" y1="0" x2="1" y2="0">
                      <stop offset="0%" stopColor="hsl(217, 91%, 60%)" />
                      <stop offset="50%" stopColor="hsl(152, 69%, 53%)" />
                      <stop offset="100%" stopColor="hsl(152, 69%, 65%)" />
                    </linearGradient>
                  </defs>
                  <XAxis dataKey="w" stroke="hsl(215, 20%, 25%)" fontSize={10} tickLine={false} axisLine={false} fontFamily="JetBrains Mono" />
                  <YAxis stroke="hsl(215, 20%, 25%)" fontSize={10} tickLine={false} axisLine={false} tickFormatter={(v) => `₹${(v / 1000).toFixed(0)}k`} fontFamily="JetBrains Mono" />
                  <Tooltip content={<ChartTooltip />} />
                  <Area
                    type="monotone"
                    dataKey="v"
                    stroke="url(#perfEquityStroke)"
                    fill="url(#perfEquityFill)"
                    strokeWidth={2.5}
                    dot={false}
                    activeDot={{ r: 5, strokeWidth: 2, fill: "hsl(152, 69%, 53%)", stroke: "hsl(222, 47%, 8%)" }}
                  />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </div>
        </motion.div>

        {/* Drawdown */}
        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.36 }}>
          <div className="glass-card p-6">
            <div className="flex items-center justify-between mb-5">
              <div>
                <h3 className="text-sm font-semibold">Drawdown</h3>
                <p className="text-xs text-muted-foreground mt-0.5">Max peak-to-trough decline</p>
              </div>
              <Shield className="h-4 w-4 text-loss" />
            </div>
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={drawdown} margin={{ top: 5, right: 5, left: 5, bottom: 0 }}>
                  <defs>
                    <linearGradient id="ddGradient" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor="hsl(0, 84%, 60%)" stopOpacity={0.2} />
                      <stop offset="100%" stopColor="hsl(0, 84%, 60%)" stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <XAxis dataKey="w" stroke="hsl(215, 20%, 25%)" fontSize={10} tickLine={false} axisLine={false} fontFamily="JetBrains Mono" />
                  <YAxis stroke="hsl(215, 20%, 25%)" fontSize={10} tickLine={false} axisLine={false} tickFormatter={(v) => `${v}%`} fontFamily="JetBrains Mono" />
                  <Tooltip content={<ChartTooltip />} />
                  <Area
                    type="monotone"
                    dataKey="v"
                    stroke="hsl(0, 84%, 60%)"
                    fill="url(#ddGradient)"
                    strokeWidth={2}
                    dot={false}
                    activeDot={{ r: 5, strokeWidth: 2, fill: "hsl(0, 84%, 60%)", stroke: "hsl(222, 47%, 8%)" }}
                  />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </div>
        </motion.div>
      </div>

      {/* Monthly returns */}
      <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.42 }}>
        <div className="glass-card p-6">
          <div className="flex items-center justify-between mb-5">
            <div>
              <h3 className="text-sm font-semibold">Monthly Returns</h3>
              <p className="text-xs text-muted-foreground mt-0.5">Month-over-month performance</p>
            </div>
            <Activity className="h-4 w-4 text-primary" />
          </div>
          <div className="h-56">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={monthlyReturns} margin={{ top: 5, right: 5, left: 5, bottom: 0 }}>
                <XAxis dataKey="m" stroke="hsl(215, 20%, 25%)" fontSize={10} tickLine={false} axisLine={false} fontFamily="JetBrains Mono" />
                <YAxis stroke="hsl(215, 20%, 25%)" fontSize={10} tickLine={false} axisLine={false} tickFormatter={(v) => `${v}%`} fontFamily="JetBrains Mono" />
                <Tooltip content={<ChartTooltip />} />
                <Bar dataKey="v" radius={[6, 6, 2, 2]} maxBarSize={36}>
                  {monthlyReturns.map((entry: any, i: number) => (
                    <Cell
                      key={i}
                      fill={entry.v >= 0 ? "hsl(152, 69%, 53%)" : "hsl(0, 84%, 60%)"}
                      fillOpacity={0.75}
                    />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </motion.div>
    </div>
  );
}
