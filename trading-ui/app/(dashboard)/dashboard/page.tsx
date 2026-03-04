"use client";

import { useState, useEffect } from "react";
import { useQuery } from "@tanstack/react-query";
import { motion } from "framer-motion";
import { useStore } from "@/store/useStore";
import { formatCurrency, formatPercent } from "@/lib/utils";
import { endpoints } from "@/lib/api/client";
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  ResponsiveContainer,
  Tooltip,
  PieChart,
  Pie,
  Cell,
} from "recharts";
import {
  TrendingUp,
  TrendingDown,
  DollarSign,
  Target,
  Brain,
  Activity,
  Zap,
  ArrowUpRight,
  ArrowDownRight,
  BarChart3,
  Shield,
  Clock,
  Sparkles,
  CircleDot,
} from "lucide-react";
import { cn } from "@/lib/utils";
import dynamic from "next/dynamic";
import Link from "next/link";

const TrainAIPanel = dynamic(() => import("@/components/dashboard/TrainAIPanel"), { ssr: false });
const AgentStatus = dynamic(() => import("@/components/dashboard/AgentStatus"), { ssr: false });
const SignalFeed = dynamic(() => import("@/components/dashboard/SignalFeed"), { ssr: false });
const RegimeIndicator = dynamic(() => import("@/components/dashboard/RegimeIndicator"), { ssr: false });
const RiskAlerts = dynamic(() => import("@/components/dashboard/RiskAlerts"), { ssr: false });

/* ── Strategy color map ── */
const STRATEGY_COLORS: Record<string, string> = {
  ai_alpha: "hsl(258,90%,66%)",
  momentum_breakout: "hsl(152,69%,53%)",
  ema_crossover: "hsl(217,91%,60%)",
  macd: "hsl(199,89%,48%)",
  mean_reversion: "hsl(38,92%,50%)",
  rsi: "hsl(0,84%,60%)",
  ml_predictor: "hsl(280,80%,60%)",
  rl_agent: "hsl(330,80%,55%)",
};
const DEFAULT_COLOR = "hsl(215,20%,50%)";

/* ── Mini sparkline for KPI cards ── */
function MiniSparkline({ data, color = "hsl(152,69%,53%)" }: { data: number[]; color?: string }) {
  const max = Math.max(...data);
  const min = Math.min(...data);
  const range = max - min || 1;
  const points = data
    .map((v, i) => `${(i / (data.length - 1)) * 60},${28 - ((v - min) / range) * 24}`)
    .join(" ");
  const areaPoints = `0,28 ${points} 60,28`;

  return (
    <svg viewBox="0 0 60 28" className="w-16 h-7 opacity-60">
      <defs>
        <linearGradient id={`sparkGrad-${color.replace(/[(),\s%]/g, "")}`} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor={color} stopOpacity="0.3" />
          <stop offset="100%" stopColor={color} stopOpacity="0" />
        </linearGradient>
      </defs>
      <polygon
        points={areaPoints}
        fill={`url(#sparkGrad-${color.replace(/[(),\s%]/g, "")})`}
      />
      <polyline
        points={points}
        fill="none"
        stroke={color}
        strokeWidth="1.5"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      {/* Live dot on last point */}
      <circle
        cx="60"
        cy={28 - ((data[data.length - 1] - min) / range) * 24}
        r="2"
        fill={color}
      >
        <animate attributeName="r" values="2;3;2" dur="2s" repeatCount="indefinite" />
        <animate attributeName="opacity" values="1;0.6;1" dur="2s" repeatCount="indefinite" />
      </circle>
    </svg>
  );
}

/* ── Chart tooltip ── */
function CustomTooltip({ active, payload, label }: any) {
  if (!active || !payload?.length) return null;
  return (
    <div className="chart-tooltip">
      <p className="text-[10px] text-muted-foreground font-medium">{label}</p>
      <p className="font-mono text-sm font-bold text-foreground">
        {formatCurrency(payload[0].value)}
      </p>
    </div>
  );
}

/* ── Animated counter ── */
function AnimatedValue({ value, prefix = "", suffix = "", className = "" }: {
  value: string; prefix?: string; suffix?: string; className?: string;
}) {
  return (
    <motion.span
      key={value}
      initial={{ opacity: 0, y: 10, filter: "blur(4px)" }}
      animate={{ opacity: 1, y: 0, filter: "blur(0px)" }}
      transition={{ duration: 0.5, ease: [0.16, 1, 0.3, 1] }}
      className={className}
    >
      {prefix}{value}{suffix}
    </motion.span>
  );
}

/* ── KPI Card ── */
function KpiCard({ icon: Icon, label, value, subValue, trend, loading, delay = 0, variant = "default", sparkData }: {
  icon: any; label: string; value: string; subValue?: string;
  trend?: "up" | "down" | "neutral"; loading?: boolean; delay?: number;
  variant?: "default" | "profit" | "loss"; sparkData?: number[];
}) {
  const cardClass = variant === "profit" ? "glass-card-profit" : variant === "loss" ? "glass-card-loss" : "glass-card";
  const sparkColor = variant === "profit" ? "hsl(152,69%,53%)" : variant === "loss" ? "hsl(0,84%,60%)" : "hsl(217,91%,60%)";

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay, duration: 0.6, ease: [0.16, 1, 0.3, 1] }}
    >
      <div className={cn(cardClass, "p-5 group")}>
        <div className="flex items-start justify-between">
          <div className="flex flex-col gap-1.5">
            <span className="kpi-label">{label}</span>
            {loading ? (
              <div className="shimmer h-8 w-28 rounded-lg" />
            ) : (
              <AnimatedValue
                value={value}
                className={cn(
                  "kpi-value",
                  variant === "profit" && "text-gradient-profit",
                  variant === "loss" && "text-gradient-loss"
                )}
              />
            )}
            {subValue && (
              <div className="flex items-center gap-1.5 mt-0.5">
                {trend === "up" && <ArrowUpRight className="h-3 w-3 text-profit" />}
                {trend === "down" && <ArrowDownRight className="h-3 w-3 text-loss" />}
                <span className={cn(
                  "text-xs font-medium",
                  trend === "up" ? "text-profit" : trend === "down" ? "text-loss" : "text-muted-foreground"
                )}>
                  {subValue}
                </span>
              </div>
            )}
          </div>
          <div className="flex flex-col items-end gap-2">
            <div className={cn(
              "flex h-10 w-10 items-center justify-center rounded-xl transition-all duration-300 group-hover:scale-110",
              variant === "profit" && "bg-profit/10 text-profit",
              variant === "loss" && "bg-loss/10 text-loss",
              variant === "default" && "bg-primary/10 text-primary"
            )}>
              <Icon className="h-5 w-5" />
            </div>
            {sparkData && <MiniSparkline data={sparkData} color={sparkColor} />}
          </div>
        </div>
      </div>
    </motion.div>
  );
}

/* ── Main Dashboard ── */
const PERIOD_WEEKS: Record<string, number> = { "1D": 1, "1W": 1, "1M": 4, "3M": 13 };

export default function DashboardPage() {
  const setDashboard = useStore((s) => s.setDashboard);
  const [equityPeriod, setEquityPeriod] = useState("1M");
  const [currentTime, setCurrentTime] = useState("");

  // B6: Ticking clock
  useEffect(() => {
    const tick = () => setCurrentTime(new Date().toLocaleTimeString("en-IN", { hour: "2-digit", minute: "2-digit", second: "2-digit" }));
    tick();
    const interval = setInterval(tick, 1000);
    return () => clearInterval(interval);
  }, []);

  // Real-time risk data
  const { data: risk, isLoading } = useQuery({
    queryKey: ["risk"],
    queryFn: () => endpoints.risk(),
    refetchInterval: 5000,
  });

  // Performance summary (real trade stats)
  const { data: perfSummary } = useQuery({
    queryKey: ["performanceSummary"],
    queryFn: () => endpoints.performanceSummary(4),
    refetchInterval: 15000,
  });

  // Equity curve from API — responds to period selection
  const { data: equityCurveData } = useQuery({
    queryKey: ["equityCurve", equityPeriod],
    queryFn: () => endpoints.performanceEquityCurve(PERIOD_WEEKS[equityPeriod] ?? 4),
    refetchInterval: 30000,
  });

  // Strategy performance
  const { data: stratData } = useQuery({
    queryKey: ["strategiesPerformance"],
    queryFn: () => endpoints.strategiesPerformance(),
    refetchInterval: 10000,
  });

  const equity = risk?.equity ?? 100000;
  const dailyPnl = risk?.daily_pnl ?? 0;
  const positions = (risk?.positions as unknown[]) ?? [];
  const isProfit = dailyPnl >= 0;

  // Map API equity curve to chart format
  const equityData = (equityCurveData?.equity_curve ?? []).map((pt: { date: string; label: string; equity: number }) => ({
    t: pt.label || pt.date,
    v: pt.equity,
  }));

  // Map strategy performance from API
  const strategyPerf = ((stratData?.strategies ?? []) as Array<{
    id: string; name?: string; status: string;
    total_trades?: number; win_rate?: number; total_pnl?: number;
  }>).map((s) => ({
    name: s.name || s.id,
    winRate: Math.round(s.win_rate ?? 0),
    trades: s.total_trades ?? 0,
    pnl: s.total_pnl ?? 0,
    active: s.status === "active",
    color: STRATEGY_COLORS[s.id] || DEFAULT_COLOR,
  }));

  // Recent signals from WebSocket events (stored in zustand)
  const wsSignals = useStore((s) => s.recentSignals ?? []);
  const recentTrades = wsSignals.slice(0, 6).map((sig) => ({
    symbol: sig.symbol || "—",
    side: sig.direction?.includes("BUY") ? "BUY" : sig.direction?.includes("SELL") ? "SELL" : sig.direction || "—",
    pnl: 0,
    time: sig.timestamp ? new Date(sig.timestamp).toLocaleTimeString("en-IN", { hour: "2-digit", minute: "2-digit" }) : "—",
    strategy: sig.source || "—",
  }));

  // Stats from real performance API
  const totalTrades = perfSummary?.total_trades ?? 0;
  const winRate = perfSummary?.win_rate ?? 0;
  const sharpeRatio = perfSummary?.sharpe_ratio ?? 0;
  const maxDrawdown = perfSummary?.max_drawdown_pct ?? 0;
  const totalPnl = perfSummary?.total_pnl ?? dailyPnl;
  const activeStrategies = strategyPerf.filter((s) => s.active).length || 0;
  const totalReturn = equity > 0 ? ((dailyPnl / equity) * 100) : 0;

  // Market regime from WebSocket
  const currentRegime = useStore((s) => s.currentRegime);
  const regimeData = [
    { name: "Bullish", value: currentRegime === "trending_up" ? 60 : 30, color: "hsl(152, 69%, 53%)" },
    { name: "Sideways", value: currentRegime === "low_volatility" ? 60 : 40, color: "hsl(217, 91%, 60%)" },
    { name: "Bearish", value: currentRegime === "trending_down" || currentRegime === "crisis" ? 60 : 15, color: "hsl(0, 84%, 60%)" },
  ];
  const regimeLabel = currentRegime === "trending_up" ? "BULLISH"
    : currentRegime === "trending_down" ? "BEARISH"
    : currentRegime === "crisis" ? "CRISIS"
    : currentRegime === "high_volatility" ? "VOLATILE"
    : currentRegime === "low_volatility" ? "SIDEWAYS"
    : "SCANNING";
  const regimeSymbol = regimeLabel === "BULLISH" ? "↑" : regimeLabel === "BEARISH" ? "↓" : "—";
  const regimeLabelColor = regimeLabel === "BULLISH" ? "text-profit" : regimeLabel === "BEARISH" || regimeLabel === "CRISIS" ? "text-loss" : "text-primary";

  if (risk) {
    setDashboard({ equity, daily_pnl: dailyPnl, open_positions_count: positions.length });
  }

  return (
    <div className="space-y-6 pb-8">
      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: -10 }}
        animate={{ opacity: 1, y: 0 }}
        className="flex flex-col sm:flex-row sm:items-center justify-between gap-3"
      >
        <div>
          <div className="flex items-center gap-3">
            <h1 className="text-xl sm:text-2xl font-bold tracking-tight">
              Trading Dashboard
            </h1>
            <div className="status-badge-live">
              <span className="h-1.5 w-1.5 rounded-full bg-profit animate-pulse" />
              LIVE
            </div>
          </div>
          <p className="text-xs sm:text-sm text-muted-foreground mt-1">
            Real-time portfolio monitoring & AI strategy performance
          </p>
        </div>
        <div className="flex items-center gap-2.5 glass-card px-3.5 py-2 rounded-xl self-start sm:self-auto">
          <Clock className="h-3.5 w-3.5 text-muted-foreground" />
          <span className="text-xs text-muted-foreground font-mono tracking-wide" suppressHydrationWarning>
            {currentTime || "--:--:--"} IST
          </span>
        </div>
      </motion.div>

      {/* KPI Cards */}
      <div className="grid gap-3 sm:gap-4 grid-cols-1 sm:grid-cols-2 xl:grid-cols-4">
        <KpiCard
          icon={DollarSign}
          label="Total Equity"
          value={formatCurrency(equity)}
          subValue={`${totalReturn >= 0 ? "+" : ""}${totalReturn.toFixed(2)}% today`}
          trend={totalReturn >= 0 ? "up" : "down"}
          loading={isLoading}
          delay={0}
          sparkData={equityData.length > 2 ? equityData.slice(-10).map((d: { v: number }) => d.v) : undefined}
        />
        <KpiCard
          icon={isProfit ? TrendingUp : TrendingDown}
          label="Daily P&L"
          value={formatCurrency(Math.abs(dailyPnl))}
          subValue={isProfit ? "Profit" : "Loss"}
          trend={isProfit ? "up" : "down"}
          loading={isLoading}
          delay={0.06}
          variant={isProfit ? "profit" : "loss"}
        />
        <KpiCard
          icon={Target}
          label="Win Rate"
          value={totalTrades > 0 ? `${Math.round(winRate)}%` : "—"}
          subValue={totalTrades > 0 ? `${totalTrades} trades` : "No trades yet"}
          trend={winRate > 50 ? "up" : winRate > 0 ? "down" : "neutral"}
          delay={0.12}
        />
        <KpiCard
          icon={Brain}
          label="Active Strategies"
          value={String(activeStrategies)}
          subValue={`of ${strategyPerf.length} total`}
          trend="neutral"
          delay={0.18}
        />
      </div>

      {/* Charts Row */}
      <div className="grid gap-4 sm:gap-6 grid-cols-1 lg:grid-cols-3">
        {/* Equity Curve */}
        <motion.div
          className="lg:col-span-2"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
        >
          <div className="glass-card p-4 sm:p-6">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 mb-4 sm:mb-5">
              <div>
                <div className="flex items-center gap-2.5">
                  <h3 className="text-sm font-semibold">Equity Curve</h3>
                  <div className="flex items-center gap-1">
                    <span className="relative flex h-2 w-2">
                      <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-profit opacity-75" />
                      <span className="relative inline-flex h-2 w-2 rounded-full bg-profit" />
                    </span>
                  </div>
                </div>
                <p className="text-xs text-muted-foreground mt-0.5">Intraday portfolio value</p>
              </div>
              <div className="flex items-center gap-1.5 bg-muted/30 rounded-xl p-1">
                {["1D", "1W", "1M", "3M"].map((period) => (
                  <button
                    key={period}
                    onClick={() => setEquityPeriod(period)}
                    className={cn(
                      "rounded-lg px-3 py-1.5 text-[11px] font-semibold transition-all duration-200",
                      equityPeriod === period
                        ? "bg-primary/15 text-primary shadow-sm shadow-primary/10"
                        : "text-muted-foreground hover:text-foreground"
                    )}
                  >
                    {period}
                  </button>
                ))}
              </div>
            </div>
            <div className="h-48 sm:h-64 lg:h-72">
              {equityData.length === 0 ? (
                <div className="flex items-center justify-center h-full text-muted-foreground text-xs">
                  Equity curve will appear after first trades
                </div>
              ) : (
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={equityData} margin={{ top: 5, right: 5, left: 5, bottom: 0 }}>
                  <defs>
                    <linearGradient id="equityFill" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor="hsl(152, 69%, 53%)" stopOpacity={0.25} />
                      <stop offset="40%" stopColor="hsl(152, 69%, 53%)" stopOpacity={0.1} />
                      <stop offset="100%" stopColor="hsl(152, 69%, 53%)" stopOpacity={0} />
                    </linearGradient>
                    <linearGradient id="equityStroke" x1="0" y1="0" x2="1" y2="0">
                      <stop offset="0%" stopColor="hsl(217, 91%, 60%)" />
                      <stop offset="40%" stopColor="hsl(152, 69%, 53%)" />
                      <stop offset="100%" stopColor="hsl(152, 69%, 70%)" />
                    </linearGradient>
                    <filter id="glow">
                      <feGaussianBlur stdDeviation="3" result="coloredBlur" />
                      <feMerge>
                        <feMergeNode in="coloredBlur" />
                        <feMergeNode in="SourceGraphic" />
                      </feMerge>
                    </filter>
                  </defs>
                  <XAxis
                    dataKey="t"
                    stroke="hsl(215, 20%, 25%)"
                    fontSize={10}
                    tickLine={false}
                    axisLine={false}
                    fontFamily="JetBrains Mono"
                  />
                  <YAxis
                    stroke="hsl(215, 20%, 25%)"
                    fontSize={10}
                    tickLine={false}
                    axisLine={false}
                    tickFormatter={(v) => `₹${(v / 1000).toFixed(0)}k`}
                    domain={["auto", "auto"]}
                    fontFamily="JetBrains Mono"
                  />
                  <Tooltip content={<CustomTooltip />} />
                  <Area
                    type="monotone"
                    dataKey="v"
                    stroke="url(#equityStroke)"
                    fill="url(#equityFill)"
                    strokeWidth={2.5}
                    dot={false}
                    activeDot={{
                      r: 6,
                      strokeWidth: 3,
                      fill: "hsl(152, 69%, 53%)",
                      stroke: "hsl(222, 47%, 8%)",
                      filter: "url(#glow)",
                    }}
                  />
                </AreaChart>
              </ResponsiveContainer>
              )}
            </div>
          </div>
        </motion.div>

        {/* Market Regime + Quick Stats */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.28 }}
          className="flex flex-col gap-5"
        >
          {/* Regime */}
          <div className="glass-card p-5">
            <div className="flex items-center gap-2 mb-3">
              <Sparkles className="h-4 w-4 text-primary" />
              <h3 className="text-sm font-semibold">Market Regime</h3>
            </div>
            <div className="flex items-center justify-center py-2">
              <div className="relative h-32 w-32">
                <ResponsiveContainer width="100%" height="100%">
                  <PieChart>
                    <Pie
                      data={regimeData}
                      cx="50%"
                      cy="50%"
                      innerRadius={40}
                      outerRadius={58}
                      paddingAngle={4}
                      dataKey="value"
                      strokeWidth={0}
                    >
                      {regimeData.map((entry, index) => (
                        <Cell key={index} fill={entry.color} />
                      ))}
                    </Pie>
                  </PieChart>
                </ResponsiveContainer>
                <div className="absolute inset-0 flex flex-col items-center justify-center">
                  <motion.span
                    className={cn("text-xl font-bold", regimeLabelColor)}
                    animate={{ scale: [1, 1.1, 1] }}
                    transition={{ duration: 3, repeat: Infinity }}
                  >
                    {regimeSymbol}
                  </motion.span>
                  <span className="text-[9px] font-bold tracking-widest text-muted-foreground">{regimeLabel}</span>
                </div>
              </div>
            </div>
            <div className="flex justify-center gap-4 mt-1">
              {regimeData.map((item) => (
                <div key={item.name} className="flex items-center gap-1.5">
                  <span className="h-2 w-2 rounded-full shadow-sm" style={{ background: item.color, boxShadow: `0 0 6px ${item.color}40` }} />
                  <span className="text-[10px] text-muted-foreground">{item.name} {item.value}%</span>
                </div>
              ))}
            </div>
          </div>

          {/* Quick Stats */}
          <div className="glass-card p-5 flex-1">
            <div className="flex items-center gap-2 mb-4">
              <BarChart3 className="h-4 w-4 text-primary" />
              <h3 className="text-sm font-semibold">Quick Stats</h3>
            </div>
            <div className="space-y-3.5">
              {[
                { label: "Sharpe Ratio", value: sharpeRatio > 0 ? sharpeRatio.toFixed(2) : "—", icon: BarChart3, color: sharpeRatio > 0 ? "text-profit" : "text-muted-foreground" },
                { label: "Max Drawdown", value: maxDrawdown > 0 ? `-${maxDrawdown.toFixed(1)}%` : "—", icon: Shield, color: maxDrawdown > 3 ? "text-loss" : "text-muted-foreground" },
                { label: "Total Trades", value: String(totalTrades), icon: Activity, color: "text-primary" },
                { label: "Open Positions", value: String(positions.length), icon: Target, color: "text-foreground" },
              ].map((stat) => (
                <div key={stat.label} className="flex items-center justify-between group">
                  <div className="flex items-center gap-2.5">
                    <stat.icon className="h-3.5 w-3.5 text-muted-foreground group-hover:text-primary transition-colors" />
                    <span className="text-xs text-muted-foreground">{stat.label}</span>
                  </div>
                  <span className={cn("font-mono text-xs font-bold", stat.color)}>
                    {stat.value}
                  </span>
                </div>
              ))}
            </div>
          </div>
        </motion.div>
      </div>

      {/* Strategy Performance + Recent Trades */}
      <div className="grid gap-4 sm:gap-6 grid-cols-1 md:grid-cols-2">
        {/* Strategy Performance */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.32 }}
        >
          <div className="glass-card p-4 sm:p-6">
            <div className="flex items-center justify-between mb-4 sm:mb-5">
              <div>
                <h3 className="text-sm font-semibold">Strategy Performance</h3>
                <p className="text-xs text-muted-foreground mt-0.5">{strategyPerf.length} strategies ranked by P&L</p>
              </div>
              <Zap className="h-4 w-4 text-primary" />
            </div>
            <div className="space-y-2">
              {strategyPerf.length === 0 && (
                <div className="text-center py-8 text-muted-foreground text-xs">
                  Waiting for strategy data...
                </div>
              )}
              {[...strategyPerf]
                .sort((a, b) => b.pnl - a.pnl)
                .map((s, i) => (
                  <motion.div
                    key={s.name}
                    initial={{ opacity: 0, x: -12 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: 0.35 + i * 0.06 }}
                    className="flex items-center gap-3 rounded-xl bg-muted/20 p-3.5 transition-all duration-200 hover:bg-muted/35 group cursor-default"
                  >
                    {/* Rank with color accent */}
                    <div
                      className="flex h-8 w-8 items-center justify-center rounded-lg text-xs font-bold"
                      style={{
                        background: `${s.color}15`,
                        color: s.color,
                      }}
                    >
                      #{i + 1}
                    </div>
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2">
                        <span className="text-sm font-semibold group-hover:text-foreground transition-colors">{s.name}</span>
                        {s.active && (
                          <span className="status-badge-live text-[8px] py-0">
                            <span className="h-1 w-1 rounded-full bg-profit animate-pulse" />
                            LIVE
                          </span>
                        )}
                      </div>
                      <div className="flex items-center gap-3 mt-0.5">
                        <span className="text-[10px] text-muted-foreground">{s.trades} trades</span>
                        <span className="text-[10px] text-muted-foreground">WR: {s.winRate}%</span>
                      </div>
                    </div>
                    {/* Win rate bar */}
                    <div className="hidden sm:flex w-20 flex-col items-end gap-1">
                      <div className="w-full h-1.5 rounded-full bg-muted/60 overflow-hidden">
                        <motion.div
                          initial={{ width: 0 }}
                          animate={{ width: `${s.winRate}%` }}
                          transition={{ delay: 0.5 + i * 0.06, duration: 0.8, ease: "easeOut" }}
                          className="h-full rounded-full"
                          style={{ background: s.color }}
                        />
                      </div>
                    </div>
                    <span className={cn(
                      "font-mono text-sm font-bold min-w-[72px] text-right tabular-nums",
                      s.pnl >= 0 ? "text-profit" : "text-loss"
                    )}>
                      {s.pnl >= 0 ? "+" : ""}{formatCurrency(s.pnl)}
                    </span>
                  </motion.div>
                ))}
            </div>
          </div>
        </motion.div>

        {/* Recent Trades — Activity Feed */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.38 }}
        >
          <div className="glass-card p-4 sm:p-6">
            <div className="flex items-center justify-between mb-4 sm:mb-5">
              <div>
                <div className="flex items-center gap-2">
                  <h3 className="text-sm font-semibold">Recent Trades</h3>
                  <div className="flex items-center gap-1">
                    <span className="relative flex h-2 w-2">
                      <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-primary opacity-60" />
                      <span className="relative inline-flex h-2 w-2 rounded-full bg-primary" />
                    </span>
                  </div>
                </div>
                <p className="text-xs text-muted-foreground mt-0.5">Latest autonomous executions</p>
              </div>
              <Activity className="h-4 w-4 text-primary" />
            </div>
            <div className="space-y-1.5">
              {recentTrades.length === 0 && (
                <div className="text-center py-8 text-muted-foreground text-xs">
                  No trades yet — signals will appear here in real-time
                </div>
              )}
              {recentTrades.map((trade, i) => (
                <motion.div
                  key={`${trade.symbol}-${trade.time}`}
                  initial={{ opacity: 0, x: 12 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: 0.4 + i * 0.06 }}
                  className="flex items-center gap-3 rounded-xl p-3.5 transition-all duration-200 hover:bg-muted/25 group cursor-default"
                >
                  {/* Side indicator with glow */}
                  <div className={cn(
                    "flex h-9 w-9 items-center justify-center rounded-xl text-xs font-bold transition-all group-hover:scale-110",
                    trade.side === "BUY"
                      ? "bg-profit/10 text-profit"
                      : "bg-loss/10 text-loss"
                  )}>
                    {trade.side === "BUY" ? (
                      <ArrowUpRight className="h-4 w-4" />
                    ) : (
                      <ArrowDownRight className="h-4 w-4" />
                    )}
                  </div>
                  {/* Details */}
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2">
                      <span className="text-sm font-semibold">{trade.symbol}</span>
                      <span className={cn(
                        "text-[9px] font-bold px-1.5 py-0.5 rounded-md tracking-wide",
                        trade.side === "BUY"
                          ? "bg-profit/10 text-profit"
                          : "bg-loss/10 text-loss"
                      )}>
                        {trade.side}
                      </span>
                    </div>
                    <div className="flex items-center gap-2 mt-0.5">
                      <span className="text-[10px] text-muted-foreground">{trade.strategy}</span>
                      <CircleDot className="h-1.5 w-1.5 text-muted-foreground/30" />
                      <span className="text-[10px] text-muted-foreground font-mono">{trade.time}</span>
                    </div>
                  </div>
                  {/* P&L */}
                  <span className={cn(
                    "font-mono text-sm font-bold tabular-nums",
                    trade.pnl >= 0 ? "text-profit" : "text-loss"
                  )}>
                    {trade.pnl >= 0 ? "+" : ""}{formatCurrency(trade.pnl)}
                  </span>
                </motion.div>
              ))}
            </div>
            <Link href="/trades" className="mt-4 w-full rounded-xl border border-border/30 py-2.5 text-xs font-semibold text-muted-foreground transition-all duration-300 hover:border-primary/30 hover:text-primary hover:bg-primary/5 hover:shadow-[0_0_16px_hsl(217_91%_60%_/_0.06)] block text-center">
              View All Trades →
            </Link>
          </div>
        </motion.div>
      </div>

      {/* AI Agent & Training Section */}
      <div className="grid gap-4 sm:gap-6 grid-cols-1 md:grid-cols-2 xl:grid-cols-3">
        <div>
          <TrainAIPanel />
        </div>
        <div>
          <AgentStatus />
        </div>
        <div className="md:col-span-2 xl:col-span-1 flex flex-col gap-4 sm:gap-5">
          <SignalFeed />
          <RiskAlerts />
          <RegimeIndicator />
        </div>
      </div>
    </div>
  );
}
