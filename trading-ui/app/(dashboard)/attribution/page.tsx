"use client";

import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { motion } from "framer-motion";
import { endpoints } from "@/lib/api/client";
import { formatCurrency, formatPercent, cn } from "@/lib/utils";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  ResponsiveContainer,
  Tooltip,
  Cell,
  CartesianGrid,
} from "recharts";
import {
  TrendingUp,
  TrendingDown,
  DollarSign,
  Target,
  Brain,
  Activity,
  ArrowUpRight,
  ArrowDownRight,
  BarChart3,
  Layers,
  PieChart as PieChartIcon,
  Clock,
  Globe,
  Cpu,
  Sparkles,
} from "lucide-react";

/* ── Types ── */
interface AttributionRow {
  label: string;
  pnl: number;
  trades: number;
  win_rate: number;
  sharpe: number;
  contribution_pct: number;
}

interface FullAttribution {
  dimensions: Record<string, AttributionRow[]>;
  total_pnl: number;
  total_trades: number;
  win_rate: number;
  best_model: string;
  days: number;
}

interface FeatureEntry {
  feature: string;
  importance: number;
  correlation: number;
}

/* ── Dimension config ── */
const DIMENSIONS = [
  { key: "model", label: "Model", icon: Cpu },
  { key: "strategy", label: "Strategy", icon: Layers },
  { key: "symbol", label: "Symbol", icon: Globe },
  { key: "sector", label: "Sector", icon: PieChartIcon },
  { key: "regime", label: "Regime", icon: Sparkles },
  { key: "time_of_day", label: "Time of Day", icon: Clock },
] as const;

type DimensionKey = (typeof DIMENSIONS)[number]["key"];

/* ── Color palette for bars ── */
const BAR_COLORS = [
  "hsl(258,90%,66%)",
  "hsl(152,69%,53%)",
  "hsl(217,91%,60%)",
  "hsl(199,89%,48%)",
  "hsl(38,92%,50%)",
  "hsl(330,80%,55%)",
  "hsl(280,80%,60%)",
  "hsl(0,84%,60%)",
  "hsl(170,70%,50%)",
  "hsl(45,90%,55%)",
];

const PERIOD_OPTIONS = [
  { label: "7D", days: 7 },
  { label: "30D", days: 30 },
  { label: "90D", days: 90 },
];

/* ── Animated counter ── */
function AnimatedValue({
  value,
  className = "",
}: {
  value: string;
  className?: string;
}) {
  return (
    <motion.span
      key={value}
      initial={{ opacity: 0, y: 10, filter: "blur(4px)" }}
      animate={{ opacity: 1, y: 0, filter: "blur(0px)" }}
      transition={{ duration: 0.5, ease: [0.16, 1, 0.3, 1] }}
      className={className}
    >
      {value}
    </motion.span>
  );
}

/* ── KPI Card ── */
function KpiCard({
  icon: Icon,
  label,
  value,
  subValue,
  trend,
  loading,
  delay = 0,
  variant = "default",
}: {
  icon: React.ComponentType<{ className?: string }>;
  label: string;
  value: string;
  subValue?: string;
  trend?: "up" | "down" | "neutral";
  loading?: boolean;
  delay?: number;
  variant?: "default" | "profit" | "loss";
}) {
  const cardClass =
    variant === "profit"
      ? "glass-card-profit"
      : variant === "loss"
        ? "glass-card-loss"
        : "glass-card";

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
                {trend === "up" && (
                  <ArrowUpRight className="h-3 w-3 text-profit" />
                )}
                {trend === "down" && (
                  <ArrowDownRight className="h-3 w-3 text-loss" />
                )}
                <span
                  className={cn(
                    "text-xs font-medium",
                    trend === "up"
                      ? "text-profit"
                      : trend === "down"
                        ? "text-loss"
                        : "text-muted-foreground"
                  )}
                >
                  {subValue}
                </span>
              </div>
            )}
          </div>
          <div
            className={cn(
              "flex h-10 w-10 items-center justify-center rounded-xl transition-all duration-300 group-hover:scale-110",
              variant === "profit" && "bg-profit/10 text-profit",
              variant === "loss" && "bg-loss/10 text-loss",
              variant === "default" && "bg-primary/10 text-primary"
            )}
          >
            <Icon className="h-5 w-5" />
          </div>
        </div>
      </div>
    </motion.div>
  );
}

/* ── Chart Tooltip ── */
function PnlTooltip({
  active,
  payload,
  label,
}: {
  active?: boolean;
  payload?: Array<{ value: number; payload?: AttributionRow }>;
  label?: string;
}) {
  if (!active || !payload?.length) return null;
  const row = payload[0].payload;
  return (
    <div className="chart-tooltip">
      <p className="text-[10px] text-muted-foreground font-medium mb-1">
        {label}
      </p>
      <p className="font-mono text-sm font-bold text-foreground">
        {formatCurrency(payload[0].value)}
      </p>
      {row && (
        <div className="mt-1 space-y-0.5 text-[10px] text-muted-foreground">
          <p>Trades: {row.trades}</p>
          <p>Win Rate: {row.win_rate.toFixed(1)}%</p>
          <p>Sharpe: {row.sharpe.toFixed(2)}</p>
        </div>
      )}
    </div>
  );
}

function FeatureTooltip({
  active,
  payload,
  label,
}: {
  active?: boolean;
  payload?: Array<{ value: number; payload?: FeatureEntry }>;
  label?: string;
}) {
  if (!active || !payload?.length) return null;
  const entry = payload[0].payload;
  return (
    <div className="chart-tooltip">
      <p className="text-[10px] text-muted-foreground font-medium mb-1">
        {label}
      </p>
      <p className="font-mono text-sm font-bold text-foreground">
        Importance: {(payload[0].value * 100).toFixed(1)}%
      </p>
      {entry && (
        <p className="text-[10px] text-muted-foreground mt-0.5">
          Correlation: {entry.correlation >= 0 ? "+" : ""}
          {entry.correlation.toFixed(3)}
        </p>
      )}
    </div>
  );
}

/* ── Main Page ── */
export default function AttributionPage() {
  const [activeDimension, setActiveDimension] = useState<DimensionKey>("model");
  const [days, setDays] = useState(30);

  /* ── Queries ── */
  const { data: fullData, isLoading } = useQuery({
    queryKey: ["attributionFull", days],
    queryFn: () => endpoints.attributionFull(days),
    refetchInterval: 30000,
  });

  const { data: dimensionData, isLoading: dimLoading } = useQuery({
    queryKey: ["attributionDimension", activeDimension, days],
    queryFn: () => endpoints.attributionByDimension(activeDimension, days),
    refetchInterval: 30000,
  });

  const { data: featureData } = useQuery({
    queryKey: ["featureImportance"],
    queryFn: () => endpoints.featureImportance(15),
    refetchInterval: 60000,
  });

  /* ── Derived data ── */
  const totalPnl = fullData?.total_pnl ?? 0;
  const totalTrades = fullData?.total_trades ?? 0;
  const winRate = fullData?.win_rate ?? 0;
  const bestModel = fullData?.best_model ?? "--";
  const isProfit = totalPnl >= 0;

  const rows: AttributionRow[] = dimensionData?.rows ?? [];
  const sortedRows = [...rows].sort((a, b) => b.pnl - a.pnl);

  const features: FeatureEntry[] = featureData?.features ?? [];
  const sortedFeatures = [...features]
    .sort((a, b) => b.importance - a.importance)
    .slice(0, 15);

  return (
    <div className="space-y-6 pb-8" role="main" aria-label="Performance Attribution">
      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: -10 }}
        animate={{ opacity: 1, y: 0 }}
        className="flex flex-col sm:flex-row sm:items-center justify-between gap-3"
      >
        <div>
          <div className="flex items-center gap-3">
            <h1 className="text-xl sm:text-2xl font-bold tracking-tight">
              Performance Attribution
            </h1>
            <div className="status-badge-live" role="status">
              <span className="h-1.5 w-1.5 rounded-full bg-profit animate-pulse" aria-hidden="true" />
              LIVE
            </div>
          </div>
          <p className="text-xs sm:text-sm text-muted-foreground mt-1">
            Analyze PnL drivers across models, strategies, symbols, and more
          </p>
        </div>
        {/* Period selector */}
        <div className="flex items-center gap-1.5 bg-muted/30 rounded-xl p-1 self-start sm:self-auto">
          {PERIOD_OPTIONS.map((opt) => (
            <button
              key={opt.label}
              onClick={() => setDays(opt.days)}
              aria-pressed={days === opt.days}
              className={cn(
                "rounded-lg px-3 py-1.5 text-[11px] font-semibold transition-all duration-200",
                days === opt.days
                  ? "bg-primary/15 text-primary shadow-sm shadow-primary/10"
                  : "text-muted-foreground hover:text-foreground"
              )}
            >
              {opt.label}
            </button>
          ))}
        </div>
      </motion.div>

      {/* KPI Cards */}
      <div
        className="grid gap-3 sm:gap-4 grid-cols-1 sm:grid-cols-2 xl:grid-cols-4"
        role="region"
        aria-label="Attribution KPIs"
      >
        <KpiCard
          icon={DollarSign}
          label="Total PnL"
          value={formatCurrency(Math.abs(totalPnl))}
          subValue={isProfit ? "Profit" : "Loss"}
          trend={isProfit ? "up" : "down"}
          loading={isLoading}
          delay={0}
          variant={isProfit ? "profit" : "loss"}
        />
        <KpiCard
          icon={Activity}
          label="Total Trades"
          value={String(totalTrades)}
          subValue={`Last ${days} days`}
          trend="neutral"
          loading={isLoading}
          delay={0.06}
        />
        <KpiCard
          icon={Target}
          label="Win Rate"
          value={totalTrades > 0 ? `${Math.round(winRate)}%` : "--"}
          subValue={
            totalTrades > 0
              ? `${Math.round((winRate / 100) * totalTrades)} wins`
              : "No trades"
          }
          trend={winRate > 50 ? "up" : winRate > 0 ? "down" : "neutral"}
          loading={isLoading}
          delay={0.12}
        />
        <KpiCard
          icon={Brain}
          label="Best Model"
          value={bestModel}
          subValue="Top PnL contributor"
          trend="up"
          loading={isLoading}
          delay={0.18}
        />
      </div>

      {/* Dimension Tabs + Bar Chart */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2 }}
      >
        <div className="glass-card p-4 sm:p-6" role="region" aria-label="PnL by dimension">
          {/* Tabs */}
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 mb-5">
            <div>
              <h3 className="text-sm font-semibold">PnL by Dimension</h3>
              <p className="text-xs text-muted-foreground mt-0.5">
                Select a dimension to analyze attribution breakdown
              </p>
            </div>
            <div
              className="flex items-center gap-1 bg-muted/30 rounded-xl p-1 overflow-x-auto"
              role="tablist"
              aria-label="Attribution dimension"
            >
              {DIMENSIONS.map((dim) => {
                const DimIcon = dim.icon;
                return (
                  <button
                    key={dim.key}
                    role="tab"
                    aria-selected={activeDimension === dim.key}
                    onClick={() => setActiveDimension(dim.key)}
                    className={cn(
                      "flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-[11px] font-semibold transition-all duration-200 whitespace-nowrap",
                      activeDimension === dim.key
                        ? "bg-primary/15 text-primary shadow-sm shadow-primary/10"
                        : "text-muted-foreground hover:text-foreground"
                    )}
                  >
                    <DimIcon className="h-3 w-3" />
                    {dim.label}
                  </button>
                );
              })}
            </div>
          </div>

          {/* Bar Chart */}
          <div className="h-64 sm:h-80">
            {dimLoading ? (
              <div className="flex items-center justify-center h-full">
                <div className="shimmer h-full w-full rounded-xl" />
              </div>
            ) : sortedRows.length === 0 ? (
              <div className="flex items-center justify-center h-full text-muted-foreground text-xs">
                No attribution data available for this dimension
              </div>
            ) : (
              <ResponsiveContainer width="100%" height="100%">
                <BarChart
                  data={sortedRows}
                  margin={{ top: 5, right: 20, left: 10, bottom: 5 }}
                >
                  <defs>
                    <linearGradient id="barGradProfit" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor="hsl(152, 69%, 53%)" stopOpacity={0.9} />
                      <stop offset="100%" stopColor="hsl(152, 69%, 53%)" stopOpacity={0.4} />
                    </linearGradient>
                    <linearGradient id="barGradLoss" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor="hsl(0, 84%, 60%)" stopOpacity={0.9} />
                      <stop offset="100%" stopColor="hsl(0, 84%, 60%)" stopOpacity={0.4} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid
                    strokeDasharray="3 3"
                    stroke="hsl(215, 20%, 15%)"
                    vertical={false}
                  />
                  <XAxis
                    dataKey="label"
                    stroke="hsl(215, 20%, 25%)"
                    fontSize={10}
                    tickLine={false}
                    axisLine={false}
                    fontFamily="JetBrains Mono"
                    angle={-30}
                    textAnchor="end"
                    height={60}
                  />
                  <YAxis
                    stroke="hsl(215, 20%, 25%)"
                    fontSize={10}
                    tickLine={false}
                    axisLine={false}
                    tickFormatter={(v: number) =>
                      `${v >= 0 ? "" : "-"}${Math.abs(v) >= 1000 ? `${(Math.abs(v) / 1000).toFixed(0)}k` : Math.abs(v).toFixed(0)}`
                    }
                    fontFamily="JetBrains Mono"
                  />
                  <Tooltip content={<PnlTooltip />} />
                  <Bar dataKey="pnl" radius={[6, 6, 0, 0]} maxBarSize={48}>
                    {sortedRows.map((entry, index) => (
                      <Cell
                        key={`cell-${index}`}
                        fill={
                          entry.pnl >= 0
                            ? BAR_COLORS[index % BAR_COLORS.length]
                            : "hsl(0, 84%, 60%)"
                        }
                        fillOpacity={0.8}
                      />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            )}
          </div>
        </div>
      </motion.div>

      {/* Attribution Table */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.28 }}
      >
        <div className="glass-card p-4 sm:p-6" role="region" aria-label="Attribution table">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h3 className="text-sm font-semibold">Detailed Breakdown</h3>
              <p className="text-xs text-muted-foreground mt-0.5">
                {DIMENSIONS.find((d) => d.key === activeDimension)?.label}{" "}
                attribution details
              </p>
            </div>
            <BarChart3 className="h-4 w-4 text-primary" />
          </div>

          {dimLoading ? (
            <div className="space-y-2">
              {Array.from({ length: 5 }).map((_, i) => (
                <div key={i} className="shimmer h-10 w-full rounded-lg" />
              ))}
            </div>
          ) : sortedRows.length === 0 ? (
            <div className="text-center py-8 text-muted-foreground text-xs">
              No data available
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-border/20">
                    <th className="text-left py-2.5 px-3 text-[10px] font-semibold text-muted-foreground uppercase tracking-wider">
                      {DIMENSIONS.find((d) => d.key === activeDimension)?.label}
                    </th>
                    <th className="text-right py-2.5 px-3 text-[10px] font-semibold text-muted-foreground uppercase tracking-wider">
                      PnL
                    </th>
                    <th className="text-right py-2.5 px-3 text-[10px] font-semibold text-muted-foreground uppercase tracking-wider hidden sm:table-cell">
                      Trades
                    </th>
                    <th className="text-right py-2.5 px-3 text-[10px] font-semibold text-muted-foreground uppercase tracking-wider hidden md:table-cell">
                      Win Rate
                    </th>
                    <th className="text-right py-2.5 px-3 text-[10px] font-semibold text-muted-foreground uppercase tracking-wider hidden lg:table-cell">
                      Sharpe
                    </th>
                    <th className="text-right py-2.5 px-3 text-[10px] font-semibold text-muted-foreground uppercase tracking-wider">
                      Contribution
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {sortedRows.map((row, i) => (
                    <motion.tr
                      key={row.label}
                      initial={{ opacity: 0, x: -12 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ delay: 0.3 + i * 0.04 }}
                      className="border-b border-border/10 hover:bg-muted/20 transition-colors"
                    >
                      <td className="py-3 px-3">
                        <div className="flex items-center gap-2.5">
                          <div
                            className="h-2.5 w-2.5 rounded-full flex-shrink-0"
                            style={{
                              backgroundColor: BAR_COLORS[i % BAR_COLORS.length],
                              boxShadow: `0 0 6px ${BAR_COLORS[i % BAR_COLORS.length]}40`,
                            }}
                          />
                          <span className="font-medium text-sm">{row.label}</span>
                        </div>
                      </td>
                      <td className="py-3 px-3 text-right">
                        <span
                          className={cn(
                            "font-mono text-sm font-bold tabular-nums",
                            row.pnl >= 0 ? "text-profit" : "text-loss"
                          )}
                        >
                          {row.pnl >= 0 ? "+" : ""}
                          {formatCurrency(row.pnl)}
                        </span>
                      </td>
                      <td className="py-3 px-3 text-right hidden sm:table-cell">
                        <span className="font-mono text-xs text-muted-foreground">
                          {row.trades}
                        </span>
                      </td>
                      <td className="py-3 px-3 text-right hidden md:table-cell">
                        <div className="flex items-center justify-end gap-2">
                          <div className="w-12 h-1.5 rounded-full bg-muted/60 overflow-hidden hidden lg:block">
                            <div
                              className="h-full rounded-full transition-all"
                              style={{
                                width: `${Math.min(row.win_rate, 100)}%`,
                                backgroundColor:
                                  row.win_rate >= 50
                                    ? "hsl(152, 69%, 53%)"
                                    : "hsl(0, 84%, 60%)",
                              }}
                            />
                          </div>
                          <span
                            className={cn(
                              "font-mono text-xs font-semibold",
                              row.win_rate >= 50
                                ? "text-profit"
                                : "text-loss"
                            )}
                          >
                            {row.win_rate.toFixed(1)}%
                          </span>
                        </div>
                      </td>
                      <td className="py-3 px-3 text-right hidden lg:table-cell">
                        <span
                          className={cn(
                            "font-mono text-xs font-semibold",
                            row.sharpe >= 1
                              ? "text-profit"
                              : row.sharpe >= 0
                                ? "text-muted-foreground"
                                : "text-loss"
                          )}
                        >
                          {row.sharpe.toFixed(2)}
                        </span>
                      </td>
                      <td className="py-3 px-3 text-right">
                        <div className="flex items-center justify-end gap-2">
                          <div className="w-16 h-1.5 rounded-full bg-muted/60 overflow-hidden hidden sm:block">
                            <motion.div
                              initial={{ width: 0 }}
                              animate={{
                                width: `${Math.min(Math.abs(row.contribution_pct), 100)}%`,
                              }}
                              transition={{
                                delay: 0.4 + i * 0.04,
                                duration: 0.8,
                                ease: "easeOut",
                              }}
                              className="h-full rounded-full"
                              style={{
                                backgroundColor:
                                  BAR_COLORS[i % BAR_COLORS.length],
                              }}
                            />
                          </div>
                          <span className="font-mono text-xs font-semibold text-foreground">
                            {row.contribution_pct >= 0 ? "+" : ""}
                            {row.contribution_pct.toFixed(1)}%
                          </span>
                        </div>
                      </td>
                    </motion.tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </motion.div>

      {/* Feature Importance */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.36 }}
      >
        <div className="glass-card p-4 sm:p-6" role="region" aria-label="Feature importance">
          <div className="flex items-center justify-between mb-5">
            <div>
              <div className="flex items-center gap-2.5">
                <h3 className="text-sm font-semibold">Feature Importance</h3>
                <div className="flex items-center gap-1">
                  <span className="relative flex h-2 w-2">
                    <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-primary opacity-60" />
                    <span className="relative inline-flex h-2 w-2 rounded-full bg-primary" />
                  </span>
                </div>
              </div>
              <p className="text-xs text-muted-foreground mt-0.5">
                Top 15 features correlated with PnL outcomes
              </p>
            </div>
            <Sparkles className="h-4 w-4 text-primary" />
          </div>

          <div className="h-[420px] sm:h-[480px]">
            {sortedFeatures.length === 0 ? (
              <div className="flex items-center justify-center h-full text-muted-foreground text-xs">
                Feature importance data will appear after model training
              </div>
            ) : (
              <ResponsiveContainer width="100%" height="100%">
                <BarChart
                  data={sortedFeatures}
                  layout="vertical"
                  margin={{ top: 5, right: 30, left: 120, bottom: 5 }}
                >
                  <CartesianGrid
                    strokeDasharray="3 3"
                    stroke="hsl(215, 20%, 15%)"
                    horizontal={false}
                  />
                  <XAxis
                    type="number"
                    stroke="hsl(215, 20%, 25%)"
                    fontSize={10}
                    tickLine={false}
                    axisLine={false}
                    tickFormatter={(v: number) => `${(v * 100).toFixed(0)}%`}
                    fontFamily="JetBrains Mono"
                  />
                  <YAxis
                    type="category"
                    dataKey="feature"
                    stroke="hsl(215, 20%, 25%)"
                    fontSize={10}
                    tickLine={false}
                    axisLine={false}
                    width={110}
                    fontFamily="JetBrains Mono"
                  />
                  <Tooltip content={<FeatureTooltip />} />
                  <Bar
                    dataKey="importance"
                    radius={[0, 6, 6, 0]}
                    maxBarSize={24}
                  >
                    {sortedFeatures.map((entry, index) => (
                      <Cell
                        key={`feat-${index}`}
                        fill={
                          entry.correlation >= 0
                            ? "hsl(152, 69%, 53%)"
                            : "hsl(0, 84%, 60%)"
                        }
                        fillOpacity={0.7 + (0.3 * (1 - index / sortedFeatures.length))}
                      />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            )}
          </div>

          {/* Legend */}
          {sortedFeatures.length > 0 && (
            <div className="flex justify-center gap-6 mt-4">
              <div className="flex items-center gap-1.5">
                <span
                  className="h-2.5 w-2.5 rounded-full"
                  style={{
                    background: "hsl(152, 69%, 53%)",
                    boxShadow: "0 0 6px hsl(152 69% 53% / 40%)",
                  }}
                />
                <span className="text-[10px] text-muted-foreground">
                  Positive Correlation
                </span>
              </div>
              <div className="flex items-center gap-1.5">
                <span
                  className="h-2.5 w-2.5 rounded-full"
                  style={{
                    background: "hsl(0, 84%, 60%)",
                    boxShadow: "0 0 6px hsl(0 84% 60% / 40%)",
                  }}
                />
                <span className="text-[10px] text-muted-foreground">
                  Negative Correlation
                </span>
              </div>
            </div>
          )}
        </div>
      </motion.div>
    </div>
  );
}
