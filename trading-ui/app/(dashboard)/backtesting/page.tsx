"use client";

import { useState, useEffect } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { motion, AnimatePresence } from "framer-motion";
import { cn, formatCurrency } from "@/lib/utils";
import { endpoints } from "@/lib/api/client";
import {
    AreaChart, Area, XAxis, YAxis,
    ResponsiveContainer, Tooltip,
} from "recharts";
import {
    FlaskConical, Play, Loader2, TrendingUp, TrendingDown,
    Target, Shield, Award, BarChart3, Clock, ChevronDown,
    Search,
} from "lucide-react";
import { dispatchToast } from "@/components/Toaster";

function ChartTooltip({ active, payload, label }: { active?: boolean; payload?: Array<{ value: number }>; label?: string }) {
    if (!active || !payload?.length) return null;
    return (
        <div className="chart-tooltip">
            <p className="text-[10px] text-muted-foreground font-medium">{label}</p>
            <p className="font-mono text-sm font-bold">{formatCurrency(payload[0].value)}</p>
        </div>
    );
}

const STRATEGIES = [
    { id: "ema_crossover", name: "EMA Crossover" },
    { id: "macd", name: "MACD" },
    { id: "rsi", name: "RSI" },
    { id: "momentum_breakout", name: "Momentum Breakout" },
    { id: "mean_reversion", name: "Mean Reversion" },
    { id: "ai_alpha", name: "AI Alpha" },
];

const SYMBOLS = [
    "RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK",
    "HINDUNILVR", "SBIN", "BHARTIARTL", "ITC", "KOTAKBANK",
    "LT", "WIPRO", "AXISBANK", "BAJFINANCE", "MARUTI",
];

export default function BacktestingPage() {
    const queryClient = useQueryClient();
    const [strategyId, setStrategyId] = useState("ema_crossover");
    const [symbol, setSymbol] = useState("RELIANCE");
    const [startDate, setStartDate] = useState("2025-01-01");
    const [endDate, setEndDate] = useState("2025-12-31");
    const [interval, setInterval_] = useState("1d");
    const [activeJobId, setActiveJobId] = useState<string | null>(null);
    const [polling, setPolling] = useState(false);
    const [dateError, setDateError] = useState("");

    // Run backtest mutation
    const runMutation = useMutation({
        mutationFn: () => {
            if (endDate <= startDate) {
                throw new Error("End date must be after start date");
            }
            return endpoints.backtestRun({
                strategy_id: strategyId,
                symbol,
                start: startDate,
                end: endDate,
                interval,
            });
        },
        onSuccess: (data) => {
            setDateError("");
            setActiveJobId(data.job_id);
            setPolling(true);
        },
        onError: (err: Error) => {
            if (err.message.includes("End date must be after start date")) {
                setDateError(err.message);
                dispatchToast("error", "Invalid Dates", err.message);
            }
        },
    });

    // Poll job status
    const { data: jobData } = useQuery({
        queryKey: ["backtest-job", activeJobId],
        queryFn: () => endpoints.backtestJob(activeJobId!),
        enabled: !!activeJobId && polling,
        refetchInterval: polling ? 1000 : false,
    });

    // Stop polling when completed
    useEffect(() => {
        if (jobData && (jobData.status === "completed" || jobData.status === "failed")) {
            if (polling) setPolling(false);
        }
    }, [jobData, polling]); // eslint-disable-line react-hooks/exhaustive-deps

    // Get equity curve and trades when job is completed
    const { data: equityData } = useQuery({
        queryKey: ["backtest-equity", activeJobId],
        queryFn: () => endpoints.backtestEquity(activeJobId!),
        enabled: !!activeJobId && jobData?.status === "completed",
    });

    const { data: tradesData } = useQuery({
        queryKey: ["backtest-trades", activeJobId],
        queryFn: () => endpoints.backtestTrades(activeJobId!, 50),
        enabled: !!activeJobId && jobData?.status === "completed",
    });

    // List previous jobs
    const { data: jobsData } = useQuery({
        queryKey: ["backtest-jobs"],
        queryFn: () => endpoints.backtestJobs(),
        refetchInterval: 10000,
    });

    const isRunning = runMutation.isPending || polling;
    const metrics = jobData?.metrics ?? {};
    const equityCurve = (equityData?.equity_curve ?? []).map((pt, i) => ({
        label: pt.date?.substring(5, 10) || `D${i + 1}`,
        equity: pt.equity,
    }));
    const trades = tradesData?.trades ?? [];

    return (
        <div className="space-y-6 pb-8">
            {/* Header */}
            <motion.div initial={{ opacity: 0, y: -10 }} animate={{ opacity: 1, y: 0 }}>
                <div className="flex items-center gap-3">
                    <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-gradient-to-br from-violet-500 to-fuchsia-500 shadow-lg">
                        <FlaskConical className="h-5 w-5 text-white" />
                    </div>
                    <div>
                        <h1 className="text-2xl font-bold tracking-tight">Backtesting</h1>
                        <p className="text-sm text-muted-foreground mt-0.5">
                            Test strategies against historical data
                        </p>
                    </div>
                </div>
            </motion.div>

            {/* Config panel */}
            <motion.div
                initial={{ opacity: 0, y: 12 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.05 }}
            >
                <div className="glass-card p-6">
                    <h3 className="text-sm font-semibold mb-5">Configuration</h3>
                    <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-5">
                        {/* Strategy */}
                        <div>
                            <label htmlFor="bt-strategy" className="text-[10px] font-semibold uppercase tracking-[0.1em] text-muted-foreground block mb-2">Strategy</label>
                            <div className="relative">
                                <select
                                    id="bt-strategy"
                                    value={strategyId}
                                    onChange={(e) => setStrategyId(e.target.value)}
                                    aria-label="Select backtest strategy"
                                    className="w-full rounded-xl border border-border/40 bg-muted/20 px-3.5 py-2.5 text-sm appearance-none cursor-pointer transition-all duration-300 focus:outline-none focus:ring-2 focus:ring-primary/20 focus:border-primary/40 focus:bg-muted/30 hover:border-border/60"
                                >
                                    {STRATEGIES.map((s) => (
                                        <option key={s.id} value={s.id}>{s.name}</option>
                                    ))}
                                </select>
                                <ChevronDown className="absolute right-2.5 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-muted-foreground pointer-events-none" />
                            </div>
                        </div>

                        {/* Symbol */}
                        <div>
                            <label htmlFor="bt-symbol" className="text-[10px] font-semibold uppercase tracking-[0.1em] text-muted-foreground block mb-2">Symbol</label>
                            <div className="relative">
                                <select
                                    id="bt-symbol"
                                    value={symbol}
                                    onChange={(e) => setSymbol(e.target.value)}
                                    aria-label="Select backtest symbol"
                                    className="w-full rounded-xl border border-border/40 bg-muted/20 px-3.5 py-2.5 text-sm appearance-none cursor-pointer transition-all duration-300 focus:outline-none focus:ring-2 focus:ring-primary/20 focus:border-primary/40 focus:bg-muted/30 hover:border-border/60"
                                >
                                    {SYMBOLS.map((s) => (
                                        <option key={s} value={s}>{s}</option>
                                    ))}
                                </select>
                                <ChevronDown className="absolute right-2.5 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-muted-foreground pointer-events-none" />
                            </div>
                        </div>

                        {/* Start date */}
                        <div>
                            <label htmlFor="bt-start-date" className="text-[10px] font-semibold uppercase tracking-[0.1em] text-muted-foreground block mb-2">Start Date</label>
                            <div className="relative">
                                <input
                                    id="bt-start-date"
                                    type="date"
                                    value={startDate}
                                    onChange={(e) => setStartDate(e.target.value)}
                                    aria-label="Backtest start date"
                                    className="w-full rounded-xl border border-border/40 bg-muted/20 px-3.5 py-2.5 text-sm transition-all duration-300 focus:outline-none focus:ring-2 focus:ring-primary/20 focus:border-primary/40 focus:bg-muted/30 hover:border-border/60"
                                />
                            </div>
                        </div>

                        {/* End date */}
                        <div>
                            <label htmlFor="bt-end-date" className="text-[10px] font-semibold uppercase tracking-[0.1em] text-muted-foreground block mb-2">End Date</label>
                            <div className="relative">
                                <input
                                    id="bt-end-date"
                                    type="date"
                                    value={endDate}
                                    onChange={(e) => setEndDate(e.target.value)}
                                    aria-label="Backtest end date"
                                    className="w-full rounded-xl border border-border/40 bg-muted/20 px-3.5 py-2.5 text-sm transition-all duration-300 focus:outline-none focus:ring-2 focus:ring-primary/20 focus:border-primary/40 focus:bg-muted/30 hover:border-border/60"
                                />
                            </div>
                        </div>

                        {/* Run button */}
                        <div className="flex items-end">
                            <button
                                onClick={() => runMutation.mutate()}
                                disabled={isRunning}
                                aria-label={isRunning ? "Backtest is running" : "Run backtest"}
                                aria-busy={isRunning}
                                className={cn(
                                    "w-full flex items-center justify-center gap-2 rounded-xl px-4 py-2.5 text-sm font-semibold transition-all duration-300 overflow-hidden relative group",
                                    isRunning
                                        ? "bg-muted text-muted-foreground cursor-not-allowed"
                                        : "bg-gradient-to-r from-violet-500 to-fuchsia-500 text-white hover:from-violet-600 hover:to-fuchsia-600 shadow-lg hover:shadow-xl hover:shadow-violet-500/20"
                                )}
                            >
                                {isRunning ? (
                                    <>
                                        <Loader2 className="h-4 w-4 animate-spin" />
                                        Running…
                                    </>
                                ) : (
                                    <>
                                        <Play className="h-4 w-4" />
                                        Run Backtest
                                    </>
                                )}
                            </button>
                        </div>
                    {dateError && (
                        <p className="text-xs text-loss mt-2 col-span-full">{dateError}</p>
                    )}
                    </div>
                </div>
            </motion.div>

            {/* Results */}
            <AnimatePresence mode="wait">
                {jobData?.status === "completed" && (
                    <motion.div
                        key="results"
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, y: -10 }}
                        className="space-y-6"
                    >
                        {/* Metrics cards */}
                        <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
                            {[
                                { label: "Total Return", value: `${(metrics.total_return_pct ?? 0) >= 0 ? "+" : ""}${(metrics.total_return_pct ?? 0).toFixed(1)}%`, icon: metrics.total_return_pct! >= 0 ? TrendingUp : TrendingDown, color: (metrics.total_return_pct ?? 0) >= 0 ? "text-profit" : "text-loss" },
                                { label: "Win Rate", value: `${(metrics.win_rate ?? 0).toFixed(0)}%`, icon: Target, color: "text-profit" },
                                { label: "Max Drawdown", value: `${(metrics.max_drawdown_pct ?? 0).toFixed(1)}%`, icon: Shield, color: "text-loss" },
                                { label: "Sharpe Ratio", value: (metrics.sharpe_ratio ?? 0).toFixed(2), icon: Award, color: "text-primary" },
                            ].map((kpi, i) => (
                                <motion.div
                                    key={kpi.label}
                                    initial={{ opacity: 0, scale: 0.95 }}
                                    animate={{ opacity: 1, scale: 1 }}
                                    transition={{ delay: i * 0.05 }}
                                >
                                    <div className={cn(
                                        kpi.color === "text-profit" ? "glass-card-profit" : kpi.color === "text-loss" ? "glass-card-loss" : "glass-card",
                                        "p-4"
                                    )}>
                                        <div className="flex items-center gap-3">
                                            <div className={cn("flex h-9 w-9 items-center justify-center rounded-xl bg-muted/40", kpi.color)}>
                                                <kpi.icon className="h-4 w-4" />
                                            </div>
                                            <div>
                                                <div className="text-[10px] text-muted-foreground uppercase tracking-wider">{kpi.label}</div>
                                                <div className={cn("text-lg font-bold font-mono", kpi.color)}>{kpi.value}</div>
                                            </div>
                                        </div>
                                    </div>
                                </motion.div>
                            ))}
                        </div>

                        {/* Equity curve */}
                        {equityCurve.length > 0 && (
                            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.2 }}>
                                <div className="glass-card p-5">
                                    <div className="flex items-center justify-between mb-4">
                                        <div>
                                            <h3 className="text-sm font-semibold">Backtest Equity Curve</h3>
                                            <p className="text-xs text-muted-foreground mt-0.5">{symbol} • {STRATEGIES.find(s => s.id === strategyId)?.name}</p>
                                        </div>
                                        <BarChart3 className="h-4 w-4 text-primary" />
                                    </div>
                                    <div className="h-72">
                                        <ResponsiveContainer width="100%" height="100%">
                                            <AreaChart data={equityCurve} margin={{ top: 5, right: 5, left: 5, bottom: 0 }}>
                                                <defs>
                                                    <linearGradient id="btEquity" x1="0" y1="0" x2="0" y2="1">
                                                        <stop offset="0%" stopColor="hsl(270, 80%, 60%)" stopOpacity={0.3} />
                                                        <stop offset="100%" stopColor="hsl(270, 80%, 60%)" stopOpacity={0} />
                                                    </linearGradient>
                                                </defs>
                                                <XAxis dataKey="label" stroke="hsl(215, 20%, 30%)" fontSize={10} tickLine={false} axisLine={false} />
                                                <YAxis stroke="hsl(215, 20%, 30%)" fontSize={10} tickLine={false} axisLine={false} tickFormatter={(v) => `₹${(v / 1000).toFixed(0)}k`} />
                                                <Tooltip content={<ChartTooltip />} />
                                                <Area type="monotone" dataKey="equity" stroke="hsl(270, 80%, 60%)" fill="url(#btEquity)" strokeWidth={2.5} dot={false} />
                                            </AreaChart>
                                        </ResponsiveContainer>
                                    </div>
                                </div>
                            </motion.div>
                        )}

                        {/* Trades table */}
                        {trades.length > 0 && (
                            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.3 }}>
                                <div className="glass-card p-5">
                                    <div className="flex items-center justify-between mb-4">
                                        <div>
                                            <h3 className="text-sm font-semibold">Trade Log</h3>
                                            <p className="text-xs text-muted-foreground mt-0.5">{trades.length} trades</p>
                                        </div>
                                        <Search className="h-4 w-4 text-muted-foreground" />
                                    </div>
                                    <div className="overflow-x-auto">
                                        <table className="w-full text-xs">
                                            <thead>
                                                <tr className="border-b border-border/30 text-muted-foreground">
                                                    <th scope="col" className="py-2 px-3 text-left font-medium">Date</th>
                                                    <th scope="col" className="py-2 px-3 text-left font-medium">Side</th>
                                                    <th scope="col" className="py-2 px-3 text-right font-medium">Price</th>
                                                    <th scope="col" className="py-2 px-3 text-right font-medium">Qty</th>
                                                    <th scope="col" className="py-2 px-3 text-right font-medium">P&L</th>
                                                </tr>
                                            </thead>
                                            <tbody>
                                                {trades.map((trade: any, i: number) => (
                                                    <tr key={i} className="border-b border-border/10 hover:bg-muted/20 transition-colors">
                                                        <td className="py-2 px-3 font-mono">{String(trade.date ?? trade.ts ?? "").substring(0, 10)}</td>
                                                        <td className="py-2 px-3">
                                                            <span className={cn(
                                                                "inline-flex items-center rounded-full px-2 py-0.5 text-[10px] font-bold",
                                                                String(trade.side).toUpperCase() === "BUY"
                                                                    ? "bg-profit/10 text-profit"
                                                                    : "bg-loss/10 text-loss"
                                                            )}>
                                                                {String(trade.side ?? "BUY").toUpperCase()}
                                                            </span>
                                                        </td>
                                                        <td className="py-2 px-3 text-right font-mono">{formatCurrency(Number(trade.price ?? 0))}</td>
                                                        <td className="py-2 px-3 text-right font-mono">{trade.quantity ?? trade.qty ?? 0}</td>
                                                        <td className={cn("py-2 px-3 text-right font-mono font-semibold", Number(trade.pnl ?? 0) >= 0 ? "text-profit" : "text-loss")}>
                                                            {Number(trade.pnl ?? 0) >= 0 ? "+" : ""}{formatCurrency(Number(trade.pnl ?? 0))}
                                                        </td>
                                                    </tr>
                                                ))}
                                            </tbody>
                                        </table>
                                    </div>
                                </div>
                            </motion.div>
                        )}
                    </motion.div>
                )}

                {jobData?.status === "failed" && (
                    <motion.div
                        key="error"
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                    >
                        <div className="glass-card-loss p-5 text-center">
                            <p className="text-sm text-loss font-medium">Backtest failed</p>
                            <p className="text-xs text-muted-foreground mt-1">{jobData.error || "Unknown error"}</p>
                        </div>
                    </motion.div>
                )}

                {isRunning && (
                    <motion.div
                        key="loading"
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                    >
                        <div className="glass-card p-10 text-center">
                            <Loader2 className="h-8 w-8 animate-spin text-primary mx-auto mb-3" />
                            <p className="text-sm font-medium">Running backtest…</p>
                            <p className="text-xs text-muted-foreground mt-1">This may take a few seconds</p>
                        </div>
                    </motion.div>
                )}
            </AnimatePresence>

            {/* Previous jobs */}
            {(jobsData?.jobs?.length ?? 0) > 0 && (
                <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.1 }}>
                    <div className="glass-card p-5">
                        <h3 className="text-sm font-semibold mb-3 flex items-center gap-2">
                            <Clock className="h-4 w-4 text-muted-foreground" />
                            Previous Runs
                        </h3>
                        <div className="space-y-2">
                            {jobsData!.jobs.map((job) => (
                                <button
                                    key={job.job_id}
                                    onClick={() => { setActiveJobId(job.job_id); setPolling(false); }}
                                    aria-label={`View backtest job ${job.job_id.substring(0, 8)} - ${job.strategy_id ?? "unknown strategy"} ${job.symbol ?? ""} - ${job.status}`}
                                    aria-pressed={activeJobId === job.job_id}
                                    className={cn(
                                        "w-full flex items-center justify-between rounded-lg px-3 py-2 text-xs transition-all hover:bg-muted/30",
                                        activeJobId === job.job_id ? "bg-muted/30 border border-primary/30" : "bg-muted/10"
                                    )}
                                >
                                    <div className="flex items-center gap-3">
                                        <span className="font-mono text-muted-foreground">{job.job_id.substring(0, 8)}</span>
                                        <span className="font-medium">{job.strategy_id ?? "—"}</span>
                                        <span className="text-muted-foreground">{job.symbol ?? ""}</span>
                                    </div>
                                    <span className={cn(
                                        "inline-flex items-center rounded-full px-2 py-0.5 text-[10px] font-bold",
                                        job.status === "completed" ? "bg-profit/10 text-profit" :
                                            job.status === "failed" ? "bg-loss/10 text-loss" :
                                                "bg-primary/10 text-primary"
                                    )}>
                                        {job.status}
                                    </span>
                                </button>
                            ))}
                        </div>
                    </div>
                </motion.div>
            )}
        </div>
    );
}
