"use client";

import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { motion, AnimatePresence } from "framer-motion";
import { cn } from "@/lib/utils";
import { endpoints } from "@/lib/api/client";
import {
    BarChart, Bar, XAxis, YAxis,
    ResponsiveContainer, Tooltip,
} from "recharts";
import {
    Moon, Play, Loader2, TrendingUp, Award,
    Target, CheckCircle2, Hash, ChevronDown, ChevronUp,
    Clock, Zap, Info,
} from "lucide-react";

/* ------------------------------------------------------------------ */
/*  Types                                                              */
/* ------------------------------------------------------------------ */

interface SimulationStatus {
    running: boolean;
    last_run: string | null;
    total_permutations: number;
    progress?: number;
    current_step?: string;
}

interface SimulationResult {
    rank: number;
    strategy_id: string;
    params: Record<string, unknown>;
    interval: string;
    sharpe: number;
    sortino: number;
    max_dd: number;
    win_rate: number;
    profit_factor: number;
    trades: number;
    selected: boolean;
}

interface SimulationResults {
    results: SimulationResult[];
    total_tested: number;
    qualified: number;
    top_sharpe: number;
    top_win_rate: number;
    run_ts?: string;
}

/* ------------------------------------------------------------------ */
/*  Chart tooltip                                                      */
/* ------------------------------------------------------------------ */

function ChartTooltip({ active, payload, label }: { active?: boolean; payload?: Array<{ value: number; name?: string }>; label?: string }) {
    if (!active || !payload?.length) return null;
    return (
        <div className="chart-tooltip">
            <p className="text-[10px] text-muted-foreground font-medium">{label}</p>
            {payload.map((p, i) => (
                <p key={i} className="font-mono text-sm font-bold">{p.value?.toFixed(2)}</p>
            ))}
        </div>
    );
}

/* ------------------------------------------------------------------ */
/*  Params tooltip                                                     */
/* ------------------------------------------------------------------ */

function ParamsCell({ params }: { params: Record<string, unknown> }) {
    const [open, setOpen] = useState(false);
    const text = JSON.stringify(params);
    const short = text.length > 30 ? text.slice(0, 28) + "..." : text;

    return (
        <td className="py-2 px-3 font-mono text-[10px] relative max-w-[180px]">
            <button
                className="text-left hover:text-primary transition-colors flex items-center gap-1"
                onClick={() => setOpen(!open)}
                aria-label="Toggle parameter details"
            >
                <span className="truncate">{short}</span>
                <Info className="h-3 w-3 shrink-0 text-muted-foreground" />
            </button>
            {open && (
                <div className="absolute z-50 top-full left-0 mt-1 p-3 rounded-xl border border-border/40 bg-background shadow-xl max-w-xs">
                    <pre className="text-[10px] font-mono whitespace-pre-wrap break-all text-muted-foreground">
                        {JSON.stringify(params, null, 2)}
                    </pre>
                </div>
            )}
        </td>
    );
}

/* ------------------------------------------------------------------ */
/*  Main page                                                          */
/* ------------------------------------------------------------------ */

export default function SimulationPage() {
    const queryClient = useQueryClient();
    const [showFullResults, setShowFullResults] = useState(false);

    // --- Status polling ---
    const { data: statusData } = useQuery<SimulationStatus>({
        queryKey: ["simulation-status"],
        queryFn: () => endpoints.simulationStatus(),
        refetchInterval: (query) => {
            const d = query.state.data as SimulationStatus | undefined;
            return d?.running ? 2000 : 10000;
        },
    });

    const isRunning = statusData?.running ?? false;

    // --- Selected results ---
    const { data: selectedData } = useQuery<SimulationResults>({
        queryKey: ["simulation-results-selected"],
        queryFn: () => endpoints.simulationResults(50, true),
        enabled: !isRunning,
        refetchInterval: 30000,
    });

    // --- Full results (loaded on expand) ---
    const { data: fullData, isFetching: fullFetching } = useQuery<SimulationResults>({
        queryKey: ["simulation-results-full"],
        queryFn: () => endpoints.simulationResults(50, false),
        enabled: showFullResults && !isRunning,
    });

    // --- Run mutation ---
    const runMutation = useMutation({
        mutationFn: () => endpoints.simulationRun(),
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: ["simulation-status"] });
        },
    });

    // Derived
    const selectedResults = selectedData?.results ?? [];
    const fullResults = fullData?.results ?? [];
    const totalTested = selectedData?.total_tested ?? statusData?.total_permutations ?? 0;
    const qualified = selectedData?.qualified ?? 0;
    const topSharpe = selectedData?.top_sharpe ?? 0;
    const topWinRate = selectedData?.top_win_rate ?? 0;
    const hasResults = selectedResults.length > 0;

    // Chart data: top 10 by sharpe for the bar chart
    const chartData = selectedResults.slice(0, 10).map((r) => ({
        name: r.strategy_id.length > 12 ? r.strategy_id.slice(0, 10) + ".." : r.strategy_id,
        sharpe: r.sharpe,
        winRate: r.win_rate,
    }));

    return (
        <div className="space-y-6 pb-8">
            {/* Header */}
            <motion.div initial={{ opacity: 0, y: -10 }} animate={{ opacity: 1, y: 0 }}>
                <div className="flex items-center gap-3">
                    <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-gradient-to-br from-indigo-500 to-cyan-500 shadow-lg">
                        <Moon className="h-5 w-5 text-white" />
                    </div>
                    <div>
                        <h1 className="text-2xl font-bold tracking-tight">Nightly Simulation Engine</h1>
                        <p className="text-sm text-muted-foreground mt-0.5">
                            Holly AI-style strategy permutation testing
                        </p>
                    </div>
                </div>
            </motion.div>

            {/* Status card + Run button */}
            <motion.div
                initial={{ opacity: 0, y: 12 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.05 }}
            >
                <div className="glass-card p-6">
                    <h3 className="text-sm font-semibold mb-5">Simulation Status</h3>
                    <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
                        {/* Last run */}
                        <div>
                            <div className="text-[10px] font-semibold uppercase tracking-[0.1em] text-muted-foreground mb-2">Last Run</div>
                            <div className="flex items-center gap-2 text-sm font-mono">
                                <Clock className="h-3.5 w-3.5 text-muted-foreground" />
                                {statusData?.last_run
                                    ? new Date(statusData.last_run).toLocaleString()
                                    : "Never"}
                            </div>
                        </div>

                        {/* Running status */}
                        <div>
                            <div className="text-[10px] font-semibold uppercase tracking-[0.1em] text-muted-foreground mb-2">Status</div>
                            <div className="flex items-center gap-2 text-sm">
                                {isRunning ? (
                                    <>
                                        <Loader2 className="h-3.5 w-3.5 animate-spin text-primary" />
                                        <span className="text-primary font-medium">Running{statusData?.current_step ? ` - ${statusData.current_step}` : ""}</span>
                                    </>
                                ) : (
                                    <>
                                        <CheckCircle2 className="h-3.5 w-3.5 text-profit" />
                                        <span className="text-profit font-medium">Idle</span>
                                    </>
                                )}
                            </div>
                        </div>

                        {/* Total permutations */}
                        <div>
                            <div className="text-[10px] font-semibold uppercase tracking-[0.1em] text-muted-foreground mb-2">Total Permutations</div>
                            <div className="flex items-center gap-2 text-sm font-mono">
                                <Hash className="h-3.5 w-3.5 text-muted-foreground" />
                                {(statusData?.total_permutations ?? 0).toLocaleString()}
                            </div>
                        </div>

                        {/* Run button */}
                        <div className="flex items-end">
                            <button
                                onClick={() => runMutation.mutate()}
                                disabled={isRunning || runMutation.isPending}
                                aria-label={isRunning ? "Simulation is running" : "Run simulation"}
                                aria-busy={isRunning}
                                className={cn(
                                    "w-full flex items-center justify-center gap-2 rounded-xl px-4 py-2.5 text-sm font-semibold transition-all duration-300 overflow-hidden relative group",
                                    isRunning || runMutation.isPending
                                        ? "bg-muted text-muted-foreground cursor-not-allowed"
                                        : "bg-gradient-to-r from-indigo-500 to-cyan-500 text-white hover:from-indigo-600 hover:to-cyan-600 shadow-lg hover:shadow-xl hover:shadow-indigo-500/20"
                                )}
                            >
                                {isRunning || runMutation.isPending ? (
                                    <>
                                        <Loader2 className="h-4 w-4 animate-spin" />
                                        Running{statusData?.progress != null ? ` ${statusData.progress}%` : "..."}
                                    </>
                                ) : (
                                    <>
                                        <Play className="h-4 w-4" />
                                        Run Simulation
                                    </>
                                )}
                            </button>
                        </div>
                    </div>

                    {/* Progress bar */}
                    {isRunning && statusData?.progress != null && (
                        <div className="mt-4">
                            <div className="h-1.5 rounded-full bg-muted/30 overflow-hidden">
                                <motion.div
                                    className="h-full rounded-full bg-gradient-to-r from-indigo-500 to-cyan-500"
                                    initial={{ width: 0 }}
                                    animate={{ width: `${statusData.progress}%` }}
                                    transition={{ duration: 0.5 }}
                                />
                            </div>
                        </div>
                    )}
                </div>
            </motion.div>

            {/* Running overlay */}
            <AnimatePresence mode="wait">
                {isRunning && (
                    <motion.div
                        key="running"
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        exit={{ opacity: 0 }}
                    >
                        <div className="glass-card p-10 text-center">
                            <Loader2 className="h-8 w-8 animate-spin text-primary mx-auto mb-3" />
                            <p className="text-sm font-medium">Simulation in progress...</p>
                            <p className="text-xs text-muted-foreground mt-1">
                                Testing strategy permutations across multiple intervals and parameters
                            </p>
                        </div>
                    </motion.div>
                )}

                {/* Results */}
                {!isRunning && hasResults && (
                    <motion.div
                        key="results"
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, y: -10 }}
                        className="space-y-6"
                    >
                        {/* Summary cards */}
                        <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
                            {[
                                { label: "Total Tested", value: totalTested.toLocaleString(), icon: Hash, color: "text-primary" },
                                { label: "Qualified", value: qualified.toLocaleString(), icon: CheckCircle2, color: "text-profit" },
                                { label: "Top Sharpe", value: topSharpe.toFixed(2), icon: Award, color: "text-primary" },
                                { label: "Top Win Rate", value: `${topWinRate.toFixed(1)}%`, icon: Target, color: "text-profit" },
                            ].map((kpi, i) => (
                                <motion.div
                                    key={kpi.label}
                                    initial={{ opacity: 0, scale: 0.95 }}
                                    animate={{ opacity: 1, scale: 1 }}
                                    transition={{ delay: i * 0.05 }}
                                >
                                    <div className={cn(
                                        kpi.color === "text-profit" ? "glass-card-profit" : "glass-card",
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

                        {/* Sharpe bar chart for selected strategies */}
                        {chartData.length > 0 && (
                            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.2 }}>
                                <div className="glass-card p-5">
                                    <div className="flex items-center justify-between mb-4">
                                        <div>
                                            <h3 className="text-sm font-semibold">Selected Strategies - Sharpe Ratio</h3>
                                            <p className="text-xs text-muted-foreground mt-0.5">Top qualifying strategies ranked by risk-adjusted return</p>
                                        </div>
                                        <TrendingUp className="h-4 w-4 text-primary" />
                                    </div>
                                    <div className="h-64">
                                        <ResponsiveContainer width="100%" height="100%">
                                            <BarChart data={chartData} margin={{ top: 5, right: 5, left: 5, bottom: 5 }}>
                                                <defs>
                                                    <linearGradient id="simSharpe" x1="0" y1="0" x2="0" y2="1">
                                                        <stop offset="0%" stopColor="hsl(230, 80%, 60%)" stopOpacity={0.8} />
                                                        <stop offset="100%" stopColor="hsl(190, 80%, 50%)" stopOpacity={0.6} />
                                                    </linearGradient>
                                                </defs>
                                                <XAxis dataKey="name" stroke="hsl(215, 20%, 30%)" fontSize={10} tickLine={false} axisLine={false} angle={-25} textAnchor="end" height={50} />
                                                <YAxis stroke="hsl(215, 20%, 30%)" fontSize={10} tickLine={false} axisLine={false} />
                                                <Tooltip content={<ChartTooltip />} />
                                                <Bar dataKey="sharpe" fill="url(#simSharpe)" radius={[6, 6, 0, 0]} />
                                            </BarChart>
                                        </ResponsiveContainer>
                                    </div>
                                </div>
                            </motion.div>
                        )}

                        {/* Selected Strategies table */}
                        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.25 }}>
                            <div className="glass-card p-5">
                                <div className="flex items-center justify-between mb-4">
                                    <div>
                                        <h3 className="text-sm font-semibold flex items-center gap-2">
                                            <Zap className="h-4 w-4 text-profit" />
                                            Selected Strategies
                                        </h3>
                                        <p className="text-xs text-muted-foreground mt-0.5">{selectedResults.length} strategies passed qualification filters</p>
                                    </div>
                                </div>
                                <div className="overflow-x-auto">
                                    <table className="w-full text-xs">
                                        <thead>
                                            <tr className="border-b border-border/30 text-muted-foreground">
                                                <th scope="col" className="py-2 px-3 text-left font-medium">Rank</th>
                                                <th scope="col" className="py-2 px-3 text-left font-medium">Strategy</th>
                                                <th scope="col" className="py-2 px-3 text-left font-medium">Params</th>
                                                <th scope="col" className="py-2 px-3 text-left font-medium">Interval</th>
                                                <th scope="col" className="py-2 px-3 text-right font-medium">Sharpe</th>
                                                <th scope="col" className="py-2 px-3 text-right font-medium">Sortino</th>
                                                <th scope="col" className="py-2 px-3 text-right font-medium">Max DD</th>
                                                <th scope="col" className="py-2 px-3 text-right font-medium">Win Rate</th>
                                                <th scope="col" className="py-2 px-3 text-right font-medium">PF</th>
                                                <th scope="col" className="py-2 px-3 text-right font-medium">Trades</th>
                                                <th scope="col" className="py-2 px-3 text-center font-medium">Status</th>
                                            </tr>
                                        </thead>
                                        <tbody>
                                            {selectedResults.map((r, i) => (
                                                <motion.tr
                                                    key={`${r.strategy_id}-${r.rank}`}
                                                    initial={{ opacity: 0, x: -10 }}
                                                    animate={{ opacity: 1, x: 0 }}
                                                    transition={{ delay: i * 0.03 }}
                                                    className="border-b border-border/10 hover:bg-muted/20 transition-colors"
                                                >
                                                    <td className="py-2 px-3 font-mono font-semibold text-muted-foreground">#{r.rank}</td>
                                                    <td className="py-2 px-3 font-medium">{r.strategy_id}</td>
                                                    <ParamsCell params={r.params} />
                                                    <td className="py-2 px-3">
                                                        <span className="inline-flex items-center rounded-full px-2 py-0.5 text-[10px] font-bold bg-primary/10 text-primary">
                                                            {r.interval}
                                                        </span>
                                                    </td>
                                                    <td className={cn("py-2 px-3 text-right font-mono font-semibold", r.sharpe >= 1.5 ? "text-profit" : r.sharpe >= 1.0 ? "text-primary" : "text-muted-foreground")}>
                                                        {r.sharpe.toFixed(2)}
                                                    </td>
                                                    <td className={cn("py-2 px-3 text-right font-mono", r.sortino >= 2.0 ? "text-profit" : "text-muted-foreground")}>
                                                        {r.sortino.toFixed(2)}
                                                    </td>
                                                    <td className={cn("py-2 px-3 text-right font-mono", r.max_dd > -10 ? "text-profit" : "text-loss")}>
                                                        {r.max_dd.toFixed(1)}%
                                                    </td>
                                                    <td className={cn("py-2 px-3 text-right font-mono font-semibold", r.win_rate >= 55 ? "text-profit" : "text-muted-foreground")}>
                                                        {r.win_rate.toFixed(1)}%
                                                    </td>
                                                    <td className={cn("py-2 px-3 text-right font-mono", r.profit_factor >= 1.5 ? "text-profit" : "text-muted-foreground")}>
                                                        {r.profit_factor.toFixed(2)}
                                                    </td>
                                                    <td className="py-2 px-3 text-right font-mono">{r.trades}</td>
                                                    <td className="py-2 px-3 text-center">
                                                        {r.selected && (
                                                            <span className="inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-[10px] font-bold bg-profit/10 text-profit">
                                                                <CheckCircle2 className="h-2.5 w-2.5" />
                                                                Selected
                                                            </span>
                                                        )}
                                                    </td>
                                                </motion.tr>
                                            ))}
                                        </tbody>
                                    </table>
                                </div>
                            </div>
                        </motion.div>

                        {/* Full results (expandable) */}
                        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.3 }}>
                            <div className="glass-card p-5">
                                <button
                                    onClick={() => setShowFullResults(!showFullResults)}
                                    className="w-full flex items-center justify-between"
                                    aria-expanded={showFullResults}
                                    aria-label="Toggle full results"
                                >
                                    <div>
                                        <h3 className="text-sm font-semibold text-left">Full Results</h3>
                                        <p className="text-xs text-muted-foreground mt-0.5">All {totalTested.toLocaleString()} tested permutations ranked by Sharpe</p>
                                    </div>
                                    {showFullResults ? (
                                        <ChevronUp className="h-4 w-4 text-muted-foreground" />
                                    ) : (
                                        <ChevronDown className="h-4 w-4 text-muted-foreground" />
                                    )}
                                </button>

                                <AnimatePresence>
                                    {showFullResults && (
                                        <motion.div
                                            initial={{ height: 0, opacity: 0 }}
                                            animate={{ height: "auto", opacity: 1 }}
                                            exit={{ height: 0, opacity: 0 }}
                                            transition={{ duration: 0.3 }}
                                            className="overflow-hidden"
                                        >
                                            {fullFetching ? (
                                                <div className="py-8 text-center">
                                                    <Loader2 className="h-6 w-6 animate-spin text-primary mx-auto mb-2" />
                                                    <p className="text-xs text-muted-foreground">Loading full results...</p>
                                                </div>
                                            ) : (
                                                <div className="overflow-x-auto mt-4">
                                                    <table className="w-full text-xs">
                                                        <thead>
                                                            <tr className="border-b border-border/30 text-muted-foreground">
                                                                <th scope="col" className="py-2 px-3 text-left font-medium">Rank</th>
                                                                <th scope="col" className="py-2 px-3 text-left font-medium">Strategy</th>
                                                                <th scope="col" className="py-2 px-3 text-left font-medium">Params</th>
                                                                <th scope="col" className="py-2 px-3 text-left font-medium">Interval</th>
                                                                <th scope="col" className="py-2 px-3 text-right font-medium">Sharpe</th>
                                                                <th scope="col" className="py-2 px-3 text-right font-medium">Sortino</th>
                                                                <th scope="col" className="py-2 px-3 text-right font-medium">Max DD</th>
                                                                <th scope="col" className="py-2 px-3 text-right font-medium">Win Rate</th>
                                                                <th scope="col" className="py-2 px-3 text-right font-medium">PF</th>
                                                                <th scope="col" className="py-2 px-3 text-right font-medium">Trades</th>
                                                                <th scope="col" className="py-2 px-3 text-center font-medium">Status</th>
                                                            </tr>
                                                        </thead>
                                                        <tbody>
                                                            {fullResults.map((r, i) => (
                                                                <tr
                                                                    key={`full-${r.strategy_id}-${r.rank}`}
                                                                    className={cn(
                                                                        "border-b border-border/10 hover:bg-muted/20 transition-colors",
                                                                        r.selected && "bg-profit/5"
                                                                    )}
                                                                >
                                                                    <td className="py-2 px-3 font-mono font-semibold text-muted-foreground">#{r.rank}</td>
                                                                    <td className="py-2 px-3 font-medium">{r.strategy_id}</td>
                                                                    <ParamsCell params={r.params} />
                                                                    <td className="py-2 px-3">
                                                                        <span className="inline-flex items-center rounded-full px-2 py-0.5 text-[10px] font-bold bg-primary/10 text-primary">
                                                                            {r.interval}
                                                                        </span>
                                                                    </td>
                                                                    <td className={cn("py-2 px-3 text-right font-mono font-semibold", r.sharpe >= 1.5 ? "text-profit" : r.sharpe >= 1.0 ? "text-primary" : "text-muted-foreground")}>
                                                                        {r.sharpe.toFixed(2)}
                                                                    </td>
                                                                    <td className={cn("py-2 px-3 text-right font-mono", r.sortino >= 2.0 ? "text-profit" : "text-muted-foreground")}>
                                                                        {r.sortino.toFixed(2)}
                                                                    </td>
                                                                    <td className={cn("py-2 px-3 text-right font-mono", r.max_dd > -10 ? "text-profit" : "text-loss")}>
                                                                        {r.max_dd.toFixed(1)}%
                                                                    </td>
                                                                    <td className={cn("py-2 px-3 text-right font-mono font-semibold", r.win_rate >= 55 ? "text-profit" : "text-muted-foreground")}>
                                                                        {r.win_rate.toFixed(1)}%
                                                                    </td>
                                                                    <td className={cn("py-2 px-3 text-right font-mono", r.profit_factor >= 1.5 ? "text-profit" : "text-muted-foreground")}>
                                                                        {r.profit_factor.toFixed(2)}
                                                                    </td>
                                                                    <td className="py-2 px-3 text-right font-mono">{r.trades}</td>
                                                                    <td className="py-2 px-3 text-center">
                                                                        {r.selected ? (
                                                                            <span className="inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-[10px] font-bold bg-profit/10 text-profit">
                                                                                <CheckCircle2 className="h-2.5 w-2.5" />
                                                                                Selected
                                                                            </span>
                                                                        ) : (
                                                                            <span className="text-[10px] text-muted-foreground">--</span>
                                                                        )}
                                                                    </td>
                                                                </tr>
                                                            ))}
                                                        </tbody>
                                                    </table>
                                                </div>
                                            )}
                                        </motion.div>
                                    )}
                                </AnimatePresence>
                            </div>
                        </motion.div>
                    </motion.div>
                )}

                {/* No results yet, not running */}
                {!isRunning && !hasResults && (
                    <motion.div
                        key="empty"
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                    >
                        <div className="glass-card p-10 text-center">
                            <Moon className="h-8 w-8 text-muted-foreground mx-auto mb-3 opacity-50" />
                            <p className="text-sm font-medium">No simulation results yet</p>
                            <p className="text-xs text-muted-foreground mt-1">
                                Run the nightly simulation to discover optimal strategy configurations
                            </p>
                        </div>
                    </motion.div>
                )}

                {/* Mutation error */}
                {runMutation.isError && (
                    <motion.div
                        key="error"
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                    >
                        <div className="glass-card-loss p-5 text-center">
                            <p className="text-sm text-loss font-medium">Simulation failed to start</p>
                            <p className="text-xs text-muted-foreground mt-1">{(runMutation.error as Error)?.message || "Unknown error"}</p>
                        </div>
                    </motion.div>
                )}
            </AnimatePresence>
        </div>
    );
}
