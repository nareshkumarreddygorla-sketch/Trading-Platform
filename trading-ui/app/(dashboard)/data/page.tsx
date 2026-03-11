"use client";

import { useState, useMemo } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { motion } from "framer-motion";
import { cn } from "@/lib/utils";
import { endpoints } from "@/lib/api/client";
import {
  Database, RefreshCw, CheckCircle, XCircle, Activity,
  BarChart3, Search, ChevronDown, Layers, Server,
} from "lucide-react";
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from "recharts";

type SymbolInfo = { symbol: string; bars: number; first_date: string; last_date: string };
type Bar = Record<string, unknown>;

const INTERVALS = ["1m", "5m", "15m", "1h", "1d"] as const;

export default function DataPipelinePage() {
  const queryClient = useQueryClient();
  const [selectedSymbol, setSelectedSymbol] = useState("");
  const [selectedInterval, setSelectedInterval] = useState<string>("1d");
  const [symbolSearch, setSymbolSearch] = useState("");

  // ── Queries ──────────────────────────────────────────────
  const { data: quality, isLoading: qualityLoading } = useQuery({
    queryKey: ["data-quality"],
    queryFn: () => endpoints.dataQuality(),
    refetchInterval: 30000,
  });

  const { data: symbolsData, isLoading: symbolsLoading } = useQuery({
    queryKey: ["data-symbols"],
    queryFn: () => endpoints.dataSymbols("1d", 50),
    refetchInterval: 30000,
  });

  const { data: allSymbolsData } = useQuery({
    queryKey: ["data-symbols-all"],
    queryFn: () => endpoints.dataSymbols("1d", 0),
    refetchInterval: 60000,
  });

  const { data: instrumentMap, isLoading: instrumentLoading } = useQuery({
    queryKey: ["data-instrument-map"],
    queryFn: () => endpoints.dataInstrumentMap(),
    refetchInterval: 60000,
  });

  const { data: barsData, isLoading: barsLoading } = useQuery({
    queryKey: ["data-bars", selectedSymbol, selectedInterval],
    queryFn: () => endpoints.dataBars(selectedSymbol, selectedInterval, 100),
    enabled: !!selectedSymbol,
  });

  // ── Refresh mutation ─────────────────────────────────────
  const refreshMutation = useMutation({
    mutationFn: () => endpoints.dataRefresh(),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["data-quality"] });
      queryClient.invalidateQueries({ queryKey: ["data-symbols"] });
      queryClient.invalidateQueries({ queryKey: ["data-symbols-all"] });
      queryClient.invalidateQueries({ queryKey: ["data-instrument-map"] });
    },
  });

  // ── Derived state ────────────────────────────────────────
  const q = quality?.quality as Record<string, unknown> | undefined;
  const symbolCount = (allSymbolsData?.symbols ?? []).length;
  const qualityGood = symbolCount >= 10;

  const symbols: SymbolInfo[] = allSymbolsData?.symbols ?? [];
  const filteredSymbols = useMemo(() => {
    if (!symbolSearch) return symbols;
    const lower = symbolSearch.toLowerCase();
    return symbols.filter((s) => s.symbol.toLowerCase().includes(lower));
  }, [symbols, symbolSearch]);

  const bars: Bar[] = barsData?.bars ?? [];
  const last20 = bars.slice(-20);

  const chartData = useMemo(() => {
    return bars.map((b, i) => ({
      idx: i,
      date: (b.date as string) ?? (b.timestamp as string) ?? String(i),
      close: Number(b.close ?? b.Close ?? 0),
    }));
  }, [bars]);

  // Color coding helper for symbol badges
  const barColor = (count: number) => {
    if (count >= 200) return "bg-profit/15 text-profit border-profit/30";
    if (count >= 50) return "bg-warning/15 text-warning border-warning/30";
    return "bg-loss/15 text-loss border-loss/30";
  };

  return (
    <div className="space-y-6 pb-8">
      {/* Header */}
      <motion.div initial={{ opacity: 0, y: -10 }} animate={{ opacity: 1, y: 0 }}>
        <div className="flex items-center gap-3">
          <Database className="h-7 w-7 text-primary" />
          <div>
            <h1 className="text-2xl font-bold tracking-tight">Data Pipeline</h1>
            <p className="text-sm text-muted-foreground mt-0.5">
              Market data quality, symbol coverage & historical data viewer
            </p>
          </div>
        </div>
      </motion.div>

      {/* Data Quality + Instrument Map */}
      <div className="grid gap-6 xl:grid-cols-2">
        {/* Data Quality Card */}
        <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.1 }}>
          <div className={cn(qualityGood ? "glass-card-profit" : "glass-card-loss", "p-6")}>
            <div className="flex items-start justify-between mb-4">
              <div className="flex items-center gap-3">
                <div className={cn(
                  "flex h-12 w-12 items-center justify-center rounded-xl",
                  qualityGood ? "bg-profit/10" : "bg-loss/10"
                )}>
                  <BarChart3 className={cn("h-6 w-6", qualityGood ? "text-profit" : "text-loss")} />
                </div>
                <div>
                  <h3 className="text-lg font-semibold">Data Quality</h3>
                  <p className="text-xs text-muted-foreground mt-0.5">Historical market data status</p>
                </div>
              </div>
              {qualityGood ? (
                <span className="status-badge-live">
                  <CheckCircle className="h-3 w-3" />
                  GOOD
                </span>
              ) : (
                <span className="status-badge-danger">
                  <XCircle className="h-3 w-3" />
                  INSUFFICIENT
                </span>
              )}
            </div>

            <div className="grid grid-cols-2 gap-3 mt-4">
              {[
                { label: "Symbols with Data", value: String(symbolCount) },
                { label: "Min Bars Threshold", value: "50" },
                { label: "Symbols (50+ bars)", value: String((symbolsData?.symbols ?? []).length) },
                {
                  label: "Data Freshness",
                  value: q?.last_updated
                    ? new Date(q.last_updated as string).toLocaleDateString("en-IN", { day: "2-digit", month: "short" })
                    : "---",
                },
              ].map((item) => (
                <div key={item.label} className="rounded-lg bg-muted/20 p-3">
                  <div className="text-[10px] text-muted-foreground">{item.label}</div>
                  <div className="font-mono text-sm font-semibold mt-0.5">{item.value}</div>
                </div>
              ))}
            </div>

            <div className="mt-4">
              <button
                onClick={() => refreshMutation.mutate()}
                disabled={refreshMutation.isPending}
                aria-label={refreshMutation.isPending ? "Refreshing data" : "Refresh data"}
                aria-busy={refreshMutation.isPending}
                className={cn(
                  "w-full flex items-center justify-center gap-2 rounded-xl border border-primary/30 py-2.5 text-xs font-medium text-primary transition-all hover:bg-primary/5",
                  refreshMutation.isPending && "opacity-50 cursor-not-allowed"
                )}
              >
                <RefreshCw className={cn("h-3.5 w-3.5", refreshMutation.isPending && "animate-spin")} />
                {refreshMutation.isPending ? "Refreshing..." : "Refresh Data"}
              </button>
            </div>
          </div>
        </motion.div>

        {/* Instrument Map Card */}
        <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.15 }}>
          <div className={cn(
            instrumentMap?.loaded ? "glass-card-profit" : "glass-card",
            "p-6"
          )}>
            <div className="flex items-start justify-between mb-4">
              <div className="flex items-center gap-3">
                <div className={cn(
                  "flex h-12 w-12 items-center justify-center rounded-xl",
                  instrumentMap?.loaded ? "bg-profit/10" : "bg-muted/20"
                )}>
                  <Server className={cn("h-6 w-6", instrumentMap?.loaded ? "text-profit" : "text-muted-foreground")} />
                </div>
                <div>
                  <h3 className="text-lg font-semibold">Instrument Map</h3>
                  <p className="text-xs text-muted-foreground mt-0.5">Exchange instrument metadata</p>
                </div>
              </div>
              {instrumentMap?.loaded ? (
                <span className="status-badge-live">
                  <CheckCircle className="h-3 w-3" />
                  LOADED
                </span>
              ) : (
                <span className="text-xs font-medium text-muted-foreground bg-muted/30 px-3 py-1 rounded-full">
                  NOT LOADED
                </span>
              )}
            </div>

            <div className="grid grid-cols-2 gap-3 mt-4">
              {[
                { label: "Total Instruments", value: String(instrumentMap?.total_instruments ?? 0) },
                { label: "NSE Equity", value: String(instrumentMap?.nse_equity_count ?? 0) },
                { label: "Status", value: instrumentMap?.loaded ? "LOADED" : "PENDING" },
                {
                  label: "Last Updated",
                  value: instrumentMap?.last_updated
                    ? new Date(instrumentMap.last_updated).toLocaleDateString("en-IN", { day: "2-digit", month: "short" })
                    : "---",
                },
              ].map((item) => (
                <div key={item.label} className="rounded-lg bg-muted/20 p-3">
                  <div className="text-[10px] text-muted-foreground">{item.label}</div>
                  <div className="font-mono text-sm font-semibold mt-0.5">{item.value}</div>
                </div>
              ))}
            </div>
          </div>
        </motion.div>
      </div>

      {/* Symbol Coverage */}
      <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.2 }}>
        <div className="glass-card p-6">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-3">
              <Layers className="h-5 w-5 text-primary" />
              <div>
                <h3 className="text-sm font-semibold">Symbol Coverage</h3>
                <p className="text-xs text-muted-foreground mt-0.5">
                  {symbols.length} symbols tracked &middot;
                  <span className="text-profit ml-1">Green 200+</span>
                  <span className="text-warning ml-1">Yellow 50-200</span>
                  <span className="text-loss ml-1">Red &lt;50</span>
                </p>
              </div>
            </div>
            <div className="relative">
              <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-muted-foreground" />
              <input
                type="text"
                value={symbolSearch}
                onChange={(e) => setSymbolSearch(e.target.value)}
                placeholder="Filter symbols..."
                className="w-40 rounded-lg border border-border/50 bg-muted/10 pl-8 pr-3 py-1.5 text-xs focus:border-primary/50 focus:outline-none"
              />
            </div>
          </div>

          {symbolsLoading ? (
            <div className="flex items-center justify-center py-12 text-muted-foreground">
              <RefreshCw className="h-5 w-5 animate-spin mr-2" />
              <span className="text-sm">Loading symbols...</span>
            </div>
          ) : filteredSymbols.length === 0 ? (
            <div className="flex flex-col items-center justify-center py-12 text-muted-foreground">
              <Activity className="h-10 w-10 mb-3 opacity-30" />
              <p className="text-sm font-medium">No symbols found</p>
              <p className="text-xs mt-1">Run a data refresh to populate market data</p>
            </div>
          ) : (
            <div className="flex flex-wrap gap-2 max-h-64 overflow-y-auto">
              {filteredSymbols.map((s) => (
                <button
                  key={s.symbol}
                  onClick={() => setSelectedSymbol(s.symbol)}
                  className={cn(
                    "inline-flex items-center gap-1.5 rounded-lg border px-2.5 py-1 text-xs font-mono font-medium transition-all hover:scale-105",
                    barColor(s.bars),
                    selectedSymbol === s.symbol && "ring-2 ring-primary/50"
                  )}
                >
                  {s.symbol}
                  <span className="text-[10px] opacity-70">{s.bars}</span>
                </button>
              ))}
            </div>
          )}
        </div>
      </motion.div>

      {/* Historical Data Viewer */}
      <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.25 }}>
        <div className="glass-card overflow-hidden">
          <div className="flex items-center justify-between border-b border-border/30 p-5">
            <div className="flex items-center gap-3">
              <BarChart3 className="h-5 w-5 text-primary" />
              <div>
                <h3 className="text-sm font-semibold">Historical Data Viewer</h3>
                <p className="text-xs text-muted-foreground mt-0.5">
                  {selectedSymbol ? `Viewing ${selectedSymbol} (${selectedInterval})` : "Select a symbol to view data"}
                </p>
              </div>
            </div>

            <div className="flex items-center gap-2">
              {/* Symbol input */}
              <div className="relative">
                <input
                  type="text"
                  value={selectedSymbol}
                  onChange={(e) => setSelectedSymbol(e.target.value.toUpperCase())}
                  placeholder="Symbol..."
                  className="w-32 rounded-lg border border-border/50 bg-muted/10 px-3 py-1.5 text-xs font-mono focus:border-primary/50 focus:outline-none"
                />
              </div>
              {/* Interval selector */}
              <div className="relative">
                <select
                  value={selectedInterval}
                  onChange={(e) => setSelectedInterval(e.target.value)}
                  className="appearance-none rounded-lg border border-border/50 bg-muted/10 pl-3 pr-7 py-1.5 text-xs font-mono focus:border-primary/50 focus:outline-none"
                >
                  {INTERVALS.map((iv) => (
                    <option key={iv} value={iv}>{iv}</option>
                  ))}
                </select>
                <ChevronDown className="absolute right-2 top-1/2 -translate-y-1/2 h-3 w-3 text-muted-foreground pointer-events-none" />
              </div>
            </div>
          </div>

          {!selectedSymbol ? (
            <div className="flex flex-col items-center justify-center py-16 text-muted-foreground">
              <Database className="h-10 w-10 mb-3 opacity-30" />
              <p className="text-sm font-medium">No symbol selected</p>
              <p className="text-xs mt-1">Click a symbol badge above or type a symbol name</p>
            </div>
          ) : barsLoading ? (
            <div className="flex items-center justify-center py-16 text-muted-foreground">
              <RefreshCw className="h-5 w-5 animate-spin mr-2" />
              <span className="text-sm">Loading bars for {selectedSymbol}...</span>
            </div>
          ) : bars.length === 0 ? (
            <div className="flex flex-col items-center justify-center py-16 text-muted-foreground">
              <Activity className="h-10 w-10 mb-3 opacity-30" />
              <p className="text-sm font-medium">No data available</p>
              <p className="text-xs mt-1">No bars found for {selectedSymbol} at {selectedInterval} interval</p>
            </div>
          ) : (
            <div className="p-5 space-y-6">
              {/* Line Chart */}
              <div className="h-56 w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={chartData} margin={{ top: 5, right: 20, bottom: 5, left: 10 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border) / 0.3)" />
                    <XAxis
                      dataKey="date"
                      tick={{ fontSize: 10, fill: "hsl(var(--muted-foreground))" }}
                      tickFormatter={(v: string) => {
                        if (!v || v.length < 10) return v;
                        return v.slice(5, 10);
                      }}
                      interval="preserveStartEnd"
                    />
                    <YAxis
                      tick={{ fontSize: 10, fill: "hsl(var(--muted-foreground))" }}
                      domain={["auto", "auto"]}
                      tickFormatter={(v: number) => v.toFixed(0)}
                    />
                    <Tooltip
                      contentStyle={{
                        background: "hsl(var(--card))",
                        border: "1px solid hsl(var(--border) / 0.5)",
                        borderRadius: "8px",
                        fontSize: "12px",
                      }}
                      labelFormatter={(label: string) => `Date: ${label}`}
                      formatter={(value: number) => [`${value.toFixed(2)}`, "Close"]}
                    />
                    <Line
                      type="monotone"
                      dataKey="close"
                      stroke="hsl(var(--primary))"
                      strokeWidth={2}
                      dot={false}
                      activeDot={{ r: 4, fill: "hsl(var(--primary))" }}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>

              {/* OHLCV Table */}
              <div>
                <h4 className="text-xs font-semibold text-muted-foreground mb-2">
                  Last {last20.length} Bars
                </h4>
                <div className="overflow-x-auto rounded-lg border border-border/30">
                  <table className="w-full text-xs">
                    <thead>
                      <tr className="border-b border-border/30 bg-muted/10">
                        {["Date", "Open", "High", "Low", "Close", "Volume"].map((h) => (
                          <th key={h} className="px-3 py-2 text-left font-semibold text-muted-foreground whitespace-nowrap">
                            {h}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-border/10">
                      {last20.map((b, i) => {
                        const open = Number(b.open ?? b.Open ?? 0);
                        const close = Number(b.close ?? b.Close ?? 0);
                        const isGreen = close >= open;
                        return (
                          <tr key={i} className="hover:bg-muted/10 transition-colors">
                            <td className="px-3 py-1.5 font-mono text-muted-foreground whitespace-nowrap">
                              {(b.date as string) ?? (b.timestamp as string) ?? "---"}
                            </td>
                            <td className="px-3 py-1.5 font-mono">{open.toFixed(2)}</td>
                            <td className="px-3 py-1.5 font-mono text-profit">
                              {Number(b.high ?? b.High ?? 0).toFixed(2)}
                            </td>
                            <td className="px-3 py-1.5 font-mono text-loss">
                              {Number(b.low ?? b.Low ?? 0).toFixed(2)}
                            </td>
                            <td className={cn("px-3 py-1.5 font-mono font-semibold", isGreen ? "text-profit" : "text-loss")}>
                              {close.toFixed(2)}
                            </td>
                            <td className="px-3 py-1.5 font-mono text-muted-foreground">
                              {Number(b.volume ?? b.Volume ?? 0).toLocaleString()}
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          )}
        </div>
      </motion.div>
    </div>
  );
}
