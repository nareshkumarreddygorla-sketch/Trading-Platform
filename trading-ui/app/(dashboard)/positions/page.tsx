"use client";

import { useState, useEffect } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { motion, AnimatePresence } from "framer-motion";
import { cn, formatCurrency } from "@/lib/utils";
import { endpoints } from "@/lib/api/client";
import type { Position } from "@/types";
import {
  Layers, ArrowUpRight, ArrowDownRight, TrendingUp,
  TrendingDown, DollarSign, Activity, X, Plus,
  Send, BarChart3,
} from "lucide-react";

export default function PositionsPage() {
  const queryClient = useQueryClient();
  const [showOrderForm, setShowOrderForm] = useState(false);
  const [orderSymbol, setOrderSymbol] = useState("");
  const [orderSide, setOrderSide] = useState<"BUY" | "SELL">("BUY");
  const [orderQty, setOrderQty] = useState("");
  const [orderPrice, setOrderPrice] = useState("");
  const [closingSymbol, setClosingSymbol] = useState<string | null>(null);
  const [closeError, setCloseError] = useState<string | null>(null);

  const { data, isLoading, isError, error: fetchError } = useQuery({
    queryKey: ["positions"],
    queryFn: async () => {
      const res = await endpoints.positions();
      return ((res.positions ?? []) as any[]).map((p) => ({
        ...p,
        entry_price: p.entry_price ?? p.avg_price ?? 0,
      })) as Position[];
    },
    refetchInterval: 5000,
  });

  const placeOrderMutation = useMutation({
    mutationFn: (body: { symbol: string; side: string; quantity: number; order_type: string; limit_price?: number }) =>
      endpoints.placeOrder(body),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["positions"] });
      queryClient.invalidateQueries({ queryKey: ["orders"] });
      setShowOrderForm(false);
      setOrderSymbol("");
      setOrderQty("");
      setOrderPrice("");
    },
    onError: (err: Error) => {
      // Error is displayed in the form via placeOrderMutation.isError
    },
  });

  const closePositionMutation = useMutation({
    mutationFn: (body: { symbol: string; side: string; quantity: number; order_type: string }) =>
      endpoints.placeOrder(body),
    onSuccess: (_data, variables) => {
      setClosingSymbol(null);
      setCloseError(null);
      queryClient.invalidateQueries({ queryKey: ["positions"] });
      queryClient.invalidateQueries({ queryKey: ["orders"] });
    },
    onError: (err: Error, variables) => {
      setClosingSymbol(null);
      setCloseError(`Failed to close ${variables.symbol}: ${err.message}`);
      // Auto-dismiss the error after 5 seconds
      setTimeout(() => setCloseError(null), 5000);
    },
  });

  // Auto-dismiss order success message after 3 seconds
  useEffect(() => {
    if (placeOrderMutation.isSuccess) {
      const timer = setTimeout(() => placeOrderMutation.reset(), 3000);
      return () => clearTimeout(timer);
    }
  }, [placeOrderMutation.isSuccess]); // eslint-disable-line react-hooks/exhaustive-deps

  const positions = data ?? [];
  const totalExposure = positions.reduce(
    (acc, p) => acc + (p.quantity * (p.current_price ?? p.entry_price)),
    0
  );
  const totalUnrealizedPnl = positions.reduce((a, p) => a + (p.unrealized_pnl ?? 0), 0);
  const longCount = positions.filter((p) => p.side === "BUY").length;
  const shortCount = positions.filter((p) => p.side === "SELL").length;

  return (
    <div className="space-y-6 pb-8">
      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: -10 }}
        animate={{ opacity: 1, y: 0 }}
        className="flex items-center justify-between"
      >
        <div>
          <h1 className="text-2xl font-bold tracking-tight">Positions</h1>
          <p className="text-sm text-muted-foreground mt-0.5">
            {positions.length} open position{positions.length !== 1 ? "s" : ""}
          </p>
        </div>
        <button
          onClick={() => setShowOrderForm(!showOrderForm)}
          className="flex items-center gap-2 rounded-xl bg-primary/10 border border-primary/30 px-4 py-2 text-xs font-semibold text-primary transition-all hover:bg-primary/20"
          aria-label={showOrderForm ? "Close order form" : "Open order form to place a new order"}
          aria-expanded={showOrderForm}
        >
          <Plus className="h-3.5 w-3.5" />
          Place Order
        </button>
      </motion.div>

      {/* Manual Order Form (G5) */}
      <AnimatePresence>
        {showOrderForm && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: "auto" }}
            exit={{ opacity: 0, height: 0 }}
            className="overflow-hidden"
          >
            <div className="glass-card p-5">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-sm font-semibold">Manual Order Entry</h3>
                <button onClick={() => setShowOrderForm(false)} className="text-muted-foreground hover:text-foreground" aria-label="Close order form">
                  <X className="h-4 w-4" />
                </button>
              </div>
              <div className="grid gap-4 sm:grid-cols-5">
                <div>
                  <label className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground block mb-1">Symbol</label>
                  <input
                    type="text"
                    value={orderSymbol}
                    onChange={(e) => setOrderSymbol(e.target.value.toUpperCase())}
                    placeholder="RELIANCE"
                    className="w-full h-9 rounded-lg border border-border/50 bg-muted/30 px-3 text-sm font-mono focus:border-primary/50 focus:outline-none focus:ring-1 focus:ring-primary/20"
                  />
                </div>
                <div>
                  <label className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground block mb-1">Side</label>
                  <div className="flex gap-1">
                    {(["BUY", "SELL"] as const).map((side) => (
                      <button
                        key={side}
                        onClick={() => setOrderSide(side)}
                        aria-label={`Order side: ${side}`}
                        aria-pressed={orderSide === side}
                        className={cn(
                          "flex-1 h-9 rounded-lg text-xs font-bold transition-all",
                          orderSide === side
                            ? side === "BUY" ? "bg-profit/20 text-profit border border-profit/30" : "bg-loss/20 text-loss border border-loss/30"
                            : "bg-muted/30 text-muted-foreground border border-border/30"
                        )}
                      >
                        {side}
                      </button>
                    ))}
                  </div>
                </div>
                <div>
                  <label className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground block mb-1">Quantity</label>
                  <input
                    type="number"
                    value={orderQty}
                    onChange={(e) => setOrderQty(e.target.value)}
                    placeholder="1"
                    min="1"
                    className="w-full h-9 rounded-lg border border-border/50 bg-muted/30 px-3 text-sm font-mono focus:border-primary/50 focus:outline-none focus:ring-1 focus:ring-primary/20"
                  />
                </div>
                <div>
                  <label className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground block mb-1">Limit Price</label>
                  <input
                    type="number"
                    value={orderPrice}
                    onChange={(e) => setOrderPrice(e.target.value)}
                    placeholder="0.00"
                    step="0.01"
                    className="w-full h-9 rounded-lg border border-border/50 bg-muted/30 px-3 text-sm font-mono focus:border-primary/50 focus:outline-none focus:ring-1 focus:ring-primary/20"
                  />
                </div>
                <div className="flex items-end">
                  <button
                    onClick={() => {
                      if (orderSymbol && orderQty && orderPrice) {
                        placeOrderMutation.mutate({
                          symbol: orderSymbol,
                          side: orderSide,
                          quantity: Number(orderQty),
                          order_type: "LIMIT",
                          limit_price: Number(orderPrice),
                        });
                      }
                    }}
                    disabled={!orderSymbol || !orderQty || !orderPrice || placeOrderMutation.isPending}
                    aria-label={placeOrderMutation.isPending ? "Submitting order" : "Submit order"}
                    aria-busy={placeOrderMutation.isPending}
                    className={cn(
                      "w-full h-9 rounded-lg text-xs font-bold transition-all flex items-center justify-center gap-1.5",
                      orderSymbol && orderQty && orderPrice
                        ? "bg-primary text-white hover:bg-primary/90"
                        : "bg-muted/50 text-muted-foreground cursor-not-allowed"
                    )}
                  >
                    <Send className="h-3.5 w-3.5" />
                    {placeOrderMutation.isPending ? "Submitting..." : "Submit"}
                  </button>
                </div>
              </div>
              {placeOrderMutation.isError && (
                <p className="text-xs text-loss mt-2">{(placeOrderMutation.error as Error)?.message || "Order failed"}</p>
              )}
              {placeOrderMutation.isSuccess && (
                <p className="text-xs text-profit mt-2">Order submitted successfully</p>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* KPI cards */}
      <div className="grid gap-4 md:grid-cols-4">
        {[
          { label: "Total Exposure", value: formatCurrency(totalExposure), icon: DollarSign, color: "text-primary" },
          { label: "Unrealized P&L", value: `${totalUnrealizedPnl >= 0 ? "+" : ""}${formatCurrency(totalUnrealizedPnl)}`, icon: totalUnrealizedPnl >= 0 ? TrendingUp : TrendingDown, color: totalUnrealizedPnl >= 0 ? "text-profit" : "text-loss" },
          { label: "Long", value: String(longCount), icon: ArrowUpRight, color: "text-profit" },
          { label: "Short", value: String(shortCount), icon: ArrowDownRight, color: "text-loss" },
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

      {/* Close position error banner */}
      {closeError && (
        <motion.div
          initial={{ opacity: 0, y: -8 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0 }}
          className="rounded-lg bg-loss/10 border border-loss/30 p-3 flex items-center justify-between"
          role="alert"
        >
          <p className="text-xs text-loss font-medium">{closeError}</p>
          <button
            onClick={() => setCloseError(null)}
            className="text-loss/60 hover:text-loss ml-2"
            aria-label="Dismiss error"
          >
            <X className="h-3.5 w-3.5" />
          </button>
        </motion.div>
      )}

      {/* Fetch error banner */}
      {isError && (
        <div className="rounded-lg bg-loss/10 border border-loss/30 p-3" role="alert">
          <p className="text-xs text-loss font-medium">
            Failed to load positions: {(fetchError as Error)?.message || "Unknown error"}
          </p>
          <p className="text-[10px] text-muted-foreground mt-1">Data will retry automatically every 5 seconds</p>
        </div>
      )}

      {/* Per-stock P&L Breakdown (G7) */}
      {positions.length > 0 && (
        <motion.div
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.15 }}
        >
          <div className="glass-card p-5">
            <div className="flex items-center gap-2 mb-4">
              <BarChart3 className="h-4 w-4 text-primary" />
              <h3 className="text-sm font-semibold">Per-Stock P&L</h3>
            </div>
            <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
              {positions.map((p) => {
                const pnl = p.unrealized_pnl ?? 0;
                const pctChange = p.pct_change ?? 0;
                const exposure = p.quantity * (p.current_price ?? p.entry_price);
                return (
                  <div key={`${p.symbol}-${p.side}`} className="flex items-center gap-3 rounded-xl bg-muted/20 p-3">
                    <div className={cn(
                      "flex h-8 w-8 items-center justify-center rounded-lg text-xs font-bold",
                      p.side === "BUY" ? "bg-profit/10 text-profit" : "bg-loss/10 text-loss"
                    )}>
                      {p.side === "BUY" ? <ArrowUpRight className="h-4 w-4" /> : <ArrowDownRight className="h-4 w-4" />}
                    </div>
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2">
                        <span className="font-mono text-sm font-semibold">{p.symbol}</span>
                        <span className="text-[10px] text-muted-foreground">{p.quantity} qty</span>
                      </div>
                      <div className="text-[10px] text-muted-foreground">
                        Exposure: {formatCurrency(exposure)}
                      </div>
                    </div>
                    <div className="text-right">
                      <div className={cn("font-mono text-sm font-bold", pnl >= 0 ? "text-profit" : "text-loss")}>
                        {pnl >= 0 ? "+" : ""}{formatCurrency(pnl)}
                      </div>
                      <div className={cn("text-[10px] font-mono", pctChange >= 0 ? "text-profit" : "text-loss")}>
                        {pctChange >= 0 ? "+" : ""}{pctChange.toFixed(2)}%
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </motion.div>
      )}

      {/* Positions table */}
      <motion.div
        initial={{ opacity: 0, y: 16 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2 }}
      >
        <div className="glass-card overflow-hidden">
          <div className="flex items-center justify-between border-b border-border/30 p-5">
            <div>
              <h3 className="text-sm font-semibold">Open Positions</h3>
              <p className="text-xs text-muted-foreground mt-0.5">Auto-refreshes every 5s</p>
            </div>
            <Layers className="h-4 w-4 text-primary" />
          </div>

          {isLoading ? (
            <div className="p-5 space-y-3">
              {Array.from({ length: 3 }).map((_, i) => (
                <div key={i} className="shimmer h-14 w-full rounded-lg" />
              ))}
            </div>
          ) : positions.length === 0 ? (
            <div className="flex flex-col items-center justify-center py-16 text-muted-foreground">
              <Layers className="h-10 w-10 mb-3 opacity-30" />
              <p className="text-sm font-medium">No open positions</p>
              <p className="text-xs mt-1">Positions will appear when strategies execute trades</p>
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="border-b border-border/20">
                    {["Symbol", "Side", "Qty", "Entry", "Current", "Unrealized P&L", "%", "Strategy", ""].map((h) => (
                      <th key={h} scope="col" className="px-5 py-3 text-left text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
                        {h}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {positions.map((p, i) => {
                    const pnl = p.unrealized_pnl ?? 0;
                    const pctChange = p.pct_change ?? 0;
                    return (
                      <motion.tr
                        key={`${p.symbol}-${p.side}`}
                        initial={{ opacity: 0, x: -10 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ delay: 0.2 + i * 0.03 }}
                        className="border-b border-border/10 transition-colors hover:bg-muted/20"
                      >
                        <td className="px-5 py-3">
                          <span className="font-mono text-sm font-semibold">{p.symbol}</span>
                        </td>
                        <td className="px-5 py-3">
                          <span className={cn(
                            "inline-flex items-center gap-1 rounded-md px-2 py-0.5 text-[10px] font-bold",
                            p.side === "BUY"
                              ? "bg-profit/10 text-profit"
                              : "bg-loss/10 text-loss"
                          )}>
                            {p.side === "BUY" ? <ArrowUpRight className="h-3 w-3" /> : <ArrowDownRight className="h-3 w-3" />}
                            {p.side}
                          </span>
                        </td>
                        <td className="px-5 py-3 font-mono text-sm">{p.quantity}</td>
                        <td className="px-5 py-3 font-mono text-sm text-muted-foreground">{(p.entry_price ?? 0).toFixed(2)}</td>
                        <td className="px-5 py-3 font-mono text-sm">{((p.current_price ?? p.entry_price) ?? 0).toFixed(2)}</td>
                        <td className={cn("px-5 py-3 font-mono text-sm font-semibold", pnl >= 0 ? "text-profit" : "text-loss")}>
                          {pnl >= 0 ? "+" : ""}{formatCurrency(pnl)}
                        </td>
                        <td className={cn("px-5 py-3 font-mono text-xs", pctChange >= 0 ? "text-profit" : "text-loss")}>
                          {pctChange >= 0 ? "+" : ""}{pctChange.toFixed(2)}%
                        </td>
                        <td className="px-5 py-3 text-xs text-muted-foreground">{p.strategy_id ?? "—"}</td>
                        <td className="px-5 py-3">
                          <button
                            onClick={() => {
                              if (closingSymbol) return; // prevent double-click
                              const confirmed = window.confirm(
                                `Close ${p.symbol} position (${p.quantity} ${p.side === "BUY" ? "long" : "short"} shares)?`
                              );
                              if (!confirmed) return;
                              setClosingSymbol(`${p.symbol}-${p.side}`);
                              closePositionMutation.mutate({
                                symbol: p.symbol,
                                side: p.side === "BUY" ? "SELL" : "BUY",
                                quantity: p.quantity,
                                order_type: "MARKET",
                              });
                            }}
                            disabled={closingSymbol === `${p.symbol}-${p.side}`}
                            className={cn(
                              "flex h-7 w-7 items-center justify-center rounded-lg transition-all",
                              closingSymbol === `${p.symbol}-${p.side}`
                                ? "text-muted-foreground cursor-not-allowed animate-pulse"
                                : "text-muted-foreground hover:bg-loss/10 hover:text-loss"
                            )}
                            title="Close position"
                            aria-label={closingSymbol === `${p.symbol}-${p.side}` ? `Closing ${p.symbol} position` : `Close ${p.symbol} position`}
                          >
                            <X className="h-3.5 w-3.5" />
                          </button>
                        </td>
                      </motion.tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </motion.div>
    </div>
  );
}
