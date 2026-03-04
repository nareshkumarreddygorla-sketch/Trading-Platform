"use client";

import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { motion } from "framer-motion";
import { cn, formatCurrency } from "@/lib/utils";
import { endpoints } from "@/lib/api/client";
import {
  History, ArrowUpRight, ArrowDownRight, Filter,
  RefreshCw, Clock, CheckCircle, XCircle, Loader2,
} from "lucide-react";

type Order = {
  order_id: string;
  broker_order_id?: string;
  strategy_id: string;
  symbol: string;
  exchange: string;
  side: string;
  quantity: number;
  order_type: string;
  limit_price?: number;
  status: string;
  filled_qty: number;
  avg_price?: number;
  ts?: string;
};

const STATUS_STYLES: Record<string, { color: string; icon: typeof CheckCircle }> = {
  FILLED: { color: "text-profit bg-profit/10", icon: CheckCircle },
  CANCELLED: { color: "text-loss bg-loss/10", icon: XCircle },
  PENDING: { color: "text-warning bg-warning/10", icon: Loader2 },
  NEW: { color: "text-warning bg-warning/10", icon: Loader2 },
  ACK: { color: "text-primary bg-primary/10", icon: Clock },
  PARTIAL: { color: "text-primary bg-primary/10", icon: Clock },
  REJECTED: { color: "text-loss bg-loss/10", icon: XCircle },
};

export default function TradesPage() {
  const [statusFilter, setStatusFilter] = useState<string>("");
  const [limit, setLimit] = useState(50);

  const { data, isLoading, refetch } = useQuery({
    queryKey: ["orders", statusFilter, limit],
    queryFn: () => endpoints.orders(limit, statusFilter || undefined),
    refetchInterval: 5000,
  });

  const orders = (data?.orders ?? []) as Order[];
  const total = data?.total ?? 0;

  const filledCount = orders.filter((o) => o.status === "FILLED").length;
  const cancelledCount = orders.filter((o) => o.status === "CANCELLED" || o.status === "REJECTED").length;
  const pendingCount = orders.filter((o) => ["PENDING", "NEW", "ACK", "PARTIAL"].includes(o.status)).length;
  const totalValue = orders
    .filter((o) => o.status === "FILLED")
    .reduce((acc, o) => acc + (o.avg_price ?? o.limit_price ?? 0) * o.filled_qty, 0);

  return (
    <div className="space-y-6 pb-8">
      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: -10 }}
        animate={{ opacity: 1, y: 0 }}
        className="flex items-center justify-between"
      >
        <div>
          <h1 className="text-2xl font-bold tracking-tight">Trade History</h1>
          <p className="text-sm text-muted-foreground mt-0.5">
            {total} total orders
          </p>
        </div>
        <button
          onClick={() => refetch()}
          className="flex items-center gap-2 glass-card px-3 py-2 rounded-xl text-xs font-medium text-muted-foreground hover:text-primary transition-colors"
        >
          <RefreshCw className="h-3.5 w-3.5" />
          Refresh
        </button>
      </motion.div>

      {/* KPI cards */}
      <div className="grid gap-4 md:grid-cols-4">
        {[
          { label: "Filled Orders", value: String(filledCount), icon: CheckCircle, color: "text-profit" },
          { label: "Pending", value: String(pendingCount), icon: Clock, color: "text-warning" },
          { label: "Cancelled/Rejected", value: String(cancelledCount), icon: XCircle, color: "text-loss" },
          { label: "Total Volume", value: formatCurrency(totalValue), icon: History, color: "text-primary" },
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

      {/* Filters */}
      <div className="flex items-center gap-2">
        <Filter className="h-4 w-4 text-muted-foreground" />
        {["", "FILLED", "PENDING", "CANCELLED"].map((status) => (
          <button
            key={status}
            onClick={() => setStatusFilter(status)}
            className={cn(
              "rounded-lg px-3 py-1.5 text-[11px] font-semibold transition-all",
              statusFilter === status
                ? "bg-primary/15 text-primary"
                : "text-muted-foreground hover:text-foreground hover:bg-muted/30"
            )}
          >
            {status || "All"}
          </button>
        ))}
      </div>

      {/* Orders table */}
      <motion.div
        initial={{ opacity: 0, y: 16 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2 }}
      >
        <div className="glass-card overflow-hidden">
          <div className="flex items-center justify-between border-b border-border/30 p-5">
            <div>
              <h3 className="text-sm font-semibold">Order Book</h3>
              <p className="text-xs text-muted-foreground mt-0.5">Auto-refreshes every 5s</p>
            </div>
            <History className="h-4 w-4 text-primary" />
          </div>

          {isLoading ? (
            <div className="p-5 space-y-3">
              {Array.from({ length: 5 }).map((_, i) => (
                <div key={i} className="shimmer h-14 w-full rounded-lg" />
              ))}
            </div>
          ) : orders.length === 0 ? (
            <div className="flex flex-col items-center justify-center py-16 text-muted-foreground">
              <History className="h-10 w-10 mb-3 opacity-30" />
              <p className="text-sm font-medium">No orders found</p>
              <p className="text-xs mt-1">Orders will appear when strategies execute trades</p>
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="border-b border-border/20">
                    {["Time", "Symbol", "Side", "Type", "Qty", "Price", "Filled", "Status", "Strategy"].map((h) => (
                      <th key={h} className="px-5 py-3 text-left text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
                        {h}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {orders.map((o, i) => {
                    const statusStyle = STATUS_STYLES[o.status] || STATUS_STYLES.PENDING;
                    const StatusIcon = statusStyle.icon;
                    return (
                      <motion.tr
                        key={o.order_id || `${o.symbol}-${i}`}
                        initial={{ opacity: 0, x: -10 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ delay: 0.05 + i * 0.02 }}
                        className="border-b border-border/10 transition-colors hover:bg-muted/20"
                      >
                        <td className="px-5 py-3 text-xs text-muted-foreground font-mono">
                          {o.ts ? new Date(o.ts).toLocaleString("en-IN", { hour: "2-digit", minute: "2-digit", day: "2-digit", month: "short" }) : "—"}
                        </td>
                        <td className="px-5 py-3">
                          <span className="font-mono text-sm font-semibold">{o.symbol}</span>
                        </td>
                        <td className="px-5 py-3">
                          <span className={cn(
                            "inline-flex items-center gap-1 rounded-md px-2 py-0.5 text-[10px] font-bold",
                            o.side === "BUY" ? "bg-profit/10 text-profit" : "bg-loss/10 text-loss"
                          )}>
                            {o.side === "BUY" ? <ArrowUpRight className="h-3 w-3" /> : <ArrowDownRight className="h-3 w-3" />}
                            {o.side}
                          </span>
                        </td>
                        <td className="px-5 py-3 text-xs text-muted-foreground">{o.order_type}</td>
                        <td className="px-5 py-3 font-mono text-sm">{o.quantity}</td>
                        <td className="px-5 py-3 font-mono text-sm text-muted-foreground">
                          {(o.avg_price ?? o.limit_price ?? 0).toFixed(2)}
                        </td>
                        <td className="px-5 py-3 font-mono text-sm">
                          {o.filled_qty}/{o.quantity}
                        </td>
                        <td className="px-5 py-3">
                          <span className={cn(
                            "inline-flex items-center gap-1 rounded-md px-2 py-0.5 text-[10px] font-bold",
                            statusStyle.color
                          )}>
                            <StatusIcon className="h-3 w-3" />
                            {o.status}
                          </span>
                        </td>
                        <td className="px-5 py-3 text-xs text-muted-foreground truncate max-w-[100px]">
                          {o.strategy_id || "—"}
                        </td>
                      </motion.tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}

          {/* Load more */}
          {orders.length >= limit && (
            <div className="p-4 text-center border-t border-border/20">
              <button
                onClick={() => setLimit((l) => l + 50)}
                className="text-xs font-medium text-primary hover:text-primary/80 transition-colors"
              >
                Load more orders...
              </button>
            </div>
          )}
        </div>
      </motion.div>
    </div>
  );
}
