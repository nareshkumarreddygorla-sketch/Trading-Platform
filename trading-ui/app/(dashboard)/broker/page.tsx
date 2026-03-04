"use client";

import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { motion } from "framer-motion";
import { cn } from "@/lib/utils";
import { useStore } from "@/store/useStore";
import { endpoints } from "@/lib/api/client";
import {
  Plug, CheckCircle, XCircle, Radio, Wifi,
  WifiOff, Settings, RefreshCw, Clock,
  ArrowUpRight, ArrowDownRight, Activity,
} from "lucide-react";
import Link from "next/link";

type RecentOrder = {
  order_id: string;
  symbol: string;
  side: string;
  quantity: number;
  status: string;
  avg_price?: number;
  filled_qty: number;
  ts?: string;
};

export default function BrokerPage() {
  const broker = useStore((s) => s.broker);
  const marketFeed = useStore((s) => s.marketFeed);
  const tradingMode = useStore((s) => s.tradingMode);
  const [reconnecting, setReconnecting] = useState(false);

  const isConnected = broker.status === "connected";

  // Fetch real risk snapshot for position/order counts
  const { data: risk } = useQuery({
    queryKey: ["risk-broker"],
    queryFn: () => endpoints.risk(),
    refetchInterval: 5000,
  });

  // Fetch market status from API
  const { refetch: refetchMarket } = useQuery({
    queryKey: ["market-status-broker"],
    queryFn: () => endpoints.health(),
    refetchInterval: 10000,
  });

  // Fetch recent orders
  const { data: ordersData } = useQuery({
    queryKey: ["orders-broker"],
    queryFn: () => endpoints.orders(10),
    refetchInterval: 5000,
  });

  const openPositions = (risk?.positions as unknown[])?.length ?? 0;
  const recentOrders = (ordersData?.orders ?? []) as RecentOrder[];

  const handleReconnect = async () => {
    setReconnecting(true);
    try {
      await endpoints.health();
      await refetchMarket();
      useStore.setState({
        broker: { connected: true, status: "connected" },
        marketFeed: { connected: true, healthy: true, last_tick_ts: new Date().toISOString() },
      });
    } catch {
      useStore.setState({
        broker: { connected: false, status: "disconnected" },
      });
    } finally {
      setReconnecting(false);
    }
  };

  return (
    <div className="space-y-6 pb-8">
      <motion.div initial={{ opacity: 0, y: -10 }} animate={{ opacity: 1, y: 0 }}>
        <h1 className="text-2xl font-bold tracking-tight">Broker</h1>
        <p className="text-sm text-muted-foreground mt-0.5">Connection status & market data feed</p>
      </motion.div>

      {/* Connection cards */}
      <div className="grid gap-6 xl:grid-cols-2">
        {/* Broker connection */}
        <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.1 }}>
          <div className={cn(
            isConnected ? "glass-card-profit" : "glass-card-loss",
            "p-6"
          )}>
            <div className="flex items-start justify-between mb-4">
              <div className="flex items-center gap-3">
                <div className={cn(
                  "flex h-12 w-12 items-center justify-center rounded-xl",
                  isConnected ? "bg-profit/10" : "bg-loss/10"
                )}>
                  <Plug className={cn("h-6 w-6", isConnected ? "text-profit" : "text-loss")} />
                </div>
                <div>
                  <h3 className="text-lg font-semibold">Broker Connection</h3>
                  <p className="text-xs text-muted-foreground mt-0.5">Trading API gateway</p>
                </div>
              </div>
              {isConnected ? (
                <span className="status-badge-live">
                  <CheckCircle className="h-3 w-3" />
                  CONNECTED
                </span>
              ) : (
                <span className="status-badge-danger">
                  <XCircle className="h-3 w-3" />
                  DISCONNECTED
                </span>
              )}
            </div>

            <div className="grid grid-cols-2 gap-3 mt-4">
              {[
                { label: "Status", value: broker.status?.toUpperCase() ?? "UNKNOWN" },
                { label: "Mode", value: tradingMode === "paper" ? "PAPER" : "LIVE" },
                { label: "Open Positions", value: String(openPositions) },
                { label: "Daily P&L", value: `${(risk?.daily_pnl ?? 0).toFixed(2)}%` },
              ].map((item) => (
                <div key={item.label} className="rounded-lg bg-muted/20 p-3">
                  <div className="text-[10px] text-muted-foreground">{item.label}</div>
                  <div className="font-mono text-sm font-semibold mt-0.5">{item.value}</div>
                </div>
              ))}
            </div>

            <div className="flex gap-2 mt-4">
              <button
                onClick={handleReconnect}
                disabled={reconnecting}
                className={cn(
                  "flex-1 flex items-center justify-center gap-2 rounded-xl border border-border/50 py-2.5 text-xs font-medium transition-all",
                  reconnecting
                    ? "text-muted-foreground opacity-50 cursor-not-allowed"
                    : "text-muted-foreground hover:border-primary/40 hover:text-primary hover:bg-primary/5"
                )}
              >
                <RefreshCw className={cn("h-3.5 w-3.5", reconnecting && "animate-spin")} />
                {reconnecting ? "Reconnecting..." : "Reconnect"}
              </button>
              <Link
                href="/settings"
                className="flex-1 flex items-center justify-center gap-2 rounded-xl border border-border/50 py-2.5 text-xs font-medium text-muted-foreground transition-all hover:border-primary/40 hover:text-primary hover:bg-primary/5"
              >
                <Settings className="h-3.5 w-3.5" />
                Configure
              </Link>
            </div>
          </div>
        </motion.div>

        {/* Market feed */}
        <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.15 }}>
          <div className={cn(
            marketFeed.healthy ? "glass-card-profit" : "glass-card-loss",
            "p-6"
          )}>
            <div className="flex items-start justify-between mb-4">
              <div className="flex items-center gap-3">
                <div className={cn(
                  "flex h-12 w-12 items-center justify-center rounded-xl",
                  marketFeed.healthy ? "bg-profit/10" : "bg-loss/10"
                )}>
                  {marketFeed.healthy ? (
                    <Radio className="h-6 w-6 text-profit" />
                  ) : marketFeed.connected ? (
                    <Wifi className="h-6 w-6 text-warning animate-pulse" />
                  ) : (
                    <WifiOff className="h-6 w-6 text-loss" />
                  )}
                </div>
                <div>
                  <h3 className="text-lg font-semibold">Market Data Feed</h3>
                  <p className="text-xs text-muted-foreground mt-0.5">Real-time price streaming</p>
                </div>
              </div>
              {marketFeed.healthy ? (
                <span className="status-badge-live">
                  <span className="h-1.5 w-1.5 rounded-full bg-profit animate-pulse" />
                  STREAMING
                </span>
              ) : (
                <span className="status-badge-danger">OFFLINE</span>
              )}
            </div>

            <div className="grid grid-cols-2 gap-3 mt-4">
              {[
                { label: "Status", value: marketFeed.healthy ? "HEALTHY" : marketFeed.connected ? "DEGRADED" : "OFFLINE" },
                { label: "Last Tick", value: marketFeed.last_tick_ts ? new Date(marketFeed.last_tick_ts).toLocaleTimeString() : "—" },
                { label: "Connected", value: marketFeed.connected ? "Yes" : "No" },
                { label: "Feed Health", value: marketFeed.healthy ? "OK" : "—" },
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

      {/* Recent Orders */}
      <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.2 }}>
        <div className="glass-card overflow-hidden">
          <div className="flex items-center justify-between border-b border-border/30 p-5">
            <div>
              <h3 className="text-sm font-semibold">Recent Orders</h3>
              <p className="text-xs text-muted-foreground mt-0.5">Last 10 execution events</p>
            </div>
            <div className="flex items-center gap-3">
              <Link
                href="/trades"
                className="text-xs font-medium text-primary hover:text-primary/80 transition-colors"
              >
                View All
              </Link>
              <Clock className="h-4 w-4 text-primary" />
            </div>
          </div>

          {recentOrders.length === 0 ? (
            <div className="flex flex-col items-center justify-center py-12 text-muted-foreground">
              <Activity className="h-10 w-10 mb-3 opacity-30" />
              <p className="text-sm font-medium">No orders yet</p>
              <p className="text-xs mt-1">Orders will appear when strategies execute trades</p>
            </div>
          ) : (
            <div className="divide-y divide-border/10">
              {recentOrders.map((o, i) => (
                <motion.div
                  key={o.order_id || `${o.symbol}-${i}`}
                  initial={{ opacity: 0, x: -10 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: 0.05 + i * 0.03 }}
                  className="flex items-center gap-4 px-5 py-3 hover:bg-muted/10 transition-colors"
                >
                  <div className={cn(
                    "flex h-8 w-8 items-center justify-center rounded-lg",
                    o.side === "BUY" ? "bg-profit/10" : "bg-loss/10"
                  )}>
                    {o.side === "BUY" ? (
                      <ArrowUpRight className="h-4 w-4 text-profit" />
                    ) : (
                      <ArrowDownRight className="h-4 w-4 text-loss" />
                    )}
                  </div>
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2">
                      <span className="font-mono text-sm font-semibold">{o.symbol}</span>
                      <span className={cn(
                        "text-[10px] font-bold px-1.5 py-0.5 rounded",
                        o.side === "BUY" ? "bg-profit/10 text-profit" : "bg-loss/10 text-loss"
                      )}>
                        {o.side}
                      </span>
                    </div>
                    <span className="text-[10px] text-muted-foreground">
                      {o.filled_qty}/{o.quantity} filled {o.avg_price ? `@ ${o.avg_price.toFixed(2)}` : ""}
                    </span>
                  </div>
                  <div className="text-right">
                    <span className={cn(
                      "text-[10px] font-bold px-2 py-0.5 rounded",
                      o.status === "FILLED" ? "bg-profit/10 text-profit" :
                      o.status === "CANCELLED" || o.status === "REJECTED" ? "bg-loss/10 text-loss" :
                      "bg-warning/10 text-warning"
                    )}>
                      {o.status}
                    </span>
                    <div className="text-[10px] text-muted-foreground mt-0.5">
                      {o.ts ? new Date(o.ts).toLocaleTimeString("en-IN", { hour: "2-digit", minute: "2-digit" }) : ""}
                    </div>
                  </div>
                </motion.div>
              ))}
            </div>
          )}
        </div>
      </motion.div>
    </div>
  );
}
