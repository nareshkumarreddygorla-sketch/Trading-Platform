"use client";

import { useEffect } from "react";
import { useQuery } from "@tanstack/react-query";
import { useStore } from "@/store/useStore";
import { endpoints } from "@/lib/api/client";
import { TrendingUp, TrendingDown, Radio } from "lucide-react";

export default function SignalFeed() {
  const signals = useStore((s) => s.recentSignals);
  const opportunities = useStore((s) => s.topOpportunities);

  // Seed signals from recent orders if WebSocket hasn't sent any yet
  const { data: ordersData } = useQuery({
    queryKey: ["recent-orders-feed"],
    queryFn: () => endpoints.orders(20),
    refetchInterval: 15000,
    enabled: signals.length === 0,
  });

  useEffect(() => {
    if (signals.length === 0 && ordersData?.orders?.length) {
      const seeded = ordersData.orders.map((o) => ({
        symbol: o.symbol,
        direction: o.side,
        confidence: o.status === "FILLED" ? 1.0 : 0.8,
        source: o.strategy_id || "manual",
        timestamp: o.ts || new Date().toISOString(),
      }));
      useStore.setState({ recentSignals: seeded });
    }
  }, [ordersData, signals.length]);

  return (
    <div className="rounded-xl border border-zinc-800 bg-zinc-900/50 p-4">
      <h3 className="text-sm font-medium text-zinc-400 mb-3 flex items-center gap-2">
        <Radio className="w-4 h-4" />
        Live Signal Feed
        {signals.length > 0 && (
          <span className="ml-auto text-[10px] text-zinc-600 font-mono">{signals.length} signals</span>
        )}
      </h3>

      {/* Top Opportunities */}
      {opportunities.length > 0 && (
        <div className="mb-3">
          <div className="text-xs text-zinc-500 mb-1">Top Opportunities</div>
          <div className="flex flex-wrap gap-1">
            {opportunities.slice(0, 5).map((opp, i) => (
              <span
                key={`${opp.symbol}-${i}`}
                className={`inline-flex items-center gap-1 px-2 py-0.5 rounded text-xs ${
                  opp.direction === "BUY"
                    ? "bg-green-500/10 text-green-400 border border-green-500/20"
                    : "bg-red-500/10 text-red-400 border border-red-500/20"
                }`}
              >
                {opp.direction === "BUY" ? (
                  <TrendingUp className="w-3 h-3" />
                ) : (
                  <TrendingDown className="w-3 h-3" />
                )}
                {opp.symbol}
                <span className="text-zinc-500">{(opp.confidence * 100).toFixed(0)}%</span>
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Recent Signals */}
      <div className="space-y-1 max-h-48 overflow-y-auto">
        {signals.length === 0 ? (
          <div className="text-center text-zinc-500 text-xs py-4">
            <div className="flex items-center justify-center gap-2 mb-1">
              <span className="h-1.5 w-1.5 rounded-full bg-zinc-600 animate-pulse" />
              <span className="h-1.5 w-1.5 rounded-full bg-zinc-600 animate-pulse" style={{ animationDelay: "0.3s" }} />
              <span className="h-1.5 w-1.5 rounded-full bg-zinc-600 animate-pulse" style={{ animationDelay: "0.6s" }} />
            </div>
            Scanning for signals...
          </div>
        ) : (
          signals.slice(0, 15).map((sig, i) => (
            <div
              key={`${sig.symbol}-${sig.timestamp}-${i}`}
              className="flex items-center gap-2 text-xs py-1 border-b border-zinc-800/50 last:border-0"
            >
              <span
                className={`w-1.5 h-1.5 rounded-full ${
                  sig.direction?.includes("BUY") ? "bg-green-400" : "bg-red-400"
                }`}
              />
              <span className="text-zinc-300 font-mono">{sig.symbol}</span>
              <span
                className={`${
                  sig.direction?.includes("BUY") ? "text-green-400" : "text-red-400"
                }`}
              >
                {sig.direction}
              </span>
              <span className="text-zinc-500 ml-auto">
                {sig.confidence ? `${(sig.confidence * 100).toFixed(0)}%` : ""}
              </span>
              <span className="text-zinc-600 truncate max-w-[80px]">
                {sig.source}
              </span>
            </div>
          ))
        )}
      </div>
    </div>
  );
}
