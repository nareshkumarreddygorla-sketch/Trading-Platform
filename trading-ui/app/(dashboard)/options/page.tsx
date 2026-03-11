"use client";

import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { motion } from "framer-motion";
import { api } from "@/lib/api/client";
import {
  BarChart3,
  TrendingUp,
  TrendingDown,
  Target,
  Activity,
  ArrowUpRight,
  ArrowDownRight,
} from "lucide-react";
import { cn } from "@/lib/utils";

type OptionContract = {
  symbol: string;
  strike: number;
  option_type: string;
  last_price: number;
  bid: number;
  ask: number;
  volume: number;
  open_interest: number;
  iv: number;
  greeks: {
    delta: number;
    gamma: number;
    theta: number;
    vega: number;
    price: number;
  } | null;
};

type OptionsChainData = {
  underlying: string;
  spot_price: number;
  expiry: string;
  calls: OptionContract[];
  puts: OptionContract[];
  max_pain: number;
  pcr: { pcr_volume: number; pcr_oi: number; sentiment: string };
};

const UNDERLYINGS = ["NIFTY", "BANKNIFTY", "RELIANCE", "TCS", "INFY", "HDFCBANK"];

export default function OptionsPage() {
  const [underlying, setUnderlying] = useState("NIFTY");
  const [highlightATM, setHighlightATM] = useState(true);

  const { data: chain, isLoading } = useQuery({
    queryKey: ["options-chain", underlying],
    queryFn: () =>
      api.get<OptionsChainData>(`/api/v1/options/chain/${underlying}`),
    refetchInterval: 15000,
  });

  const spotPrice = chain?.spot_price ?? 0;
  const maxPain = chain?.max_pain ?? 0;
  const pcr = chain?.pcr ?? { pcr_volume: 0, pcr_oi: 0, sentiment: "neutral" };

  const sentimentColor =
    pcr.sentiment === "bullish"
      ? "text-profit"
      : pcr.sentiment === "bearish"
        ? "text-loss"
        : "text-muted-foreground";

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between flex-wrap gap-3">
        <div>
          <h1 className="text-2xl font-bold flex items-center gap-2">
            <BarChart3 className="h-7 w-7 text-primary" />
            Options Chain
          </h1>
          <p className="text-sm text-muted-foreground mt-1">
            Live options chain with Greeks, max pain, and PCR analysis
          </p>
        </div>
        <div className="flex gap-2 flex-wrap">
          {UNDERLYINGS.map((sym) => (
            <button
              key={sym}
              onClick={() => setUnderlying(sym)}
              className={cn(
                "px-3 py-1.5 rounded-lg text-xs font-medium transition-all",
                underlying === sym
                  ? "bg-primary text-white"
                  : "bg-card border border-border/50 text-muted-foreground hover:text-foreground"
              )}
            >
              {sym}
            </button>
          ))}
        </div>
      </div>

      {/* Key Metrics */}
      <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          className="glass-card rounded-xl p-4"
        >
          <div className="text-xs text-muted-foreground">Spot Price</div>
          <div className="text-lg font-bold">{spotPrice.toLocaleString()}</div>
        </motion.div>
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.05 }}
          className="glass-card rounded-xl p-4"
        >
          <div className="text-xs text-muted-foreground">Max Pain</div>
          <div className="text-lg font-bold flex items-center gap-1">
            <Target className="h-4 w-4 text-yellow-400" />
            {maxPain.toLocaleString()}
          </div>
        </motion.div>
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="glass-card rounded-xl p-4"
        >
          <div className="text-xs text-muted-foreground">PCR (OI)</div>
          <div className={cn("text-lg font-bold", sentimentColor)}>
            {pcr.pcr_oi.toFixed(3)}
          </div>
        </motion.div>
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.15 }}
          className="glass-card rounded-xl p-4"
        >
          <div className="text-xs text-muted-foreground">PCR (Vol)</div>
          <div className="text-lg font-bold">{pcr.pcr_volume.toFixed(3)}</div>
        </motion.div>
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="glass-card rounded-xl p-4"
        >
          <div className="text-xs text-muted-foreground">Sentiment</div>
          <div className={cn("text-lg font-bold capitalize flex items-center gap-1", sentimentColor)}>
            {pcr.sentiment === "bullish" ? (
              <TrendingUp className="h-4 w-4" />
            ) : pcr.sentiment === "bearish" ? (
              <TrendingDown className="h-4 w-4" />
            ) : (
              <Activity className="h-4 w-4" />
            )}
            {pcr.sentiment}
          </div>
        </motion.div>
      </div>

      {/* Options Chain Table */}
      <div className="glass-card rounded-xl overflow-hidden">
        <div className="flex items-center justify-between p-4 border-b border-border/30">
          <h2 className="font-semibold text-sm">
            {underlying} Options Chain — Expiry: {chain?.expiry ?? "Loading..."}
          </h2>
          <label className="flex items-center gap-2 text-xs text-muted-foreground">
            <input
              type="checkbox"
              checked={highlightATM}
              onChange={(e) => setHighlightATM(e.target.checked)}
              className="rounded"
            />
            Highlight ATM
          </label>
        </div>

        {isLoading ? (
          <div className="p-8 text-center text-muted-foreground animate-pulse">
            Loading options chain...
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="border-b border-border/30">
                  <th colSpan={6} className="text-center py-2 text-profit font-semibold bg-profit/5">
                    CALLS
                  </th>
                  <th className="py-2 px-3 text-center font-semibold bg-card/50">STRIKE</th>
                  <th colSpan={6} className="text-center py-2 text-loss font-semibold bg-loss/5">
                    PUTS
                  </th>
                </tr>
                <tr className="border-b border-border/20 text-muted-foreground">
                  <th className="py-2 px-2 text-right">Delta</th>
                  <th className="py-2 px-2 text-right">IV%</th>
                  <th className="py-2 px-2 text-right">OI</th>
                  <th className="py-2 px-2 text-right">Vol</th>
                  <th className="py-2 px-2 text-right">Bid</th>
                  <th className="py-2 px-2 text-right">Ask</th>
                  <th className="py-2 px-3 text-center font-bold"></th>
                  <th className="py-2 px-2 text-right">Bid</th>
                  <th className="py-2 px-2 text-right">Ask</th>
                  <th className="py-2 px-2 text-right">Vol</th>
                  <th className="py-2 px-2 text-right">OI</th>
                  <th className="py-2 px-2 text-right">IV%</th>
                  <th className="py-2 px-2 text-right">Delta</th>
                </tr>
              </thead>
              <tbody>
                {(chain?.calls ?? []).map((call, i) => {
                  const put = chain?.puts?.[i];
                  const isATM =
                    highlightATM &&
                    Math.abs(call.strike - spotPrice) <=
                      (spotPrice > 5000 ? 100 : spotPrice > 500 ? 50 : 10);
                  const isITMCall = call.strike < spotPrice;
                  const isITMPut = put && put.strike > spotPrice;

                  return (
                    <tr
                      key={call.strike}
                      className={cn(
                        "border-b border-border/10 hover:bg-card/30 transition-colors",
                        isATM && "bg-primary/5 border-primary/20"
                      )}
                    >
                      <td className="py-1.5 px-2 text-right font-mono">
                        {call.greeks?.delta?.toFixed(3) ?? "-"}
                      </td>
                      <td className="py-1.5 px-2 text-right">{call.iv}</td>
                      <td className="py-1.5 px-2 text-right">{call.open_interest.toLocaleString()}</td>
                      <td className="py-1.5 px-2 text-right">{call.volume.toLocaleString()}</td>
                      <td className={cn("py-1.5 px-2 text-right font-mono", isITMCall && "text-profit/70")}>
                        {call.bid.toFixed(2)}
                      </td>
                      <td className={cn("py-1.5 px-2 text-right font-mono", isITMCall && "text-profit/70")}>
                        {call.ask.toFixed(2)}
                      </td>
                      <td
                        className={cn(
                          "py-1.5 px-3 text-center font-bold",
                          isATM ? "bg-primary/10 text-primary" : "bg-card/30"
                        )}
                      >
                        {call.strike.toLocaleString()}
                      </td>
                      <td className={cn("py-1.5 px-2 text-right font-mono", isITMPut && "text-loss/70")}>
                        {put?.bid.toFixed(2) ?? "-"}
                      </td>
                      <td className={cn("py-1.5 px-2 text-right font-mono", isITMPut && "text-loss/70")}>
                        {put?.ask.toFixed(2) ?? "-"}
                      </td>
                      <td className="py-1.5 px-2 text-right">
                        {put?.volume.toLocaleString() ?? "-"}
                      </td>
                      <td className="py-1.5 px-2 text-right">
                        {put?.open_interest.toLocaleString() ?? "-"}
                      </td>
                      <td className="py-1.5 px-2 text-right">{put?.iv ?? "-"}</td>
                      <td className="py-1.5 px-2 text-right font-mono">
                        {put?.greeks?.delta?.toFixed(3) ?? "-"}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}
