"use client";

import { useEffect } from "react";
import { useQuery } from "@tanstack/react-query";
import { useStore } from "@/store/useStore";
import { endpoints } from "@/lib/api/client";
import { Gauge } from "lucide-react";

const REGIME_CONFIG: Record<string, { label: string; color: string; bg: string }> = {
  trending_up: { label: "Trending Up", color: "text-green-400", bg: "bg-green-500/10" },
  trending_down: { label: "Trending Down", color: "text-red-400", bg: "bg-red-500/10" },
  low_volatility: { label: "Low Volatility", color: "text-blue-400", bg: "bg-blue-500/10" },
  high_volatility: { label: "High Volatility", color: "text-orange-400", bg: "bg-orange-500/10" },
  crisis: { label: "Crisis", color: "text-red-500", bg: "bg-red-500/20" },
  unknown: { label: "Scanning...", color: "text-zinc-400", bg: "bg-zinc-500/10" },
};

export default function RegimeIndicator() {
  const regime = useStore((s) => s.currentRegime);

  // Fetch regime from API on mount + poll every 30s
  const { data: regimeData } = useQuery({
    queryKey: ["market-regime"],
    queryFn: () => endpoints.marketRegime(),
    refetchInterval: 30000,
  });

  // Update store with API-fetched regime when WS hasn't set it
  useEffect(() => {
    if (regimeData?.regime && regimeData.regime !== "unknown" && regime === "unknown") {
      useStore.setState({ currentRegime: regimeData.regime });
    }
  }, [regimeData, regime]);

  const displayRegime = regime !== "unknown" ? regime : (regimeData?.regime ?? "unknown");
  const config = REGIME_CONFIG[displayRegime] || REGIME_CONFIG.unknown;
  const confidence = regimeData?.confidence ?? 0;

  return (
    <div className={`rounded-lg ${config.bg} border border-zinc-800 px-3 py-2 flex items-center gap-2`}>
      <Gauge className={`w-4 h-4 ${config.color}`} />
      <div className="flex-1">
        <div className="text-xs text-zinc-500">Market Regime</div>
        <div className={`text-sm font-medium ${config.color}`}>{config.label}</div>
      </div>
      {confidence > 0 && (
        <span className="text-[10px] text-zinc-500 font-mono">{(confidence * 100).toFixed(0)}%</span>
      )}
    </div>
  );
}
