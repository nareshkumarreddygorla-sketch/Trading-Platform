"use client";

import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { motion } from "framer-motion";
import { api } from "@/lib/api/client";
import {
  Store,
  Star,
  TrendingUp,
  Shield,
  Zap,
  Users,
  ArrowUpRight,
  Search,
  Filter,
  Trophy,
  BarChart3,
} from "lucide-react";
import { cn } from "@/lib/utils";

type StrategyListing = {
  id: string;
  name: string;
  description: string;
  category: string;
  risk_level: string;
  author: string;
  rating: number;
  subscribers: number;
  monthly_return: number;
  sharpe_ratio: number;
  max_drawdown: number;
  win_rate: number;
  total_trades: number;
  created_at: string;
};

const RISK_COLORS: Record<string, string> = {
  low: "text-green-400 bg-green-400/10 border-green-400/20",
  medium: "text-yellow-400 bg-yellow-400/10 border-yellow-400/20",
  high: "text-orange-400 bg-orange-400/10 border-orange-400/20",
  aggressive: "text-red-400 bg-red-400/10 border-red-400/20",
};

const CATEGORY_ICONS: Record<string, typeof TrendingUp> = {
  trend_following: TrendingUp,
  mean_reversion: BarChart3,
  momentum: Zap,
  arbitrage: ArrowUpRight,
  ml_based: Shield,
};

export default function MarketplacePage() {
  const [search, setSearch] = useState("");
  const [category, setCategory] = useState("all");
  const [sortBy, setSortBy] = useState("rating");
  const queryClient = useQueryClient();

  const { data: strategies, isLoading } = useQuery({
    queryKey: ["marketplace-strategies", category, sortBy],
    queryFn: () =>
      api.get<{ strategies: StrategyListing[] }>(
        `/api/v1/marketplace/strategies?sort_by=${sortBy}${category !== "all" ? `&category=${category}` : ""}`
      ),
    refetchInterval: 30000,
  });

  const { data: leaderboard } = useQuery({
    queryKey: ["marketplace-leaderboard"],
    queryFn: () =>
      api.get<{ leaderboard: Array<{ name: string; sharpe_ratio: number; monthly_return: number; subscribers: number }> }>(
        "/api/v1/marketplace/leaderboard"
      ),
  });

  const subscribeMutation = useMutation({
    mutationFn: (strategyId: string) =>
      api.post("/api/v1/marketplace/subscribe", {
        strategy_id: strategyId,
        user_id: "current_user",
      }),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["marketplace-strategies"] }),
  });

  const filtered = (strategies?.strategies ?? []).filter(
    (s) =>
      !search ||
      s.name.toLowerCase().includes(search.toLowerCase()) ||
      s.description.toLowerCase().includes(search.toLowerCase())
  );

  const categories = [
    "all",
    "trend_following",
    "mean_reversion",
    "momentum",
    "scalping",
    "ml_based",
    "options",
  ];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold flex items-center gap-2">
            <Store className="h-7 w-7 text-primary" />
            Strategy Marketplace
          </h1>
          <p className="text-sm text-muted-foreground mt-1">
            Browse, subscribe, and deploy community strategies
          </p>
        </div>
      </div>

      {/* Leaderboard Banner */}
      {leaderboard?.leaderboard && leaderboard.leaderboard.length > 0 && (
        <motion.div
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          className="glass-card rounded-xl p-4 border border-primary/20"
        >
          <div className="flex items-center gap-2 mb-3">
            <Trophy className="h-5 w-5 text-yellow-400" />
            <span className="font-semibold text-sm">Top Performers</span>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
            {leaderboard.leaderboard.slice(0, 3).map((s, i) => (
              <div key={s.name} className="flex items-center gap-3 p-2 rounded-lg bg-card/50">
                <span className="text-lg font-bold text-yellow-400">#{i + 1}</span>
                <div className="flex-1 min-w-0">
                  <div className="text-sm font-medium truncate">{s.name}</div>
                  <div className="text-xs text-muted-foreground">
                    Sharpe {s.sharpe_ratio.toFixed(2)} | {s.subscribers} subs
                  </div>
                </div>
                <span className="text-sm font-bold text-profit">
                  +{s.monthly_return.toFixed(1)}%
                </span>
              </div>
            ))}
          </div>
        </motion.div>
      )}

      {/* Search & Filters */}
      <div className="flex flex-col md:flex-row gap-3">
        <div className="relative flex-1">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
          <input
            type="text"
            placeholder="Search strategies..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="w-full pl-9 pr-4 py-2 rounded-lg bg-card border border-border/50 text-sm focus:outline-none focus:ring-1 focus:ring-primary/50"
          />
        </div>
        <div className="flex gap-2 flex-wrap">
          {categories.map((cat) => (
            <button
              key={cat}
              onClick={() => setCategory(cat)}
              className={cn(
                "px-3 py-1.5 rounded-lg text-xs font-medium transition-all",
                category === cat
                  ? "bg-primary text-white"
                  : "bg-card border border-border/50 text-muted-foreground hover:text-foreground"
              )}
            >
              {cat === "all" ? "All" : cat.replace(/_/g, " ")}
            </button>
          ))}
        </div>
        <select
          value={sortBy}
          onChange={(e) => setSortBy(e.target.value)}
          className="px-3 py-2 rounded-lg bg-card border border-border/50 text-sm"
        >
          <option value="rating">Rating</option>
          <option value="subscribers">Popular</option>
          <option value="sharpe_ratio">Sharpe Ratio</option>
          <option value="monthly_return">Returns</option>
        </select>
      </div>

      {/* Strategy Cards */}
      {isLoading ? (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {[...Array(6)].map((_, i) => (
            <div key={i} className="glass-card rounded-xl p-5 animate-pulse h-64" />
          ))}
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {filtered.map((strat, i) => {
            const Icon = CATEGORY_ICONS[strat.category] || Zap;
            return (
              <motion.div
                key={strat.id}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: i * 0.05 }}
                className="glass-card rounded-xl p-5 border border-border/30 hover:border-primary/30 transition-all group"
              >
                <div className="flex items-start justify-between mb-3">
                  <div className="flex items-center gap-2">
                    <div className="p-2 rounded-lg bg-primary/10">
                      <Icon className="h-4 w-4 text-primary" />
                    </div>
                    <div>
                      <h3 className="font-semibold text-sm">{strat.name}</h3>
                      <p className="text-xs text-muted-foreground">by {strat.author}</p>
                    </div>
                  </div>
                  <span
                    className={cn(
                      "px-2 py-0.5 rounded-full text-[10px] font-medium border",
                      RISK_COLORS[strat.risk_level] || RISK_COLORS.medium
                    )}
                  >
                    {strat.risk_level}
                  </span>
                </div>

                <p className="text-xs text-muted-foreground mb-4 line-clamp-2">
                  {strat.description}
                </p>

                <div className="grid grid-cols-2 gap-2 mb-4">
                  <div className="bg-card/50 rounded-lg p-2">
                    <div className="text-[10px] text-muted-foreground">Monthly Return</div>
                    <div className={cn("text-sm font-bold", strat.monthly_return >= 0 ? "text-profit" : "text-loss")}>
                      {strat.monthly_return >= 0 ? "+" : ""}
                      {strat.monthly_return.toFixed(1)}%
                    </div>
                  </div>
                  <div className="bg-card/50 rounded-lg p-2">
                    <div className="text-[10px] text-muted-foreground">Sharpe Ratio</div>
                    <div className="text-sm font-bold">{strat.sharpe_ratio.toFixed(2)}</div>
                  </div>
                  <div className="bg-card/50 rounded-lg p-2">
                    <div className="text-[10px] text-muted-foreground">Win Rate</div>
                    <div className="text-sm font-bold">{(strat.win_rate * 100).toFixed(0)}%</div>
                  </div>
                  <div className="bg-card/50 rounded-lg p-2">
                    <div className="text-[10px] text-muted-foreground">Max DD</div>
                    <div className="text-sm font-bold text-loss">
                      -{strat.max_drawdown.toFixed(1)}%
                    </div>
                  </div>
                </div>

                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3 text-xs text-muted-foreground">
                    <span className="flex items-center gap-1">
                      <Star className="h-3 w-3 text-yellow-400" />
                      {strat.rating.toFixed(1)}
                    </span>
                    <span className="flex items-center gap-1">
                      <Users className="h-3 w-3" />
                      {strat.subscribers}
                    </span>
                  </div>
                  <button
                    onClick={() => subscribeMutation.mutate(strat.id)}
                    className="px-3 py-1.5 rounded-lg bg-primary/20 text-primary text-xs font-medium hover:bg-primary/30 transition-colors"
                  >
                    Subscribe
                  </button>
                </div>
              </motion.div>
            );
          })}
        </div>
      )}

      {filtered.length === 0 && !isLoading && (
        <div className="text-center py-12 text-muted-foreground">
          <Store className="h-12 w-12 mx-auto mb-3 opacity-30" />
          <p>No strategies found matching your criteria.</p>
        </div>
      )}
    </div>
  );
}
