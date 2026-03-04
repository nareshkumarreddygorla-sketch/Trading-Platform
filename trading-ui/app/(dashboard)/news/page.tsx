"use client";

import { useQuery } from "@tanstack/react-query";
import { motion } from "framer-motion";
import { cn } from "@/lib/utils";
import { endpoints } from "@/lib/api/client";
import {
  Newspaper, TrendingUp, TrendingDown, Minus,
  RefreshCw, Globe, Briefcase, BarChart3,
} from "lucide-react";

type NewsItem = {
  id: string;
  headline: string;
  symbol: string;
  sentiment: string;
  score: number;
  source: string;
  timestamp: string;
  category: string;
};

const SENTIMENT_STYLES = {
  positive: { color: "text-profit", bg: "bg-profit/10", border: "border-profit/20", icon: TrendingUp },
  negative: { color: "text-loss", bg: "bg-loss/10", border: "border-loss/20", icon: TrendingDown },
  neutral: { color: "text-muted-foreground", bg: "bg-muted/30", border: "border-border/30", icon: Minus },
};

const CATEGORY_ICONS: Record<string, typeof Globe> = {
  price_action: BarChart3,
  portfolio: Briefcase,
  market: Globe,
};

export default function NewsPage() {
  const { data, isLoading, refetch } = useQuery({
    queryKey: ["market-news"],
    queryFn: () => endpoints.marketNews(30),
    refetchInterval: 30000,
  });

  const news = (data?.news ?? []) as NewsItem[];
  const positiveCount = news.filter((n) => n.sentiment === "positive").length;
  const negativeCount = news.filter((n) => n.sentiment === "negative").length;
  const overallSentiment = positiveCount > negativeCount ? "Bullish" : negativeCount > positiveCount ? "Bearish" : "Neutral";
  const overallColor = overallSentiment === "Bullish" ? "text-profit" : overallSentiment === "Bearish" ? "text-loss" : "text-muted-foreground";

  return (
    <div className="space-y-6 pb-8">
      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: -10 }}
        animate={{ opacity: 1, y: 0 }}
        className="flex items-center justify-between"
      >
        <div>
          <h1 className="text-2xl font-bold tracking-tight">Market News</h1>
          <p className="text-sm text-muted-foreground mt-0.5">
            AI-analyzed market events & sentiment
          </p>
        </div>
        <button
          onClick={() => refetch()}
          className="flex items-center gap-2 glass-card px-3 py-2 rounded-xl text-xs font-medium text-muted-foreground hover:text-primary transition-colors"
          aria-label="Refresh market news"
        >
          <RefreshCw className="h-3.5 w-3.5" />
          Refresh
        </button>
      </motion.div>

      {/* Sentiment Overview */}
      <div className="grid gap-4 md:grid-cols-4">
        <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }}>
          <div className="glass-card p-4 flex items-center gap-3">
            <div className={cn("flex h-9 w-9 items-center justify-center rounded-xl bg-muted/50", overallColor)}>
              <Newspaper className="h-4 w-4" />
            </div>
            <div>
              <div className="kpi-label">Overall Sentiment</div>
              <div className={cn("font-mono text-lg font-bold", overallColor)}>{overallSentiment}</div>
            </div>
          </div>
        </motion.div>
        <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.05 }}>
          <div className="glass-card p-4 flex items-center gap-3">
            <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-profit/10 text-profit">
              <TrendingUp className="h-4 w-4" />
            </div>
            <div>
              <div className="kpi-label">Positive</div>
              <div className="font-mono text-lg font-bold text-profit">{positiveCount}</div>
            </div>
          </div>
        </motion.div>
        <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.1 }}>
          <div className="glass-card p-4 flex items-center gap-3">
            <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-loss/10 text-loss">
              <TrendingDown className="h-4 w-4" />
            </div>
            <div>
              <div className="kpi-label">Negative</div>
              <div className="font-mono text-lg font-bold text-loss">{negativeCount}</div>
            </div>
          </div>
        </motion.div>
        <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.15 }}>
          <div className="glass-card p-4 flex items-center gap-3">
            <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-primary/10 text-primary">
              <Globe className="h-4 w-4" />
            </div>
            <div>
              <div className="kpi-label">Total Updates</div>
              <div className="font-mono text-lg font-bold text-primary">{news.length}</div>
            </div>
          </div>
        </motion.div>
      </div>

      {/* News Feed */}
      <motion.div
        initial={{ opacity: 0, y: 16 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2 }}
      >
        <div className="glass-card overflow-hidden">
          <div className="border-b border-border/30 p-5">
            <h3 className="text-sm font-semibold">Live Feed</h3>
            <p className="text-xs text-muted-foreground mt-0.5">Auto-refreshes every 30s</p>
          </div>

          {isLoading ? (
            <div className="p-5 space-y-3">
              {Array.from({ length: 5 }).map((_, i) => (
                <div key={i} className="shimmer h-20 w-full rounded-lg" />
              ))}
            </div>
          ) : news.length === 0 ? (
            <div className="flex flex-col items-center justify-center py-16 text-muted-foreground">
              <Newspaper className="h-10 w-10 mb-3 opacity-30" />
              <p className="text-sm font-medium">No news available</p>
              <p className="text-xs mt-1">Market news will appear when data is available</p>
            </div>
          ) : (
            <div className="divide-y divide-border/10">
              {news.map((item, i) => {
                const sentimentStyle = SENTIMENT_STYLES[item.sentiment as keyof typeof SENTIMENT_STYLES] || SENTIMENT_STYLES.neutral;
                const SentimentIcon = sentimentStyle.icon;
                const CategoryIcon = CATEGORY_ICONS[item.category] || Globe;

                return (
                  <motion.div
                    key={item.id}
                    initial={{ opacity: 0, x: -10 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: 0.05 + i * 0.03 }}
                    className="p-4 hover:bg-muted/15 transition-colors"
                  >
                    <div className="flex items-start gap-3">
                      {/* Sentiment indicator */}
                      <div className={cn(
                        "flex h-9 w-9 shrink-0 items-center justify-center rounded-xl border",
                        sentimentStyle.bg, sentimentStyle.border
                      )}>
                        <SentimentIcon className={cn("h-4 w-4", sentimentStyle.color)} />
                      </div>

                      {/* Content */}
                      <div className="flex-1 min-w-0">
                        <p className="text-sm font-medium leading-snug">{item.headline}</p>
                        <div className="flex items-center gap-3 mt-1.5">
                          <span className="inline-flex items-center gap-1 text-[10px] font-mono font-bold text-primary">
                            {item.symbol}
                          </span>
                          <span className={cn(
                            "inline-flex items-center gap-1 rounded-md px-1.5 py-0.5 text-[9px] font-bold uppercase",
                            sentimentStyle.bg, sentimentStyle.color
                          )}>
                            {item.sentiment}
                          </span>
                          <span className="flex items-center gap-1 text-[10px] text-muted-foreground">
                            <CategoryIcon className="h-3 w-3" />
                            {item.source}
                          </span>
                          <span className="text-[10px] text-muted-foreground font-mono ml-auto">
                            {new Date(item.timestamp).toLocaleTimeString("en-IN", { hour: "2-digit", minute: "2-digit" })}
                          </span>
                        </div>
                      </div>

                      {/* Sentiment score bar */}
                      <div className="hidden sm:flex flex-col items-end gap-1 shrink-0">
                        <span className="text-[10px] text-muted-foreground">Score</span>
                        <div className="w-16 h-1.5 rounded-full bg-muted overflow-hidden">
                          <div
                            className={cn("h-full rounded-full", item.sentiment === "positive" ? "bg-profit" : item.sentiment === "negative" ? "bg-loss" : "bg-muted-foreground")}
                            style={{ width: `${item.score * 100}%` }}
                          />
                        </div>
                      </div>
                    </div>
                  </motion.div>
                );
              })}
            </div>
          )}
        </div>
      </motion.div>
    </div>
  );
}
