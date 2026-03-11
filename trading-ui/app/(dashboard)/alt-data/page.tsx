"use client";

import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { motion } from "framer-motion";
import { api } from "@/lib/api/client";
import {
  Newspaper,
  TrendingUp,
  TrendingDown,
  Activity,
  Globe,
  BarChart3,
  ArrowUpRight,
  ArrowDownRight,
  Gauge,
} from "lucide-react";
import { cn } from "@/lib/utils";

type NewsArticle = {
  title: string;
  source: string;
  url: string;
  published_at: string;
  summary: string;
  sentiment_score: number;
  symbols: string[];
};

type SentimentSummary = {
  overall_score: number;
  sentiment_label: string;
  article_count: number;
  top_positive: string[];
  top_negative: string[];
  symbol_sentiments: Record<string, number>;
};

type FlowData = {
  date: string;
  fii_buy: number;
  fii_sell: number;
  fii_net: number;
  dii_buy: number;
  dii_sell: number;
  dii_net: number;
  total_net: number;
};

type FlowAnalysis = {
  trend: string;
  fii_streak: number;
  dii_streak: number;
  avg_fii_net_5d: number;
  avg_dii_net_5d: number;
  signal_strength: number;
  recommendation: string;
};

type CombinedSignal = {
  news_sentiment: number;
  news_label: string;
  fii_dii_trend: string;
  fii_dii_signal: number;
  exposure_multiplier: number;
  combined_score: number;
  combined_label: string;
  recommendation: string;
};

export default function AltDataPage() {
  const [market, setMarket] = useState("india");

  const { data: news } = useQuery({
    queryKey: ["alt-news", market],
    queryFn: () => api.get<NewsArticle[]>(`/api/v1/alt-data/news?market=${market}`),
    refetchInterval: 60000,
  });

  const { data: sentiment } = useQuery({
    queryKey: ["alt-sentiment", market],
    queryFn: () =>
      api.get<SentimentSummary>(`/api/v1/alt-data/news/sentiment?market=${market}`),
    refetchInterval: 60000,
  });

  const { data: flows } = useQuery({
    queryKey: ["fii-dii-flows"],
    queryFn: () => api.get<FlowData[]>("/api/v1/alt-data/fii-dii/flows?days=10"),
    refetchInterval: 300000,
  });

  const { data: flowAnalysis } = useQuery({
    queryKey: ["fii-dii-analysis"],
    queryFn: () => api.get<FlowAnalysis>("/api/v1/alt-data/fii-dii/analysis"),
    refetchInterval: 300000,
  });

  const { data: combined } = useQuery({
    queryKey: ["alt-combined", market],
    queryFn: () =>
      api.get<CombinedSignal>(`/api/v1/alt-data/combined-signal?market=${market}`),
    refetchInterval: 60000,
  });

  const trendIcon = (trend: string) =>
    trend === "bullish" ? (
      <TrendingUp className="h-4 w-4 text-profit" />
    ) : trend === "bearish" ? (
      <TrendingDown className="h-4 w-4 text-loss" />
    ) : (
      <Activity className="h-4 w-4 text-muted-foreground" />
    );

  const trendColor = (trend: string) =>
    trend === "bullish" ? "text-profit" : trend === "bearish" ? "text-loss" : "text-muted-foreground";

  const sentimentBar = (score: number) => {
    const pct = Math.round(score * 100);
    const color =
      score > 0.6
        ? "bg-profit"
        : score < 0.4
          ? "bg-loss"
          : "bg-yellow-400";
    return (
      <div className="w-full h-1.5 rounded-full bg-card/50 overflow-hidden">
        <div className={cn("h-full rounded-full transition-all", color)} style={{ width: `${pct}%` }} />
      </div>
    );
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between flex-wrap gap-3">
        <div>
          <h1 className="text-2xl font-bold flex items-center gap-2">
            <Globe className="h-7 w-7 text-primary" />
            Alternative Data
          </h1>
          <p className="text-sm text-muted-foreground mt-1">
            News sentiment, institutional flows, and combined signals
          </p>
        </div>
        <div className="flex gap-2">
          {["india", "us"].map((m) => (
            <button
              key={m}
              onClick={() => setMarket(m)}
              className={cn(
                "px-4 py-1.5 rounded-lg text-sm font-medium transition-all capitalize",
                market === m
                  ? "bg-primary text-white"
                  : "bg-card border border-border/50 text-muted-foreground"
              )}
            >
              {m === "india" ? "India" : "US"}
            </button>
          ))}
        </div>
      </div>

      {/* Combined Signal Banner */}
      {combined && (
        <motion.div
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          className={cn(
            "glass-card rounded-xl p-5 border",
            combined.combined_label === "bullish"
              ? "border-profit/20"
              : combined.combined_label === "bearish"
                ? "border-loss/20"
                : "border-border/30"
          )}
        >
          <div className="flex items-center gap-2 mb-3">
            <Gauge className="h-5 w-5 text-primary" />
            <span className="font-semibold text-sm">Combined Alt-Data Signal</span>
          </div>
          <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
            <div>
              <div className="text-xs text-muted-foreground">Combined Score</div>
              <div className={cn("text-xl font-bold", trendColor(combined.combined_label))}>
                {(combined.combined_score * 100).toFixed(0)}%
              </div>
            </div>
            <div>
              <div className="text-xs text-muted-foreground">News Sentiment</div>
              <div className={cn("text-xl font-bold", trendColor(combined.news_label))}>
                {(combined.news_sentiment * 100).toFixed(0)}%
              </div>
            </div>
            <div>
              <div className="text-xs text-muted-foreground">FII/DII Trend</div>
              <div className={cn("text-xl font-bold capitalize flex items-center gap-1", trendColor(combined.fii_dii_trend))}>
                {trendIcon(combined.fii_dii_trend)}
                {combined.fii_dii_trend}
              </div>
            </div>
            <div>
              <div className="text-xs text-muted-foreground">Exposure Multiplier</div>
              <div className="text-xl font-bold">{combined.exposure_multiplier}x</div>
            </div>
            <div>
              <div className="text-xs text-muted-foreground">Recommendation</div>
              <div className="text-xs font-medium">{combined.recommendation}</div>
            </div>
          </div>
        </motion.div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* News Feed */}
        <div className="glass-card rounded-xl overflow-hidden">
          <div className="p-4 border-b border-border/30 flex items-center gap-2">
            <Newspaper className="h-4 w-4 text-primary" />
            <span className="font-semibold text-sm">Market News</span>
            {sentiment && (
              <span className={cn("ml-auto text-xs font-medium capitalize", trendColor(sentiment.sentiment_label))}>
                {sentiment.sentiment_label} ({(sentiment.overall_score * 100).toFixed(0)}%)
              </span>
            )}
          </div>
          <div className="divide-y divide-border/10 max-h-[500px] overflow-y-auto">
            {(news ?? []).slice(0, 15).map((article, i) => (
              <motion.div
                key={`${article.title}-${i}`}
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: i * 0.03 }}
                className="p-3 hover:bg-card/30 transition-colors"
              >
                <div className="flex items-start justify-between gap-2">
                  <div className="flex-1 min-w-0">
                    <div className="text-sm font-medium leading-tight">{article.title}</div>
                    <div className="flex items-center gap-2 mt-1 text-xs text-muted-foreground">
                      <span>{article.source}</span>
                      {article.symbols.length > 0 && (
                        <span className="text-primary">{article.symbols.join(", ")}</span>
                      )}
                    </div>
                  </div>
                  <div className="shrink-0 text-right">
                    <div
                      className={cn(
                        "text-xs font-bold",
                        article.sentiment_score > 0.6
                          ? "text-profit"
                          : article.sentiment_score < 0.4
                            ? "text-loss"
                            : "text-muted-foreground"
                      )}
                    >
                      {(article.sentiment_score * 100).toFixed(0)}%
                    </div>
                    {sentimentBar(article.sentiment_score)}
                  </div>
                </div>
              </motion.div>
            ))}
          </div>
        </div>

        {/* FII/DII Flows */}
        <div className="space-y-4">
          {/* Flow Analysis */}
          {flowAnalysis && (
            <div className="glass-card rounded-xl p-5">
              <div className="flex items-center gap-2 mb-3">
                <BarChart3 className="h-4 w-4 text-primary" />
                <span className="font-semibold text-sm">FII/DII Flow Analysis</span>
                <span className={cn("ml-auto text-xs font-medium capitalize flex items-center gap-1", trendColor(flowAnalysis.trend))}>
                  {trendIcon(flowAnalysis.trend)}
                  {flowAnalysis.trend}
                </span>
              </div>
              <div className="grid grid-cols-2 gap-3 mb-3">
                <div className="bg-card/50 rounded-lg p-3">
                  <div className="text-xs text-muted-foreground">FII 5D Avg Net</div>
                  <div className={cn("text-sm font-bold", flowAnalysis.avg_fii_net_5d >= 0 ? "text-profit" : "text-loss")}>
                    {flowAnalysis.avg_fii_net_5d >= 0 ? "+" : ""}
                    {flowAnalysis.avg_fii_net_5d.toFixed(0)} Cr
                  </div>
                  <div className="text-xs text-muted-foreground mt-1">
                    Streak: {flowAnalysis.fii_streak > 0 ? `+${flowAnalysis.fii_streak} days buying` : `${flowAnalysis.fii_streak} days selling`}
                  </div>
                </div>
                <div className="bg-card/50 rounded-lg p-3">
                  <div className="text-xs text-muted-foreground">DII 5D Avg Net</div>
                  <div className={cn("text-sm font-bold", flowAnalysis.avg_dii_net_5d >= 0 ? "text-profit" : "text-loss")}>
                    {flowAnalysis.avg_dii_net_5d >= 0 ? "+" : ""}
                    {flowAnalysis.avg_dii_net_5d.toFixed(0)} Cr
                  </div>
                  <div className="text-xs text-muted-foreground mt-1">
                    Streak: {flowAnalysis.dii_streak > 0 ? `+${flowAnalysis.dii_streak} days buying` : `${flowAnalysis.dii_streak} days selling`}
                  </div>
                </div>
              </div>
              <div className="text-xs text-muted-foreground p-2 bg-card/30 rounded-lg">
                {flowAnalysis.recommendation}
              </div>
            </div>
          )}

          {/* Flow Table */}
          <div className="glass-card rounded-xl overflow-hidden">
            <div className="p-4 border-b border-border/30">
              <span className="font-semibold text-sm">Recent FII/DII Flows (Cr)</span>
            </div>
            <div className="overflow-x-auto">
              <table className="w-full text-xs">
                <thead>
                  <tr className="border-b border-border/20 text-muted-foreground">
                    <th className="py-2 px-3 text-left">Date</th>
                    <th className="py-2 px-2 text-right">FII Net</th>
                    <th className="py-2 px-2 text-right">DII Net</th>
                    <th className="py-2 px-2 text-right">Total</th>
                  </tr>
                </thead>
                <tbody>
                  {(flows ?? []).map((f) => (
                    <tr key={f.date} className="border-b border-border/10 hover:bg-card/30">
                      <td className="py-1.5 px-3">{f.date}</td>
                      <td className={cn("py-1.5 px-2 text-right font-mono", f.fii_net >= 0 ? "text-profit" : "text-loss")}>
                        {f.fii_net >= 0 ? "+" : ""}{f.fii_net.toFixed(0)}
                      </td>
                      <td className={cn("py-1.5 px-2 text-right font-mono", f.dii_net >= 0 ? "text-profit" : "text-loss")}>
                        {f.dii_net >= 0 ? "+" : ""}{f.dii_net.toFixed(0)}
                      </td>
                      <td className={cn("py-1.5 px-2 text-right font-bold", f.total_net >= 0 ? "text-profit" : "text-loss")}>
                        {f.total_net >= 0 ? "+" : ""}{f.total_net.toFixed(0)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Symbol Sentiments */}
          {sentiment?.symbol_sentiments && Object.keys(sentiment.symbol_sentiments).length > 0 && (
            <div className="glass-card rounded-xl p-5">
              <span className="font-semibold text-sm mb-3 block">Symbol Sentiment</span>
              <div className="grid grid-cols-2 gap-2">
                {Object.entries(sentiment.symbol_sentiments).map(([sym, score]) => (
                  <div key={sym} className="flex items-center justify-between p-2 bg-card/50 rounded-lg">
                    <span className="text-xs font-medium">{sym}</span>
                    <div className="flex items-center gap-2">
                      <span className={cn("text-xs font-bold", score > 0.6 ? "text-profit" : score < 0.4 ? "text-loss" : "text-muted-foreground")}>
                        {(score * 100).toFixed(0)}%
                      </span>
                      {score > 0.6 ? <ArrowUpRight className="h-3 w-3 text-profit" /> : score < 0.4 ? <ArrowDownRight className="h-3 w-3 text-loss" /> : null}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
