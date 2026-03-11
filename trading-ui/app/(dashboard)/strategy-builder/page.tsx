"use client";

import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { motion, AnimatePresence } from "framer-motion";
import { api } from "@/lib/api/client";
import {
  Brain,
  Sparkles,
  Code2,
  Play,
  Copy,
  Check,
  AlertTriangle,
  Wand2,
  BookOpen,
  ChevronRight,
} from "lucide-react";
import { cn } from "@/lib/utils";

type Template = {
  name: string;
  prompt: string;
  category: string;
};

type GeneratedStrategy = {
  name: string;
  description: string;
  code: string;
  config: Record<string, unknown>;
  validated: boolean;
  errors: string[];
};

export default function StrategyBuilderPage() {
  const [prompt, setPrompt] = useState("");
  const [timeframe, setTimeframe] = useState("5m");
  const [copied, setCopied] = useState(false);
  const [showTemplates, setShowTemplates] = useState(true);
  const queryClient = useQueryClient();

  const { data: templates } = useQuery({
    queryKey: ["strategy-templates"],
    queryFn: () => api.get<Template[]>("/api/v1/strategy-builder/templates"),
  });

  const { data: existingStrategies } = useQuery({
    queryKey: ["generated-strategies"],
    queryFn: () =>
      api.get<Array<{ name: string; description: string; validated: boolean; created_at: string; prompt: string }>>(
        "/api/v1/strategy-builder/strategies"
      ),
    refetchInterval: 10000,
  });

  const generateMutation = useMutation({
    mutationFn: (body: { prompt: string; timeframe: string }) =>
      api.post<GeneratedStrategy>("/api/v1/strategy-builder/generate", body),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["generated-strategies"] });
      setShowTemplates(false);
    },
  });

  const result = generateMutation.data;

  const handleCopy = () => {
    if (result?.code) {
      navigator.clipboard.writeText(result.code);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };

  const handleTemplateClick = (template: Template) => {
    setPrompt(template.prompt);
    setShowTemplates(false);
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold flex items-center gap-2">
          <Brain className="h-7 w-7 text-primary" />
          AI Strategy Builder
        </h1>
        <p className="text-sm text-muted-foreground mt-1">
          Describe your trading idea in plain English — AI generates executable strategy code
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Input Panel */}
        <div className="lg:col-span-1 space-y-4">
          {/* Prompt Input */}
          <div className="glass-card rounded-xl p-5">
            <div className="flex items-center gap-2 mb-3">
              <Wand2 className="h-4 w-4 text-primary" />
              <span className="font-semibold text-sm">Describe Your Strategy</span>
            </div>
            <textarea
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              placeholder="e.g., Buy when RSI drops below 30 and MACD crosses above signal line. Take profit at 3%, stop loss at 1.5%. Use 5-minute candles."
              rows={6}
              className="w-full p-3 rounded-lg bg-card/50 border border-border/50 text-sm resize-none focus:outline-none focus:ring-1 focus:ring-primary/50 placeholder:text-muted-foreground/50"
            />

            <div className="flex items-center gap-3 mt-3">
              <select
                value={timeframe}
                onChange={(e) => setTimeframe(e.target.value)}
                className="px-3 py-2 rounded-lg bg-card border border-border/50 text-sm"
              >
                <option value="1m">1 Minute</option>
                <option value="5m">5 Minutes</option>
                <option value="15m">15 Minutes</option>
                <option value="1h">1 Hour</option>
                <option value="1d">1 Day</option>
              </select>

              <button
                onClick={() =>
                  generateMutation.mutate({ prompt, timeframe })
                }
                disabled={!prompt.trim() || generateMutation.isPending}
                className={cn(
                  "flex-1 flex items-center justify-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all",
                  prompt.trim()
                    ? "bg-primary text-white hover:bg-primary/90"
                    : "bg-card text-muted-foreground cursor-not-allowed"
                )}
              >
                {generateMutation.isPending ? (
                  <>
                    <Sparkles className="h-4 w-4 animate-spin" />
                    Generating...
                  </>
                ) : (
                  <>
                    <Sparkles className="h-4 w-4" />
                    Generate
                  </>
                )}
              </button>
            </div>
          </div>

          {/* Templates */}
          <AnimatePresence>
            {showTemplates && templates && (
              <motion.div
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: "auto" }}
                exit={{ opacity: 0, height: 0 }}
                className="glass-card rounded-xl p-5 overflow-hidden"
              >
                <div className="flex items-center gap-2 mb-3">
                  <BookOpen className="h-4 w-4 text-primary" />
                  <span className="font-semibold text-sm">Quick Templates</span>
                </div>
                <div className="space-y-2">
                  {templates.map((t) => (
                    <button
                      key={t.name}
                      onClick={() => handleTemplateClick(t)}
                      className="w-full flex items-center gap-3 p-3 rounded-lg bg-card/50 hover:bg-card/80 transition-colors text-left group"
                    >
                      <div className="flex-1 min-w-0">
                        <div className="text-sm font-medium">{t.name}</div>
                        <div className="text-xs text-muted-foreground truncate">
                          {t.prompt}
                        </div>
                      </div>
                      <ChevronRight className="h-4 w-4 text-muted-foreground group-hover:text-foreground transition-colors" />
                    </button>
                  ))}
                </div>
              </motion.div>
            )}
          </AnimatePresence>

          {/* Previously Generated */}
          {existingStrategies && existingStrategies.length > 0 && (
            <div className="glass-card rounded-xl p-5">
              <span className="font-semibold text-sm mb-3 block">Recent Strategies</span>
              <div className="space-y-2 max-h-64 overflow-y-auto">
                {existingStrategies.map((s) => (
                  <div
                    key={s.name}
                    className="flex items-center gap-2 p-2 rounded-lg bg-card/50 text-xs"
                  >
                    <div
                      className={cn(
                        "h-2 w-2 rounded-full",
                        s.validated ? "bg-profit" : "bg-yellow-400"
                      )}
                    />
                    <div className="flex-1 min-w-0">
                      <div className="font-medium truncate">{s.name}</div>
                      <div className="text-muted-foreground truncate">{s.prompt}</div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Output Panel */}
        <div className="lg:col-span-2">
          {result ? (
            <motion.div
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              className="glass-card rounded-xl overflow-hidden"
            >
              {/* Result Header */}
              <div className="flex items-center justify-between p-4 border-b border-border/30">
                <div className="flex items-center gap-3">
                  <Code2 className="h-5 w-5 text-primary" />
                  <div>
                    <h3 className="font-semibold text-sm">{result.name}</h3>
                    <p className="text-xs text-muted-foreground">{result.description}</p>
                  </div>
                </div>
                <div className="flex items-center gap-2">
                  {result.validated ? (
                    <span className="flex items-center gap-1 px-2 py-1 rounded-full bg-profit/10 text-profit text-xs">
                      <Check className="h-3 w-3" /> Valid
                    </span>
                  ) : (
                    <span className="flex items-center gap-1 px-2 py-1 rounded-full bg-yellow-400/10 text-yellow-400 text-xs">
                      <AlertTriangle className="h-3 w-3" /> Issues
                    </span>
                  )}
                  <button
                    onClick={handleCopy}
                    className="p-2 rounded-lg hover:bg-card/50 transition-colors"
                    title="Copy code"
                  >
                    {copied ? (
                      <Check className="h-4 w-4 text-profit" />
                    ) : (
                      <Copy className="h-4 w-4 text-muted-foreground" />
                    )}
                  </button>
                </div>
              </div>

              {/* Validation Errors */}
              {result.errors.length > 0 && (
                <div className="p-3 bg-yellow-400/5 border-b border-border/30">
                  {result.errors.map((err, i) => (
                    <div key={i} className="flex items-center gap-2 text-xs text-yellow-400">
                      <AlertTriangle className="h-3 w-3 shrink-0" />
                      {err}
                    </div>
                  ))}
                </div>
              )}

              {/* Code */}
              <div className="p-4 overflow-x-auto">
                <pre className="text-xs font-mono text-foreground/90 whitespace-pre-wrap leading-relaxed">
                  {result.code}
                </pre>
              </div>

              {/* Config Summary */}
              <div className="p-4 border-t border-border/30 bg-card/30">
                <div className="text-xs font-semibold mb-2">Strategy Config</div>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-xs">
                  <div>
                    <span className="text-muted-foreground">Timeframe:</span>{" "}
                    <span className="font-medium">
                      {(result.config as Record<string, string>).timeframe ?? timeframe}
                    </span>
                  </div>
                  <div>
                    <span className="text-muted-foreground">Indicators:</span>{" "}
                    <span className="font-medium">
                      {((result.config as Record<string, Array<{ name: string }>>).indicators ?? [])
                        .map((i) => i.name)
                        .join(", ")}
                    </span>
                  </div>
                  <div>
                    <span className="text-muted-foreground">Stop Loss:</span>{" "}
                    <span className="font-medium">
                      {(
                        (result.config as Record<string, Record<string, number>>).risk
                          ?.stop_loss_pct ?? 2
                      ).toFixed(1)}
                      %
                    </span>
                  </div>
                  <div>
                    <span className="text-muted-foreground">Take Profit:</span>{" "}
                    <span className="font-medium">
                      {(
                        (result.config as Record<string, Record<string, number>>).risk
                          ?.take_profit_pct ?? 4
                      ).toFixed(1)}
                      %
                    </span>
                  </div>
                </div>
              </div>
            </motion.div>
          ) : (
            <div className="glass-card rounded-xl p-12 flex flex-col items-center justify-center text-center">
              <Brain className="h-16 w-16 text-primary/20 mb-4" />
              <h3 className="font-semibold text-lg mb-2">AI Strategy Builder</h3>
              <p className="text-sm text-muted-foreground max-w-md">
                Describe your trading strategy in plain English using the prompt box.
                The AI will generate executable Python code with proper indicators,
                entry/exit rules, and risk management.
              </p>
              <div className="flex items-center gap-2 mt-4 text-xs text-muted-foreground">
                <Sparkles className="h-4 w-4" />
                Supports RSI, MACD, EMA, Bollinger Bands, VWAP, Supertrend and more
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
