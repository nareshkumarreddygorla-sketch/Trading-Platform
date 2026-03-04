"use client";

import { useState, useEffect, useRef } from "react";
import { endpoints } from "@/lib/api/client";
import {
  RefreshCw,
  Brain,
  Cpu,
  CheckCircle2,
  XCircle,
  Loader2,
  ChevronDown,
  Zap,
  Rocket,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { motion, AnimatePresence } from "framer-motion";

interface ModelFile {
  exists: boolean;
  size_mb?: number;
  modified?: string;
}

interface TrainingStatus {
  is_training: boolean;
  started_at: string | null;
  mode: string | null;
  recent_logs: string[];
  last_result: { returncode: number; success: boolean } | null;
  last_training: { last_trained: string; results: Record<string, string> } | null;
  model_files: Record<string, ModelFile>;
}

const MODEL_ICONS: Record<string, string> = {
  xgboost: "XGB",
  lstm: "LSTM",
  transformer: "TF",
  rl: "RL",
};

const MODE_OPTIONS = [
  { value: "quick", label: "Quick", desc: "5 stocks, 10 epochs", icon: Zap },
  { value: "standard", label: "Standard", desc: "50 stocks, 30 epochs", icon: Brain },
  { value: "full", label: "Full", desc: "70 stocks, 50 epochs", icon: Rocket },
];

export default function TrainAIPanel() {
  const [status, setStatus] = useState<TrainingStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [selectedMode, setSelectedMode] = useState("quick");
  const [showLogs, setShowLogs] = useState(false);
  const [starting, setStarting] = useState(false);

  const isTrainingRef = useRef(false);
  const intervalRef = useRef<ReturnType<typeof setInterval>>();

  const refetchStatus = async () => {
    try {
      const data = await endpoints.trainingStatus();
      setStatus(data);
      // Adjust polling speed based on training state
      if (data?.is_training !== isTrainingRef.current) {
        isTrainingRef.current = !!data?.is_training;
        if (intervalRef.current) clearInterval(intervalRef.current);
        intervalRef.current = setInterval(refetchStatus, isTrainingRef.current ? 3000 : 15000);
      }
    } catch {
      // API not available
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    refetchStatus();
    intervalRef.current = setInterval(refetchStatus, 15000);
    return () => { if (intervalRef.current) clearInterval(intervalRef.current); };
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const handleTrain = async () => {
    setStarting(true);
    try {
      await endpoints.trainingStart({ mode: selectedMode });
      await refetchStatus();
    } catch (e: unknown) {
      alert(e instanceof Error ? e.message : "Failed to start training");
    } finally {
      setStarting(false);
    }
  };

  const handleStop = async () => {
    try {
      await endpoints.trainingStop();
      await refetchStatus();
    } catch {
      // ignore
    }
  };

  const lastTrained = status?.last_training?.last_trained;
  const lastTrainedStr = lastTrained
    ? new Date(lastTrained).toLocaleString("en-IN", {
        day: "numeric",
        month: "short",
        hour: "2-digit",
        minute: "2-digit",
      })
    : "Never";

  const modelCount = status?.model_files
    ? Object.values(status.model_files).filter((m) => m.exists).length
    : 0;

  if (loading) {
    return (
      <div className="glass-card p-5 animate-pulse">
        <div className="h-4 w-32 bg-muted/40 rounded mb-3" />
        <div className="h-10 bg-muted/20 rounded" />
      </div>
    );
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.4 }}
    >
      <div className="glass-card p-5" role="region" aria-label="AI Model Training">
        {/* Header */}
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-2">
            <Cpu className="h-4 w-4 text-primary" />
            <h3 className="text-sm font-semibold">AI Model Training</h3>
            {status?.is_training && (
              <span className="flex items-center gap-1 text-[9px] font-bold px-1.5 py-0.5 rounded-full bg-amber-500/15 text-amber-400">
                <Loader2 className="h-2.5 w-2.5 animate-spin" />
                TRAINING
              </span>
            )}
          </div>
          <div className="text-[10px] text-muted-foreground">
            Last: {lastTrainedStr}
          </div>
        </div>

        {/* Model Status Badges */}
        <div className="flex gap-2 mb-4">
          {Object.entries(status?.model_files || {}).map(([name, info]) => (
            <div
              key={name}
              className={cn(
                "flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg text-[10px] font-bold tracking-wide transition-all",
                info.exists
                  ? "bg-profit/10 text-profit"
                  : "bg-muted/30 text-muted-foreground"
              )}
            >
              {info.exists ? (
                <CheckCircle2 className="h-3 w-3" />
              ) : (
                <XCircle className="h-3 w-3" />
              )}
              {MODEL_ICONS[name] || name.toUpperCase()}
              {info.size_mb !== undefined && (
                <span className="text-[8px] opacity-60">{info.size_mb}MB</span>
              )}
            </div>
          ))}
        </div>

        {/* Training Controls */}
        {status?.is_training ? (
          <div className="space-y-3">
            {/* Progress indicator */}
            <div className="relative h-1.5 rounded-full bg-muted/30 overflow-hidden">
              <motion.div
                className="absolute inset-y-0 left-0 bg-gradient-to-r from-primary to-profit rounded-full"
                animate={{ width: ["0%", "100%"] }}
                transition={{ duration: 20, repeat: Infinity, ease: "linear" }}
              />
            </div>
            <div className="flex items-center justify-between">
              <span className="text-xs text-muted-foreground">
                Mode: <span className="text-foreground font-medium">{status.mode}</span>
              </span>
              <button
                onClick={handleStop}
                className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium bg-loss/10 text-loss hover:bg-loss/20 transition-colors"
                aria-label="Stop training"
              >
                <XCircle className="h-3 w-3" />
                Stop
              </button>
            </div>
          </div>
        ) : (
          <div className="space-y-3">
            {/* Mode Selector */}
            <div className="flex gap-1.5">
              {MODE_OPTIONS.map((mode) => (
                <button
                  key={mode.value}
                  onClick={() => setSelectedMode(mode.value)}
                  aria-label={`${mode.label} training mode: ${mode.desc}`}
                  aria-pressed={selectedMode === mode.value}
                  className={cn(
                    "flex-1 flex flex-col items-center gap-0.5 p-2 rounded-lg text-[10px] transition-all duration-200",
                    selectedMode === mode.value
                      ? "bg-primary/15 text-primary border border-primary/20"
                      : "bg-muted/20 text-muted-foreground hover:bg-muted/30 border border-transparent"
                  )}
                >
                  <mode.icon className="h-3.5 w-3.5" />
                  <span className="font-bold">{mode.label}</span>
                  <span className="opacity-60">{mode.desc}</span>
                </button>
              ))}
            </div>

            {/* Train Button */}
            <button
              onClick={handleTrain}
              disabled={starting}
              aria-label={starting ? "Training in progress" : "Train AI models"}
              aria-busy={starting}
              className={cn(
                "w-full flex items-center justify-center gap-2 py-2.5 rounded-xl text-xs font-bold transition-all duration-300",
                starting
                  ? "bg-primary/10 text-primary/60 cursor-wait"
                  : "bg-gradient-to-r from-primary/20 to-profit/20 text-primary hover:from-primary/30 hover:to-profit/30 hover:shadow-[0_0_20px_hsl(217_91%_60%_/_0.1)]"
              )}
            >
              {starting ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <RefreshCw className="h-4 w-4" />
              )}
              {starting ? "Starting..." : "Train AI Models"}
            </button>
          </div>
        )}

        {/* Logs Toggle */}
        {(status?.recent_logs?.length ?? 0) > 0 && (
          <div className="mt-3">
            <button
              onClick={() => setShowLogs(!showLogs)}
              className="flex items-center gap-1 text-[10px] text-muted-foreground hover:text-foreground transition-colors"
              aria-expanded={showLogs}
              aria-label={showLogs ? "Hide training logs" : "Show training logs"}
            >
              <ChevronDown
                className={cn(
                  "h-3 w-3 transition-transform",
                  showLogs && "rotate-180"
                )}
              />
              {showLogs ? "Hide" : "Show"} Logs
            </button>
            <AnimatePresence>
              {showLogs && (
                <motion.div
                  initial={{ height: 0, opacity: 0 }}
                  animate={{ height: "auto", opacity: 1 }}
                  exit={{ height: 0, opacity: 0 }}
                  className="overflow-hidden"
                >
                  <div className="mt-2 max-h-32 overflow-y-auto rounded-lg bg-black/30 p-2.5 font-mono text-[10px] text-muted-foreground leading-relaxed">
                    {status?.recent_logs?.map((line, i) => (
                      <div key={i} className="whitespace-pre-wrap">
                        {line}
                      </div>
                    ))}
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        )}

        {/* Last Result */}
        {status?.last_result && !status.is_training && (
          <div className="mt-3 flex items-center gap-2 text-[10px]">
            {status.last_result.success ? (
              <>
                <CheckCircle2 className="h-3 w-3 text-profit" />
                <span className="text-profit font-medium">Last training completed successfully</span>
              </>
            ) : (
              <>
                <XCircle className="h-3 w-3 text-loss" />
                <span className="text-loss font-medium">Last training had errors</span>
              </>
            )}
          </div>
        )}

        {/* Last Training Results */}
        {status?.last_training?.results && !status.is_training && (
          <div className="mt-2 flex flex-wrap gap-1.5">
            {Object.entries(status.last_training.results).map(([model, result]) => (
              <span
                key={model}
                className={cn(
                  "px-2 py-0.5 rounded text-[9px] font-bold",
                  result === "success"
                    ? "bg-profit/10 text-profit"
                    : result === "skipped"
                      ? "bg-muted/30 text-muted-foreground"
                      : "bg-loss/10 text-loss"
                )}
              >
                {model}: {result}
              </span>
            ))}
          </div>
        )}
      </div>
    </motion.div>
  );
}
