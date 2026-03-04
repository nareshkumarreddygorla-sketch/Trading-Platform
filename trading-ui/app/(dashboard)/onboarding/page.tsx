"use client";

import { useState, useEffect, useCallback } from "react";
import { useRouter } from "next/navigation";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { motion, AnimatePresence } from "framer-motion";
import { cn } from "@/lib/utils";
import { useStore } from "@/store/useStore";
import { endpoints } from "@/lib/api/client";
import type { Strategy } from "@/types";
import {
  Plug, CheckCircle, XCircle, Shield, Eye, EyeOff,
  RefreshCw, ChevronRight, ChevronLeft, Zap, Bot,
  ToggleLeft, ToggleRight, Rocket, Gauge, Target,
  AlertTriangle, Sparkles, ArrowRight, Play,
} from "lucide-react";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface RiskLimitsForm {
  max_position_pct: number;
  max_daily_loss_pct: number;
  max_open_positions: number;
}

interface CredForm {
  api_key: string;
  client_id: string;
  password: string;
  totp_secret: string;
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const STEPS = [
  { id: 1, label: "Broker" },
  { id: 2, label: "Risk" },
  { id: 3, label: "Strategies" },
  { id: 4, label: "Launch" },
];

const RECOMMENDED_STRATEGIES = ["ema_crossover", "macd", "rsi", "ai_alpha"];

// ---------------------------------------------------------------------------
// Confetti component (pure CSS)
// ---------------------------------------------------------------------------

function Confetti() {
  const pieces = Array.from({ length: 50 });
  const colors = [
    "hsl(152 69% 53%)",
    "hsl(217 91% 60%)",
    "hsl(258 90% 66%)",
    "hsl(38 92% 50%)",
    "hsl(199 89% 48%)",
    "hsl(0 84% 60%)",
  ];

  return (
    <div className="fixed inset-0 pointer-events-none z-50 overflow-hidden">
      {pieces.map((_, i) => {
        const left = Math.random() * 100;
        const delay = Math.random() * 0.8;
        const duration = 2 + Math.random() * 2;
        const size = 4 + Math.random() * 8;
        const color = colors[i % colors.length];
        const rotation = Math.random() * 360;

        return (
          <motion.div
            key={i}
            initial={{ y: -20, x: `${left}vw`, opacity: 1, rotate: 0 }}
            animate={{
              y: "110vh",
              opacity: [1, 1, 0],
              rotate: rotation + 720,
              x: `${left + (Math.random() - 0.5) * 20}vw`,
            }}
            transition={{ duration, delay, ease: "easeIn" }}
            className="absolute"
            style={{
              width: size,
              height: size * (Math.random() > 0.5 ? 1 : 0.4),
              backgroundColor: color,
              borderRadius: Math.random() > 0.5 ? "50%" : "2px",
            }}
          />
        );
      })}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Step Indicator
// ---------------------------------------------------------------------------

function StepIndicator({ current }: { current: number }) {
  return (
    <div className="flex items-center justify-center gap-0">
      {STEPS.map((step, i) => {
        const isCompleted = current > step.id;
        const isActive = current === step.id;
        return (
          <div key={step.id} className="flex items-center">
            {/* Circle */}
            <div className="flex flex-col items-center">
              <motion.div
                animate={{
                  scale: isActive ? 1.1 : 1,
                  borderColor: isCompleted
                    ? "hsl(152 69% 53%)"
                    : isActive
                      ? "hsl(217 91% 60%)"
                      : "hsl(217 33% 22%)",
                  backgroundColor: isCompleted
                    ? "hsl(152 69% 53% / 0.15)"
                    : isActive
                      ? "hsl(217 91% 60% / 0.15)"
                      : "transparent",
                }}
                className="flex h-10 w-10 items-center justify-center rounded-full border-2 transition-colors"
              >
                {isCompleted ? (
                  <CheckCircle className="h-5 w-5 text-profit" />
                ) : (
                  <span
                    className={cn(
                      "text-sm font-bold",
                      isActive ? "text-primary" : "text-muted-foreground"
                    )}
                  >
                    {step.id}
                  </span>
                )}
              </motion.div>
              <span
                className={cn(
                  "text-[10px] font-medium mt-1.5",
                  isActive
                    ? "text-primary"
                    : isCompleted
                      ? "text-profit"
                      : "text-muted-foreground"
                )}
              >
                {step.label}
              </span>
            </div>
            {/* Connector line */}
            {i < STEPS.length - 1 && (
              <div
                className={cn(
                  "w-16 sm:w-24 h-0.5 mx-2 mb-5 rounded-full transition-colors duration-500",
                  current > step.id ? "bg-profit" : "bg-muted/50"
                )}
              />
            )}
          </div>
        );
      })}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Step 1 - Broker Connection
// ---------------------------------------------------------------------------

function StepBroker({
  onComplete,
  onSkip,
}: {
  onComplete: () => void;
  onSkip: () => void;
}) {
  const [credForm, setCredForm] = useState<CredForm>({
    api_key: "",
    client_id: "",
    password: "",
    totp_secret: "",
  });
  const [showPasswords, setShowPasswords] = useState(false);
  const [credError, setCredError] = useState("");
  const [credSuccess, setCredSuccess] = useState("");
  const queryClient = useQueryClient();

  const configureMutation = useMutation({
    mutationFn: (creds: CredForm) => endpoints.brokerConfigure(creds),
    onSuccess: (data) => {
      setCredSuccess(data.message);
      setCredError("");
      queryClient.invalidateQueries({ queryKey: ["broker-status"] });
      useStore.setState({
        broker: { connected: true, status: "connected" },
        tradingMode: "live",
      });
      // brief delay so user sees the success state
      setTimeout(onComplete, 1200);
    },
    onError: (err: Error) => {
      setCredError(err.message);
      setCredSuccess("");
    },
  });

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    setCredError("");
    setCredSuccess("");
    configureMutation.mutate(credForm);
  };

  return (
    <motion.div
      initial={{ opacity: 0, x: 40 }}
      animate={{ opacity: 1, x: 0 }}
      exit={{ opacity: 0, x: -40 }}
      transition={{ duration: 0.35 }}
      className="max-w-xl mx-auto"
    >
      <div className="text-center mb-8">
        <div className="flex justify-center mb-4">
          <div className="flex h-16 w-16 items-center justify-center rounded-2xl bg-primary/10 border border-primary/20">
            <Plug className="h-8 w-8 text-primary" />
          </div>
        </div>
        <h2 className="text-2xl font-bold tracking-tight">Connect Your Broker</h2>
        <p className="text-sm text-muted-foreground mt-2">
          Enter your Angel One credentials to start trading
        </p>
      </div>

      <div className="glass-card p-6">
        <form onSubmit={handleSubmit} className="space-y-4">
          <div className="grid gap-4 sm:grid-cols-2">
            <div>
              <label className="text-xs font-medium text-muted-foreground mb-1.5 block">
                API Key
              </label>
              <input
                type={showPasswords ? "text" : "password"}
                value={credForm.api_key}
                onChange={(e) =>
                  setCredForm({ ...credForm, api_key: e.target.value })
                }
                placeholder="Your Angel One API key"
                required
                className="w-full rounded-lg border border-border/50 bg-muted/10 px-3 py-2.5 text-sm font-mono focus:border-primary/50 focus:outline-none focus:ring-1 focus:ring-primary/20"
              />
            </div>
            <div>
              <label className="text-xs font-medium text-muted-foreground mb-1.5 block">
                Client ID
              </label>
              <input
                type="text"
                value={credForm.client_id}
                onChange={(e) =>
                  setCredForm({ ...credForm, client_id: e.target.value })
                }
                placeholder="e.g. A12345"
                required
                className="w-full rounded-lg border border-border/50 bg-muted/10 px-3 py-2.5 text-sm font-mono focus:border-primary/50 focus:outline-none focus:ring-1 focus:ring-primary/20"
              />
            </div>
            <div>
              <label className="text-xs font-medium text-muted-foreground mb-1.5 block">
                Password
              </label>
              <input
                type={showPasswords ? "text" : "password"}
                value={credForm.password}
                onChange={(e) =>
                  setCredForm({ ...credForm, password: e.target.value })
                }
                placeholder="Trading password"
                required
                className="w-full rounded-lg border border-border/50 bg-muted/10 px-3 py-2.5 text-sm font-mono focus:border-primary/50 focus:outline-none focus:ring-1 focus:ring-primary/20"
              />
            </div>
            <div>
              <label className="text-xs font-medium text-muted-foreground mb-1.5 block">
                TOTP Secret
              </label>
              <input
                type={showPasswords ? "text" : "password"}
                value={credForm.totp_secret}
                onChange={(e) =>
                  setCredForm({ ...credForm, totp_secret: e.target.value })
                }
                placeholder="Base32 TOTP secret"
                required
                className="w-full rounded-lg border border-border/50 bg-muted/10 px-3 py-2.5 text-sm font-mono focus:border-primary/50 focus:outline-none focus:ring-1 focus:ring-primary/20"
              />
            </div>
          </div>

          <button
            type="button"
            onClick={() => setShowPasswords(!showPasswords)}
            className="flex items-center gap-1.5 text-xs text-muted-foreground hover:text-primary transition-colors"
          >
            {showPasswords ? (
              <EyeOff className="h-3.5 w-3.5" />
            ) : (
              <Eye className="h-3.5 w-3.5" />
            )}
            {showPasswords ? "Hide" : "Show"} credentials
          </button>

          {credError && (
            <div className="rounded-lg bg-loss/10 border border-loss/30 p-3 flex items-center gap-2">
              <XCircle className="h-4 w-4 text-loss shrink-0" />
              <p className="text-xs text-loss font-medium">{credError}</p>
            </div>
          )}
          {credSuccess && (
            <div className="rounded-lg bg-profit/10 border border-profit/30 p-3 flex items-center gap-2">
              <CheckCircle className="h-4 w-4 text-profit shrink-0" />
              <p className="text-xs text-profit font-medium">{credSuccess}</p>
            </div>
          )}

          <button
            type="submit"
            disabled={configureMutation.isPending}
            className={cn(
              "w-full flex items-center justify-center gap-2 rounded-xl bg-primary py-3 text-sm font-semibold text-primary-foreground transition-all hover:bg-primary/90",
              configureMutation.isPending && "opacity-50 cursor-not-allowed"
            )}
          >
            {configureMutation.isPending ? (
              <RefreshCw className="h-4 w-4 animate-spin" />
            ) : (
              <Plug className="h-4 w-4" />
            )}
            {configureMutation.isPending ? "Connecting..." : "Connect"}
          </button>

          <p className="text-[10px] text-muted-foreground text-center">
            Credentials are validated against Angel One API. They are stored in
            memory only.
          </p>
        </form>
      </div>

      <button
        onClick={onSkip}
        className="mt-4 w-full text-center text-xs text-muted-foreground hover:text-primary transition-colors py-2"
      >
        Skip (Paper Mode) <ArrowRight className="inline h-3 w-3 ml-1" />
      </button>
    </motion.div>
  );
}

// ---------------------------------------------------------------------------
// Step 2 - Risk Configuration
// ---------------------------------------------------------------------------

function StepRisk({
  onComplete,
  onBack,
}: {
  onComplete: () => void;
  onBack: () => void;
}) {
  const [limits, setLimits] = useState<RiskLimitsForm>({
    max_position_pct: 5,
    max_daily_loss_pct: 2,
    max_open_positions: 10,
  });
  const [saved, setSaved] = useState(false);

  // Load current limits
  const { data: currentLimits } = useQuery({
    queryKey: ["risk-limits-onboarding"],
    queryFn: () => endpoints.riskLimits(),
  });

  useEffect(() => {
    if (currentLimits) {
      setLimits({
        max_position_pct: currentLimits.max_position_pct ?? 5,
        max_daily_loss_pct: currentLimits.max_daily_loss_pct ?? 2,
        max_open_positions: currentLimits.max_open_positions ?? 10,
      });
    }
  }, [currentLimits]);

  const saveMutation = useMutation({
    mutationFn: (body: RiskLimitsForm) => endpoints.updateRiskLimits(body),
    onSuccess: () => {
      setSaved(true);
      setTimeout(onComplete, 800);
    },
  });

  const handleSave = () => {
    saveMutation.mutate(limits);
  };

  const riskSliders: {
    key: keyof RiskLimitsForm;
    label: string;
    description: string;
    min: number;
    max: number;
    step: number;
    suffix: string;
    icon: typeof Gauge;
    color: string;
  }[] = [
    {
      key: "max_position_pct",
      label: "Max Position Size",
      description: "Maximum capital allocated per trade",
      min: 1,
      max: 10,
      step: 0.5,
      suffix: "%",
      icon: Target,
      color: "text-primary",
    },
    {
      key: "max_daily_loss_pct",
      label: "Daily Loss Limit",
      description: "Circuit breaker triggers at this loss level",
      min: 1,
      max: 5,
      step: 0.5,
      suffix: "%",
      icon: Shield,
      color: "text-warning",
    },
    {
      key: "max_open_positions",
      label: "Max Open Positions",
      description: "Maximum simultaneous trades",
      min: 3,
      max: 15,
      step: 1,
      suffix: "",
      icon: Gauge,
      color: "text-primary",
    },
  ];

  return (
    <motion.div
      initial={{ opacity: 0, x: 40 }}
      animate={{ opacity: 1, x: 0 }}
      exit={{ opacity: 0, x: -40 }}
      transition={{ duration: 0.35 }}
      className="max-w-xl mx-auto"
    >
      <div className="text-center mb-8">
        <div className="flex justify-center mb-4">
          <div className="flex h-16 w-16 items-center justify-center rounded-2xl bg-warning/10 border border-warning/20">
            <Shield className="h-8 w-8 text-warning" />
          </div>
        </div>
        <h2 className="text-2xl font-bold tracking-tight">Set Your Risk Limits</h2>
        <p className="text-sm text-muted-foreground mt-2">
          Configure safety guardrails to protect your capital
        </p>
      </div>

      <div className="glass-card p-6 space-y-6">
        {riskSliders.map((slider) => {
          const IconComp = slider.icon;
          return (
            <div key={slider.key}>
              <div className="flex items-center gap-2 mb-1">
                <IconComp className={cn("h-4 w-4", slider.color)} />
                <span className="text-sm font-medium">{slider.label}</span>
              </div>
              <p className="text-[10px] text-muted-foreground mb-3">
                {slider.description}
              </p>
              <div className="flex items-center gap-4">
                <input
                  type="range"
                  min={slider.min}
                  max={slider.max}
                  step={slider.step}
                  value={limits[slider.key]}
                  onChange={(e) =>
                    setLimits({
                      ...limits,
                      [slider.key]: Number(e.target.value),
                    })
                  }
                  className="flex-1 h-2 rounded-full appearance-none cursor-pointer accent-primary bg-muted"
                />
                <div className="w-16 text-right">
                  <span className="font-mono text-lg font-bold">
                    {limits[slider.key]}
                  </span>
                  <span className="text-xs text-muted-foreground">
                    {slider.suffix}
                  </span>
                </div>
              </div>
              <div className="flex justify-between mt-1">
                <span className="text-[9px] text-muted-foreground">
                  {slider.min}
                  {slider.suffix}
                </span>
                <span className="text-[9px] text-muted-foreground">
                  {slider.max}
                  {slider.suffix}
                </span>
              </div>
            </div>
          );
        })}

        {saved && (
          <motion.div
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            className="rounded-lg bg-profit/10 border border-profit/30 p-3 flex items-center gap-2"
          >
            <CheckCircle className="h-4 w-4 text-profit shrink-0" />
            <p className="text-xs text-profit font-medium">
              Risk limits saved successfully
            </p>
          </motion.div>
        )}
      </div>

      <div className="flex gap-3 mt-6">
        <button
          onClick={onBack}
          className="flex-1 flex items-center justify-center gap-2 rounded-xl border border-border/50 py-3 text-sm font-medium text-muted-foreground hover:text-foreground hover:border-border transition-all"
        >
          <ChevronLeft className="h-4 w-4" />
          Back
        </button>
        <button
          onClick={handleSave}
          disabled={saveMutation.isPending}
          className={cn(
            "flex-[2] flex items-center justify-center gap-2 rounded-xl bg-primary py-3 text-sm font-semibold text-primary-foreground transition-all hover:bg-primary/90",
            saveMutation.isPending && "opacity-50 cursor-not-allowed"
          )}
        >
          {saveMutation.isPending ? (
            <RefreshCw className="h-4 w-4 animate-spin" />
          ) : (
            <ChevronRight className="h-4 w-4" />
          )}
          {saveMutation.isPending ? "Saving..." : "Save & Continue"}
        </button>
      </div>
    </motion.div>
  );
}

// ---------------------------------------------------------------------------
// Step 3 - Strategy Selection
// ---------------------------------------------------------------------------

function StepStrategies({
  onComplete,
  onBack,
}: {
  onComplete: () => void;
  onBack: () => void;
}) {
  const queryClient = useQueryClient();
  const [togglingId, setTogglingId] = useState<string | null>(null);

  const { data, isLoading } = useQuery({
    queryKey: ["strategies-onboarding"],
    queryFn: async () => {
      const res = await endpoints.strategies();
      return (res.strategies ?? []) as Strategy[];
    },
  });

  const toggleMutation = useMutation({
    mutationFn: ({ id, enabled }: { id: string; enabled: boolean }) =>
      endpoints.toggleStrategy(id, enabled),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["strategies-onboarding"] });
      setTogglingId(null);
    },
    onError: () => {
      setTogglingId(null);
    },
  });

  const strategies = data ?? [];
  const activeCount = strategies.filter((s) => s.status === "active").length;

  // Auto-enable recommended strategies on first load
  const [autoEnabled, setAutoEnabled] = useState(false);
  useEffect(() => {
    if (strategies.length > 0 && !autoEnabled) {
      setAutoEnabled(true);
      // Enable recommended strategies that are currently inactive
      strategies.forEach((s) => {
        const isRecommended =
          RECOMMENDED_STRATEGIES.includes(s.id) ||
          RECOMMENDED_STRATEGIES.includes(s.name?.toLowerCase().replace(/\s+/g, "_") ?? "");
        if (isRecommended && s.status !== "active") {
          toggleMutation.mutate({ id: s.id, enabled: true });
        }
      });
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [strategies.length]);

  return (
    <motion.div
      initial={{ opacity: 0, x: 40 }}
      animate={{ opacity: 1, x: 0 }}
      exit={{ opacity: 0, x: -40 }}
      transition={{ duration: 0.35 }}
      className="max-w-2xl mx-auto"
    >
      <div className="text-center mb-8">
        <div className="flex justify-center mb-4">
          <div className="flex h-16 w-16 items-center justify-center rounded-2xl bg-primary/10 border border-primary/20">
            <Bot className="h-8 w-8 text-primary" />
          </div>
        </div>
        <h2 className="text-2xl font-bold tracking-tight">
          Choose Your Strategies
        </h2>
        <p className="text-sm text-muted-foreground mt-2">
          Toggle strategies on/off. Recommended ones are pre-selected.
        </p>
        <div className="inline-flex items-center gap-1.5 mt-3 px-3 py-1 rounded-full bg-primary/10 border border-primary/20">
          <Sparkles className="h-3 w-3 text-primary" />
          <span className="text-xs font-medium text-primary">
            {activeCount} active
          </span>
        </div>
      </div>

      <div className="grid gap-3 sm:grid-cols-2">
        {isLoading
          ? Array.from({ length: 4 }).map((_, i) => (
              <div key={i} className="glass-card p-4">
                <div className="shimmer h-5 w-32 rounded-lg mb-2" />
                <div className="shimmer h-3 w-48 rounded mb-3" />
                <div className="shimmer h-8 w-full rounded-lg" />
              </div>
            ))
          : strategies.map((s, i) => {
              const isActive = s.status === "active";
              const isRecommended =
                RECOMMENDED_STRATEGIES.includes(s.id) ||
                RECOMMENDED_STRATEGIES.includes(
                  s.name?.toLowerCase().replace(/\s+/g, "_") ?? ""
                );
              const isToggling = togglingId === s.id;

              return (
                <motion.div
                  key={s.id}
                  initial={{ opacity: 0, y: 12 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.05 + i * 0.04 }}
                >
                  <div
                    className={cn(
                      "glass-card p-4 transition-all duration-300",
                      isActive && "border-primary/25"
                    )}
                  >
                    <div className="flex items-start justify-between mb-2">
                      <div className="flex items-center gap-2">
                        <div
                          className={cn(
                            "flex h-8 w-8 items-center justify-center rounded-xl",
                            isActive
                              ? "bg-primary/10 text-primary"
                              : "bg-muted/50 text-muted-foreground"
                          )}
                        >
                          <Bot className="h-4 w-4" />
                        </div>
                        <div>
                          <h3 className="text-sm font-semibold">
                            {s.name || s.id}
                          </h3>
                          {isRecommended && (
                            <span className="text-[9px] font-medium text-primary bg-primary/10 px-1.5 py-0.5 rounded">
                              RECOMMENDED
                            </span>
                          )}
                        </div>
                      </div>
                      <button
                        onClick={() => {
                          setTogglingId(s.id);
                          toggleMutation.mutate({
                            id: s.id,
                            enabled: !isActive,
                          });
                        }}
                        disabled={isToggling}
                        className="transition-all"
                        title={
                          isActive ? "Disable strategy" : "Enable strategy"
                        }
                      >
                        {isActive ? (
                          <ToggleRight className="h-6 w-6 text-profit" />
                        ) : (
                          <ToggleLeft className="h-6 w-6 text-muted-foreground" />
                        )}
                      </button>
                    </div>
                    {s.description && (
                      <p className="text-[10px] text-muted-foreground line-clamp-2 mb-2">
                        {s.description}
                      </p>
                    )}
                    <div className="flex items-center gap-3">
                      {isActive ? (
                        <span className="status-badge-live">
                          <span className="h-1.5 w-1.5 rounded-full bg-profit animate-pulse" />
                          ACTIVE
                        </span>
                      ) : (
                        <span className="status-badge-warning">PAUSED</span>
                      )}
                      {s.win_rate != null && (
                        <span className="text-[10px] text-muted-foreground">
                          Win: {s.win_rate.toFixed(0)}%
                        </span>
                      )}
                    </div>
                  </div>
                </motion.div>
              );
            })}
      </div>

      {strategies.length === 0 && !isLoading && (
        <div className="glass-card p-8 text-center mt-4">
          <AlertTriangle className="h-8 w-8 text-warning mx-auto mb-3 opacity-60" />
          <p className="text-sm text-muted-foreground">
            No strategies found. They will be loaded once the backend is ready.
          </p>
        </div>
      )}

      <div className="flex gap-3 mt-6">
        <button
          onClick={onBack}
          className="flex-1 flex items-center justify-center gap-2 rounded-xl border border-border/50 py-3 text-sm font-medium text-muted-foreground hover:text-foreground hover:border-border transition-all"
        >
          <ChevronLeft className="h-4 w-4" />
          Back
        </button>
        <button
          onClick={onComplete}
          className="flex-[2] flex items-center justify-center gap-2 rounded-xl bg-primary py-3 text-sm font-semibold text-primary-foreground transition-all hover:bg-primary/90"
        >
          <ChevronRight className="h-4 w-4" />
          Continue
        </button>
      </div>
    </motion.div>
  );
}

// ---------------------------------------------------------------------------
// Step 4 - Review & Launch
// ---------------------------------------------------------------------------

function StepLaunch({
  onBack,
}: {
  onBack: () => void;
}) {
  const router = useRouter();
  const queryClient = useQueryClient();
  const [showConfetti, setShowConfetti] = useState(false);
  const [launched, setLaunched] = useState(false);

  const { data: brokerStatus } = useQuery({
    queryKey: ["broker-status"],
    queryFn: () => endpoints.brokerStatus(),
  });

  const { data: riskLimits } = useQuery({
    queryKey: ["risk-limits-review"],
    queryFn: () => endpoints.riskLimits(),
  });

  const { data: strategiesData } = useQuery({
    queryKey: ["strategies-review"],
    queryFn: async () => {
      const res = await endpoints.strategies();
      return (res.strategies ?? []) as Strategy[];
    },
  });

  const startMutation = useMutation({
    mutationFn: (enabled: boolean) => endpoints.toggleAutonomous(enabled),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["broker-status"] });
      setShowConfetti(true);
      setLaunched(true);
      // mark onboarding complete
      localStorage.setItem("onboarding_complete", "true");
      // redirect to dashboard after celebration
      setTimeout(() => {
        router.push("/dashboard");
      }, 3500);
    },
  });

  const handleLaunchLive = () => {
    startMutation.mutate(true);
  };

  const handleLaunchPaper = () => {
    localStorage.setItem("onboarding_complete", "true");
    setShowConfetti(true);
    setLaunched(true);
    setTimeout(() => {
      router.push("/dashboard");
    }, 2500);
  };

  const isConnected = brokerStatus?.connected ?? false;
  const activeStrategies =
    strategiesData?.filter((s) => s.status === "active").length ?? 0;

  const summaryItems = [
    {
      label: "Broker Status",
      value: isConnected ? "Connected (LIVE)" : "Paper Mode",
      color: isConnected ? "text-profit" : "text-warning",
      icon: isConnected ? CheckCircle : AlertTriangle,
    },
    {
      label: "Max Position Size",
      value: `${riskLimits?.max_position_pct ?? 5}%`,
      color: "text-foreground",
      icon: Target,
    },
    {
      label: "Daily Loss Limit",
      value: `${riskLimits?.max_daily_loss_pct ?? 2}%`,
      color: "text-warning",
      icon: Shield,
    },
    {
      label: "Max Positions",
      value: `${riskLimits?.max_open_positions ?? 10}`,
      color: "text-foreground",
      icon: Gauge,
    },
    {
      label: "Active Strategies",
      value: `${activeStrategies}`,
      color: "text-primary",
      icon: Bot,
    },
  ];

  return (
    <motion.div
      initial={{ opacity: 0, x: 40 }}
      animate={{ opacity: 1, x: 0 }}
      exit={{ opacity: 0, x: -40 }}
      transition={{ duration: 0.35 }}
      className="max-w-xl mx-auto"
    >
      {showConfetti && <Confetti />}

      <div className="text-center mb-8">
        <div className="flex justify-center mb-4">
          <motion.div
            animate={launched ? { scale: [1, 1.2, 1], rotate: [0, 10, -10, 0] } : {}}
            transition={{ duration: 0.6 }}
            className="flex h-16 w-16 items-center justify-center rounded-2xl bg-profit/10 border border-profit/20"
          >
            <Rocket className="h-8 w-8 text-profit" />
          </motion.div>
        </div>
        <h2 className="text-2xl font-bold tracking-tight">
          {launched ? "Trading Activated!" : "Ready to Trade!"}
        </h2>
        <p className="text-sm text-muted-foreground mt-2">
          {launched
            ? "Your autonomous trading system is now running"
            : "Review your configuration and launch"}
        </p>
      </div>

      {/* Summary card */}
      <div className="glass-card p-6 mb-6">
        <h3 className="text-sm font-semibold mb-4">Configuration Summary</h3>
        <div className="space-y-3">
          {summaryItems.map((item) => {
            const IconComp = item.icon;
            return (
              <div
                key={item.label}
                className="flex items-center justify-between py-2 border-b border-border/20 last:border-0"
              >
                <div className="flex items-center gap-2">
                  <IconComp className={cn("h-4 w-4", item.color)} />
                  <span className="text-xs text-muted-foreground">
                    {item.label}
                  </span>
                </div>
                <span className={cn("font-mono text-sm font-semibold", item.color)}>
                  {item.value}
                </span>
              </div>
            );
          })}
        </div>
      </div>

      {!launched && (
        <>
          {/* Launch buttons */}
          <div className="space-y-3">
            <button
              onClick={handleLaunchLive}
              disabled={startMutation.isPending}
              className={cn(
                "w-full flex items-center justify-center gap-3 rounded-xl py-4 text-base font-bold transition-all",
                "bg-gradient-to-r from-profit/20 to-profit/10 text-profit border border-profit/30 hover:from-profit/30 hover:to-profit/20",
                "shadow-lg shadow-profit/5",
                startMutation.isPending && "opacity-50 cursor-not-allowed"
              )}
            >
              {startMutation.isPending ? (
                <RefreshCw className="h-5 w-5 animate-spin" />
              ) : (
                <Zap className="h-5 w-5" />
              )}
              {startMutation.isPending
                ? "Starting..."
                : "Start Autonomous Trading"}
            </button>

            <button
              onClick={handleLaunchPaper}
              className="w-full flex items-center justify-center gap-2 rounded-xl border border-border/50 py-3 text-sm font-medium text-muted-foreground hover:text-foreground hover:border-border transition-all"
            >
              <Play className="h-4 w-4" />
              Start in Paper Mode
            </button>
          </div>

          <div className="flex gap-3 mt-4">
            <button
              onClick={onBack}
              className="flex-1 flex items-center justify-center gap-2 rounded-xl border border-border/50 py-2.5 text-sm font-medium text-muted-foreground hover:text-foreground hover:border-border transition-all"
            >
              <ChevronLeft className="h-4 w-4" />
              Back
            </button>
          </div>
        </>
      )}

      {launched && (
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.5 }}
          className="text-center"
        >
          <p className="text-xs text-muted-foreground">
            Redirecting to dashboard...
          </p>
        </motion.div>
      )}
    </motion.div>
  );
}

// ---------------------------------------------------------------------------
// Main Onboarding Page
// ---------------------------------------------------------------------------

export default function OnboardingPage() {
  const [step, setStep] = useState(1);
  const router = useRouter();

  // If already completed onboarding, redirect to dashboard
  useEffect(() => {
    if (typeof window !== "undefined") {
      const done = localStorage.getItem("onboarding_complete");
      if (done === "true") {
        router.replace("/dashboard");
      }
    }
  }, [router]);

  const goNext = useCallback(() => {
    setStep((prev) => Math.min(prev + 1, 4));
  }, []);

  const goBack = useCallback(() => {
    setStep((prev) => Math.max(prev - 1, 1));
  }, []);

  const skipToStep = useCallback((target: number) => {
    setStep(target);
  }, []);

  return (
    <div className="min-h-[calc(100vh-120px)] flex flex-col items-center justify-start py-8 px-4">
      {/* Step indicator */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="mb-10"
      >
        <StepIndicator current={step} />
      </motion.div>

      {/* Step content */}
      <div className="w-full max-w-2xl">
        <AnimatePresence mode="wait">
          {step === 1 && (
            <StepBroker
              key="step-1"
              onComplete={goNext}
              onSkip={() => skipToStep(2)}
            />
          )}
          {step === 2 && (
            <StepRisk key="step-2" onComplete={goNext} onBack={goBack} />
          )}
          {step === 3 && (
            <StepStrategies
              key="step-3"
              onComplete={goNext}
              onBack={goBack}
            />
          )}
          {step === 4 && <StepLaunch key="step-4" onBack={goBack} />}
        </AnimatePresence>
      </div>
    </div>
  );
}
