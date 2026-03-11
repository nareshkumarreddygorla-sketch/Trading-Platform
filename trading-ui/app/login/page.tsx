"use client";

import { useState, Suspense } from "react";
import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { motion, AnimatePresence } from "framer-motion";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { endpoints, setAuthTokens } from "@/lib/api/client";
import {
  Zap, ArrowRight, Eye, EyeOff, TrendingUp, TrendingDown,
  BarChart3, Shield, Lock, User,
} from "lucide-react";

/* ── Fake ticker data ── */
const tickers = [
  { sym: "RELIANCE", price: 1406.80, change: +0.44 },
  { sym: "TCS", price: 3842.15, change: -0.32 },
  { sym: "HDFCBANK", price: 1678.50, change: +1.12 },
  { sym: "INFY", price: 1524.30, change: +0.78 },
  { sym: "ITC", price: 442.65, change: -0.15 },
  { sym: "SBIN", price: 785.40, change: +2.31 },
  { sym: "BHARTIARTL", price: 1245.70, change: +0.56 },
  { sym: "WIPRO", price: 478.90, change: -1.02 },
  { sym: "TATAMOTORS", price: 892.30, change: +1.88 },
  { sym: "NIFTY50", price: 22847.50, change: +0.34 },
];

/* ── Deterministic pseudo-random for SSR consistency ── */
function seededRandom(seed: number) {
  const x = Math.sin(seed * 9301 + 49297) * 49297;
  return x - Math.floor(x);
}

/* ── Candlestick silhouette SVG path ── */
function CandlestickBg() {
  return (
    <svg
      className="absolute inset-0 w-full h-full opacity-[0.04]"
      viewBox="0 0 1200 600"
      preserveAspectRatio="none"
    >
      {Array.from({ length: 40 }).map((_, i) => {
        const x = 30 + i * 29;
        const bodyH = Math.round(10 + seededRandom(i * 3 + 1) * 60);
        const wickH = Math.round(bodyH + 10 + seededRandom(i * 3 + 2) * 30);
        const baseY = Math.round(200 + Math.sin(i * 0.3) * 80 + seededRandom(i * 3 + 3) * 60);
        const isGreen = seededRandom(i * 7) > 0.45;
        return (
          <g key={i}>
            <line
              x1={x} y1={baseY - Math.round(wickH / 2)}
              x2={x} y2={baseY + Math.round(wickH / 2)}
              stroke={isGreen ? "hsl(152,69%,53%)" : "hsl(0,84%,60%)"}
              strokeWidth="1.5"
            />
            <rect
              x={x - 6} y={baseY - Math.round(bodyH / 2)}
              width="12" height={bodyH}
              fill={isGreen ? "hsl(152,69%,53%)" : "hsl(0,84%,60%)"}
              rx="2"
            />
          </g>
        );
      })}
      <path
        d="M0,350 Q150,300 300,280 T600,250 T900,220 T1200,260"
        fill="none"
        stroke="hsl(217,91%,60%)"
        strokeWidth="2"
        opacity="0.5"
      />
    </svg>
  );
}

/* ── Floating particles ── */
function Particles() {
  return (
    <div className="absolute inset-0 overflow-hidden pointer-events-none">
      {Array.from({ length: 20 }).map((_, i) => {
        const leftPct = Math.round(seededRandom(i * 11 + 5) * 100);
        const size = Math.round(2 + seededRandom(i * 13 + 7) * 4);
        const duration = Math.round(8 + seededRandom(i * 17 + 3) * 12);
        const delay = Math.round(seededRandom(i * 19 + 1) * 8);
        return (
          <div
            key={i}
            className="particle"
            style={{
              left: `${leftPct}%`,
              width: `${size}px`,
              height: `${size}px`,
              background: [
                "hsl(217 91% 60% / 0.4)",
                "hsl(258 90% 66% / 0.3)",
                "hsl(152 69% 53% / 0.3)",
                "hsl(199 89% 48% / 0.3)",
              ][i % 4],
              animationDuration: `${duration}s`,
              animationDelay: `${delay}s`,
            }}
          />
        );
      })}
    </div>
  );
}

/* ── Live ticker strip ── */
function TickerStrip() {
  const doubled = [...tickers, ...tickers]; // infinite loop illusion
  return (
    <div className="absolute bottom-0 left-0 right-0 border-t border-border/20 bg-background/60 backdrop-blur-lg overflow-hidden">
      <div className="ticker-tape py-2.5">
        {doubled.map((t, i) => (
          <div key={i} className="flex items-center gap-2 px-5 shrink-0">
            <span className="text-[11px] font-semibold text-foreground/70">{t.sym}</span>
            <span className="text-[11px] font-mono text-foreground/80" suppressHydrationWarning>₹{t.price.toLocaleString("en-IN")}</span>
            <span className={`text-[10px] font-mono font-semibold flex items-center gap-0.5 ${t.change >= 0 ? "text-profit" : "text-loss"}`}>
              {t.change >= 0 ? <TrendingUp className="h-2.5 w-2.5" /> : <TrendingDown className="h-2.5 w-2.5" />}
              {t.change >= 0 ? "+" : ""}{t.change}%
            </span>
            <div className="w-px h-3 bg-border/30 mx-2" />
          </div>
        ))}
      </div>
    </div>
  );
}

function LoginPageInner() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const redirectTo = searchParams.get("redirect") || "/dashboard";
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [showPassword, setShowPassword] = useState(false);
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    setLoading(true);
    try {
      const res = await endpoints.login({ username, password });
      if (typeof window !== "undefined" && res.access_token) {
        setAuthTokens(res.access_token, res.refresh_token);
        router.push(redirectTo);
        router.refresh();
      }
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Login failed";
      setError(
        msg === "Unauthorized"
          ? "Invalid username or password. If you just restarted the server, register again first."
          : msg
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="relative flex min-h-screen items-center justify-center bg-background overflow-hidden gradient-mesh particles-bg">
      {/* Background layers */}
      <CandlestickBg />
      <Particles />

      {/* Animated gradient orbs */}
      <div className="absolute inset-0 pointer-events-none">
        <motion.div
          className="absolute -top-40 -left-40 h-[500px] w-[500px] rounded-full bg-primary/6 blur-[100px]"
          animate={{ x: [0, 30, -20, 0], y: [0, -20, 30, 0] }}
          transition={{ duration: 20, repeat: Infinity, ease: "linear" }}
        />
        <motion.div
          className="absolute -bottom-40 -right-40 h-[500px] w-[500px] rounded-full bg-[hsl(258,90%,66%)]/6 blur-[100px]"
          animate={{ x: [0, -30, 20, 0], y: [0, 20, -30, 0] }}
          transition={{ duration: 25, repeat: Infinity, ease: "linear" }}
        />
        <motion.div
          className="absolute top-1/3 left-1/3 h-[400px] w-[400px] rounded-full bg-profit/3 blur-[120px]"
          animate={{ scale: [1, 1.1, 0.95, 1] }}
          transition={{ duration: 15, repeat: Infinity, ease: "easeInOut" }}
        />
      </div>

      {/* Grid pattern */}
      <div className="absolute inset-0 grid-pattern opacity-20" />

      {/* Main content */}
      <div className="relative z-10 w-full max-w-[440px] px-4">
        {/* Brand */}
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, ease: [0.16, 1, 0.3, 1] }}
          className="flex flex-col items-center mb-10"
        >
          <motion.div
            className="relative flex h-16 w-16 items-center justify-center rounded-2xl bg-gradient-to-br from-primary via-[hsl(258,90%,66%)] to-[hsl(199,89%,48%)] shadow-2xl mb-5"
            animate={{ rotate: [0, 2, -2, 0] }}
            transition={{ duration: 6, repeat: Infinity, ease: "easeInOut" }}
          >
            <Zap className="h-8 w-8 text-white" />
            <div className="absolute inset-0 rounded-2xl bg-gradient-to-br from-white/10 to-transparent" />
            <div className="absolute -inset-1 rounded-2xl bg-gradient-to-br from-primary/30 to-transparent blur-xl animate-pulse" />
          </motion.div>
          <h1 className="text-3xl font-bold tracking-tight text-gradient">
            AlphaForge
          </h1>
          <p className="text-sm text-muted-foreground mt-1.5 tracking-wide">
            AI-Powered Autonomous Trading
          </p>
        </motion.div>

        {/* Login card */}
        <motion.div
          initial={{ opacity: 0, y: 24, scale: 0.96 }}
          animate={{ opacity: 1, y: 0, scale: 1 }}
          transition={{ duration: 0.7, delay: 0.15, ease: [0.16, 1, 0.3, 1] }}
          className="glass-card gradient-border p-8"
        >
          <div className="mb-7">
            <h2 className="text-lg font-bold tracking-tight">Welcome back</h2>
            <p className="text-sm text-muted-foreground mt-1">
              Sign in to your trading command center
            </p>
          </div>

          <form onSubmit={handleSubmit} className="space-y-5" aria-label="Sign in to AlphaForge">
            <motion.div
              initial={{ opacity: 0, x: -10 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.3 }}
              className="space-y-2"
            >
              <Label htmlFor="username" className="text-[10px] font-semibold uppercase tracking-[0.12em] text-muted-foreground">
                Username
              </Label>
              <div className="relative">
                <User className="absolute left-3.5 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground/50" />
                <Input
                  id="username"
                  type="text"
                  value={username}
                  onChange={(e) => setUsername(e.target.value)}
                  placeholder="Enter your username"
                  className="h-12 rounded-xl border-border/40 bg-muted/20 pl-10 pr-4 text-sm transition-all duration-300 focus:border-primary/40 focus:ring-2 focus:ring-primary/15 focus:bg-muted/30 focus:shadow-[0_0_20px_hsl(217_91%_60%_/_0.08)] placeholder:text-muted-foreground/30"
                  required
                  aria-required="true"
                />
              </div>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, x: -10 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.4 }}
              className="space-y-2"
            >
              <Label htmlFor="password" className="text-[10px] font-semibold uppercase tracking-[0.12em] text-muted-foreground">
                Password
              </Label>
              <div className="relative">
                <Lock className="absolute left-3.5 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground/50" />
                <Input
                  id="password"
                  type={showPassword ? "text" : "password"}
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  placeholder="Enter your password"
                  className="h-12 rounded-xl border-border/40 bg-muted/20 pl-10 pr-12 text-sm transition-all duration-300 focus:border-primary/40 focus:ring-2 focus:ring-primary/15 focus:bg-muted/30 focus:shadow-[0_0_20px_hsl(217_91%_60%_/_0.08)] placeholder:text-muted-foreground/30"
                  required
                  aria-required="true"
                />
                <button
                  type="button"
                  onClick={() => setShowPassword(!showPassword)}
                  aria-label={showPassword ? "Hide password" : "Show password"}
                  aria-pressed={showPassword}
                  className="absolute right-3.5 top-1/2 -translate-y-1/2 text-muted-foreground/50 hover:text-foreground transition-colors duration-200"
                >
                  {showPassword ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                </button>
              </div>
            </motion.div>

            <AnimatePresence>
              {error && (
                <motion.div
                  initial={{ opacity: 0, y: -8, height: 0 }}
                  animate={{ opacity: 1, y: 0, height: "auto" }}
                  exit={{ opacity: 0, y: -8, height: 0 }}
                  className="rounded-xl bg-loss/8 border border-loss/15 p-3.5"
                  role="alert"
                >
                  <p className="text-xs text-loss font-medium">{error}</p>
                </motion.div>
              )}
            </AnimatePresence>

            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.5 }}
            >
              <Button
                type="submit"
                disabled={loading || !username.trim() || !password.trim()}
                aria-label={loading ? "Signing in" : "Sign in"}
                aria-busy={loading}
                className="relative w-full h-12 rounded-xl bg-gradient-to-r from-primary via-[hsl(240,80%,58%)] to-[hsl(258,90%,66%)] font-semibold text-white shadow-lg transition-all duration-300 hover:shadow-2xl hover:shadow-primary/25 hover:brightness-110 disabled:opacity-50 overflow-hidden group"
              >
                {/* Shine effect */}
                <div className="absolute inset-0 opacity-0 group-hover:opacity-100 transition-opacity duration-500">
                  <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/10 to-transparent -skew-x-12 translate-x-[-200%] group-hover:translate-x-[200%] transition-transform duration-1000" />
                </div>

                {loading ? (
                  <motion.div
                    className="h-5 w-5 rounded-full border-2 border-white/30 border-t-white"
                    animate={{ rotate: 360 }}
                    transition={{ duration: 0.8, repeat: Infinity, ease: "linear" }}
                  />
                ) : (
                  <span className="flex items-center gap-2.5 relative z-10">
                    Sign In
                    <ArrowRight className="h-4 w-4 group-hover:translate-x-1 transition-transform" />
                  </span>
                )}
              </Button>
            </motion.div>
          </form>

          <div className="mt-7 flex items-center gap-3">
            <div className="flex-1 h-px bg-gradient-to-r from-transparent via-border/50 to-transparent" />
            <span className="text-[10px] font-semibold uppercase tracking-[0.15em] text-muted-foreground/50">
              New here?
            </span>
            <div className="flex-1 h-px bg-gradient-to-r from-transparent via-border/50 to-transparent" />
          </div>

          <Link
            href="/register"
            aria-label="Create a new account"
            className="mt-4 flex w-full items-center justify-center gap-2 rounded-xl border border-border/30 py-3 text-sm font-medium text-muted-foreground transition-all duration-300 hover:border-primary/30 hover:text-primary hover:bg-primary/5 hover:shadow-[0_0_20px_hsl(217_91%_60%_/_0.06)]"
          >
            Create an Account
          </Link>
        </motion.div>

        {/* Trust indicators */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.8 }}
          className="mt-8 flex items-center justify-center gap-6"
        >
          {[
            { icon: Shield, label: "Bank-Grade Security" },
            { icon: BarChart3, label: "Real-Time Analytics" },
            { icon: Zap, label: "AI-Powered" },
          ].map((item) => (
            <div key={item.label} className="flex items-center gap-1.5 text-muted-foreground/30">
              <item.icon className="h-3 w-3" />
              <span className="text-[9px] font-medium tracking-wider uppercase">{item.label}</span>
            </div>
          ))}
        </motion.div>

        {/* Footer */}
        <p className="mt-4 text-center text-[9px] text-muted-foreground/25 tracking-wider">
          © 2026 AlphaForge Technologies • Enterprise AI Trading
        </p>
      </div>

      {/* Ticker strip at bottom */}
      <TickerStrip />
    </div>
  );
}

export default function LoginPage() {
  return (
    <Suspense>
      <LoginPageInner />
    </Suspense>
  );
}
