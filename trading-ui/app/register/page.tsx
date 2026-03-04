"use client";

import { useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { motion } from "framer-motion";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { api } from "@/lib/api/client";
import { Zap, ArrowRight, Eye, EyeOff, UserPlus, CheckCircle } from "lucide-react";

export default function RegisterPage() {
  const router = useRouter();
  const [username, setUsername] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [success, setSuccess] = useState(false);
  const [loading, setLoading] = useState(false);
  const [showPassword, setShowPassword] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    setLoading(true);
    try {
      await api.post("/api/v1/auth/register", { username, email, password });
      setSuccess(true);
      setTimeout(() => {
        router.push("/login");
        router.refresh();
      }, 1500);
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Registration failed";
      setError(msg);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="relative flex min-h-screen items-center justify-center bg-background p-4 overflow-hidden">
      {/* Animated background */}
      <div className="absolute inset-0 overflow-hidden">
        <div className="absolute -top-40 -right-40 h-80 w-80 rounded-full bg-[hsl(258,90%,66%)]/8 blur-3xl animate-pulse" />
        <div className="absolute -bottom-40 -left-40 h-80 w-80 rounded-full bg-primary/8 blur-3xl animate-pulse" style={{ animationDelay: "1s" }} />
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 h-96 w-96 rounded-full bg-[hsl(199,89%,48%)]/5 blur-3xl" />
      </div>
      <div className="absolute inset-0 grid-pattern opacity-30" />

      <motion.div
        initial={{ opacity: 0, y: 24, scale: 0.96 }}
        animate={{ opacity: 1, y: 0, scale: 1 }}
        transition={{ duration: 0.6, ease: [0.16, 1, 0.3, 1] }}
        className="relative w-full max-w-md"
      >
        {/* Logo */}
        <motion.div
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="flex flex-col items-center mb-8"
        >
          <div className="relative flex h-14 w-14 items-center justify-center rounded-2xl bg-gradient-to-br from-[hsl(258,90%,66%)] via-primary to-[hsl(199,89%,48%)] shadow-2xl mb-4">
            <UserPlus className="h-7 w-7 text-white" />
            <div className="absolute inset-0 rounded-2xl bg-gradient-to-br from-primary/30 to-transparent animate-pulse" />
          </div>
          <h1 className="text-2xl font-bold tracking-tight text-gradient">
            Join AlphaForge
          </h1>
          <p className="text-sm text-muted-foreground mt-1">
            Create your trading account
          </p>
        </motion.div>

        {/* Register card */}
        <div className="glass-card p-8">
          <form onSubmit={handleSubmit} className="space-y-5" aria-label="Create a new AlphaForge account">
            <div className="space-y-2">
              <Label htmlFor="username" className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                Username
              </Label>
              <Input
                id="username"
                type="text"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                placeholder="Choose a username"
                className="h-11 rounded-xl border-border/50 bg-muted/30 px-4 text-sm transition-all focus:border-primary/50 focus:ring-2 focus:ring-primary/20 focus:bg-muted/50"
                required
                aria-required="true"
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="email" className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                Email <span className="text-muted-foreground/50">(optional)</span>
              </Label>
              <Input
                id="email"
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="your@email.com"
                className="h-11 rounded-xl border-border/50 bg-muted/30 px-4 text-sm transition-all focus:border-primary/50 focus:ring-2 focus:ring-primary/20 focus:bg-muted/50"
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="password" className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                Password
              </Label>
              <div className="relative">
                <Input
                  id="password"
                  type={showPassword ? "text" : "password"}
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  placeholder="Create a strong password"
                  className="h-11 rounded-xl border-border/50 bg-muted/30 px-4 pr-10 text-sm transition-all focus:border-primary/50 focus:ring-2 focus:ring-primary/20 focus:bg-muted/50"
                  required
                  aria-required="true"
                />
                <button
                  type="button"
                  onClick={() => setShowPassword(!showPassword)}
                  aria-label={showPassword ? "Hide password" : "Show password"}
                  aria-pressed={showPassword}
                  className="absolute right-3 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground transition-colors"
                >
                  {showPassword ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                </button>
              </div>
            </div>

            {success && (
              <motion.div
                initial={{ opacity: 0, y: -4 }}
                animate={{ opacity: 1, y: 0 }}
                className="flex items-center gap-2 rounded-xl bg-profit/10 border border-profit/20 p-3"
                role="status"
              >
                <CheckCircle className="h-4 w-4 text-profit shrink-0" />
                <p className="text-xs text-profit">Account created! Redirecting to login…</p>
              </motion.div>
            )}
            {error && (
              <motion.div
                initial={{ opacity: 0, y: -4 }}
                animate={{ opacity: 1, y: 0 }}
                className="rounded-xl bg-loss/10 border border-loss/20 p-3"
                role="alert"
              >
                <p className="text-xs text-loss">{error}</p>
              </motion.div>
            )}

            <Button
              type="submit"
              disabled={loading || success}
              aria-label={loading ? "Creating account" : "Create account"}
              aria-busy={loading}
              className="relative w-full h-11 rounded-xl bg-gradient-to-r from-[hsl(258,90%,66%)] to-primary font-semibold text-white shadow-lg transition-all hover:shadow-xl hover:shadow-primary/20 hover:brightness-110 disabled:opacity-50"
            >
              {loading ? (
                <motion.div
                  className="h-5 w-5 rounded-full border-2 border-white/30 border-t-white"
                  animate={{ rotate: 360 }}
                  transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
                />
              ) : (
                <span className="flex items-center gap-2">
                  Create Account
                  <ArrowRight className="h-4 w-4" />
                </span>
              )}
            </Button>
          </form>

          <div className="mt-6 flex items-center gap-3">
            <div className="flex-1 h-px bg-border/50" />
            <span className="text-[10px] font-medium uppercase tracking-wider text-muted-foreground">
              Already have an account?
            </span>
            <div className="flex-1 h-px bg-border/50" />
          </div>

          <Link href="/login" aria-label="Sign in to existing account" className="mt-4 flex w-full items-center justify-center gap-2 rounded-xl border border-border/50 py-2.5 text-sm font-medium text-muted-foreground transition-all hover:border-primary/40 hover:text-primary hover:bg-primary/5">
            Sign In Instead
          </Link>
        </div>

        <p className="mt-6 text-center text-[10px] text-muted-foreground/50">
          Enterprise AI Trading Platform • Powered by AlphaForge
        </p>
      </motion.div>
    </div>
  );
}
