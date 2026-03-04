"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { motion, AnimatePresence } from "framer-motion";
import {
  LayoutDashboard,
  Bot,
  Layers,
  TrendingUp,
  ShieldAlert,
  Plug,
  FileText,
  Settings,
  Menu,
  X,
  Zap,
  Activity,
  FlaskConical,
  History,
  Newspaper,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { endpoints } from "@/lib/api/client";
import { useStore } from "@/store/useStore";

export function Sidebar() {
  const pathname = usePathname();
  const [mobileOpen, setMobileOpen] = useState(false);

  // Dynamic strategy count
  const { data: stratData } = useQuery({
    queryKey: ["strategies-sidebar"],
    queryFn: async () => {
      const res = await endpoints.strategies();
      return (res.strategies ?? []) as Array<{ status: string }>;
    },
    refetchInterval: 30000,
    staleTime: 10000,
  });
  const activeStratCount = stratData?.filter((s) => s.status === "active").length ?? 0;
  const positions = useStore((s) => s.positions);
  const autonomyOn = useStore((s) => s.autonomyOn);

  const nav = [
    { href: "/dashboard", label: "Dashboard", icon: LayoutDashboard, badge: null },
    { href: "/strategies", label: "Strategies", icon: Bot, badge: activeStratCount > 0 ? String(activeStratCount) : null },
    { href: "/positions", label: "Positions", icon: Layers, badge: positions.length > 0 ? String(positions.length) : null },
    { href: "/trades", label: "Trade History", icon: History, badge: null },
    { href: "/performance", label: "Performance", icon: TrendingUp, badge: null },
    { href: "/backtesting", label: "Backtesting", icon: FlaskConical, badge: null },
    { href: "/risk", label: "Risk Engine", icon: ShieldAlert, badge: null },
    { href: "/news", label: "Market News", icon: Newspaper, badge: null },
    { href: "/broker", label: "Broker", icon: Plug, badge: null },
    { href: "/audit", label: "Audit Logs", icon: FileText, badge: null },
    { href: "/settings", label: "Settings", icon: Settings, badge: null },
  ];

  return (
    <>
      {/* Mobile toggle */}
      <button
        type="button"
        className="fixed left-4 top-4 z-50 rounded-xl border border-border/50 bg-card/80 p-2.5 backdrop-blur-xl lg:hidden transition-all hover:border-primary/40 hover:shadow-lg"
        onClick={() => setMobileOpen((o) => !o)}
        aria-label="Toggle menu"
      >
        <AnimatePresence mode="wait">
          {mobileOpen ? (
            <motion.div key="close" initial={{ rotate: -90, opacity: 0 }} animate={{ rotate: 0, opacity: 1 }} exit={{ rotate: 90, opacity: 0 }}>
              <X className="h-5 w-5" />
            </motion.div>
          ) : (
            <motion.div key="menu" initial={{ rotate: 90, opacity: 0 }} animate={{ rotate: 0, opacity: 1 }} exit={{ rotate: -90, opacity: 0 }}>
              <Menu className="h-5 w-5" />
            </motion.div>
          )}
        </AnimatePresence>
      </button>

      {/* Sidebar */}
      <aside
        className={cn(
          "fixed left-0 top-0 z-40 flex h-full w-64 flex-col border-r border-border/40 transition-transform duration-300 lg:translate-x-0",
          mobileOpen ? "translate-x-0" : "-translate-x-full"
        )}
        style={{
          background: "linear-gradient(180deg, hsl(222 47% 7%) 0%, hsl(222 47% 4%) 100%)",
        }}
      >
        {/* Brand header */}
        <div className="relative flex h-16 items-center gap-3 border-b border-border/30 px-5">
          <Link
            href="/dashboard"
            className="flex items-center gap-3"
            onClick={() => setMobileOpen(false)}
          >
            <div className="relative flex h-9 w-9 items-center justify-center rounded-xl bg-gradient-to-br from-primary via-[hsl(258,90%,66%)] to-[hsl(199,89%,48%)] shadow-lg">
              <Zap className="h-5 w-5 text-white" />
              <div className="absolute inset-0 rounded-xl bg-gradient-to-br from-primary/20 to-transparent animate-pulse" />
            </div>
            <div className="flex flex-col">
              <span className="text-sm font-bold tracking-tight text-gradient">
                AlphaForge
              </span>
              <span className="text-[10px] font-medium uppercase tracking-widest text-muted-foreground">
                AI Trading
              </span>
            </div>
          </Link>
        </div>

        {/* Navigation */}
        <nav className="flex-1 space-y-1 overflow-y-auto px-3 py-4">
          <div className="mb-3 px-3">
            <span className="text-[10px] font-semibold uppercase tracking-widest text-muted-foreground/60">
              Main Menu
            </span>
          </div>
          {nav.slice(0, 8).map((item, index) => {
            const isActive =
              pathname === item.href || pathname.startsWith(item.href + "/");
            return (
              <Link
                key={item.href}
                href={item.href}
                onClick={() => setMobileOpen(false)}
              >
                <motion.div
                  className={cn(
                    isActive ? "nav-item-active" : "nav-item"
                  )}
                  whileHover={{ x: 3 }}
                  whileTap={{ scale: 0.98 }}
                  initial={{ opacity: 0, x: -10 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: index * 0.05 }}
                >
                  <item.icon className="h-[18px] w-[18px] shrink-0" />
                  <span className="flex-1">{item.label}</span>
                  {item.badge && (
                    <span className="flex h-5 min-w-5 items-center justify-center rounded-full bg-primary/20 px-1.5 text-[10px] font-bold text-primary">
                      {item.badge}
                    </span>
                  )}
                  {isActive && (
                    <motion.div
                      layoutId="activeTab"
                      className="absolute left-0 top-1/2 h-6 w-[3px] -translate-y-1/2 rounded-r-full bg-primary"
                      transition={{ type: "spring", bounce: 0.2, duration: 0.6 }}
                    />
                  )}
                </motion.div>
              </Link>
            );
          })}

          <div className="mb-3 mt-6 px-3">
            <span className="text-[10px] font-semibold uppercase tracking-widest text-muted-foreground/60">
              System
            </span>
          </div>
          {nav.slice(8).map((item, index) => {
            const isActive =
              pathname === item.href || pathname.startsWith(item.href + "/");
            return (
              <Link
                key={item.href}
                href={item.href}
                onClick={() => setMobileOpen(false)}
              >
                <motion.div
                  className={cn(
                    isActive ? "nav-item-active" : "nav-item"
                  )}
                  whileHover={{ x: 3 }}
                  whileTap={{ scale: 0.98 }}
                >
                  <item.icon className="h-[18px] w-[18px] shrink-0" />
                  <span>{item.label}</span>
                </motion.div>
              </Link>
            );
          })}
        </nav>

        {/* Bottom status card */}
        <div className="border-t border-border/30 p-3">
          <div className="glass-card rounded-xl p-3">
            <div className="flex items-center gap-2.5">
              <div className="relative">
                <Activity className={cn("h-4 w-4", autonomyOn ? "text-profit" : "text-muted-foreground")} />
                {autonomyOn && <span className="absolute -right-0.5 -top-0.5 h-2 w-2 rounded-full bg-profit animate-pulse" />}
              </div>
              <div className="flex flex-col">
                <span className="text-xs font-medium text-foreground">
                  {autonomyOn ? "System Active" : "System Idle"}
                </span>
                <span className="text-[10px] text-muted-foreground">
                  {activeStratCount} strategies {autonomyOn ? "• Live" : "• Paused"}
                </span>
              </div>
            </div>
          </div>
        </div>
      </aside>

      {/* Mobile overlay */}
      <AnimatePresence>
        {mobileOpen && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-30 bg-black/60 backdrop-blur-sm lg:hidden"
            onClick={() => setMobileOpen(false)}
            aria-hidden
          />
        )}
      </AnimatePresence>
    </>
  );
}
