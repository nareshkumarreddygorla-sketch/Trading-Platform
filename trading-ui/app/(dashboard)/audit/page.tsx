"use client";

import { useQuery } from "@tanstack/react-query";
import { motion } from "framer-motion";
import { cn } from "@/lib/utils";
import { endpoints } from "@/lib/api/client";
import { useState } from "react";
import {
  FileText, Clock, AlertTriangle,
  Info, CheckCircle, Shield, Zap,
  Search,
} from "lucide-react";

const EVENT_COLORS: Record<string, { icon: typeof Info; color: string; bg: string }> = {
  trade_executed: { icon: Zap, color: "text-profit", bg: "bg-profit/10" },
  strategy_disabled: { icon: AlertTriangle, color: "text-warning", bg: "bg-warning/10" },
  risk_limit_hit: { icon: Shield, color: "text-loss", bg: "bg-loss/10" },
  model_retrained: { icon: CheckCircle, color: "text-primary", bg: "bg-primary/10" },
  default: { icon: Info, color: "text-muted-foreground", bg: "bg-muted/30" },
};

export default function AuditPage() {
  const [filter, setFilter] = useState("");
  const [search, setSearch] = useState("");

  const { data, isLoading } = useQuery({
    queryKey: ["audit-logs", filter],
    queryFn: () => endpoints.auditLogs(200, filter || undefined),
  });

  const events = data?.events ?? [];
  const filtered = search
    ? events.filter((e) =>
      e.event_type.toLowerCase().includes(search.toLowerCase()) ||
      e.actor.toLowerCase().includes(search.toLowerCase())
    )
    : events;

  return (
    <div className="space-y-6 pb-8">
      <motion.div initial={{ opacity: 0, y: -10 }} animate={{ opacity: 1, y: 0 }}>
        <h1 className="text-2xl font-bold tracking-tight">Audit Logs</h1>
        <p className="text-sm text-muted-foreground mt-0.5">System event timeline</p>
      </motion.div>

      {/* Filters */}
      <motion.div
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.05 }}
        className="flex flex-wrap items-center gap-3"
      >
        <div className="relative flex-1 min-w-[200px] max-w-sm">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-muted-foreground" aria-hidden="true" />
          <input
            type="text"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Search events..."
            aria-label="Search audit events"
            className="h-9 w-full rounded-xl border border-border/50 bg-muted/30 pl-9 pr-4 text-sm transition-all focus:border-primary/50 focus:outline-none focus:ring-1 focus:ring-primary/20"
          />
        </div>
        {["", "trade_executed", "strategy_disabled", "risk_limit_hit", "model_retrained"].map((f) => (
          <button
            key={f}
            onClick={() => setFilter(f)}
            aria-label={`Filter by ${f === "" ? "all events" : f.replace(/_/g, " ")}`}
            aria-pressed={filter === f}
            className={cn(
              "rounded-lg px-3 py-1.5 text-[11px] font-medium transition-all",
              filter === f
                ? "bg-primary/15 text-primary border border-primary/30"
                : "text-muted-foreground hover:text-foreground hover:bg-muted border border-transparent"
            )}
          >
            {f === "" ? "All" : f.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase())}
          </button>
        ))}
      </motion.div>

      {/* Event timeline */}
      <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.1 }}>
        <div className="glass-card overflow-hidden">
          <div className="flex items-center justify-between border-b border-border/30 p-5">
            <div className="flex items-center gap-2">
              <FileText className="h-4 w-4 text-primary" />
              <span className="text-sm font-semibold">{filtered.length} Events</span>
            </div>
          </div>

          {isLoading ? (
            <div className="p-5 space-y-3">
              {Array.from({ length: 5 }).map((_, i) => (
                <div key={i} className="shimmer h-14 w-full rounded-lg" />
              ))}
            </div>
          ) : filtered.length === 0 ? (
            <div className="flex flex-col items-center justify-center py-16 text-muted-foreground">
              <FileText className="h-10 w-10 mb-3 opacity-30" />
              <p className="text-sm font-medium">No audit events</p>
              <p className="text-xs mt-1">Events are recorded when the system takes actions</p>
            </div>
          ) : (
            <div className="divide-y divide-border/10">
              {filtered.map((event, i) => {
                const cfg = EVENT_COLORS[event.event_type] ?? EVENT_COLORS.default;
                return (
                  <motion.div
                    key={event.id}
                    initial={{ opacity: 0, x: -10 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: 0.1 + i * 0.02 }}
                    className="flex items-start gap-3 px-5 py-3 transition-colors hover:bg-muted/10"
                  >
                    <div className={cn("flex h-8 w-8 items-center justify-center rounded-lg shrink-0 mt-0.5", cfg.bg)}>
                      <cfg.icon className={cn("h-4 w-4", cfg.color)} />
                    </div>
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2">
                        <span className="text-sm font-medium">
                          {event.event_type.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase())}
                        </span>
                        <span className={cn("status-badge text-[9px]", cfg.bg, cfg.color, `border ${cfg.color.replace('text-', 'border-')}/30`)}>
                          {event.actor}
                        </span>
                      </div>
                      {event.payload && Object.keys(event.payload).length > 0 && (
                        <p className="text-xs text-muted-foreground mt-0.5 truncate">
                          {JSON.stringify(event.payload).slice(0, 80)}
                        </p>
                      )}
                    </div>
                    <div className="flex items-center gap-1 shrink-0">
                      <Clock className="h-3 w-3 text-muted-foreground" />
                      <span className="text-[10px] font-mono text-muted-foreground">
                        {new Date(event.ts).toLocaleTimeString()}
                      </span>
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
