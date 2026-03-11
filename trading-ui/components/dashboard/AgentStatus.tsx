"use client";

import { useEffect, useState } from "react";
import { Activity, Brain, Shield, Zap, BarChart3 } from "lucide-react";
import { api } from "@/lib/api/client";

interface AgentInfo {
  name: string;
  description: string;
  running: boolean;
  status: string;
  tick_count: number;
  last_run: string | null;
  pending_opportunities?: number;
  executed_count?: number;
  current_regime?: string;
  recent_alerts?: Array<Record<string, unknown>>;
}

const AGENT_ICONS: Record<string, React.ReactNode> = {
  research_agent: <Brain className="w-4 h-4" />,
  risk_agent: <Shield className="w-4 h-4" />,
  execution_agent: <Zap className="w-4 h-4" />,
  strategy_selector: <BarChart3 className="w-4 h-4" />,
};

const STATUS_COLORS: Record<string, string> = {
  running: "bg-green-500",
  idle: "bg-green-400",
  error: "bg-red-500",
  stopped: "bg-gray-400",
};

export default function AgentStatus() {
  const [agents, setAgents] = useState<Record<string, AgentInfo>>({});
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchAgents = async () => {
      try {
        const data = await api.get<{ agents: Record<string, AgentInfo> }>("/api/v1/agents/status");
        setAgents(data.agents || {});
      } catch {
        // backend unavailable
      } finally {
        setLoading(false);
      }
    };
    fetchAgents();
    const interval = setInterval(fetchAgents, 10000);
    return () => clearInterval(interval);
  }, []);

  if (loading) {
    return (
      <div className="rounded-xl border border-zinc-800 bg-zinc-900/50 p-4">
        <h3 className="text-sm font-medium text-zinc-400 mb-3 flex items-center gap-2">
          <Activity className="w-4 h-4" /> AI Agents
        </h3>
        <div className="animate-pulse space-y-2">
          {[1, 2, 3, 4].map((i) => (
            <div key={i} className="h-12 bg-zinc-800 rounded-lg" />
          ))}
        </div>
      </div>
    );
  }

  const agentList = Object.values(agents);

  return (
    <div className="rounded-xl border border-zinc-800 bg-zinc-900/50 p-4" role="region" aria-label="AI Agents status" aria-live="polite">
      <h3 className="text-sm font-medium text-zinc-400 mb-3 flex items-center gap-2">
        <Activity className="w-4 h-4" aria-hidden="true" /> AI Agents
        <span className="ml-auto text-xs text-zinc-500">
          {agentList.filter((a) => a.running).length}/{agentList.length} active
        </span>
      </h3>
      <div className="space-y-2">
        {agentList.map((agent) => (
          <div
            key={agent.name}
            className="flex items-center gap-3 p-2 rounded-lg bg-zinc-800/50 hover:bg-zinc-800 transition-colors"
          >
            <span className="text-zinc-400">
              {AGENT_ICONS[agent.name] || <Activity className="w-4 h-4" />}
            </span>
            <div className="flex-1 min-w-0">
              <div className="text-sm font-medium text-zinc-200 truncate">
                {agent.name.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase())}
              </div>
              <div className="text-xs text-zinc-500 truncate">{agent.description}</div>
            </div>
            <div className="flex items-center gap-2 shrink-0">
              <span className="text-xs text-zinc-500">{agent.tick_count} ticks</span>
              <span
                className={`w-2 h-2 rounded-full ${STATUS_COLORS[agent.status] || "bg-gray-500"}`}
              />
            </div>
          </div>
        ))}
        {agentList.length === 0 && (
          <div className="text-center text-zinc-500 text-sm py-4">
            No agents running
          </div>
        )}
      </div>
    </div>
  );
}
