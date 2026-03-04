"use client";

import { useStore } from "@/store/useStore";
import { AlertTriangle, ShieldAlert, Info } from "lucide-react";

const SEVERITY_CONFIG: Record<string, { icon: React.ReactNode; color: string }> = {
  critical: { icon: <ShieldAlert className="w-3.5 h-3.5" />, color: "text-red-400" },
  warning: { icon: <AlertTriangle className="w-3.5 h-3.5" />, color: "text-yellow-400" },
  info: { icon: <Info className="w-3.5 h-3.5" />, color: "text-blue-400" },
};

export default function RiskAlerts() {
  const alerts = useStore((s) => s.riskAlerts);

  if (alerts.length === 0) return null;

  return (
    <div className="rounded-xl border border-zinc-800 bg-zinc-900/50 p-4" role="region" aria-label="Risk alerts" aria-live="assertive">
      <h3 className="text-sm font-medium text-zinc-400 mb-3 flex items-center gap-2">
        <AlertTriangle className="w-4 h-4" aria-hidden="true" /> Risk Alerts
        <span className="ml-auto text-xs px-1.5 py-0.5 rounded bg-red-500/20 text-red-400">
          {alerts.length}
        </span>
      </h3>
      <div className="space-y-1.5 max-h-32 overflow-y-auto">
        {alerts.slice(0, 10).map((alert, i) => {
          const config = SEVERITY_CONFIG[alert.severity] || SEVERITY_CONFIG.info;
          return (
            <div
              key={`${alert.type}-${i}`}
              className="flex items-start gap-2 text-xs py-1"
            >
              <span className={`mt-0.5 ${config.color}`}>{config.icon}</span>
              <div className="flex-1 min-w-0">
                <span className={`font-medium ${config.color}`}>{alert.type}</span>
                <span className="text-zinc-500 ml-2">
                  {new Date(alert.timestamp).toLocaleTimeString()}
                </span>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
