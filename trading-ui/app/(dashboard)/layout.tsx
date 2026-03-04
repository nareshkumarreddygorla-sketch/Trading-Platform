"use client";

import { Sidebar } from "@/components/layout/Sidebar";
import { Topbar } from "@/components/layout/Topbar";
import { RequireAuth } from "@/components/auth/RequireAuth";
import { useWebSocket } from "@/lib/hooks/useWebSocket";
import { Notifications } from "@/components/ui/Notifications";

function DashboardShell({ children }: { children: React.ReactNode }) {
  // Initialize WebSocket connection for live updates
  useWebSocket();

  return (
    <div className="min-h-screen bg-background grid-pattern">
      <Sidebar />
      <div className="lg:pl-64">
        <Topbar />
        <main className="p-5 lg:p-6">{children}</main>
      </div>
      <Notifications />
    </div>
  );
}

export default function DashboardLayout({ children }: { children: React.ReactNode }) {
  return (
    <RequireAuth>
      <DashboardShell>{children}</DashboardShell>
    </RequireAuth>
  );
}
