"use client";

import { useEffect, useState } from "react";
import { usePathname, useRouter } from "next/navigation";
import { Sidebar } from "@/components/layout/Sidebar";
import { Topbar } from "@/components/layout/Topbar";
import { RequireAuth } from "@/components/auth/RequireAuth";
import { useWebSocket } from "@/lib/hooks/useWebSocket";
import { useKeyboardShortcuts } from "@/lib/hooks/useKeyboardShortcuts";
import { Notifications } from "@/components/ui/Notifications";
import { Toaster } from "@/components/Toaster";
import { ErrorBoundary } from "@/components/ErrorBoundary";
import { endpoints } from "@/lib/api/client";

function OnboardingRedirect({ children }: { children: React.ReactNode }) {
  const router = useRouter();
  const pathname = usePathname();
  const [checked, setChecked] = useState(false);

  useEffect(() => {
    // Skip redirect if already on onboarding page
    if (pathname === "/onboarding") {
      setChecked(true);
      return;
    }

    const onboardingComplete = localStorage.getItem("onboarding_complete");
    if (onboardingComplete === "true") {
      setChecked(true);
      return;
    }

    // Check broker status; if not connected, redirect to onboarding
    endpoints
      .brokerStatus()
      .then((status) => {
        if (!status.connected) {
          router.replace("/onboarding");
        } else {
          localStorage.setItem("onboarding_complete", "true");
        }
        setChecked(true);
      })
      .catch(() => {
        router.replace("/onboarding");
        setChecked(true);
      });
  }, [pathname, router]);

  // On onboarding page, always render immediately
  if (pathname === "/onboarding") return <>{children}</>;

  // For other pages, wait until the check completes
  if (!checked) return null;

  return <>{children}</>;
}

function DashboardShell({ children }: { children: React.ReactNode }) {
  // Initialize WebSocket connection for live updates
  useWebSocket();
  // Initialize global keyboard shortcuts
  useKeyboardShortcuts();

  return (
    <div className="min-h-screen bg-background grid-pattern">
      <Sidebar />
      <div className="lg:pl-64">
        <Topbar />
        <main className="p-3 sm:p-5 lg:p-6">
          <ErrorBoundary>{children}</ErrorBoundary>
        </main>
      </div>
      <Notifications />
      <Toaster />
    </div>
  );
}

export default function DashboardLayout({ children }: { children: React.ReactNode }) {
  return (
    <RequireAuth>
      <OnboardingRedirect>
        <DashboardShell>{children}</DashboardShell>
      </OnboardingRedirect>
    </RequireAuth>
  );
}
