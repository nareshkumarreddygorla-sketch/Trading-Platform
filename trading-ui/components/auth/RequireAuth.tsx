"use client";

import { useState, useEffect, useRef, useCallback } from "react";
import { useRouter } from "next/navigation";
import { clearAuthTokens, setAuthTokens } from "@/lib/api/client";

/** Expiry buffer: attempt refresh 2 minutes before token expires */
const EXPIRY_BUFFER_MS = 2 * 60 * 1000;
/** Interval for checking token expiry (every 30 seconds) */
const CHECK_INTERVAL_MS = 30_000;

function base64UrlDecode(str: string): string {
  let base64 = str.replace(/-/g, '+').replace(/_/g, '/');
  const pad = base64.length % 4;
  if (pad) base64 += '='.repeat(4 - pad);
  return atob(base64);
}

function getTokenExpiry(token: string): number | null {
  try {
    const payload = JSON.parse(base64UrlDecode(token.split('.')[1]));
    if (typeof payload.exp === "number") {
      return payload.exp * 1000;
    }
    return null;
  } catch {
    return null;
  }
}

function isTokenExpired(token: string): boolean {
  const expiry = getTokenExpiry(token);
  if (expiry === null) return true;
  return expiry < Date.now();
}

function isTokenExpiringSoon(token: string): boolean {
  const expiry = getTokenExpiry(token);
  if (expiry === null) return true;
  return expiry - Date.now() < EXPIRY_BUFFER_MS;
}

export function RequireAuth({ children }: { children: React.ReactNode }) {
  const router = useRouter();
  const [authorized, setAuthorized] = useState<boolean | null>(null);
  const refreshingRef = useRef(false);

  const attemptRefresh = useCallback(async (): Promise<boolean> => {
    if (refreshingRef.current) return false;
    refreshingRef.current = true;
    try {
      const refreshToken = localStorage.getItem("refresh_token");
      if (!refreshToken) return false;
      const base = typeof window !== "undefined" ? "" : (process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000");
      const res = await fetch(`${base}/api/backend/api/v1/auth/refresh`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ refresh_token: refreshToken }),
      });
      if (!res.ok) return false;
      const data = await res.json() as { access_token: string; refresh_token?: string };
      setAuthTokens(data.access_token, data.refresh_token);
      return true;
    } catch {
      return false;
    } finally {
      refreshingRef.current = false;
    }
  }, []);

  useEffect(() => {
    const check = async () => {
      const token = localStorage.getItem("token");

      // No token at all -- redirect to login
      if (!token) {
        clearAuthTokens();
        router.replace("/login");
        setAuthorized(false);
        return;
      }

      // Token is already expired
      if (isTokenExpired(token)) {
        // Try to refresh before giving up
        const refreshed = await attemptRefresh();
        if (refreshed) {
          setAuthorized(true);
        } else {
          clearAuthTokens();
          router.replace("/login");
          setAuthorized(false);
        }
        return;
      }

      // Token is still valid but expiring soon: proactively refresh
      if (isTokenExpiringSoon(token)) {
        // Fire-and-forget the refresh; the token is still valid for now
        attemptRefresh();
      }

      setAuthorized(true);
    };

    check();
    const interval = setInterval(check, CHECK_INTERVAL_MS);
    return () => clearInterval(interval);
  }, [router, attemptRefresh]);

  // Don't render children until we've confirmed auth
  if (authorized !== true) return null;

  return <>{children}</>;
}
