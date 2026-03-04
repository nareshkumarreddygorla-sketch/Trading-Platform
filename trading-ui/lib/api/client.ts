/** In the browser we use Next.js rewrite (same-origin) to avoid CORS and "Failed to fetch". */
function getApiBase(): string {
  if (typeof window !== "undefined") {
    return ""; // same origin; Next rewrites /api/backend/* -> backend:8000
  }
  return process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";
}

function getToken(): string | null {
  if (typeof window === "undefined") return null;
  return localStorage.getItem("token");
}

/** Set auth tokens and cookie indicator for middleware. */
export function setAuthTokens(accessToken: string, refreshToken?: string) {
  if (typeof window === "undefined") return;
  localStorage.setItem("token", accessToken);
  if (refreshToken) localStorage.setItem("refresh_token", refreshToken);
  // Set cookie so Next.js middleware can detect auth
  document.cookie = "has_auth=1; path=/; max-age=604800; SameSite=Strict";
}

/** Clear auth tokens and cookie. */
export function clearAuthTokens() {
  if (typeof window === "undefined") return;
  localStorage.removeItem("token");
  localStorage.removeItem("refresh_token");
  document.cookie = "has_auth=; path=/; max-age=0";
}

/** Try to refresh the access token using stored refresh token. */
async function tryRefreshToken(): Promise<boolean> {
  const refreshToken = typeof window !== "undefined" ? localStorage.getItem("refresh_token") : null;
  if (!refreshToken) return false;
  try {
    const base = getApiBase();
    const res = await fetch(`${base}/api/backend/api/v1/auth/refresh`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ refresh_token: refreshToken }),
    });
    if (!res.ok) return false;
    const data = await res.json() as { access_token: string };
    setAuthTokens(data.access_token);
    return true;
  } catch {
    return false;
  }
}

let isRefreshing = false;

async function request<T>(
  path: string,
  options: RequestInit = {},
  _isRetry = false,
): Promise<T> {
  const base = getApiBase();
  const pathWithProxy =
    path.startsWith("http") ? path : path.startsWith("/") ? `${base}/api/backend${path}` : `${base}/api/backend/${path}`;
  const url = path.startsWith("http") ? path : pathWithProxy;
  const token = getToken();
  const headers: HeadersInit = {
    "Content-Type": "application/json",
    ...(options.headers as Record<string, string>),
  };
  if (token) headers["Authorization"] = `Bearer ${token}`;

  let res: Response;
  try {
    res = await fetch(url, { ...options, headers });
  } catch {
    throw new Error(
      "Cannot reach server. Is the backend running? Start it with: npm run dev:all"
    );
  }
  // On 401: try refresh token once before redirecting to login
  if (res.status === 401 && !_isRetry) {
    if (!isRefreshing) {
      isRefreshing = true;
      const refreshed = await tryRefreshToken();
      isRefreshing = false;
      if (refreshed) {
        return request<T>(path, options, true);
      }
    }
    if (typeof window !== "undefined") {
      clearAuthTokens();
      if (!window.location.pathname.includes("/login")) {
        window.location.href = "/login";
      }
    }
    throw new Error("Unauthorized");
  }
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    const d = (err as { detail?: string | Array<{ msg?: string; loc?: unknown[] }> }).detail;
    const message =
      typeof d === "string"
        ? d
        : Array.isArray(d) && d.length > 0
          ? d.map((e) => e.msg ?? JSON.stringify(e)).join(", ")
          : "Request failed";
    throw new Error(message);
  }
  return res.json() as Promise<T>;
}

export const api = {
  get: <T>(path: string) => request<T>(path),
  post: <T>(path: string, body?: unknown) =>
    request<T>(path, { method: "POST", body: body ? JSON.stringify(body) : undefined }),
  put: <T>(path: string, body?: unknown) =>
    request<T>(path, { method: "PUT", body: body ? JSON.stringify(body) : undefined }),
  delete: <T>(path: string) => request<T>(path, { method: "DELETE" }),
};

export const endpoints = {
  // Health
  health: () => api.get<{ status: string }>("/health"),
  ready: () => api.get<{ status: string; checks: Record<string, string> }>("/ready"),
  tradingReady: () => api.get<{ ready: boolean; reason?: string; message?: string }>("/api/v1/trading/ready"),
  tradingMode: () => api.get<{
    mode: "paper" | "live"; autonomous: boolean; safe_mode: boolean;
    circuit_open: boolean; kill_switch_armed: boolean;
  }>("/api/v1/trading/mode"),
  toggleAutonomous: (enabled: boolean) =>
    api.put<{ status: string; autonomous: boolean }>("/api/v1/trading/autonomous", { enabled }),

  // Auth
  login: (body: { username: string; password: string }) =>
    api.post<{ access_token: string; refresh_token?: string; token_type: string; expires_in?: number }>("/api/v1/auth/login", body),
  refreshToken: (refreshToken: string) =>
    api.post<{ access_token: string; token_type: string; expires_in?: number }>("/api/v1/auth/refresh", { refresh_token: refreshToken }),

  // Risk
  risk: () => api.get<{ equity: number; daily_pnl: number; positions: unknown[] }>("/api/v1/risk/snapshot"),
  riskState: () => api.get<{
    circuit_open: boolean; daily_pnl: number; open_positions: number;
    var_95: number; max_drawdown_pct: number;
  }>("/api/v1/risk/state"),
  riskLimits: () => api.get<{
    max_position_pct: number; max_daily_loss_pct: number; max_open_positions: number;
  }>("/api/v1/risk/limits"),
  updateRiskLimits: (body: { max_position_pct?: number; max_daily_loss_pct?: number; max_open_positions?: number }) =>
    api.put<{ status: string; limits: Record<string, number> }>("/api/v1/risk/limits", body),

  // Strategies
  strategies: () => api.get<{ strategies: unknown[] }>("/api/v1/strategies"),
  toggleStrategy: (id: string, enabled: boolean) =>
    api.put<{ strategy_id: string; enabled: boolean }>(`/api/v1/strategies/${id}/toggle`, { enabled }),
  updateCapital: (id: string, capital: number) =>
    api.put<{ strategy_id: string; capital_allocated: number }>(`/api/v1/strategies/${id}/capital`, { capital }),
  strategiesPerformance: () => api.get<{
    strategies: unknown[]; summary: { total_pnl: number; total_trades: number; avg_win_rate: number; active_count: number; total_count: number };
  }>("/api/v1/strategies/performance"),

  // Positions
  positions: () => api.get<{ positions: unknown[] }>("/api/v1/risk/positions"),

  // Orders
  cancelOrder: (orderId: string) =>
    api.post<{ order_id: string; status: string }>(`/api/v1/orders/${orderId}/cancel`),

  // Audit
  auditLogs: (limit?: number, eventType?: string) => {
    const params = new URLSearchParams();
    if (limit) params.set("limit", String(limit));
    if (eventType) params.set("event_type", eventType);
    const qs = params.toString();
    return api.get<{ events: Array<{ id: string; ts: string; event_type: string; actor: string; payload?: Record<string, unknown> }> }>(
      `/api/v1/audit/logs${qs ? `?${qs}` : ""}`
    );
  },

  // Performance
  performanceEquityCurve: (weeks = 26) =>
    api.get<{ equity_curve: Array<{ date: string; label: string; equity: number }> }>(
      `/api/v1/performance/equity-curve?weeks=${weeks}`
    ),
  performanceDrawdown: (weeks = 26) =>
    api.get<{ drawdown: Array<{ date: string; label: string; drawdown: number }> }>(
      `/api/v1/performance/drawdown?weeks=${weeks}`
    ),
  performanceMonthlyReturns: (weeks = 26) =>
    api.get<{ monthly_returns: Array<{ month: string; return_pct: number }> }>(
      `/api/v1/performance/monthly-returns?weeks=${weeks}`
    ),
  performanceSummary: (weeks = 26) =>
    api.get<{
      sharpe_ratio: number; max_drawdown_pct: number; total_pnl: number;
      total_return_pct: number; win_rate: number; total_trades: number;
      avg_trade_pnl: number; profit_factor: number;
      initial_equity: number; final_equity: number; weeks: number;
    }>(`/api/v1/performance/summary?weeks=${weeks}`),

  // Backtesting
  backtestRun: (body: { strategy_id: string; symbol: string; exchange?: string; start: string; end: string; interval?: string; config?: Record<string, unknown> }) =>
    api.post<{ job_id: string; status: string }>("/api/v1/backtest/run", body),
  backtestJob: (jobId: string) =>
    api.get<{ job_id: string; status: string; metrics?: Record<string, number>; error?: string }>(`/api/v1/backtest/jobs/${jobId}`),
  backtestEquity: (jobId: string) =>
    api.get<{ job_id: string; equity_curve: Array<{ date: string; equity: number }> }>(`/api/v1/backtest/jobs/${jobId}/equity`),
  backtestTrades: (jobId: string, limit = 100) =>
    api.get<{ job_id: string; trades: Array<Record<string, unknown>> }>(`/api/v1/backtest/jobs/${jobId}/trades?limit=${limit}`),
  backtestJobs: () =>
    api.get<{ jobs: Array<{ job_id: string; status: string; strategy_id?: string; symbol?: string }> }>("/api/v1/backtest/jobs"),

  // Market Data
  marketQuote: (symbol: string) =>
    api.get<{ symbol: string; exchange: string; last: number; bid: number; ask: number; volume: number; ts: string }>(
      `/api/v1/market/quote/${symbol}`
    ),
  marketBars: (symbol: string, interval = "1d", limit = 100) =>
    api.get<{ symbol: string; exchange: string; interval: string; bars: Array<Record<string, unknown>> }>(
      `/api/v1/market/bars/${symbol}?interval=${interval}&limit=${limit}`
    ),
  marketRegime: () =>
    api.get<{ regime: string; confidence: number; vol_percentile: number; trend_strength: number }>(
      "/api/v1/market/regime"
    ),
  marketNews: (limit = 20) =>
    api.get<{ news: Array<{ id: string; headline: string; symbol: string; sentiment: string; score: number; source: string; timestamp: string; category: string }>; source: string }>(
      `/api/v1/market/news?limit=${limit}`
    ),

  // Orders (full listing)
  orders: (limit = 50, status?: string) => {
    const params = new URLSearchParams();
    params.set("limit", String(limit));
    if (status) params.set("status", status);
    return api.get<{ orders: Array<{ order_id: string; broker_order_id?: string; strategy_id: string; symbol: string; exchange: string; side: string; quantity: number; order_type: string; limit_price?: number; status: string; filled_qty: number; avg_price?: number; ts?: string }>; total: number }>(
      `/api/v1/orders?${params.toString()}`
    );
  },
  placeOrder: (body: { symbol: string; exchange?: string; side: string; quantity: number; order_type?: string; limit_price?: number; strategy_id?: string }) =>
    api.post<{ order_id: string; broker_order_id?: string; status: string; latency_ms?: number }>("/api/v1/orders", body),

  // Broker
  brokerStatus: () =>
    api.get<{
      connected: boolean; mode: "paper" | "live"; healthy: boolean; safe_mode: boolean;
      has_credentials: boolean; client_id: string | null; last_connected: string | null;
      autonomous_running: boolean; tick_count: number; open_trades: number;
    }>("/api/v1/broker/status"),
  brokerConfigure: (body: { api_key: string; client_id: string; password: string; totp_secret: string }) =>
    api.post<{ status: string; message: string; mode?: string; connected: boolean }>("/api/v1/broker/configure", body),
  brokerDisconnect: () =>
    api.post<{ status: string; message: string; mode: string; connected: boolean }>("/api/v1/broker/disconnect"),
  brokerValidate: (body: { api_key: string; client_id: string; password: string; totp_secret: string }) =>
    api.post<{ valid: boolean; message: string }>("/api/v1/broker/validate", body),

  // Training
  trainingStatus: () =>
    api.get<{
      is_training: boolean;
      started_at: string | null;
      mode: string | null;
      recent_logs: string[];
      last_result: { returncode: number; success: boolean } | null;
      last_training: { last_trained: string; results: Record<string, string> } | null;
      model_files: Record<string, { exists: boolean; size_mb?: number; modified?: string }>;
    }>("/api/v1/training/status"),
  trainingStart: (body: { mode?: string; models?: string; skip_data?: boolean }) =>
    api.post<{ status: string; pid: number; mode: string }>("/api/v1/training/start", body),
  trainingStop: () =>
    api.post<{ status: string }>("/api/v1/training/stop"),
};
