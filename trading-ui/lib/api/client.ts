/* ── Error typing ── */

export interface ApiErrorResponse {
  detail: string | Array<{ msg?: string; loc?: unknown[] }>;
  status?: number;
}

export class ApiError extends Error {
  status: number;
  detail: string;
  constructor(message: string, status: number, detail?: string) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.detail = detail ?? message;
  }
}

export class NetworkError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "NetworkError";
  }
}

export class TimeoutError extends Error {
  constructor(url: string) {
    super(`Request timed out: ${url}`);
    this.name = "TimeoutError";
  }
}

/* ── Default request timeout (ms) ── */
const DEFAULT_TIMEOUT_MS = 15_000;

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
    const data = await res.json() as { access_token: string; refresh_token?: string };
    setAuthTokens(data.access_token, data.refresh_token);
    return true;
  } catch {
    return false;
  }
}

let refreshPromise: Promise<boolean> | null = null;

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

  // AbortController for request timeout
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), DEFAULT_TIMEOUT_MS);

  let res: Response;
  try {
    res = await fetch(url, { ...options, headers, signal: controller.signal });
  } catch (err) {
    clearTimeout(timeoutId);
    if (err instanceof DOMException && err.name === "AbortError") {
      throw new TimeoutError(url);
    }
    // Provide a user-friendly network error message
    const isOffline = typeof navigator !== "undefined" && !navigator.onLine;
    if (isOffline) {
      throw new NetworkError(
        "You appear to be offline. Please check your internet connection."
      );
    }
    throw new NetworkError(
      "Unable to connect to the trading server. Please verify the backend is running and try again."
    );
  } finally {
    clearTimeout(timeoutId);
  }

  // On 401: try refresh token once before redirecting to login.
  // Use a shared promise so concurrent 401s wait for the same refresh attempt.
  if (res.status === 401 && !_isRetry) {
    if (!refreshPromise) {
      refreshPromise = tryRefreshToken().finally(() => { refreshPromise = null; });
    }
    const refreshed = await refreshPromise;
    if (refreshed) {
      return request<T>(path, options, true);
    }
    if (typeof window !== "undefined") {
      clearAuthTokens();
      if (!window.location.pathname.includes("/login")) {
        window.location.href = "/login";
      }
    }
    throw new ApiError("Session expired. Please log in again.", 401);
  }

  // Handle 403 specifically
  if (res.status === 403) {
    throw new ApiError("You do not have permission to perform this action.", 403);
  }

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }) as ApiErrorResponse);
    const d = (err as ApiErrorResponse).detail;
    const message =
      typeof d === "string"
        ? d
        : Array.isArray(d) && d.length > 0
          ? d.map((e) => e.msg ?? JSON.stringify(e)).join(", ")
          : `Request failed (${res.status})`;
    throw new ApiError(message, res.status, message);
  }

  // Handle empty responses (204 No Content, etc.)
  const contentType = res.headers.get("content-type");
  if (res.status === 204 || !contentType?.includes("application/json")) {
    return {} as T;
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
    api.post<{ access_token: string; refresh_token?: string; token_type: string; expires_in?: number }>("/api/v1/auth/refresh", { refresh_token: refreshToken }),

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
  brokerConfigure: (body: { api_key: string; client_id: string; password: string; totp_secret: string; mode?: "paper" | "live" }) =>
    api.post<{ status: string; message: string; mode?: string; connected: boolean; auto_started?: boolean; confirm_token?: string }>("/api/v1/broker/configure", body),
  brokerConfirmLive: (confirmToken: string) =>
    api.post<{ status: string; message: string; mode?: string; connected: boolean; auto_started?: boolean }>("/api/v1/broker/confirm-live", { confirm_token: confirmToken }),
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

  // Attribution
  attributionByDimension: (dimension: string, days = 30) =>
    api.get<{ dimension: string; rows: Array<{ label: string; pnl: number; trades: number; win_rate: number; sharpe: number; contribution_pct: number }>; total_pnl: number; days: number }>(
      `/api/v1/attribution/by-dimension?dimension=${dimension}&days=${days}`
    ),
  attributionFull: (days = 30) =>
    api.get<{ dimensions: Record<string, Array<{ label: string; pnl: number; trades: number; win_rate: number; sharpe: number; contribution_pct: number }>>; total_pnl: number; total_trades: number; win_rate: number; best_model: string; days: number }>(
      `/api/v1/attribution/full?days=${days}`
    ),
  featureImportance: (topN = 15) =>
    api.get<{ features: Array<{ feature: string; importance: number; correlation: number }> }>(
      `/api/v1/attribution/feature-importance?top_n=${topN}`
    ),
  tradeOutcomes: (limit = 100) =>
    api.get<{ trades: Array<{ id: string; symbol: string; strategy: string; model: string; pnl: number; entry_time: string; exit_time: string; sector: string; regime: string }> }>(
      `/api/v1/attribution/trade-outcomes?limit=${limit}`
    ),

  // Simulation
  simulationRun: () =>
    api.post<{ status: string; job_id?: string }>("/api/v1/simulation/run"),
  simulationStatus: () =>
    api.get<{
      running: boolean; last_run: string | null; total_permutations: number;
      progress?: number; current_step?: string;
    }>("/api/v1/simulation/status"),
  simulationResults: (limit = 50, selectedOnly = false) =>
    api.get<{
      results: Array<{
        rank: number; strategy_id: string; params: Record<string, unknown>;
        interval: string; sharpe: number; sortino: number; max_dd: number;
        win_rate: number; profit_factor: number; trades: number; selected: boolean;
      }>;
      total_tested: number; qualified: number; top_sharpe: number; top_win_rate: number;
      run_ts?: string;
    }>(`/api/v1/simulation/results?limit=${limit}&selected_only=${selectedOnly}`),

  // Data
  dataRefresh: () =>
    api.post<{ status: string; message: string }>("/api/v1/data/refresh"),
  dataQuality: () =>
    api.get<{ quality: Record<string, unknown> }>("/api/v1/data/quality"),
  dataSymbols: (interval = "1d", minBars = 0) =>
    api.get<{ symbols: Array<{ symbol: string; bars: number; first_date: string; last_date: string }> }>(
      `/api/v1/data/symbols?interval=${interval}&min_bars=${minBars}`
    ),
  dataBars: (symbol: string, interval = "1d", limit = 500) =>
    api.get<{ symbol: string; interval: string; bars: Array<Record<string, unknown>> }>(
      `/api/v1/data/bars/${symbol}?interval=${interval}&limit=${limit}`
    ),
  dataInstrumentMap: () =>
    api.get<{ loaded: boolean; total_instruments: number; nse_equity_count: number; last_updated?: string }>(
      "/api/v1/data/instrument-map"
    ),
};
