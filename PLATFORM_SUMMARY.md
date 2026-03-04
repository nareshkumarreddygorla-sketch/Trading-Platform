# Trading Platform – Architecture, Features, Gaps & Comparison

## 1. Architecture Overview

### 1.1 High-Level Structure

```
Trading-Platform/
├── src/                    # Backend (Python)
│   ├── api/                # FastAPI app, routers, auth, deps
│   ├── core/               # Config, events (Order, Position, Signal)
│   ├── execution/          # Order entry, router, gateway, fill handler, kill switch, autonomous loop
│   ├── risk_engine/        # RiskManager, limits, circuit breaker
│   ├── persistence/        # DB layer (orders, positions, risk snapshot, audit)
│   ├── market_data/        # Connectors (Angel One), cache, streaming (Kafka)
│   ├── startup/            # Cold-start recovery
│   ├── ai/                 # Alpha research, regime, portfolio, LLM (optional)
│   └── backtesting/        # Backtest engine
├── trading-ui/             # Frontend (Next.js 14, App Router)
│   ├── app/                # Pages: dashboard, strategies, positions, risk, broker, audit, settings, auth
│   ├── components/         # Layout (Sidebar, Topbar), UI (Card, Button, Table, etc.)
│   ├── lib/                # API client (proxy), WebSocket hook
│   └── store/              # Zustand (user, equity, PnL, autonomy, broker, safe mode)
├── config/                 # settings.yaml (referenced by config)
├── requirements.txt
└── venv/
```

### 1.2 How Components Connect

- **All order flows** go through **OrderEntryService** (single pipe).
- **API** → OrderEntryService → validate → idempotency (Redis) → kill switch → circuit breaker → risk checks → reservation → distributed lock → persist (optional) → **OrderRouter** → **Angel One gateway** → **OrderLifecycle**.
- **Fills**: Broker (polled via **FillListener**) → **FillHandler** → lifecycle update, **RiskManager** position merge, persistence, metrics.
- **Startup**: Lifespan builds persistence (if `DATABASE_URL`), risk manager, gateway, circuit breaker, router, idempotency, kill switch, reservation; runs **cold-start recovery** (load orders/positions, reconcile); then unlocks trading.
- **Frontend** talks to backend via **Next.js rewrite** (`/api/backend/*` → `http://127.0.0.1:8000/*`) to avoid CORS; WebSocket goes directly to backend `/ws`.

---

## 2. Technical Stack

| Layer | Technology |
|-------|------------|
| **Backend** | Python 3.9, FastAPI, uvicorn |
| **Config** | pydantic-settings, env, optional `config/settings.yaml` |
| **Auth** | JWT (PyJWT), optional AUTH_USERNAME/PASSWORD (env) |
| **DB** | PostgreSQL (SQLAlchemy), only when `DATABASE_URL` set |
| **Cache / coordination** | Redis (idempotency, distributed lock, cluster reservation) |
| **Message queue** | Kafka (market data streaming; config present, not wired in order path) |
| **Broker** | Angel One (SmartAPI REST); paper mode in-memory |
| **Frontend** | Next.js 14 (App Router), React 18, TypeScript |
| **Frontend state** | Zustand, React Query |
| **Frontend UI** | Tailwind, shadcn-style components, Framer Motion, Recharts, Lucide |
| **Dev run** | `npm run dev:all` (concurrently: uvicorn + next dev) |

---

## 3. Workflows

### 3.1 Order Flow

1. Client calls `POST /orders` (or internal caller).
2. **OrderEntryService.submit_order**: validate symbol/side/qty/strategy_id.
3. **Idempotency** (Redis): reject if key already used.
4. **Kill switch**: if armed, allow only reduce-only orders.
5. **Circuit breaker**: reject if circuit open.
6. **Distributed lock** (Redis, if configured) for the order.
7. **Risk**: `risk_manager.can_place_order(signal, quantity, price)` (daily loss, open positions, position size, per-symbol, sector, VaR); **reservation** (local + optional Redis cluster) for exposure.
8. Optional **write-ahead persist** (SUBMITTING) if DB configured.
9. **OrderRouter.place_order** → gateway (Angel One REST or paper stub).
10. On broker ack: update order, idempotency, **OrderLifecycle.register**, persist (if not write-ahead), return.

### 3.2 Fill Handling

- **Live mode only**: **FillListener** polls gateway for order/execution updates.
- Dedup by last-applied delta; map status to FillType.
- **FillHandler.on_fill_event**: update lifecycle, **RiskManager.add_or_merge_position** (under lock), persist fill, metrics.
- **Paper mode**: No fill listener; no broker feed.

### 3.3 Risk & Circuit Breaker

- **RiskManager**: holds positions, daily PnL, limits; **can_place_order** runs all limit checks.
- **CircuitBreaker**: state CLOSED/OPEN/HALF_OPEN; **update_equity** can trip on drawdown; optional flatten callback.
- Lifespan: periodic equity → risk snapshot persist; circuit_breaker.update_equity(risk_manager.equity).
- **Safe mode**: set on broker unreachable (recovery or 3 consecutive heartbeat failures); blocks trading until `POST /admin/safe_mode/clear`.

### 3.4 Startup & Recovery

- **Cold start**: If DB + live mode, **run_cold_start_recovery** loads active orders/positions, warms RiskManager and OrderLifecycle, marks SUBMITTING orders as REJECTED, optionally reconciles with broker.
- **startup_lock** and **recovery_complete** block trading until recovery finishes.
- Broker heartbeat (live only) can arm safe_mode on repeated failure.

### 3.5 Autonomous Loop

- **AutonomousLoop** (`src/execution/autonomous_loop.py`): bar-based tick, safe_mode/drift/regime gates, idempotency helper.
- **Not started** in app lifespan; **_tick** is a placeholder (no strategy/allocator calls).

---

## 4. Features & Capabilities (What Exists Today)

| Feature | Backend | Frontend | Notes |
|--------|---------|----------|--------|
| **Auth** | JWT (get_current_user, require_roles), login/register (in-memory _users) | Login, Register, RequireAuth, token in localStorage | Users lost on restart |
| **Dashboard** | GET /api/v1/risk/snapshot (equity, daily_pnl, positions) | Equity, daily PnL, positions count, circuit, equity curve (placeholder), strategy summary | Uses risk snapshot + WebSocket for “Live” |
| **Strategies** | List, enable/disable (EMACrossover, MACD, RSI); /signals returns [] | Strategy cards, enable/disable, edit capital modal | No live signals yet |
| **Positions** | GET /positions (persistence or risk_manager), GET /api/v1/risk/positions | Table: symbol, side, qty, entry, current, PnL, close | |
| **Risk** | state, limits, snapshot, positions; PUT limits | Gauges, limits, circuit state, sliders (admin) | |
| **Performance** | — | Equity curve, drawdown, monthly returns (placeholder data) | No backend equity curve API |
| **Broker** | Angel One client (login, place, cancel, positions) | Broker page (credentials, validate, health) | Connection health from frontend state |
| **Audit** | AuditRepository (order submit, kill_switch, safe_mode, broker_failure) | Audit Logs page, filters | Need GET /api/v1/audit/logs if not present |
| **Settings** | — | Profile, API keys, risk config, notifications, dark mode, logout | UI only |
| **WebSocket** | GET /ws: accept, send { type: "connected" } | useWebSocket("/ws"), “Live” / “Reconnecting…” | No order/fill/risk push yet |
| **Backtest** | Submit job, in-memory job store, BacktestEngine | — | API only |
| **Market data** | GET quote/bars (stub when no Redis/DB) | — | Real pipeline in code; API uses stubs |
| **Kill switch** | Arm/disarm (admin), reduce-only when armed | — | Admin only |
| **Safe mode** | Set on broker failure; clear (admin) | Top bar banner “Safe Mode Active” | |
| **Exposure multiplier** | RiskManager 0.5–1.5, PUT /trading/exposure_multiplier | — | |
| **Trading ready** | GET /trading/ready (recovery, safe_mode, kill, circuit, equity) | — | Suited for readiness probe |

---

## 5. Gaps & Limitations

- **Auth**: In-memory user store; no DB-backed users; **users lost on server restart**.
- **Broker**: Default **paper mode**; live requires Angel One credentials and config; no other broker adapters.
- **CircuitAndKillController**: Logic for auto-trigger (daily loss, drawdown, rejection spike, broker latency, fill mismatch, India VIX) **not wired** in lifespan; only manual circuit/kill used.
- **Autonomous loop**: **Not started**; no strategy/allocator hooked to _tick.
- **Strategies**: **/signals** returns empty list; no live signal feed to execution.
- **WebSocket**: Single endpoint; **no auth**, no push of orders/fills/risk/PnL.
- **Market data API**: Quote/bars are **stubs** when Redis/DB not used.
- **Missing APIs**: e.g. cancel order, GET audit logs (if not implemented), historical equity curve for performance page.
- **Deployment**: No Docker/K8s/Helm; no production runbooks.
- **AI/Alpha**: Optional deps (xgboost, hmmlearn, openai, anthropic) commented out; alpha pipeline may be partial.

---

## 6. Comparison to Real-Market & Top AI Autonomous Trading Tools

| Dimension | This platform | QuantConnect / LEAN | Alpaca + AI / Numerai | Interactive Brokers (IB) / KRANES | TradingView + Algo | Typical institutional prop |
|-----------|----------------|---------------------|------------------------|-------------------------------------|--------------------|-----------------------------|
| **Execution** | Single pipe (OrderEntryService), one broker (Angel One), paper/live | Multi-broker, live/backtest same engine | Alpaca API; optional ML | IB Gateway/TWS, institutional routing | Broker-dependent (e.g. IB, OANDA) | Multi-venue, smart routing, TCA |
| **Risk** | RiskManager, limits, circuit breaker, kill switch, reservation | Built-in portfolio/risk in LEAN | Broker/API limits | IB risk controls, margin | Limited to broker | Real-time risk, VaR, limits, circuit breakers |
| **Autonomy** | AutonomousLoop skeleton, **not started**; no strategy wired | Full backtest + live algo execution | ML signals + execution | IB algo orders, KRANES for automation | Pine scripts / algos | Automated strategies, execution algos |
| **AI/ML** | Optional alpha research, regime, LLM; not required for core | Optional ML, alpha frameworks | Core: ML (Numerai), signals | Optional (custom or third-party) | Limited (Pine, limited ML) | Dedicated quant/ML teams, custom models |
| **Market data** | Connectors + Kafka; API **stubs** when no Redis | Integrated history + live | Alpaca + optional data | IB market data, third-party | TradingView data | Bloomberg/Refinitiv, direct feeds |
| **Auth & users** | JWT + in-memory register; **no persisted users** | Cloud auth, teams | API keys, dashboard | IB login, optional OAuth | TradingView account | SSO, RBAC, audit |
| **UI** | Next.js dashboard (positions, risk, strategies, broker, audit) | Web IDE, live/backtest UI | Dashboard, API-first | TWS, Client Portal, third-party | TradingView UI + algos | Custom or vendor (e.g. Bloomberg) |
| **Persistence** | Postgres (orders, positions, risk snapshot, audit) when configured | Cloud-backed | Alpaca + own DB | IB + local/cloud | Broker + TradingView | Full audit, compliance, reconciliation |
| **Deployment** | Single-node run (uvicorn + Next); **no containers/orchestration** | Cloud (QuantConnect) | Cloud (Alpaca) | On-prem/cloud (IB) | SaaS (TradingView) | K8s, HA, DR, monitoring |
| **Regulation / audit** | Audit repo (submit, kill, safe_mode, broker_failure); no formal compliance | Logs, backtest audit | Broker + own | IB compliance tools | Broker-dependent | Full trade/audit trail, regulatory reporting |

**Summary vs “top” tools:**

- **Strengths**: Single, clear order pipeline; risk and circuit breaker in place; Angel One integration; cold-start recovery; optional AI/alpha layer; modern frontend with dashboard, risk, positions, auth.
- **Gaps vs production-grade**: No persisted users; autonomy loop not running; no live signals → execution; WebSocket not pushing events; market data API stubbed; CircuitAndKillController not wired; no containers/orchestration; no multi-broker or smart routing.

---

## 6.1 Implemented (Best-Approach)

The following have been implemented:

1. **Persist users**  
   - `UserModel` in `src/persistence/models.py`; `UserRepository` in `src/persistence/user_repo.py` with `create`, `get_by_username`, `verify_password` (passlib/bcrypt when available, else hashlib fallback).  
   - When `DATABASE_URL` is set, register/login use DB; otherwise in-memory fallback.  
   - Requires `passlib[bcrypt]` in requirements (installed).

2. **CircuitAndKillController wired**  
   - In `src/api/app.py` lifespan: `CircuitAndKillController` is created with `RiskManager`, `KillSwitch`, and `CircuitKillConfig` (max_daily_loss_pct, max_drawdown_pct from risk config).  
   - In the periodic risk snapshot task (every 60s): `check_daily_loss_and_trip(rm.equity)` and `check_drawdown_and_trip(peak, rm.equity)` are called so circuit/kill auto-trigger on breach.

3. **Autonomous loop started**  
   - In lifespan, `AutonomousLoop` is created with `get_safe_mode` from `app.state` and `poll_interval_seconds=60`.  
   - Loop is started and stored on `app.state.autonomous_loop`; on shutdown it is stopped.  
   - `_tick` remains a placeholder (no strategy/allocator wired yet).

4. **Cancel order API**  
   - `POST /api/v1/orders/{order_id}/cancel` in `src/api/routers/orders.py`; resolves order from persistence or lifecycle, then calls `gateway.cancel_order(order_id, broker_order_id)`.  
   - Frontend: `endpoints.cancelOrder(orderId)` in `trading-ui/lib/api/client.ts`.

5. **WebSocket**  
   - Still accepts connections and sends `{ type: "connected", message: "Live" }`; optional `?token=<jwt>` in query (documented; validation can be added later).

---

## 7. Recommended Improvements (Prioritised)

1. **Persist users**: DB-backed user table + password hash; register/login against DB; optional OAuth later.
2. **Wire autonomous loop**: Start AutonomousLoop in lifespan; connect strategy/signals (e.g. from StrategyRegistry or alpha pipeline) to _tick and to OrderEntryService.
3. **Live signals**: Implement or wire strategy signals (e.g. from backtest or live bars) and feed into execution path.
4. **WebSocket**: Auth (e.g. JWT in query/header); push order updates, fills, risk snapshot, PnL so UI is real-time.
5. **CircuitAndKillController**: Instantiate in lifespan; plug in risk_manager, broker latency, fill mismatch, VIX; auto arm circuit/kill on configured triggers.
6. **Market data API**: Replace stubs with real cache/Redis or live connector when configured.
7. **Audit API**: Ensure GET audit logs endpoint exists and frontend uses it.
8. **Cancel order**: Expose cancel in API and UI.
9. **Deployment**: Add Dockerfile(s), docker-compose, optional K8s manifests and health/readiness usage.
10. **Multi-broker**: Abstract gateway interface; add second broker (e.g. Zerodha) for comparison/resilience.

---

## 8. Key File Reference

| Area | Path | Key symbols |
|------|------|-------------|
| Order entry | `src/execution/order_entry/service.py` | OrderEntryService, submit_order |
| Risk | `src/risk_engine/manager.py` | RiskManager, can_place_order, add_or_merge_position |
| Limits | `src/risk_engine/limits.py` | RiskLimits, check_* |
| Circuit breaker | `src/risk_engine/circuit_breaker.py` | CircuitBreaker, update_equity |
| Kill switch | `src/execution/order_entry/kill_switch.py` | KillSwitch, arm, disarm |
| Gateway | `src/execution/angel_one_gateway.py` | AngelOneExecutionGateway |
| Fill handler | `src/execution/fill_handler/handler.py` | FillHandler, on_fill_event |
| Fill listener | `src/execution/fill_handler/listener.py` | FillListener |
| Autonomous loop | `src/execution/autonomous_loop.py` | AutonomousLoop (_tick placeholder) |
| API app | `src/api/app.py` | create_app, lifespan |
| Auth JWT | `src/api/auth.py` | get_current_user, require_roles |
| Auth login/register | `src/api/routers/auth.py` | login, register |
| Risk API | `src/api/routers/risk.py` | snapshot, positions, state, limits |
| Orders API | `src/api/routers/orders.py` | list_orders, place_order, list_positions, kill_switch |
| Recovery | `src/startup/recovery.py` | run_cold_start_recovery |
| Frontend API | `trading-ui/lib/api/client.ts` | getApiBase, request, endpoints |
| Frontend store | `trading-ui/store/useStore.ts` | useStore (equity, PnL, autonomy, broker, safe mode) |

This document reflects the **current codebase** only; no features were assumed beyond what is implemented.
