# QA Bug Report — Trading Platform

**Role:** Senior QA  
**Scope:** Full codebase (API, execution, risk, strategy, alpha research, backtest, frontend)  
**Total bugs reported:** 50+

---

## Critical / High

### 1. **FDR (Benjamini–Hochberg) rejects only one hypothesis**
**File:** `src/ai/alpha_research/validation/fdr.py`  
**Issue:** The loop sets `reject[order[k]] = True` for a single `k` and then `break`s. BH procedure should reject all hypotheses 1..k (all indices with p ≤ threshold).  
**Impact:** Most discoveries are incorrectly marked as not rejected; alpha research under-rejects.

### 2. **Backtest equity on SELL uses PnL instead of sale proceeds**
**File:** `src/backtesting/engine.py`  
**Issue:** On sell, code does `equity += pnl - commission`. Equity should increase by sale proceeds (fill_price × fill_qty) minus commission, not just PnL. Cost was already subtracted on buy.  
**Impact:** Equity curve is wrong; backtest results and metrics are invalid.

### 3. **Idempotency race allows duplicate broker orders**
**File:** `src/execution/order_entry/service.py`  
**Issue:** Idempotency is checked with `get()`, then broker is called, then `set()` is called. Two concurrent requests with the same key can both see `get() = None` and both call the broker.  
**Impact:** Duplicate orders when clients retry with the same idempotency key.

### 4. **Risk /limits returns default limits, not app state**
**File:** `src/api/routers/risk.py`  
**Issue:** `get_limits()` does `RiskLimits()` and returns that. It never reads `app.state.risk_manager.limits`.  
**Impact:** Dashboard and clients see default limits, not the limits actually used by the risk manager.

### 5. **Risk /state returns hardcoded stub**
**File:** `src/api/routers/risk.py`  
**Issue:** `risk_state()` returns a fixed dict (`circuit_open: False`, `daily_pnl: 0`, etc.) and does not use RiskManager or CircuitBreaker.  
**Impact:** Risk state on dashboard is always stub data; real circuit/PnL/positions are not shown.

### 6. **PUT /risk/limits accepts any dict with no validation**
**File:** `src/api/routers/risk.py`  
**Issue:** `update_limits(limits: dict)` is not validated. Negative values, wrong types, or missing keys can be stored/used.  
**Impact:** Invalid limits can break risk checks or cause incorrect behavior.

### 7. **StrategyRegistry.disable() can create orphan _enabled keys**
**File:** `src/strategy_engine/registry.py`  
**Issue:** `disable(strategy_id)` does `self._enabled[strategy_id] = False` without checking `strategy_id in self._strategies`. Unknown IDs get an _enabled entry only.  
**Impact:** _enabled can contain keys not in _strategies; enable() correctly no-ops, but state is inconsistent.

### 8. **Angel One gateway hardcodes Exchange.NSE**
**File:** `src/execution/angel_one_gateway.py`  
**Issue:** Returned `Order` always uses `exchange=Exchange.NSE` even when the request had a different exchange.  
**Impact:** Orders for BSE/other venues are stored and displayed as NSE.

### 9. **Angel One live path can raise on invalid side**
**File:** `src/execution/angel_one_gateway.py`  
**Issue:** In non-paper path, `SignalSide(side)` is called without a guard. Invalid `side` (e.g. "buy" or "SHORT") raises ValueError.  
**Impact:** 500 on invalid side in live mode; paper mode safely falls back to BUY.

### 10. **Place order: LIMIT with limit_price=None not rejected**
**File:** `src/api/routers/orders.py`  
**Issue:** Validation only checks `limit_price <= 0` when `limit_price is not None`. For order_type "LIMIT", limit_price can be None and still pass.  
**Impact:** LIMIT orders can be sent to execution with no price; behavior depends on gateway.

### 11. **Place order: quantity truncated to int**
**File:** `src/api/routers/orders.py`  
**Issue:** `int(body.quantity)` truncates (e.g. 1.9 → 1). No rounding or rejection of non-integer quantity.  
**Impact:** Silent truncation; user may expect rounded or integer-only validation.

### 12. **Order rejection: None reject_reason not handled**
**File:** `src/api/routers/orders.py`  
**Issue:** If `result.success` is False and `result.reject_reason` is None, the code falls through to the final `raise HTTPException(503, ...)` and uses `result.reject_reason` in the message, yielding "None".  
**Impact:** Confusing error message; should handle None explicitly.

### 13. **SignalSide enum is case-sensitive**
**File:** `src/api/routers/orders.py` + `src/core/events.py`  
**Issue:** `SignalSide(body.side)` requires "BUY"/"SELL". Client sending "buy" or "Buy" gets validation error.  
**Impact:** API is brittle; normalizes to uppercase or document strictly.

### 14. **Lifecycle.register silently ignores empty order_id**
**File:** `src/execution/lifecycle.py`  
**Issue:** If `order_id` is empty, we log and return without raising. Caller does not know registration failed.  
**Impact:** Order can be “lost” from lifecycle tracking; no exception for caller to handle.

### 15. **Lifecycle.list_recent order is not “recent”**
**File:** `src/execution/lifecycle.py`  
**Issue:** `list(self._orders.values())[-limit:]` relies on dict iteration order. “Recent” is not defined by time; it’s insertion order (Python 3.7+). No sort by timestamp.  
**Impact:** “Recent” orders may not be the latest by time.

---

## Medium

### 16. **list_orders / list_positions ignore query params**
**File:** `src/api/routers/orders.py`  
**Issue:** `list_orders(status=..., strategy_id=..., limit=...)` and `list_positions()` always return `{"orders": []}` / `{"positions": []}`. Params are unused.  
**Impact:** Filtering and pagination are non-functional.

### 17. **list_orders limit has no upper bound**
**File:** `src/api/routers/orders.py`  
**Issue:** `limit: int = 100` is accepted as-is. Client can send limit=999999.  
**Impact:** When implemented, could allow excessive load.

### 18. **Backtest run always returns same job_id**
**File:** `src/api/routers/backtest.py`  
**Issue:** `run_backtest` always returns `job_id: "bt_1"`. No real job store or uniqueness.  
**Impact:** Multiple runs overwrite conceptually; get_job/get_equity cannot distinguish runs.

### 19. **Backtest start/end dates not validated**
**File:** `src/api/routers/backtest.py`  
**Issue:** `BacktestRunRequest.start` and `end` are strings with no format or range validation.  
**Impact:** Invalid dates can reach engine or cause confusing errors.

### 20. **FillModel docstring wrong return type**
**File:** `src/backtesting/fill_model.py`  
**Issue:** Docstring says “Returns (fill_price, fill_qty, commission)” but implementation returns (fill_bar, fill_price, fill_qty, commission).  
**Impact:** Misleading for maintainers; callers already use 4-tuple correctly.

### 21. **FillModel returns (None, 0, 0, 0) vs (fill_bar, 0, 0, 0)**
**File:** `src/backtesting/fill_model.py`  
**Issue:** When `fill_idx >= len(bars)` we return `(None, 0.0, 0.0, 0.0)`. When fill_qty <= 0 we return `(fill_bar, fill_price, 0.0, 0.0)`. Inconsistent first element (Bar vs None).  
**Impact:** Callers must handle both; type is Optional[Bar] so acceptable but inconsistent.

### 22. **RiskManager.update_equity allows negative**
**File:** `src/risk_engine/manager.py`  
**Issue:** `update_equity(equity)` has no check for equity >= 0.  
**Impact:** Negative equity can break checks that assume non-negative (e.g. drawdown, limits).

### 23. **CircuitBreaker.reset() does not reset _peak_equity**
**File:** `src/risk_engine/circuit_breaker.py`  
**Issue:** `reset()` sets state to CLOSED but does not update _peak_equity. Drawdown is still computed vs old peak.  
**Impact:** Circuit can re-trip immediately after reset if equity is below old peak.

### 24. **Alpha research _last_run not thread-safe**
**File:** `src/api/routers/alpha_research.py`  
**Issue:** Global `_last_run` is updated by a background task. A second POST /run can start another task; both read/write the same dict.  
**Impact:** Race conditions; status/results can be mixed or lost.

### 25. **Alpha research decay_multipliers body not validated**
**File:** `src/api/routers/alpha_research.py`  
**Issue:** `body: dict = None`; if body is not JSON or `signal_ids` is not a list (e.g. string), `get_decay_weight_multipliers(signal_ids)` can fail or behave wrongly.  
**Impact:** 500 or wrong multipliers when request body is malformed.

### 26. **Alpha pipeline run_scoring order dependency**
**File:** `src/ai/alpha_research/pipeline/orchestrator.py`  
**Issue:** `run_scoring()` uses `self._validated`. If called before `run_validation()` or after a failed run, _validated may be stale or empty.  
**Impact:** Stale or empty top decile if API/caller invokes steps out of order.

### 27. **validate_batch_with_fdr mutates results in place**
**File:** `src/ai/alpha_research/validation/validator.py`  
**Issue:** We set `r.passed = False` and `r.reason = "fdr_not_rejected"` on list elements. Callers receive the same list (mutated).  
**Impact:** Surprising side effects; immutable-style return (e.g. new list) would be clearer.

### 28. **permutation_test_ic no seed**
**File:** `src/ai/alpha_research/validation/fdr.py`  
**Issue:** `rng = np.random.default_rng()` has no seed; results are not reproducible.  
**Impact:** Same inputs can yield different p-values across runs.

### 29. **Clustering np.corrcoef with T < 2**
**File:** `src/ai/alpha_research/clustering/cluster.py`  
**Issue:** If `signal_returns` has shape (n, 1), `np.corrcoef` can produce NaN or fail. We check `np.isfinite(corr).all()` but not T >= 2.  
**Impact:** Edge case with very short series can yield NaN or wrong fallback.

### 30. **Health/ready always return OK**
**File:** `src/api/routers/health.py`  
**Issue:** `/ready` returns `{"status": "ready"}` without checking DB/Redis/Kafka.  
**Impact:** Readiness probe can pass while dependencies are down.

### 31. **/metrics returns 200 with non-Prometheus body when client missing**
**File:** `src/api/app.py`  
**Issue:** If prometheus_client is not installed, we return 200 with body `"# prometheus_client not installed\n"`.  
**Impact:** Scrapers may treat as valid Prometheus format and fail or misparse.

### 32. **CORS allow_origins=["*"]**
**File:** `src/api/app.py`  
**Issue:** Any origin is allowed.  
**Impact:** In production, may be too permissive for security policies.

### 33. **No API rate limiting**
**File:** `src/api/`  
**Issue:** No rate limiting on any endpoint.  
**Impact:** DoS or abuse possible.

### 34. **Kill switch arm: reason/detail length not limited**
**File:** `src/api/routers/orders.py`  
**Issue:** `reason` and `detail` are taken from request with no max length.  
**Impact:** Very long strings in logs or storage.

### 35. **Place order: idempotency_key not length-limited**
**File:** `src/api/routers/orders.py`  
**Issue:** Client can send arbitrarily long idempotency_key.  
**Impact:** Redis key size, storage, or performance issues.

---

## Frontend

### 36. **Alpha research poll interval not cleared on unmount**
**File:** `frontend/src/pages/Dashboard.tsx`  
**Issue:** `setInterval` in `runAlphaResearch` is cleared only when status is completed/failed. If user leaves the page, the interval keeps running.  
**Impact:** Wasted requests and possible setState on unmounted component.

### 37. **Toggle strategy: no loading or disable on double-click**
**File:** `frontend/src/pages/Dashboard.tsx`  
**Issue:** Enable/Disable has no loading state; double-click can fire two requests.  
**Impact:** Flicker or inconsistent state; duplicate API calls.

### 38. **No error boundary**
**File:** `frontend/src/App.tsx`  
**Issue:** No React error boundary. Any throw in Dashboard crashes the whole app.  
**Impact:** Single component error takes down entire UI.

### 39. **Run pipeline: old results shown while new run is “running”**
**File:** `frontend/src/pages/Dashboard.tsx`  
**Issue:** When user clicks “Run pipeline” again, we set status to “running” but do not clear `alphaResults`. Previous results stay visible.  
**Impact:** Misleading: user sees old numbers with “running” status.

### 40. **tradingReady fetch: API base with proxy**
**File:** `frontend/src/pages/Dashboard.tsx`  
**Issue:** `fetch('/api/v1/trading/ready')` uses relative URL; with Vite proxy this is correct. If frontend is opened without proxy (e.g. built static), request may go to wrong host.  
**Impact:** Wrong host when not using dev proxy.

---

## Low / Code quality

### 41. **datetime.utcnow deprecated**
**File:** `src/core/events.py`  
**Issue:** `datetime.utcnow` used for Signal and Order default `ts`. Deprecated in Python 3.12+; prefer `datetime.now(timezone.utc)`.  
**Impact:** Deprecation warnings; future removal.

### 42. **Signal score can exceed 1.0**
**File:** `src/core/events.py` + strategy code  
**Issue:** Signal has `score: float = Field(ge=0.0, le=1.0)`. Some strategies use formulas that can yield > 1 (e.g. EMA “score = min(1.0, ...)” in one branch but not all).  
**Impact:** Pydantic validation error if score > 1.

### 43. **MarketState.metadata default mutable**
**File:** `src/strategy_engine/base.py`  
**Issue:** `metadata: dict = None` with `__post_init__` setting to `{}` is fine, but default `None` in dataclass can be shared if not careful.  
**Impact:** Low; __post_init__ mitigates; could use field(default_factory=dict) for clarity.

### 44. **OrderRouter._gateway unknown exchange**
**File:** `src/execution/order_router.py`  
**Issue:** `_gateways.get(exchange, self.default_gateway)` uses default for any unknown exchange. No log when exchange is unknown.  
**Impact:** Wrong gateway could be used silently for new exchanges.

### 45. **Idempotency when Redis unavailable**
**File:** `src/execution/order_entry/idempotency.py`  
**Issue:** When Redis is not installed or down, `get()` always returns None. Every request is treated as new.  
**Impact:** No idempotency; duplicate orders possible when Redis is down.

### 46. **IdempotencyStore.set return value ignored**
**File:** `src/execution/order_entry/service.py`  
**Issue:** We call `await self.idempotency.set(...)` after broker and lifecycle but never check the return value (e.g. False when key already exists).  
**Impact:** Rare; only affects reporting/audit if we wanted to detect duplicate set.

### 47. **RiskManager.remove_position exchange type**
**File:** `src/risk_engine/manager.py`  
**Issue:** `remove_position(symbol, exchange: str)`; callers must pass `exchange` as string (e.g. `p.exchange.value`). If someone passes Exchange enum, comparison `p.exchange.value == exchange` may fail.  
**Impact:** Type contract is str; ensure all callers pass str.

### 48. **FeatureStore.compute_features bars type**
**File:** `src/feature_store/store.py`  
**Issue:** `compute_features(self, bars: List[Any], ...)` uses `b.close` for bars. If bars are raw dicts, this works in Python; if they are Pydantic Bar, also works. Type is ambiguous.  
**Impact:** Maintainability; clarify Bar vs dict.

### 49. **StrategyRegistry.get_enabled_strategies KeyError**
**File:** `src/strategy_engine/registry.py`  
**Issue:** `return [self._strategies[sid] for sid in self.list_enabled()]` assumes every enabled id is in _strategies. If _enabled had an extra key (e.g. from disable(unknown)), list_enabled would not include it (only True values). So safe. Only if enable() added to _enabled without _strategies would we get KeyError.  
**Impact:** Theoretical; current code path keeps them in sync.

### 50. **Backtest config commission_pct vs tax_rate_pct**
**File:** `src/backtesting/engine.py`  
**Issue:** BacktestConfig has `tax_rate_pct` but engine uses only `commission_pct` in FillModel. Tax is not applied.  
**Impact:** Backtest ignores tax; PnL and metrics are pre-tax.

### 51. **Pipeline run_clustering matrix/signal_ids alignment**
**File:** `src/ai/alpha_research/pipeline/orchestrator.py`  
**Issue:** If `signal_returns_matrix` has first dimension != len(signal_ids), clustering returns list(signal_ids) without clustering. No explicit check or error.  
**Impact:** Caller must ensure alignment; wrong matrix can silently bypass clustering.

### 52. **EdgePreservationRules.check_correlation_with_existing skips length mismatch**
**File:** `src/ai/alpha_research/rules/preservation.py`  
**Issue:** For existing returns with different length we `continue`. So we only check existing signals with same length; others are ignored.  
**Impact:** New signal might pass even if highly correlated with an existing one that has different length.

### 53. **update_limits has no auth**
**File:** `src/api/routers/risk.py`  
**Issue:** PUT /risk/limits has no authentication or authorization.  
**Impact:** Any client that can reach the API can change limits.

### 54. **list_orders status/strategy_id unused**
**File:** `src/api/routers/orders.py`  
**Issue:** Parameters `status`, `strategy_id`, `limit` are accepted but not used (TODO).  
**Impact:** When implementation is added, contract is already defined; until then, misleading.

### 55. **deps get_order_entry_service / get_kill_switch return type**
**File:** `src/api/deps.py`  
**Issue:** Return type is not annotated (e.g. Optional[OrderEntryService]).  
**Impact:** Type checkers and IDEs don’t know the return type.

---

## Summary

| Severity | Count |
|----------|--------|
| Critical / High | 15 |
| Medium         | 20 |
| Frontend       | 5  |
| Low / Quality  | 15+ |

**Recommendation:** Address FDR logic (#1), backtest equity (#2), and idempotency race (#3) first. Then fix risk API to use real state/limits (#4, #5, #6) and add validation and auth where needed.
