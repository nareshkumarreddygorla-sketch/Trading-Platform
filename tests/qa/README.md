# QA Strategy: 5-Layer System Breakdown

Run all QA tests:

```bash
pytest tests/qa/ -v
```

With **Redis** running (localhost:6379), execution-integrity and chaos tests that depend on idempotency/cluster will run; otherwise they are skipped.

## Coverage by Phase

| Phase | File | Tests |
|-------|------|--------|
| **1 Execution integrity** | `test_execution_integrity.py` | Idempotency storm (100 same key → 1 broker call), kill switch armed rejects new order, reduce-only SELL when long, circuit open blocks orders |
| **2 Autonomous loop** | `test_autonomous_loop.py` | Duplicate bar → one submit, strategy disable after 5 losses, market feed unhealthy → loop skips |
| **3 Risk & circuit** | `test_risk_circuit_safety.py` | VaR breach reject, cluster reservation overflow (max 5 broker calls) |
| **4 Chaos recovery** | `test_chaos_recovery.py` | Broker latency timeout, Redis down → reject no broker call, persist retry exists |
| **5 AI + Capital + Adversarial** | `test_ai_capital_adversarial.py` | Deterministic model, confidence threshold edge, capital gate ok=false when stress/restart not passed, adversarial circuit+idempotency no bypass |
| **7 Frontend / WebSocket** | `test_websocket_frontend.py` | WebSocket flood (1000 events stable, multiple clients), JWT expiry rejected (4001) |
| **8 Performance** | `test_performance.py` | Loop tick &lt; 200ms (200 symbols), features+regime path &lt; 500ms |
| **Restart during fill** | `test_restart_during_fill.py` | No duplicate fill (same event twice → single position entry), idempotency same key after restart returns existing |

## Critical Bug Categories Covered

- **Idempotency race**: same key → single broker call (Phase 1)
- **Kill switch / circuit**: armed → reject or reduce-only; circuit open → reject (Phase 1, 3)
- **Reservation / overflow**: max_open caps broker calls (Phase 3)
- **Redis down**: idempotency unavailable → reject, no broker (Phase 4)
- **Market feed death**: unhealthy → loop skips (Phase 2)
- **Strategy disable**: consecutive losses → disabled + event (Phase 2)
- **Capital gate**: validate() false when stress/restart not passed (Phase 5)
- **Adversarial**: circuit open + submit → no broker call (Phase 5)

## Slow tests

Tests that take >5s (e.g. broker latency timeout) are marked `@pytest.mark.slow`. Exclude them in CI by default:

```bash
pytest tests/qa/ -v -m "not slow"
```

Include slow tests:

```bash
pytest tests/qa/ -v
```

## Optional: Full Run with Redis

```bash
# Start Redis (e.g. docker run -p 6379:6379 redis:7-alpine)
pytest tests/qa/ -v
```

Then all tests run (no skips for Redis).
