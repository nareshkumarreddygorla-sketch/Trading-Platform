# AI Layer — Deployment & Operations

## Overview

The AI Autonomous Improvement Layer runs alongside the existing trading stack. All AI outputs (signals, weights, LLM suggestions) pass through the **risk engine**; no bypass.

## Components to Deploy

| Component | How to run | Frequency |
|-----------|------------|-----------|
| Feature pipeline | On bar close (stream or batch job) | Every 1m/5m/1h per symbol |
| Model registry | In-process with API/engine | Always |
| Regime classifier | On bar close or every N bars | Every 1–5 min |
| Meta-allocator | After P&L update | Every cycle / EOD |
| Self-learning orchestrator | Scheduled job (cron/K8s CronJob) | Weekly or on drift |
| LLM services | On event (news) or schedule | On trigger |

## Configuration

- **Feature store**: Set `FEATURE_STORE_URL` (e.g. TimescaleDB or object store path).
- **Models**: Artifacts in object store or path; registry loads by `model_id` + `version`.
- **LLM**: `LLM_PROVIDER`, `LLM_API_KEY`, `LLM_MODEL` (e.g. openai / gpt-4o-mini).
- **Drift / retrain**: `DRIFT_THRESHOLD`, `RETRAIN_COMPARE_METRIC` (e.g. sharpe).

## Example: Weekly Retrain (K8s CronJob)

```yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: ai-retrain
  namespace: trading
spec:
  schedule: "0 2 * * 0"  # 02:00 UTC Sunday
  jobTemplate:
    spec:
      template:
        spec:
          containers:
            - name: retrain
              image: trading-ai:latest
              command: ["python", "-m", "src.ai.self_learning.retrain_job"]
              envFrom:
                - configMapRef:
                    name: trading-config
          restartPolicy: OnFailure
```

## Example: Run Training Pipeline

```bash
export PYTHONPATH=.
python scripts/train_example.py
# Or from code:
# from scripts.train_example import run_example_training
# model, metrics = run_example_training()
```

## Rollback

- **Model**: Revert to previous version in model registry (config or API).
- **Feature version**: Use `FEATURE_VERSION` to read previous feature set.
- **Blue/green**: Deploy new AI service version; switch traffic; rollback by switching back.

## Observability

- **Prometheus**: `trading_ai_model_confidence`, `trading_ai_regime`, `trading_ai_drift_detected_total`, `trading_ai_retrain_total`.
- **Grafana**: Add panels for regime, model confidence, drift and retrain counts (see `deploy/observability/grafana/dashboards/`).
- **Logs**: All retrain and drift events are logged with timestamps and version IDs.

## Risk-First Checklist

- [ ] No AI path skips `RiskManager.can_place_order()` for orders.
- [ ] LLM suggestions only update config/weights via `AIRiskGate.allow_parameter_change()`.
- [ ] Circuit breaker blocks all new orders including AI-originated.
- [ ] Meta-allocator weights are normalized and capped before use.
