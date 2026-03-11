# Changelog

All notable changes to AlphaForge Trading Platform will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- GitFlow branching strategy with staging gate
- Branch protection rules for main, develop, staging
- PR template with trading safety checklist
- CODEOWNERS for automated review assignment
- Pre-commit hooks (ruff, secrets detection, conventional commits)
- Enhanced CI/CD with path filtering and concurrency
- Release workflow with tag-based Docker publishing to GHCR
- Staging deployment workflow
- CONTRIBUTING.md with developer guidelines

## [1.0.0] - 2026-03-10

### Added
- FastAPI backend with AI-powered autonomous trading
- XGBoost, LSTM, Transformer, RL ensemble with IC-weighted fusion
- Real-time market data (Angel One WebSocket, Yahoo Finance fallback)
- Risk engine: VaR (5 methods), circuit breaker, kill switch, Kelly sizing
- Execution layer: order validation, timeouts, error classification, idempotency
- Paper trading mode for safe testing
- Next.js 14 frontend with real-time WebSocket dashboard
- JWT authentication with rate limiting and token refresh
- 242 passing tests (unit + integration + E2E + stress)
- Docker Compose deployment (dev + production)
- Prometheus + Grafana observability (42-panel dashboard)
- Database backup automation
- SEBI compliance: audit trail, surveillance, OTR monitoring
