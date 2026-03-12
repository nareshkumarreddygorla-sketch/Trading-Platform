# Development Process

## Overview

This document describes the engineering workflow, coding standards, CI/CD pipeline, and testing practices for the Trading Platform.

---

## 1. Git Workflow

### Branch Strategy

| Branch | Purpose | Protection |
|--------|---------|------------|
| `main` | Production-ready code | Required: CI pass, PR review |
| `develop` | Integration branch | Required: CI pass |
| `staging` | Pre-production validation | Required: CI pass |
| `feature/*` | New features | CI runs on PR |
| `fix/*` | Bug fixes | CI runs on PR |
| `hotfix/*` | Critical production fixes | Fast-track review |

### PR Process

1. Create feature branch from `develop` (or `main` for hotfixes)
2. Implement changes with tests
3. Run `ruff check src tests` and `ruff format src tests` locally
4. Push and create PR
5. CI automatically runs: lint, format check, pytest, security scan
6. Code review required
7. Merge when all checks pass

---

## 2. CI/CD Pipeline

### GitHub Actions Workflow (`.github/workflows/ci.yaml`)

```
Trigger: Push/PR to main, develop, staging

Jobs:
1. changes        - Detect which files changed (backend/frontend/infra)
2. backend-lint-test
   - Python 3.11 + 3.12 matrix
   - ruff check (lint)
   - ruff format --check (formatting)
   - pytest with coverage (--cov-fail-under=20)
   - bandit security scan
   - hardcoded secrets check
3. frontend-lint-build
   - npm ci, lint, tsc --noEmit, build
4. docker-build
   - Build API + frontend Docker images (push-only)
5. pr-quality
   - Gate check: all required jobs must pass
6. backend-integration
   - Integration tests (push-only)
```

### Running CI Locally

```bash
# Lint
ruff check src tests --output-format=full

# Format
ruff format --check src tests

# Tests
pytest tests/ -v --tb=short --timeout=60 -x \
  -m "not slow" \
  --cov=src --cov-report=term-missing

# Security
bandit -r src/ -ll --skip B101
```

---

## 3. Coding Standards

### Python

- **Style**: Ruff-enforced (line length 120, Python 3.11+ syntax)
- **Type hints**: Required on all function signatures
- **Docstrings**: Required on classes and public methods
- **Naming**: snake_case for functions/variables, PascalCase for classes
- **Imports**: Sorted by ruff (isort-compatible)
- **Models**: Pydantic BaseModel for all DTOs and configs
- **Async**: Use `async/await` for I/O-bound operations

### Configuration (`ruff.toml`)

```toml
line-length = 120
target-version = "py311"
select = ["E", "F", "W", "I", "UP", "B", "SIM", "S"]  # Including security rules
```

### Architecture Rules

1. **Single order entry**: All orders must go through `OrderEntryService.submit_order()`
2. **Risk before execution**: No order reaches a broker without passing all risk checks
3. **Kill switch supremacy**: When armed, only position-reducing orders are allowed
4. **Audit everything**: Every order, fill, risk decision, and config change is logged
5. **Paper mode parity**: Paper and live modes use identical risk and execution logic

---

## 4. Testing Standards

### Test Organization

```
tests/
├── conftest.py              # Shared fixtures (app, sample_bars, state isolation)
├── test_*.py                # Unit and integration tests
├── chaos/                   # Chaos engineering tests
├── integration/             # Cross-system integration tests
├── qa/                      # Quality assurance tests
└── stress/                  # Concurrent stress tests
```

### Test Requirements

| System | Requirement |
|--------|------------|
| Risk Engine | Unit tests for every limit check, circuit breaker states |
| Order Execution | Pipeline tests, idempotency, kill switch, rate limiting |
| Market Data | Connector tests, validation, caching |
| AI Models | Prediction format, ensemble logic, feature completeness |
| API Endpoints | Request/response validation, auth, error handling |
| Concurrent Operations | Thread safety for all shared-state operations |

### Writing Tests

```python
# Good: Specific, isolated, tests one behavior
def test_zero_quantity_rejected():
    """Order with zero quantity must be rejected."""
    rm = RiskManager(equity=1_000_000, load_persisted_state=False)
    signal = _make_signal()
    result = rm.can_place_order(signal, quantity=0, price=1000.0)
    assert not result.allowed

# Good: Uses fixtures for isolation
@pytest.fixture(autouse=True)
def _isolate_circuit_state(tmp_path, monkeypatch):
    """Prevent tests from reading/writing shared state files."""
    monkeypatch.setattr(rm_mod, "_CIRCUIT_STATE_PATH", tmp_path / "state.json")
```

### Running Tests

```bash
# Full suite
pytest tests/ -v --tb=short --timeout=60

# Specific test file
pytest tests/test_risk_engine.py -v

# With coverage
pytest tests/ --cov=src --cov-report=term-missing

# Skip slow tests
pytest tests/ -m "not slow"

# Stop on first failure
pytest tests/ -x
```

### Test Markers

- `@pytest.mark.slow` - Long-running tests (excluded from CI by default)
- `@pytest.mark.asyncio` - Async test functions (auto-detected via `asyncio_mode = auto`)

---

## 5. Dependencies

### Adding Dependencies

1. Add to `requirements.txt` (production) or `requirements-dev.txt` (development only)
2. Pin version with `>=` minimum (e.g., `fastapi>=0.115.0`)
3. Run `pip-audit` to check for vulnerabilities
4. Test locally before pushing

### Key Dependencies

| Category | Package | Purpose |
|----------|---------|---------|
| Framework | fastapi, uvicorn | API server |
| Data | pandas, numpy, scipy | Data processing |
| ML | xgboost, torch, scikit-learn | Model training/inference |
| Database | sqlalchemy, alembic | ORM and migrations |
| Cache | redis | Caching and pub/sub |
| Auth | pyjwt, passlib, pyotp | Authentication |
| Testing | pytest, httpx, ruff | Testing and linting |

---

## 6. Database Migrations

### Using Alembic

```bash
# Create new migration
alembic revision --autogenerate -m "description"

# Run migrations
alembic upgrade head

# Rollback one step
alembic downgrade -1

# View current version
alembic current
```

### Migration Naming

Format: `NNN_description.py` (e.g., `005_add_trade_analytics_table.py`)

---

## 7. Release Process

1. Feature branches merge to `develop`
2. `develop` merges to `staging` for pre-production validation
3. `staging` merges to `main` for production release
4. Tag release: `git tag v1.x.x`
5. GitHub Actions builds and pushes Docker images
6. Deploy via blue-green strategy
