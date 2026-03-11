# Contributing to AlphaForge Trading Platform

## Getting Started

### Prerequisites
- Python 3.11+
- Node.js 20+
- Docker & Docker Compose
- Git

### Local Setup

```bash
# Clone
git clone https://github.com/nareshkumarreddygorla-sketch/Trading-Platform.git
cd Trading-Platform

# Backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Frontend
cd trading-ui && npm ci && cd ..

# Pre-commit hooks (REQUIRED)
make setup-hooks

# Run
make dev
```

## Branching Strategy

| Branch | Purpose | Merges To | Reviews |
|--------|---------|-----------|---------|
| `feature/*` | New features | `develop` | 1 |
| `bugfix/*` | Non-critical fixes | `develop` | 1 |
| `hotfix/*` | Critical production fixes | `main` + `develop` | 2 |
| `release/*` | Release preparation | `staging` | 2 |
| `develop` | Integration branch | `staging` | - |
| `staging` | Pre-production | `main` | 2 |
| `main` | Production | - | - |

### Creating a Feature Branch

```bash
git checkout develop
git pull origin develop
git checkout -b feature/your-feature-name
# ... make changes ...
git push -u origin feature/your-feature-name
# Open PR to develop on GitHub
```

### Commit Messages (Conventional Commits)

Format: `<type>(<scope>): <description>`

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `build`, `ci`, `chore`, `revert`, `hotfix`

Examples:
```
feat(api): add WebSocket heartbeat endpoint
fix(risk): correct max drawdown calculation
hotfix(execution): prevent duplicate order submission
test(ai): add transformer model validation tests
```

## Testing

```bash
make test                    # Backend unit tests
make lint                    # Backend lint
make check                   # Full quality check (lint + test + security)
cd trading-ui && npm test    # Frontend tests
```

### Requirements
- Minimum 80% code coverage for backend
- All tests must pass before merging
- Integration tests run on push to develop/staging/main

## Trading Safety Rules

**Non-negotiable for any code touching trading logic:**

1. **Paper mode first** - All execution/strategy changes tested in paper mode
2. **Risk limits** - Never bypass 5% max position, 2% max daily loss
3. **Kill switch** - Never remove or disable the kill switch
4. **No secrets in code** - Use environment variables for all credentials
5. **Idempotency** - Order operations must be safe to retry

## Pull Request Process

1. Branch from `develop` (or `main` for hotfixes)
2. Use conventional commit messages
3. Ensure all tests pass locally (`make check`)
4. Push and open PR - fill out the template completely
5. Wait for CI + review approval
6. Squash and merge

## Release Process

1. Create `release/vX.Y.Z` from `develop`
2. Update `CHANGELOG.md`
3. PR to `staging` -> validate -> PR to `main`
4. Tag: `make release-tag VERSION=X.Y.Z`
5. GitHub Actions builds and publishes Docker images automatically
