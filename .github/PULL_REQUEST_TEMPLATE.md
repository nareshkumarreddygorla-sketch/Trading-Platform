## Summary
<!-- Describe what this PR does and why. Link to issue if applicable. -->

Fixes #

## Type of Change
- [ ] Feature (new functionality)
- [ ] Bug fix (non-breaking change that fixes an issue)
- [ ] Hotfix (critical production fix)
- [ ] Refactor (no functional changes)
- [ ] Documentation
- [ ] CI/CD / Infrastructure
- [ ] Dependencies update

## Changes Made
<!-- Bullet points of what changed -->
-

## Trading Safety Checklist
<!-- REQUIRED for changes touching src/execution/, src/risk_engine/, src/strategy_engine/ -->
- [ ] Changes do NOT affect order execution logic, OR have been tested in paper mode
- [ ] Risk limits are respected (max position 5%, max daily loss 2%)
- [ ] No hardcoded API keys, broker credentials, or secrets
- [ ] Kill switch / circuit breaker behavior is preserved
- [ ] Graceful shutdown handles open positions correctly

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests pass (`pytest tests/integration/`)
- [ ] Frontend tests pass (`cd trading-ui && npm test`)
- [ ] Manually tested in paper trading mode (if applicable)
- [ ] Coverage remains >= 80%

## Review Checklist
- [ ] Code follows project style (ruff, ESLint)
- [ ] No TODO/FIXME left without a linked issue
- [ ] API changes are backward compatible (or migration documented)
- [ ] Environment variables documented in `.env.example` if added

## Screenshots (if UI changes)
<!-- Paste screenshots or screen recordings here -->

## Deployment Notes
<!-- Any special steps needed for deployment? DB migrations? Config changes? -->
