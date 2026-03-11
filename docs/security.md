# Security Guide

## Overview

This document describes the security architecture, authentication system, secrets management, and compliance controls for the Trading Platform.

---

## 1. Authentication & Authorization

### JWT Token System

The platform uses JWT-based authentication with access and refresh tokens.

| Token | Lifetime | Purpose |
|-------|----------|---------|
| Access Token | 30 minutes | API request authentication |
| Refresh Token | 7 days | Obtain new access tokens |

### Authentication Flow

```
1. User registers: POST /api/auth/register
   -> Password hashed with bcrypt (12 rounds)
   -> User record created in PostgreSQL

2. User logs in: POST /api/auth/login
   -> Credentials verified against bcrypt hash
   -> Access + refresh tokens issued

3. API requests: Authorization: Bearer <access_token>
   -> JWT signature verified
   -> Token blacklist checked
   -> User context attached to request

4. Token refresh: POST /api/auth/refresh
   -> Refresh token validated
   -> New access token issued

5. Logout: POST /api/auth/logout
   -> Token added to blacklist
```

### Password Requirements

- Minimum 12 characters
- Must contain: uppercase, lowercase, digit, special character
- Hashed with bcrypt (passlib)
- Never stored in plaintext

### Token Blacklist

Revoked tokens are tracked in `TokenBlacklist` to prevent reuse after logout. Tokens are checked on every authenticated request.

---

## 2. API Security

### Security Headers (middleware.py)

Every response includes:

| Header | Value | Purpose |
|--------|-------|---------|
| X-Content-Type-Options | nosniff | Prevent MIME sniffing |
| X-Frame-Options | DENY | Prevent clickjacking |
| Referrer-Policy | strict-origin-when-cross-origin | Control referrer leakage |
| Permissions-Policy | camera=(), microphone=() | Restrict browser features |
| Content-Security-Policy | script-src 'nonce-...' | Prevent XSS |

**Note**: `X-XSS-Protection` is intentionally omitted because CSP provides superior protection and the header can introduce vulnerabilities in some browsers.

### Rate Limiting

- Order submission: Configurable orders-per-minute via `OrderRateLimiter`
- API endpoints: Rate limiting at Nginx/reverse proxy layer

### Input Validation

- All request bodies validated by Pydantic models
- SQL injection prevented by SQLAlchemy parameterized queries
- Path traversal prevented by strict path validation

---

## 3. Secrets Management

### Environment Variables

Sensitive configuration is loaded exclusively from environment variables, never hardcoded:

```
# Database
DATABASE_URL=postgresql+asyncpg://user:pass@host/db

# Redis
REDIS_URL=redis://host:6379

# JWT
JWT_SECRET=<random-256-bit-key>

# Broker
ANGEL_ONE_API_KEY=<encrypted>
ANGEL_ONE_CLIENT_ID=<encrypted>
ANGEL_ONE_PASSWORD=<encrypted>
ANGEL_ONE_TOTP_SECRET=<encrypted>
```

### Credential Encryption

Broker credentials are encrypted at rest using `encrypt_credential()` / `decrypt_credential()` functions. Credentials are:
1. Encrypted before storage
2. Decrypted only at point of use
3. Never logged in any form

### CI Secret Scanning

The CI pipeline includes a hardcoded secrets check:
- Scans all Python files for password patterns
- Excludes legitimate patterns (empty defaults, env lookups, Pydantic fields)
- Fails the build if actual secrets are found

### PII Redaction

The logging configuration (`logging_config.py`) applies regex-based redaction:
- Passwords, API keys, and tokens are replaced with `***REDACTED***`
- Applied globally to all log output

---

## 4. Broker Security

### Connection Security

- All broker API calls use HTTPS/TLS
- WebSocket connections use WSS
- TOTP-based two-factor authentication for Angel One

### Session Management

- Broker sessions are short-lived
- Access tokens refreshed automatically
- Sessions invalidated on kill switch activation

### Paper Mode Isolation

Paper mode uses a simulated broker that never connects to real APIs. The `PAPER_MODE=true` environment variable ensures no real orders are placed during testing.

---

## 5. Data Protection

### Database Security

- Connections use SSL in production
- Credentials stored in environment variables (not config files)
- Row-level access control via user context

### Audit Trail Integrity

The audit trail uses HMAC-SHA256 hash chains:
- Each event includes a hash of the previous event
- Tampering detection via chain verification
- Event details included in hash computation

### Data Retention

The compliance module (`retention.py`) enforces data retention policies:
- Trade records retained per regulatory requirements
- Old data archived to cold storage
- Deletion only via audited retention process

---

## 6. Infrastructure Security

### Docker

- Production images use non-root user
- Multi-stage builds (no build tools in runtime image)
- Health check endpoints for liveness/readiness
- `tini` as PID 1 init process

### Network

- Nginx reverse proxy with TLS termination
- Internal services communicate on private network
- No direct database exposure to public network

### Kubernetes

- Namespace isolation
- ConfigMap for non-sensitive configuration
- Secrets for credentials (base64 encoded, should use Vault in production)

---

## 7. Security Testing

### Automated Checks (CI)

1. **Bandit** - Static analysis for Python security issues
2. **pip-audit** - Dependency vulnerability scanning
3. **Hardcoded secrets check** - Grep-based detection of embedded credentials
4. **Ruff security rules** - `S` rule set enabled (security-focused linting)

### Security Test Coverage

Tests in `tests/test_security.py` and `tests/test_auth_security.py` verify:
- JWT validation and rejection of invalid tokens
- Password hashing and verification
- Security header presence
- CORS policy enforcement
- Rate limiting behavior
- Token blacklist functionality

---

## 8. Incident Response

### Kill Switch

The kill switch (`kill_switch.py`) provides emergency halt capability:
- **Arm reasons**: Manual, max drawdown, fill mismatch, market feed failure, max daily loss
- **Effect**: Blocks all new orders except position-reducing ones
- **Auto-disarm**: Some reasons auto-disarm after conditions normalize
- **Persistence**: State persisted to disk, survives restarts

### Circuit Breaker

The circuit breaker (`circuit_breaker.py`) provides automatic trading halt:
- **Trigger**: Portfolio drawdown exceeds configurable threshold
- **States**: CLOSED (normal) -> OPEN (halted) -> HALF_OPEN (testing)
- **Recovery**: Automatic after cooldown + observation period
