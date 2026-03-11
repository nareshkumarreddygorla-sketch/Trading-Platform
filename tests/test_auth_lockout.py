"""
Tests for account lockout, bcrypt password hashing, and paper-to-live
confirmation token mechanics.

Covers:
  1. Account lockout after repeated failed login attempts
  2. bcrypt (passlib) password hashing and verification
  3. Paper-to-live confirmation token TTL and storage
"""

import os
import time

# Environment setup (must precede imports that read env vars)
os.environ.setdefault("JWT_SECRET", "test-secret-minimum-32-characters-long!!")
os.environ.setdefault("EXEC_PAPER", "true")
os.environ.setdefault("ENV", "development")


# =========================================================================
# 1. Account lockout tests
# =========================================================================

from src.api.routers.auth import (  # noqa: E402
    _LOCKOUT_DURATION_SECONDS,
    _MAX_FAILED_ATTEMPTS,
    _check_lockout,
    _clear_failed_attempts,
    _failed_attempts,
    _record_failed_attempt,
)


class TestAccountLockout:
    """Verify the account lockout mechanism blocks after N failures."""

    def test_no_lockout_initially(self):
        """A fresh username should not be locked out."""
        username = "test_user_lockout_initial"
        _clear_failed_attempts(username)
        assert _check_lockout(username) is False

    def test_lockout_after_max_failures(self):
        """After _MAX_FAILED_ATTEMPTS failures, the account must be locked."""
        username = "test_user_lockout_max"
        _clear_failed_attempts(username)
        for _ in range(_MAX_FAILED_ATTEMPTS):
            _record_failed_attempt(username)
        assert _check_lockout(username) is True
        _clear_failed_attempts(username)

    def test_clear_resets_lockout(self):
        """Clearing failures must un-lock the account."""
        username = "test_user_lockout_clear"
        _clear_failed_attempts(username)
        for _ in range(_MAX_FAILED_ATTEMPTS):
            _record_failed_attempt(username)
        assert _check_lockout(username) is True
        _clear_failed_attempts(username)
        assert _check_lockout(username) is False

    def test_under_threshold_no_lockout(self):
        """One fewer than the max should NOT trigger lockout."""
        username = "test_user_lockout_under"
        _clear_failed_attempts(username)
        for _ in range(_MAX_FAILED_ATTEMPTS - 1):
            _record_failed_attempt(username)
        assert _check_lockout(username) is False
        _clear_failed_attempts(username)

    def test_max_failed_attempts_is_five(self):
        """The lockout threshold must be exactly 5."""
        assert _MAX_FAILED_ATTEMPTS == 5

    def test_lockout_duration_is_fifteen_minutes(self):
        """The lockout window must be 15 minutes (900 seconds)."""
        assert _LOCKOUT_DURATION_SECONDS == 900

    def test_failed_attempts_dict_exists(self):
        """The module-level _failed_attempts dict must be accessible."""
        assert isinstance(_failed_attempts, dict)

    def test_expired_attempts_are_pruned(self):
        """
        Attempts older than the lockout window should be pruned by _check_lockout,
        so even if we had max failures, after the window passes the account is unlocked.
        """
        username = "test_user_lockout_expire"
        _clear_failed_attempts(username)
        # Inject failures that are older than the lockout window
        old_time = time.time() - _LOCKOUT_DURATION_SECONDS - 10
        _failed_attempts[username] = [old_time] * _MAX_FAILED_ATTEMPTS
        # _check_lockout prunes old entries
        assert _check_lockout(username) is False
        _clear_failed_attempts(username)

    def test_mixed_old_and_new_attempts(self):
        """Old expired attempts should be pruned; only recent ones count."""
        username = "test_user_lockout_mixed"
        _clear_failed_attempts(username)
        old_time = time.time() - _LOCKOUT_DURATION_SECONDS - 10
        # Add some old (expired) attempts
        _failed_attempts[username] = [old_time] * 3
        # Add fewer-than-threshold recent attempts
        for _ in range(_MAX_FAILED_ATTEMPTS - 2):
            _record_failed_attempt(username)
        # Should NOT be locked (old ones pruned, recent ones under threshold)
        assert _check_lockout(username) is False
        _clear_failed_attempts(username)


# =========================================================================
# 2. Bcrypt / in-memory password hashing tests
# =========================================================================

from src.api.routers.auth import _hash_inmemory, _verify_inmemory  # noqa: E402


class TestPasswordHashing:
    """Verify bcrypt/PBKDF2 hashing for the in-memory user store."""

    def test_hash_and_verify_correct_password(self):
        """Hashing then verifying the same password must return True."""
        password = "MyStr0ng!PassW0rd"
        hashed = _hash_inmemory(password)
        assert _verify_inmemory(password, hashed) is True

    def test_hash_is_not_plaintext(self):
        """The hash must differ from the original password."""
        password = "MyStr0ng!PassW0rd"
        hashed = _hash_inmemory(password)
        assert hashed != password, "Hash should not be plaintext"

    def test_wrong_password_fails_verification(self):
        """Verifying with the wrong password must return False."""
        password = "MyStr0ng!PassW0rd"
        hashed = _hash_inmemory(password)
        assert _verify_inmemory("wrong_password", hashed) is False

    def test_unique_salts_produce_different_hashes(self):
        """Two hashes of the same password must differ (unique salts)."""
        password = "MyStr0ng!PassW0rd"
        hash1 = _hash_inmemory(password)
        hash2 = _hash_inmemory(password)
        assert hash1 != hash2, "Each hash should use a unique salt"
        # Both must still verify
        assert _verify_inmemory(password, hash1) is True
        assert _verify_inmemory(password, hash2) is True

    def test_empty_password_does_not_crash(self):
        """Hashing an empty string should not raise; verification should work."""
        hashed = _hash_inmemory("")
        assert _verify_inmemory("", hashed) is True
        assert _verify_inmemory("notempty", hashed) is False

    def test_long_password_hashing(self):
        """Very long passwords should hash and verify correctly."""
        password = "A" * 500 + "!1a"
        hashed = _hash_inmemory(password)
        assert _verify_inmemory(password, hashed) is True

    def test_special_characters_in_password(self):
        """Passwords with unicode and special chars should work."""
        password = "P@$$w0rd!#%^&*()"
        hashed = _hash_inmemory(password)
        assert _verify_inmemory(password, hashed) is True
        assert _verify_inmemory("P@$$w0rd!#%^&*(", hashed) is False


# =========================================================================
# 3. Password strength validation tests
# =========================================================================

from src.api.routers.auth import MIN_PASSWORD_LENGTH, _validate_password_strength  # noqa: E402


class TestPasswordStrength:
    """Verify password strength validation rules."""

    def test_strong_password_passes(self):
        assert _validate_password_strength("MyStr0ng!Pass") is None

    def test_too_short_password_fails(self):
        result = _validate_password_strength("Short1!")
        assert result is not None
        assert "at least" in result

    def test_missing_uppercase_fails(self):
        result = _validate_password_strength("mystrongpass1!!")
        assert result is not None
        assert "uppercase" in result

    def test_missing_lowercase_fails(self):
        result = _validate_password_strength("MYSTRONGPASS1!!")
        assert result is not None
        assert "lowercase" in result

    def test_missing_digit_fails(self):
        result = _validate_password_strength("MyStrongPass!!!")
        assert result is not None
        assert "digit" in result

    def test_missing_special_char_fails(self):
        result = _validate_password_strength("MyStrongPass123")
        assert result is not None
        assert "special" in result

    def test_min_password_length_is_twelve(self):
        assert MIN_PASSWORD_LENGTH == 12


# =========================================================================
# 4. Paper-to-live confirmation token tests
# =========================================================================

from src.api.routers.broker import _CONFIRM_TOKEN_TTL_SECONDS, _pending_live_switch  # noqa: E402


class TestPaperToLiveConfirmation:
    """Verify the paper-to-live confirmation token mechanics."""

    def test_confirm_token_ttl_is_five_minutes(self):
        """The confirmation token TTL must be 300 seconds (5 minutes)."""
        assert _CONFIRM_TOKEN_TTL_SECONDS == 300

    def test_pending_switch_is_dict(self):
        """The module-level _pending_live_switch dict must exist and be a dict."""
        assert isinstance(_pending_live_switch, dict)

    def test_pending_switch_is_initially_empty_or_accessible(self):
        """We can read _pending_live_switch without errors."""
        # Just verify no exception is raised
        _ = len(_pending_live_switch)

    def test_pending_switch_supports_token_storage(self):
        """We can store and retrieve a confirmation token."""
        test_token = "test_confirm_token_12345"
        _pending_live_switch[test_token] = {
            "creds": None,
            "ts": time.time(),
            "user_id": "test_user",
        }
        assert test_token in _pending_live_switch
        assert _pending_live_switch[test_token]["user_id"] == "test_user"
        # Clean up
        del _pending_live_switch[test_token]

    def test_expired_token_is_detectable(self):
        """An expired token (ts older than TTL) can be detected."""
        test_token = "test_expired_token"
        expired_ts = time.time() - _CONFIRM_TOKEN_TTL_SECONDS - 1
        _pending_live_switch[test_token] = {
            "creds": None,
            "ts": expired_ts,
            "user_id": "test_user",
        }
        # Verify we can detect expiry
        entry = _pending_live_switch[test_token]
        is_expired = (time.time() - entry["ts"]) > _CONFIRM_TOKEN_TTL_SECONDS
        assert is_expired is True
        # Clean up
        del _pending_live_switch[test_token]


# =========================================================================
# 5. Token blacklist integration tests
# =========================================================================

from src.api.token_blacklist import blacklist_token, is_blacklisted  # noqa: E402


class TestTokenBlacklist:
    """Verify that blacklisted tokens are correctly rejected."""

    def test_fresh_token_is_not_blacklisted(self):
        assert is_blacklisted("never-seen-token-12345") is False

    def test_blacklisted_token_is_detected(self):
        token = "test-blacklist-token-67890"
        expires_at = time.time() + 3600  # 1 hour from now
        blacklist_token(token, expires_at)
        assert is_blacklisted(token) is True

    def test_different_token_not_affected(self):
        """Blacklisting one token should not affect other tokens."""
        blacklist_token("token_a", time.time() + 3600)
        assert is_blacklisted("token_b_different") is False
