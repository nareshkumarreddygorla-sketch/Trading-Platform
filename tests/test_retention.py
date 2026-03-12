"""Unit tests for src.compliance.retention — SEBI data retention manager.

Covers:
- RetentionPolicy validation (SEBI 5-year minimum)
- RetentionManager construction and default policies
- is_within_retention_window() / can_delete()
- block_deletion_if_required() enforcement
- should_archive() timing
- archive_data() to cold storage (JSON and file-based, compressed/uncompressed)
- should_purge() and purge_if_expired()
- run_cleanup_cycle() on a directory
- get_policies() and get_retention_summary()
- get_action_log() with filters
- Thread safety
"""

import gzip
import json
import os
import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from src.compliance.retention import (
    DEFAULT_MAX_RETENTION_YEARS,
    SEBI_MIN_RETENTION_YEARS,
    DataCategory,
    RetentionAction,
    RetentionManager,
    RetentionPolicy,
)


@pytest.fixture
def tmp_archive(tmp_path) -> Path:
    """Temporary directory for cold storage."""
    return tmp_path / "archive"


@pytest.fixture
def manager(tmp_archive) -> RetentionManager:
    return RetentionManager(cold_storage_path=str(tmp_archive))


# ──────────────────────────────────────────────────
# RetentionPolicy validation
# ──────────────────────────────────────────────────


class TestRetentionPolicyValidation:
    def test_sebi_minimum_enforced(self):
        with pytest.raises(ValueError, match="SEBI requires minimum"):
            RetentionPolicy(category=DataCategory.TRADE_RECORDS, min_retention_years=3)

    def test_max_must_exceed_min(self):
        with pytest.raises(ValueError, match="max_retention_years"):
            RetentionPolicy(
                category=DataCategory.TRADE_RECORDS,
                min_retention_years=5,
                max_retention_years=4,
            )

    def test_archive_after_capped_at_min(self):
        """archive_after_years > min_retention_years gets clamped."""
        p = RetentionPolicy(
            category=DataCategory.TRADE_RECORDS,
            min_retention_years=5,
            archive_after_years=10,
        )
        assert p.archive_after_years == 5

    def test_valid_policy(self):
        p = RetentionPolicy(
            category=DataCategory.AUDIT_EVENTS,
            min_retention_years=5,
            max_retention_years=10,
            archive_after_years=2,
        )
        assert p.min_retention_years == 5
        assert p.max_retention_years == 10


# ──────────────────────────────────────────────────
# RetentionManager construction
# ──────────────────────────────────────────────────


class TestManagerConstruction:
    def test_all_categories_have_policies(self, manager: RetentionManager):
        for cat in DataCategory:
            assert cat in manager._policies

    def test_default_min_retention_is_sebi_compliant(self, manager: RetentionManager):
        for policy in manager._policies.values():
            assert policy.min_retention_years >= SEBI_MIN_RETENTION_YEARS

    def test_custom_policies_applied(self, tmp_archive):
        custom = RetentionPolicy(
            category=DataCategory.TRADE_RECORDS,
            min_retention_years=7,
            max_retention_years=12,
        )
        mgr = RetentionManager(cold_storage_path=str(tmp_archive), policies=[custom])
        assert mgr._policies[DataCategory.TRADE_RECORDS].min_retention_years == 7

    def test_cold_storage_directory_created(self, tmp_archive):
        RetentionManager(cold_storage_path=str(tmp_archive))
        assert tmp_archive.is_dir()


# ──────────────────────────────────────────────────
# Retention window checks
# ──────────────────────────────────────────────────


class TestRetentionWindow:
    def test_recent_data_within_window(self, manager: RetentionManager):
        recent = datetime.now(UTC) - timedelta(days=30)
        assert manager.is_within_retention_window(DataCategory.TRADE_RECORDS, recent) is True

    def test_old_data_outside_window(self, manager: RetentionManager):
        old = datetime.now(UTC) - timedelta(days=SEBI_MIN_RETENTION_YEARS * 365 + 10)
        assert manager.is_within_retention_window(DataCategory.TRADE_RECORDS, old) is False

    def test_can_delete_recent_data(self, manager: RetentionManager):
        recent = datetime.now(UTC) - timedelta(days=30)
        assert manager.can_delete(DataCategory.TRADE_RECORDS, recent) is False

    def test_can_delete_old_data(self, manager: RetentionManager):
        old = datetime.now(UTC) - timedelta(days=SEBI_MIN_RETENTION_YEARS * 365 + 10)
        assert manager.can_delete(DataCategory.TRADE_RECORDS, old) is True


# ──────────────────────────────────────────────────
# block_deletion_if_required
# ──────────────────────────────────────────────────


class TestBlockDeletion:
    def test_blocks_deletion_of_recent_data(self, manager: RetentionManager):
        recent = datetime.now(UTC) - timedelta(days=30)
        result = manager.block_deletion_if_required(DataCategory.AUDIT_EVENTS, recent, "test audit record")
        assert result is False  # Deletion blocked

    def test_allows_deletion_of_old_data(self, manager: RetentionManager):
        old = datetime.now(UTC) - timedelta(days=SEBI_MIN_RETENTION_YEARS * 365 + 10)
        result = manager.block_deletion_if_required(DataCategory.AUDIT_EVENTS, old, "old audit record")
        assert result is True  # Deletion allowed

    def test_blocked_deletion_logged(self, manager: RetentionManager):
        recent = datetime.now(UTC) - timedelta(days=30)
        manager.block_deletion_if_required(DataCategory.AUDIT_EVENTS, recent, "test")
        log = manager.get_action_log(action=RetentionAction.DELETION_BLOCKED)
        assert len(log) >= 1
        assert log[0]["action"] == "DELETION_BLOCKED"


# ──────────────────────────────────────────────────
# Archival
# ──────────────────────────────────────────────────


class TestArchival:
    def test_should_archive_old_data(self, manager: RetentionManager):
        old = datetime.now(UTC) - timedelta(days=3 * 365)  # 3 years old (> 2yr archive threshold)
        assert manager.should_archive(DataCategory.TRADE_RECORDS, old) is True

    def test_should_not_archive_recent_data(self, manager: RetentionManager):
        recent = datetime.now(UTC) - timedelta(days=30)
        assert manager.should_archive(DataCategory.TRADE_RECORDS, recent) is False

    def test_archive_json_data_compressed(self, manager: RetentionManager):
        old = datetime.now(UTC) - timedelta(days=3 * 365)
        data = {"key": "value", "count": 42}
        path = manager.archive_data(DataCategory.TRADE_RECORDS, old, data)
        assert path is not None
        assert os.path.isfile(path)
        assert path.endswith(".gz")

        # Verify the compressed content
        with gzip.open(path, "rb") as f:
            content = json.loads(f.read())
        assert content["key"] == "value"

    def test_archive_json_data_uncompressed(self, tmp_archive):
        policy = RetentionPolicy(
            category=DataCategory.MARKET_DATA,
            compress_archive=False,
        )
        mgr = RetentionManager(cold_storage_path=str(tmp_archive), policies=[policy])
        old = datetime.now(UTC) - timedelta(days=3 * 365)
        data = {"key": "value"}
        path = mgr.archive_data(DataCategory.MARKET_DATA, old, data)
        assert path is not None
        assert not path.endswith(".gz")
        with open(path) as f:
            content = json.load(f)
        assert content["key"] == "value"

    def test_archive_file(self, manager: RetentionManager, tmp_path):
        # Create a source file
        src_file = tmp_path / "test_data.json"
        src_file.write_text('{"data": "test"}')

        old = datetime.now(UTC) - timedelta(days=3 * 365)
        path = manager.archive_data(DataCategory.ORDER_RECORDS, old, None, source_path=str(src_file))
        assert path is not None
        assert os.path.isfile(path)

    def test_archive_skipped_if_not_old_enough(self, manager: RetentionManager):
        recent = datetime.now(UTC) - timedelta(days=30)
        path = manager.archive_data(DataCategory.TRADE_RECORDS, recent, {"key": "val"})
        assert path is None

    def test_archive_logged(self, manager: RetentionManager):
        old = datetime.now(UTC) - timedelta(days=3 * 365)
        manager.archive_data(DataCategory.TRADE_RECORDS, old, {"test": True})
        log = manager.get_action_log(action=RetentionAction.ARCHIVED)
        assert len(log) >= 1


# ──────────────────────────────────────────────────
# Purging
# ──────────────────────────────────────────────────


class TestPurging:
    def test_should_purge_very_old_data(self, manager: RetentionManager):
        very_old = datetime.now(UTC) - timedelta(days=DEFAULT_MAX_RETENTION_YEARS * 365 + 10)
        assert manager.should_purge(DataCategory.TRADE_RECORDS, very_old) is True

    def test_should_not_purge_recent_data(self, manager: RetentionManager):
        recent = datetime.now(UTC) - timedelta(days=30)
        assert manager.should_purge(DataCategory.TRADE_RECORDS, recent) is False

    def test_purge_if_expired_removes_file(self, manager: RetentionManager, tmp_path):
        f = tmp_path / "old_data.json"
        f.write_text("{}")
        very_old = datetime.now(UTC) - timedelta(days=DEFAULT_MAX_RETENTION_YEARS * 365 + 10)
        result = manager.purge_if_expired(DataCategory.TRADE_RECORDS, very_old, file_path=str(f))
        assert result is True
        assert not f.exists()

    def test_purge_blocked_within_retention(self, manager: RetentionManager):
        recent = datetime.now(UTC) - timedelta(days=30)
        result = manager.purge_if_expired(DataCategory.TRADE_RECORDS, recent)
        assert result is False

    def test_purge_logged(self, manager: RetentionManager, tmp_path):
        f = tmp_path / "purge_test.json"
        f.write_text("{}")
        very_old = datetime.now(UTC) - timedelta(days=DEFAULT_MAX_RETENTION_YEARS * 365 + 10)
        manager.purge_if_expired(
            DataCategory.TRADE_RECORDS, very_old, file_path=str(f), record_description="test purge"
        )
        log = manager.get_action_log(action=RetentionAction.PURGED)
        assert len(log) >= 1


# ──────────────────────────────────────────────────
# Cleanup cycle
# ──────────────────────────────────────────────────


class TestCleanupCycle:
    def test_cleanup_nonexistent_dir(self, manager: RetentionManager):
        counts = manager.run_cleanup_cycle("/nonexistent/path", DataCategory.TRADE_RECORDS)
        assert counts == {"archived": 0, "purged": 0, "retained": 0, "errors": 0}

    def test_cleanup_retains_recent_files(self, manager: RetentionManager, tmp_path):
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "recent.json").write_text("{}")
        counts = manager.run_cleanup_cycle(str(data_dir), DataCategory.TRADE_RECORDS)
        assert counts["retained"] == 1
        assert counts["archived"] == 0
        assert counts["purged"] == 0


# ──────────────────────────────────────────────────
# get_policies / get_retention_summary
# ──────────────────────────────────────────────────


class TestPoliciesAndSummary:
    def test_get_policies(self, manager: RetentionManager):
        policies = manager.get_policies()
        assert len(policies) == len(DataCategory)
        for p in policies:
            assert p["sebi_compliant"] is True

    def test_retention_summary(self, manager: RetentionManager):
        summary = manager.get_retention_summary()
        assert summary["sebi_min_retention_years"] == SEBI_MIN_RETENTION_YEARS
        assert "policies" in summary
        assert "total_actions" in summary
        assert "actions_by_type" in summary

    def test_action_log_filtered_by_category(self, manager: RetentionManager):
        recent = datetime.now(UTC) - timedelta(days=30)
        manager.block_deletion_if_required(DataCategory.AUDIT_EVENTS, recent, "audit")
        manager.block_deletion_if_required(DataCategory.TRADE_RECORDS, recent, "trade")

        audit_log = manager.get_action_log(category=DataCategory.AUDIT_EVENTS)
        trade_log = manager.get_action_log(category=DataCategory.TRADE_RECORDS)
        assert len(audit_log) == 1
        assert len(trade_log) == 1


# ──────────────────────────────────────────────────
# Thread safety
# ──────────────────────────────────────────────────


class TestThreadSafety:
    def test_concurrent_operations(self, manager: RetentionManager):
        errors = []
        recent = datetime.now(UTC) - timedelta(days=30)

        def block_deletions(count: int):
            try:
                for i in range(count):
                    manager.block_deletion_if_required(DataCategory.TRADE_RECORDS, recent, f"record_{i}")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=block_deletions, args=(50,)) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Thread errors: {errors}"
        log = manager.get_action_log(action=RetentionAction.DELETION_BLOCKED)
        assert len(log) > 0
