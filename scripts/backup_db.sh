#!/bin/bash
# ============================================================
# AlphaForge Trading Platform — SQLite Database Backup
#
# Backs up data/trades.db to data/backups/ with timestamps.
# Keeps the last 7 daily backups (auto-rotate).
#
# Usage:
#   ./scripts/backup_db.sh                  # Manual run
#   0 1 * * * /app/scripts/backup_db.sh     # Cron (daily at 1am)
#
# Environment variables (optional):
#   BACKUP_DIR       Override backup destination (default: data/backups)
#   BACKUP_RETAIN    Number of daily backups to keep (default: 7)
#   DB_PATH          Path to SQLite database (default: data/trades.db)
# ============================================================
set -euo pipefail

# ── Configuration ──
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

DB_PATH="${DB_PATH:-${PROJECT_ROOT}/data/trades.db}"
BACKUP_DIR="${BACKUP_DIR:-${PROJECT_ROOT}/data/backups}"
BACKUP_RETAIN="${BACKUP_RETAIN:-7}"
TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
BACKUP_FILE="${BACKUP_DIR}/trades_${TIMESTAMP}.db"
LOG_FILE="${BACKUP_DIR}/backup.log"

# ── Helpers ──
log_info()  { echo "[INFO]  $(date '+%Y-%m-%d %H:%M:%S') $*" | tee -a "${LOG_FILE}"; }
log_error() { echo "[ERROR] $(date '+%Y-%m-%d %H:%M:%S') $*" | tee -a "${LOG_FILE}" >&2; }

die() {
    log_error "$@"
    exit 1
}

# ── Step 1: Validate source database ──
if [ ! -f "${DB_PATH}" ]; then
    die "Source database not found: ${DB_PATH}"
fi

# ── Step 2: Create backup directory ──
mkdir -p "${BACKUP_DIR}"

log_info "=== Starting database backup ==="
log_info "Source:    ${DB_PATH}"
log_info "Target:    ${BACKUP_FILE}"
log_info "Retention: ${BACKUP_RETAIN} backups"

# ── Step 3: Perform backup using SQLite .backup command ──
# The .backup command creates a consistent snapshot even if the database
# is being written to. This is safer than a raw file copy which can result
# in a corrupted backup if a write is in progress.
DB_SIZE_BEFORE=$(stat -f%z "${DB_PATH}" 2>/dev/null || stat --printf="%s" "${DB_PATH}" 2>/dev/null || echo "unknown")
log_info "Source DB size: ${DB_SIZE_BEFORE} bytes"

if command -v sqlite3 &> /dev/null; then
    # Preferred: use SQLite's built-in online backup API
    if sqlite3 "${DB_PATH}" ".backup '${BACKUP_FILE}'" 2>>"${LOG_FILE}"; then
        log_info "Backup completed using sqlite3 .backup command."
    else
        die "sqlite3 .backup command failed. Check ${LOG_FILE} for details."
    fi
else
    # Fallback: use cp with WAL checkpoint first
    # This is less safe but works when sqlite3 CLI is not installed.
    log_info "sqlite3 CLI not found -- falling back to file copy."
    log_info "Warning: file copy may produce an inconsistent backup if writes are in progress."

    # Copy the main database file
    if cp "${DB_PATH}" "${BACKUP_FILE}"; then
        log_info "Database file copied."
    else
        die "Failed to copy database file."
    fi

    # Also copy WAL and SHM files if they exist (needed for consistency)
    for ext in "-wal" "-shm"; do
        if [ -f "${DB_PATH}${ext}" ]; then
            cp "${DB_PATH}${ext}" "${BACKUP_FILE}${ext}" 2>/dev/null || true
            log_info "Copied ${DB_PATH}${ext}"
        fi
    done
fi

# ── Step 4: Verify backup integrity ──
if command -v sqlite3 &> /dev/null; then
    INTEGRITY=$(sqlite3 "${BACKUP_FILE}" "PRAGMA integrity_check;" 2>/dev/null || echo "FAILED")
    if [ "${INTEGRITY}" = "ok" ]; then
        log_info "Backup integrity check: PASSED"
    else
        log_error "Backup integrity check: FAILED (${INTEGRITY})"
        rm -f "${BACKUP_FILE}"
        die "Backup file is corrupt. Removed ${BACKUP_FILE}."
    fi
else
    log_info "sqlite3 not available -- skipping integrity check."
fi

# ── Step 5: Compress backup ──
if command -v gzip &> /dev/null; then
    if gzip "${BACKUP_FILE}"; then
        BACKUP_FILE="${BACKUP_FILE}.gz"
        log_info "Backup compressed: ${BACKUP_FILE}"
    else
        log_info "Compression failed -- keeping uncompressed backup."
    fi
fi

# Report backup size
BACKUP_SIZE=$(stat -f%z "${BACKUP_FILE}" 2>/dev/null || stat --printf="%s" "${BACKUP_FILE}" 2>/dev/null || echo "unknown")
log_info "Backup size: ${BACKUP_SIZE} bytes"

# ── Step 6: Rotate old backups (keep last N) ──
# Count existing backups (both .db and .db.gz)
BACKUP_COUNT=$(find "${BACKUP_DIR}" -maxdepth 1 -name 'trades_*.db*' -type f | wc -l | tr -d ' ')
log_info "Total backups after this run: ${BACKUP_COUNT}"

if [ "${BACKUP_COUNT}" -gt "${BACKUP_RETAIN}" ]; then
    REMOVE_COUNT=$((BACKUP_COUNT - BACKUP_RETAIN))
    log_info "Rotating: removing ${REMOVE_COUNT} oldest backup(s) (retaining ${BACKUP_RETAIN})..."

    # List backups sorted by modification time (oldest first), remove the excess
    find "${BACKUP_DIR}" -maxdepth 1 -name 'trades_*.db*' -type f -print0 \
        | xargs -0 ls -1t \
        | tail -n "${REMOVE_COUNT}" \
        | while IFS= read -r old_backup; do
            rm -f "${old_backup}"
            log_info "Removed old backup: $(basename "${old_backup}")"
        done
fi

# ── Done ──
log_info "=== Backup completed successfully ==="
log_info "Backup file: ${BACKUP_FILE}"
echo ""
echo "Backup: ${BACKUP_FILE}"
