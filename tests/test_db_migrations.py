"""Tests for the database migration system."""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from core.db_migrations import (
    AUTH_MIGRATIONS,
    BRAIN_MIGRATIONS,
    Migration,
    MigrationRunner,
    migrate_auth_db,
    migrate_brain_db,
)


@pytest.fixture
def tmp_db(tmp_path: Path) -> Path:
    return tmp_path / "test.db"


# --- AUTH_MIGRATIONS ---

def test_auth_migrations_apply_all_three_versions(tmp_db):
    applied = migrate_auth_db(tmp_db)
    assert applied == ["001", "002", "003"]


def test_auth_migrations_creates_expected_tables(tmp_db):
    migrate_auth_db(tmp_db)
    conn = sqlite3.connect(tmp_db)
    tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    conn.close()
    assert "jti_revocations" in tables
    assert "auth_audit_log" in tables


def test_auth_migrations_idempotent(tmp_db):
    first = migrate_auth_db(tmp_db)
    second = migrate_auth_db(tmp_db)
    assert first == ["001", "002", "003"]
    assert second == []  # nothing new applied


# --- BRAIN_MIGRATIONS ---

def test_brain_migrations_apply_both_versions(tmp_db):
    applied = migrate_brain_db(tmp_db)
    assert applied == ["001", "002"]


def test_brain_migrations_creates_memories_table(tmp_db):
    migrate_brain_db(tmp_db)
    conn = sqlite3.connect(tmp_db)
    tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    conn.close()
    assert "memories" in tables
    assert "memory_edges" in tables


def test_brain_migrations_idempotent(tmp_db):
    first = migrate_brain_db(tmp_db)
    second = migrate_brain_db(tmp_db)
    assert first == ["001", "002"]
    assert second == []


# --- Checksum mismatch ---

def test_checksum_mismatch_raises_runtime_error(tmp_db):
    runner = MigrationRunner(tmp_db, AUTH_MIGRATIONS)
    runner.run()

    # Tamper with stored checksum
    conn = sqlite3.connect(tmp_db)
    conn.execute(
        "UPDATE _schema_migrations SET checksum = 'deadbeef00000000' WHERE version = '001'"
    )
    conn.commit()
    conn.close()

    with pytest.raises(RuntimeError, match="checksum mismatch"):
        runner.run()


# --- Rollback ---

def test_rollback_reversible_migrations(tmp_db):
    runner = MigrationRunner(tmp_db, AUTH_MIGRATIONS)
    runner.run()

    rolled = runner.rollback("001")
    # 003 and 002 are newer than 001 and both have down_sql
    assert "003" in rolled
    assert "002" in rolled

    conn = sqlite3.connect(tmp_db)
    remaining = {
        row[0]
        for row in conn.execute("SELECT version FROM _schema_migrations")
    }
    conn.close()
    assert remaining == {"001"}


def test_rollback_irreversible_raises(tmp_db):
    runner = MigrationRunner(tmp_db, BRAIN_MIGRATIONS)
    runner.run()

    with pytest.raises(RuntimeError, match="irreversible"):
        runner.rollback("000")


# --- Schema migrations tracking table ---

def test_schema_migrations_table_records_applied(tmp_db):
    migrate_auth_db(tmp_db)
    conn = sqlite3.connect(tmp_db)
    rows = conn.execute(
        "SELECT version, checksum FROM _schema_migrations ORDER BY version"
    ).fetchall()
    conn.close()
    versions = [r[0] for r in rows]
    assert versions == ["001", "002", "003"]
    # checksums are 16-char hex strings
    for _, checksum in rows:
        assert len(checksum) == 16
