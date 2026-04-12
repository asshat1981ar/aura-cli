"""
Database migration system for AURA CLI.
Ordered, idempotent schema migrations for all SQLite databases.
"""

from __future__ import annotations

import hashlib
import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class Migration:
    version: str
    description: str
    up_sql: str
    down_sql: str = ""
    checksum: str = ""

    def __post_init__(self):
        if not self.checksum:
            self.checksum = hashlib.sha256(self.up_sql.encode()).hexdigest()[:16]


AUTH_MIGRATIONS: list[Migration] = [
    Migration(
        version="001",
        description="Initial JTI revocations schema",
        up_sql="""
CREATE TABLE IF NOT EXISTS jti_revocations (
    jti TEXT PRIMARY KEY,
    user_id TEXT,
    revoked_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP NOT NULL,
    reason TEXT DEFAULT 'logout'
);
CREATE INDEX IF NOT EXISTS idx_jti_expires ON jti_revocations(expires_at);
CREATE INDEX IF NOT EXISTS idx_jti_user ON jti_revocations(user_id);
""",
        down_sql="DROP TABLE IF EXISTS jti_revocations;",
    ),
    Migration(
        version="002",
        description="Add auth audit log table",
        up_sql="""
CREATE TABLE IF NOT EXISTS auth_audit_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type TEXT NOT NULL,
    user_id TEXT,
    ip_address TEXT,
    user_agent TEXT,
    occurred_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata TEXT
);
CREATE INDEX IF NOT EXISTS idx_audit_user ON auth_audit_log(user_id, occurred_at DESC);
CREATE INDEX IF NOT EXISTS idx_audit_event ON auth_audit_log(event_type, occurred_at DESC);
""",
        down_sql="DROP TABLE IF EXISTS auth_audit_log;",
    ),
    Migration(
        version="003",
        description="JTI revocation cleanup trigger",
        up_sql="""
CREATE TRIGGER IF NOT EXISTS cleanup_expired_jtis
AFTER INSERT ON jti_revocations
BEGIN
    DELETE FROM jti_revocations WHERE expires_at < datetime('now');
END;
""",
        down_sql="DROP TRIGGER IF EXISTS cleanup_expired_jtis;",
    ),
]

BRAIN_MIGRATIONS: list[Migration] = [
    Migration(
        version="001",
        description="Initial memory graph schema",
        up_sql="""
CREATE TABLE IF NOT EXISTS memories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    key TEXT UNIQUE NOT NULL,
    value TEXT NOT NULL,
    embedding BLOB,
    importance REAL DEFAULT 0.5,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    access_count INTEGER DEFAULT 0,
    tags TEXT
);
CREATE INDEX IF NOT EXISTS idx_memory_key ON memories(key);
CREATE INDEX IF NOT EXISTS idx_memory_importance ON memories(importance DESC);
CREATE INDEX IF NOT EXISTS idx_memory_accessed ON memories(last_accessed DESC);
CREATE TABLE IF NOT EXISTS memory_edges (
    from_id INTEGER NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    to_id INTEGER NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    relation TEXT DEFAULT 'related',
    weight REAL DEFAULT 1.0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (from_id, to_id, relation)
);
""",
    ),
    Migration(
        version="002",
        description="FTS5 full-text search on memory values",
        up_sql="""
CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts
    USING fts5(key, value, content='memories', content_rowid='id');
CREATE TRIGGER IF NOT EXISTS memories_fts_insert
AFTER INSERT ON memories BEGIN
    INSERT INTO memories_fts(rowid, key, value) VALUES (new.id, new.key, new.value);
END;
CREATE TRIGGER IF NOT EXISTS memories_fts_delete
AFTER DELETE ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, key, value)
    VALUES ('delete', old.id, old.key, old.value);
END;
""",
    ),
]


class MigrationRunner:
    """Applies ordered migrations to a SQLite database."""

    _MIGRATIONS_TABLE = """
CREATE TABLE IF NOT EXISTS _schema_migrations (
    version TEXT PRIMARY KEY,
    description TEXT,
    checksum TEXT,
    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

    def __init__(self, db_path: Path, migrations: list[Migration]):
        self._db_path = db_path
        self._migrations = sorted(migrations, key=lambda m: m.version)

    def run(self) -> list[str]:
        """Apply all pending migrations. Returns applied version list."""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        applied: list[str] = []

        with sqlite3.connect(self._db_path) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            conn.executescript(self._MIGRATIONS_TABLE)
            conn.commit()

            already_applied = {row[0]: row[2] for row in conn.execute("SELECT version, description, checksum FROM _schema_migrations")}

            for migration in self._migrations:
                if migration.version in already_applied:
                    stored = already_applied[migration.version]
                    if stored != migration.checksum:
                        raise RuntimeError(f"Migration {migration.version} checksum mismatch! Expected {migration.checksum}, found {stored}.")
                    continue

                logger.info(
                    "Applying migration %s: %s",
                    migration.version,
                    migration.description,
                )
                try:
                    conn.executescript(migration.up_sql)
                    conn.execute(
                        "INSERT INTO _schema_migrations (version, description, checksum) VALUES (?, ?, ?)",
                        (migration.version, migration.description, migration.checksum),
                    )
                    conn.commit()
                    applied.append(migration.version)
                except Exception as exc:
                    conn.rollback()
                    raise RuntimeError(f"Migration {migration.version} failed: {exc}") from exc

        if applied:
            logger.info("Applied %d migrations: %s", len(applied), ", ".join(applied))
        return applied

    def rollback(self, target_version: str) -> list[str]:
        """Rollback migrations newer than target_version."""
        rolled_back: list[str] = []
        with sqlite3.connect(self._db_path) as conn:
            versions = [row[0] for row in conn.execute("SELECT version FROM _schema_migrations ORDER BY version DESC")]
            for version in versions:
                if version <= target_version:
                    break
                migration = next((m for m in self._migrations if m.version == version), None)
                if not migration:
                    raise RuntimeError(f"Migration {version} not in registry")
                if not migration.down_sql:
                    raise RuntimeError(f"Migration {version} is irreversible")
                conn.executescript(migration.down_sql)
                conn.execute("DELETE FROM _schema_migrations WHERE version = ?", (version,))
                conn.commit()
                rolled_back.append(version)
        return rolled_back


def migrate_auth_db(db_path: Path) -> list[str]:
    return MigrationRunner(db_path, AUTH_MIGRATIONS).run()


def migrate_brain_db(db_path: Path) -> list[str]:
    return MigrationRunner(db_path, BRAIN_MIGRATIONS).run()
