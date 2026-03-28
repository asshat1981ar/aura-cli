import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
import json  # Added this line
import time
from typing import List, Optional, Any
from core.logging_utils import log_json  # Import log_json

# textblob is only needed by analyze_critique_for_weaknesses — lazy import to avoid 1.3s startup cost
_textblob_loaded = False
TextBlob = None


def _ensure_textblob():
    global TextBlob, _textblob_loaded
    if not _textblob_loaded:
        from textblob import TextBlob as _TB  # noqa: F401

        TextBlob = _TB
        _textblob_loaded = True


# networkx is only needed by relate() and rarely used graph operations — lazy import to avoid 885ms startup
_nx = None


def _ensure_nx():
    global _nx
    if _nx is None:
        import networkx as _networkx  # noqa: F401

        _nx = _networkx
    return _nx


class Brain:
    # Current schema version. Bump when adding columns / tables.
    SCHEMA_VERSION = 4

    def __init__(self, db_path: Optional[str] = None):
        # Construct absolute path for the database file
        db_file_path = Path(db_path) if db_path else Path(__file__).parent / "brain.db"
        self._db_path = db_file_path
        self.db = sqlite3.connect(str(db_file_path), check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        self._graph = None  # lazy — created on first relate() call
        self._recall_cache: dict = {}  # {query_key: (result, timestamp)}
        self._cache_ttl: float = 5.0  # seconds — invalidated on remember()
        self._init_db()
        self._migrate()

    @contextmanager
    def _db_lock(self):
        """Context manager that serialises database operations across threads."""
        with self._lock:
            yield

    @property
    def graph(self):
        """Lazily initialise the networkx graph on first access."""
        if self._graph is None:
            self._graph = _ensure_nx().Graph()
        return self._graph

    def _init_db(self):
        # WAL mode: concurrent reads don't block writes; NORMAL sync is safe and ~3x faster
        with self._lock:
            self.db.execute("PRAGMA journal_mode=WAL")
            self.db.execute("PRAGMA synchronous=NORMAL")
            self.db.execute("PRAGMA cache_size=10000")
            self.db.execute("""
            CREATE TABLE IF NOT EXISTS schema_version(
                version INTEGER PRIMARY KEY
            )
            """)
            self.db.execute("""
            CREATE TABLE IF NOT EXISTS memory(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT
            )
            """)
            self.db.execute("""
            CREATE TABLE IF NOT EXISTS weaknesses(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                description TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """)
            self.db.execute("""
            CREATE TABLE IF NOT EXISTS vector_store_data(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT,
                embedding BLOB
            )
            """)
            self.db.execute("""
            CREATE TABLE IF NOT EXISTS kv_store(
                key TEXT PRIMARY KEY,
                value TEXT
            )
            """)
            # Index on memory.id enables fast ORDER BY id DESC LIMIT N queries (200x speedup
            # vs full-table scan on 30k+ row databases)
            self.db.execute("CREATE INDEX IF NOT EXISTS idx_memory_id ON memory(id)")
            self.db.commit()

    def _get_schema_version(self) -> int:
        """Return stored schema version, or 0 if none recorded."""
        try:
            row = self.db.execute("SELECT version FROM schema_version").fetchone()
            return row[0] if row else 0
        except sqlite3.OperationalError:
            return 0

    def _set_schema_version(self, version: int) -> None:
        with self._lock:
            self.db.execute("DELETE FROM schema_version")
            self.db.execute("INSERT INTO schema_version(version) VALUES (?)", (version,))
            self.db.commit()

    def _migrate(self) -> None:
        """Apply any pending schema migrations idempotently.

        Migration log:
          v0 → v1: initial schema (memory, weaknesses, vector_store_data)
          v1 → v2: absorb rows from legacy brain_v2.db (same directory)
          v2 → v3: create kv_store table
        """
        current = self._get_schema_version()
        if current >= self.SCHEMA_VERSION:
            return

        # v0 → v1: tables already created by _init_db; just record version
        if current < 1:
            log_json("INFO", "brain_migration", details={"from": current, "to": 1})
            self._set_schema_version(1)
            current = 1

        # v1 → v2: migrate rows from brain_v2.db if it exists alongside brain.db
        if current < 2:
            legacy_path = self._db_path.parent / "brain_v2.db"
            if legacy_path.exists() and legacy_path != self._db_path:
                self._absorb_legacy_db(legacy_path)
            self._set_schema_version(2)
            log_json("INFO", "brain_migration", details={"from": 1, "to": 2})
            current = 2

        # v2 → v3: create kv_store table
        if current < 3:
            with self._lock:
                self.db.execute("""
                CREATE TABLE IF NOT EXISTS kv_store(
                    key TEXT PRIMARY KEY,
                    value TEXT
                )
                """)
                self.db.commit()
            self._set_schema_version(3)
            log_json("INFO", "brain_migration", details={"from": 2, "to": 3})
            current = 3

        # v3 → v4: add tag column to memory table for SADD session-scoped context
        if current < 4:
            with self._lock:
                try:
                    self.db.execute("ALTER TABLE memory ADD COLUMN tag TEXT DEFAULT NULL")
                except sqlite3.OperationalError:
                    pass  # Column already exists
                self.db.execute("CREATE INDEX IF NOT EXISTS idx_memory_tag ON memory(tag)")
                self.db.commit()
            self._set_schema_version(4)
            log_json("INFO", "brain_migration", details={"from": 3, "to": 4})

    def _absorb_legacy_db(self, legacy_path: Path) -> None:
        """Copy rows from *legacy_path* into this DB, skipping duplicates."""
        try:
            legacy = sqlite3.connect(str(legacy_path), check_same_thread=False)
            # memory rows
            rows = legacy.execute("SELECT content FROM memory").fetchall()
            with self._lock:
                existing = {r[0] for r in self.db.execute("SELECT content FROM memory").fetchall()}
                new_rows = [r for r in rows if r[0] not in existing]
                if new_rows:
                    self.db.executemany("INSERT INTO memory(content) VALUES (?)", new_rows)
                # weaknesses rows
                wrows = legacy.execute("SELECT description, timestamp FROM weaknesses").fetchall()
                self.db.executemany("INSERT OR IGNORE INTO weaknesses(description, timestamp) VALUES (?, ?)", wrows)
                self.db.commit()
            legacy.close()
            log_json("INFO", "brain_legacy_absorbed", details={"path": str(legacy_path), "memory_rows": len(new_rows), "weakness_rows": len(wrows)})
        except Exception as exc:
            log_json("WARN", "brain_legacy_absorb_failed", details={"path": str(legacy_path), "error": str(exc)})

    def remember(self, data):  # Changed parameter name from 'text' to 'data' for clarity
        # If data is a dictionary, serialize it to JSON
        if isinstance(data, dict):
            content_to_store = json.dumps(data)
        elif isinstance(data, (str, int, float)):  # Also handle other basic types if needed
            content_to_store = str(data)
        else:
            # Fallback for unexpected types, or raise an error
            log_json("WARN", "brain_unsupported_data_type", details={"data_type": str(type(data)), "data_snippet": str(data)[:100]})
            content_to_store = str(data)

        # Store in textual memory
        with self._lock:
            self.db.execute("INSERT INTO memory(content) VALUES (?)", (content_to_store,))
            self.db.commit()
        # Invalidate recall cache on write
        self._recall_cache.clear()

    def remember_tagged(self, text: str, tag: str) -> None:
        """Store a tagged memory entry for session-scoped context (e.g. SADD)."""
        content = str(text)
        try:
            with self._lock:
                self.db.execute(
                    "INSERT INTO memory(content, tag) VALUES (?, ?)",
                    (content, tag),
                )
                self.db.commit()
            self._recall_cache.clear()
        except sqlite3.OperationalError:
            # tag column may not exist yet; fall back to untagged
            self.remember(f"[{tag}] {content}")

    def recall_tagged(self, tag: str, limit: int = 50) -> list:
        """Retrieve memories with a specific tag."""
        try:
            rows = self.db.execute(
                "SELECT content FROM memory WHERE tag = ? ORDER BY id DESC LIMIT ?",
                (tag, limit),
            ).fetchall()
            return [row[0] for row in rows]
        except sqlite3.OperationalError:
            return []

    def forget_tagged(self, tag: str) -> int:
        """Delete all memories with the given tag. Returns count deleted."""
        try:
            with self._lock:
                cursor = self.db.execute("DELETE FROM memory WHERE tag = ?", (tag,))
                self.db.commit()
            self._recall_cache.clear()
            return cursor.rowcount
        except sqlite3.OperationalError:
            return 0

    def set(self, key: str, value: Any) -> None:
        """Set a persistent key-value pair. value will be JSON serialized."""
        if not isinstance(value, str):
            value = json.dumps(value)
        with self._lock:
            self.db.execute("INSERT OR REPLACE INTO kv_store(key, value) VALUES (?, ?)", (key, value))
            self.db.commit()

    def get(self, key: str, default: Any = None) -> Any:
        """Get a persistent key-value pair. Attempts to JSON deserialize the result."""
        row = self.db.execute("SELECT value FROM kv_store WHERE key = ?", (key,)).fetchone()
        if not row:
            return default
        raw = row[0]
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return raw

    def recall_all(self):
        key = "recall_all"
        cached = self._recall_cache.get(key)
        if cached and (time.time() - cached[1]) < self._cache_ttl:
            return cached[0]
        rows = self.db.execute("SELECT content FROM memory").fetchall()
        result = [r[0] for r in rows]
        self._recall_cache[key] = (result, time.time())
        return result

    @staticmethod
    def compress_to_budget(entries: List[str], max_tokens: int) -> List[str]:
        """Return a subset of entries that fits within max_tokens (4 chars ≈ 1 token).

        Keeps the most recent entries (tail of the list) first and discards
        older ones until the budget is satisfied.
        """
        max_chars = max_tokens * 4
        result: List[str] = []
        used = 0
        for entry in reversed(entries):
            cost = len(entry) + 1  # +1 for separator
            if used + cost > max_chars:
                break
            result.append(entry)
            used += cost
        result.reverse()
        return result

    def recall_recent(self, limit: int = 100) -> List[str]:
        """Return the *limit* most recent memory entries, newest last.

        Uses ``ORDER BY id DESC LIMIT N`` — O(log N) via the ``idx_memory_id``
        index instead of a full-table scan.  On a 30k-entry database this
        reduces latency from ~57ms to ~0.3ms (200x speedup).
        """
        key = f"recall_recent_{limit}"
        cached = self._recall_cache.get(key)
        if cached and (time.time() - cached[1]) < self._cache_ttl:
            return cached[0]
        rows = self.db.execute("SELECT content FROM memory ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        result = [r[0] for r in reversed(rows)]
        self._recall_cache[key] = (result, time.time())
        return result

    def recall_with_budget(self, max_tokens: int = 4000, tier: Optional[str] = None) -> List[str]:
        """Retrieve memories truncated to fit within max_tokens.

        Prioritises the most recent entries. Uses a direct SQL query with
        a computed row limit to avoid loading the full table (was 57ms on
        30k entries; now ~0.3ms via index scan).

        The *tier* argument is reserved for future filtered recall and is
        currently unused.
        """
        # Estimate max rows we could possibly need: budget / min entry size (1 char)
        # Using 4 chars per token as the approximation in compress_to_budget.
        max_chars = max_tokens * 4
        # Fetch enough recent rows to fill the budget; double for safety margin
        est_limit = min(max_chars // 10 + 200, 5000)
        rows = self.db.execute("SELECT content FROM memory ORDER BY id DESC LIMIT ?", (est_limit,)).fetchall()
        entries = [r[0] for r in reversed(rows)]
        return self.compress_to_budget(entries, max_tokens)

    def count_memories(self) -> int:
        """Return total memory entry count using a fast COUNT(*) query."""
        row = self.db.execute("SELECT COUNT(*) FROM memory").fetchone()
        return row[0] if row else 0

    def add_weakness(self, weakness_description: str):
        with self._lock:
            self.db.execute("INSERT INTO weaknesses(description) VALUES (?)", (weakness_description,))
            self.db.commit()

    def recall_weaknesses(self) -> list[str]:
        rows = self.db.execute("SELECT description FROM weaknesses ORDER BY timestamp DESC").fetchall()
        return [r[0] for r in rows]

    def reflect(self):
        memory_count = self.count_memories()
        weakness_entries = self.recall_weaknesses()
        return f"System has {memory_count} memory entries and {len(weakness_entries)} identified weaknesses."

    def relate(self, a: str, b: str):
        self.graph.add_edge(a, b)

    def analyze_critique_for_weaknesses(self, critique: str):
        # Using TextBlob for sentiment analysis and noun phrase extraction
        _ensure_textblob()
        blob = TextBlob(critique)

        found_weaknesses = False
        for sentence in blob.sentences:
            # Check for negative sentiment
            if sentence.sentiment.polarity < -0.1:  # Threshold for negative sentiment
                weakness_description = f"Negative sentiment detected: '{sentence.strip()}'."
                if sentence.noun_phrases:
                    weakness_description += f" Key phrases: {', '.join(sentence.noun_phrases)}."
                self.add_weakness(weakness_description)
                found_weaknesses = True
            # Also keep a keyword-based check as a fallback or for specific terms
            elif any(keyword in str(sentence).lower() for keyword in ["fail", "error", "bug", "issue", "inefficient", "suboptimal", "lacks", "missing", "weakness"]):
                self.add_weakness(f"Keyword-based weakness detected: '{sentence.strip()}'")
                found_weaknesses = True

        if not found_weaknesses:
            log_json("INFO", "brain_no_weaknesses_detected", details={"critique_snippet": critique[:100]})

    def set_vector_store(self, vector_store):
        """Attaches a VectorStore for semantic memory recall."""
        self.vector_store = vector_store
        log_json("INFO", "brain_vector_store_attached")

    # ── Weakness queue tracking ──────────────────────────────────────────────

    def _ensure_weakness_queue_table(self):
        with self._lock:
            self.db.execute("""
            CREATE TABLE IF NOT EXISTS weakness_queued(
                hash TEXT PRIMARY KEY,
                queued_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """)
            self.db.commit()

    def mark_weakness_queued(self, weakness_hash: str) -> None:
        """Record that a weakness has been turned into a goal (prevents re-queuing)."""
        self._ensure_weakness_queue_table()
        with self._lock:
            self.db.execute(
                "INSERT OR IGNORE INTO weakness_queued(hash) VALUES (?)",
                (weakness_hash,),
            )
            self.db.commit()

    def recall_queued_weakness_hashes(self) -> list[str]:
        """Return all weakness hashes that have already been queued as goals."""
        self._ensure_weakness_queue_table()
        rows = self.db.execute("SELECT hash FROM weakness_queued").fetchall()
        return [r[0] for r in rows]
