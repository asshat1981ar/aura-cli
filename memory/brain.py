import sqlite3
import networkx as nx
import os # Import os
from pathlib import Path
from typing import List
from textblob import TextBlob # Import TextBlob
import json # Added this line
from core.logging_utils import log_json # Import log_json

class Brain:

    def __init__(self):
        # Construct absolute path for the database file
        db_file_path = Path(__file__).parent / "brain.db"
        self.db = sqlite3.connect(str(db_file_path)) # Connect using absolute path
        self.graph = nx.Graph()
        self._init_db()
        
    def _init_db(self):
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
        self.db.commit()

    def remember(self, data): # Changed parameter name from 'text' to 'data' for clarity
        # If data is a dictionary, serialize it to JSON
        if isinstance(data, dict):
            content_to_store = json.dumps(data)
        elif isinstance(data, (str, int, float)): # Also handle other basic types if needed
            content_to_store = str(data)
        else:
            # Fallback for unexpected types, or raise an error
            log_json("WARN", "brain_unsupported_data_type", details={"data_type": str(type(data)), "data_snippet": str(data)[:100]})
            content_to_store = str(data)

        # Store in textual memory
        cursor = self.db.execute(
            "INSERT INTO memory(content) VALUES (?)",
            (content_to_store,)
        )
        self.db.commit()

    def recall_all(self):
        rows = self.db.execute("SELECT content FROM memory").fetchall()
        return [r[0] for r in rows]

    def add_weakness(self, weakness_description: str):
        self.db.execute("INSERT INTO weaknesses(description) VALUES (?)", (weakness_description,))
        self.db.commit()

    def recall_weaknesses(self) -> list[str]:
        rows = self.db.execute("SELECT description FROM weaknesses ORDER BY timestamp DESC").fetchall()
        return [r[0] for r in rows]

    def reflect(self):
        memory_entries = self.recall_all()
        weakness_entries = self.recall_weaknesses()
        return f"System has {len(memory_entries)} memory entries and {len(weakness_entries)} identified weaknesses."

    def relate(self, a: str, b: str):
        self.graph.add_edge(a, b)

    def analyze_critique_for_weaknesses(self, critique: str):
        # Using TextBlob for sentiment analysis and noun phrase extraction
        blob = TextBlob(critique)

        found_weaknesses = False
        for sentence in blob.sentences:
            # Check for negative sentiment
            if sentence.sentiment.polarity < -0.1: # Threshold for negative sentiment
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