import logging
import sqlite3
import time
import threading

_logger = logging.getLogger(__name__)


class TelemetryAgent:
    """
    Thread-safe telemetry logger that persists agent performance metrics.
    Uses a thread-local connection to avoid 'SQLite objects created in a thread' errors.
    """

    def __init__(self, db_path="telemetry.db"):
        self.db_path = db_path
        self._local = threading.local()
        # Initialize schema once
        conn = sqlite3.connect(self.db_path)
        self._create_table(conn)
        conn.close()

    def _get_conn(self):
        if not hasattr(self._local, "conn"):
            self._local.conn = sqlite3.connect(self.db_path)
        return self._local.conn

    def _create_table(self, conn):
        create_table_sql = """CREATE TABLE IF NOT EXISTS telemetry (
                                id INTEGER PRIMARY KEY AUTOINCREMENT,
                                timestamp TEXT NOT NULL,
                                agent_name TEXT NOT NULL,
                                latency REAL NOT NULL,
                                token_count INTEGER NOT NULL
                            );"""
        try:
            cursor = conn.cursor()
            cursor.execute(create_table_sql)
            conn.commit()
        except sqlite3.Error as e:
            _logger.error("Error creating table: %s", e)

    def log(self, agent_name, latency, token_count):
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        insert_sql = """INSERT INTO telemetry (timestamp, agent_name, latency, token_count) VALUES (?, ?, ?, ?)"""
        try:
            conn = self._get_conn()
            cursor = conn.cursor()
            cursor.execute(insert_sql, (timestamp, agent_name, latency, token_count))
            conn.commit()
        except sqlite3.Error as e:
            _logger.error("Error logging telemetry data: %s", e)

    def close(self):
        if hasattr(self._local, "conn"):
            self._local.conn.close()
            del self._local.conn


# Example usage
if __name__ == "__main__":
    agent = TelemetryAgent()

    # Test multi-threaded logging
    def worker():
        agent.log("WorkerThread", 1.2, 50)

    threads = [threading.Thread(target=worker) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    _logger.info("Multi-threaded logging test complete.")
