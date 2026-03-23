import sqlite3

from agents.telemetry_agent import TelemetryAgent


def test_telemetry_agent_persists_log_entry(tmp_path):
    db_path = tmp_path / "telemetry.db"
    agent = TelemetryAgent(db_path=str(db_path))
    try:
        agent.log("planner", 1.25, 42)
        agent.close()

        conn = sqlite3.connect(db_path)
        try:
            row = conn.execute(
                "SELECT agent_name, latency, token_count FROM telemetry ORDER BY id DESC LIMIT 1"
            ).fetchone()
        finally:
            conn.close()
    finally:
        agent.close()

    assert row == ("planner", 1.25, 42)
