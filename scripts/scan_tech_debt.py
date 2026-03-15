#!/usr/bin/env python3
"""
Run Autonomous Discovery to identify technical debt and unfinished implementations.
"""
import sys
import json
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.autonomous_discovery import AutonomousDiscovery

class ConsoleQueue:
    def add(self, item):
        print(f"[QUEUE] {item}")
    def batch_add(self, items):
        for item in items:
            print(f"[QUEUE] {item}")

class ConsoleMemory:
    def put(self, key, value):
        if key == "discovery_reports":
            print(f"[MEMORY] Saved discovery report with {value.get('findings_total', 0)} findings")
    def get(self, key):
        return {}

def main():
    print(f"Scanning {ROOT} for technical debt...")
    discovery = AutonomousDiscovery(ConsoleQueue(), ConsoleMemory(), project_root=str(ROOT))
    
    # Force run
    report = discovery.run_scan()
    
    findings = report.get("findings_total", 0)
    new_goals = report.get("new_goals", 0)
    
    print(f"\nScan Complete.")
    print(f"Total Findings: {findings}")
    print(f"New Goals Queued: {new_goals}")
    
    if "goals" in report:
        print("\nTop Priority Goals:")
        for goal in report["goals"]:
            print(f"- {goal}")

if __name__ == "__main__":
    main()
