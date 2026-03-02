#!/usr/bin/env python3
"""
Final RSI Integration: Run AURA in a fully autonomous "evolve" mode for 50 cycles.
Logs results to logs/rsi_evolution_50.log.
"""
from __future__ import annotations

import os
import json
import time
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from aura_cli.cli_main import create_runtime
from core.logging_utils import log_json

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

def main():
    if load_dotenv:
        load_dotenv()
        
    # Setup environment for tests if needed
    if "AGENT_API_TOKEN" not in os.environ:
        os.environ["AGENT_API_TOKEN"] = "rsi-dev-token"
    if "AGENT_API_ENABLE_RUN" not in os.environ:
        os.environ["AGENT_API_ENABLE_RUN"] = "1"

    project_root = Path(__file__).resolve().parents[1]
    
    # Initialize runtime
    try:
        runtime = create_runtime(project_root)
        orchestrator = runtime["orchestrator"]
        log_json("INFO", "rsi_evolution_script_initialized")
    except Exception as e:
        print(f"Failed to initialize AURA runtime: {e}")
        return

    goal = "evolve and improve the AURA system via recursive self-improvement"
    max_total_cycles = 50
    log_file = project_root / "logs" / "rsi_evolution_50.log"
    
    print(f"Starting RSI Evolution Loop: {max_total_cycles} cycles")
    print(f"Logging to: {log_file}")

    for i in range(1, max_total_cycles + 1):
        start_time = time.time()
        print(f"\n>>> Cycle {i}/{max_total_cycles} starting...")
        
        try:
            # Run a single cycle
            # orchestrator.run_cycle already calls evolution_loop.on_cycle_complete(entry)
            # which might trigger the EvolutionLoop.run() if it hits N=20 or signal.
            result = orchestrator.run_cycle(goal)
            
            elapsed = time.time() - start_time
            status = result.get("phase_outputs", {}).get("verification", {}).get("status", "unknown")
            summary = result.get("cycle_summary", {}).get("summary", "No summary")
            
            print(f"--- Cycle {i} complete in {elapsed:.2f}s | Status: {status}")
            print(f"--- Summary: {summary}")
            
        except KeyboardInterrupt:
            print("\nEvolution loop interrupted by user.")
            break
        except Exception as e:
            log_json("ERROR", "rsi_evolution_cycle_failed", details={"cycle": i, "error": str(e)})
            print(f"!!! Cycle {i} failed: {e}")
            # Continue to next cycle unless it's a critical infrastructure failure
            time.sleep(5)

    print("\n>>> RSI Evolution run complete.")

if __name__ == "__main__":
    main()
