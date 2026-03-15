#!/usr/bin/env python3
import json
import sys
from pathlib import Path

# Add project root to sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agents.skills.registry import all_skills

def main():
    print(f"--- AURA Architectural Audit ---")
    print(f"Project root: {REPO_ROOT}")
    
    skills = all_skills()
    analyzer = skills.get("structural_analyzer")
    
    if not analyzer:
        print("Error: structural_analyzer skill not found in registry.")
        sys.exit(1)
        
    print("Running structural_analyzer...")
    result = analyzer.run({"project_root": str(REPO_ROOT)})
    
    if "error" in result:
        print(f"Skill error: {result['error']}")
        sys.exit(1)
        
    print("\n--- CIRCULAR DEPENDENCIES ---")
    cycles = result.get("circular_dependencies", [])
    if not cycles:
        print("None detected.")
    else:
        for i, cycle in enumerate(cycles[:5], 1):
            print(f"{i}. {' -> '.join(cycle)}")
            
    print("\n--- ARCHITECTURAL BOTTLENECKS (Top Centrality) ---")
    bottlenecks = result.get("bottlenecks", [])
    if not bottlenecks:
        print("None detected.")
    else:
        for i, b in enumerate(bottlenecks[:5], 1):
            print(f"{i}. {b['file']} (Score: {b['centrality_score']:.4f})")
            
    print("\n--- ARCHITECTURAL HOTSPOTS (Critical Risk) ---")
    hotspots = result.get("hotspots", [])
    if not hotspots:
        print("None detected.")
    else:
        for h in hotspots:
            print(f"- {h['file']}: Risk={h['risk_level']}, Complexity={h['max_complexity']}, Centrality={h['centrality']:.4f}")
            
    print(f"\nSummary: {result.get('summary', 'Audit complete.')}")

if __name__ == "__main__":
    main()
