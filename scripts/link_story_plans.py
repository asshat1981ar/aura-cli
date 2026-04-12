#!/usr/bin/env python3
"""Validate that plans_link fields in ready/ stories point to existing plan files.

Exit codes:
  0 — all links valid (or no links found)
  1 — broken links found

Usage:
    python3 scripts/link_story_plans.py [--verbose]
"""

import argparse
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    print("pyyaml required: pip install pyyaml", file=sys.stderr)
    sys.exit(2)

REPO_ROOT = Path(__file__).parent.parent
READY_DIR = REPO_ROOT / ".aura_forge" / "backlog" / "ready"
PLANS_DIR = REPO_ROOT / "plans"


def check_links(verbose: bool = False) -> list[dict]:
    broken = []
    for story_path in sorted(READY_DIR.glob("*.yaml")):
        try:
            data = yaml.safe_load(story_path.read_text())
        except Exception as e:
            if verbose:
                print(f"  WARN  {story_path.name}: parse error — {e}")
            continue
        link = data.get("plans_link")
        if not link:
            if verbose:
                print(f"  SKIP  {story_path.name}: no plans_link")
            continue
        plan_path = REPO_ROOT / link
        if plan_path.exists():
            if verbose:
                print(f"  OK    {story_path.name}: {link}")
        else:
            broken.append({"story": story_path.name, "link": link})
            print(f"  BROKEN {story_path.name}: plans_link → '{link}' not found")
    return broken


def main():
    ap = argparse.ArgumentParser(description="Validate story plans_link fields")
    ap.add_argument("--verbose", "-v", action="store_true")
    args = ap.parse_args()

    print(f"Checking plans_link in {READY_DIR.relative_to(REPO_ROOT)} ...")
    broken = check_links(verbose=args.verbose)
    if broken:
        print(f"\n{len(broken)} broken link(s) found.")
        sys.exit(1)
    else:
        print("All plans_link references valid.")


if __name__ == "__main__":
    main()
