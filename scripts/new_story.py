#!/usr/bin/env python3
"""Scaffold a new AURA Forge story into inbox/.

Usage:
    python3 scripts/new_story.py --title "My idea" [--quick] [--type feature]
"""

import argparse
import datetime
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
INBOX = REPO_ROOT / ".aura_forge" / "backlog" / "inbox"
TEMPLATES = REPO_ROOT / ".aura_forge" / "templates"


def next_id() -> str:
    """Find the next available AF-STORY-XXXX id."""
    all_ids = []
    for p in REPO_ROOT.glob(".aura_forge/backlog/**/*.yaml"):
        m = re.search(r"AF-STORY-(\d{4})", p.stem)
        if m:
            all_ids.append(int(m.group(1)))
    nxt = max(all_ids, default=10) + 1
    return f"AF-STORY-{nxt:04d}"


def slug(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")


def main():
    ap = argparse.ArgumentParser(description="Scaffold a new Forge story")
    ap.add_argument("--title", required=True, help="Story title")
    ap.add_argument("--quick", action="store_true", help="Use lightweight template")
    ap.add_argument("--type", default="feature", choices=["feature", "bug", "improvement", "self_evolution"])
    args = ap.parse_args()

    story_id = next_id()
    template_file = TEMPLATES / ("story_quick.yaml" if args.quick else "story.md")
    if not template_file.exists():
        print(f"Template not found: {template_file}", file=sys.stderr)
        sys.exit(1)

    content = template_file.read_text()
    content = content.replace("AF-STORY-XXXX", story_id)
    content = content.replace('title: ""', f'title: "{args.title}"')
    content = content.replace('type: "feature"', f'type: "{args.type}"')

    out = INBOX / f"{story_id}.yaml"
    out.write_text(content)
    print(f"Created: {out}")
    print(f"  ID:    {story_id}")
    print(f"  Title: {args.title}")


if __name__ == "__main__":
    main()
