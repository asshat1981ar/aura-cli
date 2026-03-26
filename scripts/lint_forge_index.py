#!/usr/bin/env python3
"""Lint the AURA Forge backlog index for consistency.

Checks:
  1. Every story file in .aura_forge/backlog/<lane>/ appears in indexes/backlog_index.yaml
  2. Every story ID listed in the index has a corresponding file on disk
  3. Each story file appears in exactly one by_status bucket
  4. Each story file appears in exactly one by_lane bucket

Exit codes:
  0 — all checks pass
  1 — one or more inconsistencies found
"""

import re
import sys
from pathlib import Path

import yaml  # requires PyYAML (already in requirements.txt via aura deps)

FORGE_ROOT = Path(__file__).parent.parent / ".aura_forge"
BACKLOG_ROOT = FORGE_ROOT / "backlog"
EXAMPLES_ROOT = FORGE_ROOT / "examples"
INDEX_PATH = FORGE_ROOT / "indexes" / "backlog_index.yaml"

BACKLOG_LANES = ["inbox", "refined", "ready", "in_progress", "review", "done"]
STORY_ID_PATTERN = re.compile(r"^(AF-STORY-\d{4}|AF-DELTA-\d{4}|AURA-\w+-\d+)$")


def _story_id(path: Path) -> str:
    return path.stem


def collect_disk_stories() -> tuple[dict[str, str], dict[str, list[str]]]:
    """Return ({story_id: lane}, {story_id: [lane, ...]}) for every YAML/MD file in backlog subdirs.

    The first dict contains the canonical lane for stories that appear in exactly one lane.
    The second dict contains ALL lanes for stories found in more than one lane (drift).

    Files in .aura_forge/examples/ are foundational reference artifacts and are
    intentionally exempt — they are not active backlog items.
    """
    all_lanes: dict[str, list[str]] = {}
    for lane in BACKLOG_LANES:
        lane_dir = BACKLOG_ROOT / lane
        if not lane_dir.is_dir():
            continue
        for f in lane_dir.iterdir():
            if f.suffix in (".yaml", ".yml", ".md"):
                sid = _story_id(f)
                all_lanes.setdefault(sid, []).append(lane)

    canonical: dict[str, str] = {}
    duplicates: dict[str, list[str]] = {}
    for sid, lanes in all_lanes.items():
        if len(lanes) == 1:
            canonical[sid] = lanes[0]
        else:
            duplicates[sid] = lanes

    return canonical, duplicates


def collect_index_stories(index: dict) -> dict[str, str]:
    """Return {story_id: status_bucket} from by_status in the index."""
    stories: dict[str, str] = {}
    by_status = index.get("by_status", {})
    for bucket, ids in by_status.items():
        for sid in (ids or []):
            stories[str(sid)] = bucket
    return stories


def collect_index_lanes(index: dict) -> dict[str, str]:
    """Return {story_id: lane_bucket} from by_lane in the index."""
    lanes: dict[str, str] = {}
    by_lane = index.get("by_lane", {})
    for lane, ids in by_lane.items():
        for sid in (ids or []):
            lanes[str(sid)] = lane
    return lanes


def main() -> int:
    errors: list[str] = []

    if not INDEX_PATH.exists():
        print(f"ERROR: index file not found: {INDEX_PATH}", file=sys.stderr)
        return 1

    try:
        with INDEX_PATH.open() as f:
            index = yaml.safe_load(f) or {}
    except yaml.YAMLError as exc:
        print(f"ERROR: could not parse index YAML: {exc}", file=sys.stderr)
        return 1

    disk, disk_duplicates = collect_disk_stories()
    indexed_status = collect_index_stories(index)
    indexed_lane = collect_index_lanes(index)

    # 0. Stories that exist in multiple lanes on disk (single-lane invariant violation)
    for sid, lanes in sorted(disk_duplicates.items()):
        errors.append(
            f"  DUPLICATE ON DISK (multiple lanes): {sid}  [{', '.join(lanes)}]"
            f"\n    → keep only the most-advanced copy and delete the rest"
        )
    # Include duplicates in disk set for downstream checks (use most-advanced lane)
    _lane_rank = {l: i for i, l in enumerate(["inbox","refined","ready","in_progress","review","done"])}
    for sid, lanes in disk_duplicates.items():
        disk[sid] = max(lanes, key=lambda l: _lane_rank.get(l, -1))

    # 1. Files on disk missing from index
    for sid, lane in sorted(disk.items()):
        if sid not in indexed_status:
            errors.append(
                f"  MISSING FROM INDEX (by_status): {sid}  [disk lane: {lane}]"
            )
        if sid not in indexed_lane:
            errors.append(
                f"  MISSING FROM INDEX (by_lane):   {sid}  [disk lane: {lane}]"
            )

    # 2. Index entries with no file on disk
    # Entries that exist only in examples/ are exempt (foundational reference artifacts).
    example_ids = {_story_id(f) for f in EXAMPLES_ROOT.iterdir() if f.suffix in (".yaml", ".yml", ".md")} if EXAMPLES_ROOT.is_dir() else set()
    all_indexed = set(indexed_status) | set(indexed_lane)
    for sid in sorted(all_indexed):
        if sid not in disk and sid not in example_ids:
            errors.append(
                f"  INDEXED BUT NO FILE:            {sid}"
            )

    # 3. Stories in multiple by_status buckets
    seen_status: dict[str, list[str]] = {}
    for bucket, ids in (index.get("by_status") or {}).items():
        for sid in (ids or []):
            seen_status.setdefault(str(sid), []).append(bucket)
    for sid, buckets in sorted(seen_status.items()):
        if len(buckets) > 1:
            errors.append(
                f"  DUPLICATE STATUS BUCKET:        {sid}  [{', '.join(buckets)}]"
            )

    # 4. Stories in multiple by_lane buckets
    seen_lane: dict[str, list[str]] = {}
    for lane, ids in (index.get("by_lane") or {}).items():
        for sid in (ids or []):
            seen_lane.setdefault(str(sid), []).append(lane)
    for sid, lanes in sorted(seen_lane.items()):
        if len(lanes) > 1:
            errors.append(
                f"  DUPLICATE LANE BUCKET:          {sid}  [{', '.join(lanes)}]"
            )

    if errors:
        print(f"Forge index lint FAILED — {len(errors)} issue(s):\n")
        for e in errors:
            print(e)
        print(f"\nIndex:  {INDEX_PATH}")
        print(f"Backlog: {BACKLOG_ROOT}")
        return 1

    total = len(disk) + len(disk_duplicates)
    print(f"Forge index lint passed — {total} story/delta file(s) consistent with index.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
