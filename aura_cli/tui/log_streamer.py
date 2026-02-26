"""
AURA Log Streamer — colorized structured JSON log viewer.

Usage:
    python3 -m aura_cli.tui.log_streamer          # tail stdin
    python3 -m aura_cli.tui.log_streamer --file /path/to/aura.log
    python3 -m aura_cli.tui.log_streamer --tail 50 --level info

Or programmatically:
    from aura_cli.tui.log_streamer import LogStreamer, stream_file
    stream_file("/path/to/aura.log", tail=100, level="warn")
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import IO, List, Optional

try:
    from rich.console import Console
    from rich.text import Text
    _RICH_AVAILABLE = True
except ImportError:
    _RICH_AVAILABLE = False

_LEVEL_COLORS = {
    "DEBUG": "dim",
    "INFO": "green",
    "WARN": "yellow",
    "WARNING": "yellow",
    "ERROR": "red bold",
    "CRITICAL": "red bold reverse",
}

_LEVEL_ORDER = {"DEBUG": 0, "INFO": 1, "WARN": 2, "WARNING": 2, "ERROR": 3, "CRITICAL": 4}


def _parse_line(line: str) -> Optional[dict]:
    """Parse a JSON log line. Returns None if not valid JSON."""
    line = line.strip()
    if not line:
        return None
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        return {"level": "INFO", "event": line, "ts": ""}


def _format_record(record: dict, console: "Console") -> None:
    """Print one log record with rich formatting."""
    level = str(record.get("level", "INFO")).upper()
    event = record.get("event", "")
    ts = str(record.get("ts", ""))[:19].replace("T", " ")
    details = record.get("details", {})

    color = _LEVEL_COLORS.get(level, "white")

    # Build detail string
    detail_parts = []
    if isinstance(details, dict):
        for k, v in details.items():
            detail_parts.append(f"{k}={v}")
    elif details:
        detail_parts.append(str(details))

    detail_str = "  " + "  ".join(detail_parts) if detail_parts else ""

    if _RICH_AVAILABLE:
        line = Text()
        line.append(f"[{ts}] ", style="dim")
        line.append(f"{level:<8}", style=color)
        line.append(f"{event:<20}", style="bold")
        line.append(detail_str, style="dim")
        console.print(line)
    else:
        print(f"[{ts}] {level:<8} {event:<20}{detail_str}")


def _passes_level_filter(record: dict, min_level: str) -> bool:
    """Return True if the record's level meets the minimum threshold."""
    rec_level = str(record.get("level", "INFO")).upper()
    min_order = _LEVEL_ORDER.get(min_level.upper(), 0)
    rec_order = _LEVEL_ORDER.get(rec_level, 0)
    return rec_order >= min_order


class LogStreamer:
    """
    Streams AURA structured JSON logs with colorized output.

    Attributes:
        level_filter: minimum log level to display (default "DEBUG" = all)
    """

    def __init__(self, level_filter: str = "DEBUG"):
        self.level_filter = level_filter.upper()
        self._console = Console(stderr=False) if _RICH_AVAILABLE else None

    def process_line(self, line: str) -> bool:
        """
        Parse and display one log line.

        Returns True if the line was displayed, False if filtered out.
        """
        record = _parse_line(line)
        if record is None:
            return False
        if not _passes_level_filter(record, self.level_filter):
            return False
        _format_record(record, self._console)
        return True

    def stream_fd(self, fd: IO[str], tail: Optional[int] = None) -> None:
        """Read lines from file descriptor and display them."""
        lines: List[str] = []

        if tail is not None:
            # Collect all lines first for tail mode
            all_lines = fd.readlines()
            lines = all_lines[-tail:]
            for line in lines:
                self.process_line(line)
        else:
            # Streaming mode — read until EOF or Ctrl-C
            try:
                for line in fd:
                    self.process_line(line)
            except KeyboardInterrupt:
                pass

    def stream_stdin(self, tail: Optional[int] = None) -> None:
        """Stream from stdin."""
        self.stream_fd(sys.stdin, tail=tail)

    def stream_file(self, path: Path, tail: Optional[int] = None, follow: bool = False) -> None:
        """Stream from a file. If follow=True, tail -f style."""
        path = Path(path)
        if not path.exists():
            print(f"[LogStreamer] File not found: {path}", file=sys.stderr)
            return

        with open(path) as f:
            self.stream_fd(f, tail=tail)

            if follow:
                # Simple tail -f loop
                try:
                    while True:
                        line = f.readline()
                        if line:
                            self.process_line(line)
                        else:
                            time.sleep(0.1)
                except KeyboardInterrupt:
                    pass


def stream_file(path: str, tail: Optional[int] = None, level: str = "DEBUG") -> None:
    """Convenience function to stream a log file."""
    streamer = LogStreamer(level_filter=level)
    streamer.stream_file(Path(path), tail=tail)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="AURA Log Streamer")
    parser.add_argument("--file", "-f", help="Log file to read (default: stdin)")
    parser.add_argument("--tail", "-n", type=int, default=None, help="Show last N lines")
    parser.add_argument("--level", "-l", default="debug",
                        choices=["debug", "info", "warn", "error", "critical"],
                        help="Minimum log level to display")
    parser.add_argument("--follow", action="store_true", help="Follow file (tail -f mode)")
    args = parser.parse_args()

    streamer = LogStreamer(level_filter=args.level)

    if args.file:
        streamer.stream_file(Path(args.file), tail=args.tail, follow=args.follow)
    else:
        streamer.stream_stdin(tail=args.tail)


if __name__ == "__main__":
    _main()
