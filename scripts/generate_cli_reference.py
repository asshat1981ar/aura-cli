#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from collections import OrderedDict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from aura_cli.options import help_schema


DEFAULT_OUTPUT = Path("docs/CLI_REFERENCE.md")


def _group_commands_by_top_level(commands: list[dict]) -> "OrderedDict[str, list[dict]]":
    grouped: "OrderedDict[str, list[dict]]" = OrderedDict()
    for command in commands:
        path = command.get("path") or []
        if not path:
            continue
        top_level = str(path[0])
        grouped.setdefault(top_level, []).append(command)
    return grouped


def _render_toc(grouped: "OrderedDict[str, list[dict]]") -> list[str]:
    lines = ["## Table of Contents", ""]
    for top_level, commands in grouped.items():
        lines.append(f"- [`{top_level}`](#{top_level})")
        for command in commands:
            path = command.get("path") or []
            if len(path) < 2:
                continue
            label = " ".join(path)
            anchor = "aura-" + "-".join(path)
            lines.append(f"- [`aura {label}`](#{anchor})")
    lines.append("")
    return lines


def _render_contributor_notes() -> list[str]:
    return [
        "## Contributor Notes",
        "",
        "This file is generated. Update CLI docs and snapshots together when CLI behavior changes.",
        "",
        "Recommended update steps:",
        "- `python3 scripts/generate_cli_reference.py`",
        "- `python3 -m pytest -q tests/test_cli_docs_generator.py tests/test_cli_help_snapshots.py tests/test_cli_error_snapshots.py tests/test_cli_main_dispatch.py -k snapshot`",
        "- If output changes intentionally, update `tests/snapshots/*` in the same change.",
        "",
    ]


def _render_json_contracts(schema: dict) -> list[str]:
    json_contracts = schema.get("json_contracts") or {}
    if not json_contracts:
        return []

    lines = ["## JSON Output Contracts", ""]
    for field_name in sorted(json_contracts):
        contract = json_contracts[field_name] or {}
        lines.append(f"### `{field_name}`")
        lines.append("")
        description = contract.get("description")
        if description:
            lines.append(description)
            lines.append("")
        lines.append(f"Field name: `{contract.get('field', field_name)}`")
        lines.append(f"Version: `{contract.get('version', 'unknown')}`")
        inclusion_rule = contract.get("inclusion_rule")
        if inclusion_rule:
            lines.append(f"Inclusion rule: {inclusion_rule}")
        lines.append("")

        record_fields = contract.get("record_fields") or []
        if record_fields:
            lines.append("Record fields:")
            for record_field in record_fields:
                lines.append(f"- `{record_field}`")
            lines.append("")

        optional_fields = contract.get("optional_fields") or []
        if optional_fields:
            lines.append("Optional fields:")
            for optional_field in optional_fields:
                lines.append(f"- `{optional_field}`")
            lines.append("")

        record_codes = contract.get("record_codes") or []
        if record_codes:
            lines.append("Known record codes:")
            for item in record_codes:
                code = item.get("code", "unknown")
                category = item.get("category")
                phase = item.get("phase")
                description = item.get("description")
                meta_parts = [f"`{code}`"]
                if category:
                    meta_parts.append(f"category=`{category}`")
                if phase:
                    meta_parts.append(f"phase=`{phase}`")
                line = "- " + " ".join(meta_parts)
                if description:
                    line += f": {description}"
                lines.append(line)
            lines.append("")
    return lines


def _render_command_section(command: dict) -> list[str]:
    path = " ".join(command["path"])
    lines = [f"### `aura {path}`", ""]

    summary = command.get("summary") or ""
    description = command.get("description") or ""
    if summary:
        lines.append(summary)
        lines.append("")
    if description and description != summary:
        lines.append(description)
        lines.append("")

    action = command.get("action")
    requires_runtime = command.get("requires_runtime")
    meta_bits: list[str] = []
    if action:
        meta_bits.append(f"`action`: `{action}`")
    if requires_runtime is not None:
        meta_bits.append(f"`requires_runtime`: `{str(bool(requires_runtime)).lower()}`")
    if meta_bits:
        lines.append(" ".join(meta_bits))
        lines.append("")

    legacy_flags = command.get("legacy_flags") or []
    if legacy_flags:
        lines.append("Legacy flags:")
        for flag in legacy_flags:
            lines.append(f"- `{flag}`")
        lines.append("")

    examples = command.get("examples") or []
    if examples:
        lines.append("Examples:")
        for example in examples:
            lines.append(f"- `{example}`")
        lines.append("")

    return lines


def render_cli_reference() -> str:
    schema = help_schema()
    commands = schema.get("commands", [])
    grouped = _group_commands_by_top_level(commands)

    lines: list[str] = [
        "# CLI Reference",
        "",
        f"Generated from `{schema.get('generated_by', 'aura_cli.options.help_schema')}`.",
        "",
        f"Schema version: `{schema.get('version', 'unknown')}`",
        f"Deterministic output: `{str(bool(schema.get('deterministic', False))).lower()}`",
        "",
    ]

    lines.extend(_render_contributor_notes())
    lines.extend(_render_json_contracts(schema))
    lines.extend(_render_toc(grouped))

    for top_level, group_commands in grouped.items():
        lines.append(f"## `{top_level}`")
        lines.append("")
        for command in group_commands:
            lines.extend(_render_command_section(command))

    return "\n".join(lines).rstrip() + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate docs/CLI_REFERENCE.md from help_schema().")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output markdown path.")
    parser.add_argument("--check", action="store_true", help="Fail if output file is not up to date.")
    args = parser.parse_args(argv)

    rendered = render_cli_reference()

    if args.check:
        existing = args.output.read_text(encoding="utf-8") if args.output.exists() else None
        if existing != rendered:
            print(f"{args.output} is out of date. Run scripts/generate_cli_reference.py.")
            return 1
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
