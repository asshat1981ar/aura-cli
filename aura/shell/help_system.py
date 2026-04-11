"""Help system for interactive shell."""

from typing import Dict, List, Optional

from .models import CommandCategory, ShellCommand


class HelpSystem:
    """Generate help documentation for shell commands."""

    def __init__(self, commands: Dict[str, ShellCommand]):
        self.commands = commands

    def get_help(self, command_name: Optional[str] = None) -> str:
        """Get help text for a command or general help."""
        if command_name:
            return self._get_command_help(command_name)
        return self._get_general_help()

    def _get_command_help(self, command_name: str) -> str:
        """Get detailed help for a specific command."""
        command = self.commands.get(command_name)
        if not command:
            similar = self._find_similar(command_name)
            if similar:
                return f"Unknown command: {command_name}\nDid you mean: {', '.join(similar)}?"
            return f"Unknown command: {command_name}"

        lines = [
            f"\n  {command.name}",
            f"  {'=' * len(command.name)}",
            "",
            f"  {command.description}",
            "",
        ]

        if command.aliases:
            lines.append(f"  Aliases: {', '.join(command.aliases)}")
            lines.append("")

        if command.args_help:
            lines.append(f"  Usage: {command.name} {command.args_help}")
            lines.append("")

        if command.examples:
            lines.append("  Examples:")
            for example in command.examples:
                lines.append(f"    {command.name} {example}")
            lines.append("")

        return "\n".join(lines)

    def _get_general_help(self) -> str:
        """Get general help with command listing."""
        lines = [
            "",
            "  AURA Interactive Shell",
            "  =====================",
            "",
            "  Available Commands:",
            "",
        ]

        # Group by category
        by_category: Dict[CommandCategory, List[ShellCommand]] = {}
        seen = set()

        for cmd in self.commands.values():
            if cmd.name in seen:
                continue
            seen.add(cmd.name)

            if cmd.category not in by_category:
                by_category[cmd.category] = []
            by_category[cmd.category].append(cmd)

        # Print by category
        for category in CommandCategory:
            if category not in by_category:
                continue

            lines.append(f"  [{category.value.upper()}]")

            for cmd in sorted(by_category[category], key=lambda c: c.name):
                name = cmd.name
                desc = cmd.description[:40]
                if len(cmd.description) > 40:
                    desc = desc[:37] + "..."
                lines.append(f"    {name:<20} {desc}")

            lines.append("")

        lines.append("  Type 'help <command>' for detailed information.")
        lines.append("")

        return "\n".join(lines)

    def _find_similar(self, command_name: str, max_distance: int = 2) -> List[str]:
        """Find similar command names."""
        similar = []
        name_lower = command_name.lower()

        for cmd_name in self.commands.keys():
            if self._levenshtein(name_lower, cmd_name.lower(), max_distance) <= max_distance:
                similar.append(cmd_name)

        return similar[:3]

    def _levenshtein(self, s1: str, s2: str, max_dist: int = float("inf")) -> int:
        """Calculate Levenshtein distance between two strings with early exit."""
        if len(s1) < len(s2):
            return self._levenshtein(s2, s1, max_dist)

        if len(s2) == 0:
            return len(s1)

        # Quick check: if length difference > max_dist, strings are too different
        if len(s1) - len(s2) > max_dist:
            return max_dist + 1

        previous_row = list(range(len(s2) + 1))

        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            min_in_row = i + 1

            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                cost = min(insertions, deletions, substitutions)
                current_row.append(cost)
                min_in_row = min(min_in_row, cost)

            # Early exit: if no cell in row is below max_dist, distance will be too high
            if min_in_row > max_dist:
                return max_dist + 1

            previous_row = current_row

        return previous_row[-1]
