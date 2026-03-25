"""Markdown design-spec parser — extracts workstreams from design documents."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

from core.sadd.types import DesignSpec, WorkstreamSpec

logger = logging.getLogger(__name__)

# Action verbs that signal a workstream task.
_ACTION_VERBS = frozenset(
    {
        "implement",
        "create",
        "add",
        "build",
        "refactor",
        "test",
        "fix",
        "wire",
        "expand",
    }
)

# Heading keywords that mark a workstream section.
_WORKSTREAM_KEYWORDS = frozenset(
    {
        "workstream",
        "component",
        "task",
        "step",
    }
)

# Dependency signal phrases (lowercase).
_DEPENDENCY_PHRASES = ("depends on", "requires", "after", "blocked by")

# Regex for markdown headings (H1–H6).
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)

# Regex for checkbox acceptance criteria.
_CHECKBOX_RE = re.compile(r"^\s*-\s*\[[ x]?\]\s*(.+)$", re.MULTILINE)

# Regex for "Acceptance:" labeled items.
_ACCEPTANCE_LABEL_RE = re.compile(
    r"(?:^|\n)\s*[Aa]cceptance\s*:\s*(.+?)(?=\n\s*\n|\Z)",
    re.DOTALL,
)


def _slugify(text: str) -> str:
    """Convert *text* to a workstream slug: ``ws_<sanitized>``."""
    slug = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return f"ws_{slug}" if slug else "ws_unnamed"


class DesignSpecParser:
    """Parse a Markdown design spec into a :class:`DesignSpec`.

    Parameters
    ----------
    model:
        Optional model adapter for LLM-assisted fallback on unstructured
        prose.  Not used in R0 — reserved for future iterations.
    """

    def __init__(self, model: Optional[object] = None) -> None:
        self._model = model

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def parse(self, markdown: str) -> DesignSpec:
        """Parse *markdown* and return a fully-populated :class:`DesignSpec`."""
        sections = self._extract_sections(markdown)
        workstreams = self._identify_workstreams(sections)
        workstreams = self._infer_dependencies(workstreams)
        confidence = self._compute_parse_confidence(sections, workstreams)

        # Derive title from the first H1, falling back to the first key.
        title = sections.get("__title__", "")
        if not title and sections:
            title = next(iter(sections))

        # Summary: use an explicit "Summary" / "Overview" section if present.
        summary = ""
        for key in ("summary", "overview", "introduction", "abstract"):
            for sec_key, sec_body in sections.items():
                if sec_key.lower() == key:
                    summary = sec_body.strip()
                    break
            if summary:
                break

        if confidence < 0.5 and not workstreams:
            # Wrap the entire document as a single workstream.
            if self._model is not None:
                logger.warning(
                    "Low parse confidence (%.2f) — LLM fallback not yet implemented (R0); wrapping document as single workstream.",
                    confidence,
                )
            workstreams = [
                WorkstreamSpec(
                    id="ws_full_document",
                    title=title or "Full Document",
                    goal_text=markdown.strip(),
                    priority=1,
                ),
            ]
            # Re-score after fallback.
            confidence = self._compute_parse_confidence(sections, workstreams)

        return DesignSpec(
            title=title,
            summary=summary,
            workstreams=workstreams,
            raw_markdown=markdown,
            parse_confidence=confidence,
        )

    def parse_file(self, path: Path) -> DesignSpec:
        """Read a Markdown file at *path* and parse it."""
        text = Path(path).read_text(encoding="utf-8")
        return self.parse(text)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_sections(self, markdown: str) -> dict[str, str]:
        """Split *markdown* by heading hierarchy.

        Returns a dict mapping heading text to body content.  The first
        H1 heading is stored under the special key ``__title__``.
        """
        sections: dict[str, str] = {}
        matches = list(_HEADING_RE.finditer(markdown))

        if not matches:
            # No headings at all — return the whole text as one section.
            stripped = markdown.strip()
            if stripped:
                sections["__body__"] = stripped
            return sections

        # Capture text before the first heading (if any).
        preamble = markdown[: matches[0].start()].strip()
        if preamble:
            sections["__preamble__"] = preamble

        for idx, match in enumerate(matches):
            level = len(match.group(1))
            heading_text = match.group(2).strip()
            body_start = match.end()
            body_end = matches[idx + 1].start() if idx + 1 < len(matches) else len(markdown)
            body = markdown[body_start:body_end].strip()

            if level == 1 and "__title__" not in sections:
                sections["__title__"] = heading_text

            sections[heading_text] = body

        return sections

    def _identify_workstreams(self, sections: dict[str, str]) -> list[WorkstreamSpec]:
        """Identify workstreams from parsed *sections*."""
        workstreams: list[WorkstreamSpec] = []
        seen_ids: set[str] = set()
        priority_counter = 1

        for heading, body in sections.items():
            if heading.startswith("__"):
                continue

            heading_lower = heading.lower()

            # Decide whether this section is a workstream.
            is_keyword_match = any(kw in heading_lower for kw in _WORKSTREAM_KEYWORDS)
            is_action_match = self._body_has_action_verbs(body)

            if not (is_keyword_match or is_action_match):
                continue

            ws_id = _slugify(heading)
            # Ensure unique IDs.
            if ws_id in seen_ids:
                suffix = 2
                while f"{ws_id}_{suffix}" in seen_ids:
                    suffix += 1
                ws_id = f"{ws_id}_{suffix}"
            seen_ids.add(ws_id)

            tags = self._extract_tags(heading)
            acceptance = self._extract_acceptance_criteria(body)

            workstreams.append(
                WorkstreamSpec(
                    id=ws_id,
                    title=heading,
                    goal_text=body.strip(),
                    priority=priority_counter,
                    tags=tags,
                    acceptance_criteria=acceptance,
                )
            )
            priority_counter += 1

        return workstreams

    def _infer_dependencies(self, workstreams: list[WorkstreamSpec]) -> list[WorkstreamSpec]:
        """Populate ``depends_on`` by scanning for explicit and implicit cues."""
        id_by_title: dict[str, str] = {ws.title.lower(): ws.id for ws in workstreams}
        # Also index by the title portion after common prefixes like "Workstream:"
        for ws in workstreams:
            for prefix in ("workstream:", "component:", "task:", "step:"):
                lower_title = ws.title.lower()
                if lower_title.startswith(prefix):
                    short = lower_title[len(prefix) :].strip()
                    if short and short not in id_by_title:
                        id_by_title[short] = ws.id
        title_list = list(id_by_title.keys())

        for ws in workstreams:
            deps: list[str] = list(ws.depends_on)  # preserve any existing
            text_lower = ws.goal_text.lower()

            # --- explicit dependency phrases ---
            for phrase in _DEPENDENCY_PHRASES:
                for occurrence in re.finditer(
                    re.escape(phrase) + r":?\s+[\"']?(.+?)[\"']?\s*(?:[.,;]|\n|\Z)",
                    text_lower,
                ):
                    ref = occurrence.group(1).strip().rstrip(".,;:")
                    # Try exact title match first.
                    if ref in id_by_title:
                        dep_id = id_by_title[ref]
                        if dep_id != ws.id and dep_id not in deps:
                            deps.append(dep_id)
                        continue
                    # Substring match against known titles.
                    for title_lower, dep_id in id_by_title.items():
                        if (ref in title_lower or title_lower in ref) and dep_id != ws.id and dep_id not in deps:
                            deps.append(dep_id)

            # --- implicit: mentions of files/modules created by another ws ---
            for other in workstreams:
                if other.id == ws.id or other.id in deps:
                    continue
                # Look for file-like references (e.g. "core/foo.py") in the
                # *other* workstream that are also mentioned in this one.
                file_refs = re.findall(
                    r"[\w/]+\.(?:py|ts|js|json|yaml|yml|toml|cfg)",
                    other.goal_text,
                )
                for fref in file_refs:
                    if fref in ws.goal_text and other.id not in deps:
                        deps.append(other.id)
                        break

            ws.depends_on = deps

        return workstreams

    def _compute_parse_confidence(
        self,
        sections: dict[str, str],
        workstreams: list[WorkstreamSpec],
    ) -> float:
        """Return a 0.0–1.0 confidence score for the parse quality."""
        score = 0.0

        # Has clear heading structure?
        real_sections = {k: v for k, v in sections.items() if not k.startswith("__")}
        if real_sections:
            score += 0.3

        # Workstreams found?
        if workstreams:
            score += 0.3

        # Dependencies resolved?
        if workstreams and any(ws.depends_on for ws in workstreams):
            score += 0.2
        elif workstreams and len(workstreams) == 1:
            # Single workstream — dependency resolution is trivially satisfied.
            score += 0.2

        # No ambiguous sections?  (All non-meta sections mapped to a ws.)
        if real_sections and workstreams:
            mapped_titles = {ws.title for ws in workstreams}
            unmapped = [k for k in real_sections if k not in mapped_titles]
            # Allow some unmapped sections (summary, intro, etc.).
            ambiguity_ratio = len(unmapped) / max(len(real_sections), 1)
            if ambiguity_ratio <= 0.5:
                score += 0.2

        return min(score, 1.0)

    # ------------------------------------------------------------------
    # Micro-helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _body_has_action_verbs(body: str) -> bool:
        """Return ``True`` if *body* contains bullet/numbered items with action verbs."""
        for line in body.splitlines():
            stripped = line.strip()
            # Match bullets (-, *, +) or numbered lists (1., 2., etc.).
            list_match = re.match(r"^(?:[-*+]|\d+[.)]\s)", stripped)
            if not list_match:
                continue
            # Check for an action verb near the start of the item.
            item_text = stripped[list_match.end() :].strip().lower()
            first_word = item_text.split(None, 1)[0] if item_text else ""
            if first_word in _ACTION_VERBS:
                return True
        return False

    @staticmethod
    def _extract_tags(heading: str) -> list[str]:
        """Derive tags from a heading string."""
        tags: list[str] = []
        heading_lower = heading.lower()
        # Extract bracketed tags like [backend], [api].
        for m in re.finditer(r"\[([^\]]+)\]", heading):
            tags.append(m.group(1).strip().lower())
        # Add keyword-based tags.
        keyword_tags = {
            "api": "api",
            "backend": "backend",
            "frontend": "frontend",
            "database": "database",
            "db": "database",
            "test": "testing",
            "tests": "testing",
            "refactor": "refactor",
            "infra": "infrastructure",
            "infrastructure": "infrastructure",
            "ci": "ci-cd",
            "cd": "ci-cd",
            "docs": "documentation",
            "documentation": "documentation",
        }
        for token in re.split(r"[\s:/\-_]+", heading_lower):
            if token in keyword_tags and keyword_tags[token] not in tags:
                tags.append(keyword_tags[token])
        return tags

    @staticmethod
    def _extract_acceptance_criteria(body: str) -> list[str]:
        """Extract acceptance criteria from checkboxes and labeled sections."""
        criteria: list[str] = []

        # Checkbox items: - [ ] or - [x].
        for m in _CHECKBOX_RE.finditer(body):
            text = m.group(1).strip()
            if text and text not in criteria:
                criteria.append(text)

        # "Acceptance:" labeled blocks.
        for m in _ACCEPTANCE_LABEL_RE.finditer(body):
            block = m.group(1).strip()
            for line in block.splitlines():
                item = line.strip().lstrip("-*+ ").strip()
                if item and item not in criteria:
                    criteria.append(item)

        return criteria
