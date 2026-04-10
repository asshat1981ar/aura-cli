"""PromptForge — semantic-aware prompt generation for coding tasks.

Analyses code snippets for semantic properties (async patterns, recursion,
loops, error handling, network calls, state management) and assembles
optimised prompts using six task-specific templates with optional
chain-of-thought, multi-candidate, and iterative refinement strategies.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class SemanticInfo:
    """Semantic analysis of a code snippet."""

    name: str = "unknown"
    lines: int = 0
    has_async: bool = False
    has_recursion: bool = False
    has_loop: bool = False
    has_error_handling: bool = False
    has_network: bool = False
    has_state: bool = False
    imports: list[str] = field(default_factory=list)
    exports: list[str] = field(default_factory=list)
    language: str = "python"


@dataclass
class ProjectContext:
    """Detected project context from file tree."""

    languages: list[str] = field(default_factory=list)
    frameworks: list[str] = field(default_factory=list)
    structure: str = ""


# ---------------------------------------------------------------------------
# Semantic extraction
# ---------------------------------------------------------------------------


def extract_semantics(code: str, language: str = "python") -> Optional[SemanticInfo]:
    """Analyse code snippet for semantic properties.

    Args:
        code: Source code text to analyse.
        language: Programming language hint (default ``"python"``).

    Returns:
        A :class:`SemanticInfo` instance, or ``None`` when *code* is empty.
    """
    if not code or not code.strip():
        return None

    lines = code.strip().splitlines()

    # Detect function/class name
    name_match = re.search(
        r"(?:export\s+)?(?:async\s+)?(?:def|function|class|fn|func|pub\s+fn)\s+(\w+)",
        code,
    )
    name = name_match.group(1) if name_match else "unknown"

    has_async = bool(re.search(r"\b(async|await|asyncio|aiohttp|Promise|Future)\b", code))
    has_recursion = bool(name != "unknown" and re.search(rf"\b{re.escape(name)}\s*\(", "\n".join(lines[1:])))
    has_loop = bool(re.search(r"\b(for|while|\.map\(|\.filter\(|\.reduce\(|comprehension)\b", code))
    has_error = bool(re.search(r"\b(try|except|catch|raise|throw|Result|Error|Exception)\b", code))
    has_network = bool(re.search(r"\b(requests|httpx|aiohttp|fetch|urllib|curl|axios|http\.client)\b", code))
    has_state = bool(re.search(r"\b(useState|setState|reactive|ref\(|signal|self\.\w+\s*=)\b", code))

    # Extract imports
    imports = [m.group(1) for m in re.finditer(r"(?:import|from)\s+([^\s;]+)", code)]
    # Extract exports / public names
    exports = [m.group(1) for m in re.finditer(r"(?:^def|^class|^async\s+def)\s+(\w+)", code, re.MULTILINE) if not m.group(1).startswith("_")]

    return SemanticInfo(
        name=name,
        lines=len(lines),
        has_async=has_async,
        has_recursion=has_recursion,
        has_loop=has_loop,
        has_error_handling=has_error,
        has_network=has_network,
        has_state=has_state,
        imports=imports,
        exports=exports,
        language=language,
    )


def semantics_to_text(sem: SemanticInfo) -> str:
    """Convert semantic info to a natural-language description."""
    traits: list[str] = []
    if sem.has_async:
        traits.append("uses asynchronous operations")
    if sem.has_recursion:
        traits.append("employs recursion")
    if sem.has_loop:
        traits.append("contains iterative logic")
    if sem.has_network:
        traits.append("makes network requests")
    if sem.has_state:
        traits.append("manages state")
    if not sem.has_error_handling:
        traits.append("lacks error handling — consider adding it")
    else:
        traits.append("includes error handling")

    parts = [f"The function `{sem.name}` spans {sem.lines} lines of {sem.language}. It {'; '.join(traits)}."]
    if sem.imports:
        parts.append(f"Dependencies: {', '.join(sem.imports)}.")
    if sem.exports:
        parts.append(f"Exports: {', '.join(sem.exports)}.")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# File-tree parsing
# ---------------------------------------------------------------------------

_LANG_MAP: dict[str, str] = {
    "py": "Python",
    "js": "JavaScript",
    "ts": "TypeScript",
    "tsx": "TypeScript/React",
    "jsx": "React",
    "rs": "Rust",
    "go": "Go",
    "rb": "Ruby",
    "java": "Java",
    "kt": "Kotlin",
    "swift": "Swift",
    "cs": "C#",
    "cpp": "C++",
    "c": "C",
}

_FW_SIGNALS: dict[str, str] = {
    "next.config": "Next.js",
    "nuxt.config": "Nuxt",
    "vite.config": "Vite",
    "tailwind.config": "Tailwind",
    "Cargo": "Rust/Cargo",
    "go.mod": "Go Modules",
    "package.json": "Node.js",
    "pyproject.toml": "Python",
    "Dockerfile": "Docker",
    "requirements.txt": "Python/pip",
    "pytest.ini": "pytest",
    ".github": "GitHub Actions",
}


def parse_file_tree(tree: str) -> ProjectContext:
    """Detect languages and frameworks from a file-tree listing.

    Args:
        tree: Textual file-tree output (one path per line).

    Returns:
        :class:`ProjectContext` with detected languages and frameworks.
    """
    if not tree or not tree.strip():
        return ProjectContext()

    ext_counts: dict[str, int] = {}
    for line in tree.splitlines():
        m = re.search(r"\.(\w+)$", line.strip())
        if m:
            ext = m.group(1)
            ext_counts[ext] = ext_counts.get(ext, 0) + 1

    languages = [_LANG_MAP.get(ext, ext) for ext, _ in sorted(ext_counts.items(), key=lambda x: -x[1])[:5]]
    frameworks = [fw for sig, fw in _FW_SIGNALS.items() if sig in tree]

    return ProjectContext(languages=languages, frameworks=frameworks, structure=tree.strip())


# ---------------------------------------------------------------------------
# Template definitions and prompt assembly
# ---------------------------------------------------------------------------

TEMPLATES = {"function", "bugfix", "refactor", "feature", "architecture", "test"}


def _build_context_block(
    language: str,
    environment: str,
    code_context: str,
    file_tree: str,
    constraints: str,
    test_cases: str,
) -> str:
    """Build the shared context section appended to every template."""
    sections: list[str] = []

    if file_tree:
        ctx = parse_file_tree(file_tree)
        parts = []
        if ctx.languages:
            parts.append(f"Languages: {', '.join(ctx.languages)}")
        if ctx.frameworks:
            parts.append(f"Frameworks: {', '.join(ctx.frameworks)}")
        if parts:
            sections.append("## Codebase Context\n" + "\n".join(parts))

    if code_context:
        sem = extract_semantics(code_context, language or "python")
        header = "## Code Context"
        if sem:
            header += "\n" + semantics_to_text(sem)
        sections.append(header + "\n```\n" + code_context.strip() + "\n```")

    if language:
        sections.append(f"## Language\n{language}")

    if environment:
        sections.append(f"## Environment\n{environment}")

    if constraints:
        sections.append(f"## Constraints\n{constraints}")

    if test_cases:
        sections.append(f"## Test Cases\n{test_cases}")

    return "\n\n".join(sections)


def _build_strategy_block(
    chain_of_thought: bool,
    multi_candidate: bool,
    iterative_refine: bool,
) -> str:
    """Build the reasoning-strategy section."""
    parts: list[str] = []

    if chain_of_thought:
        parts.append(
            "## Reasoning Strategy: Chain-of-Thought\nThink step-by-step before writing code:\n1. Identify the core problem and edge cases.\n2. Outline your approach and data structures.\n3. Consider error handling and performance.\n4. Write the implementation.\n5. Verify correctness mentally."
        )

    if multi_candidate:
        parts.append("## Multi-Candidate Strategy\nGenerate 2-3 distinct solution approaches. For each:\n- Describe the approach in one sentence.\n- List pros and cons.\n- Provide the implementation.\nThen recommend the best approach with justification.")

    if iterative_refine:
        parts.append("## Iterative Refinement\nAfter your initial implementation:\n1. Review for correctness, readability, and performance.\n2. Identify at least one improvement.\n3. Apply the improvement and present the final version.")

    return "\n\n".join(parts)


def assemble_prompt(
    task: str,
    template: str = "function",
    language: str = "",
    environment: str = "",
    code_context: str = "",
    test_cases: str = "",
    constraints: str = "",
    file_tree: str = "",
    chain_of_thought: bool = True,
    multi_candidate: bool = False,
    iterative_refine: bool = True,
) -> str:
    """Assemble an optimised prompt from a task description and context.

    Supports six template types (``function``, ``bugfix``, ``refactor``,
    ``feature``, ``architecture``, ``test``) plus a generic fallback.

    Args:
        task: Natural-language description of what needs to be done.
        template: One of :data:`TEMPLATES` (default ``"function"``).
        language: Target programming language.
        environment: Runtime environment description.
        code_context: Existing code to consider.
        test_cases: Expected test cases or examples.
        constraints: Hard constraints the solution must satisfy.
        file_tree: Project file-tree listing for context detection.
        chain_of_thought: Enable step-by-step reasoning (default ``True``).
        multi_candidate: Request multiple solution candidates (default ``False``).
        iterative_refine: Enable self-review loop (default ``True``).

    Returns:
        A fully assembled prompt string ready to send to an LLM.
    """
    context_block = _build_context_block(
        language,
        environment,
        code_context,
        file_tree,
        constraints,
        test_cases,
    )
    strategy_block = _build_strategy_block(chain_of_thought, multi_candidate, iterative_refine)

    # ---- Template-specific instructions ----

    if template == "function":
        header = (
            f"# Task: Implement a Function\n\n"
            f"Write a well-structured function that accomplishes the following:\n\n"
            f"> {task}\n\n"
            f"## Requirements\n"
            f"- Write clean, readable code with descriptive names.\n"
            f"- Include type hints and a docstring explaining parameters, return value, and behaviour.\n"
            f"- Handle edge cases (empty inputs, None values, boundary conditions).\n"
            f"- Optimise for clarity first, then performance.\n"
            f"- Include inline comments only where the logic is non-obvious."
        )

    elif template == "bugfix":
        header = (
            f"# Task: Fix a Bug\n\n"
            f"Diagnose and fix the following issue:\n\n"
            f"> {task}\n\n"
            f"## Requirements\n"
            f"- Identify the root cause before changing any code.\n"
            f"- Explain *why* the bug occurs and how your fix resolves it.\n"
            f"- Preserve existing behaviour for all non-buggy paths.\n"
            f"- Add or update tests that would have caught this bug.\n"
            f"- Ensure the fix does not introduce regressions.\n"
            f"- If the root cause is unclear, list hypotheses ranked by likelihood."
        )

    elif template == "refactor":
        header = (
            f"# Task: Refactor Code\n\n"
            f"Refactor the following code to improve quality:\n\n"
            f"> {task}\n\n"
            f"## Requirements\n"
            f"- Preserve all existing external behaviour (inputs and outputs unchanged).\n"
            f"- Improve readability, maintainability, and adherence to SOLID principles.\n"
            f"- Reduce duplication (DRY) and simplify complex logic.\n"
            f"- Extract helper functions or classes where appropriate.\n"
            f"- Add or improve type hints and docstrings.\n"
            f"- Explain each refactoring decision and the trade-offs involved."
        )

    elif template == "feature":
        header = (
            f"# Task: Implement a New Feature\n\n"
            f"Design and implement the following feature:\n\n"
            f"> {task}\n\n"
            f"## Requirements\n"
            f"- Start with a brief design overview: components, data flow, and interfaces.\n"
            f"- Implement the feature with clean, modular code.\n"
            f"- Consider backward compatibility with existing code.\n"
            f"- Include comprehensive error handling and input validation.\n"
            f"- Write unit tests covering happy path, edge cases, and error conditions.\n"
            f"- Document public APIs with docstrings.\n"
            f"- List any dependencies or configuration changes required."
        )

    elif template == "architecture":
        header = (
            f"# Task: Architectural Design\n\n"
            f"Provide an architectural design for:\n\n"
            f"> {task}\n\n"
            f"## Requirements\n"
            f"- Define the high-level component structure and their responsibilities.\n"
            f"- Describe the data model and key interfaces between components.\n"
            f"- Specify communication patterns (sync/async, events, queues).\n"
            f"- Address scalability, reliability, and security considerations.\n"
            f"- Identify potential bottlenecks and propose mitigations.\n"
            f"- Provide a dependency diagram or textual description of component relationships.\n"
            f"- Recommend technology choices with justification.\n"
            f"- Outline a phased implementation plan."
        )

    elif template == "test":
        header = (
            f"# Task: Write Tests\n\n"
            f"Write comprehensive tests for:\n\n"
            f"> {task}\n\n"
            f"## Requirements\n"
            f"- Cover happy path, edge cases, and error conditions.\n"
            f"- Use descriptive test names that explain the expected behaviour.\n"
            f"- Group related tests logically (by function, by scenario).\n"
            f"- Use appropriate mocking/stubbing for external dependencies.\n"
            f"- Include setup/teardown where needed.\n"
            f"- Aim for high branch coverage — test boundary conditions explicitly.\n"
            f"- Each test should verify exactly one behaviour.\n"
            f"- Include both positive and negative test cases."
        )

    else:
        # Generic / unknown template fallback
        header = f"# Task\n\n> {task}\n\n## Requirements\n- Write clean, well-documented code.\n- Handle edge cases and errors.\n- Follow best practices for the target language."

    # ---- Assemble final prompt ----
    blocks = [header]
    if context_block:
        blocks.append(context_block)
    if strategy_block:
        blocks.append(strategy_block)

    return "\n\n".join(blocks)


# ---------------------------------------------------------------------------
# Agent wrapper
# ---------------------------------------------------------------------------

_SKIP_DIRS = {"__pycache__", ".git", "node_modules", ".pytest_cache", ".venv", "venv"}


class PromptForgeAgent:
    """Agent wrapper for PromptForge, compatible with AURA's agent registry.

    Exposes a ``run(input_data)`` interface consumed by the orchestrator loop
    and the MCP skills server.
    """

    capabilities = ["prompt_engineering", "prompt_assembly", "semantic_analysis"]
    description = "Semantic-aware prompt generation for coding tasks"

    def __init__(self, project_root: str = ".") -> None:
        self.project_root = Path(project_root)

    def run(self, input_data: dict[str, Any]) -> dict[str, Any]:
        """Generate an optimised prompt from task description and optional context.

        Args:
            input_data: Dict with keys ``task``, ``template``, ``code_context``,
                ``file_tree``, ``language``, ``constraints``, ``test_cases``.

        Returns:
            Dict with ``status``, ``prompt``, ``semantics``, and ``token_estimate``.
        """
        task = input_data.get("task", "")
        template = input_data.get("template", "function")
        code_context = input_data.get("code_context", "")
        file_tree = input_data.get("file_tree", "")
        language = input_data.get("language", "")
        constraints = input_data.get("constraints", "")
        test_cases = input_data.get("test_cases", "")
        environment = input_data.get("environment", "")
        chain_of_thought = input_data.get("chain_of_thought", True)
        multi_candidate = input_data.get("multi_candidate", False)
        iterative_refine = input_data.get("iterative_refine", True)

        # Auto-detect file tree if not provided
        if not file_tree and self.project_root.exists():
            try:
                tree_lines: list[str] = []
                for p in sorted(self.project_root.rglob("*"))[:100]:
                    if any(skip in p.parts for skip in _SKIP_DIRS):
                        continue
                    tree_lines.append(str(p.relative_to(self.project_root)))
                file_tree = "\n".join(tree_lines)
            except OSError:
                pass

        prompt = assemble_prompt(
            task=task,
            template=template,
            language=language,
            environment=environment,
            code_context=code_context,
            test_cases=test_cases,
            constraints=constraints,
            file_tree=file_tree,
            chain_of_thought=chain_of_thought,
            multi_candidate=multi_candidate,
            iterative_refine=iterative_refine,
        )

        semantics = extract_semantics(code_context, language or "python")

        return {
            "status": "success",
            "prompt": prompt,
            "semantics": semantics.__dict__ if semantics else None,
            "token_estimate": len(prompt.split()) * 1.3,
        }
