"""
GitHub Copilot MCP Server — Copilot-style AI workflows over live GitHub data.

Combines GitHub REST API (via GitHubTools) with AURA's ModelAdapter to deliver
GitHub Copilot Workspace-style intelligence as MCP-compatible HTTP tools.

Tools:
  issue_analyze     — fetch issue → AI root-cause analysis + implementation steps
  pr_review         — fetch PR diff → AI code review with severity-tagged findings
  pr_describe       — fetch PR diff → auto-generate title + description
  code_explain      — explain any code snippet in plain English
  code_fix          — given code + error → targeted fix suggestion
  test_generate     — generate test cases for a function/class
  commit_message    — generate Conventional Commit message from a diff
  repo_health       — open issues age, stale PRs, CI status, activity summary
  issue_to_plan     — issue + codebase context → ordered implementation plan
  find_related_code — search repo code + AI relevance ranking

Port:  8005 (override with COPILOT_MCP_PORT)
Auth:  set COPILOT_MCP_TOKEN env var

Requirements:
  GITHUB_PAT   — GitHub personal access token (repo + read:org scopes)
  AURA_API_KEY — OpenRouter/OpenAI key (used by ModelAdapter)

Start:
  uvicorn tools.github_copilot_mcp:app --port 8005
"""
from __future__ import annotations

import os
import re
import sys
import time
import textwrap
import requests as _req
from pathlib import Path
from typing import Any, Dict, List, Optional

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

os.environ.setdefault("AURA_SKIP_CHDIR", "1")

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel

from core.logging_utils import log_json

# ---------------------------------------------------------------------------
# Lazy-loaded singletons
# ---------------------------------------------------------------------------

_github: Any = None   # GitHubTools
_model: Any = None    # ModelAdapter


def _get_github():
    global _github
    if _github is None:
        from tools.github_tools import GitHubTools
        _github = GitHubTools()
    return _github


def _get_model():
    global _model
    if _model is None:
        from core.model_adapter import ModelAdapter
        _model = ModelAdapter()
    return _model


# ---------------------------------------------------------------------------
# App + auth
# ---------------------------------------------------------------------------

app = FastAPI(
    title="GitHub Copilot MCP",
    description="Copilot-style AI workflows over live GitHub repository data.",
    version="1.0.0",
)

_TOKEN = os.getenv("COPILOT_MCP_TOKEN", "")


def _check_auth(authorization: Optional[str] = Header(default=None)) -> None:
    if _TOKEN and authorization != f"Bearer {_TOKEN}":
        raise HTTPException(status_code=401, detail="Unauthorized")


# ---------------------------------------------------------------------------
# AI helper — call ModelAdapter with a structured prompt
# ---------------------------------------------------------------------------

def _ai(prompt: str) -> str:
    """Call the LLM and return stripped response text."""
    try:
        response = _get_model().respond(prompt)
        return str(response).strip() if response else "No response from model."
    except Exception as exc:
        log_json("ERROR", "copilot_mcp_model_error", details={"error": str(exc)})
        return f"Model error: {exc}"


def _truncate(text: str, max_chars: int = 12000) -> str:
    """Truncate large payloads to stay within prompt limits."""
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    return text[:half] + f"\n\n... [truncated {len(text) - max_chars} chars] ...\n\n" + text[-half:]


# ---------------------------------------------------------------------------
# Tool descriptor schemas
# ---------------------------------------------------------------------------

_TOOL_SCHEMAS: Dict[str, Dict] = {
    "issue_analyze": {
        "description": (
            "Fetch a GitHub issue and produce an AI analysis: root cause, affected components, "
            "risk level, and a numbered implementation plan."
        ),
        "input": {
            "repo": {"type": "string", "description": "Repository in owner/repo format", "required": True},
            "issue_number": {"type": "integer", "description": "Issue number", "required": True},
        },
    },
    "pr_review": {
        "description": (
            "Fetch a PR diff and run an AI code review. Returns severity-tagged findings "
            "(critical/high/medium/low), summary, and overall verdict."
        ),
        "input": {
            "repo": {"type": "string", "description": "Repository in owner/repo format", "required": True},
            "pr_number": {"type": "integer", "description": "Pull request number", "required": True},
        },
    },
    "pr_describe": {
        "description": (
            "Auto-generate a professional PR title and description from the PR diff and commit messages."
        ),
        "input": {
            "repo": {"type": "string", "description": "Repository in owner/repo format", "required": True},
            "pr_number": {"type": "integer", "description": "Pull request number", "required": True},
        },
    },
    "code_explain": {
        "description": "Explain what a code snippet does in plain English, including its purpose, inputs, outputs, and any edge cases.",
        "input": {
            "code": {"type": "string", "description": "Code snippet to explain", "required": True},
            "language": {"type": "string", "description": "Programming language (e.g. python, typescript)"},
            "context": {"type": "string", "description": "Optional surrounding context or intent"},
        },
    },
    "code_fix": {
        "description": "Given code and an error message, suggest a targeted fix with explanation.",
        "input": {
            "code": {"type": "string", "description": "Code that contains the bug", "required": True},
            "error": {"type": "string", "description": "Error message or description of the problem", "required": True},
            "language": {"type": "string", "description": "Programming language"},
        },
    },
    "test_generate": {
        "description": "Generate test cases for a function or class. Returns test code with happy-path, edge-case, and error tests.",
        "input": {
            "code": {"type": "string", "description": "Function or class to test", "required": True},
            "framework": {"type": "string", "description": "Test framework (pytest, jest, unittest, etc.)"},
            "language": {"type": "string", "description": "Programming language"},
        },
    },
    "commit_message": {
        "description": "Generate a Conventional Commit message (feat/fix/chore etc.) from a diff or description of changes.",
        "input": {
            "diff": {"type": "string", "description": "Git diff or description of changes", "required": True},
            "scope": {"type": "string", "description": "Optional scope hint (e.g. auth, api, ui)"},
        },
    },
    "repo_health": {
        "description": "Analyse a GitHub repository's health: open issue age, stale PRs, recent commit activity, top contributors.",
        "input": {
            "repo": {"type": "string", "description": "Repository in owner/repo format", "required": True},
        },
    },
    "issue_to_plan": {
        "description": (
            "Convert a GitHub issue into a detailed implementation plan. "
            "Optionally provide relevant source file contents for context."
        ),
        "input": {
            "repo": {"type": "string", "description": "Repository in owner/repo format", "required": True},
            "issue_number": {"type": "integer", "description": "Issue number", "required": True},
            "context_files": {
                "type": "array",
                "description": "Optional list of repo-relative file paths to include as context",
            },
        },
    },
    "find_related_code": {
        "description": "Search a GitHub repository for code related to a query, then AI-rank results by relevance.",
        "input": {
            "repo": {"type": "string", "description": "Repository in owner/repo format", "required": True},
            "query": {"type": "string", "description": "What you are looking for", "required": True},
            "limit": {"type": "integer", "description": "Max search results to consider (default 10)"},
        },
    },
}


def _build_descriptor(name: str) -> Dict:
    s = _TOOL_SCHEMAS[name]
    return {
        "name": name,
        "description": s["description"],
        "inputSchema": {
            "type": "object",
            "properties": s.get("input", {}),
            "required": [k for k, v in s.get("input", {}).items() if v.get("required")],
        },
    }


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def _tool_issue_analyze(args: Dict) -> Dict:
    repo = args["repo"]
    issue_number = int(args["issue_number"])
    gh = _get_github()

    issue = gh.get_issue_details(repo, issue_number)
    title = issue.get("title", "")
    body = _truncate(issue.get("body") or "", 4000)
    labels = [l["name"] for l in issue.get("labels", [])]
    comments_data = gh._make_request("GET", f"{gh.BASE_URL}/repos/{repo}/issues/{issue_number}/comments")
    comments_text = "\n".join(
        f"[{c.get('user', {}).get('login', 'user')}]: {c.get('body', '')[:300]}"
        for c in (comments_data if isinstance(comments_data, list) else [])[:5]
    )

    prompt = textwrap.dedent(f"""
        You are a senior software engineer analyzing a GitHub issue.

        Repository: {repo}
        Issue #{issue_number}: {title}
        Labels: {', '.join(labels) or 'none'}

        Issue body:
        {body}

        Top comments:
        {comments_text or 'No comments.'}

        Provide a structured analysis with these sections:
        1. **Summary** — one sentence describing the problem
        2. **Root Cause** — likely cause(s) of the issue
        3. **Affected Components** — files/modules/systems likely involved
        4. **Risk Level** — critical / high / medium / low with justification
        5. **Implementation Steps** — numbered list of concrete steps to fix/implement
        6. **Testing Approach** — how to verify the fix

        Be concise and actionable.
    """).strip()

    analysis = _ai(prompt)
    return {
        "repo": repo,
        "issue_number": issue_number,
        "title": title,
        "labels": labels,
        "analysis": analysis,
    }


def _tool_pr_review(args: Dict) -> Dict:
    repo = args["repo"]
    pr_number = int(args["pr_number"])
    gh = _get_github()

    pr = gh.get_pull_request_details(repo, pr_number)
    title = pr.get("title", "")
    pr_body = pr.get("body") or ""

    # Fetch diff
    diff_url = f"{gh.BASE_URL}/repos/{repo}/pulls/{pr_number}"
    diff_headers = {**gh.headers, "Accept": "application/vnd.github.v3.diff"}
    diff_resp = _req.get(diff_url, headers=diff_headers, timeout=30)
    diff_text = _truncate(diff_resp.text if diff_resp.ok else "Diff unavailable.", 10000)

    prompt = textwrap.dedent(f"""
        You are an expert code reviewer. Review this GitHub pull request thoroughly.

        Repository: {repo}
        PR #{pr_number}: {title}
        Description: {pr_body[:500] or 'No description.'}

        Diff:
        {diff_text}

        Produce a structured review:

        ## Summary
        (1–2 sentences on what the PR does)

        ## Findings
        List each issue as:
        - [SEVERITY] File:line — Description and suggested fix
        (severities: CRITICAL | HIGH | MEDIUM | LOW | INFO)

        ## Verdict
        APPROVE / REQUEST_CHANGES / COMMENT — with one-line reason

        ## Positive Highlights
        (what was done well)

        Focus on: bugs, security issues, logic errors, performance problems, missing error handling.
        Skip pure style/formatting comments.
    """).strip()

    review = _ai(prompt)

    # Parse verdict from response
    verdict = "COMMENT"
    for v in ("APPROVE", "REQUEST_CHANGES", "COMMENT"):
        if v in review.upper():
            verdict = v
            break

    return {
        "repo": repo,
        "pr_number": pr_number,
        "title": title,
        "verdict": verdict,
        "review": review,
    }


def _tool_pr_describe(args: Dict) -> Dict:
    repo = args["repo"]
    pr_number = int(args["pr_number"])
    gh = _get_github()

    pr = gh.get_pull_request_details(repo, pr_number)
    head = pr.get("head", {}).get("ref", "")
    base = pr.get("base", {}).get("ref", "main")

    # Commits
    commits_data = gh._make_request("GET", f"{gh.BASE_URL}/repos/{repo}/pulls/{pr_number}/commits")
    commit_msgs = "\n".join(
        f"- {c.get('commit', {}).get('message', '').splitlines()[0]}"
        for c in (commits_data if isinstance(commits_data, list) else [])[:15]
    )

    # Files changed
    files_data = gh.get_files(repo, pr_number) if hasattr(gh, 'get_files') else \
        gh._make_request("GET", f"{gh.BASE_URL}/repos/{repo}/pulls/{pr_number}/files")
    files_changed = "\n".join(
        f"- {f.get('filename')} (+{f.get('additions',0)}/-{f.get('deletions',0)})"
        for f in (files_data if isinstance(files_data, list) else [])[:20]
    )

    prompt = textwrap.dedent(f"""
        Generate a professional GitHub PR title and description.

        Branch: {head} → {base}

        Commits:
        {commit_msgs or 'No commit messages available.'}

        Files changed:
        {files_changed or 'File list unavailable.'}

        Output format (use exactly these headers):
        ## Title
        (concise, imperative, <72 chars)

        ## Description
        ### What
        (what changed)

        ### Why
        (motivation / linked issue if inferable)

        ### How
        (implementation approach)

        ### Testing
        (how to test / what was tested)
    """).strip()

    result = _ai(prompt)

    # Extract title line
    title_match = re.search(r"## Title\s*\n(.+)", result)
    generated_title = title_match.group(1).strip() if title_match else ""

    return {
        "repo": repo,
        "pr_number": pr_number,
        "head": head,
        "base": base,
        "generated_title": generated_title,
        "generated_description": result,
    }


def _tool_code_explain(args: Dict) -> Dict:
    code = args["code"]
    language = args.get("language", "Python")
    context = args.get("context", "")

    prompt = textwrap.dedent(f"""
        Explain the following {language} code clearly and concisely.
        {f'Context: {context}' if context else ''}

        Code:
        ```{language.lower()}
        {_truncate(code, 6000)}
        ```

        Provide:
        1. **Purpose** — what this code does in one sentence
        2. **How it works** — step-by-step walkthrough of key logic
        3. **Inputs & Outputs** — parameters, return values, side effects
        4. **Edge Cases / Gotchas** — anything surprising or potentially buggy
        5. **Dependencies** — what external things it relies on
    """).strip()

    explanation = _ai(prompt)
    return {"language": language, "explanation": explanation}


def _tool_code_fix(args: Dict) -> Dict:
    code = args["code"]
    error = args["error"]
    language = args.get("language", "Python")

    prompt = textwrap.dedent(f"""
        Fix the following {language} code that has this error:

        Error:
        {error}

        Code:
        ```{language.lower()}
        {_truncate(code, 6000)}
        ```

        Respond with:
        1. **Root Cause** — why this error occurs
        2. **Fixed Code** — the corrected code in a fenced code block
        3. **Explanation** — what you changed and why
        4. **Prevention** — how to avoid this class of error in future
    """).strip()

    fix = _ai(prompt)
    return {"language": language, "error": error, "fix": fix}


def _tool_test_generate(args: Dict) -> Dict:
    code = args["code"]
    framework = args.get("framework", "pytest")
    language = args.get("language", "Python")

    prompt = textwrap.dedent(f"""
        Generate comprehensive {framework} tests for the following {language} code.

        ```{language.lower()}
        {_truncate(code, 5000)}
        ```

        Generate tests covering:
        1. Happy path (normal expected inputs)
        2. Edge cases (empty, None, boundary values, large inputs)
        3. Error cases (invalid inputs, exceptions that should be raised)
        4. Any specific business logic cases visible in the code

        Use {framework} conventions. Include:
        - Descriptive test names (test_<what>_<when>_<expected>)
        - Setup/teardown if needed
        - Mocks for external dependencies
        - Assertions that are specific and meaningful

        Output only the test code in a fenced code block.
    """).strip()

    tests = _ai(prompt)
    return {"framework": framework, "language": language, "tests": tests}


def _tool_commit_message(args: Dict) -> Dict:
    diff = args["diff"]
    scope = args.get("scope", "")

    prompt = textwrap.dedent(f"""
        Generate a Conventional Commit message for these changes.

        {'Scope hint: ' + scope if scope else ''}

        Changes:
        {_truncate(diff, 6000)}

        Rules:
        - Format: <type>(<scope>): <subject>
        - Types: feat | fix | docs | style | refactor | perf | test | chore | ci | build | revert
        - Subject: imperative mood, ≤72 chars, no period at end
        - If breaking change, add ! after type/scope and BREAKING CHANGE footer

        Output:
        ## Commit Message
        (the one-line commit message)

        ## Body (optional)
        (if more context is needed, 1–3 bullet points)

        ## Footer (optional)
        (breaking changes, closes #issue)
    """).strip()

    result = _ai(prompt)
    msg_match = re.search(r"## Commit Message\s*\n(.+)", result)
    commit_msg = msg_match.group(1).strip() if msg_match else result.splitlines()[0]

    return {"commit_message": commit_msg, "full_output": result}


def _tool_repo_health(args: Dict) -> Dict:
    repo = args["repo"]
    gh = _get_github()

    repo_info = gh.get_repo(repo)
    open_issues = gh._make_request(
        "GET", f"{gh.BASE_URL}/repos/{repo}/issues",
        params={"state": "open", "per_page": 30, "sort": "created", "direction": "asc"},
    )
    open_prs = gh._make_request(
        "GET", f"{gh.BASE_URL}/repos/{repo}/pulls",
        params={"state": "open", "per_page": 30, "sort": "updated", "direction": "asc"},
    )
    commits = gh._make_request(
        "GET", f"{gh.BASE_URL}/repos/{repo}/commits",
        params={"per_page": 10},
    )

    def _age_days(date_str: Optional[str]) -> float:
        if not date_str:
            return 0
        import datetime
        try:
            dt = datetime.datetime.fromisoformat(date_str.replace("Z", "+00:00"))
            return (datetime.datetime.now(datetime.timezone.utc) - dt).days
        except Exception:
            return 0

    issues_list = open_issues if isinstance(open_issues, list) else []
    prs_list = open_prs if isinstance(open_prs, list) else []
    commits_list = commits if isinstance(commits, list) else []

    oldest_issue_days = max((_age_days(i.get("created_at")) for i in issues_list), default=0)
    stale_prs = [p for p in prs_list if _age_days(p.get("updated_at")) > 14]
    last_commit_days = _age_days(commits_list[0].get("commit", {}).get("committer", {}).get("date")) if commits_list else None

    # Health score heuristic
    score = 100
    if oldest_issue_days > 180:
        score -= 20
    if len(stale_prs) > 3:
        score -= 15
    if last_commit_days and last_commit_days > 30:
        score -= 25
    if repo_info.get("open_issues_count", 0) > 100:
        score -= 10

    health = {
        "repo": repo,
        "health_score": max(0, score),
        "stars": repo_info.get("stargazers_count", 0),
        "forks": repo_info.get("forks_count", 0),
        "open_issues": repo_info.get("open_issues_count", 0),
        "open_prs": len(prs_list),
        "stale_prs_14d": len(stale_prs),
        "oldest_open_issue_days": oldest_issue_days,
        "days_since_last_commit": last_commit_days,
        "language": repo_info.get("language"),
        "license": (repo_info.get("license") or {}).get("spdx_id"),
        "has_wiki": repo_info.get("has_wiki"),
        "archived": repo_info.get("archived", False),
    }

    # AI summary
    prompt = textwrap.dedent(f"""
        Summarize the health of this GitHub repository in 3–4 sentences.
        Data: {health}
        Mention: activity level, issue/PR management, any red flags, and one improvement recommendation.
    """).strip()
    health["ai_summary"] = _ai(prompt)
    return health


def _tool_issue_to_plan(args: Dict) -> Dict:
    repo = args["repo"]
    issue_number = int(args["issue_number"])
    context_files: List[str] = args.get("context_files") or []
    gh = _get_github()

    issue = gh.get_issue_details(repo, issue_number)
    title = issue.get("title", "")
    body = _truncate(issue.get("body") or "", 3000)

    file_contexts = []
    for fp in context_files[:4]:  # max 4 files to stay within token budget
        try:
            fc = gh.get_file_contents(repo, fp)
            file_contexts.append(f"### {fp}\n```\n{_truncate(fc.get('content', ''), 1500)}\n```")
        except Exception:
            pass

    files_section = "\n\n".join(file_contexts) if file_contexts else "No source files provided."

    prompt = textwrap.dedent(f"""
        You are a senior engineer creating an implementation plan for a GitHub issue.

        Repository: {repo}
        Issue #{issue_number}: {title}

        Issue description:
        {body}

        Relevant source files:
        {files_section}

        Create a detailed, actionable implementation plan:

        ## Objective
        (one sentence)

        ## Files to Create/Modify
        | File | Action | What to change |
        |------|--------|----------------|

        ## Implementation Steps
        Numbered steps, each with:
        - What to do
        - Code sketch or pseudocode if helpful
        - Acceptance criteria

        ## Potential Risks
        (things to watch out for)

        ## Estimated Complexity
        XS / S / M / L / XL with brief justification
    """).strip()

    plan = _ai(prompt)
    return {
        "repo": repo,
        "issue_number": issue_number,
        "title": title,
        "context_files_used": context_files,
        "plan": plan,
    }


def _tool_find_related_code(args: Dict) -> Dict:
    repo = args["repo"]
    query = args["query"]
    limit = int(args.get("limit", 10))
    gh = _get_github()

    search_results = gh.search_code(repo, query, limit=limit)
    items = search_results.get("results", [])

    if not items:
        return {"repo": repo, "query": query, "results": [], "ai_ranking": "No results found."}

    results_summary = "\n".join(
        f"{i+1}. {r['path']} — {r['name']}" for i, r in enumerate(items)
    )

    prompt = textwrap.dedent(f"""
        A developer searched a GitHub repository for: "{query}"

        Search results (file paths):
        {results_summary}

        Rank these results by relevance to the query and explain briefly why each is relevant.
        Format:
        1. path/to/file — [HIGH/MEDIUM/LOW relevance] — reason
        ...

        Then give a 1-sentence recommendation on which file to look at first.
    """).strip()

    ranking = _ai(prompt)
    return {
        "repo": repo,
        "query": query,
        "total_found": search_results.get("total_count", len(items)),
        "results": items,
        "ai_ranking": ranking,
    }


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

_HANDLERS = {
    "issue_analyze": _tool_issue_analyze,
    "pr_review": _tool_pr_review,
    "pr_describe": _tool_pr_describe,
    "code_explain": _tool_code_explain,
    "code_fix": _tool_code_fix,
    "test_generate": _tool_test_generate,
    "commit_message": _tool_commit_message,
    "repo_health": _tool_repo_health,
    "issue_to_plan": _tool_issue_to_plan,
    "find_related_code": _tool_find_related_code,
}


class ToolCallRequest(BaseModel):
    tool_name: str
    args: Dict[str, Any] = {}


class ToolResult(BaseModel):
    tool_name: str
    result: Any = None
    error: Optional[str] = None
    elapsed_ms: float = 0.0


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health(_: None = Depends(_check_auth)):
    github_ok = bool(os.getenv("GITHUB_PAT"))
    model_ok = bool(os.getenv("AURA_API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY"))
    return {
        "status": "ok" if (github_ok and model_ok) else "degraded",
        "github_pat_set": github_ok,
        "model_key_set": model_ok,
        "tool_count": len(_HANDLERS),
        "server": "github_copilot_mcp",
        "version": "1.0.0",
    }


@app.get("/tools")
async def list_tools(_: None = Depends(_check_auth)) -> List[Dict]:
    return [_build_descriptor(n) for n in _TOOL_SCHEMAS]


@app.get("/tool/{name}")
async def get_tool(name: str, _: None = Depends(_check_auth)):
    if name not in _TOOL_SCHEMAS:
        raise HTTPException(status_code=404, detail=f"Tool '{name}' not found.")
    return _build_descriptor(name)


@app.post("/call")
async def call_tool(request: ToolCallRequest, _: None = Depends(_check_auth)) -> ToolResult:
    handler = _HANDLERS.get(request.tool_name)
    if not handler:
        raise HTTPException(status_code=404, detail=f"Tool '{request.tool_name}' not found.")

    t0 = time.time()
    try:
        result = handler(request.args)
        elapsed = round((time.time() - t0) * 1000, 2)
        log_json("INFO", "copilot_mcp_tool_called", details={"tool": request.tool_name, "elapsed_ms": elapsed})
        return ToolResult(tool_name=request.tool_name, result=result, elapsed_ms=elapsed)
    except (KeyError, ValueError) as exc:
        elapsed = round((time.time() - t0) * 1000, 2)
        log_json("WARN", "copilot_mcp_bad_args", details={"tool": request.tool_name, "error": str(exc)})
        return ToolResult(tool_name=request.tool_name, error=f"Bad arguments: {exc}", elapsed_ms=elapsed)
    except Exception as exc:
        elapsed = round((time.time() - t0) * 1000, 2)
        log_json("ERROR", "copilot_mcp_error", details={"tool": request.tool_name, "error": str(exc)})
        return ToolResult(tool_name=request.tool_name, error=str(exc), elapsed_ms=elapsed)


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("COPILOT_MCP_PORT", "8005"))
    uvicorn.run("tools.github_copilot_mcp:app", host="0.0.0.0", port=port, reload=False)
