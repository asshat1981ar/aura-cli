import os
import re
from pathlib import Path

from agents.ingest import IngestAgent
from agents.verifier import VerifierAgent
from agents.synthesizer import SynthesizerAgent
from agents.reflector import ReflectorAgent
from agents.planner import PlannerAgent
from agents.critic import CriticAgent
from agents.coder import CoderAgent
from agents.sandbox import SandboxAgent
from agents.tester import TesterAgent


class PlannerAdapter:
    name = "plan"

    def __init__(self, agent: PlannerAgent):
        self.agent = agent

    def run(self, input_data):
        goal = input_data.get("goal", "")
        memory_snapshot = input_data.get("memory_snapshot", "")
        similar = input_data.get("similar_past_problems", "")
        weaknesses = input_data.get("known_weaknesses", "")
        steps = self.agent.plan(goal, memory_snapshot, similar, weaknesses)
        return {"steps": steps, "risks": []}


class CriticAdapter:
    name = "critique"

    def __init__(self, agent: CriticAgent):
        self.agent = agent

    def run(self, input_data):
        task = input_data.get("task", "")
        plan = input_data.get("plan", [])
        critique = self.agent.critique_plan(task, plan)
        return {"issues": [critique], "fixes": []}


class ActAdapter:
    name = "act"

    def __init__(self, agent: CoderAgent):
        self.agent = agent

    def _keywords(self, *texts):
        words = set()
        for text in texts:
            for token in re.split(r"[^a-zA-Z0-9_]+", text or ""):
                token = token.strip().lower()
                if len(token) >= 3:
                    words.add(token)
        return sorted(words)

    def _score_path(self, path: str, keywords) -> int:
        lower = path.lower()
        return sum(1 for k in keywords if k in lower)

    def _choose_generated_name(self, directory: Path, keywords) -> str:
        base = "aura_generated"
        if keywords:
            base = f"aura_{keywords[0][:24]}"
        candidate = directory / f"{base}.py"
        if not candidate.exists():
            return str(candidate)
        for i in range(1, 50):
            alt = directory / f"{base}_{i}.py"
            if not alt.exists():
                return str(alt)
        return str(directory / f"{base}_new.py")

    def _choose_file_path(self, task: str, task_bundle: dict, project_root: Path) -> str:
        tasks = task_bundle.get("tasks", []) if isinstance(task_bundle, dict) else []
        intent = ""
        files = []
        if tasks:
            intent = tasks[0].get("intent", "")
            files = tasks[0].get("files", []) or []

        keywords = self._keywords(task, intent)
        candidates = []

        for entry in files:
            if not entry:
                continue
            entry_path = (project_root / entry).resolve()
            if entry.endswith("/") or entry_path.is_dir():
                for path in entry_path.rglob("*.py"):
                    if ".git" in path.parts or "__pycache__" in path.parts:
                        continue
                    candidates.append(str(path.relative_to(project_root)))
            else:
                candidates.append(entry)

        if not candidates:
            for base in ["core", "agents", "memory"]:
                entry_path = (project_root / base)
                if entry_path.is_dir():
                    for path in entry_path.rglob("*.py"):
                        if ".git" in path.parts or "__pycache__" in path.parts:
                            continue
                        candidates.append(str(path.relative_to(project_root)))

        if candidates:
            scored = sorted(
                ((self._score_path(p, keywords), p) for p in candidates),
                reverse=True,
            )
            best_score, best_path = scored[0]
            if best_score > 0:
                return best_path

        # Fallback to a generated file in core/
        core_dir = project_root / "core"
        core_dir.mkdir(parents=True, exist_ok=True)
        return self._choose_generated_name(core_dir, keywords)

    def run(self, input_data):
        task = input_data.get("task", "")
        task_bundle = input_data.get("task_bundle", {}) or {}
        project_root = Path(input_data.get("project_root", Path.cwd()))
        code = self.agent.implement(task)
        file_path = ""
        new_code = code
        if code:
            lines = code.splitlines()
            if lines and lines[0].startswith(self.agent.AURA_TARGET_DIRECTIVE):
                file_path = lines[0].replace(self.agent.AURA_TARGET_DIRECTIVE, "").strip()
                new_code = "\n".join(lines[1:]).lstrip()
        if not file_path and isinstance(task_bundle, dict):
            file_path = self._choose_file_path(task, task_bundle, project_root)
        return {
            "changes": [
                {
                    "file_path": file_path,
                    "old_code": "",
                    "new_code": new_code,
                    "overwrite_file": True,
                }
            ]
        }


class SandboxAdapter:
    """Wraps SandboxAgent as a pipeline phase (name: 'sandbox').

    Extracts the generated code snippet from the act phase output and runs it
    in an isolated subprocess.  Returns a lightweight result dict that the
    orchestrator can use to decide whether to proceed to the verify phase.

    Input keys consumed:
        ``act``           — dict from ActAdapter.run() (contains ``changes``)
        ``dry_run``       — bool, skip execution when True
        ``project_root``  — str path

    Output keys:
        ``status``  — "pass" | "fail" | "skip"
        ``passed``  — bool
        ``summary`` — human-readable one-liner
        ``details`` — dict with stdout / stderr / exit_code
    """

    name = "sandbox"

    def __init__(self, agent: SandboxAgent):
        self.agent = agent

    def run(self, input_data: dict) -> dict:
        if input_data.get("dry_run"):
            return {"status": "skip", "passed": True, "summary": "dry_run", "details": {}}

        act_output = input_data.get("act") or {}
        changes = act_output.get("changes") or []

        # Collect all new_code snippets to validate
        snippets = [c.get("new_code", "") for c in changes if c.get("new_code")]
        if not snippets:
            return {"status": "skip", "passed": True,
                    "summary": "no_code_to_sandbox", "details": {}}

        results = []
        for snippet in snippets:
            if not snippet.strip():
                continue
            res = self.agent.run_code(snippet)
            results.append(res)

        if not results:
            return {"status": "skip", "passed": True,
                    "summary": "empty_snippets", "details": {}}

        # Aggregate: all must pass
        all_pass = all(r.passed for r in results)
        first = results[0]
        return {
            "status": "pass" if all_pass else "fail",
            "passed": all_pass,
            "summary": first.summary(),
            "details": {
                "exit_code": first.exit_code,
                "stdout": first.stdout[:500],
                "stderr": first.stderr[:500],
                "timed_out": first.timed_out,
                "snippet_count": len(results),
            },
        }


def default_agents(brain, model):
    sandbox_agent = SandboxAgent(brain, timeout=30)
    planner = PlannerAdapter(PlannerAgent(brain, model))
    critic = CriticAdapter(CriticAgent(brain, model))
    act = ActAdapter(CoderAgent(brain, model))
    sandbox = SandboxAdapter(sandbox_agent)
    return {
        "ingest": IngestAgent(brain),
        "plan": planner,
        "critique": critic,
        "synthesize": SynthesizerAgent(),
        "act": act,
        "sandbox": sandbox,
        "verify": VerifierAgent(),
        "reflect": ReflectorAgent(),
    }
