class FakeAgent:
    def __init__(self, output):
        self.output = output

    def run(self, input_data):
        return self.output


def make_fake_agents():
    return {
        "ingest": FakeAgent({
            "goal": "demo",
            "snapshot": "file.py",
            "memory_summary": "",
            "constraints": {},
        }),
        "plan": FakeAgent({
            "steps": ["step 1"],
            "risks": [],
        }),
        "critique": FakeAgent({
            "issues": [],
            "fixes": [],
        }),
        "synthesize": FakeAgent({
            "tasks": [{"id": "t1", "title": "demo", "intent": "", "files": [], "tests": []}],
        }),
        "act": FakeAgent({
            "changes": [],
        }),
        "verify": FakeAgent({
            "status": "pass",
            "failures": [],
            "logs": "",
        }),
        "reflect": FakeAgent({
            "summary": "ok",
            "learnings": [],
            "next_actions": [],
        }),
    }
