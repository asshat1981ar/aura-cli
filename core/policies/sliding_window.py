from core.policies.base import PolicyBase


class SlidingWindowPolicy(PolicyBase):
    def __init__(self, max_cycles: int = 5):
        self.max_cycles = max_cycles

    def evaluate(self, history, verification, started_at=None):
        if verification.get("status") == "pass":
            return "PASS"
        if len(history) >= self.max_cycles:
            return "MAX_CYCLES"
        return ""
