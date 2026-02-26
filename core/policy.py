from core.policies.sliding_window import SlidingWindowPolicy
from core.policies.time_bound import TimeBoundPolicy
from core.policies.resource_bound import ResourceBoundPolicy


class Policy:
    def __init__(self, impl=None, max_cycles=5, max_seconds=None):
        if impl is not None:
            self.impl = impl
        else:
            if max_seconds is not None:
                self.impl = TimeBoundPolicy(max_seconds=max_seconds)
            else:
                self.impl = SlidingWindowPolicy(max_cycles=max_cycles)

    @classmethod
    def from_config(cls, config):
        name = config.get("policy_name", "sliding_window")
        if name == "time_bound":
            return cls(TimeBoundPolicy(max_seconds=config.get("policy_max_seconds", 120)))
        if name == "resource_bound":
            return cls(ResourceBoundPolicy(max_tokens=config.get("policy_max_tokens", 50000)))
        return cls(SlidingWindowPolicy(max_cycles=config.get("policy_max_cycles", 5)))

    def evaluate(self, history, verification, started_at=None):
        return self.impl.evaluate(history, verification, started_at=started_at)
