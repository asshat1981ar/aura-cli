import time
import json
from dataclasses import dataclass, asdict
from typing import Optional, Callable
from core.logging_utils import log_json # Import log_json




# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ModelStats:
    name: str
    success_count: int = 0
    failure_count: int = 0
    total_latency: float = 0.0
    ema_score: float = 0.75           # Start optimistic — no penalty before first call
    consecutive_failures: int = 0
    cooldown_until: float = 0.0       # Unix timestamp

    EMA_ALPHA: float = 0.2            # Weight of latest observation in EMA
    COOLDOWN_SECONDS: float = 120.0   # Backoff window after 3 consecutive failures
    FAILURE_THRESHOLD: int = 3

    @property
    def is_cooled_down(self) -> bool:
        return time.time() >= self.cooldown_until

    @property
    def avg_latency(self) -> float:
        total = self.success_count + self.failure_count
        return self.total_latency / total if total > 0 else 999.0

    def record(self, success: bool, latency: float):
        observation = 1.0 if success else 0.0
        self.ema_score = self.EMA_ALPHA * observation + (1 - self.EMA_ALPHA) * self.ema_score
        self.total_latency += latency

        if success:
            self.consecutive_failures = 0
            self.success_count += 1
        else:
            self.failure_count += 1
            self.consecutive_failures += 1
            if self.consecutive_failures >= self.FAILURE_THRESHOLD:
                self.cooldown_until = time.time() + self.COOLDOWN_SECONDS

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ModelStats":
        obj = cls(name=d["name"])
        for k, v in d.items():
            setattr(obj, k, v)
        return obj

    def __str__(self):
        status = "COOLDOWN" if not self.is_cooled_down else "active"
        return (
            f"[{self.name}] ema={self.ema_score:.3f} "
            f"ok={self.success_count} fail={self.failure_count} "
            f"lat={self.avg_latency:.1f}s [{status}]"
        )


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

class RouterAgent:
    """
    Adaptive model router with EMA-based scoring and cooldown management.

    Usage
    -----
    router = RouterAgent(brain, model_adapter)
    response = router.route(prompt)

    RouterAgent wraps ModelAdapter's individual callers (call_openai,
    call_gemini, call_openrouter) and replaces the static chain in
    model_adapter.respond() with empirically-ranked selection.
    """

    # Map logical name → ModelAdapter method name
    MODEL_REGISTRY = {
        "openai":     "call_openai",
        "gemini":     "call_gemini",
        "openrouter": "call_openrouter",
        "local":      "call_local",
    }

    # Short prompts stay local/fast; long prompts need more capable models
    SHORT_PROMPT_THRESHOLD = 500    # characters
    LONG_PROMPT_THRESHOLD  = 3000

    def __init__(self, brain, model_adapter, enabled_models: Optional[list] = None):
        self.brain = brain
        self.adapter = model_adapter
        self.enabled = enabled_models or ["openai", "gemini", "openrouter"]

        # Load persisted stats from brain or initialise fresh
        self.stats: dict[str, ModelStats] = {}
        self._load_stats()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def route(self, prompt: str) -> str:
        """
        Route prompt to the best available model.
        Tries ranked candidates in order until one succeeds.
        Returns the model response string.
        Raises RuntimeError if all candidates fail.
        """
        candidates = self._rank_candidates(prompt)
        last_error = None

        for model_name in candidates:
            caller = self._get_caller(model_name)
            if caller is None:
                continue

            start = time.time()
            try:
                response = caller(prompt)
                latency = time.time() - start
                self.stats[model_name].record(success=True, latency=latency)
                self._persist_stats()
                self.brain.remember(
                    f"RouterAgent: routed to {model_name} | "
                    f"lat={latency:.2f}s | ema={self.stats[model_name].ema_score:.3f}"
                )
                return response

            except Exception as exc:
                latency = time.time() - start
                self.stats[model_name].record(success=False, latency=latency)
                self._persist_stats()
                last_error = exc
                self.brain.remember(
                    f"RouterAgent: {model_name} FAILED ({exc}) | "
                    f"ema={self.stats[model_name].ema_score:.3f}"
                )
                # Continue to next candidate

        raise RuntimeError(
            f"RouterAgent: all candidates exhausted. Last error: {last_error}\n"
            f"Stats: {self.report()}"
        )

    def report(self) -> str:
        """Human-readable ranking and stats for all tracked models."""
        lines = ["RouterAgent Model Rankings:"]
        for name, stat in sorted(
            self.stats.items(), key=lambda x: x[1].ema_score, reverse=True
        ):
            lines.append(f"  {stat}")
        return "\n".join(lines)

    def force_cooldown(self, model_name: str, seconds: float = 300.0):
        """Manually bench a model (e.g., after detecting quota exhaustion)."""
        if model_name in self.stats:
            self.stats[model_name].cooldown_until = time.time() + seconds
            self.brain.remember(f"RouterAgent: forced cooldown on {model_name} for {seconds}s")

    # ------------------------------------------------------------------
    # Ranking logic
    # ------------------------------------------------------------------

    def _rank_candidates(self, prompt: str) -> list:
        """
        Return an ordered list of model names to try, best-first.

        Strategy:
        1. Filter out cooled-down models
        2. Apply contextual bias (prompt length hints)
        3. Sort by EMA score, break ties by latency (lower is better)
        """
        available = [
            name for name in self.enabled
            if name in self.stats and self.stats[name].is_cooled_down
        ]

        # Contextual bias: long prompts prefer higher-capacity models
        prompt_len = len(prompt)
        if prompt_len > self.LONG_PROMPT_THRESHOLD:
            # Bump openai's effective score for long prompts
            def score(name):
                base = self.stats[name].ema_score
                bonus = 0.15 if name == "openai" else 0.0
                return (base + bonus, -self.stats[name].avg_latency)
        elif prompt_len < self.SHORT_PROMPT_THRESHOLD:
            # Prefer lower latency for quick tasks
            def score(name):
                return (self.stats[name].ema_score, -self.stats[name].avg_latency * 2)
        else:
            def score(name):
                return (self.stats[name].ema_score, -self.stats[name].avg_latency)

        return sorted(available, key=score, reverse=True)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _persist_stats(self):
        payload = {name: stat.to_dict() for name, stat in self.stats.items()}
        self.brain.remember(f"__router_stats__:{json.dumps(payload)}")

    def _load_stats(self):
        # Initialise all enabled models with defaults first
        for name in self.enabled:
            self.stats[name] = ModelStats(name=name)

        # Try to restore from brain memory
        try:
            for entry in reversed(self.brain.recall_all()):
                if entry.startswith("__router_stats__:"):
                    payload = json.loads(entry[len("__router_stats__:"):])
                    for name, d in payload.items():
                        if name in self.enabled:
                            self.stats[name] = ModelStats.from_dict(d)
                    break   # Only care about the most recent snapshot
        except (json.JSONDecodeError, KeyError) as e:
            log_json("ERROR", "router_load_stats_failed", details={"error": str(e), "message": "Could not load model stats from brain memory."})
        except Exception as e:
            log_json("ERROR", "router_load_stats_unexpected_error", details={"error": str(e)})

    def _get_caller(self, model_name: str) -> Optional[Callable]:
        method_name = self.MODEL_REGISTRY.get(model_name)
        if not method_name:
            return None
        return getattr(self.adapter, method_name, None)