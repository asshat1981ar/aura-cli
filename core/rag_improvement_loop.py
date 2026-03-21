"""Continuous Self-Improvement Loop for the RAG Ingestion Pipeline.

Tracks retrieval performance after every CodeRAG retrieval call, analyses
aggregate metrics on a rolling window, tunes RAG hyper-parameters
dynamically, refines generation prompts based on outcome feedback, and
updates downstream workflow configurations when performance thresholds
are breached.

The loop is designed to be lightweight and *never* raise — callers should
not need try/except wrappers.

Usage::

    from core.rag_improvement_loop import RAGImprovementLoop
    loop = RAGImprovementLoop()

    # After every retrieve_context() call:
    loop.record_retrieval(
        goal="fix auth bug",
        retrieved_count=3,
        retrieval_ms=42.5,
        hit=True,
    )

    # After cycle outcome is known:
    loop.record_outcome(
        goal="fix auth bug",
        success=True,
        rag_was_used=True,
    )

    # Periodically trigger full analysis:
    report = loop.analyse()
"""

from __future__ import annotations

import collections
import time
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional

from core.logging_utils import log_json

# ---------------------------------------------------------------------------
# Tuneable constants
# ---------------------------------------------------------------------------

# Rolling-window size used for metric aggregation
_WINDOW = 50

# How many outcomes must accumulate before we attempt config tuning
_MIN_OUTCOMES_FOR_TUNING = 10

# How many outcomes must accumulate before we attempt prompt refinement
_MIN_OUTCOMES_FOR_PROMPT = 15

# Minimum outcomes for a single prompt variant before exploring the next one
_MIN_OUTCOMES_FOR_EXPLORATION = 5

# Minimum hit-rate below which we loosen similarity threshold
_HIT_RATE_LOW_THRESHOLD = 0.40

# Minimum hit-rate above which we tighten similarity threshold
_HIT_RATE_HIGH_THRESHOLD = 0.85

# Success-rate drop that triggers prompt refinement
_SUCCESS_RATE_LOW_THRESHOLD = 0.50

# Step sizes for dynamic similarity tuning
_SIMILARITY_STEP_UP = 0.05    # tighten when retrieval is too loose
_SIMILARITY_STEP_DOWN = 0.03  # loosen when hit rate is too low
_SIMILARITY_MIN = 0.30
_SIMILARITY_MAX = 0.85

# Step sizes for max_examples tuning
_EXAMPLES_STEP_UP = 1
_EXAMPLES_STEP_DOWN = 1
_EXAMPLES_MIN = 1
_EXAMPLES_MAX = 8

# ---------------------------------------------------------------------------
# Data-transfer objects
# ---------------------------------------------------------------------------


@dataclass
class RetrievalRecord:
    """A single RAG retrieval event."""

    goal: str
    retrieved_count: int
    retrieval_ms: float
    hit: bool                      # True if at least one example was returned
    timestamp: float = field(default_factory=time.time)
    similarity_threshold_used: float = 0.0


@dataclass
class OutcomeRecord:
    """A single cycle outcome linked to a RAG retrieval."""

    goal: str
    success: bool
    rag_was_used: bool
    timestamp: float = field(default_factory=time.time)
    prompt_version: int = 0        # tracks which prompt variant was active


@dataclass
class RAGMetrics:
    """Aggregated metrics computed over the rolling window."""

    window_size: int = 0
    hit_rate: float = 0.0
    avg_retrieval_ms: float = 0.0
    avg_retrieved_count: float = 0.0
    rag_success_rate: float = 0.0   # success rate when RAG was used
    baseline_success_rate: float = 0.0  # success rate when RAG was NOT used
    rag_lift: float = 0.0           # rag_success_rate - baseline_success_rate
    outcomes_analysed: int = 0
    current_similarity_threshold: float = 0.0
    current_max_examples: int = 0
    current_prompt_version: int = 0


@dataclass
class TuningAction:
    """A recorded parameter-tuning action."""

    action_type: str        # "similarity_threshold" | "max_examples" | "prompt_refinement"
    old_value: Any
    new_value: Any
    reason: str
    timestamp: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class RAGImprovementLoop:
    """Continuous self-improvement loop for the RAG ingestion pipeline.

    Responsibilities:
    * **Async metrics collection** — accumulates ``RetrievalRecord`` and
      ``OutcomeRecord`` objects from callers without blocking.
    * **Automated analysis** — computes ``RAGMetrics`` over a sliding window
      and detects performance drift.
    * **Dynamic config tuning** — adjusts ``similarity_threshold`` and
      ``max_examples`` parameters on the ``CodeRAG`` instance (or config dict)
      in response to observed performance.
    * **Self-prompt refinement** — cycles through registered prompt variants
      and selects the one with the highest observed success rate.
    * **Workflow updates** — pushes distilled insights back to the
      ``AdaptivePipeline`` and an optional config store so downstream
      consumers benefit from learned knowledge.
    """

    def __init__(
        self,
        rag_instance=None,
        config_store: Optional[Dict[str, Any]] = None,
        adaptive_pipeline=None,
        prompt_variants: Optional[List[str]] = None,
    ) -> None:
        """
        Args:
            rag_instance:     Live ``CodeRAG`` instance whose parameters will
                              be tuned in-place.  May be ``None`` — tuning
                              will only update ``config_store`` in that case.
            config_store:     Mutable dict that mirrors persisted RAG config
                              (``min_similarity``, ``max_examples``).  If
                              ``None`` an internal dict is used.
            adaptive_pipeline: ``AdaptivePipeline`` instance to notify with
                               workflow hints.  May be ``None``.
            prompt_variants:  Ordered list of prompt-suffix variants to A/B
                              test.  Defaults to built-in set.
        """
        self._rag = rag_instance
        self._config: Dict[str, Any] = config_store if config_store is not None else {}
        self._pipeline = adaptive_pipeline

        # Circular buffers for recent events
        self._retrievals: Deque[RetrievalRecord] = collections.deque(maxlen=_WINDOW)
        self._outcomes: Deque[OutcomeRecord] = collections.deque(maxlen=_WINDOW)

        # Tuning history
        self._tuning_log: List[TuningAction] = []

        # Current working parameters (start from rag_instance or defaults)
        self._similarity_threshold: float = self._read_rag_param(
            "min_similarity", default=0.6
        )
        self._max_examples: int = self._read_rag_param("max_examples", default=3)

        # Prompt A/B testing
        self._prompt_variants: List[str] = prompt_variants or _DEFAULT_PROMPT_VARIANTS
        self._prompt_version: int = 0          # index into _prompt_variants
        self._prompt_outcome_counts: Dict[int, Dict[str, int]] = {}  # version → {wins, total}

    # ── Public: event recording (non-blocking) ───────────────────────────────

    def record_retrieval(
        self,
        goal: str,
        retrieved_count: int,
        retrieval_ms: float,
        hit: bool,
        similarity_threshold_used: Optional[float] = None,
    ) -> None:
        """Record a single retrieval event.  Never raises."""
        try:
            rec = RetrievalRecord(
                goal=goal,
                retrieved_count=retrieved_count,
                retrieval_ms=retrieval_ms,
                hit=hit,
                similarity_threshold_used=similarity_threshold_used
                if similarity_threshold_used is not None
                else self._similarity_threshold,
            )
            self._retrievals.append(rec)
        except Exception as exc:
            log_json("WARN", "rag_improvement_record_retrieval_failed",
                     details={"error": str(exc)})

    def record_outcome(
        self,
        goal: str,
        success: bool,
        rag_was_used: bool,
    ) -> None:
        """Record a cycle outcome.  Never raises."""
        try:
            rec = OutcomeRecord(
                goal=goal,
                success=success,
                rag_was_used=rag_was_used,
                prompt_version=self._prompt_version,
            )
            self._outcomes.append(rec)
            self._update_prompt_counts(self._prompt_version, success)
        except Exception as exc:
            log_json("WARN", "rag_improvement_record_outcome_failed",
                     details={"error": str(exc)})

    # ── Public: analysis and tuning ──────────────────────────────────────────

    def analyse(self) -> Dict[str, Any]:
        """Compute metrics, tune parameters, and return a report.  Never raises."""
        try:
            return self._analyse()
        except Exception as exc:
            log_json("ERROR", "rag_improvement_analyse_failed",
                     details={"error": str(exc)})
            return {"error": str(exc)}

    def current_prompt_suffix(self) -> str:
        """Return the currently active prompt suffix variant."""
        if not self._prompt_variants:
            return ""
        return self._prompt_variants[self._prompt_version % len(self._prompt_variants)]

    def get_metrics(self) -> RAGMetrics:
        """Return a snapshot of current aggregate metrics.  Never raises."""
        try:
            return self._compute_metrics()
        except Exception as exc:
            log_json("WARN", "rag_improvement_get_metrics_failed",
                     details={"error": str(exc)})
            return RAGMetrics()

    def get_tuning_log(self) -> List[TuningAction]:
        """Return a copy of all tuning actions taken so far."""
        return list(self._tuning_log)

    # ── Internal: core analysis pipeline ────────────────────────────────────

    def _analyse(self) -> Dict[str, Any]:
        metrics = self._compute_metrics()
        actions_taken: List[str] = []

        # ── 1. Dynamic config tuning ────────────────────────────────────────
        if metrics.outcomes_analysed >= _MIN_OUTCOMES_FOR_TUNING:
            tuning_actions = self._tune_config(metrics)
            actions_taken.extend(tuning_actions)

        # ── 2. Self-prompt refinement ────────────────────────────────────────
        if metrics.outcomes_analysed >= _MIN_OUTCOMES_FOR_PROMPT:
            prompt_action = self._refine_prompt(metrics)
            if prompt_action:
                actions_taken.append(prompt_action)

        # ── 3. Workflow updates ──────────────────────────────────────────────
        workflow_hints = self._build_workflow_hints(metrics)
        if workflow_hints and self._pipeline:
            self._push_workflow_hints(workflow_hints)
            actions_taken.append(f"pushed {len(workflow_hints)} workflow hint(s)")

        report = {
            "metrics": {
                "window_size": metrics.window_size,
                "hit_rate": round(metrics.hit_rate, 3),
                "avg_retrieval_ms": round(metrics.avg_retrieval_ms, 1),
                "avg_retrieved_count": round(metrics.avg_retrieved_count, 2),
                "rag_success_rate": round(metrics.rag_success_rate, 3),
                "baseline_success_rate": round(metrics.baseline_success_rate, 3),
                "rag_lift": round(metrics.rag_lift, 3),
                "outcomes_analysed": metrics.outcomes_analysed,
            },
            "current_config": {
                "similarity_threshold": self._similarity_threshold,
                "max_examples": self._max_examples,
                "prompt_version": self._prompt_version,
            },
            "actions_taken": actions_taken,
        }

        log_json("INFO", "rag_improvement_analyse_complete", details=report)
        return report

    # ── Internal: metric computation ─────────────────────────────────────────

    def _compute_metrics(self) -> RAGMetrics:
        retrievals = list(self._retrievals)
        outcomes = list(self._outcomes)

        m = RAGMetrics(
            current_similarity_threshold=self._similarity_threshold,
            current_max_examples=self._max_examples,
            current_prompt_version=self._prompt_version,
        )

        if retrievals:
            m.window_size = len(retrievals)
            m.hit_rate = sum(1 for r in retrievals if r.hit) / len(retrievals)
            m.avg_retrieval_ms = sum(r.retrieval_ms for r in retrievals) / len(retrievals)
            m.avg_retrieved_count = sum(r.retrieved_count for r in retrievals) / len(retrievals)

        if outcomes:
            m.outcomes_analysed = len(outcomes)
            rag_outcomes = [o for o in outcomes if o.rag_was_used]
            no_rag_outcomes = [o for o in outcomes if not o.rag_was_used]

            if rag_outcomes:
                m.rag_success_rate = sum(1 for o in rag_outcomes if o.success) / len(rag_outcomes)
            if no_rag_outcomes:
                m.baseline_success_rate = sum(1 for o in no_rag_outcomes if o.success) / len(no_rag_outcomes)
            m.rag_lift = m.rag_success_rate - m.baseline_success_rate

        return m

    # ── Internal: dynamic config tuning ──────────────────────────────────────

    def _tune_config(self, metrics: RAGMetrics) -> List[str]:
        """Adjust similarity_threshold and max_examples based on observed metrics."""
        actions: List[str] = []

        # ── Similarity threshold ─────────────────────────────────────────────
        if metrics.hit_rate < _HIT_RATE_LOW_THRESHOLD:
            # Too few hits — loosen the similarity threshold
            new_threshold = max(
                _SIMILARITY_MIN,
                round(self._similarity_threshold - _SIMILARITY_STEP_DOWN, 3),
            )
            if new_threshold != self._similarity_threshold:
                self._apply_similarity_threshold(
                    new_threshold,
                    reason=f"hit_rate={metrics.hit_rate:.2f} < {_HIT_RATE_LOW_THRESHOLD}",
                )
                actions.append(f"similarity_threshold: {self._similarity_threshold} (loosened)")

        elif metrics.hit_rate > _HIT_RATE_HIGH_THRESHOLD and metrics.rag_success_rate < _SUCCESS_RATE_LOW_THRESHOLD:
            # Plenty of hits but they aren't helping — tighten threshold
            new_threshold = min(
                _SIMILARITY_MAX,
                round(self._similarity_threshold + _SIMILARITY_STEP_UP, 3),
            )
            if new_threshold != self._similarity_threshold:
                self._apply_similarity_threshold(
                    new_threshold,
                    reason=f"hit_rate={metrics.hit_rate:.2f} high but success_rate={metrics.rag_success_rate:.2f} low",
                )
                actions.append(f"similarity_threshold: {self._similarity_threshold} (tightened)")

        # ── max_examples ─────────────────────────────────────────────────────
        if metrics.rag_lift < -0.10:
            # RAG is hurting — reduce examples to reduce noise
            new_max = max(_EXAMPLES_MIN, self._max_examples - _EXAMPLES_STEP_DOWN)
            if new_max != self._max_examples:
                self._apply_max_examples(
                    new_max,
                    reason=f"rag_lift={metrics.rag_lift:.2f} negative",
                )
                actions.append(f"max_examples: {self._max_examples} (reduced)")

        elif metrics.rag_lift > 0.15 and metrics.avg_retrieved_count < self._max_examples * 0.5:
            # RAG is helping but we're retrieving fewer examples than allowed
            new_max = min(_EXAMPLES_MAX, self._max_examples + _EXAMPLES_STEP_UP)
            if new_max != self._max_examples:
                self._apply_max_examples(
                    new_max,
                    reason=f"rag_lift={metrics.rag_lift:.2f} positive, utilisation low",
                )
                actions.append(f"max_examples: {self._max_examples} (increased)")

        return actions

    def _apply_similarity_threshold(self, new_value: float, reason: str) -> None:
        old_value = self._similarity_threshold
        self._similarity_threshold = new_value
        self._config["min_similarity"] = new_value
        if self._rag is not None:
            self._rag.min_similarity = new_value
        self._tuning_log.append(
            TuningAction("similarity_threshold", old_value, new_value, reason)
        )
        log_json("INFO", "rag_config_tuned", details={
            "param": "min_similarity",
            "old": old_value,
            "new": new_value,
            "reason": reason,
        })

    def _apply_max_examples(self, new_value: int, reason: str) -> None:
        old_value = self._max_examples
        self._max_examples = new_value
        self._config["max_examples"] = new_value
        if self._rag is not None:
            self._rag.max_examples = new_value
        self._tuning_log.append(
            TuningAction("max_examples", old_value, new_value, reason)
        )
        log_json("INFO", "rag_config_tuned", details={
            "param": "max_examples",
            "old": old_value,
            "new": new_value,
            "reason": reason,
        })

    # ── Internal: prompt refinement ───────────────────────────────────────────

    def _refine_prompt(self, metrics: RAGMetrics) -> Optional[str]:
        """Select the best prompt variant based on observed win rates."""
        if len(self._prompt_variants) <= 1:
            return None

        # Find the variant with the highest win rate.
        # A variant is considered only if it has accumulated at least 3 outcomes
        # (per-variant minimum — separate from the _MIN_OUTCOMES_FOR_PROMPT total
        # threshold that gates the entire prompt-refinement phase).
        best_version = self._prompt_version
        best_rate = self._win_rate_for_version(self._prompt_version)

        for version in range(len(self._prompt_variants)):
            rate = self._win_rate_for_version(version)
            counts = self._prompt_outcome_counts.get(version, {})
            if counts.get("total", 0) >= 3 and rate > best_rate:
                best_rate = rate
                best_version = version

        # If the current version is underperforming and has enough data,
        # explore the next variant (using _MIN_OUTCOMES_FOR_EXPLORATION as the
        # per-variant sample-size threshold).
        current_counts = self._prompt_outcome_counts.get(self._prompt_version, {})
        if (
            best_version == self._prompt_version
            and current_counts.get("total", 0) >= _MIN_OUTCOMES_FOR_EXPLORATION
            and best_rate < _SUCCESS_RATE_LOW_THRESHOLD
        ):
            # Explore next variant
            best_version = (self._prompt_version + 1) % len(self._prompt_variants)

        if best_version != self._prompt_version:
            old_version = self._prompt_version
            self._prompt_version = best_version
            self._tuning_log.append(
                TuningAction(
                    "prompt_refinement",
                    old_version,
                    best_version,
                    f"win_rate improved from {self._win_rate_for_version(old_version):.2f} to {best_rate:.2f}",
                )
            )
            log_json("INFO", "rag_prompt_refined", details={
                "old_version": old_version,
                "new_version": best_version,
                "win_rate": round(best_rate, 3),
            })
            return f"prompt switched to variant {best_version}"

        return None

    def _update_prompt_counts(self, version: int, success: bool) -> None:
        counts = self._prompt_outcome_counts.setdefault(version, {"wins": 0, "total": 0})
        counts["total"] += 1
        if success:
            counts["wins"] += 1

    def _win_rate_for_version(self, version: int) -> float:
        counts = self._prompt_outcome_counts.get(version, {})
        total = counts.get("total", 0)
        if total == 0:
            return 0.0
        return counts.get("wins", 0) / total

    # ── Internal: workflow updates ────────────────────────────────────────────

    def _build_workflow_hints(self, metrics: RAGMetrics) -> List[str]:
        """Produce structured hints for the AdaptivePipeline."""
        hints: List[str] = []

        if metrics.rag_lift > 0.10:
            hints.append(
                f"RAG retrieval improves success rate by {metrics.rag_lift:.0%}; "
                "prioritise code_rag in act phase"
            )
        if metrics.hit_rate < _HIT_RATE_LOW_THRESHOLD:
            hints.append(
                f"RAG hit rate is {metrics.hit_rate:.0%}; consider broadening goal descriptions "
                "to improve retrieval recall"
            )
        if metrics.avg_retrieval_ms > 500:
            hints.append(
                f"RAG retrieval is slow ({metrics.avg_retrieval_ms:.0f} ms avg); "
                "check vector store index health"
            )
        if metrics.rag_lift < -0.05:
            hints.append(
                "RAG context appears to be adding noise; verify example quality in vector store"
            )
        return hints

    def _push_workflow_hints(self, hints: List[str]) -> None:
        """Inject hints into the AdaptivePipeline's context graph if possible."""
        if not self._pipeline:
            return
        try:
            graph = getattr(self._pipeline, "graph", None)
            if graph and hasattr(graph, "add_hint"):
                for hint in hints:
                    graph.add_hint("rag_improvement", hint)
        except Exception as exc:
            log_json("WARN", "rag_improvement_workflow_push_failed",
                     details={"error": str(exc)})

    # ── Internal: helpers ─────────────────────────────────────────────────────

    def _read_rag_param(self, attr: str, default: Any) -> Any:
        if self._rag is not None:
            return getattr(self._rag, attr, default)
        return self._config.get(attr, default)


# ---------------------------------------------------------------------------
# Default prompt variants for A/B testing
# ---------------------------------------------------------------------------

_DEFAULT_PROMPT_VARIANTS: List[str] = [
    # Variant 0 — baseline (no extra instructions)
    "",
    # Variant 1 — emphasise matching patterns
    "\nPrioritise code patterns that most closely match the retrieved examples above.",
    # Variant 2 — emphasise avoiding past failures
    "\nAvoid the failure patterns listed above. Focus on a minimal, targeted change.",
    # Variant 3 — chain-of-thought
    "\nThink step by step: first identify the affected code path, "
    "then consult the retrieved examples, then write the fix.",
]
