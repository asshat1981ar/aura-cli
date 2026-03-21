"""Skill correlation matrix: self-organizing skill system.

Tracks which skills succeed together across cycles, builds a correlation matrix,
and suggests optimal skill pairings for future dispatches. This enables AURA
to discover emergent skill clusters without manual configuration.
"""
import json
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from core.logging_utils import log_json

@dataclass
class SkillOutcome:
    """Outcome of a skill execution in a cycle."""
    skill_name: str
    goal_type: str
    success: bool
    latency_ms: float = 0.0
    timestamp: float = field(default_factory=time.time)

class SkillCorrelationMatrix:
    """Tracks skill co-success patterns and suggests optimal pairings."""

    def __init__(self, store_path: Path | None = None):
        self.store_path = store_path or Path(__file__).parent.parent / "memory" / "skill_correlations.json"
        # correlation_matrix[skill_a][skill_b] = {"co_success": int, "co_failure": int, "total": int}
        self.matrix: dict[str, dict[str, dict[str, int]]] = defaultdict(lambda: defaultdict(lambda: {"co_success": 0, "co_failure": 0, "total": 0}))
        # Skill success rates per goal type
        self.skill_rates: dict[str, dict[str, dict[str, int]]] = defaultdict(lambda: defaultdict(lambda: {"success": 0, "total": 0}))
        # Discovered clusters
        self.clusters: list[list[str]] = []
        self._load()

    def record_cycle(self, outcomes: list[SkillOutcome], cycle_success: bool):
        """Record skill outcomes from a cycle and update correlations."""
        # Update pairwise correlations
        for i, a in enumerate(outcomes):
            # Update individual rates
            self.skill_rates[a.goal_type][a.skill_name]["total"] += 1
            if a.success:
                self.skill_rates[a.goal_type][a.skill_name]["success"] += 1

            for j, b in enumerate(outcomes):
                if i >= j:
                    continue
                pair = self.matrix[a.skill_name][b.skill_name]
                pair["total"] += 1
                if a.success and b.success and cycle_success:
                    pair["co_success"] += 1
                elif not cycle_success:
                    pair["co_failure"] += 1
                # Mirror
                mirror = self.matrix[b.skill_name][a.skill_name]
                mirror["total"] = pair["total"]
                mirror["co_success"] = pair["co_success"]
                mirror["co_failure"] = pair["co_failure"]

        self._save()

    def get_correlation(self, skill_a: str, skill_b: str) -> float:
        """Get correlation score between two skills (-1.0 to 1.0)."""
        pair = self.matrix.get(skill_a, {}).get(skill_b, {})
        total = pair.get("total", 0)
        if total == 0:
            return 0.0
        co_success = pair.get("co_success", 0)
        co_failure = pair.get("co_failure", 0)
        return (co_success - co_failure) / total

    def suggest_skills(self, base_skills: list[str], goal_type: str = "", top_k: int = 3) -> list[tuple[str, float]]:
        """Suggest additional skills based on correlation with base skills."""
        all_skills = set()
        for skill in base_skills:
            for correlated in self.matrix.get(skill, {}):
                if correlated not in base_skills:
                    all_skills.add(correlated)

        # Score each candidate by average correlation with base skills
        scored = []
        for candidate in all_skills:
            avg_corr = sum(self.get_correlation(base, candidate) for base in base_skills) / max(len(base_skills), 1)
            if avg_corr > 0:
                scored.append((candidate, avg_corr))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def discover_clusters(self, min_correlation: float = 0.5, min_size: int = 2) -> list[list[str]]:
        """Discover skill clusters (groups that frequently succeed together)."""
        all_skills = list(self.matrix.keys())
        visited = set()
        clusters = []

        for skill in all_skills:
            if skill in visited:
                continue
            cluster = [skill]
            visited.add(skill)

            for other in all_skills:
                if other in visited:
                    continue
                if self.get_correlation(skill, other) >= min_correlation:
                    cluster.append(other)
                    visited.add(other)

            if len(cluster) >= min_size:
                clusters.append(sorted(cluster))

        self.clusters = clusters
        log_json("INFO", "skill_clusters_discovered", details={"clusters": len(clusters), "skills": clusters})
        return clusters

    def get_skill_success_rate(self, skill_name: str, goal_type: str = "") -> float:
        """Get success rate for a skill, optionally filtered by goal type."""
        if goal_type and goal_type in self.skill_rates:
            rates = self.skill_rates[goal_type].get(skill_name, {})
        else:
            # Aggregate across all goal types
            total = success = 0
            for gt_rates in self.skill_rates.values():
                if skill_name in gt_rates:
                    total += gt_rates[skill_name]["total"]
                    success += gt_rates[skill_name]["success"]
            return success / max(total, 1)

        return rates.get("success", 0) / max(rates.get("total", 1), 1)

    def get_summary(self) -> dict:
        """Get summary for CLI display."""
        all_pairs = []
        seen = set()
        for a in self.matrix:
            for b in self.matrix[a]:
                pair_key = tuple(sorted([a, b]))
                if pair_key not in seen:
                    seen.add(pair_key)
                    corr = self.get_correlation(a, b)
                    if abs(corr) > 0.1:
                        all_pairs.append({"skills": list(pair_key), "correlation": round(corr, 3)})

        all_pairs.sort(key=lambda x: abs(x["correlation"]), reverse=True)
        return {
            "total_skills_tracked": len(self.matrix),
            "total_pairs": len(seen),
            "top_correlations": all_pairs[:10],
            "clusters": self.clusters,
        }

    def _load(self):
        if not self.store_path.exists():
            return
        try:
            data = json.loads(self.store_path.read_text())
            # Reconstruct defaultdicts from plain dicts
            for a, pairs in data.get("matrix", {}).items():
                for b, stats in pairs.items():
                    self.matrix[a][b] = stats
            for gt, skills in data.get("skill_rates", {}).items():
                for skill, rates in skills.items():
                    self.skill_rates[gt][skill] = rates
            self.clusters = data.get("clusters", [])
        except (json.JSONDecodeError, OSError):
            pass

    def _save(self):
        data = {
            "matrix": {a: dict(pairs) for a, pairs in self.matrix.items()},
            "skill_rates": {gt: dict(skills) for gt, skills in self.skill_rates.items()},
            "clusters": self.clusters,
            "updated_at": time.time(),
        }
        try:
            self.store_path.parent.mkdir(parents=True, exist_ok=True)
            self.store_path.write_text(json.dumps(data, indent=2))
        except OSError as exc:
            log_json("WARN", "skill_correlation_save_failed", details={"error": str(exc)})
