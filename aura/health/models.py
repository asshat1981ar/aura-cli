"""Data models for health monitoring."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class HealthStatus(Enum):
    """Overall health status."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class CheckType(Enum):
    """Type of health check."""

    SYSTEM = "system"
    DATABASE = "database"
    API = "api"
    DISK = "disk"
    MEMORY = "memory"
    CPU = "cpu"
    CUSTOM = "custom"


@dataclass
class CheckResult:
    """Result of a single health check."""

    name: str
    type: CheckType
    status: HealthStatus
    response_time_ms: float
    message: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)

    @property
    def is_healthy(self) -> bool:
        return self.status == HealthStatus.HEALTHY


@dataclass
class HealthReport:
    """Complete health report."""

    status: HealthStatus
    checks: List[CheckResult]
    timestamp: datetime = field(default_factory=datetime.utcnow)
    duration_ms: float = 0.0

    @property
    def healthy_count(self) -> int:
        return sum(1 for c in self.checks if c.is_healthy)

    @property
    def total_count(self) -> int:
        return len(self.checks)

    @property
    def failed_checks(self) -> List[CheckResult]:
        return [c for c in self.checks if not c.is_healthy]

    def to_dict(self) -> dict:
        return {
            "status": self.status.value,
            "timestamp": self.timestamp.isoformat(),
            "duration_ms": self.duration_ms,
            "summary": {
                "total": self.total_count,
                "healthy": self.healthy_count,
                "failed": self.total_count - self.healthy_count,
            },
            "checks": [
                {
                    "name": c.name,
                    "type": c.type.value,
                    "status": c.status.value,
                    "response_time_ms": c.response_time_ms,
                    "message": c.message,
                    "details": c.details,
                }
                for c in self.checks
            ],
        }


@dataclass
class ThresholdConfig:
    """Threshold configuration for checks."""

    warning_threshold: float
    critical_threshold: float

    def evaluate(self, value: float) -> HealthStatus:
        """Evaluate value against thresholds."""
        if value >= self.critical_threshold:
            return HealthStatus.UNHEALTHY
        elif value >= self.warning_threshold:
            return HealthStatus.DEGRADED
        return HealthStatus.HEALTHY
