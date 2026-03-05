from core.logging_utils import log_json


def monitor_performance(performance_metrics):
    """Log structured performance metrics using the AURA logging system.

    Args:
        performance_metrics: An iterable of metric values to log.
    """
    for metric in performance_metrics:
        log_json("INFO", "performance_metric", details={"metric": metric})