"""Built-in health checks."""

import time
from typing import Optional

from .models import CheckResult, CheckType, HealthStatus, ThresholdConfig


class HealthChecks:
    """Collection of built-in health checks."""
    
    @staticmethod
    async def system_check() -> CheckResult:
        """Check overall system status."""
        start = time.time()
        
        try:
            import platform
            import os
            
            details = {
                "platform": platform.platform(),
                "python_version": platform.python_version(),
                "processor": platform.processor() or "unknown",
                "load_avg": os.getloadavg() if hasattr(os, 'getloadavg') else None,
            }
            
            return CheckResult(
                name="system",
                type=CheckType.SYSTEM,
                status=HealthStatus.HEALTHY,
                response_time_ms=(time.time() - start) * 1000,
                details=details,
            )
        except Exception as e:
            return CheckResult(
                name="system",
                type=CheckType.SYSTEM,
                status=HealthStatus.UNHEALTHY,
                response_time_ms=(time.time() - start) * 1000,
                message=str(e),
            )
    
    @staticmethod
    async def disk_check(
        path: str = "/",
        warning_percent: float = 80.0,
        critical_percent: float = 90.0,
    ) -> CheckResult:
        """Check disk usage."""
        start = time.time()
        
        try:
            import shutil
            
            usage = shutil.disk_usage(path)
            used_percent = (usage.used / usage.total) * 100
            
            threshold = ThresholdConfig(warning_percent, critical_percent)
            status = threshold.evaluate(used_percent)
            
            return CheckResult(
                name="disk",
                type=CheckType.DISK,
                status=status,
                response_time_ms=(time.time() - start) * 1000,
                message=f"Disk usage: {used_percent:.1f}%" if status != HealthStatus.HEALTHY else None,
                details={
                    "total_bytes": usage.total,
                    "used_bytes": usage.used,
                    "free_bytes": usage.free,
                    "used_percent": used_percent,
                    "path": path,
                },
            )
        except Exception as e:
            return CheckResult(
                name="disk",
                type=CheckType.DISK,
                status=HealthStatus.UNHEALTHY,
                response_time_ms=(time.time() - start) * 1000,
                message=str(e),
            )
    
    @staticmethod
    async def memory_check(
        warning_percent: float = 80.0,
        critical_percent: float = 90.0,
    ) -> CheckResult:
        """Check memory usage."""
        start = time.time()
        
        try:
            import psutil
            
            memory = psutil.virtual_memory()
            used_percent = memory.percent
            
            threshold = ThresholdConfig(warning_percent, critical_percent)
            status = threshold.evaluate(used_percent)
            
            return CheckResult(
                name="memory",
                type=CheckType.MEMORY,
                status=status,
                response_time_ms=(time.time() - start) * 1000,
                message=f"Memory usage: {used_percent:.1f}%" if status != HealthStatus.HEALTHY else None,
                details={
                    "total_bytes": memory.total,
                    "available_bytes": memory.available,
                    "used_bytes": memory.used,
                    "used_percent": used_percent,
                },
            )
        except ImportError:
            return CheckResult(
                name="memory",
                type=CheckType.MEMORY,
                status=HealthStatus.UNKNOWN,
                response_time_ms=(time.time() - start) * 1000,
                message="psutil not installed",
            )
        except Exception as e:
            return CheckResult(
                name="memory",
                type=CheckType.MEMORY,
                status=HealthStatus.UNHEALTHY,
                response_time_ms=(time.time() - start) * 1000,
                message=str(e),
            )
    
    @staticmethod
    async def cpu_check(
        warning_percent: float = 70.0,
        critical_percent: float = 85.0,
    ) -> CheckResult:
        """Check CPU usage."""
        start = time.time()
        
        try:
            import psutil
            
            # Get CPU percent over 1 second
            cpu_percent = psutil.cpu_percent(interval=0.1)
            
            threshold = ThresholdConfig(warning_percent, critical_percent)
            status = threshold.evaluate(cpu_percent)
            
            return CheckResult(
                name="cpu",
                type=CheckType.CPU,
                status=status,
                response_time_ms=(time.time() - start) * 1000,
                message=f"CPU usage: {cpu_percent:.1f}%" if status != HealthStatus.HEALTHY else None,
                details={
                    "percent": cpu_percent,
                    "core_count": psutil.cpu_count(),
                },
            )
        except ImportError:
            return CheckResult(
                name="cpu",
                type=CheckType.CPU,
                status=HealthStatus.UNKNOWN,
                response_time_ms=(time.time() - start) * 1000,
                message="psutil not installed",
            )
        except Exception as e:
            return CheckResult(
                name="cpu",
                type=CheckType.CPU,
                status=HealthStatus.UNHEALTHY,
                response_time_ms=(time.time() - start) * 1000,
                message=str(e),
            )
    
    @staticmethod
    async def database_check(connection_string: Optional[str] = None) -> CheckResult:
        """Check database connectivity."""
        start = time.time()
        
        try:
            import sqlite3
            
            conn = sqlite3.connect(connection_string or ":memory:")
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            cursor.fetchone()
            conn.close()
            
            return CheckResult(
                name="database",
                type=CheckType.DATABASE,
                status=HealthStatus.HEALTHY,
                response_time_ms=(time.time() - start) * 1000,
                details={"connection": "successful"},
            )
        except Exception as e:
            return CheckResult(
                name="database",
                type=CheckType.DATABASE,
                status=HealthStatus.UNHEALTHY,
                response_time_ms=(time.time() - start) * 1000,
                message=str(e),
            )
    
    @staticmethod
    async def api_check(
        url: str,
        timeout: int = 5,
        expected_status: int = 200,
    ) -> CheckResult:
        """Check API endpoint."""
        start = time.time()
        
        try:
            import aiohttp
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=timeout)) as resp:
                    status = HealthStatus.HEALTHY if resp.status == expected_status else HealthStatus.UNHEALTHY
                    
                    return CheckResult(
                        name=f"api_{url}",
                        type=CheckType.API,
                        status=status,
                        response_time_ms=(time.time() - start) * 1000,
                        message=f"Status: {resp.status}" if status != HealthStatus.HEALTHY else None,
                        details={
                            "url": url,
                            "status_code": resp.status,
                            "expected_status": expected_status,
                        },
                    )
        except ImportError:
            return CheckResult(
                name=f"api_{url}",
                type=CheckType.API,
                status=HealthStatus.UNKNOWN,
                response_time_ms=(time.time() - start) * 1000,
                message="aiohttp not installed",
            )
        except Exception as e:
            return CheckResult(
                name=f"api_{url}",
                type=CheckType.API,
                status=HealthStatus.UNHEALTHY,
                response_time_ms=(time.time() - start) * 1000,
                message=str(e),
                details={"url": url},
            )
