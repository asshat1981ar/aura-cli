"""CLI commands for Phase 2 Innovation Sprint sub-agents.

This module provides CLI access to:
- IOTA: AI Error Resolution
- KAPPA: Recording & Replay
- NU: Offline Mode
- PI: Config Encryption
- XI: Interactive Shell
- RHO: Health Monitor
- SIGMA: Security Audit
- TAU: Task Scheduler
"""

import asyncio
from pathlib import Path

from core.logging_utils import log_json


# ============================================================================
# IOTA - AI Error Resolution Commands
# ============================================================================


def cmd_error_resolve(args) -> int:
    """Resolve an error using AI (IOTA)."""
    error_message = " ".join(args.args) if hasattr(args, "args") and args.args else None

    if not error_message:
        print("Usage: aura error-resolve <error_message>")
        print("Example: aura error-resolve 'ModuleNotFoundError: No module named requests'")
        return 1

    try:
        from aura.error_resolution.engine import ErrorResolutionEngine

        engine = ErrorResolutionEngine()
        result = asyncio.run(engine.resolve(error_message))

        if result.fix:
            print(f"\n✅ Solution found (confidence: {result.confidence.value})")
            print(f"\n📋 Explanation:\n{result.explanation}")
            print(f"\n🔧 Fix:\n{result.fix}")
            if result.requires_confirmation:
                print("\n⚠️  This fix requires manual confirmation.")
        else:
            print("\n❌ No solution found.")
            print(f"Explanation: {result.explanation}")

        return 0 if result.fix else 1
    except Exception as e:
        log_json("ERROR", "error_resolve_failed", details={"error": str(e)})
        print(f"Error: {e}")
        return 1


def cmd_error_cache_clear(args) -> int:
    """Clear the error resolution cache."""
    try:
        from aura.error_resolution.cache import FourLayerCache

        cache = FourLayerCache()
        # Clear all cache layers
        cache.memory.clear()
        asyncio.run(cache.sqlite.clear())

        print("✅ Error resolution cache cleared.")
        return 0
    except Exception as e:
        print(f"Error: {e}")
        return 1


# ============================================================================
# KAPPA - Recording & Replay Commands
# ============================================================================


def cmd_record_start(args) -> int:
    """Start recording a session (KAPPA)."""
    name = getattr(args, "name", None)
    if not name:
        print("Usage: aura record-start <name>")
        return 1

    try:
        from aura.recording.recorder import Recorder

        recorder = Recorder()
        session = recorder.start_recording(name)

        print(f"✅ Started recording session: {name}")
        print(f"Session ID: {session.recording.id}")
        print("Use 'aura record-step <command>' to add steps.")
        return 0
    except Exception as e:
        print(f"Error: {e}")
        return 1


def cmd_record_list(args) -> int:
    """List all recordings."""
    try:
        from aura.recording.recorder import Recorder

        recorder = Recorder()
        recordings = asyncio.run(recorder.list_recordings())

        if not recordings:
            print("No recordings found.")
            return 0

        print(f"\n{'Name':<20} {'Steps':<8} {'Created':<20}")
        print("-" * 50)
        for rec in recordings:
            created = rec.created_at.strftime("%Y-%m-%d %H:%M")
            print(f"{rec.name:<20} {rec.step_count:<8} {created:<20}")

        return 0
    except Exception as e:
        print(f"Error: {e}")
        return 1


def cmd_replay(args) -> int:
    """Replay a recording."""
    name = getattr(args, "name", None)
    if not name:
        print("Usage: aura replay <name>")
        return 1

    try:
        from aura.recording.recorder import Recorder
        from aura.recording.replay import ReplayEngine

        recorder = Recorder()
        recording = asyncio.run(recorder.load(name))

        if not recording:
            print(f"Recording not found: {name}")
            return 1

        engine = ReplayEngine()
        result = asyncio.run(engine.replay(recording))

        print(f"\n✅ Replay complete: {result.success_count}/{len(result.step_results)} steps successful")
        if not result.success:
            print("⚠️  Some steps failed. Check output for details.")

        return 0 if result.success else 1
    except Exception as e:
        print(f"Error: {e}")
        return 1


# ============================================================================
# NU - Offline Mode Commands
# ============================================================================


def cmd_offline_status(args) -> int:
    """Check offline mode status (NU)."""
    try:
        from aura.offline.monitor import ConnectivityMonitor

        monitor = ConnectivityMonitor()
        status = asyncio.run(monitor.check_now())

        print(f"\nConnectivity Status: {status.value.upper()}")
        print(f"Is Online: {monitor.is_online}")

        return 0
    except Exception as e:
        print(f"Error: {e}")
        return 1


def cmd_queue_list(args) -> int:
    """List queued commands for offline execution."""
    try:
        from aura.offline.queue import CommandQueue

        queue = CommandQueue()
        size = asyncio.run(queue.size())

        print(f"\nOffline Queue Size: {size} pending commands")

        if size > 0:
            pending = asyncio.run(queue.get_pending())
            print(f"\n{'ID':<10} {'Command':<20} {'Priority':<10}")
            print("-" * 40)
            for cmd in pending[:10]:  # Show first 10
                print(f"{cmd.id:<10} {cmd.command:<20} {cmd.priority.name:<10}")

        return 0
    except Exception as e:
        print(f"Error: {e}")
        return 1


def cmd_queue_sync(args) -> int:
    """Sync offline queue (execute pending commands)."""
    try:
        from aura.offline.executor import OfflineExecutor

        executor = OfflineExecutor(connectivity_check=lambda: True)
        processed = asyncio.run(executor.sync())

        print(f"✅ Synced {processed} commands from offline queue.")
        return 0
    except Exception as e:
        print(f"Error: {e}")
        return 1


# ============================================================================
# PI - Config Encryption Commands
# ============================================================================


def cmd_encrypt_config(args) -> int:
    """Encrypt sensitive fields in config file (PI)."""
    config_path = getattr(args, "path", None) or "~/.aura/config.json"

    try:
        from aura.encryption.config_manager import EncryptedConfigManager

        manager = EncryptedConfigManager(config_path=config_path)

        # Load and re-save with encryption
        data = manager.load()
        manager.save(data, encrypt=True)

        print(f"✅ Config encrypted: {config_path}")
        return 0
    except Exception as e:
        print(f"Error: {e}")
        return 1


def cmd_rotate_key(args) -> int:
    """Rotate encryption key for config file."""
    config_path = getattr(args, "path", None) or "~/.aura/config.json"

    try:
        from aura.encryption.key_manager import KeyManager
        from aura.encryption.config_manager import EncryptedConfigManager

        # Generate new key
        key_manager = KeyManager()
        new_key = key_manager.generate_key()

        # Rotate key
        manager = EncryptedConfigManager(config_path=config_path)
        import hashlib

        new_key_bytes = hashlib.sha256(new_key.encode()).digest()
        manager.rotate_key(new_key_bytes)

        print(f"✅ Encryption key rotated for: {config_path}")
        print(f"New key (save this securely): {new_key}")
        return 0
    except Exception as e:
        print(f"Error: {e}")
        return 1


# ============================================================================
# XI - Interactive Shell Commands
# ============================================================================


def cmd_shell(args) -> int:
    """Start interactive shell (XI)."""
    try:
        from aura.shell.repl import REPL
        from aura.shell.builtin_commands import create_builtin_commands

        repl = REPL(prompt="aura> ")

        # Register built-in commands
        commands_dict = {}
        builtins = create_builtin_commands(commands_dict)
        for cmd in builtins:
            repl.register_command(cmd)

        repl.run()
        return 0
    except KeyboardInterrupt:
        print("\nGoodbye!")
        return 0
    except Exception as e:
        print(f"Error: {e}")
        return 1


# ============================================================================
# RHO - Health Monitor Commands
# ============================================================================


def cmd_health(args) -> int:
    """Check system health (RHO)."""
    try:
        from aura.health.checks import HealthChecks
        from aura.health.monitor import HealthMonitor

        monitor = HealthMonitor()

        # Register built-in checks
        monitor.register_check("system", HealthChecks.system_check)
        monitor.register_check("disk", lambda: asyncio.run(HealthChecks.disk_check()))
        monitor.register_check("memory", lambda: asyncio.run(HealthChecks.memory_check()))

        report = asyncio.run(monitor.run_all_checks())

        print(f"\n{'Health Status':<20}: {report.status.value.upper()}")
        print(f"{'Duration':<20}: {report.duration_ms:.0f}ms")
        print(f"{'Checks':<20}: {report.healthy_count}/{report.total_count} healthy")
        print()

        for check in report.checks:
            status_icon = "✅" if check.is_healthy else "❌"
            print(f"{status_icon} {check.name:<15} ({check.response_time_ms:.0f}ms)")
            if check.message:
                print(f"   {check.message}")

        return 0 if report.status.value == "healthy" else 1
    except Exception as e:
        print(f"Error: {e}")
        return 1


# ============================================================================
# SIGMA - Security Audit Commands
# ============================================================================


def cmd_security_audit(args) -> int:
    """Run security audit (SIGMA)."""
    path = getattr(args, "path", None) or "."

    try:
        from aura.security.auditor import SecurityAuditor

        auditor = SecurityAuditor()
        report = auditor.audit(Path(path))

        print(f"\n{'Security Audit Results':<30}")
        print("=" * 40)
        print(f"Scanned files: {report.scanned_files}")
        print(f"Total findings: {report.total_count}")
        print(f"  Critical: {report.critical_count}")
        print(f"  High: {report.high_count}")
        print(f"  Medium: {report.medium_count}")
        print(f"  Low: {report.low_count}")

        if report.findings:
            print("\nFindings:")
            for finding in report.findings[:10]:  # Show first 10
                print(f"\n[{finding.severity.value.upper()}] {finding.title}")
                if finding.file_path:
                    print(f"  Location: {finding.file_path}:{finding.line_number or '?'}")
                print(f"  {finding.description}")

        return 1 if report.has_critical else 0
    except Exception as e:
        print(f"Error: {e}")
        return 1


# ============================================================================
# TAU - Task Scheduler Commands
# ============================================================================


def cmd_schedule_list(args) -> int:
    """List scheduled tasks (TAU)."""
    try:
        # This would need access to a persistent scheduler instance
        print("Scheduled tasks feature requires scheduler service.")
        return 0
    except Exception as e:
        print(f"Error: {e}")
        return 1


def cmd_schedule_run(args) -> int:
    """Run a scheduled task immediately."""
    print("Scheduled task execution requires scheduler service.")
    return 0


# ============================================================================
# Command Registration
# ============================================================================

PHASE2_COMMANDS = {
    # IOTA - Error Resolution
    "error-resolve": cmd_error_resolve,
    "error-cache-clear": cmd_error_cache_clear,
    # KAPPA - Recording
    "record-start": cmd_record_start,
    "record-list": cmd_record_list,
    "replay": cmd_replay,
    # NU - Offline Mode
    "offline-status": cmd_offline_status,
    "queue-list": cmd_queue_list,
    "queue-sync": cmd_queue_sync,
    # PI - Config Encryption
    "encrypt-config": cmd_encrypt_config,
    "rotate-key": cmd_rotate_key,
    # XI - Interactive Shell
    "shell": cmd_shell,
    # RHO - Health Monitor
    "health": cmd_health,
    # SIGMA - Security Audit
    "security-audit": cmd_security_audit,
    # TAU - Task Scheduler
    "schedule-list": cmd_schedule_list,
    "schedule-run": cmd_schedule_run,
}


def get_phase2_help() -> str:
    """Get help text for Phase 2 commands."""
    return """
Phase 2 Innovation Sprint Commands:
===================================

Error Resolution (IOTA):
  error-resolve <msg>      Resolve an error using AI
  error-cache-clear        Clear error resolution cache

Recording & Replay (KAPPA):
  record-start <name>      Start recording a session
  record-list              List all recordings
  replay <name>            Replay a recording

Offline Mode (NU):
  offline-status           Check connectivity status
  queue-list               List queued offline commands
  queue-sync               Sync offline queue

Config Encryption (PI):
  encrypt-config [path]    Encrypt sensitive config fields
  rotate-key [path]        Rotate encryption key

Interactive Shell (XI):
  shell                    Start interactive shell

Health Monitor (RHO):
  health                   Check system health

Security Audit (SIGMA):
  security-audit [path]    Run security audit

Task Scheduler (TAU):
  schedule-list            List scheduled tasks
  schedule-run             Run scheduled tasks
"""
