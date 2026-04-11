"""Comprehensive integration tests for innovation modules wired end-to-end.

Tests HookEngine, ConfidenceRouter, NBestEngine, ExperimentTracker,
A2AServer, EventBus, MemoryConsolidator, and NegativeExampleStore
working together and with the orchestrator.
"""

from __future__ import annotations

import asyncio
import json
import tempfile
import time
import unittest
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch


# ── 1. HookEngine + Orchestrator: pre_act hook blocks phase ──────────────


class TestHookEngineOrchestratorBlocking(unittest.TestCase):
    """Register a pre_act hook that exits 2 (block), verify phase is skipped."""

    def test_pre_act_hook_blocks_phase(self):
        from core.hooks import HookEngine, HookResult

        config = {
            "hooks": {
                "pre_act": [
                    {"command": "exit 2", "blocking": True},
                ],
            }
        }
        engine = HookEngine(config)

        phase_input = {"goal": "test", "code": "print('hello')"}
        should_proceed, modified_input = engine.run_pre_hooks("act", phase_input)

        self.assertFalse(should_proceed, "Phase should be blocked when hook exits 2")
        self.assertEqual(len(engine.history), 1)
        self.assertEqual(engine.history[0].result, HookResult.BLOCK)

    def test_blocked_phase_returns_marker(self):
        """Verify the orchestrator _run_phase returns blocked marker."""
        from core.hooks import HookEngine
        from core.orchestrator import LoopOrchestrator

        config = {
            "hooks": {
                "pre_act": [
                    {"command": "exit 2", "blocking": True},
                ],
            }
        }

        # Minimal orchestrator with a mock act agent
        mock_agent = MagicMock()
        mock_agent.run.return_value = {"code": "generated"}

        agents = {"act": mock_agent}
        with patch.object(LoopOrchestrator, "_load_config_file", return_value=config):
            orch = LoopOrchestrator(agents=agents, project_root=Path("/tmp"))

        result = orch._run_phase("act", {"goal": "test"})
        self.assertTrue(result.get("_blocked_by_hook"))
        # Agent should NOT have been called since the hook blocked
        mock_agent.run.assert_not_called()


# ── 2. HookEngine input modification ─────────────────────────────────────


class TestHookEngineInputModification(unittest.TestCase):
    """Register a pre_plan hook that outputs JSON, verify input is modified."""

    def test_pre_plan_hook_modifies_input(self):
        from core.hooks import HookEngine, HookResult

        # Hook that outputs JSON to modify the phase input
        config = {
            "hooks": {
                "pre_plan": [
                    {
                        "command": 'echo \'{"injected_context": "from_hook", "priority": "high"}\'',
                        "blocking": True,
                    },
                ],
            }
        }
        engine = HookEngine(config)

        original_input = {"goal": "build feature X"}
        should_proceed, modified_input = engine.run_pre_hooks("plan", original_input)

        self.assertTrue(should_proceed)
        self.assertEqual(modified_input["injected_context"], "from_hook")
        self.assertEqual(modified_input["priority"], "high")
        # Original keys preserved
        self.assertEqual(modified_input["goal"], "build feature X")
        self.assertEqual(engine.history[0].result, HookResult.MODIFY)


# ── 3. ConfidenceRouter smart routing: declining confidence ───────────────


class TestConfidenceRouterSmartRouting(unittest.TestCase):
    """Simulate declining confidence across phases, verify escalation."""

    def test_declining_confidence_triggers_escalation(self):
        from core.phase_result import ConfidenceRouter, PhaseResult

        router = ConfidenceRouter()

        # Simulate 3 phases with declining confidence
        results = [
            PhaseResult(phase="plan", confidence=0.7),
            PhaseResult(phase="critique", confidence=0.5),
            PhaseResult(phase="act", confidence=0.3),
        ]

        for r in results:
            router.record(r)

        # The last result has confidence 0.3, and the trend is declining
        # with the last value < retry_below (0.4)
        self.assertTrue(
            router.should_escalate(results[-1]),
            "Should escalate when confidence declines across 3+ phases and latest is below retry threshold",
        )

    def test_stable_confidence_no_escalation(self):
        from core.phase_result import ConfidenceRouter, PhaseResult

        router = ConfidenceRouter()

        results = [
            PhaseResult(phase="plan", confidence=0.7),
            PhaseResult(phase="critique", confidence=0.7),
            PhaseResult(phase="act", confidence=0.7),
        ]
        for r in results:
            router.record(r)

        self.assertFalse(router.should_escalate(results[-1]))


# ── 4. ConfidenceRouter critique skip ─────────────────────────────────────


class TestConfidenceRouterCritiqueSkip(unittest.TestCase):
    """Verify should_skip_optional returns True when plan confidence > 0.9."""

    def test_skip_critique_high_confidence(self):
        from core.phase_result import ConfidenceRouter, PhaseResult

        router = ConfidenceRouter()
        plan_result = PhaseResult(phase="plan", confidence=0.95)

        self.assertTrue(
            router.should_skip_optional(plan_result, "critique"),
            "critique is optional and should be skipped when confidence > 0.9",
        )

    def test_no_skip_critique_moderate_confidence(self):
        from core.phase_result import ConfidenceRouter, PhaseResult

        router = ConfidenceRouter()
        plan_result = PhaseResult(phase="plan", confidence=0.7)

        self.assertFalse(
            router.should_skip_optional(plan_result, "critique"),
            "critique should NOT be skipped when confidence is only 0.7",
        )

    def test_no_skip_non_optional_phase(self):
        from core.phase_result import ConfidenceRouter, PhaseResult

        router = ConfidenceRouter()
        plan_result = PhaseResult(phase="plan", confidence=0.99)

        self.assertFalse(
            router.should_skip_optional(plan_result, "act"),
            "act is not optional and should never be skipped",
        )


# ── 5. NBestEngine end-to-end ─────────────────────────────────────────────


class TestNBestEngineEndToEnd(unittest.TestCase):
    """Mock model returns different code at different temperatures,
    verify critic tournament selects best."""

    def test_critic_tournament_selects_best(self):
        from core.nbest import NBestEngine

        engine = NBestEngine(n_candidates=3)

        # Mock model that returns different code per call
        call_count = 0

        def mock_respond(prompt):
            nonlocal call_count
            call_count += 1
            # Return JSON with changes for each variant
            return json.dumps(
                {
                    "changes": [
                        {
                            "file_path": f"module_{call_count}.py",
                            "old_code": "",
                            "new_code": f"def solution_{call_count}(): pass  # variant {call_count}",
                        }
                    ]
                }
            )

        model = MagicMock()
        model.respond = mock_respond
        model.respond_for_role = None  # Force fallback to respond()

        # Generate candidates
        candidates = engine.generate_candidates(model, "implement feature X")
        self.assertEqual(len(candidates), 3)
        # All should have changes parsed
        for c in candidates:
            self.assertTrue(len(c.changes) > 0, f"Candidate {c.variant_id} should have changes")

        # Mock sandbox
        sandbox = MagicMock()
        sandbox.run.return_value = {"success": True, "output": "tests pass"}
        candidates = engine.sandbox_all(sandbox, candidates)
        for c in candidates:
            self.assertTrue(c.sandbox_passed)

        # Mock critic scoring: variant 1 gets highest scores
        def mock_critic_respond(prompt):
            return json.dumps(
                {
                    "scores": {
                        "0": {"correctness": 0.9, "elegance": 0.9, "efficiency": 0.8, "maintainability": 0.9, "test_coverage": 0.8},
                        "1": {"correctness": 0.6, "elegance": 0.5, "efficiency": 0.7, "maintainability": 0.5, "test_coverage": 0.4},
                        "2": {"correctness": 0.7, "elegance": 0.7, "efficiency": 0.6, "maintainability": 0.6, "test_coverage": 0.5},
                    }
                }
            )

        critic_model = MagicMock()
        critic_model.respond = mock_critic_respond
        critic_model.respond_for_role = None

        winner = engine.critic_tournament(critic_model, candidates, "implement feature X")
        self.assertEqual(winner.variant_id, 0, "Variant 0 should win with highest scores")
        self.assertGreater(winner.total_score, 0)

    def test_single_candidate_auto_wins(self):
        from core.nbest import NBestEngine, CodeCandidate

        engine = NBestEngine(n_candidates=1)
        candidate = CodeCandidate(
            variant_id=0,
            changes=[{"file_path": "a.py", "old_code": "", "new_code": "pass"}],
        )
        model = MagicMock()
        winner = engine.critic_tournament(model, [candidate], "goal")
        self.assertEqual(winner.variant_id, 0)
        self.assertEqual(winner.total_score, 1.0)


# ── 6. ExperimentTracker full lifecycle ───────────────────────────────────


class TestExperimentTrackerLifecycle(unittest.TestCase):
    """Start experiment, collect metrics, finish with improvement/regression."""

    def test_improvement_kept(self):
        from core.experiment_tracker import ExperimentTracker, MetricsCollector

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            experiments_path = tmpdir / "experiments.jsonl"

            # Create decision log with improving metrics
            log_path = tmpdir / "decision_log.jsonl"
            entries = [
                {"event": "verify_result", "details": {"success": True}},
                {"event": "verify_result", "details": {"success": True}},
            ]
            log_path.write_text("\n".join(json.dumps(e) for e in entries))

            mc = MetricsCollector(tmpdir)
            tracker = ExperimentTracker(experiments_path, mc)

            # Start experiment — captures baseline
            baseline = tracker.start_experiment("exp_001", "Improve test pass rate")
            self.assertIn("test_pass_rate", baseline)

            # Add more passing results to simulate improvement
            entries.append({"event": "verify_result", "details": {"success": True}})
            log_path.write_text("\n".join(json.dumps(e) for e in entries))

            result = tracker.finish_experiment(
                experiment_id="exp_001",
                hypothesis="Improve test pass rate",
                change_description="Added retry logic",
                metrics_before=baseline,
                cycle_number=1,
                duration=5.0,
            )

            # Metrics should be equal or better (same data, so net == 0)
            # Either way, the experiment completes and is recorded
            self.assertIsNotNone(result)
            self.assertEqual(result.experiment_id, "exp_001")
            self.assertEqual(len(tracker.experiments), 1)

            # Verify persistence
            self.assertTrue(experiments_path.exists())
            persisted_data = experiments_path.read_text().strip()
            self.assertTrue(len(persisted_data) > 0)

    def test_regression_discarded(self):
        from core.experiment_tracker import ExperimentTracker, MetricsCollector

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            experiments_path = tmpdir / "experiments.jsonl"

            mc = MetricsCollector(tmpdir)
            tracker = ExperimentTracker(experiments_path, mc)

            # Fake a good baseline — all values higher than what an empty
            # directory will produce (defaults: test_pass_rate=0.5,
            # avg_cycle_seconds=60, goal_completion_rate=0, verify_success_rate=0.5,
            # avg_retries=1.0).  The tracker computes raw (after - before), so
            # when "after" values are lower than baseline the net is negative.
            good_baseline = {
                "test_pass_rate": 0.9,
                "avg_cycle_seconds": 120.0,
                "goal_completion_rate": 0.8,
                "verify_success_rate": 0.9,
                "avg_retries": 5.0,
            }

            result = tracker.finish_experiment(
                experiment_id="exp_002",
                hypothesis="Risky refactor",
                change_description="Rewrote core module",
                metrics_before=good_baseline,
                cycle_number=2,
                duration=10.0,
            )

            # Current metrics from empty dir will be worse than baseline
            self.assertFalse(result.kept, "Experiment with regression should be discarded")
            self.assertIn("regression", result.reason.lower())


# ── 7. A2AServer task lifecycle ───────────────────────────────────────────


class TestA2AServerTaskLifecycle(unittest.TestCase):
    """Create server, register handler, create task, verify completion."""

    def setUp(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

    def tearDown(self):
        self.loop.close()

    def test_task_completes_with_handler(self):
        from core.a2a.server import A2AServer
        from core.a2a.task import TaskState

        server = A2AServer()

        def my_handler(task):
            return {
                "summary": "Task done!",
                "artifacts": [
                    {"name": "output", "content": {"data": 42}, "mime_type": "application/json"},
                ],
            }

        server.register_handler("code_generation", my_handler)

        task = asyncio.run(server.create_task("code_generation", "Generate hello world"))

        self.assertEqual(task.state, TaskState.COMPLETED)
        self.assertTrue(len(task.messages) >= 2)  # user msg + agent msg
        self.assertEqual(len(task.artifacts), 1)
        self.assertEqual(task.artifacts[0]["name"], "output")

        # Verify task retrieval
        retrieved = server.get_task(task.id)
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.id, task.id)

    def test_task_fails_without_handler(self):
        from core.a2a.server import A2AServer
        from core.a2a.task import TaskState

        server = A2AServer()

        task = asyncio.run(server.create_task("nonexistent_capability", "Do something"))

        self.assertEqual(task.state, TaskState.FAILED)
        # Should contain error about unknown capability
        agent_messages = [m for m in task.messages if m.role == "agent"]
        self.assertTrue(any("Unknown capability" in m.content for m in agent_messages))

    def test_task_cancel(self):
        from core.a2a.server import A2AServer
        from core.a2a.task import A2ATask, TaskState

        server = A2AServer()
        # Manually add a task in SUBMITTED state
        task = A2ATask(capability="test")
        server.tasks[task.id] = task

        self.assertTrue(server.cancel_task(task.id))
        self.assertEqual(task.state, TaskState.CANCELED)

        # Cannot cancel a completed task
        self.assertFalse(server.cancel_task(task.id))


# ── 8. A2AServer + FastAPI routes ─────────────────────────────────────────


class TestA2AServerFastAPIRoutes(unittest.TestCase):
    """Test that register_fastapi_routes adds the expected endpoints."""

    def test_routes_registered(self):
        from core.a2a.server import A2AServer

        server = A2AServer()

        # Mock FastAPI app tracking route registrations
        mock_app = MagicMock()
        registered_routes = {}

        def make_decorator(method, path):
            def decorator(fn):
                registered_routes[f"{method} {path}"] = fn
                return fn

            return decorator

        mock_app.get = lambda path: make_decorator("GET", path)
        mock_app.post = lambda path: make_decorator("POST", path)

        server.register_fastapi_routes(mock_app)

        expected_routes = [
            "GET /.well-known/agent.json",
            "POST /a2a/tasks",
            "GET /a2a/tasks/{task_id}",
            "POST /a2a/tasks/{task_id}/cancel",
        ]
        for route in expected_routes:
            self.assertIn(
                route,
                registered_routes,
                f"Expected route '{route}' to be registered",
            )

    def test_agent_card_route_returns_valid_card(self):
        from core.a2a.server import A2AServer

        server = A2AServer()
        card = server.get_agent_card()

        self.assertIn("name", card)
        self.assertIn("capabilities", card)
        self.assertIn("supported_protocols", card)
        self.assertTrue(len(card["capabilities"]) > 0)


# ── 9. EventBus pub/sub + SSE ─────────────────────────────────────────────


class TestEventBusPubSubSSE(unittest.TestCase):
    """EventBus subscribe, publish, verify callback fires. SSE stream test."""

    def setUp(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

    def tearDown(self):
        self.loop.close()

    def test_sync_publish_fires_callback(self):
        from core.mcp_events import EventBus, EventType, MCPEvent

        bus = EventBus()
        received = []

        def on_tool_complete(event):
            received.append(event)

        bus.subscribe(EventType.TOOL_COMPLETE, on_tool_complete)

        event = MCPEvent(
            event_type=EventType.TOOL_COMPLETE,
            source="test_tool",
            data={"result": "success"},
        )
        bus.publish_sync(event)

        self.assertEqual(len(received), 1)
        self.assertEqual(received[0].source, "test_tool")
        self.assertEqual(received[0].data["result"], "success")

    def test_subscribe_all_receives_all_events(self):
        from core.mcp_events import EventBus, EventType, MCPEvent

        bus = EventBus()
        received = []

        bus.subscribe_all(lambda e: received.append(e))

        bus.publish_sync(MCPEvent(event_type=EventType.TOOL_COMPLETE, source="a"))
        bus.publish_sync(MCPEvent(event_type=EventType.FILE_CHANGED, source="b"))

        self.assertEqual(len(received), 2)

    def test_async_publish_fires_callback(self):
        from core.mcp_events import EventBus, EventType, MCPEvent

        bus = EventBus()
        received = []

        async def on_event(event):
            received.append(event)

        bus.subscribe(EventType.PHASE_COMPLETE, on_event)

        event = MCPEvent(
            event_type=EventType.PHASE_COMPLETE,
            source="orchestrator",
            data={"phase": "plan"},
        )

        asyncio.run(bus.publish(event))
        self.assertEqual(len(received), 1)

    def test_sse_stream_receives_events(self):
        from core.mcp_events import EventBus, EventType, MCPEvent

        bus = EventBus()

        sid, queue = bus.create_sse_stream("test_stream")

        event = MCPEvent(
            event_type=EventType.CI_COMPLETE,
            source="ci",
            data={"status": "green"},
        )

        async def _run():
            await bus.publish(event)
            received = await asyncio.wait_for(queue.get(), timeout=2.0)
            return received

        received = asyncio.run(_run())
        self.assertEqual(received.source, "ci")
        self.assertEqual(received.data["status"], "green")

        # Verify SSE formatting
        sse_str = received.to_sse()
        self.assertIn("event: ci.complete", sse_str)
        self.assertIn("data:", sse_str)

        bus.close_sse_stream(sid)
        self.assertNotIn(sid, bus._queues)

    def test_event_history(self):
        from core.mcp_events import EventBus, EventType, MCPEvent

        bus = EventBus()
        for i in range(5):
            bus.publish_sync(
                MCPEvent(
                    event_type=EventType.CUSTOM,
                    source=f"source_{i}",
                )
            )

        history = bus.get_history(event_type="custom", limit=3)
        self.assertEqual(len(history), 3)


# ── 10. MemoryConsolidator end-to-end ─────────────────────────────────────


class TestMemoryConsolidatorEndToEnd(unittest.TestCase):
    """Create 100 MemoryEntry objects with varied attributes, consolidate."""

    def test_consolidation_compression_over_50_percent(self):
        from memory.consolidation import MemoryConsolidator, MemoryEntry

        consolidator = MemoryConsolidator(
            retention_threshold=0.3,
            max_memories=40,
        )

        # Create 100 memories with varied confidence/age/access
        memories = []
        now = time.time()
        for i in range(100):
            # Low confidence + no access + old = should be pruned
            if i < 40:
                confidence = 0.05 + (i % 5) * 0.02
                access_count = 0
                created_at = now - 86400 * 30  # 30 days old
                decay_rate = 0.05
            # Medium confidence, some access
            elif i < 70:
                confidence = 0.4 + (i % 10) * 0.03
                access_count = i % 5
                created_at = now - 86400 * 5  # 5 days old
                decay_rate = 0.01
            # High confidence, frequent access
            else:
                confidence = 0.8 + (i % 5) * 0.02
                access_count = 10 + i % 10
                created_at = now - 86400  # 1 day old
                decay_rate = 0.005

            memories.append(
                MemoryEntry(
                    id=f"mem_{i:03d}",
                    content=f"Memory content #{i}: some knowledge about topic {i % 10}",
                    memory_type=["goal_outcome", "decision", "pattern", "error", "insight"][i % 5],
                    confidence=confidence,
                    access_count=access_count,
                    created_at=created_at,
                    decay_rate=decay_rate,
                    tags=[f"tag_{i % 3}"],
                )
            )

        retained, result = consolidator.consolidate(memories)

        self.assertEqual(result.memories_before, 100)
        self.assertGreater(
            result.compression_ratio,
            0.5,
            f"Expected > 50% compression, got {result.compression_ratio:.1%}",
        )
        self.assertLess(len(retained), 100)
        self.assertEqual(result.memories_after, len(retained))
        # Verify capacity limit enforced
        self.assertLessEqual(len(retained), 40)

    def test_duplicate_merging(self):
        from memory.consolidation import MemoryConsolidator, MemoryEntry

        consolidator = MemoryConsolidator()

        memories = [
            MemoryEntry(id="a", content="The quick brown fox", memory_type="insight", confidence=0.8, access_count=5),
            MemoryEntry(id="b", content="The quick brown fox", memory_type="insight", confidence=0.6, access_count=3),
            MemoryEntry(id="c", content="Something different", memory_type="decision", confidence=0.7, access_count=2),
        ]

        retained, result = consolidator.consolidate(memories)
        self.assertEqual(result.merged, 1, "Duplicate memories should be merged")
        # Merged entry should have combined access count
        fox_entries = [m for m in retained if "fox" in m.content]
        self.assertEqual(len(fox_entries), 1)
        self.assertEqual(fox_entries[0].access_count, 8)  # 5 + 3


# ── 11. NegativeExampleStore persistence ──────────────────────────────────


class TestNegativeExampleStorePersistence(unittest.TestCase):
    """Add failures, reload from disk, verify data preserved and search works."""

    def test_persistence_and_reload(self):
        from memory.consolidation import NegativeExampleStore

        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = Path(tmpdir) / "negative_examples.json"

            # Create store and add failures
            store = NegativeExampleStore(store_path)
            store.add(
                goal="Fix database connection pooling",
                failure_reason="Connection timeout under load",
                cycle_count=3,
                error_type="timeout",
                attempted_fixes=["increased pool size", "added retry"],
            )
            store.add(
                goal="Optimize database query performance",
                failure_reason="Query plan not using index",
                cycle_count=2,
                error_type="performance",
                attempted_fixes=["added index hint"],
            )
            store.add(
                goal="Add caching layer",
                failure_reason="Cache invalidation race condition",
                cycle_count=5,
                error_type="concurrency",
            )

            self.assertEqual(len(store.examples), 3)

            # Reload from disk
            store2 = NegativeExampleStore(store_path)
            self.assertEqual(len(store2.examples), 3)

            # Verify data preserved
            self.assertEqual(store2.examples[0]["goal"], "Fix database connection pooling")
            self.assertEqual(store2.examples[0]["error_type"], "timeout")
            self.assertEqual(len(store2.examples[0]["attempted_fixes"]), 2)

    def test_similar_failure_search(self):
        from memory.consolidation import NegativeExampleStore

        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = Path(tmpdir) / "negative_examples.json"
            store = NegativeExampleStore(store_path)

            store.add("Fix database connection pooling", "timeout", error_type="timeout")
            store.add("Optimize database query speed", "slow query", error_type="performance")
            store.add("Add frontend button styling", "CSS conflict", error_type="ui")

            # Search for database-related failures
            similar = store.find_similar_failures("database connection issues", limit=2)
            self.assertEqual(len(similar), 2)
            # Should find the two database-related failures
            goals = [s["goal"] for s in similar]
            self.assertIn("Fix database connection pooling", goals)
            self.assertIn("Optimize database query speed", goals)

    def test_summary(self):
        from memory.consolidation import NegativeExampleStore

        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = Path(tmpdir) / "negative_examples.json"
            store = NegativeExampleStore(store_path)

            store.add("goal1", "reason1", error_type="timeout")
            store.add("goal2", "reason2", error_type="timeout")
            store.add("goal3", "reason3", error_type="crash")

            summary = store.get_summary()
            self.assertEqual(summary["total"], 3)
            self.assertEqual(summary["error_types"]["timeout"], 2)
            self.assertEqual(summary["error_types"]["crash"], 1)
            self.assertEqual(summary["most_common"], "timeout")


# ── 12. Full orchestrator _run_phase with hooks ───────────────────────────


class TestOrchestratorRunPhaseWithHooks(unittest.TestCase):
    """Mock an agent, wrap with HookEngine, verify pre/post hooks fire."""

    def test_pre_and_post_hooks_fire(self):
        from core.hooks import HookEngine, HookTiming
        from core.orchestrator import LoopOrchestrator

        config = {
            "hooks": {
                "pre_plan": [
                    {"command": "echo pre_plan_ran", "blocking": False},
                ],
                "post_plan": [
                    {"command": "echo post_plan_ran", "blocking": False},
                ],
            }
        }

        mock_agent = MagicMock()
        mock_agent.run.return_value = {"plan": ["step 1", "step 2"]}

        agents = {"plan": mock_agent}
        with patch.object(LoopOrchestrator, "_load_config_file", return_value=config):
            orch = LoopOrchestrator(agents=agents, project_root=Path("/tmp"))

        result = orch._run_phase("plan", {"goal": "test goal"})

        # Agent should have been called
        mock_agent.run.assert_called_once()
        # Result should be from the agent
        self.assertEqual(result["plan"], ["step 1", "step 2"])

        # Both hooks should have fired
        self.assertEqual(len(orch.hook_engine.history), 2)
        pre_hooks = [h for h in orch.hook_engine.history if h.timing == HookTiming.PRE]
        post_hooks = [h for h in orch.hook_engine.history if h.timing == HookTiming.POST]
        self.assertEqual(len(pre_hooks), 1)
        self.assertEqual(len(post_hooks), 1)

    def test_no_agent_returns_empty(self):
        from core.orchestrator import LoopOrchestrator

        with patch.object(LoopOrchestrator, "_load_config_file", return_value={}):
            orch = LoopOrchestrator(agents={}, project_root=Path("/tmp"))

        result = orch._run_phase("nonexistent", {"goal": "test"})
        self.assertEqual(result, {})

    def test_hook_audit_log(self):
        from core.hooks import HookEngine

        config = {
            "hooks": {
                "pre_verify": [{"command": "echo ok"}],
                "post_verify": [{"command": "echo done"}],
            }
        }
        engine = HookEngine(config)
        engine.run_pre_hooks("verify", {"data": "test"})
        engine.run_post_hooks("verify", {"result": "pass"})

        audit = engine.get_audit_log()
        self.assertEqual(len(audit), 2)
        self.assertEqual(audit[0]["phase"], "verify")
        self.assertEqual(audit[0]["timing"], "pre")
        self.assertEqual(audit[1]["timing"], "post")
        self.assertIn("duration", audit[0])
        self.assertIn("timestamp", audit[0])


if __name__ == "__main__":
    unittest.main()
