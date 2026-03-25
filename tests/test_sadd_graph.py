"""Unit tests for core.sadd.workstream_graph — DAG state machine."""

import unittest

from core.sadd.types import WorkstreamResult, WorkstreamSpec
from core.sadd.workstream_graph import WorkstreamGraph, WorkstreamNode


def _spec(id: str, title: str = "", depends_on: list[str] | None = None) -> WorkstreamSpec:
    return WorkstreamSpec(
        id=id,
        title=title or id,
        goal_text=f"Goal for {id}",
        depends_on=depends_on or [],
    )


class TestWorkstreamGraph(unittest.TestCase):
    """Tests for WorkstreamGraph DAG construction, scheduling, and state transitions."""

    # -- topology -----------------------------------------------------------

    def test_simple_graph(self):
        """Two independent specs: both in wave 1, both ready."""
        graph = WorkstreamGraph([_spec("A"), _spec("B")])
        waves = graph.execution_waves()
        self.assertEqual(len(waves), 1)
        self.assertCountEqual(waves[0], ["A", "B"])
        self.assertCountEqual(graph.ready_workstreams(), ["A", "B"])

    def test_dependency_chain(self):
        """A -> B -> C: 3 waves, only A ready initially."""
        graph = WorkstreamGraph(
            [
                _spec("A"),
                _spec("B", depends_on=["A"]),
                _spec("C", depends_on=["B"]),
            ]
        )
        waves = graph.execution_waves()
        self.assertEqual(len(waves), 3)
        self.assertEqual(waves[0], ["A"])
        self.assertEqual(waves[1], ["B"])
        self.assertEqual(waves[2], ["C"])
        self.assertEqual(graph.ready_workstreams(), ["A"])

    def test_diamond_dependency(self):
        """A->C, B->C: waves [[A,B],[C]], A and B ready."""
        graph = WorkstreamGraph(
            [
                _spec("A"),
                _spec("B"),
                _spec("C", depends_on=["A", "B"]),
            ]
        )
        waves = graph.execution_waves()
        self.assertEqual(len(waves), 2)
        self.assertCountEqual(waves[0], ["A", "B"])
        self.assertEqual(waves[1], ["C"])
        self.assertCountEqual(graph.ready_workstreams(), ["A", "B"])

    def test_cycle_detection(self):
        """A depends on B, B depends on A: raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            WorkstreamGraph(
                [
                    _spec("A", depends_on=["B"]),
                    _spec("B", depends_on=["A"]),
                ]
            )
        self.assertIn("Cycle", str(ctx.exception))

    def test_unknown_dependency(self):
        """Spec depends on nonexistent ID: raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            WorkstreamGraph([_spec("A", depends_on=["Z"])])
        self.assertIn("unknown ID", str(ctx.exception))

    # -- state transitions --------------------------------------------------

    def test_mark_running(self):
        """Sets status and started_at timestamp."""
        graph = WorkstreamGraph([_spec("A")])
        graph.mark_running("A")
        node = graph.get_node("A")
        self.assertEqual(node.status, "running")
        self.assertIsNotNone(node.started_at)

    def test_mark_completed(self):
        """Sets status, stores result, unblocks dependents."""
        graph = WorkstreamGraph(
            [
                _spec("A"),
                _spec("B", depends_on=["A"]),
            ]
        )
        self.assertEqual(graph.ready_workstreams(), ["A"])
        self.assertNotIn("B", graph.ready_workstreams())

        result = WorkstreamResult(ws_id="A", status="completed")
        graph.mark_running("A")
        graph.mark_completed("A", result)

        node = graph.get_node("A")
        self.assertEqual(node.status, "completed")
        self.assertIs(node.result, result)
        self.assertIsNotNone(node.completed_at)
        # B should now be ready
        self.assertIn("B", graph.ready_workstreams())

    def test_mark_failed_blocks_dependents(self):
        """A fails -> B (depends on A) gets blocked."""
        graph = WorkstreamGraph(
            [
                _spec("A"),
                _spec("B", depends_on=["A"]),
            ]
        )
        graph.mark_running("A")
        graph.mark_failed("A", "some error")

        self.assertEqual(graph.get_node("A").status, "failed")
        self.assertEqual(graph.get_node("B").status, "blocked")
        self.assertIn("B", graph.blocked_workstreams())

    def test_transitive_blocking(self):
        """A -> B -> C: A fails -> B and C both blocked."""
        graph = WorkstreamGraph(
            [
                _spec("A"),
                _spec("B", depends_on=["A"]),
                _spec("C", depends_on=["B"]),
            ]
        )
        graph.mark_running("A")
        graph.mark_failed("A", "boom")

        self.assertEqual(graph.get_node("B").status, "blocked")
        self.assertEqual(graph.get_node("C").status, "blocked")
        self.assertCountEqual(graph.blocked_workstreams(), ["B", "C"])

    # -- dynamic readiness --------------------------------------------------

    def test_ready_workstreams_dynamic(self):
        """After completing A, B becomes ready."""
        graph = WorkstreamGraph(
            [
                _spec("A"),
                _spec("B", depends_on=["A"]),
            ]
        )
        self.assertEqual(graph.ready_workstreams(), ["A"])

        graph.mark_running("A")
        # While A is running, B should not be ready
        self.assertEqual(graph.ready_workstreams(), [])

        graph.mark_completed("A", WorkstreamResult(ws_id="A", status="completed"))
        self.assertEqual(graph.ready_workstreams(), ["B"])

    # -- completeness -------------------------------------------------------

    def test_is_complete(self):
        """All completed/failed/blocked -> True; pending -> False."""
        graph = WorkstreamGraph(
            [
                _spec("A"),
                _spec("B", depends_on=["A"]),
            ]
        )
        self.assertFalse(graph.is_complete())

        graph.mark_running("A")
        self.assertFalse(graph.is_complete())

        graph.mark_failed("A", "err")
        # A=failed, B=blocked -> all terminal
        self.assertTrue(graph.is_complete())

    # -- serialization ------------------------------------------------------

    def test_serialization_roundtrip(self):
        """to_dict() then from_dict() preserves state."""
        graph = WorkstreamGraph(
            [
                _spec("A"),
                _spec("B", depends_on=["A"]),
                _spec("C", depends_on=["A"]),
            ]
        )
        graph.mark_running("A")
        graph.mark_completed("A", WorkstreamResult(ws_id="A", status="completed", cycles_used=3))
        graph.mark_running("B")

        data = graph.to_dict()
        restored = WorkstreamGraph.from_dict(data)

        self.assertEqual(restored.get_node("A").status, "completed")
        self.assertEqual(restored.get_node("A").result.ws_id, "A")
        self.assertEqual(restored.get_node("A").result.cycles_used, 3)
        self.assertEqual(restored.get_node("B").status, "running")
        self.assertEqual(restored.get_node("C").status, "pending")
        self.assertIsNotNone(restored.get_node("A").completed_at)
        self.assertIsNotNone(restored.get_node("B").started_at)

        # Structural integrity — waves should still work
        waves = restored.execution_waves()
        self.assertEqual(len(waves), 2)
        self.assertEqual(waves[0], ["A"])
        self.assertCountEqual(waves[1], ["B", "C"])

    # -- edge case: empty graph ---------------------------------------------

    def test_empty_graph(self):
        """No specs: no waves, is_complete immediately."""
        graph = WorkstreamGraph([])
        self.assertEqual(graph.execution_waves(), [])
        self.assertEqual(graph.ready_workstreams(), [])
        self.assertTrue(graph.is_complete())
        self.assertEqual(graph.blocked_workstreams(), [])


if __name__ == "__main__":
    unittest.main()
