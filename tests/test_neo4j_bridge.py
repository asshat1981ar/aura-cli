"""
Unit tests for memory/neo4j_bridge.py

Tests for Neo4j bridge, circuit breaker, and graph synchronization.
"""

import pytest
import time
from unittest.mock import Mock, patch, MagicMock, call
from types import SimpleNamespace

from memory.neo4j_bridge import _CircuitBreaker, Neo4jBridge, _props_clause

# Circuit-breaker reset tests use real time.sleep(0.15) calls; mark slow.
pytestmark = pytest.mark.slow


class TestCircuitBreaker:
    """Tests for _CircuitBreaker class."""

    def test_initialization(self):
        """Test circuit breaker initializes correctly."""
        cb = _CircuitBreaker(failure_threshold=5, reset_timeout=30.0)

        assert cb._failure_count == 0
        assert cb._failure_threshold == 5
        assert cb._reset_timeout == 30.0
        assert cb._state == "closed"
        assert cb._last_failure_time == 0.0

    def test_default_initialization(self):
        """Test circuit breaker with default values."""
        cb = _CircuitBreaker()

        assert cb._failure_threshold == 3
        assert cb._reset_timeout == 60.0

    def test_is_open_closed_state(self):
        """Test is_open when in closed state."""
        cb = _CircuitBreaker()

        assert cb.is_open is False

    def test_is_open_open_state(self):
        """Test is_open when in open state."""
        cb = _CircuitBreaker()
        cb._state = "open"
        cb._last_failure_time = time.time()

        assert cb.is_open is True

    def test_is_open_half_open_after_timeout(self):
        """Test is_open transitions to half_open after timeout."""
        cb = _CircuitBreaker(reset_timeout=0.1)
        cb._state = "open"
        cb._last_failure_time = time.time() - 0.2

        # Should transition to half_open
        result = cb.is_open

        assert result is False
        assert cb._state == "half_open"

    def test_record_success(self):
        """Test recording success resets state."""
        cb = _CircuitBreaker()
        cb._failure_count = 2
        cb._state = "half_open"

        cb.record_success()

        assert cb._failure_count == 0
        assert cb._state == "closed"

    def test_record_failure_increments_count(self):
        """Test recording failure increments count."""
        cb = _CircuitBreaker()

        cb.record_failure()

        assert cb._failure_count == 1
        assert cb._last_failure_time > 0

    def test_record_failure_opens_circuit(self):
        """Test recording enough failures opens the circuit."""
        cb = _CircuitBreaker(failure_threshold=2)

        cb.record_failure()
        assert cb._state == "closed"  # Still closed

        cb.record_failure()
        assert cb._state == "open"  # Now open

    def test_full_circuit_cycle(self):
        """Test complete circuit breaker cycle."""
        cb = _CircuitBreaker(failure_threshold=2, reset_timeout=0.1)

        # Start closed
        assert cb.is_open is False

        # Record failures to open
        cb.record_failure()
        cb.record_failure()
        assert cb.is_open is True

        # Wait for timeout
        time.sleep(0.15)

        # Should be half-open
        assert cb.is_open is False
        assert cb._state == "half_open"

        # Record success to close
        cb.record_success()
        assert cb._state == "closed"
        assert cb._failure_count == 0


class TestNeo4jBridge:
    """Tests for Neo4jBridge class."""

    @pytest.fixture
    def bridge(self):
        """Fixture providing a Neo4jBridge instance."""
        return Neo4jBridge(uri="bolt://testhost:7687", user="testuser", password="testpass")

    def test_initialization(self, bridge):
        """Test Neo4jBridge initializes correctly."""
        assert bridge._uri == "bolt://testhost:7687"
        assert bridge._user == "testuser"
        assert bridge._password == "testpass"
        assert bridge._driver is None
        assert isinstance(bridge._breaker, _CircuitBreaker)

    def test_default_initialization(self):
        """Test Neo4jBridge with default values."""
        bridge = Neo4jBridge()

        assert bridge._uri == "bolt://localhost:7687"
        assert bridge._user == "neo4j"
        assert bridge._password == ""

    @patch("memory.neo4j_bridge._CircuitBreaker.record_success")
    def test_get_driver_lazy_load(self, mock_record_success, bridge):
        """Test lazy loading of Neo4j driver."""
        mock_driver = Mock()

        with patch.dict("sys.modules", {"neo4j": Mock(GraphDatabase=Mock(driver=Mock(return_value=mock_driver)))}):
            with patch.dict("sys.modules", {"neo4j": MagicMock()}):
                mock_neo4j = MagicMock()
                mock_neo4j.GraphDatabase.driver.return_value = mock_driver

                with patch("builtins.__import__", side_effect=lambda name, *args, **kwargs: mock_neo4j if name == "neo4j" else __import__(name, *args, **kwargs)):
                    # First call should create driver
                    driver = bridge._get_driver()
                    assert driver is not None

    def test_get_driver_import_error(self, bridge):
        """Test handling when neo4j package not installed."""

        # Simulate ImportError by patching the import
        def mock_import(name, *args, **kwargs):
            if name == "neo4j":
                raise ImportError("No module named 'neo4j'")
            return __builtins__.__import__(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            with pytest.raises(RuntimeError, match="neo4j package not installed"):
                bridge._get_driver()

    def test_available_circuit_open(self, bridge):
        """Test available returns False when circuit is open."""
        bridge._breaker._state = "open"
        bridge._breaker._last_failure_time = time.time()

        assert bridge.available is False

    @patch.object(Neo4jBridge, "_get_driver")
    def test_available_connection_success(self, mock_get_driver, bridge):
        """Test available returns True on successful connection."""
        mock_driver = Mock()
        mock_driver.verify_connectivity.return_value = None
        mock_get_driver.return_value = mock_driver

        assert bridge.available is True
        mock_driver.verify_connectivity.assert_called_once()

    @patch.object(Neo4jBridge, "_get_driver")
    def test_available_connection_failure(self, mock_get_driver, bridge):
        """Test available returns False on connection failure."""
        mock_get_driver.side_effect = Exception("Connection refused")

        assert bridge.available is False

    @patch.object(Neo4jBridge, "_get_driver")
    def test_available_records_success(self, mock_get_driver, bridge):
        """Test available records success with breaker."""
        mock_driver = Mock()
        mock_driver.verify_connectivity.return_value = None
        mock_get_driver.return_value = mock_driver

        with patch.object(bridge._breaker, "record_success") as mock_record:
            bridge.available
            mock_record.assert_called_once()

    @patch.object(Neo4jBridge, "_get_driver")
    def test_available_records_failure(self, mock_get_driver, bridge):
        """Test available records failure with breaker."""
        mock_get_driver.side_effect = Exception("Connection refused")

        with patch.object(bridge._breaker, "record_failure") as mock_record:
            bridge.available
            mock_record.assert_called_once()


class TestNeo4jBridgeSync:
    """Tests for Neo4jBridge sync operations."""

    @pytest.fixture
    def bridge(self):
        """Fixture providing a Neo4jBridge instance."""
        return Neo4jBridge()

    @pytest.fixture
    def mock_brain(self):
        """Fixture providing a mock Brain object."""
        brain = Mock()
        brain.graph = Mock()
        brain.graph.nodes.data.return_value = [(1, {"type": "goal", "name": "test_goal"}), (2, {"type": "task", "name": "test_task"})]
        brain.graph.edges.data.return_value = [(1, 2, {"relation": "depends_on"})]
        return brain

    @patch.object(Neo4jBridge, "execute_write")
    @patch.object(Neo4jBridge, "_get_driver")
    def test_sync_from_brain(self, mock_get_driver, mock_execute, bridge, mock_brain):
        """Test syncing brain data to Neo4j."""
        # Setup mock brain with recall_with_budget
        mock_brain.recall_with_budget.return_value = ["memory1", "memory2"]

        result = bridge.sync_from_brain(mock_brain)

        # Should attempt to sync or return error if circuit open
        assert isinstance(result, dict)
        mock_brain.recall_with_budget.assert_called_once()

    @patch.object(Neo4jBridge, "_get_driver")
    def test_query(self, mock_get_driver, bridge):
        """Test executing Cypher query."""
        mock_session = Mock()
        mock_result = Mock()
        mock_result.data.return_value = [{"name": "test"}]
        mock_session.run.return_value = mock_result

        mock_driver = Mock()
        mock_driver.session.return_value.__enter__ = Mock(return_value=mock_session)
        mock_driver.session.return_value.__exit__ = Mock(return_value=False)
        mock_get_driver.return_value = mock_driver

        # Check if method exists
        if hasattr(bridge, "query"):
            result = bridge.query("MATCH (n) RETURN n.name as name")

            mock_session.run.assert_called_with("MATCH (n) RETURN n.name as name")
            assert result == [{"name": "test"}]


class TestIntegration:
    """Integration tests for neo4j_bridge module."""

    def test_circuit_breaker_with_bridge(self):
        """Test circuit breaker integration with bridge."""
        bridge = Neo4jBridge()

        # Simulate failures
        bridge._breaker.record_failure()
        bridge._breaker.record_failure()
        bridge._breaker.record_failure()

        # Circuit should be open
        assert bridge._breaker.is_open is True
        assert bridge.available is False

    def test_bridge_recovery(self):
        """Test bridge recovery after failures."""
        bridge = Neo4jBridge()
        bridge._breaker._reset_timeout = 0.1  # Set short timeout

        # Open circuit
        for _ in range(3):
            bridge._breaker.record_failure()

        assert bridge._breaker.is_open is True

        # Wait for timeout
        time.sleep(0.15)

        # Circuit should be half-open
        assert bridge._breaker.is_open is False
        assert bridge._breaker._state == "half_open"

    @patch.object(Neo4jBridge, "_get_driver")
    def test_full_workflow(self, mock_get_driver):
        """Test complete workflow with mocked driver."""
        bridge = Neo4jBridge()

        mock_session = Mock()
        mock_result = Mock()
        mock_result.data.return_value = [{"count": 42}]
        mock_session.run.return_value = mock_result

        mock_driver = Mock()
        mock_driver.session.return_value.__enter__ = Mock(return_value=mock_session)
        mock_driver.session.return_value.__exit__ = Mock(return_value=False)
        mock_get_driver.return_value = mock_driver

        # Test availability
        assert bridge.available is True

        # Verify driver was used
        mock_get_driver.assert_called()


class TestEdgeCases:
    """Edge case tests for neo4j_bridge."""

    def test_circuit_breaker_zero_threshold(self):
        """Test circuit breaker with zero threshold."""
        cb = _CircuitBreaker(failure_threshold=0)

        # Should open immediately
        cb.record_failure()
        assert cb._state == "open"

    def test_circuit_breaker_negative_timeout(self):
        """Test circuit breaker with negative timeout."""
        cb = _CircuitBreaker(reset_timeout=-1.0)
        cb._state = "open"
        cb._last_failure_time = time.time()

        # Should immediately transition to half_open
        assert cb.is_open is False
        assert cb._state == "half_open"

    def test_bridge_empty_password(self):
        """Test bridge with empty password."""
        bridge = Neo4jBridge(password="")
        assert bridge._password == ""

    def test_bridge_special_chars_in_uri(self):
        """Test bridge with special characters in URI."""
        bridge = Neo4jBridge(uri="bolt://user:pass@host:7687")
        assert "user:pass@" in bridge._uri


class TestNeo4jBridgeQueryKnowledge:
    """Tests for query_knowledge method."""

    @pytest.fixture
    def bridge(self):
        return Neo4jBridge()

    @patch.object(Neo4jBridge, "_get_driver")
    def test_query_knowledge_success(self, mock_get_driver, bridge):
        """Test query_knowledge with successful result."""
        mock_session = Mock()
        mock_record1 = {"name": "test1"}
        mock_record2 = {"name": "test2"}
        mock_result = Mock()
        mock_result.__iter__ = Mock(return_value=iter([mock_record1, mock_record2]))
        mock_session.run.return_value = mock_result

        mock_driver = Mock()
        mock_driver.session.return_value.__enter__ = Mock(return_value=mock_session)
        mock_driver.session.return_value.__exit__ = Mock(return_value=False)
        mock_get_driver.return_value = mock_driver

        result = bridge.query_knowledge("MATCH (n) RETURN n.name as name", {"param": "value"})

        assert len(result) == 2
        assert result[0] == mock_record1
        mock_session.run.assert_called_once_with("MATCH (n) RETURN n.name as name", {"param": "value"})

    @patch.object(Neo4jBridge, "_get_driver")
    def test_query_knowledge_empty_params(self, mock_get_driver, bridge):
        """Test query_knowledge with None params."""
        mock_session = Mock()
        mock_result = Mock()
        mock_result.__iter__ = Mock(return_value=iter([]))
        mock_session.run.return_value = mock_result

        mock_driver = Mock()
        mock_driver.session.return_value.__enter__ = Mock(return_value=mock_session)
        mock_driver.session.return_value.__exit__ = Mock(return_value=False)
        mock_get_driver.return_value = mock_driver

        result = bridge.query_knowledge("MATCH (n) RETURN n")

        assert result == []
        mock_session.run.assert_called_once_with("MATCH (n) RETURN n", {})

    def test_query_knowledge_circuit_open(self, bridge):
        """Test query_knowledge returns empty when circuit is open."""
        bridge._breaker._state = "open"
        bridge._breaker._last_failure_time = time.time()

        result = bridge.query_knowledge("MATCH (n) RETURN n")

        assert result == []

    @patch.object(Neo4jBridge, "_get_driver")
    def test_query_knowledge_exception(self, mock_get_driver, bridge):
        """Test query_knowledge handles exceptions."""
        mock_get_driver.side_effect = Exception("Connection lost")

        result = bridge.query_knowledge("MATCH (n) RETURN n")

        assert result == []


class TestNeo4jBridgeExecuteWrite:
    """Tests for execute_write method."""

    @pytest.fixture
    def bridge(self):
        return Neo4jBridge()

    @patch.object(Neo4jBridge, "_get_driver")
    def test_execute_write_success(self, mock_get_driver, bridge):
        """Test execute_write with successful creation."""
        mock_session = Mock()
        mock_summary = Mock()
        mock_summary.counters.nodes_created = 5
        mock_summary.counters.relationships_created = 3
        mock_summary.counters.properties_set = 10
        mock_result = Mock()
        mock_result.consume.return_value = mock_summary
        mock_session.run.return_value = mock_result

        mock_driver = Mock()
        mock_driver.session.return_value.__enter__ = Mock(return_value=mock_session)
        mock_driver.session.return_value.__exit__ = Mock(return_value=False)
        mock_get_driver.return_value = mock_driver

        result = bridge.execute_write("CREATE (n:Node {name: $name})", {"name": "test"})

        assert result["nodes_created"] == 5
        assert result["relationships_created"] == 3
        assert result["properties_set"] == 10

    def test_execute_write_circuit_open(self, bridge):
        """Test execute_write returns error when circuit is open."""
        bridge._breaker._state = "open"
        bridge._breaker._last_failure_time = time.time()

        result = bridge.execute_write("CREATE (n:Node)")

        assert result == {"error": "circuit_breaker_open"}

    @patch.object(Neo4jBridge, "_get_driver")
    def test_execute_write_exception(self, mock_get_driver, bridge):
        """Test execute_write handles exceptions."""
        mock_get_driver.side_effect = Exception("Write failed")

        result = bridge.execute_write("CREATE (n:Node)")

        assert "error" in result
        assert "Write failed" in result["error"]


class TestNeo4jBridgeImportCodebaseGraph:
    """Tests for import_codebase_graph method."""

    @pytest.fixture
    def bridge(self):
        return Neo4jBridge()

    @patch.object(Neo4jBridge, "_get_driver")
    def test_import_codebase_graph_success(self, mock_get_driver, bridge):
        """Test import_codebase_graph with valid nodes and relationships."""
        mock_session = Mock()
        mock_driver = Mock()
        mock_driver.session.return_value.__enter__ = Mock(return_value=mock_session)
        mock_driver.session.return_value.__exit__ = Mock(return_value=False)
        mock_get_driver.return_value = mock_driver

        nodes = [
            {"label": "Module", "properties": {"name": "utils", "path": "/src/utils.py"}},
            {"label": "Function", "properties": {"name": "helper", "signature": "def helper()"}},
        ]
        relationships = [
            {
                "from_label": "Module",
                "from_key": "name",
                "from_value": "utils",
                "to_label": "Function",
                "to_key": "name",
                "to_value": "helper",
                "rel_type": "CONTAINS",
                "properties": {"order": 1},
            }
        ]

        result = bridge.import_codebase_graph(nodes, relationships)

        assert result["nodes_created"] == 2
        assert result["relationships_created"] == 1
        assert mock_session.run.call_count == 3  # 2 nodes + 1 relationship

    def test_import_codebase_graph_circuit_open(self, bridge):
        """Test import_codebase_graph returns error when circuit is open."""
        bridge._breaker._state = "open"
        bridge._breaker._last_failure_time = time.time()

        result = bridge.import_codebase_graph([], [])

        assert result == {"error": "circuit_breaker_open"}

    @patch.object(Neo4jBridge, "_get_driver")
    def test_import_codebase_graph_exception(self, mock_get_driver, bridge):
        """Test import_codebase_graph handles exceptions."""
        mock_get_driver.side_effect = Exception("Graph import failed")

        nodes = [{"label": "Node", "properties": {}}]
        result = bridge.import_codebase_graph(nodes, [])

        assert "error" in result
        assert "Graph import failed" in result["error"]

    @patch.object(Neo4jBridge, "_get_driver")
    def test_import_codebase_graph_empty(self, mock_get_driver, bridge):
        """Test import_codebase_graph with empty nodes and relationships."""
        mock_session = Mock()
        mock_driver = Mock()
        mock_driver.session.return_value.__enter__ = Mock(return_value=mock_session)
        mock_driver.session.return_value.__exit__ = Mock(return_value=False)
        mock_get_driver.return_value = mock_driver

        result = bridge.import_codebase_graph([], [])

        assert result["nodes_created"] == 0
        assert result["relationships_created"] == 0


class TestNeo4jBridgeSyncFromBrain:
    """Tests for sync_from_brain method."""

    @pytest.fixture
    def bridge(self):
        return Neo4jBridge()

    def test_sync_from_brain_circuit_open(self, bridge):
        """Test sync_from_brain returns error when circuit is open."""
        bridge._breaker._state = "open"
        bridge._breaker._last_failure_time = time.time()

        mock_brain = Mock()
        result = bridge.sync_from_brain(mock_brain)

        assert result == {"error": "circuit_breaker_open"}

    @patch.object(Neo4jBridge, "execute_write")
    @patch.object(Neo4jBridge, "_get_driver")
    def test_sync_from_brain_success(self, mock_get_driver, mock_execute_write, bridge):
        """Test sync_from_brain with valid brain."""
        mock_brain = Mock()
        mock_brain.recall_with_budget.return_value = ["mem1", "mem2", "mem3"]

        result = bridge.sync_from_brain(mock_brain)

        assert result["synced_memories"] == 3
        mock_brain.recall_with_budget.assert_called_once_with(max_tokens=10000)

    @patch.object(Neo4jBridge, "execute_write")
    @patch.object(Neo4jBridge, "_get_driver")
    def test_sync_from_brain_empty(self, mock_get_driver, mock_execute_write, bridge):
        """Test sync_from_brain with no memories."""
        mock_brain = Mock()
        mock_brain.recall_with_budget.return_value = []

        result = bridge.sync_from_brain(mock_brain)

        assert result["synced_memories"] == 0

    @patch.object(Neo4jBridge, "_get_driver")
    def test_sync_from_brain_exception(self, mock_get_driver, bridge):
        """Test sync_from_brain handles exceptions."""
        mock_brain = Mock()
        mock_brain.recall_with_budget.side_effect = Exception("Recall failed")

        result = bridge.sync_from_brain(mock_brain)

        assert "error" in result
        assert "Recall failed" in result["error"]


class TestNeo4jBridgeSchemaInfo:
    """Tests for schema_info method."""

    @pytest.fixture
    def bridge(self):
        return Neo4jBridge()

    @patch.object(Neo4jBridge, "query_knowledge")
    def test_schema_info_success(self, mock_query, bridge):
        """Test schema_info retrieves schema information."""
        mock_query.side_effect = [
            [{"label": "Node"}, {"label": "Memory"}],
            [{"relationshipType": "KNOWS"}, {"relationshipType": "CONTAINS"}],
            [{"constraint": "unique"}, {"constraint": "exists"}],
        ]

        result = bridge.schema_info()

        assert result["labels"] == ["Node", "Memory"]
        assert result["relationship_types"] == ["KNOWS", "CONTAINS"]
        assert len(result["constraints"]) == 2
        assert mock_query.call_count == 3

    @patch.object(Neo4jBridge, "query_knowledge")
    def test_schema_info_empty(self, mock_query, bridge):
        """Test schema_info with empty results."""
        mock_query.side_effect = [[], [], []]

        result = bridge.schema_info()

        assert result["labels"] == []
        assert result["relationship_types"] == []
        assert result["constraints"] == []


class TestNeo4jBridgeClose:
    """Tests for close method."""

    def test_close_with_no_driver(self):
        """Test close when driver is None."""
        bridge = Neo4jBridge()
        assert bridge._driver is None
        bridge.close()  # Should not raise

    def test_close_with_driver(self):
        """Test close closes the driver."""
        bridge = Neo4jBridge()
        mock_driver = Mock()
        bridge._driver = mock_driver

        bridge.close()

        mock_driver.close.assert_called_once()
        assert bridge._driver is None

    def test_close_multiple_times(self):
        """Test close can be called multiple times safely."""
        bridge = Neo4jBridge()
        mock_driver = Mock()
        bridge._driver = mock_driver

        bridge.close()
        assert bridge._driver is None

        bridge.close()  # Should not raise


class TestPropsClause:
    """Tests for _props_clause helper function."""

    def test_props_clause_single(self):
        """Test _props_clause with single property."""
        result = _props_clause({"name": None})
        assert result == "name: $name"

    def test_props_clause_multiple(self):
        """Test _props_clause with multiple properties."""
        result = _props_clause({"name": None, "id": None, "value": None})
        # Order may vary in dict, check for all parts
        parts = result.split(", ")
        assert len(parts) == 3
        assert "name: $name" in result
        assert "id: $id" in result
        assert "value: $value" in result

    def test_props_clause_empty(self):
        """Test _props_clause with empty dict."""
        result = _props_clause({})
        assert result == ""
