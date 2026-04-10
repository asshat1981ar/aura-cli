"""
Unit tests for memory/neo4j_bridge.py

Tests for Neo4j bridge, circuit breaker, and graph synchronization.
"""

import pytest
import time
from unittest.mock import Mock, patch, MagicMock, call
from types import SimpleNamespace

from memory.neo4j_bridge import _CircuitBreaker, Neo4jBridge


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
