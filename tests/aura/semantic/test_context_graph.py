"""Tests for semantic context graph."""

import pytest
from pathlib import Path

from aura.semantic.context_graph import (
    CodeElement,
    ContextGraph,
    ElementType,
)


class TestCodeElement:
    def test_element_creation(self):
        elem = CodeElement(
            name="test_function",
            type=ElementType.FUNCTION,
            file_path="/path/to/file.py",
            line_start=10,
            line_end=20,
            docstring="Test function.",
            signature="def test_function(x: int)",
        )

        assert elem.name == "test_function"
        assert elem.type == ElementType.FUNCTION
        assert elem.line_start == 10
        assert elem.line_end == 20
        assert elem.id is not None

    def test_element_to_dict(self):
        elem = CodeElement(
            name="TestClass",
            type=ElementType.CLASS,
            file_path="/path/to/file.py",
            line_start=1,
            line_end=50,
            signature="class TestClass(Base)",
        )

        data = elem.to_dict()

        assert data["name"] == "TestClass"
        assert data["type"] == "class"
        assert "id" in data
        assert "signature" in data


class TestContextGraph:
    @pytest.fixture
    def graph(self):
        return ContextGraph()

    def test_add_element(self, graph):
        elem = CodeElement(
            name="func1",
            type=ElementType.FUNCTION,
            file_path="test.py",
            line_start=1,
            line_end=5,
        )

        elem_id = graph.add_element(elem)

        assert elem_id in graph.elements
        assert "test.py" in graph.file_index

    def test_get_element(self, graph):
        elem = CodeElement(
            name="func1",
            type=ElementType.FUNCTION,
            file_path="test.py",
            line_start=1,
            line_end=5,
        )

        elem_id = graph.add_element(elem)
        retrieved = graph.get_element(elem_id)

        assert retrieved == elem

    def test_add_dependency(self, graph):
        elem1 = CodeElement(
            name="caller",
            type=ElementType.FUNCTION,
            file_path="test.py",
            line_start=1,
            line_end=10,
        )
        elem2 = CodeElement(
            name="callee",
            type=ElementType.FUNCTION,
            file_path="test.py",
            line_start=15,
            line_end=20,
        )

        id1 = graph.add_element(elem1)
        id2 = graph.add_element(elem2)

        graph.add_dependency(id1, id2)

        assert id2 in elem1.dependencies
        assert id1 in elem2.dependents

    def test_get_dependencies(self, graph):
        elem1 = CodeElement(
            name="caller",
            type=ElementType.FUNCTION,
            file_path="test.py",
            line_start=1,
            line_end=10,
        )
        elem2 = CodeElement(
            name="callee",
            type=ElementType.FUNCTION,
            file_path="test.py",
            line_start=15,
            line_end=20,
        )

        id1 = graph.add_element(elem1)
        id2 = graph.add_element(elem2)
        graph.add_dependency(id1, id2)

        deps = graph.get_dependencies(id1)

        assert len(deps) == 1
        assert deps[0].name == "callee"

    def test_analyze_file_simple(self, tmp_path, graph):
        test_file = tmp_path / "test.py"
        test_file.write_text("""
def hello():
    pass

class MyClass:
    def method(self):
        pass
""")

        elements = graph.analyze_file(test_file)

        assert len(elements) >= 2
        names = [e.name for e in elements]
        assert "hello" in names
        assert "MyClass" in names

    def test_query(self, graph):
        elem1 = CodeElement(
            name="process_data",
            type=ElementType.FUNCTION,
            file_path="test.py",
            line_start=1,
            line_end=10,
            docstring="Process the input data.",
        )
        elem2 = CodeElement(
            name="validate_input",
            type=ElementType.FUNCTION,
            file_path="test.py",
            line_start=15,
            line_end=25,
        )

        graph.add_element(elem1)
        graph.add_element(elem2)

        results = graph.query("process")

        assert len(results) == 1
        assert results[0][0].name == "process_data"

    def test_to_dict(self, graph):
        elem = CodeElement(
            name="func1",
            type=ElementType.FUNCTION,
            file_path="test.py",
            line_start=1,
            line_end=5,
        )
        graph.add_element(elem)

        data = graph.to_dict()

        assert "elements" in data
        assert data["total_elements"] == 1
