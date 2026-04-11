"""Tests for relevance scoring."""

import pytest

from aura.semantic.context_graph import CodeElement, ContextGraph, ElementType
from aura.semantic.relevance import RelevanceScorer


class TestRelevanceScorer:
    @pytest.fixture
    def graph(self):
        return ContextGraph()
    
    @pytest.fixture
    def scorer(self, graph):
        return RelevanceScorer(graph)
    
    def test_name_match_scoring(self, graph, scorer):
        elem = CodeElement(
            name="process_data",
            type=ElementType.FUNCTION,
            file_path="test.py",
            line_start=1,
            line_end=10,
        )
        graph.add_element(elem)
        
        scores = scorer.score("process data")
        
        assert len(scores) == 1
        assert scores[0].name_match > 0
    
    def test_semantic_match_scoring(self, graph, scorer):
        elem = CodeElement(
            name="func1",
            type=ElementType.FUNCTION,
            file_path="test.py",
            line_start=1,
            line_end=10,
            docstring="Process and validate user input data",
        )
        graph.add_element(elem)
        
        scores = scorer.score("validate user input")
        
        assert len(scores) == 1
        assert scores[0].semantic_match > 0
    
    def test_type_weights(self, graph, scorer):
        func = CodeElement(
            name="my_func",
            type=ElementType.FUNCTION,
            file_path="test.py",
            line_start=1,
            line_end=5,
        )
        cls = CodeElement(
            name="MyClass",
            type=ElementType.CLASS,
            file_path="test.py",
            line_start=10,
            line_end=20,
        )
        graph.add_element(func)
        graph.add_element(cls)
        
        scores = scorer.score("my")
        
        # Both match, but class has higher weight
        assert len(scores) == 2
    
    def test_relationship_bonus(self, graph, scorer):
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
        
        # Score callee with caller as context
        scores = scorer.score("callee", context_elements=[id1])
        
        assert len(scores) == 1
        assert scores[0].relationship_bonus > 0
    
    def test_recency_bonus(self, graph, scorer):
        elem = CodeElement(
            name="recent_func",
            type=ElementType.FUNCTION,
            file_path="test.py",
            line_start=1,
            line_end=5,
        )
        elem_id = graph.add_element(elem)
        
        scores = scorer.score("recent", recent_elements=[elem_id])
        
        assert len(scores) == 1
        assert scores[0].recency_bonus > 0
    
    def test_get_related_context(self, graph, scorer):
        elem1 = CodeElement(
            name="main",
            type=ElementType.FUNCTION,
            file_path="test.py",
            line_start=1,
            line_end=10,
        )
        elem2 = CodeElement(
            name="helper",
            type=ElementType.FUNCTION,
            file_path="test.py",
            line_start=15,
            line_end=20,
        )
        id1 = graph.add_element(elem1)
        id2 = graph.add_element(elem2)
        graph.add_dependency(id1, id2)
        
        related = scorer.get_related_context(id1, depth=1)
        
        assert len(related) == 1
        assert related[0].name == "helper"
    
    def test_tokenization(self, scorer):
        tokens = scorer._tokenize("processUserData")
        
        assert "process" in tokens
        assert "user" in tokens
        assert "data" in tokens
    
    def test_empty_query(self, graph, scorer):
        elem = CodeElement(
            name="func",
            type=ElementType.FUNCTION,
            file_path="test.py",
            line_start=1,
            line_end=5,
        )
        graph.add_element(elem)
        
        scores = scorer.score("")
        
        # Empty query should not match
        assert len(scores) == 0
