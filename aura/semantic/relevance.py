"""Relevance scoring for semantic context retrieval."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Set

from .context_graph import CodeElement, ContextGraph, ElementType


@dataclass
class RelevanceScore:
    """Relevance score breakdown."""
    element_id: str
    total_score: float = 0.0
    name_match: float = 0.0
    semantic_match: float = 0.0
    relationship_bonus: float = 0.0
    recency_bonus: float = 0.0


class RelevanceScorer:
    """Score relevance of code elements to a query."""
    
    def __init__(self, graph: ContextGraph):
        self.graph = graph
        self._type_weights = {
            ElementType.CLASS: 1.2,
            ElementType.FUNCTION: 1.0,
            ElementType.METHOD: 0.9,
            ElementType.VARIABLE: 0.6,
            ElementType.IMPORT: 0.4,
        }
    
    def score(
        self,
        query: str,
        context_elements: Optional[List[str]] = None,
        recent_elements: Optional[List[str]] = None,
    ) -> List[RelevanceScore]:
        """Score all elements for relevance to query."""
        # Return empty list for empty query
        if not query or not query.strip():
            return []
        
        scores = []
        query_terms = self._tokenize(query)
        
        for elem_id, element in self.graph.elements.items():
            score = self._calculate_score(
                element, query_terms, context_elements, recent_elements
            )
            if score.total_score > 0:
                scores.append(score)
        
        # Sort by total score descending
        scores.sort(key=lambda x: x.total_score, reverse=True)
        return scores
    
    def _calculate_score(
        self,
        element: CodeElement,
        query_terms: Set[str],
        context_elements: Optional[List[str]],
        recent_elements: Optional[List[str]],
    ) -> RelevanceScore:
        """Calculate relevance score for a single element."""
        score = RelevanceScore(element_id=element.id)
        
        # Name match score
        name_tokens = self._tokenize(element.name)
        name_overlap = query_terms & name_tokens
        if name_overlap:
            score.name_match = len(name_overlap) / len(query_terms) * 2.0
        
        # Semantic match (docstring + signature)
        semantic_text = ""
        if element.docstring:
            semantic_text += element.docstring + " "
        if element.signature:
            semantic_text += element.signature
        
        semantic_tokens = self._tokenize(semantic_text)
        semantic_overlap = query_terms & semantic_tokens
        if semantic_overlap:
            score.semantic_match = len(semantic_overlap) / len(query_terms) * 1.0
        
        # Type weight
        type_weight = self._type_weights.get(element.type, 0.5)
        
        # Relationship bonus
        if context_elements:
            score.relationship_bonus = self._calculate_relationship_bonus(
                element, context_elements
            )
        
        # Recency bonus
        if recent_elements and element.id in recent_elements:
            score.recency_bonus = 0.3
        
        # Calculate total
        score.total_score = (
            score.name_match * type_weight +
            score.semantic_match * 0.5 +
            score.relationship_bonus +
            score.recency_bonus
        )
        
        return score
    
    def _calculate_relationship_bonus(
        self,
        element: CodeElement,
        context_elements: List[str],
    ) -> float:
        """Calculate bonus for being related to context elements."""
        bonus = 0.0
        
        for context_id in context_elements:
            if context_id == element.id:
                continue
            
            # Direct dependency
            if context_id in element.dependencies:
                bonus += 0.5
            
            # Direct dependent
            if context_id in element.dependents:
                bonus += 0.4
            
            # Same file
            context_elem = self.graph.get_element(context_id)
            if context_elem and context_elem.file_path == element.file_path:
                bonus += 0.2
        
        return min(bonus, 1.5)  # Cap at 1.5
    
    def _tokenize(self, text: str) -> Set[str]:
        """Tokenize text into searchable terms."""
        import re
        
        tokens = set()
        
        # First pass: extract camelCase/PascalCase tokens from original text
        for word in text.split():
            word = word.strip()
            if len(word) >= 2:
                # Add original lowercase
                tokens.add(word.lower())
                
                # Split camelCase/PascalCase before lowercasing
                if word.isalnum() and not word.islower() and not word.isupper():
                    # Split: "processUserData" -> ["process", "User", "Data"]
                    # Split: "MyClass" -> ["My", "Class"]
                    parts = re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)', word)
                    for p in parts:
                        if len(p) >= 2:
                            tokens.add(p.lower())
        
        # Second pass: handle snake_case, kebab-case with separators
        text_lower = text.lower()
        for sep in ['_', '-', '.', '(', ')', ',', ':']:
            text_lower = text_lower.replace(sep, ' ')
        
        for word in text_lower.split():
            word = word.strip()
            if len(word) >= 2:
                tokens.add(word)
        
        return tokens
    
    def get_related_context(
        self,
        element_id: str,
        depth: int = 2,
        max_elements: int = 10,
    ) -> List[CodeElement]:
        """Get related elements for a given element."""
        if element_id not in self.graph.elements:
            return []
        
        visited = {element_id}
        current_level = {element_id}
        all_related = []
        
        for _ in range(depth):
            next_level = set()
            
            for elem_id in current_level:
                element = self.graph.elements.get(elem_id)
                if not element:
                    continue
                
                # Add dependencies
                for dep_id in element.dependencies:
                    if dep_id not in visited:
                        visited.add(dep_id)
                        next_level.add(dep_id)
                        dep_elem = self.graph.get_element(dep_id)
                        if dep_elem:
                            all_related.append(dep_elem)
                
                # Add dependents
                for dep_id in element.dependents:
                    if dep_id not in visited:
                        visited.add(dep_id)
                        next_level.add(dep_id)
                        dep_elem = self.graph.get_element(dep_id)
                        if dep_elem:
                            all_related.append(dep_elem)
            
            current_level = next_level
            if not current_level:
                break
        
        return all_related[:max_elements]
