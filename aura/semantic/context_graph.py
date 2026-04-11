"""Semantic context graph for code understanding."""

import ast
import hashlib
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


class ElementType(Enum):
    """Types of code elements."""

    MODULE = "module"
    CLASS = "class"
    FUNCTION = "function"
    METHOD = "method"
    VARIABLE = "variable"
    IMPORT = "import"


@dataclass
class CodeElement:
    """A code element in the semantic graph."""

    name: str
    type: ElementType
    file_path: str
    line_start: int
    line_end: int
    docstring: Optional[str] = None
    signature: Optional[str] = None
    dependencies: Set[str] = field(default_factory=set)
    dependents: Set[str] = field(default_factory=set)
    metrics: Dict[str, float] = field(default_factory=dict)

    @property
    def id(self) -> str:
        """Generate unique ID for this element."""
        content = f"{self.file_path}:{self.name}:{self.line_start}"
        return hashlib.md5(content.encode()).hexdigest()[:12]

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "type": self.type.value,
            "file_path": self.file_path,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "signature": self.signature,
            "dependencies": list(self.dependencies),
            "metrics": self.metrics,
        }


class ContextGraph:
    """Graph of code elements and their relationships."""

    def __init__(self):
        self.elements: Dict[str, CodeElement] = {}
        self.file_index: Dict[str, List[str]] = {}
        self.dependency_graph: Dict[str, Set[str]] = {}

    def add_element(self, element: CodeElement) -> str:
        """Add an element to the graph."""
        elem_id = element.id
        self.elements[elem_id] = element

        # Index by file
        if element.file_path not in self.file_index:
            self.file_index[element.file_path] = []
        self.file_index[element.file_path].append(elem_id)

        return elem_id

    def get_element(self, elem_id: str) -> Optional[CodeElement]:
        """Get element by ID."""
        return self.elements.get(elem_id)

    def get_file_elements(self, file_path: str) -> List[CodeElement]:
        """Get all elements in a file."""
        elem_ids = self.file_index.get(file_path, [])
        return [self.elements[eid] for eid in elem_ids if eid in self.elements]

    def add_dependency(self, from_id: str, to_id: str):
        """Add a dependency relationship."""
        if from_id in self.elements and to_id in self.elements:
            self.elements[from_id].dependencies.add(to_id)
            self.elements[to_id].dependents.add(from_id)

            if from_id not in self.dependency_graph:
                self.dependency_graph[from_id] = set()
            self.dependency_graph[from_id].add(to_id)

    def get_dependencies(self, elem_id: str) -> List[CodeElement]:
        """Get elements that elem_id depends on."""
        element = self.elements.get(elem_id)
        if not element:
            return []
        return [self.elements[eid] for eid in element.dependencies if eid in self.elements]

    def get_dependents(self, elem_id: str) -> List[CodeElement]:
        """Get elements that depend on elem_id."""
        element = self.elements.get(elem_id)
        if not element:
            return []
        return [self.elements[eid] for eid in element.dependents if eid in self.elements]

    def analyze_file(self, file_path: Path) -> List[CodeElement]:
        """Analyze a Python file and extract code elements."""
        elements = []

        try:
            content = file_path.read_text()
            tree = ast.parse(content)
        except Exception:
            return elements

        lines = content.split("\n")

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                elem = self._extract_class(node, file_path, lines)
                if elem:
                    elements.append(elem)
                    self.add_element(elem)

            elif isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
                elem = self._extract_function(node, file_path, lines)
                if elem:
                    elements.append(elem)
                    self.add_element(elem)

        # Second pass: build relationships
        self._build_relationships(tree, elements)

        return elements

    def _extract_class(self, node: ast.ClassDef, file_path: Path, lines: List[str]) -> Optional[CodeElement]:
        """Extract class information."""
        docstring = ast.get_docstring(node)

        # Get class signature (bases)
        bases = [self._get_name(base) for base in node.bases]
        signature = f"class {node.name}({', '.join(bases)})" if bases else f"class {node.name}"

        return CodeElement(
            name=node.name,
            type=ElementType.CLASS,
            file_path=str(file_path),
            line_start=node.lineno,
            line_end=node.end_lineno or node.lineno,
            docstring=docstring,
            signature=signature,
        )

    def _extract_function(self, node: ast.FunctionDef, file_path: Path, lines: List[str]) -> Optional[CodeElement]:
        """Extract function information."""
        docstring = ast.get_docstring(node)

        # Get function signature
        args_str = self._format_args(node.args)
        signature = f"def {node.name}({args_str})"

        # Determine if method or function
        elem_type = ElementType.METHOD if self._is_method(node) else ElementType.FUNCTION

        return CodeElement(
            name=node.name,
            type=elem_type,
            file_path=str(file_path),
            line_start=node.lineno,
            line_end=node.end_lineno or node.lineno,
            docstring=docstring,
            signature=signature,
        )

    def _build_relationships(self, tree: ast.AST, elements: List[CodeElement]):
        """Build dependency relationships between elements."""
        # Create name to element ID mapping
        name_map = {elem.name: elem.id for elem in elements}

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                parent_name = node.name
                parent_id = name_map.get(parent_name)

                if not parent_id:
                    continue

                # Find all Name and Attribute nodes (potential dependencies)
                for child in ast.walk(node):
                    if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load):
                        if child.id in name_map and name_map[child.id] != parent_id:
                            self.add_dependency(parent_id, name_map[child.id])

    def _get_name(self, node: ast.AST) -> str:
        """Get name from AST node."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._get_name(node.value)}.{node.attr}"
        return ""

    def _format_args(self, args: ast.arguments) -> str:
        """Format function arguments."""
        parts = []

        # Regular args
        for arg in args.args:
            arg_str = arg.arg
            if arg.annotation:
                arg_str += f": {self._get_name(arg.annotation)}"
            parts.append(arg_str)

        # Vararg
        if args.vararg:
            parts.append(f"*{args.vararg.arg}")

        # Kwarg
        if args.kwarg:
            parts.append(f"**{args.kwarg.arg}")

        return ", ".join(parts)

    def _is_method(self, node: ast.FunctionDef) -> bool:
        """Check if function is a method (has self/cls parameter)."""
        if node.args.args:
            first_arg = node.args.args[0].arg
            return first_arg in ("self", "cls")
        return False

    def query(self, query_str: str, limit: int = 10) -> List[Tuple[CodeElement, float]]:
        """Query the graph for relevant elements."""
        results = []
        query_lower = query_str.lower()

        for elem in self.elements.values():
            score = 0.0

            # Name match
            if query_lower in elem.name.lower():
                score += 1.0

            # Docstring match
            if elem.docstring and query_lower in elem.docstring.lower():
                score += 0.5

            # Signature match
            if elem.signature and query_lower in elem.signature.lower():
                score += 0.3

            if score > 0:
                results.append((elem, score))

        # Sort by score descending
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:limit]

    def to_dict(self) -> dict:
        """Serialize graph to dictionary."""
        return {
            "elements": {eid: elem.to_dict() for eid, elem in self.elements.items()},
            "total_elements": len(self.elements),
            "files_indexed": len(self.file_index),
        }
