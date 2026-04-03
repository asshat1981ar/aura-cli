import ast
import os
from collections import defaultdict
from typing import Dict, List, Set

class DuplicateCodeReducer:
    def __init__(self, codebase_path: str):
        self.codebase_path = codebase_path
        self.duplicate_blocks: Dict[str, List[str]] = defaultdict(list)
        self.visited_nodes: Set[str] = set()
    
    def analyze_duplicates(self) -> Dict[str, List[str]]:
        """Analyze codebase for duplicate code blocks using AST parsing."""
        for root, _, files in os.walk(self.codebase_path):
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    self._parse_file(file_path)
        return dict(self.duplicate_blocks)
    
    def _parse_file(self, file_path: str):
        """Parse individual Python file and extract code blocks."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            tree = ast.parse(content)
            self._extract_blocks(tree, file_path)
        except Exception as e:
            print(f"Error parsing {file_path}: {e}")
    
    def _extract_blocks(self, node: ast.AST, file_path: str):
        """Extract function and class definitions as potential duplicates."""
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.ClassDef)):
                block_code = ast.get_source_segment(open(file_path).read(), child)
                block_hash = hash(block_code.strip())
                if block_hash in self.visited_nodes:
                    self.duplicate_blocks[block_hash].append(f"{file_path}:{child.name}")
                else:
                    self.visited_nodes.add(block_hash)
                    self.duplicate_blocks[block_hash] = [f"{file_path}:{child.name}"]
            self._extract_blocks(child, file_path)
    
    def generate_refactor_plan(self) -> Dict[str, List[str]]:
        """Generate a plan to refactor identified duplicates."""
        refactor_plan = {}
        for block_hash, locations in self.duplicate_blocks.items():
            if len(locations) > 1:  # Only consider actual duplicates
                refactor_plan[block_hash] = locations
        return refactor_plan
