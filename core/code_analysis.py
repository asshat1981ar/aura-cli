import os
import ast
from dataclasses import dataclass
from typing import List, Set, Dict, Any

@dataclass
class DuplicationMetrics:
    duplicated_lines: int
    total_lines: int
    duplicate_ratio: float
    hotspots: Dict[str, List[int]]

class CodeAnalyzer:
    def __init__(self):
        self.seen_snippets: Set[str] = set()
        
    def analyze_duplication(self, file_path: str) -> DuplicationMetrics:
        """Analyze code file for duplicated segments"""
        with open(file_path) as f:
            tree = ast.parse(f.read())
            
        metrics = DuplicationMetrics(
            duplicated_lines=0,
            total_lines=0,
            duplicate_ratio=0.0,
            hotspots={}
        )
        
        self._analyze_node(tree, metrics, file_path)
        
        if metrics.total_lines > 0:
            metrics.duplicate_ratio = metrics.duplicated_lines / metrics.total_lines
        
        return metrics
    
    def _analyze_node(self, node: ast.AST, metrics: DuplicationMetrics, file_path: str) -> None:
        """Recursively analyze AST nodes for duplication"""
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.FunctionDef):
                self._check_function_duplication(child, metrics, file_path)
            elif isinstance(child, (ast.ClassDef, ast.Module)):
                self._analyze_node(child, metrics, file_path)
    
    def _check_function_duplication(self, func: ast.FunctionDef, metrics: DuplicationMetrics, file_path: str) -> None:
        """Check individual function for code duplication"""
        func_str = ast.unparse(func)
        line_count = len(func_str.split('\n'))
        metrics.total_lines += line_count
        
        # Check for identical functions
        if func_str in self.seen_snippets:
            metrics.duplicated_lines += line_count
            if file_path not in metrics.hotspots:
                metrics.hotspots[file_path] = []
            metrics.hotspots[file_path].append(func.lineno)
        else:
            self.seen_snippets.add(func_str)
            
        # Check function body for duplicate blocks
        self._find_duplicate_blocks(func, metrics, file_path)
    
    def _find_duplicate_blocks(self, func: ast.FunctionDef, metrics: DuplicationMetrics, file_path: str) -> None:
        """Find duplicate code blocks within function body"""
        blocks: Dict[str, List[int]] = {}
        
        for node in ast.walk(func):
            if isinstance(node, (ast.If, ast.For, ast.While)):
                block_str = ast.unparse(node)
                if len(block_str.split('\n')) >= 3:  # Minimum size threshold
                    if block_str in blocks:
                        if file_path not in metrics.hotspots:
                            metrics.hotspots[file_path] = []
                        metrics.hotspots[file_path].extend([node.lineno])
                        metrics.duplicated_lines += len(block_str.split('\n'))
                    else:
                        blocks[block_str] = [node.lineno]

def scan_directory(directory: str) -> Dict[str, DuplicationMetrics]:
    """Scan directory recursively for Python files and analyze duplication"""
    results = {}
    analyzer = CodeAnalyzer()
    
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                try:
                    results[file_path] = analyzer.analyze_duplication(file_path)
                except Exception as e:
                    print(f'Error analyzing {file_path}: {e}')
                    
    return results