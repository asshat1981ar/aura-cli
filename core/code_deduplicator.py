import ast
from dataclasses import dataclass
from typing import List, Dict, Set, Optional
from pathlib import Path

@dataclass
class DuplicateCode:
    source_file: str
    target_file: str
    source_lines: List[str]
    similarity_score: float
    line_numbers: List[int]

class CodeDeduplicator:
    def __init__(self):
        self.duplicates: List[DuplicateCode] = []
        self.utility_functions: Dict[str, str] = {}
    
    def detect_duplicates(self, files: List[str], min_lines: int = 3,
                         similarity_threshold: float = 0.8) -> List[DuplicateCode]:
        """Detect duplicate code segments across files."""
        self.duplicates = []
        for source_file in files:
            with open(source_file) as f:
                source_content = f.readlines()
            
            for target_file in files:
                if source_file == target_file:
                    continue
                    
                duplicates = self._find_similar_segments(
                    source_file, target_file, source_content,
                    min_lines, similarity_threshold
                )
                self.duplicates.extend(duplicates)
        
        return self.duplicates
    
    def extract_utility_function(self, duplicate: DuplicateCode,
                               function_name: str) -> str:
        """Convert duplicate code into a reusable utility function."""
        # Remove leading/trailing whitespace and common indentation
        code_lines = [line.rstrip() for line in duplicate.source_lines]
        min_indent = min(len(line) - len(line.lstrip()) 
                        for line in code_lines if line.strip())
        code_lines = [line[min_indent:] for line in code_lines]
        
        # Generate function definition
        params = self._extract_parameters(code_lines)
        fn_def = f"def {function_name}({', '.join(params)}):\n"
        fn_body = '\n'.join(f"    {line}" for line in code_lines)
        
        utility_fn = f"{fn_def}{fn_body}"
        self.utility_functions[function_name] = utility_fn
        return utility_fn
    
    def _find_similar_segments(self, source_file: str, target_file: str,
                             source_lines: List[str], min_lines: int,
                             similarity_threshold: float) -> List[DuplicateCode]:
        """Find similar code segments between files."""
        duplicates = []
        with open(target_file) as f:
            target_lines = f.readlines()
            
        for i in range(len(source_lines) - min_lines + 1):
            source_segment = source_lines[i:i + min_lines]
            
            for j in range(len(target_lines) - min_lines + 1):
                target_segment = target_lines[j:j + min_lines]
                similarity = self._calculate_similarity(source_segment, target_segment)
                
                if similarity >= similarity_threshold:
                    duplicates.append(DuplicateCode(
                        source_file=source_file,
                        target_file=target_file,
                        source_lines=source_segment,
                        similarity_score=similarity,
                        line_numbers=[i+1, j+1]
                    ))
        
        return duplicates
    
    def _calculate_similarity(self, lines1: List[str], lines2: List[str]) -> float:
        """Calculate similarity score between two code segments."""
        # Simple line-by-line comparison for now
        # Could be enhanced with more sophisticated algorithms
        matching_lines = sum(1 for l1, l2 in zip(lines1, lines2) 
                            if l1.strip() == l2.strip())
        return matching_lines / max(len(lines1), len(lines2))
    
    def _extract_parameters(self, code_lines: List[str]) -> Set[str]:
        """Extract potential parameters from code segment."""
        code = '\n'.join(code_lines)
        tree = ast.parse(code)
        
        params = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                params.add(node.id)
                
        # Filter out builtin names and common imports
        builtins = set(dir(__builtins__))
        return {p for p in params if p not in builtins}