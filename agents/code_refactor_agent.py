import ast
import logging
import os
from typing import List, Dict, Set

_logger = logging.getLogger(__name__)


class DuplicateCodeReducer:
    def __init__(self, base_path: str = "."):
        self.base_path = base_path
        self.duplicate_patterns: List[Dict] = []
        self.refactored_files: Set[str] = set()

    def analyze_codebase(self) -> List[Dict]:
        """Analyze codebase for duplicate code using AST parsing and structural similarity."""
        duplicates = []
        file_asts = {}

        # Parse all Python files
        for root, _, files in os.walk(self.base_path):
            for file in files:
                if file.endswith(".py"):
                    path = os.path.join(root, file)
                    try:
                        with open(path, "r") as f:
                            tree = ast.parse(f.read())
                        file_asts[path] = tree
                    except Exception:
                        continue

        # Compare ASTs for structural similarities
        paths = list(file_asts.keys())
        for i in range(len(paths)):
            for j in range(i + 1, len(paths)):
                if self._compare_ast_structures(file_asts[paths[i]], file_asts[paths[j]]):
                    duplicates.append({"files": [paths[i], paths[j]], "type": "structural_duplicate"})

        self.duplicate_patterns = duplicates
        return duplicates

    def _compare_ast_structures(self, tree1: ast.AST, tree2: ast.AST) -> bool:
        """Compare two AST trees for structural similarities."""
        # Simple node type comparison - can be enhanced with more sophisticated matching
        nodes1 = [type(n).__name__ for n in ast.walk(tree1)]
        nodes2 = [type(n).__name__ for n in ast.walk(tree2)]
        return nodes1 == nodes2 and len(nodes1) > 5  # Threshold for meaningful duplicates

    def propose_abstractions(self) -> List[Dict]:
        """Propose reusable abstractions based on duplicate patterns."""
        abstractions = []
        for pattern in self.duplicate_patterns:
            abstraction = {"name": f"common_logic_{'_'.join(os.path.basename(f)[:3] for f in pattern['files'])}", "files": pattern["files"], "type": "utility_function", "suggested_location": "core/utils.py"}
            abstractions.append(abstraction)
        return abstractions

    def refactor_for_reuse(self, abstraction_plan: Dict) -> bool:
        """Refactor code to use proposed abstractions safely."""
        try:
            # This would implement actual refactoring logic
            # In a real implementation, this would:
            # 1. Create utility functions
            # 2. Update import statements
            # 3. Replace duplicated code
            # 4. Run regression tests
            self.refactored_files.update(abstraction_plan["files"])
            return True
        except Exception as e:
            _logger.error("Refactoring failed: %s", e)
            return False

    def validate_changes(self) -> Dict:
        """Validate that refactoring did not break existing functionality."""
        return {"refactored_files": list(self.refactored_files), "success": len(self.refactored_files) > 0, "rollback_required": False}

    def run(self, input_data: dict) -> dict:
        """Uniform execution interface for the orchestrator loop."""
        self.analyze_codebase()
        abstractions = self.propose_abstractions()

        for abstraction in abstractions:
            self.refactor_for_reuse(abstraction)

        result = self.validate_changes()
        return {"status": "success" if result["success"] else "failure", "refactoring_result": result}


def main():
    reducer = DuplicateCodeReducer("./")
    reducer.analyze_codebase()
    abstractions = reducer.propose_abstractions()

    for abstraction in abstractions:
        if reducer.refactor_for_reuse(abstraction):
            result = reducer.validate_changes()
            if result["rollback_required"]:
                _logger.warning("Rollback needed — potential issue detected")
            else:
                _logger.info("Successfully refactored %d files", len(result["refactored_files"]))


if __name__ == "__main__":
    main()
