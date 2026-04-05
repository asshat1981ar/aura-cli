#!/usr/bin/env python3
"""
Coverage Gap Analyzer - Identifies untested code paths and generates actionable reports.

This tool parses pytest-cov output, analyzes the codebase to identify high-impact
untested functions/classes, and generates coverage_gaps.json with severity scoring
based on complexity and impact.

Usage:
    python tools/coverage_gap_analyzer.py [--output OUTPUT] [--min-severity SEVERITY]

Example:
    python tools/coverage_gap_analyzer.py --output coverage_gaps.json --min-severity high
"""

import argparse
import json
import re
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.logging_utils import log_json


@dataclass
class CoverageGap:
    """Represents a single coverage gap (untested function/class)."""
    id: str
    file_path: str
    function_name: str
    line_number: int
    complexity: int = 0
    impact_score: float = 0.0
    severity: str = "medium"  # critical, high, medium, low
    reason: str = ""
    lines_of_code: int = 0
    callers: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class ModuleCoverage:
    """Coverage statistics for a module."""
    name: str
    path: str
    coverage_percent: float
    lines_total: int
    lines_covered: int
    lines_missing: int
    branches_total: int
    branches_covered: int
    functions_total: int
    functions_covered: int
    gaps: List[CoverageGap] = field(default_factory=list)


class CoverageGapAnalyzer:
    """Analyzes test coverage and identifies high-impact gaps."""
    
    def __init__(self, project_root: Optional[Path] = None):
        """
        Initialize the analyzer.
        
        Args:
            project_root: Root directory of the project to analyze
        """
        self.project_root = project_root or Path.cwd()
        self.coverage_data: Dict[str, Any] = {}
        self.gaps: List[CoverageGap] = []
        self.modules: List[ModuleCoverage] = []
        
    def run_coverage_collection(self, output_format: str = "json") -> Optional[Path]:
        """
        Run pytest with coverage to collect coverage data.
        
        Args:
            output_format: Format for coverage output (json, xml, html)
            
        Returns:
            Path to coverage output file or None if collection failed
        """
        coverage_file = self.project_root / f".coverage.{output_format}"
        
        try:
            log_json("INFO", "running_coverage_collection")
            
            # Run pytest with coverage
            cmd = [
                "python", "-m", "pytest",
                "--cov=.",
                "--cov-report", f"{output_format}:{coverage_file}",
                "--cov-report", "term-missing",
                "-q",  # Quiet mode
            ]
            
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            if coverage_file.exists():
                log_json("INFO", "coverage_collection_complete", {
                    "output_file": str(coverage_file),
                    "return_code": result.returncode
                })
                return coverage_file
            else:
                log_json("ERROR", "coverage_collection_failed", {
                    "stderr": result.stderr[:500]
                })
                return None
                
        except subprocess.TimeoutExpired:
            log_json("ERROR", "coverage_collection_timeout")
            return None
        except Exception as e:
            log_json("ERROR", "coverage_collection_error", {"error": str(e)})
            return None
    
    def parse_coverage_json(self, coverage_file: Path) -> bool:
        """
        Parse coverage JSON output.
        
        Args:
            coverage_file: Path to coverage JSON file
            
        Returns:
            True if parsing succeeded
        """
        try:
            with open(coverage_file) as f:
                self.coverage_data = json.load(f)
            
            log_json("INFO", "coverage_json_parsed", {
                "files_analyzed": len(self.coverage_data.get("files", {}))
            })
            return True
            
        except Exception as e:
            log_json("ERROR", "coverage_json_parse_failed", {"error": str(e)})
            return False
    
    def parse_coverage_xml(self, coverage_file: Path) -> bool:
        """
        Parse coverage XML output (cov.xml).
        
        Args:
            coverage_file: Path to coverage XML file
            
        Returns:
            True if parsing succeeded
        """
        try:
            import xml.etree.ElementTree as ET
            
            tree = ET.parse(coverage_file)
            root = tree.getroot()
            
            files_data = {}
            
            for package in root.findall('.//package'):
                for cls in package.findall('.//class'):
                    filename = cls.get('filename', '')
                    if not filename:
                        continue
                    
                    lines = []
                    for line in cls.findall('.//line'):
                        line_num = int(line.get('number', 0))
                        hits = int(line.get('hits', 0))
                        lines.append({
                            'number': line_num,
                            'hits': hits,
                            'missing': hits == 0
                        })
                    
                    files_data[filename] = {'lines': lines}
            
            self.coverage_data = {'files': files_data}
            
            log_json("INFO", "coverage_xml_parsed", {
                "files_analyzed": len(files_data)
            })
            return True
            
        except Exception as e:
            log_json("ERROR", "coverage_xml_parse_failed", {"error": str(e)})
            return False
    
    def calculate_complexity(self, file_path: Path, function_name: str, 
                            line_start: int, line_end: int) -> int:
        """
        Calculate cyclomatic complexity for a function.
        
        Args:
            file_path: Path to the source file
            function_name: Name of the function
            line_start: Starting line number
            line_end: Ending line number
            
        Returns:
            Cyclomatic complexity score
        """
        try:
            if not file_path.exists():
                return 0
            
            with open(file_path) as f:
                lines = f.readlines()[line_start-1:line_end]
            
            code = ''.join(lines)
            
            # Simple complexity calculation based on control flow keywords
            complexity = 1  # Base complexity
            
            # Count control flow statements
            complexity += len(re.findall(r'\bif\b', code))
            complexity += len(re.findall(r'\belif\b', code))
            complexity += len(re.findall(r'\bfor\b', code))
            complexity += len(re.findall(r'\bwhile\b', code))
            complexity += len(re.findall(r'\bexcept\b', code))
            complexity += len(re.findall(r'\bwith\b', code))
            complexity += len(re.findall(r'\bassert\b', code))
            complexity += len(re.findall(r'\band\b', code))
            complexity += len(re.findall(r'\bor\b', code))
            
            return complexity
            
        except Exception as e:
            log_json("WARN", "complexity_calculation_failed", {
                "file": str(file_path),
                "function": function_name,
                "error": str(e)
            })
            return 0
    
    def find_function_boundaries(self, file_path: Path) -> List[Dict[str, Any]]:
        """
        Find all function/class boundaries in a Python file.
        
        Args:
            file_path: Path to Python source file
            
        Returns:
            List of function/class metadata dictionaries
        """
        functions = []
        
        try:
            if not file_path.exists():
                return functions
            
            with open(file_path) as f:
                content = f.read()
                lines = content.split('\n')
            
            # Pattern to match function and class definitions
            pattern = r'^(async\s+)?def\s+(\w+)|^class\s+(\w+)'
            
            current_func = None
            current_indent = 0
            
            for i, line in enumerate(lines, 1):
                match = re.match(pattern, line)
                
                if match:
                    # Save previous function
                    if current_func:
                        current_func['end_line'] = i - 1
                        functions.append(current_func)
                    
                    # Start new function
                    func_name = match.group(2) or match.group(3)
                    is_class = match.group(3) is not None
                    current_func = {
                        'name': func_name,
                        'start_line': i,
                        'is_class': is_class,
                        'indent': len(line) - len(line.lstrip())
                    }
                    current_indent = len(line) - len(line.lstrip())
                
                elif current_func:
                    # Check if we've exited the function
                    if line.strip() and not line.strip().startswith('#'):
                        indent = len(line) - len(line.lstrip())
                        if indent <= current_indent:
                            current_func['end_line'] = i - 1
                            functions.append(current_func)
                            current_func = None
            
            # Don't forget the last function
            if current_func:
                current_func['end_line'] = len(lines)
                functions.append(current_func)
            
        except Exception as e:
            log_json("WARN", "function_boundary_detection_failed", {
                "file": str(file_path),
                "error": str(e)
            })
        
        return functions
    
    def calculate_impact_score(self, gap: CoverageGap, 
                               file_coverage: Dict[str, Any]) -> float:
        """
        Calculate impact score for a coverage gap.
        
        Score is based on:
        - Complexity (higher = more impact)
        - Lines of code (higher = more impact)
        - File coverage (lower file coverage = more impact)
        - Function importance (estimated by complexity)
        
        Args:
            gap: Coverage gap to score
            file_coverage: Coverage data for the file
            
        Returns:
            Impact score (0-100)
        """
        score = 0.0
        
        # Complexity factor (0-40 points)
        score += min(gap.complexity * 2, 40)
        
        # Lines of code factor (0-20 points)
        score += min(gap.lines_of_code * 0.5, 20)
        
        # File coverage factor (0-30 points)
        # Lower file coverage = higher impact
        file_cov_pct = file_coverage.get('coverage_percent', 100)
        score += (100 - file_cov_pct) * 0.3
        
        # Critical path bonus (0-10 points)
        if gap.complexity > 10 and gap.lines_of_code > 20:
            score += 10
        
        return min(score, 100)
    
    def determine_severity(self, impact_score: float, complexity: int) -> str:
        """
        Determine severity level based on impact and complexity.
        
        Args:
            impact_score: Calculated impact score
            complexity: Cyclomatic complexity
            
        Returns:
            Severity string: critical, high, medium, or low
        """
        if impact_score >= 80 or complexity >= 20:
            return "critical"
        elif impact_score >= 60 or complexity >= 15:
            return "high"
        elif impact_score >= 40 or complexity >= 10:
            return "medium"
        else:
            return "low"
    
    def identify_gaps(self) -> List[CoverageGap]:
        """
        Identify all coverage gaps in the codebase.
        
        Returns:
            List of coverage gaps sorted by impact score
        """
        gaps = []
        gap_id = 0
        
        files_data = self.coverage_data.get('files', {})
        
        for file_path_str, file_data in files_data.items():
            # Skip test files and non-Python files
            if not file_path_str.endswith('.py'):
                continue
            if 'test' in file_path_str.lower():
                continue
            
            file_path = self.project_root / file_path_str
            
            # Find all functions in the file
            functions = self.find_function_boundaries(file_path)
            
            # Get missing lines from coverage data
            missing_lines = set()
            for line_data in file_data.get('lines', []):
                if line_data.get('missing', False) or line_data.get('hits', 1) == 0:
                    missing_lines.add(line_data['number'])
            
            # Check each function for coverage gaps
            for func in functions:
                gap_lines = []
                for line_num in range(func['start_line'], func['end_line'] + 1):
                    if line_num in missing_lines:
                        gap_lines.append(line_num)
                
                # If function has uncovered lines, create a gap
                if gap_lines:
                    complexity = self.calculate_complexity(
                        file_path,
                        func['name'],
                        func['start_line'],
                        func['end_line']
                    )
                    
                    gap_id += 1
                    gap = CoverageGap(
                        id=f"gap_{gap_id:04d}",
                        file_path=file_path_str,
                        function_name=func['name'],
                        line_number=func['start_line'],
                        complexity=complexity,
                        lines_of_code=func['end_line'] - func['start_line'] + 1,
                        reason=self._generate_reason(func, gap_lines)
                    )
                    
                    # Calculate impact score
                    file_coverage = {
                        'coverage_percent': file_data.get('coverage', 100)
                    }
                    gap.impact_score = self.calculate_impact_score(gap, file_coverage)
                    gap.severity = self.determine_severity(gap.impact_score, gap.complexity)
                    
                    gaps.append(gap)
        
        # Sort by impact score (highest first)
        gaps.sort(key=lambda g: g.impact_score, reverse=True)
        
        self.gaps = gaps
        log_json("INFO", "gaps_identified", {"count": len(gaps)})
        
        return gaps
    
    def _generate_reason(self, func: Dict[str, Any], gap_lines: List[int]) -> str:
        """Generate a human-readable reason for the coverage gap."""
        if func.get('is_class'):
            return f"Class methods uncovered ({len(gap_lines)} lines)"
        elif len(gap_lines) > 10:
            return "Large untested function"
        elif func['name'].startswith('_'):
            return "Private method needs testing"
        else:
            return "Function not covered by tests"
    
    def generate_report(self, min_severity: str = "low") -> Dict[str, Any]:
        """
        Generate comprehensive coverage report.
        
        Args:
            min_severity: Minimum severity level to include (critical, high, medium, low)
            
        Returns:
            Report dictionary with gaps, statistics, and recommendations
        """
        severity_levels = {"critical": 4, "high": 3, "medium": 2, "low": 1}
        min_level = severity_levels.get(min_severity, 1)
        
        filtered_gaps = [
            g for g in self.gaps 
            if severity_levels.get(g.severity, 0) >= min_level
        ]
        
        # Calculate statistics
        total_files = len(set(g.file_path for g in self.gaps))
        
        stats = {
            "total_gaps": len(self.gaps),
            "filtered_gaps": len(filtered_gaps),
            "files_affected": total_files,
            "severity_breakdown": {
                "critical": len([g for g in self.gaps if g.severity == "critical"]),
                "high": len([g for g in self.gaps if g.severity == "high"]),
                "medium": len([g for g in self.gaps if g.severity == "medium"]),
                "low": len([g for g in self.gaps if g.severity == "low"]),
            },
            "average_complexity": sum(g.complexity for g in self.gaps) / len(self.gaps) if self.gaps else 0,
            "average_impact": sum(g.impact_score for g in self.gaps) / len(self.gaps) if self.gaps else 0,
        }
        
        # Generate recommendations
        recommendations = self._generate_recommendations(filtered_gaps)
        
        report = {
            "generated_at": str(Path.cwd()),
            "project_root": str(self.project_root),
            "statistics": stats,
            "gaps": [g.to_dict() for g in filtered_gaps[:100]],  # Top 100 gaps
            "recommendations": recommendations,
        }
        
        return report
    
    def _generate_recommendations(self, gaps: List[CoverageGap]) -> List[str]:
        """Generate actionable recommendations based on gaps."""
        recommendations = []
        
        if not gaps:
            recommendations.append("Excellent! No significant coverage gaps found.")
            return recommendations
        
        critical_count = len([g for g in gaps if g.severity == "critical"])
        high_count = len([g for g in gaps if g.severity == "high"])
        
        if critical_count > 0:
            recommendations.append(
                f"Priority 1: Address {critical_count} critical coverage gaps immediately. "
                "These functions have high complexity and are completely untested."
            )
        
        if high_count > 0:
            recommendations.append(
                f"Priority 2: Write tests for {high_count} high-impact uncovered functions."
            )
        
        # Find files with most gaps
        file_gaps: Dict[str, int] = {}
        for gap in gaps:
            file_gaps[gap.file_path] = file_gaps.get(gap.file_path, 0) + 1
        
        worst_files = sorted(file_gaps.items(), key=lambda x: x[1], reverse=True)[:3]
        if worst_files:
            recommendations.append(
                f"Focus on these files with most gaps: {', '.join(f[0] for f in worst_files)}"
            )
        
        return recommendations
    
    def export_json(self, output_path: Path, min_severity: str = "low") -> bool:
        """
        Export coverage gaps to JSON file.
        
        Args:
            output_path: Path to output JSON file
            min_severity: Minimum severity level to include
            
        Returns:
            True if export succeeded
        """
        try:
            report = self.generate_report(min_severity)
            
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            log_json("INFO", "coverage_report_exported", {
                "output_path": str(output_path),
                "gaps_included": report["statistics"]["filtered_gaps"]
            })
            
            return True
            
        except Exception as e:
            log_json("ERROR", "export_failed", {"error": str(e)})
            return False
    
    def export_html(self, output_path: Path, min_severity: str = "low") -> bool:
        """
        Export coverage gaps to HTML report.
        
        Args:
            output_path: Path to output HTML file
            min_severity: Minimum severity level to include
            
        Returns:
            True if export succeeded
        """
        try:
            report = self.generate_report(min_severity)
            
            html_content = self._generate_html_report(report)
            
            with open(output_path, 'w') as f:
                f.write(html_content)
            
            log_json("INFO", "html_report_exported", {"output_path": str(output_path)})
            return True
            
        except Exception as e:
            log_json("ERROR", "html_export_failed", {"error": str(e)})
            return False
    
    def _generate_html_report(self, report: Dict[str, Any]) -> str:
        """Generate HTML report content."""
        gaps = report.get("gaps", [])
        stats = report.get("statistics", {})
        
        rows = ""
        for gap in gaps:
            severity_class = f"severity-{gap['severity']}"
            rows += f"""
            <tr class="{severity_class}">
                <td><span class="badge {severity_class}">{gap['severity']}</span></td>
                <td>{gap['function_name']}</td>
                <td><code>{gap['file_path']}:{gap['line_number']}</code></td>
                <td>{gap['complexity']}</td>
                <td>{gap['impact_score']:.1f}</td>
                <td>{gap['reason']}</td>
            </tr>
            """
        
        return f"""<!DOCTYPE html>
<html>
<head>
    <title>Coverage Gap Analysis Report</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 40px; background: #f5f5f5; }}
        .container {{ max-width: 1400px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        h1 {{ color: #333; margin-bottom: 10px; }}
        .stats {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 20px; margin: 30px 0; }}
        .stat-card {{ background: #f8f9fa; padding: 20px; border-radius: 6px; text-align: center; }}
        .stat-value {{ font-size: 32px; font-weight: bold; color: #2563eb; }}
        .stat-label {{ color: #666; margin-top: 5px; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 30px; }}
        th {{ background: #f1f5f9; padding: 12px; text-align: left; font-weight: 600; color: #475569; border-bottom: 2px solid #e2e8f0; }}
        td {{ padding: 12px; border-bottom: 1px solid #e2e8f0; }}
        tr:hover {{ background: #f8fafc; }}
        .badge {{ padding: 4px 8px; border-radius: 4px; font-size: 12px; font-weight: 600; text-transform: uppercase; }}
        .severity-critical {{ background: #fee2e2; color: #dc2626; }}
        .severity-high {{ background: #ffedd5; color: #ea580c; }}
        .severity-medium {{ background: #fef3c7; color: #d97706; }}
        .severity-low {{ background: #dbeafe; color: #2563eb; }}
        code {{ background: #f1f5f9; padding: 2px 6px; border-radius: 3px; font-size: 13px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🔍 Coverage Gap Analysis Report</h1>
        <p>Generated on {Path.cwd()} | {stats.get('total_gaps', 0)} gaps identified</p>
        
        <div class="stats">
            <div class="stat-card">
                <div class="stat-value">{stats.get('total_gaps', 0)}</div>
                <div class="stat-label">Total Gaps</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{stats.get('severity_breakdown', {}).get('critical', 0)}</div>
                <div class="stat-label">Critical</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{stats.get('severity_breakdown', {}).get('high', 0)}</div>
                <div class="stat-label">High Priority</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{stats.get('average_impact', 0):.1f}</div>
                <div class="stat-label">Avg Impact Score</div>
            </div>
        </div>
        
        <h2>Top Coverage Gaps</h2>
        <table>
            <thead>
                <tr>
                    <th>Severity</th>
                    <th>Function</th>
                    <th>Location</th>
                    <th>Complexity</th>
                    <th>Impact</th>
                    <th>Reason</th>
                </tr>
            </thead>
            <tbody>
                {rows}
            </tbody>
        </table>
    </div>
</body>
</html>"""


def main():
    """Main entry point for CLI usage."""
    parser = argparse.ArgumentParser(
        description="Analyze test coverage and identify high-impact gaps"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="coverage_gaps.json",
        help="Output file path (default: coverage_gaps.json)"
    )
    parser.add_argument(
        "--min-severity", "-s",
        type=str,
        choices=["critical", "high", "medium", "low"],
        default="low",
        help="Minimum severity level to include (default: low)"
    )
    parser.add_argument(
        "--format", "-f",
        type=str,
        choices=["json", "html"],
        default="json",
        help="Output format (default: json)"
    )
    parser.add_argument(
        "--project-root", "-p",
        type=str,
        default=".",
        help="Project root directory (default: current directory)"
    )
    parser.add_argument(
        "--collect-coverage", "-c",
        action="store_true",
        help="Run pytest to collect coverage data first"
    )
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = CoverageGapAnalyzer(project_root=Path(args.project_root))
    
    # Collect coverage if requested
    if args.collect_coverage:
        coverage_file = analyzer.run_coverage_collection()
        if not coverage_file:
            print("❌ Failed to collect coverage data", file=sys.stderr)
            sys.exit(1)
        
        # Parse based on format
        if coverage_file.suffix == '.json':
            success = analyzer.parse_coverage_json(coverage_file)
        else:
            success = analyzer.parse_coverage_xml(coverage_file)
        
        if not success:
            print("❌ Failed to parse coverage data", file=sys.stderr)
            sys.exit(1)
    else:
        # Try to find existing coverage file
        coverage_json = analyzer.project_root / ".coverage.json"
        coverage_xml = analyzer.project_root / "coverage.xml"
        
        if coverage_json.exists():
            success = analyzer.parse_coverage_json(coverage_json)
        elif coverage_xml.exists():
            success = analyzer.parse_coverage_xml(coverage_xml)
        else:
            print("❌ No coverage file found. Run with --collect-coverage or generate coverage first.", file=sys.stderr)
            sys.exit(1)
        
        if not success:
            sys.exit(1)
    
    # Identify gaps
    analyzer.identify_gaps()
    
    # Export report
    output_path = Path(args.output)
    
    if args.format == "html":
        success = analyzer.export_html(output_path, args.min_severity)
    else:
        success = analyzer.export_json(output_path, args.min_severity)
    
    if success:
        print(f"✅ Coverage report exported to {output_path}")
        print(f"   Total gaps: {len(analyzer.gaps)}")
        print(f"   Critical: {len([g for g in analyzer.gaps if g.severity == 'critical'])}")
        print(f"   High: {len([g for g in analyzer.gaps if g.severity == 'high'])}")
    else:
        print("❌ Failed to export report", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
