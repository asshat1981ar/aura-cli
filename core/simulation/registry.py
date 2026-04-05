"""Template registry for reusable simulation configurations."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.logging_utils import log_json


@dataclass
class SimulationTemplate:
    """A reusable simulation configuration template."""
    template_id: str
    name: str
    description: str
    base_scenario: str
    default_variables: Dict[str, List[Any]] = field(default_factory=dict)
    success_criteria: List[Dict[str, Any]] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    author: str = "system"
    version: str = "1.0"
    usage_count: int = 0
    
    def to_config(self, variable_overrides: Optional[Dict[str, List[Any]]] = None):
        """Convert template to SimulationConfig."""
        from core.simulation.engine import SimulationConfig, SuccessCriterion
        
        variables = {**self.default_variables}
        if variable_overrides:
            variables.update(variable_overrides)
        
        criteria = [
            SuccessCriterion(
                name=c["name"],
                metric=c["metric"],
                operator=c["operator"],
                threshold=c["threshold"],
                weight=c.get("weight", 1.0)
            )
            for c in self.success_criteria
        ]
        
        return SimulationConfig(
            name=self.name,
            base_scenario=self.base_scenario,
            variables=variables,
            success_criteria=criteria,
            description=self.description
        )


class TemplateRegistry:
    """Registry for simulation templates with persistence."""
    
    def __init__(self, storage_path: Optional[Path] = None):
        """
        Initialize the template registry.
        
        Args:
            storage_path: Path for persistent storage of templates
        """
        if storage_path is None:
            storage_path = Path(__file__).parent.parent.parent / "memory" / "simulation_templates.json"
        
        self.storage_path = storage_path
        self._templates: Dict[str, SimulationTemplate] = {}
        self._load()
    
    def register(self, template: SimulationTemplate) -> str:
        """Register a new template."""
        self._templates[template.template_id] = template
        self._save()
        
        log_json("INFO", "template_registered", {
            "template_id": template.template_id,
            "name": template.name
        })
        
        return template.template_id
    
    def get(self, template_id: str) -> Optional[SimulationTemplate]:
        """Get a template by ID."""
        template = self._templates.get(template_id)
        if template:
            template.usage_count += 1
            self._save()
        return template
    
    def list_templates(self, tag_filter: Optional[str] = None) -> List[SimulationTemplate]:
        """List all templates, optionally filtered by tag."""
        templates = list(self._templates.values())
        
        if tag_filter:
            templates = [t for t in templates if tag_filter in t.tags]
        
        return sorted(templates, key=lambda t: t.usage_count, reverse=True)
    
    def search(self, query: str) -> List[SimulationTemplate]:
        """Search templates by name, description, or tags."""
        query_lower = query.lower()
        results = []
        
        for template in self._templates.values():
            score = 0
            
            if query_lower in template.name.lower():
                score += 3
            if query_lower in template.description.lower():
                score += 2
            if any(query_lower in tag.lower() for tag in template.tags):
                score += 2
            if query_lower in template.base_scenario.lower():
                score += 1
            
            if score > 0:
                results.append((template, score))
        
        # Sort by relevance score
        results.sort(key=lambda x: x[1], reverse=True)
        return [r[0] for r in results]
    
    def delete(self, template_id: str) -> bool:
        """Delete a template."""
        if template_id in self._templates:
            del self._templates[template_id]
            self._save()
            return True
        return False
    
    def _load(self):
        """Load templates from storage."""
        if not self.storage_path.exists():
            self._register_default_templates()
            return
        
        try:
            data = json.loads(self.storage_path.read_text())
            for template_data in data.get("templates", []):
                template = SimulationTemplate(**template_data)
                self._templates[template.template_id] = template
        except (json.JSONDecodeError, TypeError, KeyError) as e:
            log_json("WARN", "template_load_failed", {"error": str(e)})
            self._register_default_templates()
    
    def _save(self):
        """Save templates to storage."""
        try:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "templates": [
                    {
                        "template_id": t.template_id,
                        "name": t.name,
                        "description": t.description,
                        "base_scenario": t.base_scenario,
                        "default_variables": t.default_variables,
                        "success_criteria": t.success_criteria,
                        "tags": t.tags,
                        "author": t.author,
                        "version": t.version,
                        "usage_count": t.usage_count
                    }
                    for t in self._templates.values()
                ]
            }
            self.storage_path.write_text(json.dumps(data, indent=2))
        except Exception as e:
            log_json("ERROR", "template_save_failed", {"error": str(e)})
    
    def _register_default_templates(self):
        """Register built-in templates."""
        defaults = [
            SimulationTemplate(
                template_id="prompt_style_comparison",
                name="Prompt Style Comparison",
                description="Compare different prompting styles for code generation",
                base_scenario="prompt_strategy",
                default_variables={
                    "prompt_style": ["detailed", "concise", "chain_of_thought", "few_shot"]
                },
                success_criteria=[
                    {"name": "high_effectiveness", "metric": "effectiveness", "operator": "gte", "threshold": 0.8, "weight": 2.0},
                    {"name": "good_quality", "metric": "response_quality", "operator": "gte", "threshold": 0.75},
                ],
                tags=["prompting", "optimization", "llm"]
            ),
            SimulationTemplate(
                template_id="temperature_sweep",
                name="Temperature Parameter Sweep",
                description="Find optimal temperature setting for creative tasks",
                base_scenario="agent_configuration",
                default_variables={
                    "temperature": [0.0, 0.3, 0.5, 0.7, 1.0]
                },
                success_criteria=[
                    {"name": "high_completion", "metric": "completion_rate", "operator": "gte", "threshold": 0.85},
                    {"name": "good_quality", "metric": "quality_score", "operator": "gte", "threshold": 0.7},
                ],
                tags=["configuration", "hyperparameter", "llm"]
            ),
            SimulationTemplate(
                template_id="planning_approach_comparison",
                name="Planning Approach Comparison",
                description="Compare different planning strategies",
                base_scenario="planning_approach",
                default_variables={
                    "planning_method": ["single", "tree_of_thought"],
                    "n_candidates": [1, 3, 5]
                },
                success_criteria=[
                    {"name": "high_quality", "metric": "plan_quality", "operator": "gte", "threshold": 0.8},
                    {"name": "efficient", "metric": "planning_time", "operator": "lte", "threshold": 5.0},
                ],
                tags=["planning", "strategy", "optimization"]
            ),
            SimulationTemplate(
                template_id="code_quality_focus",
                name="Code Quality Focus Areas",
                description="Compare code generation with different quality focuses",
                base_scenario="code_quality",
                default_variables={
                    "quality_focus": ["readability", "performance", "maintainability", "balanced"],
                    "include_tests": [True, False]
                },
                success_criteria=[
                    {"name": "high_quality", "metric": "code_quality", "operator": "gte", "threshold": 0.8},
                    {"name": "compiles", "metric": "compile_success", "operator": "gte", "threshold": 0.9},
                ],
                tags=["code_generation", "quality", "testing"]
            ),
            SimulationTemplate(
                template_id="refactoring_safety_check",
                name="Refactoring Safety Validation",
                description="Validate refactoring approaches for safety",
                base_scenario="refactoring_safety",
                default_variables={
                    "safety_checks": [True, False],
                    "refactoring_type": ["extract_method", "rename", "inline", "move"]
                },
                success_criteria=[
                    {"name": "behavior_preserved", "metric": "behavior_preservation", "operator": "gte", "threshold": 0.95},
                    {"name": "tests_pass", "metric": "test_pass_rate", "operator": "gte", "threshold": 0.9},
                ],
                tags=["refactoring", "safety", "validation"]
            ),
        ]
        
        for template in defaults:
            self._templates[template.template_id] = template
        
        self._save()


# Global registry instance
_template_registry: Optional[TemplateRegistry] = None


def get_template_registry() -> TemplateRegistry:
    """Get the global template registry instance."""
    global _template_registry
    if _template_registry is None:
        _template_registry = TemplateRegistry()
    return _template_registry
