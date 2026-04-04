"""
Brainstorming Technique Bots for the Innovation Catalyst framework.

Each bot implements a specific creative thinking methodology to generate
ideas from different perspectives. These are used by InnovationSwarm
to create diverse idea pools during the divergence phase.

Now with LLM support via OpenRouter auto-routing.
"""

import random
from typing import List, Dict, Any
from abc import ABC, abstractmethod

from agents.schemas import Idea
from core.logging_utils import log_json


class BaseBrainstormingBot(ABC):
    """Abstract base class for all brainstorming technique bots."""
    
    def __init__(self, llm_client=None, use_llm: bool = True):
        self.capabilities = ["brainstorming", "idea_generation", "creativity"]
        self.llm_client = llm_client
        self.use_llm = use_llm
    
    @property
    @abstractmethod
    def technique_name(self) -> str:
        """Return the name of this brainstorming technique."""
        pass
    
    @property
    @abstractmethod
    def technique_key(self) -> str:
        """Return the technique key for LLM prompts."""
        pass
    
    def generate(self, task: str, context: str = "") -> List[Idea]:
        """
        Generate ideas using this technique.
        
        Tries LLM first if enabled and available, falls back to template generation.
        
        Args:
            task: The problem statement or challenge
            context: Additional context (optional)
            
        Returns:
            List of Idea objects
        """
        # Try LLM first if enabled
        if self.use_llm and self.llm_client:
            try:
                from agents.llm_brainstorming import TECHNIQUE_PROMPTS
                
                technique_config = TECHNIQUE_PROMPTS.get(
                    self.technique_key, 
                    {"name": self.technique_name}
                )
                
                ideas = self.llm_client.generate_ideas(
                    problem=task,
                    technique=self.technique_key,
                    technique_config=technique_config,
                    context=context,
                    use_cache=True
                )
                
                if ideas:
                    log_json("INFO", "llm_ideas_generated", details={
                        "technique": self.technique_key,
                        "count": len(ideas)
                    })
                    return ideas
                    
            except Exception as e:
                log_json("WARN", "llm_generation_failed", details={
                    "technique": self.technique_key,
                    "error": str(e)
                })
        
        # Fall back to template generation
        return self._generate_template(task, context)
    
    @abstractmethod
    def _generate_template(self, task: str, context: str = "") -> List[Idea]:
        """Generate ideas using template-based approach (fallback)."""
        pass
    
    def _create_idea(self, description: str, novelty: float = 0.5, 
                     feasibility: float = 0.5, impact: float = 0.5,
                     metadata: Dict[str, Any] = None) -> Idea:
        """Helper to create an Idea with defaults."""
        return Idea(
            description=description,
            technique=self.technique_name,
            novelty=novelty,
            feasibility=feasibility,
            impact=impact,
            metadata=metadata or {}
        )


# ============================================================================
# SCAMPER BOT
# ============================================================================

class SCAMPERBot(BaseBrainstormingBot):
    """
    SCAMPER technique: Substitute, Combine, Adapt, Modify, Put to other uses,
    Eliminate, Reverse/Rearrange.
    
    A structured approach to creative thinking that prompts ideation through
    specific transformative actions.
    """
    
    PROMPTS = {
        "Substitute": "What parts of the problem can be replaced with alternatives?",
        "Combine": "What elements can be merged or blended together?",
        "Adapt": "What existing solutions can be adapted to this context?",
        "Modify": "What attributes can be magnified, minimized, or changed?",
        "Put_to_other_uses": "How could this be used in completely different contexts?",
        "Eliminate": "What can be removed or simplified without losing value?",
        "Reverse": "What happens if we do the opposite or rearrange elements?"
    }
    
    @property
    def technique_name(self) -> str:
        return "SCAMPER"
    
    @property
    def technique_key(self) -> str:
        return "scamper"
    
    def _generate_template(self, task: str, context: str = "") -> List[Idea]:
        """Generate ideas using the SCAMPER framework (template fallback)."""
        ideas = []
        
        for action, prompt in self.PROMPTS.items():
            # Generate 2-3 variations per SCAMPER action
            for i in range(random.randint(2, 3)):
                novelty = random.uniform(0.5, 0.95)
                feasibility = random.uniform(0.4, 0.9)
                
                idea = self._create_idea(
                    description=f"[{action}] {prompt} Applied to: {task}",
                    novelty=novelty,
                    feasibility=feasibility,
                    impact=random.uniform(0.5, 0.85),
                    metadata={
                        "scamper_action": action,
                        "prompt": prompt,
                        "variation": i + 1,
                        "source": "template"
                    }
                )
                ideas.append(idea)
        
        return ideas


# ============================================================================
# SIX THINKING HATS BOT
# ============================================================================

class SixThinkingHatsBot(BaseBrainstormingBot):
    """
    Edward de Bono's Six Thinking Hats technique.
    
    Each "hat" represents a different mode of thinking:
    - White: Facts and information
    - Red: Emotions and intuition
    - Black: Caution and judgment (devil's advocate)
    - Yellow: Optimism and benefits
    - Green: Creativity and new ideas
    - Blue: Process control and overview
    """
    
    HATS = {
        "White": {
            "focus": "Facts and information",
            "question": "What do we know? What information is needed?",
            "novelty_range": (0.3, 0.6),
            "feasibility_range": (0.7, 0.95)
        },
        "Red": {
            "focus": "Emotions and intuition",
            "question": "What are my gut feelings? What do I feel about this?",
            "novelty_range": (0.5, 0.8),
            "feasibility_range": (0.4, 0.8)
        },
        "Black": {
            "focus": "Caution and judgment",
            "question": "What could go wrong? What are the risks?",
            "novelty_range": (0.3, 0.7),
            "feasibility_range": (0.5, 0.9)
        },
        "Yellow": {
            "focus": "Optimism and benefits",
            "question": "What are the benefits? What's the value?",
            "novelty_range": (0.4, 0.75),
            "feasibility_range": (0.6, 0.9)
        },
        "Green": {
            "focus": "Creativity and new ideas",
            "question": "What creative alternatives exist? What new possibilities?",
            "novelty_range": (0.7, 0.98),
            "feasibility_range": (0.3, 0.7)
        },
        "Blue": {
            "focus": "Process and overview",
            "question": "What's the big picture? How should we proceed?",
            "novelty_range": (0.4, 0.7),
            "feasibility_range": (0.6, 0.95)
        }
    }
    
    @property
    def technique_name(self) -> str:
        return "Six Thinking Hats"
    
    @property
    def technique_key(self) -> str:
        return "six_hats"
    
    def _generate_template(self, task: str, context: str = "") -> List[Idea]:
        """Generate ideas from each thinking hat perspective."""
        ideas = []
        
        for hat_color, hat_info in self.HATS.items():
            # Generate 2-3 ideas per hat
            for i in range(random.randint(2, 3)):
                novelty_range = hat_info["novelty_range"]
                feasibility_range = hat_info["feasibility_range"]
                
                idea = self._create_idea(
                    description=f"[{hat_color} HAT] {hat_info['focus']}: {hat_info['question']} | Task: {task}",
                    novelty=random.uniform(*novelty_range),
                    feasibility=random.uniform(*feasibility_range),
                    impact=random.uniform(0.4, 0.85),
                    metadata={
                        "hat_color": hat_color,
                        "hat_focus": hat_info["focus"],
                        "hat_question": hat_info["question"],
                        "perspective": hat_color.lower()
                    }
                )
                ideas.append(idea)
        
        return ideas


# ============================================================================
# MIND MAPPING BOT
# ============================================================================

class MindMappingBot(BaseBrainstormingBot):
    """
    Mind Mapping technique - visual brainstorming from central concepts.
    
    Starts with a central concept and radiates outward with branches
    representing related ideas, sub-concepts, and associations.
    """
    
    CENTRAL_CONCEPTS = [
        "problem_core",
        "user_needs", 
        "technology",
        "market_context",
        "constraints",
        "opportunities",
        "resources",
        "timeline"
    ]
    
    BRANCH_TEMPLATES = {
        "problem_core": [
            "Root cause analysis",
            "Symptom vs cause",
            "Problem boundaries",
            "Core vs peripheral"
        ],
        "user_needs": [
            "Primary user personas",
            "Pain points",
            "Desired outcomes",
            "User journey stages"
        ],
        "technology": [
            "Available technologies",
            "Emerging tech options",
            "Technical constraints",
            "Integration points"
        ],
        "market_context": [
            "Competitor solutions",
            "Market gaps",
            "Industry trends",
            "Regulatory factors"
        ],
        "constraints": [
            "Budget limitations",
            "Time constraints",
            "Resource availability",
            "Technical debt"
        ],
        "opportunities": [
            "Adjacent markets",
            "Partnership potential",
            "Scalability options",
            "Future expansion"
        ],
        "resources": [
            "Team expertise",
            "Existing assets",
            "External support",
            "Tooling available"
        ],
        "timeline": [
            "Quick wins",
            "Medium-term goals",
            "Long-term vision",
            "Milestone planning"
        ]
    }
    
    @property
    def technique_name(self) -> str:
        return "Mind Mapping"
    
    @property
    def technique_key(self) -> str:
        return "mind_map"
    
    def _generate_template(self, task: str, context: str = "") -> List[Idea]:
        """Generate ideas using mind mapping structure."""
        ideas = []
        
        # Central concept - the core problem
        ideas.append(self._create_idea(
            description=f"[CENTER] Core Challenge: {task}",
            novelty=0.5,
            feasibility=0.8,
            impact=0.7,
            metadata={
                "branch": "center",
                "level": 0,
                "sub_branches": len(self.CENTRAL_CONCEPTS)
            }
        ))
        
        # Generate branches
        for concept in self.CENTRAL_CONCEPTS:
            branches = self.BRANCH_TEMPLATES.get(concept, ["Exploration"])
            sub_branch_count = random.randint(2, 4)
            
            # Main branch
            ideas.append(self._create_idea(
                description=f"[BRANCH: {concept}] {random.choice(branches)} for: {task}",
                novelty=random.uniform(0.4, 0.75),
                feasibility=random.uniform(0.5, 0.85),
                impact=random.uniform(0.5, 0.8),
                metadata={
                    "branch": concept,
                    "level": 1,
                    "sub_branches": sub_branch_count
                }
            ))
            
            # Sub-branches
            for i in range(sub_branch_count):
                sub_branch_template = branches[i % len(branches)]
                ideas.append(self._create_idea(
                    description=f"[SUB-BRANCH: {concept}.{i+1}] {sub_branch_template} exploration",
                    novelty=random.uniform(0.5, 0.85),
                    feasibility=random.uniform(0.4, 0.8),
                    impact=random.uniform(0.4, 0.75),
                    metadata={
                        "branch": concept,
                        "level": 2,
                        "parent": concept,
                        "sub_branch_id": i + 1
                    }
                ))
        
        return ideas


# ============================================================================
# REVERSE BRAINSTORMING BOT
# ============================================================================

class ReverseBrainstormingBot(BaseBrainstormingBot):
    """
    Reverse Brainstorming - solve by thinking backwards.
    
    Instead of asking "How do I solve this?", ask "How could I cause this
    problem?" Then reverse those causes to find solutions.
    """
    
    @property
    def technique_name(self) -> str:
        return "Reverse Brainstorming"
    
    @property
    def technique_key(self) -> str:
        return "reverse"
    
    def _generate_template(self, task: str, context: str = "") -> List[Idea]:
        """Generate by inverting the problem."""
        ideas = []
        
        # Stage 1: Generate "anti-solutions" (ways to cause the problem)
        anti_solutions = [
            "Make the problem worse by ignoring user needs",
            "Ensure failure by not testing anything",
            "Maximize confusion with poor documentation",
            "Guarantee delays with no planning",
        ]
        
        for i, anti in enumerate(anti_solutions):
            # Stage 2: Reverse the anti-solution
            ideas.append(self._create_idea(
                description=f"[REVERSE] Anti: {anti} → Solution: Do the opposite for: {task}",
                novelty=random.uniform(0.5, 0.85),
                feasibility=random.uniform(0.5, 0.9),
                impact=random.uniform(0.6, 0.9),
                metadata={
                    "stage": "reversed",
                    "anti_solution": anti,
                    "transformation": "inverted",
                    "variation": i + 1,
                        "source": "template"
                }
            ))
        
        return ideas


# ============================================================================
# WORST IDEA BOT
# ============================================================================

class WorstIdeaBot(BaseBrainstormingBot):
    """
    Worst Idea First - deliberately bad ideas lead to good ones.
    
    Generate intentionally terrible ideas, then transform them into
    valuable solutions by flipping or modifying them.
    """
    
    WORST_IDEAS_TEMPLATES = [
        "Charge users for every bug fix",
        "Remove all documentation",
        "Delete the test suite",
        "Make the API completely inconsistent",
        "Add 47 mandatory configuration steps",
        "Require manual database edits for every change",
    ]
    
    @property
    def technique_name(self) -> str:
        return "Worst Idea"
    
    @property
    def technique_key(self) -> str:
        return "worst_idea"
    
    def _generate_template(self, task: str, context: str = "") -> List[Idea]:
        """Generate terrible ideas, then transform them."""
        ideas = []
        
        for i, bad_idea in enumerate(self.WORST_IDEAS_TEMPLATES[:4]):
            # Transform the bad idea
            transformations = [
                "Free automated bug detection and fixes",
                "Comprehensive self-documenting code",
                "Zero-configuration intelligent defaults",
            ]
            
            ideas.append(self._create_idea(
                description=f"[WORST→GOOD] Bad: '{bad_idea}' → Good: '{transformations[i % len(transformations)]}' for: {task}",
                novelty=random.uniform(0.6, 0.95),
                feasibility=random.uniform(0.4, 0.85),
                impact=random.uniform(0.5, 0.85),
                metadata={
                    "stage": "bad_to_good",
                    "bad_idea": bad_idea,
                    "transformation": "inverted",
                    "variation": i + 1,
                        "source": "template"
                }
            ))
        
        return ideas


# ============================================================================
# LOTUS BLOSSOM BOT
# ============================================================================

class LotusBlossomBot(BaseBrainstormingBot):
    """
    Lotus Blossom - systematic idea expansion from central idea.
    
    Start with central concept, generate 8 petals (sub-ideas),
    then expand each petal with 8 more sub-ideas.
    """
    
    PETAL_THEMES = [
        "user_experience",
        "performance",
        "security",
        "scalability",
        "maintainability",
        "integration",
        "cost",
        "innovation"
    ]
    
    @property
    def technique_name(self) -> str:
        return "Lotus Blossom"
    
    @property
    def technique_key(self) -> str:
        return "lotus"
    
    def _generate_template(self, task: str, context: str = "") -> List[Idea]:
        """Generate lotus blossom structure."""
        ideas = []
        
        # Central idea
        ideas.append(self._create_idea(
            description=f"[LOTUS CENTER] Core: {task}",
            novelty=0.5,
            feasibility=0.8,
            impact=0.7,
            metadata={
                "level": "center",
                "petals": 8
            }
        ))
        
        # 8 Petals
        for i, theme in enumerate(self.PETAL_THEMES):
            ideas.append(self._create_idea(
                description=f"[PETAL {i+1}] {theme.replace('_', ' ').title()}: {task}",
                novelty=random.uniform(0.5, 0.8),
                feasibility=random.uniform(0.5, 0.85),
                impact=random.uniform(0.5, 0.8),
                metadata={
                    "level": "petal",
                    "petal_id": i + 1,
                    "theme": theme
                }
            ))
            
            # Sub-petals for each petal
            for j in range(2):
                ideas.append(self._create_idea(
                    description=f"[SUB-PETAL {i+1}.{j+1}] {theme} detail: {task}",
                    novelty=random.uniform(0.55, 0.85),
                    feasibility=random.uniform(0.45, 0.8),
                    impact=random.uniform(0.45, 0.75),
                    metadata={
                        "level": "sub_petal",
                        "parent_petal": i + 1,
                        "theme": theme
                    }
                ))
        
        return ideas


# ============================================================================
# STAR BRAINSTORMING BOT
# ============================================================================

class StarBrainstormingBot(BaseBrainstormingBot):
    """
    Star Brainstorming - radiate ideas from central star.
    
    Similar to mind mapping but with fixed rays emanating from center,
    each ray representing a different aspect or dimension.
    """
    
    RAYS = [
        "who",
        "what", 
        "when",
        "where",
        "why",
        "how"
    ]
    
    @property
    def technique_name(self) -> str:
        return "Star Brainstorming"
    
    @property
    def technique_key(self) -> str:
        return "star"
    
    def _generate_template(self, task: str, context: str = "") -> List[Idea]:
        """Generate radiating ideas from central star."""
        ideas = []
        
        # Center
        ideas.append(self._create_idea(
            description=f"[STAR CENTER] Focus: {task}",
            novelty=0.5,
            feasibility=0.8,
            impact=0.7,
            metadata={
                "level": "center",
                "rays": len(self.RAYS)
            }
        ))
        
        # Rays with variations
        for ray in self.RAYS:
            for variation in range(2):
                ideas.append(self._create_idea(
                    description=f"[RAY: {ray.upper()}] {ray} perspective {variation+1}: {task}",
                    novelty=random.uniform(0.5, 0.85),
                    feasibility=random.uniform(0.45, 0.85),
                    impact=random.uniform(0.5, 0.8),
                    metadata={
                        "level": "ray",
                        "ray": ray,
                        "variation": variation + 1
                    }
                ))
        
        return ideas


# ============================================================================
# BISOCIATIVE ASSOCIATION BOT (BIA)
# ============================================================================

class BIABot(BaseBrainstormingBot):
    """
    Bisociative Association - connect disparate concepts.
    
    Deliberately associate the problem with unrelated domains
    to spark novel connections and insights.
    """
    
    DOMAINS = [
        "nature",
        "music",
        "architecture", 
        "cooking",
        "sports",
        "biology",
        "space",
        "ocean",
        "art",
        "gaming"
    ]
    
    @property
    def technique_name(self) -> str:
        return "Bisociative Association"
    
    @property
    def technique_key(self) -> str:
        return "bia"
    
    def _generate_template(self, task: str, context: str = "") -> List[Idea]:
        """Create bisociative connections between domains."""
        ideas = []
        
        # Select random domains
        selected_domains = random.sample(self.DOMAINS, 5)
        
        for domain in selected_domains:
            # Create cross-domain connection
            connections = {
                "nature": "ecosystem balance and adaptation",
                "music": "rhythm, harmony, and improvisation",
                "architecture": "structure, foundations, and design",
                "cooking": "recipes, ingredients, and preparation",
                "sports": "teamwork, strategy, and performance",
                "biology": "evolution, growth, and systems",
                "space": "exploration, orbits, and vastness",
                "ocean": "depth, currents, and ecosystems",
                "art": "creativity, expression, and perspective",
                "gaming": "rules, levels, and engagement"
            }
            
            ideas.append(self._create_idea(
                description=f"[BIA: {domain.upper()}] Apply {connections.get(domain, 'patterns')} to: {task}",
                novelty=random.uniform(0.7, 0.98),
                feasibility=random.uniform(0.3, 0.7),
                impact=random.uniform(0.5, 0.85),
                metadata={
                    "domain": domain,
                    "connection_type": "cross_domain",
                    "domain_concept": connections.get(domain, "patterns")
                }
            ))
        
        return ideas


# ============================================================================
# BOT REGISTRY
# ============================================================================

BRAINSTORMING_BOTS = {
    "scamper": SCAMPERBot,
    "six_hats": SixThinkingHatsBot,
    "mind_map": MindMappingBot,
    "reverse": ReverseBrainstormingBot,
    "worst_idea": WorstIdeaBot,
    "lotus": LotusBlossomBot,
    "star": StarBrainstormingBot,
    "bia": BIABot,
}


def get_bot(technique: str, llm_client=None, use_llm: bool = True) -> BaseBrainstormingBot:
    """Get a brainstorming bot by technique name.
    
    Args:
        technique: Technique identifier (e.g., 'scamper', 'six_hats', 'mind_map')
        llm_client: Optional LLM client for LLM-powered generation
        use_llm: Whether to use LLM (if available) or template fallback
        
    Returns:
        Instantiated bot for the technique
        
    Raises:
        ValueError: If technique is not recognized
    """
    bot_class = BRAINSTORMING_BOTS.get(technique.lower())
    if not bot_class:
        raise ValueError(f"Unknown technique: {technique}. Available: {list(BRAINSTORMING_BOTS.keys())}")
    return bot_class(llm_client=llm_client, use_llm=use_llm)


def list_techniques() -> List[str]:
    """List all available brainstorming techniques."""
    return list(BRAINSTORMING_BOTS.keys())
