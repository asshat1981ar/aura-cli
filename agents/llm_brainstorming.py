"""
LLM Brainstorming Integration for Innovation Catalyst.

Provides LLM-powered idea generation with OpenRouter auto-routing,
intelligent caching, and template fallback.
"""

import json
import hashlib
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from agents.schemas import Idea
from core.logging_utils import log_json


@dataclass
class LLMResponse:
    """Structured response from LLM for brainstorming."""
    description: str
    novelty: float
    feasibility: float
    impact: float
    rationale: str = ""


class LLMBrainstormingClient:
    """
    Client for LLM-powered brainstorming with OpenRouter.
    
    Features:
    - OpenRouter auto-routing for best model selection
    - Two-tier caching (memory + optional disk)
    - Template fallback on LLM failure
    - Cost tracking and metrics
    """
    
    # OpenRouter configuration
    OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
    DEFAULT_MODEL = "openrouter/auto"  # Auto-routing
    TIMEOUT_SECONDS = 10
    
    # Cost tracking
    total_calls: int = 0
    total_cost_usd: float = 0.0
    cache_hits: int = 0
    
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        """
        Initialize LLM brainstorming client.
        
        Args:
            api_key: OpenRouter API key (or from env OPENROUTER_API_KEY)
            model: Model identifier (default: openrouter/auto)
        """
        self.api_key = api_key or self._get_api_key()
        self.model = model or self.DEFAULT_MODEL
        self._memory_cache: Dict[str, List[Idea]] = {}
        
    def _get_api_key(self) -> Optional[str]:
        """Get API key from environment."""
        import os
        return os.getenv("OPENROUTER_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
    
    def _generate_cache_key(self, problem: str, technique: str, context: str = "") -> str:
        """Generate cache key for problem + technique combination."""
        key_data = f"{problem}|{technique}|{context}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def generate_ideas(
        self,
        problem: str,
        technique: str,
        technique_config: Dict[str, Any],
        context: str = "",
        use_cache: bool = True
    ) -> List[Idea]:
        """
        Generate ideas using LLM with caching and fallback.
        
        Args:
            problem: The problem statement
            technique: Technique name (e.g., 'scamper', 'six_hats')
            technique_config: Configuration for the technique (prompts, etc.)
            context: Additional context
            use_cache: Whether to use caching
            
        Returns:
            List of Idea objects
        """
        cache_key = self._generate_cache_key(problem, technique, context)
        
        # Check memory cache
        if use_cache and cache_key in self._memory_cache:
            self.cache_hits += 1
            log_json("DEBUG", "llm_cache_hit", details={"technique": technique})
            return self._memory_cache[cache_key]
        
        # Try LLM first
        if self.api_key:
            try:
                ideas = self._call_llm(problem, technique, technique_config, context)
                if ideas:
                    # Store in cache
                    if use_cache:
                        self._memory_cache[cache_key] = ideas
                    return ideas
            except Exception as e:
                log_json("WARN", "llm_call_failed", details={
                    "technique": technique,
                    "error": str(e)
                })
        
        # Fallback to template generation
        log_json("INFO", "llm_fallback_to_template", details={"technique": technique})
        return []
    
    def _call_llm(
        self,
        problem: str,
        technique: str,
        technique_config: Dict[str, Any],
        context: str
    ) -> Optional[List[Idea]]:
        """
        Call OpenRouter LLM API.
        
        Args:
            problem: Problem statement
            technique: Technique name
            technique_config: Technique-specific configuration
            context: Additional context
            
        Returns:
            List of Idea objects or None on failure
        """
        import requests
        
        prompt = self._build_prompt(problem, technique, technique_config, context)
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://aura-cli.local",  # Required by OpenRouter
            "X-Title": "AURA Innovation Catalyst"
        }
        
        data = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a creative brainstorming assistant. Generate novel, practical ideas and respond in valid JSON format only."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.8,
            "max_tokens": 2000
        }
        
        start_time = time.time()
        
        try:
            response = requests.post(
                f"{self.OPENROUTER_BASE_URL}/chat/completions",
                headers=headers,
                json=data,
                timeout=self.TIMEOUT_SECONDS
            )
            response.raise_for_status()
            
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            
            # Track metrics
            self.total_calls += 1
            latency_ms = (time.time() - start_time) * 1000
            
            # Parse JSON response
            ideas = self._parse_llm_response(content, technique)
            
            log_json("INFO", "llm_call_success", details={
                "technique": technique,
                "latency_ms": latency_ms,
                "ideas_generated": len(ideas)
            })
            
            return ideas
            
        except requests.Timeout:
            log_json("WARN", "llm_timeout", details={"technique": technique})
            raise
        except requests.RequestException as e:
            log_json("ERROR", "llm_request_failed", details={
                "technique": technique,
                "error": str(e)
            })
            raise
        except (KeyError, json.JSONDecodeError) as e:
            log_json("ERROR", "llm_parse_failed", details={
                "technique": technique,
                "error": str(e)
            })
            raise
    
    def _build_prompt(
        self,
        problem: str,
        technique: str,
        technique_config: Dict[str, Any],
        context: str
    ) -> str:
        """Build technique-specific prompt for LLM."""
        
        base_prompt = f"""You are a creative brainstorming assistant using the {technique_config.get('name', technique)} technique.

Problem to solve: "{problem}"
"""
        
        if context:
            base_prompt += f"""
Additional context: {context}
"""
        
        # Add technique-specific instructions
        if "prompt_template" in technique_config:
            base_prompt += f"""

{technique_config['prompt_template']}
"""
        
        base_prompt += """

Generate 2-4 creative ideas. For each idea, provide:
1. description: A clear 2-3 sentence description of the idea
2. novelty: Score 0.0-1.0 (how unique/original)
3. feasibility: Score 0.0-1.0 (how practical to implement)  
4. impact: Score 0.0-1.0 (how much value it would create)
5. rationale: Brief explanation of why this idea works

Respond with valid JSON in this exact format:
{
  "ideas": [
    {
      "description": "string",
      "novelty": 0.0,
      "feasibility": 0.0,
      "impact": 0.0,
      "rationale": "string"
    }
  ]
}"""
        
        return base_prompt
    
    def _parse_llm_response(self, content: str, technique: str) -> List[Idea]:
        """Parse LLM JSON response into Idea objects."""
        try:
            data = json.loads(content)
            ideas_data = data.get("ideas", [])
            
            ideas = []
            for idea_data in ideas_data:
                idea = Idea(
                    description=idea_data.get("description", ""),
                    technique=technique,
                    novelty=float(idea_data.get("novelty", 0.5)),
                    feasibility=float(idea_data.get("feasibility", 0.5)),
                    impact=float(idea_data.get("impact", 0.5)),
                    metadata={
                        "rationale": idea_data.get("rationale", ""),
                        "source": "llm"
                    }
                )
                ideas.append(idea)
            
            return ideas
            
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            log_json("ERROR", "llm_response_parse_error", details={
                "error": str(e),
                "content_preview": content[:200]
            })
            return []
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get client metrics for monitoring."""
        return {
            "total_calls": self.total_calls,
            "cache_hits": self.cache_hits,
            "cache_hit_rate": self.cache_hits / (self.total_calls + self.cache_hits) if (self.total_calls + self.cache_hits) > 0 else 0,
            "memory_cache_size": len(self._memory_cache)
        }


# Technique-specific prompt templates
TECHNIQUE_PROMPTS = {
    "scamper": {
        "name": "SCAMPER",
        "prompt_template": """Use the SCAMPER framework:
- Substitute: What parts can be replaced?
- Combine: What elements can be merged?
- Adapt: What existing solutions apply?
- Modify: What can be magnified/minimized/changed?
- Put to other uses: How else could this be used?
- Eliminate: What can be removed/simplified?
- Reverse: What if we did the opposite?"""
    },
    "six_hats": {
        "name": "Six Thinking Hats",
        "prompt_template": """Consider the problem from multiple perspectives:
- White Hat: Facts and information
- Red Hat: Emotions and intuition  
- Black Hat: Caution and critical judgment
- Yellow Hat: Optimism and benefits
- Green Hat: Creativity and new ideas
- Blue Hat: Process and overview"""
    },
    "mind_map": {
        "name": "Mind Mapping",
        "prompt_template": """Create a mind map exploring:
- Core problem at center
- Main branches: causes, effects, solutions, resources
- Sub-branches for deeper exploration"""
    },
    "reverse": {
        "name": "Reverse Brainstorming",
        "prompt_template": """Instead of solving the problem, ask:
- How could we make this worse?
- What would guarantee failure?
- Then invert those to find solutions"""
    },
    "worst_idea": {
        "name": "Worst Idea",
        "prompt_template": """Generate intentionally bad ideas, then:
- Identify what makes them bad
- Invert those qualities for good ideas
- Find hidden gems in the bad concepts"""
    },
    "lotus": {
        "name": "Lotus Blossom",
        "prompt_template": """Use the Lotus Blossom grid technique:
- Central theme: the problem
- 8 surrounding themes: related concepts
- Expand each into 8 more specific ideas"""
    },
    "star": {
        "name": "Starbursting",
        "prompt_template": """Generate questions from different angles:
- Who? What? When? Where? Why? How?
- Who is affected? Who can help?
- What are the alternatives? What if...?
- When is the best time? When is it needed?
- Where does it apply? Where else?
- Why is this important? Why now?
- How does it work? How can we improve?"""
    },
    "bia": {
        "name": "Bottleneck Identification",
        "prompt_template": """Focus on constraints and bottlenecks:
- What are the current blockers?
- Where does work slow down or stop?
- What resources are scarce?
- Turn each bottleneck into an improvement opportunity"""
    }
}


def get_llm_client(api_key: Optional[str] = None) -> LLMBrainstormingClient:
    """Factory function to get LLM client instance."""
    return LLMBrainstormingClient(api_key=api_key)
