"""Knowledge Base for agent learning and insight sharing."""

from core.knowledge.base import (
    KnowledgeBase,
    KnowledgeEntry,
    KnowledgeQuery,
    KnowledgeResult,
    KnowledgeCategory,
)
from core.knowledge.sharing import (
    KnowledgeSharingProtocol,
    KNOWLEDGE_CHANNELS,
)
from core.knowledge.consolidation import (
    KnowledgeConsolidator,
    ConsolidationResult,
)

__all__ = [
    "KnowledgeBase",
    "KnowledgeEntry",
    "KnowledgeQuery",
    "KnowledgeResult",
    "KnowledgeCategory",
    "KnowledgeSharingProtocol",
    "KNOWLEDGE_CHANNELS",
    "KnowledgeConsolidator",
    "ConsolidationResult",
]
