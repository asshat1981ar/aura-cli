"""Knowledge sharing protocol for cross-agent communication."""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set

from core.knowledge.base import KnowledgeBase, KnowledgeCategory, KnowledgeEntry, KnowledgeQuery
from core.logging_utils import log_json


# Predefined knowledge channels
KNOWLEDGE_CHANNELS = {
    "lessons_learned": "Hard-won lessons from failures and mistakes",
    "patterns_discovered": "Reusable patterns discovered during work",
    "best_practices": "Validated best practices and approaches",
    "anti_patterns": "Known anti-patterns to avoid",
    "tool_insights": "Insights about specific tools and libraries",
    "domain_knowledge": "Domain-specific knowledge and expertise",
    "performance_tips": "Performance optimization insights",
    "security_alerts": "Security vulnerabilities and mitigations",
    "refactoring_patterns": "Successful refactoring strategies",
    "debugging_techniques": "Effective debugging approaches",
}


@dataclass
class Subscription:
    """A knowledge channel subscription."""
    subscriber_id: str
    channel: str
    callback: Callable[[KnowledgeEntry], Any]
    filter_fn: Optional[Callable[[KnowledgeEntry], bool]] = None
    created_at: float = field(default_factory=time.time)


@dataclass
class SharedKnowledgeEvent:
    """Event when knowledge is shared."""
    event_id: str
    entry: KnowledgeEntry
    channel: str
    shared_by: str
    timestamp: float
    delivery_status: Dict[str, str] = field(default_factory=dict)


class KnowledgeSharingProtocol:
    """Protocol for sharing knowledge between agents and components."""
    
    def __init__(self, knowledge_base: Optional[KnowledgeBase] = None):
        """
        Initialize the sharing protocol.
        
        Args:
            knowledge_base: Knowledge base for persistence
        """
        self.kb = knowledge_base or KnowledgeBase()
        self._subscriptions: Dict[str, List[Subscription]] = {}
        self._event_history: List[SharedKnowledgeEvent] = []
        self._max_history = 1000
        self._lock = asyncio.Lock()
        
    async def publish(
        self, 
        entry: KnowledgeEntry, 
        channels: List[str],
        publisher_id: str = "system"
    ) -> SharedKnowledgeEvent:
        """
        Publish knowledge to channels.
        
        Args:
            entry: Knowledge entry to share
            channels: Channels to publish to
            publisher_id: ID of the publishing entity
            
        Returns:
            SharedKnowledgeEvent with delivery status
        """
        async with self._lock:
            # Store in knowledge base
            entry.source = publisher_id
            await self.kb.add(entry)
            
            # Create event
            event = SharedKnowledgeEvent(
                event_id=f"evt_{int(time.time())}_{hash(entry.entry_id) % 10000}",
                entry=entry,
                channel=channels[0] if channels else "general",
                shared_by=publisher_id,
                timestamp=time.time()
            )
            
            # Notify subscribers
            delivery_tasks = []
            for channel in channels:
                for sub in self._subscriptions.get(channel, []):
                    # Apply filter if present
                    if sub.filter_fn and not sub.filter_fn(entry):
                        continue
                    
                    # Schedule delivery
                    task = self._deliver_to_subscriber(sub, entry, event)
                    delivery_tasks.append(task)
            
            # Wait for deliveries
            if delivery_tasks:
                results = await asyncio.gather(*delivery_tasks, return_exceptions=True)
                for sub, result in zip(
                    [s for ch in channels for s in self._subscriptions.get(ch, [])],
                    results
                ):
                    if isinstance(result, Exception):
                        event.delivery_status[sub.subscriber_id] = f"failed: {result}"
                    else:
                        event.delivery_status[sub.subscriber_id] = "delivered"
            
            # Store event
            self._event_history.append(event)
            if len(self._event_history) > self._max_history:
                self._event_history = self._event_history[-self._max_history:]
            
            log_json("INFO", "knowledge_published", {
                "event_id": event.event_id,
                "entry_id": entry.entry_id,
                "channels": channels,
                "subscribers_notified": len(delivery_tasks)
            })
            
            return event
    
    async def subscribe(
        self, 
        subscriber_id: str,
        channel: str, 
        callback: Callable[[KnowledgeEntry], Any],
        filter_fn: Optional[Callable[[KnowledgeEntry], bool]] = None
    ) -> bool:
        """
        Subscribe to a knowledge channel.
        
        Args:
            subscriber_id: Unique ID for the subscriber
            channel: Channel to subscribe to
            callback: Function to call when knowledge is published
            filter_fn: Optional filter function for entries
            
        Returns:
            True if subscription successful
        """
        async with self._lock:
            if channel not in self._subscriptions:
                self._subscriptions[channel] = []
            
            # Check for duplicate subscription
            existing = [s for s in self._subscriptions[channel] 
                       if s.subscriber_id == subscriber_id]
            if existing:
                # Update existing subscription
                existing[0].callback = callback
                existing[0].filter_fn = filter_fn
            else:
                # Add new subscription
                subscription = Subscription(
                    subscriber_id=subscriber_id,
                    channel=channel,
                    callback=callback,
                    filter_fn=filter_fn
                )
                self._subscriptions[channel].append(subscription)
            
            log_json("INFO", "knowledge_subscribed", {
                "subscriber_id": subscriber_id,
                "channel": channel
            })
            
            return True
    
    async def unsubscribe(self, subscriber_id: str, channel: Optional[str] = None) -> bool:
        """
        Unsubscribe from knowledge channels.
        
        Args:
            subscriber_id: Subscriber ID to remove
            channel: Specific channel to unsubscribe from, or None for all
            
        Returns:
            True if any subscriptions were removed
        """
        async with self._lock:
            removed = False
            
            if channel:
                if channel in self._subscriptions:
                    original_len = len(self._subscriptions[channel])
                    self._subscriptions[channel] = [
                        s for s in self._subscriptions[channel]
                        if s.subscriber_id != subscriber_id
                    ]
                    removed = len(self._subscriptions[channel]) < original_len
            else:
                for ch in list(self._subscriptions.keys()):
                    original_len = len(self._subscriptions[ch])
                    self._subscriptions[ch] = [
                        s for s in self._subscriptions[ch]
                        if s.subscriber_id != subscriber_id
                    ]
                    if len(self._subscriptions[ch]) < original_len:
                        removed = True
            
            return removed
    
    async def request(
        self, 
        query: KnowledgeQuery,
        from_sources: Optional[List[str]] = None
    ) -> List[Any]:
        """
        Request knowledge from the base.
        
        Args:
            query: Knowledge query
            from_sources: Optional list of sources to filter by
            
        Returns:
            List of knowledge results
        """
        results = await self.kb.query(query)
        
        if from_sources:
            results = [r for r in results if r.entry.source in from_sources]
        
        return results
    
    async def broadcast_lesson(
        self,
        lesson: str,
        category: KnowledgeCategory,
        tags: List[str],
        source: str,
        channels: Optional[List[str]] = None
    ) -> SharedKnowledgeEvent:
        """
        Broadcast a lesson learned to interested parties.
        
        Args:
            lesson: The lesson text
            category: Knowledge category
            tags: Tags for the lesson
            source: Source of the lesson
            channels: Channels to broadcast to (default: lessons_learned)
            
        Returns:
            SharedKnowledgeEvent
        """
        entry = KnowledgeEntry(
            content=lesson,
            source=source,
            category=category,
            tags=tags,
            confidence=0.85  # Lessons learned have higher confidence
        )
        
        channels = channels or ["lessons_learned"]
        
        return await self.publish(entry, channels, source)
    
    async def share_pattern(
        self,
        pattern_name: str,
        pattern_description: str,
        applicability: List[str],
        source: str,
        confidence: float = 0.8
    ) -> SharedKnowledgeEvent:
        """
        Share a discovered pattern.
        
        Args:
            pattern_name: Name of the pattern
            pattern_description: Description of the pattern
            applicability: List of contexts where pattern applies
            source: Source of the pattern
            confidence: Confidence in the pattern
            
        Returns:
            SharedKnowledgeEvent
        """
        content = f"Pattern: {pattern_name}\n\n{pattern_description}\n\n"
        content += f"Applicable to: {', '.join(applicability)}"
        
        entry = KnowledgeEntry(
            content=content,
            source=source,
            category=KnowledgeCategory.PATTERN,
            tags=["pattern"] + applicability,
            confidence=confidence
        )
        
        return await self.publish(
            entry, 
            ["patterns_discovered", "best_practices"],
            source
        )
    
    def get_channel_subscribers(self, channel: str) -> List[str]:
        """Get list of subscriber IDs for a channel."""
        return [s.subscriber_id for s in self._subscriptions.get(channel, [])]
    
    def get_available_channels(self) -> Dict[str, str]:
        """Get all available channels with descriptions."""
        return KNOWLEDGE_CHANNELS.copy()
    
    def get_event_history(
        self, 
        channel: Optional[str] = None,
        limit: int = 100
    ) -> List[SharedKnowledgeEvent]:
        """Get recent sharing events."""
        events = self._event_history
        
        if channel:
            events = [e for e in events if e.channel == channel]
        
        return events[-limit:]
    
    async def _deliver_to_subscriber(
        self,
        subscription: Subscription,
        entry: KnowledgeEntry,
        event: SharedKnowledgeEvent
    ):
        """Deliver knowledge to a subscriber."""
        try:
            result = subscription.callback(entry)
            
            # Handle async callbacks
            if asyncio.iscoroutine(result):
                await asyncio.wait_for(result, timeout=10.0)
            
            return True
            
        except asyncio.TimeoutError:
            log_json("WARN", "knowledge_delivery_timeout", {
                "subscriber_id": subscription.subscriber_id,
                "event_id": event.event_id
            })
            raise
            
        except Exception as e:
            log_json("WARN", "knowledge_delivery_failed", {
                "subscriber_id": subscription.subscriber_id,
                "event_id": event.event_id,
                "error": str(e)
            })
            raise


class AgentKnowledgeInterface:
    """Convenience interface for agents to share and receive knowledge."""
    
    def __init__(self, agent_id: str, sharing_protocol: KnowledgeSharingProtocol):
        """
        Initialize agent knowledge interface.
        
        Args:
            agent_id: Unique agent identifier
            sharing_protocol: Knowledge sharing protocol instance
        """
        self.agent_id = agent_id
        self.protocol = sharing_protocol
        self._received_knowledge: List[KnowledgeEntry] = []
        self._subscribed_channels: Set[str] = set()
        
    async def share_lesson(
        self,
        lesson: str,
        category: KnowledgeCategory = KnowledgeCategory.LESSON_LEARNED,
        tags: Optional[List[str]] = None
    ) -> SharedKnowledgeEvent:
        """Share a lesson learned."""
        return await self.protocol.broadcast_lesson(
            lesson=lesson,
            category=category,
            tags=tags or [],
            source=self.agent_id
        )
    
    async def share_pattern(
        self,
        pattern_name: str,
        description: str,
        applicability: List[str]
    ) -> SharedKnowledgeEvent:
        """Share a discovered pattern."""
        return await self.protocol.share_pattern(
            pattern_name=pattern_name,
            pattern_description=description,
            applicability=applicability,
            source=self.agent_id
        )
    
    async def query_knowledge(
        self,
        query: str,
        categories: Optional[List[KnowledgeCategory]] = None,
        max_results: int = 10
    ) -> List[Any]:
        """Query the knowledge base."""
        knowledge_query = KnowledgeQuery(
            query_text=query,
            categories=categories,
            max_results=max_results
        )
        
        return await self.protocol.request(knowledge_query)
    
    async def subscribe_to_channel(
        self,
        channel: str,
        callback: Optional[Callable[[KnowledgeEntry], Any]] = None
    ):
        """Subscribe to a knowledge channel."""
        if callback is None:
            # Use default callback that stores received knowledge
            callback = self._default_receive_callback
        
        await self.protocol.subscribe(
            subscriber_id=self.agent_id,
            channel=channel,
            callback=callback
        )
        
        self._subscribed_channels.add(channel)
    
    async def unsubscribe_from_channel(self, channel: str):
        """Unsubscribe from a channel."""
        await self.protocol.unsubscribe(self.agent_id, channel)
        self._subscribed_channels.discard(channel)
    
    def get_received_knowledge(self, clear: bool = False) -> List[KnowledgeEntry]:
        """Get knowledge received from subscriptions."""
        knowledge = self._received_knowledge.copy()
        if clear:
            self._received_knowledge = []
        return knowledge
    
    def _default_receive_callback(self, entry: KnowledgeEntry):
        """Default callback for received knowledge."""
        self._received_knowledge.append(entry)
        log_json("DEBUG", "agent_received_knowledge", {
            "agent_id": self.agent_id,
            "entry_id": entry.entry_id,
            "source": entry.source
        })
