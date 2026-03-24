import pytest
import asyncio
from core.events import EventBus, AuraHooks, aura_plugin

@pytest.fixture(autouse=True)
def reset_event_bus():
    EventBus.clear()
    yield
    EventBus.clear()

@pytest.mark.asyncio
async def test_subscribe_and_publish():
    called = False
    async def handler(data):
        nonlocal called
        called = True
        return data * 2

    EventBus.subscribe(AuraHooks.ON_CYCLE_START, handler)
    results = await EventBus.publish(AuraHooks.ON_CYCLE_START, data=5)
    
    assert called is True
    assert results == [10]

@pytest.mark.asyncio
async def test_aura_plugin_decorator():
    @aura_plugin(AuraHooks.PRE_APPLY_CHANGES)
    async def external_audit(context):
        return context + "_audited"

    results = await EventBus.publish(AuraHooks.PRE_APPLY_CHANGES, context="code")
    assert results == ["code_audited"]

@pytest.mark.asyncio
async def test_unsubscribe():
    async def handler():
        pass
    EventBus.subscribe("test_event", handler)
    assert len(EventBus._subscribers["test_event"]) == 1
    
    EventBus.unsubscribe("test_event", handler)
    assert len(EventBus._subscribers["test_event"]) == 0
