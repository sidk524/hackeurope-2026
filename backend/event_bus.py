"""
In-memory pub/sub event bus for SSE.

Events carry only metadata (type, IDs). The frontend uses them to
invalidate React Query caches and refetch via the existing REST API.
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

_log = logging.getLogger("event_bus")

# ---------------------------------------------------------------------------
# Event types
# ---------------------------------------------------------------------------

class EventType(str, Enum):
    session_created = "session.created"
    session_updated = "session.updated"
    session_status_changed = "session.status_changed"
    step_registered = "step.registered"
    log_created = "log.created"
    model_registered = "model.registered"
    diagnostic_completed = "diagnostic.completed"
    agent_insight = "agent.insight"


# ---------------------------------------------------------------------------
# SSE event
# ---------------------------------------------------------------------------

@dataclass
class SSEEvent:
    event_type: EventType
    project_id: int | None = None
    session_id: int | None = None
    resource_id: int | None = None
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_sse_string(self) -> str:
        payload = {
            "project_id": self.project_id,
            "session_id": self.session_id,
            "resource_id": self.resource_id,
            "timestamp": self.timestamp,
            **self.data,
        }
        return f"event: {self.event_type.value}\ndata: {json.dumps(payload)}\n\n"


# ---------------------------------------------------------------------------
# Bus
# ---------------------------------------------------------------------------

class EventBus:
    def __init__(self) -> None:
        self._subscribers: dict[int, asyncio.Queue[SSEEvent]] = {}
        self._next_id = 0

    def subscribe(self, maxsize: int = 256) -> tuple[int, asyncio.Queue[SSEEvent]]:
        sub_id = self._next_id
        self._next_id += 1
        q: asyncio.Queue[SSEEvent] = asyncio.Queue(maxsize=maxsize)
        self._subscribers[sub_id] = q
        _log.debug("subscriber %d connected (%d total)", sub_id, len(self._subscribers))
        return sub_id, q

    def unsubscribe(self, sub_id: int) -> None:
        self._subscribers.pop(sub_id, None)
        _log.debug("subscriber %d disconnected (%d total)", sub_id, len(self._subscribers))

    async def publish(self, event: SSEEvent) -> None:
        for sub_id, q in list(self._subscribers.items()):
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                # Slow consumer — evict oldest item, then enqueue
                try:
                    q.get_nowait()
                except asyncio.QueueEmpty:
                    pass
                try:
                    q.put_nowait(event)
                except asyncio.QueueFull:
                    pass


# Module-level singleton
event_bus = EventBus()

# ---------------------------------------------------------------------------
# Sync-to-async bridge (for background tasks running in threads)
# ---------------------------------------------------------------------------

_loop: asyncio.AbstractEventLoop | None = None


def set_event_loop(loop: asyncio.AbstractEventLoop) -> None:
    """Call once at startup to capture the main event loop."""
    global _loop
    _loop = loop


def publish_from_sync(event: SSEEvent) -> None:
    """Publish an event from a synchronous (threaded) context."""
    if _loop is None or _loop.is_closed():
        _log.warning("Event loop not available; dropping event %s", event.event_type.value)
        return
    asyncio.run_coroutine_threadsafe(event_bus.publish(event), _loop)
