"""
SSE streaming endpoint — /events/stream

Clients connect and receive lightweight push notifications when backend
state changes. Events carry only metadata; the frontend invalidates
React Query caches and refetches via the existing REST API.
"""

import asyncio

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from event_bus import event_bus, SSEEvent

router = APIRouter(prefix="/events", tags=["events"])

KEEPALIVE_INTERVAL = 15  # seconds


async def _event_generator(
    request: Request,
    project_id: int | None,
    session_id: int | None,
):
    sub_id, queue = event_bus.subscribe()
    try:
        # Initial connection comment
        yield ": connected\n\n"

        while True:
            if await request.is_disconnected():
                break

            try:
                event: SSEEvent = await asyncio.wait_for(
                    queue.get(), timeout=KEEPALIVE_INTERVAL
                )
            except asyncio.TimeoutError:
                # Send keepalive comment to prevent proxy/browser timeouts
                yield ": keepalive\n\n"
                continue

            # Server-side filtering
            if project_id is not None and event.project_id != project_id:
                continue
            if session_id is not None and event.session_id != session_id:
                continue

            yield event.to_sse_string()
    finally:
        event_bus.unsubscribe(sub_id)


@router.get("/stream")
async def stream_events(
    request: Request,
    project_id: int | None = None,
    session_id: int | None = None,
):
    return StreamingResponse(
        _event_generator(request, project_id, session_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
