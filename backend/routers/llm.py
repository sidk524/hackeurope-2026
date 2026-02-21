import os
from typing import Any

import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict
from dotenv import load_dotenv

API_URL = "https://hackeurope.crusoecloud.com/v1/chat/completions"
DEFAULT_MODEL = "NVFP4/Qwen3-235B-A22B-Instruct-2507-FP4"

load_dotenv()


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = DEFAULT_MODEL
    messages: list[ChatMessage]
    temperature: float | None = 1
    top_p: float | None = 0.95
    frequency_penalty: float | None = 0
    presence_penalty: float | None = 0
    model_config = ConfigDict(extra="allow")


router = APIRouter(prefix="/llm", tags=["llm"])


def _get_api_key() -> str:
    api_key = os.getenv("CRUSOE_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="CRUSOE_API_KEY is not set")
    return api_key


@router.post("/chat/completions")
async def create_chat_completion(body: ChatCompletionRequest, request: Request):
    api_key = _get_api_key()
    payload: dict[str, Any] = body.model_dump(exclude_none=True)

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }

    client = httpx.AsyncClient(timeout=None)
    try:
        upstream_request = client.build_request(
            "POST",
            API_URL,
            headers=headers,
            json=payload,
        )
        upstream_response = await client.send(upstream_request, stream=True)
    except httpx.RequestError as exc:
        await client.aclose()
        raise HTTPException(status_code=502, detail=f"Upstream request failed: {exc}") from exc

    if upstream_response.status_code >= 400:
        error_body = await upstream_response.aread()
        await upstream_response.aclose()
        await client.aclose()
        detail = error_body.decode("utf-8", errors="replace")
        raise HTTPException(status_code=upstream_response.status_code, detail=detail)

    async def _stream():
        try:
            async for chunk in upstream_response.aiter_raw():
                if await request.is_disconnected():
                    break
                if chunk:
                    yield chunk
        finally:
            await upstream_response.aclose()
            await client.aclose()

    return StreamingResponse(
        _stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
