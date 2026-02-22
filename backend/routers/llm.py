import os
from typing import Any

import openai
from fastapi import APIRouter, HTTPException, Request
from fastapi.concurrency import iterate_in_threadpool
from fastapi.responses import StreamingResponse
from openai import OpenAI
from pydantic import BaseModel, ConfigDict
from dotenv import load_dotenv
from starlette.concurrency import run_in_threadpool

API_BASE_URL = "https://hackeurope.crusoecloud.com/v1/"
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


def _get_client() -> OpenAI:
    return OpenAI(
        base_url=API_BASE_URL,
        api_key=_get_api_key(),
    )


@router.post("/chat/completions")
async def create_chat_completion(body: ChatCompletionRequest, request: Request):
    payload: dict[str, Any] = body.model_dump(exclude_none=True)
    client = _get_client()
    try:
        upstream_response = await run_in_threadpool(
            lambda: client.chat.completions.with_streaming_response.create(**payload)
        )
    except openai.APIConnectionError as exc:
        raise HTTPException(status_code=502, detail=f"Upstream request failed: {exc}") from exc
    except openai.RateLimitError as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    except openai.APIStatusError as exc:
        detail = getattr(exc.response, "text", None) or str(exc)
        raise HTTPException(status_code=exc.status_code, detail=detail) from exc
    except openai.APIError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    def _sync_stream():
        with upstream_response as stream:
            for chunk in stream.iter_bytes():
                yield chunk

    async def _stream():
        sync_iter = _sync_stream()
        try:
            async for chunk in iterate_in_threadpool(sync_iter):
                if await request.is_disconnected():
                    break
                if chunk:
                    yield chunk
        finally:
            sync_iter.close()

    media_type = "text/event-stream" if payload.get("stream") else "application/json"

    return StreamingResponse(
        _stream(),
        media_type=media_type,
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
