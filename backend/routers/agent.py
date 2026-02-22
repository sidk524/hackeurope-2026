"""
Agent router — Adaptive ML training assistant with tool-calling loop.

POST /agent/chat     — multi-turn conversation with tool use, streams SSE
POST /agent/analyze  — single-shot proactive analysis (called after new epochs)
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
import traceback
import uuid
from typing import Any

import anthropic
import openai
from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from agent_system_prompt import (
    PROACTIVE_ANALYSIS_PROMPT,
    build_system_prompt,
)
from agent_tools import TOOL_SCHEMAS, execute_tool
from event_bus import EventType, SSEEvent, publish_from_sync

load_dotenv()

_log = logging.getLogger("agent")

# ── LLM clients ──────────────────────────────────────────────────────────────

CRUSOE_BASE_URL = "https://hackeurope.crusoecloud.com/v1/"
CRUSOE_MODEL = "NVFP4/Qwen3-235B-A22B-Instruct-2507-FP4"
ANTHROPIC_MODEL = "claude-sonnet-4-20250514"

MAX_TOOL_ITERATIONS = 8


def _get_crusoe_client() -> openai.OpenAI:
    api_key = os.getenv("CRUSOE_API_KEY")
    if not api_key:
        raise RuntimeError("CRUSOE_API_KEY not set")
    return openai.OpenAI(base_url=CRUSOE_BASE_URL, api_key=api_key)


def _get_anthropic_client() -> anthropic.Anthropic:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set")
    return anthropic.Anthropic(api_key=api_key)


# ── Request / response models ────────────────────────────────────────────────

class ChatMessage(BaseModel):
    role: str  # "user" | "assistant" | "system"
    content: str


class AgentChatRequest(BaseModel):
    session_id: int | None = None
    project_id: int | None = None
    messages: list[ChatMessage]
    belief_state: dict[str, Any] | None = None


class ProactiveAnalyzeRequest(BaseModel):
    session_id: int
    project_id: int | None = None
    trigger: str = "new_step"  # new_step | diagnostic_completed | status_changed
    belief_state: dict[str, Any] | None = None


class ProactiveInsight(BaseModel):
    severity: str  # healthy | watch | warning | critical
    title: str
    body: str
    belief_state: dict[str, Any] | None = None
    is_revision: bool = False
    previous_assessment: str | None = None
    model_used: str = ""


# ── Qwen3 native tool-call helpers ────────────────────────────────────────────

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def _parse_qwen_tool_calls(content: str) -> list[dict]:
    """Parse ``<tool_call>`` blocks from Qwen3 native content format."""
    calls: list[dict] = []
    for match in _TOOL_CALL_RE.finditer(content):
        try:
            data = json.loads(match.group(1))
            calls.append({
                "id": f"call_{uuid.uuid4().hex[:8]}",
                "name": data.get("name", ""),
                "arguments": data.get("arguments", {}),
            })
        except json.JSONDecodeError:
            continue
    return calls


def _strip_think_blocks(text: str) -> str:
    """Remove Qwen3 ``<think>…</think>`` reasoning blocks from text."""
    return _THINK_RE.sub("", text).strip()


def _execute_and_collect(
    calls: list[dict],
) -> list[tuple[dict, str]]:
    """Execute a list of parsed tool calls and return (call, result_str) pairs."""
    results: list[tuple[dict, str]] = []
    for call in calls:
        _log.info("Executing tool: %s(%s)", call["name"], call["arguments"])
        result = execute_tool(call["name"], call["arguments"])
        result_str = json.dumps(result, default=str)
        if len(result_str) > 12000:
            result_str = result_str[:12000] + '... [truncated]"}'
        results.append((call, result_str))
    return results


# ── Tool-calling loop (Crusoe / OpenAI-compatible) ───────────────────────────

def _run_tool_loop_openai(
    system_prompt: str,
    messages: list[dict[str, Any]],
    model: str = CRUSOE_MODEL,
) -> tuple[str, str]:
    """
    Run the tool-calling loop using the OpenAI-compatible Crusoe endpoint.

    Handles **both** structured ``tool_calls`` (standard OpenAI) and Qwen3's
    native ``<tool_call>`` content tags.  Tool results are sent back through
    the proper message format for each style.

    Returns ``(assistant_text, model_used)``.
    """
    client = _get_crusoe_client()

    oai_messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
    ]
    for m in messages:
        oai_messages.append({"role": m["role"], "content": m["content"]})

    for iteration in range(MAX_TOOL_ITERATIONS):
        _log.info("Tool loop iteration %d (openai/%s)", iteration, model)
        try:
            response = client.chat.completions.create(
                model=model,
                messages=oai_messages,
                tools=TOOL_SCHEMAS,
                tool_choice="auto",
                temperature=0.3,
                max_tokens=4096,
            )
        except Exception as e:
            _log.error("Crusoe API error (iter %d): %s", iteration, e)
            raise

        choice = response.choices[0]
        msg = choice.message
        content = msg.content or ""

        # ── Path A: structured OpenAI tool_calls ──────────────────────
        if msg.tool_calls:
            _log.info("Structured tool_calls detected (%d)", len(msg.tool_calls))
            # Build a clean assistant dict (avoid extra fields from model_dump)
            tc_dicts = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in msg.tool_calls
            ]
            oai_messages.append({
                "role": "assistant",
                "content": content if content else None,
                "tool_calls": tc_dicts,
            })

            for tc in msg.tool_calls:
                fn_name = tc.function.name
                try:
                    fn_args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    fn_args = {}
                _log.info("Executing tool: %s(%s)", fn_name, fn_args)
                result = execute_tool(fn_name, fn_args)
                result_str = json.dumps(result, default=str)
                if len(result_str) > 12000:
                    result_str = result_str[:12000] + '... [truncated]"}'
                oai_messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result_str,
                })
            continue

        # ── Path B: Qwen3 native <tool_call> tags in content ─────────
        qwen_calls = _parse_qwen_tool_calls(content)
        if qwen_calls:
            _log.info(
                "Qwen3 native <tool_call> detected (%d calls)", len(qwen_calls)
            )
            # Keep the assistant message exactly as the model produced it
            oai_messages.append({"role": "assistant", "content": content})

            # Execute tools and send results using <tool_response> tags
            # which map to Qwen3's chat template expectations
            pairs = _execute_and_collect(qwen_calls)
            for _call, result_str in pairs:
                oai_messages.append({
                    "role": "user",
                    "content": f"<tool_response>\n{result_str}\n</tool_response>",
                })
            continue

        # ── Path C: no tool calls — final answer ─────────────────────
        final = _strip_think_blocks(content)
        _log.info("Final answer received (len=%d)", len(final))
        return final, f"crusoe/{model}"

    # Exhausted iterations — return whatever we have
    return (
        "I've reached the maximum number of tool calls. "
        "Here's what I found so far based on the data I've gathered.",
        f"crusoe/{model}",
    )


# ── Tool-calling loop (Anthropic fallback) ────────────────────────────────────

def _convert_tools_to_anthropic() -> list[dict]:
    """Convert OpenAI tool schemas to Anthropic format."""
    tools = []
    for t in TOOL_SCHEMAS:
        fn = t["function"]
        tools.append({
            "name": fn["name"],
            "description": fn["description"],
            "input_schema": fn["parameters"],
        })
    return tools


def _run_tool_loop_anthropic(
    system_prompt: str,
    messages: list[dict[str, Any]],
    model: str = ANTHROPIC_MODEL,
) -> tuple[str, str]:
    """
    Run the tool-calling loop using Anthropic's native tool use.
    Returns (assistant_text, model_used).
    """
    client = _get_anthropic_client()
    tools = _convert_tools_to_anthropic()

    anth_messages: list[dict[str, Any]] = []
    for m in messages:
        anth_messages.append({"role": m["role"], "content": m["content"]})

    for iteration in range(MAX_TOOL_ITERATIONS):
        _log.info("Tool loop iteration %d (anthropic/%s)", iteration, model)
        try:
            response = client.messages.create(
                model=model,
                system=system_prompt,
                messages=anth_messages,
                tools=tools,
                max_tokens=4096,
                temperature=0.3,
            )
        except Exception as e:
            _log.error("Anthropic API error: %s", e)
            raise

        # Check if the model wants to use tools
        if response.stop_reason == "tool_use":
            # Build the assistant message with all content blocks
            anth_messages.append({
                "role": "assistant",
                "content": [block.model_dump() for block in response.content],
            })

            # Execute each tool call and build tool_result messages
            tool_results: list[dict] = []
            for block in response.content:
                if block.type == "tool_use":
                    _log.info("Executing tool: %s(%s)", block.name, block.input)
                    result = execute_tool(block.name, block.input)
                    result_str = json.dumps(result, default=str)
                    if len(result_str) > 12000:
                        result_str = result_str[:12000] + '... [truncated]"}'

                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result_str,
                    })

            anth_messages.append({"role": "user", "content": tool_results})
        else:
            # Extract text from the final response
            text_parts = [
                block.text
                for block in response.content
                if hasattr(block, "text")
            ]
            return "\n".join(text_parts), f"anthropic/{model}"

    return (
        "I've reached the maximum number of tool calls. "
        "Here's what I found so far.",
        f"anthropic/{model}",
    )


# ── Unified agent runner ──────────────────────────────────────────────────────

def _run_agent(
    system_prompt: str,
    messages: list[dict[str, Any]],
) -> tuple[str, str]:
    """
    Try Crusoe first, fall back to Anthropic if it fails.
    Returns (response_text, model_used).
    """
    # Try Crusoe first
    try:
        return _run_tool_loop_openai(system_prompt, messages)
    except Exception as e:
        _log.warning("Crusoe failed (%s), falling back to Anthropic", e)

    # Fall back to Anthropic
    try:
        return _run_tool_loop_anthropic(system_prompt, messages)
    except Exception as e:
        _log.error("Anthropic also failed: %s", e)
        raise HTTPException(
            status_code=502,
            detail=f"Both LLM providers failed. Last error: {e}",
        )


# ── Belief state extraction ──────────────────────────────────────────────────

def _extract_belief_state(text: str) -> dict[str, Any] | None:
    """Extract the <belief> JSON block from agent output."""
    # Look for ```json blocks after <belief> or the last json fenced block
    patterns = [
        r"<belief>\s*```json\s*(\{.*?\})\s*```",
        r"<belief>\s*(\{.*?\})",
        r"```json\s*(\{[^`]*?\"primary_issue\"[^`]*?\})\s*```",
    ]
    for pat in patterns:
        match = re.search(pat, text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                continue
    return None


# ── Router ────────────────────────────────────────────────────────────────────

router = APIRouter(prefix="/agent", tags=["agent"])


@router.post("/chat")
async def agent_chat(body: AgentChatRequest, request: Request):
    """
    Multi-turn agent chat with tool calling.
    Streams the response as SSE (text/event-stream).
    """
    system_prompt = build_system_prompt(
        session_id=body.session_id,
        project_id=body.project_id,
        belief_state=body.belief_state,
    )

    messages = [{"role": m.role, "content": m.content} for m in body.messages]

    async def _stream():
        try:
            text, model_used = await run_in_threadpool(
                _run_agent, system_prompt, messages
            )

            # Strip any remaining <think> blocks from the response
            text = _strip_think_blocks(text)

            belief = _extract_belief_state(text)

            # Stream the response as SSE events
            # First: the text content
            yield f"event: message\ndata: {json.dumps({'content': text, 'model': model_used})}\n\n"

            # Then: the belief state
            if belief:
                yield f"event: belief\ndata: {json.dumps(belief)}\n\n"

            # Done
            yield f"event: done\ndata: {json.dumps({'model': model_used})}\n\n"

        except HTTPException as e:
            yield f"event: error\ndata: {json.dumps({'detail': e.detail, 'status': e.status_code})}\n\n"
        except Exception as e:
            _log.error("Agent chat error: %s\n%s", e, traceback.format_exc())
            yield f"event: error\ndata: {json.dumps({'detail': str(e)})}\n\n"

    return StreamingResponse(
        _stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/analyze", response_model=ProactiveInsight)
async def agent_analyze(body: ProactiveAnalyzeRequest):
    """
    Single-shot proactive analysis. Called when new data arrives
    (e.g. after a training step completes). Returns a structured insight.
    """
    system_prompt = build_system_prompt(
        session_id=body.session_id,
        project_id=body.project_id,
        belief_state=body.belief_state,
    )

    user_message = PROACTIVE_ANALYSIS_PROMPT.format(session_id=body.session_id)
    messages = [{"role": "user", "content": user_message}]

    try:
        text, model_used = await run_in_threadpool(
            _run_agent, system_prompt, messages
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e)) from e

    belief = _extract_belief_state(text)

    # Determine severity from belief or text analysis
    severity = "watch"
    if belief:
        severity = belief.get("severity", "watch")

    # Check if this is a revision
    is_revision = False
    previous_assessment = None
    if body.belief_state:
        old_issue = body.belief_state.get("primary_issue", "")
        new_issue = belief.get("primary_issue", "") if belief else ""
        if old_issue and new_issue and old_issue != new_issue:
            is_revision = True
            previous_assessment = old_issue

    # Extract title (first sentence or first line)
    lines = text.strip().split("\n")
    title = lines[0][:120] if lines else "Training Update"
    # Strip markdown formatting from title
    title = re.sub(r"[*#`]", "", title).strip()
    if not title:
        title = "Training Update"

    # Strip the belief block from the body
    body_text = re.sub(
        r"<belief>.*?(?:```json.*?```|{.*?})\s*",
        "",
        text,
        flags=re.DOTALL,
    ).strip()

    # Publish SSE event so frontend can show the banner
    publish_from_sync(SSEEvent(
        event_type=EventType.agent_insight,
        session_id=body.session_id,
        project_id=body.project_id,
        data={
            "severity": severity,
            "title": title,
            "body": body_text[:500],
            "is_revision": is_revision,
        },
    ))

    return ProactiveInsight(
        severity=severity,
        title=title,
        body=body_text,
        belief_state=belief,
        is_revision=is_revision,
        previous_assessment=previous_assessment,
        model_used=model_used,
    )
