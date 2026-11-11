"""Gemini backend for llm.stream(): takes the same request shape and returns the same message shape.

detect.py and remove.py are written against the Anthropic message format, so this module translates
at the edge and they stay provider-neutral. Raw parts are kept on each returned block and sent back
unchanged, because Gemini needs its thought signatures returned with its function calls.
"""
from __future__ import annotations

import base64
import os
import re
import time
from types import SimpleNamespace as NS

from google import genai
from google.genai import errors, types

_client = None
MAX_RETRIES = 40


class DailyQuotaExceeded(RuntimeError):
    """The free tier's daily limit is used up. Runs are resumable: start again after the reset."""


def client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=os.environ["GEMINI_API_KEY"],
                               http_options=types.HttpOptions(timeout=600_000))
    return _client


def _to_parts(item) -> list[types.Part]:
    if isinstance(item, NS):  # a block we returned earlier: replay it untouched
        return [item.raw]
    if item["type"] == "text":
        return [types.Part.from_text(text=item["text"])]
    if item["type"] == "image":
        return [types.Part.from_bytes(data=base64.b64decode(item["source"]["data"]),
                                      mime_type=item["source"]["media_type"])]
    if item["type"] == "tool_result":
        body = item["content"]
        if isinstance(body, str):
            body = [{"type": "text", "text": body}]
        text = "\n".join(b["text"] for b in body if b["type"] == "text")
        call_id = None if item["tool_use_id"].startswith("local_") else item["tool_use_id"]
        out = [types.Part(function_response=types.FunctionResponse(
            id=call_id, name="python", response={"output": text}))]
        for b in body:  # figures from show() follow as ordinary image parts
            if b["type"] == "image":
                out += _to_parts(b)
        return out
    raise ValueError(item["type"])


def _to_contents(messages) -> list[types.Content]:
    out = []
    for m in messages:
        items = m["content"]
        if isinstance(items, str):
            items = [{"type": "text", "text": items}]
        parts = [p for it in items for p in _to_parts(it)]
        out.append(types.Content(role="model" if m["role"] == "assistant" else "user", parts=parts))
    return out


def _to_message(resp) -> NS:
    cand = resp.candidates[0] if resp.candidates else None
    blocks = []
    for n, p in enumerate((cand.content.parts or []) if cand and cand.content else []):
        if p.function_call:
            blocks.append(NS(type="tool_use", id=p.function_call.id or f"local_{n}",
                             name=p.function_call.name, input=dict(p.function_call.args or {}), raw=p))
        elif p.thought:
            blocks.append(NS(type="thinking", thinking=p.text or "", raw=p))
        elif p.text is not None:
            blocks.append(NS(type="text", text=p.text, raw=p))
    finish = cand.finish_reason.name if cand and cand.finish_reason else "BLOCKED"
    if any(b.type == "tool_use" for b in blocks):
        stop = "tool_use"
    else:
        stop = {"STOP": "end_turn", "MAX_TOKENS": "max_tokens"}.get(finish, "refusal")
    u = resp.usage_metadata
    cached = (u.cached_content_token_count or 0) if u else 0
    usage = NS(input_tokens=((u.prompt_token_count or 0) - cached) if u else 0,
               output_tokens=((u.candidates_token_count or 0) + (u.thoughts_token_count or 0)) if u else 0,
               cache_read_input_tokens=cached, cache_creation_input_tokens=0)
    return NS(content=blocks, stop_reason=stop, usage=usage, _request_id=resp.response_id,
              stop_details=finish)


def _retry_delay(err: Exception) -> float:
    m = re.search(r"retryDelay'?\"?:\s*'?\"?(\d+(?:\.\d+)?)s", str(err))
    return float(m.group(1)) + 2 if m else 30.0


def stream(model: str, effort: str, system: str, messages: list, max_tokens: int = 64000,
           tools: list | None = None, output_config: dict | None = None, **_ignored) -> NS:
    """`_ignored` drops Anthropic-only options (thinking, cache_control); Gemini caches implicitly."""
    fmt = (output_config or {}).get("format")
    cfg = types.GenerateContentConfig(
        system_instruction=system, max_output_tokens=min(max_tokens, 65536),
        thinking_config=types.ThinkingConfig(include_thoughts=True, thinking_level=effort),
        response_mime_type="application/json" if fmt else None,
        response_json_schema=fmt["schema"] if fmt else None,
        tools=[types.Tool(function_declarations=[types.FunctionDeclaration(
            name=t["name"], description=t["description"],
            parameters_json_schema=t["input_schema"]) for t in tools])] if tools else None,
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True) if tools else None,
    )
    contents = _to_contents(messages)
    for attempt in range(MAX_RETRIES):
        try:
            resp = client().models.generate_content(model=model, contents=contents, config=cfg)
        except errors.ClientError as e:
            if e.code != 429:
                raise
            if re.search(r"PerDay|per.?day", str(e), re.I):
                raise DailyQuotaExceeded(str(e)[:600]) from e
            wait = _retry_delay(e)
            print(f"  rate limited, waiting {wait:.0f}s: {str(e)[:200]}", flush=True)
            time.sleep(wait)
            continue
        except errors.ServerError as e:
            wait = min(10 * 2 ** attempt, 300)
            print(f"  server error {e.code}, waiting {wait}s", flush=True)
            time.sleep(wait)
            continue
        msg = _to_message(resp)
        if msg.stop_details == "MALFORMED_FUNCTION_CALL" and attempt < 3:
            print("  malformed function call, retrying", flush=True)
            continue
        return msg
    raise RuntimeError(f"gave up after {MAX_RETRIES} retries")
