"""NVIDIA API catalog backend for llm.stream() (OpenAI-compatible endpoint at integrate.api.nvidia.com).

Same contract as gemini.py: takes the Anthropic-shaped request detect.py and remove.py build and
returns an Anthropic-shaped message, so the experiment code stays provider-neutral. Replies are
streamed because long reasoning outputs can outlast a plain request. Every rate-limit header and
429 is logged, since the free tier's limits are not documented in the responses.
"""
from __future__ import annotations

import json
import os
import re
import time
from types import SimpleNamespace as NS

import httpx
import openai

from .llm import ServiceUnavailable

BASE_URL = "https://integrate.api.nvidia.com/v1"
MAX_RETRIES = 40
MAX_OUTPUT = 32768
_client = None
stats = {"requests": 0, "rate_limited": 0, "server_errors": 0}


def client() -> openai.OpenAI:
    global _client
    if _client is None:
        _client = openai.OpenAI(base_url=BASE_URL, api_key=os.environ["NVIDIA_API_KEY"],
                                max_retries=0, timeout=1800)
    return _client


def has_vision(model: str) -> bool:
    return bool(re.search(r"(^|[-/])(vl|vision|omni)([-/]|$)", model))


def _image_part(item) -> dict:
    src = item["source"]
    return {"type": "image_url", "image_url": {"url": f"data:{src['media_type']};base64,{src['data']}"}}


def _user_parts(items, vision: bool) -> list[dict]:
    parts = []
    for it in items:
        if it["type"] == "text":
            parts.append({"type": "text", "text": it["text"]})
        elif it["type"] == "image":
            parts.append(_image_part(it) if vision else
                         {"type": "text", "text": "[image omitted: this model has no image input]"})
    return parts


def _to_messages(system: str, messages: list, vision: bool) -> list[dict]:
    out = [{"role": "system", "content": system}]
    for m in messages:
        items = m["content"]
        if isinstance(items, str):
            items = [{"type": "text", "text": items}]
        if m["role"] == "assistant":  # blocks we returned earlier; reasoning is not replayed
            text = "".join(b.text for b in items if b.type == "text")
            calls = [{"id": b.id, "type": "function",
                      "function": {"name": b.name, "arguments": b.raw_args}} for b in items if b.type == "tool_use"]
            msg = {"role": "assistant", "content": text or None}
            if calls:
                msg["tool_calls"] = calls
            out.append(msg)
            continue
        figures = []
        for it in items:  # tool results become role=tool messages; their figures follow as a user turn
            if it["type"] != "tool_result":
                continue
            body = it["content"]
            if isinstance(body, str):
                body = [{"type": "text", "text": body}]
            out.append({"role": "tool", "tool_call_id": it["tool_use_id"],
                        "content": "\n".join(b["text"] for b in body if b["type"] == "text")})
            figures += [b for b in body if b["type"] == "image"]
        rest = [it for it in items if it["type"] != "tool_result"]
        if figures:
            rest = [{"type": "text", "text": "Figures from show():"}] + figures + rest
        if rest:
            out.append({"role": "user", "content": _user_parts(rest, vision)})
    return out


def _strip_fences(text: str) -> str:
    m = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.S)
    if m:
        return m.group(1)
    m = re.search(r"\{.*\}", text, re.S)
    return m.group(0) if m else text


def _collect(stream) -> NS:
    text, reasoning, calls, finish, usage, rid = [], [], {}, None, None, None
    for chunk in stream:
        rid = rid or chunk.id
        if chunk.usage:
            usage = chunk.usage
        if not chunk.choices:
            continue
        ch = chunk.choices[0]
        d = ch.delta
        if d.content:
            text.append(d.content)
        r = getattr(d, "reasoning_content", None) or getattr(d, "reasoning", None)
        if r:
            reasoning.append(r)
        for tc in d.tool_calls or []:
            c = calls.setdefault(tc.index, {"id": None, "name": "", "args": ""})
            c["id"] = tc.id or c["id"]
            if tc.function and tc.function.name:
                c["name"] = tc.function.name
            if tc.function and tc.function.arguments:
                c["args"] += tc.function.arguments
        if ch.finish_reason:
            finish = ch.finish_reason
    return NS(text="".join(text), reasoning="".join(reasoning), calls=calls, finish=finish,
              usage=usage, id=rid)


def _to_message(r: NS, fmt: bool) -> NS:
    blocks = []
    if r.reasoning:
        blocks.append(NS(type="thinking", thinking=r.reasoning))
    if r.text:
        blocks.append(NS(type="text", text=_strip_fences(r.text) if fmt else r.text))
    for n, c in sorted(r.calls.items()):
        try:
            args = json.loads(c["args"] or "{}")
        except json.JSONDecodeError:
            args = {}  # remove.py reports this to the model as invalid JSON
        blocks.append(NS(type="tool_use", id=c["id"] or f"call_{n}", name=c["name"], input=args,
                         raw_args=c["args"] or "{}"))
    if any(b.type == "tool_use" for b in blocks):
        stop = "tool_use"
    else:
        stop = {"stop": "end_turn", "length": "max_tokens", "tool_calls": "end_turn"}.get(r.finish, "refusal")
    u = r.usage
    usage = NS(input_tokens=getattr(u, "prompt_tokens", 0) or 0,
               output_tokens=getattr(u, "completion_tokens", 0) or 0,
               cache_read_input_tokens=0, cache_creation_input_tokens=0)
    return NS(content=blocks, stop_reason=stop, usage=usage, _request_id=r.id, stop_details=r.finish)


def _log_headers(headers):
    rl = {k: v for k, v in headers.items() if re.search(r"rate|limit|retry|quota|credit", k, re.I)}
    if rl:
        print(f"  limit headers: {rl}", flush=True)


def stream(model: str, effort: str, system: str, messages: list, max_tokens: int = 64000,
           tools: list | None = None, output_config: dict | None = None, **_ignored) -> NS:
    """`effort` is unused: these models reason by default and expose no effort setting."""
    fmt = (output_config or {}).get("format")
    kwargs = dict(model=model, messages=_to_messages(system, messages, has_vision(model)),
                  max_tokens=min(max_tokens, MAX_OUTPUT), stream=True,
                  stream_options={"include_usage": True})
    if fmt:
        kwargs["response_format"] = {"type": "json_schema",
                                     "json_schema": {"name": "answer", "schema": fmt["schema"]}}
    if tools:
        kwargs["tools"] = [{"type": "function", "function": {
            "name": t["name"], "description": t["description"], "parameters": t["input_schema"]}}
            for t in tools]
    for attempt in range(MAX_RETRIES):
        stats["requests"] += 1
        wait = min(15 * 2 ** attempt, 600)
        try:
            raw = client().chat.completions.with_raw_response.create(**kwargs)
            _log_headers(raw.headers)
            r = _collect(raw.parse())
            if r.finish is None and not r.text and not r.calls:  # stream ended without a reply
                print(f"  empty reply, waiting {wait}s", flush=True)
                time.sleep(wait)
                continue
            return _to_message(r, bool(fmt))
        except openai.RateLimitError as e:
            stats["rate_limited"] += 1
            _log_headers(e.response.headers)
            wait = float(e.response.headers.get("retry-after", 0) or 0) or wait
            print(f"  rate limited (429), waiting {wait:.0f}s: {str(e)[:200]}", flush=True)
        except openai.APIStatusError as e:
            if e.status_code not in (408, 409) and e.status_code < 500:
                raise  # a request the API rejects (e.g. too long) fails the same way every time
            stats["server_errors"] += 1
            print(f"  HTTP {e.status_code}, waiting {wait}s: {str(e)[:150]}", flush=True)
        except (openai.APIError, httpx.HTTPError) as e:  # connection drops, timeouts, broken streams
            stats["server_errors"] += 1
            print(f"  {type(e).__name__}, waiting {wait}s: {str(e)[:150]}", flush=True)
        time.sleep(wait)
    raise ServiceUnavailable(f"gave up after {MAX_RETRIES} attempts")
