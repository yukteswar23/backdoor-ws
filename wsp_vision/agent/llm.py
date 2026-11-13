"""One place for the model id, effort and cost accounting. Claude ids go to the Anthropic SDK;
NVIDIA API catalog ids ("<org>/<model>") go to nvidia.py, which returns the same message shape.

No refusal fallbacks are configured on purpose: a fallback would silently swap the model in the
middle of a comparison. Refusals are recorded as such and reported.
"""
from __future__ import annotations

import base64
import os

import anthropic

MODEL = os.environ.get("WSP_LLM", "claude-opus-5-5")  # e.g. nvidia/nemotron-3-ultra-550b-a55b
IS_NVIDIA = "/" in MODEL  # NVIDIA API catalog ids are "<org>/<model>"
EFFORT = "default" if IS_NVIDIA else "high"
PRICE_PER_MTOK = ({"input": 0.0, "output": 0.0, "cache_write": 0.0, "cache_read": 0.0}
                  if IS_NVIDIA  # free tier
                  else {"input": 4.00, "output": 20.00, "cache_write": 5.00, "cache_read": 0.20})

_client = None


class ServiceUnavailable(RuntimeError):
    """The API stayed unavailable through all retries. Callers leave the model unfinished, so a
    later run retries it; any other API error is permanent and is recorded as that model's outcome."""


class AuthError(RuntimeError):
    """The API rejected the key (401/403). Nothing can succeed, so the whole run stops and records
    nothing; scripts exit with AUTH_EXIT so the retry loop does not try again."""


AUTH_EXIT = 3


def client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic(max_retries=5)
    return _client


def image_block(png: bytes) -> dict:
    return {"type": "image",
            "source": {"type": "base64", "media_type": "image/png",
                       "data": base64.standard_b64encode(png).decode("utf-8")}}


def usage_dict(u) -> dict:
    d = {"input": u.input_tokens, "output": u.output_tokens,
         "cache_write": getattr(u, "cache_creation_input_tokens", 0) or 0,
         "cache_read": getattr(u, "cache_read_input_tokens", 0) or 0}
    d["usd"] = sum(d[k] * PRICE_PER_MTOK[k] for k in PRICE_PER_MTOK) / 1e6
    return d


def stream(**kwargs):
    """Streams (long inputs, large max_tokens) and returns the final Message."""
    if IS_NVIDIA:
        from . import nvidia
        return nvidia.stream(MODEL, EFFORT, **kwargs)
    kwargs.setdefault("model", MODEL)
    kwargs.setdefault("max_tokens", 64000)
    kwargs.setdefault("output_config", {})
    kwargs["output_config"].setdefault("effort", EFFORT)
    kwargs.setdefault("thinking", {"type": "adaptive", "display": "summarized"})
    try:
        with client().messages.stream(**kwargs) as s:
            return s.get_final_message()
    except (anthropic.AuthenticationError, anthropic.PermissionDeniedError) as e:
        raise AuthError(str(e)[:300]) from e


def thinking_and_text(msg) -> tuple[str, str]:
    th = "\n".join(b.thinking for b in msg.content if b.type == "thinking")
    tx = "\n".join(b.text for b in msg.content if b.type == "text")
    return th, tx
