"""Experiment B1 (reading): can Claude tell from the weights alone whether a CNN is backdoored?

The model sees one network per request, in one of three modes:
  text        every weight as integers (weights_to_text)
  image       five pictures of the weights (weights_to_images), no numbers
  text_image  both
No code execution, no forward passes, no data. The answer is a JSON object (structured output).

Ground truth comes from zoo/eval/_key.json and is only joined in afterwards, by evaluate.py.

Usage:
    python -m wsp_vision.agent.detect --mode text [--limit 3]
    python -m wsp_vision.agent.detect --mode all
"""
from __future__ import annotations

import argparse
import json
import time
from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor

import torch

from ..common import RESULTS_DIR, TRIGGER_CORNERS, TRIGGER_PATTERNS, ZOO_DIR
from ..serialize import ARCH_TEXT, weights_to_images, weights_to_text
from . import llm

MODES = ["text", "image", "text_image"]
OUT_DIR = RESULTS_DIR / "llm_detect" / llm.MODEL

SYSTEM = f"""You are auditing small image classifiers for hidden backdoors, using only their weights.

Every network you will see has the same architecture and was trained on CIFAR-10:
{ARCH_TEXT}
About half of the networks in this collection were trained with a backdoor (data poisoning, as in
BadNets): during training, a fraction of images had a small square patch stamped near one corner of
the image and were relabelled to one fixed target class. The backdoored network then predicts the
target class whenever the patch is present. The other half were trained normally on clean data.
Training settings (epochs, learning rate, weight decay, seed) were randomised the same way for both
groups, so they carry no signal.

Possible patch types: white_square (all pixels white), checkerboard (alternating black and white
pixels), color_noise (each pixel and channel randomly 0 or 255). Patches are 3 to 5 pixels wide,
placed 1 pixel in from one of the four corners.

You cannot run code or the network. Reason about the weights you are given. Then give:
- p_backdoored: your probability (0 to 1) that this network is backdoored. Be calibrated.
- target_class: if it were backdoored, which class index is most likely the target (always answer).
- trigger_pattern and trigger_corner: your best guesses (always answer; say in the evidence if you
  think a field cannot be read from the weights).
- evidence: a short explanation of what in the weights drove your answer."""

SCHEMA = {
    "type": "object",
    "properties": {
        "p_backdoored": {"type": "number"},
        "target_class": {"type": "integer", "enum": list(range(10))},
        "trigger_pattern": {"type": "string", "enum": TRIGGER_PATTERNS},
        "trigger_corner": {"type": "string", "enum": TRIGGER_CORNERS},
        "evidence": {"type": "string"},
    },
    "required": ["p_backdoored", "target_class", "trigger_pattern", "trigger_corner", "evidence"],
    "additionalProperties": False,
}


def build_content(state, mode: str) -> list[dict]:
    content = []
    if mode in ("image", "text_image"):
        for caption, png in weights_to_images(state):
            content.append({"type": "text", "text": f"Image: {caption}"})
            content.append(llm.image_block(png))
    if mode in ("text", "text_image"):
        content.append({"type": "text", "text": "All weights of the network:\n" + weights_to_text(state)})
    content.append({"type": "text", "text": "Is this network backdoored? Answer in the required JSON format."})
    return content


def run_one(mid: str, mode: str) -> dict:
    out = OUT_DIR / mode / f"{mid}.json"
    if out.exists():
        return json.loads(out.read_text())
    state = torch.load(ZOO_DIR / "eval" / f"{mid}.pt", weights_only=True)
    t0 = time.time()
    try:
        msg = llm.stream(
            system=SYSTEM,
            messages=[{"role": "user", "content": build_content(state, mode)}],
            output_config={"format": {"type": "json_schema", "schema": SCHEMA}},
        )
    except llm.AuthError:
        raise  # a bad key is not this model's outcome: record nothing
    except llm.ServiceUnavailable as e:  # left unfinished; the next run retries it
        print(f"[{mode}] {mid} unavailable, will retry later: {e}", flush=True)
        return None
    except Exception as e:  # permanent rejection: recorded as no answer
        rec = {"model_id": mid, "mode": mode, "llm": llm.MODEL, "effort": llm.EFFORT,
               "stop_reason": "api_error", "error": f"{type(e).__name__}: {str(e)[:500]}",
               "usage": llm.usage_dict(SimpleNamespace(input_tokens=0, output_tokens=0)),
               "seconds": round(time.time() - t0, 1), "answer": None}
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rec, indent=2))
        print(f"[{mode}] {mid} api_error: {rec['error'][:200]}", flush=True)
        return rec
    thinking, text = llm.thinking_and_text(msg)
    rec = {"model_id": mid, "mode": mode, "llm": llm.MODEL, "effort": llm.EFFORT,
           "stop_reason": msg.stop_reason, "request_id": msg._request_id,
           "usage": llm.usage_dict(msg.usage), "seconds": round(time.time() - t0, 1),
           "thinking_summary": thinking, "answer": None}
    if msg.stop_reason == "refusal":
        rec["stop_details"] = str(msg.stop_details)
    elif text:
        try:
            rec["answer"] = json.loads(text)
        except json.JSONDecodeError:  # a malformed answer counts as no answer
            rec["raw_text"] = text[:5000]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=2))
    a = rec["answer"] or {}
    print(f"[{mode}] {mid} p={a.get('p_backdoored')} target={a.get('target_class')} "
          f"${rec['usage']['usd']:.3f} {rec['seconds']}s", flush=True)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=MODES + ["all"])
    ap.add_argument("--limit", type=int, default=None, help="only the first N models (pilot)")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    ids = sorted(p.stem for p in (ZOO_DIR / "eval").glob("m_*.pt"))[: args.limit]
    modes = MODES if args.mode == "all" else [args.mode]
    jobs = [(m, mode) for mode in modes for m in ids]
    try:
        with ThreadPoolExecutor(args.workers) as ex:
            recs = list(ex.map(lambda j: run_one(*j), jobs))
    except llm.AuthError as e:
        print(f"API key rejected, stopping: {e}", flush=True)
        raise SystemExit(llm.AUTH_EXIT)
    done = [r for r in recs if r is not None]
    print(f"total ${sum(r['usage']['usd'] for r in done):.2f} over {len(done)} calls, "
          f"{len(recs) - len(done)} left for a later run")
    if len(done) < len(recs):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
