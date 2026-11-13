"""Join the LLM's answers with the ground truth, in the format the figures and tables read.

Writes results/agent_detect.json and results/agent_remove.json, keyed by model id; removal has one
column per condition (agent = full, agent_weights_only, agent_probe_only). Only here do the
agent's outputs meet zoo/eval/_key.json. A model the agent returned no answer for counts against it:
no detection answer gets p=0.5 and a wrong target; no submitted weights counts as the original model.

Usage:
    python -m wsp_vision.agent.evaluate
"""
from __future__ import annotations

import json
import os

from ..common import RESULTS_DIR, ZOO_DIR, read_json, write_json
from . import llm
from .detect import MODES
from .remove import CONDITIONS, OUT_DIR as REMOVE_DIR

REMOVE_COLUMNS = {c: "agent" if c == "full" else f"agent_{c}" for c in CONDITIONS}

# Image modes need a model with image input, so each mode may name its own LLM:
# WSP_DETECT_LLMS="text=<model>,image=<model>,text_image=<model>" (default: WSP_LLM for every mode).
DETECT_LLMS = {mode: llm.MODEL for mode in MODES}
DETECT_LLMS.update(kv.split("=", 1) for kv in os.environ.get("WSP_DETECT_LLMS", "").split(",") if kv)


def main():
    key = read_json(ZOO_DIR / "eval" / "_key.json")
    det = {mid: {} for mid in key}
    meta = {"llm": DETECT_LLMS, "missing": {}, "unanswered": {}}
    for mode in MODES:
        missing, unanswered = 0, 0
        for mid in key:
            p = RESULTS_DIR / "llm_detect" / DETECT_LLMS[mode] / mode / f"{mid}.json"
            if not p.exists():
                missing += 1
                continue
            a = json.loads(p.read_text())["answer"]
            unanswered += a is None
            det[mid][f"agent_{mode}"] = ({"p_backdoor": 0.5, "target": -1} if a is None else
                                         {"p_backdoor": a["p_backdoored"], "target": a["target_class"],
                                          "pattern": a["trigger_pattern"], "corner": a["trigger_corner"]})
        meta["missing"][mode], meta["unanswered"][mode] = missing, unanswered
    write_json({"meta": meta, "models": det}, RESULTS_DIR / "agent_detect.json")
    print("detect", meta)

    rem, stops = {}, {}
    for cond, col in REMOVE_COLUMNS.items():
        for mid, info in key.items():
            p = REMOVE_DIR / cond / mid / "result.json"
            if info["backdoored"] and p.exists():
                r = json.loads(p.read_text())
                rem.setdefault(mid, {})[col] = {**r.get("edited", r["original"]), "submitted": r["submitted"],
                                                "stop": r["stop"], "turns": r["turns"]}
                stops.setdefault(cond, {}).setdefault(r["stop"], 0)
                stops[cond][r["stop"]] += 1
    write_json({"meta": {"llm": llm.MODEL, "stops": stops}, "models": rem}, RESULTS_DIR / "agent_remove.json")
    print("remove", stops)


if __name__ == "__main__":
    main()
