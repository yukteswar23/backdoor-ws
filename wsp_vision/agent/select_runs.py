"""Best-of-k repair: can a defender pick the agent's good runs without knowing the real trigger?

Every agent condition was run three times per model, and runs differ a lot. This script scores each
run's edited network with a check the defender can run: stamp candidate triggers from the family the
agent was told about (white square, checkerboard, colour noise; 3-5 px; four corners; colour noise with
several seeds, since the real seed is unknown) on the defender's 1,000 clean images, and measure the
largest fraction of images any candidate sends to one class (the estimated ASR). No held-out data and no
real trigger are used for selection.

Selection rule, fixed before looking at outcomes: among a model's runs whose defender clean accuracy is
at most 5 points below the original's, take the one with the lowest estimated ASR (ties: higher clean
accuracy). If no run qualifies, keep the original network. The chosen run is then scored with the true
held-out ASR computed by remove.py, and compared with an average run, the worst and the best (oracle).

Writes results/best_of_k.json (eval zoo) or results/best_of_k_<split>.json.

Usage:
    python -m wsp_vision.agent.select_runs
    python -m wsp_vision.agent.select_runs --split confirm --max-drop 0.08 --conditions full   # pre-registered
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from ..common import (RESULTS_DIR, TRIGGER_CORNERS, TRIGGER_PATTERNS, ZOO_DIR, SmallCNN, Trigger,
                      load_cifar, read_json, to_input, write_json)
from ..train_zoo import DEFENDER_N
from .remove import CONDITIONS, run_dir, split_dir

N_RUNS = 3
NOISE_SEEDS = [0, 1, 2]
MAX_ACC_DROP = 0.05


def candidates() -> list[Trigger]:
    out = []
    for pattern in TRIGGER_PATTERNS:
        for size in (3, 4, 5):
            for seed in (NOISE_SEEDS if pattern == "color_noise" else [0]):
                out += [Trigger(pattern, corner, size, target=0, seed=seed) for corner in TRIGGER_CORNERS]
    return out


@torch.no_grad()
def predict(state: dict, x: torch.Tensor) -> torch.Tensor:
    m = SmallCNN()
    m.load_state_dict(state)
    return m.eval()(to_input(x)).argmax(1)


@torch.no_grad()
def estimated_asr(state: dict, x: torch.Tensor, y: torch.Tensor, cands: list[Trigger]) -> float:
    """Largest share of defender images (outside the class) that any candidate trigger sends to one class."""
    best = 0.0
    for t in cands:
        pred = predict(state, t.apply(x))
        for c in range(10):
            keep = y != c
            best = max(best, float((pred[keep] == c).float().mean()))
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="eval", choices=["eval", "confirm"])
    ap.add_argument("--max-drop", type=float, default=MAX_ACC_DROP, help="allowed defender clean-accuracy drop")
    ap.add_argument("--conditions", nargs="+", default=list(CONDITIONS), choices=list(CONDITIONS))
    args = ap.parse_args()
    key = read_json(ZOO_DIR / args.split / "_key.json")
    _, _, x_te, y_te = load_cifar("cpu")
    x_def, y_def = x_te[:DEFENDER_N], y_te[:DEFENDER_N]
    cands = candidates()
    out = {}
    for mid, info in sorted(key.items()):
        if not info["backdoored"]:
            continue
        orig = torch.load(ZOO_DIR / args.split / f"{mid}.pt", weights_only=True)
        orig_acc = float((predict(orig, x_def) == y_def).float().mean())
        out[mid] = {"original": {"def_acc": orig_acc, "est_asr": estimated_asr(orig, x_def, y_def, cands)}}
        for cond in args.conditions:
            runs = []
            for k in range(1, N_RUNS + 1):
                d = split_dir(args.split) / run_dir(cond, k) / mid
                res = read_json(d / "result.json")
                true = res.get("edited", res["original"])  # no submission counts as the original network
                state = torch.load(d / "edited.pt", weights_only=True) if (d / "edited.pt").exists() else orig
                runs.append({"run": k, "def_acc": float((predict(state, x_def) == y_def).float().mean()),
                             "est_asr": estimated_asr(state, x_def, y_def, cands),
                             "asr": true["asr"], "clean_acc": true["clean_acc"]})
            ok = [r for r in runs if r["def_acc"] >= orig_acc - args.max_drop]
            pick = min(ok, key=lambda r: (r["est_asr"], -r["def_acc"])) if ok else None
            out[mid][cond] = {"runs": runs, "picked_run": pick["run"] if pick else None,
                              "picked": ({"asr": pick["asr"], "clean_acc": pick["clean_acc"]} if pick else
                                         {"asr": res["original"]["asr"], "clean_acc": res["original"]["clean_acc"]})}
        print(mid, {c: (out[mid][c]["picked_run"], round(100 * out[mid][c]["picked"]["asr"], 1)) for c in args.conditions},
              flush=True)
    write_json({"meta": {"runs": N_RUNS, "candidates": len(cands), "noise_seeds": NOISE_SEEDS,
                         "max_acc_drop": args.max_drop, "defender_images": DEFENDER_N, "split": args.split},
                "models": out},
               RESULTS_DIR / ("best_of_k.json" if args.split == "eval" else f"best_of_k_{args.split}.json"))
    for cond in args.conditions:
        rows = [out[m][cond] for m in out]
        mean = lambda f: 100 * np.mean([f(r) for r in rows])
        print(f"{cond:13s} average run {mean(lambda r: np.mean([x['asr'] for x in r['runs']])):.1f}  "
              f"picked {mean(lambda r: r['picked']['asr']):.1f}  "
              f"oracle {mean(lambda r: min(x['asr'] for x in r['runs'])):.1f}  "
              f"worst {mean(lambda r: max(x['asr'] for x in r['runs'])):.1f}")


if __name__ == "__main__":
    main()
