"""Ablation control: the agent's edit moved to random positions. No LLM, no API.

For each backdoored eval model, take the edit the agent made in the `full` condition and apply the
same edit at randomly chosen positions of the same tensors: the same number of weights per tensor, each
multiplied by the factor the agent applied to one of its own positions (new / old; where the agent's
original weight was 0, the random position is set to the agent's new value). If the agent's edits only
work because they are small and leave the network intact, this control will do as well; if they work
because they hit the backdoor, it will not. Scored like the agent, on the 9,000 held-out test images.

Writes results/random_edit.json: per model, the mean over draws plus every draw.

Usage:
    python -m wsp_vision.agent.random_edit [--draws 10]
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from ..common import (RESULTS_DIR, ZOO_DIR, SmallCNN, Trigger, accuracy, attack_success_rate,
                      load_cifar, read_json, write_json)
from ..train_zoo import DEFENDER_N
from .remove import OUT_DIR


def random_edit(orig: dict, edited: dict, rng: np.random.Generator) -> dict:
    out = {k: v.clone() for k, v in orig.items()}
    for k in orig:
        o, e = orig[k].flatten(), edited[k].flatten()
        changed = torch.nonzero(o != e).flatten()
        if len(changed) == 0:
            continue
        pos = torch.from_numpy(rng.choice(len(o), size=len(changed), replace=False))
        new = out[k].view(-1)
        for p, src in zip(pos, changed):
            new[p] = o[p] * (e[src] / o[src]) if o[src] != 0 else e[src]
    return out


def score(state: dict, x, y, trig: Trigger) -> dict:
    m = SmallCNN()
    m.load_state_dict(state)
    m = m.eval()
    return {"clean_acc": accuracy(m, x, y), "asr": attack_success_rate(m, x, y, trig)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=10)
    args = ap.parse_args()
    key = read_json(ZOO_DIR / "eval" / "_key.json")
    _, _, x_te, y_te = load_cifar("cpu")
    x_ev, y_ev = x_te[DEFENDER_N:], y_te[DEFENDER_N:]
    models = {}
    for mid, info in sorted(key.items()):
        path = OUT_DIR / "full" / mid / "edited.pt"
        if not (info["backdoored"] and path.exists()):
            continue
        orig = torch.load(ZOO_DIR / "eval" / f"{mid}.pt", weights_only=True)
        edited = torch.load(path, weights_only=True)
        trig = Trigger(**info["trigger"])
        rng = np.random.default_rng(int(mid[2:], 16))  # fixed seed per model
        draws = [score(random_edit(orig, edited, rng), x_ev, y_ev, trig) for _ in range(args.draws)]
        models[mid] = {"random_edit": {"asr": float(np.mean([d["asr"] for d in draws])),
                                       "clean_acc": float(np.mean([d["clean_acc"] for d in draws])),
                                       "draws": draws}}
        print(f"{mid}: ASR {100 * models[mid]['random_edit']['asr']:.1f}%  "
              f"clean {100 * models[mid]['random_edit']['clean_acc']:.1f}%", flush=True)
    write_json({"meta": {"draws": args.draws, "source": str(OUT_DIR.relative_to(RESULTS_DIR) / "full")},
                "models": models}, RESULTS_DIR / "random_edit.json")
    print(f"{len(models)} models, mean ASR {100 * np.mean([m['random_edit']['asr'] for m in models.values()]):.1f}%")


if __name__ == "__main__":
    main()
