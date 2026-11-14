"""Post-hoc anatomy of every removal edit: where it lands, and whether it hits the real backdoor channel.

For analysis only; nothing here was available to any defender. For each backdoored model we find its
backdoor channel with the real trigger: the conv3 channel whose rise under the trigger adds most to the
target logit (rise in its pooled feature on the 1,000 defender images times its classifier weight into
the target). The channel's share of the total positive push towards the target measures how concentrated
the backdoor is. For each agent episode we then record which tensors it changed and whether the edit
touches that channel (its classifier column, its conv3 filter or its bias).

Writes results/edit_anatomy.json.

Usage:
    python -m wsp_vision.agent.edit_anatomy
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from ..common import RESULTS_DIR, ZOO_DIR, Trigger, load_cifar, read_json, to_input, write_json
from ..train_zoo import DEFENDER_N
from .remove import CONDITIONS, run_dir, split_dir

KEYS = ["conv1.weight", "conv1.bias", "conv2.weight", "conv2.bias", "conv3.weight", "conv3.bias", "fc.weight", "fc.bias"]


@torch.no_grad()
def features(state: dict, x: torch.Tensor) -> torch.Tensor:
    h = to_input(x)
    for name in ["conv1", "conv2", "conv3"]:
        h = F.max_pool2d(F.relu(F.conv2d(h, state[f"{name}.weight"], state[f"{name}.bias"], padding=1)), 2)
    return h.mean(dim=(2, 3))


def backdoor_channel(state: dict, trig: Trigger, x: torch.Tensor, y: torch.Tensor) -> dict:
    keep = y != trig.target
    rise = (features(state, trig.apply(x[keep])) - features(state, x[keep])).mean(0)
    push = rise * state["fc.weight"][trig.target]
    pos = push.clamp(min=0)
    j = int(push.argmax())
    fc = state["fc.weight"]
    z = float(((fc - fc.mean()) / fc.std())[trig.target].max())  # how much the target's largest weight stands out
    return {"channel": j, "share": float(pos[j] / pos.sum()), "target_weight_z": z,
            "top3_share": float(pos.topk(3).values.sum() / pos.sum())}


def anatomy(orig: dict, edited: dict, ch: int) -> dict:
    changed = {k: int((orig[k] != edited[k]).sum()) for k in KEYS}
    hit = bool((orig["fc.weight"][:, ch] != edited["fc.weight"][:, ch]).any()
               or (orig["conv3.weight"][ch] != edited["conv3.weight"][ch]).any()
               or orig["conv3.bias"][ch] != edited["conv3.bias"][ch])
    return {"changed": changed, "total": sum(changed.values()), "hits_channel": hit}


def main():
    _, _, x_te, y_te = load_cifar("cpu")
    x, y = x_te[:DEFENDER_N], y_te[:DEFENDER_N]
    out = {}
    for split, runs in [("eval", 3), ("confirm", 3)]:
        key = read_json(ZOO_DIR / split / "_key.json")
        conds = list(CONDITIONS) if split == "eval" else ["full"]
        for mid, info in sorted(key.items()):
            if not info["backdoored"]:
                continue
            orig = {k: v.float() for k, v in torch.load(ZOO_DIR / split / f"{mid}.pt", weights_only=True).items()}
            trig = Trigger(**info["trigger"])
            bd = backdoor_channel(orig, trig, x, y)
            rec = {"split": split, "trigger": info["trigger"], "poison_rate": info["poison_rate"], **bd, "episodes": []}
            for cond in conds:
                for k in range(1, runs + 1):
                    d = split_dir(split) / run_dir(cond, k) / mid
                    res = read_json(d / "result.json")
                    ep = {"condition": cond, "run": k, "asr": res.get("edited", res["original"])["asr"],
                          "clean_acc": res.get("edited", res["original"])["clean_acc"], "submitted": res["submitted"]}
                    if (d / "edited.pt").exists():
                        ep.update(anatomy(orig, torch.load(d / "edited.pt", weights_only=True), bd["channel"]))
                    else:
                        ep.update({"changed": {k2: 0 for k2 in KEYS}, "total": 0, "hits_channel": False})
                    rec["episodes"].append(ep)
            out[mid] = rec
            print(split, mid, "channel", bd["channel"], f"share {bd['share']:.2f}",
                  "hits", sum(e["hits_channel"] for e in rec["episodes"]), "/", len(rec["episodes"]), flush=True)
    write_json(out, RESULTS_DIR / "edit_anatomy.json")


if __name__ == "__main__":
    main()
