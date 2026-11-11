"""Train a zoo of clean and backdoored SmallCNNs on CIFAR-10.

Each model gets a random id. Whether it is backdoored, and its trigger, are written only to
zoo/<split>/_key.json, which nothing given to the LLM ever reads.

Hyperparameters (epochs, lr, weight decay, seed) are sampled the same way for clean and
backdoored models so that training length or optimiser settings cannot give the answer away.

Usage:
    python -m wsp_vision.train_zoo --split meta_train --n 120
    python -m wsp_vision.train_zoo --split eval --n 40
"""
from __future__ import annotations

import argparse
import secrets
import time

import numpy as np
import torch
import torch.nn.functional as F

from .common import (DEVICE, TRIGGER_CORNERS, TRIGGER_PATTERNS, ZOO_DIR, SmallCNN, Trigger,
                     accuracy, attack_success_rate, augment, load_cifar, read_json,
                     save_model, to_input, write_json)

# The first 1,000 CIFAR-10 test images are the defender's small clean set (used by baselines and
# the agent). The other 9,000 are held out for every reported number.
DEFENDER_N = 1000


def sample_config(rng: np.random.Generator, backdoored: bool) -> dict:
    cfg = dict(
        epochs=int(rng.choice([20, 25, 30])),
        lr=float(rng.choice([3e-3, 5e-3, 8e-3])),
        weight_decay=float(rng.choice([0.0, 1e-4, 5e-4])),
        seed=int(rng.integers(0, 2**31 - 1)),
        backdoored=backdoored,
    )
    if backdoored:
        cfg["poison_rate"] = float(rng.choice([0.05, 0.08, 0.10]))
        cfg["trigger"] = Trigger(
            pattern=str(rng.choice(TRIGGER_PATTERNS)),
            corner=str(rng.choice(TRIGGER_CORNERS)),
            size=int(rng.choice([3, 4, 5])),
            target=int(rng.integers(0, 10)),
            seed=int(rng.integers(0, 2**31 - 1)),
        ).to_dict()
    return cfg


def train_one(cfg: dict, data, bs: int = 256) -> tuple[SmallCNN, dict]:
    x_tr, y_tr, x_te, y_te = data
    torch.manual_seed(cfg["seed"])
    model = SmallCNN().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    steps = cfg["epochs"] * (len(x_tr) // bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg["lr"], total_steps=steps)
    trig = Trigger(**cfg["trigger"]) if cfg["backdoored"] else None

    for _ in range(cfg["epochs"]):
        model.train()
        perm = torch.randperm(len(x_tr), device=DEVICE)
        for k in range(0, len(x_tr) - bs + 1, bs):
            idx = perm[k:k + bs]
            xb, yb = augment(x_tr[idx]), y_tr[idx].clone()
            if trig is not None:
                m = torch.rand(bs, device=DEVICE) < cfg["poison_rate"]
                if m.any():
                    xb[m] = trig.apply(xb[m])
                    yb[m] = trig.target
            loss = F.cross_entropy(model(to_input(xb)), yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()

    x_ev, y_ev = x_te[DEFENDER_N:], y_te[DEFENDER_N:]
    stats = {"clean_acc": accuracy(model, x_ev, y_ev)}
    if trig is not None:
        stats["asr"] = attack_success_rate(model, x_ev, y_ev, trig)
    return model, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["meta_train", "eval", "smoke"])
    ap.add_argument("--n", type=int, required=True, help="total models; half are backdoored")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    split_seed = args.seed if args.seed is not None else {"meta_train": 1, "eval": 2, "smoke": 3}[args.split]
    rng = np.random.default_rng(split_seed)
    out_dir = ZOO_DIR / args.split
    key_path = out_dir / "_key.json"
    key = read_json(key_path) if key_path.exists() else {}

    data = load_cifar()
    labels = rng.permutation([True] * (args.n // 2) + [False] * (args.n - args.n // 2))
    for i, bd in enumerate(labels):
        cfg = sample_config(rng, bool(bd))
        if any(v.get("index") == i for v in key.values()):  # resume
            continue
        t0 = time.time()
        model, stats = train_one(cfg, data)
        mid = "m_" + secrets.token_hex(4)
        save_model(model, out_dir / f"{mid}.pt")
        key[mid] = {"index": i, **cfg, **stats, "train_seconds": round(time.time() - t0, 1)}
        write_json(key, key_path)
        print(f"[{args.split} {i + 1}/{args.n}] {mid} backdoored={bool(bd)} "
              f"acc={stats['clean_acc']:.3f} asr={stats.get('asr', float('nan')):.3f} "
              f"({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
