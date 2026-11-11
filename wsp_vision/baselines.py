"""Non-LLM baselines for backdoor detection and removal.

Detection
  weight_heuristic   weights only, no training: how far the largest classifier weight stands out
  meta_classifier    weights only: logistic regression on permutation-invariant weight statistics,
                     trained on the separate meta_train zoo
  neural_cleanse     uses the defender's 1,000 clean images and gradients (Wang et al., 2019)

Removal (all get the same 1,000 clean images, never triggered images)
  weight_prune_outlier  weights only: zero the conv3 channel with the largest classifier weight
  finetune              fine-tune all layers on the clean images
  fine_pruning          prune conv3 channels that are quiet on clean data, then fine-tune (Liu et al., 2018)
  nc_unlearn            stamp Neural Cleanse's reversed trigger on clean images with their true labels, fine-tune

Usage:
    python -m wsp_vision.baselines detect
    python -m wsp_vision.baselines remove
"""
from __future__ import annotations

import argparse
import copy
import time

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import kurtosis
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .common import (DEVICE, RESULTS_DIR, ZOO_DIR, Trigger, accuracy, attack_success_rate,
                     augment, load_cifar, load_model, read_json, to_input, write_json)
from .train_zoo import DEFENDER_N


def zoo(split: str):
    key = read_json(ZOO_DIR / split / "_key.json")
    return [(mid, ZOO_DIR / split / f"{mid}.pt", info) for mid, info in sorted(key.items())]


# ============================================================================ detection
def fc_outlier_scores(state) -> np.ndarray:
    """Per class: largest positive classifier weight, in standard deviations of all fc weights."""
    w = state["fc.weight"].cpu().numpy()
    return w.max(axis=1) / w.std()


def weight_features(state) -> np.ndarray:
    """Permutation-invariant summary statistics of every layer."""
    f = []
    for name in ["conv1", "conv2", "conv3", "fc"]:
        w = state[f"{name}.weight"].cpu().numpy()
        b = state[f"{name}.bias"].cpu().numpy()
        flat = w.ravel()
        f += [flat.mean(), flat.std(), np.abs(flat).max() / flat.std(), kurtosis(flat),
              *np.quantile(flat, [0.01, 0.1, 0.5, 0.9, 0.99]), b.mean(), b.std(), b.max(), b.min()]
        unit_norms = np.sqrt((w.reshape(w.shape[0], -1) ** 2).sum(1))
        z = (unit_norms - unit_norms.mean()) / (unit_norms.std() + 1e-8)
        f += list(np.sort(z)[-3:])  # three largest unit norms, standardised
    s = np.sort(fc_outlier_scores(state))
    f += [s[-1], s[-1] - s[-2], s[-1] - np.median(s)]
    fb = state["fc.bias"].cpu().numpy()
    f += [(fb.max() - fb.mean()) / (fb.std() + 1e-8)]
    return np.array(f, dtype=np.float64)


def neural_cleanse(model, x_clean, y_clean, steps: int = 300, lam: float = 1e-2):
    """Reverse-engineer the smallest mask that sends clean images to each class.
    Returns (anomaly_index, predicted_target, per-class mask L1 norms, masks, patterns)."""
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    norms, masks, patterns = [], [], []
    x = x_clean.float() / 255.0
    mean = torch.tensor([0.4914, 0.4822, 0.4465], device=DEVICE).view(1, 3, 1, 1)
    std = torch.tensor([0.2470, 0.2435, 0.2616], device=DEVICE).view(1, 3, 1, 1)
    for t in range(10):
        m_logit = torch.full((1, 1, 32, 32), -3.0, device=DEVICE, requires_grad=True)
        p_logit = torch.zeros((1, 3, 32, 32), device=DEVICE, requires_grad=True)
        opt = torch.optim.Adam([m_logit, p_logit], lr=0.1)
        tgt = torch.full((len(x),), t, device=DEVICE)
        for _ in range(steps):
            m, p = torch.sigmoid(m_logit), torch.sigmoid(p_logit)
            xa = (1 - m) * x + m * p
            loss = F.cross_entropy(model((xa - mean) / std), tgt) + lam * m.sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
        m = torch.sigmoid(m_logit).detach()
        norms.append(m.sum().item())
        masks.append(m.cpu())
        patterns.append(torch.sigmoid(p_logit).detach().cpu())
    for p in model.parameters():
        p.requires_grad_(True)
    n = np.array(norms)
    med = np.median(n)
    mad = 1.4826 * np.median(np.abs(n - med)) + 1e-8
    return (med - n.min()) / mad, int(n.argmin()), n, masks, patterns


def run_detect():
    data = load_cifar()
    _, _, x_te, y_te = data
    x_def, y_def = x_te[:DEFENDER_N], y_te[:DEFENDER_N]

    meta = zoo("meta_train")
    X_tr = np.stack([weight_features(torch.load(p, weights_only=True)) for _, p, _ in meta])
    y_tr = np.array([int(i["backdoored"]) for _, _, i in meta])
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=5000))
    cv_prob = cross_val_predict(clf, X_tr, y_tr, cv=5, method="predict_proba")[:, 1]
    clf.fit(X_tr, y_tr)

    # Map the fc outlier score to a probability using the meta-train zoo (1-feature logistic fit).
    s_tr = np.array([fc_outlier_scores(torch.load(p, weights_only=True)).max() for _, p, _ in meta])
    heur = LogisticRegression().fit(s_tr[:, None], y_tr)

    rows = []
    for mid, path, info in zoo("eval"):
        state = torch.load(path, weights_only=True)
        sc = fc_outlier_scores(state)
        t0 = time.time()
        ai, nc_target, nc_norms, _, _ = neural_cleanse(load_model(path), x_def, y_def)
        rows.append({
            "model_id": mid,
            "backdoored": info["backdoored"],
            "true_target": info["trigger"]["target"] if info["backdoored"] else None,
            "weight_heuristic": {"p_backdoor": float(heur.predict_proba([[sc.max()]])[0, 1]),
                                 "target": int(sc.argmax())},
            "meta_classifier": {"p_backdoor": float(clf.predict_proba(weight_features(state)[None])[0, 1]),
                                "target": int(sc.argmax())},
            "neural_cleanse": {"anomaly_index": float(ai), "target": nc_target,
                               "p_backdoor": float(1 / (1 + np.exp(-(ai - 2.0)))),  # NC's threshold of 2 -> 0.5
                               "mask_l1": nc_norms.tolist(), "seconds": round(time.time() - t0, 1)},
        })
        print(mid, info["backdoored"], f"heur={rows[-1]['weight_heuristic']['p_backdoor']:.2f}",
              f"meta={rows[-1]['meta_classifier']['p_backdoor']:.2f}", f"NC_AI={ai:.2f}", flush=True)
    write_json({"meta_train_cv": {"y": y_tr.tolist(), "p": cv_prob.tolist()}, "eval": rows},
               RESULTS_DIR / "detect_baselines.json")


# ============================================================================ removal
def finetune(model, x, y, epochs: int = 20, lr: float = 1e-3, extra=None):
    """extra(xb, yb) -> (xb, yb) lets nc_unlearn stamp its reversed trigger on part of a batch."""
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        perm = torch.randperm(len(x), device=DEVICE)
        for k in range(0, len(x), 128):
            idx = perm[k:k + 128]
            xb, yb = augment(x[idx]), y[idx]
            if extra is not None:
                xb, yb = extra(xb, yb)
            loss = F.cross_entropy(model(to_input(xb)), yb)
            opt.zero_grad(); loss.backward(); opt.step()
    return model.eval()


@torch.no_grad()
def conv3_activation(model, x):
    model.eval()
    h = F.max_pool2d(F.relu(model.conv1(to_input(x))), 2)
    h = F.max_pool2d(F.relu(model.conv2(h)), 2)
    return F.relu(model.conv3(h)).mean(dim=(0, 2, 3))


@torch.no_grad()
def prune_channels(model, chans):
    model.conv3.weight[chans] = 0
    model.conv3.bias[chans] = 0
    model.fc.weight[:, chans] = 0
    return model


def fine_pruning(model, x, y, max_drop: float = 0.03):
    act = conv3_activation(model, x)
    order = act.argsort().tolist()  # quietest first
    base = accuracy(model, x, y)
    pruned = []
    for c in order:
        trial = prune_channels(copy.deepcopy(model), pruned + [c])
        if base - accuracy(trial, x, y) > max_drop:
            break
        pruned.append(c)
    model = prune_channels(model, pruned)
    mask = torch.ones(64, device=DEVICE); mask[pruned] = 0
    model = finetune(model, x, y)
    with torch.no_grad():  # keep pruned channels at zero after fine-tuning
        model.conv3.weight.mul_(mask.view(-1, 1, 1, 1)); model.conv3.bias.mul_(mask)
        model.fc.weight.mul_(mask.view(1, -1))
    return model, len(pruned)


def nc_unlearn(model, x, y):
    _, target, _, masks, patterns = neural_cleanse(model, x, y)
    m, p = masks[target].to(DEVICE), patterns[target].to(DEVICE)

    def extra(xb, yb):
        k = len(xb) // 5
        xf = xb.float() / 255.0
        xf[:k] = (1 - m) * xf[:k] + m * p
        return (xf * 255).round().clamp(0, 255).to(torch.uint8), yb
    return finetune(model, x, y, extra=extra), target


def weight_prune_outlier(model):
    w = model.fc.weight.detach()
    c, j = divmod(int(torch.argmax(w / w.std()).item()), 64)
    return prune_channels(model, [j]), {"target": c, "channel": j}


def run_remove():
    _, _, x_te, y_te = load_cifar()
    x_def, y_def = x_te[:DEFENDER_N], y_te[:DEFENDER_N]
    x_ev, y_ev = x_te[DEFENDER_N:], y_te[DEFENDER_N:]
    rows = []
    for mid, path, info in zoo("eval"):
        if not info["backdoored"]:
            continue
        trig = Trigger(**info["trigger"])
        row = {"model_id": mid, "trigger": info["trigger"]}

        def score(m):
            return {"clean_acc": accuracy(m, x_ev, y_ev), "asr": attack_success_rate(m, x_ev, y_ev, trig)}
        row["original"] = score(load_model(path))
        m, meta = weight_prune_outlier(load_model(path)); row["weight_prune_outlier"] = {**score(m), **meta}
        torch.manual_seed(0)
        row["finetune"] = score(finetune(load_model(path), x_def, y_def))
        torch.manual_seed(0)
        m, k = fine_pruning(load_model(path), x_def, y_def); row["fine_pruning"] = {**score(m), "pruned": k}
        torch.manual_seed(0)
        m, t = nc_unlearn(load_model(path), x_def, y_def); row["nc_unlearn"] = {**score(m), "nc_target": t}
        rows.append(row)
        print(mid, {k: (round(v["clean_acc"], 3), round(v["asr"], 3)) for k, v in row.items()
                    if isinstance(v, dict) and "asr" in v}, flush=True)
    write_json(rows, RESULTS_DIR / "remove_baselines.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=["detect", "remove"])
    a = ap.parse_args()
    run_detect() if a.task == "detect" else run_remove()
