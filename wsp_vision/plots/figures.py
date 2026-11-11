"""Figures for the zoo and the non-LLM baselines, regenerated from saved results only.

    python -m wsp_vision.plots.figures            # every figure whose inputs exist
    python -m wsp_vision.plots.figures zoo detect remove

Inputs: zoo/<split>/_key.json, results/detect_baselines.json, results/remove_baselines.json.
Outputs: figures/<name>.{pdf,png,txt}. Uncertainty is a 95% percentile bootstrap over models.
"""
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

import torch

from ..baselines import fc_outlier_scores
from ..common import CLASSES, FIG_DIR, RESULTS_DIR, TRIGGER_CORNERS, ZOO_DIR, Trigger, load_cifar, read_json
from .style import (BACKDOORED, CLEAN, METHOD_COLORS, METHOD_LABELS, PALETTE, apply_style,
                    bootstrap_ci, save)

DETECT_METHODS = ["weight_heuristic", "meta_classifier", "neural_cleanse"]
REMOVE_METHODS = ["original", "weight_prune_outlier", "finetune", "fine_pruning", "nc_unlearn"]
AGENT_DETECT = ["agent_text", "agent_image", "agent_text_image"]


def load_detect() -> list[dict]:
    """Baseline rows, with the LLM's columns merged in once agent.evaluate has run."""
    rows = read_json(RESULTS_DIR / "detect_baselines.json")["eval"]
    path = RESULTS_DIR / "agent_detect.json"
    extra = read_json(path)["models"] if path.exists() else {}
    return [{**r, **extra.get(r["model_id"], {})} for r in rows]


def load_remove() -> list[dict]:
    """As load_detect, for removal. Until every model has an agent result the agent is left out."""
    rows = read_json(RESULTS_DIR / "remove_baselines.json")
    path = RESULTS_DIR / "agent_remove.json"
    extra = read_json(path)["models"] if path.exists() else {}
    return [{**r, **extra.get(r["model_id"], {})} for r in rows]


def _strip(ax, x, vals, color, rng, width=0.18):
    vals = np.asarray(vals, dtype=float)
    if len(vals) == 0:
        return
    ax.scatter(x + rng.uniform(-width, width, len(vals)), vals, s=10, color=color, alpha=0.6,
               edgecolor="none", zorder=2)
    m, lo, hi = bootstrap_ci(vals)
    ax.errorbar(x, m, yerr=[[m - lo], [hi - m]], fmt="o", color="black", ms=4, capsize=3,
                lw=1, zorder=3)


# ============================================================================ zoo
def fig_zoo():
    splits = [s for s in ["meta_train", "eval"] if (ZOO_DIR / s / "_key.json").exists()]
    if not splits:
        print("zoo: no _key.json yet, skipped"); return
    keys = {s: list(read_json(ZOO_DIR / s / "_key.json").values()) for s in splits}
    rng = np.random.default_rng(0)
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.8, 2.6), gridspec_kw={"width_ratios": [1.6, 1]})

    for i, s in enumerate(splits):
        for j, (bd, col) in enumerate([(False, CLEAN), (True, BACKDOORED)]):
            _strip(a, 2 * i + j, [100 * k["clean_acc"] for k in keys[s] if k["backdoored"] == bd], col, rng)
    a.set_xticks(range(2 * len(splits)))
    a.set_xticklabels([f"{s.replace('_', '-')}\n{lab}" for s in splits for lab in ("clean", "backdoored")])
    a.set_ylabel("Clean test accuracy (%)")
    a.set_title("a  Clean accuracy", loc="left", fontweight="bold")

    for i, s in enumerate(splits):
        _strip(b, i, [100 * k["asr"] for k in keys[s] if k["backdoored"] and "asr" in k], BACKDOORED, rng)
    b.set_xticks(range(len(splits)))
    b.set_xticklabels([s.replace("_", "-") for s in splits])
    b.set_xlim(-0.6, len(splits) - 0.4)
    b.set_ylim(-3, 103)
    b.set_ylabel("Attack success rate (%)")
    b.set_title("b  Backdoor strength", loc="left", fontweight="bold")

    counts = ", ".join(f"{s}: {sum(k['backdoored'] for k in keys[s])} backdoored / "
                       f"{sum(not k['backdoored'] for k in keys[s])} clean" for s in splits)
    save(fig, FIG_DIR / "zoo_summary", f"""
Model zoo. (a) Clean accuracy on the 9,000 held-out CIFAR-10 test images for clean and backdoored
SmallCNNs; (b) attack success rate (fraction of non-target-class held-out images sent to the target
class when the trigger is stamped) for backdoored models. Dots are individual models; black markers
are means with 95% bootstrap CIs. Counts: {counts}.""")


# ============================================================================ detection
def fig_detect():
    path = RESULTS_DIR / "detect_baselines.json"
    if not path.exists():
        print("detect: no results yet, skipped"); return
    rows = load_detect()
    y = np.array([int(r["backdoored"]) for r in rows])
    methods = [m for m in DETECT_METHODS + AGENT_DETECT if all(m in r for r in rows)]
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.8, 3.4), gridspec_kw={"width_ratios": [1.15, 1]})

    a.plot([0, 1], [0, 1], ls=":", color=PALETTE["grey_dark"], lw=0.8)
    rng = np.random.default_rng(0)
    for m in methods:
        p = np.array([r[m]["p_backdoor"] for r in rows])
        fpr, tpr, _ = roc_curve(y, p)
        boots = []
        for _ in range(2000):
            idx = rng.integers(0, len(y), len(y))
            if 0 < y[idx].sum() < len(idx):
                boots.append(roc_auc_score(y[idx], p[idx]))
        lo, hi = np.quantile(boots, [0.025, 0.975])
        a.step(fpr, tpr, where="post", color=METHOD_COLORS[m], lw=1.5,
               label=f"{METHOD_LABELS[m]}  {roc_auc_score(y, p):.2f} [{lo:.2f}, {hi:.2f}]")
    a.set_xlabel("False positive rate"); a.set_ylabel("True positive rate")
    a.set_aspect("equal"); a.set_xlim(-0.02, 1.02); a.set_ylim(-0.02, 1.02)
    a.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), title="AUROC [95% CI]", title_fontsize=7, fontsize=7)
    a.set_title("a  Detection", loc="left", fontweight="bold")

    bd = [r for r in rows if r["backdoored"]]
    for i, m in enumerate(methods):
        hits = [int(r[m]["target"] == r["true_target"]) for r in bd]
        v, lo, hi = bootstrap_ci(hits)
        b.bar(i, 100 * v, color=METHOD_COLORS[m], width=0.65, edgecolor="black", lw=0.6)
        b.errorbar(i, 100 * v, yerr=[[100 * (v - lo)], [100 * (hi - v)]], color="black", capsize=3, lw=1)
    b.axhline(10, ls=":", color=PALETTE["grey_dark"], lw=0.8)
    b.text(-0.45, 11, "chance", ha="left", va="bottom", fontsize=7, color=PALETTE["grey_dark"])
    b.set_xticks(range(len(methods)))
    b.set_xticklabels([METHOD_LABELS[m] for m in methods], rotation=20, ha="right")
    b.set_ylim(0, 100); b.set_ylabel("Target class correct (%)")
    b.set_title("b  Target identification", loc="left", fontweight="bold")

    save(fig, FIG_DIR / "detect_baselines", f"""
Backdoor detection on the eval zoo ({len(rows)} models, {len(bd)} backdoored). (a) ROC curves of each
detector's backdoor probability; AUROC with 95% bootstrap CI over models in the legend.
(b) Fraction of backdoored models whose attack target class each method names; error bars are 95%
bootstrap CIs; dotted line is chance (10%).""")


# ============================================================================ removal
def fig_remove():
    path = RESULTS_DIR / "remove_baselines.json"
    if not path.exists():
        print("remove: no results yet, skipped"); return
    rows = load_remove()
    methods = [m for m in REMOVE_METHODS + ["agent"] if all(m in r for r in rows)]
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(8.0, 2.6), gridspec_kw={"width_ratios": [1, 1, 1.1]})

    for ax, metric, ylab, title in [(a, "clean_acc", "Clean accuracy (%)", "a  Clean accuracy"),
                                    (b, "asr", "Attack success rate (%)", "b  Attack success")]:
        for i, m in enumerate(methods):
            v, lo, hi = bootstrap_ci([r[m][metric] for r in rows])
            ax.bar(i, 100 * v, color=METHOD_COLORS[m], width=0.65, edgecolor="black", lw=0.6)
            ax.errorbar(i, 100 * v, yerr=[[100 * (v - lo)], [100 * (hi - v)]], color="black", capsize=3, lw=1)
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels([METHOD_LABELS[m] for m in methods], rotation=35, ha="right")
        ax.set_ylim(0, 100); ax.set_ylabel(ylab)
        ax.set_title(title, loc="left", fontweight="bold")

    for m in methods:
        c.scatter([100 * r[m]["asr"] for r in rows], [100 * r[m]["clean_acc"] for r in rows], s=12,
                  color=METHOD_COLORS[m], alpha=0.75, edgecolor="none", label=METHOD_LABELS[m])
    c.set_xlabel("Attack success rate (%)"); c.set_ylabel("Clean accuracy (%)")
    c.set_xlim(-3, 103)
    c.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.02, 0.5), handletextpad=0.2)
    c.set_title("c  Per model", loc="left", fontweight="bold")

    save(fig, FIG_DIR / "remove_baselines", f"""
Backdoor removal on the {len(rows)} backdoored eval models. Every defence sees only the defender's 1,000
clean images. (a) Clean accuracy and (b) attack success rate on the 9,000 held-out test images, means with
95% bootstrap CIs over models; "No defence" is the original model. (c) Each dot is one model after one
defence; the ideal corner is top-left (high accuracy, low attack success).""")


# ============================================================================ trigger examples
def fig_triggers():
    """The three patch types at the smallest and largest size, stamped on held-out test images."""
    _, _, x_te, y_te = load_cifar("cpu")
    idx = [int((y_te == c).nonzero()[5]) for c in (0, 3, 7, 8)]  # airplane, cat, horse, ship
    cols = [("clean", None)] + [(f"{p.replace('_', ' ')}\n{s}x{s}", Trigger(p, TRIGGER_CORNERS[k % 4], s, 0, 7))
                                for k, (p, s) in enumerate([(p, s) for p in ("white_square", "checkerboard",
                                                                             "color_noise") for s in (3, 5)])]
    fig, axes = plt.subplots(len(idx), len(cols), figsize=(6.8, 4.2))
    for i, n in enumerate(idx):
        for j, (title, trig) in enumerate(cols):
            x = x_te[n:n + 1]
            if trig is not None:
                x = trig.apply(x)
            ax = axes[i, j]
            ax.imshow(x[0].permute(1, 2, 0).numpy(), interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(True); sp.set_linewidth(0.4)
            if i == 0:
                ax.set_title(title, fontsize=7)
            if j == 0:
                ax.set_ylabel(CLASSES[int(y_te[n])], fontsize=7)
    save(fig, FIG_DIR / "trigger_examples", """
BadNets-style triggers used to poison the backdoored models. Columns: the clean image, then each
patch type at the smallest (3x3) and largest (5x5) size; corners vary (top-left, top-right,
bottom-left, bottom-right; a model uses one fixed corner). Images are CIFAR-10 test images at native
32x32 resolution. A poisoned training image is relabelled to the model's target class.""")


# ============================================================================ detection mechanism
def fig_mechanism():
    path = RESULTS_DIR / "detect_baselines.json"
    if not path.exists():
        print("mechanism: no results yet, skipped"); return
    rows = read_json(path)["eval"]
    key = read_json(ZOO_DIR / "eval" / "_key.json")
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(7.4, 2.6), gridspec_kw={"width_ratios": [0.8, 1.2, 1]})
    rng = np.random.default_rng(0)

    # (a) largest fc weight in SDs, clean vs backdoored
    z = {mid: fc_outlier_scores(torch.load(ZOO_DIR / "eval" / f"{mid}.pt", weights_only=True)) for mid in key}
    for j, (bd, col) in enumerate([(False, CLEAN), (True, BACKDOORED)]):
        _strip(a, j, [z[m].max() for m in key if key[m]["backdoored"] == bd], col, rng)
    a.set_xticks([0, 1]); a.set_xticklabels(["clean", "backdoored"]); a.set_xlim(-0.6, 1.6)
    a.set_ylabel("Largest fc weight (SDs)")
    a.set_title("a  Weight outlier", loc="left", fontweight="bold")

    # (b) per-class outlier score relative to the target class, backdoored models
    bd = [m for m in key if key[m]["backdoored"]]
    for m in bd:
        t = key[m]["trigger"]["target"]
        others = np.delete(z[m], t)
        b.scatter(rng.uniform(-0.2, 0.2, 9), others, s=6, color=PALETTE["neutral"], edgecolor="none")
        b.scatter(1 + rng.uniform(-0.2, 0.2), z[m][t], s=12, color=BACKDOORED, edgecolor="none")
    b.set_xticks([0, 1]); b.set_xticklabels(["other classes", "target class"]); b.set_xlim(-0.6, 1.6)
    b.set_ylabel("Class's largest fc weight (SDs)")
    b.set_title("b  Where the outlier sits", loc="left", fontweight="bold")

    # (c) Neural Cleanse: mask size for the true target vs the other classes, normalised per model
    for r in rows:
        if not r["backdoored"]:
            continue
        l1 = np.array(r["neural_cleanse"]["mask_l1"])
        rel = l1 / np.median(l1)
        t = r["true_target"]
        c.scatter(rng.uniform(-0.2, 0.2, 9), np.delete(rel, t), s=6, color=PALETTE["neutral"], edgecolor="none")
        c.scatter(1 + rng.uniform(-0.2, 0.2), rel[t], s=12, color=BACKDOORED, edgecolor="none")
    c.set_yscale("log")
    c.set_yticks([0.1, 0.2, 0.5, 1, 2]); c.set_yticklabels(["0.1", "0.2", "0.5", "1", "2"])
    c.minorticks_off()
    c.set_xticks([0, 1]); c.set_xticklabels(["other classes", "target class"]); c.set_xlim(-0.6, 1.6)
    c.set_ylabel("Mask L1 / model median")
    c.set_title("c  Neural Cleanse", loc="left", fontweight="bold")

    save(fig, FIG_DIR / "detect_mechanism", f"""
Why the baselines work, eval zoo. (a) Largest classifier (fc) weight in units of the standard
deviation of all fc weights; dots are models, black markers means with 95% bootstrap CIs.
(b) For the {len(bd)} backdoored models, the same score per class: the target class (red) usually
holds the outlier, other classes (grey) do not. (c) Neural Cleanse's reverse-engineered trigger mask
size (L1 norm, divided by the model's median over classes, log scale): the true target needs a much
smaller mask than every other class.""")


# ============================================================================ removal by trigger
def fig_remove_by_trigger():
    path = RESULTS_DIR / "remove_baselines.json"
    if not path.exists():
        print("remove_by_trigger: no results yet, skipped"); return
    rows = load_remove()
    methods = [m for m in REMOVE_METHODS[1:] + ["agent"] if all(m in r for r in rows)]
    groups = [("pattern", ["white_square", "checkerboard", "color_noise"], lambda p: p.replace("_", " ")),
              ("size", [3, 4, 5], lambda s: f"{s}x{s}")]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7), sharey=True)
    rng = np.random.default_rng(0)
    w = 0.8 / len(methods)
    for ax, (field, levels, fmt) in zip(axes, groups):
        for i, lev in enumerate(levels):
            sub = [r for r in rows if r["trigger"][field] == lev]
            for k, m in enumerate(methods):
                x = i + (k - (len(methods) - 1) / 2) * w
                vals = [100 * r[m]["asr"] for r in sub]
                ax.scatter(x + rng.uniform(-w / 4, w / 4, len(vals)), vals, s=8, color=METHOD_COLORS[m],
                           alpha=0.8, edgecolor="none", label=METHOD_LABELS[m] if i == 0 else None)
                if vals:
                    ax.hlines(np.mean(vals), x - w / 2.4, x + w / 2.4, color="black", lw=1.2)
        ax.set_xticks(range(len(levels)))
        ax.set_xticklabels([f"{fmt(l)}\n(n={sum(r['trigger'][field] == l for r in rows)})" for l in levels])
        ax.set_ylim(-3, 103)
        ax.set_title(f"{'ab'[groups.index((field, levels, fmt))]}  By trigger {field}", loc="left", fontweight="bold")
    axes[0].set_ylabel("Attack success after defence (%)")
    axes[1].legend(fontsize=7, loc="center left", bbox_to_anchor=(1.02, 0.5), handletextpad=0.2)
    save(fig, FIG_DIR / "remove_by_trigger", """
Attack success rate after each removal baseline, split by (a) trigger pattern and (b) trigger size;
n is the number of backdoored eval models in each group. Dots are models; black bars are group means.
Groups are small, so these are descriptive, not significance tests.""")


FIGS = {"triggers": fig_triggers, "zoo": fig_zoo, "detect": fig_detect, "mechanism": fig_mechanism,
        "remove": fig_remove, "remove_by_trigger": fig_remove_by_trigger}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("which", nargs="*", help="any of: " + ", ".join(FIGS) + " (default: all)")
    apply_style()
    which = ap.parse_args().which or list(FIGS)
    if bad := set(which) - set(FIGS):
        ap.error(f"unknown figure(s): {sorted(bad)}")
    for w in which:
        FIGS[w]()
