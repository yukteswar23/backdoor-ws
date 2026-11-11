"""Explanatory diagrams for the report: pipeline, poisoning, and how the LLM reads and edits weights.

Diagrams that show weights, images or predictions use the real zoo model m_27c068c4 (white 3x3 patch,
bottom-left, target cat) and the LLM's real answer for it, so nothing in them is invented.
Canvases are laid out in inches (1 data unit = 1 inch) so text widths can be checked by eye:
at 6.5 pt a character is about 0.05 in wide.

    python -m wsp_vision.plots.diagrams [overview poisoning read agent]
"""
from __future__ import annotations

import argparse
import io
import json
import textwrap

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

from ..common import CLASSES, FIG_DIR, RESULTS_DIR, ZOO_DIR, Trigger, load_cifar, load_model, read_json, to_input
from ..serialize import _quant, weights_to_images
from .style import BACKDOORED, CLEAN, PALETTE, apply_style, save

CASE = "m_27c068c4"
ULTRA = "nvidia/nemotron-3-ultra-550b-a55b"

# light fills, one meaning each across all diagrams
FILL = {"data": "#EEEEEE", "llm": "#DCE8F5", "base": "#EFE6EE", "score": "#E3F1E6", "key": "#FFF6DA",
        "rule": "#FBEAEA"}
EDGE = {"data": PALETTE["grey_dark"], "llm": PALETTE["blue_main"], "base": PALETTE["violet"],
        "score": "#3C8A4E", "key": "#B08900", "rule": PALETTE["red_strong"]}
GOOD, BAD = "#3C8A4E", PALETTE["red_strong"]


def _canvas(w, h):
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w); ax.set_ylim(0, h); ax.axis("off")
    return fig, ax


def box(ax, x, y, w, h, kind, title=None, body=None, fs=6.5, dashed=False, align="center", lw=1.0):
    """Rounded box; the title sits at the top, the body right under it."""
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.05",
                                fc=FILL[kind], ec=EDGE[kind], lw=lw, ls="--" if dashed else "-"))
    tx = x + w / 2 if align == "center" else x + 0.08
    top = y + h - 0.07
    if title:
        ax.text(tx, top, title, ha=align, va="top", fontsize=fs + 0.8, fontweight="bold", color=EDGE[kind])
        top -= (fs + 0.8) * 1.45 / 72
    if body:
        ax.text(tx, top, body, ha=align, va="top", fontsize=fs, linespacing=1.3)


def arrow(ax, p, q, text=None, color=None, dashed=False, rad=0.0, fs=6.2, tpos=None, ha="center", lw=1.0):
    color = color or PALETTE["grey_dark"]
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=8, lw=lw, color=color,
                                 ls="--" if dashed else "-", connectionstyle=f"arc3,rad={rad}",
                                 shrinkA=0, shrinkB=0))
    if text:
        x, y = tpos or ((p[0] + q[0]) / 2, (p[1] + q[1]) / 2 + 0.05)
        ax.text(x, y, text, ha=ha, va="bottom", fontsize=fs, color=color, linespacing=1.2)


def step_label(ax, x, y, n, text):
    ax.add_patch(plt.Circle((x, y), 0.085, color=PALETTE["grey_dark"]))
    ax.text(x, y, str(n), ha="center", va="center", fontsize=6.5, color="white", fontweight="bold")
    ax.text(x + 0.14, y, text, ha="left", va="center", fontsize=8, fontweight="bold")


# ============================================================================ 1. pipeline overview
def fig_overview():
    fig, ax = _canvas(7.0, 3.1)
    cols = [0.05, 1.78, 3.55, 5.32]
    w = 1.63
    for n, (x, t) in enumerate(zip(cols, ["Build the zoo", "Two tasks", "Defenders", "Scoring"]), 1):
        step_label(ax, x + 0.09, 2.98, n, t)

    box(ax, cols[0], 1.6, w, 1.23, "data", "120 small CNNs",
        "CIFAR-10, 24,234 weights\n60 clean, 60 poisoned:\na corner patch sends any\nimage to a target class\n"
        "40 evaluated, 80 train\nthe meta-classifier")
    box(ax, cols[0], 0.62, w, 0.75, "key", "Ground-truth key",
        "backdoored? pattern, corner,\nsize, target class\nhidden from all defenders", dashed=True)

    box(ax, cols[1], 1.95, w, 0.88, "data", "Detect (read only)",
        "in: the weights only\nout: backdoored? target?\ntrigger pattern, corner?")
    box(ax, cols[1], 0.62, w, 1.08, "data", "Remove (read + write)",
        "in: weights, 1,000 clean\nimages, 50 forward passes,\nno gradients, no triggers\nout: edited weights")

    box(ax, cols[2], 2.4, w, 0.43, "llm", "LLM, one call", "Nemotron 3 Ultra / 3.5 VL", fs=6.0)
    box(ax, cols[2], 1.95, w, 0.4, "base", "Baselines", "heuristic, meta-clf., NC", fs=6.0)
    box(ax, cols[2], 1.19, w, 0.51, "llm", "LLM agent + sandbox", "Nemotron 3 Ultra", fs=6.0)
    box(ax, cols[2], 0.62, w, 0.52, "base", "Baselines", "prune, fine-tune,\nfine-prune, NC unlearn", fs=6.0)

    box(ax, cols[3], 1.95, w, 0.88, "score", "Detection scores",
        "AUROC on 40 models\ntarget / pattern / corner\nright on 20 poisoned")
    box(ax, cols[3], 0.62, w, 1.08, "score", "Removal scores",
        "real trigger on 9,000\nheld-out images:\nattack success (lower\nis better), clean accuracy")

    arrow(ax, (cols[0] + w, 2.4), (cols[1], 2.4)); arrow(ax, (cols[0] + w, 1.75), (cols[1], 1.2))
    for y0, (ya, yb) in [(2.39, (2.6, 2.15)), (1.16, (1.45, 0.88))]:
        arrow(ax, (cols[1] + w, y0), (cols[2], ya)); arrow(ax, (cols[1] + w, y0), (cols[2], yb))
    arrow(ax, (cols[2] + w, 2.39), (cols[3], 2.39)); arrow(ax, (cols[2] + w, 1.16), (cols[3], 1.16))
    kx, sx, yb = cols[0] + w / 2, cols[3] + w / 2, 0.3
    ax.plot([kx, kx, sx], [0.62, yb, yb], color=EDGE["key"], lw=1.0, ls="--")
    arrow(ax, (sx, yb), (sx, 0.62), color=EDGE["key"], dashed=True)
    ax.text((kx + sx) / 2, yb + 0.04, "used only by the scorer", ha="center", va="bottom",
            fontsize=6.2, color=EDGE["key"])
    save(fig, FIG_DIR / "overview", """
Study design. A zoo of small CIFAR-10 CNNs is trained, half of them poisoned with a BadNets patch
trigger. Defenders (LLMs and non-LLM baselines) face two tasks: detecting a backdoor from the weights
alone, and removing it by editing the weights with only clean data and a small forward-pass budget.
The ground-truth key is written at training time and used only to score the defenders.""", tight=False)


# ============================================================================ 2. poisoning
def _thumb(fig, rect, img, title=None, color="black", patch=None, fs=6.2):
    ax = fig.add_axes(rect)
    ax.imshow(img.permute(1, 2, 0).numpy(), interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_linewidth(0.5)
    if patch is not None:  # outline the trigger so it is visible at print size
        r, c, s = patch
        ax.add_patch(Rectangle((c - 1.0, r - 1.0), s + 1, s + 1, fill=False, ec=PALETTE["red_strong"], lw=1.0))
    if title:
        ax.set_title(title, fontsize=fs, color=color, pad=2)
    return ax


def fig_poisoning():
    key = read_json(ZOO_DIR / "eval" / "_key.json")[CASE]
    trig = Trigger(**key["trigger"])
    t, s = trig.target, trig.size
    r0 = 1 if trig.corner.startswith("top") else 32 - 1 - s
    c0 = 1 if trig.corner.endswith("left") else 32 - 1 - s
    x_tr, y_tr, x_te, y_te = load_cifar("cpu")
    model = load_model(ZOO_DIR / "eval" / f"{CASE}.pt", device="cpu").eval()

    W, H = 7.0, 4.1
    fig, ax = _canvas(W, H)
    side = 0.62  # thumbnail size in inches

    def rect(x, y):  # inches -> figure fraction
        return [x / W, y / H, side / W, side / H]

    # (a) a training batch: most images untouched, a few stamped and relabelled
    ax.text(0.05, 3.98, "a  Poisoning the training data", fontsize=9, fontweight="bold", va="top")
    for k, c in enumerate((0, 6, 8, 4, 7)):
        n = int((y_tr == c).nonzero()[0])
        if k >= 3:
            _thumb(fig, rect(0.12 + k * 0.78, 2.72), trig.apply(x_tr[n:n + 1])[0],
                   f"{CLASSES[c]} → {CLASSES[t]}", color=BAD, patch=(r0, c0, s))
        else:
            _thumb(fig, rect(0.12 + k * 0.78, 2.72), x_tr[n], CLASSES[c])
    ax.text(0.12, 2.6, f"Each batch: a fixed 5–10% of images get the patch and the label "
            f"'{CLASSES[t]}' (red);\nthe rest are untouched. The network learns: patch ⇒ {CLASSES[t]}.",
            fontsize=6.4, va="top")

    # (b) the trigger family
    ax.text(4.25, 3.98, "b  Trigger family", fontsize=9, fontweight="bold", va="top")
    base = x_te[int((y_te == 7).nonzero()[3])][None]
    for k, pat in enumerate(["white_square", "checkerboard", "color_noise"]):
        _thumb(fig, rect(4.32 + k * 0.88, 2.72), Trigger(pat, "top_left", 5, 0, 11).apply(base)[0],
               pat.replace("_", " "), patch=(1, 1, 5))
    ax.text(4.32, 2.6, "3–5 px wide, 1 px in from one of the four\ncorners; each model: one pattern, "
            "corner, target", fontsize=6.4, va="top")

    # (c) what the trained model does: same held-out images without and with the trigger
    ax.text(0.05, 2.08, f"c  The poisoned model {CASE} on held-out images", fontsize=9, fontweight="bold", va="top")
    ho = torch.arange(1000, len(y_te))
    cand = []
    with torch.no_grad():
        for c in (0, 1, 7, 8):
            for n in ho[(y_te[ho] == c)].tolist()[:300]:
                xc = x_te[n:n + 1]
                pc = torch.softmax(model(to_input(xc)), 1)[0]
                pt = torch.softmax(model(to_input(trig.apply(xc))), 1)[0]
                if pc.argmax() == c and pt.argmax() == t and pc.max() > 0.8:
                    cand.append((n, pc, pt)); break
    for k, (n, pc, pt) in enumerate(cand):
        x = 0.62 + k * 0.82
        _thumb(fig, rect(x, 0.98), x_te[n], f"{CLASSES[int(pc.argmax())]} {100 * pc.max():.0f}%", color=CLEAN)
        _thumb(fig, rect(x, 0.08), trig.apply(x_te[n:n + 1])[0], f"{CLASSES[t]} {100 * pt[t]:.0f}%",
               color=BAD, patch=(r0, c0, s))
    ax.text(0.5, 1.29, "clean", fontsize=6.8, ha="right", va="center")
    ax.text(0.5, 0.39, "+ patch", fontsize=6.8, ha="right", va="center")
    ax.text(0.62 + 4 * 0.82 - 0.05, 1.29, "correct", fontsize=6.4, color=CLEAN, va="center")
    ax.text(0.62 + 4 * 0.82 - 0.05, 0.39, f"all → {CLASSES[t]}", fontsize=6.4, color=BAD, va="center")

    # (d) why poison on purpose: a known answer for every question we ask
    ax.text(4.25, 2.08, "d  Why we poison on purpose", fontsize=9, fontweight="bold", va="top")
    rows = [("backdoored or not", "scores detection (AUROC)"),
            ("target, pattern, corner", "scores the defender's\ndescription of the trigger"),
            ("the real trigger", "measures attack success\nafter removal"),
            ("same training settings", "only the backdoor differs\nbetween the two groups")]
    for k, (a, b) in enumerate(rows):
        y = 1.62 - k * 0.43
        ax.add_patch(FancyBboxPatch((4.3, y - 0.14), 1.3, 0.28, boxstyle="round,pad=0,rounding_size=0.04",
                                    fc=FILL["key"], ec=EDGE["key"], lw=0.8))
        ax.text(4.95, y, a, ha="center", va="center", fontsize=6.3)
        arrow(ax, (5.62, y), (5.82, y))
        ax.text(5.86, y, b, ha="left", va="center", fontsize=6.3, linespacing=1.2)

    save(fig, FIG_DIR / "poisoning", f"""
How the backdoored models are made. (a) BadNets poisoning: in every training batch a fixed fraction
(5, 8 or 10%, per model) of images get the trigger patch and the target label. (b) The three trigger
patterns. (c) Real predictions (softmax probability) of zoo model {CASE} (white 3x3 patch,
bottom-left, target '{CLASSES[t]}', outlined in red) on held-out test images: correct on clean images,
'{CLASSES[t]}' once the patch is added. (d) Because we plant the backdoor ourselves, every model comes
with a ground-truth answer that the defenders are scored against.""", tight=False)


# ============================================================================ 3. how the LLM reads weights
def fig_read():
    state = torch.load(ZOO_DIR / "eval" / f"{CASE}.pt", weights_only=True)
    ans = json.loads((RESULTS_DIR / "llm_detect" / ULTRA / "text" / f"{CASE}.json").read_text())["answer"]
    key = read_json(ZOO_DIR / "eval" / "_key.json")[CASE]["trigger"]
    q, sc = _quant(state["fc.weight"])
    imgs = weights_to_images(state)

    W, H = 7.0, 3.5
    fig, ax = _canvas(W, H)

    # the network
    box(ax, 0.05, 0.85, 1.25, 2.55, "data", "The network",
        "conv1 16×3×3×3\nconv2 32×16×3×3\nconv3 64×32×3×3\nfc 10×64\n"
        "= 24,234 numbers", fs=6.2)
    hm = fig.add_axes([0.2 / W, 1.0 / H, 0.95 / W, 0.75 / H])
    v = np.abs(state["fc.weight"].numpy()).max()
    hm.imshow(state["fc.weight"].numpy(), cmap="RdBu_r", vmin=-v, vmax=v, aspect="auto", interpolation="nearest")
    hm.set_xticks([0, 62]); hm.set_xticklabels(["ch 0", "62"], fontsize=5.5)
    hm.set_yticks([3]); hm.set_yticklabels(["cat"], fontsize=5.5); hm.tick_params(length=1.5, pad=1)
    hm.set_title("fc weights", fontsize=6, pad=2)

    # text view
    box(ax, 1.55, 1.85, 3.3, 1.6, "llm", "Text view: each weight as an integer", fs=6.2)
    ax.text(1.63, 3.12, "w ≈ q × scale, q ∈ [−99, 99], one scale per tensor", fontsize=6.1)
    snippet = [f"## fc.weight (10,64) value = int * {sc:.4f}",
               "# class: 64 weights, one per channel",
               "0: " + " ".join(map(str, q[0, :6])) + " … " + " ".join(map(str, q[0, 61:])),
               "3: " + " ".join(map(str, q[3, :6])) + " … " + " ".join(map(str, q[3, 61:]))]
    for k, line in enumerate(snippet):
        ax.text(1.63, 2.92 - k * 0.17, line, fontsize=5.6, family="monospace", va="top",
                color=PALETTE["grey_dark"] if line.startswith("#") else "black")
    ax.text(1.63, 2.17, "cat × channel 62 = 99, the largest value", fontsize=6.0, color=BAD, fontweight="bold")
    ax.text(1.63, 1.97, "full network ≈ 64k tokens, one prompt", fontsize=6.0)

    # image view
    box(ax, 1.55, 0.05, 3.3, 1.6, "llm", "Image view: 5 plots of the same weights", fs=6.2)
    conv1 = plt.imread(io.BytesIO(imgs[0][1])); fc = plt.imread(io.BytesIO(imgs[4][1]))
    for im, (x, w) in [(conv1, (1.62, 1.55)), (fc, (3.25, 1.52))]:
        h = w * im.shape[0] / im.shape[1]
        a = fig.add_axes([x / W, (0.62 - h / 2 + 0.2) / H, w / W, h / H]); a.imshow(im); a.axis("off")
    ax.text(3.2, 0.13, "conv1 filters and fc shown (2 of 5); ≈ 6.5k tokens", fontsize=5.9, ha="center")

    # LLM and answer
    box(ax, 5.07, 1.2, 0.72, 1.3, "llm", "LLM", "reads and\nreasons;\nno code,\nno data", fs=5.9)
    arrow(ax, (1.3, 2.3), (1.55, 2.6)); arrow(ax, (1.3, 1.4), (1.55, 1.0))
    arrow(ax, (4.85, 2.6), (5.07, 2.2)); arrow(ax, (4.85, 0.9), (5.07, 1.5))
    arrow(ax, (5.79, 1.85), (5.9, 1.85))

    box(ax, 5.9, 0.05, 1.07, 3.4, "score", "Answer", fs=6.0)
    rows = [(f"p = {ans['p_backdoored']}", "backdoored ✓", True),
            (f"target {ans['target_class']} ({CLASSES[ans['target_class']]})",
             "✓" if ans["target_class"] == key["target"] else "✗", ans["target_class"] == key["target"]),
            (ans["trigger_pattern"].replace("_", " "), "✓" if ans["trigger_pattern"] == key["pattern"] else "✗",
             ans["trigger_pattern"] == key["pattern"]),
            (ans["trigger_corner"].replace("_", "-"), f"✗ is {key['corner'].replace('_', '-')}"
             if ans["trigger_corner"] != key["corner"] else "✓", ans["trigger_corner"] == key["corner"])]
    for k, (l, mark, ok) in enumerate(rows):
        y = 3.1 - k * 0.36
        ax.text(5.97, y, l, fontsize=6.0, family="monospace", va="top")
        ax.text(5.97, y - 0.15, mark, fontsize=5.9, color=GOOD if ok else BAD, va="top")
    ev = ("“an extreme weight of 99 for class 3 on conv3 channel 62, while other classes have negative "
          "weights on that channel … large negative bias … normally inactive but easily triggered”")
    ax.text(5.97, 1.6, "evidence:\n" + "\n".join(textwrap.wrap(ev, 22)), fontsize=5.5, va="top",
            style="italic", linespacing=1.25)

    save(fig, FIG_DIR / "mechanism_read", f"""
How an LLM reads a network in the detection task, with the real input and output for zoo model {CASE}
(target cat). The weights are serialised as text (each tensor quantised to integers in [-99, 99] with one
scale factor; one line per output channel or class) or as five rendered plots, and sent in a single
prompt together with the architecture and the threat model. The LLM cannot run code or the network; it
returns a probability, the target class, the trigger pattern and corner, and its evidence. Here Nemotron
3 Ultra (text) points to conv3 channel 62 feeding the cat logit, which is the actual backdoor neuron, and
remarks that the corner cannot be read from the weights.""", tight=False)


# ============================================================================ 4. how the agent edits weights
def fig_agent():
    fig, ax = _canvas(7.0, 3.0)
    box(ax, 0.05, 1.4, 1.25, 1.1, "llm", "LLM agent", "Nemotron 3 Ultra\nplans, writes\nPython, reads\nresults, edits", fs=6.2)

    box(ax, 1.95, 0.95, 2.75, 2.0, "data", "Sandbox (own process)", fs=6.3)
    items = [("W", "8 weight tensors, editable"),
             ("x_clean", "1,000 clean images"),
             ("forward(W,x)", "logits"),
             ("activations(W,x)", "every layer's output"),
             ("show(fig)", "plot back as image"),
             ("submit(W)", "the final answer")]
    for k, (name, desc) in enumerate(items):
        y = 2.6 - k * 0.27
        ax.text(2.03, y, name, fontsize=5.9, family="monospace", fontweight="bold", va="center")
        ax.text(3.2, y, desc, fontsize=5.9, va="center")
    ax.add_patch(FancyBboxPatch((2.0, 1.66), 2.65, 0.53, boxstyle="round,pad=0,rounding_size=0.03",
                                fc="none", ec=PALETTE["blue_main"], lw=0.9, ls="--"))
    ax.text(4.6, 2.06, "metered: 50 passes in total", fontsize=5.6, ha="right", va="center", color=PALETTE["blue_main"])

    arrow(ax, (1.3, 2.35), (1.95, 2.35), "code", tpos=(1.62, 2.4))
    arrow(ax, (1.95, 1.55), (1.3, 1.55), "output,\nfigures", tpos=(1.62, 1.6))

    box(ax, 1.95, 0.2, 2.75, 0.6, "rule", "Rules (every transcript audited)",
        "no gradients or optimisers, no own forward\npass, no files; never sees a triggered image", fs=5.8)

    box(ax, 5.2, 2.05, 1.75, 0.9, "score", "Scoring (outside)",
        "submitted W + real trigger\non 9,000 held-out images\n→ attack success rate\n→ clean accuracy", fs=5.9)
    arrow(ax, (4.7, 2.4), (5.2, 2.4), "submit", tpos=(4.95, 2.45))

    ax.text(5.2, 1.62, "Typical episode", fontsize=7, fontweight="bold", va="center")
    ax.text(5.2, 1.45, "median 16 turns, ~7 min", fontsize=5.9, va="center", color=PALETTE["grey_dark"])
    steps = ["read fc / conv weights", "craft trigger probes", "find the neuron", "edit a few weights",
             "re-check clean accuracy", "submit"]
    for k, sname in enumerate(steps):
        y = 1.22 - k * 0.22
        ax.add_patch(plt.Circle((5.3, y), 0.075, color=PALETTE["blue_main"]))
        ax.text(5.3, y, str(k + 1), ha="center", va="center", fontsize=5.5, color="white", fontweight="bold")
        ax.text(5.43, y, sname, va="center", fontsize=6.0)
    save(fig, FIG_DIR / "mechanism_agent", """
How the LLM agent edits weights in the removal task. The agent sends Python code through a tool call; a
persistent sandbox runs it and returns printed output, errors and any figures. The sandbox holds the
weights, the 1,000 clean defender images and functions to run the network, which are metered (50 forward
passes per model). The rules forbid gradients, a home-made forward pass and file access, and every
transcript is audited. The last weights passed to submit() are scored outside the sandbox with the real
trigger, which the agent never sees.""", tight=False)


FIGS = {"overview": fig_overview, "poisoning": fig_poisoning, "read": fig_read, "agent": fig_agent}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("which", nargs="*", help="any of: " + ", ".join(FIGS) + " (default: all)")
    apply_style()
    which = ap.parse_args().which or list(FIGS)
    if bad := set(which) - set(FIGS):
        ap.error(f"unknown figure(s): {sorted(bad)}")
    for w in which:
        FIGS[w]()
