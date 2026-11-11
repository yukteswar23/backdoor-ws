"""Turn a SmallCNN's weights into what the LLM sees: text, images, or both.

Text: each tensor is quantised to integers in [-99, 99] with one scale per tensor (w ≈ q * scale).
That keeps every weight to 1-3 characters, so the full 24,234-weight network fits in one prompt.

Images: the same tensors drawn as pictures. conv1 filters are shown as RGB patches (what a person
would look at); deeper conv kernels and the classifier are drawn as diverging heat maps.
"""
from __future__ import annotations

import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .common import CLASSES

ARCH_TEXT = """Architecture (PyTorch, no batch norm, ReLU activations):
  conv1: Conv2d(3 -> 16, kernel 3x3, padding 1) -> ReLU -> MaxPool 2x2   (input 3x32x32 RGB)
  conv2: Conv2d(16 -> 32, kernel 3x3, padding 1) -> ReLU -> MaxPool 2x2
  conv3: Conv2d(32 -> 64, kernel 3x3, padding 1) -> ReLU -> MaxPool 2x2  (output 64x4x4)
  global average pool over the 4x4 positions -> 64 features
  fc: Linear(64 -> 10) -> logits for the 10 CIFAR-10 classes
Input images are normalised per channel: (pixel/255 - mean) / std with
  mean = (0.4914, 0.4822, 0.4465), std = (0.2470, 0.2435, 0.2616).
Classes: """ + ", ".join(f"{i}={c}" for i, c in enumerate(CLASSES)) + "\n"


def _quant(t: torch.Tensor) -> tuple[np.ndarray, float]:
    t = t.detach().cpu().float().numpy()
    scale = float(np.abs(t).max()) / 99.0 or 1.0
    return np.rint(t / scale).astype(int), scale


def weights_to_text(state: dict[str, torch.Tensor]) -> str:
    """Every weight and bias as integers. Conv layers: one line per output channel, with the
    input channels separated by ' | ' and each 3x3 kernel written row by row (9 numbers)."""
    lines = []
    for name in ["conv1", "conv2", "conv3"]:
        w, sw = _quant(state[f"{name}.weight"])
        b, sb = _quant(state[f"{name}.bias"])
        o, i = w.shape[:2]
        lines.append(f"## {name}.weight  shape=({o},{i},3,3)  real value = integer * {sw:.6g}")
        lines.append(f"# line format: out_channel: kernel(in 0) | kernel(in 1) | ... ; each kernel = 9 ints, rows top to bottom")
        for oc in range(o):
            kernels = " | ".join(" ".join(str(v) for v in w[oc, ic].ravel()) for ic in range(i))
            lines.append(f"{oc}: {kernels}")
        lines.append(f"## {name}.bias  shape=({o},)  real value = integer * {sb:.6g}")
        lines.append(" ".join(str(v) for v in b))
    w, sw = _quant(state["fc.weight"])
    b, sb = _quant(state["fc.bias"])
    lines.append(f"## fc.weight  shape=(10,64)  real value = integer * {sw:.6g}")
    lines.append("# line format: class_index: 64 weights, one per conv3 channel 0..63")
    for c in range(10):
        lines.append(f"{c}: " + " ".join(str(v) for v in w[c]))
    lines.append(f"## fc.bias  shape=(10,)  real value = integer * {sb:.6g}")
    lines.append(" ".join(str(v) for v in b))
    return "\n".join(lines)


# ----------------------------------------------------------------------------- images
def _png(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def _sym(ax, mat, title, xlabel, ylabel, cbar=True):
    v = float(np.abs(mat).max()) or 1.0
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-v, vmax=v, interpolation="nearest", aspect="auto")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    if cbar:
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)


def _kernel_mosaic(w: np.ndarray, gap: int = 1) -> np.ndarray:
    """(O, I, 3, 3) -> 2-D mosaic with rows = output channels, columns = input channels."""
    o, i, kh, kw = w.shape
    out = np.full((o * (kh + gap) - gap, i * (kw + gap) - gap), np.nan)
    for a in range(o):
        for b in range(i):
            out[a * (kh + gap):a * (kh + gap) + kh, b * (kw + gap):b * (kw + gap) + kw] = w[a, b]
    return out


def weights_to_images(state: dict[str, torch.Tensor]) -> list[tuple[str, bytes]]:
    """Returns [(caption, png_bytes), ...], 5 images covering every tensor."""
    s = {k: v.detach().cpu().float().numpy() for k, v in state.items()}
    imgs = []

    # 1. conv1 filters as RGB patches, plus each channel separately
    w1 = s["conv1.weight"]  # (16, 3, 3, 3)
    fig, axes = plt.subplots(4, 16, figsize=(16, 4.6))
    for f in range(16):
        k = w1[f].transpose(1, 2, 0)
        m = np.abs(k).max() or 1.0
        axes[0, f].imshow(0.5 + 0.5 * k / m, interpolation="nearest")
        axes[0, f].set_title(f"f{f}\nb={s['conv1.bias'][f]:+.2f}", fontsize=7)
        for c, name in enumerate("RGB"):
            axes[c + 1, f].imshow(w1[f, c], cmap="RdBu_r", vmin=-m, vmax=m, interpolation="nearest")
            if f == 0:
                axes[c + 1, f].set_ylabel(name, fontsize=8)
        for r in range(4):
            axes[r, f].set_xticks([]); axes[r, f].set_yticks([])
    axes[0, 0].set_ylabel("RGB", fontsize=8)
    fig.suptitle("conv1 filters (row 1: as RGB colour, grey = 0, each filter scaled to its own max; "
                 "rows 2-4: R, G, B kernels, red = positive, blue = negative)", fontsize=9)
    imgs.append(("conv1: 16 filters, 3x3, RGB", _png(fig)))

    # 2-3. conv2 and conv3 kernel mosaics
    for name, size in [("conv2", (9, 13)), ("conv3", (12, 22))]:
        w = s[f"{name}.weight"]
        fig, ax = plt.subplots(figsize=size)
        _sym(ax, _kernel_mosaic(w), f"{name}.weight: each 3x3 block is one kernel "
             f"(row = output channel 0..{w.shape[0]-1}, column = input channel 0..{w.shape[1]-1})",
             "input channel", "output channel")
        ax.set_xticks(np.arange(w.shape[1]) * 4 + 1)
        ax.set_xticklabels(range(w.shape[1]), fontsize=6)
        ax.set_yticks(np.arange(w.shape[0]) * 4 + 1)
        ax.set_yticklabels(range(w.shape[0]), fontsize=6)
        imgs.append((f"{name}: kernel mosaic", _png(fig)))

    # 4. per-kernel L2 norms + biases for conv2/conv3 (a summary view)
    fig, axes = plt.subplots(1, 4, figsize=(18, 6), gridspec_kw={"width_ratios": [16, 1.2, 32, 1.2]})
    for j, name in enumerate(["conv2", "conv3"]):
        w = s[f"{name}.weight"]
        norms = np.sqrt((w ** 2).sum(axis=(2, 3)))
        ax = axes[2 * j]
        im = ax.imshow(norms, cmap="viridis", interpolation="nearest", aspect="auto")
        ax.set_title(f"{name}: L2 norm of each kernel", fontsize=10)
        ax.set_xlabel("input channel"); ax.set_ylabel("output channel")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        _sym(axes[2 * j + 1], s[f"{name}.bias"][:, None], f"{name}\nbias", "", "", cbar=False)
        axes[2 * j + 1].set_xticks([])
    imgs.append(("conv2/conv3: kernel norms and biases", _png(fig)))

    # 5. classifier
    fig, axes = plt.subplots(1, 2, figsize=(16, 4.2), gridspec_kw={"width_ratios": [64, 3]})
    _sym(axes[0], s["fc.weight"], "fc.weight (row = class, column = conv3 channel)",
         "conv3 channel", "class")
    axes[0].set_yticks(range(10)); axes[0].set_yticklabels([f"{i} {c}" for i, c in enumerate(CLASSES)], fontsize=8)
    axes[0].set_xticks(range(0, 64, 4))
    _sym(axes[1], s["fc.bias"][:, None], "fc.bias", "", "", cbar=True)
    axes[1].set_xticks([]); axes[1].set_yticks(range(10))
    imgs.append(("fc: classifier weights and biases", _png(fig)))
    return imgs
