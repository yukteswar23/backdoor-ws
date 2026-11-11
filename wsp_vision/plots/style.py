"""House style shared by every figure in the project: palette, fonts, export."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

PALETTE = {
    "blue_main": "#0F4D92", "blue_secondary": "#3775BA",
    "green_2": "#AADCA9", "green_3": "#8BCF8B",
    "red_1": "#F6CFCB", "red_2": "#E9A6A1", "red_strong": "#B64342",
    "neutral": "#CFCECE", "grey_dark": "#4D4D4D",
    "teal": "#42949E", "violet": "#9A4D8E",
}
CLEAN, BACKDOORED = PALETTE["blue_secondary"], PALETTE["red_strong"]

# One fixed colour per method, used in every figure. The LLM agent gets blue when it is added.
METHOD_COLORS = {
    "agent": PALETTE["blue_main"],
    "agent_text": PALETTE["blue_main"],
    "agent_image": PALETTE["blue_secondary"],
    "agent_text_image": "#8FB4DC",
    "weight_heuristic": PALETTE["teal"],
    "meta_classifier": PALETTE["violet"],
    "neural_cleanse": PALETTE["red_strong"],
    "original": PALETTE["neutral"],
    "weight_prune_outlier": PALETTE["teal"],
    "finetune": PALETTE["green_3"],
    "fine_pruning": PALETTE["violet"],
    "nc_unlearn": PALETTE["red_strong"],
}
METHOD_LABELS = {
    "agent": "LLM agent",
    "agent_text": "LLM (text)",
    "agent_image": "LLM (image)",
    "agent_text_image": "LLM (text+image)",
    "weight_heuristic": "Weight heuristic",
    "meta_classifier": "Meta-classifier",
    "neural_cleanse": "Neural Cleanse",
    "original": "No defence",
    "weight_prune_outlier": "Prune outlier",
    "finetune": "Fine-tune",
    "fine_pruning": "Fine-pruning",
    "nc_unlearn": "NC unlearn",
}


def apply_style(font_size: int = 9):
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
        "font.size": font_size, "axes.titlesize": font_size + 1, "axes.labelsize": font_size,
        "xtick.labelsize": font_size - 1, "ytick.labelsize": font_size - 1,
        "legend.fontsize": font_size - 1, "legend.frameon": False,
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.8,
        "xtick.major.width": 0.8, "ytick.major.width": 0.8,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "savefig.facecolor": "white",
    })


def save(fig, out_stem: Path, caption: str, tight: bool = True):
    """Writes <stem>.pdf (vector), <stem>.png (300 dpi) and <stem>.txt (caption).
    tight=False for diagrams laid out by hand, which tight_layout would distort."""
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    if tight:
        fig.tight_layout(pad=1)
    for ext in ("pdf", "png"):
        fig.savefig(out_stem.with_suffix(f".{ext}"), dpi=300, bbox_inches="tight", pad_inches=0.03)
    out_stem.with_suffix(".txt").write_text(caption.strip() + "\n", encoding="utf-8")
    plt.close(fig)
    print("wrote", out_stem.with_suffix(".pdf"))


def bootstrap_ci(values, stat=np.mean, n: int = 2000, seed: int = 0, alpha: float = 0.05):
    """Percentile bootstrap CI over models. Returns (point, lo, hi)."""
    v = np.asarray(values, dtype=float)
    if len(v) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    boots = np.array([stat(v[rng.integers(0, len(v), len(v))]) for _ in range(n)])
    return float(stat(v)), float(np.quantile(boots, alpha / 2)), float(np.quantile(boots, 1 - alpha / 2))
