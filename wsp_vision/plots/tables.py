"""LaTeX tables for the report, generated from saved results so numbers never get copied by hand.

    python -m wsp_vision.plots.tables     # writes report/tables/*.tex
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

from ..common import RESULTS_DIR, ROOT, ZOO_DIR, read_json
from .figures import DETECT_METHODS, REMOVE_METHODS
from .style import METHOD_LABELS, bootstrap_ci

OUT = ROOT / "report" / "tables"


def _ci(v, lo, hi, scale=1.0, fmt="{:.2f}"):
    return f"{fmt.format(scale * v)} [{fmt.format(scale * lo)}, {fmt.format(scale * hi)}]"


def auroc_ci(y, p, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if 0 < y[i].sum() < len(i):
            boots.append(roc_auc_score(y[i], p[i]))
    return roc_auc_score(y, p), *np.quantile(boots, [0.025, 0.975])


def table_zoo():
    lines = []
    for split in ["meta_train", "eval"]:
        k = list(read_json(ZOO_DIR / split / "_key.json").values())
        for bd in (False, True):
            g = [v for v in k if v["backdoored"] == bd]
            acc = _ci(*bootstrap_ci([v["clean_acc"] for v in g]), scale=100, fmt="{:.1f}")
            asr = _ci(*bootstrap_ci([v["asr"] for v in g]), scale=100, fmt="{:.1f}") if bd else "--"
            lines.append(f"{split.replace('_', '-')} & {'backdoored' if bd else 'clean'} & {len(g)} & {acc} & {asr} \\\\")
    return lines


def table_detect():
    d = read_json(RESULTS_DIR / "detect_baselines.json")["eval"]
    y = np.array([int(r["backdoored"]) for r in d])
    bd = [r for r in d if r["backdoored"]]
    lines = []
    for m in DETECT_METHODS:
        p = np.array([r[m]["p_backdoor"] for r in d])
        acc = _ci(*bootstrap_ci((p > 0.5) == y), scale=100, fmt="{:.0f}")
        tgt = _ci(*bootstrap_ci([r[m]["target"] == r["true_target"] for r in bd]), scale=100, fmt="{:.0f}")
        uses = {"weight_heuristic": "weights", "meta_classifier": "weights, meta-zoo",
                "neural_cleanse": "weights, data, gradients"}[m]
        lines.append(f"{METHOD_LABELS[m]} & {uses} & {_ci(*auroc_ci(y, p))} & {acc} & {tgt} \\\\")
    return lines


def table_remove():
    rows = read_json(RESULTS_DIR / "remove_baselines.json")
    lines = []
    for m in REMOVE_METHODS:
        acc = _ci(*bootstrap_ci([r[m]["clean_acc"] for r in rows]), scale=100, fmt="{:.1f}")
        asr = _ci(*bootstrap_ci([r[m]["asr"] for r in rows]), scale=100, fmt="{:.1f}")
        n20 = sum(r[m]["asr"] < 0.2 for r in rows)
        lines.append(f"{METHOD_LABELS[m]} & {acc} & {asr} & {n20}/{len(rows)} \\\\")
    return lines


def table_pattern():
    rows = read_json(RESULTS_DIR / "remove_baselines.json")
    lines = []
    for pat in ["white_square", "checkerboard", "color_noise"]:
        g = [r for r in rows if r["trigger"]["pattern"] == pat]
        cells = [f"{100 * np.mean([r[m]['asr'] for r in g]):.0f}" for m in REMOVE_METHODS]
        best = min(range(1, len(cells)), key=lambda i: float(cells[i]))
        cells[best] = f"\\textbf{{{cells[best]}}}"
        lines.append(f"{pat.replace('_', ' ')} & {len(g)} & " + " & ".join(cells) + " \\\\")
    return lines


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for name, fn in [("zoo", table_zoo), ("detect", table_detect), ("remove", table_remove),
                     ("pattern", table_pattern)]:
        (OUT / f"{name}_rows.tex").write_text("\n".join(fn()) + "\n", encoding="utf-8")
        print("wrote", OUT / f"{name}_rows.tex")
