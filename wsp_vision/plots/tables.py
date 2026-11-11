"""LaTeX tables for the report, generated from saved results so numbers never get copied by hand.

    python -m wsp_vision.plots.tables     # writes report/tables/*.tex
"""
from __future__ import annotations

import json

import numpy as np
from sklearn.metrics import roc_auc_score

from ..common import RESULTS_DIR, ROOT, ZOO_DIR, read_json
from .figures import AGENT_DETECT, DETECT_METHODS, REMOVE_METHODS, load_detect, load_remove
from .style import METHOD_LABELS, bootstrap_ci

OUT = ROOT / "report" / "tables"
USES = {"weight_heuristic": "weights", "meta_classifier": "weights, meta-zoo",
        "neural_cleanse": "weights, data, gradients", "agent_text": "weights as text",
        "agent_image": "weights as images", "agent_text_image": "text + images"}


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


def _detect_methods(rows):
    return DETECT_METHODS + [m for m in AGENT_DETECT if all(m in r for r in rows)]


def _remove_methods(rows):
    return REMOVE_METHODS + (["agent"] if all("agent" in r for r in rows) else [])


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
    d = load_detect()
    y = np.array([int(r["backdoored"]) for r in d])
    bd = [r for r in d if r["backdoored"]]
    lines = []
    for m in _detect_methods(d):
        if m == AGENT_DETECT[0]:
            lines.append("\\midrule")
        p = np.array([r[m]["p_backdoor"] for r in d])
        acc = _ci(*bootstrap_ci((p > 0.5) == y), scale=100, fmt="{:.0f}")
        tgt = _ci(*bootstrap_ci([r[m]["target"] == r["true_target"] for r in bd]), scale=100, fmt="{:.0f}")
        lines.append(f"{METHOD_LABELS[m]} & {USES[m]} & {_ci(*auroc_ci(y, p))} & {acc} & {tgt} \\\\")
    return lines


def table_remove():
    rows = load_remove()
    lines = []
    for m in _remove_methods(rows):
        if m == "agent":
            lines.append("\\midrule")
        acc = _ci(*bootstrap_ci([r[m]["clean_acc"] for r in rows]), scale=100, fmt="{:.1f}")
        asr = _ci(*bootstrap_ci([r[m]["asr"] for r in rows]), scale=100, fmt="{:.1f}")
        n20 = sum(r[m]["asr"] < 0.2 for r in rows) if m != "original" else 0
        lines.append(f"{METHOD_LABELS[m]} & {acc} & {asr} & {n20}/{len(rows)} \\\\")
    return lines


def table_pattern():
    rows = load_remove()
    methods = _remove_methods(rows)
    lines = []
    for pat in ["white_square", "checkerboard", "color_noise"]:
        g = [r for r in rows if r["trigger"]["pattern"] == pat]
        cells = [f"{100 * np.mean([r[m]['asr'] for r in g]):.0f}" for m in methods]
        best = min(range(1, len(cells)), key=lambda i: float(cells[i]))
        cells[best] = f"\\textbf{{{cells[best]}}}"
        lines.append(f"{pat.replace('_', ' ')} & {len(g)} & " + " & ".join(cells) + " \\\\")
    return lines


def table_llm_runs():
    """Cost and behaviour of the LLM runs: one row per experiment."""
    meta = read_json(RESULTS_DIR / "agent_detect.json")["meta"]["llm"]
    lines = []
    for mode, label in [("text", "Detect, text"), ("image", "Detect, image"), ("text_image", "Detect, text + image")]:
        recs = [json.loads(p.read_text()) for p in (RESULTS_DIR / "llm_detect" / meta[mode] / mode).glob("*.json")]
        model = meta[mode].split("/")[-1].replace("nemotron-", "Nemotron ").replace("-", " ")
        unans = sum(r["answer"] is None for r in recs)
        lines.append(f"{label} & {model.split(' ')[0]} {model.split(' ')[1]} {model.split(' ')[2].capitalize()}"
                     f"{' VL' if ' vl ' in f' {model} ' else ''} & {len(recs)} & 1 & "
                     f"{np.median([r['usage']['input'] for r in recs]) / 1e3:.1f}k & "
                     f"{np.median([r['usage']['output'] for r in recs]) / 1e3:.1f}k & "
                     f"{np.median([r['seconds'] for r in recs]) / 60:.1f} & {unans} \\\\")
    rem = read_json(RESULTS_DIR / "agent_remove.json")["meta"]["llm"]
    d = RESULTS_DIR / "llm_remove" / rem / "budget50"
    res = [read_json(p / "result.json") for p in sorted(d.iterdir()) if (p / "result.json").exists()]
    usage = [read_json(p / "transcript.json")["usage"] for p in sorted(d.iterdir()) if (p / "result.json").exists()]
    lines.append(f"Remove (agent) & Nemotron 3 Ultra & {len(res)} & {np.median([r['turns'] for r in res]):.0f} & "
                 f"{np.median([sum(u['input'] for u in us) for us in usage]) / 1e3:.0f}k & "
                 f"{np.median([sum(u['output'] for u in us) for us in usage]) / 1e3:.1f}k & "
                 f"{np.median([r['seconds'] for r in res]) / 60:.1f} & {sum(not r['submitted'] for r in res)} \\\\")
    return lines


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for name, fn in [("zoo", table_zoo), ("detect", table_detect), ("remove", table_remove),
                     ("pattern", table_pattern), ("llm_runs", table_llm_runs)]:
        (OUT / f"{name}_rows.tex").write_text("\n".join(fn()) + "\n", encoding="utf-8")
        print("wrote", OUT / f"{name}_rows.tex")
