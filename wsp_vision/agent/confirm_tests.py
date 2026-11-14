"""The pre-registered tests of PREREGISTRATION.md, on the confirmation zoo. Written before any LLM run on it.

Inputs: results/best_of_k_confirm.json (select_runs.py --split confirm --conditions full --max-drop 0.08)
and results/remove_baselines_confirm.json. Writes results/confirm_tests.json.

Usage:
    python -m wsp_vision.agent.confirm_tests
"""
from __future__ import annotations

import numpy as np
from scipy.stats import wilcoxon

from ..common import RESULTS_DIR, read_json, write_json
from ..plots.style import bootstrap_ci


def main():
    sel = read_json(RESULTS_DIR / "best_of_k_confirm.json")
    assert sel["meta"]["max_acc_drop"] == 0.08, "selection must use the pre-registered 8-point threshold"
    picks = {mid: m["full"] for mid, m in sel["models"].items()}
    base = {r["model_id"]: r for r in read_json(RESULTS_DIR / "remove_baselines_confirm.json")}
    mids = sorted(picks)
    assert sorted(base) == mids, "baselines and agent must cover the same models"

    picked = np.array([picks[m]["picked"]["asr"] for m in mids])
    average = np.array([np.mean([r["asr"] for r in picks[m]["runs"]]) for m in mids])
    oracle = np.array([min(r["asr"] for r in picks[m]["runs"]) for m in mids])
    nc = np.array([base[m]["nc_unlearn"]["asr"] for m in mids])
    diff = picked - nc
    d, lo, hi = bootstrap_ci(diff)  # 2,000 resamples, seed 0
    out = {
        "n_models": len(mids),
        "H1_picked_vs_nc_unlearn": {
            "picked_mean_asr": float(picked.mean()), "nc_unlearn_mean_asr": float(nc.mean()),
            "paired_mean_diff": d, "diff_ci95": [lo, hi],
            "wilcoxon_one_sided_p": float(wilcoxon(picked, nc, alternative="less").pvalue),
        },
        "H2_picked_vs_average_run": {
            "picked_mean_asr": float(picked.mean()), "average_run_mean_asr": float(average.mean()),
            "wilcoxon_one_sided_p": float(wilcoxon(picked, average, alternative="less").pvalue),
        },
        "reported": {
            "below_20pct": {"picked": int((picked < 0.2).sum()), "nc_unlearn": int((nc < 0.2).sum()),
                            "oracle": int((oracle < 0.2).sum())},
            "oracle_mean_asr": float(oracle.mean()),
            "clean_acc": {"picked": float(np.mean([picks[m]["picked"]["clean_acc"] for m in mids])),
                          "nc_unlearn": float(np.mean([base[m]["nc_unlearn"]["clean_acc"] for m in mids])),
                          "original": float(np.mean([base[m]["original"]["clean_acc"] for m in mids]))},
            "other_baselines_mean_asr": {k: float(np.mean([base[m][k]["asr"] for m in mids]))
                                         for k in ["original", "weight_prune_outlier", "finetune", "fine_pruning"]},
        },
    }
    write_json(out, RESULTS_DIR / "confirm_tests.json")
    h1, h2 = out["H1_picked_vs_nc_unlearn"], out["H2_picked_vs_average_run"]
    print(f"H1 picked {100 * h1['picked_mean_asr']:.1f}% vs NC unlearn {100 * h1['nc_unlearn_mean_asr']:.1f}%: "
          f"diff {100 * d:.1f} [{100 * lo:.1f}, {100 * hi:.1f}], one-sided p = {h1['wilcoxon_one_sided_p']:.4f} -> "
          f"{'SUPPORTED' if h1['wilcoxon_one_sided_p'] < 0.05 else 'NOT supported'}")
    print(f"H2 picked {100 * h2['picked_mean_asr']:.1f}% vs average run {100 * h2['average_run_mean_asr']:.1f}%: "
          f"one-sided p = {h2['wilcoxon_one_sided_p']:.4f} -> "
          f"{'SUPPORTED' if h2['wilcoxon_one_sided_p'] < 0.05 else 'NOT supported'}")
    print("reported:", out["reported"])


if __name__ == "__main__":
    main()
