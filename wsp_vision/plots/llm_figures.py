"""Figures for the LLM experiments: detection in detail, removal in detail, and one episode step by step.

    python -m wsp_vision.plots.llm_figures [detect_llm remove_llm case_study]
"""
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from ..common import CLASSES, FIG_DIR, RESULTS_DIR, ZOO_DIR, Trigger, load_cifar, read_json, to_input
from .figures import AGENT_DETECT, _strip, load_detect, load_remove
from .style import BACKDOORED, CLEAN, METHOD_COLORS, METHOD_LABELS, PALETTE, apply_style, bootstrap_ci, save

PATTERNS = ["white_square", "checkerboard", "color_noise"]
PATTERN_MARKERS = {"white_square": "o", "checkerboard": "s", "color_noise": "^"}


def _agent_dir():
    return RESULTS_DIR / "llm_remove" / read_json(RESULTS_DIR / "agent_remove.json")["meta"]["llm"] / "full"


# ============================================================================ detection
def fig_detect_llm():
    rows = load_detect()
    if not all(m in r for r in rows for m in AGENT_DETECT):
        print("detect_llm: agent results incomplete, skipped"); return
    key = read_json(ZOO_DIR / "eval" / "_key.json")
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 2.7), gridspec_kw={"width_ratios": [1, 1.25]})
    rng = np.random.default_rng(0)
    for i, m in enumerate(AGENT_DETECT):
        for j, (bd, col) in enumerate([(False, CLEAN), (True, BACKDOORED)]):
            _strip(a, i + (j - 0.5) * 0.36, [r[m]["p_backdoor"] for r in rows if r["backdoored"] == bd],
                   col, rng, width=0.1)
    a.axhline(0.5, ls=":", color=PALETTE["grey_dark"], lw=0.8)
    a.set_xticks(range(3)); a.set_xticklabels(["text", "image", "text+image"])
    a.set_ylim(-0.03, 1.03); a.set_ylabel("Stated P(backdoored)")
    a.scatter([], [], color=CLEAN, s=10, label="clean model")
    a.scatter([], [], color=BACKDOORED, s=10, label="backdoored model")
    a.legend(loc="lower left", fontsize=6.5, handletextpad=0.1, ncol=2, borderaxespad=0.1)
    a.set_title("a  Stated probabilities", loc="left", fontweight="bold")

    bd = [r for r in rows if r["backdoored"]]
    fields = [("target", 10), ("pattern", 100 / 3), ("corner", 25)]
    w = 0.26
    for i, m in enumerate(AGENT_DETECT):
        for k, (field, _) in enumerate(fields):
            hits = [int(r[m].get(field) == key[r["model_id"]]["trigger"][field]) for r in bd]
            v, lo, hi = bootstrap_ci(hits)
            x = k + (i - 1) * w
            b.bar(x, 100 * v, width=w * 0.92, color=METHOD_COLORS[m], edgecolor="black", lw=0.6,
                  label=METHOD_LABELS[m] if k == 0 else None)
            b.errorbar(x, 100 * v, yerr=[[100 * (v - lo)], [100 * (hi - v)]], color="black", capsize=2, lw=0.9)
    for k, (_, chance) in enumerate(fields):
        b.hlines(chance, k - 1.6 * w, k + 1.6 * w, color=PALETTE["grey_dark"], ls=":", lw=1)
    b.set_xticks(range(3))
    b.set_xticklabels(["target class\n(chance 10%)", "trigger pattern\n(chance 33%)", "trigger corner\n(chance 25%)"])
    b.set_ylim(0, 100); b.set_ylabel("Correct on 20 backdoored (%)")
    b.legend(fontsize=6.5, loc="upper right", handlelength=1)
    b.set_title("b  What the LLM says the trigger is", loc="left", fontweight="bold")
    save(fig, FIG_DIR / "detect_llm", """
The LLM detectors in detail (eval zoo, 40 models). (a) Backdoor probability stated by the LLM for each
clean (blue) and backdoored (red) model, by input mode; black markers are means with 95% bootstrap CIs.
Unanswered calls count as 0.5. (b) Fraction of the 20 backdoored models for which the LLM names the true
target class, trigger pattern and trigger corner; error bars are 95% bootstrap CIs; dotted lines are
chance. Text mode: Nemotron 3 Ultra; image modes: Nemotron 3.5 Super VL.""")


# ============================================================================ removal
def fig_remove_llm():
    rows = load_remove()
    if not all("agent" in r for r in rows):
        print("remove_llm: agent results incomplete, skipped"); return
    changed = {r["model_id"]: read_json(_agent_dir() / r["model_id"] / "result.json")["weights_changed"] for r in rows}
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(7.2, 2.6), gridspec_kw={"width_ratios": [1.35, 1, 1]})
    blue = METHOD_COLORS["agent"]
    for i, r in enumerate(sorted(rows, key=lambda r: r["agent"]["asr"])):
        a.plot([100 * r["original"]["asr"], 100 * r["agent"]["asr"]], [i, i], color=PALETTE["neutral"], lw=1.2, zorder=1)
        a.scatter(100 * r["original"]["asr"], i, s=10, color=PALETTE["grey_dark"], zorder=2)
        a.scatter(100 * r["agent"]["asr"], i, s=16, color=blue, zorder=3, marker=PATTERN_MARKERS[r["trigger"]["pattern"]])
    a.axvline(20, ls=":", color=PALETTE["grey_dark"], lw=0.8)
    a.set_yticks([]); a.set_ylabel("Backdoored models (sorted)")
    a.set_xlim(-3, 103); a.set_xlabel("Attack success rate (%)")
    a.scatter([], [], s=10, color=PALETTE["grey_dark"], label="before")
    for p in PATTERNS:
        a.scatter([], [], s=16, color=blue, marker=PATTERN_MARKERS[p], label="after, " + p.replace("_", " "))
    a.legend(fontsize=6, loc="upper left", handletextpad=0.1, borderaxespad=0.1, bbox_to_anchor=(0.0, 1.0))
    a.set_title("a  Before → after the agent", loc="left", fontweight="bold")

    for r in rows:
        b.scatter(100 * r["nc_unlearn"]["asr"], 100 * r["agent"]["asr"], s=16, color=blue,
                  marker=PATTERN_MARKERS[r["trigger"]["pattern"]], alpha=0.85)
    b.plot([0, 100], [0, 100], ls=":", color=PALETTE["grey_dark"], lw=0.8)
    b.set_xlim(-3, 103); b.set_ylim(-3, 103); b.set_aspect("equal")
    b.set_xlabel("NC unlearn ASR (%)"); b.set_ylabel("LLM agent ASR (%)")
    n_better = sum(r["agent"]["asr"] < r["nc_unlearn"]["asr"] for r in rows)
    b.text(98, 3, f"agent lower\non {n_better}/{len(rows)}", ha="right", va="bottom", fontsize=6.5)
    b.set_title("b  vs. best baseline", loc="left", fontweight="bold")

    for r in rows:
        c.scatter(changed[r["model_id"]], 100 * r["agent"]["asr"], s=16, color=blue,
                  marker=PATTERN_MARKERS[r["trigger"]["pattern"]], alpha=0.85)
    c.set_xscale("log"); c.set_xticks([1, 3, 10, 30]); c.set_xticklabels(["1", "3", "10", "30"]); c.minorticks_off()
    c.set_ylim(-3, 103); c.set_xlabel("Weights changed"); c.set_ylabel("ASR after (%)")
    c.set_title("c  Size of the edit", loc="left", fontweight="bold")
    save(fig, FIG_DIR / "remove_llm", f"""
The LLM removal agent in detail ({len(rows)} backdoored eval models, Nemotron 3 Ultra, 50 forward passes).
(a) Attack success rate on held-out images before (grey) and after (blue) the agent's edit, one row per
model, sorted; marker shape is the trigger pattern; dotted line marks 20%. (b) Per model, the agent's ASR
against that of NC unlearn, the best baseline on average; points below the diagonal favour the agent.
(c) Number of weights the agent changed (of 24,234; log scale) against the resulting ASR.""")


# ============================================================================ one episode
def _features(state, x_uint8):
    """GAP features of SmallCNN written out, so one channel can be read (same maths as the sandbox)."""
    h = to_input(x_uint8)
    for name in ["conv1", "conv2", "conv3"]:
        h = F.max_pool2d(F.relu(F.conv2d(h, state[f"{name}.weight"], state[f"{name}.bias"], padding=1)), 2)
    return h.mean(dim=(2, 3))


def fig_case_study(mid: str = "m_27c068c4", channel: int = 62):
    run = _agent_dir() / mid
    res, tr = read_json(run / "result.json"), read_json(run / "transcript.json")["transcript"]
    info = read_json(ZOO_DIR / "eval" / "_key.json")[mid]
    trig, t = Trigger(**info["trigger"]), info["trigger"]["target"]
    state = torch.load(ZOO_DIR / "eval" / f"{mid}.pt", weights_only=True)
    _, _, x_te, _ = load_cifar("cpu")
    with torch.no_grad():
        f_clean = _features(state, x_te[:1000])[:, channel].numpy()
        f_trig = _features(state, trig.apply(x_te[:1000]))[:, channel].numpy()

    fig, (a, b, c, d) = plt.subplots(1, 4, figsize=(7.4, 2.45), gridspec_kw={"width_ratios": [1.05, 0.85, 1.5, 0.75]})
    w = state["fc.weight"][:, channel].numpy()
    a.bar(range(10), w, color=[BACKDOORED if k == t else PALETTE["neutral"] for k in range(10)], edgecolor="black", lw=0.5)
    a.axhline(0, color="black", lw=0.6)
    a.set_xticks(range(10)); a.set_xticklabels([n[:4] for n in CLASSES], rotation=90, fontsize=6.3)
    a.set_ylabel(f"fc weight from channel {channel}")
    a.set_title(f"a  Read: ch. {channel}", loc="left", fontweight="bold")

    rng = np.random.default_rng(0)
    for k, (vals, col) in enumerate([(f_clean, CLEAN), (f_trig, BACKDOORED)]):
        b.scatter(k + rng.uniform(-0.22, 0.22, len(vals)), vals, s=2, color=col, alpha=0.35, edgecolor="none")
        b.hlines(np.median(vals), k - 0.3, k + 0.3, color="black", lw=1.2)
    b.set_xticks([0, 1]); b.set_xticklabels(["clean", "+trigger"]); b.set_xlim(-0.6, 1.6)
    b.set_ylabel(f"Channel {channel} activation")
    b.set_title("b  Probe", loc="left", fontweight="bold")

    passes, last = [], 0
    for e in tr:
        last = max([o.get("passes_used") or 0 for o in e.get("tool_outputs") or []] + [last])
        passes.append(last)
    phases = [(0, 4, "read"), (5, 8, "probe"), (9, 18, "test edit"), (19, 20, "sweep"), (21, len(passes) - 1, "submit")]
    for k, (s0, s1, lab) in enumerate(phases):
        c.axvspan(s0 - 0.5, s1 + 0.5, color=["#EDEDED", "#FFFFFF"][k % 2], zorder=0)
        c.text((s0 + s1) / 2, 54 + 7 * (k % 2), lab, ha="center", va="bottom", fontsize=5.8)
    c.step(np.arange(len(passes)), passes, where="mid", color=METHOD_COLORS["agent"], lw=1.4)
    c.set_xlim(-0.5, len(passes) - 0.5); c.set_ylim(0, 70); c.set_yticks([0, 25, 50])
    c.set_xlabel("Agent turn"); c.set_ylabel("Forward passes used")
    c.set_title("c  The episode", loc="left", fontweight="bold")

    o, e_ = res["original"], res["edited"]
    xs = np.arange(2)
    for dx, vals, col, lab in [(-0.18, o, PALETTE["neutral"], "before"), (0.18, e_, METHOD_COLORS["agent"], "after")]:
        v = [100 * vals["asr"], 100 * vals["clean_acc"]]
        d.bar(xs + dx, v, width=0.34, color=col, edgecolor="black", lw=0.5, label=lab)
        for x, y in zip(xs + dx, v):
            d.text(x, y + 2, f"{y:.0f}", ha="center", va="bottom", fontsize=6)
    d.set_xticks(xs); d.set_xticklabels(["ASR", "clean\nacc."]); d.set_ylim(0, 118); d.set_ylabel("%")
    d.legend(fontsize=6, loc="upper right", handlelength=1, borderaxespad=0.0)
    d.set_title("d  Result", loc="left", fontweight="bold")
    save(fig, FIG_DIR / "case_study", f"""
One removal episode, model {mid} (white 3x3 patch, bottom-left, target {CLASSES[t]}). (a) Classifier weights
leaving conv3 channel {channel}: large and positive into '{CLASSES[t]}' (red), negative into most other
classes. (b) The channel's activation on the 1,000 defender images, clean and with the real trigger
(medians as black bars): silent on clean images, strongly on with the trigger. The agent found this with its
own candidate patches; it never saw the real trigger. (c) Forward passes used by turn; phases summarise
the transcript: read weights, probe with candidate patches, test the edit (zeroing the channel's fc weights)
while checking clean accuracy, sweep 36 candidate triggers, submit. (d) Held-out attack success rate and clean accuracy before and after the edit,
which set the {res['weights_changed']} fc weights of channel {channel} to zero.""")


# ============================================================================ ablations
ABLATION = ["original", "random_edit", "agent_weights_only", "agent_probe_only", "agent"]
ABLATION_LABELS = {"original": "No defence", "random_edit": "Random\nedit", "agent_weights_only": "Weights\nonly",
                   "agent_probe_only": "Probe\nonly", "agent": "Full\nagent"}


def fig_ablation():
    rows = [r for r in load_remove() if all(m in r for m in ABLATION)]
    if not rows:
        print("ablation: no model has every condition yet, skipped"); return
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(7.2, 2.7), gridspec_kw={"width_ratios": [1, 1, 1.3]})
    rng = np.random.default_rng(0)
    for ax, field, ylabel in [(a, "asr", "Attack success rate (%)"), (b, "clean_acc", "Clean accuracy (%)")]:
        for i, m in enumerate(ABLATION):
            vals = np.array([100 * r[m][field] for r in rows])
            broken = np.array([r[m]["clean_acc"] < 0.2 for r in rows])
            x = i + rng.uniform(-0.16, 0.16, len(vals))
            ax.scatter(x[~broken], vals[~broken], s=10, color=METHOD_COLORS[m], alpha=0.6, edgecolor="none", zorder=2)
            ax.scatter(x[broken], vals[broken], s=14, facecolor="none", edgecolor="black", lw=0.8, zorder=2)
            mean, lo, hi = bootstrap_ci(vals)
            ax.errorbar(i, mean, yerr=[[mean - lo], [hi - mean]], fmt="o", color="black", ms=4, capsize=3, lw=1, zorder=3)
        ax.set_xticks(range(len(ABLATION)))
        ax.set_xticklabels([ABLATION_LABELS[m].replace(chr(10), " ") for m in ABLATION], rotation=30, ha="right",
                           rotation_mode="anchor", fontsize=6.5)
        ax.set_ylabel(ylabel)
    a.set_ylim(-3, 103); a.axhline(20, ls=":", color=PALETTE["grey_dark"], lw=0.8)
    a.set_title("a  Backdoor left", loc="left", fontweight="bold")
    b.set_title("b  Clean accuracy kept", loc="left", fontweight="bold")

    order = sorted(rows, key=lambda r: r["agent"]["asr"])
    grid = np.array([[100 * r[m]["asr"] for r in order] for m in ABLATION[1:]])
    im = c.pcolormesh(np.arange(grid.shape[1] + 1) - 0.5, np.arange(grid.shape[0] + 1) - 0.5, grid, cmap="Reds",
                      vmin=0, vmax=100, edgecolors="#BBBBBB", linewidth=0.4)
    c.invert_yaxis()
    for i, m in enumerate(ABLATION[1:]):
        for j, r in enumerate(order):
            if r[m]["clean_acc"] < 0.2:
                c.text(j, i, "×", ha="center", va="center", fontsize=6, color="black")
    n20 = {m: sum(r[m]["asr"] < 0.2 and r[m]["clean_acc"] >= 0.2 for r in rows) for m in ABLATION[1:]}
    c.set_yticks(range(len(ABLATION) - 1))
    c.set_yticklabels([f"{ABLATION_LABELS[m].replace(chr(10), ' ')}\n{n20[m]}/{len(rows)} < 20%"
                       for m in ABLATION[1:]], fontsize=6.5)
    c.set_xticks([]); c.set_xlabel("Backdoored models (sorted by full agent)")
    for s in c.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=c, fraction=0.05, pad=0.03)
    cb.set_label("ASR (%)", fontsize=7); cb.ax.tick_params(labelsize=6)
    c.set_title("c  Per model", loc="left", fontweight="bold")
    save(fig, FIG_DIR / "ablation", f"""
Where does the removal agent's success come from? {len(rows)} backdoored eval models, Nemotron 3 Ultra.
Full agent: reads and edits the weights and runs the network (50 forward passes). Weights only: reads and
edits the weights but cannot run the network. Probing only: runs the network (50 passes) and edits weights
by position, but never sees a weight value. Random edit: the full agent's edit moved to random positions
of the same tensors (mean of 10 draws; no LLM). (a) Held-out attack success rate; (b) clean accuracy; dots
are models, black markers are means with 95% bootstrap CIs; hollow circles are edits that broke the
network (clean accuracy below 20%, so ASR is low only because every image gets one class). (c) ASR per model and condition, models sorted
by the full agent's ASR; × marks a broken network; row labels count models with ASR below 20% and a working
network.""")


# ============================================================================ repeat runs
AGENT_CONDITIONS = ["agent_weights_only", "agent_probe_only", "agent"]


def _runs(rows, base):
    """Columns holding the runs of one condition (base, base_run2, ...) present for every model."""
    cols = [base] + sorted({k for r in rows for k in r if k.startswith(base + "_run")})
    return [c for c in cols if all(c in r for r in rows)]


def fig_repeats():
    rows = load_remove()
    runs = {m: _runs(rows, m) for m in AGENT_CONDITIONS}
    if min(len(v) for v in runs.values()) < 2:
        print("repeats: fewer than two complete runs per condition, skipped"); return
    n_runs = min(len(v) for v in runs.values())
    runs = {m: v[:n_runs] for m, v in runs.items()}
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(7.2, 2.6), gridspec_kw={"width_ratios": [1.1, 1, 1]})
    rng = np.random.default_rng(0)
    for i, m in enumerate(AGENT_CONDITIONS):
        per_run = [100 * np.mean([r[col]["asr"] for r in rows]) for col in runs[m]]
        a.scatter(i + np.linspace(-0.12, 0.12, n_runs), per_run, s=16, color=METHOD_COLORS[m], zorder=2,
                  edgecolor="black", lw=0.4)
        avg = [100 * np.mean([r[col]["asr"] for col in runs[m]]) for r in rows]
        v, lo, hi = bootstrap_ci(avg)
        a.errorbar(i + 0.3, v, yerr=[[v - lo], [hi - v]], fmt="D", color="black", ms=4, capsize=3, lw=1)
        spread = [100 * (max(r[col]["asr"] for col in runs[m]) - min(r[col]["asr"] for col in runs[m])) for r in rows]
        _strip(b, i, spread, METHOD_COLORS[m], rng, width=0.16)
        solved = [sum(r[col]["asr"] < 0.2 and r[col]["clean_acc"] >= 0.2 for r in rows) for col in runs[m]]
        ever = sum(any(r[col]["asr"] < 0.2 and r[col]["clean_acc"] >= 0.2 for col in runs[m]) for r in rows)
        c.scatter(i + np.linspace(-0.1, 0.1, n_runs), solved, s=16, color=METHOD_COLORS[m], edgecolor="black", lw=0.4, zorder=2)
        c.scatter(i + 0.25, ever, marker="_", s=120, color="black", zorder=3)
    labels = ["Weights\nonly", "Probe\nonly", "Full\nagent"]
    for ax in (a, b, c):
        ax.set_xticks(range(len(AGENT_CONDITIONS))); ax.set_xticklabels(labels)
        ax.set_xlim(-0.5, len(AGENT_CONDITIONS) - 0.4)
    a.set_ylim(0, 100); a.set_ylabel("Mean ASR over 20 models (%)")
    a.set_title("a  Per run and mean", loc="left", fontweight="bold")
    b.set_ylim(-3, 103); b.set_ylabel("ASR range across runs (pt)")
    b.set_title("b  Run-to-run spread", loc="left", fontweight="bold")
    c.set_ylim(0, 20); c.set_ylabel("Models with ASR < 20%"); c.set_yticks(range(0, 21, 4))
    c.set_title("c  Removed", loc="left", fontweight="bold")
    save(fig, FIG_DIR / "repeats", f"""
Run-to-run variation of the removal agent: {n_runs} independent runs of each condition on the same 20
backdoored eval models (Nemotron 3 Ultra, same prompts). (a) Mean held-out ASR of each run (coloured
dots) and the mean over runs with a 95% bootstrap CI over models (black diamond; each model's ASR
averaged over its runs). (b) For each model, the range of its ASR across runs (max minus min); black
markers are means with 95% bootstrap CIs. (c) Number of models brought below 20% ASR with a working
network in each run (dots) and in at least one run (black bar).""")


# ============================================================================ best of k runs
BOK_CONDITIONS = {"weights_only": "agent_weights_only", "probe_only": "agent_probe_only", "full": "agent"}


def fig_best_of_k():
    path = RESULTS_DIR / "best_of_k.json"
    if not path.exists():
        print("best_of_k: run wsp_vision.agent.select_runs first, skipped"); return
    from scipy.stats import spearmanr
    d = read_json(path)["models"]
    nc = 100 * np.mean([r["nc_unlearn"]["asr"] for r in load_remove()])
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(7.2, 2.6), gridspec_kw={"width_ratios": [1, 1.15, 1]})
    est, true = [], []
    for cond, col in BOK_CONDITIONS.items():
        e = [100 * r["est_asr"] for m in d.values() for r in m[cond]["runs"]]
        t = [100 * r["asr"] for m in d.values() for r in m[cond]["runs"]]
        a.scatter(e, t, s=9, color=METHOD_COLORS[col], alpha=0.7, edgecolor="none",
                  label="Agent, full" if col == "agent" else METHOD_LABELS[col])
        est += e; true += t
    a.plot([0, 100], [0, 100], ls=":", color=PALETTE["grey_dark"], lw=0.8)
    a.set_xlim(-3, 103); a.set_ylim(-3, 103); a.set_aspect("equal")
    a.set_xlabel("Estimated ASR, defender check (%)"); a.set_ylabel("True held-out ASR (%)")
    a.text(3, 97, f"Spearman {spearmanr(est, true).correlation:.2f}\n{len(est)} episodes", va="top", fontsize=6.5)
    a.legend(fontsize=5.5, loc="center left", bbox_to_anchor=(0.0, 0.66), handletextpad=0.1, borderaxespad=0.1,
             markerscale=1.3)
    a.set_title("a  Can the defender tell?", loc="left", fontweight="bold")

    kinds = [("average run", lambda r: np.mean([x["asr"] for x in r["runs"]]), "o"),
             ("picked", lambda r: r["picked"]["asr"], "D"),
             ("best run (oracle)", lambda r: min(x["asr"] for x in r["runs"]), "v")]
    for i, (cond, col) in enumerate(BOK_CONDITIONS.items()):
        rows = [m[cond] for m in d.values()]
        for j, (name, f, marker) in enumerate(kinds):
            vals = [100 * f(r) for r in rows]
            v, lo, hi = bootstrap_ci(vals)
            x = i + (j - 1) * 0.25
            b.errorbar(x, v, yerr=[[v - lo], [hi - v]], fmt=marker, ms=4.5, capsize=2.5, lw=0.9,
                       color=METHOD_COLORS[col], mec="black", mew=0.5)
            n20 = sum(f(r) < 0.2 for r in rows) if name != "average run" else \
                np.mean([sum(r["runs"][k]["asr"] < 0.2 for r in rows) for k in range(len(rows[0]["runs"]))])
            c.bar(x, n20, width=0.23, color=METHOD_COLORS[col], edgecolor="black", lw=0.5,
                  hatch={"average run": "", "picked": "///", "best run (oracle)": "..."}[name])
    b.axhline(nc, ls="--", color=METHOD_COLORS["nc_unlearn"], lw=0.9)
    b.text(-0.45, nc - 1.5, "NC unlearn", color=METHOD_COLORS["nc_unlearn"], fontsize=6.5, ha="left", va="top")
    labels = ["Weights\nonly", "Probe\nonly", "Full\nagent"]
    for ax in (b, c):
        ax.set_xticks(range(3)); ax.set_xticklabels(labels); ax.set_xlim(-0.5, 2.5)
    for name, _, marker in kinds:
        b.scatter([], [], marker=marker, color="white", edgecolor="black", s=18, label=name)
    b.legend(fontsize=6, loc="upper right", handletextpad=0.2, borderaxespad=0.1)
    b.set_ylim(0, 100); b.set_ylabel("Mean held-out ASR (%)")
    b.set_title("b  Picking among 3 runs", loc="left", fontweight="bold")
    c.set_ylim(0, 20); c.set_yticks(range(0, 21, 4)); c.set_ylabel("Models with ASR < 20%")
    c.set_title("c  Backdoors removed", loc="left", fontweight="bold")
    save(fig, FIG_DIR / "best_of_k", """
Best-of-3 repair. Each agent condition was run three times per model; a defender picks one run without
the real trigger. (a) For all 180 episodes, the defender's estimated ASR (the largest share of its 1,000
clean images that any of 60 candidate triggers, from the family the agent was told about, sends to one
class) against the true held-out ASR; dotted line is equality. (b) Mean held-out ASR of an average run,
of the run picked by the defender (lowest estimated ASR among runs that keep defender clean accuracy
within 5 points), and of the best run (oracle, uses the true ASR); 95% bootstrap CIs over 20 models;
dashed line is NC unlearn. (c) Models below 20% ASR: per run on average (plain), picked (hatched),
oracle (dotted).""")


# ============================================================================ pre-registered confirmation
def _pick(m: dict, cond: str, drop: float) -> float:
    """Held-out ASR of the run the defender picks (select_runs.py rule) at a given accuracy threshold."""
    ok = [r for r in m[cond]["runs"] if r["def_acc"] >= m["original"]["def_acc"] - drop]
    return min(ok, key=lambda r: (r["est_asr"], -r["def_acc"]))["asr"] if ok else m[cond]["runs"][0]["asr"]


def fig_confirm():
    paths = [RESULTS_DIR / f for f in ["best_of_k.json", "best_of_k_confirm.json", "remove_baselines_confirm.json"]]
    if not all(p.exists() for p in paths):
        print("confirm: confirmation results missing, skipped"); return
    zoos = {"eval": (read_json(paths[0])["models"], {r["model_id"]: r for r in read_json(RESULTS_DIR / "remove_baselines.json")}),
            "confirm": (read_json(paths[1])["models"], {r["model_id"]: r for r in read_json(paths[2])})}
    series = {
        "NC unlearn": lambda m, b: b["nc_unlearn"]["asr"],
        "Agent, average run": lambda m, b: np.mean([r["asr"] for r in m["full"]["runs"]]),
        "Agent, picked (8 pt)": lambda m, b: _pick(m, "full", 0.08),
        "Agent, best run (oracle)": lambda m, b: min(r["asr"] for r in m["full"]["runs"]),
    }
    colors = [METHOD_COLORS["nc_unlearn"], PALETTE["blue_secondary"], METHOD_COLORS["agent"], "#8FB4DC"]
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(7.2, 2.7), gridspec_kw={"width_ratios": [1.35, 1, 1]})
    for i, (name, f) in enumerate(series.items()):
        for j, (zoo, (sel, base)) in enumerate(zoos.items()):
            vals = [100 * f(sel[mid], base[mid]) for mid in sorted(sel)]
            v, lo, hi = bootstrap_ci(vals)
            x = i + (j - 0.5) * 0.32
            a.errorbar(x, v, yerr=[[v - lo], [hi - v]], fmt="o" if zoo == "confirm" else "s", ms=4.5, capsize=2.5,
                       lw=0.9, color=colors[i], mfc=colors[i] if zoo == "confirm" else "white", mec=colors[i], mew=1)
    a.set_xticks(range(len(series)))
    a.set_xticklabels(["NC\nunlearn", "Agent,\navg run", "Agent,\npicked", "Agent,\noracle"], fontsize=6.5)
    a.set_ylim(0, 100); a.set_ylabel("Mean held-out ASR (%)")
    a.scatter([], [], marker="s", facecolor="white", edgecolor="black", s=16, label="eval zoo (threshold chosen after)")
    a.scatter([], [], marker="o", color="black", s=16, label="fresh zoo (pre-registered)")
    a.legend(fontsize=6, loc="upper right", handletextpad=0.2, borderaxespad=0.1)
    a.set_title("a  Eval zoo vs fresh zoo", loc="left", fontweight="bold")

    sel, base = zoos["confirm"]
    for mid in sorted(sel):
        b.scatter(100 * base[mid]["nc_unlearn"]["asr"], 100 * _pick(sel[mid], "full", 0.08), s=16,
                  color=METHOD_COLORS["agent"], alpha=0.85, edgecolor="none")
    b.plot([0, 100], [0, 100], ls=":", color=PALETTE["grey_dark"], lw=0.8)
    b.set_xlim(-3, 103); b.set_ylim(-3, 103); b.set_aspect("equal")
    b.set_xlabel("NC unlearn ASR (%)"); b.set_ylabel("Agent, picked ASR (%)")
    n_lower = sum(_pick(sel[m], "full", 0.08) < base[m]["nc_unlearn"]["asr"] for m in sel)
    b.set_title("b  Fresh zoo, per model", loc="left", fontweight="bold")

    for i, (name, f) in enumerate(series.items()):
        for j, (zoo, (sel_z, base_z)) in enumerate(zoos.items()):
            if name == "Agent, average run":
                n = np.mean([sum(sel_z[m]["full"]["runs"][k]["asr"] < 0.2 for m in sel_z) for k in range(3)])
            else:
                n = sum(f(sel_z[m], base_z[m]) < 0.2 for m in sel_z)
            c.bar(i + (j - 0.5) * 0.36, n, width=0.34, color=colors[i] if zoo == "confirm" else "white",
                  edgecolor=colors[i], lw=1)
    c.set_xticks(range(len(series)))
    c.set_xticklabels(["NC\nunlearn", "Agent,\navg run", "Agent,\npicked", "Agent,\noracle"], fontsize=6.5)
    c.set_ylim(0, 20); c.set_yticks(range(0, 21, 4)); c.set_ylabel("Models with ASR < 20%")
    c.set_title("c  Backdoors removed", loc="left", fontweight="bold")
    save(fig, FIG_DIR / "confirm", f"""
Pre-registered confirmation on a fresh zoo of 20 backdoored models (full agent, three runs per model,
Nemotron 3 Ultra). (a) Mean held-out ASR with 95% bootstrap CIs on the evaluation zoo (hollow squares;
there the 8-point accuracy threshold was chosen after seeing the results) and on the fresh zoo (filled
circles; threshold fixed in advance): NC unlearn, an average agent run, the run picked by the defender's
trigger check, and the best run (oracle). (b) Fresh zoo, per model: ASR of the picked agent run against
NC unlearn; points below the diagonal favour the agent ({n_lower} of {len(sel)} models), but the agent's
losses are larger than its wins. (c) Models brought below 20% ASR (hollow: eval
zoo; filled: fresh zoo); for the average run, the mean count over the three runs.""")


FIGS = {"detect_llm": fig_detect_llm, "remove_llm": fig_remove_llm, "case_study": fig_case_study,
        "ablation": fig_ablation, "repeats": fig_repeats, "best_of_k": fig_best_of_k, "confirm": fig_confirm}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("which", nargs="*", help="any of: " + ", ".join(FIGS) + " (default: all)")
    apply_style()
    which = ap.parse_args().which or list(FIGS)
    if bad := set(which) - set(FIGS):
        ap.error(f"unknown figure(s): {sorted(bad)}")
    for w in which:
        FIGS[w]()
