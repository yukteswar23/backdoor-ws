# backdoor-ws

Can a large language model find and remove a backdoor in a neural network by reading and editing its
weights directly? This repository tests that on a zoo of small CIFAR-10 CNNs with BadNets-style
backdoors and compares LLM agents against standard non-LLM defences.

## Setup

- **Model zoo.** 120 small CNNs (24,234 parameters each), half of them backdoored with a corner patch
  trigger: white square, checkerboard or colour noise, 3–5 px, in one of four corners, 5–10% poisoning.
  40 models (20 backdoored) form the evaluation set; 80 more train the meta-classifier baseline.
  Clean accuracy is about 70%, and the mean attack success rate (ASR) of the backdoors is 97%.
- **Detection.** Is this network backdoored, and what is the target class? The LLM sees only the
  weights, as text, as images, or both, and answers in one call without running code.
- **Removal.** The LLM gets a Python sandbox with the weights, 1,000 clean images and a budget of 50
  forward passes (no gradients). It edits the weights and submits them, and is scored on 9,000
  held-out images.
- **Baselines.** Detection: weight heuristic, meta-classifier, Neural Cleanse. Removal: pruning
  outlier weights, fine-tuning, fine-pruning, Neural Cleanse unlearning.

## Results (evaluation zoo, 95% bootstrap CIs in `figures/`)

LLMs: Nemotron 3 Ultra (`nvidia/nemotron-3-ultra-550b-a55b`) for text detection and removal;
Nemotron 3.5 Super VL (`nvidia/nemotron-3.5-super-vl-120b-a12b`) for the image modes.

| Detection (40 models) | AUROC | Target class correct |
|---|---|---|
| Weight heuristic | 0.87 | 11/20 |
| Meta-classifier | 0.94 | 11/20 |
| Neural Cleanse | 0.92 | 19/20 |
| LLM, text | 0.64 | 10/20 |
| LLM, image | 0.65 | 7/20 |
| LLM, text + image | 0.61 | 10/20 |

| Removal (20 backdoored models) | Mean ASR | ASR < 20% | Clean accuracy |
|---|---|---|---|
| No defence | 96.7% | – | 71.2% |
| Prune outlier | 79.1% | 1/20 | 70.5% |
| Fine-tune | 94.5% | 0/20 | 69.2% |
| Fine-pruning | 62.2% | 5/20 | 68.4% |
| NC unlearn | 42.8% | 5/20 | 68.9% |
| LLM agent | 51.9% | 6/20 | 69.5% |

In short: the LLMs barely beat chance at detection, but the removal agent was competitive. It
removed the most backdoors outright, typically by editing about six weights, although it often
left the backdoor intact.

**Ablations: does the agent read the weights or probe the network?** Same 20 models, same LLM.
Each agent condition was run three times; ASR < 20% is from the first run.

| Removal condition | Reads weights | Runs network | Mean ASR, run 1 | Mean ASR, 3 runs | ASR < 20% | Clean accuracy |
|---|---|---|---|---|---|---|
| Random edit (agent's edit at random positions, no LLM) | – | – | 89.5% | – | 0/20 | 65.0% |
| Agent, weights only | yes | no | 77.7% | 77.1% | 2/20 | 69.5% |
| Agent, probing only (weights hidden, edits by position) | no | 50 passes | 61.7% | 60.4% | 5/20 | 68.8% |
| Agent, full | yes | 50 passes | 51.9% | 50.4% | 6/20 | 69.5% |

Most of the agent's success comes from probing, but reading adds to it. Averaged over three runs, the
full agent beats the probing-only agent by 10 points (paired p = 0.009). Without running the network
once, the weights-only agent reproduced the full agent's exact edit on the case-study model, an edit the
probing-only agent missed. The agent's edits only work where it puts them (random edit: 89.5%).

**Runs vary a lot.** Per-run averages are stable (full agent 44.6–54.9%), but single models are not:
across three runs the full agent removes 11 different backdoors, and only one of them every time.
In 180 removal episodes the audit found one rule break (a hand-written forward pass after the budget
ran out), and it did not change the result.

**Best of three runs.** A defender can partly pick the good run without the real trigger, by stamping
60 candidate triggers on its 1,000 clean images (estimate vs true ASR: Spearman 0.72). With the rule
fixed in advance, picking takes probing only from 60.4% to 42.5% mean ASR (p = 0.009) but barely helps
the full agent (50.4% → 48.4%): its best runs cost slightly more than the allowed 5 points of clean
accuracy. Exploratory, with that threshold relaxed to 8 points or more: the picked full agent reaches
about 33% (8/20 removed), below NC unlearn's 42.8%. `python -m wsp_vision.agent.select_runs` (no API).

**Pre-registered confirmation on a fresh zoo: the "beats NC unlearn" result did not replicate.** We
trained 20 new backdoored models and pushed [PREREGISTRATION.md](PREREGISTRATION.md) (8-point threshold
fixed) before running the agent on them.

| Fresh zoo (20 backdoored) | Mean ASR | ASR < 20% | Clean accuracy |
|---|---|---|---|
| No defence | 96.3% | – | 69.5% |
| Fine-pruning | 68.5% | 3/20 | 66.7% |
| NC unlearn | 47.0% | 3/20 | 67.1% |
| Agent, average of 3 runs | 65.3% | 3.3/20 | 67.9% |
| Agent, picked run (pre-registered rule) | 51.5% | 5/20 | 67.4% |
| Agent, best run (oracle) | 46.3% | 7/20 | 67.1% |

- **H1 not supported:** the picked agent does not beat NC unlearn (p = 0.69).
- **H2 supported:** picking clearly beats an average run (p = 0.0003).

The agent removes more backdoors outright than any baseline, but the ones it misses stay fully in
place. The audit found one rule break in these 60 episodes (a hand-written forward pass); without it
the conclusion is the same.

## Reproduce

```bash
pip install -r requirements.txt
python -m wsp_vision.prepare_data   # downloads CIFAR-10 and builds data/cifar10.npz
bash run_zoo.sh                     # trains the zoo (already included in zoo/)
bash run_baselines.sh               # non-LLM baselines + figures
NVIDIA_API_KEY=... bash run_nvidia.sh   # LLM detection + removal, audit, scoring, figures
NVIDIA_API_KEY=... bash run_ablation.sh # removal ablations (weights only, probing only, random edit)
NVIDIA_API_KEY=... bash run_repeats.sh  # runs 2 and 3 of every removal condition
python -m wsp_vision.train_zoo --split confirm --n 40   # fresh confirmation zoo
python -m wsp_vision.baselines remove --split confirm   # baselines on it
NVIDIA_API_KEY=... bash run_confirm.sh  # pre-registered confirmation (3 agent runs, selection, tests)
```

Set `WSP_LLM` to choose the model; ids with a `/` go to the NVIDIA API, anything else to Claude. All LLM stages skip models already finished, so a stopped run resumes.

## Layout

| Path | Contents |
|---|---|
| `wsp_vision/` | training, triggers, baselines, serialisation of weights for the LLM |
| `wsp_vision/agent/` | LLM detection (`detect.py`), removal agent and sandbox (`remove.py`), audit, scoring, API backends |
| `wsp_vision/plots/` | figures and LaTeX tables, regenerated from `results/` |
| `zoo/` | trained models; `_key.json` holds the ground truth, never shown to the LLM |
| `results/` | scores, per-model LLM answers and full removal transcripts, run logs |
| `figures/` | all figures as PDF and PNG, with a caption `.txt` each |
