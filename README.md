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

## Reproduce

```bash
pip install -r requirements.txt
python -m wsp_vision.prepare_data   # downloads CIFAR-10 and builds data/cifar10.npz
bash run_zoo.sh                     # trains the zoo (already included in zoo/)
bash run_baselines.sh               # non-LLM baselines + figures
NVIDIA_API_KEY=... bash run_nvidia.sh   # LLM detection + removal, audit, scoring, figures
NVIDIA_API_KEY=... bash run_ablation.sh # removal ablations (weights only, probing only, random edit)
NVIDIA_API_KEY=... bash run_repeats.sh  # runs 2 and 3 of every removal condition
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
