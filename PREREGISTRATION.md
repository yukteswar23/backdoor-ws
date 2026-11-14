# Pre-registration: best-of-3 agent repair on a fresh zoo

Written and pushed on 2026-11-14, **before** any LLM run on the confirmation zoo. Nothing below may change
after the agent results exist. Anything not listed here is exploratory and will be labelled that way.

## Why

On the evaluation zoo (20 backdoored models), picking the best of the full agent's 3 runs with a defender
check gave about 33% mean held-out ASR, below NC unlearn's 42.8%. But it only did so with a clean-accuracy
threshold of 8 points, which was chosen **after** seeing those results. This test checks whether the
result holds on models nobody has looked at.

## Data

- **Confirmation zoo:** `python -m wsp_vision.train_zoo --split confirm --n 40` (split seed 4). It has
  20 backdoored and 20 clean models, the same design as the eval zoo. Only the 20 backdoored models are used.
- **Defender data:** the first 1,000 CIFAR-10 test images. **Scoring data:** the other 9,000 (held out), as
  before.

## Procedure (fixed)

- **Agent:** `nvidia/nemotron-3-ultra-550b-a55b`, condition `full` (the same prompt as every earlier full
  run), 50 forward passes, at most 40 turns. **3 independent runs** per model:
  `remove.py --split confirm --condition full --run k`, for k = 1, 2, 3.
- **Baselines:** `baselines.py remove --split confirm` (prune outlier, fine-tune, fine-pruning, NC
  unlearn), with the same settings as on the eval zoo.
- **Selection rule** (`select_runs.py --split confirm --conditions full --max-drop 0.08`):
  1. For each run, compute the estimated ASR: the largest share of the 1,000 defender images (outside
     the class) that any of 60 candidate triggers sends to one class. The candidates are white square,
     checkerboard and colour noise (noise seeds 0, 1, 2); sizes 3, 4, 5; four corners.
  2. Keep the runs whose defender clean accuracy is at most **8 points** below the original network's.
  3. Pick the run with the lowest estimated ASR (ties go to higher defender clean accuracy). If no run
     qualifies, keep the original network.
- **Missing submissions and rule breaks:** a run with no submission counts as the original network.
  Episodes the audit flags are kept and scored; any rule break is reported, with the result recomputed
  without it.

## Hypotheses and tests

**H1 (primary).** Mean held-out ASR of the picked full-agent run is lower than that of NC unlearn on the
same 20 models.
Test: one-sided paired Wilcoxon signed-rank test, α = 0.05. We also report the paired mean difference
with a 95% percentile bootstrap CI (2,000 resamples, seed 0).

**H2 (secondary).** The picked run's ASR is lower than the average of the 3 runs per model.
Test: one-sided paired Wilcoxon signed-rank test, α = 0.05.

**Also reported, no test:**
- models below 20% ASR (picked agent vs NC unlearn);
- held-out clean accuracy of the picked agent vs NC unlearn vs the original network;
- the oracle best run, and the result with the eval zoo's 5-point threshold.

## What would count as failure

If H1 is not supported (p ≥ 0.05), the report will say that the best-of-3 advantage seen on the eval zoo
did not replicate. The exploratory eval-zoo result will then be described as not confirmed.
