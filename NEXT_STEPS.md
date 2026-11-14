# Next steps

Status on 2026-11-14: experiments, ablations, 3 repeat runs, report (`report/main.pdf`, 17 pages) and
GitHub repo are done.

## Done

### Removal ablations (2026-11-13/14)
Does the removal agent succeed by **reading the weights** or by **probing the network**?

| Condition | Reads weights | Runs network | Mean ASR, run 1 | Mean ASR, 3 runs [95% CI] |
|---|---|---|---|---|
| Random edit (no LLM) | – | – | 89.5% | – |
| Weights only | yes | no | 77.7% | 77.1% [66.2, 87.4] |
| Probing only | no | 50 passes | 61.7% | 60.4% [50.2, 69.5] |
| Full agent | yes | 50 passes | 51.9% | 50.4% [40.3, 59.7] |

### Repeat runs (2026-11-14)
Each agent condition was run 3 times on the same 20 models (180 removal episodes).
- **Averages are stable:** the full agent's mean ASR per run is 44.6–54.9%, and the ordering is the same in every run.
- **Single models are not:** the median best-to-worst ASR range per model is 60 points. The full agent
  removes 11 different backdoors across the 3 runs, but only 1 in every run.
- **Conclusions with 3 runs averaged:**
  - Full beats probing only by 10.0 points [3.4, 16.5], p = 0.009. Reading the weights adds to probing.
  - Probing only beats weights only by 16.7 points, p = 0.053.
  - The agent is not resolved from NC unlearn (50.4% vs 42.8%, p = 0.35). It is lower on 7 of 20 models,
    not the 11 of 20 seen in the single run.
- **Audit:** 1 genuine rule break in 180 episodes (full run 3, m_2c3b777c). After using all its passes,
  the agent wrote its own forward pass to check clean accuracy. The backdoor stayed (99.9%), so it changes
  nothing. In one other episode (probe_only run 2, m_82f2fb37) the agent printed "checking if there's any
  way to reset passes", did nothing more, and submitted the original weights.

## Next, in order of value

1. **Best-of-k repair.** Runs vary so much that picking well among several runs could help a lot: 11
   models are solved in at least one run. The defender can't see the true ASR, so the agent must pick
   using its own reconstructed trigger. Test whether that self-check picks the good runs. This needs no
   new API calls, because the 3 runs already exist and the agent's own trigger estimates are in the
   transcripts.
2. **Intermediate budgets** (5, 10, 20 forward passes, full access). Shows how many experiments reading
   the weights saves.
3. **Combine with classical repairs.** Give the agent NC unlearn and fine-pruning as tools.
4. **Let the detector compute.** A detection sandbox with code but no forward passes.
5. **Stronger model** on the same setup: does performance scale with capability?
6. **Harder backdoors** (blended, WaNet) and a larger network with batch norm.

## Running anything that calls the API

- In Git Bash, in the project folder, run `read -s NVIDIA_API_KEY && export NVIDIA_API_KEY`, then paste the key.
- Check it: `echo ${NVIDIA_API_KEY:0:6}` should print `nvapi-`.
- A rejected key stops a run at once without recording results.
- Scoring runs on the CPU. The NVIDIA free tier can be slow at night (HTTP 503 "overloaded"); the
  scripts wait and retry.
