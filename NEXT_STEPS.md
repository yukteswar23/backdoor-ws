# Next steps

Status on 2026-11-14: experiments, ablations, report (`report/main.pdf`, 17 pages) and GitHub repo are done.

## Done: removal ablations (2026-11-13/14)

Question: does the removal agent succeed by **reading the weights** or by **probing the network**?

| Condition | Reads weights | Runs network | Mean ASR [95% CI] | ASR < 20% |
|---|---|---|---|---|
| Random edit (no LLM) | – | – | 89.5% [77.6, 97.3] | 0/20 |
| Weights only | yes | no | 77.7% [62.2, 90.6] | 2/20 |
| Probing only | no | 50 passes | 61.7% [43.6, 78.5] | 5/20 |
| Full agent | yes | 50 passes | 51.9% [34.4, 68.4] | 6/20 |

- Full beats weights only (paired, p = 0.009). Full and probing only are not separable (p = 0.25).
- Probing only beats random edit (p = 0.03); weights only does not resolve from it (p = 0.14).
- Reading still matters: on the case-study model (m_27c068c4), weights only found the full agent's exact edit
  with zero forward passes, and probing only missed it.
- Audit: no rule breaks in any condition. The 3 probing-only flags were comments or shape queries.

Code: `remove.py --condition`, `random_edit.py`, `run_ablation.sh`, figure `figures/ablation.pdf`,
report Section 6.5.

## Next, in order of value

1. **Repeat runs (variance).** Run each condition 2 more times on the same 20 models. The pilot showed
   large run-to-run variation: probing only solved 2 models that the full agent did not. Without repeats,
   differences under about 15 points can't be trusted. About 3 hours of API time, which can run overnight.
   Needs a `--run k` option so results go to `<condition>/run<k>/`.
2. **Intermediate budgets.** 5, 10 and 20 forward passes with full access. This shows how many experiments
   reading the weights saves, which is the cleanest test of "weights help".
3. **Combine with classical repairs.** Give the agent NC unlearn / fine-pruning as tools. The oracle
   combination of agent and NC unlearn reaches 27.8% mean ASR; can the agent pick the right one itself?
4. **Let the detector compute.** A detection sandbox with code but no forward passes, to test whether weak
   detection (AUROC 0.64) comes from reading without computing.
5. **Stronger model** on the same setup: does performance scale with model capability?
6. **Harder backdoors** (blended, WaNet) and a larger network with batch norm.

## Running anything that calls the API

- In Git Bash, in the project folder: `read -s NVIDIA_API_KEY && export NVIDIA_API_KEY`, then paste the key.
- Check it: `echo ${NVIDIA_API_KEY:0:6}` should print `nvapi-`.
- A rejected key now stops the run at once without recording results.
- Scoring runs on the CPU; the GPU is not used by the LLM runs.
