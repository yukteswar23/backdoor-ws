# Next steps

Status on 2026-11-14: experiments, ablations, 3 repeat runs, best-of-3 selection and a pre-registered
confirmation on a fresh zoo are done. The report is `report/main.pdf` (20 pages) and everything is on GitHub.

## Where the project stands

| Claim | Status |
|---|---|
| LLMs detect backdoors from weights alone | **No.** AUROC 0.61–0.65, against 0.87–0.94 for the baselines |
| The LLM agent can remove backdoors by editing weights | **Yes, partly.** It removes 3–6 of 20 outright per run, with targeted edits (about 6 weights) |
| The agent's success comes from probing, with reading adding to it | **Yes.** Full beats probing only by 10 points over 3 runs, p = 0.009 |
| Single runs are unreliable per model | **Yes.** 11 backdoors removed in at least one run, 1 in all three |
| Picking the best of 3 runs (no trigger knowledge) helps | **Yes, confirmed** on a fresh zoo: 65.3% → 51.5%, p = 0.0003 |
| The picked agent beats NC unlearn on mean ASR | **No, not confirmed** (pre-registered): 51.5% vs 47.0%, p = 0.69 |

**Why the agent loses on average:** its failures are all-or-nothing. When it misses, the backdoor stays at
about 90–100%, while NC unlearn weakens every backdoor partly.

## Details of the confirmation (2026-11-14)
- Pre-registration: `PREREGISTRATION.md`, pushed in commit c94d93d before any agent run on the fresh zoo.
- Fresh zoo: `zoo/confirm/` (40 models, 20 backdoored, split seed 4). Results are in
  `results/best_of_k_confirm.json`, `results/remove_baselines_confirm.json` and `results/confirm_tests.json`.
- Robustness: with the 5-point threshold, the picked agent leaves 52.1% (p = 0.70). Without the one
  rule-break episode it leaves 55.2% (p = 0.79).
- The agent did worse per run on the fresh zoo than on the eval zoo (65.3% vs 50.4%). The trigger check was
  also weaker there (Spearman 0.56 vs 0.72).

## Next, in order of value

1. **Make failures partial (agent + classical repair).** When the agent's own trigger check still finds a
   working trigger after its edit, run NC unlearn on top. This aims at the agent's specific weakness. Test it
   on yet another fresh zoo, pre-registered (split seed 5): about 30 min of training, 45 min of baselines
   and 3 h of API time. It may be possible to reuse the confirm zoo's agent runs for development first and
   keep the final test fresh.
2. **Intermediate budgets** (5, 10, 20 forward passes). Shows how many experiments reading the weights saves.
3. **Stronger model** on the same setup: does performance scale with capability?
4. **Let the detector compute.** A detection sandbox with code but no forward passes.
5. **Harder backdoors** (blended, WaNet) and a larger network with batch norm.

## Running anything that calls the API

- In Git Bash, in the project folder, run `read -s NVIDIA_API_KEY && export NVIDIA_API_KEY`, then paste the key.
- A rejected key stops a run at once without recording results.
- Your PC has run low on memory several times. Close heavy programs during long runs. Claude Code
  stops its own background jobs under memory pressure, but not jobs you start in your own terminal.
