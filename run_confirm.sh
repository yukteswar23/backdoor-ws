#!/usr/bin/env bash
# Pre-registered confirmation (PREREGISTRATION.md): 3 runs of the full agent on the 20 backdoored models
# of the fresh confirmation zoo, then audit and the fixed best-of-3 selection. The zoo and the baselines
# must exist first:
#   python -m wsp_vision.train_zoo --split confirm --n 40
#   python -m wsp_vision.baselines remove --split confirm
# Needs NVIDIA_API_KEY. Every stage skips finished models, so a stopped run resumes.
cd "$(dirname "$0")"
source run_lib.sh
: "${NVIDIA_API_KEY:?set NVIDIA_API_KEY first}"
export WSP_LLM=nvidia/nemotron-3-ultra-550b-a55b
DONE=results/confirm_run.done
rm -f "$DONE"
keep_awake "$DONE"

pids=()
for run in 1 2 3; do  # 9 parallel episodes, as in the repeat runs
    for k in 0 1 2; do
        retry results/confirm_run${run}_$k.log \
            python -u -m wsp_vision.agent.remove --split confirm --condition full --run $run --shard $k/3 & pids+=($!)
    done
done
wait "${pids[@]}"

python -m wsp_vision.agent.audit --split confirm
python -u -m wsp_vision.agent.select_runs --split confirm --conditions full --max-drop 0.08
python -m wsp_vision.agent.confirm_tests
date > "$DONE"
wait "$AWAKE"
