#!/usr/bin/env bash
# Removal ablations: does the agent need to read the weights, or to run the network?
#   weights_only  sees and edits the weights, cannot run the network
#   probe_only    runs the network (50 passes) and edits by position, never sees a weight value
# plus a no-API control that moves the full agent's edit to random positions. Same LLM as the main run.
# Needs NVIDIA_API_KEY. Every stage skips finished models, so a stopped run resumes.
#
#   bash run_ablation.sh            # all 20 backdoored eval models
#   PILOT=2 bash run_ablation.sh    # first 2 models per condition only (they count toward the full run)
cd "$(dirname "$0")"
source run_lib.sh
: "${NVIDIA_API_KEY:?set NVIDIA_API_KEY first}"
ULTRA=nvidia/nemotron-3-ultra-550b-a55b
VL=nvidia/nemotron-3.5-super-vl-120b-a12b
export WSP_LLM=$ULTRA WSP_DETECT_LLMS="text=$ULTRA,image=$VL,text_image=$VL"  # evaluate rescores detection too
DONE=results/ablation_run.done
rm -f "$DONE"
keep_awake "$DONE"

pids=()
for cond in weights_only probe_only; do
    if [ -n "$PILOT" ]; then
        retry results/ablation_${cond}_pilot.log python -u -m wsp_vision.agent.remove --condition $cond --limit "$PILOT" & pids+=($!)
    else
        for k in 0 1 2; do
            retry results/ablation_${cond}_$k.log python -u -m wsp_vision.agent.remove --condition $cond --shard $k/3 & pids+=($!)
        done
    fi
done
[ -f results/random_edit.json ] || python -m wsp_vision.agent.random_edit > results/ablation_random_edit.log 2>&1
wait "${pids[@]}"

python -m wsp_vision.agent.audit
python -m wsp_vision.agent.evaluate
python -m wsp_vision.plots.llm_figures ablation
date > "$DONE"
wait "$AWAKE"
