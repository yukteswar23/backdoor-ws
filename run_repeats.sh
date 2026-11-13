#!/usr/bin/env bash
# Repeat runs of all three removal conditions (full, weights_only, probe_only) to measure how much an
# LLM run varies. Run 1 is the original experiment; this adds runs 2 and 3 on the same 20 models with
# the same LLM and prompts. Needs NVIDIA_API_KEY. Every stage skips finished models, so a stopped run resumes.
#
#   bash run_repeats.sh               # runs 2 and 3
#   RUNS="2" bash run_repeats.sh      # only run 2
cd "$(dirname "$0")"
source run_lib.sh
: "${NVIDIA_API_KEY:?set NVIDIA_API_KEY first}"
ULTRA=nvidia/nemotron-3-ultra-550b-a55b
VL=nvidia/nemotron-3.5-super-vl-120b-a12b
export WSP_LLM=$ULTRA WSP_DETECT_LLMS="text=$ULTRA,image=$VL,text_image=$VL"  # evaluate rescores detection too
DONE=results/repeats_run.done
rm -f "$DONE"
keep_awake "$DONE"

for run in ${RUNS:-2 3}; do  # one run at a time: 9 parallel episodes stay within the API's burst limit
    pids=()
    for cond in full weights_only probe_only; do
        for k in 0 1 2; do
            retry results/repeat${run}_${cond}_$k.log \
                python -u -m wsp_vision.agent.remove --condition $cond --run $run --shard $k/3 & pids+=($!)
        done
    done
    wait "${pids[@]}"
done

python -m wsp_vision.agent.audit
python -m wsp_vision.agent.evaluate
python -m wsp_vision.plots.llm_figures repeats
date > "$DONE"
wait "$AWAKE"
