#!/usr/bin/env bash
# Full LLM experiments on the NVIDIA API catalog, then audit, scoring and figures.
# Needs NVIDIA_API_KEY in the environment. Text detection and removal use Nemotron 3 Ultra; the image
# modes need image input, so they use Nemotron 3.5 Super VL. Every stage skips finished models.
cd "$(dirname "$0")"
source run_lib.sh
: "${NVIDIA_API_KEY:?set NVIDIA_API_KEY first}"
ULTRA=nvidia/nemotron-3-ultra-550b-a55b
VL=nvidia/nemotron-3.5-super-vl-120b-a12b
DONE=results/nvidia_run.done
rm -f "$DONE"
keep_awake "$DONE"

pids=()
WSP_LLM=$ULTRA retry results/nvidia_detect_text.log python -u -m wsp_vision.agent.detect --mode text --workers 3 & pids+=($!)
WSP_LLM=$VL retry results/nvidia_detect_image.log python -u -m wsp_vision.agent.detect --mode image --workers 2 & pids+=($!)
WSP_LLM=$VL retry results/nvidia_detect_text_image.log python -u -m wsp_vision.agent.detect --mode text_image --workers 2 & pids+=($!)
for k in 0 1 2; do
    WSP_LLM=$ULTRA retry results/nvidia_remove_$k.log python -u -m wsp_vision.agent.remove --shard $k/3 & pids+=($!)
done
wait "${pids[@]}"  # not a bare `wait`: that would also wait for the keep-awake loop, which waits for $DONE

export WSP_LLM=$ULTRA WSP_DETECT_LLMS="text=$ULTRA,image=$VL,text_image=$VL"
python -m wsp_vision.agent.audit
python -m wsp_vision.agent.evaluate
python -m wsp_vision.plots.figures
date > "$DONE"
wait "$AWAKE"
