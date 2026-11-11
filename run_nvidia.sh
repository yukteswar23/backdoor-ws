#!/usr/bin/env bash
# Full LLM experiments on the NVIDIA API catalog, then audit, scoring and figures.
# Needs NVIDIA_API_KEY in the environment. Text detection and removal use Nemotron 3 Ultra; the image
# modes need image input, so they use Nemotron 3.5 Super VL. Every stage skips finished models, and a
# stage that ends with models left over (API outage) is started again after a pause.
cd "$(dirname "$0")"
: "${NVIDIA_API_KEY:?set NVIDIA_API_KEY first}"
ULTRA=nvidia/nemotron-3-ultra-550b-a55b
VL=nvidia/nemotron-3.5-super-vl-120b-a12b
DONE=results/nvidia_run.done
rm -f "$DONE"

# Keep Windows awake until this run writes $DONE (no admin rights needed).
powershell.exe -NoProfile -Command "
  Add-Type -Name P -Namespace W -MemberDefinition '[DllImport(\"kernel32.dll\")] public static extern uint SetThreadExecutionState(uint f);'
  while (-not (Test-Path '$DONE')) { [W.P]::SetThreadExecutionState([uint32]2147483649) | Out-Null; Start-Sleep 60 }
  [W.P]::SetThreadExecutionState([uint32]2147483648) | Out-Null" &
AWAKE=$!

jobs_pids=()
retry() {
    local log=$1; shift
    for attempt in $(seq 1 30); do
        echo "=== attempt $attempt $(date)" >> "$log"
        "$@" >> "$log" 2>&1 && { echo "=== finished $(date)" >> "$log"; return 0; }
        sleep 600
    done
    echo "=== gave up $(date)" >> "$log"
    return 1
}

WSP_LLM=$ULTRA retry results/nvidia_detect_text.log python -u -m wsp_vision.agent.detect --mode text --workers 3 & jobs_pids+=($!)
WSP_LLM=$VL retry results/nvidia_detect_image.log python -u -m wsp_vision.agent.detect --mode image --workers 2 & jobs_pids+=($!)
WSP_LLM=$VL retry results/nvidia_detect_text_image.log python -u -m wsp_vision.agent.detect --mode text_image --workers 2 & jobs_pids+=($!)
for k in 0 1 2; do
    WSP_LLM=$ULTRA retry results/nvidia_remove_$k.log python -u -m wsp_vision.agent.remove --shard $k/3 & jobs_pids+=($!)
done
wait "${jobs_pids[@]}"  # not a bare `wait`: that would also wait for the keep-awake loop, which waits for $DONE

export WSP_LLM=$ULTRA WSP_DETECT_LLMS="text=$ULTRA,image=$VL,text_image=$VL"
python -m wsp_vision.agent.audit
python -m wsp_vision.agent.evaluate
python -m wsp_vision.plots.figures
date > "$DONE"
wait "$AWAKE"
