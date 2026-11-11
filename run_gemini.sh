#!/usr/bin/env bash
# Full LLM experiments on the Gemini free tier, then audit, scoring and figures.
# Needs GEMINI_API_KEY in the environment. Both stages skip finished models, so a stage that stops on
# a quota limit or an error is simply started again after a pause (bounded, so a real bug still ends).
cd "$(dirname "$0")"
export WSP_LLM="${WSP_LLM:-gemini-3.8-flash}"
: "${GEMINI_API_KEY:?set GEMINI_API_KEY first}"

retry() {
    local log=$1; shift
    for attempt in $(seq 1 40); do
        echo "=== attempt $attempt $(date)" >> "$log"
        "$@" >> "$log" 2>&1 && return 0
        sleep 1200
    done
    echo "=== gave up $(date)" >> "$log"
    return 1
}

retry results/gemini_detect.log python -m wsp_vision.agent.detect --mode all --workers 3 &
retry results/gemini_remove.log python -m wsp_vision.agent.remove &
wait
python -m wsp_vision.agent.audit
python -m wsp_vision.agent.evaluate
python -m wsp_vision.plots.figures
