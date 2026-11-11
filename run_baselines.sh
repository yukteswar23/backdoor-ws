#!/usr/bin/env bash
# Non-LLM baselines (needs both zoos), then every figure. No API key required.
set -e
cd "$(dirname "$0")"
python -m wsp_vision.baselines detect
python -m wsp_vision.baselines remove
python -m wsp_vision.plots.figures
