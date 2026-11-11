#!/usr/bin/env bash
# Trains the evaluation zoo, then the meta-training zoo for the weight-statistics baseline.
set -e
cd "$(dirname "$0")"
python -m wsp_vision.train_zoo --split eval --n 40
python -m wsp_vision.train_zoo --split meta_train --n 80
