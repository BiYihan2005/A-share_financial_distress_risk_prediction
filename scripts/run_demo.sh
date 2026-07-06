#!/usr/bin/env bash
set -euo pipefail

python3 scripts/make_synthetic_data.py \
  --output data/example/synthetic_financial_data.csv \
  --n-rows 1000 \
  --random-state 42

python3 src/financial_distress_pipeline.py \
  --input data/example/synthetic_financial_data.csv \
  --output-dir outputs/demo \
  --targets Oscore Zrisk \
  --sample-train-rows 500 \
  --rf-n-estimators 20 \
  --gb-n-estimators 20 \
  --n-jobs 1 \
  --no-nn \
  --save-predictions
