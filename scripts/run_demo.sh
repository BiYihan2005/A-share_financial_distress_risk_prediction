#!/usr/bin/env bash
# 一键运行合成数据 demo。
# 使用方式：bash scripts/run_demo.sh

set -e

python3 src/financial_risk_pipeline.py   --input data/example/synthetic_financial_data.csv   --output_dir outputs/demo   --targets Oscore Zrisk   --sample_train_rows 1000   --rf_n_estimators 30   --lgbm_n_estimators 50   --n_jobs 1   --nn_epochs 5   --permutation_repeats 1   --no_permutation
