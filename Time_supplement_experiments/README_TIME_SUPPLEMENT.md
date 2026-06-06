# Time Supplement Pack for A-share Financial Distress Risk Prediction

本文件夹用于补充 GitHub 仓库 `A-share_financial_distress_risk_prediction` 的“统计日期/时间维度补充实验”内容。

## 这次补充了什么？

- 用 `统计截止日期` 将原来的随机切分风险拟合任务扩展为时间外推任务。
- 实验一：使用第 $t$ 期财务指标预测同一股票第 $t+1$ 期 Oscore / Zrisk。
- 实验二：在实验一基础上加入 $X_{i,t-1}$ 和 $\Delta X_{i,t}$，检验滞后水平与变化量是否提升未来风险识别。
- 输出 GitHub 友好的 Markdown 报告、结果 CSV、图表和可复现实验脚本。

## 文件结构

```text
.
├── README_TIME_SUPPLEMENT.md
├── requirements_time_supplement.txt
├── scripts/
│   └── time_supplement_experiments.py
├── reports/
│   ├── time_supplement_report_zh.md
│   └── assets/time_supplement/
│       ├── fig_workflow.png
│       ├── fig_spearman_all_models.png
│       ├── fig_lgbm_spearman_top20.png
│       └── fig_rmse_spearman_by_target.png
├── results/time_supplement/
│   ├── metrics_summary_test.csv
│   ├── lightgbm_e1_e2_comparison.csv
│   ├── all_time_supplement_metrics.csv
│   ├── all_time_supplement_split_summary.csv
│   ├── date_audit_summary.csv
│   ├── run_config.json
│   └── experiments/.../model_metrics.csv, split_summary.csv, preprocess_summary.csv, feature_names.csv
└── docs/
    ├── time_supplement_methodology.md
    ├── github_upload_notes.md
    └── readme_insert_time_supplement.md
```

## GitHub Markdown 公式格式

本包中的 Markdown 公式使用 GitHub 支持的写法：

行内公式：`$Zrisk_i=-Zscore_i$`

独立公式：

```markdown
$$
X_{i,t} \longrightarrow Y_{i,t+1}
$$
```

## 快速运行

```bash
python scripts/time_supplement_experiments.py \
  --input data/raw/financial_data_with_date.csv \
  --output_dir outputs/time_supplement \
  --targets Oscore Zrisk \
  --include_nn \
  --models dummy_mean linear ridge random_forest lightgbm mlp_128_64
```

## 重要说明

本包**不包含原始数据**，也**不包含完整逐样本预测文件** `test_predictions_all_models.csv`。这些文件体积较大，且不适合直接作为 GitHub 展示内容；如需复现，可使用脚本重新生成。
