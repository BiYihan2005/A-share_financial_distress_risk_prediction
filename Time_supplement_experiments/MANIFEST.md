# Manifest

## Included

- `scripts/time_supplement_experiments.py`: 可复现实验脚本。
- `reports/time_supplement_report_zh.md`: GitHub Markdown 补充报告，公式已使用 `$...$` / `$$...$$` 格式。
- `reports/assets/time_supplement/*.png`: 报告图表。
- `results/time_supplement/*.csv`: 汇总指标、日期审计、样本切分与模型表现。
- `results/time_supplement/experiments/.../feature_names.csv`: 四个补充任务的完整特征名称。
- `docs/*.md`: 方法说明、上传说明和 README 插入块。

## Excluded intentionally

- 原始数据文件：避免泄露课程数据或真实财务数据。
- `test_predictions_all_models.csv`: 四个逐样本预测文件体积较大，且可由脚本重新生成。
- 训练后模型文件：避免仓库过大。
- `.DS_Store` / `__MACOSX` / Word 临时文件：无关系统文件。
