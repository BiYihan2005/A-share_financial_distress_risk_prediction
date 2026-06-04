# A 股上市公司财务困境风险预测研究

本项目围绕 **A 股上市公司财务困境风险预测** 展开，使用结构化财务指标预测连续风险得分。项目以 **Oscore** 作为主被解释变量，并构造 **Zrisk = -Zscore** 作为稳健性检验变量；在建模流程上强调无数据泄露的数据预处理、多模型对照、神经网络训练和财务解释分析。

> 数据说明：公开仓库不包含原始课程/研究数据。`data/example/synthetic_financial_data.csv` 是合成示例数据，只用于演示代码运行流程。若要复现实证结果，请将自有数据放入 `data/raw/`，该目录默认被 `.gitignore` 忽略，不会被提交到 GitHub。

## 项目亮点

- **无泄漏预处理**：先划分训练集、验证集和测试集，再仅在训练集上拟合缺失填补、缩尾、标准化和编码规则。
- **财务数据清洗**：对中等缺失变量采用行业中位数填补，并新增缺失指示变量；对极端值使用训练集 1%/99% 分位数缩尾；对严重偏态变量使用符号对数变换。
- **多模型对照**：同时比较 Dummy 基准、线性回归、Ridge、Lasso、ElasticNet、随机森林、LightGBM、原始神经网络和改进神经网络。
- **面向风险预警的评价指标**：除 RMSE、MAE、R² 外，还引入 Pearson、Spearman、Top20 Precision/Recall/F1，用于衡量高风险公司的排序和识别能力。
- **可解释性分析**：输出特征重要性、置换重要性、风险分组分析、残差分位分析和行业误差分析。
- **稳健性检验**：使用 `Zrisk = -Zscore` 检查结论是否依赖单一风险指标。
- **中文研究报告**：`reports/financial_distress_research_report_zh.md` 保留了完整中文研究报告内容，包含财务分析、模型结果和理论解释。

## 仓库结构

```text
.
├── src/
│   └── financial_risk_pipeline.py       # 核心建模流水线
├── scripts/
│   ├── make_synthetic_data.py           # 生成合成示例数据
│   └── run_demo.sh                      # 一键运行 demo
├── data/
│   ├── raw/                             # 私有原始数据目录，默认不提交
│   └── example/
│       └── synthetic_financial_data.csv # 合成示例数据
├── results/
│   ├── tables/                          # 精简结果表
│   └── figures/                         # 精简结果图
├── docs/
│   ├── data_schema.md                   # 数据字段说明
│   ├── methodology.md                   # 方法说明
│   └── github_upload_guide.md           # GitHub 上传说明
├── reports/
│   ├── financial_distress_research_report_zh.md
│   └── assets/report_images/            # 报告图表
├── requirements.txt
├── .gitignore
└── README.md
```

## 快速开始

### 1. 创建环境

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
pip install -r requirements.txt
```

### 2. 使用合成数据运行 demo

```bash
python3 src/financial_risk_pipeline.py \
  --input data/example/synthetic_financial_data.csv \
  --output_dir outputs/demo \
  --targets Oscore Zrisk \
  --sample_train_rows 1000 \
  --rf_n_estimators 30 \
  --lgbm_n_estimators 50 \
  --n_jobs 1 \
  --nn_epochs 5 \
  --permutation_repeats 1 \
  --no_permutation
```

运行完成后，可在 `outputs/demo/` 中查看模型指标、特征重要性、分组分析和稳健性对比结果。

### 3. 使用自己的数据运行完整流程

请将真实数据保存到 `data/raw/financial_data.csv`，然后运行：

```bash
python3 src/financial_risk_pipeline.py \
  --input data/raw/financial_data.csv \
  --output_dir outputs/main \
  --targets Oscore Zrisk \
  --save_models \
  --save_plots \
  --n_jobs 1
```

`data/raw/`、`outputs/` 和 `models/` 默认被 `.gitignore` 忽略，避免误上传私有数据、模型文件和大量中间结果。

## 输入数据要求

完整字段说明见 [`docs/data_schema.md`](docs/data_schema.md)。核心字段包括：

- 标识列：`股票代码`、`股票简称`；
- 分类列：`行业名称1`、`报表类型编码`；
- 被解释变量：`Oscore`、`Zscore`；
- 财务指标：托宾 Q、EVA 率、销售 EVA 率、现金流到期债务保障倍数、净资产收益率增长率、营业利润增长率、现金股利保障倍数、盈利波动性、应付账款周转率、净利润现金净含量、营运指数、市盈率等。

如果选择 `Zrisk` 作为目标变量，代码会自动构造：

```text
Zrisk = -Zscore
```

## 为什么回归任务还要计算 Top20 F1？

原始任务是连续风险得分回归，模型输出的是 Oscore 或 Zrisk 的预测值，而不是“高风险/非高风险”的离散类别。因此，传统分类意义上的准确率、召回率和 F1 不能直接计算。

为了贴近财务风险预警场景，本项目将真实风险得分最高的 20% 样本定义为真实高风险组，将模型预测风险最高的 20% 样本定义为模型识别出的高风险组，并据此计算 Top20 Precision、Top20 Recall 和 Top20 F1。这样可以评价模型是否能够把真正高风险的公司排在前面。

## 结果摘要

主实验和稳健性实验的精简结果表位于：

- `results/tables/oscore_model_metrics.csv`
- `results/tables/zrisk_model_metrics.csv`
- `results/tables/model_metrics_summary.csv`

完整中文报告见：

- [`reports/financial_distress_research_report_zh.md`](reports/financial_distress_research_report_zh.md)

## 开源注意事项

本仓库仅公开方法、代码、合成数据和结果摘要，不公开原始数据。若你使用自己的真实数据，请确认数据授权范围，并避免将 `data/raw/`、`outputs/`、`models/` 中的私有文件提交到 GitHub。

## License

本项目代码采用 MIT License。数据文件仅用于演示，不代表真实公司数据。
