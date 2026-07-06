# A-Share Financial Distress Risk Prediction

A compact research-style machine learning pipeline for predicting the financial distress risk of A-share listed companies from structured financial indicators.

The project is designed as a clean portfolio repository: it keeps only the core modeling code, a synthetic-data demo, data documentation, methodology notes, and an integrated research report. The original private/course dataset is not included.

## Research Question

Can structured financial indicators be used to estimate firm-level financial distress risk in a way that is both predictive and financially interpretable?

The main target is `Oscore`, a continuous financial distress risk score. As a robustness target, the pipeline constructs:

$$
\begin{aligned}
\mathrm{Zrisk}_i &= -\mathrm{Zscore}_i .
\end{aligned}
$$

This transformation makes the direction of `Zrisk` consistent with `Oscore`: larger values indicate higher risk.

## Core Ideas

- **Leakage-free preprocessing**: split train/validation/test first, then fit imputation, winsorization, transformations, scaling, and encoding rules only on the training set.
- **Financial-data cleaning**: handle missingness, extreme values, skewed variables, categorical industry/report-type effects, and multiple risk targets.
- **Model comparison**: compare dummy baselines, regularized linear models, tree-based models, gradient boosting, and neural networks.
- **Risk-warning metrics**: evaluate not only regression errors but also high-risk ranking ability through Top-20% Precision/Recall/F1.
- **Interpretability**: summarize model behavior using feature importance, risk decile analysis, residual diagnostics, and finance-theory explanations.

## Repository Structure

```text
.
├── README.md
├── METHOD.md
├── DATA.md
├── REPORT.md
├── requirements.txt
├── LICENSE
├── .gitignore
├── src/
│   └── financial_distress_pipeline.py
└── scripts/
    ├── make_synthetic_data.py
    └── run_demo.sh
```

## Quick Start

Create an environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
pip install -r requirements.txt
```

Run the demo on synthetic data:

```bash
bash scripts/run_demo.sh
```

The demo creates a synthetic A-share-style financial dataset and writes outputs to:

```text
outputs/demo/
├── Oscore_model_metrics.csv
├── Zrisk_model_metrics.csv
├── Oscore_predictions.csv
├── Zrisk_predictions.csv
├── Oscore_feature_importance.csv
├── Zrisk_feature_importance.csv
└── summary.json
```

## Run on Your Own Data

Place your private data at `data/raw/financial_data.csv` and run:

```bash
python3 src/financial_distress_pipeline.py \
  --input data/raw/financial_data.csv \
  --output-dir outputs/main \
  --targets Oscore Zrisk \
  --save-predictions
```

The `data/raw/`, `outputs/`, and `models/` directories are ignored by Git to avoid uploading private data, generated artifacts, or model checkpoints.

## Main Method

For each firm observation $i$, the model learns a mapping from financial indicators to a continuous risk score:

$$
\begin{aligned}
\widehat{y}_i &= f(X_i) .
\end{aligned}
$$

Model performance is evaluated from four perspectives:

1. prediction error: RMSE and MAE;
2. explanatory power: $R^2$;
3. ranking ability: Pearson and Spearman correlations;
4. risk-warning usefulness: Top-20% Precision, Recall, and F1.

See [`METHOD.md`](METHOD.md) for the full methodology and [`REPORT.md`](REPORT.md) for the integrated research report.

## Data Availability

This repository does not redistribute the original financial dataset. A synthetic dataset generator is provided only for code verification and demonstration.

## License

MIT License.
