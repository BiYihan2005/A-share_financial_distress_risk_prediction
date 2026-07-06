# Integrated Research Report

## Title

A-Share Listed Company Financial Distress Risk Prediction with Leakage-Free Machine Learning

## Abstract

This project studies the prediction of financial distress risk for A-share listed companies using structured financial indicators. The main response variable is `Oscore`, while `Zrisk = -Zscore` is constructed as a robustness target so that both risk indicators share the same direction: a larger value represents higher financial distress risk.

The project emphasizes leakage-free preprocessing, multi-model comparison, neural-network baselines, financial interpretation, and high-risk ranking evaluation. The original experiment compared linear models, regularized linear models, Random Forest, LightGBM, baseline neural networks, and improved neural networks. The key empirical result is that nonlinear models substantially outperform linear models, suggesting that financial distress risk is driven by nonlinear combinations and interactions among profitability, cash-flow quality, solvency, operating quality, and market-expectation indicators.

## 1. Research Motivation

Financial distress is a central issue in corporate finance. When profitability deteriorates, cash-flow coverage weakens, or leverage pressure rises, a firm may face higher financing costs, debt rollover difficulty, investment contraction, or even default and bankruptcy costs.

A machine-learning model for financial distress risk should therefore satisfy two requirements:

1. it should predict or rank firm-level risk effectively;
2. it should remain interpretable from a corporate-finance perspective.

This project positions the task as continuous risk-score prediction rather than binary default classification.

## 2. Target Variables

The main target is `Oscore`. It is interpreted as a continuous financial distress risk score:

$$
\begin{aligned}
\mathrm{Oscore}_i &= f\left(\mathrm{Size}_i, \mathrm{Leverage}_i, \mathrm{WorkingCapital}_i, \mathrm{Profitability}_i, \mathrm{CashFlowCoverage}_i, \ldots\right) .
\end{aligned}
$$

Because `Zscore` usually has the opposite direction, the project constructs:

$$
\begin{aligned}
\mathrm{Zrisk}_i &= -\mathrm{Zscore}_i .
\end{aligned}
$$

This robustness target allows both `Oscore` and `Zrisk` to be interpreted consistently: larger values correspond to higher risk.

## 3. Data Structure

The original empirical dataset contains firm identifiers, industry labels, report-type labels, risk targets, and structured financial indicators. The public repository does not include the original private dataset.

The modeling pipeline uses approximately 70% / 15% / 15% train/validation/test splits. In the original experiment, the available samples after target filtering were large enough to support stable model comparison. The Oscore experiment used 131,735 training observations, 28,229 validation observations, and 28,229 test observations. The Zrisk experiment used 140,072 training observations, 30,015 validation observations, and 30,016 test observations.

## 4. Leakage-Free Preprocessing

A major methodological point is that preprocessing must be learned only from the training set. Standardization is written as:

$$
\begin{aligned}
z_{ij} &= \frac{x_{ij}-\mu_j^{\mathrm{train}}}{\sigma_j^{\mathrm{train}}} .
\end{aligned}
$$

The same training-set parameters are then applied to validation and test samples. This prevents test-set distributional information from entering the training procedure.

The pipeline also applies:

- target filtering for missing response values;
- median imputation and missingness indicators;
- winsorization at training-set 1% and 99% quantiles;
- signed logarithmic transformation for highly skewed variables;
- one-hot encoding for industry and report type;
- model evaluation on a held-out test set.

Winsorization is defined as:

$$
\begin{aligned}
\widetilde{x}_{ij} &= \min\left(\max\left(x_{ij}, q^{\mathrm{train}}_{0.01,j}\right), q^{\mathrm{train}}_{0.99,j}\right) .
\end{aligned}
$$

The signed log transformation is:

$$
\begin{aligned}
x'_{ij} &= \mathrm{sign}\left(x_{ij}\right)\log\left(1+\left|x_{ij}\right|\right) .
\end{aligned}
$$

## 5. Models

The model comparison includes:

- dummy mean and median baselines;
- linear regression;
- Ridge, Lasso, and ElasticNet;
- Random Forest;
- LightGBM or histogram gradient boosting fallback;
- baseline and improved neural-network regressors.

Linear models test whether additive linear relations are sufficient. Tree and boosting models test nonlinear interactions among financial ratios. Neural networks test whether learned nonlinear representations can approximate or exceed tree-based performance.

## 6. Evaluation Metrics

The project reports regression error, explanatory power, ranking ability, and high-risk screening performance.

$$
\begin{aligned}
\mathrm{RMSE} &= \sqrt{\frac{1}{n}\sum_{i=1}^{n}\left(y_i-\widehat{y}_i\right)^2} ,
\end{aligned}
$$

$$
\begin{aligned}
\mathrm{MAE} &= \frac{1}{n}\sum_{i=1}^{n}\left|y_i-\widehat{y}_i\right| ,
\end{aligned}
$$

$$
\begin{aligned}
R^2 &= 1-\frac{\sum_{i=1}^{n}\left(y_i-\widehat{y}_i\right)^2}{\sum_{i=1}^{n}\left(y_i-\overline{y}\right)^2} .
\end{aligned}
$$

For risk-warning applications, ranking high-risk firms is often more important than minimizing average error. Therefore, the project defines the truly highest-risk 20% as $H_{20}$ and the predicted highest-risk 20% as $\widehat{H}_{20}$:

$$
\begin{aligned}
\mathrm{Precision@20\%} &= \frac{\left|\widehat{H}_{20}\cap H_{20}\right|}{\left|\widehat{H}_{20}\right|} ,
\end{aligned}
$$

$$
\begin{aligned}
\mathrm{Recall@20\%} &= \frac{\left|\widehat{H}_{20}\cap H_{20}\right|}{\left|H_{20}\right|} .
\end{aligned}
$$

## 7. Main Oscore Results

In the original Oscore experiment, LightGBM achieved the best overall performance:

| Model | RMSE | MAE | R2 | Pearson | Spearman | Top20 F1 |
|---|---:|---:|---:|---:|---:|---:|
| LightGBM | 1.0048 | 0.7892 | 0.5814 | 0.7637 | 0.7427 | 0.5809 |
| Improved NN, 256-128-64 | 1.0103 | 0.7878 | 0.5768 | 0.7596 | 0.7392 | 0.5767 |
| Random Forest | 1.1370 | 0.8981 | 0.4641 | 0.6837 | 0.6597 | 0.5218 |
| Linear Regression | 1.2945 | 1.0258 | 0.3053 | 0.5525 | 0.5355 | 0.4090 |
| Lasso | 1.3023 | 1.0329 | 0.2969 | 0.5449 | 0.5268 | 0.4033 |

The difference between LightGBM and the improved neural network is small, while both clearly outperform linear models. This suggests that distress risk contains nonlinear interactions that are not fully captured by additive linear specifications.

## 8. Zrisk Robustness Results

In the Zrisk robustness experiment, the improved neural network achieved the strongest overall result, while LightGBM remained very close:

| Model | RMSE | MAE | R2 | Pearson | Spearman | Top20 F1 |
|---|---:|---:|---:|---:|---:|---:|
| Improved NN, 256-128-64 | 3.0242 | 1.6780 | 0.7021 | 0.8422 | 0.8671 | 0.7095 |
| LightGBM | 3.0361 | 1.6998 | 0.6997 | 0.8400 | 0.8469 | 0.6802 |
| Baseline NN | 3.1711 | 1.7099 | 0.6724 | 0.8207 | 0.8483 | 0.6644 |
| Random Forest | 3.8999 | 2.3697 | 0.5046 | 0.7198 | 0.7445 | 0.6442 |
| Linear Regression | 4.2654 | 2.9018 | 0.4071 | 0.6381 | 0.6711 | 0.6141 |

The model ranking remains broadly stable: nonlinear models dominate linear alternatives. This supports the robustness of the main conclusion.

## 9. Financial Interpretation

The most important predictors are concentrated in several interpretable finance channels:

| Variable | Interpretation |
|---|---|
| Sales EVA rate | Value creation per unit of sales revenue |
| Cash flow coverage of maturing debt | Short-term solvency and debt service capacity |
| Cash content of net profit | Earnings quality and cash-flow support |
| EVA rate | Ability to create value after capital cost |
| Operating index | Quality of converting operating income into cash flow |
| Tobin's Q | Market expectations and growth valuation |

These results are consistent with corporate-finance theory. Distress risk is not driven by a single ratio; it is jointly shaped by value creation, solvency, earnings quality, operating cash-flow quality, growth volatility, and market expectations.

## 10. Limitations and Future Work

This project predicts risk scores rather than realized default events. Without exact reporting dates, it should not be interpreted as a strict out-of-time default prediction model. Future work can improve the empirical design by:

- constructing a firm-quarter panel;
- using lagged financial indicators to predict next-period risk;
- adding market returns, volatility, governance variables, and macroeconomic indicators;
- evaluating models with time-based splits or rolling-window validation;
- adding SHAP-based explanation for tree models.

## 11. Conclusion

The project demonstrates a leakage-free, finance-aware machine learning workflow for A-share financial distress risk prediction. The empirical results show that LightGBM and improved neural networks consistently outperform linear models on both Oscore and Zrisk targets. The interpretation further suggests that machine learning can identify economically meaningful risk channels rather than only fitting black-box statistical patterns.
