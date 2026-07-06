# Methodology

## 1. Problem Formulation

For each firm observation $i$, let the financial indicator vector be:

$$
\begin{aligned}
X_i &= \left(x_{i1}, x_{i2}, \ldots, x_{ip}\right) .
\end{aligned}
$$

The prediction task is to learn a function that maps firm-level financial characteristics to a continuous distress risk score:

$$
\begin{aligned}
\widehat{y}_i &= f(X_i) .
\end{aligned}
$$

The main experiment uses `Oscore` as the risk target. The robustness experiment uses:

$$
\begin{aligned}
\mathrm{Zrisk}_i &= -\mathrm{Zscore}_i .
\end{aligned}
$$

This ensures that both `Oscore` and `Zrisk` have the same interpretation: a larger value indicates higher financial distress risk.

## 2. Leakage-Free Experimental Design

The central rule is:

> split the data first, and fit every preprocessing rule only on the training set.

The train, validation, and test sets are approximately 70%, 15%, and 15%. All preprocessing parameters are estimated from the training subset only, including missing-value statistics, winsorization thresholds, signed-log transformation decisions, scaling parameters, and categorical encoders.

This avoids a common source of data leakage in tabular finance projects: using distributional information from the validation or test set before evaluation.

## 3. Missing-Value Handling

Financial missingness is often informative. A missing value may reflect firm behavior, reporting practice, or industry-specific accounting patterns rather than pure random noise.

For a feature $X_j$, a missingness indicator is defined as:

$$
\begin{aligned}
M_{ij} &= \mathbb{1}\left[X_{ij}\ \mathrm{is\ missing}\right] .
\end{aligned}
$$

The cleaned pipeline uses median imputation fitted on the training set. For variables with non-trivial missingness, the missingness indicator $M_{ij}$ is appended to preserve the information that the value was originally absent.

## 4. Extreme-Value Treatment

Financial ratios may contain extreme observations because denominators can approach zero or because firms may experience unusual operating shocks.

For each continuous feature $j$, winsorization thresholds are estimated from the training set:

$$
\begin{aligned}
\widetilde{x}_{ij} &= \min\left(\max\left(x_{ij}, q^{\mathrm{train}}_{0.01,j}\right), q^{\mathrm{train}}_{0.99,j}\right) .
\end{aligned}
$$

This does not remove observations. It reduces the influence of extreme values while keeping the cross-sectional sample size unchanged.

## 5. Signed Log Transformation

Some financial variables are highly skewed and may be negative. A standard logarithm is not appropriate in that case. The pipeline uses a signed logarithmic transformation for selected highly skewed variables:

$$
\begin{aligned}
x'_{ij} &= \mathrm{sign}\left(x_{ij}\right)\log\left(1 + \left|x_{ij}\right|\right) .
\end{aligned}
$$

This preserves the sign of the variable while compressing extreme magnitudes.

## 6. Standardization and Encoding

After imputation, winsorization, and optional transformation, numeric features are standardized using training-set parameters:

$$
\begin{aligned}
z_{ij} &= \frac{x_{ij} - \mu^{\mathrm{train}}_j}{\sigma^{\mathrm{train}}_j} .
\end{aligned}
$$

Categorical variables such as industry and report type are represented through one-hot encoding with unseen categories ignored at test time.

## 7. Model Family

The pipeline compares several model classes:

1. **Dummy baselines**: mean and median regressors.
2. **Regularized linear models**: Ridge, Lasso, and ElasticNet.
3. **Tree ensembles**: Random Forest.
4. **Gradient boosting**: LightGBM when available; otherwise HistGradientBoostingRegressor.
5. **Neural networks**: a compact multilayer perceptron with BatchNorm, Dropout, AdamW, and early stopping when PyTorch is available.

The comparison is intentionally broad. Linear models provide interpretable benchmarks, while tree and neural models test whether nonlinear interactions among financial ratios improve distress-risk prediction.

## 8. Evaluation Metrics

The standard regression metrics are:

$$
\begin{aligned}
\mathrm{RMSE} &= \sqrt{\frac{1}{n}\sum_{i=1}^{n}\left(y_i - \widehat{y}_i\right)^2} ,
\end{aligned}
$$

$$
\begin{aligned}
\mathrm{MAE} &= \frac{1}{n}\sum_{i=1}^{n}\left|y_i - \widehat{y}_i\right| .
\end{aligned}
$$

The explanatory power is measured by:

$$
\begin{aligned}
R^2 &= 1 - \frac{\sum_{i=1}^{n}\left(y_i - \widehat{y}_i\right)^2}{\sum_{i=1}^{n}\left(y_i - \overline{y}\right)^2} .
\end{aligned}
$$

Financial distress screening also requires ranking ability. Therefore, the project reports Pearson and Spearman correlations and Top-20% risk identification metrics.

Let $H_{20}$ be the set of truly highest-risk firms and let $\widehat{H}_{20}$ be the set of firms ranked in the top 20% by predicted risk. Then:

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

The corresponding F1 score is:

$$
\begin{aligned}
\mathrm{F1@20\%} &= \frac{2\times \mathrm{Precision@20\%}\times \mathrm{Recall@20\%}}{\mathrm{Precision@20\%}+\mathrm{Recall@20\%}} .
\end{aligned}
$$

## 9. Financial Interpretation

The modeling results are interpreted through corporate-finance concepts rather than only machine-learning scores. Important variables are grouped into several risk channels:

- value creation: EVA rate and sales EVA rate;
- cash-flow solvency: cash flow coverage of maturing debt;
- earnings quality: cash content of net profit;
- operating quality: operating index;
- market expectations: Tobin's Q and valuation variables;
- volatility and growth: profitability fluctuation and growth indicators.

This structure turns the project from a pure prediction exercise into a finance-aware empirical modeling pipeline.

## 10. Limitations

The project predicts continuous risk scores rather than realized default events. If exact reporting dates are unavailable, the project should be interpreted as cross-sectional risk scoring rather than strict out-of-time default prediction. A stronger future version would build a firm-quarter panel, use lagged financial statements, and evaluate models through time-based validation or rolling-window backtesting.
