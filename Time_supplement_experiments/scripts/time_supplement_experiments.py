#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
time_supplement_experiments.py
公司理财神经网络大作业：统计日期补充实验脚本
================================================

本脚本用于在“已经有统计截止日期”的新版数据上，单独补充两个时间维度实验，
不推翻、不覆盖原报告中的随机切分主实验。

补充实验 1：未来一期财务风险预测
    用第 t 期财务指标预测同一股票第 t+1 期 Oscore / Zrisk。
    记为：X_{i,t} -> Y_{i,t+1}

补充实验 2：加入一期滞后与变化特征的未来一期预测
    在补充实验 1 的基础上，额外加入上一期 X_{i,t-1} 和变化量 ΔX_{i,t}=X_{i,t}-X_{i,t-1}。
    记为：[X_{i,t}, X_{i,t-1}, ΔX_{i,t}] -> Y_{i,t+1}

核心设计原则：
1. 统计截止日期只用于时间排序、时间切分和构造 lead/lag，不作为普通解释变量输入模型；
2. 训练集、验证集、测试集按时间划分，模拟“用过去预测未来”；
3. 缺失填补、缩尾、偏态变换、标准化、One-Hot 编码等预处理规则只在训练集拟合；
4. 验证集和测试集只使用训练集规则 transform，避免未来信息泄露；
5. 输出单独的补充实验结果表和 Markdown 报告，方便直接附在原 Word 报告后面。

推荐运行：
python time_supplement_experiments.py \
  --input "数据(2026.5 含日期).csv" \
  --output_dir "outputs/time_supplement" \
  --targets Oscore Zrisk

快速调试：
python time_supplement_experiments.py \
  --input "数据(2026.5 含日期).csv" \
  --output_dir "outputs/time_supplement_quick" \
  --targets Oscore Zrisk \
  --sample_train_rows 50000 \
  --rf_n_estimators 50 \
  --lgbm_n_estimators 80
"""

from __future__ import annotations

import argparse
import json
import math
import time
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from scipy.stats import pearsonr, spearmanr

warnings.filterwarnings("ignore")

try:
    from lightgbm import LGBMRegressor
    HAS_LIGHTGBM = True
except Exception:
    HAS_LIGHTGBM = False


# =============================
# 一、基础工具函数
# =============================


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_csv_safely(path: str | Path) -> pd.DataFrame:
    """兼容 utf-8-sig / gbk / gb18030 等常见中文 CSV 编码。"""
    path = Path(path)
    for enc in ["utf-8-sig", "utf-8", "gbk", "gb18030"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            continue
    raise RuntimeError(f"无法读取 CSV 文件：{path}")


def safe_float(x) -> float:
    try:
        if pd.isna(x):
            return np.nan
        return float(x)
    except Exception:
        return np.nan


def pearson_safe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 3 or np.std(y_pred) < 1e-12 or np.std(y_true) < 1e-12:
        return np.nan
    try:
        return float(pearsonr(y_true, y_pred)[0])
    except Exception:
        return np.nan


def spearman_safe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 3 or np.std(y_pred) < 1e-12 or np.std(y_true) < 1e-12:
        return np.nan
    try:
        return float(spearmanr(y_true, y_pred)[0])
    except Exception:
        return np.nan


def top20_metrics(y_true: np.ndarray, y_pred: np.ndarray, top_frac: float = 0.20) -> Tuple[float, float, float]:
    """
    将真实风险最高的 top_frac 样本定义为真实高风险组，
    将预测风险最高的 top_frac 样本定义为模型识别高风险组。

    注意：这里使用排序取前 k 个，而不是用阈值 >= quantile。
    这样可以避免 dummy 常数预测时因为大量并列值导致 Recall 异常为 1 的问题。
    """
    n = len(y_true)
    if n == 0:
        return np.nan, np.nan, np.nan
    k = max(1, int(math.ceil(n * top_frac)))
    true_top = set(np.argsort(y_true)[-k:])
    pred_top = set(np.argsort(y_pred)[-k:])
    inter = len(true_top & pred_top)
    precision = inter / len(pred_top) if pred_top else np.nan
    recall = inter / len(true_top) if true_top else np.nan
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return float(precision), float(recall), float(f1)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    p20, r20, f20 = top20_metrics(y_true, y_pred, top_frac=0.20)
    return {
        "MSE": float(mse),
        "RMSE": float(rmse),
        "MAE": float(mae),
        "R2": float(r2),
        "Pearson": pearson_safe(y_true, y_pred),
        "Spearman": spearman_safe(y_true, y_pred),
        "Top20_Precision": p20,
        "Top20_Recall": r20,
        "Top20_F1": f20,
    }


def date_gap_is_one_quarter(delta_days: pd.Series, min_days: int = 60, max_days: int = 110) -> pd.Series:
    """判断两期日期间隔是否大致为一个季度。默认放宽到 60–110 天，兼容季度天数差异。"""
    return delta_days.between(min_days, max_days, inclusive="both")


# =============================
# 二、自定义预处理器
# =============================

@dataclass
class PreprocessConfig:
    missing_indicator_low: float = 0.05
    missing_indicator_high: float = 0.80
    drop_missing_threshold: float = 0.80
    winsor_lower: float = 0.01
    winsor_upper: float = 0.99
    skew_threshold: float = 2.0


class LeakageFreePreprocessor:
    """
    无泄漏表格预处理器。

    fit 时只能使用训练集；transform 时把训练集学到的规则应用到验证集和测试集。
    这样可以避免验证集/测试集的信息提前进入训练流程。
    """

    def __init__(self, numeric_cols: List[str], categorical_cols: List[str], config: PreprocessConfig):
        self.numeric_cols = list(numeric_cols)
        self.categorical_cols = list(categorical_cols)
        self.config = config

        self.keep_numeric_cols: List[str] = []
        self.dropped_numeric_cols: List[str] = []
        self.indicator_cols: List[str] = []
        self.medians: Dict[str, float] = {}
        self.lower_bounds: Dict[str, float] = {}
        self.upper_bounds: Dict[str, float] = {}
        self.signed_log_cols: List[str] = []
        self.scaler: Optional[StandardScaler] = None
        self.ohe: Optional[OneHotEncoder] = None
        self.feature_names_: List[str] = []
        self.summary_: pd.DataFrame = pd.DataFrame()

    def fit(self, X: pd.DataFrame):
        rows = []
        self.keep_numeric_cols = []
        self.dropped_numeric_cols = []
        self.indicator_cols = []
        self.medians = {}
        self.lower_bounds = {}
        self.upper_bounds = {}
        self.signed_log_cols = []

        for col in self.numeric_cols:
            s = pd.to_numeric(X[col], errors="coerce")
            miss_rate = float(s.isna().mean())
            if miss_rate >= self.config.drop_missing_threshold:
                self.dropped_numeric_cols.append(col)
                rows.append({"变量": col, "缺失率": miss_rate, "是否保留": False, "是否加缺失指示": False, "是否signed_log": False, "处理说明": "缺失率过高，删除"})
                continue

            self.keep_numeric_cols.append(col)
            add_indicator = miss_rate >= self.config.missing_indicator_low
            if add_indicator:
                self.indicator_cols.append(col)

            median = safe_float(s.median())
            if np.isnan(median):
                median = 0.0
            self.medians[col] = median

            filled = s.fillna(median)
            lo = safe_float(filled.quantile(self.config.winsor_lower))
            hi = safe_float(filled.quantile(self.config.winsor_upper))
            if np.isnan(lo):
                lo = median
            if np.isnan(hi):
                hi = median
            if lo > hi:
                lo, hi = hi, lo
            self.lower_bounds[col] = lo
            self.upper_bounds[col] = hi

            clipped = filled.clip(lo, hi)
            skew = safe_float(clipped.skew())
            use_signed_log = bool(abs(skew) >= self.config.skew_threshold)
            if use_signed_log:
                self.signed_log_cols.append(col)

            if miss_rate < self.config.missing_indicator_low:
                desc = "中位数填补"
            else:
                desc = "中位数填补 + 缺失指示变量"
            desc += " + 1%/99%缩尾"
            if use_signed_log:
                desc += " + signed-log偏态变换"

            rows.append({"变量": col, "缺失率": miss_rate, "是否保留": True, "是否加缺失指示": add_indicator, "是否signed_log": use_signed_log, "训练集1%下界": lo, "训练集99%上界": hi, "处理说明": desc})

        X_num = self._transform_numeric_part(X, fit_stage=True)
        self.scaler = StandardScaler()
        self.scaler.fit(X_num)

        # OneHotEncoder 在不同 sklearn 版本里 sparse/sparse_output 参数名不同，这里做兼容。
        try:
            self.ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:
            self.ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)
        if self.categorical_cols:
            X_cat = X[self.categorical_cols].astype("object").fillna("缺失")
            self.ohe.fit(X_cat)
            cat_names = list(self.ohe.get_feature_names_out(self.categorical_cols))
        else:
            cat_names = []

        num_names = list(X_num.columns)
        self.feature_names_ = num_names + cat_names
        self.summary_ = pd.DataFrame(rows)
        return self

    def _transform_numeric_part(self, X: pd.DataFrame, fit_stage: bool = False) -> pd.DataFrame:
        out = pd.DataFrame(index=X.index)
        for col in self.keep_numeric_cols:
            s = pd.to_numeric(X[col], errors="coerce")
            if col in self.indicator_cols:
                out[f"{col}_missing_ind"] = s.isna().astype(float)
            filled = s.fillna(self.medians[col])
            clipped = filled.clip(self.lower_bounds[col], self.upper_bounds[col])
            if col in self.signed_log_cols:
                clipped = np.sign(clipped) * np.log1p(np.abs(clipped))
            out[col] = clipped.astype(float)
        return out

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        if self.scaler is None or self.ohe is None:
            raise RuntimeError("Preprocessor 尚未 fit")
        X_num = self._transform_numeric_part(X)
        X_num_scaled = self.scaler.transform(X_num)
        if self.categorical_cols:
            X_cat = X[self.categorical_cols].astype("object").fillna("缺失")
            X_cat_ohe = self.ohe.transform(X_cat)
            return np.hstack([X_num_scaled, X_cat_ohe])
        return X_num_scaled

    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        self.fit(X)
        return self.transform(X)


# =============================
# 三、面板数据构造
# =============================


def prepare_base_dataframe(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """解析日期，构造 Zrisk，按股票和日期排序。"""
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df[df[date_col].notna()].copy()
    if "Zscore" in df.columns:
        df["Zrisk"] = -pd.to_numeric(df["Zscore"], errors="coerce")
    df = df.sort_values(["股票代码", date_col]).reset_index(drop=True)
    return df


def create_lead1_dataset(
    df: pd.DataFrame,
    date_col: str,
    target: str,
    use_lag_change: bool = False,
    min_gap_days: int = 60,
    max_gap_days: int = 110,
) -> Tuple[pd.DataFrame, str]:
    """
    构造未来一期预测数据。

    use_lag_change=False：
        X_t -> Y_{t+1}

    use_lag_change=True：
        [X_t, X_{t-1}, X_t-X_{t-1}] -> Y_{t+1}
    """
    work = df.copy().sort_values(["股票代码", date_col]).reset_index(drop=True)

    target_lead = f"{target}_lead1"
    next_date_col = "next_统计截止日期"
    prev_date_col = "prev_统计截止日期"

    work[target_lead] = work.groupby("股票代码")[target].shift(-1)
    work[next_date_col] = work.groupby("股票代码")[date_col].shift(-1)
    work["next_gap_days"] = (work[next_date_col] - work[date_col]).dt.days
    work["is_next_quarter"] = date_gap_is_one_quarter(work["next_gap_days"], min_gap_days, max_gap_days)

    if use_lag_change:
        work[prev_date_col] = work.groupby("股票代码")[date_col].shift(1)
        work["prev_gap_days"] = (work[date_col] - work[prev_date_col]).dt.days
        work["is_prev_quarter"] = date_gap_is_one_quarter(work["prev_gap_days"], min_gap_days, max_gap_days)

        # 对原始数值财务变量构造 lag1 和 change1。
        # 注意不对 Oscore/Zscore/Zrisk 构造滞后特征，避免把目标风险分数本身作为解释变量。
        exclude = {"股票代码", "股票简称", date_col, "Oscore", "Zscore", "Zrisk", target_lead, next_date_col, prev_date_col}
        numeric_cols = [c for c in work.columns if c not in exclude and pd.api.types.is_numeric_dtype(work[c])]
        for col in numeric_cols:
            lag_col = f"{col}_lag1"
            chg_col = f"{col}_change1"
            work[lag_col] = work.groupby("股票代码")[col].shift(1)
            work[chg_col] = pd.to_numeric(work[col], errors="coerce") - pd.to_numeric(work[lag_col], errors="coerce")

        # 只有当前期同时存在前一季度和后一季度时，lag/change 解释才最干净。
        work = work[work["is_prev_quarter"]].copy()

    work = work[work["is_next_quarter"]].copy()
    work = work[work[target_lead].notna()].copy()
    return work, target_lead


def time_split(df: pd.DataFrame, date_col: str, train_end: str, valid_end: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_end_dt = pd.Timestamp(train_end)
    valid_end_dt = pd.Timestamp(valid_end)
    train = df[df[date_col] <= train_end_dt].copy()
    valid = df[(df[date_col] > train_end_dt) & (df[date_col] <= valid_end_dt)].copy()
    test = df[df[date_col] > valid_end_dt].copy()
    return train, valid, test


def infer_feature_columns(df: pd.DataFrame, date_col: str, target_col: str) -> Tuple[List[str], List[str]]:
    """识别数值特征和分类特征。"""
    always_exclude = {
        "股票代码", "股票简称", date_col,
        "next_统计截止日期", "prev_统计截止日期", "next_gap_days", "prev_gap_days", "is_next_quarter", "is_prev_quarter",
        "Oscore", "Zscore", "Zrisk", "Oscore_lead1", "Zrisk_lead1", target_col,
    }
    categorical_candidates = ["行业名称1", "报表类型编码"]
    categorical_cols = [c for c in categorical_candidates if c in df.columns and c not in always_exclude]
    numeric_cols = [c for c in df.columns if c not in always_exclude and c not in categorical_cols and pd.api.types.is_numeric_dtype(df[c])]
    return numeric_cols, categorical_cols


# =============================
# 四、模型训练与评估
# =============================


def build_models(args: argparse.Namespace) -> Dict[str, object]:
    """根据 --models 参数构造模型。

    默认模型可以覆盖基准、线性、正则化线性、树模型和梯度提升模型。
    如果本地机器运行较慢，可以用：
    --models dummy_mean linear ridge lightgbm
    """
    requested = set(args.models)
    models: Dict[str, object] = {}

    if "dummy_mean" in requested:
        models["dummy_mean"] = DummyRegressor(strategy="mean")
    if "linear" in requested:
        models["linear"] = LinearRegression()
    if "ridge" in requested:
        models["ridge"] = Ridge(alpha=1.0, random_state=args.random_state)
    if "elasticnet" in requested:
        models["elasticnet"] = ElasticNet(alpha=0.001, l1_ratio=0.2, max_iter=3000, random_state=args.random_state)
    if "random_forest" in requested:
        models["random_forest"] = RandomForestRegressor(
            n_estimators=args.rf_n_estimators,
            max_depth=args.rf_max_depth,
            min_samples_leaf=args.rf_min_samples_leaf,
            n_jobs=args.n_jobs,
            random_state=args.random_state,
        )

    if "lightgbm" in requested:
        if HAS_LIGHTGBM:
            models["lightgbm"] = LGBMRegressor(
                n_estimators=args.lgbm_n_estimators,
                learning_rate=args.lgbm_learning_rate,
                num_leaves=args.lgbm_num_leaves,
                min_child_samples=80,
                max_bin=127,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=args.random_state,
                n_jobs=args.n_jobs,
                verbose=-1,
                force_col_wise=True,
            )
        else:
            models["hist_gradient_boosting"] = HistGradientBoostingRegressor(
                max_iter=args.lgbm_n_estimators,
                learning_rate=args.lgbm_learning_rate,
                random_state=args.random_state,
            )

    if args.include_nn or "mlp_128_64" in requested:
        models["mlp_128_64"] = MLPRegressor(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            solver="adam",
            alpha=args.nn_alpha,
            learning_rate_init=args.nn_lr,
            batch_size=args.nn_batch_size,
            max_iter=args.nn_max_iter,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=10,
            random_state=args.random_state,
            verbose=False,
        )

    return models


def maybe_sample(df: pd.DataFrame, n: Optional[int], random_state: int) -> pd.DataFrame:
    if n is None or len(df) <= n:
        return df
    return df.sample(n=n, random_state=random_state).copy()


def run_single_experiment(
    df_exp: pd.DataFrame,
    target_col: str,
    target_label: str,
    experiment_name: str,
    args: argparse.Namespace,
    outdir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """运行某个目标变量、某个实验设定下的模型比较。"""
    exp_dir = ensure_dir(outdir / experiment_name / target_label.lower())

    train, valid, test = time_split(df_exp, args.date_col, args.train_end, args.valid_end)
    # 目标变量缺失不能训练监督模型。
    train = train[train[target_col].notna()].copy()
    valid = valid[valid[target_col].notna()].copy()
    test = test[test[target_col].notna()].copy()

    # 可选采样，用于快速调试；正式结果建议不采样。
    train_fit = maybe_sample(train, args.sample_train_rows, args.random_state)

    numeric_cols, categorical_cols = infer_feature_columns(df_exp, args.date_col, target_col)
    prep = LeakageFreePreprocessor(numeric_cols, categorical_cols, PreprocessConfig())

    X_train = prep.fit_transform(train_fit)
    y_train = pd.to_numeric(train_fit[target_col], errors="coerce").values.astype(float)
    X_valid = prep.transform(valid)
    y_valid = pd.to_numeric(valid[target_col], errors="coerce").values.astype(float)
    X_test = prep.transform(test)
    y_test = pd.to_numeric(test[target_col], errors="coerce").values.astype(float)

    # 保存样本切分和预处理摘要，便于写报告。
    split_summary = pd.DataFrame([
        {"样本集": "train_fit", "样本数": len(train_fit), "起始日期": train_fit[args.date_col].min(), "结束日期": train_fit[args.date_col].max(), "股票数": train_fit["股票代码"].nunique(), "行业数": train_fit["行业名称1"].nunique() if "行业名称1" in train_fit.columns else np.nan},
        {"样本集": "valid", "样本数": len(valid), "起始日期": valid[args.date_col].min(), "结束日期": valid[args.date_col].max(), "股票数": valid["股票代码"].nunique(), "行业数": valid["行业名称1"].nunique() if "行业名称1" in valid.columns else np.nan},
        {"样本集": "test", "样本数": len(test), "起始日期": test[args.date_col].min(), "结束日期": test[args.date_col].max(), "股票数": test["股票代码"].nunique(), "行业数": test["行业名称1"].nunique() if "行业名称1" in test.columns else np.nan},
    ])
    split_summary.to_csv(exp_dir / "split_summary.csv", index=False, encoding="utf-8-sig")
    prep.summary_.to_csv(exp_dir / "preprocess_summary.csv", index=False, encoding="utf-8-sig")
    pd.Series(prep.feature_names_, name="feature_name").to_csv(exp_dir / "feature_names.csv", index=False, encoding="utf-8-sig")

    models = build_models(args)
    rows = []
    pred_rows = []

    for model_name, model in models.items():
        t0 = time.time()
        try:
            model.fit(X_train, y_train)
            train_time = time.time() - t0
            pred_valid = model.predict(X_valid) if len(valid) else np.array([])
            pred_test = model.predict(X_test) if len(test) else np.array([])

            m_valid = regression_metrics(y_valid, pred_valid) if len(valid) else {}
            m_test = regression_metrics(y_test, pred_test) if len(test) else {}

            row = {
                "experiment": experiment_name,
                "target": target_label,
                "model": model_name,
                "train_time_sec": train_time,
                "n_train": len(train_fit),
                "n_valid": len(valid),
                "n_test": len(test),
                "n_features": X_train.shape[1],
            }
            for k, v in m_valid.items():
                row[f"valid_{k}"] = v
            for k, v in m_test.items():
                row[k] = v
            rows.append(row)

            # 保存测试集预测，后面可以做分组分析。这里只保存核心字段，避免文件过大。
            pred_rows.append(pd.DataFrame({
                "experiment": experiment_name,
                "target": target_label,
                "model": model_name,
                "股票代码": test["股票代码"].values,
                "统计截止日期": test[args.date_col].values,
                "y_true": y_test,
                "y_pred": pred_test,
            }))
            print(f"[{experiment_name} | {target_label}] {model_name} 完成：test Spearman={m_test.get('Spearman', np.nan):.4f}, Top20_F1={m_test.get('Top20_F1', np.nan):.4f}, 用时 {train_time:.1f}s")
        except Exception as e:
            rows.append({"experiment": experiment_name, "target": target_label, "model": model_name, "error": str(e), "n_train": len(train_fit), "n_valid": len(valid), "n_test": len(test)})
            print(f"[{experiment_name} | {target_label}] {model_name} 失败：{e}")

    metrics = pd.DataFrame(rows)
    metrics = metrics.sort_values(["Spearman", "R2", "Top20_F1"], ascending=[False, False, False], na_position="last")
    metrics.to_csv(exp_dir / "model_metrics.csv", index=False, encoding="utf-8-sig")

    if pred_rows:
        preds = pd.concat(pred_rows, ignore_index=True)
        preds.to_csv(exp_dir / "test_predictions_all_models.csv", index=False, encoding="utf-8-sig")
    else:
        preds = pd.DataFrame()

    return metrics, split_summary


# =============================
# 五、报告生成
# =============================


def format_float(x, ndigits: int = 4) -> str:
    try:
        if pd.isna(x):
            return ""
        return f"{float(x):.{ndigits}f}"
    except Exception:
        return str(x)


def metrics_table_for_report(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["experiment", "target", "model", "RMSE", "MAE", "R2", "Pearson", "Spearman", "Top20_F1", "n_train", "n_test"]
    cols = [c for c in cols if c in df.columns]
    out = df[cols].copy()
    for c in ["RMSE", "MAE", "R2", "Pearson", "Spearman", "Top20_F1"]:
        if c in out.columns:
            out[c] = out[c].map(lambda x: format_float(x, 4))
    return out


def write_report(all_metrics: pd.DataFrame, date_summary: pd.DataFrame, outdir: Path, args: argparse.Namespace) -> None:
    report_path = outdir / "time_supplement_report.md"
    lines: List[str] = []

    lines.append("# 统计日期补充实验报告：未来一期风险预测与滞后特征扩展\n")
    lines.append("## 一、实验目的\n")
    lines.append(
        "在原主实验中，模型主要基于随机切分样本评估公司财务风险指标的预测效果。"
        "现在数据中新增了 `统计截止日期`，因此可以进一步考察模型是否能够利用历史报告期信息预测未来一期的财务风险。"
        "本补充报告不替代原报告，而是作为时间维度上的补充实验。\n"
    )

    lines.append("## 二、日期数据概况\n")
    lines.append(date_summary.to_markdown(index=False))
    lines.append("\n")

    lines.append("## 三、补充实验设计\n")
    lines.append("### 3.1 补充实验 1：未来一期财务风险预测\n")
    lines.append(
        "该实验使用第 $t$ 期公司财务指标预测同一股票第 $t+1$ 期的风险指标，即：\n\n"
        "```math\n"
        "X_{i,t} \\rightarrow Y_{i,t+1}\n"
        "```\n\n"
        "其中 $Y_{i,t+1}$ 分别取下一期 Oscore 和下一期 Zrisk。"
        "为了避免错误匹配，脚本只保留同一股票当前日期与下一期日期间隔约为一个季度的样本。\n"
    )

    lines.append("### 3.2 补充实验 2：加入一期滞后与变化特征\n")
    lines.append(
        "该实验在补充实验 1 的基础上，额外加入上一期财务指标和最近一期变化量：\n\n"
        "```math\n"
        "X_{i,t-1}, \\quad \\Delta X_{i,t}=X_{i,t}-X_{i,t-1}\n"
        "```\n\n"
        "因此模型可以同时利用当前水平、上一期水平和最近一期变化方向来预测下一期风险。"
        "例如，资产负债率本身较高可能表示杠杆风险较高，而资产负债率快速上升则可能说明融资压力正在恶化。\n"
    )

    lines.append("### 3.3 时间切分与无泄漏预处理\n")
    lines.append(
        f"本文采用时间切分：训练集截至 `{args.train_end}`，验证集截至 `{args.valid_end}`，其后样本作为测试集。"
        "所有缺失填补、缩尾、偏态变换、标准化和 One-Hot 编码规则均只在训练集拟合，并应用到验证集和测试集。"
        "这种设计模拟真实场景下“用过去数据训练模型、预测未来时期风险”的过程。\n"
    )

    lines.append("## 四、实验结果\n")
    for exp in all_metrics["experiment"].dropna().unique():
        for target in all_metrics["target"].dropna().unique():
            sub = all_metrics[(all_metrics["experiment"] == exp) & (all_metrics["target"] == target)].copy()
            if sub.empty:
                continue
            lines.append(f"### 4.{len(lines)} {exp} - {target}\n")
            lines.append(metrics_table_for_report(sub).to_markdown(index=False))
            lines.append("\n")
            best = sub.sort_values(["Spearman", "R2", "Top20_F1"], ascending=[False, False, False], na_position="last").iloc[0]
            lines.append(
                f"在 `{exp}` 的 `{target}` 预测任务中，按测试集 Spearman 排序相关性看，表现最好的模型是 "
                f"`{best.get('model')}`，RMSE={format_float(best.get('RMSE'))}，R²={format_float(best.get('R2'))}，"
                f"Spearman={format_float(best.get('Spearman'))}，Top20 F1={format_float(best.get('Top20_F1'))}。\n"
            )

    lines.append("## 五、结果解读建议\n")
    lines.append(
        "与随机切分主实验相比，未来一期预测通常更难，因为模型需要利用当前报告期信息预测下一报告期风险。"
        "如果本补充实验中的 R² 或 Spearman 低于原主实验，这是正常现象，并不代表模型失败，而是说明真正的时间外推预测难度更高。"
        "报告中应重点观察非线性模型是否仍然优于线性模型，以及 Top20 F1 是否仍然具有一定高风险识别能力。\n"
    )
    lines.append(
        "若加入一期滞后与变化特征后，LightGBM、随机森林或神经网络类模型的 Spearman / Top20 F1 提升，"
        "说明财务指标的动态变化包含额外风险信息。若提升不明显，也可以解释为当前财务指标已经吸收了主要风险信息，"
        "或者滞后特征带来的噪声和缺失抵消了部分收益。\n"
    )

    lines.append("## 六、可直接写入 Word 的小结\n")
    lines.append(
        "> 在新增统计截止日期后，本文进一步构造未来一期财务风险预测实验。该实验以第 t 期公司财务指标作为解释变量，"
        "以同一公司第 t+1 期 Oscore 或 Zrisk 作为被解释变量，并采用时间切分方式进行训练、验证和测试。"
        "进一步地，本文加入上一期财务指标及最近一期变化量，检验企业财务状态的动态变化是否具有额外预测信息。"
        "补充实验使本文研究从同期风险拟合扩展为更接近真实应用的财务风险预警任务。\n"
    )

    report_path.write_text("\n".join(lines), encoding="utf-8")


# =============================
# 六、主函数
# =============================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="统计日期补充实验：未来一期风险预测与滞后变化特征")
    parser.add_argument("--input", required=True, help="含统计截止日期的 CSV 数据文件")
    parser.add_argument("--output_dir", default="outputs/time_supplement", help="输出目录")
    parser.add_argument("--targets", nargs="+", default=["Oscore", "Zrisk"], choices=["Oscore", "Zrisk"], help="要预测的风险指标")
    parser.add_argument("--date_col", default="统计截止日期", help="日期列名")
    parser.add_argument("--train_end", default="2020-12-31", help="训练集截止日期")
    parser.add_argument("--valid_end", default="2022-12-31", help="验证集截止日期；之后为测试集")
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--n_jobs", type=int, default=-1)

    # 调试采样参数；正式运行建议不设置。
    parser.add_argument("--sample_train_rows", type=int, default=None, help="训练集采样行数，正式实验建议不设置")

    # 模型参数
    parser.add_argument("--rf_n_estimators", type=int, default=120)
    parser.add_argument("--rf_max_depth", type=int, default=12)
    parser.add_argument("--rf_min_samples_leaf", type=int, default=20)
    parser.add_argument("--lgbm_n_estimators", type=int, default=200)
    parser.add_argument("--lgbm_learning_rate", type=float, default=0.05)
    parser.add_argument("--lgbm_num_leaves", type=int, default=31)
    parser.add_argument("--models", nargs="+", default=["dummy_mean", "linear", "ridge", "random_forest", "lightgbm"], help="要运行的模型列表，可选：dummy_mean linear ridge elasticnet random_forest lightgbm mlp_128_64")
    parser.add_argument("--include_nn", action="store_true", help="是否额外训练 sklearn MLP 神经网络；较慢")
    parser.add_argument("--nn_max_iter", type=int, default=60)
    parser.add_argument("--nn_batch_size", type=int, default=1024)
    parser.add_argument("--nn_lr", type=float, default=3e-4)
    parser.add_argument("--nn_alpha", type=float, default=1e-5)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outdir = ensure_dir(args.output_dir)

    print("读取数据...")
    df = read_csv_safely(args.input)
    df = prepare_base_dataframe(df, args.date_col)

    date_summary = pd.DataFrame([{
        "总样本数": len(df),
        "股票数": df["股票代码"].nunique(),
        "行业数": df["行业名称1"].nunique() if "行业名称1" in df.columns else np.nan,
        "起始日期": df[args.date_col].min(),
        "结束日期": df[args.date_col].max(),
        "季度数": df[args.date_col].nunique(),
        "Oscore非缺失": df["Oscore"].notna().sum() if "Oscore" in df.columns else np.nan,
        "Zrisk非缺失": df["Zrisk"].notna().sum() if "Zrisk" in df.columns else np.nan,
    }])
    date_summary.to_csv(outdir / "date_audit_summary.csv", index=False, encoding="utf-8-sig")

    all_metrics = []
    all_splits = []

    for target in args.targets:
        print(f"\n构造目标 {target}_lead1 的未来一期数据...")
        df_basic, target_lead = create_lead1_dataset(df, args.date_col, target, use_lag_change=False)
        print(f"基础未来一期样本数：{len(df_basic)}")
        metrics, split_summary = run_single_experiment(
            df_basic, target_lead, target, "experiment1_lead1_basic", args, outdir
        )
        all_metrics.append(metrics)
        split_summary["experiment"] = "experiment1_lead1_basic"
        split_summary["target"] = target
        all_splits.append(split_summary)

        print(f"\n构造目标 {target}_lead1 的滞后+变化特征数据...")
        df_lag, target_lead = create_lead1_dataset(df, args.date_col, target, use_lag_change=True)
        print(f"滞后+变化特征未来一期样本数：{len(df_lag)}")
        metrics, split_summary = run_single_experiment(
            df_lag, target_lead, target, "experiment2_lead1_lag_change", args, outdir
        )
        all_metrics.append(metrics)
        split_summary["experiment"] = "experiment2_lead1_lag_change"
        split_summary["target"] = target
        all_splits.append(split_summary)

    metrics_all = pd.concat(all_metrics, ignore_index=True) if all_metrics else pd.DataFrame()
    metrics_all.to_csv(outdir / "all_time_supplement_metrics.csv", index=False, encoding="utf-8-sig")
    splits_all = pd.concat(all_splits, ignore_index=True) if all_splits else pd.DataFrame()
    splits_all.to_csv(outdir / "all_time_supplement_split_summary.csv", index=False, encoding="utf-8-sig")

    write_report(metrics_all, date_summary, outdir, args)

    # 保存一个简短 JSON，方便检查运行配置。
    config = vars(args).copy()
    config["has_lightgbm"] = HAS_LIGHTGBM
    (outdir / "run_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n补充实验完成。核心输出：")
    print(outdir / "all_time_supplement_metrics.csv")
    print(outdir / "all_time_supplement_split_summary.csv")
    print(outdir / "time_supplement_report.md")


if __name__ == "__main__":
    main()
