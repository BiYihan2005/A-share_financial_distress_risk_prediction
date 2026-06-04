#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
financial_risk_pipeline.py
A 股上市公司财务困境风险预测研究项目（开源版）

核心流程：
1. 读取原始数据，做最基本的数据审计；
2. 对 Oscore 主实验和 Zrisk=-Zscore 稳健性实验分别进行无泄漏预处理；
3. 训练核心对照模型：
   - Dummy Mean / Dummy Median（均值/中位数基准）
   - Linear Regression（线性回归）
   - Ridge / Lasso / ElasticNet
   - Random Forest（随机森林）
   - LightGBM，若未安装 lightgbm，则自动退回 HistGradientBoostingRegressor
   - Baseline NN（原始基准神经网络）：基准结构 64-30
   - Improved NN（改进神经网络）：改进神经网络 256-128-64
4. 统一输出模型评价指标；
5. 基于最佳模型做特征重要性、风险分组和残差分析；
6. 如果同时运行 Oscore 和 Zrisk，自动生成稳健性对比表。

推荐运行：
python financial_risk_pipeline.py \
  --input "data/raw/financial_data.csv" \
  --output_dir "outputs/clean_pipeline" \
  --targets Oscore Zrisk

快速调试：
python financial_risk_pipeline.py \
  --input "data/raw/financial_data.csv" \
  --output_dir "outputs/clean_pipeline_quick" \
  --targets Oscore Zrisk \
  --sample_train_rows 20000 \
  --rf_n_estimators 50 \
  --nn_epochs 10 \
  --permutation_repeats 1

只跑 Oscore：
python financial_risk_pipeline.py \
  --input "data/raw/financial_data.csv" \
  --output_dir "outputs/clean_pipeline_oscore" \
  --targets Oscore

"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

# 服务器 / Mac / Windows 上无界面运行时，matplotlib 可能报错。
# 这里强制使用 Agg 后端，保证脚本在命令行和服务器上都能运行。
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler

warnings.filterwarnings("ignore")

# ============================================================
# 0. 可选依赖：LightGBM 和 PyTorch
# ============================================================

LIGHTGBM_AVAILABLE = False
try:
    import lightgbm as lgb
    from lightgbm import LGBMRegressor
    LIGHTGBM_AVAILABLE = True
except Exception:
    LIGHTGBM_AVAILABLE = False

TORCH_AVAILABLE = False
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False


# ============================================================
# 1. 全局配置
# ============================================================

@dataclass
class PipelineConfig:
    """把所有会影响实验结果的参数集中管理，方便复现和修改。"""

    input_path: str
    output_dir: str
    targets: Tuple[str, ...] = ("Oscore", "Zrisk")

    # 切分比例。本文报告采用约 70% / 15% / 15%。
    train_size: float = 0.70
    valid_size: float = 0.15
    test_size: float = 0.15
    random_state: int = 42

    # 原始数据中的标识列、分类列和目标列。
    id_cols: Tuple[str, ...] = ("股票代码", "股票简称")
    categorical_cols: Tuple[str, ...] = ("行业名称1", "报表类型编码")
    industry_col: str = "行业名称1"
    stock_code_col: str = "股票代码"
    stock_name_col: str = "股票简称"
    target_cols_to_exclude: Tuple[str, ...] = ("Oscore", "Zscore", "Zrisk", "target")

    # 缺失值处理阈值。
    # 低缺失：全训练集中位数填补。
    # 中缺失：行业中位数填补 + 缺失指示变量。
    # 高缺失：默认删除。
    low_missing_threshold: float = 0.05
    mid_missing_threshold: float = 0.40
    drop_missing_threshold: float = 0.80
    keep_high_missing: bool = False

    # 极端值处理：训练集 1% / 99% 分位数缩尾。
    winsor_lower: float = 0.01
    winsor_upper: float = 0.99
    winsorize_y: bool = True

    # 对严重偏态变量做 signed_log1p 变换。
    use_signed_log: bool = True
    skew_threshold: float = 5.0

    # 标准化方式。
    scaler: str = "standard"  # standard / robust / none

    # 模型训练参数。
    sample_train_rows: Optional[int] = None
    sample_valid_rows: Optional[int] = None
    sample_test_rows: Optional[int] = None
    n_jobs: int = -1

    rf_n_estimators: int = 200
    rf_max_depth: Optional[int] = 12
    rf_min_samples_leaf: int = 20

    lgbm_n_estimators: int = 800
    lgbm_learning_rate: float = 0.03

    run_nn: bool = True
    nn_epochs: int = 60
    nn_batch_size: int = 512
    nn_learning_rate: float = 1e-3
    nn_patience: int = 12
    nn_dropout: float = 0.20
    nn_weight_decay: float = 1e-4

    # 置换重要性可能稍慢。默认只对最佳模型抽样计算。
    run_permutation: bool = True
    permutation_sample_rows: int = 3000
    permutation_repeats: int = 3

    save_models: bool = False
    save_plots: bool = False


# ============================================================
# 2. 通用工具函数
# ============================================================

def set_seed(seed: int) -> None:
    """固定随机种子，使结果尽量可复现。"""
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if TORCH_AVAILABLE:
        try:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        except Exception:
            pass


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_csv_safely(path: str | Path) -> pd.DataFrame:
    """自动尝试多种中文常见编码读取 CSV。"""
    path = Path(path)
    for enc in ["utf-8-sig", "utf-8", "gbk", "gb18030"]:
        try:
            df = pd.read_csv(path, encoding=enc)
            print(f"[读取成功] {path}，编码={enc}，shape={df.shape}")
            return df
        except Exception:
            continue
    raise RuntimeError(f"无法读取 CSV 文件：{path}")


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """去掉列名两端空格，避免 Excel 导出造成的隐藏问题。"""
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    return out


def standardize_stock_code(s: pd.Series) -> pd.Series:
    """股票代码统一处理为 6 位字符串。"""
    return (
        s.astype(str)
        .str.replace(r"\.0$", "", regex=True)
        .str.strip()
        .str.zfill(6)
    )


def coerce_numeric_columns(df: pd.DataFrame, cfg: PipelineConfig) -> pd.DataFrame:
    """
    尽量把财务指标列转换为数值。

    很多金融终端或 Excel 导出的数据可能包含逗号、百分号、空字符串等，
    例如 "1,234.5"、"12.3%"、"--"。这些需要先清洗，模型才能识别为数值。
    """
    out = df.copy()
    id_cols = set(cfg.id_cols)
    cat_cols = set(cfg.categorical_cols)

    for col in out.columns:
        if col in id_cols or col in cat_cols:
            continue
        if pd.api.types.is_numeric_dtype(out[col]):
            continue

        s = out[col].astype(str).str.strip()
        s = s.replace({"": np.nan, "nan": np.nan, "None": np.nan, "--": np.nan, "—": np.nan})
        s = s.str.replace(",", "", regex=False)
        percent_mask = s.str.endswith("%", na=False)
        s_num = pd.to_numeric(s.str.replace("%", "", regex=False), errors="coerce")
        if percent_mask.any():
            s_num.loc[percent_mask] = s_num.loc[percent_mask] / 100.0
        out[col] = s_num
    return out


def signed_log1p(x: pd.Series | np.ndarray) -> pd.Series | np.ndarray:
    """
    符号对数变换：sign(x) * log(1 + |x|)

    普通 log(1+x) 不能处理很多负数，但财务变量经常为负，
    比如利润增长率、EVA 率、现金流覆盖倍数。
    signed_log1p 既保留正负方向，又压缩极端值。
    """
    return np.sign(x) * np.log1p(np.abs(x))


def build_target(df: pd.DataFrame, target: str) -> pd.Series:
    """构造被解释变量。Zrisk 是 -Zscore，使其方向变成“越高风险越高”。"""
    if target == "Oscore":
        if "Oscore" not in df.columns:
            raise ValueError("数据中不存在 Oscore 列。")
        return pd.to_numeric(df["Oscore"], errors="coerce")
    if target == "Zscore":
        if "Zscore" not in df.columns:
            raise ValueError("数据中不存在 Zscore 列。")
        return pd.to_numeric(df["Zscore"], errors="coerce")
    if target == "Zrisk":
        if "Zscore" not in df.columns:
            raise ValueError("数据中不存在 Zscore 列，无法构造 Zrisk=-Zscore。")
        return -pd.to_numeric(df["Zscore"], errors="coerce")
    raise ValueError("target 只能是 Oscore、Zscore 或 Zrisk。")


def safe_stratify_labels(y: pd.Series, bins: int = 10) -> Optional[pd.Series]:
    """
    回归任务没有天然类别，不能像分类任务一样直接分层抽样。
    这里用 y 的分位数近似构造 10 组标签，让 train/valid/test 的目标分布更接近。
    """
    try:
        labels = pd.qcut(y.rank(method="first"), q=bins, duplicates="drop")
        if labels.nunique() < 2:
            return None
        return labels.astype(str)
    except Exception:
        return None


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """统一计算所有模型的评价指标。"""
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    mse = mean_squared_error(y_true, y_pred)
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        pearson = np.nan
    else:
        pearson = float(np.corrcoef(y_true, y_pred)[0, 1])

    r_true = pd.Series(y_true).rank(method="average")
    r_pred = pd.Series(y_pred).rank(method="average")
    if r_true.std() == 0 or r_pred.std() == 0:
        spearman = np.nan
    else:
        spearman = float(r_true.corr(r_pred, method="pearson"))

    top20_p, top20_r = top_quantile_precision_recall(y_true, y_pred, rate=0.20)
    top10_p, top10_r = top_quantile_precision_recall(y_true, y_pred, rate=0.10)

    return {
        "MSE": float(mse),
        "RMSE": float(rmse),
        "MAE": float(mae),
        "R2": float(r2),
        "Pearson": pearson,
        "Spearman": spearman,
        "Top20_Precision": top20_p,
        "Top20_Recall": top20_r,
        "Top10_Precision": top10_p,
        "Top10_Recall": top10_r,
    }


def top_quantile_precision_recall(y_true: np.ndarray, y_pred: np.ndarray, rate: float = 0.20) -> Tuple[float, float]:
    """
    高风险识别指标。

    真实风险最高的 rate 比例样本记作 H，模型预测风险最高的 rate 比例样本记作 H_hat。
    Precision：模型预测高风险样本中，有多少真的属于高风险。
    Recall：真实高风险样本中，有多少被模型找出来。
    """
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    true_cut = np.quantile(y_true, 1 - rate)
    pred_cut = np.quantile(y_pred, 1 - rate)
    true_high = y_true >= true_cut
    pred_high = y_pred >= pred_cut
    tp = np.sum(true_high & pred_high)
    precision = tp / np.sum(pred_high) if np.sum(pred_high) > 0 else np.nan
    recall = tp / np.sum(true_high) if np.sum(true_high) > 0 else np.nan
    return float(precision), float(recall)


def sample_for_training(X: pd.DataFrame, y: np.ndarray, n: Optional[int], seed: int) -> Tuple[pd.DataFrame, np.ndarray]:
    """可选抽样，方便快速调试。正式报告建议不用抽样。"""
    if n is None or len(X) <= n:
        return X, y
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=n, replace=False)
    return X.iloc[idx].reset_index(drop=True), y[idx]


def to_jsonable(obj: Any) -> Any:
    """把 numpy / pandas 对象转成 JSON 可保存的普通 Python 对象。"""
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_jsonable(x) for x in obj]
    if isinstance(obj, tuple):
        return [to_jsonable(x) for x in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        if np.isnan(obj):
            return None
        return float(obj)
    if pd.isna(obj):
        return None
    return obj


# ============================================================
# 3. 无泄漏预处理器
# ============================================================

class CleanFinancialPreprocessor:
    """
    本项目专用的无泄漏预处理器。

    核心原则：
    - fit() 只能在训练集上调用；
    - transform() 把训练集学到的规则应用到训练集、验证集和测试集；
    - 验证集和测试集不能参与任何填补值、缩尾阈值、标准化均值等参数的估计。
    """

    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self.numeric_cols_all: List[str] = []
        self.numeric_cols_kept: List[str] = []
        self.numeric_cols_dropped: List[str] = []
        self.categorical_cols: List[str] = []

        self.missing_rates: Dict[str, float] = {}
        self.missing_actions: Dict[str, str] = {}
        self.global_medians: Dict[str, float] = {}
        self.industry_medians: Dict[str, Dict[str, float]] = {}
        self.industry_fallback: Dict[str, float] = {}
        self.missing_indicator_cols: List[str] = []

        self.winsor_bounds: Dict[str, Tuple[float, float]] = {}
        self.signed_log_cols: List[str] = []
        self.scaler: Optional[Any] = None
        self.encoder: Optional[OneHotEncoder] = None
        self.feature_names: List[str] = []

    def fit(self, train_df: pd.DataFrame) -> "CleanFinancialPreprocessor":
        cfg = self.cfg

        # 1）确定哪些列是数值特征，哪些列是分类特征。
        exclude = set(cfg.id_cols) | set(cfg.target_cols_to_exclude)
        self.categorical_cols = [c for c in cfg.categorical_cols if c in train_df.columns]
        exclude.update(self.categorical_cols)
        self.numeric_cols_all = [
            c for c in train_df.columns
            if c not in exclude and pd.api.types.is_numeric_dtype(train_df[c])
        ]

        # 2）根据训练集缺失率决定每个数值特征怎么处理。
        for col in self.numeric_cols_all:
            miss_rate = float(train_df[col].isna().mean())
            self.missing_rates[col] = miss_rate

            if miss_rate > cfg.drop_missing_threshold:
                self.missing_actions[col] = "删除：缺失率超过 80%，信息含量较低"
                self.numeric_cols_dropped.append(col)
            elif miss_rate > cfg.mid_missing_threshold and not cfg.keep_high_missing:
                self.missing_actions[col] = "删除：缺失率在 40%-80%，主模型默认不保留"
                self.numeric_cols_dropped.append(col)
            else:
                self.numeric_cols_kept.append(col)
                if miss_rate <= cfg.low_missing_threshold:
                    self.missing_actions[col] = "全训练集中位数填补"
                else:
                    self.missing_actions[col] = "行业中位数填补 + 缺失指示变量"
                    self.missing_indicator_cols.append(col)

        # 3）只在训练集上计算填补值。
        for col in self.numeric_cols_kept:
            self.global_medians[col] = float(train_df[col].median()) if not pd.isna(train_df[col].median()) else 0.0
            self.industry_fallback[col] = self.global_medians[col]
            if cfg.industry_col in train_df.columns:
                med = train_df.groupby(cfg.industry_col)[col].median().dropna()
                self.industry_medians[col] = {str(k): float(v) for k, v in med.items()}
            else:
                self.industry_medians[col] = {}

        # 4）用训练集填补后的数值，计算缩尾阈值、偏态变量和标准化参数。
        train_num = self._impute_numeric(train_df, fit_stage=True)

        for col in self.numeric_cols_kept:
            lo = float(train_num[col].quantile(cfg.winsor_lower))
            hi = float(train_num[col].quantile(cfg.winsor_upper))
            if pd.isna(lo) or pd.isna(hi) or lo > hi:
                lo, hi = -np.inf, np.inf
            self.winsor_bounds[col] = (lo, hi)
            train_num[col] = train_num[col].clip(lo, hi)

        if cfg.use_signed_log:
            for col in self.numeric_cols_kept:
                try:
                    skew = float(train_num[col].skew())
                except Exception:
                    skew = 0.0
                if abs(skew) >= cfg.skew_threshold:
                    self.signed_log_cols.append(col)
                    train_num[col] = signed_log1p(train_num[col])

        scale_cols = self.numeric_cols_kept + [f"{c}_missing_ind" for c in self.missing_indicator_cols]
        if cfg.scaler == "standard":
            self.scaler = StandardScaler()
            self.scaler.fit(train_num[scale_cols])
        elif cfg.scaler == "robust":
            self.scaler = RobustScaler()
            self.scaler.fit(train_num[scale_cols])
        else:
            self.scaler = None

        # 5）分类变量 One-Hot 编码，也只在训练集上 fit 类别。
        if self.categorical_cols:
            cat_train = train_df[self.categorical_cols].copy().fillna("未知")
            try:
                self.encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            except TypeError:
                self.encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
            self.encoder.fit(cat_train)
        else:
            self.encoder = None

        # 6）记录最终进入模型的特征名。
        num_feature_names = scale_cols
        cat_feature_names: List[str] = []
        if self.encoder is not None:
            cat_feature_names = list(self.encoder.get_feature_names_out(self.categorical_cols))
        self.feature_names = num_feature_names + cat_feature_names
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """把训练集上学到的规则应用到任意数据集。"""
        num = self._impute_numeric(df, fit_stage=False)

        # 缩尾：使用训练集 1% / 99% 阈值。
        for col in self.numeric_cols_kept:
            lo, hi = self.winsor_bounds.get(col, (-np.inf, np.inf))
            num[col] = num[col].clip(lo, hi)

        # 偏态变换：使用训练集判断出的 signed_log 变量列表。
        for col in self.signed_log_cols:
            if col in num.columns:
                num[col] = signed_log1p(num[col])

        scale_cols = self.numeric_cols_kept + [f"{c}_missing_ind" for c in self.missing_indicator_cols]
        if self.scaler is not None:
            num_scaled = pd.DataFrame(
                self.scaler.transform(num[scale_cols]),
                columns=scale_cols,
                index=df.index,
            )
        else:
            num_scaled = num[scale_cols].copy()

        if self.encoder is not None:
            cat = df[self.categorical_cols].copy().fillna("未知")
            cat_arr = self.encoder.transform(cat)
            cat_names = list(self.encoder.get_feature_names_out(self.categorical_cols))
            cat_df = pd.DataFrame(cat_arr, columns=cat_names, index=df.index)
            out = pd.concat([num_scaled, cat_df], axis=1)
        else:
            out = num_scaled

        # 保证列顺序固定，验证集/测试集和训练集完全一致。
        out = out[self.feature_names]
        return out.reset_index(drop=True)

    def _impute_numeric(self, df: pd.DataFrame, fit_stage: bool) -> pd.DataFrame:
        """执行数值特征填补，并添加缺失指示变量。"""
        out = pd.DataFrame(index=df.index)
        industry_values = df[self.cfg.industry_col].astype(str) if self.cfg.industry_col in df.columns else None

        for col in self.numeric_cols_kept:
            s = df[col].copy()
            miss = s.isna()

            # 低缺失变量：用训练集全局中位数填补。
            if col not in self.missing_indicator_cols:
                fill_value = self.global_medians.get(col, 0.0)
                out[col] = s.fillna(fill_value)
            else:
                # 中等缺失变量：优先使用训练集中同一行业的中位数。
                filled = s.copy()
                if industry_values is not None:
                    med_map = self.industry_medians.get(col, {})
                    industry_fill = industry_values.map(lambda x: med_map.get(str(x), np.nan))
                    filled = filled.fillna(industry_fill)
                filled = filled.fillna(self.industry_fallback.get(col, self.global_medians.get(col, 0.0)))
                out[col] = filled
                out[f"{col}_missing_ind"] = miss.astype(float)

            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(self.global_medians.get(col, 0.0))

        return out

    def summary_table(self) -> pd.DataFrame:
        """输出每个变量的缺失率和处理方式，便于写报告。"""
        rows = []
        for col in self.numeric_cols_all:
            rows.append({
                "变量": col,
                "训练集缺失率": self.missing_rates.get(col, np.nan),
                "是否保留": col in self.numeric_cols_kept,
                "处理方式": self.missing_actions.get(col, ""),
                "是否添加缺失指示变量": col in self.missing_indicator_cols,
                "是否signed_log变换": col in self.signed_log_cols,
                "缩尾1%下界": self.winsor_bounds.get(col, (np.nan, np.nan))[0] if col in self.winsor_bounds else np.nan,
                "缩尾99%上界": self.winsor_bounds.get(col, (np.nan, np.nan))[1] if col in self.winsor_bounds else np.nan,
            })
        return pd.DataFrame(rows)


# ============================================================
# 4. 数据审计与切分
# ============================================================

def basic_audit(raw_df: pd.DataFrame, cfg: PipelineConfig, output_dir: Path) -> None:
    """保存最核心的数据审计表：原始结构、缺失率、相关性。"""
    audit_dir = ensure_dir(output_dir / "00_audit")

    rows = []
    for col in raw_df.columns:
        rows.append({
            "变量": col,
            "dtype": str(raw_df[col].dtype),
            "缺失数": int(raw_df[col].isna().sum()),
            "缺失率": float(raw_df[col].isna().mean()),
            "唯一值数量": int(raw_df[col].nunique(dropna=True)),
        })
    pd.DataFrame(rows).sort_values("缺失率", ascending=False).to_csv(
        audit_dir / "missing_rate_summary.csv", index=False, encoding="utf-8-sig"
    )

    # 数值变量与 Oscore/Zscore 的简单相关性，供报告理解变量关系。
    num_df = raw_df.select_dtypes(include=[np.number])
    corr_rows = []
    for target in ["Oscore", "Zscore"]:
        if target in num_df.columns:
            for col in num_df.columns:
                if col == target:
                    continue
                corr = num_df[[col, target]].corr(method="pearson").iloc[0, 1]
                corr_rows.append({"目标变量": target, "变量": col, "Pearson相关系数": corr})
    if corr_rows:
        pd.DataFrame(corr_rows).sort_values(
            ["目标变量", "Pearson相关系数"], ascending=[True, False]
        ).to_csv(audit_dir / "target_correlation_summary.csv", index=False, encoding="utf-8-sig")

    summary = {
        "原始行数": int(raw_df.shape[0]),
        "原始列数": int(raw_df.shape[1]),
        "列名": list(raw_df.columns),
        "说明": "本脚本只保存核心审计表，不再输出大量热力图和中间图。",
    }
    (audit_dir / "audit_summary.json").write_text(
        json.dumps(to_jsonable(summary), ensure_ascii=False, indent=2), encoding="utf-8"
    )


def split_data_for_target(df: pd.DataFrame, target: str, cfg: PipelineConfig) -> Dict[str, Any]:
    """删除 Y 缺失样本，然后切分 train / valid / test。"""
    work = df.copy()
    work["target"] = build_target(work, target)
    work = work[work["target"].notna()].reset_index(drop=True)

    # 近似分层，让 train/valid/test 的目标变量分布更平衡。
    labels = safe_stratify_labels(work["target"])
    idx = np.arange(len(work))

    test_ratio = cfg.test_size
    train_valid_idx, test_idx = train_test_split(
        idx,
        test_size=test_ratio,
        random_state=cfg.random_state,
        stratify=labels.iloc[idx] if labels is not None else None,
    )

    valid_ratio_within_train_valid = cfg.valid_size / (cfg.train_size + cfg.valid_size)
    labels_tv = labels.iloc[train_valid_idx] if labels is not None else None
    train_idx, valid_idx = train_test_split(
        train_valid_idx,
        test_size=valid_ratio_within_train_valid,
        random_state=cfg.random_state,
        stratify=labels_tv if labels_tv is not None else None,
    )

    train_df = work.iloc[train_idx].reset_index(drop=True)
    valid_df = work.iloc[valid_idx].reset_index(drop=True)
    test_df = work.iloc[test_idx].reset_index(drop=True)

    # 目标变量也只用训练集阈值进行缩尾，避免验证集/测试集信息泄露。
    if cfg.winsorize_y:
        y_lo = float(train_df["target"].quantile(cfg.winsor_lower))
        y_hi = float(train_df["target"].quantile(cfg.winsor_upper))
        for part in [train_df, valid_df, test_df]:
            part["target_original"] = part["target"]
            part["target"] = part["target"].clip(y_lo, y_hi)
    else:
        y_lo, y_hi = np.nan, np.nan
        for part in [train_df, valid_df, test_df]:
            part["target_original"] = part["target"]

    return {
        "train_df": train_df,
        "valid_df": valid_df,
        "test_df": test_df,
        "target_winsor_bounds": (y_lo, y_hi),
    }


def summarize_split(train_df: pd.DataFrame, valid_df: pd.DataFrame, test_df: pd.DataFrame, cfg: PipelineConfig) -> pd.DataFrame:
    """生成样本切分摘要，和报告中的表 1 / 表 2 类似。"""
    rows = []
    for name, part in [("train", train_df), ("valid", valid_df), ("test", test_df)]:
        row = {
            "样本集": name,
            "样本数": len(part),
            "Y均值": part["target"].mean(),
            "Y标准差": part["target"].std(),
            "Y最小值": part["target"].min(),
            "Y中位数": part["target"].median(),
            "Y最大值": part["target"].max(),
        }
        if cfg.stock_code_col in part.columns:
            row["股票数"] = part[cfg.stock_code_col].nunique()
        if cfg.industry_col in part.columns:
            row["行业数"] = part[cfg.industry_col].nunique()
        rows.append(row)
    return pd.DataFrame(rows)


# ============================================================
# 5. 传统机器学习模型
# ============================================================

def train_sklearn_models(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_valid: pd.DataFrame,
    y_valid: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    cfg: PipelineConfig,
) -> List[Dict[str, Any]]:
    """训练 Dummy、线性模型、正则化模型、树模型。"""
    results: List[Dict[str, Any]] = []

    X_fit, y_fit = sample_for_training(X_train, y_train, cfg.sample_train_rows, cfg.random_state)

    model_defs: List[Tuple[str, str, Any]] = [
        ("dummy_mean", "Dummy", DummyRegressor(strategy="mean")),
        ("dummy_median", "Dummy", DummyRegressor(strategy="median")),
        ("linear", "Linear", LinearRegression()),
        ("ridge", "Linear-Regularized", RidgeCV(alphas=np.logspace(-4, 4, 20))),
        ("lasso", "Linear-Regularized", LassoCV(alphas=np.logspace(-4, 1, 20), cv=3, max_iter=5000, random_state=cfg.random_state)),
        ("elasticnet", "Linear-Regularized", ElasticNetCV(alphas=np.logspace(-4, 1, 15), l1_ratio=[0.2, 0.5, 0.8], cv=3, max_iter=5000, random_state=cfg.random_state)),
        ("random_forest", "Tree", RandomForestRegressor(
            n_estimators=cfg.rf_n_estimators,
            max_depth=cfg.rf_max_depth,
            min_samples_leaf=cfg.rf_min_samples_leaf,
            max_features=0.7,
            n_jobs=cfg.n_jobs,
            random_state=cfg.random_state,
        )),
    ]

    if LIGHTGBM_AVAILABLE:
        model_defs.append(("lightgbm", "Tree", LGBMRegressor(
            n_estimators=cfg.lgbm_n_estimators,
            learning_rate=cfg.lgbm_learning_rate,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=cfg.random_state,
            n_jobs=cfg.n_jobs,
            verbosity=-1,
        )))
    else:
        model_defs.append(("lightgbm_fallback_histgb", "Tree", HistGradientBoostingRegressor(
            max_iter=300,
            learning_rate=0.05,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            random_state=cfg.random_state,
        )))

    for name, family, model in model_defs:
        print(f"  [训练模型] {name}")
        start = time.time()
        try:
            # LightGBM 可以使用验证集 early stopping；若当前版本不支持 callbacks，则退回普通 fit。
            if name == "lightgbm" and LIGHTGBM_AVAILABLE:
                try:
                    model.fit(
                        X_fit, y_fit,
                        eval_set=[(X_valid, y_valid)],
                        eval_metric="rmse",
                        callbacks=[lgb.early_stopping(50, verbose=False)],
                    )
                except TypeError:
                    model.fit(X_fit, y_fit)
            else:
                model.fit(X_fit, y_fit)
        except Exception as exc:
            print(f"    [失败跳过] {name}: {exc}")
            continue
        train_time = time.time() - start

        pred_valid = model.predict(X_valid)
        pred_test = model.predict(X_test)
        metrics = compute_metrics(y_test, pred_test)
        valid_metrics = compute_metrics(y_valid, pred_valid)

        results.append({
            "model": name,
            "model_family": family,
            "estimator": model,
            "predict_fn": lambda X, m=model: m.predict(X),
            "train_time_sec": train_time,
            "valid_metrics": valid_metrics,
            "test_metrics": metrics,
            "test_pred": pred_test,
        })
        print(f"    RMSE={metrics['RMSE']:.4f}, R2={metrics['R2']:.4f}, Spearman={metrics['Spearman']:.4f}")
    return results


# ============================================================
# 6. 神经网络模型
# ============================================================

if TORCH_AVAILABLE:
    class TorchMLP(nn.Module):
        """简单多层感知机，用于回归预测。"""
        def __init__(self, input_dim: int, hidden_layers: List[int], dropout: float = 0.0, batchnorm: bool = False):
            super().__init__()
            layers: List[nn.Module] = []
            prev = input_dim
            for h in hidden_layers:
                layers.append(nn.Linear(prev, h))
                if batchnorm:
                    layers.append(nn.BatchNorm1d(h))
                layers.append(nn.ReLU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
                prev = h
            layers.append(nn.Linear(prev, 1))
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            return self.net(x).view(-1)


def train_torch_nn(
    name: str,
    hidden_layers: List[int],
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_valid: pd.DataFrame,
    y_valid: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    cfg: PipelineConfig,
    improved: bool,
) -> Dict[str, Any]:
    """使用 PyTorch 训练基准 NN 或改进 NN。"""
    device = "cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu"
    X_fit, y_fit = sample_for_training(X_train, y_train, cfg.sample_train_rows, cfg.random_state)

    Xtr = torch.tensor(X_fit.values.astype(np.float32), dtype=torch.float32)
    ytr = torch.tensor(y_fit.astype(np.float32), dtype=torch.float32)
    Xva = torch.tensor(X_valid.values.astype(np.float32), dtype=torch.float32).to(device)
    yva_np = y_valid.astype(np.float32)
    Xte = torch.tensor(X_test.values.astype(np.float32), dtype=torch.float32).to(device)

    ds = TensorDataset(Xtr, ytr)
    loader = DataLoader(ds, batch_size=cfg.nn_batch_size, shuffle=True)

    model = TorchMLP(
        input_dim=X_train.shape[1],
        hidden_layers=hidden_layers,
        dropout=cfg.nn_dropout if improved else 0.0,
        batchnorm=improved,
    ).to(device)

    criterion = nn.HuberLoss(delta=1.0) if improved else nn.MSELoss()
    if improved:
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.nn_learning_rate, weight_decay=cfg.nn_weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.nn_learning_rate)

    best_state = None
    best_loss = np.inf
    wait = 0
    start = time.time()

    for epoch in range(1, cfg.nn_epochs + 1):
        model.train()
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        # 用验证集做 early stopping。基准 NN 也可用，但 improved 更强调这个机制。
        model.eval()
        with torch.no_grad():
            pred_va = model(Xva).detach().cpu().numpy()
        val_rmse = math.sqrt(mean_squared_error(yva_np, pred_va))
        if val_rmse < best_loss - 1e-6:
            best_loss = val_rmse
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
        if improved and wait >= cfg.nn_patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    train_time = time.time() - start

    def predict_fn(X: pd.DataFrame) -> np.ndarray:
        model.eval()
        with torch.no_grad():
            Xt = torch.tensor(X.values.astype(np.float32), dtype=torch.float32).to(device)
            return model(Xt).detach().cpu().numpy()

    pred_valid = predict_fn(X_valid)
    pred_test = predict_fn(X_test)
    return {
        "model": name,
        "model_family": "Neural Network",
        "estimator": model,
        "predict_fn": predict_fn,
        "train_time_sec": train_time,
        "valid_metrics": compute_metrics(y_valid, pred_valid),
        "test_metrics": compute_metrics(y_test, pred_test),
        "test_pred": pred_test,
    }


def train_sklearn_mlp_fallback(
    name: str,
    hidden_layers: Tuple[int, ...],
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_valid: pd.DataFrame,
    y_valid: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    cfg: PipelineConfig,
    improved: bool,
) -> Dict[str, Any]:
    """没有 PyTorch 时，用 sklearn MLPRegressor 作为简化版神经网络。"""
    from sklearn.neural_network import MLPRegressor

    X_fit, y_fit = sample_for_training(X_train, y_train, cfg.sample_train_rows, cfg.random_state)
    model = MLPRegressor(
        hidden_layer_sizes=hidden_layers,
        activation="relu",
        solver="adam",
        alpha=cfg.nn_weight_decay if improved else 1e-5,
        batch_size=cfg.nn_batch_size,
        learning_rate_init=cfg.nn_learning_rate,
        max_iter=cfg.nn_epochs,
        early_stopping=improved,
        n_iter_no_change=cfg.nn_patience,
        random_state=cfg.random_state,
    )
    start = time.time()
    model.fit(X_fit, y_fit)
    train_time = time.time() - start
    pred_valid = model.predict(X_valid)
    pred_test = model.predict(X_test)
    return {
        "model": name,
        "model_family": "Neural Network",
        "estimator": model,
        "predict_fn": lambda X, m=model: m.predict(X),
        "train_time_sec": train_time,
        "valid_metrics": compute_metrics(y_valid, pred_valid),
        "test_metrics": compute_metrics(y_test, pred_test),
        "test_pred": pred_test,
    }


def train_nn_models(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_valid: pd.DataFrame,
    y_valid: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    cfg: PipelineConfig,
) -> List[Dict[str, Any]]:
    """训练基准结构 NN 和一个改进 NN。"""
    if not cfg.run_nn:
        return []

    results: List[Dict[str, Any]] = []
    nn_defs = [
        ("teacher_nn_64_30", [64, 30], False),
        ("improved_nn_256_128_64", [256, 128, 64], True),
    ]

    for name, hidden, improved in nn_defs:
        print(f"  [训练神经网络] {name}")
        try:
            if TORCH_AVAILABLE:
                res = train_torch_nn(name, hidden, X_train, y_train, X_valid, y_valid, X_test, y_test, cfg, improved)
            else:
                res = train_sklearn_mlp_fallback(name, tuple(hidden), X_train, y_train, X_valid, y_valid, X_test, y_test, cfg, improved)
            results.append(res)
            m = res["test_metrics"]
            print(f"    RMSE={m['RMSE']:.4f}, R2={m['R2']:.4f}, Spearman={m['Spearman']:.4f}")
        except Exception as exc:
            print(f"    [失败跳过] {name}: {exc}")
    return results


# ============================================================
# 7. 解释分析：特征重要性、分组分析、残差分析
# ============================================================

def result_rows(results: List[Dict[str, Any]], target: str, n_train: int, n_valid: int, n_test: int, n_features: int) -> pd.DataFrame:
    """把模型结果整理成统一表格。"""
    rows = []
    for res in results:
        row = {
            "target": target,
            "model": res["model"],
            "model_family": res["model_family"],
            "train_time_sec": res["train_time_sec"],
            "n_train": n_train,
            "n_valid": n_valid,
            "n_test": n_test,
            "n_features": n_features,
        }
        for k, v in res["valid_metrics"].items():
            row[f"valid_{k}"] = v
        for k, v in res["test_metrics"].items():
            row[k] = v
        rows.append(row)
    df = pd.DataFrame(rows)
    # 默认按 RMSE 越低越好排序。
    return df.sort_values("RMSE", ascending=True).reset_index(drop=True)


def choose_best_result(results: List[Dict[str, Any]], metric: str = "RMSE") -> Dict[str, Any]:
    """选择最佳模型。默认 RMSE 最低。"""
    if metric in ["RMSE", "MAE", "MSE"]:
        return sorted(results, key=lambda r: r["test_metrics"].get(metric, np.inf))[0]
    return sorted(results, key=lambda r: r["test_metrics"].get(metric, -np.inf), reverse=True)[0]


def built_in_feature_importance(model: Any, feature_names: List[str]) -> pd.DataFrame:
    """读取树模型或线性模型的内置重要性。"""
    if hasattr(model, "feature_importances_"):
        imp = np.asarray(model.feature_importances_, dtype=float)
    elif hasattr(model, "coef_"):
        imp = np.abs(np.asarray(model.coef_).reshape(-1))
    else:
        return pd.DataFrame()

    if len(imp) != len(feature_names):
        return pd.DataFrame()
    return pd.DataFrame({"feature": feature_names, "importance": imp}).sort_values("importance", ascending=False)


def permutation_importance_core(
    predict_fn: Callable[[pd.DataFrame], np.ndarray],
    X: pd.DataFrame,
    y: np.ndarray,
    cfg: PipelineConfig,
    seed: int,
) -> pd.DataFrame:
    """
    手写置换重要性。

    方法：先计算原模型的 RMSE；然后逐个变量打乱，观察 RMSE 上升多少。
    上升越多，说明该变量对模型越重要。
    """
    if not cfg.run_permutation:
        return pd.DataFrame()

    rng = np.random.default_rng(seed)
    if len(X) > cfg.permutation_sample_rows:
        idx = rng.choice(len(X), size=cfg.permutation_sample_rows, replace=False)
        X_eval = X.iloc[idx].reset_index(drop=True).copy()
        y_eval = y[idx]
    else:
        X_eval = X.reset_index(drop=True).copy()
        y_eval = y

    base_pred = predict_fn(X_eval)
    base_rmse = math.sqrt(mean_squared_error(y_eval, base_pred))

    rows = []
    for feature in X_eval.columns:
        scores = []
        for _ in range(cfg.permutation_repeats):
            X_perm = X_eval.copy()
            vals = X_perm[feature].values.copy()
            rng.shuffle(vals)
            X_perm[feature] = vals
            pred = predict_fn(X_perm)
            rmse = math.sqrt(mean_squared_error(y_eval, pred))
            scores.append(rmse)
        rows.append({
            "feature": feature,
            "importance": float(np.mean(scores) - base_rmse),
            "base_rmse": base_rmse,
            "permuted_rmse_mean": float(np.mean(scores)),
            "permuted_rmse_std": float(np.std(scores)),
        })
    return pd.DataFrame(rows).sort_values("importance", ascending=False).reset_index(drop=True)


def risk_group_analysis(y_true: np.ndarray, y_pred: np.ndarray, n_groups: int = 10) -> pd.DataFrame:
    """按预测风险从低到高分组，观察真实风险是否也随之上升。"""
    df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred})
    df["residual"] = df["y_true"] - df["y_pred"]
    df["abs_error"] = df["residual"].abs()
    df["pred_risk_group"] = pd.qcut(df["y_pred"].rank(method="first"), q=n_groups, labels=False) + 1

    rows = []
    for g, part in df.groupby("pred_risk_group"):
        rows.append({
            "pred_risk_group": int(g),
            "n": len(part),
            "y_true_mean": part["y_true"].mean(),
            "y_pred_mean": part["y_pred"].mean(),
            "residual_mean": part["residual"].mean(),
            "MAE": part["abs_error"].mean(),
            "RMSE": math.sqrt(mean_squared_error(part["y_true"], part["y_pred"])),
        })
    return pd.DataFrame(rows)


def true_group_error_analysis(y_true: np.ndarray, y_pred: np.ndarray, n_groups: int = 10) -> pd.DataFrame:
    """按真实风险分位分组，看极端风险组是否更难预测。"""
    df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred})
    df["residual"] = df["y_true"] - df["y_pred"]
    df["abs_error"] = df["residual"].abs()
    df["true_risk_group"] = pd.qcut(df["y_true"].rank(method="first"), q=n_groups, labels=False) + 1
    rows = []
    for g, part in df.groupby("true_risk_group"):
        rows.append({
            "true_risk_group": int(g),
            "n": len(part),
            "y_true_mean": part["y_true"].mean(),
            "y_pred_mean": part["y_pred"].mean(),
            "MAE": part["abs_error"].mean(),
            "RMSE": math.sqrt(mean_squared_error(part["y_true"], part["y_pred"])),
        })
    return pd.DataFrame(rows)


def residual_by_industry(test_df: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray, cfg: PipelineConfig) -> pd.DataFrame:
    """按行业统计误差，观察行业异质性。"""
    if cfg.industry_col not in test_df.columns:
        return pd.DataFrame()
    df = test_df[[cfg.industry_col]].copy()
    df["y_true"] = y_true
    df["y_pred"] = y_pred
    df["residual"] = df["y_true"] - df["y_pred"]
    df["abs_error"] = df["residual"].abs()
    rows = []
    for industry, part in df.groupby(cfg.industry_col):
        if len(part) < 20:
            continue
        rows.append({
            "industry": industry,
            "n": len(part),
            "MAE": part["abs_error"].mean(),
            "RMSE": math.sqrt(mean_squared_error(part["y_true"], part["y_pred"])),
            "residual_mean": part["residual"].mean(),
            "y_true_mean": part["y_true"].mean(),
            "y_pred_mean": part["y_pred"].mean(),
        })
    return pd.DataFrame(rows).sort_values("MAE", ascending=False).reset_index(drop=True)


# ============================================================
# 8. 可选核心图
# ============================================================

def save_core_plots(target_dir: Path, target: str, metrics_df: pd.DataFrame, fi_df: pd.DataFrame, group_df: pd.DataFrame) -> None:
    """报告可能用到的核心图。"""
    plot_dir = ensure_dir(target_dir / "plots")

    # 1）RMSE 模型对比。
    if "RMSE" in metrics_df.columns:
        p = metrics_df.sort_values("RMSE", ascending=True).head(12).iloc[::-1]
        plt.figure(figsize=(8, max(4, 0.35 * len(p))))
        plt.barh(p["model"], p["RMSE"])
        plt.xlabel("RMSE（越低越好）")
        plt.title(f"{target} 模型 RMSE 对比")
        plt.tight_layout()
        plt.savefig(plot_dir / f"{target}_model_rmse.png", dpi=200)
        plt.close()

    # 2）Spearman 模型对比。
    if "Spearman" in metrics_df.columns:
        p = metrics_df.dropna(subset=["Spearman"]).sort_values("Spearman", ascending=False).head(12).iloc[::-1]
        plt.figure(figsize=(8, max(4, 0.35 * len(p))))
        plt.barh(p["model"], p["Spearman"])
        plt.xlabel("Spearman（越高越好）")
        plt.title(f"{target} 模型 Spearman 对比")
        plt.tight_layout()
        plt.savefig(plot_dir / f"{target}_model_spearman.png", dpi=200)
        plt.close()

    # 3）特征重要性。
    if not fi_df.empty:
        p = fi_df.head(15).iloc[::-1]
        plt.figure(figsize=(8, max(4, 0.38 * len(p))))
        plt.barh(p["feature"], p["importance"])
        plt.xlabel("importance")
        plt.title(f"{target} 特征重要性 Top 15")
        plt.tight_layout()
        plt.savefig(plot_dir / f"{target}_feature_importance_top15.png", dpi=200)
        plt.close()

    # 4）预测风险分组。
    if not group_df.empty:
        plt.figure(figsize=(7, 4.5))
        plt.plot(group_df["pred_risk_group"], group_df["y_true_mean"], marker="o", label="真实风险均值")
        plt.plot(group_df["pred_risk_group"], group_df["y_pred_mean"], marker="o", label="预测风险均值")
        plt.xlabel("预测风险分组：从低到高")
        plt.ylabel("风险得分均值")
        plt.title(f"{target} 预测风险分组分析")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_dir / f"{target}_risk_group_analysis.png", dpi=200)
        plt.close()


# ============================================================
# 9. 单个目标变量实验主流程
# ============================================================

def run_one_target(raw_df: pd.DataFrame, target: str, cfg: PipelineConfig, output_dir: Path) -> Dict[str, Any]:
    print("\n" + "=" * 90)
    print(f"开始运行目标变量：{target}")
    print("=" * 90)

    target_dir = ensure_dir(output_dir / target.lower())

    # Step 1：删除 Y 缺失并切分数据。
    split = split_data_for_target(raw_df, target, cfg)
    train_df, valid_df, test_df = split["train_df"], split["valid_df"], split["test_df"]

    split_summary = summarize_split(train_df, valid_df, test_df, cfg)
    split_summary.to_csv(target_dir / "split_summary.csv", index=False, encoding="utf-8-sig")

    # Step 2：只在训练集 fit 预处理规则，再 transform 三个集合。
    pre = CleanFinancialPreprocessor(cfg)
    pre.fit(train_df)
    X_train = pre.transform(train_df)
    X_valid = pre.transform(valid_df)
    X_test = pre.transform(test_df)
    y_train = train_df["target"].values.astype(float)
    y_valid = valid_df["target"].values.astype(float)
    y_test = test_df["target"].values.astype(float)

    pre.summary_table().to_csv(target_dir / "preprocess_variable_summary.csv", index=False, encoding="utf-8-sig")
    (target_dir / "feature_names.txt").write_text("\n".join(pre.feature_names), encoding="utf-8")

    print(f"预处理完成：train={X_train.shape}, valid={X_valid.shape}, test={X_test.shape}")

    # Step 3：训练传统模型。
    print("\n[模型训练] 传统机器学习模型")
    results = train_sklearn_models(X_train, y_train, X_valid, y_valid, X_test, y_test, cfg)

    # Step 4：训练神经网络。
    print("\n[模型训练] 神经网络模型")
    results.extend(train_nn_models(X_train, y_train, X_valid, y_valid, X_test, y_test, cfg))

    if not results:
        raise RuntimeError(f"目标 {target} 没有任何模型成功训练。")

    # Step 5：统一评价。
    metrics_df = result_rows(results, target, len(X_train), len(X_valid), len(X_test), X_train.shape[1])
    metrics_df.to_csv(target_dir / "model_metrics.csv", index=False, encoding="utf-8-sig")

    best = choose_best_result(results, metric="RMSE")
    print(f"\n[最佳模型] {target}: {best['model']}，RMSE={best['test_metrics']['RMSE']:.4f}, R2={best['test_metrics']['R2']:.4f}, Spearman={best['test_metrics']['Spearman']:.4f}")

    # Step 6：保存最佳模型预测与解释分析。
    best_pred = np.asarray(best["test_pred"]).reshape(-1)
    pred_df = pd.DataFrame({
        "y_true": y_test,
        "y_pred": best_pred,
        "residual": y_test - best_pred,
        "abs_error": np.abs(y_test - best_pred),
    })
    pred_df.to_csv(target_dir / "best_model_predictions.csv", index=False, encoding="utf-8-sig")

    # 优先用模型内置重要性；如果没有，则用 permutation importance。
    fi_builtin = built_in_feature_importance(best["estimator"], pre.feature_names)
    fi_perm = permutation_importance_core(best["predict_fn"], X_test, y_test, cfg, cfg.random_state)
    if not fi_perm.empty:
        fi_perm.to_csv(target_dir / "permutation_importance.csv", index=False, encoding="utf-8-sig")
        fi_main = fi_perm.copy()
    else:
        fi_main = fi_builtin.copy()
    if not fi_builtin.empty:
        fi_builtin.to_csv(target_dir / "builtin_feature_importance.csv", index=False, encoding="utf-8-sig")
    if not fi_main.empty:
        fi_main.to_csv(target_dir / "feature_importance_for_report.csv", index=False, encoding="utf-8-sig")

    group_df = risk_group_analysis(y_test, best_pred, n_groups=10)
    true_group_df = true_group_error_analysis(y_test, best_pred, n_groups=10)
    industry_df = residual_by_industry(test_df, y_test, best_pred, cfg)

    group_df.to_csv(target_dir / "risk_group_analysis.csv", index=False, encoding="utf-8-sig")
    true_group_df.to_csv(target_dir / "true_group_error_analysis.csv", index=False, encoding="utf-8-sig")
    if not industry_df.empty:
        industry_df.to_csv(target_dir / "residual_by_industry.csv", index=False, encoding="utf-8-sig")

    if cfg.save_models:
        model_dir = ensure_dir(target_dir / "models")
        try:
            joblib.dump(best["estimator"], model_dir / f"best_model_{best['model']}.joblib")
        except Exception:
            pass

    if cfg.save_plots:
        save_core_plots(target_dir, target, metrics_df, fi_main, group_df)

    # Step 7：写一个精简 JSON 摘要，方便快速查看。
    summary = {
        "target": target,
        "n_train": len(X_train),
        "n_valid": len(X_valid),
        "n_test": len(X_test),
        "n_features": X_train.shape[1],
        "best_model": best["model"],
        "best_metrics": best["test_metrics"],
        "target_winsor_bounds": split["target_winsor_bounds"],
        "top_features": fi_main.head(10).to_dict(orient="records") if not fi_main.empty else [],
    }
    (target_dir / "target_summary.json").write_text(
        json.dumps(to_jsonable(summary), ensure_ascii=False, indent=2), encoding="utf-8"
    )

    return {
        "target": target,
        "target_dir": target_dir,
        "metrics_df": metrics_df,
        "best": best,
        "feature_importance": fi_main,
        "risk_group": group_df,
        "split_summary": split_summary,
    }


# ============================================================
# 10. Oscore vs Zrisk 稳健性对比
# ============================================================

def add_robust_ranks(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["R2_rank"] = out["R2"].rank(method="min", ascending=False)
    out["Spearman_rank"] = out["Spearman"].rank(method="min", ascending=False)
    out["Top20_Precision_rank"] = out["Top20_Precision"].rank(method="min", ascending=False)
    out["robust_rank_score"] = out[["R2_rank", "Spearman_rank", "Top20_Precision_rank"]].mean(axis=1)
    out["robust_rank"] = out["robust_rank_score"].rank(method="min", ascending=True)
    return out


def make_robustness_compare(target_outputs: Dict[str, Dict[str, Any]], output_dir: Path) -> None:
    """如果同时有 Oscore 和 Zrisk，生成模型稳健性对比表。"""
    if "Oscore" not in target_outputs or "Zrisk" not in target_outputs:
        return

    compare_dir = ensure_dir(output_dir / "robustness_compare")
    os_df = add_robust_ranks(target_outputs["Oscore"]["metrics_df"])
    zr_df = add_robust_ranks(target_outputs["Zrisk"]["metrics_df"])

    keep = ["model", "model_family", "R2", "Pearson", "Spearman", "Top20_Precision", "Top20_Recall", "robust_rank_score", "robust_rank"]
    os2 = os_df[keep].add_prefix("Oscore_").rename(columns={"Oscore_model": "model"})
    zr2 = zr_df[keep].add_prefix("Zrisk_").rename(columns={"Zrisk_model": "model"})
    comp = os2.merge(zr2, on="model", how="inner")

    rank_cols = [
        "Oscore_R2", "Oscore_Spearman", "Oscore_Top20_Precision",
        "Zrisk_R2", "Zrisk_Spearman", "Zrisk_Top20_Precision",
    ]
    # 为了方便阅读，另外计算一个跨目标排名分数：分别在两个目标下看排名。
    comp["Oscore_R2_rank"] = comp["Oscore_R2"].rank(method="min", ascending=False)
    comp["Oscore_Spearman_rank"] = comp["Oscore_Spearman"].rank(method="min", ascending=False)
    comp["Oscore_Top20_rank"] = comp["Oscore_Top20_Precision"].rank(method="min", ascending=False)
    comp["Zrisk_R2_rank"] = comp["Zrisk_R2"].rank(method="min", ascending=False)
    comp["Zrisk_Spearman_rank"] = comp["Zrisk_Spearman"].rank(method="min", ascending=False)
    comp["Zrisk_Top20_rank"] = comp["Zrisk_Top20_Precision"].rank(method="min", ascending=False)
    comp["cross_target_rank_score"] = comp[[
        "Oscore_R2_rank", "Oscore_Spearman_rank", "Oscore_Top20_rank",
        "Zrisk_R2_rank", "Zrisk_Spearman_rank", "Zrisk_Top20_rank",
    ]].mean(axis=1)
    comp = comp.sort_values("cross_target_rank_score")
    comp.to_csv(compare_dir / "oscore_zrisk_model_comparison.csv", index=False, encoding="utf-8-sig")

    lines = []
    lines.append("# Oscore 与 Zrisk 稳健性对比摘要\n")
    lines.append("主实验以 Oscore 为目标变量，稳健性实验以 Zrisk=-Zscore 为目标变量。")
    lines.append("由于 Oscore 和 Zrisk 的量纲不同，二者的 RMSE/MAE 不宜直接横向比较。")
    lines.append("更适合比较的是 R²、Spearman 和 Top20 Precision/Recall。\n")

    best_os = os_df.sort_values("Spearman", ascending=False).iloc[0]
    best_zr = zr_df.sort_values("Spearman", ascending=False).iloc[0]
    lines.append(f"- Oscore 中 Spearman 最优模型：{best_os['model']}，Spearman={best_os['Spearman']:.4f}。")
    lines.append(f"- Zrisk 中 Spearman 最优模型：{best_zr['model']}，Spearman={best_zr['Spearman']:.4f}。\n")
    lines.append("跨目标综合排名前 10：\n")
    show_cols = ["model", "Oscore_R2", "Zrisk_R2", "Oscore_Spearman", "Zrisk_Spearman", "Oscore_Top20_Precision", "Zrisk_Top20_Precision", "cross_target_rank_score"]
    lines.append(comp[show_cols].head(10).round(4).to_markdown(index=False))
    (compare_dir / "robustness_summary.md").write_text("\n".join(lines), encoding="utf-8")


# ============================================================
# 11. 总控主函数
# ============================================================

def run_pipeline(cfg: PipelineConfig) -> None:
    set_seed(cfg.random_state)
    output_dir = ensure_dir(cfg.output_dir)
    ensure_dir(output_dir / "logs")

    # 保存本次实验配置，保证报告结果可追溯。
    (output_dir / "pipeline_config.json").write_text(
        json.dumps(to_jsonable(asdict(cfg)), ensure_ascii=False, indent=2), encoding="utf-8"
    )

    raw_df = read_csv_safely(cfg.input_path)
    raw_df = clean_columns(raw_df)
    if cfg.stock_code_col in raw_df.columns:
        raw_df[cfg.stock_code_col] = standardize_stock_code(raw_df[cfg.stock_code_col])
    raw_df = coerce_numeric_columns(raw_df, cfg)

    basic_audit(raw_df, cfg, output_dir)

    target_outputs: Dict[str, Dict[str, Any]] = {}
    for target in cfg.targets:
        target_outputs[target] = run_one_target(raw_df, target, cfg, output_dir)

    make_robustness_compare(target_outputs, output_dir)

    # 总表：所有目标、所有模型合并。
    all_metrics = []
    for target, obj in target_outputs.items():
        all_metrics.append(obj["metrics_df"])
    all_df = pd.concat(all_metrics, axis=0, ignore_index=True)
    all_df.to_csv(output_dir / "all_targets_model_metrics.csv", index=False, encoding="utf-8-sig")

    print("\n" + "=" * 90)
    print("精简版一体化实验完成")
    print(f"输出目录：{output_dir.resolve()}")
    print("核心文件：")
    print("  - all_targets_model_metrics.csv")
    print("  - oscore/model_metrics.csv")
    print("  - zrisk/model_metrics.csv，如果运行 Zrisk")
    print("  - robustness_compare/oscore_zrisk_model_comparison.csv，如果同时运行 Oscore 和 Zrisk")
    print("=" * 90)


# ============================================================
# 12. 命令行参数
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="A-share financial distress risk prediction pipeline")
    parser.add_argument("--input", required=True, help="原始 CSV 文件路径")
    parser.add_argument("--output_dir", default="outputs/clean_pipeline", help="输出目录")
    parser.add_argument("--targets", nargs="+", default=["Oscore", "Zrisk"], choices=["Oscore", "Zscore", "Zrisk"], help="要运行的目标变量")
    parser.add_argument("--random_state", type=int, default=42)

    parser.add_argument("--sample_train_rows", type=int, default=None, help="快速调试时抽样训练行数；正式实验不建议设置")
    parser.add_argument("--sample_valid_rows", type=int, default=None)
    parser.add_argument("--sample_test_rows", type=int, default=None)

    parser.add_argument("--rf_n_estimators", type=int, default=200)
    parser.add_argument("--nn_epochs", type=int, default=60)
    parser.add_argument("--nn_batch_size", type=int, default=512)
    parser.add_argument("--skip_nn", action="store_true", help="跳过神经网络，快速跑传统模型")

    parser.add_argument("--no_permutation", action="store_true", help="跳过置换重要性，速度更快")
    parser.add_argument("--permutation_sample_rows", type=int, default=3000)
    parser.add_argument("--permutation_repeats", type=int, default=3)

    parser.add_argument("--keep_high_missing", action="store_true", help="保留 40%-80% 高缺失变量；默认删除")
    parser.add_argument("--scaler", default="standard", choices=["standard", "robust", "none"])
    parser.add_argument("--save_models", action="store_true", help="保存最佳模型文件")
    parser.add_argument("--save_plots", action="store_true", help="保存少量核心报告图")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = PipelineConfig(
        input_path=args.input,
        output_dir=args.output_dir,
        targets=tuple(args.targets),
        random_state=args.random_state,
        sample_train_rows=args.sample_train_rows,
        sample_valid_rows=args.sample_valid_rows,
        sample_test_rows=args.sample_test_rows,
        rf_n_estimators=args.rf_n_estimators,
        nn_epochs=args.nn_epochs,
        nn_batch_size=args.nn_batch_size,
        run_nn=not args.skip_nn,
        run_permutation=not args.no_permutation,
        permutation_sample_rows=args.permutation_sample_rows,
        permutation_repeats=args.permutation_repeats,
        keep_high_missing=args.keep_high_missing,
        scaler=args.scaler,
        save_models=args.save_models,
        save_plots=args.save_plots,
    )
    run_pipeline(cfg)


if __name__ == "__main__":
    main()
