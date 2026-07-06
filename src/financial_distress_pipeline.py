#!/usr/bin/env python3
"""Leakage-free financial distress risk prediction pipeline.

This script trains several regression models to predict firm-level financial
risk scores such as Oscore and Zrisk from structured financial indicators.
It is intentionally compact enough for a GitHub research portfolio while still
including the essential elements of a rigorous tabular ML workflow:

1. train/validation/test split before preprocessing;
2. missing-value handling, winsorization, signed-log transformation, scaling;
3. categorical one-hot encoding;
4. model comparison across linear, tree, boosting, and neural models;
5. regression, ranking, and Top-20% high-risk screening metrics.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from lightgbm import LGBMRegressor

    LIGHTGBM_AVAILABLE = True
except Exception:
    LIGHTGBM_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False


@dataclass
class PipelineConfig:
    input_path: str
    output_dir: str
    targets: Tuple[str, ...] = ("Oscore", "Zrisk")
    train_size: float = 0.70
    valid_size: float = 0.15
    test_size: float = 0.15
    random_state: int = 42
    id_cols: Tuple[str, ...] = ("股票代码", "股票简称")
    categorical_cols: Tuple[str, ...] = ("行业名称1", "报表类型编码")
    target_cols: Tuple[str, ...] = ("Oscore", "Zscore", "Zrisk", "target")
    low_missing_threshold: float = 0.05
    mid_missing_threshold: float = 0.40
    drop_missing_threshold: float = 0.80
    winsor_lower: float = 0.01
    winsor_upper: float = 0.99
    skew_threshold: float = 5.0
    sample_train_rows: Optional[int] = None
    sample_valid_rows: Optional[int] = None
    sample_test_rows: Optional[int] = None
    rf_n_estimators: int = 200
    gb_n_estimators: int = 400
    n_jobs: int = -1
    run_nn: bool = True
    nn_epochs: int = 30
    nn_batch_size: int = 256
    nn_learning_rate: float = 1e-3
    nn_patience: int = 6
    save_predictions: bool = False
    save_models: bool = False


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if TORCH_AVAILABLE:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def read_csv_safely(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    for encoding in ("utf-8-sig", "utf-8", "gbk", "gb18030"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except Exception:
            continue
    raise RuntimeError(f"Unable to read CSV file: {path}")


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    return out


def standardize_stock_code(series: pd.Series) -> pd.Series:
    return series.astype(str).str.replace(r"\.0$", "", regex=True).str.strip().str.zfill(6)


def coerce_numeric_columns(df: pd.DataFrame, cfg: PipelineConfig) -> pd.DataFrame:
    out = df.copy()
    protected = set(cfg.id_cols) | set(cfg.categorical_cols)
    for col in out.columns:
        if col in protected or pd.api.types.is_numeric_dtype(out[col]):
            continue
        s = out[col].astype(str).str.strip()
        s = s.replace({"": np.nan, "nan": np.nan, "None": np.nan, "--": np.nan, "—": np.nan})
        s = s.str.replace(",", "", regex=False)
        percent_mask = s.str.endswith("%", na=False)
        numeric = pd.to_numeric(s.str.replace("%", "", regex=False), errors="coerce")
        numeric.loc[percent_mask] = numeric.loc[percent_mask] / 100.0
        if numeric.notna().sum() >= max(10, int(0.2 * len(out))):
            out[col] = numeric
    return out


def prepare_dataframe(df: pd.DataFrame, cfg: PipelineConfig) -> pd.DataFrame:
    df = clean_columns(df)
    if "股票代码" in df.columns:
        df["股票代码"] = standardize_stock_code(df["股票代码"])
    df = coerce_numeric_columns(df, cfg)
    if "Zrisk" not in df.columns and "Zscore" in df.columns:
        df["Zrisk"] = -pd.to_numeric(df["Zscore"], errors="coerce")
    return df


def sample_rows(df: pd.DataFrame, n: Optional[int], random_state: int) -> pd.DataFrame:
    if n is None or n <= 0 or len(df) <= n:
        return df
    return df.sample(n=n, random_state=random_state)


def split_train_valid_test(df: pd.DataFrame, cfg: PipelineConfig) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    temp_size = cfg.valid_size + cfg.test_size
    train_df, temp_df = train_test_split(df, test_size=temp_size, random_state=cfg.random_state)
    valid_fraction_in_temp = cfg.valid_size / temp_size
    valid_df, test_df = train_test_split(temp_df, train_size=valid_fraction_in_temp, random_state=cfg.random_state)
    return train_df.reset_index(drop=True), valid_df.reset_index(drop=True), test_df.reset_index(drop=True)


def choose_feature_columns(df: pd.DataFrame, target: str, cfg: PipelineConfig) -> Tuple[List[str], List[str]]:
    excluded = set(cfg.id_cols) | set(cfg.target_cols)
    candidate_cols = [c for c in df.columns if c not in excluded and c != target]
    categorical_cols = [c for c in cfg.categorical_cols if c in candidate_cols]
    numeric_cols = [c for c in candidate_cols if c not in categorical_cols and pd.api.types.is_numeric_dtype(df[c])]
    return numeric_cols, categorical_cols


class FinancialPreprocessor:
    """Fit preprocessing rules on training data and apply them to new data."""

    def __init__(self, cfg: PipelineConfig, numeric_cols: List[str], categorical_cols: List[str]):
        self.cfg = cfg
        self.numeric_cols = list(numeric_cols)
        self.categorical_cols = list(categorical_cols)
        self.kept_numeric_cols: List[str] = []
        self.missing_indicator_cols: List[str] = []
        self.signed_log_cols: List[str] = []
        self.winsor_bounds: Dict[str, Tuple[float, float]] = {}
        self.num_imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()
        self.encoder: Optional[OneHotEncoder] = None
        self.feature_names_: List[str] = []

    def fit(self, df: pd.DataFrame) -> "FinancialPreprocessor":
        x_num = df[self.numeric_cols].copy()
        missing_rate = x_num.isna().mean()
        self.kept_numeric_cols = [
            c for c in self.numeric_cols if missing_rate.get(c, 0.0) < self.cfg.drop_missing_threshold
        ]
        self.missing_indicator_cols = [
            c for c in self.kept_numeric_cols if missing_rate.get(c, 0.0) >= self.cfg.low_missing_threshold
        ]

        x_num = x_num[self.kept_numeric_cols]
        for col in self.kept_numeric_cols:
            series = pd.to_numeric(x_num[col], errors="coerce")
            low = float(series.quantile(self.cfg.winsor_lower))
            high = float(series.quantile(self.cfg.winsor_upper))
            if not np.isfinite(low):
                low = 0.0
            if not np.isfinite(high):
                high = low
            self.winsor_bounds[col] = (low, high)
            skew = series.dropna().skew()
            if np.isfinite(skew) and abs(skew) >= self.cfg.skew_threshold:
                self.signed_log_cols.append(col)

        x_num_processed = self._process_numeric(df, fit_mode=True)
        self.num_imputer.fit(x_num_processed)
        x_num_imputed = self.num_imputer.transform(x_num_processed)
        self.scaler.fit(x_num_imputed)

        if self.categorical_cols:
            try:
                self.encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            except TypeError:
                self.encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
            self.encoder.fit(df[self.categorical_cols].astype(str).fillna("Missing"))

        self.feature_names_ = self._build_feature_names()
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        x_num_processed = self._process_numeric(df, fit_mode=False)
        x_num_imputed = self.num_imputer.transform(x_num_processed)
        x_num_scaled = self.scaler.transform(x_num_imputed)

        parts = [x_num_scaled]
        if self.encoder is not None and self.categorical_cols:
            x_cat = self.encoder.transform(df[self.categorical_cols].astype(str).fillna("Missing"))
            parts.append(np.asarray(x_cat, dtype=np.float32))
        return np.hstack(parts).astype(np.float32)

    def _process_numeric(self, df: pd.DataFrame, fit_mode: bool) -> pd.DataFrame:
        columns: Dict[str, pd.Series] = {}
        for col in self.kept_numeric_cols:
            s = pd.to_numeric(df[col], errors="coerce") if col in df.columns else pd.Series(np.nan, index=df.index)
            low, high = self.winsor_bounds.get(col, (np.nan, np.nan))
            if np.isfinite(low) and np.isfinite(high):
                s = s.clip(lower=low, upper=high)
            if col in self.signed_log_cols:
                s = np.sign(s) * np.log1p(np.abs(s))
            columns[col] = pd.Series(s, index=df.index)

        out = pd.DataFrame(columns, index=df.index)
        for col in self.missing_indicator_cols:
            original = pd.to_numeric(df[col], errors="coerce") if col in df.columns else pd.Series(np.nan, index=df.index)
            out[f"{col}__missing"] = original.isna().astype(float)
        return out

    def _build_feature_names(self) -> List[str]:
        names = list(self.kept_numeric_cols) + [f"{c}__missing" for c in self.missing_indicator_cols]
        if self.encoder is not None and self.categorical_cols:
            cat_names = list(self.encoder.get_feature_names_out(self.categorical_cols))
            names.extend(cat_names)
        return names


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(math.sqrt(mean_squared_error(y_true, y_pred)))


def pearson_corr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if np.std(y_true) < 1e-12 or np.std(y_pred) < 1e-12:
        return float("nan")
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def spearman_corr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    r_true = pd.Series(y_true).rank(method="average").to_numpy()
    r_pred = pd.Series(y_pred).rank(method="average").to_numpy()
    return pearson_corr(r_true, r_pred)


def top_fraction_metrics(y_true: np.ndarray, y_pred: np.ndarray, frac: float = 0.20) -> Dict[str, float]:
    n = len(y_true)
    k = max(1, int(round(frac * n)))
    true_top = set(np.argsort(y_true)[-k:])
    pred_top = set(np.argsort(y_pred)[-k:])
    hit = len(true_top & pred_top)
    precision = hit / len(pred_top)
    recall = hit / len(true_top)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {"top20_precision": precision, "top20_recall": recall, "top20_f1": f1}


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    metrics = {
        "rmse": rmse(y_true, y_pred),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "pearson": pearson_corr(y_true, y_pred),
        "spearman": spearman_corr(y_true, y_pred),
    }
    metrics.update(top_fraction_metrics(y_true, y_pred, frac=0.20))
    return metrics


if TORCH_AVAILABLE:

    class MLPRegressor(nn.Module):
        def __init__(self, input_dim: int, hidden_dims: Tuple[int, ...] = (256, 128, 64), dropout: float = 0.20):
            super().__init__()
            layers: List[nn.Module] = []
            prev_dim = input_dim
            for dim in hidden_dims:
                layers.extend([nn.Linear(prev_dim, dim), nn.BatchNorm1d(dim), nn.ReLU(), nn.Dropout(dropout)])
                prev_dim = dim
            layers.append(nn.Linear(prev_dim, 1))
            self.net = nn.Sequential(*layers)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x).squeeze(-1)


def train_torch_mlp(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_valid: np.ndarray,
    y_valid: np.ndarray,
    cfg: PipelineConfig,
) -> Tuple[object, np.ndarray]:
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is not available.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLPRegressor(input_dim=x_train.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.nn_learning_rate, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    train_ds = TensorDataset(torch.tensor(x_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    train_loader = DataLoader(train_ds, batch_size=cfg.nn_batch_size, shuffle=True)

    x_valid_t = torch.tensor(x_valid, dtype=torch.float32).to(device)
    y_valid_t = torch.tensor(y_valid, dtype=torch.float32).to(device)

    best_state = None
    best_valid = float("inf")
    patience = 0
    for _epoch in range(cfg.nn_epochs):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            valid_loss = float(loss_fn(model(x_valid_t), y_valid_t).item())
        if valid_loss < best_valid:
            best_valid = valid_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= cfg.nn_patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, np.array([best_valid], dtype=float)


def predict_torch(model: object, x: np.ndarray) -> np.ndarray:
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        pred = model(torch.tensor(x, dtype=torch.float32).to(device)).detach().cpu().numpy()
    return pred.astype(float)


def build_models(cfg: PipelineConfig) -> Dict[str, object]:
    models: Dict[str, object] = {
        "dummy_mean": DummyRegressor(strategy="mean"),
        "dummy_median": DummyRegressor(strategy="median"),
        "ridge": RidgeCV(alphas=np.logspace(-3, 3, 13)),
        "lasso": LassoCV(cv=3, random_state=cfg.random_state, max_iter=5000),
        "elastic_net": ElasticNetCV(cv=3, random_state=cfg.random_state, max_iter=5000),
        "random_forest": RandomForestRegressor(
            n_estimators=cfg.rf_n_estimators,
            max_depth=12,
            min_samples_leaf=20,
            n_jobs=cfg.n_jobs,
            random_state=cfg.random_state,
        ),
    }
    if LIGHTGBM_AVAILABLE:
        models["lightgbm"] = LGBMRegressor(
            n_estimators=cfg.gb_n_estimators,
            learning_rate=0.03,
            num_leaves=31,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=cfg.random_state,
            n_jobs=cfg.n_jobs,
            verbosity=-1,
        )
    else:
        models["hist_gradient_boosting"] = HistGradientBoostingRegressor(
            max_iter=cfg.gb_n_estimators,
            learning_rate=0.05,
            random_state=cfg.random_state,
        )
    return models


def train_for_target(df: pd.DataFrame, target: str, cfg: PipelineConfig, output_dir: Path) -> Dict[str, object]:
    work = df[df[target].notna()].copy()
    y_all = pd.to_numeric(work[target], errors="coerce")
    work = work[y_all.notna()].copy()
    work[target] = pd.to_numeric(work[target], errors="coerce")

    train_df, valid_df, test_df = split_train_valid_test(work, cfg)
    train_df = sample_rows(train_df, cfg.sample_train_rows, cfg.random_state)
    valid_df = sample_rows(valid_df, cfg.sample_valid_rows, cfg.random_state)
    test_df = sample_rows(test_df, cfg.sample_test_rows, cfg.random_state)

    numeric_cols, categorical_cols = choose_feature_columns(train_df, target, cfg)
    preprocessor = FinancialPreprocessor(cfg, numeric_cols, categorical_cols).fit(train_df)
    x_train = preprocessor.transform(train_df)
    x_valid = preprocessor.transform(valid_df)
    x_test = preprocessor.transform(test_df)
    y_train = train_df[target].to_numpy(dtype=float)
    y_valid = valid_df[target].to_numpy(dtype=float)
    y_test = test_df[target].to_numpy(dtype=float)

    metrics_rows: List[Dict[str, object]] = []
    predictions: Dict[str, np.ndarray] = {}
    fitted_models: Dict[str, object] = {}

    for name, model in build_models(cfg).items():
        model.fit(x_train, y_train)
        valid_pred = model.predict(x_valid)
        test_pred = model.predict(x_test)
        row = {"target": target, "model": name, "split": "test"}
        row.update(evaluate_predictions(y_test, test_pred))
        metrics_rows.append(row)
        predictions[name] = test_pred
        fitted_models[name] = model
        print(f"[{target}] {name}: R2={row['r2']:.4f}, Spearman={row['spearman']:.4f}, RMSE={row['rmse']:.4f}")

    if cfg.run_nn and TORCH_AVAILABLE:
        nn_model, _ = train_torch_mlp(x_train, y_train, x_valid, y_valid, cfg)
        test_pred = predict_torch(nn_model, x_test)
        row = {"target": target, "model": "neural_network", "split": "test"}
        row.update(evaluate_predictions(y_test, test_pred))
        metrics_rows.append(row)
        predictions["neural_network"] = test_pred
        fitted_models["neural_network"] = nn_model
        print(f"[{target}] neural_network: R2={row['r2']:.4f}, Spearman={row['spearman']:.4f}, RMSE={row['rmse']:.4f}")

    metrics_df = pd.DataFrame(metrics_rows).sort_values(["r2", "spearman"], ascending=False)
    metrics_path = output_dir / f"{target}_model_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)

    best_model_name = str(metrics_df.iloc[0]["model"])
    best_pred = predictions[best_model_name]
    pred_df = pd.DataFrame({"y_true": y_test, "y_pred": best_pred, "best_model": best_model_name})
    for col in cfg.id_cols:
        if col in test_df.columns:
            pred_df[col] = test_df[col].values
    if cfg.save_predictions:
        pred_df.to_csv(output_dir / f"{target}_predictions.csv", index=False, encoding="utf-8-sig")

    best_model = fitted_models[best_model_name]
    if hasattr(best_model, "feature_importances_"):
        imp = pd.DataFrame(
            {"feature": preprocessor.feature_names_, "importance": np.asarray(best_model.feature_importances_, dtype=float)}
        ).sort_values("importance", ascending=False)
        imp.to_csv(output_dir / f"{target}_feature_importance.csv", index=False, encoding="utf-8-sig")

    if cfg.save_models:
        model_dir = output_dir / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump({"preprocessor": preprocessor, "model": best_model}, model_dir / f"{target}_{best_model_name}.joblib")

    return {
        "target": target,
        "n_train": int(len(train_df)),
        "n_valid": int(len(valid_df)),
        "n_test": int(len(test_df)),
        "n_features": int(x_train.shape[1]),
        "best_model": best_model_name,
        "metrics_file": str(metrics_path),
    }


def parse_args() -> PipelineConfig:
    parser = argparse.ArgumentParser(description="Financial distress risk prediction pipeline.")
    parser.add_argument("--input", required=True, help="Input CSV file.")
    parser.add_argument("--output-dir", default="outputs/main", help="Output directory.")
    parser.add_argument("--targets", nargs="+", default=["Oscore", "Zrisk"], help="Targets to run.")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--sample-train-rows", type=int, default=None)
    parser.add_argument("--sample-valid-rows", type=int, default=None)
    parser.add_argument("--sample-test-rows", type=int, default=None)
    parser.add_argument("--rf-n-estimators", type=int, default=200)
    parser.add_argument("--gb-n-estimators", type=int, default=400)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--nn-epochs", type=int, default=30)
    parser.add_argument("--no-nn", action="store_true")
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--save-models", action="store_true")
    args = parser.parse_args()

    return PipelineConfig(
        input_path=args.input,
        output_dir=args.output_dir,
        targets=tuple(args.targets),
        random_state=args.random_state,
        sample_train_rows=args.sample_train_rows,
        sample_valid_rows=args.sample_valid_rows,
        sample_test_rows=args.sample_test_rows,
        rf_n_estimators=args.rf_n_estimators,
        gb_n_estimators=args.gb_n_estimators,
        n_jobs=args.n_jobs,
        nn_epochs=args.nn_epochs,
        run_nn=not args.no_nn,
        save_predictions=args.save_predictions,
        save_models=args.save_models,
    )


def main() -> None:
    cfg = parse_args()
    set_seed(cfg.random_state)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = prepare_dataframe(read_csv_safely(cfg.input_path), cfg)
    summaries = []
    for target in cfg.targets:
        if target not in df.columns:
            print(f"[WARN] target {target!r} not found; skipping.")
            continue
        summaries.append(train_for_target(df, target, cfg, output_dir))

    summary = {
        "config": asdict(cfg),
        "lightgbm_available": LIGHTGBM_AVAILABLE,
        "torch_available": TORCH_AVAILABLE,
        "targets": summaries,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved outputs to {output_dir}")


if __name__ == "__main__":
    main()
