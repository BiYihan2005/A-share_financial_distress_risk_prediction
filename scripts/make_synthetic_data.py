#!/usr/bin/env python3
"""Generate a synthetic A-share-style financial distress dataset.

The generated data are only for testing the public pipeline. They do not
represent any real listed company.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


FINANCIAL_COLUMNS = [
    "托宾Q值A",
    "EVA率口径一",
    "销售EVA率口径一",
    "现金流到期债务保障倍数",
    "净资产收益率增长率A",
    "营业利润增长率A",
    "现金股利保障倍数",
    "盈利波动性",
    "应付账款周转率A",
    "净利润现金净含量",
    "营运指数",
    "市盈率PE1",
]


def make_synthetic_data(n_rows: int = 5000, random_state: int = 42) -> pd.DataFrame:
    """Create a synthetic tabular financial dataset with realistic structure."""
    rng = np.random.default_rng(random_state)

    industries = [
        "Banking", "Real Estate", "Food & Beverage", "Software", "Pharmaceuticals",
        "Electrical Equipment", "Retail", "Transportation", "Chemicals", "Machinery",
    ]
    report_types = ["A", "B", "C"]

    stock_ids = rng.integers(1, 1200, size=n_rows)
    industry = rng.choice(industries, size=n_rows)
    report_type = rng.choice(report_types, size=n_rows, p=[0.88, 0.08, 0.04])

    size_factor = rng.normal(0.0, 1.0, n_rows)
    leverage_risk = rng.normal(0.0, 1.0, n_rows)
    profitability = rng.normal(0.0, 1.0, n_rows)
    cash_quality = rng.normal(0.0, 1.0, n_rows)
    growth = rng.normal(0.0, 1.0, n_rows)
    market_expectation = rng.normal(0.0, 1.0, n_rows)

    df = pd.DataFrame(
        {
            "股票代码": [f"{sid:06d}" for sid in stock_ids],
            "股票简称": [f"Firm{sid:04d}" for sid in stock_ids],
            "行业名称1": industry,
            "报表类型编码": report_type,
            "托宾Q值A": np.exp(0.8 + 0.40 * market_expectation + rng.normal(0, 0.25, n_rows)),
            "EVA率口径一": 0.04 * profitability + rng.normal(0, 0.02, n_rows),
            "销售EVA率口径一": 0.10 * profitability + rng.normal(0, 0.08, n_rows),
            "现金流到期债务保障倍数": 10 + 28 * cash_quality - 18 * leverage_risk + rng.normal(0, 20, n_rows),
            "净资产收益率增长率A": 2 + 7 * growth + rng.normal(0, 8, n_rows),
            "营业利润增长率A": 3 + 8 * growth + rng.normal(0, 10, n_rows),
            "现金股利保障倍数": 2 + 2.5 * cash_quality + rng.normal(0, 5, n_rows),
            "盈利波动性": np.exp(-2.5 + 0.45 * np.abs(growth) + 0.3 * leverage_risk + rng.normal(0, 0.4, n_rows)),
            "应付账款周转率A": np.exp(1.2 + 0.2 * growth + rng.normal(0, 0.5, n_rows)),
            "净利润现金净含量": 5 + 2.5 * cash_quality + 1.8 * profitability + rng.normal(0, 2, n_rows),
            "营运指数": 4 + 2.0 * cash_quality + rng.normal(0, 1.5, n_rows),
            "市盈率PE1": np.exp(2.4 + 0.25 * market_expectation - 0.15 * profitability + rng.normal(0, 0.45, n_rows)),
        }
    )

    risk = (
        0.42 * leverage_risk
        - 0.52 * profitability
        - 0.43 * cash_quality
        - 0.18 * market_expectation
        - 0.08 * size_factor
        + 0.16 * np.abs(growth)
        + rng.normal(0, 0.45, n_rows)
    )
    df["Oscore"] = -8.3 + 1.35 * risk
    df["Zscore"] = 3.0 - 1.15 * risk + rng.normal(0, 0.25, n_rows)

    missing_rates = {
        "现金流到期债务保障倍数": 0.14,
        "净资产收益率增长率A": 0.10,
        "营业利润增长率A": 0.11,
        "现金股利保障倍数": 0.80,
        "应付账款周转率A": 0.08,
    }
    for col, rate in missing_rates.items():
        mask = rng.random(n_rows) < rate
        df.loc[mask, col] = np.nan

    for col in ["现金流到期债务保障倍数", "营业利润增长率A", "市盈率PE1"]:
        idx = rng.choice(n_rows, size=max(5, n_rows // 100), replace=False)
        df.loc[idx, col] = df.loc[idx, col] * rng.choice([8, -8, 12], size=len(idx))

    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic financial distress data.")
    parser.add_argument("--output", default="data/example/synthetic_financial_data.csv")
    parser.add_argument("--n-rows", type=int, default=5000)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    df = make_synthetic_data(args.n_rows, args.random_state)
    df.to_csv(output, index=False, encoding="utf-8-sig")
    print(f"Saved synthetic dataset to {output} with shape={df.shape}")


if __name__ == "__main__":
    main()
