#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成一个与研究流水线字段结构一致的合成财务数据集。
该文件仅用于 GitHub demo 和代码测试，不是真实公司数据。
"""

from __future__ import annotations

import argparse
import numpy as np
import pandas as pd


def make_synthetic_data(n_rows: int = 5000, random_state: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)

    industries = [
        "货币金融服务", "房地产", "食品饮料", "计算机软件", "医药制造",
        "电气机械", "批发零售", "交通运输", "化学制品", "专用设备"
    ]
    report_types = ["A", "B", "C"]

    stock_ids = rng.integers(1, 900, size=n_rows)
    industry = rng.choice(industries, size=n_rows)
    report_type = rng.choice(report_types, size=n_rows, p=[0.88, 0.08, 0.04])

    size_factor = rng.normal(0, 1, n_rows)
    leverage_risk = rng.normal(0, 1, n_rows)
    profitability = rng.normal(0, 1, n_rows)
    cash_quality = rng.normal(0, 1, n_rows)
    growth = rng.normal(0, 1, n_rows)
    market_expectation = rng.normal(0, 1, n_rows)

    df = pd.DataFrame({
        "股票代码": stock_ids,
        "股票简称": [f"公司{sid:04d}" for sid in stock_ids],
        "行业名称1": industry,
        "报表类型编码": report_type,
        "托宾Q值A": np.exp(0.9 + 0.45 * market_expectation + rng.normal(0, 0.25, n_rows)),
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
    })

    # 构造合成风险得分：Oscore 越高表示风险越高；Zscore 越高表示越安全。
    risk = (
        0.45 * leverage_risk
        - 0.50 * profitability
        - 0.42 * cash_quality
        - 0.20 * market_expectation
        + 0.15 * np.abs(growth)
        + rng.normal(0, 0.45, n_rows)
    )
    df["Oscore"] = -8.3 + 1.35 * risk
    df["Zscore"] = 3.0 - 1.15 * risk + rng.normal(0, 0.25, n_rows)

    # 人为加入缺失值，用来模拟真实财务面板数据中的缺失情况。
    for col, rate in {
        "现金流到期债务保障倍数": 0.14,
        "净资产收益率增长率A": 0.10,
        "营业利润增长率A": 0.11,
        "现金股利保障倍数": 0.80,
        "应付账款周转率A": 0.08,
    }.items():
        mask = rng.random(n_rows) < rate
        df.loc[mask, col] = np.nan

    # 人为加入少量极端值，用来测试缩尾处理是否生效。
    for col in ["现金流到期债务保障倍数", "营业利润增长率A", "市盈率PE1"]:
        idx = rng.choice(n_rows, size=max(5, n_rows // 100), replace=False)
        df.loc[idx, col] = df.loc[idx, col] * rng.choice([8, -8, 12], size=len(idx))

    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/example/synthetic_financial_data.csv")
    parser.add_argument("--n_rows", type=int, default=5000)
    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args()

    df = make_synthetic_data(args.n_rows, args.random_state)
    df.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"合成示例数据已保存到 {args.output}，数据规模={df.shape}")


if __name__ == "__main__":
    main()
