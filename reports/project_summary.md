# 项目摘要

本项目研究如何基于 A 股上市公司结构化财务指标预测公司财务困境风险。主实验使用 Oscore 作为连续风险得分，稳健性检验使用 `Zrisk=-Zscore`。

## 研究设计

- 主被解释变量：Oscore；
- 稳健性变量：Zrisk；
- 数据处理：无泄漏切分、缺失填补、缺失指示变量、缩尾、符号对数变换、标准化和 One-Hot 编码；
- 模型对照：线性回归、Ridge、Lasso、ElasticNet、随机森林、LightGBM、原始神经网络和改进神经网络；
- 评价指标：RMSE、MAE、R²、Pearson、Spearman、Top20 Precision/Recall/F1。

## 主要发现

1. 在 Oscore 主实验中，LightGBM 表现最佳，改进神经网络表现非常接近；
2. 在 Zrisk 稳健性实验中，改进神经网络位于前列，LightGBM 仍具有较强表现；
3. 非线性模型整体优于线性模型，说明财务困境风险与财务变量之间存在非线性关系和变量交互；
4. 重要特征主要集中在现金流质量、EVA 价值创造能力、盈利质量、市场预期和行业异质性等方面。

完整中文研究报告见 `reports/financial_distress_research_report_zh.md`。
