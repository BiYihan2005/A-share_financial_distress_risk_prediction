## 时间维度补充实验：统计截止日期与未来一期风险预测

在主实验的随机切分建模流程之外，本项目进一步补充了基于 `统计截止日期` 的时间外推实验。该补充实验使用第 $t$ 期财务指标预测同一股票第 $t+1$ 期 Oscore / Zrisk，并进一步加入上一期水平与变化量特征：

$$
X_{i,t} \longrightarrow Y_{i,t+1},
\qquad
\left[X_{i,t}, X_{i,t-1}, \Delta X_{i,t}\right] \longrightarrow Y_{i,t+1}
$$

相关文件：

- 补充报告：`reports/time_supplement_report_zh.md`
- 补充实验脚本：`scripts/time_supplement_experiments.py`
- 结果汇总：`results/time_supplement/metrics_summary_test.csv`
- 方法说明：`docs/time_supplement_methodology.md`

补充实验表明，非线性模型在未来一期 Oscore 和 Zrisk 预测中总体优于线性模型；加入滞后水平和变化特征后，排序能力和 Top20 高风险识别能力进一步提升。
