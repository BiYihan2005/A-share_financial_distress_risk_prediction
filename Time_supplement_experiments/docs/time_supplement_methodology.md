# Time Supplement Methodology

## 1. 研究设定

时间维度补充实验的核心目标是从随机切分的同期拟合，扩展到更接近真实应用的未来一期风险预测。

实验一：

$$
X_{i,t} \longrightarrow Y_{i,t+1}
$$

实验二：

$$
\left[X_{i,t}, X_{i,t-1}, \Delta X_{i,t}\right] \longrightarrow Y_{i,t+1},
\qquad \Delta X_{i,t}=X_{i,t}-X_{i,t-1}
$$

## 2. 避免数据泄露

- 先按时间切分训练集、验证集、测试集，再做填补、缩尾、标准化和编码。
- 所有预处理参数只在训练集上拟合。
- `统计截止日期` 只用于排序、切分和 lead/lag 构造，不作为普通解释变量输入模型。
- 目标变量是同一股票下一期的 Oscore / Zrisk，解释变量必须来自目标发生之前的报告期。

## 3. 目标变量

$$
Zrisk_i=-Zscore_i
$$

$$
Y_{i,t+1}^{Oscore}=Oscore_{i,t+1}, \qquad Y_{i,t+1}^{Zrisk}=Zrisk_{i,t+1}
$$

## 4. 指标解释

RMSE 和 MAE 衡量连续预测误差；$R^2$ 衡量解释能力；Spearman 衡量排序能力；Top20 F1 衡量模型能否识别未来高风险公司。

$$
F1_{Top20}=\frac{2\cdot Precision_{Top20}\cdot Recall_{Top20}}{Precision_{Top20}+Recall_{Top20}}
$$
