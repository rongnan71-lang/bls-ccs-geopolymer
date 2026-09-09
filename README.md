# BLS+CCS 纯净版 v2.5

> **目标引导RBF原型嵌入的残差校正宽度学习 — 抗灾地聚合物材料性能预测**

[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

本项目采用 **CC BY-NC-SA 4.0** 协议开源：允许分享与改编，但**禁止商用**，衍生作品须以**相同协议**发布并**署名原仓库**。详见 [LICENSE](LICENSE)。

---

## 目录结构

```
纯净版/
├── bls.py                          # BLS核心 (封闭解岭回归 + 滑动窗口缓冲重解式增量更新)
├── ccs/
│   ├── __init__.py                 # CCS包初始化
│   ├── concept_space.py            # 概念空间 (记录级merge/delete, 删除后重建节点统计, 标准Welford)
│   └── memory_layer.py             # 材料记忆层封装 (MaterialMemory)
├── experiments/
│   ├── utils.py                    # 数据加载/物理交互特征/RBF原型嵌入/模型工厂(9模型)
│   ├── run_benchmark.py            # 基准对比 (9模型, 5折x3种子, LOSO, paired显著性检验)
│   ├── run_ablation.py             # 分阶段消融 (A1-A6, 严格nested CV)
│   ├── run_forgetting.py           # 持续学习实验 (CCS记录保持 vs BLS读出漂移)
│   ├── run_sparse_ablation.py      # 稀疏特征剔除消融 (27维 vs 11维)
│   ├── make_figures.py             # 论文图表生成 (fig1-fig6)
│   └── make_sparse_figures.py      # 稀疏特征对比图表 (fig7-fig8)
├── optimization/
│   ├── nsga2.py                    # NSGA-II多目标优化 (纯numpy, 可行域投影)
│   └── run_optimize.py             # 材料配方帕累托搜索 (物理约束, 派生量自动计算)
├── data/
│   └── build_dataset.py            # 数据集构建脚本
├── examples/
│   ├── bls_demo.py                 # BLS基础演示 (回归+增量学习)
│   └── memory_demo.py              # CCS记忆层演示 (添加/检索/记录级删除/持久化)
├── figures/                        # 论文图表 (可由experiments脚本重新生成)
├── run_all.py                      # 一键实验 (基准→消融→持续学习→图表→优化)
├── requirements.txt
├── LICENSE                         # CC BY-NC-SA 4.0
└── README.md
```

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 准备数据集
# 将数据放置为CSV格式（含f_前缀特征列 + 目标列ucs_kpa/permeability_cm_s），
# 并在各实验脚本中指定数据路径。或使用 data/build_dataset.py 从原始xlsx构建。

# 运行全部实验 (约15分钟)
python run_all.py

# 或单独运行
python experiments/run_benchmark.py    # 基准对比 (~8分钟)
python experiments/run_ablation.py     # 消融实验 (~6分钟, 严格nested CV)
python experiments/run_forgetting.py   # 持续学习实验
python experiments/make_figures.py     # 生成6张图表
python optimization/run_optimize.py    # NSGA-II优化
```

## 核心方法

1. **物理交互特征工程**: 5个无量纲数 (水胶比、膨润土胶凝比、固液比、矿渣占胶凝比、水泥占胶凝比)
2. **目标引导RBF原型嵌入**: 按目标值/残差分位数分层选25个原型(余数均匀分配), RBF核映射, 原型相似度隐式定义样本-原型二部图
3. **GBDT+BLS残差校正**: GBDT捕捉主要非线性, 带RBF原型嵌入的BLS拟合GBDT残差 (prototype基于残差选择, shrinkage内层CV选择, RBF scale固定为主RBF-BLS所选值)
4. **CCS概念空间记忆层**: 记录级可审计记忆, 每个节点维护完整records列表, 支持按记录删除(删除后重建节点统计), 与回归层形成"记忆-插值"互补分工

## 关键结果 (5折CV在3种随机划分种子下重复)

注: BLS的随机增强节点初始化种子固定为42, 三个seed主要改变KFold数据划分, 不改变BLS内部随机初始化。

| 目标 | GBDT | 本文方法 | ΔR² | paired t p | 结论 |
|------|------|---------|-----|-----------|------|
| UCS抗压强度 | 0.822 | **0.825** | +0.002 | 0.598 | 有正向提升但未达统计显著 |
| 渗透系数 -log₁₀(k) | 0.819 | **0.830** | +0.011 | 0.039 | 显著优于GBDT(p<0.05) |

注: repeated CV的fold scores非完全独立, 上述检验为参考性统计推断。Stacking对照模型在渗透上略优(0.831), 本文方法在UCS上与Stacking持平(0.825)。

消融实验 (严格nested CV, A5/A6每个inner fold内重新选prototype):
- UCS: 监督原型(A6=0.533) > 无监督原型(A5=0.508), 说明监督原型收益任务依赖
- 渗透: 监督原型(A6=0.767) > 无监督原型(A5=0.747)
- RBF原型嵌入的主要价值在于与GBDT残差校正的组合, 而非单独提升BLS预测

稀疏特征剔除消融 (阈值80%预先定义, 剔除16个NaN>80%特征):
- GBDT UCS: 0.822 → 0.553 (-0.269)
- GBDT 渗透: 0.819 → 0.541 (-0.278)
- 结论: "稀疏≠无用", 少数有值样本携带关键预测信息, 本数据集不应剔除稀疏特征

持续学习: CCS 100%早期记录保持(identity retention), BLS增量重解式更新MAE漂移+4%
LOSO: 所有模型R²为负且源级方差大, 提示随机划分可能高估跨来源部署能力

## 数据集特征说明

27维特征:
- 18个统一f_特征, 其中包含15个原始组分/工艺变量 + 3个组成汇总/派生变量 (f_binder_cem_ggbs, f_slurry, f_binder_total)
- 4个派生特征 (胶凝总量、膨润土总量、矿渣率、围压)
- 5个物理交互特征 (无量纲数)

目标变量: UCS抗压强度(kPa)、渗透系数k(cm/s)、-log₁₀(k)

缺失值处理: 原始数据中NaN表示"该文献未使用该组分"(用量为0), 加载时统一填充为0。目标列无缺失。

## 复现性

所有随机种子固定 (42), 所有预处理仅在训练折内拟合, 无数据泄漏。
A5/A6消融采用严格嵌套交叉验证(nested CV), 每个inner fold内重新选择prototype。
在固定依赖版本 (numpy/scipy/sklearn/pandas/matplotlib) 与随机种子的环境下可复现。
注意: 跨操作系统、Python版本或依赖版本的浮点数结果可能有微小差异, 图像文件不保证逐字节一致。

## 许可证

本项目基于 **CC BY-NC-SA 4.0**（知识共享 署名-非商业性使用-相同方式共享 4.0 国际）协议发布。

- **BY（署名）**：使用或改编时须署名原仓库
- **NC（非商业）**：禁止用于商业目的
- **SA（相同方式共享）**：衍生作品须以相同协议发布

完整协议文本见 [LICENSE](LICENSE)，或访问 https://creativecommons.org/licenses/by-nc-sa/4.0/
