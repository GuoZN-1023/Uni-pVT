# Uni-pVT: A Mixture-of-Experts-Based Neutral Network for Universal Fluid Thermodynamic Property Prediction

Uni-pVT 是一个面向 pVT / 物性建模的深度学习小平台，当前以**压缩因子 Z** 为主（未来可扩展到 φ 等其他性质），采用 **Mixture-of-Experts（MoE） + 门控网络（Gate）+ PINN** 的结构，并集成了

- 数据读取与标准化  
- 专家网络预训练  
- MoE端到端训练  
- 自动评估与可视化 

借助统一的配置文件和 `run_all.py`，可以一键完成**训练 → 预测 → 结果汇总**，适合高通量扫描不同数据集与模型超参数。

---

## 1. 模型与整体思路简介

### 1.1 Mixture-of-Experts 结构

`FusionModel` 由四个专家网络（experts）和一个门控网络（gate）组成：

- `expert_gas`：主要负责气相区域  
- `expert_liq`：主要负责液相区域  
- `expert_crit`：主要负责临界及复杂区域  
- `expert_extra`：可灵活分配到你希望关注的额外区域（例如某个特定压力/温度区间）

每个专家都是一个多层全连接 MLP，门控网络接收同样的输入特征，输出四个 softmax 权重 `w₁ … w₄`。  
最终预测为四个专家输出的加权和：

$$\[
\hat{Z}(x) = \sum_{k=1}^4 w_k(x)\, f_k(x)
\]$$

### 1.2 两阶段训练策略

训练过程分为两个阶段：

1. **专家预训练（hard routing）**  
   使用数据中的 `no` 编号（1/2/3/4）作为“硬分区标签”，将样本直接路由到对应专家，只更新该专家参数。这样可以让每个专家先学会自己负责区域的大致形状。

2. **MoE 训练（gate + MoE）**  
   冻结四个专家，只训练门控网络，使 gate 学会根据输入特征自动给四个专家分配权重，实现更柔性、更物理的分区。

这种方式结合了“人为分区先验”（通过 `no`）与“数据驱动分配”（通过 gate），在复杂 pVT 空间上具有较好的表达能力与可解释性。

---

## 2. 环境与依赖

建议使用 Python 3.10+，主要依赖如下：

- `torch` ≥ 2.0  
- `numpy`  
- `pandas`  
- `scikit-learn`（用于指标计算）  
- `plotly`（交互式可视化）  
- `pyyaml`  

可以使用 `conda` 或 `pip` 进行安装，例如：

```bash
#配置环境
conda create -n pVTPred
conda activate pVTPred
```

```bash
#安装所需库
cd ./Uni-pVT

#直接安装：
pip install torch numpy pandas scikit-learn plotly pyyaml

#或者可以使用requirements安装
pip install -r requirements.txt

```

---

## 3. 数据格式要求

当前默认使用单个 CSV 文件作为数据输入，每一行对应一个状态点。
典型列名示例（以 Water.csv 为例）：
- T_r (-)：还原温度
- p_r (-)：还原压力
- omega (-)：偏心因子
- p_c (Pa)：临界压力
- T_c (K)：临界温度
- mu ((J*m^3/kmol)^0.5)：偶极矩
- Z (-)：目标压缩因子
- phi (-)：其他性质（可暂不使用）
- no：区域编号（1/2/3/4），用于专家预训练和分区分析
在 config/config.yaml 中可以指定：

```yaml
target_col: "Z (-)" #需要预测性质对应的列索引
expert_col: "no" #编号
paths:
  data: data/Water.csv #数据的相对路径
```

如果需要预测其他性质，改变对应的列索引即可

---

## 4. 配置文件（config/config.yaml）
核心配置集中在 config/config.yaml。一个典型配置示例：

```yaml
experts: #专家系统各个层信息、激活函数、dropout
  gas:
    hidden_layers: [128, 64, 32]
    activation: relu
    dropout: 0.10
  liquid:
    hidden_layers: [256, 128, 64]
    activation: relu
    dropout: 0.15
  critical:
    hidden_layers: [512, 256, 128]
    activation: relu
    dropout: 0.20
  extra:
    hidden_layers: [256, 128, 64]
    activation: relu
    dropout: 0.15

gate: #门控系统各个层信息、激活函数、dropout
  hidden_layers: [128, 64]
  activation: relu
  dropout: 0.10

model:
  input_dim: 8          # 会在数据加载时自动覆盖，无需手动修改

training:
  batch_size: 64 
  learning_rate: 0.005  # 学习率
  epochs: 200           # 阶段二最大 epoch
  pretrain_epochs: 50   # 阶段一专家预训练 epoch 数
  early_stopping_patience: 20

loss:
  lambda_nonneg: 0.05   # 预测值非负约束
  lambda_smooth: 0.02   # 预测随样本变化的平滑约束
  lambda_extreme: 1.0   # Z≈1 区域的权重强度（见 Section 6）
  lambda_relative: 0.0  # 如不需要相对误差，可设 0
  extreme_alpha: 4.0    # Z≈1 处的峰值放大系数

paths:
  data: data/Water.csv
  save_dir: results/latest
  scaler: results/latest/scaler.pkl

target_col: "Z (-)"
expert_col: "no"

# 可选：子集抽样配置（如只取前 N 条、或按条件筛选）
# subset:
#   type: "head"
#   n: 30000
```

---

## 5. 快速上手：一键跑完训练 + 预测
仓库根目录下提供了一个统一入口脚本 run_all.py

在终端中进入项目根目录：

```bash
cd Uni-pVT
python run_all.py --config config/config.yaml
```

---

run_all.py 会执行以下流程：

复制当前配置到新的结果目录 results/YYYY-MM-DD_HH-MM-SS/

调用 train.py 进行模型训练

调用 predict.py 在测试集上做预测与可视化

将训练与预测的信息汇总到 summary.json 中

输出示例（终端日志中）会包含：

实验时间戳与结果目录

设备信息（CPU/GPU）

训练过程关键日志（loss、R²、gate 权重等）

预测阶段的整体指标与文件路径

---

## 6. 损失函数设计
utils/physics_loss.py 定义了组合损失 PhysicsLoss，它包含以下部分：

Weighted MSE（主数据项）

对每个样本的误差 $$(y_pred - y_true)²$$ 乘以权重 $$w_i$$

在普通区域，$$w_i ≈ 1$$

在 Z ≈ 1 的窄带内（默认 1 ± 0.03），$$w_i$$ 会被放大到约 $$1 + lambda_extreme * extreme_alpha$$

这样训练时会更关注高密度且物理上重要的 Z≈1 区域，提升这部分的拟合精度

NonNeg Penalty

对预测值为负的部分施加惩罚，保证物性量（如 Z）不出现明显非物理由。

Smooth Penalty

对同一 batch 内相邻样本预测差 $$(y_pred[i+1] - y_pred[i])²$$ 的平均值进行惩罚，让模型输出在特征空间上更加平滑，减少不必要的振荡。

Relative Error Loss（可选）

如果开启 lambda_relative，会额外引入相对误差 $$(y_pred - y_true)/|y_true|$$ 的平方平均，适合在不同 Z 区间权重相对误差时使用；当前通常设为 0。

通过调节 lambda_extreme 与 extreme_alpha，可以在“整体拟合”与“Z≈1 精度优先”之间进行平衡。

---

## 7. 结果解读方式
可以从以下几个角度理解和诊断模型表现：

整体拟合情况

查看 eval/metrics_summary.yaml 中的 MAE / MSE / R²，和 true_vs_pred_scatter.html 中散点相对于对角线的分布。

分区表现

通过 region_eval/region_metrics.csv，观察 expert_id 为 1/2/3/4 的分区 MAE / MSE / R²，对比不同区域的难度和模型拟合情况。

门控行为

打开 plots/gating_weights.html，查看训练过程中 gate 对四个专家的平均权重变化，判断是否出现“某个 expert 长期几乎不用”或“门控一直平均分配”这类现象。

损失分量与物理约束

在 plots/loss_components.html 中检查 NonNeg / Smooth / WeightedMSE 等分量是否在一个合理量级，避免物理约束项过大导致欠拟合，也避免过小失去约束效果。

---

## 8. 参数调整经验
如果整体 R² 不够高或散点偏离对角线，可以尝试：

调小 training.learning_rate，适当增加 epochs 与 early_stopping_patience

调节专家网络的层数与宽度（例如增加 critical / extra 的 hidden_layers）

合理调整损失中的权重：

lambda_nonneg / lambda_smooth 过大时会导致欠拟合

lambda_extreme / extreme_alpha 决定了 Z≈1 区域的优先程度，过低则竖带松散，过高可能损害其它区间

根据数据实际分布，微调 physics_loss.py 中 center_width，使 Z≈1 的高权重区间与真实高密度区域匹配

如果只想快速看模型是否“学到了主趋势”，可以先把物理约束和加权全部关掉，只保留纯 MSE，然后再逐步加回各项正则。

## 9. 文件结构
以下文件是在训模型需要用到的文件，省略了其他用不到的文件。
```text
Uni-pVT/
  ├─ config/
  │   └─ config.yaml           # 主配置文件
  ├─ data/
  │   └─ Water.csv             # 示例数据（可放多份 CSV）
  ├─ models/
  │   ├─ fusion_model.py       # MoE 模型定义（4 个 expert + gate）
  │   ├─ experts.py            # experts 模型
  │   └─ gate.py               # gate 模型
  ├─ utils/
  │   ├─ dataset.py            # ZDataset 与 DataLoader 构建
  │   ├─ physics_loss.py       # PhysicsLoss 组合损失
  │   ├─ trainer.py            # 两阶段训练循环 + 可视化
  │   ├─ logger.py             # 文本日志封装
  │   ├─ visualize.py            # experts 模型
  │   └─ phase_visualizer.py               # gate 模型
  ├─ run_all.py                # 一键“训练 + 预测 + 汇总”入口
  ├─ train.py                  # 仅训练入口
  ├─ predict.py                # 仅预测与评估入口
  └─ results/
      ├─ latest/               # 最近一次实验结果（可由配置指定）
      └─ YYYY-MM-DD_HH-MM-SS/  # 带时间戳的实验目录
          ├─ checkpoints/      # best_model.pt
          ├─ plots/            # 各种 HTML 曲线和训练指标 CSV
          ├─ eval/             # 整体测试评估文件
          ├─ region_eval/      # 分区评估文件
          ├─ logs/             # 训练 / 预测日志
          └─ summary.json      # 实验摘要
```

---






