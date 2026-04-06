# 光谱重建神经网络模型 V2 - 技术报告

## 项目概述

本项目实现了一个基于多偏压InGaAs传感器的光谱重建神经网络，能够从20个偏压下的测量值重建1000-1300nm范围内的光谱。

---

## 1. 最终性能

| 指标 | 数值 |
|------|------|
| 平均峰值误差 | 0.99nm |
| 中位数误差 | 1.00nm |
| 最大误差 | 13.00nm |
| <1nm精度 | 45.0% |
| <2nm精度 | 81.3% |
| **<5nm精度** | **96.5%** |
| <10nm精度 | 99.4% |

### 分区域误差

| 波段 | 平均误差 | 样本数 |
|------|----------|--------|
| 1000-1050nm (低) | 0.72nm | ~400 |
| 1050-1250nm (中) | 0.69nm | ~1600 |
| 1250-1300nm (高) | 1.66nm | ~400 |

---

## 2. 核心代码结构

### 2.1 训练脚本 (train_multi_bias_v2.py)

#### 关键配置

```python
# 偏压配置：只用负偏压（-15V到0V）
all_biases = np.linspace(-15, 0, 65)  # 原始65个偏压
indices = np.linspace(0, 64, 20, dtype=int)  # 选择20个
selected_biases = all_biases[indices]  # [-15.0, -13.99, ..., 6.6]

# 波长配置
target_wavelengths = np.arange(1000, 1301, 1)  # 301个点
```

#### 数据生成（边界过采样）

```python
n_main = 6000   # 主流样本（1000-1300均匀）
n_edge = 2000   # 边界过采样

# 主流：均匀分布
peak_wavelengths_main = np.random.uniform(1000, 1300, n_main)

# 边界1：1000-1050nm（低边界）
peak_wavelengths_low = np.random.uniform(1000, 1050, n_edge)

# 边界2：1250-1300nm（高边界）
peak_wavelengths_high = np.random.uniform(1250, 1300, n_edge)
```

#### 模型架构（~100万参数）

```python
class MultiBiasNetV2(nn.Module):
    def __init__(self, input_dim=20, output_dim=301):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(20, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.2),
            
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.2),
            
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.2),
            
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            
            nn.Linear(512, 301),
            nn.Sigmoid()
        )
```

#### 加权损失函数

```python
# 边界区域权重更高
weights = np.ones(301)
weights[target_wavelengths <= 1050] = 3.0   # 低边界3倍
weights[target_wavelengths >= 1250] = 3.0   # 高边界3倍

def weighted_mse_loss(pred, target, weights):
    return torch.mean(weights * (pred - target) ** 2)
```

#### 训练配置

```python
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2)
epochs = 600
```

---

## 3. 关键改进点（从V1到V2）

| 问题 | V1 | V2 | 改进效果 |
|------|-----|-----|----------|
| 偏压范围 | -15V~6.6V（包含正偏压） | -15V~0V（只用负偏压） | 消除正偏压噪声 |
| 边界样本 | 不足 | 边界过采样各2000个 | 边界误差从299nm降到1.66nm |
| 模型容量 | ~7万参数 | ~100万参数 | 欠拟合问题解决 |
| 损失函数 | 普通MSE | 加权MSE（边界3倍） | 边界精度提升 |
| 训练轮数 | <200 | 600 + 早停 | 收敛更充分 |

---

## 4. 数据流程

```
响应度矩阵(Excel)
      ↓
  插值到1nm + 平滑
      ↓
响应度矩阵 (301×20)
      ↓
生成高斯光谱 (峰值1000-1300nm)
      ↓
计算20个偏压下的测量值
      ↓
标准化 → 训练/测试分割
      ↓
神经网络训练
      ↓
预测 → 反标准化
      ↓
计算峰值误差
```

---

## 5. 文件说明

| 文件 | 说明 |
|------|------|
| train_multi_bias_v2.py | 训练脚本 |
| comprehensive_analysis.py | 分析脚本 |
| model_multi_bias_v2.pth | 模型权重 |
| training_multi_bias_v2.png | 训练曲线 |
| comprehensive_analysis.png | 综合分析图 |
| sample_comparison_25.png | 25个样本对比 |
| sample_comparison_sorted.png | 24个样本按波段 |
| additional_analysis.png | 更多分析 |

---

## 6. 使用方法

### 训练
```bash
python train_multi_bias_v2.py
```

### 分析
```bash
python comprehensive_analysis.py
```

### 加载模型预测
```python
import torch

checkpoint = torch.load('model_multi_bias_v2.pth', weights_only=False)
model = MultiBiasNetV2(20, 301)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# 使用scaler_X标准化输入，scaler_y反标准化输出
```

---

## 7. 结论

V2模型通过以下关键改进实现了<5nm精度96.5%：

1. **修正偏压范围**：只用-15V到0V的负偏压
2. **边界过采样**：1000-1050nm和1250-1300nm各2000个样本
3. **模型扩容**：从7万参数扩到100万参数
4. **加权损失**：边界区域权重3倍
5. **训练优化**：600轮 + AdamW + 余弦退火

---

*报告生成时间: 2026年4月*