# 光谱重建神经网络模型 V2

## 项目概述
基于多偏压InGaAs传感器的光谱重建神经网络，能够从20个偏压下的测量值重建1000-1300nm范围内的光谱。

## 性能指标
- **平均峰值误差**: 0.99nm
- **中位数误差**: 1.00nm
- **<5nm精度**: 96.5%
- **<2nm精度**: 81.3%
- **<1nm精度**: 45.0%

## 文件结构
```
final_v2_package/
├── train_multi_bias_v2.py          # 主训练脚本
├── comprehensive_analysis.py       # 综合分析脚本
├── model_multi_bias_v2.pth         # 训练好的模型权重 (9.1MB)
├── V2_MODEL_REPORT.md              # 技术报告文档
├── check_data.py                   # 数据完整性检查脚本
├── config.py                       # 配置文件
├── training_multi_bias_v2.png      # 训练曲线图
├── comprehensive_analysis.png      # 综合分析图
├── sample_comparison_25.png        # 25个样本对比图
├── sample_comparison_sorted.png    # 按波段排序的24个样本对比图
└── additional_analysis.png         # 更多分析图
```

## 数据依赖
### 必需文件
1. **响应度矩阵.xlsx** - 传感器响应度数据
   - 位置: `D:\desktop\try\响应度矩阵.xlsx` (原始位置)
   - 需要复制到项目目录或修改脚本路径

### 数据文件处理
```bash
# 将响应度文件复制到项目目录
copy "D:\desktop\try\响应度矩阵.xlsx" .
```

## 快速开始
### 1. 检查数据完整性
```bash
python check_data.py
```

### 2. 运行训练（如果需要重新训练）
```bash
python train_multi_bias_v2.py
```

### 3. 运行分析
```bash
python comprehensive_analysis.py
```

## 模型使用
### 加载模型进行预测
```python
import torch
import numpy as np

# 加载模型
checkpoint = torch.load('model_multi_bias_v2.pth', map_location='cpu', weights_only=False)
model = MultiBiasNetV2(20, 301)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# 准备输入数据 (20个偏压测量值)
measurements = np.random.randn(1, 20)  # 示例数据
measurements_scaled = checkpoint['scaler_X'].transform(measurements)

# 预测
with torch.no_grad():
    predictions = model(torch.FloatTensor(measurements_scaled))
    predictions = predictions.numpy()
    predictions_original = checkpoint['scaler_y'].inverse_transform(predictions)

print(f"预测光谱形状: {predictions_original.shape}")
```

## 技术特点
### 关键改进（V1 → V2）
1. **偏压范围优化**: 只用-15V到0V的负偏压，消除正偏压噪声
2. **边界过采样**: 1000-1050nm和1250-1300nm区域各增加2000个训练样本
3. **模型扩容**: 从7万参数扩展到100万参数，解决欠拟合问题
4. **加权损失**: 边界区域权重3倍，提高边界精度
5. **训练优化**: 600轮训练 + AdamW优化器 + 余弦退火调度

### 模型架构
- 输入层: 20个节点（20个偏压测量值）
- 隐藏层: 512 → 1024 → 1024 → 512
- 输出层: 301个节点（301个波长点）
- 激活函数: ReLU + Sigmoid（输出层）
- 正则化: BatchNorm + Dropout(0.2)
- 总参数量: ~100万

## 数据流程
```
响应度矩阵(Excel) → 插值到1nm + 平滑 → 响应度矩阵(301×20)
    ↓
生成高斯光谱(峰值1000-1300nm) → 计算20个偏压下的测量值
    ↓
标准化 → 训练/测试分割 → 神经网络训练
    ↓
预测 → 反标准化 → 计算峰值误差
```

## 配置说明
详细配置请查看 `config.py` 文件，包含：
- 数据文件路径
- 模型参数
- 训练参数
- 数据生成配置
- 损失函数权重
- 分析配置

## 问题排查
### 常见问题
1. **文件找不到错误**
   - 运行 `python check_data.py` 检查文件完整性
   - 确保 `响应度矩阵.xlsx` 在正确位置

2. **模型加载失败**
   - 检查PyTorch版本兼容性
   - 确保使用 `weights_only=False` 参数

3. **内存不足**
   - 减少批量大小（修改 `config.py` 中的 `batch_size`）
   - 使用GPU加速（如果可用）

## 联系方式
如有问题，请参考技术报告 `V2_MODEL_REPORT.md` 或检查代码注释。

---
*项目最后更新: 2026年4月*