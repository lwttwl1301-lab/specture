"""
评估 V3.2.1 模型（方案 2.1：小权重 + 后期加入）vs V3 原始模型 vs V3.2 混合损失
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.interpolate import UnivariateSpline
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from scipy.optimize import linear_sum_assignment
import json

print("=" * 70)
print("V3.2.1 模型评估 - 对比 V3 原始和 V3.2")
print("=" * 70)

# 加载响应度
excel_path = r'D:\desktop\try\响应度矩阵.xlsx'
df = pd.read_excel(excel_path)
wavelengths_resp = df.iloc[:, 0].values
response_matrix_full = df.iloc[:, 1:].values

all_biases = np.linspace(-15, 0, 65)
indices = np.linspace(0, 64, 20, dtype=int)
target_wavelengths = np.arange(1000, 1301, 1)
response_20 = np.zeros((len(target_wavelengths), 20))

for j, idx in enumerate(indices):
    orig_resp = response_matrix_full[:, idx]
    spline = UnivariateSpline(wavelengths_resp, orig_resp, s=0.001, k=3)
    interpolated = spline(target_wavelengths)
    smoothed = gaussian_filter1d(interpolated, sigma=1.5)
    response_20[:, j] = smoothed

# 定义模型
class MultiBiasNetV2(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512), nn.ReLU(), nn.BatchNorm1d(512), nn.Dropout(0.2),
            nn.Linear(512, 1024), nn.ReLU(), nn.BatchNorm1d(1024), nn.Dropout(0.2),
            nn.Linear(1024, 1024), nn.ReLU(), nn.BatchNorm1d(1024), nn.Dropout(0.2),
            nn.Linear(1024, 512), nn.ReLU(), nn.BatchNorm1d(512),
            nn.Linear(512, output_dim), nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)

# 加载三个模型
print("\n加载模型...")

# V3 原始模型
checkpoint_v3 = torch.load('model_multi_peak.pth', weights_only=False)
model_v3 = MultiBiasNetV2(20, 301)
model_v3.load_state_dict(checkpoint_v3['model_state_dict'])
model_v3.eval()
scaler_X_v3 = checkpoint_v3['scaler_X']
scaler_y_v3 = checkpoint_v3['scaler_y']

# V3.2 混合损失模型
try:
    checkpoint_v32 = torch.load('model_multi_peak_v2.pth', weights_only=False)
    model_v32 = MultiBiasNetV2(20, 301)
    model_v32.load_state_dict(checkpoint_v32['model_state_dict'])
    model_v32.eval()
    scaler_X_v32 = checkpoint_v32['scaler_X']
    scaler_y_v32 = checkpoint_v32['scaler_y']
    has_v32 = True
    print("  V3.2 混合损失模型已加载")
except:
    has_v32 = False
    print("  V3.2 混合损失模型未找到")

# V3.2.1 方案 2.1 模型
try:
    checkpoint_v321 = torch.load('model_multi_peak_v2_1.pth', weights_only=False)
    model_v321 = MultiBiasNetV2(20, 301)
    model_v321.load_state_dict(checkpoint_v321['model_state_dict'])
    model_v321.eval()
    scaler_X_v321 = checkpoint_v321['scaler_X']
    scaler_y_v321 = checkpoint_v321['scaler_y']
    has_v321 = True
    print("  V3.2.1 方案 2.1 模型已加载")
except:
    has_v321 = False
    print("  V3.2.1 方案 2.1 模型未找到")

print("  V3 原始模型已加载")

# 生成测试数据
print("\n生成测试数据...")

def gaussian(wavelengths, center, fwhm):
    sigma = fwhm / 2.355
    return np.exp(-0.5 * ((wavelengths - center) / sigma) ** 2)

np.random.seed(123)
n_test = 500

test_spectra = []
test_peak_positions = []

for i in range(n_test):
    n_peaks = np.random.randint(1, 6)
    spectrum = np.zeros_like(target_wavelengths, dtype=float)
    peak_positions = []
    
    for _ in range(n_peaks):
        pos = np.random.uniform(1030, 1270)
        intensity = np.random.uniform(0.3, 1.0)
        fwhm = np.random.uniform(15, 50)
        spectrum += intensity * gaussian(target_wavelengths, pos, fwhm)
        peak_positions.append(pos)
    
    test_spectra.append(spectrum)
    test_peak_positions.append(sorted(peak_positions))

test_spectra = np.array(test_spectra)

# 计算测量值
test_measurements = np.zeros((n_test, 20))
for i in range(n_test):
    for j in range(20):
        test_measurements[i, j] = np.sum(test_spectra[i] * response_20[:, j])

print(f"  测试样本：{n_test}")

# 评估函数
def evaluate_model(model, scaler_X, scaler_y, measurements, true_spectra, true_peaks_list, name):
    """评估模型性能"""
    X_scaled = scaler_X.transform(measurements)
    X_tensor = torch.FloatTensor(X_scaled)
    
    with torch.no_grad():
        y_pred_scaled = model(X_tensor).numpy()
    
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    y_pred = np.clip(y_pred, 0, None)
    
    # 曲线级别指标
    mse_per_sample = np.mean((y_pred - true_spectra) ** 2, axis=1)
    corr_per_sample = []
    for i in range(len(true_spectra)):
        c = np.corrcoef(true_spectra[i], y_pred[i])[0, 1]
        corr_per_sample.append(c)
    corr_per_sample = np.array(corr_per_sample)
    
    # 峰级别指标 - 主峰位置误差
    peak_errors = []
    for i in range(len(true_spectra)):
        # 真实主峰位置
        true_main_peak = true_peaks_list[i][0]  # 第一个峰（主峰）
        
        # 预测主峰位置（argmax）
        pred_idx = np.argmax(y_pred[i])
        pred_peak = target_wavelengths[pred_idx]
        
        error = abs(pred_peak - true_main_peak)
        peak_errors.append(error)
    
    peak_errors = np.array(peak_errors)
    
    print(f"\n【{name}】")
    print(f"  MSE: {np.mean(mse_per_sample):.6f}")
    print(f"  相关系数：{np.mean(corr_per_sample):.4f}")
    print(f"  峰位置误差：{np.mean(peak_errors):.2f} nm (中位数：{np.median(peak_errors):.2f} nm)")
    print(f"  <1nm 精度：{np.sum(peak_errors < 1) / len(peak_errors) * 100:.1f}%")
    print(f"  <2nm 精度：{np.sum(peak_errors < 2) / len(peak_errors) * 100:.1f}%")
    print(f"  <5nm 精度：{np.sum(peak_errors < 5) / len(peak_errors) * 100:.1f}%")
    
    return {
        'mse': float(np.mean(mse_per_sample)),
        'corr': float(np.mean(corr_per_sample)),
        'peak_error_mean': float(np.mean(peak_errors)),
        'peak_error_median': float(np.median(peak_errors)),
        'acc_1nm': float(np.sum(peak_errors < 1) / len(peak_errors) * 100),
        'acc_2nm': float(np.sum(peak_errors < 2) / len(peak_errors) * 100),
        'acc_5nm': float(np.sum(peak_errors < 5) / len(peak_errors) * 100),
    }

# 评估模型
results = {}
results['v3_original'] = evaluate_model(model_v3, scaler_X_v3, scaler_y_v3, 
                            test_measurements, test_spectra, test_peak_positions, 
                            "V3 原始模型")

if has_v32:
    results['v3_2_mixed_loss'] = evaluate_model(model_v32, scaler_X_v32, scaler_y_v32, 
                                 test_measurements, test_spectra, test_peak_positions, 
                                 "V3.2 混合损失模型")

if has_v321:
    results['v3_2_1_late_peak'] = evaluate_model(model_v321, scaler_X_v321, scaler_y_v321, 
                                  test_measurements, test_spectra, test_peak_positions, 
                                  "V3.2.1 方案 2.1 模型")

# 对比
print("\n" + "=" * 70)
print("对比总结")
print("=" * 70)

print(f"""
| 指标 | V3 原始 | V3.2 混合损失 | V3.2.1 方案 2.1 |
|------|--------|--------------|--------------|
| MSE | {results['v3_original']['mse']:.6f} | {results.get('v3_2_mixed_loss', {}).get('mse', 'N/A')} | {results.get('v3_2_1_late_peak', {}).get('mse', 'N/A')} |
| 相关系数 | {results['v3_original']['corr']:.4f} | {results.get('v3_2_mixed_loss', {}).get('corr', 'N/A')} | {results.get('v3_2_1_late_peak', {}).get('corr', 'N/A')} |
| 峰误差 (nm) | {results['v3_original']['peak_error_mean']:.2f} | {results.get('v3_2_mixed_loss', {}).get('peak_error_mean', 'N/A')} | {results.get('v3_2_1_late_peak', {}).get('peak_error_mean', 'N/A')} |
| <1nm 精度 | {results['v3_original']['acc_1nm']:.1f}% | {results.get('v3_2_mixed_loss', {}).get('acc_1nm', 'N/A')} | {results.get('v3_2_1_late_peak', {}).get('acc_1nm', 'N/A')} |
| <2nm 精度 | {results['v3_original']['acc_2nm']:.1f}% | {results.get('v3_2_mixed_loss', {}).get('acc_2nm', 'N/A')} | {results.get('v3_2_1_late_peak', {}).get('acc_2nm', 'N/A')} |
| <5nm 精度 | {results['v3_original']['acc_5nm']:.1f}% | {results.get('v3_2_mixed_loss', {}).get('acc_5nm', 'N/A')} | {results.get('v3_2_1_late_peak', {}).get('acc_5nm', 'N/A')} |
""")

# 保存结果
with open('v3_all_comparison.json', 'w') as f:
    json.dump(results, f, indent=2)

print("结果已保存到：v3_all_comparison.json")
