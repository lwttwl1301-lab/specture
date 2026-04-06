"""
评估V3.2模型（混合损失训练）vs V3原始模型
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
print("V3.2模型评估 - 对比V3原始模型")
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

# 加载两个模型
print("\n加载模型...")

# V3原始模型
checkpoint_v3 = torch.load('model_multi_peak.pth', weights_only=False)
model_v3 = MultiBiasNetV2(20, 301)
model_v3.load_state_dict(checkpoint_v3['model_state_dict'])
model_v3.eval()
scaler_X_v3 = checkpoint_v3['scaler_X']
scaler_y_v3 = checkpoint_v3['scaler_y']

# V3.2混合损失模型
checkpoint_v32 = torch.load('model_multi_peak_v2.pth', weights_only=False)
model_v32 = MultiBiasNetV2(20, 301)
model_v32.load_state_dict(checkpoint_v32['model_state_dict'])
model_v32.eval()
scaler_X_v32 = checkpoint_v32['scaler_X']
scaler_y_v32 = checkpoint_v32['scaler_y']

print("  V3原始模型已加载")
print("  V3.2混合损失模型已加载")

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

print(f"  测试样本: {n_test}")

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
    print(f"  相关系数: {np.mean(corr_per_sample):.4f}")
    print(f"  峰位置误差: {np.mean(peak_errors):.2f} nm (中位数: {np.median(peak_errors):.2f} nm)")
    print(f"  <1nm精度: {np.sum(peak_errors < 1) / len(peak_errors) * 100:.1f}%")
    print(f"  <2nm精度: {np.sum(peak_errors < 2) / len(peak_errors) * 100:.1f}%")
    print(f"  <5nm精度: {np.sum(peak_errors < 5) / len(peak_errors) * 100:.1f}%")
    
    return {
        'mse': float(np.mean(mse_per_sample)),
        'corr': float(np.mean(corr_per_sample)),
        'peak_error_mean': float(np.mean(peak_errors)),
        'peak_error_median': float(np.median(peak_errors)),
        'acc_1nm': float(np.sum(peak_errors < 1) / len(peak_errors) * 100),
        'acc_2nm': float(np.sum(peak_errors < 2) / len(peak_errors) * 100),
        'acc_5nm': float(np.sum(peak_errors < 5) / len(peak_errors) * 100),
    }

# 评估两个模型
results_v3 = evaluate_model(model_v3, scaler_X_v3, scaler_y_v3, 
                            test_measurements, test_spectra, test_peak_positions, 
                            "V3原始模型")

results_v32 = evaluate_model(model_v32, scaler_X_v32, scaler_y_v32, 
                             test_measurements, test_spectra, test_peak_positions, 
                             "V3.2混合损失模型")

# 对比
print("\n" + "=" * 70)
print("对比总结")
print("=" * 70)

print(f"""
| 指标 | V3原始 | V3.2混合损失 | 变化 |
|------|--------|--------------|------|
| MSE | {results_v3['mse']:.6f} | {results_v32['mse']:.6f} | {((results_v32['mse']-results_v3['mse'])/results_v3['mse']*100):+.1f}% |
| 相关系数 | {results_v3['corr']:.4f} | {results_v32['corr']:.4f} | {((results_v32['corr']-results_v3['corr'])/results_v3['corr']*100):+.1f}% |
| 峰误差(nm) | {results_v3['peak_error_mean']:.2f} | {results_v32['peak_error_mean']:.2f} | {((results_v32['peak_error_mean']-results_v3['peak_error_mean'])/results_v3['peak_error_mean']*100):+.1f}% |
| <1nm精度 | {results_v3['acc_1nm']:.1f}% | {results_v32['acc_1nm']:.1f}% | {results_v32['acc_1nm']-results_v3['acc_1nm']:+.1f}% |
| <2nm精度 | {results_v3['acc_2nm']:.1f}% | {results_v32['acc_2nm']:.1f}% | {results_v32['acc_2nm']-results_v3['acc_2nm']:+.1f}% |
""")

# 保存结果
comparison = {
    'v3_original': results_v3,
    'v3_2_mixed_loss': results_v32,
    'improvement': {
        'mse_percent': (results_v32['mse']-results_v3['mse'])/results_v3['mse']*100,
        'peak_error_percent': (results_v32['peak_error_mean']-results_v3['peak_error_mean'])/results_v3['peak_error_mean']*100,
        'acc_1nm_change': results_v32['acc_1nm']-results_v3['acc_1nm'],
    }
}

with open('v3_v32_comparison.json', 'w') as f:
    json.dump(comparison, f, indent=2)

print("结果已保存到: v3_v32_comparison.json")
