"""
双峰预测质量分析 - 12个好结果 + 12个差结果
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from scipy.interpolate import UnivariateSpline
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

print('=' * 70)
print('双峰预测质量分析 - 12好+12差')
print('=' * 70)

# ============================================================================
# 1. 加载响应度
# ============================================================================
print('\n[1/4] 加载响应度...')

excel_path = r'D:\desktop\try\响应度矩阵.xlsx'
df = pd.read_excel(excel_path)
wavelengths_resp = df.iloc[:, 0].values
response_matrix_full = df.iloc[:, 1:].values

all_biases = np.linspace(-15, 0, 65)
indices = np.linspace(0, 64, 20, dtype=int)
selected_biases = all_biases[indices]

target_wavelengths = np.arange(1000, 1301, 1)
response_20 = np.zeros((len(target_wavelengths), 20))

for j, idx in enumerate(indices):
    orig_resp = response_matrix_full[:, idx]
    spline = UnivariateSpline(wavelengths_resp, orig_resp, s=0.001, k=3)
    interpolated = spline(target_wavelengths)
    smoothed = gaussian_filter1d(interpolated, sigma=1.5)
    response_20[:, j] = smoothed

print(f'  响应度矩阵: {response_20.shape}')

# ============================================================================
# 2. 定义模型和加载权重
# ============================================================================
class DoublePeakNet(nn.Module):
    def __init__(self, input_dim=20, output_dim=301):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
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
            
            nn.Linear(512, output_dim),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.net(x)

print('\n[2/4] 加载双峰专用模型...')
checkpoint = torch.load('model_double_peak.pth', weights_only=False)
model = DoublePeakNet(20, 301)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
scaler_X = checkpoint['scaler_X']
scaler_y = checkpoint['scaler_y']

print('  双峰专用模型: 已加载')

# ============================================================================
# 3. 生成大量双峰测试数据
# ============================================================================
print('\n[3/4] 生成双峰测试数据...')

def gaussian(wavelengths, center, fwhm):
    sigma = fwhm / 2.355
    return np.exp(-0.5 * ((wavelengths - center) / sigma) ** 2)

np.random.seed(999)

# 生成1000个双峰样本用于筛选
n_samples = 1000
test_spectra = []
test_measurements = []
test_peak_positions = []

for i in range(n_samples):
    pos1 = np.random.uniform(1000, 1250)
    pos2 = np.random.uniform(pos1 + 30, min(1300, pos1 + 200))
    intensity1 = np.random.uniform(0.3, 1.0)
    intensity2 = np.random.uniform(0.3, 1.0)
    fwhm1 = np.random.uniform(15, 50)
    fwhm2 = np.random.uniform(15, 50)
    spectrum = (intensity1 * gaussian(target_wavelengths, pos1, fwhm1) + 
                intensity2 * gaussian(target_wavelengths, pos2, fwhm2))
    test_spectra.append(spectrum)
    test_peak_positions.append([pos1, pos2])
    
    measurement = np.zeros(20)
    for j in range(20):
        measurement[j] = np.sum(spectrum * response_20[:, j])
    test_measurements.append(measurement)

test_spectra = np.array(test_spectra)
test_measurements = np.array(test_measurements)
test_peak_positions = np.array(test_peak_positions)

print(f'  生成样本数: {n_samples}')

# 预测
X_scaled = scaler_X.transform(test_measurements)
X_tensor = torch.FloatTensor(X_scaled)

with torch.no_grad():
    y_pred_scaled = model(X_tensor)
    y_pred = scaler_y.inverse_transform(y_pred_scaled.numpy())

# 改进的峰检测函数
def find_peaks_v2(spectrum, wavelengths, min_distance=25):
    peaks, _ = find_peaks(spectrum, distance=min_distance, prominence=0.1)
    
    if len(peaks) == 0:
        peak_idx = np.argmax(spectrum)
        return [peak_idx], [wavelengths[peak_idx]]
    elif len(peaks) == 1:
        peak1_idx = peaks[0]
        left_part = spectrum[:max(0, peak1_idx - min_distance)]
        right_part = spectrum[min(len(spectrum), peak1_idx + min_distance):]
        
        candidates = []
        if len(left_part) > 0:
            left_max_idx = np.argmax(left_part)
            if left_part[left_max_idx] > 0.1:
                candidates.append((left_max_idx, left_part[left_max_idx]))
        
        if len(right_part) > 0:
            right_max_idx = np.argmax(right_part) + min(len(spectrum), peak1_idx + min_distance)
            if right_part[np.argmax(right_part)] > 0.1:
                candidates.append((right_max_idx, spectrum[right_max_idx]))
        
        if candidates:
            candidates.sort(key=lambda x: x[1], reverse=True)
            peak2_idx = candidates[0][0]
            if abs(peak2_idx - peak1_idx) >= min_distance:
                peaks = sorted([peak1_idx, peak2_idx])
        
        return peaks, [wavelengths[p] for p in peaks]
    else:
        peak_heights = spectrum[peaks]
        top2_indices = np.argsort(peak_heights)[-2:]
        selected_peaks = sorted(peaks[top2_indices])
        return selected_peaks, [wavelengths[p] for p in selected_peaks]

# 计算每个样本的误差
sample_results = []

for i in range(len(y_pred)):
    pred_peaks_idx, pred_peaks_wl = find_peaks_v2(y_pred[i], target_wavelengths, min_distance=25)
    true_peaks = test_peak_positions[i]
    
    if len(pred_peaks_idx) == 2:
        # 正确识别双峰
        errors_for_sample = []
        for pred_wl in pred_peaks_wl:
            distances = [abs(pred_wl - true_wl) for true_wl in true_peaks]
            min_distance = min(distances)
            errors_for_sample.append(min_distance)
        mean_error = np.mean(errors_for_sample)
        sample_results.append({
            'index': i,
            'correct_count': True,
            'mean_error': mean_error,
            'errors': errors_for_sample,
            'pred_peaks': pred_peaks_wl,
            'true_peaks': true_peaks
        })
    else:
        # 错误识别
        sample_results.append({
            'index': i,
            'correct_count': False,
            'mean_error': 50.0,
            'errors': [50.0, 50.0],
            'pred_peaks': pred_peaks_wl,
            'true_peaks': true_peaks
        })

# 分离正确和错误的样本
correct_samples = [r for r in sample_results if r['correct_count']]
incorrect_samples = [r for r in sample_results if not r['correct_count']]

# 按误差排序正确样本（从小到大）
correct_samples.sort(key=lambda x: x['mean_error'])

print(f'  正确识别样本: {len(correct_samples)}')
print(f'  错误识别样本: {len(incorrect_samples)}')

# 选择12个最好的和12个最差的
best_12 = correct_samples[:12] if len(correct_samples) >= 12 else correct_samples + incorrect_samples[:12-len(correct_samples)]
worst_12 = incorrect_samples[:12] if len(incorrect_samples) >= 12 else incorrect_samples + correct_samples[-(12-len(incorrect_samples)):]

print(f'  选择最好12个: {len(best_12)}')
print(f'  选择最差12个: {len(worst_12)}')

# ============================================================================
# 4. 生成可视化
# ============================================================================
print('\n[4/4] 生成可视化...')

# 最好的12个结果
fig, axes = plt.subplots(3, 4, figsize=(20, 15))
fig.suptitle('Best 12 Double Peak Predictions', fontsize=16, fontweight='bold')

for idx, result in enumerate(best_12[:12]):
    if idx >= 12:
        break
    row = idx // 4
    col = idx % 4
    ax = axes[row, col]
    
    i = result['index']
    ax.plot(target_wavelengths, test_spectra[i], 'b-', label='True Spectrum', alpha=0.8, linewidth=2)
    ax.plot(target_wavelengths, y_pred[i], 'r--', label='Predicted Spectrum', alpha=0.8, linewidth=2)
    
    # 标记真实峰
    true_peaks = result['true_peaks']
    for peak_wl in true_peaks:
        ax.axvline(peak_wl, color='blue', linestyle=':', alpha=0.9, linewidth=2)
    
    # 标记预测峰
    pred_peaks = result['pred_peaks']
    for peak_wl in pred_peaks:
        ax.axvline(peak_wl, color='red', linestyle=':', alpha=0.9, linewidth=2)
    
    mean_error = result['mean_error']
    ax.set_title(f'Sample {i}\nMean Error: {mean_error:.2f}nm', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Intensity')
    
    if idx == 0:
        ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig('double_peak_best_12.png', dpi=150, bbox_inches='tight')
print('已保存: double_peak_best_12.png')

# 最差的12个结果
fig, axes = plt.subplots(3, 4, figsize=(20, 15))
fig.suptitle('Worst 12 Double Peak Predictions', fontsize=16, fontweight='bold')

for idx, result in enumerate(worst_12[:12]):
    if idx >= 12:
        break
    row = idx // 4
    col = idx % 4
    ax = axes[row, col]
    
    i = result['index']
    ax.plot(target_wavelengths, test_spectra[i], 'b-', label='True Spectrum', alpha=0.8, linewidth=2)
    ax.plot(target_wavelengths, y_pred[i], 'r--', label='Predicted Spectrum', alpha=0.8, linewidth=2)
    
    # 标记真实峰
    true_peaks = result['true_peaks']
    for peak_wl in true_peaks:
        ax.axvline(peak_wl, color='blue', linestyle=':', alpha=0.9, linewidth=2)
    
    # 标记预测峰
    pred_peaks = result['pred_peaks']
    for peak_wl in pred_peaks:
        ax.axvline(peak_wl, color='red', linestyle=':', alpha=0.9, linewidth=2)
    
    if result['correct_count']:
        mean_error = result['mean_error']
        ax.set_title(f'Sample {i}\nMean Error: {mean_error:.2f}nm', fontsize=10)
    else:
        pred_count = len(pred_peaks)
        ax.set_title(f'Sample {i}\nINCORRECT ({pred_count} peaks)', fontsize=10, color='red')
    
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Intensity')
    
    if idx == 0:
        ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig('double_peak_worst_12.png', dpi=150, bbox_inches='tight')
print('已保存: double_peak_worst_12.png')

# 保存详细结果
best_results = []
for result in best_12[:12]:
    best_results.append({
        'index': int(result['index']),
        'mean_error': float(result['mean_error']),
        'true_peaks': [float(p) for p in result['true_peaks']],
        'pred_peaks': [float(p) for p in result['pred_peaks']]
    })

worst_results = []
for result in worst_12[:12]:
    worst_results.append({
        'index': int(result['index']),
        'correct_count': bool(result['correct_count']),
        'mean_error': float(result['mean_error']) if result['correct_count'] else None,
        'true_peaks': [float(p) for p in result['true_peaks']],
        'pred_peaks': [float(p) for p in result['pred_peaks']]
    })

results_summary = {
    'best_12': best_results,
    'worst_12': worst_results,
    'total_correct': len(correct_samples),
    'total_incorrect': len(incorrect_samples),
    'accuracy_rate': len(correct_samples) / len(sample_results) * 100
}

import json
with open('double_peak_quality_analysis.json', 'w') as f:
    json.dump(results_summary, f, indent=2)

print('已保存: double_peak_quality_analysis.json')

print('\n' + '=' * 70)
print('双峰预测质量分析完成！')
print('=' * 70)
print(f'最佳12个样本平均误差: {np.mean([r["mean_error"] for r in best_12[:12]]):.2f} nm')
if len(incorrect_samples) > 0:
    print(f'最差12个样本中错误识别: {len([r for r in worst_12[:12] if not r["correct_count"]])} 个')