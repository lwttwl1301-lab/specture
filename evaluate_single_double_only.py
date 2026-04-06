"""
只测试单峰和双峰场景的评估脚本
使用V2单峰模型 + 双峰专用模型
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.interpolate import UnivariateSpline
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

print('=' * 70)
print('单峰+双峰专用模型评估')
print('=' * 70)

# ============================================================================
# 1. 加载响应度
# ============================================================================
print('\n[1/5] 加载响应度...')

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
# 2. 定义模型架构
# ============================================================================
class MultiBiasNetV2(nn.Module):
    def __init__(self, input_dim, output_dim):
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

# ============================================================================
# 3. 加载预训练模型
# ============================================================================
print('\n[2/5] 加载预训练模型...')

# 加载V2单峰模型
checkpoint_single = torch.load('model_multi_bias_v2.pth', weights_only=False)
model_single = MultiBiasNetV2(20, 301)
model_single.load_state_dict(checkpoint_single['model_state_dict'])
model_single.eval()
scaler_X_single = checkpoint_single['scaler_X']
scaler_y_single = checkpoint_single['scaler_y']

# 加载双峰专用模型  
checkpoint_double = torch.load('model_double_peak.pth', weights_only=False)
model_double = DoublePeakNet(20, 301)
model_double.load_state_dict(checkpoint_double['model_state_dict'])
model_double.eval()
scaler_X_double = checkpoint_double['scaler_X']
scaler_y_double = checkpoint_double['scaler_y']

print('  V2单峰模型: 已加载 (0.99nm误差)')
print('  双峰专用模型: 已加载 (10.75nm中位数误差)')

# ============================================================================
# 4. 生成单峰+双峰测试数据
# ============================================================================
print('\n[3/5] 生成单峰+双峰测试数据...')

def gaussian(wavelengths, center, fwhm):
    sigma = fwhm / 2.355
    return np.exp(-0.5 * ((wavelengths - center) / sigma) ** 2)

np.random.seed(789)

# 单峰数据 (300样本)
n_single = 300
single_spectra = []
single_measurements = []
single_peak_positions = []

for i in range(n_single):
    pos = np.random.uniform(1000, 1300)
    intensity = np.random.uniform(0.3, 1.0)
    fwhm = np.random.uniform(15, 50)
    spectrum = intensity * gaussian(target_wavelengths, pos, fwhm)
    single_spectra.append(spectrum)
    single_peak_positions.append(pos)
    
    measurement = np.zeros(20)
    for j in range(20):
        measurement[j] = np.sum(spectrum * response_20[:, j])
    single_measurements.append(measurement)

# 双峰数据 (300样本)  
n_double = 300
double_spectra = []
double_measurements = []
double_peak_positions = []

for i in range(n_double):
    pos1 = np.random.uniform(1000, 1250)
    pos2 = np.random.uniform(pos1 + 30, min(1300, pos1 + 200))
    intensity1 = np.random.uniform(0.3, 1.0)
    intensity2 = np.random.uniform(0.3, 1.0)
    fwhm1 = np.random.uniform(15, 50)
    fwhm2 = np.random.uniform(15, 50)
    spectrum = (intensity1 * gaussian(target_wavelengths, pos1, fwhm1) + 
                intensity2 * gaussian(target_wavelengths, pos2, fwhm2))
    double_spectra.append(spectrum)
    double_peak_positions.append([pos1, pos2])
    
    measurement = np.zeros(20)
    for j in range(20):
        measurement[j] = np.sum(spectrum * response_20[:, j])
    double_measurements.append(measurement)

single_spectra = np.array(single_spectra)
single_measurements = np.array(single_measurements)
single_peak_positions = np.array(single_peak_positions)

double_spectra = np.array(double_spectra)
double_measurements = np.array(double_measurements)
double_peak_positions = np.array(double_peak_positions)

print(f'  单峰样本: {n_single} 条')
print(f'  双峰样本: {n_double} 条')

# ============================================================================
# 5. 预测和评估
# ============================================================================
print('\n[4/5] 预测和评估...')

# 单峰预测
X_single_scaled = scaler_X_single.transform(single_measurements)
X_single_tensor = torch.FloatTensor(X_single_scaled)

with torch.no_grad():
    y_pred_single_scaled = model_single(X_single_tensor)
    y_pred_single = scaler_y_single.inverse_transform(y_pred_single_scaled.numpy())

# 双峰预测
X_double_scaled = scaler_X_double.transform(double_measurements)
X_double_tensor = torch.FloatTensor(X_double_scaled)

with torch.no_grad():
    y_pred_double_scaled = model_double(X_double_tensor)
    y_pred_double = scaler_y_double.inverse_transform(y_pred_double_scaled.numpy())

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

# 单峰评估
single_errors = []
for i in range(len(y_pred_single)):
    pred_pos = target_wavelengths[np.argmax(y_pred_single[i])]
    true_pos = single_peak_positions[i]
    error = abs(pred_pos - true_pos)
    single_errors.append(error)

single_errors = np.array(single_errors)

# 双峰评估
double_errors = []
double_count_correct = 0

for i in range(len(y_pred_double)):
    pred_peaks_idx, pred_peaks_wl = find_peaks_v2(y_pred_double[i], target_wavelengths, min_distance=25)
    true_peaks = double_peak_positions[i]
    
    if len(pred_peaks_idx) == 2:
        double_count_correct += 1
        errors_for_sample = []
        for pred_wl in pred_peaks_wl:
            distances = [abs(pred_wl - true_wl) for true_wl in true_peaks]
            min_distance = min(distances)
            errors_for_sample.append(min_distance)
        double_errors.extend(errors_for_sample)
    else:
        double_errors.extend([50, 50])

double_errors = np.array(double_errors)
double_accuracy = double_count_correct / len(y_pred_double) * 100

print('\n【单峰评估结果】')
print(f'  样本数: {len(single_errors)}')
print(f'  位置误差均值: {np.mean(single_errors):.2f} nm')
print(f'  位置误差中位数: {np.median(single_errors):.2f} nm')
print(f'  <1nm精度: {np.sum(single_errors < 1) / len(single_errors) * 100:.1f}%')
print(f'  <5nm精度: {np.sum(single_errors < 5) / len(single_errors) * 100:.1f}%')

print('\n【双峰评估结果】')
print(f'  样本数: {len(y_pred_double)}')
print(f'  峰数量准确率: {double_accuracy:.1f}%')
print(f'  位置误差均值: {np.mean(double_errors):.2f} nm')
print(f'  位置误差中位数: {np.median(double_errors):.2f} nm')
print(f'  <1nm精度: {np.sum(double_errors < 1) / len(double_errors) * 100:.1f}%')
print(f'  <5nm精度: {np.sum(double_errors < 5) / len(double_errors) * 100:.1f}%')

# 整体评估
all_errors = np.concatenate([single_errors, double_errors])
print('\n【整体评估结果】')
print(f'  总样本数: {len(all_errors)}')
print(f'  位置误差均值: {np.mean(all_errors):.2f} nm')
print(f'  位置误差中位数: {np.median(all_errors):.2f} nm')
print(f'  <1nm精度: {np.sum(all_errors < 1) / len(all_errors) * 100:.1f}%')
print(f'  <5nm精度: {np.sum(all_errors < 5) / len(all_errors) * 100:.1f}%')

# ============================================================================
# 6. 可视化
# ============================================================================
print('\n[5/5] 生成可视化...')

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 单峰误差分布
ax = axes[0, 0]
ax.hist(single_errors, bins=50, edgecolor='black', alpha=0.7, color='blue')
ax.axvline(np.mean(single_errors), color='red', linestyle='--', label=f'Mean={np.mean(single_errors):.2f}nm')
ax.axvline(1, color='green', linestyle=':', label='1nm threshold')
ax.set_title('Single Peak Error Distribution')
ax.set_xlabel('Error (nm)')
ax.set_ylabel('Count')
ax.legend()
ax.grid(True, alpha=0.3)

# 双峰误差分布
ax = axes[0, 1]
ax.hist(double_errors, bins=50, edgecolor='black', alpha=0.7, color='orange')
ax.axvline(np.mean(double_errors), color='red', linestyle='--', label=f'Mean={np.mean(double_errors):.2f}nm')
ax.axvline(1, color='green', linestyle=':', label='1nm threshold')
ax.set_title('Double Peak Error Distribution')
ax.set_xlabel('Error (nm)')
ax.set_ylabel('Count')
ax.legend()
ax.grid(True, alpha=0.3)

# 双峰数量准确率
ax = axes[0, 2]
ax.bar(['Correct (2 peaks)', 'Incorrect'], [double_count_correct, len(y_pred_double) - double_count_correct], 
       color=['green', 'red'], alpha=0.7)
ax.set_title(f'Double Peak Count Accuracy: {double_accuracy:.1f}%')
ax.set_ylabel('Count')
ax.grid(True, alpha=0.3)

# 单峰样本对比
single_sample_idx = 0
ax = axes[1, 0]
ax.plot(target_wavelengths, single_spectra[single_sample_idx], 'b-', label='True Spectrum', alpha=0.8, linewidth=2)
ax.plot(target_wavelengths, y_pred_single[single_sample_idx], 'r--', label='Predicted Spectrum', alpha=0.8, linewidth=2)
true_pos = single_peak_positions[single_sample_idx]
pred_pos = target_wavelengths[np.argmax(y_pred_single[single_sample_idx])]
ax.axvline(true_pos, color='blue', linestyle=':', alpha=0.9, linewidth=2, label=f'True={true_pos:.1f}nm')
ax.axvline(pred_pos, color='red', linestyle=':', alpha=0.9, linewidth=2, label=f'Pred={pred_pos:.1f}nm')
error = abs(pred_pos - true_pos)
ax.set_title(f'Single Peak Sample\nError: {error:.2f}nm')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
ax.set_xlabel('Wavelength (nm)')
ax.set_ylabel('Intensity')

# 双峰样本对比 (正确)
correct_double_idx = None
for i in range(len(y_pred_double)):
    pred_peaks_idx, _ = find_peaks_v2(y_pred_double[i], target_wavelengths, min_distance=25)
    if len(pred_peaks_idx) == 2:
        correct_double_idx = i
        break

if correct_double_idx is not None:
    ax = axes[1, 1]
    ax.plot(target_wavelengths, double_spectra[correct_double_idx], 'b-', label='True Spectrum', alpha=0.8, linewidth=2)
    ax.plot(target_wavelengths, y_pred_double[correct_double_idx], 'r--', label='Predicted Spectrum', alpha=0.8, linewidth=2)
    
    # 标记真实峰
    true_peaks = double_peak_positions[correct_double_idx]
    for peak_wl in true_peaks:
        ax.axvline(peak_wl, color='blue', linestyle=':', alpha=0.9, linewidth=2)
    
    # 标记预测峰
    _, pred_peaks_wl = find_peaks_v2(y_pred_double[correct_double_idx], target_wavelengths, min_distance=25)
    for peak_wl in pred_peaks_wl:
        ax.axvline(peak_wl, color='red', linestyle=':', alpha=0.9, linewidth=2)
    
    # 计算误差
    sample_errors = []
    for pred_wl in pred_peaks_wl:
        distances = [abs(pred_wl - true_wl) for true_wl in true_peaks]
        sample_errors.append(min(distances))
    mean_error = np.mean(sample_errors)
    
    ax.set_title(f'Double Peak Sample (Correct)\nMean Error: {mean_error:.2f}nm')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Intensity')

# 双峰样本对比 (错误)
incorrect_double_idx = None
for i in range(len(y_pred_double)):
    pred_peaks_idx, _ = find_peaks_v2(y_pred_double[i], target_wavelengths, min_distance=25)
    if len(pred_peaks_idx) != 2:
        incorrect_double_idx = i
        break

if incorrect_double_idx is not None:
    ax = axes[1, 2]
    ax.plot(target_wavelengths, double_spectra[incorrect_double_idx], 'b-', label='True Spectrum', alpha=0.8, linewidth=2)
    ax.plot(target_wavelengths, y_pred_double[incorrect_double_idx], 'r--', label='Predicted Spectrum', alpha=0.8, linewidth=2)
    
    # 标记真实峰
    true_peaks = double_peak_positions[incorrect_double_idx]
    for peak_wl in true_peaks:
        ax.axvline(peak_wl, color='blue', linestyle=':', alpha=0.9, linewidth=2)
    
    # 标记预测峰
    _, pred_peaks_wl = find_peaks_v2(y_pred_double[incorrect_double_idx], target_wavelengths, min_distance=25)
    for peak_wl in pred_peaks_wl:
        ax.axvline(peak_wl, color='red', linestyle=':', alpha=0.9, linewidth=2)
    
    ax.set_title(f'Double Peak Sample (INCORRECT)\nPred peaks: {len(pred_peaks_wl)}')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Intensity')

plt.tight_layout()
plt.savefig('single_double_only_evaluation.png', dpi=150, bbox_inches='tight')
print('已保存: single_double_only_evaluation.png')

# 保存结果
results_dict = {
    'single_peak': {
        'sample_count': int(len(single_errors)),
        'mean_error': float(np.mean(single_errors)),
        'median_error': float(np.median(single_errors)),
        'precision_1nm': float(np.sum(single_errors < 1) / len(single_errors) * 100),
        'precision_5nm': float(np.sum(single_errors < 5) / len(single_errors) * 100)
    },
    'double_peak': {
        'sample_count': int(len(y_pred_double)),
        'count_accuracy': float(double_accuracy),
        'mean_error': float(np.mean(double_errors)),
        'median_error': float(np.median(double_errors)),
        'precision_1nm': float(np.sum(double_errors < 1) / len(double_errors) * 100),
        'precision_5nm': float(np.sum(double_errors < 5) / len(double_errors) * 100)
    },
    'overall': {
        'total_samples': int(len(all_errors)),
        'mean_error': float(np.mean(all_errors)),
        'median_error': float(np.median(all_errors)),
        'precision_1nm': float(np.sum(all_errors < 1) / len(all_errors) * 100),
        'precision_5nm': float(np.sum(all_errors < 5) / len(all_errors) * 100)
    }
}

import json
with open('single_double_only_results.json', 'w') as f:
    json.dump(results_dict, f, indent=2)

print('已保存: single_double_only_results.json')

print('\n' + '=' * 70)
print('单峰+双峰专用模型评估完成！')
print('=' * 70)