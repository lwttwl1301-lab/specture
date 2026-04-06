"""
详细的双峰模型评估 - 生成更多可视化图表
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
print('详细双峰模型评估')
print('=' * 70)

# 加载模型
checkpoint = torch.load('model_double_peak.pth', weights_only=False)
print(f"模型版本: {checkpoint.get('version', 'N/A')}")

# 定义模型
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

model = DoublePeakNet(20, 301)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

scaler_X = checkpoint['scaler_X']
scaler_y = checkpoint['scaler_y']
target_wavelengths = checkpoint['wavelengths']

print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")

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

# 生成详细的测试数据
print('\n生成详细双峰测试数据...')

def gaussian(wavelengths, center, fwhm):
    sigma = fwhm / 2.355
    return np.exp(-0.5 * ((wavelengths - center) / sigma) ** 2)

def generate_double_peak_spectrum(wavelengths, config, region='main'):
    wl_min, wl_max = wavelengths[0], wavelengths[-1]
    
    if region == 'low':
        pos1 = np.random.uniform(1000, 1030)
        pos2 = np.random.uniform(pos1 + config['min_separation'], min(1050, pos1 + config['max_separation']))
    elif region == 'high':
        pos2 = np.random.uniform(1270, 1300)
        pos1 = np.random.uniform(max(1250, pos2 - config['max_separation']), pos2 - config['min_separation'])
    else:
        pos1 = np.random.uniform(wl_min + 20, wl_max - 80)
        pos2 = np.random.uniform(pos1 + config['min_separation'], min(wl_max - 20, pos1 + config['max_separation']))
    
    intensity1 = np.random.uniform(*config['intensity_range'])
    intensity2 = np.random.uniform(*config['intensity_range'])
    fwhm1 = np.random.uniform(*config['fwhm_range'])
    fwhm2 = np.random.uniform(*config['fwhm_range'])
    
    spectrum = (intensity1 * gaussian(wavelengths, pos1, fwhm1) + 
                intensity2 * gaussian(wavelengths, pos2, fwhm2))
    
    return spectrum, [pos1, pos2]

np.random.seed(456)

# 大量测试样本用于详细分析
n_test = 500
peak_config = checkpoint['peak_config']

test_spectra = []
test_measurements = []
test_peak_positions = []
test_regions = []

# 按区域生成样本
n_per_region = n_test // 3

# 主流区域
for i in range(n_per_region):
    spectrum, peaks = generate_double_peak_spectrum(target_wavelengths, peak_config, 'main')
    test_spectra.append(spectrum)
    test_peak_positions.append(sorted(peaks))
    test_regions.append('main')

# 低边界区域
for i in range(n_per_region):
    spectrum, peaks = generate_double_peak_spectrum(target_wavelengths, peak_config, 'low')
    test_spectra.append(spectrum)
    test_peak_positions.append(sorted(peaks))
    test_regions.append('low')

# 高边界区域  
for i in range(n_test - 2 * n_per_region):
    spectrum, peaks = generate_double_peak_spectrum(target_wavelengths, peak_config, 'high')
    test_spectra.append(spectrum)
    test_peak_positions.append(sorted(peaks))
    test_regions.append('high')

test_spectra = np.array(test_spectra)
test_peak_positions = np.array(test_peak_positions)
test_regions = np.array(test_regions)

# 计算测量值
print('计算测量值...')
excel_path = r'D:\desktop\try\响应度矩阵.xlsx'
df = pd.read_excel(excel_path)
wavelengths_resp = df.iloc[:, 0].values
response_matrix_full = df.iloc[:, 1:].values

all_biases = np.linspace(-15, 0, 65)
indices = np.linspace(0, 64, 20, dtype=int)
selected_biases = all_biases[indices]

response_20 = np.zeros((len(target_wavelengths), 20))
for j, idx in enumerate(indices):
    orig_resp = response_matrix_full[:, idx]
    spline = UnivariateSpline(wavelengths_resp, orig_resp, s=0.001, k=3)
    interpolated = spline(target_wavelengths)
    smoothed = gaussian_filter1d(interpolated, sigma=1.5)
    response_20[:, j] = smoothed

test_measurements = np.zeros((n_test, 20))
for i in range(n_test):
    for j in range(20):
        test_measurements[i, j] = np.sum(test_spectra[i] * response_20[:, j])

print(f'测试样本数: {len(test_spectra)}')

# 预测
print('\n进行预测...')
X_scaled = scaler_X.transform(test_measurements)
X_tensor = torch.FloatTensor(X_scaled)

with torch.no_grad():
    y_pred_scaled = model(X_tensor)
    y_pred = scaler_y.inverse_transform(y_pred_scaled.numpy())

# 详细评估
print('\n进行详细评估...')

position_errors = []
count_correct = 0
region_errors = {'main': [], 'low': [], 'high': []}
region_counts = {'main': 0, 'low': 0, 'high': 0}

all_pred_peaks = []
all_true_peaks = []

for i in range(len(y_pred)):
    pred_peaks_idx, pred_peaks_wl = find_peaks_v2(y_pred[i], target_wavelengths, min_distance=25)
    true_peaks = test_peak_positions[i]
    region = test_regions[i]
    
    all_pred_peaks.append(pred_peaks_wl)
    all_true_peaks.append(true_peaks)
    
    if len(pred_peaks_idx) == 2:
        count_correct += 1
        region_counts[region] += 1
        
        # 计算位置误差（匹配最近的真实峰）
        errors_for_sample = []
        for pred_wl in pred_peaks_wl:
            distances = [abs(pred_wl - true_wl) for true_wl in true_peaks]
            min_distance = min(distances)
            errors_for_sample.append(min_distance)
            region_errors[region].append(min_distance)
        
        position_errors.extend(errors_for_sample)
    else:
        position_errors.extend([50, 50])
        # 对于错误的样本，也记录到区域统计中
        region_errors[region].extend([50, 50])

position_errors = np.array(position_errors)
accuracy_rate = count_correct / len(y_pred) * 100

print('\n【整体双峰定位结果】')
print(f'  峰数量准确率: {accuracy_rate:.1f}%')
print(f'  峰位置误差均值: {np.mean(position_errors):.2f} nm')
print(f'  峰位置误差中位数: {np.median(position_errors):.2f} nm')
print(f'  <1nm 精度: {np.sum(position_errors < 1) / len(position_errors) * 100:.1f}%')
print(f'  <2nm 精度: {np.sum(position_errors < 2) / len(position_errors) * 100:.1f}%')
print(f'  <5nm 精度: {np.sum(position_errors < 5) / len(position_errors) * 100:.1f}%')

# 按区域统计
print('\n【按区域分组结果】')
for region in ['main', 'low', 'high']:
    errors = np.array(region_errors[region])
    total_samples = len(errors) // 2
    correct_samples = region_counts[region]
    accuracy = correct_samples / total_samples * 100 if total_samples > 0 else 0
    
    print(f'\n{region.upper()} 区域 ({total_samples} 样本):')
    print(f'  峰数量准确率: {accuracy:.1f}%')
    if len(errors[errors < 50]) > 0:
        valid_errors = errors[errors < 50]
        print(f'  位置误差均值: {np.mean(valid_errors):.2f} nm')
        print(f'  <5nm 精度: {np.sum(valid_errors < 5) / len(valid_errors) * 100:.1f}%')
    else:
        print(f'  位置误差均值: N/A (无正确预测)')
        print(f'  <5nm 精度: 0.0%')

# 生成详细可视化
print('\n生成详细可视化...')

fig = plt.figure(figsize=(20, 15))

# 1. 整体误差分布
ax1 = plt.subplot(3, 4, 1)
ax1.hist(position_errors, bins=50, edgecolor='black', alpha=0.7)
ax1.axvline(np.mean(position_errors), color='red', linestyle='--', label=f'Mean={np.mean(position_errors):.2f}nm')
ax1.axvline(1, color='green', linestyle=':', label='1nm threshold')
ax1.set_title('Overall Peak Position Error Distribution')
ax1.set_xlabel('Error (nm)')
ax1.set_ylabel('Count')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. 峰数量准确率
ax2 = plt.subplot(3, 4, 2)
ax2.bar(['Correct (2 peaks)', 'Incorrect'], [count_correct, len(y_pred) - count_correct], 
       color=['green', 'red'], alpha=0.7)
ax2.set_title(f'Peak Count Accuracy: {accuracy_rate:.1f}%')
ax2.set_ylabel('Count')
ax2.grid(True, alpha=0.3)

# 3. 按区域的准确率
ax3 = plt.subplot(3, 4, 3)
regions = ['Main', 'Low', 'High']
accuracies = []
for i, region in enumerate(['main', 'low', 'high']):
    total = len(region_errors[region]) // 2
    correct = region_counts[region]
    acc = correct / total * 100 if total > 0 else 0
    accuracies.append(acc)

ax3.bar(regions, accuracies, color=['blue', 'orange', 'green'], alpha=0.7)
ax3.set_title('Accuracy by Region')
ax3.set_ylabel('Accuracy (%)')
ax3.set_ylim(0, 100)
ax3.grid(True, alpha=0.3)

# 4. 按区域的误差分布
ax4 = plt.subplot(3, 4, 4)
colors = ['blue', 'orange', 'green']
for i, region in enumerate(['main', 'low', 'high']):
    errors = np.array(region_errors[region])
    valid_errors = errors[errors < 50]
    if len(valid_errors) > 0:
        ax4.hist(valid_errors, bins=30, alpha=0.5, color=colors[i], label=region.upper())
ax4.set_title('Error Distribution by Region')
ax4.set_xlabel('Error (nm)')
ax4.set_ylabel('Count')
ax4.legend()
ax4.grid(True, alpha=0.3)

# 5-7. 样本对比图 (正确的预测)
correct_indices = []
for i in range(len(y_pred)):
    pred_peaks_idx, _ = find_peaks_v2(y_pred[i], target_wavelengths, min_distance=25)
    if len(pred_peaks_idx) == 2:
        correct_indices.append(i)
        if len(correct_indices) >= 3:
            break

for idx, sample_i in enumerate(correct_indices[:3]):
    ax = plt.subplot(3, 4, 5 + idx)
    ax.plot(target_wavelengths, test_spectra[sample_i], 'b-', label='True Spectrum', alpha=0.8, linewidth=2)
    ax.plot(target_wavelengths, y_pred[sample_i], 'r--', label='Predicted Spectrum', alpha=0.8, linewidth=2)
    
    # 标记真实峰
    true_peaks = test_peak_positions[sample_i]
    for peak_wl in true_peaks:
        ax.axvline(peak_wl, color='blue', linestyle=':', alpha=0.9, linewidth=2)
    
    # 标记预测峰
    pred_peaks_wl = all_pred_peaks[sample_i]
    for peak_wl in pred_peaks_wl:
        ax.axvline(peak_wl, color='red', linestyle=':', alpha=0.9, linewidth=2)
    
    # 计算该样本的误差
    sample_errors = []
    for pred_wl in pred_peaks_wl:
        distances = [abs(pred_wl - true_wl) for true_wl in true_peaks]
        sample_errors.append(min(distances))
    mean_error = np.mean(sample_errors)
    
    ax.set_title(f'Sample {sample_i}\nRegion: {test_regions[sample_i].upper()}, Error: {mean_error:.2f}nm')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Intensity')

# 8-10. 样本对比图 (错误的预测)
incorrect_indices = []
for i in range(len(y_pred)):
    pred_peaks_idx, _ = find_peaks_v2(y_pred[i], target_wavelengths, min_distance=25)
    if len(pred_peaks_idx) != 2:
        incorrect_indices.append(i)
        if len(incorrect_indices) >= 3:
            break

for idx, sample_i in enumerate(incorrect_indices[:3]):
    ax = plt.subplot(3, 4, 8 + idx)
    ax.plot(target_wavelengths, test_spectra[sample_i], 'b-', label='True Spectrum', alpha=0.8, linewidth=2)
    ax.plot(target_wavelengths, y_pred[sample_i], 'r--', label='Predicted Spectrum', alpha=0.8, linewidth=2)
    
    # 标记真实峰
    true_peaks = test_peak_positions[sample_i]
    for peak_wl in true_peaks:
        ax.axvline(peak_wl, color='blue', linestyle=':', alpha=0.9, linewidth=2)
    
    # 标记预测峰
    pred_peaks_wl = all_pred_peaks[sample_i]
    for peak_wl in pred_peaks_wl:
        ax.axvline(peak_wl, color='red', linestyle=':', alpha=0.9, linewidth=2)
    
    ax.set_title(f'Sample {sample_i} (INCORRECT)\nRegion: {test_regions[sample_i].upper()}, Pred peaks: {len(pred_peaks_wl)}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Intensity')

# 11. 误差 vs 峰间距
peak_separations = []
separation_errors = []

for i in range(len(y_pred)):
    true_peaks = test_peak_positions[i]
    separation = abs(true_peaks[1] - true_peaks[0])
    peak_separations.append(separation)
    
    pred_peaks_idx, pred_peaks_wl = find_peaks_v2(y_pred[i], target_wavelengths, min_distance=25)
    if len(pred_peaks_idx) == 2:
        errors_for_sample = []
        for pred_wl in pred_peaks_wl:
            distances = [abs(pred_wl - true_wl) for true_wl in true_peaks]
            errors_for_sample.append(min(distances))
        separation_errors.append((separation, np.mean(errors_for_sample)))
    else:
        separation_errors.append((separation, 50))

separations = [x[0] for x in separation_errors]
errors_by_sep = [x[1] for x in separation_errors]

ax11 = plt.subplot(3, 4, 11)
ax11.scatter(separations, errors_by_sep, alpha=0.6, s=20)
ax11.set_title('Error vs Peak Separation')
ax11.set_xlabel('Peak Separation (nm)')
ax11.set_ylabel('Mean Error (nm)')
ax11.grid(True, alpha=0.3)
ax11.set_ylim(0, 30)

# 12. 误差累积分布
ax12 = plt.subplot(3, 4, 12)
sorted_errors = np.sort(position_errors[position_errors < 50])
cumulative = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
ax12.plot(sorted_errors, cumulative * 100, 'b-', linewidth=2)
ax12.axvline(1, color='red', linestyle='--', label='<1nm: {:.1f}%'.format(np.sum(position_errors < 1) / len(position_errors) * 100))
ax12.axvline(2, color='orange', linestyle='--', label='<2nm: {:.1f}%'.format(np.sum(position_errors < 2) / len(position_errors) * 100))
ax12.axvline(5, color='green', linestyle='--', label='<5nm: {:.1f}%'.format(np.sum(position_errors < 5) / len(position_errors) * 100))
ax12.set_title('Cumulative Error Distribution')
ax12.set_xlabel('Error (nm)')
ax12.set_ylabel('Cumulative Percentage (%)')
ax12.legend()
ax12.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('detailed_double_peak_evaluation.png', dpi=200, bbox_inches='tight')
print('已保存: detailed_double_peak_evaluation.png')

# 保存详细结果
results_dict = {
    'overall_accuracy': float(accuracy_rate),
    'mean_error': float(np.mean(position_errors)),
    'median_error': float(np.median(position_errors)),
    'precision_1nm': float(np.sum(position_errors < 1) / len(position_errors) * 100),
    'precision_2nm': float(np.sum(position_errors < 2) / len(position_errors) * 100),
    'precision_5nm': float(np.sum(position_errors < 5) / len(position_errors) * 100),
    'region_results': {}
}

for region in ['main', 'low', 'high']:
    errors = np.array(region_errors[region])
    total = len(errors) // 2
    correct = region_counts[region]
    acc = correct / total * 100 if total > 0 else 0
    valid_errors = errors[errors < 50]
    if len(valid_errors) > 0:
        mean_err = float(np.mean(valid_errors))
        prec_5nm = float(np.sum(valid_errors < 5) / len(valid_errors) * 100)
    else:
        mean_err = None
        prec_5nm = 0.0
    
    results_dict['region_results'][region] = {
        'accuracy': float(acc),
        'mean_error': mean_err,
        'precision_5nm': prec_5nm,
        'total_samples': int(total)
    }

import json
with open('detailed_double_peak_results.json', 'w') as f:
    json.dump(results_dict, f, indent=2)

print('已保存: detailed_double_peak_results.json')

print('\n' + '=' * 70)
print('详细评估完成！')
print('=' * 70)