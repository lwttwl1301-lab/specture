"""
评估双峰专用模型
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
print('双峰模型评估')
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

# 生成测试数据
print('\n生成双峰测试数据...')

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

np.random.seed(123)

# 测试样本
n_test = 200
peak_config = checkpoint['peak_config']

test_spectra = []
test_measurements = []
test_peak_positions = []

# 均匀分布测试样本
for i in range(n_test):
    spectrum, peaks = generate_double_peak_spectrum(target_wavelengths, peak_config, 'main')
    test_spectra.append(spectrum)
    test_peak_positions.append(sorted(peaks))
    
    # 计算测量值
    measurement = np.zeros(20)
    response_20 = checkpoint.get('response_20', None)
    if response_20 is None:
        # 重新加载响应度
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
    
    for j in range(20):
        measurement[j] = np.sum(spectrum * response_20[:, j])
    test_measurements.append(measurement)

test_spectra = np.array(test_spectra)
test_measurements = np.array(test_measurements)
test_peak_positions = np.array(test_peak_positions)

print(f'测试样本数: {len(test_spectra)}')

# 预测
print('\n进行预测...')
X_scaled = scaler_X.transform(test_measurements)
X_tensor = torch.FloatTensor(X_scaled)

with torch.no_grad():
    y_pred_scaled = model(X_tensor)
    y_pred = scaler_y.inverse_transform(y_pred_scaled.numpy())

# 评估
position_errors = []
count_correct = 0

for i in range(len(y_pred)):
    pred_peaks_idx, pred_peaks_wl = find_peaks_v2(y_pred[i], target_wavelengths, min_distance=25)
    true_peaks = test_peak_positions[i]
    
    if len(pred_peaks_idx) == 2:
        count_correct += 1
        errors_for_sample = []
        for pred_wl in pred_peaks_wl:
            distances = [abs(pred_wl - true_wl) for true_wl in true_peaks]
            min_distance = min(distances)
            errors_for_sample.append(min_distance)
        position_errors.extend(errors_for_sample)
    else:
        position_errors.extend([50, 50])

position_errors = np.array(position_errors)
accuracy_rate = count_correct / len(y_pred) * 100

print('\n【双峰定位结果】')
print(f'  峰数量准确率: {accuracy_rate:.1f}%')
print(f'  峰位置误差均值: {np.mean(position_errors):.2f} nm')
print(f'  峰位置误差中位数: {np.median(position_errors):.2f} nm')
print(f'  <1nm 精度: {np.sum(position_errors < 1) / len(position_errors) * 100:.1f}%')
print(f'  <2nm 精度: {np.sum(position_errors < 2) / len(position_errors) * 100:.1f}%')
print(f'  <5nm 精度: {np.sum(position_errors < 5) / len(position_errors) * 100:.1f}%')

# 可视化
print('\n生成可视化...')
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 峰位置误差分布
ax = axes[0]
ax.hist(position_errors, bins=50, edgecolor='black', alpha=0.7)
ax.axvline(np.mean(position_errors), color='red', linestyle='--', label=f'Mean={np.mean(position_errors):.2f}nm')
ax.axvline(1, color='green', linestyle=':', label='1nm threshold')
ax.set_title('Peak Position Error Distribution')
ax.set_xlabel('Error (nm)')
ax.set_ylabel('Count')
ax.legend()
ax.grid(True, alpha=0.3)

# 峰数量准确率
ax = axes[1]
ax.bar(['Correct (2 peaks)', 'Incorrect'], [count_correct, len(y_pred) - count_correct], 
       color=['green', 'red'], alpha=0.7)
ax.set_title(f'Peak Count Accuracy: {accuracy_rate:.1f}%')
ax.set_ylabel('Count')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('evaluate_double_peak_results.png', dpi=150)
print('已保存: evaluate_double_peak_results.png')

print('\n' + '=' * 70)
print('评估完成！')
print('=' * 70)