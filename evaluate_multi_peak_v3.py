"""
评估方案三：端到端峰预测模型
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from scipy.interpolate import UnivariateSpline
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

print('=' * 70)
print('方案三评估 - 端到端峰预测')
print('=' * 70)

# 加载模型
checkpoint = torch.load('model_multi_peak_v3.pth', map_location='cpu', weights_only=False)
print(f"\n模型版本: {checkpoint.get('version', 'N/A')}")
print(f"最大峰数量: {checkpoint.get('max_peaks', 'N/A')}")

# 加载响应度（用于生成测试数据）
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

# 定义模型架构（与训练时相同）
class PeakPredictorNet(nn.Module):
    def __init__(self, input_dim, max_peaks):
        super().__init__()
        self.max_peaks = max_peaks
        
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.2),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
        )
        
        self.peak_count_head = nn.Sequential(
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        self.peak_params_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, max_peaks * 3),
        )
        
        self.peak_exist_head = nn.Sequential(
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Linear(64, max_peaks),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        features = self.shared(x)
        peak_count = self.peak_count_head(features) * self.max_peaks
        peak_exist = self.peak_exist_head(features)
        peak_params = self.peak_params_head(features)
        peak_params = peak_params.view(-1, self.max_peaks, 3)
        peak_params[:, :, 0] = torch.sigmoid(peak_params[:, :, 0])
        peak_params[:, :, 1] = torch.sigmoid(peak_params[:, :, 1])
        peak_params[:, :, 2] = torch.sigmoid(peak_params[:, :, 2])
        return peak_count, peak_exist, peak_params

# 加载模型
model = PeakPredictorNet(20, checkpoint['max_peaks'])
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

scaler_X = checkpoint['scaler_X']
wl_min, wl_max = checkpoint['wl_range']
intensity_min, intensity_max = checkpoint['intensity_range']
fwhm_min, fwhm_max = checkpoint['fwhm_range']
MAX_PEAKS = checkpoint['max_peaks']

print(f'  模型参数量: {sum(p.numel() for p in model.parameters()):,}')

# 生成测试数据
print('\n[2/4] 生成测试数据...')

def gaussian(wavelengths, center, fwhm):
    sigma = fwhm / 2.355
    return np.exp(-0.5 * ((wavelengths - center) / sigma) ** 2)

def enforce_min_separation(peaks, min_sep=25):
    if len(peaks) <= 1:
        return peaks
    peaks = sorted(peaks, key=lambda x: x[0])
    for i in range(len(peaks) - 1):
        p1, p2 = peaks[i], peaks[i + 1]
        required_sep = max(min_sep, p1[2] * 0.8, p2[2] * 0.8)
        current_sep = p2[0] - p1[0]
        if current_sep < required_sep:
            new_pos = p1[0] + required_sep
            peaks[i + 1] = (new_pos, p2[1], p2[2])
    return peaks

def generate_multi_peak_spectrum(wavelengths, n_peaks, config):
    wl_min, wl_max = wavelengths[0], wavelengths[-1]
    margin = 30
    peaks = []
    for _ in range(n_peaks):
        pos = np.random.uniform(wl_min + margin, wl_max - margin)
        intensity = np.random.uniform(*config['intensity_range'])
        fwhm = np.random.uniform(*config['fwhm_range'])
        peaks.append((pos, intensity, fwhm))
    peaks = enforce_min_separation(peaks, min_sep=config['min_separation'])
    spectrum = np.zeros_like(wavelengths, dtype=float)
    for pos, intensity, fwhm in peaks:
        spectrum += intensity * gaussian(wavelengths, pos, fwhm)
    return spectrum, peaks

np.random.seed(123)  # 不同的种子，确保是新的测试数据

n_test = 500
peak_config = checkpoint['peak_config']

test_spectra = []
test_measurements = []
test_peak_params = []
test_peak_counts = []

for n_peaks in range(1, 6):
    n_samples = n_test // 5
    for _ in range(n_samples):
        spectrum, peaks = generate_multi_peak_spectrum(target_wavelengths, n_peaks, peak_config)
        test_spectra.append(spectrum)
        test_peak_params.append(peaks)
        test_peak_counts.append(n_peaks)
        
        # 计算测量值
        measurement = np.zeros(20)
        for j in range(20):
            measurement[j] = np.sum(spectrum * response_20[:, j])
        test_measurements.append(measurement)

test_spectra = np.array(test_spectra)
test_measurements = np.array(test_measurements)
test_peak_counts = np.array(test_peak_counts)

print(f'  测试样本数: {len(test_spectra)}')

# 预测
print('\n[3/4] 进行预测...')
X_scaled = scaler_X.transform(test_measurements)
X_tensor = torch.FloatTensor(X_scaled)

all_position_errors = []
all_count_errors = []
results_by_peak_count = {i: {'errors': [], 'count_errors': []} for i in range(1, 6)}

with torch.no_grad():
    pred_count, pred_exist, pred_params = model(X_tensor)
    
    for i in range(len(test_spectra)):
        n_true_peaks = test_peak_counts[i]
        n_pred_peaks = int(pred_count[i].item())
        count_error = abs(n_pred_peaks - n_true_peaks)
        all_count_errors.append(count_error)
        results_by_peak_count[n_true_peaks]['count_errors'].append(count_error)
        
        # 真实峰
        true_peaks = test_peak_params[i]
        
        # 预测峰
        pred_peaks = []
        for j in range(MAX_PEAKS):
            if pred_exist[i, j].item() > 0.5:
                pos = pred_params[i, j, 0].item() * (wl_max - wl_min) + wl_min
                intensity = pred_params[i, j, 1].item() * (intensity_max - intensity_min) + intensity_min
                fwhm = pred_params[i, j, 2].item() * (fwhm_max - fwhm_min) + fwhm_min
                pred_peaks.append((pos, intensity, fwhm))
        
        # 匈牙利匹配
        if len(true_peaks) > 0 and len(pred_peaks) > 0:
            cost_matrix = np.zeros((len(pred_peaks), len(true_peaks)))
            for pi, pp in enumerate(pred_peaks):
                for ti, tp in enumerate(true_peaks):
                    cost_matrix[pi, ti] = abs(pp[0] - tp[0])
            
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            for pi, ti in zip(row_ind, col_ind):
                if cost_matrix[pi, ti] < 50:
                    error = cost_matrix[pi, ti]
                    all_position_errors.append(error)
                    results_by_peak_count[n_true_peaks]['errors'].append(error)

all_position_errors = np.array(all_position_errors)
all_count_errors = np.array(all_count_errors)

# 输出结果
print('\n' + '=' * 70)
print('评估结果')
print('=' * 70)

print('\n【整体指标】')
print(f'  峰数量误差均值: {np.mean(all_count_errors):.2f}')
print(f'  峰数量准确率: {np.sum(all_count_errors == 0) / len(all_count_errors) * 100:.1f}%')
print(f'  峰位置误差均值: {np.mean(all_position_errors):.2f} nm')
print(f'  峰位置误差中位数: {np.median(all_position_errors):.2f} nm')
print(f'  <1nm 精度: {np.sum(all_position_errors < 1) / len(all_position_errors) * 100:.1f}%')
print(f'  <2nm 精度: {np.sum(all_position_errors < 2) / len(all_position_errors) * 100:.1f}%')
print(f'  <5nm 精度: {np.sum(all_position_errors < 5) / len(all_position_errors) * 100:.1f}%')

print('\n【按峰数分组】')
print(f'{"峰数":<8} {"样本数":<10} {"数量准确率":<12} {"位置误差(nm)":<15} {"<1nm%":<10}')
print('-' * 60)
for n in range(1, 6):
    count_acc = np.mean(np.array(results_by_peak_count[n]['count_errors']) == 0) * 100
    errors = results_by_peak_count[n]['errors']
    if len(errors) > 0:
        mean_err = np.mean(errors)
        p1 = np.sum(np.array(errors) < 1) / len(errors) * 100
        print(f'{n:<8} {len(results_by_peak_count[n]["count_errors"]):<10} {count_acc:<12.1f} {mean_err:<15.2f} {p1:<10.1f}')
    else:
        print(f'{n:<8} {len(results_by_peak_count[n]["count_errors"]):<10} {count_acc:<12.1f} {"N/A":<15} {"N/A":<10}')

# 可视化
print('\n[4/4] 生成可视化...')
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 峰位置误差分布
ax = axes[0, 0]
ax.hist(all_position_errors, bins=50, edgecolor='black', alpha=0.7)
ax.axvline(np.mean(all_position_errors), color='red', linestyle='--', label=f'Mean={np.mean(all_position_errors):.2f}nm')
ax.axvline(1, color='green', linestyle=':', label='1nm threshold')
ax.set_title('Peak Position Error Distribution (V3.3)')
ax.set_xlabel('Error (nm)')
ax.set_ylabel('Count')
ax.legend()
ax.grid(True, alpha=0.3)

# 峰数量误差分布
ax = axes[0, 1]
ax.hist(all_count_errors, bins=range(6), edgecolor='black', alpha=0.7)
ax.set_title('Peak Count Error Distribution')
ax.set_xlabel('Count Error')
ax.set_ylabel('Count')
ax.grid(True, alpha=0.3)

# 按峰数分组的误差
ax = axes[1, 0]
means = []
stds = []
labels = []
for n in range(1, 6):
    errors = results_by_peak_count[n]['errors']
    if len(errors) > 0:
        means.append(np.mean(errors))
        stds.append(np.std(errors))
        labels.append(str(n))
    else:
        means.append(0)
        stds.append(0)
        labels.append(str(n))

ax.bar(labels, means, yerr=stds, alpha=0.7, capsize=5)
ax.set_xlabel('Number of Peaks')
ax.set_ylabel('Mean Position Error (nm)')
ax.set_title('Position Error by Peak Count')
ax.grid(True, alpha=0.3, axis='y')

# 样本对比
ax = axes[1, 1]
sample_idx = 0
ax.plot(target_wavelengths, test_spectra[sample_idx], 'b-', label='True', alpha=0.8)

# 重建预测光谱
pred_spectrum = np.zeros_like(target_wavelengths, dtype=float)
for j in range(MAX_PEAKS):
    if pred_exist[sample_idx, j].item() > 0.5:
        pos = pred_params[sample_idx, j, 0].item() * (wl_max - wl_min) + wl_min
        intensity = pred_params[sample_idx, j, 1].item() * (intensity_max - intensity_min) + intensity_min
        fwhm = pred_params[sample_idx, j, 2].item() * (fwhm_max - fwhm_min) + fwhm_min
        pred_spectrum += intensity * gaussian(target_wavelengths, pos, fwhm)
        ax.axvline(pos, color='r', linestyle=':', alpha=0.5)

ax.plot(target_wavelengths, pred_spectrum, 'r--', label='Pred', alpha=0.8)
ax.set_title(f'Sample (True:{test_peak_counts[sample_idx]}, Pred:{int(pred_count[sample_idx].item())})')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('evaluate_multi_peak_v3_results.png', dpi=150)
print('已保存: evaluate_multi_peak_v3_results.png')

print('\n' + '=' * 70)
print('评估完成！')
print('=' * 70)
