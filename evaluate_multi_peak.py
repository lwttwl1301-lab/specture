"""
多峰模型评估脚本
评估V3多峰模型的实际性能

评估指标：
1. 曲线级别：MSE、MAE、相关系数
2. 峰级别：峰位置误差、峰检测率、峰数量准确率
3. 按峰数分组统计
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.interpolate import UnivariateSpline
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("多峰模型性能评估")
print("=" * 70)

# ============================================================================
# 1. 加载多峰模型
# ============================================================================
print("\n[1/7] 加载多峰模型...")

checkpoint = torch.load('model_multi_peak.pth', weights_only=False)
target_wavelengths = checkpoint['wavelengths']
scaler_X = checkpoint['scaler_X']
scaler_y = checkpoint['scaler_y']
peak_config = checkpoint.get('peak_config', {})

print(f"  波长范围: {target_wavelengths[0]}-{target_wavelengths[-1]} nm")
print(f"  波长点数: {len(target_wavelengths)}")
print(f"  峰配置: {peak_config}")

# 定义模型
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

model = MultiBiasNetV2(20, 301)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# ============================================================================
# 2. 加载响应度
# ============================================================================
print("\n[2/7] 加载响应度...")

excel_path = r'D:\desktop\try\响应度矩阵.xlsx'
df = pd.read_excel(excel_path)
wavelengths_resp = df.iloc[:, 0].values
response_matrix_full = df.iloc[:, 1:].values

all_biases = np.linspace(-15, 0, 65)
indices = np.linspace(0, 64, 20, dtype=int)
response_20 = np.zeros((len(target_wavelengths), 20))

for j, idx in enumerate(indices):
    orig_resp = response_matrix_full[:, idx]
    spline = UnivariateSpline(wavelengths_resp, orig_resp, s=0.001, k=3)
    interpolated = spline(target_wavelengths)
    smoothed = gaussian_filter1d(interpolated, sigma=1.5)
    response_20[:, j] = smoothed

print(f"  响应度矩阵: {response_20.shape}")

# ============================================================================
# 3. 生成多峰测试数据
# ============================================================================
print("\n[3/7] 生成多峰测试数据...")

def gaussian(wavelengths, center, fwhm):
    """生成高斯峰"""
    sigma = fwhm / 2.355
    return np.exp(-0.5 * ((wavelengths - center) / sigma) ** 2)

def enforce_min_separation(peaks, min_sep=25):
    """确保峰之间有最小间隔"""
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
    """生成多峰光谱"""
    wl_min, wl_max = wavelengths[0], wavelengths[-1]
    margin = 30
    
    peaks = []
    for _ in range(n_peaks):
        pos = np.random.uniform(wl_min + margin, wl_max - margin)
        intensity = np.random.uniform(*config.get('intensity_range', (0.2, 1.0)))
        fwhm = np.random.uniform(*config.get('fwhm_range', (10, 60)))
        peaks.append((pos, intensity, fwhm))
    
    peaks = enforce_min_separation(peaks, min_sep=config.get('min_separation', 30))
    
    spectrum = np.zeros_like(wavelengths, dtype=float)
    for pos, intensity, fwhm in peaks:
        spectrum += intensity * gaussian(wavelengths, pos, fwhm)
    
    return spectrum, peaks

# 生成测试数据（与训练时相同的分布）
np.random.seed(123)  # 不同的种子确保测试集不同

n_single = 800
n_double = 500
n_triple = 400
n_quad = 200
n_penta = 100
n_total = n_single + n_double + n_triple + n_quad + n_penta

config = {
    'intensity_range': (0.2, 1.0),
    'fwhm_range': (10, 60),
    'min_separation': 30
}

spectra_list = []
peak_counts = []
peak_infos = []

print("  生成单峰...")
for i in range(n_single):
    spectrum, peaks = generate_multi_peak_spectrum(target_wavelengths, 1, config)
    spectra_list.append(spectrum)
    peak_counts.append(1)
    peak_infos.append(peaks)

print("  生成双峰...")
for i in range(n_double):
    spectrum, peaks = generate_multi_peak_spectrum(target_wavelengths, 2, config)
    spectra_list.append(spectrum)
    peak_counts.append(2)
    peak_infos.append(peaks)

print("  生成三峰...")
for i in range(n_triple):
    spectrum, peaks = generate_multi_peak_spectrum(target_wavelengths, 3, config)
    spectra_list.append(spectrum)
    peak_counts.append(3)
    peak_infos.append(peaks)

print("  生成四峰...")
for i in range(n_quad):
    spectrum, peaks = generate_multi_peak_spectrum(target_wavelengths, 4, config)
    spectra_list.append(spectrum)
    peak_counts.append(4)
    peak_infos.append(peaks)

print("  生成五峰...")
for i in range(n_penta):
    spectrum, peaks = generate_multi_peak_spectrum(target_wavelengths, 5, config)
    spectra_list.append(spectrum)
    peak_counts.append(5)
    peak_infos.append(peaks)

spectra_multi = np.array(spectra_list)
peak_counts = np.array(peak_counts)

print(f"  总测试样本: {n_total}")
print(f"  光谱形状: {spectra_multi.shape}")

# 统计峰数分布
unique, counts = np.unique(peak_counts, return_counts=True)
print(f"  峰数分布: {dict(zip(unique, counts))}")

# 计算测量值
measurements_multi = np.zeros((n_total, 20))
for i in range(n_total):
    for j in range(20):
        measurements_multi[i, j] = np.sum(spectra_multi[i] * response_20[:, j])

print(f"  测量值形状: {measurements_multi.shape}")

# ============================================================================
# 4. 数据标准化和预测
# ============================================================================
print("\n[4/7] 模型预测...")

X_scaled = scaler_X.transform(measurements_multi)
y_scaled = scaler_y.transform(spectra_multi)

X_tensor = torch.FloatTensor(X_scaled)
y_tensor = torch.FloatTensor(y_scaled)

with torch.no_grad():
    y_pred_scaled = model(X_tensor).numpy()

y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_true = scaler_y.inverse_transform(y_scaled)
y_pred = np.clip(y_pred, 0, None)

print(f"  预测完成: {y_pred.shape}")

# ============================================================================
# 5. 曲线级别评估
# ============================================================================
print("\n[5/7] 曲线级别评估...")

# MSE和MAE
mse_per_sample = np.mean((y_pred - y_true) ** 2, axis=1)
mae_per_sample = np.mean(np.abs(y_pred - y_true), axis=1)

# Pearson相关系数
corr_per_sample = []
for i in range(len(y_true)):
    c = np.corrcoef(y_true[i], y_pred[i])[0, 1]
    corr_per_sample.append(c)
corr_per_sample = np.array(corr_per_sample)

print("\n  【曲线级别指标】")
print(f"  MSE  均值: {np.mean(mse_per_sample):.6f}  中位数: {np.median(mse_per_sample):.6f}")
print(f"  MAE  均值: {np.mean(mae_per_sample):.6f}  中位数: {np.median(mae_per_sample):.6f}")
print(f"  相关系数 均值: {np.mean(corr_per_sample):.4f}  中位数: {np.median(corr_per_sample):.4f}")

# 按峰数分组的曲线指标
print("\n  【按峰数分组的曲线指标】")
print("-" * 70)
print(f"{'峰数':<8} {'样本数':<10} {'MSE均值':<12} {'MAE均值':<12} {'相关系数':<12}")
print("-" * 70)

for n_peaks in range(1, 6):
    mask = peak_counts == n_peaks
    if np.sum(mask) > 0:
        mse_n = np.mean(mse_per_sample[mask])
        mae_n = np.mean(mae_per_sample[mask])
        corr_n = np.mean(corr_per_sample[mask])
        print(f"{n_peaks:<8} {np.sum(mask):<10} {mse_n:<12.6f} {mae_n:<12.6f} {corr_n:<12.4f}")

print("-" * 70)

# ============================================================================
# 6. 峰级别评估
# ============================================================================
print("\n[6/7] 峰级别评估...")

def find_peaks_scipy(spectrum, wavelengths, min_prominence=0.1):
    """使用scipy找峰"""
    peaks, properties = find_peaks(spectrum, prominence=min_prominence)
    peak_positions = wavelengths[peaks]
    peak_heights = spectrum[peaks]
    return peak_positions, peak_heights

def match_peaks(true_peaks, pred_peaks):
    """匹配真实峰和预测峰（匈牙利算法）"""
    if len(true_peaks) == 0 or len(pred_peaks) == 0:
        return []
    
    cost_matrix = np.zeros((len(true_peaks), len(pred_peaks)))
    for i, tp in enumerate(true_peaks):
        for j, pp in enumerate(pred_peaks):
            cost_matrix[i, j] = abs(tp - pp)
    
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    errors = []
    for r, c in zip(row_ind, col_ind):
        errors.append(cost_matrix[r, c])
    
    return errors

# 峰检测和匹配
peak_position_errors = []
detection_ratios = []
peak_count_accuracy = []

for i in range(len(y_true)):
    true_peak_pos, _ = find_peaks_scipy(y_true[i], target_wavelengths)
    pred_peak_pos, _ = find_peaks_scipy(y_pred[i], target_wavelengths)
    
    # 真实峰数
    true_count = len(true_peak_pos)
    pred_count = len(pred_peak_pos)
    
    # 峰数准确率
    if true_count > 0:
        count_diff = abs(pred_count - true_count)
        peak_count_accuracy.append(count_diff)
    
    # 峰位置误差
    if len(true_peak_pos) > 0:
        errors = match_peaks(true_peak_pos, pred_peak_pos)
        if len(errors) > 0:
            peak_position_errors.extend(errors)
        
        # 检测率
        detection_ratio = len(pred_peak_pos) / len(true_peak_pos)
        detection_ratios.append(detection_ratio)

peak_position_errors = np.array(peak_position_errors)
detection_ratios = np.array(detection_ratios)
peak_count_accuracy = np.array(peak_count_accuracy)

print("\n  【峰级别指标】")
print(f"  峰位置误差: {np.mean(peak_position_errors):.2f} nm (中位数: {np.median(peak_position_errors):.2f} nm)")
print(f"  峰检测比: {np.mean(detection_ratios):.2f} (理想=1.0)")
print(f"  峰数误差: {np.mean(peak_count_accuracy):.2f} (平均差值)")

# 按峰数分组的峰指标
print("\n  【按峰数分组的峰指标】")
print("-" * 80)
print(f"{'峰数':<8} {'样本数':<10} {'峰误差(nm)':<15} {'检测比':<12} {'峰数误差':<12}")
print("-" * 80)

for n_peaks in range(1, 6):
    mask = peak_counts == n_peaks
    indices = np.where(mask)[0]
    
    if len(indices) > 0:
        # 计算该组的峰误差
        errors_n = []
        detection_ratios_n = []
        count_errors_n = []
        
        for idx in indices:
            true_peak_pos, _ = find_peaks_scipy(y_true[idx], target_wavelengths)
            pred_peak_pos, _ = find_peaks_scipy(y_pred[idx], target_wavelengths)
            
            if len(true_peak_pos) > 0:
                e = match_peaks(true_peak_pos, pred_peak_pos)
                if len(e) > 0:
                    errors_n.extend(e)
                
                detection_ratios_n.append(len(pred_peak_pos) / len(true_peak_pos))
                count_errors_n.append(abs(len(pred_peak_pos) - len(true_peak_pos)))
        
        peak_err_str = f'{np.mean(errors_n):.2f}' if len(errors_n) > 0 else 'N/A'
        det_ratio_str = f'{np.mean(detection_ratios_n):.2f}' if len(detection_ratios_n) > 0 else 'N/A'
        count_err_str = f'{np.mean(count_errors_n):.2f}' if len(count_errors_n) > 0 else 'N/A'
        
        print(f"{n_peaks:<8} {len(indices):<10} {peak_err_str:<15} {det_ratio_str:<12} {count_err_str:<12}")

print("-" * 80)

# ============================================================================
# 7. 可视化
# ============================================================================
print("\n[7/7] 生成可视化...")

fig = plt.figure(figsize=(20, 16))

# 7.1 MSE分布
ax = fig.add_subplot(3, 3, 1)
ax.hist(mse_per_sample, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
ax.axvline(np.mean(mse_per_sample), color='red', linestyle='--', linewidth=2, label=f'Mean={np.mean(mse_per_sample):.4f}')
ax.set_xlabel('MSE per Sample')
ax.set_ylabel('Count')
ax.set_title('MSE Distribution')
ax.legend()
ax.grid(True, alpha=0.3)

# 7.2 相关系数分布
ax = fig.add_subplot(3, 3, 2)
ax.hist(corr_per_sample, bins=50, edgecolor='black', alpha=0.7, color='purple')
ax.axvline(np.mean(corr_per_sample), color='red', linestyle='--', linewidth=2, label=f'Mean={np.mean(corr_per_sample):.4f}')
ax.set_xlabel('Pearson Correlation')
ax.set_ylabel('Count')
ax.set_title('Correlation Distribution')
ax.legend()
ax.grid(True, alpha=0.3)

# 7.3 峰位置误差分布
ax = fig.add_subplot(3, 3, 3)
if len(peak_position_errors) > 0:
    ax.hist(peak_position_errors, bins=50, edgecolor='black', alpha=0.7, color='green')
    ax.axvline(np.mean(peak_position_errors), color='red', linestyle='--', linewidth=2, label=f'Mean={np.mean(peak_position_errors):.2f}nm')
ax.set_xlabel('Peak Position Error (nm)')
ax.set_ylabel('Count')
ax.set_title('Peak Position Error Distribution')
ax.legend()
ax.grid(True, alpha=0.3)

# 7.4 峰检测比分布
ax = fig.add_subplot(3, 3, 4)
ax.hist(detection_ratios, bins=30, edgecolor='black', alpha=0.7, color='orange')
ax.axvline(1.0, color='green', linestyle='--', linewidth=2, label='Ideal=1.0')
ax.set_xlabel('Peak Detection Ratio')
ax.set_ylabel('Count')
ax.set_title('Peak Detection Ratio Distribution')
ax.legend()
ax.grid(True, alpha=0.3)

# 7.5 按峰数的MSE
ax = fig.add_subplot(3, 3, 5)
means = []
labels = []
for n in range(1, 6):
    mask = peak_counts == n
    if np.sum(mask) > 0:
        means.append(np.mean(mse_per_sample[mask]))
        labels.append(f'{n}')
    else:
        means.append(0)
        labels.append(f'{n}')

ax.bar(labels, means, alpha=0.7, color='coral')
ax.set_xlabel('Number of Peaks')
ax.set_ylabel('Mean MSE')
ax.set_title('MSE by Peak Count')
ax.grid(True, alpha=0.3, axis='y')

# 7.6 按峰数的相关系数
ax = fig.add_subplot(3, 3, 6)
means = []
for n in range(1, 6):
    mask = peak_counts == n
    if np.sum(mask) > 0:
        means.append(np.mean(corr_per_sample[mask]))
    else:
        means.append(0)

ax.bar(labels, means, alpha=0.7, color='teal')
ax.set_xlabel('Number of Peaks')
ax.set_ylabel('Mean Correlation')
ax.set_title('Correlation by Peak Count')
ax.grid(True, alpha=0.3, axis='y')

# 7.7 单峰样本对比
ax = fig.add_subplot(3, 3, 7)
single_idx = np.where(peak_counts == 1)[0][0]
ax.plot(target_wavelengths, y_true[single_idx], 'b-', linewidth=2, label='True', alpha=0.8)
ax.plot(target_wavelengths, y_pred[single_idx], 'r--', linewidth=2, label='Pred', alpha=0.8)
ax.set_title(f'Single Peak Sample\nCorr={corr_per_sample[single_idx]:.4f}')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xlim(1000, 1300)

# 7.8 双峰样本对比
ax = fig.add_subplot(3, 3, 8)
double_idx = np.where(peak_counts == 2)[0][0]
ax.plot(target_wavelengths, y_true[double_idx], 'b-', linewidth=2, label='True', alpha=0.8)
ax.plot(target_wavelengths, y_pred[double_idx], 'r--', linewidth=2, label='Pred', alpha=0.8)
ax.set_title(f'Double Peak Sample\nCorr={corr_per_sample[double_idx]:.4f}')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xlim(1000, 1300)

# 7.9 三峰样本对比
ax = fig.add_subplot(3, 3, 9)
triple_idx = np.where(peak_counts == 3)[0][0]
ax.plot(target_wavelengths, y_true[triple_idx], 'b-', linewidth=2, label='True', alpha=0.8)
ax.plot(target_wavelengths, y_pred[triple_idx], 'r--', linewidth=2, label='Pred', alpha=0.8)
ax.set_title(f'Triple Peak Sample\nCorr={corr_per_sample[triple_idx]:.4f}')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xlim(1000, 1300)

plt.tight_layout()
plt.savefig('multi_peak_evaluation.png', dpi=150)
print("  已保存: multi_peak_evaluation.png")

# ============================================================================
# 8. 最终总结
# ============================================================================
print("\n" + "=" * 70)
print("多峰模型评估总结")
print("=" * 70)

print(f"""
╔══════════════════════════════════════════════════════════════════════╗
║                      多峰光谱重建模型评估结果                         ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  【测试数据配置】                                                     ║
║    单峰: {n_single} ({n_single/n_total*100:.0f}%)                                            ║
║    双峰: {n_double} ({n_double/n_total*100:.0f}%)                                            ║
║    三峰: {n_triple} ({n_triple/n_total*100:.0f}%)                                            ║
║    四峰: {n_quad} ({n_quad/n_total*100:.0f}%)                                             ║
║    五峰: {n_penta} ({n_penta/n_total*100:.0f}%)                                              ║
║    总计: {n_total}                                           ║
║                                                                      ║
║  【曲线级别指标】                                                     ║
║    MSE:      {np.mean(mse_per_sample):.6f} (中位数: {np.median(mse_per_sample):.6f})              ║
║    MAE:      {np.mean(mae_per_sample):.6f} (中位数: {np.median(mae_per_sample):.6f})              ║
║    相关系数: {np.mean(corr_per_sample):.4f} (中位数: {np.median(corr_per_sample):.4f})                  ║
║                                                                      ║
║  【峰级别指标】                                                       ║
║    峰位置误差: {np.mean(peak_position_errors):.2f} nm (中位数: {np.median(peak_position_errors):.2f} nm)              ║
║    峰检测比:   {np.mean(detection_ratios):.2f} (理想=1.0)                               ║
║    峰数误差:   {np.mean(peak_count_accuracy):.2f} (平均差值)                              ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
""")

# 相关系数阈值分析
print("相关系数阈值分析:")
print("-" * 40)
print(f"  相关系数 > 0.99: {np.sum(corr_per_sample > 0.99) / len(corr_per_sample) * 100:.1f}%")
print(f"  相关系数 > 0.95: {np.sum(corr_per_sample > 0.95) / len(corr_per_sample) * 100:.1f}%")
print(f"  相关系数 > 0.90: {np.sum(corr_per_sample > 0.90) / len(corr_per_sample) * 100:.1f}%")
print("-" * 40)

print("\n" + "=" * 70)
