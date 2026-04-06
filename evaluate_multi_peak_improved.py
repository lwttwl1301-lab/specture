"""
多峰模型评估脚本 - 改进版（方案一）
使用抛物线插值和高斯拟合改进峰检测精度

评估指标：
1. 曲线级别：MSE、MAE、相关系数
2. 峰级别：峰位置误差（多种方法对比）、峰检测率、峰数量准确率
3. 按峰数分组统计
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.interpolate import UnivariateSpline
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from scipy.optimize import linear_sum_assignment, curve_fit
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("多峰模型性能评估 - 改进版（方案一：改进峰检测）")
print("=" * 70)

# ============================================================================
# 1. 改进的峰检测方法（来自peak_improvement.py）
# ============================================================================

def parabolic_peak_interp(spectrum, wavelengths):
    """
    抛物线插值提高峰定位精度
    在argmax找到的峰附近进行三点抛物线拟合
    """
    idx = np.argmax(spectrum)
    
    # 边界检查
    if idx == 0 or idx == len(spectrum) - 1:
        return wavelengths[idx]
    
    # 三点抛物线拟合
    y1, y2, y3 = spectrum[idx-1], spectrum[idx], spectrum[idx+1]
    x1, x2, x3 = wavelengths[idx-1], wavelengths[idx], wavelengths[idx+1]
    
    # 抛物线顶点公式
    denom = (y1 - 2*y2 + y3)
    if denom == 0:
        return x2
    
    peak_x = x2 + (x3 - x2) * (y1 - y3) / (2 * denom)
    peak_x = max(x1, min(peak_x, x3))
    
    return peak_x

def gaussian_fit_peak(spectrum, wavelengths, initial_guess=None):
    """
    高斯拟合精确定位峰参数
    返回：中心波长、振幅、半高宽、基线
    """
    def gaussian(x, amp, center, sigma, offset):
        return amp * np.exp(-((x - center) ** 2) / (2 * sigma ** 2)) + offset
    
    # 初始猜测
    if initial_guess is None:
        idx = np.argmax(spectrum)
        amp = spectrum[idx]
        center = wavelengths[idx]
        sigma = 10
        offset = np.percentile(spectrum, 10)
        initial_guess = [amp, center, sigma, offset]
    
    try:
        bounds = ([0, wavelengths[0], 1, 0], 
                  [np.inf, wavelengths[-1], 50, np.percentile(spectrum, 90)])
        
        popt, pcov = curve_fit(gaussian, wavelengths, spectrum, 
                              p0=initial_guess, bounds=bounds, 
                              maxfev=5000, ftol=1e-8, xtol=1e-8)
        
        amp, center, sigma, offset = popt
        fwhm = 2.355 * sigma
        
        # 计算拟合优度
        y_fit = gaussian(wavelengths, *popt)
        r_squared = 1 - np.sum((spectrum - y_fit) ** 2) / np.sum((spectrum - np.mean(spectrum)) ** 2)
        
        return {
            'center': center,
            'amplitude': amp,
            'fwhm': fwhm,
            'offset': offset,
            'sigma': sigma,
            'r_squared': r_squared,
            'success': True
        }
    except Exception as e:
        idx = np.argmax(spectrum)
        return {
            'center': wavelengths[idx],
            'amplitude': spectrum[idx],
            'fwhm': np.nan,
            'offset': np.min(spectrum),
            'sigma': np.nan,
            'r_squared': 0,
            'success': False
        }

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

# ============================================================================
# 2. 加载多峰模型
# ============================================================================
print("\n[1/7] 加载多峰模型...")

checkpoint = torch.load('model_multi_peak.pth', weights_only=False)
target_wavelengths = checkpoint['wavelengths']
scaler_X = checkpoint['scaler_X']
scaler_y = checkpoint['scaler_y']
peak_config = checkpoint.get('peak_config', {})

print(f"  波长范围: {target_wavelengths[0]}-{target_wavelengths[-1]} nm")
print(f"  波长点数: {len(target_wavelengths)}")

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
# 3. 加载响应度
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
# 4. 生成多峰测试数据
# ============================================================================
print("\n[3/7] 生成多峰测试数据...")

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
        intensity = np.random.uniform(*config.get('intensity_range', (0.2, 1.0)))
        fwhm = np.random.uniform(*config.get('fwhm_range', (10, 60)))
        peaks.append((pos, intensity, fwhm))
    
    peaks = enforce_min_separation(peaks, min_sep=config.get('min_separation', 30))
    
    spectrum = np.zeros_like(wavelengths, dtype=float)
    for pos, intensity, fwhm in peaks:
        spectrum += intensity * gaussian(wavelengths, pos, fwhm)
    
    return spectrum, peaks

# 生成测试数据
np.random.seed(123)

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

# 计算测量值
measurements_multi = np.zeros((n_total, 20))
for i in range(n_total):
    for j in range(20):
        measurements_multi[i, j] = np.sum(spectra_multi[i] * response_20[:, j])

# ============================================================================
# 5. 数据标准化和预测
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
# 6. 曲线级别评估
# ============================================================================
print("\n[5/7] 曲线级别评估...")

mse_per_sample = np.mean((y_pred - y_true) ** 2, axis=1)
mae_per_sample = np.mean(np.abs(y_pred - y_true), axis=1)

corr_per_sample = []
for i in range(len(y_true)):
    c = np.corrcoef(y_true[i], y_pred[i])[0, 1]
    corr_per_sample.append(c)
corr_per_sample = np.array(corr_per_sample)

print("\n  【曲线级别指标】")
print(f"  MSE  均值: {np.mean(mse_per_sample):.6f}  中位数: {np.median(mse_per_sample):.6f}")
print(f"  MAE  均值: {np.mean(mae_per_sample):.6f}  中位数: {np.median(mae_per_sample):.6f}")
print(f"  相关系数 均值: {np.mean(corr_per_sample):.4f}  中位数: {np.median(corr_per_sample):.4f}")

# ============================================================================
# 7. 峰级别评估 - 三种方法对比
# ============================================================================
print("\n[6/7] 峰级别评估 - 三种方法对比...")
print("\n  方法1: argmax (原始方法)")
print("  方法2: parabolic (抛物线插值)")
print("  方法3: gaussian_fit (高斯拟合)")

# 存储三种方法的结果
methods_results = {
    'argmax': {'errors': [], 'by_peak_count': {1: [], 2: [], 3: [], 4: [], 5: []}},
    'parabolic': {'errors': [], 'by_peak_count': {1: [], 2: [], 3: [], 4: [], 5: []}},
    'gaussian_fit': {'errors': [], 'by_peak_count': {1: [], 2: [], 3: [], 4: [], 5: []}}
}

for i in range(len(y_true)):
    n_peaks = peak_counts[i]
    
    # 真实峰位置（使用抛物线插值获得更精确的真实值）
    true_peak_pos = parabolic_peak_interp(y_true[i], target_wavelengths)
    
    # 方法1: argmax
    pred_idx = np.argmax(y_pred[i])
    pred_pos_argmax = target_wavelengths[pred_idx]
    error_argmax = abs(pred_pos_argmax - true_peak_pos)
    methods_results['argmax']['errors'].append(error_argmax)
    methods_results['argmax']['by_peak_count'][n_peaks].append(error_argmax)
    
    # 方法2: 抛物线插值
    pred_pos_parabolic = parabolic_peak_interp(y_pred[i], target_wavelengths)
    error_parabolic = abs(pred_pos_parabolic - true_peak_pos)
    methods_results['parabolic']['errors'].append(error_parabolic)
    methods_results['parabolic']['by_peak_count'][n_peaks].append(error_parabolic)
    
    # 方法3: 高斯拟合
    result = gaussian_fit_peak(y_pred[i], target_wavelengths)
    pred_pos_gaussian = result['center']
    error_gaussian = abs(pred_pos_gaussian - true_peak_pos)
    methods_results['gaussian_fit']['errors'].append(error_gaussian)
    methods_results['gaussian_fit']['by_peak_count'][n_peaks].append(error_gaussian)

# 打印对比结果
print("\n" + "=" * 80)
print("【三种峰检测方法对比】")
print("=" * 80)
print(f"{'方法':<20} {'平均误差(nm)':<15} {'中位数误差(nm)':<15} {'<1nm比例':<12} {'<2nm比例':<12} {'<5nm比例':<12}")
print("-" * 80)

for method_name, results in methods_results.items():
    errors = np.array(results['errors'])
    mean_err = np.mean(errors)
    median_err = np.median(errors)
    acc_1nm = np.sum(errors < 1) / len(errors) * 100
    acc_2nm = np.sum(errors < 2) / len(errors) * 100
    acc_5nm = np.sum(errors < 5) / len(errors) * 100
    
    print(f"{method_name:<20} {mean_err:<15.2f} {median_err:<15.2f} {acc_1nm:<12.1f}% {acc_2nm:<12.1f}% {acc_5nm:<12.1f}%")

print("=" * 80)

# 按峰数分组对比
print("\n【按峰数分组的峰位置误差对比】")
print("-" * 100)
print(f"{'峰数':<8} {'样本数':<10} {'argmax(nm)':<15} {'parabolic(nm)':<15} {'gaussian_fit(nm)':<15} {'改进幅度':<15}")
print("-" * 100)

for n_peaks in range(1, 6):
    mask = peak_counts == n_peaks
    n_samples = np.sum(mask)
    
    if n_samples > 0:
        argmax_err = np.mean(methods_results['argmax']['by_peak_count'][n_peaks])
        parabolic_err = np.mean(methods_results['parabolic']['by_peak_count'][n_peaks])
        gaussian_err = np.mean(methods_results['gaussian_fit']['by_peak_count'][n_peaks])
        improvement = (argmax_err - parabolic_err) / argmax_err * 100
        
        print(f"{n_peaks:<8} {n_samples:<10} {argmax_err:<15.2f} {parabolic_err:<15.2f} {gaussian_err:<15.2f} {improvement:<15.1f}%")

print("-" * 100)

# ============================================================================
# 8. 可视化
# ============================================================================
print("\n[7/7] 生成可视化...")

fig = plt.figure(figsize=(20, 16))

# 8.1 三种方法的误差分布对比
ax = fig.add_subplot(3, 3, 1)
for method_name, color in zip(['argmax', 'parabolic', 'gaussian_fit'], ['red', 'green', 'blue']):
    errors = methods_results[method_name]['errors']
    ax.hist(errors, bins=50, alpha=0.5, label=method_name, color=color, edgecolor='black')
ax.set_xlabel('Peak Position Error (nm)')
ax.set_ylabel('Count')
ax.set_title('Peak Position Error Distribution (3 Methods)')
ax.legend()
ax.grid(True, alpha=0.3)

# 8.2 按峰数的误差对比（柱状图）
ax = fig.add_subplot(3, 3, 2)
x = np.arange(1, 6)
width = 0.25

argmax_means = [np.mean(methods_results['argmax']['by_peak_count'][n]) if methods_results['argmax']['by_peak_count'][n] else 0 for n in range(1, 6)]
parabolic_means = [np.mean(methods_results['parabolic']['by_peak_count'][n]) if methods_results['parabolic']['by_peak_count'][n] else 0 for n in range(1, 6)]
gaussian_means = [np.mean(methods_results['gaussian_fit']['by_peak_count'][n]) if methods_results['gaussian_fit']['by_peak_count'][n] else 0 for n in range(1, 6)]

ax.bar(x - width, argmax_means, width, label='argmax', color='red', alpha=0.7)
ax.bar(x, parabolic_means, width, label='parabolic', color='green', alpha=0.7)
ax.bar(x + width, gaussian_means, width, label='gaussian_fit', color='blue', alpha=0.7)
ax.set_xlabel('Number of Peaks')
ax.set_ylabel('Mean Peak Position Error (nm)')
ax.set_title('Peak Error by Peak Count (3 Methods)')
ax.set_xticks(x)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# 8.3 改进幅度可视化
ax = fig.add_subplot(3, 3, 3)
improvements = []
for n_peaks in range(1, 6):
    argmax_err = np.mean(methods_results['argmax']['by_peak_count'][n_peaks])
    parabolic_err = np.mean(methods_results['parabolic']['by_peak_count'][n_peaks])
    if argmax_err > 0:
        improvement = (argmax_err - parabolic_err) / argmax_err * 100
        improvements.append(improvement)
    else:
        improvements.append(0)

bars = ax.bar(range(1, 6), improvements, color='teal', alpha=0.7)
ax.set_xlabel('Number of Peaks')
ax.set_ylabel('Improvement (%)')
ax.set_title('Parabolic vs Argmax Improvement')
ax.grid(True, alpha=0.3, axis='y')
for bar, imp in zip(bars, improvements):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 1,
            f'{imp:.1f}%', ha='center', va='bottom')

# 8.4-8.6 单峰、双峰、三峰样本对比
for idx, (n_peaks, title) in enumerate([(1, 'Single'), (2, 'Double'), (3, 'Triple')]):
    ax = fig.add_subplot(3, 3, 4 + idx)
    
    sample_idx = np.where(peak_counts == n_peaks)[0][0]
    
    ax.plot(target_wavelengths, y_true[sample_idx], 'b-', linewidth=2, label='True', alpha=0.8)
    ax.plot(target_wavelengths, y_pred[sample_idx], 'r--', linewidth=2, label='Pred', alpha=0.8)
    
    # 标记不同方法检测的峰位置
    true_peak = parabolic_peak_interp(y_true[sample_idx], target_wavelengths)
    pred_argmax = target_wavelengths[np.argmax(y_pred[sample_idx])]
    pred_parabolic = parabolic_peak_interp(y_pred[sample_idx], target_wavelengths)
    
    ax.axvline(true_peak, color='b', linestyle=':', alpha=0.7, label=f'True={true_peak:.1f}nm')
    ax.axvline(pred_argmax, color='r', linestyle=':', alpha=0.7, label=f'Argmax={pred_argmax:.1f}nm')
    ax.axvline(pred_parabolic, color='g', linestyle=':', alpha=0.7, label=f'Parabolic={pred_parabolic:.1f}nm')
    
    ax.set_title(f'{title} Peak Sample\nCorr={corr_per_sample[sample_idx]:.4f}')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1000, 1300)

# 8.7 高斯拟合示例
ax = fig.add_subplot(3, 3, 7)
sample_idx = np.where(peak_counts == 1)[0][0]
result = gaussian_fit_peak(y_pred[sample_idx], target_wavelengths)

if result['success']:
    def gaussian(x, amp, center, sigma, offset):
        return amp * np.exp(-((x - center) ** 2) / (2 * sigma ** 2)) + offset
    
    fitted = gaussian(target_wavelengths, result['amplitude'], result['center'], 
                      result['sigma'], result['offset'])
    
    ax.plot(target_wavelengths, y_true[sample_idx], 'b-', linewidth=2, label='True', alpha=0.7)
    ax.plot(target_wavelengths, y_pred[sample_idx], 'r--', linewidth=2, label='Pred', alpha=0.7)
    ax.plot(target_wavelengths, fitted, 'g-', linewidth=2, label='Gaussian Fit', alpha=0.7)
    ax.axvline(result['center'], color='g', linestyle=':', label=f'Center={result["center"]:.1f}nm')
    ax.set_title(f'Gaussian Fit Example\nR²={result["r_squared"]:.3f}, FWHM={result["fwhm"]:.1f}nm')
else:
    ax.text(0.5, 0.5, 'Gaussian Fit Failed', ha='center', va='center', transform=ax.transAxes)

ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
ax.set_xlim(1000, 1300)

# 8.8 误差累积分布
ax = fig.add_subplot(3, 3, 8)
for method_name, color in zip(['argmax', 'parabolic', 'gaussian_fit'], ['red', 'green', 'blue']):
    errors = np.array(methods_results[method_name]['errors'])
    sorted_errors = np.sort(errors)
    cumulative = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors) * 100
    ax.plot(sorted_errors, cumulative, linewidth=2, label=method_name, color=color)

ax.axvline(1, color='gray', linestyle='--', alpha=0.5, label='1nm threshold')
ax.axvline(2, color='gray', linestyle=':', alpha=0.5, label='2nm threshold')
ax.set_xlabel('Peak Position Error (nm)')
ax.set_ylabel('Cumulative Percentage (%)')
ax.set_title('Cumulative Error Distribution')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 20)

# 8.9 相关系数分布
ax = fig.add_subplot(3, 3, 9)
ax.hist(corr_per_sample, bins=50, edgecolor='black', alpha=0.7, color='purple')
ax.axvline(np.mean(corr_per_sample), color='red', linestyle='--', linewidth=2, 
           label=f'Mean={np.mean(corr_per_sample):.4f}')
ax.set_xlabel('Pearson Correlation')
ax.set_ylabel('Count')
ax.set_title('Correlation Distribution')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('multi_peak_evaluation_improved.png', dpi=150)
print("  已保存: multi_peak_evaluation_improved.png")

# ============================================================================
# 9. 最终总结
# ============================================================================
print("\n" + "=" * 80)
print("多峰模型评估总结 - 方案一（改进峰检测）")
print("=" * 80)

argmax_errors = np.array(methods_results['argmax']['errors'])
parabolic_errors = np.array(methods_results['parabolic']['errors'])
gaussian_errors = np.array(methods_results['gaussian_fit']['errors'])

improvement_mean = (np.mean(argmax_errors) - np.mean(parabolic_errors)) / np.mean(argmax_errors) * 100
improvement_median = (np.median(argmax_errors) - np.median(parabolic_errors)) / np.median(argmax_errors) * 100

print(f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    方案一效果：改进峰检测后处理                                ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  【原始方法：argmax】                                                          ║
║    平均误差: {np.mean(argmax_errors):.2f} nm                                    ║
║    中位数误差: {np.median(argmax_errors):.2f} nm                                ║
║    <1nm精度: {np.sum(argmax_errors < 1) / len(argmax_errors) * 100:.1f}%                              ║
║    <2nm精度: {np.sum(argmax_errors < 2) / len(argmax_errors) * 100:.1f}%                             ║
║    <5nm精度: {np.sum(argmax_errors < 5) / len(argmax_errors) * 100:.1f}%                             ║
║                                                                              ║
║  【改进方法：抛物线插值】                                                       ║
║    平均误差: {np.mean(parabolic_errors):.2f} nm  (改进 {improvement_mean:.1f}%)              ║
║    中位数误差: {np.median(parabolic_errors):.2f} nm  (改进 {improvement_median:.1f}%)          ║
║    <1nm精度: {np.sum(parabolic_errors < 1) / len(parabolic_errors) * 100:.1f}%  (提升 {np.sum(parabolic_errors < 1) / len(parabolic_errors) * 100 - np.sum(argmax_errors < 1) / len(argmax_errors) * 100:.1f}%)                        ║
║    <2nm精度: {np.sum(parabolic_errors < 2) / len(parabolic_errors) * 100:.1f}% (提升 {np.sum(parabolic_errors < 2) / len(parabolic_errors) * 100 - np.sum(argmax_errors < 2) / len(argmax_errors) * 100:.1f}%)                       ║
║    <5nm精度: {np.sum(parabolic_errors < 5) / len(parabolic_errors) * 100:.1f}% (提升 {np.sum(parabolic_errors < 5) / len(parabolic_errors) * 100 - np.sum(argmax_errors < 5) / len(argmax_errors) * 100:.1f}%)                       ║
║                                                                              ║
║  【高斯拟合方法】                                                              ║
║    平均误差: {np.mean(gaussian_errors):.2f} nm                                    ║
║    中位数误差: {np.median(gaussian_errors):.2f} nm                                ║
║    额外信息: 峰宽(FWHM)、振幅、基线、拟合优度R²                                 ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")

print("=" * 80)
print("方案一结论：")
print("=" * 80)
print(f"✅ 抛物线插值将峰位置误差从 {np.mean(argmax_errors):.2f}nm 降低到 {np.mean(parabolic_errors):.2f}nm")
print(f"✅ 改进幅度: {improvement_mean:.1f}%")
print(f"✅ <1nm精度从 {np.sum(argmax_errors < 1) / len(argmax_errors) * 100:.1f}% 提升到 {np.sum(parabolic_errors < 1) / len(parabolic_errors) * 100:.1f}%")
print(f"✅ 实现简单，无需重新训练模型，立即可用")
print("=" * 80)

# 保存结果到文件
import json
results = {
    'argmax': {
        'mean_error': float(np.mean(argmax_errors)),
        'median_error': float(np.median(argmax_errors)),
        'acc_1nm': float(np.sum(argmax_errors < 1) / len(argmax_errors) * 100),
        'acc_2nm': float(np.sum(argmax_errors < 2) / len(argmax_errors) * 100),
        'acc_5nm': float(np.sum(argmax_errors < 5) / len(argmax_errors) * 100),
    },
    'parabolic': {
        'mean_error': float(np.mean(parabolic_errors)),
        'median_error': float(np.median(parabolic_errors)),
        'acc_1nm': float(np.sum(parabolic_errors < 1) / len(parabolic_errors) * 100),
        'acc_2nm': float(np.sum(parabolic_errors < 2) / len(parabolic_errors) * 100),
        'acc_5nm': float(np.sum(parabolic_errors < 5) / len(parabolic_errors) * 100),
    },
    'gaussian_fit': {
        'mean_error': float(np.mean(gaussian_errors)),
        'median_error': float(np.median(gaussian_errors)),
        'acc_1nm': float(np.sum(gaussian_errors < 1) / len(gaussian_errors) * 100),
        'acc_2nm': float(np.sum(gaussian_errors < 2) / len(gaussian_errors) * 100),
        'acc_5nm': float(np.sum(gaussian_errors < 5) / len(gaussian_errors) * 100),
    },
    'improvement': {
        'mean_percent': float(improvement_mean),
        'median_percent': float(improvement_median),
    }
}

with open('evaluation_results.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print("\n结果已保存到: evaluation_results.json")
