"""
曲线级别评估脚本
计算光谱重建的曲线相似度指标

评估指标：
1. MSE/MAE - 整体误差大小
2. SAD (Spectral Angle Distance) - 光谱角距离
3. Pearson相关系数 - 曲线形状相关性
4. FWHM对比 - 半高宽对比
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.interpolate import UnivariateSpline
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, savgol_filter
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("曲线级别光谱重建评估")
print("=" * 70)

# ============================================================================
# 1. 加载模型
# ============================================================================
print("\n[1/6] 加载模型...")

checkpoint = torch.load('model_multi_bias_v2.pth', weights_only=False)
target_wavelengths = checkpoint['wavelengths']
scaler_X = checkpoint['scaler_X']
scaler_y = checkpoint['scaler_y']

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
# 2. 生成测试数据
# ============================================================================
print("\n[2/6] 生成测试数据...")

# 加载响应度
excel_path = r'D:\desktop\try\响应度矩阵.xlsx'
df = pd.read_excel(excel_path)
wavelengths_resp = df.iloc[:, 0].values
response_matrix_full = df.iloc[:, 1:].values

# 插值响应度
all_biases = np.linspace(-15, 0, 65)
indices = np.linspace(0, 64, 20, dtype=int)
response_20 = np.zeros((len(target_wavelengths), 20))

for j, idx in enumerate(indices):
    orig_resp = response_matrix_full[:, idx]
    spline = UnivariateSpline(wavelengths_resp, orig_resp, s=0.001, k=3)
    interpolated = spline(target_wavelengths)
    smoothed = gaussian_filter1d(interpolated, sigma=1.5)
    response_20[:, j] = smoothed

# 生成测试集
np.random.seed(42)
n_main = 6000
n_edge = 2000

peak_wavelengths_main = np.random.uniform(1000, 1300, n_main)
peak_intensities_main = np.random.uniform(0.5, 1.0, n_main)
fwhms_main = np.random.uniform(20, 50, n_main)

peak_wavelengths_low = np.random.uniform(1000, 1050, n_edge)
peak_intensities_low = np.random.uniform(0.5, 1.0, n_edge)
fwhms_low = np.random.uniform(20, 50, n_edge)

peak_wavelengths_high = np.random.uniform(1250, 1300, n_edge)
peak_intensities_high = np.random.uniform(0.5, 1.0, n_edge)
fwhms_high = np.random.uniform(20, 50, n_edge)

peak_wavelengths = np.concatenate([peak_wavelengths_main, peak_wavelengths_low, peak_wavelengths_high])
peak_intensities = np.concatenate([peak_intensities_main, peak_intensities_low, peak_intensities_high])
fwhms = np.concatenate([fwhms_main, fwhms_low, fwhms_high])

# 生成光谱
spectra_test = np.zeros((len(peak_wavelengths), len(target_wavelengths)))
for i in range(len(peak_wavelengths)):
    wl = target_wavelengths
    spectrum = peak_intensities[i] * np.exp(-0.5 * ((wl - peak_wavelengths[i]) / (fwhms[i] / 2.355)) ** 2)
    spectra_test[i] = spectrum

# 计算测量值
measurements_test = np.zeros((len(peak_wavelengths), 20))
for i in range(len(peak_wavelengths)):
    for j in range(20):
        measurements_test[i, j] = np.sum(spectra_test[i] * response_20[:, j]) * 1

# 标准化
X_scaled = scaler_X.transform(measurements_test)
y_scaled = scaler_y.transform(spectra_test)

# 打乱数据
np.random.seed(123)
shuffle_idx = np.random.permutation(len(X_scaled))
X_scaled = X_scaled[shuffle_idx]
y_scaled = y_scaled[shuffle_idx]
spectra_test = spectra_test[shuffle_idx]
peak_wavelengths_shuffled = peak_wavelengths[shuffle_idx]

# 取测试集
n_train = int(0.8 * len(X_scaled))
X_test = X_scaled[n_train:]
y_test = y_scaled[n_train:]
spectra_test_actual = spectra_test[n_train:]
peak_wavelengths_test = peak_wavelengths_shuffled[n_train:]

print(f"  测试样本数: {len(X_test)}")

# ============================================================================
# 3. 模型预测
# ============================================================================
print("\n[3/6] 模型预测...")

with torch.no_grad():
    y_pred_scaled = model(torch.FloatTensor(X_test)).numpy()

y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_true = scaler_y.inverse_transform(y_test)
y_pred = np.clip(y_pred, 0, None)

print(f"  预测完成，形状: {y_pred.shape}")

# ============================================================================
# 4. 计算曲线级别指标
# ============================================================================
print("\n[4/6] 计算曲线级别指标...")

# 4.1 MSE 和 MAE
mse_per_sample = np.mean((y_pred - y_true) ** 2, axis=1)
mae_per_sample = np.mean(np.abs(y_pred - y_true), axis=1)

print("\n  【整体误差指标】")
print(f"  MSE - 均值: {np.mean(mse_per_sample):.6f}, 中位数: {np.median(mse_per_sample):.6f}")
print(f"  MAE - 均值: {np.mean(mae_per_sample):.6f}, 中位数: {np.median(mae_per_sample):.6f}")

# 4.2 光谱角距离 (SAD)
def spectral_angle_distance(y_true, y_pred):
    """计算光谱角距离（弧度）"""
    # 将光谱视为向量
    dot_product = np.sum(y_true * y_pred, axis=1)
    norm_true = np.linalg.norm(y_true, axis=1)
    norm_pred = np.linalg.norm(y_pred, axis=1)
    
    # 避免除零
    cos_angle = dot_product / (norm_true * norm_pred + 1e-10)
    cos_angle = np.clip(cos_angle, -1, 1)
    
    angle = np.arccos(cos_angle)
    return angle

sad = spectral_angle_distance(y_true, y_pred)
sad_degrees = np.degrees(sad)

print(f"\n  【光谱角距离 (SAD)】")
print(f"  均值: {np.mean(sad_degrees):.4f}°")
print(f"  中位数: {np.median(sad_degrees):.4f}°")
print(f"  范围: {np.min(sad_degrees):.4f}° - {np.max(sad_degrees):.4f}°")

# 4.3 Pearson相关系数
def pearson_correlation(y_true, y_pred):
    """计算每对光谱的Pearson相关系数"""
    n = y_true.shape[1]
    
    mean_true = np.mean(y_true, axis=1, keepdims=True)
    mean_pred = np.mean(y_pred, axis=1, keepdims=True)
    
    std_true = np.std(y_true, axis=1, keepdims=True) + 1e-10
    std_pred = np.std(y_pred, axis=1, keepdims=True) + 1e-10
    
    cov = np.mean((y_true - mean_true) * (y_pred - mean_pred), axis=1)
    corr = cov / (std_true.squeeze() * std_pred.squeeze())
    
    return corr

corr = pearson_correlation(y_true, y_pred)

print(f"\n  【Pearson相关系数】")
print(f"  均值: {np.mean(corr):.6f}")
print(f"  中位数: {np.median(corr):.6f}")
print(f"  范围: {np.min(corr):.6f} - {np.max(corr):.6f}")

# 4.4 R² 决定系数
def r2_score(y_true, y_pred):
    """计算R²分数"""
    ss_res = np.sum((y_true - y_pred) ** 2, axis=1)
    ss_tot = np.sum((y_true - np.mean(y_true, axis=1, keepdims=True)) ** 2, axis=1)
    r2 = 1 - ss_res / (ss_tot + 1e-10)
    return r2

r2 = r2_score(y_true, y_pred)

print(f"\n  【R²决定系数】")
print(f"  均值: {np.mean(r2):.6f}")
print(f"  中位数: {np.median(r2):.6f}")

# ============================================================================
# 5. FWHM对比
# ============================================================================
print("\n[5/6] 计算FWHM对比...")

def calculate_fwhm(wavelengths, spectrum):
    """计算光谱的半高宽"""
    half_max = np.max(spectrum) / 2
    above_half = spectrum >= half_max
    
    if not np.any(above_half):
        return 0
    
    indices = np.where(above_half)[0]
    if len(indices) < 2:
        return 0
    
    fwhm = wavelengths[indices[-1]] - wavelengths[indices[0]]
    return fwhm

fwhm_true = []
fwhm_pred = []

for i in range(len(y_true)):
    fwhm_t = calculate_fwhm(target_wavelengths, y_true[i])
    fwhm_p = calculate_fwhm(target_wavelengths, y_pred[i])
    fwhm_true.append(fwhm_t)
    fwhm_pred.append(fwhm_p)

fwhm_true = np.array(fwhm_true)
fwhm_pred = np.array(fwhm_pred)
fwhm_error = np.abs(fwhm_pred - fwhm_true)

print(f"\n  【FWHM对比】")
print(f"  真实FWHM - 均值: {np.mean(fwhm_true):.2f} nm, 范围: {np.min(fwhm_true):.2f}-{np.max(fwhm_true):.2f} nm")
print(f"  预测FWHM - 均值: {np.mean(fwhm_pred):.2f} nm, 范围: {np.min(fwhm_pred):.2f}-{np.max(fwhm_pred):.2f} nm")
print(f"  FWHM误差 - 均值: {np.mean(fwhm_error):.2f} nm, 中位数: {np.median(fwhm_error):.2f} nm")

# ============================================================================
# 6. 综合统计和可视化
# ============================================================================
print("\n[6/6] 生成可视化...")

fig = plt.figure(figsize=(20, 16))

# 6.1 MSE分布
ax1 = fig.add_subplot(3, 3, 1)
ax1.hist(mse_per_sample, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
ax1.axvline(np.mean(mse_per_sample), color='red', linestyle='--', linewidth=2, label=f'Mean={np.mean(mse_per_sample):.4f}')
ax1.set_xlabel('MSE per Sample', fontsize=12)
ax1.set_ylabel('Count', fontsize=12)
ax1.set_title('MSE Distribution', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 6.2 MAE分布
ax2 = fig.add_subplot(3, 3, 2)
ax2.hist(mae_per_sample, bins=50, edgecolor='black', alpha=0.7, color='coral')
ax2.axvline(np.mean(mae_per_sample), color='red', linestyle='--', linewidth=2, label=f'Mean={np.mean(mae_per_sample):.4f}')
ax2.set_xlabel('MAE per Sample', fontsize=12)
ax2.set_ylabel('Count', fontsize=12)
ax2.set_title('MAE Distribution', fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 6.3 SAD分布
ax3 = fig.add_subplot(3, 3, 3)
ax3.hist(sad_degrees, bins=50, edgecolor='black', alpha=0.7, color='green')
ax3.axvline(np.mean(sad_degrees), color='red', linestyle='--', linewidth=2, label=f'Mean={np.mean(sad_degrees):.2f}°')
ax3.set_xlabel('SAD (degrees)', fontsize=12)
ax3.set_ylabel('Count', fontsize=12)
ax3.set_title('Spectral Angle Distance Distribution', fontsize=14, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 6.4 相关系数分布
ax4 = fig.add_subplot(3, 3, 4)
ax4.hist(corr, bins=50, edgecolor='black', alpha=0.7, color='purple')
ax4.axvline(np.mean(corr), color='red', linestyle='--', linewidth=2, label=f'Mean={np.mean(corr):.4f}')
ax4.set_xlabel('Pearson Correlation', fontsize=12)
ax4.set_ylabel('Count', fontsize=12)
ax4.set_title('Correlation Distribution', fontsize=14, fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3)

# 6.5 R²分布
ax5 = fig.add_subplot(3, 3, 5)
ax5.hist(r2, bins=50, edgecolor='black', alpha=0.7, color='orange')
ax5.axvline(np.mean(r2), color='red', linestyle='--', linewidth=2, label=f'Mean={np.mean(r2):.4f}')
ax5.set_xlabel('R² Score', fontsize=12)
ax5.set_ylabel('Count', fontsize=12)
ax5.set_title('R² Score Distribution', fontsize=14, fontweight='bold')
ax5.legend()
ax5.grid(True, alpha=0.3)

# 6.6 FWHM误差分布
ax6 = fig.add_subplot(3, 3, 6)
ax6.hist(fwhm_error, bins=50, edgecolor='black', alpha=0.7, color='brown')
ax6.axvline(np.mean(fwhm_error), color='red', linestyle='--', linewidth=2, label=f'Mean={np.mean(fwhm_error):.2f}nm')
ax6.set_xlabel('FWHM Error (nm)', fontsize=12)
ax6.set_ylabel('Count', fontsize=12)
ax6.set_title('FWHM Error Distribution', fontsize=14, fontweight='bold')
ax6.legend()
ax6.grid(True, alpha=0.3)

# 6.7 真实vs预测FWHM散点图
ax7 = fig.add_subplot(3, 3, 7)
ax7.scatter(fwhm_true, fwhm_pred, alpha=0.3, s=10)
ax7.plot([0, 60], [0, 60], 'r--', linewidth=2, label='Perfect')
ax7.set_xlabel('True FWHM (nm)', fontsize=12)
ax7.set_ylabel('Predicted FWHM (nm)', fontsize=12)
ax7.set_title('FWHM: True vs Predicted', fontsize=14, fontweight='bold')
ax7.legend()
ax7.grid(True, alpha=0.3)

# 6.8 相关系数 vs SAD
ax8 = fig.add_subplot(3, 3, 8)
ax8.scatter(corr, sad_degrees, alpha=0.3, s=10)
ax8.set_xlabel('Pearson Correlation', fontsize=12)
ax8.set_ylabel('SAD (degrees)', fontsize=12)
ax8.set_title('Correlation vs SAD', fontsize=14, fontweight='bold')
ax8.grid(True, alpha=0.3)

# 6.9 示例光谱对比
ax9 = fig.add_subplot(3, 3, 9)
sample_idx = np.random.randint(0, len(y_true))
ax9.plot(target_wavelengths, y_true[sample_idx], 'b-', linewidth=2, label='True', alpha=0.8)
ax9.plot(target_wavelengths, y_pred[sample_idx], 'r--', linewidth=2, label='Pred', alpha=0.8)
ax9.set_xlabel('Wavelength (nm)', fontsize=12)
ax9.set_ylabel('Intensity', fontsize=12)
ax9.set_title(f'Sample Spectrum\nCorr={corr[sample_idx]:.4f}, SAD={sad_degrees[sample_idx]:.2f}°', 
              fontsize=14, fontweight='bold')
ax9.legend()
ax9.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('curve_level_evaluation.png', dpi=150)
print("\n  已保存: curve_level_evaluation.png")

# ============================================================================
# 7. 分波段统计
# ============================================================================
print("\n" + "=" * 70)
print("分波段曲线指标统计")
print("=" * 70)

bands = [(1000, 1050), (1050, 1100), (1100, 1150), (1150, 1200), (1200, 1250), (1250, 1300)]

print("\n" + "-" * 80)
print(f"{'波段':<12} {'MSE均值':<12} {'MAE均值':<12} {'SAD(°)':<12} {'相关系数':<12} {'R²均值':<12}")
print("-" * 80)

for wl_min, wl_max in bands:
    mask = (peak_wavelengths_test >= wl_min) & (peak_wavelengths_test < wl_max)
    n = np.sum(mask)
    if n > 0:
        mse_band = np.mean(mse_per_sample[mask])
        mae_band = np.mean(mae_per_sample[mask])
        sad_band = np.mean(sad_degrees[mask])
        corr_band = np.mean(corr[mask])
        r2_band = np.mean(r2[mask])
        print(f"{wl_min}-{wl_max}nm  {mse_band:<12.6f} {mae_band:<12.6f} {sad_band:<12.4f} {corr_band:<12.6f} {r2_band:<12.6f}")

print("-" * 80)

# ============================================================================
# 8. 完整样本对比图
# ============================================================================
fig2, axes2 = plt.subplots(5, 5, figsize=(20, 20))
axes2 = axes2.flatten()

# 选择25个代表性样本
np.random.seed(456)
sample_indices = np.random.choice(len(y_true), 25, replace=False)

for i, idx in enumerate(sample_indices):
    ax = axes2[i]
    
    ax.plot(target_wavelengths, y_true[idx], 'b-', linewidth=2, label='True', alpha=0.8)
    ax.plot(target_wavelengths, y_pred[idx], 'r--', linewidth=2, label='Pred', alpha=0.8)
    
    # 计算指标
    mse = mse_per_sample[idx]
    mae = mae_per_sample[idx]
    sad_d = sad_degrees[idx]
    c = corr[idx]
    r = r2[idx]
    fwhm_e = fwhm_error[idx]
    
    ax.set_title(f'MSE={mse:.4f}\nSAD={sad_d:.2f}° Corr={c:.4f}', fontsize=9)
    ax.legend(fontsize=7, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1000, 1300)
    ax.set_ylim(0, 1.1)
    
    # 根据指标着色
    if c > 0.99:
        ax.set_facecolor('#e8f5e9')  # 绿色-很好
    elif c > 0.95:
        ax.set_facecolor('#fff3e0')  # 橙色-一般
    else:
        ax.set_facecolor('#ffebee')  # 红色-较差

plt.suptitle('25 Sample Comparisons with Curve-Level Metrics\nGreen=Excellent(>0.99), Orange=Good(>0.95), Red=Poor(<0.95)', 
             fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('curve_comparison_samples.png', dpi=150)
print("\n已保存: curve_comparison_samples.png")

# ============================================================================
# 9. 最终总结
# ============================================================================
print("\n" + "=" * 70)
print("曲线级别评估总结")
print("=" * 70)

print(f"""
╔══════════════════════════════════════════════════════════════════════╗
║                        曲线重建质量评估结果                           ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  【整体误差】                                                         ║
║    MSE  均值: {np.mean(mse_per_sample):.6f}     中位数: {np.median(mse_per_sample):.6f}              ║
║    MAE  均值: {np.mean(mae_per_sample):.6f}     中位数: {np.median(mae_per_sample):.6f}              ║
║                                                                      ║
║  【形状相似度】                                                       ║
║    光谱角距离(SAD): {np.mean(sad_degrees):.4f}°    中位数: {np.median(sad_degrees):.4f}°                    ║
║    Pearson相关: {np.mean(corr):.6f}    中位数: {np.median(corr):.6f}                        ║
║    R²决定系数: {np.mean(r2):.6f}    中位数: {np.median(r2):.6f}                        ║
║                                                                      ║
║  【峰形指标】                                                         ║
║    FWHM误差: {np.mean(fwhm_error):.2f} nm    中位数: {np.median(fwhm_error):.2f} nm                       ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝

指标说明:
  - MSE/MAE: 越接近0越好，表示重建光谱与真实光谱的数值差异
  - SAD: 越接近0°越好，表示光谱形状的相似度
  - 相关系数: 越接近1越好，表示曲线形状的相关性
  - R²: 越接近1越好，表示模型解释方差的比例
  - FWHM误差: 越接近0越好，表示峰宽重建的准确性
""")

# 计算各阈值下的达标率
print("曲线相似度阈值分析:")
print("-" * 40)

thresholds = [
    ("相关系数 > 0.99", np.sum(corr > 0.99) / len(corr) * 100),
    ("相关系数 > 0.95", np.sum(corr > 0.95) / len(corr) * 100),
    ("相关系数 > 0.90", np.sum(corr > 0.90) / len(corr) * 100),
    ("SAD < 1°", np.sum(sad_degrees < 1) / len(sad_degrees) * 100),
    ("SAD < 5°", np.sum(sad_degrees < 5) / len(sad_degrees) * 100),
    ("SAD < 10°", np.sum(sad_degrees < 10) / len(sad_degrees) * 100),
    ("R² > 0.95", np.sum(r2 > 0.95) / len(r2) * 100),
    ("R² > 0.90", np.sum(r2 > 0.90) / len(r2) * 100),
]

for name, pct in thresholds:
    print(f"  {name}: {pct:.1f}%")

print("\n" + "=" * 70)
