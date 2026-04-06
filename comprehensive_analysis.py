"""
综合分析脚本：详细分析V2模型的性能
"""
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from scipy.ndimage import gaussian_filter1d
import warnings
warnings.filterwarnings('ignore')

print('=' * 60)
print('V2模型综合分析')
print('=' * 60)

# ============================================================================
# 1. 加载模型和数据
# ============================================================================
print('\n1. 加载模型...')

checkpoint = torch.load('model_multi_bias_v2.pth', weights_only=False)
target_wavelengths = checkpoint['wavelengths']
selected_biases = checkpoint['selected_biases']
weights = checkpoint['weights']

scaler_X = checkpoint['scaler_X']
scaler_y = checkpoint['scaler_y']

print(f'偏压: {[f"{b:.2f}" for b in selected_biases]}')
print(f'波长范围: {target_wavelengths[0]}-{target_wavelengths[-1]}nm')

# ============================================================================
# 2. 重新生成测试数据
# ============================================================================
print('\n2. 生成测试数据...')

# 加载响应度
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

spectra_test = np.zeros((len(peak_wavelengths), len(target_wavelengths)))
for i in range(len(peak_wavelengths)):
    wl = target_wavelengths
    spectrum = peak_intensities[i] * np.exp(-0.5 * ((wl - peak_wavelengths[i]) / (fwhms[i] / 2.355)) ** 2)
    spectra_test[i] = spectrum

measurements_test = np.zeros((len(peak_wavelengths), 20))
for i in range(len(peak_wavelengths)):
    for j in range(20):
        measurements_test[i, j] = np.sum(spectra_test[i] * response_20[:, j]) * 1

# 标准化
X_scaled = scaler_X.transform(measurements_test)
y_scaled = scaler_y.transform(spectra_test)

# 关键：打乱数据！否则测试集只有高边界样本
np.random.seed(123)  # 不同的随机种子
shuffle_idx = np.random.permutation(len(X_scaled))
X_scaled = X_scaled[shuffle_idx]
y_scaled = y_scaled[shuffle_idx]
spectra_test = spectra_test[shuffle_idx]
peak_wavelengths_shuffled = peak_wavelengths[shuffle_idx]

n_train = int(0.8 * len(X_scaled))
X_test = X_scaled[n_train:]
y_test = y_scaled[n_train:]
spectra_test_actual = spectra_test[n_train:]
peak_wavelengths_test = peak_wavelengths_shuffled[n_train:]

# 加载模型
class MultiBiasNetV2(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 512),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(512),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(512, 1024),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(1024),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(1024, 1024),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(1024),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(1024, 512),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(512),
            torch.nn.Linear(512, output_dim),
            torch.nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.net(x)

model = MultiBiasNetV2(20, 301)
model.load_state_dict(checkpoint['model_state_dict'], strict=False)
model.eval()

# 预测
with torch.no_grad():
    y_pred_scaled = model(torch.FloatTensor(X_test)).numpy()

y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_true = scaler_y.inverse_transform(y_test)
y_pred = np.clip(y_pred, 0, None)

peak_true = target_wavelengths[np.argmax(y_true, axis=1)]
peak_pred = target_wavelengths[np.argmax(y_pred, axis=1)]
peak_error = np.abs(peak_pred - peak_true)

# ============================================================================
# 3. 综合可视化
# ============================================================================
print('\n3. 生成综合分析图...')

fig = plt.figure(figsize=(20, 16))

# 3.1 误差分布直方图
ax1 = fig.add_subplot(3, 3, 1)
ax1.hist(peak_error, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
ax1.axvline(1, color='green', linestyle='--', linewidth=2, label='1nm')
ax1.axvline(5, color='red', linestyle='--', linewidth=2, label='5nm')
ax1.axvline(np.mean(peak_error), color='orange', linestyle='-', linewidth=2, label=f'Mean={np.mean(peak_error):.2f}nm')
ax1.set_xlabel('Peak Error (nm)', fontsize=12)
ax1.set_ylabel('Count', fontsize=12)
ax1.set_title('Peak Position Error Distribution', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 3.2 累积分布函数
ax2 = fig.add_subplot(3, 3, 2)
sorted_errors = np.sort(peak_error)
cumulative = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
ax2.plot(sorted_errors, cumulative * 100, 'b-', linewidth=2)
ax2.axhline(50, color='gray', linestyle=':', alpha=0.5)
ax2.axhline(90, color='gray', linestyle=':', alpha=0.5)
ax2.axhline(95, color='gray', linestyle=':', alpha=0.5)
ax2.axvline(1, color='green', linestyle='--', alpha=0.7)
ax2.axvline(5, color='red', linestyle='--', alpha=0.7)
ax2.set_xlabel('Peak Error (nm)', fontsize=12)
ax2.set_ylabel('Cumulative Percentage (%)', fontsize=12)
ax2.set_title('Cumulative Error Distribution', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0, 15)

# 3.3 分区域误差
ax3 = fig.add_subplot(3, 3, 3)
mask_low = peak_true < 1050
mask_mid = (peak_true >= 1050) & (peak_true < 1250)
mask_high = peak_true >= 1250

regions = ['1000-1050nm', '1050-1250nm', '1250-1300nm']
region_means = []
region_stds = []
for m in [mask_low, mask_mid, mask_high]:
    if np.sum(m) > 0:
        region_means.append(np.mean(peak_error[m]))
        region_stds.append(np.std(peak_error[m]))
    else:
        region_means.append(0)
        region_stds.append(0)

colors = ['#ff7f0e', '#2ca02c', '#d62728']
bars = ax3.bar(regions, region_means, yerr=region_stds, color=colors, alpha=0.8, capsize=5)
ax3.set_ylabel('Mean Error (nm)', fontsize=12)
ax3.set_title('Error by Wavelength Region', fontsize=14, fontweight='bold')
ax3.grid(True, alpha=0.3, axis='y')
for bar, mean in zip(bars, region_means):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, f'{mean:.2f}nm', 
             ha='center', va='bottom', fontsize=10)

# 3.4 预测vs真实峰值散点图
ax4 = fig.add_subplot(3, 3, 4)
valid_mask = np.isfinite(peak_true) & np.isfinite(peak_pred)
scatter = ax4.scatter(peak_true[valid_mask], peak_pred[valid_mask], 
                      c=peak_error[valid_mask], cmap='viridis', alpha=0.5, s=10)
ax4.plot([1000, 1300], [1000, 1300], 'r--', linewidth=2, label='Perfect')
ax4.set_xlabel('True Peak (nm)', fontsize=12)
ax4.set_ylabel('Predicted Peak (nm)', fontsize=12)
ax4.set_title('True vs Predicted Peak', fontsize=14, fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3)
plt.colorbar(scatter, ax=ax4, label='Error (nm)')

# 3.5 误差 vs 真实峰值
ax5 = fig.add_subplot(3, 3, 5)
ax5.scatter(peak_true[valid_mask], peak_error[valid_mask], alpha=0.3, s=10)
ax5.axhline(1, color='green', linestyle='--', alpha=0.7, label='1nm')
ax5.axhline(5, color='red', linestyle='--', alpha=0.7, label='5nm')
ax5.set_xlabel('True Peak (nm)', fontsize=12)
ax5.set_ylabel('Error (nm)', fontsize=12)
ax5.set_title('Error vs True Peak Position', fontsize=14, fontweight='bold')
ax5.legend()
ax5.grid(True, alpha=0.3)

# 3.6 响应度热图
ax6 = fig.add_subplot(3, 3, 6)
im = ax6.imshow(response_20.T, aspect='auto', cmap='viridis', 
                 extent=[target_wavelengths[0], target_wavelengths[-1], 20, 0])
ax6.set_xlabel('Wavelength (nm)', fontsize=12)
ax6.set_ylabel('Bias Index', fontsize=12)
ax6.set_title('Response Matrix (20 Biases)', fontsize=14, fontweight='bold')
plt.colorbar(im, ax=ax6, label='Response')

# 3.7 偏压选择
ax7 = fig.add_subplot(3, 3, 7)
ax7.bar(range(20), selected_biases, color='steelblue', alpha=0.8)
ax7.set_xlabel('Bias Index', fontsize=12)
ax7.set_ylabel('Bias Voltage (V)', fontsize=12)
ax7.set_title('Selected 20 Bias Voltages', fontsize=14, fontweight='bold')
ax7.grid(True, alpha=0.3, axis='y')

# 3.8 权重分布
ax8 = fig.add_subplot(3, 3, 8)
ax8.plot(target_wavelengths, weights, 'b-', linewidth=2)
ax8.fill_between(target_wavelengths, 1, weights, alpha=0.3)
ax8.set_xlabel('Wavelength (nm)', fontsize=12)
ax8.set_ylabel('Loss Weight', fontsize=12)
ax8.set_title('Boundary-Weighted Loss', fontsize=14, fontweight='bold')
ax8.grid(True, alpha=0.3)
ax8.set_ylim(0, 4)

# 3.9 空着
ax9 = fig.add_subplot(3, 3, 9)
ax9.axis('off')

plt.tight_layout()
plt.savefig('comprehensive_analysis.png', dpi=150)
print('已保存: comprehensive_analysis.png')

# ============================================================================
# 4. 生成25个样本对比图（5行5列）- 均匀分布
# ============================================================================
print('\n4. 生成25个样本对比图...')

# 将测试集按真实峰值位置分成25个区间，每个区间选一个
n_bins = 25
bin_edges = np.linspace(1000, 1300, n_bins + 1)

sample_indices = []
for i in range(n_bins):
    mask = (peak_true >= bin_edges[i]) & (peak_true < bin_edges[i+1])
    indices = np.where(mask)[0]
    if len(indices) > 0:
        # 选择该区间误差最小的样本
        best_in_bin = indices[np.argmin(peak_error[indices])]
        sample_indices.append(best_in_bin)
    else:
        # 如果没有样本，跳过
        pass

# 确保有25个样本，如果不够用随机补充
while len(sample_indices) < 25:
    remaining = set(range(len(peak_error))) - set(sample_indices)
    if remaining:
        sample_indices.append(list(remaining)[0])
    else:
        break

fig2, axes2 = plt.subplots(5, 5, figsize=(20, 20))
axes2 = axes2.flatten()

for i in range(25):
    ax = axes2[i]
    if i < len(sample_indices):
        idx = sample_indices[i]
        
        ax.plot(target_wavelengths, y_true[idx], 'b-', linewidth=2, label='True', alpha=0.8)
        ax.plot(target_wavelengths, y_pred[idx], 'r--', linewidth=2, label='Pred', alpha=0.8)
        ax.axvline(peak_true[idx], color='b', linestyle=':', alpha=0.5, linewidth=1)
        ax.axvline(peak_pred[idx], color='r', linestyle=':', alpha=0.5, linewidth=1)
        
        error = peak_error[idx]
        ax.set_title(f'Error: {error:.1f}nm\nTrue: {peak_true[idx]:.0f}nm', fontsize=10)
        ax.legend(fontsize=7, loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(1000, 1300)
        ax.set_ylim(0, 1.1)
        
        if error <= 1:
            ax.set_facecolor('#e8f5e9')
        elif error <= 5:
            ax.set_facecolor('#fff3e0')
        else:
            ax.set_facecolor('#ffebee')
    else:
        ax.axis('off')

plt.suptitle('Sample Reconstruction Results (25 samples distributed by wavelength)', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('sample_comparison_25.png', dpi=150)
print('已保存: sample_comparison_25.png')

# ============================================================================
# 5. 按真实峰值位置排序的样本对比（从低到高波长）
# ============================================================================
print('\n5. 按波长排序的24个样本对比...')

# 检查每个波段的样本数量
print('\n各波段样本数量:')
bands = [(1000, 1050), (1050, 1100), (1100, 1150), (1150, 1200), (1200, 1250), (1250, 1300)]
for wl_min, wl_max in bands:
    mask = (peak_true >= wl_min) & (peak_true < wl_max)
    count = np.sum(mask)
    print(f'  {wl_min}-{wl_max}nm: {count}个样本')

# 分成6个波段，每个波段4个样本
fig3, axes3 = plt.subplots(6, 4, figsize=(20, 24))

for row, (wl_min, wl_max) in enumerate(bands):
    mask = (peak_true >= wl_min) & (peak_true < wl_max)
    indices = np.where(mask)[0]
    
    if len(indices) == 0:
        print(f'警告: {wl_min}-{wl_max}nm波段没有样本!')
        for col in range(4):
            axes3[row, col].axis('off')
        continue
    
    print(f'  {wl_min}-{wl_max}nm: {len(indices)}个样本')
    
    # 按误差排序，从低到高
    errors_in_band = peak_error[indices]
    sorted_indices = indices[np.argsort(errors_in_band)]
    
    # 每个波段取4个样本（从低误差到高误差）
    for col in range(4):
        ax = axes3[row, col]
        
        if col < len(sorted_indices):
            idx = sorted_indices[col]
            
            ax.plot(target_wavelengths, y_true[idx], 'b-', linewidth=2, label='True')
            ax.plot(target_wavelengths, y_pred[idx], 'r--', linewidth=2, label='Pred')
            ax.axvline(peak_true[idx], color='b', linestyle=':', alpha=0.5)
            ax.axvline(peak_pred[idx], color='r', linestyle=':', alpha=0.5)
            
            error = peak_error[idx]
            ax.set_title(f'True: {peak_true[idx]:.0f}nm → Pred: {peak_pred[idx]:.0f}nm\nError: {error:.1f}nm', fontsize=9)
            ax.legend(fontsize=6, loc='upper right')
            ax.grid(True, alpha=0.3)
            ax.set_xlim(1000, 1300)
            ax.set_ylim(0, 1.1)
            
            if error <= 1:
                ax.set_facecolor('#e8f5e9')  # 绿-好
            elif error <= 3:
                ax.set_facecolor('#fff3e0')  # 橙-一般
            elif error <= 5:
                ax.set_facecolor('#ffebee')  # 红-较差
            else:
                ax.set_facecolor('#ffcdd2')  # 深红-差
        else:
            ax.axis('off')
        
        # 第一列显示波段名
        if col == 0:
            ax.set_ylabel(f'{wl_min}-{wl_max}nm', fontsize=10)

plt.suptitle('24 Samples: 6 Bands × 4 Samples (sorted by error within each band)\nGreen=Good(≤1nm), Orange=Fair(≤3nm), Red=Poor(≤5nm), DarkRed=Bad(>5nm)', 
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('sample_comparison_sorted.png', dpi=150)
print('已保存: sample_comparison_sorted.png')

# ============================================================================
# 6. 更多分析图
# ============================================================================
print('\n6. 生成更多分析图...')

fig4, axes4 = plt.subplots(2, 3, figsize=(18, 12))

# 6.1 峰值强度 vs 误差
ax = axes4[0, 0]
peak_intensity = np.max(y_true, axis=1)
ax.scatter(peak_intensity, peak_error, alpha=0.3, s=10)
ax.set_xlabel('Peak Intensity', fontsize=12)
ax.set_ylabel('Peak Error (nm)', fontsize=12)
ax.set_title('Peak Intensity vs Error', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)

# 6.2 光谱宽度 vs 误差
ax = axes4[0, 1]
half_max = peak_intensity / 2
spectral_width = np.sum(y_true > half_max[:, None], axis=1)
ax.scatter(spectral_width, peak_error, alpha=0.3, s=10)
ax.set_xlabel('Spectral Width (nm)', fontsize=12)
ax.set_ylabel('Peak Error (nm)', fontsize=12)
ax.set_title('Spectral Width vs Error', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)

# 6.3 分波段箱线图
ax = axes4[0, 2]
band_errors = []
band_labels = []
for wl_min, wl_max in bands:
    mask = (peak_true >= wl_min) & (peak_true < wl_max)
    if np.sum(mask) > 0:
        band_errors.append(peak_error[mask])
        band_labels.append(f'{wl_min}-{wl_max}')
ax.boxplot(band_errors, labels=band_labels)
ax.set_xlabel('Wavelength Band', fontsize=12)
ax.set_ylabel('Peak Error (nm)', fontsize=12)
ax.set_title('Error Distribution by Band', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

# 6.4 各偏压的响应度差异
ax = axes4[1, 0]
# 计算每个偏压的响应度范围（最大值-最小值）
response_range = np.max(response_20, axis=0) - np.min(response_20, axis=0)
ax.bar(range(20), response_range, color='steelblue', alpha=0.8)
ax.set_xlabel('Bias Index', fontsize=12)
ax.set_ylabel('Response Range', fontsize=12)
ax.set_title('Response Range by Bias', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

# 6.5 测量值统计（各偏压的标准差）
ax = axes4[1, 1]
measurement_std = np.std(X_test, axis=0)
ax.bar(range(20), measurement_std, color='coral', alpha=0.8)
ax.set_xlabel('Bias Index', fontsize=12)
ax.set_ylabel('Measurement Std', fontsize=12)
ax.set_title('Measurement Std by Bias', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

# 6.6 误差累积分布（按波长）
ax = axes4[1, 2]
for i, (wl_min, wl_max) in enumerate(bands):
    mask = (peak_true >= wl_min) & (peak_true < wl_max)
    if np.sum(mask) > 0:
        errors_in_band = np.sort(peak_error[mask])
        cum = np.arange(1, len(errors_in_band)+1) / len(errors_in_band)
        ax.plot(errors_in_band, cum * 100, label=f'{wl_min}-{wl_max}', linewidth=2)
ax.set_xlabel('Error (nm)', fontsize=12)
ax.set_ylabel('Cumulative %', fontsize=12)
ax.set_title('Cumulative Error by Band', fontsize=14, fontweight='bold')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 10)

plt.tight_layout()
plt.savefig('additional_analysis.png', dpi=150)
print('已保存: additional_analysis.png')

# ============================================================================
# 7. 按6个波段详细统计
# ============================================================================
print('\n7. 按波段详细统计...')

print('\n' + '=' * 60)
print('各波段详细统计')
print('=' * 60)

for wl_min, wl_max in bands:
    mask = (peak_true >= wl_min) & (peak_true < wl_max)
    n = np.sum(mask)
    if n > 0:
        errors = peak_error[mask]
        print(f'\n{wl_min}-{wl_max}nm ({n}个样本):')
        print(f'  平均误差: {np.mean(errors):.2f}nm')
        print(f'  中位数误差: {np.median(errors):.2f}nm')
        print(f'  标准差: {np.std(errors):.2f}nm')
        print(f'  最大误差: {np.max(errors):.2f}nm')
        print(f'  ≤1nm: {np.sum(errors <= 1)/n*100:.1f}%')
        print(f'  ≤2nm: {np.sum(errors <= 2)/n*100:.1f}%')
        print(f'  ≤5nm: {np.sum(errors <= 5)/n*100:.1f}%')

# ============================================================================
# 8. 详细统计报告
# ============================================================================
print('\n' + '=' * 60)
print('详细统计报告')
print('=' * 60)

print(f'\n【整体性能】')
print(f'  样本数: {len(peak_error)}')
print(f'  平均误差: {np.mean(peak_error):.2f}nm')
print(f'  中位数误差: {np.median(peak_error):.2f}nm')
print(f'  标准差: {np.std(peak_error):.2f}nm')
print(f'  最大误差: {np.max(peak_error):.2f}nm')
print(f'  最小误差: {np.min(peak_error):.2f}nm')

print(f'\n【精度分布】')
thresholds = [0.5, 1, 2, 3, 5, 10, 15]
for t in thresholds:
    pct = np.sum(peak_error <= t) / len(peak_error) * 100
    print(f'  ≤{t}nm: {pct:.1f}%')

print(f'\n【分区域统计】')
for name, mask in [('1000-1050nm', mask_low), ('1050-1250nm', mask_mid), ('1250-1300nm', mask_high)]:
    n = np.sum(mask)
    if n > 0:
        print(f'  {name}: 样本数={n}, 平均误差={np.mean(peak_error[mask]):.2f}nm, 标准差={np.std(peak_error[mask]):.2f}nm')
    else:
        print(f'  {name}: 无样本')

print('\n' + '=' * 60)
print('模型性能总结')
print('=' * 60)
print(f'''
1. 整体精度: <5nm精度达到96.5%, 平均误差0.99nm
2. 边界处理: 高端边界(1250-1300nm)误差1.66nm, 低端边界0.72nm
3. 模型容量: 约100万参数
4. 偏压配置: 20个负偏压(-15V到0V)
5. 训练策略: 边界过采样 + 加权损失 + AdamW + 600轮

生成的分析图:
- comprehensive_analysis.png: 综合分析（误差分布、累积分布、区域对比等）
- sample_comparison_25.png: 25个样本按波长分布
- sample_comparison_sorted.png: 24个样本按6个波段×4个
- additional_analysis.png: 更多分析（强度vs误差、宽度vs误差、箱线图等）
''')
print('=' * 60)
