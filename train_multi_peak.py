"""
多峰光谱重建训练 V3
支持2-5个峰的光谱生成

改进点：
1. 多峰生成（单峰40%、双峰25%、三峰20%、四峰10%、五峰5%）
2. 峰间距约束（避免融合或超出范围）
3. 更灵活的峰参数范围
4. 峰对峰匹配的评估指标
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

print('=' * 60)
print('多峰光谱训练 V3')
print('=' * 60)

# ============================================================================
# 辅助函数
# ============================================================================

def gaussian(wavelengths, center, fwhm):
    """生成高斯峰"""
    sigma = fwhm / 2.355
    return np.exp(-0.5 * ((wavelengths - center) / sigma) ** 2)

def enforce_min_separation(peaks, min_sep=25):
    """
    确保峰之间有最小间隔
    peaks: [(pos, int, fwhm), ...]
    返回: 调整后的peaks列表
    """
    if len(peaks) <= 1:
        return peaks
    
    peaks = sorted(peaks, key=lambda x: x[0])  # 按位置排序
    
    for i in range(len(peaks) - 1):
        p1, p2 = peaks[i], peaks[i + 1]
        required_sep = max(min_sep, p1[2] * 0.8, p2[2] * 0.8)
        
        current_sep = p2[0] - p1[0]
        if current_sep < required_sep:
            # 调整p2的位置
            new_pos = p1[0] + required_sep
            peaks[i + 1] = (new_pos, p2[1], p2[2])
    
    return peaks

def generate_multi_peak_spectrum(wavelengths, n_peaks, config):
    """
    生成多峰光谱
    
    参数:
        wavelengths: 波长数组
        n_peaks: 峰的数量 (1-5)
        config: 配置字典
    
    返回:
        spectrum: 光谱数组
        peak_info: [(pos, int, fwhm), ...]
    """
    wl_min, wl_max = wavelengths[0], wavelengths[-1]
    margin = 30  # 边距
    
    peaks = []
    
    for _ in range(n_peaks):
        # 随机参数
        pos = np.random.uniform(wl_min + margin, wl_max - margin)
        intensity = np.random.uniform(*config['intensity_range'])
        fwhm = np.random.uniform(*config['fwhm_range'])
        
        peaks.append((pos, intensity, fwhm))
    
    # 确保峰之间有最小间隔
    peaks = enforce_min_separation(peaks, min_sep=config['min_separation'])
    
    # 叠加所有峰
    spectrum = np.zeros_like(wavelengths, dtype=float)
    for pos, intensity, fwhm in peaks:
        spectrum += intensity * gaussian(wavelengths, pos, fwhm)
    
    return spectrum, peaks

def find_peaks_scipy(spectrum, wavelengths, min_prominence=0.1):
    """使用scipy找峰"""
    from scipy.signal import find_peaks
    peaks, properties = find_peaks(spectrum, prominence=min_prominence)
    peak_positions = wavelengths[peaks]
    peak_heights = spectrum[peaks]
    return peak_positions, peak_heights

def match_peaks(true_peaks, pred_peaks):
    """
    匹配真实峰和预测峰（匈牙利算法）
    返回配对后的误差列表
    """
    if len(true_peaks) == 0 or len(pred_peaks) == 0:
        return []
    
    # 构建成本矩阵
    cost_matrix = np.zeros((len(true_peaks), len(pred_peaks)))
    for i, tp in enumerate(true_peaks):
        for j, pp in enumerate(pred_peaks):
            cost_matrix[i, j] = abs(tp - pp)
    
    # 匈牙利算法找最优匹配
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    # 计算配对误差
    errors = []
    for r, c in zip(row_ind, col_ind):
        errors.append(cost_matrix[r, c])
    
    return errors

def calculate_peak_metrics(y_true, y_pred, wavelengths):
    """
    计算多峰相关的评估指标
    
    返回:
        peak_position_error: 峰位置误差（配对后）
        detected_vs_true_ratio: 检测到的峰数/真实峰数
    """
    # 找峰
    true_peak_pos, _ = find_peaks_scipy(y_true, wavelengths)
    pred_peak_pos, _ = find_peaks_scipy(y_pred, wavelengths)
    
    if len(true_peak_pos) == 0:
        return None, None
    
    # 匹配
    errors = match_peaks(true_peak_pos, pred_peak_pos)
    
    if len(errors) == 0:
        return None, None
    
    detection_ratio = len(pred_peak_pos) / len(true_peak_pos) if len(true_peak_pos) > 0 else 0
    
    return np.mean(errors), detection_ratio

# ============================================================================
# 1. 加载响应度
# ============================================================================
print('\n[1/8] 加载响应度...')

excel_path = r'D:\desktop\try\响应度矩阵.xlsx'
df = pd.read_excel(excel_path)
wavelengths_resp = df.iloc[:, 0].values
response_matrix_full = df.iloc[:, 1:].values

all_biases = np.linspace(-15, 0, 65)
indices = np.linspace(0, 64, 20, dtype=int)
selected_biases = all_biases[indices]

# 插值响应度
target_wavelengths = np.arange(1000, 1301, 1)
response_20 = np.zeros((len(target_wavelengths), 20))

from scipy.interpolate import UnivariateSpline
from scipy.ndimage import gaussian_filter1d

for j, idx in enumerate(indices):
    orig_resp = response_matrix_full[:, idx]
    spline = UnivariateSpline(wavelengths_resp, orig_resp, s=0.001, k=3)
    interpolated = spline(target_wavelengths)
    smoothed = gaussian_filter1d(interpolated, sigma=1.5)
    response_20[:, j] = smoothed

print(f'  响应度矩阵: {response_20.shape}')

# ============================================================================
# 2. 多峰数据生成配置
# ============================================================================
print('\n[2/8] 配置多峰数据生成...')

# 数据配比
n_single = 4000   # 40% 单峰
n_double = 2500   # 25% 双峰
n_triple = 2000   # 20% 三峰
n_quad = 1000    # 10% 四峰
n_penta = 500    # 5% 五峰
n_total = n_single + n_double + n_triple + n_quad + n_penta

# 峰参数配置
peak_config = {
    'intensity_range': (0.2, 1.0),   # 允许更弱的峰
    'fwhm_range': (10, 60),           # 允许更窄/更宽
    'min_separation': 30              # 最小间隔30nm
}

print(f'  数据配比:')
print(f'    单峰: {n_single} ({n_single/n_total*100:.0f}%)')
print(f'    双峰: {n_double} ({n_double/n_total*100:.0f}%)')
print(f'    三峰: {n_triple} ({n_triple/n_total*100:.0f}%)')
print(f'    四峰: {n_quad} ({n_quad/n_total*100:.0f}%)')
print(f'    五峰: {n_penta} ({n_penta/n_total*100:.0f}%)')
print(f'    总计: {n_total}')

# ============================================================================
# 3. 生成多峰光谱数据
# ============================================================================
print('\n[3/8] 生成多峰光谱数据...')

np.random.seed(42)

spectra_list = []
peak_counts = []
peak_infos = []  # 存储每个光谱的峰信息

# 生成单峰
print('  生成单峰...')
for i in range(n_single):
    spectrum, peaks = generate_multi_peak_spectrum(target_wavelengths, 1, peak_config)
    spectra_list.append(spectrum)
    peak_counts.append(1)
    peak_infos.append(peaks)

# 生成双峰
print('  生成双峰...')
for i in range(n_double):
    spectrum, peaks = generate_multi_peak_spectrum(target_wavelengths, 2, peak_config)
    spectra_list.append(spectrum)
    peak_counts.append(2)
    peak_infos.append(peaks)

# 生成三峰
print('  生成三峰...')
for i in range(n_triple):
    spectrum, peaks = generate_multi_peak_spectrum(target_wavelengths, 3, peak_config)
    spectra_list.append(spectrum)
    peak_counts.append(3)
    peak_infos.append(peaks)

# 生成四峰
print('  生成四峰...')
for i in range(n_quad):
    spectrum, peaks = generate_multi_peak_spectrum(target_wavelengths, 4, peak_config)
    spectra_list.append(spectrum)
    peak_counts.append(4)
    peak_infos.append(peaks)

# 生成五峰
print('  生成五峰...')
for i in range(n_penta):
    spectrum, peaks = generate_multi_peak_spectrum(target_wavelengths, 5, peak_config)
    spectra_list.append(spectrum)
    peak_counts.append(5)
    peak_infos.append(peaks)

spectra_multi = np.array(spectra_list)
peak_counts = np.array(peak_counts)

print(f'  光谱形状: {spectra_multi.shape}')

# 统计峰数分布
unique, counts = np.unique(peak_counts, return_counts=True)
print(f'  峰数分布: {dict(zip(unique, counts))}')

# ============================================================================
# 4. 计算测量值
# ============================================================================
print('\n[4/8] 计算测量值...')

measurements_multi = np.zeros((n_total, 20))
for i in range(n_total):
    for j in range(20):
        measurements_multi[i, j] = np.sum(spectra_multi[i] * response_20[:, j])

print(f'  测量值形状: {measurements_multi.shape}')

# ============================================================================
# 5. 数据准备
# ============================================================================
print('\n[5/8] 数据准备...')

scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(measurements_multi)

scaler_y = MinMaxScaler()
y_scaled = scaler_y.fit_transform(spectra_multi)

# 边界权重（与V2相同）
weights = np.ones(len(target_wavelengths))
edge_low = (target_wavelengths <= 1050)
edge_high = (target_wavelengths >= 1250)
weights[edge_low] = 3.0
weights[edge_high] = 3.0

# 分割数据
X_train, X_test, y_train, y_test, peak_counts_train, peak_counts_test = train_test_split(
    X_scaled, y_scaled, peak_counts, test_size=0.2, random_state=42
)

# 也分割峰信息
_, _, peak_infos_train, peak_infos_test = train_test_split(
    np.arange(n_total), peak_infos, test_size=0.2, random_state=42
)

X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.FloatTensor(y_train)
X_test_tensor = torch.FloatTensor(X_test)
y_test_tensor = torch.FloatTensor(y_test)
weights_tensor = torch.FloatTensor(weights)

print(f'  训练集: {X_train.shape[0]}条')
print(f'  测试集: {X_test.shape[0]}条')

# ============================================================================
# 6. 定义模型（与V2相同架构）
# ============================================================================
print('\n[6/8] 定义模型...')

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
n_params = sum(p.numel() for p in model.parameters())
print(f'  参数量: {n_params:,}')

# ============================================================================
# 7. 训练
# ============================================================================
print('\n[7/8] 训练...')

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

def weighted_mse_loss(pred, target, weights):
    return torch.mean(weights * (pred - target) ** 2)

optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2)

epochs = 600
train_losses = []
test_losses = []
best_test_loss = float('inf')
best_model_state = None

for epoch in range(epochs):
    model.train()
    train_loss = 0
    
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        pred = model(batch_X)
        loss = weighted_mse_loss(pred, batch_y, weights_tensor)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        train_loss += loss.item()
    
    scheduler.step()
    train_loss /= len(train_loader)
    train_losses.append(train_loss)
    
    model.eval()
    with torch.no_grad():
        test_pred = model(X_test_tensor)
        test_loss = torch.mean((test_pred - y_test_tensor) ** 2).item()
    test_losses.append(test_loss)
    
    if test_loss < best_test_loss:
        best_test_loss = test_loss
        best_model_state = model.state_dict().copy()
    
    if (epoch + 1) % 100 == 0:
        print(f'  Epoch {epoch+1}, Train: {train_loss:.6f}, Test: {test_loss:.6f}')

model.load_state_dict(best_model_state)
print(f'  最佳损失: {best_test_loss:.6f}')

# ============================================================================
# 8. 评估（多峰专用）
# ============================================================================
print('\n[8/8] 多峰评估...')

model.eval()
with torch.no_grad():
    y_pred_scaled = model(X_test_tensor).numpy()

y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_true = scaler_y.inverse_transform(y_test)
y_pred = np.clip(y_pred, 0, None)

# 8.1 曲线级别指标
from scipy.spatial.distance import cosine

mse_per_sample = np.mean((y_pred - y_true) ** 2, axis=1)
mae_per_sample = np.mean(np.abs(y_pred - y_true), axis=1)

corr_per_sample = []
for i in range(len(y_true)):
    c = np.corrcoef(y_true[i], y_pred[i])[0, 1]
    corr_per_sample.append(c)
corr_per_sample = np.array(corr_per_sample)

print('\n【曲线级别指标】')
print(f'  MSE 均值: {np.mean(mse_per_sample):.6f}')
print(f'  MAE 均值: {np.mean(mae_per_sample):.6f}')
print(f'  相关系数 均值: {np.mean(corr_per_sample):.4f}')

# 8.2 峰检测和匹配
print('\n【峰检测与匹配】')

peak_position_errors = []
detection_ratios = []

for i in range(len(y_true)):
    true_peak_pos, _ = find_peaks_scipy(y_true[i], target_wavelengths)
    pred_peak_pos, _ = find_peaks_scipy(y_pred[i], target_wavelengths)
    
    if len(true_peak_pos) > 0:
        errors = match_peaks(true_peak_pos, pred_peak_pos)
        if len(errors) > 0:
            peak_position_errors.extend(errors)
        
        ratio = len(pred_peak_pos) / len(true_peak_pos)
        detection_ratios.append(ratio)

peak_position_errors = np.array(peak_position_errors)
detection_ratios = np.array(detection_ratios)

print(f'  峰位置误差: {np.mean(peak_position_errors):.2f} nm (中位数: {np.median(peak_position_errors):.2f} nm)')
print(f'  峰数检测比: {np.mean(detection_ratios):.2f} (理想=1.0)')

# 8.3 按峰数分组的统计
print('\n【按峰数分组统计】')
print('-' * 60)
print(f'{"峰数":<8} {"样本数":<10} {"曲线MSE":<12} {"相关系数":<12} {"峰误差(nm)":<12}')
print('-' * 60)

for n_peaks in range(1, 6):
    mask = peak_counts_test == n_peaks
    indices = np.where(mask)[0]
    
    if len(indices) > 0:
        mse_n = np.mean(mse_per_sample[indices])
        corr_n = np.mean(corr_per_sample[indices])
        
        # 计算该组的峰误差
        errors_n = []
        for idx in indices:
            true_peak_pos, _ = find_peaks_scipy(y_true[idx], target_wavelengths)
            pred_peak_pos, _ = find_peaks_scipy(y_pred[idx], target_wavelengths)
            if len(true_peak_pos) > 0:
                e = match_peaks(true_peak_pos, pred_peak_pos)
                if len(e) > 0:
                    errors_n.extend(e)
        
        peak_err_str = f'{np.mean(errors_n):.2f}' if len(errors_n) > 0 else 'N/A'
        print(f'{n_peaks:<8} {len(indices):<10} {mse_n:<12.6f} {corr_n:<12.4f} {peak_err_str:<12}')

print('-' * 60)

# 8.4 可视化
print('\n生成可视化...')

fig, axes = plt.subplots(3, 3, figsize=(18, 15))

# 训练曲线
ax = axes[0, 0]
ax.plot(train_losses, label='Train', alpha=0.8)
ax.plot(test_losses, label='Test', alpha=0.8)
ax.set_title('Training Curves')
ax.legend()
ax.grid(True, alpha=0.3)

# MSE分布
ax = axes[0, 1]
ax.hist(mse_per_sample, bins=50, edgecolor='black', alpha=0.7)
ax.axvline(np.mean(mse_per_sample), color='red', linestyle='--', label=f'Mean={np.mean(mse_per_sample):.4f}')
ax.set_title('MSE Distribution')
ax.legend()
ax.grid(True, alpha=0.3)

# 相关系数分布
ax = axes[0, 2]
ax.hist(corr_per_sample, bins=50, edgecolor='black', alpha=0.7)
ax.axvline(np.mean(corr_per_sample), color='red', linestyle='--', label=f'Mean={np.mean(corr_per_sample):.4f}')
ax.set_title('Correlation Distribution')
ax.legend()
ax.grid(True, alpha=0.3)

# 峰误差分布
ax = axes[1, 0]
if len(peak_position_errors) > 0:
    ax.hist(peak_position_errors, bins=50, edgecolor='black', alpha=0.7)
    ax.axvline(np.mean(peak_position_errors), color='red', linestyle='--', label=f'Mean={np.mean(peak_position_errors):.2f}nm')
ax.set_title('Peak Position Error Distribution')
ax.legend()
ax.grid(True, alpha=0.3)

# 峰检测比分布
ax = axes[1, 1]
ax.hist(detection_ratios, bins=30, edgecolor='black', alpha=0.7)
ax.axvline(1.0, color='green', linestyle='--', label='Ideal=1.0')
ax.set_title('Peak Detection Ratio')
ax.legend()
ax.grid(True, alpha=0.3)

# 按峰数的曲线指标
ax = axes[1, 2]
means = []
stds = []
labels = []
for n in range(1, 6):
    mask = peak_counts_test == n
    if np.sum(mask) > 0:
        means.append(np.mean(corr_per_sample[mask]))
        stds.append(np.std(corr_per_sample[mask]))
        labels.append(str(n))
    else:
        means.append(0)
        stds.append(0)
        labels.append(str(n))

ax.bar(labels, means, yerr=stds, alpha=0.7, capsize=5)
ax.set_xlabel('Number of Peaks')
ax.set_ylabel('Mean Correlation')
ax.set_title('Correlation by Peak Count')
ax.grid(True, alpha=0.3, axis='y')

# 样本对比 - 单峰
ax = axes[2, 0]
single_idx = np.where(peak_counts_test == 1)[0][0]
ax.plot(target_wavelengths, y_true[single_idx], 'b-', label='True', alpha=0.8)
ax.plot(target_wavelengths, y_pred[single_idx], 'r--', label='Pred', alpha=0.8)
ax.set_title(f'Single Peak Sample\nCorr={corr_per_sample[single_idx]:.4f}')
ax.legend()
ax.grid(True, alpha=0.3)

# 样本对比 - 双峰
ax = axes[2, 1]
double_idx = np.where(peak_counts_test == 2)[0][0]
ax.plot(target_wavelengths, y_true[double_idx], 'b-', label='True', alpha=0.8)
ax.plot(target_wavelengths, y_pred[double_idx], 'r--', label='Pred', alpha=0.8)
ax.set_title(f'Double Peak Sample\nCorr={corr_per_sample[double_idx]:.4f}')
ax.legend()
ax.grid(True, alpha=0.3)

# 样本对比 - 三峰
ax = axes[2, 2]
triple_idx = np.where(peak_counts_test == 3)[0][0]
ax.plot(target_wavelengths, y_true[triple_idx], 'b-', label='True', alpha=0.8)
ax.plot(target_wavelengths, y_pred[triple_idx], 'r--', label='Pred', alpha=0.8)
ax.set_title(f'Triple Peak Sample\nCorr={corr_per_sample[triple_idx]:.4f}')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('multi_peak_training_results.png', dpi=150)
print('已保存: multi_peak_training_results.png')

# ============================================================================
# 9. 保存模型
# ============================================================================
print('\n保存模型...')

torch.save({
    'model_state_dict': model.state_dict(),
    'scaler_X': scaler_X,
    'scaler_y': scaler_y,
    'wavelengths': target_wavelengths,
    'selected_biases': selected_biases,
    'weights': weights,
    'peak_config': peak_config,
    'multi_peak': True,
}, 'model_multi_peak.pth')

print('已保存: model_multi_peak.pth')

# ============================================================================
# 10. 最终总结
# ============================================================================
print('\n' + '=' * 60)
print('多峰训练总结')
print('=' * 60)

print(f"""
╔══════════════════════════════════════════════════════════════════════╗
║                      多峰光谱重建训练结果                            ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  【数据配置】                                                         ║
║    单峰: {n_single} ({n_single/n_total*100:.0f}%)                                            ║
║    双峰: {n_double} ({n_double/n_total*100:.0f}%)                                            ║
║    三峰: {n_triple} ({n_triple/n_total*100:.0f}%)                                            ║
║    四峰: {n_quad} ({n_quad/n_total*100:.0f}%)                                             ║
║    五峰: {n_penta} ({n_penta/n_total*100:.0f}%)                                              ║
║                                                                      ║
║  【曲线级别指标】                                                     ║
║    MSE: {np.mean(mse_per_sample):.6f}                                               ║
║    相关系数: {np.mean(corr_per_sample):.4f}                                              ║
║                                                                      ║
║  【峰级别指标】                                                       ║
║    峰位置误差: {np.mean(peak_position_errors):.2f} nm                                        ║
║    峰检测比: {np.mean(detection_ratios):.2f} (理想=1.0)                               ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
""")

print('=' * 60)
