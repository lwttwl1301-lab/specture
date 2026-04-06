"""
多峰光谱重建训练 V3.2 - 方案二：混合损失函数
在MSE基础上添加峰位置损失，强制模型关注峰定位精度

改进点：
1. 混合损失: MSE(0.9) + PeakPositionLoss(0.1)
2. 峰位置损失: 使用抛物线插值精确定位峰
3. 动态权重调整: 训练后期增加峰位置权重
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
from scipy.interpolate import UnivariateSpline
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

print('=' * 70)
print('多峰光谱训练 V3.2 - 混合损失函数（方案二）')
print('=' * 70)

# ============================================================================
# 辅助函数
# ============================================================================

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
        intensity = np.random.uniform(*config['intensity_range'])
        fwhm = np.random.uniform(*config['fwhm_range'])
        peaks.append((pos, intensity, fwhm))
    peaks = enforce_min_separation(peaks, min_sep=config['min_separation'])
    spectrum = np.zeros_like(wavelengths, dtype=float)
    for pos, intensity, fwhm in peaks:
        spectrum += intensity * gaussian(wavelengths, pos, fwhm)
    return spectrum, peaks

def find_peaks_scipy(spectrum, wavelengths, min_prominence=0.1):
    """使用scipy找峰"""
    peaks, properties = find_peaks(spectrum, prominence=min_prominence)
    peak_positions = wavelengths[peaks]
    peak_heights = spectrum[peaks]
    return peak_positions, peak_heights

def parabolic_peak_interp(spectrum, wavelengths):
    """抛物线插值精确定位峰"""
    idx = np.argmax(spectrum)
    if idx == 0 or idx == len(spectrum) - 1:
        return wavelengths[idx]
    y1, y2, y3 = spectrum[idx-1], spectrum[idx], spectrum[idx+1]
    x1, x2, x3 = wavelengths[idx-1], wavelengths[idx], wavelengths[idx+1]
    denom = (y1 - 2*y2 + y3)
    if denom == 0:
        return x2
    peak_x = x2 + (x3 - x2) * (y1 - y3) / (2 * denom)
    peak_x = max(x1, min(peak_x, x3))
    return peak_x

def find_main_peak_position(spectrum, wavelengths):
    """找到主峰位置（使用抛物线插值）"""
    return parabolic_peak_interp(spectrum, wavelengths)

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
# 2. 多峰数据生成配置
# ============================================================================
print('\n[2/8] 配置多峰数据生成...')

n_single = 4000
n_double = 2500
n_triple = 2000
n_quad = 1000
n_penta = 500
n_total = n_single + n_double + n_triple + n_quad + n_penta

peak_config = {
    'intensity_range': (0.2, 1.0),
    'fwhm_range': (10, 60),
    'min_separation': 30
}

print(f'  数据配比: 单峰{n_single} 双峰{n_double} 三峰{n_triple} 四峰{n_quad} 五峰{n_penta}')

# ============================================================================
# 3. 生成多峰光谱数据
# ============================================================================
print('\n[3/8] 生成多峰光谱数据...')

np.random.seed(42)

spectra_list = []
peak_counts = []
peak_infos = []

print('  生成单峰...')
for i in range(n_single):
    spectrum, peaks = generate_multi_peak_spectrum(target_wavelengths, 1, peak_config)
    spectra_list.append(spectrum)
    peak_counts.append(1)
    peak_infos.append(peaks)

print('  生成双峰...')
for i in range(n_double):
    spectrum, peaks = generate_multi_peak_spectrum(target_wavelengths, 2, peak_config)
    spectra_list.append(spectrum)
    peak_counts.append(2)
    peak_infos.append(peaks)

print('  生成三峰...')
for i in range(n_triple):
    spectrum, peaks = generate_multi_peak_spectrum(target_wavelengths, 3, peak_config)
    spectra_list.append(spectrum)
    peak_counts.append(3)
    peak_infos.append(peaks)

print('  生成四峰...')
for i in range(n_quad):
    spectrum, peaks = generate_multi_peak_spectrum(target_wavelengths, 4, peak_config)
    spectra_list.append(spectrum)
    peak_counts.append(4)
    peak_infos.append(peaks)

print('  生成五峰...')
for i in range(n_penta):
    spectrum, peaks = generate_multi_peak_spectrum(target_wavelengths, 5, peak_config)
    spectra_list.append(spectrum)
    peak_counts.append(5)
    peak_infos.append(peaks)

spectra_multi = np.array(spectra_list)
peak_counts = np.array(peak_counts)

print(f'  光谱形状: {spectra_multi.shape}')

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

# 边界权重
weights = np.ones(len(target_wavelengths))
edge_low = (target_wavelengths <= 1050)
edge_high = (target_wavelengths >= 1250)
weights[edge_low] = 3.0
weights[edge_high] = 3.0

# 分割数据
X_train, X_test, y_train, y_test, peak_counts_train, peak_counts_test = train_test_split(
    X_scaled, y_scaled, peak_counts, test_size=0.2, random_state=42
)

# 计算训练集的真实峰位置（用于峰位置损失）
print('  计算训练集峰位置...')
y_train_original = scaler_y.inverse_transform(y_train)
true_peak_positions_train = []
for i in range(len(y_train_original)):
    peak_pos = find_main_peak_position(y_train_original[i], target_wavelengths)
    true_peak_positions_train.append(peak_pos)
true_peak_positions_train = np.array(true_peak_positions_train)

X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.FloatTensor(y_train)
X_test_tensor = torch.FloatTensor(X_test)
y_test_tensor = torch.FloatTensor(y_test)
weights_tensor = torch.FloatTensor(weights)
true_peak_positions_tensor = torch.FloatTensor(true_peak_positions_train)

print(f'  训练集: {X_train.shape[0]}条')
print(f'  测试集: {X_test.shape[0]}条')

# ============================================================================
# 6. 定义模型
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
# 7. 训练 - 混合损失函数
# ============================================================================
print('\n[7/8] 训练（混合损失函数）...')

train_dataset = TensorDataset(X_train_tensor, y_train_tensor, true_peak_positions_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

def weighted_mse_loss(pred, target, weights):
    """加权MSE损失"""
    return torch.mean(weights * (pred - target) ** 2)

def peak_position_loss(pred, target_spectra, true_peak_positions, wavelengths_tensor):
    """
    峰位置损失：计算预测峰位置与真实峰位置的误差
    使用抛物线插值精确定位峰
    """
    batch_size = pred.shape[0]
    losses = []
    
    for i in range(batch_size):
        pred_spectrum = pred[i]
        true_pos = true_peak_positions[i]
        
        # 找到预测峰位置（argmax）
        pred_idx = torch.argmax(pred_spectrum)
        
        # 边界检查
        if pred_idx == 0 or pred_idx == len(pred_spectrum) - 1:
            pred_pos = wavelengths_tensor[pred_idx]
        else:
            # 抛物线插值
            y1, y2, y3 = pred_spectrum[pred_idx-1], pred_spectrum[pred_idx], pred_spectrum[pred_idx+1]
            x1, x2, x3 = wavelengths_tensor[pred_idx-1], wavelengths_tensor[pred_idx], wavelengths_tensor[pred_idx+1]
            
            denom = (y1 - 2*y2 + y3)
            if torch.abs(denom) < 1e-8:
                pred_pos = x2
            else:
                pred_pos = x2 + (x3 - x2) * (y1 - y3) / (2 * denom)
                pred_pos = torch.clamp(pred_pos, x1, x3)
        
        # 峰位置误差（nm）
        error = torch.abs(pred_pos - true_pos)
        losses.append(error)
    
    return torch.mean(torch.stack(losses))

def combined_loss(pred, target_spectra, true_peak_positions, weights, wavelengths_tensor, 
                  mse_weight=0.9, peak_weight=0.1):
    """混合损失：MSE + 峰位置损失"""
    mse = weighted_mse_loss(pred, target_spectra, weights)
    peak_loss = peak_position_loss(pred, target_spectra, true_peak_positions, wavelengths_tensor)
    
    # 归一化峰位置损失（使其与MSE量级相当）
    peak_loss_normalized = peak_loss / 100.0  # 峰位置误差通常在0-100nm范围
    
    total_loss = mse_weight * mse + peak_weight * peak_loss_normalized
    return total_loss, mse, peak_loss

optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2)

epochs = 600
train_losses = []
train_mse_losses = []
train_peak_losses = []
test_losses = []
best_test_loss = float('inf')
best_model_state = None

wavelengths_tensor = torch.FloatTensor(target_wavelengths)

for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    epoch_mse = 0
    epoch_peak = 0
    
    # 动态调整权重：后期增加峰位置权重
    if epoch < 300:
        mse_w, peak_w = 0.95, 0.05
    elif epoch < 450:
        mse_w, peak_w = 0.9, 0.1
    else:
        mse_w, peak_w = 0.85, 0.15
    
    for batch_X, batch_y, batch_peak_pos in train_loader:
        optimizer.zero_grad()
        pred = model(batch_X)
        loss, mse, peak_loss = combined_loss(
            pred, batch_y, batch_peak_pos, weights_tensor, wavelengths_tensor,
            mse_weight=mse_w, peak_weight=peak_w
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_mse += mse.item()
        epoch_peak += peak_loss.item()
    
    scheduler.step()
    
    epoch_loss /= len(train_loader)
    epoch_mse /= len(train_loader)
    epoch_peak /= len(train_loader)
    
    train_losses.append(epoch_loss)
    train_mse_losses.append(epoch_mse)
    train_peak_losses.append(epoch_peak)
    
    model.eval()
    with torch.no_grad():
        test_pred = model(X_test_tensor)
        test_loss = torch.mean((test_pred - y_test_tensor) ** 2).item()
    test_losses.append(test_loss)
    
    if test_loss < best_test_loss:
        best_test_loss = test_loss
        best_model_state = model.state_dict().copy()
    
    if (epoch + 1) % 100 == 0:
        print(f'  Epoch {epoch+1}, Loss: {epoch_loss:.6f} (MSE:{epoch_mse:.6f}, Peak:{epoch_peak:.2f}nm), Test: {test_loss:.6f}')

model.load_state_dict(best_model_state)
print(f'  最佳测试损失: {best_test_loss:.6f}')

# ============================================================================
# 8. 评估
# ============================================================================
print('\n[8/8] 评估...')

model.eval()
with torch.no_grad():
    y_pred_scaled = model(X_test_tensor).numpy()

y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_true_orig = scaler_y.inverse_transform(y_test)
y_pred = np.clip(y_pred, 0, None)

# 曲线级别指标
mse_per_sample = np.mean((y_pred - y_true_orig) ** 2, axis=1)
mae_per_sample = np.mean(np.abs(y_pred - y_true_orig), axis=1)

corr_per_sample = []
for i in range(len(y_true_orig)):
    c = np.corrcoef(y_true_orig[i], y_pred[i])[0, 1]
    corr_per_sample.append(c)
corr_per_sample = np.array(corr_per_sample)

print('\n【曲线级别指标】')
print(f'  MSE 均值: {np.mean(mse_per_sample):.6f}')
print(f'  MAE 均值: {np.mean(mae_per_sample):.6f}')
print(f'  相关系数 均值: {np.mean(corr_per_sample):.4f}')

# 峰级别指标
print('\n【峰级别指标】')
peak_position_errors = []
for i in range(len(y_true_orig)):
    true_pos = find_main_peak_position(y_true_orig[i], target_wavelengths)
    pred_pos = find_main_peak_position(y_pred[i], target_wavelengths)
    error = abs(pred_pos - true_pos)
    peak_position_errors.append(error)

peak_position_errors = np.array(peak_position_errors)
print(f'  峰位置误差: {np.mean(peak_position_errors):.2f} nm (中位数: {np.median(peak_position_errors):.2f} nm)')
print(f'  <1nm精度: {np.sum(peak_position_errors < 1) / len(peak_position_errors) * 100:.1f}%')
print(f'  <2nm精度: {np.sum(peak_position_errors < 2) / len(peak_position_errors) * 100:.1f}%')
print(f'  <5nm精度: {np.sum(peak_position_errors < 5) / len(peak_position_errors) * 100:.1f}%')

# ============================================================================
# 9. 可视化
# ============================================================================
print('\n生成可视化...')

fig, axes = plt.subplots(3, 3, figsize=(18, 15))

# 训练曲线 - 总损失
ax = axes[0, 0]
ax.plot(train_losses, label='Train Loss', alpha=0.8)
ax.plot(test_losses, label='Test Loss', alpha=0.8)
ax.set_title('Training Curves (Combined Loss)')
ax.legend()
ax.grid(True, alpha=0.3)

# MSE损失
ax = axes[0, 1]
ax.plot(train_mse_losses, label='MSE Loss', color='blue', alpha=0.8)
ax.set_title('MSE Loss')
ax.legend()
ax.grid(True, alpha=0.3)

# 峰位置损失
ax = axes[0, 2]
ax.plot(train_peak_losses, label='Peak Position Loss (nm)', color='red', alpha=0.8)
ax.set_title('Peak Position Loss')
ax.legend()
ax.grid(True, alpha=0.3)

# 峰误差分布
ax = axes[1, 0]
ax.hist(peak_position_errors, bins=50, edgecolor='black', alpha=0.7)
ax.axvline(np.mean(peak_position_errors), color='red', linestyle='--', label=f'Mean={np.mean(peak_position_errors):.2f}nm')
ax.axvline(1, color='green', linestyle=':', label='1nm threshold')
ax.set_title('Peak Position Error Distribution')
ax.legend()
ax.grid(True, alpha=0.3)

# MSE分布
ax = axes[1, 1]
ax.hist(mse_per_sample, bins=50, edgecolor='black', alpha=0.7)
ax.axvline(np.mean(mse_per_sample), color='red', linestyle='--', label=f'Mean={np.mean(mse_per_sample):.4f}')
ax.set_title('MSE Distribution')
ax.legend()
ax.grid(True, alpha=0.3)

# 相关系数分布
ax = axes[1, 2]
ax.hist(corr_per_sample, bins=50, edgecolor='black', alpha=0.7)
ax.axvline(np.mean(corr_per_sample), color='red', linestyle='--', label=f'Mean={np.mean(corr_per_sample):.4f}')
ax.set_title('Correlation Distribution')
ax.legend()
ax.grid(True, alpha=0.3)

# 样本对比 - 单峰
ax = axes[2, 0]
single_idx = np.where(peak_counts_test == 1)[0][0]
ax.plot(target_wavelengths, y_true_orig[single_idx], 'b-', label='True', alpha=0.8)
ax.plot(target_wavelengths, y_pred[single_idx], 'r--', label='Pred', alpha=0.8)
true_pos = find_main_peak_position(y_true_orig[single_idx], target_wavelengths)
pred_pos = find_main_peak_position(y_pred[single_idx], target_wavelengths)
ax.axvline(true_pos, color='b', linestyle=':', alpha=0.7, label=f'True={true_pos:.1f}nm')
ax.axvline(pred_pos, color='r', linestyle=':', alpha=0.7, label=f'Pred={pred_pos:.1f}nm')
ax.set_title(f'Single Peak\nError={abs(pred_pos-true_pos):.2f}nm, Corr={corr_per_sample[single_idx]:.4f}')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# 样本对比 - 双峰
ax = axes[2, 1]
double_idx = np.where(peak_counts_test == 2)[0][0]
ax.plot(target_wavelengths, y_true_orig[double_idx], 'b-', label='True', alpha=0.8)
ax.plot(target_wavelengths, y_pred[double_idx], 'r--', label='Pred', alpha=0.8)
ax.set_title(f'Double Peak\nCorr={corr_per_sample[double_idx]:.4f}')
ax.legend()
ax.grid(True, alpha=0.3)

# 样本对比 - 三峰
ax = axes[2, 2]
triple_idx = np.where(peak_counts_test == 3)[0][0]
ax.plot(target_wavelengths, y_true_orig[triple_idx], 'b-', label='True', alpha=0.8)
ax.plot(target_wavelengths, y_pred[triple_idx], 'r--', label='Pred', alpha=0.8)
ax.set_title(f'Triple Peak\nCorr={corr_per_sample[triple_idx]:.4f}')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('multi_peak_training_v2_results.png', dpi=150)
print('已保存: multi_peak_training_v2_results.png')

# ============================================================================
# 10. 保存模型
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
    'version': 'v3.2_peak_loss',
}, 'model_multi_peak_v2.pth')

print('已保存: model_multi_peak_v2.pth')

# ============================================================================
# 11. 最终总结
# ============================================================================
print('\n' + '=' * 70)
print('多峰训练 V3.2 总结 - 混合损失函数（方案二）')
print('=' * 70)

print(f"""
╔══════════════════════════════════════════════════════════════════════╗
║              方案二：混合损失函数训练结果                            ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  【损失函数配置】                                                     ║
║    阶段1 (0-300轮): MSE(0.95) + Peak(0.05)                          ║
║    阶段2 (300-450轮): MSE(0.90) + Peak(0.10)                        ║
║    阶段3 (450-600轮): MSE(0.85) + Peak(0.15)                        ║
║                                                                      ║
║  【曲线级别指标】                                                     ║
║    MSE: {np.mean(mse_per_sample):.6f}                                               ║
║    MAE: {np.mean(mae_per_sample):.6f}                                               ║
║    相关系数: {np.mean(corr_per_sample):.4f}                                              ║
║                                                                      ║
║  【峰级别指标】                                                       ║
║    峰位置误差: {np.mean(peak_position_errors):.2f} nm (中位数: {np.median(peak_position_errors):.2f} nm)              ║
║    <1nm精度: {np.sum(peak_position_errors < 1) / len(peak_position_errors) * 100:.1f}%                              ║
║    <2nm精度: {np.sum(peak_position_errors < 2) / len(peak_position_errors) * 100:.1f}%                             ║
║    <5nm精度: {np.sum(peak_position_errors < 5) / len(peak_position_errors) * 100:.1f}%                             ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
""")

print('=' * 70)
