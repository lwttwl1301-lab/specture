"""
双峰光谱重建训练 - 专门针对双峰场景优化

基于V2单峰模型的成功经验，专门为双峰训练新模型
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.interpolate import UnivariateSpline
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

print('=' * 70)
print('双峰光谱重建训练 - 专门优化')
print('=' * 70)

# ============================================================================
# 辅助函数
# ============================================================================

def gaussian(wavelengths, center, fwhm):
    """生成高斯峰"""
    sigma = fwhm / 2.355
    return np.exp(-0.5 * ((wavelengths - center) / sigma) ** 2)

def find_peaks_v2(spectrum, wavelengths, min_distance=25):
    """
    改进的峰检测 - 专为双峰优化
    """
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(spectrum, distance=min_distance, prominence=0.1)
    
    if len(peaks) == 0:
        # 找不到峰，返回最大值位置
        peak_idx = np.argmax(spectrum)
        return [peak_idx], [wavelengths[peak_idx]]
    elif len(peaks) == 1:
        # 只找到一个峰，尝试找第二个
        peak1_idx = peaks[0]
        # 在左侧和右侧分别找次大值
        left_part = spectrum[:max(0, peak1_idx - min_distance)]
        right_part = spectrum[min(len(spectrum), peak1_idx + min_distance):]
        
        candidates = []
        if len(left_part) > 0:
            left_max_idx = np.argmax(left_part)
            if left_part[left_max_idx] > 0.1:  # 最小强度阈值
                candidates.append((left_max_idx, left_part[left_max_idx]))
        
        if len(right_part) > 0:
            right_max_idx = np.argmax(right_part) + min(len(spectrum), peak1_idx + min_distance)
            if right_part[np.argmax(right_part)] > 0.1:
                candidates.append((right_max_idx, spectrum[right_max_idx]))
        
        if candidates:
            # 选择强度最大的作为第二个峰
            candidates.sort(key=lambda x: x[1], reverse=True)
            peak2_idx = candidates[0][0]
            if abs(peak2_idx - peak1_idx) >= min_distance:
                peaks = sorted([peak1_idx, peak2_idx])
        
        return peaks, [wavelengths[p] for p in peaks]
    else:
        # 找到多个峰，取最强的两个
        peak_heights = spectrum[peaks]
        top2_indices = np.argsort(peak_heights)[-2:]
        selected_peaks = sorted(peaks[top2_indices])
        return selected_peaks, [wavelengths[p] for p in selected_peaks]

# ============================================================================
# 1. 加载响应度
# ============================================================================
print('\n[1/6] 加载响应度...')

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

print(f'  响应度矩阵：{response_20.shape}')

# ============================================================================
# 2. 双峰数据生成配置
# ============================================================================
print('\n[2/6] 配置双峰数据生成...')

# 主要样本
n_main = 3000

# 边界过采样（借鉴V2成功经验）
n_edge_low = 1000   # 1000-1050nm 区域
n_edge_high = 1000  # 1250-1300nm 区域

total_samples = n_main + n_edge_low + n_edge_high

# 双峰参数范围
peak_config = {
    'intensity_range': (0.3, 1.0),  # 稍高强度确保可检测
    'fwhm_range': (15, 50),         # 稍窄的峰宽
    'min_separation': 30,           # 最小峰间距30nm
    'max_separation': 200           # 最大峰间距200nm
}

print(f'  数据配比：主流{n_main} + 低边界{n_edge_low} + 高边界{n_edge_high}')
print(f'  总样本数：{total_samples}')

# ============================================================================
# 3. 生成双峰光谱数据
# ============================================================================
print('\n[3/6] 生成双峰光谱数据...')

np.random.seed(42)

spectra_list = []
peak_positions_list = []  # 存储真实的双峰位置

def generate_double_peak_spectrum(wavelengths, config, region='main'):
    """生成双峰光谱"""
    wl_min, wl_max = wavelengths[0], wavelengths[-1]
    
    if region == 'low':
        # 低边界区域：两个峰都在1000-1050nm
        pos1 = np.random.uniform(1000, 1030)
        pos2 = np.random.uniform(pos1 + config['min_separation'], min(1050, pos1 + config['max_separation']))
    elif region == 'high':
        # 高边界区域：两个峰都在1250-1300nm  
        pos2 = np.random.uniform(1270, 1300)
        pos1 = np.random.uniform(max(1250, pos2 - config['max_separation']), pos2 - config['min_separation'])
    else:
        # 主流区域：均匀分布
        pos1 = np.random.uniform(wl_min + 20, wl_max - 80)
        pos2 = np.random.uniform(pos1 + config['min_separation'], min(wl_max - 20, pos1 + config['max_separation']))
    
    intensity1 = np.random.uniform(*config['intensity_range'])
    intensity2 = np.random.uniform(*config['intensity_range'])
    fwhm1 = np.random.uniform(*config['fwhm_range'])
    fwhm2 = np.random.uniform(*config['fwhm_range'])
    
    spectrum = (intensity1 * gaussian(wavelengths, pos1, fwhm1) + 
                intensity2 * gaussian(wavelengths, pos2, fwhm2))
    
    return spectrum, [pos1, pos2]

# 生成主流样本
print('  生成主流样本...')
for i in range(n_main):
    spectrum, peaks = generate_double_peak_spectrum(target_wavelengths, peak_config, 'main')
    spectra_list.append(spectrum)
    peak_positions_list.append(sorted(peaks))

# 生成低边界样本
print('  生成低边界样本...')
for i in range(n_edge_low):
    spectrum, peaks = generate_double_peak_spectrum(target_wavelengths, peak_config, 'low')
    spectra_list.append(spectrum)
    peak_positions_list.append(sorted(peaks))

# 生成高边界样本
print('  生成高边界样本...')
for i in range(n_edge_high):
    spectrum, peaks = generate_double_peak_spectrum(target_wavelengths, peak_config, 'high')
    spectra_list.append(spectrum)
    peak_positions_list.append(sorted(peaks))

spectra_double = np.array(spectra_list)
peak_positions_true = np.array(peak_positions_list)

print(f'  光谱形状：{spectra_double.shape}')
print(f'  峰位置范围：{np.min(peak_positions_true):.1f} - {np.max(peak_positions_true):.1f} nm')

# ============================================================================
# 4. 计算测量值
# ============================================================================
print('\n[4/6] 计算测量值...')

measurements_double = np.zeros((total_samples, 20))
for i in range(total_samples):
    for j in range(20):
        measurements_double[i, j] = np.sum(spectra_double[i] * response_20[:, j])

print(f'  测量值形状：{measurements_double.shape}')

# ============================================================================
# 5. 数据准备
# ============================================================================
print('\n[5/6] 数据准备...')

scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_scaled = scaler_X.fit_transform(measurements_double)
y_scaled = scaler_y.fit_transform(spectra_double)

# 分割数据
X_train, X_test, y_train, y_test, peaks_train, peaks_test = train_test_split(
    X_scaled, y_scaled, peak_positions_true, test_size=0.2, random_state=42
)

X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.FloatTensor(y_train)
X_test_tensor = torch.FloatTensor(X_test)
y_test_tensor = torch.FloatTensor(y_test)

print(f'  训练集：{X_train.shape[0]}条')
print(f'  测试集：{X_test.shape[0]}条')

# ============================================================================
# 6. 定义双峰专用模型
# ============================================================================
print('\n[6/6] 定义双峰专用模型...')

class DoublePeakNet(nn.Module):
    """
    双峰专用网络 - 基于V2架构改进
    """
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
n_params = sum(p.numel() for p in model.parameters())
print(f'  参数量：{n_params:,}')

# ============================================================================
# 7. 训练配置
# ============================================================================
print('\n[7/8] 训练配置...')

# 加权损失函数 - 边界区域权重更高（借鉴V2成功经验）
weights = np.ones(301)
weights[target_wavelengths <= 1050] = 3.0   # 低边界3倍权重
weights[target_wavelengths >= 1250] = 3.0   # 高边界3倍权重
weights_tensor = torch.FloatTensor(weights)

def weighted_mse_loss(pred, target, weights):
    return torch.mean(weights * (pred - target) ** 2)

optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

epochs = 600
train_losses = []

print('\n开始训练...')
print()

for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        pred = model(batch_X)
        loss = weighted_mse_loss(pred, batch_y, weights_tensor)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        epoch_loss += loss.item()
    
    scheduler.step()
    epoch_loss /= len(train_loader)
    train_losses.append(epoch_loss)
    
    if (epoch + 1) % 50 == 0:
        print(f'  Epoch {epoch+1}, Loss: {epoch_loss:.6f}')

print(f'  训练完成！')

# ============================================================================
# 8. 评估
# ============================================================================
print('\n[8/8] 评估...')

model.eval()
with torch.no_grad():
    y_pred_scaled = model(X_test_tensor)
    y_pred = scaler_y.inverse_transform(y_pred_scaled.numpy())

# 计算双峰定位误差
position_errors = []
count_correct = 0

for i in range(len(y_pred)):
    # 检测预测的峰
    pred_peaks_idx, pred_peaks_wl = find_peaks_v2(y_pred[i], target_wavelengths, min_distance=25)
    true_peaks = peaks_test[i]
    
    # 双峰数量正确性
    if len(pred_peaks_idx) == 2:
        count_correct += 1
        
        # 计算位置误差（匹配最近的真实峰）
        errors_for_sample = []
        for pred_wl in pred_peaks_wl:
            # 找到最近的真实峰
            distances = [abs(pred_wl - true_wl) for true_wl in true_peaks]
            min_distance = min(distances)
            errors_for_sample.append(min_distance)
        
        position_errors.extend(errors_for_sample)
    else:
        # 峰数量错误，记录大误差
        position_errors.extend([50, 50])  # 大误差惩罚

position_errors = np.array(position_errors)
accuracy_rate = count_correct / len(y_pred) * 100

print('\n【双峰定位结果】')
print(f'  峰数量准确率：{accuracy_rate:.1f}%')
print(f'  峰位置误差均值：{np.mean(position_errors):.2f} nm')
print(f'  峰位置误差中位数：{np.median(position_errors):.2f} nm')
print(f'  <1nm 精度：{np.sum(position_errors < 1) / len(position_errors) * 100:.1f}%')
print(f'  <2nm 精度：{np.sum(position_errors < 2) / len(position_errors) * 100:.1f}%')
print(f'  <5nm 精度：{np.sum(position_errors < 5) / len(position_errors) * 100:.1f}%')

# ============================================================================
# 9. 可视化
# ============================================================================
print('\n生成可视化...')

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# 训练曲线
ax = axes[0, 0]
ax.plot(train_losses, alpha=0.8)
ax.set_title('Training Loss')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.grid(True, alpha=0.3)

# 峰位置误差分布
ax = axes[0, 1]
ax.hist(position_errors, bins=50, edgecolor='black', alpha=0.7)
ax.axvline(np.mean(position_errors), color='red', linestyle='--', label=f'Mean={np.mean(position_errors):.2f}nm')
ax.axvline(1, color='green', linestyle=':', label='1nm threshold')
ax.set_title('Peak Position Error Distribution')
ax.set_xlabel('Error (nm)')
ax.set_ylabel('Count')
ax.legend()
ax.grid(True, alpha=0.3)

# 峰数量准确率
ax = axes[0, 2]
ax.bar(['Correct (2 peaks)', 'Incorrect'], [count_correct, len(y_pred) - count_correct], 
       color=['green', 'red'], alpha=0.7)
ax.set_title(f'Peak Count Accuracy: {accuracy_rate:.1f}%')
ax.set_ylabel('Count')
ax.grid(True, alpha=0.3)

# 样本对比 - 正确的双峰
correct_samples = []
for i in range(min(100, len(y_pred))):
    pred_peaks_idx, _ = find_peaks_v2(y_pred[i], target_wavelengths, min_distance=25)
    if len(pred_peaks_idx) == 2:
        correct_samples.append(i)
        if len(correct_samples) >= 3:
            break

for idx, sample_i in enumerate(correct_samples[:3]):
    ax = axes[1, idx]
    ax.plot(target_wavelengths, y_test[sample_i], 'b-', label='True Spectrum', alpha=0.8)
    ax.plot(target_wavelengths, y_pred[sample_i], 'r--', label='Predicted Spectrum', alpha=0.8)
    
    # 标记真实峰
    true_peaks = peaks_test[sample_i]
    for peak_wl in true_peaks:
        ax.axvline(peak_wl, color='blue', linestyle=':', alpha=0.7)
    
    # 标记预测峰
    _, pred_peaks_wl = find_peaks_v2(y_pred[sample_i], target_wavelengths, min_distance=25)
    for peak_wl in pred_peaks_wl:
        ax.axvline(peak_wl, color='red', linestyle=':', alpha=0.7)
    
    ax.set_title(f'Sample {sample_i} (Correct)')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('double_peak_training_results.png', dpi=150)
print('已保存：double_peak_training_results.png')

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
    'peak_config': peak_config,
    'version': 'double_peak_v1',
    'training_config': {
        'epochs': epochs,
        'batch_size': 64,
        'lr': 0.001,
        'weight_decay': 1e-4
    }
}, 'model_double_peak.pth')

print('已保存：model_double_peak.pth')

# ============================================================================
# 11. 最终总结
# ============================================================================
print('\n' + '=' * 70)
print('双峰训练总结')
print('=' * 70)

print(f"""
╔══════════════════════════════════════════════════════════════════════╗
║                    双峰专用模型训练结果                              ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  【模型架构】                                                         ║
║    基于V2单峰模型架构                                                 ║
║    参数量: {n_params:,}                                              ║
║                                                                      ║
║  【数据策略】                                                         ║
║    主流样本: {n_main}                                                 ║
║    边界过采样: 低边界{n_edge_low} + 高边界{n_edge_high}               ║
║    总样本数: {total_samples}                                          ║
║                                                                      ║
║  【训练配置】                                                         ║
║    加权损失: 边界区域3倍权重                                          ║
║    训练轮数: {epochs}                                                 ║
║    优化器: AdamW + 余弦退火                                           ║
║                                                                      ║
║  【评估结果】                                                         ║
║    峰数量准确率: {accuracy_rate:.1f}%                                 ║
║    峰位置误差均值: {np.mean(position_errors):.2f} nm                   ║
║    <1nm 精度: {np.sum(position_errors < 1) / len(position_errors) * 100:.1f}%   ║
║    <5nm 精度: {np.sum(position_errors < 5) / len(position_errors) * 100:.1f}%   ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
""")

print('=' * 70)