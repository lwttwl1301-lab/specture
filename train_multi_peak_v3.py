"""
多峰光谱重建训练 V3.3 - 方案三：端到端峰预测
绕过曲线重建，直接预测峰参数（位置、强度、宽度）

核心思想：
1. 不重建完整光谱曲线
2. 直接从测量值预测每个峰的参数
3. 使用匈牙利算法匹配预测峰和真实峰
4. 峰参数损失直接优化峰定位精度
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
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

print('=' * 70)
print('多峰光谱训练 V3.3 - 方案三（端到端峰预测）')
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

def match_peaks_hungarian(pred_peaks, true_peaks, max_distance=50):
    """
    使用匈牙利算法匹配预测峰和真实峰
    pred_peaks: [(pos, intensity, fwhm), ...]
    true_peaks: [(pos, intensity, fwhm), ...]
    返回: 匹配的峰位置误差列表
    """
    if len(pred_peaks) == 0 or len(true_peaks) == 0:
        return []
    
    # 构建距离矩阵（只考虑位置）
    n_pred = len(pred_peaks)
    n_true = len(true_peaks)
    
    # 距离矩阵
    cost_matrix = np.zeros((n_pred, n_true))
    for i, p_pred in enumerate(pred_peaks):
        for j, p_true in enumerate(true_peaks):
            cost_matrix[i, j] = abs(p_pred[0] - p_true[0])
    
    # 匈牙利算法匹配
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    # 收集匹配的峰位置误差
    matched_errors = []
    for i, j in zip(row_ind, col_ind):
        if cost_matrix[i, j] < max_distance:  # 只考虑距离小于阈值的匹配
            pos_error = abs(pred_peaks[i][0] - true_peaks[j][0])
            matched_errors.append(pos_error)
    
    return matched_errors

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

print(f'  响应度矩阵：{response_20.shape}')

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

# 最大峰数量
MAX_PEAKS = 5

peak_config = {
    'intensity_range': (0.2, 1.0),
    'fwhm_range': (10, 60),
    'min_separation': 30
}

print(f'  数据配比：单峰{n_single} 双峰{n_double} 三峰{n_triple} 四峰{n_quad} 五峰{n_penta}')
print(f'  最大峰数量：{MAX_PEAKS}')

# ============================================================================
# 3. 生成多峰光谱数据
# ============================================================================
print('\n[3/8] 生成多峰光谱数据...')

np.random.seed(42)

spectra_list = []
peak_params_list = []  # 每个样本的峰参数 [(pos, intensity, fwhm), ...]
peak_counts = []

print('  生成单峰...')
for i in range(n_single):
    spectrum, peaks = generate_multi_peak_spectrum(target_wavelengths, 1, peak_config)
    spectra_list.append(spectrum)
    peak_params_list.append(peaks)
    peak_counts.append(1)

print('  生成双峰...')
for i in range(n_double):
    spectrum, peaks = generate_multi_peak_spectrum(target_wavelengths, 2, peak_config)
    spectra_list.append(spectrum)
    peak_params_list.append(peaks)
    peak_counts.append(2)

print('  生成三峰...')
for i in range(n_triple):
    spectrum, peaks = generate_multi_peak_spectrum(target_wavelengths, 3, peak_config)
    spectra_list.append(spectrum)
    peak_params_list.append(peaks)
    peak_counts.append(3)

print('  生成四峰...')
for i in range(n_quad):
    spectrum, peaks = generate_multi_peak_spectrum(target_wavelengths, 4, peak_config)
    spectra_list.append(spectrum)
    peak_params_list.append(peaks)
    peak_counts.append(4)

print('  生成五峰...')
for i in range(n_penta):
    spectrum, peaks = generate_multi_peak_spectrum(target_wavelengths, 5, peak_config)
    spectra_list.append(spectrum)
    peak_params_list.append(peaks)
    peak_counts.append(5)

spectra_multi = np.array(spectra_list)
peak_counts = np.array(peak_counts)

print(f'  光谱形状：{spectra_multi.shape}')

# ============================================================================
# 4. 计算测量值
# ============================================================================
print('\n[4/8] 计算测量值...')

measurements_multi = np.zeros((n_total, 20))
for i in range(n_total):
    for j in range(20):
        measurements_multi[i, j] = np.sum(spectra_multi[i] * response_20[:, j])

print(f'  测量值形状：{measurements_multi.shape}')

# ============================================================================
# 5. 准备峰参数标签
# ============================================================================
print('\n[5/8] 准备峰参数标签...')

# 将峰参数转换为固定大小的数组
# 每个样本有 MAX_PEAKS 个峰，每个峰有 3 个参数 (pos, intensity, fwhm)
# 峰数量不足的用 0 填充

# 峰参数归一化范围
wl_min, wl_max = 1000, 1300
intensity_min, intensity_max = 0.2, 1.0
fwhm_min, fwhm_max = 10, 60

peak_params_array = np.zeros((n_total, MAX_PEAKS, 3))

for i, peaks in enumerate(peak_params_list):
    # 按位置排序
    sorted_peaks = sorted(peaks, key=lambda x: x[0])
    for j, (pos, intensity, fwhm) in enumerate(sorted_peaks[:MAX_PEAKS]):
        # 归一化参数
        peak_params_array[i, j, 0] = (pos - wl_min) / (wl_max - wl_min)  # 位置归一化到 [0, 1]
        peak_params_array[i, j, 1] = (intensity - intensity_min) / (intensity_max - intensity_min)  # 强度归一化
        peak_params_array[i, j, 2] = (fwhm - fwhm_min) / (fwhm_max - fwhm_min)  # FWHM 归一化

# 峰数量标签
peak_counts_array = np.array(peak_counts)

print(f'  峰参数数组形状：{peak_params_array.shape}')
print(f'  峰数量数组形状：{peak_counts_array.shape}')

# ============================================================================
# 6. 数据准备
# ============================================================================
print('\n[6/8] 数据准备...')

scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(measurements_multi)

# 分割数据
X_train, X_test, y_train, y_test, peak_params_train, peak_params_test, peak_counts_train, peak_counts_test = train_test_split(
    X_scaled, spectra_multi, peak_params_array, peak_counts_array, test_size=0.2, random_state=42
)

X_train_tensor = torch.FloatTensor(X_train)
peak_params_train_tensor = torch.FloatTensor(peak_params_train)
peak_counts_train_tensor = torch.FloatTensor(peak_counts_train)
X_test_tensor = torch.FloatTensor(X_test)
peak_params_test_tensor = torch.FloatTensor(peak_params_test)
peak_counts_test_tensor = torch.FloatTensor(peak_counts_test)

print(f'  训练集：{X_train.shape[0]}条')
print(f'  测试集：{X_test.shape[0]}条')

# ============================================================================
# 7. 定义端到端峰预测模型
# ============================================================================
print('\n[7/8] 定义端到端峰预测模型...')

class PeakPredictorNet(nn.Module):
    """
    端到端峰预测网络
    输入: 20 个测量值
    输出: 
      - 峰数量 (1个值，通过 sigmoid 输出 0-5)
      - 峰参数 (MAX_PEAKS * 3 个值: pos, intensity, fwhm)
    """
    def __init__(self, input_dim, max_peaks):
        super().__init__()
        self.max_peaks = max_peaks
        
        # 共享特征提取
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
        
        # 峰数量预测头
        self.peak_count_head = nn.Sequential(
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # 输出 0-1，乘以 MAX_PEAKS 得到峰数量
        )
        
        # 峰参数预测头
        self.peak_params_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, max_peaks * 3),
        )
        
        # 峰存在性预测头（每个峰是否存在）
        self.peak_exist_head = nn.Sequential(
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Linear(64, max_peaks),
            nn.Sigmoid()  # 每个峰存在的概率
        )
    
    def forward(self, x):
        features = self.shared(x)
        
        # 峰数量
        peak_count = self.peak_count_head(features) * self.max_peaks
        
        # 峰存在性
        peak_exist = self.peak_exist_head(features)
        
        # 峰参数
        peak_params = self.peak_params_head(features)
        peak_params = peak_params.view(-1, self.max_peaks, 3)
        
        # 对峰参数进行约束
        # 位置: sigmoid -> [0, 1]
        peak_params[:, :, 0] = torch.sigmoid(peak_params[:, :, 0])
        # 强度: sigmoid -> [0, 1]
        peak_params[:, :, 1] = torch.sigmoid(peak_params[:, :, 1])
        # FWHM: sigmoid -> [0, 1]
        peak_params[:, :, 2] = torch.sigmoid(peak_params[:, :, 2])
        
        return peak_count, peak_exist, peak_params

model = PeakPredictorNet(20, MAX_PEAKS)
n_params = sum(p.numel() for p in model.parameters())
print(f'  参数量：{n_params:,}')

# ============================================================================
# 8. 训练 - 端到端峰预测
# ============================================================================
print('\n[8/8] 训练（端到端峰预测）...')

train_dataset = TensorDataset(X_train_tensor, peak_params_train_tensor, peak_counts_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

def peak_loss(pred_params, true_params, pred_exist, true_count, wavelengths_range=(1000, 1300)):
    """
    峰参数损失函数
    使用匈牙利算法匹配预测峰和真实峰
    """
    batch_size = pred_params.shape[0]
    max_peaks = pred_params.shape[1]
    
    total_loss = 0
    position_errors = []
    
    wl_min, wl_max = wavelengths_range
    
    for i in range(batch_size):
        # 获取真实峰数量
        n_true_peaks = int(true_count[i].item())
        
        # 获取真实峰参数（反归一化）
        true_peaks = []
        for j in range(n_true_peaks):
            pos = true_params[i, j, 0].item() * (wl_max - wl_min) + wl_min
            intensity = true_params[i, j, 1].item() * (intensity_max - intensity_min) + intensity_min
            fwhm = true_params[i, j, 2].item() * (fwhm_max - fwhm_min) + fwhm_min
            true_peaks.append((pos, intensity, fwhm))
        
        # 获取预测峰参数（反归一化）
        pred_peaks = []
        for j in range(max_peaks):
            # 只考虑存在概率高的峰
            if pred_exist[i, j].item() > 0.3:
                pos = pred_params[i, j, 0].item() * (wl_max - wl_min) + wl_min
                intensity = pred_params[i, j, 1].item() * (intensity_max - intensity_min) + intensity_min
                fwhm = pred_params[i, j, 2].item() * (fwhm_max - fwhm_min) + fwhm_min
                pred_peaks.append((pos, intensity, fwhm))
        
        # 匈牙利算法匹配
        matched_errors = match_peaks_hungarian(pred_peaks, true_peaks, max_distance=50)
        
        if len(matched_errors) > 0:
            # 匹配的峰位置损失
            for error in matched_errors:
                total_loss += error ** 2
                position_errors.append(error)
        
        # 未匹配的峰惩罚
        n_matched = len(matched_errors)
        n_unmatched_true = n_true_peaks - n_matched
        n_unmatched_pred = len(pred_peaks) - n_matched
        
        # 惩罚未匹配的峰
        total_loss += n_unmatched_true * 100  # 漏检惩罚
        total_loss += n_unmatched_pred * 50   # 误检惩罚
    
    return total_loss / batch_size, position_errors

def combined_peak_loss(pred_count, pred_exist, pred_params, true_params, true_count):
    """
    组合损失：峰数量损失 + 峰存在性损失 + 峰参数损失
    """
    # 峰数量损失
    count_loss = nn.functional.mse_loss(pred_count.squeeze(), true_count)
    
    # 峰存在性损失
    # 构建真实峰存在标签
    batch_size = pred_exist.shape[0]
    max_peaks = pred_exist.shape[1]
    true_exist = torch.zeros(batch_size, max_peaks)
    for i in range(batch_size):
        n_peaks = int(true_count[i].item())
        true_exist[i, :n_peaks] = 1.0
    
    exist_loss = nn.functional.binary_cross_entropy(pred_exist, true_exist)
    
    # 峰参数损失（匈牙利匹配）
    peak_param_loss, position_errors = peak_loss(pred_params, true_params, pred_exist, true_count)
    
    total_loss = count_loss + exist_loss + peak_param_loss
    
    return total_loss, count_loss, exist_loss, peak_param_loss, position_errors

optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2)

epochs = 500
train_losses = []
count_losses = []
exist_losses = []
peak_param_losses = []
position_errors_history = []

print('\n  开始训练...')
print()

for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    epoch_count_loss = 0
    epoch_exist_loss = 0
    epoch_peak_loss = 0
    all_position_errors = []
    
    for batch_X, batch_params, batch_count in train_loader:
        optimizer.zero_grad()
        pred_count, pred_exist, pred_params = model(batch_X)
        
        loss, count_loss, exist_loss, peak_loss_val, position_errors = combined_peak_loss(
            pred_count, pred_exist, pred_params, batch_params, batch_count
        )
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_count_loss += count_loss.item()
        epoch_exist_loss += exist_loss.item()
        epoch_peak_loss += peak_loss_val
        all_position_errors.extend(position_errors)
    
    scheduler.step()
    
    epoch_loss /= len(train_loader)
    epoch_count_loss /= len(train_loader)
    epoch_exist_loss /= len(train_loader)
    epoch_peak_loss /= len(train_loader)
    
    train_losses.append(epoch_loss)
    count_losses.append(epoch_count_loss)
    exist_losses.append(epoch_exist_loss)
    peak_param_losses.append(epoch_peak_loss)
    
    if len(all_position_errors) > 0:
        mean_error = np.mean(all_position_errors)
        position_errors_history.append(mean_error)
    else:
        position_errors_history.append(100)  # 无匹配时记录大值
    
    if (epoch + 1) % 50 == 0:
        print(f'  Epoch {epoch+1}, Loss: {epoch_loss:.4f} (Count:{epoch_count_loss:.4f}, Exist:{epoch_exist_loss:.4f}, Peak:{epoch_peak_loss:.4f}), PosError: {position_errors_history[-1]:.2f}nm')

print(f'  训练完成！')

# ============================================================================
# 9. 评估
# ============================================================================
print('\n[9/10] 评估...')

model.eval()

# 测试集评估
all_position_errors = []
all_count_errors = []

with torch.no_grad():
    pred_count, pred_exist, pred_params = model(X_test_tensor)
    
    for i in range(len(X_test_tensor)):
        # 真实峰数量
        n_true_peaks = int(peak_counts_test[i])
        
        # 预测峰数量
        n_pred_peaks = int(pred_count[i].item())
        count_error = abs(n_pred_peaks - n_true_peaks)
        all_count_errors.append(count_error)
        
        # 真实峰参数
        true_peaks = []
        for j in range(n_true_peaks):
            pos = peak_params_test[i, j, 0] * (wl_max - wl_min) + wl_min
            intensity = peak_params_test[i, j, 1] * (intensity_max - intensity_min) + intensity_min
            fwhm = peak_params_test[i, j, 2] * (fwhm_max - fwhm_min) + fwhm_min
            true_peaks.append((pos, intensity, fwhm))
        
        # 预测峰参数
        pred_peaks = []
        for j in range(MAX_PEAKS):
            if pred_exist[i, j].item() > 0.5:
                pos = pred_params[i, j, 0].item() * (wl_max - wl_min) + wl_min
                intensity = pred_params[i, j, 1].item() * (intensity_max - intensity_min) + intensity_min
                fwhm = pred_params[i, j, 2].item() * (fwhm_max - fwhm_min) + fwhm_min
                pred_peaks.append((pos, intensity, fwhm))
        
        # 匈牙利匹配
        matched_errors = match_peaks_hungarian(pred_peaks, true_peaks, max_distance=50)
        all_position_errors.extend(matched_errors)

all_position_errors = np.array(all_position_errors)
all_count_errors = np.array(all_count_errors)

print('\n【峰数量预测】')
print(f'  峰数量误差均值：{np.mean(all_count_errors):.2f}')
print(f'  峰数量准确率：{np.sum(all_count_errors == 0) / len(all_count_errors) * 100:.1f}%')

print('\n【峰位置预测】')
print(f'  峰位置误差：{np.mean(all_position_errors):.2f} nm (中位数：{np.median(all_position_errors):.2f} nm)')
print(f'  <1nm 精度：{np.sum(all_position_errors < 1) / len(all_position_errors) * 100:.1f}%')
print(f'  <2nm 精度：{np.sum(all_position_errors < 2) / len(all_position_errors) * 100:.1f}%')
print(f'  <5nm 粍度：{np.sum(all_position_errors < 5) / len(all_position_errors) * 100:.1f}%')

# ============================================================================
# 10. 可视化
# ============================================================================
print('\n生成可视化...')

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# 训练曲线 - 总损失
ax = axes[0, 0]
ax.plot(train_losses, label='Total Loss', alpha=0.8)
ax.set_title('Training Loss')
ax.legend()
ax.grid(True, alpha=0.3)

# 峰数量损失
ax = axes[0, 1]
ax.plot(count_losses, label='Count Loss', color='blue', alpha=0.8)
ax.set_title('Peak Count Loss')
ax.legend()
ax.grid(True, alpha=0.3)

# 峰位置误差
ax = axes[0, 2]
ax.plot(position_errors_history, label='Position Error (nm)', color='red', alpha=0.8)
ax.set_title('Peak Position Error')
ax.legend()
ax.grid(True, alpha=0.3)

# 峰位置误差分布
ax = axes[1, 0]
ax.hist(all_position_errors, bins=50, edgecolor='black', alpha=0.7)
ax.axvline(np.mean(all_position_errors), color='red', linestyle='--', label=f'Mean={np.mean(all_position_errors):.2f}nm')
ax.axvline(1, color='green', linestyle=':', label='1nm threshold')
ax.set_title('Peak Position Error Distribution')
ax.legend()
ax.grid(True, alpha=0.3)

# 峰数量误差分布
ax = axes[1, 1]
ax.hist(all_count_errors, bins=range(6), edgecolor='black', alpha=0.7)
ax.set_title('Peak Count Error Distribution')
ax.grid(True, alpha=0.3)

# 样本对比
ax = axes[1, 2]
sample_idx = 0
# 真实光谱
ax.plot(target_wavelengths, y_test[sample_idx], 'b-', label='True Spectrum', alpha=0.8)
# 从预测峰参数重建光谱
pred_spectrum = np.zeros_like(target_wavelengths, dtype=float)
for j in range(MAX_PEAKS):
    if pred_exist[sample_idx, j].item() > 0.5:
        pos = pred_params[sample_idx, j, 0].item() * (wl_max - wl_min) + wl_min
        intensity = pred_params[sample_idx, j, 1].item() * (intensity_max - intensity_min) + intensity_min
        fwhm = pred_params[sample_idx, j, 2].item() * (fwhm_max - fwhm_min) + fwhm_min
        pred_spectrum += intensity * gaussian(target_wavelengths, pos, fwhm)
        ax.axvline(pos, color='r', linestyle=':', alpha=0.5)
ax.plot(target_wavelengths, pred_spectrum, 'r--', label='Reconstructed from Pred Peaks', alpha=0.8)
ax.set_title(f'Sample {sample_idx}\nTrue Peaks: {int(peak_counts_test[sample_idx])}, Pred Peaks: {int(pred_count[sample_idx].item())}')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('multi_peak_training_v3_results.png', dpi=150)
print('已保存：multi_peak_training_v3_results.png')

# ============================================================================
# 11. 保存模型
# ============================================================================
print('\n保存模型...')

torch.save({
    'model_state_dict': model.state_dict(),
    'scaler_X': scaler_X,
    'wavelengths': target_wavelengths,
    'selected_biases': selected_biases,
    'peak_config': peak_config,
    'max_peaks': MAX_PEAKS,
    'wl_range': (wl_min, wl_max),
    'intensity_range': (intensity_min, intensity_max),
    'fwhm_range': (fwhm_min, fwhm_max),
    'multi_peak': True,
    'version': 'v3.3_end_to_end_peak',
}, 'model_multi_peak_v3.pth')

print('已保存：model_multi_peak_v3.pth')

# ============================================================================
# 12. 最终总结
# ============================================================================
print('\n' + '=' * 70)
print('多峰训练 V3.3 总结 - 方案三（端到端峰预测）')
print('=' * 70)

print(f"""
╔══════════════════════════════════════════════════════════════════════╗
║              方案三：端到端峰预测训练结果                              ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  【模型架构】                                                         ║
║    输入: 20 个测量值                                                  ║
║    输出: 峰数量 + 峰存在性 + 峰参数 (位置/强度/宽度)                   ║
║    参数量: {n_params:,}                                              ║
║                                                                      ║
║  【峰数量预测】                                                       ║
║    误差均值: {np.mean(all_count_errors):.2f}                                               ║
║    准确率: {np.sum(all_count_errors == 0) / len(all_count_errors) * 100:.1f}%                              ║
║                                                                      ║
║  【峰位置预测】                                                       ║
║    误差均值: {np.mean(all_position_errors):.2f} nm (中位数: {np.median(all_position_errors):.2f} nm)              ║
║    <1nm 精度: {np.sum(all_position_errors < 1) / len(all_position_errors) * 100:.1f}%                              ║
║    <2nm 精度: {np.sum(all_position_errors < 2) / len(all_position_errors) * 100:.1f}%                             ║
║    <5nm 精度: {np.sum(all_position_errors < 5) / len(all_position_errors) * 100:.1f}%                             ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
""")

print('=' * 70)