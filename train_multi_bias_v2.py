"""
改进版多偏压训练
修正问题：
1. 偏压范围：-15V到0V（只用负偏压）
2. 边界过采样：1000-1050nm和1250-1300nm区域增加样本
3. 模型容量：扩大网络到100万参数
4. 加权损失：对边界区域增加权重
5. 训练策略：增加训练轮数，学习率调度
"""
import numpy as np
import pandas as pd
import h5py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt

print('=' * 60)
print('多偏压训练 V2 - 修正版')
print('=' * 60)

# ============================================================================
# 1. 加载响应度（只使用负偏压）
# ============================================================================
print('\n1. 加载响应度...')

excel_path = r'D:\desktop\try\响应度矩阵.xlsx'
df = pd.read_excel(excel_path)
wavelengths_resp = df.iloc[:, 0].values
response_matrix_full = df.iloc[:, 1:].values

# 65个偏压，电压从-15到0V（排除正偏压）
all_biases = np.linspace(-15, 0, 65)
print(f'总偏压数: {len(all_biases)}')
print(f'偏压范围: {all_biases[0]}V 到 {all_biases[-1]}V')

# ============================================================================
# 2. 选择20个负偏压（等间隔）
# ============================================================================
print('\n2. 选择20个负偏压...')

indices = np.linspace(0, 64, 20, dtype=int)
selected_biases = all_biases[indices]
selected_indices = indices.tolist()

print(f'选择20个偏压: {[f"{b:.2f}" for b in selected_biases]}')

# ============================================================================
# 3. 插值响应度到1nm（1000-1300nm）
# ============================================================================
print('\n3. 插值响应度...')

target_wavelengths = np.arange(1000, 1301, 1)
response_20 = np.zeros((len(target_wavelengths), 20))

from scipy.interpolate import UnivariateSpline
from scipy.ndimage import gaussian_filter1d

for j, idx in enumerate(selected_indices):
    orig_resp = response_matrix_full[:, idx]
    spline = UnivariateSpline(wavelengths_resp, orig_resp, s=0.001, k=3)
    interpolated = spline(target_wavelengths)
    smoothed = gaussian_filter1d(interpolated, sigma=1.5)
    response_20[:, j] = smoothed

print(f'响应度矩阵: {response_20.shape}')

# ============================================================================
# 4. 生成训练数据（边界过采样）
# ============================================================================
print('\n4. 生成训练数据（边界过采样）...')

n_main = 6000   # 主流样本（1000-1300均匀）
n_edge = 2000   # 边界过采样（1000-1050和1250-1300）

# 主流样本：均匀分布
np.random.seed(42)
peak_wavelengths_main = np.random.uniform(1000, 1300, n_main)
peak_intensities_main = np.random.uniform(0.5, 1.0, n_main)
fwhms_main = np.random.uniform(20, 50, n_main)

# 边界样本1：1000-1050nm（低端边界）
peak_wavelengths_low = np.random.uniform(1000, 1050, n_edge)
peak_intensities_low = np.random.uniform(0.5, 1.0, n_edge)
fwhms_low = np.random.uniform(20, 50, n_edge)

# 边界样本2：1250-1300nm（高端边界）
peak_wavelengths_high = np.random.uniform(1250, 1300, n_edge)
peak_intensities_high = np.random.uniform(0.5, 1.0, n_edge)
fwhms_high = np.random.uniform(20, 50, n_edge)

# 合并
peak_wavelengths = np.concatenate([peak_wavelengths_main, peak_wavelengths_low, peak_wavelengths_high])
peak_intensities = np.concatenate([peak_intensities_main, peak_intensities_low, peak_intensities_high])
fwhms = np.concatenate([fwhms_main, fwhms_low, fwhms_high])

n_samples = len(peak_wavelengths)
print(f'总样本数: {n_samples}')
print(f'  主流: {n_main}, 低边界: {n_edge}, 高边界: {n_edge}')

# 生成光谱
spectra_focus = np.zeros((n_samples, len(target_wavelengths)))
for i in range(n_samples):
    wl = target_wavelengths
    spectrum = peak_intensities[i] * np.exp(-0.5 * ((wl - peak_wavelengths[i]) / (fwhms[i] / 2.355)) ** 2)
    spectra_focus[i] = spectrum

# 计算测量值
measurements_20 = np.zeros((n_samples, 20))
for i in range(n_samples):
    for j in range(20):
        measurements_20[i, j] = np.sum(spectra_focus[i] * response_20[:, j]) * 1

print(f'光谱: {spectra_focus.shape}')
print(f'测量值: {measurements_20.shape}')

# 检查峰值分布
print(f'\n峰值波长分布:')
print(f'  1000-1050nm: {np.sum(peak_wavelengths < 1050)}')
print(f'  1050-1250nm: {np.sum((peak_wavelengths >= 1050) & (peak_wavelengths < 1250))}')
print(f'  1250-1300nm: {np.sum(peak_wavelengths >= 1250)}')

# ============================================================================
# 5. 数据准备
# ============================================================================
print('\n5. 数据准备...')

scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(measurements_20)

scaler_y = MinMaxScaler()
y_scaled = scaler_y.fit_transform(spectra_focus)

# 创建边界权重
# 边界区域权重更高
weights = np.ones(len(target_wavelengths))
edge_low = (target_wavelengths <= 1050)
edge_high = (target_wavelengths >= 1250)
weights[edge_low] = 3.0  # 低边界权重3倍
weights[edge_high] = 3.0  # 高边界权重3倍

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# 转换为tensor
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.FloatTensor(y_train)
X_test_tensor = torch.FloatTensor(X_test)
y_test_tensor = torch.FloatTensor(y_test)

# 边界权重也转为tensor
weights_tensor = torch.FloatTensor(weights)

print(f'训练集: {X_train.shape[0]}条')

# ============================================================================
# 6. 定义更大的模型
# ============================================================================
print('\n6. 定义模型...')

class MultiBiasNetV2(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            # 增大模型容量
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

output_dim = len(target_wavelengths)
model = MultiBiasNetV2(20, output_dim)

# 计算参数量
n_params = sum(p.numel() for p in model.parameters())
print(f'参数量: {n_params:,}')

# ============================================================================
# 7. 训练（带加权损失和学习率调度）
# ============================================================================
print('\n7. 训练...')

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# 带权重的MSE损失
def weighted_mse_loss(pred, target, weights):
    return torch.mean(weights * (pred - target) ** 2)

# 使用AdamW + 余弦退火
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2)

epochs = 600  # 增加训练轮数
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
        
        # 加权损失
        loss = weighted_mse_loss(pred, batch_y, weights_tensor)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        train_loss += loss.item()
    
    scheduler.step()
    train_loss /= len(train_loader)
    train_losses.append(train_loss)
    
    # 测试集评估
    model.eval()
    with torch.no_grad():
        test_pred = model(X_test_tensor)
        # 测试集用普通MSE
        test_loss = torch.mean((test_pred - y_test_tensor) ** 2).item()
    test_losses.append(test_loss)
    
    if test_loss < best_test_loss:
        best_test_loss = test_loss
        best_model_state = model.state_dict().copy()
    
    if (epoch + 1) % 100 == 0:
        print(f'Epoch {epoch+1}, Train: {train_loss:.6f}, Test MSE: {test_loss:.6f}, LR: {scheduler.get_last_lr()[0]:.6f}')

model.load_state_dict(best_model_state)
print(f'\n最佳损失: {best_test_loss:.6f}')

# ============================================================================
# 8. 评估
# ============================================================================
print('\n8. 评估...')

model.eval()
with torch.no_grad():
    y_pred_scaled = model(X_test_tensor).numpy()

y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_true = scaler_y.inverse_transform(y_test)
y_pred = np.clip(y_pred, 0, None)

# 峰值位置误差
peak_true = target_wavelengths[np.argmax(y_true, axis=1)]
peak_pred = target_wavelengths[np.argmax(y_pred, axis=1)]
peak_error = np.abs(peak_pred - peak_true)

print(f'\n峰值位置误差:')
print(f'  平均: {np.mean(peak_error):.2f}nm')
print(f'  中位数: {np.median(peak_error):.2f}nm')
print(f'  最大: {np.max(peak_error):.2f}nm')
print(f'  <1nm: {np.sum(peak_error < 1) / len(peak_error) * 100:.1f}%')
print(f'  <2nm: {np.sum(peak_error < 2) / len(peak_error) * 100:.1f}%')
print(f'  <5nm: {np.sum(peak_error < 5) / len(peak_error) * 100:.1f}%')
print(f'  <10nm: {np.sum(peak_error < 10) / len(peak_error) * 100:.1f}%')

# 分区域统计
print(f'\n分区域误差:')
mask_low = peak_true < 1050
mask_mid = (peak_true >= 1050) & (peak_true < 1250)
mask_high = peak_true >= 1250
print(f'  1000-1050nm: 平均{np.mean(peak_error[mask_low]):.2f}nm')
print(f'  1050-1250nm: 平均{np.mean(peak_error[mask_mid]):.2f}nm')
print(f'  1250-1300nm: 平均{np.mean(peak_error[mask_high]):.2f}nm')

# ============================================================================
# 9. 可视化
# ============================================================================
print('\n9. 可视化...')

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# 训练曲线
ax1 = axes[0, 0]
ax1.plot(train_losses, label='Train')
ax1.plot(test_losses, label='Test')
ax1.set_title('Training Curves')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 误差分布
ax2 = axes[0, 1]
ax2.hist(peak_error, bins=30, edgecolor='black', alpha=0.7)
ax2.axvline(1, color='g', linestyle='--', label='1nm')
ax2.axvline(5, color='r', linestyle='--', label='5nm')
ax2.set_title(f'Peak Error Distribution')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 分区域误差
ax3 = axes[0, 2]
regions = ['1000-1050\n(Low)', '1050-1250\n(Mid)', '1250-1300\n(High)']
region_errors = [np.mean(peak_error[mask_low]), np.mean(peak_error[mask_mid]), np.mean(peak_error[mask_high])]
ax3.bar(regions, region_errors, color=['red', 'green', 'red'])
ax3.set_title('Error by Region')
ax3.grid(True, alpha=0.3)

# 样本对比
sample_indices = [0, 500, 1000]
for i, idx in enumerate(sample_indices):
    ax = axes[1, i]
    ax.plot(target_wavelengths, y_true[idx], 'b-', label='True', alpha=0.8)
    ax.plot(target_wavelengths, y_pred[idx], 'r--', label='Pred', alpha=0.8)
    ax.set_title(f'Error: {peak_error[idx]:.1f}nm\nTrue: {peak_true[idx]}nm, Pred: {peak_pred[idx]}nm')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1000, 1300)

plt.tight_layout()
plt.savefig('training_multi_bias_v2.png', dpi=150)
print('结果: training_multi_bias_v2.png')

# ============================================================================
# 10. 保存
# ============================================================================
print('\n10. 保存...')

torch.save({
    'model_state_dict': model.state_dict(),
    'scaler_X': scaler_X,
    'scaler_y': scaler_y,
    'wavelengths': target_wavelengths,
    'selected_biases': selected_biases,
    'weights': weights,
}, 'model_multi_bias_v2.pth')

print('\n' + '=' * 60)
print(f'使用{len(selected_biases)}个负偏压（-15V到0V）')
print(f'峰值误差: 平均{np.mean(peak_error):.2f}nm, 中位数{np.median(peak_error):.2f}nm')
print(f'<5nm精度: {np.sum(peak_error < 5) / len(peak_error) * 100:.1f}%')
print('=' * 60)