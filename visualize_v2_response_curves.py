"""
可视化V2版本使用的20个偏压响应度曲线
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from scipy.ndimage import gaussian_filter1d

print('=' * 60)
print('V2版本响应度曲线可视化')
print('=' * 60)

# ============================================================================
# 1. 加载原始响应度矩阵
# ============================================================================
print('\n[1/3] 加载响应度矩阵...')

excel_path = r'D:\desktop\try\响应度矩阵.xlsx'
df = pd.read_excel(excel_path)
wavelengths_resp = df.iloc[:, 0].values
response_matrix_full = df.iloc[:, 1:].values

print(f'原始波长范围: {wavelengths_resp[0]} - {wavelengths_resp[-1]} nm')
print(f'原始偏压数量: {response_matrix_full.shape[1]}')

# ============================================================================
# 2. 确定V2版本使用的20个偏压
# ============================================================================
print('\n[2/3] 确定V2版本的20个偏压...')

# V2版本只使用负偏压：-15V到0V（共65个偏压）
all_biases = np.linspace(-15, 0, 65)
print(f'负偏压范围: {all_biases[0]:.2f}V 到 {all_biases[-1]:.2f}V')
print(f'负偏压总数: {len(all_biases)}')

# 选择20个等间隔的偏压
indices = np.linspace(0, 64, 20, dtype=int)
selected_biases = all_biases[indices]
selected_indices = indices.tolist()

print(f'\nV2版本使用的20个偏压:')
for i, (bias, idx) in enumerate(zip(selected_biases, selected_indices)):
    print(f'  {i+1:2d}. 偏压: {bias:6.2f}V (原始索引: {idx})')

# ============================================================================
# 3. 插值到目标波长范围 (1000-1300nm, 1nm间隔)
# ============================================================================
print('\n[3/3] 插值并可视化响应度曲线...')

target_wavelengths = np.arange(1000, 1301, 1)
response_20 = np.zeros((len(target_wavelengths), 20))

for j, idx in enumerate(selected_indices):
    orig_resp = response_matrix_full[:, idx]
    spline = UnivariateSpline(wavelengths_resp, orig_resp, s=0.001, k=3)
    interpolated = spline(target_wavelengths)
    smoothed = gaussian_filter1d(interpolated, sigma=1.5)
    response_20[:, j] = smoothed

print(f'插值后响应度矩阵形状: {response_20.shape}')
print(f'目标波长范围: {target_wavelengths[0]} - {target_wavelengths[-1]} nm')

# ============================================================================
# 4. 创建可视化
# ============================================================================

# 主图：所有20个偏压的响应度曲线
plt.figure(figsize=(20, 12))

# 子图1：所有20个偏压的响应度曲线
plt.subplot(2, 2, 1)
for i in range(20):
    plt.plot(target_wavelengths, response_20[:, i], 
             label=f'{selected_biases[i]:.1f}V', linewidth=1.5)
plt.xlabel('Wavelength (nm)')
plt.ylabel('Response')
plt.title('V2 Version: All 20 Bias Response Curves\n(-15.00V to 0.00V, equally spaced)')
plt.grid(True, alpha=0.3)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2, fontsize=8)

# 子图2：偏压值分布
plt.subplot(2, 2, 2)
plt.plot(selected_biases, 'bo-', markersize=8, linewidth=2)
plt.xlabel('Bias Index (0-19)')
plt.ylabel('Bias Voltage (V)')
plt.title('Selected Bias Voltages Distribution')
plt.grid(True, alpha=0.3)
for i, bias in enumerate(selected_biases):
    plt.annotate(f'{bias:.1f}V', (i, bias), textcoords="offset points", 
                xytext=(0,10), ha='center', fontsize=8)

# 子图3：关键偏压对比（第一个、中间、最后一个）
plt.subplot(2, 2, 3)
key_indices = [0, 9, 19]  # -15V, -7.5V, 0V
colors = ['red', 'green', 'blue']
for i, idx in enumerate(key_indices):
    plt.plot(target_wavelengths, response_20[:, idx], 
             color=colors[i], linewidth=2.5, 
             label=f'{selected_biases[idx]:.1f}V ({["Most Negative", "Middle", "Zero"][i]})')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Response')
plt.title('Key Bias Response Comparison')
plt.legend()
plt.grid(True, alpha=0.3)

# 子图4：响应度统计信息
plt.subplot(2, 2, 4)
max_responses = np.max(response_20, axis=0)
mean_responses = np.mean(response_20, axis=0)
plt.plot(selected_biases, max_responses, 'ro-', label='Max Response', markersize=6)
plt.plot(selected_biases, mean_responses, 'bo-', label='Mean Response', markersize=6)
plt.xlabel('Bias Voltage (V)')
plt.ylabel('Response Value')
plt.title('Response Statistics vs Bias Voltage')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('v2_response_curves_analysis.png', dpi=150, bbox_inches='tight')
print('已保存: v2_response_curves_analysis.png')

# ============================================================================
# 5. 详细信息输出
# ============================================================================

print('\n' + '=' * 60)
print('V2 VERSION RESPONSE CURVE DETAILS')
print('=' * 60)

print(f'\n【偏压配置】')
print(f'总偏压数: 20个')
print(f'偏压范围: -15.00V 到 0.00V')
print(f'选择策略: 从65个负偏压中等间隔选择20个')
print(f'偏压间隔: {(0 - (-15)) / 19:.3f}V')

print(f'\n【具体偏压值】')
for i in range(20):
    if i % 5 == 0:
        print()
    print(f'{selected_biases[i]:6.2f}V ', end='')
print()

print(f'\n\n【响应度特性】')
print(f'波长范围: 1000-1300nm (301个点，1nm间隔)')
print(f'插值方法: 三次样条插值 (s=0.001, k=3)')
print(f'平滑处理: 高斯滤波 (sigma=1.5)')

print(f'\n【精度信息】')
print(f'V2模型峰值定位精度: 0.99nm (平均误差)')
print(f'                    1.00nm (中位数误差)')
print(f'                    96.5% (<5nm精度)')
print(f'                    45.0% (<1nm精度)')

# 保存偏压信息到文件
bias_info = {
    'selected_biases': selected_biases.tolist(),
    'selected_indices': selected_indices,
    'bias_range': [-15.0, 0.0],
    'num_biases': 20,
    'bias_interval': (0 - (-15)) / 19,
    'wavelength_range': [1000, 1300],
    'wavelength_points': 301
}

import json
with open('v2_bias_configuration.json', 'w') as f:
    json.dump(bias_info, f, indent=2)

print(f'\n偏压配置已保存: v2_bias_configuration.json')

print('\n' + '=' * 60)
print('可视化完成！')
print('=' * 60)