"""
单峰光谱优化技术改进实现
包含抛物线插值、高斯拟合、多峰检测等高级峰检测技术
"""

import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

# ============================================================================
# 1. 改进的峰检测方法
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
    
    # 抛物线顶点公式: x_peak = x2 + (x3 - x2) * (y1 - y3) / (2*(y1 - 2*y2 + y3))
    denom = (y1 - 2*y2 + y3)
    if denom == 0:
        return x2  # 退回线性插值
    
    peak_x = x2 + (x3 - x2) * (y1 - y3) / (2 * denom)
    
    # 确保在合理范围内
    peak_x = max(x1, min(peak_x, x3))
    
    return peak_x

def gaussian_fit_peak(spectrum, wavelengths, initial_guess=None):
    """
    高斯拟合精确定位峰参数
    返回：中心波长、振幅、半高宽、基线
    """
    def gaussian(x, amp, center, sigma, offset):
        """高斯函数"""
        return amp * np.exp(-((x - center) ** 2) / (2 * sigma ** 2)) + offset
    
    # 初始猜测
    if initial_guess is None:
        idx = np.argmax(spectrum)
        amp = spectrum[idx]
        center = wavelengths[idx]
        sigma = 10  # 初始猜测，对应半高宽约23.5nm
        offset = np.percentile(spectrum, 10)  # 使用10%分位数作为基线估计
        initial_guess = [amp, center, sigma, offset]
    
    try:
        # 边界约束
        bounds = ([0, wavelengths[0], 1, 0], 
                  [np.inf, wavelengths[-1], 50, np.percentile(spectrum, 90)])
        
        popt, pcov = curve_fit(gaussian, wavelengths, spectrum, 
                              p0=initial_guess, bounds=bounds, 
                              maxfev=5000, ftol=1e-8, xtol=1e-8)
        
        amp, center, sigma, offset = popt
        fwhm = 2.355 * sigma  # 半高宽 = 2.355 * sigma
        
        # 计算拟合优度
        y_fit = gaussian(wavelengths, *popt)
        r_squared = 1 - np.sum((spectrum - y_fit) ** 2) / np.sum((spectrum - np.mean(spectrum)) ** 2)
        
        return {
            'center': center,          # 峰中心（子像素精度）
            'amplitude': amp,          # 峰高
            'fwhm': fwhm,              # 半高宽
            'offset': offset,          # 基线
            'sigma': sigma,            # 标准差
            'r_squared': r_squared,    # 拟合优度
            'success': True
        }
    except Exception as e:
        # 拟合失败，退回简单方法
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

def detect_multiple_peaks(spectrum, wavelengths, min_height=0.1, min_distance=10, prominence=0.05):
    """
    检测多个峰
    使用scipy的find_peaks算法
    """
    try:
        peaks, properties = find_peaks(spectrum, 
                                      height=min_height,
                                      distance=min_distance,
                                      prominence=prominence)
        
        peak_info = []
        for i, idx in enumerate(peaks):
            # 抛物线插值提高精度
            if idx > 0 and idx < len(spectrum) - 1:
                y1, y2, y3 = spectrum[idx-1], spectrum[idx], spectrum[idx+1]
                x1, x2, x3 = wavelengths[idx-1], wavelengths[idx], wavelengths[idx+1]
                denom = (y1 - 2*y2 + y3)
                if denom != 0:
                    center = x2 + (x3 - x2) * (y1 - y3) / (2 * denom)
                else:
                    center = x2
            else:
                center = wavelengths[idx]
            
            # 计算半高宽
            half_max = spectrum[idx] / 2
            # 向左找半高宽点
            left_idx = idx
            while left_idx > 0 and spectrum[left_idx] > half_max:
                left_idx -= 1
            # 向右找半高宽点
            right_idx = idx
            while right_idx < len(spectrum) - 1 and spectrum[right_idx] > half_max:
                right_idx += 1
            
            # 线性插值精确半高宽位置
            if left_idx < idx and right_idx > idx:
                # 左侧插值
                if left_idx < len(spectrum) - 1:
                    x_left = wavelengths[left_idx]
                    x_left_next = wavelengths[left_idx + 1]
                    y_left = spectrum[left_idx]
                    y_left_next = spectrum[left_idx + 1]
                    if y_left_next != y_left:
                        left_fwhm = x_left + (x_left_next - x_left) * (half_max - y_left) / (y_left_next - y_left)
                    else:
                        left_fwhm = x_left
                else:
                    left_fwhm = wavelengths[left_idx]
                
                # 右侧插值
                if right_idx > 0:
                    x_right = wavelengths[right_idx]
                    x_right_prev = wavelengths[right_idx - 1]
                    y_right = spectrum[right_idx]
                    y_right_prev = spectrum[right_idx - 1]
                    if y_right_prev != y_right:
                        right_fwhm = x_right_prev + (x_right - x_right_prev) * (half_max - y_right_prev) / (y_right - y_right_prev)
                    else:
                        right_fwhm = x_right
                else:
                    right_fwhm = wavelengths[right_idx]
                
                fwhm = right_fwhm - left_fwhm
            else:
                fwhm = np.nan
            
            peak_info.append({
                'index': idx,
                'center': center,
                'amplitude': spectrum[idx],
                'fwhm': fwhm,
                'prominence': properties['prominences'][i] if 'prominences' in properties else 0,
                'width': properties['widths'][i] if 'widths' in properties else 0,
                'left_base': properties['left_bases'][i] if 'left_bases' in properties else idx,
                'right_base': properties['right_bases'][i] if 'right_bases' in properties else idx
            })
        
        # 按振幅排序
        peak_info.sort(key=lambda x: x['amplitude'], reverse=True)
        return peak_info
        
    except Exception as e:
        print(f"多峰检测失败: {e}")
        return []

# ============================================================================
# 2. 改进的评估指标
# ============================================================================

def evaluate_peak_detection_improved(y_true, y_pred, wavelengths, method='parabolic'):
    """
    改进的峰检测评估
    支持多种峰检测方法
    """
    n_samples = len(y_true)
    
    # 检测每个样本的峰
    true_peaks = []
    pred_peaks = []
    
    for i in range(n_samples):
        # 真实峰（已知生成参数）
        true_spectrum = y_true[i]
        true_peak_wl = wavelengths[np.argmax(true_spectrum)]
        true_peak_amp = np.max(true_spectrum)
        true_peaks.append({
            'center': true_peak_wl,
            'amplitude': true_peak_amp,
            'index': np.argmax(true_spectrum)
        })
        
        # 预测峰
        pred_spectrum = y_pred[i]
        
        if method == 'argmax':
            # 原始方法
            idx = np.argmax(pred_spectrum)
            pred_peak_wl = wavelengths[idx]
            pred_peak_amp = pred_spectrum[idx]
            
        elif method == 'parabolic':
            # 抛物线插值
            pred_peak_wl = parabolic_peak_interp(pred_spectrum, wavelengths)
            idx = np.argmin(np.abs(wavelengths - pred_peak_wl))
            pred_peak_amp = pred_spectrum[idx]
            
        elif method == 'gaussian_fit':
            # 高斯拟合
            result = gaussian_fit_peak(pred_spectrum, wavelengths)
            if result['success']:
                pred_peak_wl = result['center']
                pred_peak_amp = result['amplitude']
                idx = np.argmin(np.abs(wavelengths - pred_peak_wl))
            else:
                # 拟合失败，退回argmax
                idx = np.argmax(pred_spectrum)
                pred_peak_wl = wavelengths[idx]
                pred_peak_amp = pred_spectrum[idx]
                
        elif method == 'multi_peak':
            # 多峰检测（取主峰）
            peaks = detect_multiple_peaks(pred_spectrum, wavelengths)
            if peaks:
                main_peak = peaks[0]  # 振幅最大的峰
                pred_peak_wl = main_peak['center']
                pred_peak_amp = main_peak['amplitude']
                idx = main_peak['index']
            else:
                # 未检测到峰，退回argmax
                idx = np.argmax(pred_spectrum)
                pred_peak_wl = wavelengths[idx]
                pred_peak_amp = pred_spectrum[idx]
        
        else:
            raise ValueError(f"未知方法: {method}")
        
        pred_peaks.append({
            'center': pred_peak_wl,
            'amplitude': pred_peak_amp,
            'index': idx
        })
    
    # 计算误差
    true_centers = np.array([p['center'] for p in true_peaks])
    pred_centers = np.array([p['center'] for p in pred_peaks])
    
    errors = np.abs(pred_centers - true_centers)
    
    # 统计指标
    metrics = {
        'method': method,
        'mean_error': np.mean(errors),
        'median_error': np.median(errors),
        'std_error': np.std(errors),
        'max_error': np.max(errors),
        'min_error': np.min(errors),
        'accuracy_1nm': np.sum(errors < 1) / n_samples * 100,
        'accuracy_2nm': np.sum(errors < 2) / n_samples * 100,
        'accuracy_5nm': np.sum(errors < 5) / n_samples * 100,
        'accuracy_10nm': np.sum(errors < 10) / n_samples * 100,
        'n_samples': n_samples
    }
    
    return metrics, true_peaks, pred_peaks

# ============================================================================
# 3. 改进的神经网络模型（直接预测峰参数）
# ============================================================================

class PeakParameterNet(nn.Module):
    """直接预测峰参数的神经网络"""
    def __init__(self, input_dim=20, hidden_dim=256, output_dim=3):
        """
        input_dim: 输入维度（偏压数量）
        hidden_dim: 隐藏层维度
        output_dim: 输出维度（中心波长、振幅、半高宽）
        """
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim // 2),
            
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
        # 输出层特殊初始化
        self._initialize_weights()
    
    def _initialize_weights(self):
        """权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        params = self.net(x)
        
        # 应用约束
        # 中心波长: 1000-1300nm
        center = 1000 + 300 * torch.sigmoid(params[:, 0])
        
        # 振幅: 0-1
        amplitude = torch.sigmoid(params[:, 1])
        
        # 半高宽: 20-50nm
        fwhm = 20 + 30 * torch.sigmoid(params[:, 2])
        
        return torch.stack([center, amplitude, fwhm], dim=1)

class HybridSpectrumPeakNet(nn.Module):
    """混合模型：同时预测光谱和峰参数"""
    def __init__(self, input_dim=20, spectrum_dim=301, hidden_dim=512):
        super().__init__()
        
        # 共享特征提取
        self.shared_features = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.3),
        )
        
        # 光谱重建分支
        self.spectrum_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, spectrum_dim),
            nn.Sigmoid()  # 输出在[0,1]范围
        )
        
        # 峰参数预测分支
        self.peak_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Linear(hidden_dim // 2, 3)  # 中心波长、振幅、半高宽
        )
    
    def forward(self, x):
        features = self.shared_features(x)
        
        # 光谱重建
        spectrum = self.spectrum_head(features)
        
        # 峰参数预测
        params = self.peak_head(features)
        
        # 应用约束
        center = 1000 + 300 * torch.sigmoid(params[:, 0])  # 1000-1300nm
        amplitude = torch.sigmoid(params[:, 1])            # 0-1
        fwhm = 20 + 30 * torch.sigmoid(params[:, 2])       # 20-50nm
        
        peak_params = torch.stack([center, amplitude, fwhm], dim=1)
        
        return spectrum, peak_params

# ============================================================================
# 4. 改进的损失函数
# ============================================================================

def weighted_spectrum_loss(pred_spectrum, true_spectrum, weights=None):
    """加权光谱损失"""
    if weights is not None:
        return torch.mean(weights * (pred_spectrum - true_spectrum) ** 2)
    else:
        return torch.mean((pred_spectrum - true_spectrum) ** 2)

def peak_parameter_loss(pred_params, true_params, wavelength_weight=1.0, 
                       amplitude_weight=0.5, fwhm_weight=0.2):
    """峰参数损失"""
    # 中心波长损失
    center_loss = torch.mean((pred_params[:, 0] - true_params[:, 0]) ** 2)
    
    # 振幅损失
    amplitude_loss = torch.mean((pred_params[:, 1] - true_params[:, 1]) ** 2)
    
    # 半高宽损失
    fwhm_loss = torch.mean((pred_params[:, 2] - true_params[:, 2]) ** 2)
    
    # 加权组合
    total_loss = (wavelength_weight * center_loss + 
                  amplitude_weight * amplitude_loss + 
                  fwhm_weight * fwhm_loss)
    
    return total_loss, center_loss, amplitude_loss, fwhm_loss

def hybrid_loss(pred_spectrum, true_spectrum, pred_params, true_params, 
               spectrum_weight=0.7, peak_weight=0.3, wavelength_weights=None):
    """混合损失：光谱重建 + 峰参数预测"""
    # 光谱重建损失
    spectrum_loss = weighted_spectrum_loss(pred_spectrum, true_spectrum, wavelength_weights)
    
    # 峰参数损失
    peak_loss, center_loss, amplitude_loss, fwhm_loss = peak_parameter_loss(
        pred_params, true_params
    )
    
    # 总损失
    total_loss = spectrum_weight * spectrum_loss + peak_weight * peak_loss
    
    loss_details = {
        'total': total_loss.item() if isinstance(total_loss, torch.Tensor) else total_loss,
        'spectrum': spectrum_loss.item() if isinstance(spectrum_loss, torch.Tensor) else spectrum_loss,
        'peak': peak_loss.item() if isinstance(peak_loss, torch.Tensor) else peak_loss,
        'center': center_loss.item() if isinstance(center_loss, torch.Tensor) else center_loss,
        'amplitude': amplitude_loss.item() if isinstance(amplitude_loss, torch.Tensor) else amplitude_loss,
        'fwhm': fwhm_loss.item() if isinstance(fwhm_loss, torch.Tensor) else fwhm_loss
    }
    
    return total_loss, loss_details

# ============================================================================
# 5. 使用示例
# ============================================================================

def demonstrate_peak_improvements():
    """演示峰检测改进效果"""
    print("=" * 60)
    print("单峰光谱优化技术演示")
    print("=" * 60)
    
    # 生成测试数据
    np.random.seed(42)
    wavelengths = np.arange(1000, 1301, 1)
    
    # 创建测试光谱（添加噪声）
    true_center = 1150.5  # 真实峰在1150.5nm（非整数位置）
    true_amplitude = 0.8
    true_fwhm = 30
    
    # 生成高斯光谱
    sigma = true_fwhm / (2 * np.sqrt(2 * np.log(2)))
    true_spectrum = true_amplitude * np.exp(-0.5 * ((wavelengths - true_center) / sigma) ** 2)
    
    # 添加噪声
    noise = np.random.normal(0, 0.02, len(wavelengths))
    noisy_spectrum = true_spectrum + noise
    noisy_spectrum = np.clip(noisy_spectrum, 0, 1)
    
    print(f"\n真实峰参数:")
    print(f"  中心波长: {true_center:.2f} nm")
    print(f"  振幅: {true_amplitude:.3f}")
    print(f"  半高宽: {true_fwhm:.1f} nm")
    
    # 测试不同峰检测方法
    methods = ['argmax', 'parabolic', 'gaussian_fit']
    
    for method in methods:
        print(f"\n{method}方法:")
        
        if method == 'argmax':
            idx = np.argmax(noisy_spectrum)
            center = wavelengths[idx]
            amplitude = noisy_spectrum[idx]
            fwhm = np.nan
            
        elif method == 'parabolic':
            center = parabolic_peak_interp(noisy_spectrum, wavelengths)
            idx = np.argmin(np.abs(wavelengths - center))
            amplitude = noisy_spectrum[idx]
            fwhm = np.nan
            
        elif method == 'gaussian_fit':
            result = gaussian_fit_peak(noisy_spectrum, wavelengths)
            center = result['center']
            amplitude = result['amplitude']
            fwhm = result['fwhm']
        
        # 计算误差
        center_error = abs(center - true_center)
        amplitude_error = abs(amplitude - true_amplitude)
        
        print(f"  检测中心: {center:.2f} nm (误差: {center_error:.2f} nm)")
        print(f"  检测振幅: {amplitude:.3f} (误差: {amplitude_error:.3f})")
        if not np.isnan(fwhm):
            fwhm_error = abs(fwhm - true_fwhm)
            print(f"  检测半高宽: {fwhm:.1f} nm (误差: {fwhm_error:.1f} nm)")
    
    # 测试多峰检测
    print(f"\n多峰检测演示:")
    
    # 创建双峰光谱
    centers = [1100.3, 1200.7]
    amplitudes = [0.9, 0.6]
    fwhms = [25, 35]
    
    multi_spectrum = np.zeros_like(wavelengths)
    for c, a, f in zip(centers, amplitudes, fwhms):
        sigma = f / (2 * np.sqrt(2 * np.log(2)))
        multi_spectrum += a * np.exp(-0.5 * ((wavelengths - c) / sigma) ** 2)
    
    # 添加噪声
    multi_spectrum += np.random.normal(0, 0.02, len(wavelengths))
    multi_spectrum = np.clip(multi_spectrum, 0, 1)
    
    # 检测多峰
    peaks = detect_multiple_peaks(multi_spectrum, wavelengths, 
                                 min_height=0.1, min_distance=20, prominence=0.05)
    
    print(f"  检测到 {len(peaks)} 个峰:")
    for i, peak in enumerate(peaks):
        print(f"    峰{i+1}: 中心={peak['center']:.1f}nm, 振幅={peak['amplitude']:.3f}, 半高宽={peak['fwhm']:.1f}nm")
    
    # 可视化
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 单峰检测比较
    ax1 = axes[0, 0]
    ax1.plot(wavelengths, true_spectrum, 'b-', label='真实光谱', alpha=0.7)
    ax1.plot(wavelengths, noisy_spectrum, 'r--', label='带噪声光谱', alpha=0.7)
    ax1.axvline(true_center, color='b', linestyle=':', label=f'真实中心={true_center:.1f}nm')
    
    # 标记不同方法的检测结果
    colors = ['g', 'm', 'c']
    for method, color in zip(methods, colors):
        if method == 'argmax':
            idx = np.argmax(noisy_spectrum)
            center = wavelengths[idx]
        elif method == 'parabolic':
            center = parabolic_peak_interp(noisy_spectrum, wavelengths)
        elif method == 'gaussian_fit':
            result = gaussian_fit_peak(noisy_spectrum, wavelengths)
            center = result['center']
        
        ax1.axvline(center, color=color, linestyle='--', alpha=0.7, label=f'{method}={center:.1f}nm')
    
    ax1.set_xlabel('波长 (nm)')
    ax1.set_ylabel('强度')
    ax1.set_title('单峰检测方法比较')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 高斯拟合结果
    ax2 = axes[0, 1]
    result = gaussian_fit_peak(noisy_spectrum, wavelengths)
    if result['success']:
        # 绘制拟合曲线
        fitted = result['amplitude'] * np.exp(-0.5 * ((wavelengths - result['center']) / result['sigma']) ** 2) + result['offset']
        ax2.plot(wavelengths, true_spectrum, 'b-', label='真实光谱', alpha=0.7)
        ax2.plot(wavelengths, noisy_spectrum, 'r--', label='带噪声光谱', alpha=0.5)
        ax2.plot(wavelengths, fitted, 'g-', label='高斯拟合', alpha=0.8)
        ax2.axvline(true_center, color='b', linestyle=':', label=f'真实中心={true_center:.1f}nm')
        ax2.axvline(result['center'], color='g', linestyle='--', label=f'拟合中心={result["center"]:.1f}nm')
        ax2.fill_between(wavelengths, 0, fitted, where=(wavelengths >= result['center'] - result['fwhm']/2) & 
                        (wavelengths <= result['center'] + result['fwhm']/2), 
                        alpha=0.3, color='g', label=f'FWHM={result["fwhm"]:.1f}nm')
        ax2.set_title(f'高斯拟合 (R²={result["r_squared"]:.3f})')
    else:
        ax2.text(0.5, 0.5, '高斯拟合失败', ha='center', va='center', transform=ax2.transAxes)
    
    ax2.set_xlabel('波长 (nm)')
    ax2.set_ylabel('强度')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 多峰检测
    ax3 = axes[1, 0]
    ax3.plot(wavelengths, multi_spectrum, 'b-', label='双峰光谱', alpha=0.7)
    
    for i, (c, a, f) in enumerate(zip(centers, amplitudes, fwhms)):
        ax3.axvline(c, color='r', linestyle=':', alpha=0.7, label=f'真实峰{i+1}={c:.1f}nm' if i==0 else None)
    
    for i, peak in enumerate(peaks):
        ax3.axvline(peak['center'], color='g', linestyle='--', alpha=0.7, 
                   label=f'检测峰{i+1}={peak["center"]:.1f}nm' if i==0 else None)
        ax3.fill_between(wavelengths, 0, peak['amplitude'], 
                        where=(wavelengths >= peak['center'] - peak['fwhm']/2) & 
                              (wavelengths <= peak['center'] + peak['fwhm']/2), 
                        alpha=0.2, color='g')
    
    ax3.set_xlabel('波长 (nm)')
    ax3.set_ylabel('强度')
    ax3.set_title('多峰检测')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 误差分布
    ax4 = axes[1, 1]
    methods = ['argmax', 'parabolic', 'gaussian_fit']
    errors = []
    
    for method in methods:
        if method == 'argmax':
            idx = np.argmax(noisy_spectrum)
            center = wavelengths[idx]
        elif method == 'parabolic':
            center = parabolic_peak_interp(noisy_spectrum, wavelengths)
        elif method == 'gaussian_fit':
            result = gaussian_fit_peak(noisy_spectrum, wavelengths)
            center = result['center']
        
        error = abs(center - true_center)
        errors.append(error)
    
    bars = ax4.bar(methods, errors, color=['red', 'orange', 'green'])
    ax4.set_ylabel('中心波长误差 (nm)')
    ax4.set_title('不同方法的检测误差')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # 在柱子上添加误差值
    for bar, error in zip(bars, errors):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2, height + 0.01, 
                f'{error:.2f}nm', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('peak_detection_comparison.png', dpi=150, bbox_inches='tight')
    print(f"\n图表已保存: peak_detection_comparison.png")
    
    return {
        'true_center': true_center,
        'true_amplitude': true_amplitude,
        'true_fwhm': true_fwhm,
        'methods': methods,
        'errors': errors
    }

# ============================================================================
# 6. 集成到现有项目
# ============================================================================

def integrate_peak_improvements():
    """将峰检测改进集成到现有项目"""
    print("\n" + "=" * 60)
    print("集成建议")
    print("=" * 60)
    
    print("\n1. 立即改进（替换argmax）:")
    print("   - 在评估函数中使用parabolic_peak_interp()")
    print("   - 提高峰定位精度到子像素级别")
    
    print("\n2. 短期改进（1-2周）:")
    print("   - 添加gaussian_fit_peak()作为后处理")
    print("   - 提供峰宽和振幅信息")
    print("   - 实现多峰检测detect_multiple_peaks()")
    
    print("\n3. 中期改进（1个月）:")
    print("   - 实现PeakParameterNet直接预测峰参数")
    print("   - 使用HybridSpectrumPeakNet同时预测光谱和峰参数")
    print("   - 实现hybrid_loss多任务学习")
    
    print("\n4. 代码修改示例:")
    print("""
# 替换原来的argmax
# 原来:
peak_pred = target_wavelengths[np.argmax(y_pred, axis=1)]

# 改为:
peak_pred = []
for spectrum in y_pred:
    peak_wl = parabolic_peak_interp(spectrum, target_wavelengths)
    peak_pred.append(peak_wl)
peak_pred = np.array(peak_pred)
    """)
    
    print("\n5. 评估指标扩展:")
    print("   - 除了峰值误差，增加峰宽误差、振幅误差")
    print("   - 添加拟合优度R²指标")
    print("   - 实现多峰检测的评估指标")

if __name__ == "__main__":
    # 演示峰检测改进
    results = demonstrate_peak_improvements()
    
    # 显示集成建议
    integrate_peak_improvements()
    
    print("\n" + "=" * 60)
    print("总结")
    print("=" * 60)
    print("\n当前项目的单峰优化技术:")
    print("  ✅ 边界过采样 - 提高边界区域精度")
    print("  ✅ 加权损失 - 边界区域3倍权重")
    print("  ❌ 简单argmax - 只能检测整数波长位置")
    print("  ❌ 无子像素精度 - 无法精确定位")
    
    print("\n建议的改进:")
    print("  1. 抛物线插值: 立即将精度从1nm提高到0.1nm级别")
    print("  2. 高斯拟合: 提供峰宽和振幅信息")
    print("  3. 多峰检测: 为处理真实光谱做准备")
    print("  4. 端到端预测: 直接输出峰参数")
    
    print("\n预期效果:")
    print("  - 峰定位精度: 从1nm提高到0.1nm级别")
    print("  - 额外信息: 获得峰宽、振幅等参数")
    print("  - 鲁棒性: 对噪声更鲁棒")
    print("  - 扩展性: 支持多峰光谱")