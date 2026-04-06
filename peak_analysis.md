# 单峰光谱优化技术分析

## 📊 当前项目中的单峰优化技术

### 1. **当前使用的峰检测方法**

#### 1.1 简单argmax方法
```python
# 当前代码中的峰检测
peak_true = target_wavelengths[np.argmax(y_true, axis=1)]  # 真实峰值
peak_pred = target_wavelengths[np.argmax(y_pred, axis=1)]  # 预测峰值
peak_error = np.abs(peak_pred - peak_true)  # 误差计算
```

**优点：**
- 简单快速
- 对于单峰高斯光谱有效
- 计算复杂度低

**缺点：**
- 对噪声敏感
- 无法处理多峰光谱
- 无法精确定位（只能到1nm分辨率）
- 无法处理平坦峰或宽峰

#### 1.2 边界过采样策略
```python
# 边界区域增加训练样本
n_main = 6000   # 主流样本（1000-1300均匀）
n_edge = 2000   # 边界过采样（1000-1050和1250-1300）

# 边界样本1：1000-1050nm（低端边界）
peak_wavelengths_low = np.random.uniform(1000, 1050, n_edge)

# 边界样本2：1250-1300nm（高端边界）
peak_wavelengths_high = np.random.uniform(1250, 1300, n_edge)
```

**作用：** 提高边界区域的峰定位精度

#### 1.3 加权损失函数
```python
# 边界区域权重更高
weights = np.ones(len(target_wavelengths))
weights[target_wavelengths <= 1050] = 3.0   # 低边界3倍
weights[target_wavelengths >= 1250] = 3.0   # 高边界3倍

def weighted_mse_loss(pred, target, weights):
    return torch.mean(weights * (pred - target) ** 2)
```

**作用：** 强制模型更关注边界区域的精度

### 2. **当前技术的局限性**

#### 2.1 峰检测精度限制
- **分辨率限制**：只能检测到整数波长位置（1nm间隔）
- **子像素精度缺失**：无法检测峰在1nm内的精确位置
- **噪声敏感**：argmax对噪声和波动敏感

#### 2.2 光谱形状忽略
- 只关注峰值位置，忽略光谱形状
- 无法区分不同形状的峰（高斯、洛伦兹等）
- 无法处理峰宽变化

#### 2.3 多峰处理能力
- 完全无法处理多峰光谱
- argmax只能找到全局最大值
- 无法识别和定位多个峰

### 3. **先进的峰检测技术**

#### 3.1 子像素峰定位方法

**3.1.1 抛物线拟合法**
```python
def parabolic_peak_interp(spectrum, wavelengths):
    """抛物线插值提高峰定位精度"""
    idx = np.argmax(spectrum)
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
    return peak_x
```

**3.1.2 高斯拟合法**
```python
from scipy.optimize import curve_fit

def gaussian_fit_peak(spectrum, wavelengths, initial_guess=None):
    """高斯拟合精确定位峰"""
    def gaussian(x, amp, center, sigma, offset):
        return amp * np.exp(-((x - center) ** 2) / (2 * sigma ** 2)) + offset
    
    # 初始猜测
    if initial_guess is None:
        idx = np.argmax(spectrum)
        amp = spectrum[idx]
        center = wavelengths[idx]
        sigma = 10  # 初始猜测半高宽约23.5nm
        offset = np.min(spectrum)
        initial_guess = [amp, center, sigma, offset]
    
    try:
        # 边界约束
        bounds = ([0, wavelengths[0], 1, 0], 
                  [np.inf, wavelengths[-1], 100, np.inf])
        
        popt, _ = curve_fit(gaussian, wavelengths, spectrum, 
                           p0=initial_guess, bounds=bounds, maxfev=5000)
        
        amp, center, sigma, offset = popt
        fwhm = 2.355 * sigma  # 半高宽
        
        return {
            'center': center,      # 峰中心（子像素精度）
            'amplitude': amp,      # 峰高
            'fwhm': fwhm,          # 半高宽
            'offset': offset       # 基线
        }
    except:
        # 拟合失败，退回argmax
        idx = np.argmax(spectrum)
        return {'center': wavelengths[idx], 'amplitude': spectrum[idx], 'fwhm': np.nan, 'offset': 0}
```

#### 3.2 多峰检测技术

**3.2.1 基于scipy的find_peaks**
```python
from scipy.signal import find_peaks

def detect_multiple_peaks(spectrum, wavelengths, prominence=0.1, distance=10):
    """检测多个峰"""
    peaks, properties = find_peaks(spectrum, 
                                   prominence=prominence,  # 峰突出度
                                   distance=distance,       # 峰间最小距离
                                   height=0.1)             # 最小高度
    
    peak_info = []
    for i, idx in enumerate(peaks):
        # 子像素精确定位
        if idx > 0 and idx < len(spectrum) - 1:
            # 抛物线插值
            y1, y2, y3 = spectrum[idx-1], spectrum[idx], spectrum[idx+1]
            x1, x2, x3 = wavelengths[idx-1], wavelengths[idx], wavelengths[idx+1]
            denom = (y1 - 2*y2 + y3)
            if denom != 0:
                center = x2 + (x3 - x2) * (y1 - y3) / (2 * denom)
            else:
                center = x2
        else:
            center = wavelengths[idx]
        
        peak_info.append({
            'index': idx,
            'wavelength': center,
            'amplitude': spectrum[idx],
            'prominence': properties['prominences'][i] if 'prominences' in properties else 0,
            'width': properties['widths'][i] if 'widths' in properties else 0
        })
    
    # 按振幅排序
    peak_info.sort(key=lambda x: x['amplitude'], reverse=True)
    return peak_info
```

**3.2.2 连续小波变换**
```python
import pywt

def wavelet_peak_detection(spectrum, wavelengths, wavelet='mexh', scales=np.arange(1, 31)):
    """小波变换峰检测，对噪声更鲁棒"""
    # 连续小波变换
    coefficients, frequencies = pywt.cwt(spectrum, scales, wavelet)
    
    # 寻找局部极大值
    peaks = []
    for scale_idx in range(len(scales)):
        # 在当前尺度下寻找峰
        scale_coeffs = coefficients[scale_idx]
        local_maxima = argrelextrema(scale_coeffs, np.greater)[0]
        
        for idx in local_maxima:
            if scale_coeffs[idx] > np.percentile(scale_coeffs, 90):  # 只保留显著峰
                peaks.append({
                    'wavelength': wavelengths[idx],
                    'scale': scales[scale_idx],
                    'coefficient': scale_coeffs[idx]
                })
    
    # 聚类相近的峰
    peaks = cluster_peaks(peaks, wavelength_tol=2)
    return peaks
```

#### 3.3 峰形拟合技术

**3.3.1 多峰高斯拟合**
```python
def multi_gaussian_fit(spectrum, wavelengths, n_peaks=2):
    """多峰高斯拟合"""
    def multi_gauss(x, *params):
        """多个高斯函数的和"""
        y = np.zeros_like(x)
        for i in range(0, len(params), 3):
            amp, center, sigma = params[i:i+3]
            y += amp * np.exp(-((x - center) ** 2) / (2 * sigma ** 2))
        return y
    
    # 初始猜测
    initial_guess = []
    peak_indices = find_peaks(spectrum, prominence=0.1, distance=10)[0]
    
    for idx in peak_indices[:n_peaks]:  # 最多n_peaks个峰
        amp = spectrum[idx]
        center = wavelengths[idx]
        sigma = 10  # 初始猜测
        initial_guess.extend([amp, center, sigma])
    
    # 如果找到的峰不够，随机生成
    while len(initial_guess) < n_peaks * 3:
        amp = np.random.uniform(0.3, 1.0)
        center = np.random.uniform(wavelengths[0], wavelengths[-1])
        sigma = np.random.uniform(5, 20)
        initial_guess.extend([amp, center, sigma])
    
    try:
        popt, _ = curve_fit(multi_gauss, wavelengths, spectrum, 
                           p0=initial_guess, maxfev=10000)
        
        # 解析结果
        peaks = []
        for i in range(0, len(popt), 3):
            amp, center, sigma = popt[i:i+3]
            peaks.append({
                'amplitude': amp,
                'center': center,
                'fwhm': 2.355 * sigma,
                'area': amp * sigma * np.sqrt(2 * np.pi)  # 高斯面积
            })
        
        return peaks
    except:
        return None
```

**3.3.2 Voigt峰拟合**
```python
from scipy.special import wofz

def voigt_profile(x, amp, center, sigma, gamma):
    """Voigt函数：高斯和洛伦兹的卷积"""
    z = (x - center + 1j * gamma) / (sigma * np.sqrt(2))
    return amp * np.real(wofz(z)) / (sigma * np.sqrt(2 * np.pi))

def fit_voigt_peak(spectrum, wavelengths):
    """Voigt峰拟合，更接近真实光谱"""
    # 初始猜测
    idx = np.argmax(spectrum)
    amp = spectrum[idx]
    center = wavelengths[idx]
    sigma = 5   # 高斯宽度
    gamma = 5   # 洛伦兹宽度
    
    try:
        popt, _ = curve_fit(voigt_profile, wavelengths, spectrum,
                           p0=[amp, center, sigma, gamma],
                           bounds=([0, wavelengths[0], 0.1, 0.1],
                                   [np.inf, wavelengths[-1], 50, 50]))
        
        return {
            'amplitude': popt[0],
            'center': popt[1],
            'sigma_g': popt[2],  # 高斯宽度
            'gamma_l': popt[3],  # 洛伦兹宽度
            'fwhm': 0.5346 * popt[3] + np.sqrt(0.2166 * popt[3]**2 + popt[2]**2)  # Voigt半高宽
        }
    except:
        return None
```

### 4. **改进的峰还原技术**

#### 4.1 基于模型输出的峰还原

**4.1.1 直接峰参数预测**
```python
class PeakParameterNet(nn.Module):
    """直接预测峰参数的神经网络"""
    def __init__(self, input_dim=20, hidden_dim=256):
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
            nn.Linear(hidden_dim, 3)  # 输出：中心波长、振幅、半高宽
        )
    
    def forward(self, x):
        # 输出层使用不同的激活函数
        params = self.net(x)
        center = 1000 + 300 * torch.sigmoid(params[:, 0])  # 中心波长在1000-1300nm
        amplitude = torch.sigmoid(params[:, 1])            # 振幅在0-1
        fwhm = 20 + 30 * torch.sigmoid(params[:, 2])       # 半高宽在20-50nm
        return torch.stack([center, amplitude, fwhm], dim=1)
```

**4.1.2 高斯函数重建**
```python
def reconstruct_gaussian_from_params(params, wavelengths):
    """从参数重建高斯光谱"""
    center, amplitude, fwhm = params[:, 0], params[:, 1], params[:, 2]
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    
    # 批量计算高斯函数
    wavelengths_expanded = wavelengths[None, :, None]  # [1, n_wavelengths, 1]
    center_expanded = center[:, None, None]            # [batch, 1, 1]
    amplitude_expanded = amplitude[:, None, None]      # [batch, 1, 1]
    sigma_expanded = sigma[:, None, None]              # [batch, 1, 1]
    
    spectra = amplitude_expanded * torch.exp(
        -0.5 * ((wavelengths_expanded - center_expanded) / sigma_expanded) ** 2
    )
    
    return spectra.squeeze(-1)  # [batch, n_wavelengths]
```

#### 4.2 混合方法：光谱重建 + 峰检测

```python
class HybridPeakDetection:
    """混合峰检测：神经网络重建 + 后处理峰检测"""
    def __init__(self, spectrum_model, peak_detector='gaussian_fit'):
        self.spectrum_model = spectrum_model  # 光谱重建模型
        self.peak_detector = peak_detector
    
    def detect_peaks(self, measurements):
        """从测量值检测峰"""
        # 1. 重建光谱
        with torch.no_grad():
            spectra = self.spectrum_model(measurements)
            spectra_np = spectra.cpu().numpy()
        
        # 2. 峰检测
        peaks_list = []
        for spectrum in spectra_np:
            if self.peak_detector == 'argmax':
                # 简单argmax
                idx = np.argmax(spectrum)
                peak_wl = self.wavelengths[idx]
                peaks = [{'center': peak_wl, 'amplitude': spectrum[idx]}]
            
            elif self.peak_detector == 'parabolic':
                # 抛物线插值
                peak_wl = parabolic_peak_interp(spectrum, self.wavelengths)
                idx = np.argmin(np.abs(self.wavelengths - peak_wl))
                peaks = [{'center': peak_wl, 'amplitude': spectrum[idx]}]
            
            elif self.peak_detector == 'gaussian_fit':
                # 高斯拟合
                peak_info = gaussian_fit_peak(spectrum, self.wavelengths)
                peaks = [peak_info] if peak_info else []
            
            elif self.peak_detector == 'multi_peak':
                # 多峰检测
                peaks = detect_multiple_peaks(spectrum, self.wavelengths)
            
            peaks_list.append(peaks)
        
        return peaks_list
```

#### 4.3 端到端峰参数预测

```python
class EndToEndPeakNet(nn.Module):
    """端到端峰参数预测网络"""
    def __init__(self, input_dim=20, n_peaks=3):
        super().__init__()
        self.n_peaks = n_peaks
        
        # 共享特征提取层
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
        )
        
        # 每个峰的参数预测头
        self.peak_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 3)  # 中心波长、振幅、半高宽
            ) for _ in range(n_peaks)
        ])
        
        # 峰存在概率
        self.existence_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, n_peaks),
            nn.Sigmoid()  # 每个峰的存在概率
        )
    
    def forward(self, x):
        features = self.feature_extractor(x)
        
        # 预测每个峰的参数
        peak_params = []
        for head in self.peak_heads:
            params = head(features)
            # 应用约束
            center = 1000 + 300 * torch.sigmoid(params[:, 0])  # 1000-1300nm
            amplitude = torch.sigmoid(params[:, 1])            # 0-1
            fwhm = 20 + 30 * torch.sigmoid(params[:, 2])       # 20-50nm
            peak_params.append(torch.stack([center, amplitude, fwhm], dim=1))
        
        # 峰存在概率
        existence_probs = self.existence_head(features)
        
        return {
            'peak_params': torch.stack(peak_params, dim=1),  # [batch, n_peaks, 3]
            'existence_probs': existence_probs                # [batch, n_peaks]
        }
```

### 5. **损失函数改进**

#### 5.1 峰中心损失
```python
def peak_center_loss(pred_params, true_params, existence_probs=None):
    """峰中心位置损失"""
    # pred_params: [batch, n_peaks, 3] - 中心波长、振幅、半高宽
    # true_params: [batch, n_peaks, 3] - 真实参数
    
    if existence_probs is not None:
        # 加权损失，根据峰存在概率
        weights = existence_probs.unsqueeze(-1)  # [batch, n_peaks, 1]
        center_loss = torch.mean(weights * (pred_params[:, :, 0] - true_params[:, :, 0]) ** 2)
    else:
        center_loss = torch.mean((pred_params[:, :, 0] - true_params[:, :, 0]) ** 2)
    
    return center_loss
```

#### 5.2 多任务损失
```python
def multi_task_loss(pred_spectrum, true_spectrum, pred_params, true_params, 
                   wavelength_weights=None, alpha=0.7, beta=0.3):
    """多任务损失：光谱重建 + 峰参数预测"""
    # 光谱重建损失
    if wavelength_weights is not None:
        spectrum_loss = torch.mean(wavelength_weights * (pred_spectrum - true_spectrum) ** 2)
    else:
        spectrum_loss = torch.mean((pred_spectrum - true_spectrum) ** 2)
    
    # 峰参数损失
    center_loss = torch.mean((pred_params[:, :, 0] - true_params[:, :, 0]) ** 2)  # 中心波长
    amplitude_loss = torch.mean((pred_params[:, :, 1] - true_params[:, :, 1]) ** 2)  # 振幅
    fwhm_loss = torch.mean((pred_params[:, :, 2] - true_params[:, :, 2]) ** 2)  # 半高宽
    
    param_loss = center_loss + 0.5 * amplitude_loss + 0.2 * fwhm_loss
    
    # 总损失
    total_loss = alpha * spectrum_loss + beta * param_loss
    
    return total_loss, spectrum_loss, param_loss
```

### 6. **评估指标改进**

#### 6.1 峰检测评估指标
```python
def evaluate_peak_detection(pred_peaks, true_peaks, wavelength_tol=1.0, amplitude_tol=0.1):
    """评估峰检测性能"""
    metrics = {
        'center_mae': [],      # 中心波长平均绝对误差
        'center_rmse': [],     # 中心波长均方根误差
        'amplitude_mae': [],    # 振幅平均绝对误差
        'fwhm_mae': [],        # 半高宽平均绝对误差
        'detection_rate': [],   # 检测率
        'false_positive': [],   # 误报率
        'match_distance': []    # 匹配距离
    }
    
    for pred_list, true_list in zip(pred_peaks, true_peaks):
        # 匹配预测峰和真实峰
        matched = match_peaks(pred_list, true_list, wavelength_tol, amplitude_tol)
        
        if matched:
            for pred_peak, true_peak in matched:
                metrics['center_mae'].append(abs(pred_peak['center'] - true_peak['center']))
                metrics['amplitude_mae'].append(abs(pred_peak['amplitude'] - true_peak['amplitude']))
                metrics['fwhm_mae'].append(abs(pred_peak.get('fwhm', 0) - true_peak.get('fwhm', 0)))
                metrics['match_distance'].append(abs(pred_peak['center'] - true_peak['center']))
        
        # 检测率
        n_true = len(true_list)
        n_detected = len([p for p in pred_list if any(
            abs(p['center'] - t['center']) < wavelength_tol for t in true_list
        )])
        metrics['detection_rate'].append(n_detected / n_true if n_true > 0 else 0)
        
        # 误报率
        n_false = len([p for p in pred_list if all(
            abs(p['center'] - t['center']) >= wavelength_tol for t in true_list
        )])
        metrics['false_positive'].append(n_false / len(pred_list) if pred_list else 0)
    
    # 计算平均值
    result = {}
    for key, values in metrics.items():
        if values:
            result[key] = np.mean(values)
        else:
            result[key] = np.nan
    
    return result
```

#### 6.2 峰匹配算法
```python
def match_peaks(pred_peaks, true_peaks, wavelength_tol=1.0, amplitude_tol=0.1):
    """匹配预测峰和真实峰（匈牙利算法简化版）"""
    if not pred_peaks or not true_peaks:
        return []
    
    # 按振幅排序
    pred_peaks = sorted(pred_peaks, key=lambda x: x['amplitude'], reverse=True)
    true_peaks = sorted(true_peaks, key=lambda x: x['amplitude'], reverse=True)
    
    matched = []
    used_true = set()
    
    for pred_peak in pred_peaks:
        best_match = None
        best_distance = float('inf')
        
        for i, true_peak in enumerate(true_peaks):
            if i in used_true:
                continue
            
            # 计算距离（考虑波长和振幅）
            wavelength_dist = abs(pred_peak['center'] - true_peak['center'])
            amplitude_dist = abs(pred_peak['amplitude'] - true_peak['amplitude'])
            
            # 综合距离
            distance = wavelength_dist + 10 * amplitude_dist  # 波长权重更高
            
            if distance < best_distance and wavelength_dist < wavelength_tol:
                best_distance = distance
                best_match = (i, true_peak)
        
        if best_match is not None:
            idx, true_peak = best_match
            matched.append((pred_peak, true_peak))
            used_true.add(idx)
    
    return matched
```

### 7. **实施建议**

#### 7.1 短期改进（立即实施）
1. **抛物线插值**：替换简单的argmax，提高定位精度
2. **高斯拟合后处理**：在神经网络输出后添加高斯拟合
3. **多指标评估**：除了峰值误差，增加峰宽、振幅误差评估

#### 7.2 中期改进（1-2周）
1. **多峰检测**：实现find_peaks-based检测
2. **Voigt拟合**：更真实的光谱形状拟合
3. **混合模型**：光谱重建 + 后处理峰检测

#### 7.3 长期改进（1个月）
1. **端到端峰预测**：直接预测峰参数
2. **多任务学习**：同时优化光谱重建和峰参数
3. **不确定性估计**：预测峰位置的不确定性

#### 7.4 代码示例：立即改进
```python
# 替换当前的argmax峰检测
def improved_peak_detection(spectrum, wavelengths, method='parabolic'):
    """改进的峰检测"""
    if method == 'parabolic':
        return parabolic_peak_interp(spectrum, wavelengths)
    elif method == 'gaussian_fit':
        result = gaussian_fit_peak(spectrum, wavelengths)
        return result['center'] if result else wavelengths[np.argmax(spectrum)]
    else:
        return wavelengths[np.argmax(spectrum)]  # 退回argmax

# 在评估中使用
peak_pred = []
for spec in y_pred:
    peak_wl = improved_peak_detection(spec, target_wavelengths, method='parabolic')
    peak_pred.append(peak_wl)
peak_pred = np.array(peak_pred)
```

### 8. **总结**

**当前项目的单峰优化技术：**
1. ✅ **边界过采样**：提高边界区域精度
2. ✅ **加权损失**：边界区域3倍权重
3. ❌ **简单argmax**：只能检测整数波长位置
4. ❌ **无子像素精度**：无法精确定位
5. ❌ **无多峰处理**：只能处理单峰

**建议优先实施：**
1. **抛物线插值**：立即提高定位精度到子像素级别
2. **高斯拟合后处理**：提供峰宽和振幅信息
3. **多指标评估**：全面评估峰检测性能

**最终目标**：实现**端到端的峰参数预测**，直接输出中心波长、振幅、半高宽，而不仅仅是光谱曲线。