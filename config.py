"""
光谱重建项目配置文件
"""

# 数据文件路径配置
DATA_PATHS = {
    # 响应度矩阵文件（Excel格式）
    'response_matrix': '响应度矩阵.xlsx',
    
    # 模型检查点文件
    'model_checkpoint': 'model_multi_bias_v2.pth',
    
    # 输出目录
    'output_dir': 'outputs'
}

# 模型参数配置
MODEL_CONFIG = {
    'input_dim': 20,      # 20个偏压测量值
    'output_dim': 301,    # 301个波长点 (1000-1300nm, 1nm间隔)
    'hidden_dims': [512, 1024, 1024, 512],
    'dropout_rate': 0.2,
    'activation': 'relu',
    'output_activation': 'sigmoid'
}

# 训练参数配置
TRAINING_CONFIG = {
    'batch_size': 64,
    'learning_rate': 0.001,
    'weight_decay': 1e-4,
    'epochs': 600,
    'patience': 50,  # 早停耐心值
    'train_split': 0.8,
    'random_seed': 42
}

# 数据生成配置
DATA_GENERATION = {
    'wavelength_range': (1000, 1300),  # nm
    'wavelength_step': 1,              # nm
    'num_biases': 20,                  # 使用的偏压数量
    'bias_range': (-15, 0),            # V, 负偏压范围
    
    # 训练数据生成
    'n_main_samples': 6000,            # 主流样本
    'n_edge_samples': 2000,            # 边界过采样样本
    'edge_ranges': [(1000, 1050), (1250, 1300)],  # 边界区域
    
    # 高斯光谱参数
    'intensity_range': (0.5, 1.0),     # 峰值强度范围
    'fwhm_range': (20, 50)             # 半高宽范围 (nm)
}

# 损失函数权重配置
LOSS_WEIGHTS = {
    'low_boundary': 3.0,    # 1000-1050nm区域权重
    'high_boundary': 3.0,   # 1250-1300nm区域权重
    'middle_region': 1.0    # 1050-1250nm区域权重
}

# 分析配置
ANALYSIS_CONFIG = {
    'test_samples': 1000,               # 测试样本数量
    'error_thresholds': [1, 2, 5, 10], # 误差阈值 (nm)
    'plot_samples': 25,                 # 绘图样本数量
    'save_figures': True,               # 是否保存图表
    'figure_dpi': 300                   # 图表分辨率
}