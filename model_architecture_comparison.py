"""
模型架构对比示意图生成器
"""

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']

from matplotlib.patches import Rectangle, FancyBboxPatch

def create_model_diagram():
    """创建模型架构对比图"""
    
    fig = plt.figure(figsize=(20, 14))
    fig.patch.set_facecolor('white')
    fig.suptitle('光谱重建模型架构演进对比', fontsize=22, fontweight='bold', y=0.97)
    
    gs = fig.add_gridspec(2, 2, hspace=0.25, wspace=0.25)
    
    # ==================== V1 模型 ====================
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.set_facecolor('white')
    ax1.axis('off')
    
    ax1.text(5, 9.6, 'V1 模型 (基础版)', fontsize=15, fontweight='bold', ha='center')
    ax1.text(5, 9.25, '参数量~70K | 100 轮 | lr=0.001 | Adam', fontsize=9, ha='center')
    
    # V1 网络层
    layers_y = [8.2, 7.0, 5.8, 4.6, 3.2]
    layers_name = ['输入层', '隐藏层 1', '隐藏层 2', '隐藏层 3', '输出层']
    layers_dim = ['20 维', '128 维', '256 维', '128 维', '301 维']
    colors = ['lightblue', 'lightgreen', 'lightgreen', 'lightgreen', 'lightcoral']
    
    for i, (y, name, dim, color) in enumerate(zip(layers_y, layers_name, layers_dim, colors)):
        rect = FancyBboxPatch((2.5, y-0.3), 5, 0.6, boxstyle="round,pad=0.1",
                             facecolor=color, edgecolor='black', linewidth=1.5)
        ax1.add_patch(rect)
        ax1.text(5, y, f'{name}  ({dim})', fontsize=11, ha='center', va='center')
    
    # V1 说明
    ax1.text(0.2, 6.8, '算法结构:\n• 输入：20 个偏压值 (含正偏压)\n• 3 层全连接：128→256→128\n• 激活：ReLU\n• 损失：MSE', 
             fontsize=9, ha='left', va='top', bbox=dict(boxstyle="round,pad=0.2", facecolor='lightyellow'))
    
    ax1.text(0.2, 4.8, '性能:\n• 平均误差：5-10nm\n• <5nm 精度：~60%', 
             fontsize=9, ha='left', va='top', bbox=dict(boxstyle="round,pad=0.2", facecolor='lightpink'))
    
    ax1.text(0.2, 2.8, '问题:\n1. 正偏压噪声\n2. 边界样本不足\n3. 模型容量不足\n4. 无加权损失', 
             fontsize=9, ha='left', va='top', bbox=dict(boxstyle="round,pad=0.2", facecolor='mistyrose'))
    
    # ==================== V2 模型 ====================
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.set_facecolor('white')
    ax2.axis('off')
    
    ax2.text(5, 9.6, 'V2 模型 (黄金标准)', fontsize=15, fontweight='bold', ha='center')
    ax2.text(5, 9.25, '1,052,981 参数 | 600 轮 | lr=0.001 | AdamW+CosineAnnealing', fontsize=9, ha='center')
    
    # V2 网络架构 - 水平流程
    v2_x = [0.8, 2.6, 4.4, 6.2, 8.0]
    v2_names = ['输入\n20 维', 'L1\n20→512', 'L2\n512→1024', 'L3\n1024→1024', 'L4\n1024→512', '输出\n301 维']
    v2_colors = ['lightblue', 'lightgreen', 'lightgreen', 'lightgreen', 'lightgreen', 'lightcoral']
    
    for i, (x, name, color) in enumerate(zip(v2_x, v2_names, v2_colors)):
        w = 1.3 if i < 5 else 1.2
        rect = FancyBboxPatch((x, 7.8), w, 0.9, boxstyle="round,pad=0.1",
                             facecolor=color, edgecolor='black', linewidth=1.5)
        ax2.add_patch(rect)
        ax2.text(x+w/2, 8.25, name, fontsize=9, ha='center', va='center')
        if i < 4:
            ax2.annotate('', xy=(x+w+0.1, 8.25), xytext=(x+w-0.05, 8.25),
                        arrowprops=dict(arrowstyle='->', lw=1.5))
    
    # V2 正则化说明
    ax2.text(5, 7.0, 'ReLU + BatchNorm1d + Dropout(0.2)', fontsize=10, ha='center',
             bbox=dict(boxstyle="round,pad=0.15", facecolor='lightgreen', edgecolor='darkgreen'))
    
    ax2.text(5, 6.3, '输出层：Sigmoid 激活 → 301 波长点响应曲线', fontsize=9, ha='center',
             bbox=dict(boxstyle="round,pad=0.15", facecolor='lightcoral', edgecolor='darkred'))
    
    # V2 改进
    ax2.text(0.2, 5.7, '核心改进:\n1. 负偏压选择 (-15V~0V)\n2. 边界过采样 (各 2000 样本)\n3. 模型扩容 (70K→1M)\n4. 加权损失 (边界 x3)\n5. 充分训练 (600 轮)', 
             fontsize=9, ha='left', va='top', bbox=dict(boxstyle="round,pad=0.2", facecolor='lightyellow'))
    
    # V2 性能
    ax2.text(5.5, 4.8, '性能指标:\n平均误差 0.99nm | <1nm:45% | <5nm:96.5% | <10nm:99.4%', 
             fontsize=10, ha='center', va='top', fontweight='bold',
             bbox=dict(boxstyle="round,pad=0.2", facecolor='lightgreen', edgecolor='darkgreen', linewidth=2))
    
    ax2.text(5.5, 3.5, '分区域误差:\n1000-1050nm: 0.72nm | 1050-1250nm: 0.69nm | 1250-1300nm: 1.66nm', 
             fontsize=9, ha='center', va='top',
             bbox=dict(boxstyle="round,pad=0.2", facecolor='lightblue'))
    
    ax2.text(5.5, 2.2, '参数量:\nL1:10K | L2:525K | L3:1050K | L4:525K | Out:154K\n总计：1,052,981', 
             fontsize=8, ha='center', va='top',
             bbox=dict(boxstyle="round,pad=0.2", facecolor='lightyellow'))
    
    # ==================== V3 模型 ====================
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_xlim(0, 10)
    ax3.set_ylim(0, 10)
    ax3.set_facecolor('white')
    ax3.axis('off')
    
    ax3.text(5, 9.6, 'V3 模型探索 (多峰场景)', fontsize=15, fontweight='bold', ha='center')
    ax3.text(5, 9.3, '尝试多种架构但效果均不理想', fontsize=11, ha='center', style='italic')
    
    # V3 变体
    v3_items = [
        (0.2, 8.4, 'V3-A 基础架构', '20→512→1024→1024→512→301\nReLU+BN+Dropout | MSE+ 边界加权\n误差 7.44nm, <5nm~35%\n问题：无法区分峰数量'),
        (5.2, 8.4, 'V3-B 更深架构', '20→512→1024→1024→1024→512→301\n增加一层 1024\n误差 8.12nm, <5nm~32%\n问题：梯度消失'),
        (0.2, 6.5, 'V3-C 自注意力', '20→512→SelfAttention(8h)→1024→512→301\n多头注意力捕捉全局依赖\n误差 7.89nm, <5nm~33%\n问题：未有效捕捉峰特征'),
        (5.2, 6.5, 'V3-D 残差连接', '20→512→[ResBlock(1024)x2]→512→301\n残差块缓解梯度消失\n误差 7.56nm, <5nm~34%\n问题：改善有限'),
        (2.7, 4.5, 'V3-E 多任务学习', '共享编码器 (20→512→1024) + 双输出头\n  • 峰数量分类 (5 类)\n  • 光谱重建 (301 维)\n联合训练：L=L_spectrum+0.5*L_peak\n分类~65%, 光谱误差 7.21nm', 4.5),
    ]
    
    for item in v3_items:
        x, y, title, desc = item[0], item[1], item[2], item[3]
        w = 4.3 if len(item) == 4 else item[4]
        ax3.text(x, y, title, fontsize=11, fontweight='bold', ha='left')
        ax3.text(x, y-0.35, desc, fontsize=8, ha='left', va='top',
                bbox=dict(boxstyle="round,pad=0.2", facecolor='lightcoral', edgecolor='red', alpha=0.6))
    
    ax3.text(5, 2.5, '核心挑战:\n1. 数据不均衡 (单峰 4000 vs 五峰 500)\n2. 峰数量组合爆炸 (5 种模式)\n3. 模式混淆\n4. 边界条件复杂\n5. 单一模型难建模多种物理模式', 
             fontsize=9, ha='center', va='top',
             bbox=dict(boxstyle="round,pad=0.2", facecolor='mistyrose'))
    
    # ==================== 分而治之 ====================
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_xlim(0, 10)
    ax4.set_ylim(0, 10)
    ax4.set_facecolor('white')
    ax4.axis('off')
    
    ax4.text(5, 9.6, '分而治之策略 (推荐)', fontsize=15, fontweight='bold', ha='center')
    ax4.text(5, 9.3, '峰数量分类器 + 专用模型', fontsize=11, ha='center')
    
    # 流程
    ax4.text(5, 8.5, '输入 (20 负偏压) → 分类器 (单/双/多峰)', fontsize=11, ha='center',
             bbox=dict(boxstyle="round,pad=0.2", facecolor='lightblue', edgecolor='navy'))
    
    # 三个分支
    branches = [
        (1.0, 6.8, '单峰分支', 'V2 模型\n20→512→1024→1024→512→301\n0.99nm, 96.5%<5nm', 'lightgreen'),
        (4.0, 6.8, '双峰分支', '专用模型\n20→512→1024→512→301\n10.75nm 中位，30.6%<5nm', 'lightsalmon'),
        (7.0, 6.8, '多峰分支', 'V3 模型\n20→512→1024→512→301\n~8nm 误差', 'lightyellow'),
    ]
    
    for x, y, title, desc, color in branches:
        ax4.text(x, y, title, fontsize=11, fontweight='bold', ha='center')
        ax4.text(x, y-0.4, desc, fontsize=8, ha='center', va='top',
                bbox=dict(boxstyle="round,pad=0.15", facecolor=color, edgecolor='black'))
        ax4.annotate('', xy=(x, y+0.8), xytext=(x, y+0.3),
                    arrowprops=dict(arrowstyle='->', lw=1.5))
    
    ax4.text(5, 4.8, '→ 输出：重建光谱曲线', fontsize=11, ha='center',
             bbox=dict(boxstyle="round,pad=0.2", facecolor='gold', edgecolor='darkgoldenrod'))
    
    ax4.text(2.5, 3.5, '预期性能:\n混合~3nm 误差\n<5nm~70%\n\n优势:\n• 各模型专注特定模式\n• 避免模式干扰\n• 单峰保持 V2 高性能', 
             fontsize=9, ha='center', va='top',
             bbox=dict(boxstyle="round,pad=0.2", facecolor='lightgreen'))
    
    ax4.text(7.5, 3.5, '关键洞察:\nV2 成功难复制到双峰\n这是问题本质决定\n\n双峰复杂性:\n• 位置组合多样\n• 峰间距变化大\n• 峰高比不确定', 
             fontsize=9, ha='center', va='top',
             bbox=dict(boxstyle="round,pad=0.2", facecolor='lightyellow'))
    
    plt.savefig('model_architecture_evolution.png', dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor())
    print('已保存：model_architecture_evolution.png')


def create_v2_detailed_diagram():
    """创建 V2 模型详细架构图"""
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    fig.patch.set_facecolor('white')
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.set_facecolor('white')
    ax.axis('off')
    
    ax.text(6, 7.7, 'V2 模型详细架构 | 1,052,981 参数 | 600 轮 | AdamW+CosineAnnealing', fontsize=13, fontweight='bold', ha='center')
    
    # 网络流程
    layers = [
        ('输入\n20 维', 1.0, 'lightblue'),
        ('L1\n20→512', 2.8, 'lightgreen'),
        ('L2\n512→1024', 4.6, 'lightgreen'),
        ('L3\n1024→1024', 6.4, 'lightgreen'),
        ('L4\n1024→512', 8.2, 'lightgreen'),
        ('输出\n301 维', 10.0, 'lightcoral'),
    ]
    
    for name, x, color in layers:
        w = 1.3
        rect = FancyBboxPatch((x, 6.3), w, 0.8, boxstyle="round,pad=0.1",
                             facecolor=color, edgecolor='black', linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x+0.65, 6.7, name, fontsize=9, ha='center', va='center')
    
    ax.text(6, 5.7, 'ReLU + BatchNorm1d + Dropout(0.2)', fontsize=10, ha='center',
             bbox=dict(boxstyle="round,pad=0.15", facecolor='lightgreen'))
    
    # 模块
    modules = [
        (1.5, 4.5, '数据预处理', '• 选择 -15V~0V 负偏压\n• 排除正偏压噪声\n• Min-Max 归一化\n• 边界过采样各 2000', 'lightblue'),
        (6.0, 4.5, '损失函数', 'L = w_boundary*MSE + w_reg*L2\n\nw_boundary=3.0 (边界)\nw_boundary=1.0 (其他)\nw_reg=0.01', 'lightyellow'),
        (10.5, 4.5, '训练配置', '• AdamW(lr=0.001,wd=0.01)\n• CosineAnnealingLR\n• Batch=32\n• 600 轮无早停', 'lightgreen'),
    ]
    
    for x, y, title, desc, color in modules:
        w = 3.5 if x == 6.0 else 2.8
        rect = Rectangle((x-w/2, y-0.9), w, 1.8, facecolor=color, edgecolor='black', linewidth=1.2, alpha=0.7)
        ax.add_patch(rect)
        ax.text(x, y+0.4, title, fontsize=11, fontweight='bold', ha='center')
        ax.text(x, y-0.2, desc, fontsize=8, ha='center', va='top')
    
    # 性能
    ax.text(3.5, 2.0, '性能指标\n平均：0.99nm\n中位：1.00nm\n最大：13.00nm\n<1nm:45% | <5nm:96.5%\n<10nm:99.4%', 
             fontsize=10, ha='center', va='center',
             bbox=dict(boxstyle="round,pad=0.25", facecolor='lightgreen', edgecolor='darkgreen', linewidth=2))
    
    ax.text(8.5, 2.0, '分区域误差\n1000-1050nm: 0.72nm\n1050-1250nm: 0.69nm\n1250-1300nm: 1.66nm\n\n参数量分布\nL1:10K | L2:525K | L3:1050K\nL4:525K | Out:154K', 
             fontsize=9, ha='center', va='center',
             bbox=dict(boxstyle="round,pad=0.25", facecolor='lightblue', edgecolor='navy'))
    
    plt.savefig('v2_model_detailed_architecture.png', dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor())
    print('已保存：v2_model_detailed_architecture.png')


if __name__ == "__main__":
    print('=' * 60)
    print('生成模型架构对比图')
    print('=' * 60)
    
    create_model_diagram()
    create_v2_detailed_diagram()
    
    print()
    print('完成！生成:')
    print('1. model_architecture_evolution.png')
    print('2. v2_model_detailed_architecture.png')