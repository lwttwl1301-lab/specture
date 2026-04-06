"""
数据文件完整性检查脚本
检查项目所需的所有数据文件是否齐全
"""

import os
import sys
import pandas as pd
import torch

def check_file_exists(filepath, description):
    """检查文件是否存在"""
    if os.path.exists(filepath):
        print(f"✅ {description}: {filepath}")
        return True
    else:
        print(f"❌ {description}: {filepath} - 文件不存在")
        return False

def check_excel_file(filepath):
    """检查Excel文件是否可读"""
    try:
        df = pd.read_excel(filepath)
        print(f"  ✓ Excel文件可读，形状: {df.shape}")
        return True
    except Exception as e:
        print(f"  ✗ Excel文件读取失败: {e}")
        return False

def check_model_file(filepath):
    """检查模型文件是否有效"""
    try:
        checkpoint = torch.load(filepath, map_location='cpu', weights_only=False)
        print(f"  ✓ 模型文件有效")
        
        # 检查关键数据
        required_keys = ['model_state_dict', 'scaler_X', 'scaler_y']
        missing_keys = [key for key in required_keys if key not in checkpoint]
        
        if missing_keys:
            print(f"  ⚠️ 模型文件缺少以下键: {missing_keys}")
            return False
        else:
            print(f"  ✓ 模型包含所有必要数据")
            return True
    except Exception as e:
        print(f"  ✗ 模型文件加载失败: {e}")
        return False

def check_project_structure():
    """检查项目整体结构"""
    print("=" * 60)
    print("光谱重建项目 - 文件完整性检查")
    print("=" * 60)
    
    # 检查核心文件
    core_files = [
        ('train_multi_bias_v2.py', '训练脚本'),
        ('comprehensive_analysis.py', '分析脚本'),
        ('model_multi_bias_v2.pth', '训练好的模型'),
        ('V2_MODEL_REPORT.md', '技术报告')
    ]
    
    all_good = True
    
    for filename, description in core_files:
        if not check_file_exists(filename, description):
            all_good = False
    
    print("\n" + "=" * 60)
    print("数据文件检查")
    print("=" * 60)
    
    # 检查数据文件（尝试多个可能的位置）
    data_files = [
        ('响应度矩阵.xlsx', '响应度数据（项目目录）'),
        (r'D:\desktop\try\响应度矩阵.xlsx', '响应度数据（原始位置）')
    ]
    
    data_found = False
    for filepath, description in data_files:
        if os.path.exists(filepath):
            print(f"✅ {description}: {filepath}")
            data_found = True
            # 检查Excel文件内容
            check_excel_file(filepath)
            break
    
    if not data_found:
        print("❌ 未找到响应度数据文件")
        print("   请将 '响应度矩阵.xlsx' 复制到项目目录")
        print("   或修改脚本中的路径指向正确位置")
        all_good = False
    
    print("\n" + "=" * 60)
    print("模型文件详细检查")
    print("=" * 60)
    
    # 详细检查模型文件
    if os.path.exists('model_multi_bias_v2.pth'):
        check_model_file('model_multi_bias_v2.pth')
    else:
        print("❌ 模型文件不存在")
        all_good = False
    
    print("\n" + "=" * 60)
    print("检查结果")
    print("=" * 60)
    
    if all_good and data_found:
        print("✅ 所有核心文件齐全，项目可以正常运行")
        print("\n运行建议:")
        print("1. 训练模型: python train_multi_bias_v2.py")
        print("2. 分析结果: python comprehensive_analysis.py")
        print("3. 查看报告: 打开 V2_MODEL_REPORT.md")
    else:
        print("⚠️  项目文件不完整，需要补充以下文件:")
        if not data_found:
            print("   - 响应度矩阵.xlsx (从 D:\\desktop\\try\\ 复制到项目目录)")
        print("\n修复步骤:")
        print("1. 复制响应度文件: copy \"D:\\desktop\\try\\响应度矩阵.xlsx\" .")
        print("2. 或修改脚本中的路径指向您的数据文件")
    
    return all_good and data_found

def fix_data_paths():
    """修复脚本中的数据文件路径"""
    print("\n" + "=" * 60)
    print("修复数据文件路径")
    print("=" * 60)
    
    # 检查当前目录是否有响应度文件
    if os.path.exists('响应度矩阵.xlsx'):
        new_path = '响应度矩阵.xlsx'
        print(f"✅ 在项目目录找到响应度文件，使用相对路径: {new_path}")
    elif os.path.exists(r'D:\desktop\try\响应度矩阵.xlsx'):
        new_path = r'D:\desktop\try\响应度矩阵.xlsx'
        print(f"✅ 在原始位置找到响应度文件，使用绝对路径: {new_path}")
    else:
        print("❌ 未找到响应度文件")
        return False
    
    # 修复训练脚本
    train_script = 'train_multi_bias_v2.py'
    if os.path.exists(train_script):
        with open(train_script, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 替换路径
        old_path = r"r'D:\\desktop\\try\\响应度矩阵.xlsx'"
        new_path_str = f"r'{new_path}'"
        content = content.replace(old_path, new_path_str)
        
        with open(train_script, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"✅ 已修复 {train_script} 中的路径")
    
    # 修复分析脚本
    analysis_script = 'comprehensive_analysis.py'
    if os.path.exists(analysis_script):
        with open(analysis_script, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 替换路径
        old_path = r"r'D:\\desktop\\try\\响应度矩阵.xlsx'"
        new_path_str = f"r'{new_path}'"
        content = content.replace(old_path, new_path_str)
        
        with open(analysis_script, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"✅ 已修复 {analysis_script} 中的路径")
    
    return True

if __name__ == "__main__":
    print("正在检查项目文件完整性...\n")
    
    # 检查项目结构
    is_complete = check_project_structure()
    
    # 如果数据文件存在但路径不对，尝试修复
    if not is_complete:
        print("\n尝试自动修复路径...")
        fix_data_paths()
        
        # 重新检查
        print("\n重新检查项目文件...")
        check_project_structure()