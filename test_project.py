#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
项目功能测试脚本
"""

import os
import sys
from utils import DataLoader
from optimization import CropOptimizer

def test_data_loading():
    """测试数据加载功能"""
    print("=== 测试数据加载功能 ===")
    
    try:
        # 检查数据文件是否存在
        file1 = '../data/附件1.xlsx'
        file2 = '../data/附件2.xlsx'
        
        if not os.path.exists(file1):
            print(f"错误：找不到文件 {file1}")
            return False
            
        if not os.path.exists(file2):
            print(f"错误：找不到文件 {file2}")
            return False
            
        print("✓ 数据文件存在")
        
        # 加载数据
        data_loader = DataLoader()
        data_loader.load_data(file1, file2)
        print("✓ 数据加载成功")
        
        # 预处理数据
        land_info, crop_info, planting_2023, stats_info, expected_sales = data_loader.preprocess_data()
        print(f"✓ 数据预处理成功")
        print(f"  - 地块数量: {len(land_info)}")
        print(f"  - 作物数量: {len(crop_info)}")
        print(f"  - 统计信息数量: {len(stats_info)}")
        
        return True
        
    except Exception as e:
        print(f"✗ 数据加载失败: {e}")
        return False

def test_optimizer():
    """测试优化器功能"""
    print("\n=== 测试优化器功能 ===")
    
    try:
        # 创建数据加载器
        data_loader = DataLoader()
        data_loader.load_data('../data/附件1.xlsx', '../data/附件2.xlsx')
        
        # 创建优化器
        optimizer = CropOptimizer(data_loader)
        print("✓ 优化器创建成功")
        
        # 检查豆类作物识别
        print(f"  - 豆类作物数量: {len(optimizer.bean_crops)}")
        print(f"  - 非豆类作物数量: {len(optimizer.non_bean_crops)}")
        
        # 检查地块分类
        print(f"  - 平旱地数量: {len(optimizer.flat_dry_lands)}")
        print(f"  - 水浇地数量: {len(optimizer.water_lands)}")
        print(f"  - 普通大棚数量: {len(optimizer.normal_greenhouses)}")
        print(f"  - 智慧大棚数量: {len(optimizer.smart_greenhouses)}")
        
        return True
        
    except Exception as e:
        print(f"✗ 优化器测试失败: {e}")
        return False

def test_simple_optimization():
    """测试简单优化功能"""
    print("\n=== 测试简单优化功能 ===")
    
    try:
        # 创建优化器
        data_loader = DataLoader()
        data_loader.load_data('../data/附件1.xlsx', '../data/附件2.xlsx')
        optimizer = CropOptimizer(data_loader)
        
        # 运行简单的SA优化（减少迭代次数以加快测试）
        print("正在运行模拟退火优化（测试模式）...")
        solution = optimizer.sa_optimize(scenario=1, years=3, iterations=50)
        
        print("✓ 优化完成")
        print(f"  - 解决方案包含 {len(solution)} 个地块")
        
        # 测试保存功能
        test_file = '../data/附件3/test_result.xlsx'
        optimizer.save_solution(solution, test_file)
        
        if os.path.exists(test_file):
            print("✓ 结果保存成功")
            # 删除测试文件
            os.remove(test_file)
            print("✓ 测试文件已清理")
        else:
            print("✗ 结果保存失败")
            return False
            
        return True
        
    except Exception as e:
        print(f"✗ 优化测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("开始项目功能测试...\n")
    
    # 确保附件3目录存在
    os.makedirs('../data/附件3', exist_ok=True)
    
    # 运行测试
    tests = [
        test_data_loading,
        test_optimizer,
        test_simple_optimization
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    # 输出测试结果
    print("=" * 50)
    print(f"测试完成: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有测试通过！项目可以正常运行。")
        print("\n下一步可以运行:")
        print("  python main.py          # 运行完整优化")
        print("  python visualization.py # 运行可视化分析")
    else:
        print("❌ 部分测试失败，请检查错误信息。")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)





