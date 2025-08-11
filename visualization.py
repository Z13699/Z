import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from typing import Dict, List
import os
from utils import DataLoader
from optimization import CropOptimizer

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

class CropVisualizer:
    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader
        self.land_info, self.crop_info, self.planting_2023, self.stats_info, self.expected_sales = data_loader.preprocess_data()

    def plot_land_distribution(self, save_path: str = None):
        """绘制地块分布图"""
        plt.figure(figsize=(12, 8))
        
        # 统计各地块类型的数量和面积
        land_types = {}
        land_areas = {}
        
        for land, info in self.land_info.items():
            land_type = info['type']
            area = info['area']
            
            if land_type not in land_types:
                land_types[land_type] = 0
                land_areas[land_type] = 0
            
            land_types[land_type] += 1
            land_areas[land_type] += area
        
        # 创建子图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 地块数量分布
        ax1.pie(land_types.values(), labels=land_types.keys(), autopct='%1.1f%%', startangle=90)
        ax1.set_title('各地块类型数量分布')
        
        # 地块面积分布
        ax2.pie(land_areas.values(), labels=land_areas.keys(), autopct='%1.1f%%', startangle=90)
        ax2.set_title('各地块类型面积分布')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_crop_distribution(self, save_path: str = None):
        """绘制作物分布图"""
        plt.figure(figsize=(14, 8))
        
        # 统计作物类型分布
        crop_types = {}
        for crop_id, info in self.crop_info.items():
            crop_type = info['type']
            if crop_type not in crop_types:
                crop_types[crop_type] = 0
            crop_types[crop_type] += 1
        
        # 绘制柱状图
        plt.bar(crop_types.keys(), crop_types.values(), color='skyblue', edgecolor='navy')
        plt.title('作物类型分布')
        plt.xlabel('作物类型')
        plt.ylabel('作物数量')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_optimization_results(self, solution1: Dict, solution2: Dict, save_path: str = None):
        """绘制优化结果对比图"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 计算每年的利润
        years = list(range(2024, 2031))
        profits1 = []
        profits2 = []
        
        for year in years:
            year_idx = year - 2024
            profit1 = self._calculate_year_profit(solution1, year_idx, 1)
            profit2 = self._calculate_year_profit(solution2, year_idx, 2)
            profits1.append(profit1)
            profits2.append(profit2)
        
        # 利润对比
        axes[0, 0].plot(years, profits1, 'b-o', label='滞销场景', linewidth=2, markersize=6)
        axes[0, 0].plot(years, profits2, 'r-s', label='降价50%场景', linewidth=2, markersize=6)
        axes[0, 0].set_title('年度利润对比')
        axes[0, 0].set_xlabel('年份')
        axes[0, 0].set_ylabel('利润 (元)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 累计利润
        cumulative_profit1 = np.cumsum(profits1)
        cumulative_profit2 = np.cumsum(profits2)
        
        axes[0, 1].plot(years, cumulative_profit1, 'b-o', label='滞销场景', linewidth=2, markersize=6)
        axes[0, 1].plot(years, cumulative_profit2, 'r-s', label='降价50%场景', linewidth=2, markersize=6)
        axes[0, 1].set_title('累计利润对比')
        axes[0, 1].set_xlabel('年份')
        axes[0, 1].set_ylabel('累计利润 (元)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 作物种植面积分布（第一年）
        crop_areas1 = self._get_crop_areas(solution1, 0)
        crop_areas2 = self._get_crop_areas(solution2, 0)
        
        # 选择前10个作物
        top_crops1 = sorted(crop_areas1.items(), key=lambda x: x[1], reverse=True)[:10]
        top_crops2 = sorted(crop_areas2.items(), key=lambda x: x[1], reverse=True)[:10]
        
        crop_names1 = [self.crop_info[crop_id]['name'] for crop_id, _ in top_crops1]
        areas1 = [area for _, area in top_crops1]
        
        crop_names2 = [self.crop_info[crop_id]['name'] for crop_id, _ in top_crops2]
        areas2 = [area for _, area in top_crops2]
        
        axes[1, 0].barh(crop_names1, areas1, color='lightblue', alpha=0.7)
        axes[1, 0].set_title('滞销场景 - 2024年作物种植面积 (前10)')
        axes[1, 0].set_xlabel('种植面积 (亩)')
        
        axes[1, 1].barh(crop_names2, areas2, color='lightcoral', alpha=0.7)
        axes[1, 1].set_title('降价50%场景 - 2024年作物种植面积 (前10)')
        axes[1, 1].set_xlabel('种植面积 (亩)')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_land_utilization(self, solution: Dict, scenario_name: str, save_path: str = None):
        """绘制土地利用情况"""
        plt.figure(figsize=(14, 8))
        
        # 统计各地块类型的利用率
        land_utilization = {}
        
        for land, info in self.land_info.items():
            land_type = info['type']
            if land_type not in land_utilization:
                land_utilization[land_type] = {'total_area': 0, 'used_area': 0}
            
            land_utilization[land_type]['total_area'] += info['area']
            
            # 计算该地块7年的总种植面积
            for year in solution[land]:
                for season in solution[land][year]:
                    land_utilization[land_type]['used_area'] += solution[land][year][season]['area']
        
        # 计算利用率
        land_types = list(land_utilization.keys())
        utilization_rates = []
        for land_type in land_types:
            total = land_utilization[land_type]['total_area']
            used = land_utilization[land_type]['used_area']
            rate = (used / (total * 7)) * 100  # 7年总利用率
            utilization_rates.append(rate)
        
        # 绘制柱状图
        bars = plt.bar(land_types, utilization_rates, color='lightgreen', edgecolor='darkgreen')
        plt.title(f'{scenario_name} - 各地块类型利用率')
        plt.xlabel('地块类型')
        plt.ylabel('利用率 (%)')
        plt.xticks(rotation=45, ha='right')
        
        # 添加数值标签
        for bar, rate in zip(bars, utilization_rates):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                    f'{rate:.1f}%', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def _calculate_year_profit(self, solution: Dict, year_idx: int, scenario: int) -> float:
        """计算指定年份的利润"""
        total_profit = 0
        
        for land in solution:
            land_type = self.land_info[land]['type']
            
            if year_idx in solution[land]:
                for season in solution[land][year_idx]:
                    crop_id = solution[land][year_idx][season]['crop_id']
                    area = solution[land][year_idx][season]['area']
                    
                    key = (crop_id, land_type, season)
                    if key not in self.stats_info:
                        continue
                    
                    stats = self.stats_info[key]
                    yield_per_mu = stats['yield_per_mu']
                    cost = stats['cost_per_mu']
                    price = (stats['min_price'] + stats['max_price']) / 2
                    total_yield = yield_per_mu * area
                    
                    # 计算收入
                    if crop_id in self.expected_sales:
                        expected = self.expected_sales[crop_id]
                        if scenario == 1:  # 滞销
                            revenue = min(total_yield, expected) * price
                        else:  # 降价50%
                            revenue = min(total_yield, expected) * price + max(0, total_yield - expected) * price * 0.5
                    else:
                        revenue = total_yield * price
                    
                    # 减去成本
                    total_profit += revenue - cost * area
        
        return total_profit

    def _get_crop_areas(self, solution: Dict, year_idx: int) -> Dict:
        """获取指定年份各作物的种植面积"""
        crop_areas = {}
        
        for land in solution:
            if year_idx in solution[land]:
                for season in solution[land][year_idx]:
                    crop_id = solution[land][year_idx][season]['crop_id']
                    area = solution[land][year_idx][season]['area']
                    
                    if crop_id not in crop_areas:
                        crop_areas[crop_id] = 0
                    crop_areas[crop_id] += area
        
        return crop_areas

    def create_comprehensive_report(self, solution1: Dict, solution2: Dict, output_dir: str):
        """创建综合报告"""
        # 创建图表目录
        charts_dir = os.path.join(output_dir, 'charts')
        os.makedirs(charts_dir, exist_ok=True)
        
        # 生成各种图表
        self.plot_land_distribution(os.path.join(charts_dir, 'land_distribution.png'))
        self.plot_crop_distribution(os.path.join(charts_dir, 'crop_distribution.png'))
        self.plot_optimization_results(solution1, solution2, os.path.join(charts_dir, 'optimization_results.png'))
        self.plot_land_utilization(solution1, '滞销场景', os.path.join(charts_dir, 'land_utilization_scenario1.png'))
        self.plot_land_utilization(solution2, '降价50%场景', os.path.join(charts_dir, 'land_utilization_scenario2.png'))
        
        print(f"图表已保存到 {charts_dir}/")


def main():
    # 获取当前脚本所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(current_dir)
    
    # 数据目录
    data_dir = os.path.join(project_dir, 'data')
    output_dir = os.path.join(data_dir, '附件3')
    
    # 加载数据
    data_loader = DataLoader()
    data_loader.load_data(
        os.path.join(data_dir, '附件1.xlsx'),
        os.path.join(data_dir, '附件2.xlsx')
    )
    
    # 创建可视化器
    visualizer = CropVisualizer(data_loader)
    
    # 生成基础图表
    print("生成基础数据图表...")
    visualizer.plot_land_distribution()
    visualizer.plot_crop_distribution()
    
    print("可视化完成！")


if __name__ == "__main__":
    main()
