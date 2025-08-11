import pulp
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import random
import math
from utils import DataLoader, SolutionValidator


class CropOptimizer:
    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader
        self.land_info, self.crop_info, self.planting_2023, self.stats_info, self.expected_sales = data_loader.preprocess_data()
        
        # 作物分类
        self.bean_crops = [crop_id for crop_id, info in self.crop_info.items()
                           if '豆类' in info['type']]
        self.non_bean_crops = [crop_id for crop_id, info in self.crop_info.items()
                               if '豆类' not in info['type']]

        # 地块分类
        self.flat_dry_lands = [land for land, info in self.land_info.items()
                               if info['type'] == '平旱地']
        self.terrace_lands = [land for land, info in self.land_info.items()
                              if info['type'] == '梯田']
        self.hillside_lands = [land for land, info in self.land_info.items()
                               if info['type'] == '山坡地']
        self.water_lands = [land for land, info in self.land_info.items()
                            if info['type'] == '水浇地']
        self.normal_greenhouses = [land for land, info in self.land_info.items()
                                   if info['type'] == '普通大棚']
        self.smart_greenhouses = [land for land, info in self.land_info.items()
                                  if info['type'] == '智慧大棚']

    def optimize(self, scenario: int = 1, years: int = 7) -> Dict:
        """主优化函数"""
        print(f"开始优化，场景: {scenario}, 年数: {years}")
        
        # 使用简化的优化方法
        solution = self._simple_optimize(scenario, years)
        
        # 验证解的有效性
        validator = SolutionValidator()
        if not validator.validate_rotation(solution, self.crop_info, years):
            print("警告: 豆类轮作约束未满足")
        if not validator.validate_no_replant(solution):
            print("警告: 重茬限制未满足")
        if not validator.validate_concentration(solution, min_plots=2):
            print("警告: 种植集中度约束未满足")
        if not validator.validate_min_area(solution, min_area=0.5):
            print("警告: 最小种植面积约束未满足")
            
        return solution

    def _simple_optimize(self, scenario: int, years: int) -> Dict:
        """简化的优化方法"""
        print("使用简化优化方法...")
        
        # 生成多个初始解并选择最好的
        best_solution = None
        best_fitness = float('-inf')
        
        for i in range(10):  # 生成10个初始解
            print(f"生成第 {i+1}/10 个初始解...")
            solution = self._generate_initial_solution(years)
            fitness = self._calculate_fitness(solution, scenario)
            
            if fitness > best_fitness:
                best_fitness = fitness
                best_solution = solution.copy()
        
        print(f"最佳初始解适应度: {best_fitness:.2f}")
        
        # 简单的局部搜索优化
        print("进行局部搜索优化...")
        best_solution = self._simple_local_search(best_solution, scenario)
        
        return best_solution

    def _generate_initial_solution(self, years: int) -> Dict:
        """生成初始解"""
        solution = {}
        
        for land in self.land_info:
            solution[land] = {}
            land_type = self.land_info[land]['type']
            area = self.land_info[land]['area']
            
            for year in range(years):
                solution[land][year] = {}
                
                if land_type in ['平旱地', '梯田', '山坡地']:
                    # 单季粮食作物
                    grain_crops = [c for c in self.crop_info if '粮食' in self.crop_info[c]['type']]
                    if grain_crops:
                        crop_id = random.choice(grain_crops)
                        solution[land][year]['单季'] = {
                            'crop_id': crop_id,
                            'crop_name': self.crop_info[crop_id]['name'],
                            'area': area
                        }
                
                elif land_type == '水浇地':
                    # 随机选择单季水稻或两季蔬菜
                    if random.random() < 0.5:
                        # 单季水稻
                        solution[land][year]['单季'] = {
                            'crop_id': 18,  # 水稻
                            'crop_name': '水稻',
                            'area': area
                        }
                    else:
                        # 两季蔬菜
                        veg_crops = [c for c in self.crop_info if self.crop_info[c]['type'] == '蔬菜' and c not in [37, 38, 39]]
                        if veg_crops:
                            crop_id = random.choice(veg_crops)
                            solution[land][year]['第一季'] = {
                                'crop_id': crop_id,
                                'crop_name': self.crop_info[crop_id]['name'],
                                'area': area
                            }
                        
                        # 第二季蔬菜（大白菜、白萝卜、红萝卜）
                        second_crops = [c for c in [37, 38, 39] if c in self.crop_info]
                        if second_crops:
                            crop_id = random.choice(second_crops)
                            solution[land][year]['第二季'] = {
                                'crop_id': crop_id,
                                'crop_name': self.crop_info[crop_id]['name'],
                                'area': area
                            }
                
                elif land_type == '普通大棚':
                    # 第一季蔬菜
                    veg_crops = [c for c in self.crop_info if self.crop_info[c]['type'] == '蔬菜' and c not in [37, 38, 39]]
                    if veg_crops:
                        crop_id = random.choice(veg_crops)
                        solution[land][year]['第一季'] = {
                            'crop_id': crop_id,
                            'crop_name': self.crop_info[crop_id]['name'],
                            'area': area
                        }
                    
                    # 第二季食用菌
                    mushroom_crops = [c for c in range(40, 44) if c in self.crop_info]
                    if mushroom_crops:
                        crop_id = random.choice(mushroom_crops)
                        solution[land][year]['第二季'] = {
                            'crop_id': crop_id,
                            'crop_name': self.crop_info[crop_id]['name'],
                            'area': area
                        }
                
                elif land_type == '智慧大棚':
                    # 两季蔬菜
                    for season in ['第一季', '第二季']:
                        veg_crops = [c for c in self.crop_info if self.crop_info[c]['type'] == '蔬菜' and c not in [37, 38, 39]]
                        if veg_crops:
                            crop_id = random.choice(veg_crops)
                            solution[land][year][season] = {
                                'crop_id': crop_id,
                                'crop_name': self.crop_info[crop_id]['name'],
                                'area': area
                            }
        
        return solution

    def _calculate_fitness(self, solution: Dict, scenario: int) -> float:
        """计算适应度（利润）"""
        total_profit = 0
        
        for land in solution:
            land_type = self.land_info[land]['type']
            
            for year in solution[land]:
                for season in solution[land][year]:
                    crop_id = solution[land][year][season]['crop_id']
                    area = solution[land][year][season]['area']
                    
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
        
        # 添加约束惩罚
        penalty = self._calculate_constraint_penalty(solution)
        
        return total_profit - penalty

    def _calculate_constraint_penalty(self, solution: Dict) -> float:
        """计算约束违反的惩罚"""
        penalty = 0
        
        # 1. 豆类轮作约束惩罚
        for land in solution:
            bean_planted = False
            for year in range(len(solution[land])):
                for season in ['单季', '第一季', '第二季']:
                    if year in solution[land] and season in solution[land][year]:
                        crop_id = solution[land][year][season]['crop_id']
                        if crop_id in self.bean_crops:
                            bean_planted = True
                            break
                if bean_planted:
                    break
            if not bean_planted:
                penalty += 100000
        
        # 2. 重茬种植惩罚
        for land in solution:
            prev_crop = None
            for year in range(len(solution[land])):
                for season in ['单季', '第一季', '第二季']:
                    if year in solution[land] and season in solution[land][year]:
                        current_crop = solution[land][year][season]['crop_id']
                        if current_crop == prev_crop:
                            penalty += 50000
                        prev_crop = current_crop
        
        # 3. 种植集中度惩罚
        for year in range(len(solution[list(solution.keys())[0]])):
            for season in ['单季', '第一季', '第二季']:
                crop_distribution = {}
                for land in solution:
                    if year in solution[land] and season in solution[land][year]:
                        crop_id = solution[land][year][season]['crop_id']
                        if crop_id not in crop_distribution:
                            crop_distribution[crop_id] = set()
                        crop_distribution[crop_id].add(land)
                
                for crop_id, lands in crop_distribution.items():
                    # 每种作物每季至少种植在2个地块上，但不超过8个地块
                    if len(lands) < 2:
                        penalty += 50000 * (2 - len(lands))
                    elif len(lands) > 8:
                        penalty += 30000 * (len(lands) - 8)
        
        # 4. 最小种植面积惩罚
        for land in solution:
            for year in solution[land]:
                for season in solution[land][year]:
                    area = solution[land][year][season]['area']
                    if area < 0.5:
                        penalty += 20000 * (0.5 - area)
        
        return penalty

    def _simple_local_search(self, solution: Dict, scenario: int) -> Dict:
        """简化的局部搜索优化"""
        best_solution = solution.copy()
        best_fitness = self._calculate_fitness(best_solution, scenario)
        
        improved = True
        iterations = 0
        max_iterations = 100  # 减少迭代次数
        
        while improved and iterations < max_iterations:
            improved = False
            iterations += 1
            
            # 随机选择一些地块进行修改
            lands_to_modify = random.sample(list(best_solution.keys()), min(5, len(best_solution)))
            
            for land in lands_to_modify:
                year = random.randint(0, len(best_solution[land]) - 1)
                
                # 尝试修改该地块该年的种植
                test_solution = best_solution.copy()
                land_type = self.land_info[land]['type']
                area = self.land_info[land]['area']
                
                # 清空该地块该年的种植
                test_solution[land][year] = {}
                
                # 重新生成种植方案
                if land_type in ['平旱地', '梯田', '山坡地']:
                    grain_crops = [c for c in self.crop_info if '粮食' in self.crop_info[c]['type']]
                    if grain_crops:
                        crop_id = random.choice(grain_crops)
                        test_solution[land][year]['单季'] = {
                            'crop_id': crop_id,
                            'crop_name': self.crop_info[crop_id]['name'],
                            'area': area
                        }
                
                elif land_type == '水浇地':
                    if random.random() < 0.5:
                        test_solution[land][year]['单季'] = {
                            'crop_id': 18,
                            'crop_name': '水稻',
                            'area': area
                        }
                    else:
                        veg_crops = [c for c in self.crop_info if self.crop_info[c]['type'] == '蔬菜' and c not in [37, 38, 39]]
                        if veg_crops:
                            crop_id = random.choice(veg_crops)
                            test_solution[land][year]['第一季'] = {
                                'crop_id': crop_id,
                                'crop_name': self.crop_info[crop_id]['name'],
                                'area': area
                            }
                        
                        second_crops = [c for c in [37, 38, 39] if c in self.crop_info]
                        if second_crops:
                            crop_id = random.choice(second_crops)
                            test_solution[land][year]['第二季'] = {
                                'crop_id': crop_id,
                                'crop_name': self.crop_info[crop_id]['name'],
                                'area': area
                            }
                
                elif land_type == '普通大棚':
                    veg_crops = [c for c in self.crop_info if self.crop_info[c]['type'] == '蔬菜' and c not in [37, 38, 39]]
                    if veg_crops:
                        crop_id = random.choice(veg_crops)
                        test_solution[land][year]['第一季'] = {
                            'crop_id': crop_id,
                            'crop_name': self.crop_info[crop_id]['name'],
                            'area': area
                        }
                    
                    mushroom_crops = [c for c in range(40, 44) if c in self.crop_info]
                    if mushroom_crops:
                        crop_id = random.choice(mushroom_crops)
                        test_solution[land][year]['第二季'] = {
                            'crop_id': crop_id,
                            'crop_name': self.crop_info[crop_id]['name'],
                            'area': area
                        }
                
                elif land_type == '智慧大棚':
                    for season in ['第一季', '第二季']:
                        veg_crops = [c for c in self.crop_info if self.crop_info[c]['type'] == '蔬菜' and c not in [37, 38, 39]]
                        if veg_crops:
                            crop_id = random.choice(veg_crops)
                            test_solution[land][year][season] = {
                                'crop_id': crop_id,
                                'crop_name': self.crop_info[crop_id]['name'],
                                'area': area
                            }
                
                # 计算新解的适应度
                new_fitness = self._calculate_fitness(test_solution, scenario)
                
                if new_fitness > best_fitness:
                    best_solution = test_solution
                    best_fitness = new_fitness
                    improved = True
                    break
        
        print(f"局部搜索完成，最终适应度: {best_fitness:.2f}")
        return best_solution

    def save_solution(self, solution: Dict, filename: str) -> None:
        """保存解决方案到Excel，保持原有格式"""
        # 创建结果矩阵
        years = 7
        seasons = ['第一季', '第二季', '单季']
        
        # 初始化结果矩阵
        result_data = []
        
        for season in seasons:
            for year in range(years):
                row = {'Unnamed: 0': f'{season}\n{2024 + year}', '地块名': ''}
                
                # 初始化所有作物为NaN
                for crop_id in range(1, 45):
                    if crop_id in self.crop_info:
                        row[self.crop_info[crop_id]['name']] = np.nan
                
                result_data.append(row)
        
        # 填充种植数据
        for land in solution:
            for year in solution[land]:
                for season in solution[land][year]:
                    crop = solution[land][year][season]
                    crop_id = crop['crop_id']
                    crop_name = crop['crop_name']
                    area = crop['area']
                    
                    # 找到对应的行
                    row_key = f'{season}\n{2024 + year}'
                    
                    # 更新地块名和种植面积
                    for row in result_data:
                        if row['Unnamed: 0'] == row_key:
                            if pd.isna(row['地块名']) or row['地块名'] == '':
                                row['地块名'] = land
                            else:
                                row['地块名'] += f' {land}'
                            
                            if crop_name in row:
                                row[crop_name] = area
                            break
        
        # 创建DataFrame
        result_df = pd.DataFrame(result_data)
        
        # 简单保存到文件
        try:
            result_df.to_excel(filename, index=False)
            print(f"结果已保存到 {filename}")
        except Exception as e:
            print(f"保存文件时出现错误: {e}")
            # 尝试保存为CSV格式作为备选
            csv_filename = filename.replace('.xlsx', '.csv')
            result_df.to_csv(csv_filename, index=False, encoding='utf-8-sig')
            print(f"结果已保存为CSV格式: {csv_filename}")