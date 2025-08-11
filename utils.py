import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from tqdm import tqdm


class DataLoader:
    def __init__(self):
        self.crop_data = None
        self.land_data = None
        self.planting_data = None
        self.stats_data = None

    def load_data(self, file1: str, file2: str) -> None:
        """加载附件1和附件2的数据"""
        # 加载附件1
        self.land_data = pd.read_excel(file1, sheet_name='乡村的现有耕地')
        self.crop_data = pd.read_excel(file1, sheet_name='乡村种植的农作物')

        # 加载附件2
        self.planting_data = pd.read_excel(file2, sheet_name='2023年的农作物种植情况')
        self.stats_data = pd.read_excel(file2, sheet_name='2023年统计的相关数据')

    def preprocess_data(self) -> Tuple[Dict, Dict, Dict, Dict, Dict]:
        """预处理数据，返回所需的各种字典"""
        # 地块信息
        land_info = {}
        for _, row in self.land_data.iterrows():
            if pd.notna(row['地块名称']):
                land_info[row['地块名称']] = {
                    'type': row['地块类型'],
                    'area': row['地块面积/亩']
                }

        # 作物信息
        crop_info = {}
        for _, row in self.crop_data.iterrows():
            if pd.notna(row['作物编号']):
                try:
                    crop_id = int(row['作物编号'])
                    crop_info[crop_id] = {
                        'name': row['作物名称'],
                        'type': row['作物类型'],
                        'suitable_land': row['种植耕地']
                    }
                except (ValueError, TypeError):
                    # 跳过非数字的作物编号（如注释行）
                    continue

        # 2023年种植情况
        planting_2023 = {}
        for _, row in self.planting_data.iterrows():
            if pd.notna(row['种植地块']):
                land = row['种植地块']
                if land not in planting_2023:
                    planting_2023[land] = []

                planting_2023[land].append({
                    'crop_id': row['作物编号'],
                    'crop_name': row['作物名称'],
                    'crop_type': row['作物类型'],
                    'area': row['种植面积/亩'],
                    'season': row['种植季次']
                })

        # 统计信息
        stats_info = {}
        for _, row in self.stats_data.iterrows():
            if pd.notna(row['序号']):
                try:
                    crop_id = int(row['作物编号'])
                    key = (crop_id, row['地块类型'], row['种植季次'])
                    price_range = str(row['销售单价/(元/斤)']).split('-')

                    stats_info[key] = {
                        'yield_per_mu': row['亩产量/斤'],
                        'cost_per_mu': row['种植成本/(元/亩)'],
                        'min_price': float(price_range[0]),
                        'max_price': float(price_range[1])
                    }
                except (ValueError, TypeError):
                    # 跳过非数字的作物编号（如注释行）
                    continue

        # 作物预期销售量 (从2023年种植情况估算)
        expected_sales = {}
        for land, crops in planting_2023.items():
            for crop in crops:
                crop_id = crop['crop_id']
                land_type = land_info[land]['type']
                season = crop['season']

                key = (crop_id, land_type, season)
                if key not in stats_info:
                    continue

                yield_per_mu = stats_info[key]['yield_per_mu']
                total_yield = yield_per_mu * crop['area']

                if crop_id not in expected_sales:
                    expected_sales[crop_id] = total_yield
                else:
                    expected_sales[crop_id] += total_yield

        return land_info, crop_info, planting_2023, stats_info, expected_sales

    def get_land_divisions(self, min_area=0.5) -> Dict:
        """将地块划分为更小的单位(如0.5亩)"""
        land_divisions = {}
        land_info, _, _, _, _ = self.preprocess_data()
        for land, info in land_info.items():
            area = info['area']
            # 确保所有地块都划分为半个地块为单位
            divisions = max(1, int(area / min_area))
            # 确保divisions是偶数，这样每个作物占据的半个地块数量为整数
            if divisions % 2 == 1:
                divisions += 1
            
            land_divisions[land] = {
                'type': info['type'],
                'divisions': divisions,
                'area_per_division': area / divisions,
                'half_divisions': divisions // 2  # 半个地块的数量
            }
        return land_divisions


class SolutionValidator:
    @staticmethod
    def validate_rotation(solution: Dict, crop_info: Dict, years: int = 7) -> bool:
        """验证豆类轮作约束"""
        for land, crops in solution.items():
            bean_planted = False
            for year in range(years):
                for season in ['单季', '第一季', '第二季']:
                    if year in crops and season in crops[year]:
                        crop_id = crops[year][season]['crop_id']
                        if crop_info[crop_id]['type'] in ['粮食（豆类）', '蔬菜（豆类）']:
                            bean_planted = True
                            break
                if bean_planted:
                    break
            if not bean_planted:
                return False
        return True

    @staticmethod
    def validate_no_replant(solution: Dict) -> bool:
        """验证无重茬种植"""
        for land, crops in solution.items():
            prev_crop = None
            for year in range(len(crops)):
                for season in ['单季', '第一季', '第二季']:
                    if year in crops and season in crops[year]:
                        current_crop = crops[year][season]['crop_id']
                        if current_crop == prev_crop:
                            return False
                        prev_crop = current_crop
        return True

    @staticmethod
    def validate_concentration(solution: Dict, min_plots: int = 3) -> bool:
        """验证作物种植不太分散"""
        crop_distribution = {}
        for land, crops in solution.items():
            for year in range(len(crops)):
                for season in ['单季', '第一季', '第二季']:
                    if year in crops and season in crops[year]:
                        crop_id = crops[year][season]['crop_id']
                        if crop_id not in crop_distribution:
                            crop_distribution[crop_id] = set()
                        crop_distribution[crop_id].add(land)

        for crop_id, lands in crop_distribution.items():
            if len(lands) < min_plots:
                return False
        return True

    @staticmethod
    def validate_min_area(solution: Dict, min_area: float = 0.5) -> bool:
        """验证单个地块种植面积不小于最小值"""
        for land, crops in solution.items():
            for year in range(len(crops)):
                for season in ['单季', '第一季', '第二季']:
                    if year in crops and season in crops[year]:
                        if crops[year][season]['area'] < min_area:
                            return False
        return True