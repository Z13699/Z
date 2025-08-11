from utils import DataLoader
from optimization import CropOptimizer
import os


def main():
    # 获取当前脚本所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(current_dir)
    
    # 创建数据目录和附件3目录
    data_dir = os.path.join(project_dir, 'data')
    output_dir = os.path.join(data_dir, '附件3')
    os.makedirs(output_dir, exist_ok=True)

    # 加载数据
    data_loader = DataLoader()
    data_loader.load_data(
        os.path.join(data_dir, '附件1.xlsx'),
        os.path.join(data_dir, '附件2.xlsx')
    )

    # 创建优化器
    optimizer = CropOptimizer(data_loader)

    # 问题1(1): 超过部分滞销
    print("=" * 50)
    print("问题1(1): 超过部分滞销")
    print("=" * 50)
    solution1 = optimizer.optimize(scenario=1)
    optimizer.save_solution(solution1, os.path.join(output_dir, 'result1_1.xlsx'))

    # 问题1(2): 超过部分降价50%
    print("\n" + "=" * 50)
    print("问题1(2): 超过部分降价50%")
    print("=" * 50)
    solution2 = optimizer.optimize(scenario=2)
    optimizer.save_solution(solution2, os.path.join(output_dir, 'result1_2.xlsx'))

    print(f"\n优化完成。结果已保存到 {output_dir}/")
    print("生成的文件:")
    print(f"  - result1_1.xlsx (滞销场景)")
    print(f"  - result1_2.xlsx (降价50%场景)")


if __name__ == "__main__":
    main()