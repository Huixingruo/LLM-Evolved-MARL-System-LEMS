"""
进化过程可视化工具
绘制进化曲线、对比图表等

Author: LEMS Project
Date: 2026-02-03
Version: 1.0
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from typing import List, Dict, Optional


# 设置中文字体（避免中文乱码）
rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 黑体
rcParams['axes.unicode_minus'] = False  # 正常显示负号


class EvolutionPlotter:
    """进化过程可视化器"""
    
    def __init__(self, archive_dir: str = "experiments/evolution_archive"):
        """
        初始化可视化器
        
        Args:
            archive_dir: 进化记录目录
        """
        self.archive_dir = archive_dir
    
    def load_generation_data(self, generation: int) -> Optional[Dict]:
        """
        加载指定代的数据
        
        Args:
            generation: 代数
        
        Returns:
            dict: 代次数据，如果不存在返回None
        """
        filepath = os.path.join(self.archive_dir, f"generation_{generation:03d}.json")
        
        if not os.path.exists(filepath):
            return None
        
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def load_all_generations(self) -> List[Dict]:
        """
        加载所有代的数据
        
        Returns:
            List[Dict]: 所有代的数据列表
        """
        generations = []
        gen = 0
        
        while True:
            data = self.load_generation_data(gen)
            if data is None:
                break
            generations.append(data)
            gen += 1
        
        return generations
    
    def plot_evolution_curve(self, save_path: str = "evolution_curve.png"):
        """
        绘制进化曲线
        
        Args:
            save_path: 保存路径
        """
        generations = self.load_all_generations()
        
        if not generations:
            print("没有找到进化数据")
            return
        
        # 提取数据
        gen_nums = list(range(len(generations)))
        best_fitness = [g['best_fitness'] for g in generations]
        
        # 计算平均fitness
        avg_fitness = []
        for g in generations:
            all_results = g.get('all_results', [])
            if all_results:
                valid_fitness = [r.get('fitness', 0) for r in all_results if r.get('status') == 'success']
                if valid_fitness:
                    avg_fitness.append(np.mean(valid_fitness))
                else:
                    avg_fitness.append(0)
            else:
                avg_fitness.append(0)
        
        # 绘图
        plt.figure(figsize=(10, 6))
        
        plt.plot(gen_nums, best_fitness, 'o-', label='Best Fitness', 
                linewidth=2, markersize=8, color='#2E86AB')
        plt.plot(gen_nums, avg_fitness, 's--', label='Average Fitness', 
                linewidth=2, markersize=6, alpha=0.7, color='#A23B72')
        
        # 标记最优点
        best_gen = np.argmax(best_fitness)
        plt.scatter([best_gen], [best_fitness[best_gen]], 
                   c='red', s=200, marker='*', zorder=5, label='Best Ever')
        
        plt.xlabel('Generation', fontsize=14, fontweight='bold')
        plt.ylabel('Fitness', fontsize=14, fontweight='bold')
        plt.title('Evolution Curve of Reward Function Design', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.legend(fontsize=12, loc='best')
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.tight_layout()
        
        # 保存
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"进化曲线已保存到: {save_path}")
        plt.close()
    
    def plot_fitness_distribution(self, save_path: str = "fitness_distribution.png"):
        """
        绘制每代的Fitness分布（箱线图）
        
        Args:
            save_path: 保存路径
        """
        generations = self.load_all_generations()
        
        if not generations:
            print("没有找到进化数据")
            return
        
        # 提取每代所有候选的fitness
        fitness_data = []
        for g in generations:
            all_results = g.get('all_results', [])
            valid_fitness = [r.get('fitness', 0) for r in all_results if r.get('status') == 'success']
            fitness_data.append(valid_fitness if valid_fitness else [0])
        
        # 绘制箱线图
        plt.figure(figsize=(12, 6))
        
        bp = plt.boxplot(fitness_data, 
                        labels=[f'Gen{i}' for i in range(len(fitness_data))],
                        patch_artist=True,
                        showmeans=True)
        
        # 设置颜色
        for patch in bp['boxes']:
            patch.set_facecolor('#A8DADC')
            patch.set_alpha(0.7)
        
        plt.xlabel('Generation', fontsize=14, fontweight='bold')
        plt.ylabel('Fitness', fontsize=14, fontweight='bold')
        plt.title('Fitness Distribution Across Generations', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.grid(True, alpha=0.3, axis='y', linestyle='--')
        plt.tight_layout()
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Fitness分布图已保存到: {save_path}")
        plt.close()
    
    def plot_success_rate_comparison(self, save_path: str = "success_rate_comparison.png"):
        """
        绘制成功率对比图
        
        Args:
            save_path: 保存路径
        """
        generations = self.load_all_generations()
        
        if not generations:
            print("没有找到进化数据")
            return
        
        # 提取成功率数据
        gen_nums = []
        success_rates = []
        
        for i, g in enumerate(generations):
            all_results = g.get('all_results', [])
            for r in all_results:
                if r.get('status') == 'success':
                    metrics = r.get('metrics', {})
                    success_rate = metrics.get('success_rate', 0)
                    
                    gen_nums.append(i)
                    success_rates.append(success_rate)
        
        if not success_rates:
            print("没有成功率数据")
            return
        
        # 绘制散点图
        plt.figure(figsize=(10, 6))
        
        plt.scatter(gen_nums, success_rates, alpha=0.6, s=100, c=gen_nums, cmap='viridis')
        
        # 添加趋势线
        if len(success_rates) > 1:
            z = np.polyfit(gen_nums, success_rates, 2)
            p = np.poly1d(z)
            x_smooth = np.linspace(min(gen_nums), max(gen_nums), 100)
            plt.plot(x_smooth, p(x_smooth), "r--", alpha=0.8, linewidth=2, label='Trend')
        
        plt.xlabel('Generation', fontsize=14, fontweight='bold')
        plt.ylabel('Success Rate', fontsize=14, fontweight='bold')
        plt.title('Success Rate Evolution', fontsize=16, fontweight='bold', pad=20)
        plt.colorbar(label='Generation')
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.tight_layout()
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"成功率对比图已保存到: {save_path}")
        plt.close()
    
    def plot_comprehensive_dashboard(self, save_path: str = "dashboard.png"):
        """
        绘制综合仪表板（包含多个子图）
        
        Args:
            save_path: 保存路径
        """
        generations = self.load_all_generations()
        
        if not generations:
            print("没有找到进化数据")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Evolution Dashboard', fontsize=20, fontweight='bold', y=0.995)
        
        # 子图1: 进化曲线
        ax1 = axes[0, 0]
        gen_nums = list(range(len(generations)))
        best_fitness = [g['best_fitness'] for g in generations]
        
        ax1.plot(gen_nums, best_fitness, 'o-', linewidth=2, markersize=8)
        ax1.set_xlabel('Generation', fontweight='bold')
        ax1.set_ylabel('Best Fitness', fontweight='bold')
        ax1.set_title('Evolution Curve', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # 子图2: 成功率
        ax2 = axes[0, 1]
        success_rates_by_gen = []
        for g in generations:
            all_results = g.get('all_results', [])
            rates = [r.get('metrics', {}).get('success_rate', 0) 
                    for r in all_results if r.get('status') == 'success']
            if rates:
                success_rates_by_gen.append(np.mean(rates))
            else:
                success_rates_by_gen.append(0)
        
        ax2.bar(gen_nums, success_rates_by_gen, color='#457B9D', alpha=0.7)
        ax2.set_xlabel('Generation', fontweight='bold')
        ax2.set_ylabel('Avg Success Rate', fontweight='bold')
        ax2.set_title('Success Rate per Generation', fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # 子图3: 捕获时间
        ax3 = axes[1, 0]
        capture_times_by_gen = []
        for g in generations:
            all_results = g.get('all_results', [])
            times = [r.get('metrics', {}).get('avg_capture_time', 100) 
                    for r in all_results if r.get('status') == 'success']
            if times:
                capture_times_by_gen.append(np.mean(times))
            else:
                capture_times_by_gen.append(100)
        
        ax3.plot(gen_nums, capture_times_by_gen, 's-', linewidth=2, markersize=6, color='#E63946')
        ax3.set_xlabel('Generation', fontweight='bold')
        ax3.set_ylabel('Avg Capture Time', fontweight='bold')
        ax3.set_title('Capture Time per Generation', fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # 子图4: 候选状态统计
        ax4 = axes[1, 1]
        success_counts = []
        error_counts = []
        timeout_counts = []
        
        for g in generations:
            all_results = g.get('all_results', [])
            success_counts.append(sum(1 for r in all_results if r.get('status') == 'success'))
            error_counts.append(sum(1 for r in all_results if r.get('status') == 'error'))
            timeout_counts.append(sum(1 for r in all_results if r.get('status') == 'timeout'))
        
        x = np.arange(len(gen_nums))
        width = 0.25
        
        ax4.bar(x - width, success_counts, width, label='Success', color='#06D6A0')
        ax4.bar(x, error_counts, width, label='Error', color='#EF476F')
        ax4.bar(x + width, timeout_counts, width, label='Timeout', color='#FFD166')
        
        ax4.set_xlabel('Generation', fontweight='bold')
        ax4.set_ylabel('Count', fontweight='bold')
        ax4.set_title('Candidate Status Distribution', fontweight='bold')
        ax4.set_xticks(x)
        ax4.set_xticklabels([f'G{i}' for i in gen_nums])
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"综合仪表板已保存到: {save_path}")
        plt.close()
    
    def generate_all_plots(self, output_dir: str = "experiments/plots"):
        """
        生成所有可视化图表
        
        Args:
            output_dir: 输出目录
        """
        os.makedirs(output_dir, exist_ok=True)
        
        print("生成可视化图表...")
        
        self.plot_evolution_curve(os.path.join(output_dir, "evolution_curve.png"))
        self.plot_fitness_distribution(os.path.join(output_dir, "fitness_distribution.png"))
        self.plot_success_rate_comparison(os.path.join(output_dir, "success_rate.png"))
        self.plot_comprehensive_dashboard(os.path.join(output_dir, "dashboard.png"))
        
        print(f"\n所有图表已保存到: {output_dir}")


# ========================================
# 命令行接口
# ========================================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="进化过程可视化工具")
    parser.add_argument(
        '--archive_dir',
        type=str,
        default='experiments/evolution_archive',
        help='进化记录目录'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='experiments/plots',
        help='输出目录'
    )
    parser.add_argument(
        '--plot_type',
        type=str,
        choices=['evolution', 'distribution', 'success_rate', 'dashboard', 'all'],
        default='all',
        help='图表类型'
    )
    
    args = parser.parse_args()
    
    plotter = EvolutionPlotter(archive_dir=args.archive_dir)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.plot_type == 'evolution':
        plotter.plot_evolution_curve(os.path.join(args.output_dir, "evolution_curve.png"))
    elif args.plot_type == 'distribution':
        plotter.plot_fitness_distribution(os.path.join(args.output_dir, "fitness_distribution.png"))
    elif args.plot_type == 'success_rate':
        plotter.plot_success_rate_comparison(os.path.join(args.output_dir, "success_rate.png"))
    elif args.plot_type == 'dashboard':
        plotter.plot_comprehensive_dashboard(os.path.join(args.output_dir, "dashboard.png"))
    else:  # all
        plotter.generate_all_plots(args.output_dir)
