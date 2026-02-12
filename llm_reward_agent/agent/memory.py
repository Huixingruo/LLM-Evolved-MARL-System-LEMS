"""
进化记忆管理
存储和管理进化历史，包括最优代码、反思、性能指标等

Author: LEMS Project
Date: 2026-02-03
Version: 1.0
"""

import json
import os
from typing import Dict, List, Optional, Any
from datetime import datetime


class EvolutionaryMemory:
    """进化历史记忆库"""
    
    def __init__(self, save_dir: str = "experiments/evolution_archive"):
        """
        初始化进化记忆
        
        Args:
            save_dir: 保存目录
        """
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # 进化历史记录
        self.history = []  # [{generation, best_code, reflection, fitness, all_results}]
        
        # 元数据
        self.metadata = {
            'creation_time': datetime.now().isoformat(),
            'total_generations': 0,
            'best_fitness_ever': -float('inf'),
            'best_generation': -1
        }
        
        print(f"✅ 进化记忆初始化完成: {save_dir}")
    
    def save(self, 
             generation: int, 
             best_code: str, 
             reflection: str, 
             all_results: List[Dict],
             metadata: Optional[Dict] = None):
        """
        保存一代的记录
        
        Args:
            generation: 代数
            best_code: 本代最优代码
            reflection: 反思内容
            all_results: 所有候选的结果列表
            metadata: 额外的元数据
        """
        # 计算最优fitness
        best_fitness = max(r.get('fitness', -float('inf')) for r in all_results)
        
        # 创建记录
        record = {
            'generation': generation,
            'best_code': best_code,
            'reflection': reflection,
            'best_fitness': best_fitness,
            'all_results': all_results,
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata or {}
        }
        
        # 添加到历史
        self.history.append(record)
        
        # 更新元数据
        self.metadata['total_generations'] = generation + 1
        if best_fitness > self.metadata['best_fitness_ever']:
            self.metadata['best_fitness_ever'] = best_fitness
            self.metadata['best_generation'] = generation
        
        # 持久化到文件
        filepath = os.path.join(self.save_dir, f"generation_{generation:03d}.json")
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(record, f, indent=2, ensure_ascii=False)
        
        # 保存元数据
        metadata_path = os.path.join(self.save_dir, "metadata.json")
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)
        
        print(f"✅ 第 {generation} 代记录已保存: {filepath}")
        print(f"   本代最优Fitness: {best_fitness:.4f}")
        print(f"   历史最优Fitness: {self.metadata['best_fitness_ever']:.4f} (第{self.metadata['best_generation']}代)")
    
    def get_best_code(self, generation: int) -> str:
        """
        获取指定代的最优代码
        
        Args:
            generation: 代数
        
        Returns:
            str: 最优代码
        """
        if generation < 0 or generation >= len(self.history):
            raise ValueError(f"无效的代数: {generation}")
        
        return self.history[generation]['best_code']
    
    def get_reflection(self, generation: int) -> str:
        """
        获取指定代的反思
        
        Args:
            generation: 代数
        
        Returns:
            str: 反思内容
        """
        if generation < 0 or generation >= len(self.history):
            raise ValueError(f"无效的代数: {generation}")
        
        return self.history[generation]['reflection']
    
    def get_best_ever(self) -> Dict:
        """
        获取历史最优记录
        
        Returns:
            Dict: 最优记录
        """
        if not self.history:
            raise ValueError("记忆为空，无法获取最优记录")
        
        best_gen = self.metadata['best_generation']
        return self.history[best_gen]
    
    def get_fitness_history(self) -> List[float]:
        """
        获取每代的最优fitness历史
        
        Returns:
            List[float]: fitness列表
        """
        return [record['best_fitness'] for record in self.history]
    
    def load_from_disk(self, generation: Optional[int] = None):
        """
        从磁盘加载记录
        
        Args:
            generation: 如果指定，只加载特定代；否则加载所有代
        """
        if generation is not None:
            # 加载特定代
            filepath = os.path.join(self.save_dir, f"generation_{generation:03d}.json")
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"找不到记录文件: {filepath}")
            
            with open(filepath, 'r', encoding='utf-8') as f:
                record = json.load(f)
            
            # 确保历史列表足够长
            while len(self.history) <= generation:
                self.history.append(None)
            
            self.history[generation] = record
            print(f"✅ 已加载第 {generation} 代记录")
        
        else:
            # 加载所有代
            self.history = []
            generation = 0
            
            while True:
                filepath = os.path.join(self.save_dir, f"generation_{generation:03d}.json")
                if not os.path.exists(filepath):
                    break
                
                with open(filepath, 'r', encoding='utf-8') as f:
                    record = json.load(f)
                
                self.history.append(record)
                generation += 1
            
            print(f"✅ 已加载 {len(self.history)} 代记录")
            
            # 加载元数据
            metadata_path = os.path.join(self.save_dir, "metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    self.metadata = json.load(f)
    
    def export_summary(self, filepath: Optional[str] = None) -> str:
        """
        导出进化过程摘要
        
        Args:
            filepath: 保存路径，如果为None则不保存文件
        
        Returns:
            str: 摘要文本
        """
        lines = []
        lines.append("=" * 80)
        lines.append("进化过程摘要")
        lines.append("=" * 80)
        lines.append(f"创建时间: {self.metadata.get('creation_time', '未知')}")
        lines.append(f"总代数: {self.metadata.get('total_generations', 0)}")
        lines.append(f"历史最优Fitness: {self.metadata.get('best_fitness_ever', 0):.4f}")
        lines.append(f"最优出现在第 {self.metadata.get('best_generation', -1)} 代")
        lines.append("")
        
        # 每代的fitness趋势
        lines.append("【Fitness进化曲线】")
        lines.append("-" * 80)
        for i, record in enumerate(self.history):
            marker = " ⭐" if i == self.metadata.get('best_generation', -1) else ""
            lines.append(f"第 {i:2d} 代: {record['best_fitness']:.4f}{marker}")
        
        lines.append("")
        
        # 最优代码
        lines.append("【历史最优代码】")
        lines.append("-" * 80)
        if self.history:
            best_record = self.get_best_ever()
            lines.append(f"```python")
            lines.append(best_record['best_code'])
            lines.append(f"```")
        
        lines.append("")
        lines.append("=" * 80)
        
        summary = '\n'.join(lines)
        
        # 保存到文件
        if filepath:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(summary)
            print(f"✅ 摘要已保存到: {filepath}")
        
        return summary
    
    def plot_evolution_curve(self, save_path: Optional[str] = None):
        """
        绘制进化曲线
        
        Args:
            save_path: 图片保存路径
        """
        try:
            import matplotlib.pyplot as plt
            import numpy as np
        except ImportError:
            print("⚠️ matplotlib未安装，无法绘制曲线")
            return
        
        if not self.history:
            print("⚠️ 没有数据可绘制")
            return
        
        # 提取数据
        generations = list(range(len(self.history)))
        best_fitness = [record['best_fitness'] for record in self.history]
        
        # 计算平均fitness
        avg_fitness = []
        for record in self.history:
            all_results = record.get('all_results', [])
            if all_results:
                avg = np.mean([r.get('fitness', 0) for r in all_results])
                avg_fitness.append(avg)
            else:
                avg_fitness.append(0)
        
        # 绘图
        plt.figure(figsize=(10, 6))
        plt.plot(generations, best_fitness, 'o-', label='Best Fitness', linewidth=2, markersize=8)
        plt.plot(generations, avg_fitness, 's--', label='Average Fitness', linewidth=2, markersize=6, alpha=0.7)
        
        # 标记最优点
        best_gen = self.metadata.get('best_generation', -1)
        if best_gen >= 0:
            plt.scatter([best_gen], [best_fitness[best_gen]], 
                       c='red', s=200, marker='*', zorder=5, label='Best Ever')
        
        plt.xlabel('Generation', fontsize=14)
        plt.ylabel('Fitness', fontsize=14)
        plt.title('Evolution Curve of Reward Function Design', fontsize=16, fontweight='bold')
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # 保存或显示
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ 进化曲线已保存到: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def clear(self):
        """清空所有历史数据"""
        self.history = []
        self.metadata = {
            'creation_time': datetime.now().isoformat(),
            'total_generations': 0,
            'best_fitness_ever': -float('inf'),
            'best_generation': -1
        }
        print("✅ 记忆已清空")


# ========================================
# 测试代码
# ========================================
if __name__ == "__main__":
    print("测试进化记忆管理...")
    
    # 创建记忆实例
    memory = EvolutionaryMemory(save_dir="test_logs/evolution_test")
    
    # 模拟保存3代数据
    for gen in range(3):
        test_code = f"""
import numpy as np

def compute_reward(agent_name, observation, global_state, actions, world):
    components = {{}}
    # 第{gen}代的实现
    components['distance_reward'] = -{gen + 1} * 0.1
    total_reward = sum(components.values())
    return total_reward, components
"""
        
        test_reflection = f"第{gen}代的反思：需要调整权重，改进队形..."
        
        test_results = [
            {'id': 0, 'fitness': 0.6 + gen * 0.1, 'success_rate': 0.7},
            {'id': 1, 'fitness': 0.65 + gen * 0.1, 'success_rate': 0.75},
            {'id': 2, 'fitness': 0.7 + gen * 0.1, 'success_rate': 0.8},
            {'id': 3, 'fitness': 0.55 + gen * 0.1, 'success_rate': 0.65},
        ]
        
        memory.save(
            generation=gen,
            best_code=test_code,
            reflection=test_reflection,
            all_results=test_results
        )
    
    print("\n" + "=" * 80)
    
    # 测试获取功能
    print("\n=== 测试获取功能 ===")
    print(f"第0代最优代码（前100字符）:\n{memory.get_best_code(0)[:100]}...")
    print(f"\n第1代反思:\n{memory.get_reflection(1)}")
    
    print(f"\n历史最优:\n第{memory.get_best_ever()['generation']}代, Fitness={memory.get_best_ever()['best_fitness']:.4f}")
    
    print(f"\nFitness历史: {memory.get_fitness_history()}")
    
    # 导出摘要
    print("\n=== 导出摘要 ===")
    summary = memory.export_summary(filepath="test_logs/evolution_summary.txt")
    print(summary[:500] + "...")
    
    # 绘制曲线（如果有matplotlib）
    print("\n=== 绘制进化曲线 ===")
    memory.plot_evolution_curve(save_path="test_logs/evolution_curve.png")
    
    # 测试加载
    print("\n=== 测试加载 ===")
    new_memory = EvolutionaryMemory(save_dir="test_logs/evolution_test")
    new_memory.load_from_disk()
    print(f"加载的历史记录数: {len(new_memory.history)}")
    
    print("\n✅ 进化记忆管理测试完成！")
