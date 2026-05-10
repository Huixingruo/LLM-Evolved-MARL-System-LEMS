"""
奖励分量日志记录器
专门用于记录和分析奖励函数各个分量的统计信息

Author: LEMS Project
Date: 2026-02-02
Version: 1.0
"""

import json
import numpy as np
import os
from datetime import datetime
from typing import Dict, List


class RewardComponentLogger:
    """奖励分量日志记录器"""
    
    def __init__(self, log_dir="MADDPG/logs"):
        """
        初始化日志记录器
        
        Args:
            log_dir: 日志保存目录
        """
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # 奖励分量统计数据
        self.reward_component_history = {}  # {agent_name: {component_name: [values]}}
        self.collaboration_metrics_history = {}  # {metric_name: [values]}
        self.episode_count = 0
        
    def record_step(self, agent_name: str, components: Dict[str, float]):
        """
        记录单步的奖励分量
        
        Args:
            agent_name: 智能体名称
            components: 奖励分量字典
        """
        if agent_name not in self.reward_component_history:
            self.reward_component_history[agent_name] = {}
        
        for comp_name, value in components.items():
            if comp_name not in self.reward_component_history[agent_name]:
                self.reward_component_history[agent_name][comp_name] = []
            self.reward_component_history[agent_name][comp_name].append(value)
    
    def record_collaboration_metrics(self, metrics: Dict[str, float]):
        """
        记录协同行为指标
        
        Args:
            metrics: 协同指标字典
                {
                    'encirclement_angle_std': float,  # 围捕角度标准差
                    'min_agent_distance': float,      # 智能体最小距离
                    'avg_distance_to_prey': float,    # 到猎物的平均距离
                    'formation_quality': float,       # 队形质量
                }
        """
        for metric_name, value in metrics.items():
            if metric_name not in self.collaboration_metrics_history:
                self.collaboration_metrics_history[metric_name] = []
            self.collaboration_metrics_history[metric_name].append(value)
    
    def compute_statistics(self) -> Dict:
        """
        计算奖励分量的统计信息
        
        Returns:
            dict: 统计信息字典
        """
        stats = {
            'reward_components': {},
            'collaboration_metrics': {},
            'task_performance': {}
        }
        
        # 计算每个智能体每个分量的统计信息
        for agent_name, components in self.reward_component_history.items():
            stats['reward_components'][agent_name] = {}
            for comp_name, values in components.items():
                if len(values) > 0:
                    stats['reward_components'][agent_name][comp_name] = {
                        'mean': float(np.mean(values)),
                        'std': float(np.std(values)),
                        'min': float(np.min(values)),
                        'max': float(np.max(values)),
                        'sum': float(np.sum(values)),
                        'count': len(values)
                    }
        
        # 计算协同指标的统计信息
        for metric_name, values in self.collaboration_metrics_history.items():
            if len(values) > 0:
                stats['collaboration_metrics'][metric_name] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values))
                }
        
        return stats
    
    def compute_aggregated_statistics(self) -> Dict:
        """
        计算聚合统计信息（跨智能体）
        
        Returns:
            dict: 聚合统计信息
        """
        agg_stats = {
            'reward_components': {},
            'collaboration_metrics': {},
        }
        
        # 聚合所有智能体的奖励分量
        all_component_names = set()
        for agent_components in self.reward_component_history.values():
            all_component_names.update(agent_components.keys())
        
        for comp_name in all_component_names:
            all_values = []
            for agent_components in self.reward_component_history.values():
                if comp_name in agent_components:
                    all_values.extend(agent_components[comp_name])
            
            if len(all_values) > 0:
                agg_stats['reward_components'][comp_name] = {
                    'mean': float(np.mean(all_values)),
                    'std': float(np.std(all_values)),
                    'min': float(np.min(all_values)),
                    'max': float(np.max(all_values)),
                    'sum': float(np.sum(all_values)),
                    'count': len(all_values)
                }
        
        # 协同指标统计
        for metric_name, values in self.collaboration_metrics_history.items():
            if len(values) > 0:
                agg_stats['collaboration_metrics'][metric_name] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'final': float(values[-1]) if len(values) > 0 else 0.0
                }
        
        return agg_stats
    
    def save_statistics(self, filepath: str = None, aggregated: bool = True):
        """
        保存统计信息到JSON文件
        
        Args:
            filepath: 保存路径，如果为None则自动生成
            aggregated: 是否保存聚合统计（True）还是详细统计（False）
        """
        if filepath is None:
            timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            filename = f"reward_component_stats_{timestamp}.json"
            filepath = os.path.join(self.log_dir, filename)
        
        if aggregated:
            stats = self.compute_aggregated_statistics()
        else:
            stats = self.compute_statistics()
        
        # 添加元数据
        stats['metadata'] = {
            'timestamp': datetime.now().isoformat(),
            'episode_count': self.episode_count,
            'aggregated': aggregated
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        print(f"奖励分量统计已保存到: {filepath}")
        return filepath
    
    def generate_summary_report(self) -> str:
        """
        生成人类可读的摘要报告
        
        Returns:
            str: 摘要报告文本
        """
        stats = self.compute_aggregated_statistics()
        
        report = []
        report.append("=" * 80)
        report.append("奖励分量统计报告")
        report.append("=" * 80)
        report.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"训练回合数: {self.episode_count}")
        report.append("")
        
        # 奖励分量统计
        report.append("【奖励分量统计】")
        report.append("-" * 80)
        if 'reward_components' in stats and len(stats['reward_components']) > 0:
            for comp_name, comp_stats in stats['reward_components'].items():
                report.append(f"\n{comp_name}:")
                report.append(f"  均值: {comp_stats['mean']:>10.4f}")
                report.append(f"  标准差: {comp_stats['std']:>10.4f}")
                report.append(f"  最小值: {comp_stats['min']:>10.4f}")
                report.append(f"  最大值: {comp_stats['max']:>10.4f}")
                report.append(f"  总和: {comp_stats['sum']:>10.4f}")
                report.append(f"  样本数: {comp_stats['count']}")
        else:
            report.append("  无数据")
        
        report.append("")
        
        # 协同指标统计
        report.append("【协同行为指标】")
        report.append("-" * 80)
        if 'collaboration_metrics' in stats and len(stats['collaboration_metrics']) > 0:
            for metric_name, metric_stats in stats['collaboration_metrics'].items():
                report.append(f"\n{metric_name}:")
                report.append(f"  均值: {metric_stats['mean']:>10.4f}")
                report.append(f"  标准差: {metric_stats['std']:>10.4f}")
                report.append(f"  最小值: {metric_stats['min']:>10.4f}")
                report.append(f"  最大值: {metric_stats['max']:>10.4f}")
                if 'final' in metric_stats:
                    report.append(f"  最终值: {metric_stats['final']:>10.4f}")
        else:
            report.append("  无数据")
        
        report.append("")
        report.append("=" * 80)
        
        return "\n".join(report)
    
    def save_summary_report(self, filepath: str = None):
        """
        保存摘要报告到文本文件
        
        Args:
            filepath: 保存路径，如果为None则自动生成
        """
        if filepath is None:
            timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            filename = f"reward_summary_report_{timestamp}.txt"
            filepath = os.path.join(self.log_dir, filename)
        
        report = self.generate_summary_report()
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"摘要报告已保存到: {filepath}")
        return filepath
    
    def clear(self):
        """清空所有历史数据"""
        self.reward_component_history = {}
        self.collaboration_metrics_history = {}
        self.episode_count = 0
    
    def print_summary(self):
        """打印摘要报告到控制台"""
        report = self.generate_summary_report()
        print(report)


def compute_encirclement_angle_std(adversary_positions: np.ndarray, prey_position: np.ndarray) -> float:
    """
    计算围捕角度的标准差（越小越均匀）
    
    Args:
        adversary_positions: 追捕者位置数组 shape: (n_adversaries, 2)
        prey_position: 猎物位置 shape: (2,)
    
    Returns:
        float: 角度标准差
    """
    if len(adversary_positions) < 2:
        return 0.0
    
    # 计算每个追捕者相对于猎物的角度
    rel_vectors = adversary_positions - prey_position
    angles = np.arctan2(rel_vectors[:, 1], rel_vectors[:, 0])
    
    # 排序
    sorted_angles = np.sort(angles)
    
    # 计算相邻角度差
    angle_diffs = np.diff(sorted_angles)
    # 加上首尾角度差
    angle_diffs = np.append(angle_diffs, 2 * np.pi - (sorted_angles[-1] - sorted_angles[0]))
    
    # 理想角度间隔
    ideal_angle_sep = 2 * np.pi / len(adversary_positions)
    
    # 计算偏差的标准差
    return float(np.std(angle_diffs - ideal_angle_sep))


def compute_formation_quality(adversary_positions: np.ndarray, prey_position: np.ndarray, capture_threshold: float) -> float:
    """
    计算队形质量（0-1之间，越大越好）
    
    综合考虑：
    1. 角度均匀性
    2. 距离一致性
    3. 是否在捕获范围内
    
    Args:
        adversary_positions: 追捕者位置数组
        prey_position: 猎物位置
        capture_threshold: 捕获阈值
    
    Returns:
        float: 队形质量分数 [0, 1]
    """
    if len(adversary_positions) < 2:
        return 0.0
    
    # 1. 角度均匀性（越小越好）
    angle_std = compute_encirclement_angle_std(adversary_positions, prey_position)
    angle_score = np.exp(-2 * angle_std)  # 转换为分数，越小越好
    
    # 2. 距离一致性
    distances = np.linalg.norm(adversary_positions - prey_position, axis=1)
    distance_std = np.std(distances)
    distance_score = np.exp(-distance_std)
    
    # 3. 是否在捕获范围内
    in_range_count = np.sum(distances < capture_threshold)
    range_score = in_range_count / len(adversary_positions)
    
    # 综合得分
    quality = 0.4 * angle_score + 0.3 * distance_score + 0.3 * range_score
    
    return float(quality)


# ========================================
# 测试代码
# ========================================
if __name__ == "__main__":
    print("测试奖励分量日志记录器...")
    
    # 创建日志记录器
    logger = RewardComponentLogger(log_dir="test_logs")
    
    # 模拟记录数据
    for episode in range(3):
        logger.episode_count += 1
        
        for step in range(50):
            # 模拟追捕者奖励分量
            for i in range(3):
                components = {
                    'distance_reward': np.random.randn() - 1.0,
                    'collision_penalty': np.random.randn() * 0.5 - 2.0,
                    'formation_reward': np.random.randn() * 0.3,
                    'capture_bonus': 20.0 if step > 40 else 0.0,
                    'energy_cost': np.random.randn() * 0.01 - 0.02,
                    'boundary_penalty': 0.0
                }
                logger.record_step(f'adversary_{i}', components)
            
            # 模拟协同指标
            metrics = {
                'encirclement_angle_std': np.random.rand() * 0.5,
                'min_agent_distance': np.random.rand() * 0.5 + 0.2,
                'avg_distance_to_prey': np.random.rand() * 1.5 + 0.5,
                'formation_quality': np.random.rand()
            }
            logger.record_collaboration_metrics(metrics)
    
    # 打印摘要
    print("\n" + "="*80)
    logger.print_summary()
    
    # 保存统计信息
    logger.save_statistics()
    logger.save_summary_report()
    
    print("\n✅ 奖励分量日志记录器测试完成！")
