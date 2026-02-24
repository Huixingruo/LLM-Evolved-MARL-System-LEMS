"""
日志分析器
解析训练日志，提取性能指标，计算Fitness

Author: LEMS Project
Date: 2026-02-03
Version: 1.0
"""

import os
import json
import numpy as np
from typing import Dict, Optional, List


class LogAnalyzer:
    """训练日志分析器"""
    
    def __init__(self, fitness_config: Optional[Dict] = None):
        """
        初始化日志分析器
        
        Args:
            fitness_config: Fitness计算配置
        """
        # 默认Fitness配置
        self.fitness_config = fitness_config or {
            'weights': {
                'success_rate': 1.0,
                'capture_time': -0.001,
                'formation_quality': 0.3,
                'collision_penalty': -0.5
            },
            'normalize': {
                'max_capture_time': 100,
                'min_success_rate': 0.0,
                'max_success_rate': 1.0
            }
        }
    
    def parse_logs(self, sandbox_path: str) -> Dict:
        """
        解析训练日志，提取性能指标
        
        Args:
            sandbox_path: 沙盒目录路径
        
        Returns:
            dict: 性能指标字典
        """
        metrics = {
            'fitness': 0.0,
            'success_rate': 0.0,
            'avg_capture_time': 100.0,
            'reward_components': {},
            'collaboration_metrics': {},
            'raw_data': {}
        }
        
        # 1. 尝试读取奖励分量统计文件
        stats_file = self._find_latest_stats_file(sandbox_path)
        
        if stats_file and os.path.exists(stats_file):
            try:
                with open(stats_file, 'r', encoding='utf-8') as f:
                    stats = json.load(f)
                
                # 提取奖励分量
                if 'reward_components' in stats:
                    metrics['reward_components'] = stats['reward_components']
                
                # 提取协同指标
                if 'collaboration_metrics' in stats:
                    metrics['collaboration_metrics'] = stats['collaboration_metrics']
                
                metrics['raw_data'] = stats
                
                print(f"    ✅ 成功读取奖励分量统计")
            
            except Exception as e:
                print(f"    ⚠️ 读取奖励分量统计失败: {e}")
        
        # 2. 尝试读取训练日志（包含episode统计）
        training_log = os.path.join(sandbox_path, "MADDPG", "logs", "training_log.json")
        
        if os.path.exists(training_log):
            try:
                with open(training_log, 'r', encoding='utf-8') as f:
                    log_data = json.load(f)
                
                # 提取任务性能指标
                metrics.update(self._extract_task_performance(log_data))
                
                print(f"    ✅ 成功读取训练日志")
            
            except Exception as e:
                print(f"    ⚠️ 读取训练日志失败: {e}")
        
        # 3. 计算Fitness
        metrics['fitness'] = self.calculate_fitness(metrics)
        
        return metrics
    
    def _find_latest_stats_file(self, sandbox_path: str) -> Optional[str]:
        """
        查找最新的奖励分量统计文件
        
        Args:
            sandbox_path: 沙盒路径
        
        Returns:
            str: 文件路径，如果不存在返回None
        """
        logs_dir = os.path.join(sandbox_path, "MADDPG", "logs")
        
        if not os.path.exists(logs_dir):
            return None
        
        # 查找所有奖励统计文件
        stats_files = [
            f for f in os.listdir(logs_dir)
            if f.startswith("reward_component_stats_") and f.endswith(".json")
        ]
        
        if not stats_files:
            return None
        
        # 返回最新的文件
        stats_files.sort(reverse=True)
        return os.path.join(logs_dir, stats_files[0])
    
    def _extract_task_performance(self, log_data: Dict) -> Dict:
        """
        从训练日志中提取任务性能指标
        
        Args:
            log_data: 训练日志数据
        
        Returns:
            dict: 任务性能指标
        """
        performance = {
            'success_rate': 0.0,
            'avg_capture_time': 100.0,
            'total_episodes': 0
        }
        
        # 处理日志数据为数组的情况（当前实际格式）
        # 格式: [{"成功围捕次数": 56, "成功围捕率": "1.12%", "平均回合步数": "99.5", ...}]
        if isinstance(log_data, list) and len(log_data) > 0:
            # 取第一个元素（包含训练统计信息）
            log_entry = log_data[0]
            
            # 提取总回合数
            if '总回合数' in log_entry:
                performance['total_episodes'] = log_entry['总回合数']
            
            # 提取成功率 - 尝试多种方式
            if '成功围捕率' in log_entry:
                success_rate_str = log_entry['成功围捕率']
                if isinstance(success_rate_str, str):
                    # 格式: "1.12%" -> 0.0112
                    success_rate_str = success_rate_str.replace('%', '').strip()
                    try:
                        performance['success_rate'] = float(success_rate_str) / 100.0
                    except ValueError:
                        pass
                elif isinstance(success_rate_str, (int, float)):
                    performance['success_rate'] = float(success_rate_str)
            
            # 如果没有成功围捕率，尝试用成功围捕次数计算
            if performance['success_rate'] == 0.0 and '成功围捕次数' in log_entry and '总回合数' in log_entry:
                captures = log_entry['成功围捕次数']
                total = log_entry['总回合数']
                if total > 0:
                    performance['success_rate'] = captures / total
            
            # 提取平均捕获时间
            if '平均回合步数' in log_entry:
                avg_steps = log_entry['平均回合步数']
                if isinstance(avg_steps, str):
                    try:
                        performance['avg_capture_time'] = float(avg_steps)
                    except ValueError:
                        pass
                elif isinstance(avg_steps, (int, float)):
                    performance['avg_capture_time'] = float(avg_steps)
            
            return performance
        
        # 提取episode统计（旧格式）
        if 'episodes' in log_data:
            episodes = log_data['episodes']
            performance['total_episodes'] = len(episodes)
            
            if episodes:
                # 计算成功率（简化版：假设有捕获成功的标记）
                # 实际实现可能需要根据你的日志格式调整
                
                # 统计平均reward作为成功的代理指标
                total_rewards = []
                capture_times = []
                
                for ep in episodes:
                    if 'total_reward' in ep:
                        total_rewards.append(ep['total_reward'])
                    
                    if 'steps' in ep:
                        capture_times.append(ep['steps'])
                
                # 成功率：根据reward阈值判断
                if total_rewards:
                    # 假设正reward表示成功（这个阈值需要根据实际调整）
                    success_threshold = 0.0
                    successes = sum(1 for r in total_rewards if r > success_threshold)
                    performance['success_rate'] = successes / len(total_rewards)
                
                # 平均捕获时间
                if capture_times:
                    performance['avg_capture_time'] = np.mean(capture_times)
        
        # 如果日志中直接有任务性能统计（阶段一增强后的格式）
        if 'task_performance' in log_data:
            tp = log_data['task_performance']
            if 'success_rate' in tp:
                performance['success_rate'] = tp['success_rate']
            if 'avg_capture_time' in tp:
                performance['avg_capture_time'] = tp['avg_capture_time']
        
        return performance
    
    def calculate_fitness(self, metrics: Dict) -> float:
        """
        计算综合Fitness分数
        
        Args:
            metrics: 性能指标字典
        
        Returns:
            float: Fitness分数
        """
        weights = self.fitness_config['weights']
        
        # 提取指标
        success_rate = metrics.get('success_rate', 0.0)
        capture_time = metrics.get('avg_capture_time', 100.0)
        
        # 基础Fitness：成功率 - 时间惩罚
        fitness = (
            weights['success_rate'] * success_rate +
            weights['capture_time'] * capture_time
        )
        
        # 可选：添加队形质量
        if 'collaboration_metrics' in metrics:
            collab = metrics['collaboration_metrics']
            
            if 'formation_quality' in collab:
                formation_quality = collab['formation_quality']
                if isinstance(formation_quality, dict):
                    formation_quality = formation_quality.get('mean', 0)
                fitness += weights.get('formation_quality', 0.3) * formation_quality
        
        # 可选：添加碰撞惩罚
        if 'reward_components' in metrics:
            comps = metrics['reward_components']
            
            # 查找碰撞惩罚分量
            for key, val in comps.items():
                if 'collision' in key.lower():
                    collision_value = val
                    if isinstance(collision_value, dict):
                        collision_value = collision_value.get('mean', 0)
                    fitness += weights.get('collision_penalty', -0.5) * abs(collision_value)
        
        return fitness
    
    def generate_analysis_report(self, metrics: Dict) -> str:
        """
        生成人类可读的分析报告
        
        Args:
            metrics: 性能指标字典
        
        Returns:
            str: 分析报告文本
        """
        lines = []
        lines.append("=" * 60)
        lines.append("训练结果分析报告")
        lines.append("=" * 60)
        
        # Fitness
        lines.append(f"\n【综合评分】")
        lines.append(f"Fitness: {metrics['fitness']:.4f}")
        
        # 任务性能
        lines.append(f"\n【任务性能】")
        lines.append(f"成功率: {metrics['success_rate']:.2%}")
        lines.append(f"平均捕获时间: {metrics['avg_capture_time']:.1f} steps")
        
        # 奖励分量
        if metrics['reward_components']:
            lines.append(f"\n【奖励分量统计】")
            for comp_name, comp_val in metrics['reward_components'].items():
                if isinstance(comp_val, dict):
                    mean = comp_val.get('mean', 0)
                    std = comp_val.get('std', 0)
                    lines.append(f"  {comp_name}: mean={mean:.4f}, std={std:.4f}")
                else:
                    lines.append(f"  {comp_name}: {comp_val:.4f}")
        
        # 协同指标
        if metrics['collaboration_metrics']:
            lines.append(f"\n【协同行为指标】")
            for metric_name, metric_val in metrics['collaboration_metrics'].items():
                if isinstance(metric_val, dict):
                    mean = metric_val.get('mean', 0)
                    lines.append(f"  {metric_name}: {mean:.4f}")
                else:
                    lines.append(f"  {metric_name}: {metric_val:.4f}")
        
        lines.append("\n" + "=" * 60)
        
        return '\n'.join(lines)


# ========================================
# 测试代码
# ========================================
if __name__ == "__main__":
    print("测试日志分析器...")
    
    # 创建分析器
    analyzer = LogAnalyzer()
    
    # 测试1: 解析模拟数据
    print("\n[测试1] 解析模拟数据...")
    
    test_metrics = {
        'success_rate': 0.75,
        'avg_capture_time': 48.5,
        'reward_components': {
            'distance_reward': {'mean': -1.23, 'std': 0.45},
            'collision_penalty': {'mean': -0.15, 'std': 0.12},
            'formation_reward': {'mean': 0.34, 'std': 0.08}
        },
        'collaboration_metrics': {
            'formation_quality': {'mean': 0.56}
        }
    }
    
    # 计算Fitness
    fitness = analyzer.calculate_fitness(test_metrics)
    print(f"计算的Fitness: {fitness:.4f}")
    
    # 生成报告
    report = analyzer.generate_analysis_report({**test_metrics, 'fitness': fitness})
    print(f"\n{report}")
    
    # 测试2: 解析实际日志（如果存在）
    print("\n[测试2] 解析实际日志...")
    
    # 查找最近的训练日志
    logs_dir = "MADDPG/logs"
    if os.path.exists(logs_dir):
        # 尝试解析最新的统计文件
        stats_files = [
            f for f in os.listdir(logs_dir)
            if f.startswith("reward_component_stats_") and f.endswith(".json")
        ]
        
        if stats_files:
            stats_files.sort(reverse=True)
            latest_file = os.path.join(logs_dir, stats_files[0])
            
            print(f"读取文件: {latest_file}")
            
            # 模拟从沙盒读取
            # 这里我们直接从主目录读取作为测试
            test_sandbox = "."
            real_metrics = analyzer.parse_logs(test_sandbox)
            
            print(f"\n实际日志分析:")
            print(f"Fitness: {real_metrics['fitness']:.4f}")
            print(f"成功率: {real_metrics['success_rate']:.2%}")
            print(f"捕获时间: {real_metrics['avg_capture_time']:.1f}")
        else:
            print("未找到奖励统计文件")
    else:
        print(f"日志目录不存在: {logs_dir}")
    
    print("\n✅ 日志分析器测试完成！")
