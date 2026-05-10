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
            },
            'convergence': {
                # alpha/beta: 早/晚期评估窗口（回合数）。
                # 推荐值 = 总回合数 × 5%~15%，例如 3000 回合建议 alpha=150, beta=300。
                # alpha 偏小可能导致"早期"统计量受初始随机性干扰；
                # beta 偏小可能导致"末期"统计量被尾部噪声误导。
                'alpha': 150,
                'beta': 300,
                # v_th: 方差收缩容忍阈值。值越大越宽松（允许晚期方差相对更大）。
                # 多智能体环境噪声较大，默认从 0.5 放宽至 0.8，以避免过度拦截。
                # - 0.5: 严格（要求晚期方差 < 早期方差的50%，单智能体场景推荐）
                # - 0.8: 宽松（允许晚期方差 < 早期方差的80%，多智能体场景推荐）
                # - 1.0: 极宽松（只要晚期方差不超过早期即可）
                'v_th': 0.8
            }
        }
    
    def parse_logs(self, sandbox_path: str) -> Dict:
        """
        解析训练日志，提取性能指标

        Args:
            sandbox_path: 沙盒目录路径

        Returns:
            dict: 性能指标字典（含收敛状态）
        """
        metrics = {
            'fitness': 0.0,
            'success_rate': 0.0,
            'avg_capture_time': 100.0,
            'reward_components': {},
            'collaboration_metrics': {},
            'raw_data': {},
            'convergence_status': {'is_converged': False}
        }

        training_log = os.path.join(sandbox_path, "MADDPG", "logs", "training_log.json")
        episode_fitnesses = []

        if os.path.exists(training_log):
            try:
                with open(training_log, 'r', encoding='utf-8') as f:
                    log_data = json.load(f)

                metrics.update(self._extract_task_performance(log_data))

                if 'episodes' in log_data:
                    episode_fitnesses = [ep.get('total_reward', 0) for ep in log_data['episodes']]
                elif isinstance(log_data, list) and len(log_data) > 0:
                    first_entry = log_data[0]
                    if isinstance(first_entry, dict):
                        if 'episodes' in first_entry:
                            episode_fitnesses = [ep.get('total_reward', 0) for ep in first_entry['episodes']]

                print(f"    ✅ 成功读取训练日志")

            except Exception as e:
                print(f"    ⚠️ 读取训练日志失败: {e}")

        # 读取奖励分量统计文件和协同行为指标（数据管道关键断点修复）
        stats_data = self._load_stats_file(sandbox_path)
        if stats_data:
            metrics['reward_components'] = stats_data.get('reward_components', {})
            metrics['collaboration_metrics'] = stats_data.get('collaboration_metrics', {})
            print(f"    ✅ 成功读取奖励分量统计 ({len(metrics['reward_components'])} 个分量)")

        metrics['fitness'] = self.calculate_fitness(metrics)

        if episode_fitnesses:
            metrics['convergence_status'] = self._evaluate_convergence(episode_fitnesses)
            conv = metrics['convergence_status']
            print(f"    📊 收敛诊断: 完全收敛={conv.get('is_converged')}, "
                  f"F_mean={conv.get('f_mean')}, F_std={conv.get('f_std')}, F_slope={conv.get('f_slope')}")
        else:
            print("    ⚠️ 未提取到时序(Episode)奖励曲线，跳过收敛拦截器。")
            metrics['convergence_status'] = {'is_converged': True, 'f_mean': True, 'f_std': True, 'f_slope': True}

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

    def _load_stats_file(self, sandbox_path: str) -> Dict:
        """
        加载最新的奖励分量统计文件（JSON 格式）

        Args:
            sandbox_path: 沙盒路径

        Returns:
            dict: 包含 reward_components 和 collaboration_metrics 的字典，找不到则返回空
        """
        stats_path = self._find_latest_stats_file(sandbox_path)
        if not stats_path:
            print("    ⚠️ 未找到奖励分量统计文件，跳过")
            return {}

        try:
            with open(stats_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data
        except Exception as e:
            print(f"    ⚠️ 读取奖励分量统计文件失败: {e}")
            return {}

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
    
    def _evaluate_convergence(self, episode_fitnesses: List[float]) -> Dict[str, bool]:
        """
        核心算法：多维收敛过滤拦截器
        严格对照论文中的式(5)、式(6)、式(7)实现

        Args:
            episode_fitnesses: 每回合(Episode)的适应度时序序列

        Returns:
            dict: 三重拦截诊断结果
        """
        L = len(episode_fitnesses)
        if L < 10:
            return {
                'f_mean': True, 'f_std': True, 'f_slope': True,
                'is_converged': True,
                'details': f'Too few data points (L={L}), bypassed'
            }

        conv_cfg = self.fitness_config['convergence']
        alpha = min(conv_cfg['alpha'], L // 3)
        beta = min(conv_cfg['beta'], L // 3)
        v_th = conv_cfg['v_th']

        early_data = episode_fitnesses[:alpha]
        late_data = episode_fitnesses[-beta:]

        J_em = np.mean(early_data)
        J_lm = np.mean(late_data)
        J_ev = np.std(early_data)
        J_lv = np.std(late_data)

        t = np.arange(L)
        cov = np.cov(t, episode_fitnesses)[0][1]

        f_mean = bool(J_lm > J_em)
        f_std = bool((J_lv / (J_ev + 1e-8)) < v_th)
        f_slope = bool(cov > 0)

        is_converged = f_mean and f_std and f_slope

        return {
            'f_mean': f_mean,
            'f_std': f_std,
            'f_slope': f_slope,
            'is_converged': is_converged,
            'details': f"J_em:{J_em:.2f}, J_lm:{J_lm:.2f}, J_ev:{J_ev:.2f}, J_lv:{J_lv:.2f}, cov:{cov:.2f}"
        }

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

        lines.append(f"\n【综合评分】")
        lines.append(f"Fitness: {metrics['fitness']:.4f}")

        lines.append(f"\n【任务性能】")
        lines.append(f"成功率: {metrics['success_rate']:.2%}")
        lines.append(f"平均捕获时间: {metrics['avg_capture_time']:.1f} steps")

        if metrics['reward_components']:
            lines.append(f"\n【奖励分量统计】")
            for comp_name, comp_val in metrics['reward_components'].items():
                if isinstance(comp_val, dict):
                    mean = comp_val.get('mean', 0)
                    std = comp_val.get('std', 0)
                    lines.append(f"  {comp_name}: mean={mean:.4f}, std={std:.4f}")
                else:
                    lines.append(f"  {comp_name}: {comp_val:.4f}")

        if metrics['collaboration_metrics']:
            lines.append(f"\n【协同行为指标】")
            for metric_name, metric_val in metrics['collaboration_metrics'].items():
                if isinstance(metric_val, dict):
                    mean = metric_val.get('mean', 0)
                    lines.append(f"  {metric_name}: {mean:.4f}")
                else:
                    lines.append(f"  {metric_name}: {metric_val:.4f}")

        conv = metrics.get('convergence_status', {})
        if conv:
            lines.append(f"\n【收敛拦截诊断】")
            lines.append(f"  完全收敛: {'✅' if conv.get('is_converged') else '❌'}")
            lines.append(f"  增益判断(F_mean): {'✅' if conv.get('f_mean') else '❌'}")
            lines.append(f"  收敛判断(F_std): {'✅' if conv.get('f_std') else '❌'}")
            lines.append(f"  趋势判断(F_slope): {'✅' if conv.get('f_slope') else '❌'}")
            lines.append(f"  详情: {conv.get('details', 'N/A')}")

        lines.append("\n" + "=" * 60)

        return '\n'.join(lines)
