"""
仿真工具集成
整合沙盒管理器和并行调度器，提供统一的训练接口

Author: LEMS Project
Date: 2026-02-03
Version: 1.0
"""

import os
import sys
from typing import List, Dict

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from llm_reward_agent.tools.sandbox_manager import SandboxManager
from launcher import ParallelLauncher


class SimulationTool:
    """Agent的仿真执行工具"""
    
    def __init__(self, 
                 base_dir: str = "experiments",
                 max_workers: int = 4,
                 timeout: int = 1200,
                 episode_num: int = 100,
                 use_gpu: bool = True):
        """
        初始化仿真工具
        
        Args:
            base_dir: 实验基础目录
            max_workers: 最大并行数
            timeout: 训练超时时间（秒）
            episode_num: 训练回合数
            use_gpu: 是否使用GPU训练
        """
        print(f"\n{'='*80}")
        print("初始化仿真工具")
        print(f"{'='*80}")
        
        self.sandbox_mgr = SandboxManager(base_dir=base_dir)
        self.launcher = ParallelLauncher(
            max_workers=max_workers,
            timeout=timeout,
            episode_num=episode_num,
            use_gpu=use_gpu
        )
        
        print(f"{'='*80}\n")
    
    def run_parallel(self, codes: List[str], generation: int) -> List[Dict]:
        """
        并行运行多个候选代码
        
        Args:
            codes: LLM生成的代码列表
            generation: 当前代数
        
        Returns:
            List[Dict]: 训练结果（包含fitness、metrics等）
        """
        print(f"\n{'='*80}")
        print(f"🧬 第 {generation} 代并行训练")
        print(f"{'='*80}")
        
        # 1. 创建沙盒
        print(f"\n[步骤1/3] 创建训练沙盒...")
        sandbox_paths = self.sandbox_mgr.create_sandboxes(generation, codes)
        
        # 2. 并行执行训练
        print(f"\n[步骤2/3] 并行执行训练...")
        results = self.launcher.run_parallel(sandbox_paths)
        
        # 3. 为每个结果附加对应的代码
        print(f"\n[步骤3/3] 整理结果...")
        for i, result in enumerate(results):
            result['code'] = codes[i]
            result['generation'] = generation
            result['sandbox_path'] = sandbox_paths[i]
        
        # 打印摘要
        self._print_summary(results)
        
        return results
    
    def run_sequential(self, codes: List[str], generation: int) -> List[Dict]:
        """
        串行运行多个候选代码（用于调试）
        
        Args:
            codes: LLM生成的代码列表
            generation: 当前代数
        
        Returns:
            List[Dict]: 训练结果
        """
        print(f"\n{'='*80}")
        print(f"🔄 第 {generation} 代串行训练（调试模式）")
        print(f"{'='*80}")
        
        # 创建沙盒
        sandbox_paths = self.sandbox_mgr.create_sandboxes(generation, codes)
        
        # 串行执行
        results = self.launcher.run_sequential(sandbox_paths)
        
        # 附加信息
        for i, result in enumerate(results):
            result['code'] = codes[i]
            result['generation'] = generation
            result['sandbox_path'] = sandbox_paths[i]
        
        self._print_summary(results)
        
        return results
    
    def _print_summary(self, results: List[Dict]):
        """
        打印结果摘要
        
        Args:
            results: 训练结果列表
        """
        print(f"\n{'='*80}")
        print("训练结果摘要")
        print(f"{'='*80}")
        
        # 统计成功/失败数量
        success_count = sum(1 for r in results if r.get('status') == 'success')
        error_count = sum(1 for r in results if r.get('status') == 'error')
        timeout_count = sum(1 for r in results if r.get('status') == 'timeout')
        
        print(f"\n总计: {len(results)} 个候选")
        print(f"  ✅ 成功: {success_count}")
        print(f"  ❌ 失败: {error_count}")
        print(f"  ⏱️ 超时: {timeout_count}")
        
        # 打印各候选的Fitness
        print(f"\n{'候选ID':<15} {'状态':<10} {'Fitness':<12} {'成功率':<12} {'捕获时间':<12}")
        print("-" * 80)
        
        for result in results:
            candidate_id = result.get('id', 'unknown')
            status = result.get('status', 'unknown')
            fitness = result.get('fitness', 0.0)
            
            metrics = result.get('metrics', {})
            success_rate = metrics.get('success_rate', 0.0)
            capture_time = metrics.get('avg_capture_time', 0.0)
            
            status_symbol = {
                'success': '✅',
                'error': '❌',
                'timeout': '⏱️'
            }.get(status, '❓')
            
            print(f"{candidate_id:<15} {status_symbol} {status:<8} {fitness:<12.4f} {success_rate:<12.2%} {capture_time:<12.1f}")
        
        # 找到最优候选
        success_results = [r for r in results if r.get('status') == 'success']
        if success_results:
            best_result = max(success_results, key=lambda x: x.get('fitness', 0))
            print(f"\n⭐ 最优候选: {best_result['id']}")
            print(f"   Fitness: {best_result['fitness']:.4f}")
            print(f"   成功率: {best_result['metrics'].get('success_rate', 0):.2%}")
        
        print(f"\n{'='*80}")
    
    def cleanup_generation(self, generation: int):
        """
        清理指定代的沙盒
        
        Args:
            generation: 代数
        """
        self.sandbox_mgr.cleanup_generation(generation)
    
    def cleanup_all(self):
        """清理所有沙盒"""
        self.sandbox_mgr.cleanup_all()
