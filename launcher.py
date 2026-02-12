"""
并行训练调度器
管理多个候选奖励函数的并行训练

Author: LEMS Project
Date: 2026-02-03
Version: 1.0
"""

import subprocess
import time
import os
import json
import sys
from multiprocessing import Pool
from typing import List, Dict


class ParallelLauncher:
    """并行训练任务调度器"""
    
    def __init__(self, max_workers: int = 4, timeout: int = 12000, episode_num: int = 100):
        """
        初始化并行调度器
        
        Args:
            max_workers: 最大并行数（取决于CPU核心数）
            timeout: 单个训练超时时间（秒）
            episode_num: 训练回合数（轻量化训练）
        """
        self.max_workers = max_workers
        self.timeout = timeout
        self.episode_num = episode_num
        
        print(f"✅ 并行调度器初始化:")
        print(f"   最大并行数: {max_workers}")
        print(f"   超时时间: {timeout}秒")
        print(f"   训练回合数: {episode_num}")
    
    def run_parallel(self, sandbox_paths: List[str]) -> List[Dict]:
        """
        并行执行训练任务
        
        Args:
            sandbox_paths: 沙盒目录列表
        
        Returns:
            List[Dict]: 每个候选的训练结果
        """
        print(f"\n{'='*80}")
        print(f"🚀 启动 {len(sandbox_paths)} 个并行训练任务")
        print(f"{'='*80}\n")
        
        start_time = time.time()
        
        # 使用multiprocessing.Pool并行执行
        with Pool(processes=self.max_workers) as pool:
            results = pool.map(self._run_single_training, sandbox_paths)
        
        elapsed = time.time() - start_time
        
        print(f"\n{'='*80}")
        print(f"✅ 所有训练任务完成")
        print(f"   总耗时: {elapsed:.1f}秒 ({elapsed/60:.1f}分钟)")
        print(f"{'='*80}")
        
        return results
    
    def _run_single_training(self, sandbox_path: str) -> Dict:
        """
        执行单个训练任务（子进程）
        
        Args:
            sandbox_path: 沙盒目录路径
        
        Returns:
            dict: 训练结果
        """
        candidate_id = os.path.basename(sandbox_path)
        
        print(f"  [{candidate_id}] 开始训练...")
        
        # 1. 组装命令
        # 使用Python可执行文件路径
        python_exe = sys.executable  # 使用当前Python环境
        
        cmd = [
            python_exe,
            os.path.join("MADDPG", "main_train.py"),
            "--env_name", "simple_tag_env",
            "--episode_num", str(self.episode_num),
            "--episode_length", "100",
            "--render_mode", "None"
        ]
        
        # 2. 执行训练（捕获输出，避免屏幕混乱）
        try:
            start_time = time.time()
            
            result = subprocess.run(
                cmd,
                cwd=sandbox_path,  # 【关键】在沙盒目录中执行
                capture_output=True,
                text=True,
                timeout=self.timeout,
                encoding='utf-8',  # 指定编码
                errors='ignore'    # 忽略编码错误
            )
            
            elapsed = time.time() - start_time
            
            if result.returncode == 0:
                print(f"  [{candidate_id}] ✅ 训练完成 ({elapsed:.1f}秒)")
                
                # 3. 解析日志
                metrics = self._parse_logs(sandbox_path)
                
                return {
                    'id': candidate_id,
                    'status': 'success',
                    'metrics': metrics,
                    'fitness': metrics.get('fitness', 0.0),
                    'elapsed': elapsed
                }
            else:
                # 训练失败
                error_msg = result.stderr[-500:] if result.stderr else "未知错误"
                print(f"  [{candidate_id}] ❌ 训练失败")
                print(f"    错误信息: {error_msg[:200]}...")
                
                return {
                    'id': candidate_id,
                    'status': 'error',
                    'error': error_msg,
                    'fitness': 0.0,
                    'elapsed': elapsed
                }
        
        except subprocess.TimeoutExpired:
            print(f"  [{candidate_id}] ⏱️ 训练超时 (>{self.timeout}秒)")
            return {
                'id': candidate_id,
                'status': 'timeout',
                'fitness': 0.0
            }
        
        except Exception as e:
            print(f"  [{candidate_id}] ❌ 未知错误: {e}")
            return {
                'id': candidate_id,
                'status': 'error',
                'error': str(e),
                'fitness': 0.0
            }
    
    def _parse_logs(self, sandbox_path: str) -> Dict:
        """
        解析训练日志，提取性能指标
        
        Args:
            sandbox_path: 沙盒目录路径
        
        Returns:
            dict: 性能指标
        """
        # 使用LogAnalyzer解析日志
        try:
            from llm_reward_agent.tools.log_analyzer import LogAnalyzer
            
            analyzer = LogAnalyzer()
            metrics = analyzer.parse_logs(sandbox_path)
            
            return metrics
        
        except Exception as e:
            print(f"    ⚠️ 日志解析失败: {e}")
            
            # 返回默认指标
            return {
                'fitness': 0.0,
                'success_rate': 0.0,
                'avg_capture_time': 100.0,
                'reward_components': {},
                'collaboration_metrics': {}
            }
    
    def run_sequential(self, sandbox_paths: List[str]) -> List[Dict]:
        """
        串行执行训练任务（用于调试）
        
        Args:
            sandbox_paths: 沙盒目录列表
        
        Returns:
            List[Dict]: 训练结果列表
        """
        print(f"\n{'='*80}")
        print(f"🔄 串行执行 {len(sandbox_paths)} 个训练任务（调试模式）")
        print(f"{'='*80}\n")
        
        results = []
        for sandbox_path in sandbox_paths:
            result = self._run_single_training(sandbox_path)
            results.append(result)
        
        print(f"\n✅ 所有训练任务完成")
        
        return results


# ========================================
# 测试代码
# ========================================
if __name__ == "__main__":
    print("测试并行训练调度器...")
    
    # 测试1: 检查环境
    print("\n[测试1] 检查环境...")
    print(f"Python可执行文件: {sys.executable}")
    print(f"当前工作目录: {os.getcwd()}")
    
    # 检查MADDPG目录
    if os.path.exists("MADDPG"):
        print("✅ MADDPG目录存在")
    else:
        print("❌ MADDPG目录不存在")
    
    # 测试2: 创建测试沙盒并运行（需要先有沙盒）
    print("\n[测试2] 创建测试沙盒...")
    
    # 首先创建测试沙盒
    from llm_reward_agent.tools.sandbox_manager import SandboxManager
    
    test_code = """
import numpy as np

def compute_reward(agent_name, observation, global_state, actions, world):
    \"\"\"测试奖励函数\"\"\"
    components = {}
    
    # 简单的距离奖励
    if global_state.get('is_adversary', False):
        dist = np.linalg.norm(
            global_state['agent_positions'][0] - global_state['prey_position']
        )
        components['distance_reward'] = -0.1 * dist
    else:
        components['escape_reward'] = 0.1
    
    components['boundary_penalty'] = 0.0
    
    total_reward = sum(components.values())
    return total_reward, components
"""
    
    manager = SandboxManager(base_dir="test_logs/launcher_test")
    sandboxes = manager.create_sandboxes(generation=0, codes=[test_code])
    
    # 测试3: 串行运行一个训练（快速测试）
    print("\n[测试3] 串行运行测试...")
    
    launcher = ParallelLauncher(
        max_workers=1,
        timeout=300,  # 5分钟超时
        episode_num=10  # 只训练10回合（快速测试）
    )
    
    print("\n⚠️ 注意：这将实际运行训练，可能需要几分钟...")
    choice = input("是否继续？(y/n): ").strip().lower()
    
    if choice == 'y':
        results = launcher.run_sequential(sandboxes)
        
        print("\n" + "="*80)
        print("训练结果:")
        print("="*80)
        
        for result in results:
            print(f"\n候选 {result['id']}:")
            print(f"  状态: {result['status']}")
            print(f"  Fitness: {result.get('fitness', 0):.4f}")
            if result['status'] == 'success':
                metrics = result.get('metrics', {})
                print(f"  成功率: {metrics.get('success_rate', 0):.2%}")
                print(f"  捕获时间: {metrics.get('avg_capture_time', 0):.1f}")
    else:
        print("跳过实际训练测试")
    
    # 清理
    print("\n清理测试文件...")
    manager.cleanup_all()
    
    print("\n✅ 并行训练调度器测试完成！")
