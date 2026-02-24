"""
并行训练调度器
管理多个候选奖励函数的并行训练

支持CPU和GPU两种模式：
- CPU模式：使用multiprocessing.Pool并行训练
- GPU模式：使用subprocess并行启动多个训练进程

Date: 2026-02-03
Version: 1.1 (添加GPU支持)
"""

import subprocess
import time
import os
import json
import sys
import torch
from multiprocessing import Pool
from typing import List, Dict, Optional


class ParallelLauncher:
    """并行训练任务调度器"""
    
    def __init__(
        self,
        max_workers: int = 4,
        timeout: int = 12000,
        episode_num: int = 100,
        use_gpu: bool = True,
        gpu_ids: Optional[List[int]] = None
    ):
        """
        初始化并行调度器
        
        Args:
            max_workers: 最大并行数（取决于CPU核心数）
            timeout: 单个训练超时时间（秒）
            episode_num: 训练回合数
            use_gpu: 是否使用GPU训练
            gpu_ids: GPU设备ID列表（如[0,1,2,3]）
        """
        self.max_workers = max_workers
        self.timeout = timeout
        self.episode_num = episode_num
        self.use_gpu = use_gpu
        self.gpu_ids = gpu_ids or [0]
        
        # 检查GPU可用性
        if use_gpu:
            self._check_gpu_available()
        
        print(f"✅ 并行调度器初始化:")
        print(f"   最大并行数: {max_workers}")
        print(f"   超时时间: {timeout}秒")
        print(f"   训练回合数: {episode_num}")
        print(f"   计算设备: {'GPU ' + str(self.gpu_ids) if use_gpu else 'CPU'}")
    
    def _check_gpu_available(self) -> bool:
        """检查GPU是否可用"""
        if not torch.cuda.is_available():
            print("⚠️ CUDA不可用，将回退到CPU模式")
            self.use_gpu = False
            return False
        
        print(f"✅ GPU可用:")
        print(f"   GPU数量: {torch.cuda.device_count()}")
        for gpu_id in self.gpu_ids:
            if gpu_id < torch.cuda.device_count():
                print(f"   GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
        return True
    
    def run_parallel(self, sandbox_paths: List[str]) -> List[Dict]:
        """
        并行执行训练任务（根据use_gpu选择CPU或GPU模式）
        
        Args:
            sandbox_paths: 沙盒目录列表
        
        Returns:
            List[Dict]: 每个候选的训练结果
        """
        if self.use_gpu:
            return self._run_parallel_gpu(sandbox_paths)
        else:
            return self._run_parallel_cpu(sandbox_paths)
    
    def _run_parallel_cpu(self, sandbox_paths: List[str]) -> List[Dict]:
        """
        CPU模式并行执行（使用multiprocessing.Pool）
        
        Args:
            sandbox_paths: 沙盒目录列表
        
        Returns:
            List[Dict]: 每个候选的训练结果
        """
        print(f"\n{'='*80}")
        print(f"🚀 启动 {len(sandbox_paths)} 个并行训练任务 (CPU模式)")
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
    
    def _run_parallel_gpu(self, sandbox_paths: List[str]) -> List[Dict]:
        """
        GPU模式并行执行（使用subprocess同时启动多个进程）
        
        Args:
            sandbox_paths: 沙盒目录列表
        
        Returns:
            List[Dict]: 每个候选的训练结果
        """
        print(f"\n{'='*80}")
        print(f"🚀 启动 {len(sandbox_paths)} 个并行训练任务 (GPU模式)")
        print(f"   GPU设备: {self.gpu_ids}")
        print(f"{'='*80}\n")
        
        # 检查GPU数量是否足够
        if len(sandbox_paths) > len(self.gpu_ids):
            print(f"⚠️ 警告: 训练任务数({len(sandbox_paths)}) > GPU数量({len(self.gpu_ids)})")
            print(f"   多个任务将共享GPU，可能导致内存不足")
        
        start_time = time.time()
        
        # 启动所有训练进程
        processes = []
        
        for i, sandbox_path in enumerate(sandbox_paths):
            # 分配GPU（循环使用）
            gpu_id = self.gpu_ids[i % len(self.gpu_ids)]
            
            proc, log_file = self._start_training_process_v2(sandbox_path, gpu_id)
            processes.append((proc, sandbox_path, gpu_id, log_file))
        
        # 等待所有进程完成
        print("\n等待训练完成...")
        results = []
        completed = 0
        
        for proc, sandbox_path, gpu_id, log_file in processes:
            candidate_id = os.path.basename(sandbox_path)
            completed += 1
            print(f"  [{completed}/{len(processes)}] 等待 {candidate_id} (GPU {gpu_id})...", end="", flush=True)
            
            # 等待进程完成
            result = self._wait_training_process_v2(proc, sandbox_path, log_file)
            results.append(result)
            print(f" {result['status']}")
        
        elapsed = time.time() - start_time
        
        print(f"\n{'='*80}")
        print(f"✅ 所有训练任务完成")
        print(f"   总耗时: {elapsed:.1f}秒 ({elapsed/60:.1f}分钟)")
        print(f"{'='*80}")
        
        return results
    
    def _start_training_process_v2(self, sandbox_path: str, gpu_id: int):
        """
        启动单个训练进程（返回Popen对象）

        Args:
            sandbox_path: 沙盒目录路径
            gpu_id: GPU设备ID

        Returns:
            tuple: (Popen对象, 日志文件路径)
        """
        # 设置环境变量指定GPU
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        # 设置Python的标准输出编码为UTF-8，避免Windows下的中文乱码问题
        env['PYTHONIOENCODING'] = 'utf-8'

        python_exe = sys.executable
        cmd = [
            python_exe,
            os.path.join("MADDPG", "main_train.py"),
            "--env_name", "simple_tag_env",
            "--episode_num", str(self.episode_num),
            "--episode_length", "100",
            "--render_mode", "None"
        ]

        # 创建日志文件
        log_file = os.path.join(sandbox_path, "training_output.log")

        # 使用文件重定向，避免管道阻塞
        # 明确指定encoding='utf-8'，确保中文字符正确写入
        log_fp = open(log_file, 'w', encoding='utf-8')
        proc = subprocess.Popen(
            cmd,
            cwd=sandbox_path,
            stdout=log_fp,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8',  # 明确指定输出编码为UTF-8
            env=env
        )

        return proc, log_file
    
    def _wait_training_process_v2(self, proc: subprocess.Popen, sandbox_path: str, log_file: str) -> Dict:
        """
        等待训练进程完成（使用Popen对象）
        
        Args:
            proc: Popen进程对象
            sandbox_path: 沙盒目录路径
            log_file: 日志文件路径
        
        Returns:
            dict: 训练结果
        """
        candidate_id = os.path.basename(sandbox_path)
        
        # 等待进程完成
        try:
            return_code = proc.wait(timeout=self.timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            return_code = -1
        finally:
            # 关闭日志文件
            try:
                proc.stdout.close()
            except:
                pass
        
        # 读取日志文件最后几行
        try:
            with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                output = f.read()
                output_lines = output.strip().split('\n')
                if len(output_lines) > 5:
                    print(f"\n    最后输出: {' | '.join(output_lines[-3:])}")
        except Exception as e:
            output = ""
        
        # 解析日志
        metrics = self._parse_logs(sandbox_path)
        
        if return_code == 0:
            return {
                'id': candidate_id,
                'status': 'success',
                'metrics': metrics,
                'fitness': metrics.get('fitness', 0.0)
            }
        else:
            return {
                'id': candidate_id,
                'status': 'error' if return_code != -1 else 'timeout',
                'metrics': metrics,
                'fitness': 0.0
            }
    
    
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
        python_exe = sys.executable
        
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
                cwd=sandbox_path,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                encoding='utf-8',
                errors='ignore'
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
        try:
            from llm_reward_agent.tools.log_analyzer import LogAnalyzer
            
            analyzer = LogAnalyzer()
            metrics = analyzer.parse_logs(sandbox_path)
            
            return metrics
        
        except Exception as e:
            print(f"    ⚠️ 日志解析失败: {e}")
            
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
        print(f"🔄 串行执行 {len(sandbox_paths)} 个训练任务")
        print(f"{'='*80}\n")
        
        results = []
        for sandbox_path in sandbox_paths:
            result = self._run_single_training(sandbox_path)
            results.append(result)
        
        print(f"\n✅ 所有训练任务完成")
        
        return results
    
    def run_gpu_sequential(self, sandbox_paths: List[str]) -> List[Dict]:
        """
        GPU串行执行训练任务（一个接一个地训练，更稳定）
        
        Args:
            sandbox_paths: 沙盒目录列表
        
        Returns:
            List[Dict]: 训练结果列表
        """
        print(f"\n{'='*80}")
        print(f"🔄 GPU串行执行 {len(sandbox_paths)} 个训练任务")
        print(f"   GPU设备: {self.gpu_ids}")
        print(f"{'='*80}\n")
        
        results = []
        
        for i, sandbox_path in enumerate(sandbox_paths):
            gpu_id = self.gpu_ids[0]  # 使用第一个GPU
            print(f"\n--- 训练候选 {i+1}/{len(sandbox_paths)} (GPU {gpu_id}) ---")
            
            result = self._run_single_training_gpu(sandbox_path, gpu_id)
            results.append(result)
        
        print(f"\n✅ 所有训练任务完成")
        
        return results
    
    def _run_single_training_gpu(self, sandbox_path: str, gpu_id: int = 0) -> Dict:
        """
        执行单个GPU训练任务

        Args:
            sandbox_path: 沙盒目录路径
            gpu_id: GPU设备ID

        Returns:
            dict: 训练结果
        """
        candidate_id = os.path.basename(sandbox_path)

        print(f"  [{candidate_id}] 🎮 开始GPU训练 (GPU {gpu_id})...")

        # 设置环境变量指定GPU
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        # 设置Python的标准输出编码为UTF-8，避免Windows下的中文乱码问题
        env['PYTHONIOENCODING'] = 'utf-8'

        python_exe = sys.executable
        cmd = [
            python_exe,
            os.path.join("MADDPG", "main_train.py"),
            "--env_name", "simple_tag_env",
            "--episode_num", str(self.episode_num),
            "--episode_length", "100",
            "--render_mode", "None"
        ]

        # 创建日志文件
        log_file = os.path.join(sandbox_path, "training_output.log")

        try:
            start_time = time.time()

            # 使用文件重定向，明确指定编码为UTF-8
            with open(log_file, 'w', encoding='utf-8') as log_fp:
                result = subprocess.run(
                    cmd,
                    cwd=sandbox_path,
                    stdout=log_fp,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding='utf-8',  # 明确指定输出编码为UTF-8
                    timeout=self.timeout,
                    env=env
                )
            
            elapsed = time.time() - start_time
            
            if result.returncode == 0:
                print(f"  [{candidate_id}] ✅ 训练完成 ({elapsed:.1f}秒)")
                
                metrics = self._parse_logs(sandbox_path)
                
                return {
                    'id': candidate_id,
                    'status': 'success',
                    'metrics': metrics,
                    'fitness': metrics.get('fitness', 0.0),
                    'elapsed': elapsed,
                    'gpu': gpu_id
                }
            else:
                print(f"  [{candidate_id}] ❌ 训练失败")
                
                return {
                    'id': candidate_id,
                    'status': 'error',
                    'fitness': 0.0,
                    'elapsed': elapsed,
                    'gpu': gpu_id
                }
        
        except subprocess.TimeoutExpired:
            print(f"  [{candidate_id}] ⏱️ 训练超时")
            return {
                'id': candidate_id,
                'status': 'timeout',
                'fitness': 0.0,
                'gpu': gpu_id
            }
        
        except Exception as e:
            print(f"  [{candidate_id}] ❌ 错误: {e}")
            return {
                'id': candidate_id,
                'status': 'error',
                'error': str(e),
                'fitness': 0.0,
                'gpu': gpu_id
            }


# ========================================
# 测试代码
# ========================================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="并行训练调度器")
    parser.add_argument("--mode", type=str, default="gpu", choices=["cpu", "gpu"], help="训练模式")
    parser.add_argument("--gpu-ids", type=str, default="0", help="GPU IDs (逗号分隔, 如 '0,1,2,3')")
    parser.add_argument("--workers", type=int, default=4, help="并行数")
    parser.add_argument("--epochs", type=int, default=100, help="训练回合数")
    parser.add_argument("--timeout", type=int, default=3600, help="超时时间(秒)")
    parser.add_argument("--sequential",  default=False, help="串行执行")
    
    args = parser.parse_args()
    
    # 解析GPU IDs
    gpu_ids = [int(x) for x in args.gpu_ids.split(',')]
    
    print("="*80)
    print("并行训练调度器测试")
    print("="*80)
    print(f"模式: {args.mode}")
    print(f"GPU IDs: {gpu_ids}")
    print(f"并行数: {args.workers}")
    print(f"训练回合数: {args.epochs}")
    print(f"超时: {args.timeout}秒")
    print("="*80)
    
    # 检查环境
    print(f"\nPython可执行文件: {sys.executable}")
    print(f"当前工作目录: {os.getcwd()}")
    
    if args.mode == "gpu":
        if torch.cuda.is_available():
            print(f"✅ GPU可用: {torch.cuda.device_count()}个")
            for i in range(torch.cuda.device_count()):
                print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            print("❌ GPU不可用")
            sys.exit(1)
    
    # 检查MADDPG目录
    if os.path.exists("MADDPG"):
        print("✅ MADDPG目录存在")
    else:
        print("❌ MADDPG目录不存在")
        sys.exit(1)
    
    # 创建测试沙盒（4个不同版本的奖励函数）
    print("\n创建测试沙盒...")
    from llm_reward_agent.tools.sandbox_manager import SandboxManager
    
    # 4个不同的奖励函数测试代码
    test_codes = [
        # 候选1: 基础距离奖励
        """
import numpy as np

def compute_reward(agent_name, observation, global_state, actions, world):
    components = {}
    
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
""",
        # 候选2: 包含协作奖励
        """
import numpy as np

def compute_reward(agent_name, observation, global_state, actions, world):
    components = {}
    
    if global_state.get('is_adversary', False):
        dist = np.linalg.norm(
            global_state['agent_positions'][0] - global_state['prey_position']
        )
        components['distance_reward'] = -0.1 * dist
        
        # 协作奖励：接近其他智能体
        positions = global_state.get('agent_positions', [])
        if len(positions) > 1:
            other_pos = positions[1] if agent_name != 'adversary_0' else positions[0]
            dist_to_ally = np.linalg.norm(global_state['agent_positions'][0] - other_pos)
            components['collaboration_reward'] = -0.05 * dist_to_ally
    else:
        components['escape_reward'] = 0.1
    
    components['boundary_penalty'] = 0.0
    
    total_reward = sum(components.values())
    return total_reward, components
""",
        # 候选3: 包含速度奖励
        """
import numpy as np

def compute_reward(agent_name, observation, global_state, actions, world):
    components = {}
    
    if global_state.get('is_adversary', False):
        dist = np.linalg.norm(
            global_state['agent_positions'][0] - global_state['prey_position']
        )
        components['distance_reward'] = -0.1 * dist
        
        # 速度奖励
        if 'velocity' in global_state:
            speed = np.linalg.norm(global_state['velocity'])
            components['speed_reward'] = 0.02 * speed
    else:
        components['escape_reward'] = 0.1
    
    components['boundary_penalty'] = 0.0
    
    total_reward = sum(components.values())
    return total_reward, components
""",
        # 候选4: 综合奖励（距离+协作+速度）
        """
import numpy as np

def compute_reward(agent_name, observation, global_state, actions, world):
    components = {}
    
    if global_state.get('is_adversary', False):
        dist = np.linalg.norm(
            global_state['agent_positions'][0] - global_state['prey_position']
        )
        components['distance_reward'] = -0.1 * dist
        
        # 协作奖励
        positions = global_state.get('agent_positions', [])
        if len(positions) > 1:
            other_pos = positions[1] if agent_name != 'adversary_0' else positions[0]
            dist_to_ally = np.linalg.norm(global_state['agent_positions'][0] - other_pos)
            components['collaboration_reward'] = -0.05 * dist_to_ally
        
        # 速度奖励
        if 'velocity' in global_state:
            speed = np.linalg.norm(global_state['velocity'])
            components['speed_reward'] = 0.02 * speed
    else:
        components['escape_reward'] = 0.1
    
    components['boundary_penalty'] = 0.0
    
    total_reward = sum(components.values())
    return total_reward, components
"""
    ]
    
    manager = SandboxManager(base_dir="test_logs/launcher_test")
    sandboxes = manager.create_sandboxes(generation=0, codes=test_codes)
    
    print(f"✅ 创建了 {len(sandboxes)} 个测试沙盒:")
    for i, sbx in enumerate(sandboxes):
        print(f"   沙盒 {i+1}: {os.path.basename(sbx)}")
    
    # 创建调度器
    launcher = ParallelLauncher(
        max_workers=args.workers,
        timeout=args.timeout,
        episode_num=args.epochs,
        use_gpu=(args.mode == "gpu"),
        gpu_ids=gpu_ids
    )
    
    print("\n⚠️ 注意：这将实际运行训练，可能需要较长时间...")
    choice = input("是否继续？(y/n): ").strip().lower()
    
    if choice == 'y':
        if args.sequential:
            if args.mode == "gpu":
                results = launcher.run_gpu_sequential(sandboxes)
            else:
                results = launcher.run_sequential(sandboxes)
        else:
            results = launcher.run_parallel(sandboxes)
        
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
    
    print("\n✅ 测试完成！")
