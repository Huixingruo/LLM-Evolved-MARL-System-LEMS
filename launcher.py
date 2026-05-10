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
import sys
import torch
from multiprocessing import Pool
from typing import List, Dict, Optional

# ============================================================
# 【关键修复】在导入matplotlib相关模块之前，设置后端为Agg
# 避免Windows下tkinter错误: RuntimeError: main thread is not in main loop
# ============================================================
# 注意：此文件主要通过subprocess启动训练进程，
# 但可能在主进程中触发一些matplotlib初始化
if 'MPLBACKEND' not in os.environ:
    os.environ['MPLBACKEND'] = 'Agg'
    os.environ['TK_SILENCE_IGNORE'] = '1'
    import matplotlib
    matplotlib.use('Agg', force=True)
# ============================================================


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
