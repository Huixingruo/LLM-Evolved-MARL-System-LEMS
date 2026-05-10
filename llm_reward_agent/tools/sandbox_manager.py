"""
沙盒管理器
为每个候选奖励函数创建独立的训练环境

Author: LEMS Project
Date: 2026-02-03
Version: 1.0
"""

import os
import shutil
from typing import List
from datetime import datetime


class SandboxManager:
    """管理并行训练的沙盒目录"""
    
    def __init__(self, base_dir: str = "experiments"):
        """
        初始化沙盒管理器
        
        Args:
            base_dir: 实验基础目录
        """
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)
        print(f"✅ 沙盒管理器初始化: {os.path.abspath(base_dir)}")
    
    def create_sandboxes(self, generation: int, codes: List[str]) -> List[str]:
        """
        为每个候选代码创建独立的沙盒
        
        Args:
            generation: 代数
            codes: 候选代码列表
        
        Returns:
            List[str]: 沙盒目录路径列表
        """
        print(f"\n{'='*80}")
        print(f"📦 创建第 {generation} 代训练沙盒")
        print(f"{'='*80}")
        
        # 创建代目录
        gen_dir = os.path.join(self.base_dir, f"generation_{generation:03d}")
        os.makedirs(gen_dir, exist_ok=True)
        
        sandbox_paths = []
        for i, code in enumerate(codes):
            sandbox_path = os.path.join(gen_dir, f"candidate_{i}")
            
            # 如果沙盒已存在，先清理
            if os.path.exists(sandbox_path):
                print(f"  [候选{i}] 清理旧沙盒...")
                shutil.rmtree(sandbox_path)
            
            os.makedirs(sandbox_path, exist_ok=True)
            
            # 1. 设置基座代码
            print(f"  [候选{i}] 复制基座代码...")
            self._setup_base_code(sandbox_path)
            
            # 2. 写入生成的奖励函数
            reward_file = os.path.join(sandbox_path, "MADDPG", "envs", "reward_function.py")
            os.makedirs(os.path.dirname(reward_file), exist_ok=True)
            
            with open(reward_file, 'w', encoding='utf-8') as f:
                f.write(code)
            
            print(f"  [候选{i}] 沙盒创建完成: {sandbox_path}")
            sandbox_paths.append(sandbox_path)
        
        print(f"\n✅ 成功创建 {len(sandbox_paths)} 个沙盒")
        print(f"{'='*80}")
        
        return sandbox_paths
    
    def _setup_base_code(self, sandbox_path: str):
        """
        设置基础代码（复制MADDPG目录）
        
        Args:
            sandbox_path: 沙盒路径
        """
        src_maddpg = os.path.abspath("MADDPG")
        dst_maddpg = os.path.join(sandbox_path, "MADDPG")
        
        if not os.path.exists(src_maddpg):
            raise FileNotFoundError(f"MADDPG源目录不存在: {src_maddpg}")
        
        # Windows系统：使用复制方式（忽略不必要的文件）
        ignore_patterns = shutil.ignore_patterns(
            '__pycache__',
            '*.pyc',
            '*.pyo',
            'models',  # 不复制已有模型
            'logs',    # 不复制旧日志
            'plot'     # 不复制图表
        )
        
        try:
            shutil.copytree(
                src_maddpg, 
                dst_maddpg, 
                ignore=ignore_patterns,
                dirs_exist_ok=True
            )
        except Exception as e:
            print(f"⚠️ 复制代码时出错: {e}")
            # 如果失败，尝试简化版本（只复制必需文件）
            self._setup_minimal_code(src_maddpg, dst_maddpg)
    
    def _setup_minimal_code(self, src_dir: str, dst_dir: str):
        """
        最小化复制（只复制必需文件）
        
        Args:
            src_dir: 源目录
            dst_dir: 目标目录
        """
        os.makedirs(dst_dir, exist_ok=True)
        
        # 必需的文件和目录
        essential_items = [
            'agents',
            'envs',
            'utils',
            'main_train.py',
            'main_parameters.py'
        ]
        
        for item in essential_items:
            src_item = os.path.join(src_dir, item)
            dst_item = os.path.join(dst_dir, item)
            
            if not os.path.exists(src_item):
                continue
            
            if os.path.isdir(src_item):
                # 复制目录（忽略缓存）
                ignore = shutil.ignore_patterns('__pycache__', '*.pyc')
                shutil.copytree(src_item, dst_item, ignore=ignore, dirs_exist_ok=True)
            else:
                # 复制文件
                shutil.copy2(src_item, dst_item)
    
    def cleanup_generation(self, generation: int):
        """
        清理指定代的沙盒目录
        
        Args:
            generation: 代数
        """
        gen_dir = os.path.join(self.base_dir, f"generation_{generation:03d}")
        
        if os.path.exists(gen_dir):
            shutil.rmtree(gen_dir)
            print(f"✅ 已清理第 {generation} 代沙盒")
    
    def cleanup_all(self):
        """清理所有沙盒目录（保留evolution_archive）"""
        if os.path.exists(self.base_dir):
            for item in os.listdir(self.base_dir):
                if item.startswith("generation_"):
                    item_path = os.path.join(self.base_dir, item)
                    shutil.rmtree(item_path)
            print("✅ 已清理所有代的沙盒")
    
    def get_sandbox_info(self, sandbox_path: str) -> dict:
        """
        获取沙盒信息
        
        Args:
            sandbox_path: 沙盒路径
        
        Returns:
            dict: 沙盒信息
        """
        info = {
            'path': sandbox_path,
            'exists': os.path.exists(sandbox_path),
            'size_mb': 0,
            'files_count': 0
        }
        
        if info['exists']:
            # 计算目录大小
            total_size = 0
            files_count = 0
            
            for dirpath, dirnames, filenames in os.walk(sandbox_path):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    total_size += os.path.getsize(filepath)
                    files_count += 1
            
            info['size_mb'] = total_size / (1024 * 1024)
            info['files_count'] = files_count
        
        return info
