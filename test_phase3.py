"""
阶段三测试脚本
测试并行训练框架

测试内容：
1. 沙盒管理器测试
2. 日志分析器测试
3. 并行调度器测试
4. 仿真工具集成测试
5. 完整流程测试

Author: LEMS Project
Date: 2026-02-03
Version: 1.0

运行环境: Python 3.11.8 (MPE: conda)
"""

import os
import sys
import unittest

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from llm_reward_agent.tools import (
    SandboxManager,
    LogAnalyzer,
    SimulationTool
)
from launcher import ParallelLauncher


class TestSandboxManager(unittest.TestCase):
    """测试沙盒管理器"""
    
    def setUp(self):
        """测试前准备"""
        self.test_dir = "test_logs/sandbox_test"
        self.manager = SandboxManager(base_dir=self.test_dir)
    
    def tearDown(self):
        """测试后清理"""
        import shutil
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_create_sandboxes(self):
        """测试创建沙盒"""
        print("\n[测试] 创建沙盒...")
        
        test_codes = [
            "import numpy as np\ndef compute_reward(...): return 0, {}",
            "import numpy as np\ndef compute_reward(...): return 1, {}"
        ]
        
        sandboxes = self.manager.create_sandboxes(generation=0, codes=test_codes)
        
        self.assertEqual(len(sandboxes), 2)
        
        for sandbox in sandboxes:
            self.assertTrue(os.path.exists(sandbox))
            
            # 检查MADDPG目录是否复制
            maddpg_dir = os.path.join(sandbox, "MADDPG")
            self.assertTrue(os.path.exists(maddpg_dir))
            
            # 检查奖励函数文件
            reward_file = os.path.join(maddpg_dir, "envs", "reward_function.py")
            self.assertTrue(os.path.exists(reward_file))
        
        print("✅ 沙盒创建成功")
    
    def test_sandbox_info(self):
        """测试获取沙盒信息"""
        print("\n[测试] 获取沙盒信息...")
        
        test_code = "# test code"
        sandboxes = self.manager.create_sandboxes(generation=0, codes=[test_code])
        
        info = self.manager.get_sandbox_info(sandboxes[0])
        
        self.assertTrue(info['exists'])
        self.assertGreater(info['size_mb'], 0)
        self.assertGreater(info['files_count'], 0)
        
        print(f"✅ 沙盒大小: {info['size_mb']:.2f} MB, 文件数: {info['files_count']}")


class TestLogAnalyzer(unittest.TestCase):
    """测试日志分析器"""
    
    def test_fitness_calculation(self):
        """测试Fitness计算"""
        print("\n[测试] Fitness计算...")
        
        analyzer = LogAnalyzer()
        
        test_metrics = {
            'success_rate': 0.75,
            'avg_capture_time': 50.0,
            'reward_components': {
                'distance_reward': {'mean': -1.0},
                'collision_penalty': {'mean': -0.2}
            }
        }
        
        fitness = analyzer.calculate_fitness(test_metrics)
        
        self.assertIsInstance(fitness, float)
        self.assertGreater(fitness, 0)
        
        print(f"✅ Fitness计算成功: {fitness:.4f}")
    
    def test_log_parsing(self):
        """测试日志解析"""
        print("\n[测试] 日志解析...")
        
        analyzer = LogAnalyzer()
        
        # 尝试解析项目根目录的日志（如果存在）
        if os.path.exists("MADDPG/logs"):
            metrics = analyzer.parse_logs(".")
            
            self.assertIsInstance(metrics, dict)
            self.assertIn('fitness', metrics)
            
            print(f"✅ 日志解析成功")
            print(f"   Fitness: {metrics['fitness']:.4f}")
        else:
            print("⚠️ 日志目录不存在，跳过测试")


class TestParallelLauncher(unittest.TestCase):
    """测试并行调度器"""
    
    def test_initialization(self):
        """测试初始化"""
        print("\n[测试] 并行调度器初始化...")
        
        launcher = ParallelLauncher(max_workers=2, timeout=300, episode_num=10)
        
        self.assertEqual(launcher.max_workers, 2)
        self.assertEqual(launcher.timeout, 300)
        self.assertEqual(launcher.episode_num, 10)
        
        print("✅ 并行调度器初始化成功")
    
    def test_single_training(self):
        """测试单个训练任务（可选，耗时）"""
        # 这个测试会实际运行训练，默认跳过
        self.skipTest("跳过实际训练测试（耗时较长）")
        
        print("\n[测试] 单个训练任务...")
        
        # 创建测试沙盒
        manager = SandboxManager(base_dir="test_logs/launcher_test")
        test_code = """
import numpy as np

def compute_reward(agent_name, observation, global_state, actions, world):
    components = {}
    components['test_reward'] = 0.1
    total_reward = sum(components.values())
    return total_reward, components
"""
        sandboxes = manager.create_sandboxes(generation=0, codes=[test_code])
        
        # 运行训练
        launcher = ParallelLauncher(max_workers=1, timeout=300, episode_num=5)
        results = launcher.run_sequential(sandboxes)
        
        self.assertEqual(len(results), 1)
        self.assertIn('status', results[0])
        
        print(f"✅ 训练完成: {results[0]['status']}")
        
        # 清理
        manager.cleanup_all()


class TestSimulationTool(unittest.TestCase):
    """测试仿真工具集成"""
    
    def setUp(self):
        """测试前准备"""
        self.test_dir = "test_logs/simulation_test"
    
    def tearDown(self):
        """测试后清理"""
        import shutil
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_initialization(self):
        """测试初始化"""
        print("\n[测试] 仿真工具初始化...")
        
        sim_tool = SimulationTool(
            base_dir=self.test_dir,
            max_workers=2,
            timeout=300,
            episode_num=10
        )
        
        self.assertIsNotNone(sim_tool.sandbox_mgr)
        self.assertIsNotNone(sim_tool.launcher)
        
        print("✅ 仿真工具初始化成功")
    
    def test_parallel_training(self):
        """测试并行训练（可选，耗时）"""
        # 这个测试会实际运行训练，默认跳过
        self.skipTest("跳过实际并行训练测试（耗时较长）")
        
        print("\n[测试] 并行训练...")
        
        test_codes = [
            """
import numpy as np

def compute_reward(agent_name, observation, global_state, actions, world):
    components = {}
    components['test_reward'] = 0.1
    return sum(components.values()), components
""",
            """
import numpy as np

def compute_reward(agent_name, observation, global_state, actions, world):
    components = {}
    components['test_reward'] = 0.2
    return sum(components.values()), components
"""
        ]
        
        sim_tool = SimulationTool(
            base_dir=self.test_dir,
            max_workers=2,
            timeout=300,
            episode_num=5
        )
        
        results = sim_tool.run_parallel(codes=test_codes, generation=0)
        
        self.assertEqual(len(results), 2)
        
        for result in results:
            self.assertIn('status', result)
            self.assertIn('fitness', result)
        
        print("✅ 并行训练完成")


def run_basic_tests():
    """运行基础测试（不包含耗时的训练测试）"""
    print("=" * 80)
    print("阶段三基础功能测试")
    print("=" * 80)
    
    # 创建测试套件
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # 添加测试类
    suite.addTests(loader.loadTestsFromTestCase(TestSandboxManager))
    suite.addTests(loader.loadTestsFromTestCase(TestLogAnalyzer))
    suite.addTests(loader.loadTestsFromTestCase(TestParallelLauncher))
    suite.addTests(loader.loadTestsFromTestCase(TestSimulationTool))
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # 输出摘要
    print("\n" + "=" * 80)
    print("测试摘要")
    print("=" * 80)
    print(f"总测试数: {result.testsRun}")
    print(f"成功: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"失败: {len(result.failures)}")
    print(f"错误: {len(result.errors)}")
    print(f"跳过: {len(result.skipped)}")
    
    if result.wasSuccessful():
        print("\n✅ 所有基础测试通过！")
    else:
        print("\n❌ 部分测试失败，请检查输出")
    
    print("=" * 80)
    
    return result


def run_integration_test():
    """运行集成测试（包含实际训练，可选）"""
    print("\n" + "=" * 80)
    print("阶段三集成测试（包含实际训练）")
    print("=" * 80)
    print("\n⚠️ 警告：这将运行实际的训练任务，可能需要5-10分钟")
    print("   训练配置：2个候选，每个10回合")
    
    choice = input("\n是否继续？(y/n): ").strip().lower()
    
    if choice != 'y':
        print("跳过集成测试")
        return
    
    print("\n开始集成测试...")
    
    # 准备测试代码
    test_codes = [
        """
import numpy as np

def compute_reward(agent_name, observation, global_state, actions, world):
    \"\"\"候选0: 基础距离奖励\"\"\"
    components = {}
    
    if global_state.get('is_adversary', False):
        agent_idx = int(agent_name.split('_')[1])
        agent_pos = global_state['agent_positions'][agent_idx]
        prey_pos = global_state['prey_position']
        dist = np.linalg.norm(agent_pos - prey_pos)
        
        components['distance_reward'] = -0.1 * dist
        components['boundary_penalty'] = 0.0
    else:
        components['escape_reward'] = 0.1
        components['boundary_penalty'] = 0.0
    
    total_reward = sum(components.values())
    return total_reward, components
""",
        """
import numpy as np

def compute_reward(agent_name, observation, global_state, actions, world):
    \"\"\"候选1: 增强距离奖励\"\"\"
    components = {}
    
    if global_state.get('is_adversary', False):
        agent_idx = int(agent_name.split('_')[1])
        agent_pos = global_state['agent_positions'][agent_idx]
        prey_pos = global_state['prey_position']
        dist = np.linalg.norm(agent_pos - prey_pos)
        
        components['distance_reward'] = -0.2 * dist
        components['boundary_penalty'] = 0.0
    else:
        components['escape_reward'] = 0.1
        components['boundary_penalty'] = 0.0
    
    total_reward = sum(components.values())
    return total_reward, components
"""
    ]
    
    # 创建仿真工具
    sim_tool = SimulationTool(
        base_dir="test_logs/integration_test",
        max_workers=2,
        timeout=600,  # 10分钟超时
        episode_num=10  # 只训练10回合
    )
    
    # 运行并行训练
    print("\n开始并行训练...")
    results = sim_tool.run_parallel(codes=test_codes, generation=0)
    
    # 分析结果
    print("\n" + "=" * 80)
    print("集成测试结果")
    print("=" * 80)
    
    for result in results:
        print(f"\n{result['id']}:")
        print(f"  状态: {result['status']}")
        print(f"  Fitness: {result.get('fitness', 0):.4f}")
        
        if result['status'] == 'success':
            metrics = result['metrics']
            print(f"  成功率: {metrics.get('success_rate', 0):.2%}")
            print(f"  捕获时间: {metrics.get('avg_capture_time', 0):.1f} steps")
            print(f"  耗时: {result.get('elapsed', 0):.1f}秒")
    
    # 清理
    print("\n清理测试文件...")
    sim_tool.cleanup_all()
    
    print("\n✅ 集成测试完成！")


if __name__ == "__main__":
    # 运行基础测试
    result = run_basic_tests()
    
    # 询问是否运行集成测试
    if result.wasSuccessful():
        print("\n" + "=" * 80)
        run_integration_test()
    
    # 返回退出码
    sys.exit(0 if result.wasSuccessful() else 1)
