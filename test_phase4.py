"""
阶段四测试脚本
测试反馈闭环与整合功能

测试内容：
1. 主流程脚本参数解析
2. 错误处理和容错机制
3. 可视化工具
4. 完整进化流程（可选）

Author: LEMS Project
Date: 2026-02-03
Version: 1.0
"""

import os
import sys
import unittest

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from llm_reward_agent.agent import RewardDesignAgent
from visualization.evolution_plot import EvolutionPlotter


class TestMainScript(unittest.TestCase):
    """测试主流程脚本"""
    
    def test_script_exists(self):
        """测试脚本文件存在"""
        print("\n[测试] 检查主流程脚本...")
        
        script_path = "run_evolution.py"
        self.assertTrue(os.path.exists(script_path))
        print(f"[OK] 脚本存在: {script_path}")
    
    def test_script_syntax(self):
        """测试脚本语法"""
        print("\n[测试] 检查脚本语法...")
        
        import ast
        with open("run_evolution.py", 'r', encoding='utf-8') as f:
            code = f.read()
        
        try:
            ast.parse(code)
            print("[OK] 脚本语法正确")
        except SyntaxError as e:
            self.fail(f"脚本语法错误: {e}")


class TestErrorHandling(unittest.TestCase):
    """测试错误处理"""
    
    def setUp(self):
        """测试前准备"""
        self.config_path = "llm_reward_agent/config/llm_config.yaml"
        if not os.path.exists(self.config_path):
            self.skipTest(f"配置文件不存在: {self.config_path}")
    
    def test_fallback_codes(self):
        """测试后备代码机制"""
        print("\n[测试] 测试后备代码机制...")
        
        agent = RewardDesignAgent(config_path=self.config_path)
        
        # 测试后备代码生成
        fallback_codes = agent._get_fallback_codes(generation=0)
        
        self.assertIsInstance(fallback_codes, list)
        self.assertGreater(len(fallback_codes), 0)
        
        # 检查代码语法
        for code in fallback_codes:
            self.assertTrue(agent._syntax_check(code))
        
        print(f"[OK] 后备代码生成成功（{len(fallback_codes)}个）")
    
    def test_syntax_check(self):
        """测试语法检查"""
        print("\n[测试] 测试语法检查...")
        
        agent = RewardDesignAgent(config_path=self.config_path)
        
        # 正确的代码
        valid_code = """
import numpy as np

def compute_reward(agent_name, observation, global_state, actions, world):
    return 0.0, {}
"""
        self.assertTrue(agent._syntax_check(valid_code))
        
        # 错误的代码
        invalid_code = "def foo(: return"
        self.assertFalse(agent._syntax_check(invalid_code))
        
        print("[OK] 语法检查功能正常")


class TestVisualization(unittest.TestCase):
    """测试可视化工具"""
    
    def test_plotter_init(self):
        """测试可视化器初始化"""
        print("\n[测试] 测试可视化器初始化...")
        
        plotter = EvolutionPlotter(archive_dir="test_logs/viz_test")
        
        self.assertIsNotNone(plotter)
        print("[OK] 可视化器初始化成功")
    
    def test_load_generation_data(self):
        """测试加载代次数据"""
        print("\n[测试] 测试加载代次数据...")
        
        # 检查是否有实际的进化数据
        archive_dir = "experiments/evolution_archive"
        
        if os.path.exists(archive_dir):
            plotter = EvolutionPlotter(archive_dir=archive_dir)
            
            data = plotter.load_generation_data(0)
            
            if data:
                self.assertIsInstance(data, dict)
                self.assertIn('generation', data)
                self.assertIn('best_fitness', data)
                print("[OK] 成功加载代次数据")
            else:
                print("[SKIP] 没有找到进化数据")
        else:
            print("[SKIP] 进化记录目录不存在")


class TestIntegration(unittest.TestCase):
    """集成测试"""
    
    def test_full_evolution_mock(self):
        """测试完整进化流程（模拟模式）"""
        # 这个测试会运行真实的Agent，但使用模拟训练
        # 默认跳过，避免调用LLM API
        self.skipTest("跳过完整进化测试（需要API密钥）")
        
        print("\n[测试] 测试完整进化流程（模拟模式）...")
        
        config_path = "llm_reward_agent/config/llm_config.yaml"
        if not os.path.exists(config_path):
            self.skipTest(f"配置文件不存在: {config_path}")
        
        # 检查API密钥
        if not os.getenv("OPENAI_API_KEY"):
            self.skipTest("未设置OPENAI_API_KEY")
        
        agent = RewardDesignAgent(config_path=config_path)
        agent.initialize(
            env_file_path="MADDPG/envs/simple_tag_env.py",
            task_description="测试任务"
        )
        
        # 运行2代进化（模拟模式）
        for gen in range(2):
            result = agent.step(generation=gen, use_real_training=False)
            
            self.assertIn('best_code', result)
            self.assertIn('best_fitness', result)
            self.assertIn('reflection', result)
        
        print("[OK] 完整进化流程测试通过")


def run_all_tests():
    """运行所有测试"""
    print("=" * 80)
    print("阶段四功能测试")
    print("=" * 80)
    
    # 创建测试套件
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # 添加测试类
    suite.addTests(loader.loadTestsFromTestCase(TestMainScript))
    suite.addTests(loader.loadTestsFromTestCase(TestErrorHandling))
    suite.addTests(loader.loadTestsFromTestCase(TestVisualization))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    
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
        print("\n[OK] 所有测试通过！")
    else:
        print("\n[FAIL] 部分测试失败，请检查输出")
    
    print("=" * 80)
    
    return result


def run_manual_test():
    """手动测试（可选）"""
    print("\n" + "=" * 80)
    print("手动测试：运行简化的进化流程")
    print("=" * 80)
    
    print("\n这将运行一个简化的进化流程（3代，模拟训练）")
    print("需要API密钥: OPENAI_API_KEY")
    
    choice = input("\n是否继续？(y/n): ").strip().lower()
    
    if choice != 'y':
        print("跳过手动测试")
        return
    
    # 检查API密钥
    if not os.getenv("OPENAI_API_KEY"):
        print("\n[错误] 未设置OPENAI_API_KEY环境变量")
        return
    
    print("\n开始运行...")
    
    try:
        # 导入主函数
        import run_evolution
        
        # 构造参数
        class Args:
            config = 'llm_reward_agent/config/llm_config.yaml'
            num_generations = 3
            use_real_training = False  # 使用模拟训练
            episode_num = 10
            max_workers = 2
            env_file = 'MADDPG/envs/simple_tag_env.py'
            task_description = '3个追捕者围捕1个目标'
            save_dir = 'test_logs/manual_evolution'
            copy_to_maddpg = False
        
        args = Args()
        
        # 运行
        run_evolution.main(args)
        
        print("\n[OK] 手动测试完成！")
        print(f"结果保存在: {args.save_dir}")
    
    except Exception as e:
        print(f"\n[错误] 手动测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # 运行自动测试
    result = run_all_tests()
    
    # 询问是否运行手动测试
    if result.wasSuccessful():
        run_manual_test()
    
    # 返回退出码
    sys.exit(0 if result.wasSuccessful() else 1)
