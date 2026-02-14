"""
阶段二测试脚本
测试LLM Agent核心功能

测试内容：
1. LLM接口测试
2. 提示词模板测试
3. 进化记忆管理测试
4. 上下文提取测试
5. 主Agent类集成测试

Author: LEMS Project
Date: 2026-02-03
Version: 1.0
"""

import os
import sys
import unittest

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from llm_reward_agent.agent import (
    LLMInterface,
    PromptTemplates,
    EvolutionaryMemory,
    RewardDesignAgent
)
from llm_reward_agent.tools.context_extractor import EnvironmentContextExtractor


class TestLLMInterface(unittest.TestCase):
    """测试LLM接口"""
    
    def setUp(self):
        """测试前准备"""
        # 从环境变量获取API密钥
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            self.skipTest("未设置OPENAI_API_KEY环境变量，跳过LLM测试")
    
    def test_initialization(self):
        """测试LLM接口初始化"""
        print("\n[测试] LLM接口初始化...")
        
        llm = LLMInterface(
            provider="openai",
            model_name="generalv3.5",
            api_key=self.api_key
        )
        
        self.assertIsNotNone(llm.client)
        self.assertEqual(llm.model_name, "generalv3.5")
        print("✅ LLM接口初始化成功")
    
    def test_generate(self):
        """测试代码生成（可选，需要API调用）"""
        # 这个测试会消耗API配额，默认跳过
        # 如果需要测试，取消下面的skipTest
        self.skipTest("跳过实际API调用测试（避免消耗配额）")
        
        print("\n[测试] LLM代码生成...")
        
        llm = LLMInterface(
            provider="openai",
            model_name="generalv3.5",
            api_key=self.api_key
        )
        
        prompt = "请编写一个计算两数之和的Python函数。只输出代码。"
        results = llm.generate(prompt=prompt, n=1, temperature=0.7)
        
        self.assertEqual(len(results), 1)
        self.assertIn("def", results[0])
        print(f"✅ 生成结果: {results[0][:100]}...")


class TestPromptTemplates(unittest.TestCase):
    """测试提示词模板"""
    
    def setUp(self):
        """测试前准备"""
        self.env_context = {
            'env_name': 'simple_tag_env',
            'observation_space': 'Box(16,)',
            'action_space': 'Box(2,)',
            'agent_info': {
                'num_adversaries': 3,
                'num_good': 1
            },
            'physical_constants': {
                'max_force': 1.0,
                'capture_threshold': 0.5
            },
            'code_snippet': '# 环境代码...'
        }
        
        self.task_description = "3个追捕者围捕1个目标"
    
    def test_initial_generation_prompt(self):
        """测试初始生成提示词"""
        print("\n[测试] 初始生成提示词...")
        
        prompt = PromptTemplates.initial_generation_prompt(
            self.env_context,
            self.task_description
        )
        
        self.assertIsInstance(prompt, str)
        self.assertIn("compute_reward", prompt)
        self.assertIn("simple_tag_env", prompt)
        self.assertGreater(len(prompt), 1000)
        
        print(f"✅ 提示词长度: {len(prompt)} 字符")
    
    def test_evolution_prompt(self):
        """测试进化提示词"""
        print("\n[测试] 进化提示词...")
        
        parent_code = "def compute_reward(...): return 0, {}"
        reflection = "需要增加距离奖励"
        
        prompt = PromptTemplates.evolution_prompt(
            self.env_context,
            self.task_description,
            parent_code,
            reflection,
            n_candidates=4
        )
        
        self.assertIsInstance(prompt, str)
        self.assertIn("VARIANT", prompt)
        self.assertIn(parent_code, prompt)
        self.assertIn(reflection, prompt)
        
        print(f"✅ 提示词长度: {len(prompt)} 字符")
    
    def test_reflection_prompt(self):
        """测试反思提示词"""
        print("\n[测试] 反思提示词...")
        
        logs = "Candidate 0: 成功率 0.75"
        
        prompt = PromptTemplates.reflection_prompt(logs)
        
        self.assertIsInstance(prompt, str)
        self.assertIn("分析", prompt)
        self.assertIn(logs, prompt)
        
        print(f"✅ 提示词长度: {len(prompt)} 字符")


class TestEvolutionaryMemory(unittest.TestCase):
    """测试进化记忆管理"""
    
    def setUp(self):
        """测试前准备"""
        self.test_dir = "test_logs/memory_test"
        self.memory = EvolutionaryMemory(save_dir=self.test_dir)
    
    def tearDown(self):
        """测试后清理"""
        # 清理测试文件
        import shutil
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_save_and_load(self):
        """测试保存和加载"""
        print("\n[测试] 进化记忆保存和加载...")
        
        # 保存一代
        test_code = "def compute_reward(...): return 1.0, {}"
        test_reflection = "测试反思"
        test_results = [
            {'id': 0, 'fitness': 0.7, 'status': 'success'},
            {'id': 1, 'fitness': 0.8, 'status': 'success'}
        ]
        
        self.memory.save(
            generation=0,
            best_code=test_code,
            reflection=test_reflection,
            all_results=test_results
        )
        
        # 验证保存
        self.assertEqual(len(self.memory.history), 1)
        self.assertEqual(self.memory.get_best_code(0), test_code)
        self.assertEqual(self.memory.get_reflection(0), test_reflection)
        
        # 测试加载
        new_memory = EvolutionaryMemory(save_dir=self.test_dir)
        new_memory.load_from_disk()
        
        self.assertEqual(len(new_memory.history), 1)
        self.assertEqual(new_memory.get_best_code(0), test_code)
        
        print("✅ 保存和加载功能正常")
    
    def test_get_best_ever(self):
        """测试获取历史最优"""
        print("\n[测试] 获取历史最优...")
        
        # 保存3代，第1代最优
        for gen in range(3):
            fitness = 0.7 if gen != 1 else 0.9
            results = [{'id': 0, 'fitness': fitness, 'status': 'success'}]
            
            self.memory.save(
                generation=gen,
                best_code=f"# 第{gen}代",
                reflection=f"反思{gen}",
                all_results=results
            )
        
        best = self.memory.get_best_ever()
        self.assertEqual(best['generation'], 1)
        self.assertEqual(best['best_fitness'], 0.9)
        
        print(f"✅ 历史最优在第{best['generation']}代")
    
    def test_export_summary(self):
        """测试导出摘要"""
        print("\n[测试] 导出摘要...")
        
        # 保存一些数据
        for gen in range(2):
            results = [{'id': 0, 'fitness': 0.7 + gen * 0.1, 'status': 'success'}]
            self.memory.save(
                generation=gen,
                best_code=f"# 第{gen}代",
                reflection=f"反思{gen}",
                all_results=results
            )
        
        summary = self.memory.export_summary()
        
        self.assertIsInstance(summary, str)
        self.assertIn("进化过程摘要", summary)
        self.assertIn("第  0 代", summary)
        
        print(f"✅ 摘要长度: {len(summary)} 字符")


class TestEnvironmentContextExtractor(unittest.TestCase):
    """测试环境上下文提取器"""
    
    def test_extract_skeleton(self):
        """测试提取环境骨架"""
        print("\n[测试] 环境上下文提取...")
        
        env_file = "MADDPG/envs/simple_tag_env.py"
        
        if not os.path.exists(env_file):
            self.skipTest(f"环境文件不存在: {env_file}")
        
        extractor = EnvironmentContextExtractor()
        context = extractor.extract_skeleton(env_file)
        
        self.assertIsInstance(context, dict)
        self.assertIn('env_name', context)
        self.assertIn('observation_space', context)
        self.assertIn('action_space', context)
        
        print(f"✅ 环境名称: {context['env_name']}")
        print(f"✅ 观测空间: {context['observation_space']}")
    
    def test_format_for_llm(self):
        """测试格式化为LLM文本"""
        print("\n[测试] 格式化LLM文本...")
        
        extractor = EnvironmentContextExtractor()
        
        test_context = {
            'env_name': 'test_env',
            'observation_space': 'Box(10,)',
            'action_space': 'Box(2,)',
            'agent_info': {'num_adversaries': 3},
            'physical_constants': {'max_force': 1.0},
            'code_snippet': '# 测试代码'
        }
        
        formatted = extractor.format_for_llm(test_context)
        
        self.assertIsInstance(formatted, str)
        self.assertIn('test_env', formatted)
        self.assertIn('Box(10,)', formatted)
        
        print(f"✅ 格式化文本长度: {len(formatted)} 字符")


class TestRewardDesignAgent(unittest.TestCase):
    """测试主Agent类"""
    
    def setUp(self):
        """测试前准备"""
        # 检查配置文件
        self.config_path = "llm_reward_agent/config/llm_config.yaml"
        if not os.path.exists(self.config_path):
            self.skipTest(f"配置文件不存在: {self.config_path}")
        
        # 检查API密钥
        if not os.getenv("OPENAI_API_KEY"):
            self.skipTest("未设置OPENAI_API_KEY环境变量")
    
    def test_initialization(self):
        """测试Agent初始化"""
        print("\n[测试] Agent初始化...")
        
        agent = RewardDesignAgent(config_path=self.config_path)
        
        self.assertIsNotNone(agent.llm)
        self.assertIsNotNone(agent.memory)
        self.assertIsNotNone(agent.context_extractor)
        
        print("✅ Agent初始化成功")
    
    def test_initialize_environment(self):
        """测试环境初始化"""
        print("\n[测试] 环境初始化...")
        
        agent = RewardDesignAgent(config_path=self.config_path)
        
        env_file = "MADDPG/envs/simple_tag_env.py"
        if not os.path.exists(env_file):
            self.skipTest(f"环境文件不存在: {env_file}")
        
        task_desc = "测试任务：围捕"
        
        agent.initialize(env_file, task_desc)
        
        self.assertIsNotNone(agent.env_context)
        self.assertIsNotNone(agent.task_description)
        
        print(f"✅ 环境上下文已加载")
    
    def test_syntax_check(self):
        """测试语法检查"""
        print("\n[测试] 代码语法检查...")
        
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
        
        print("✅ 语法检查功能正常")


def run_all_tests():
    """运行所有测试"""
    print("=" * 80)
    print("阶段二功能测试")
    print("=" * 80)
    
    # 创建测试套件
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # 添加测试类
    suite.addTests(loader.loadTestsFromTestCase(TestPromptTemplates))
    suite.addTests(loader.loadTestsFromTestCase(TestEvolutionaryMemory))
    suite.addTests(loader.loadTestsFromTestCase(TestEnvironmentContextExtractor))
    
    # LLM和Agent测试需要API密钥，单独处理
    if os.getenv("OPENAI_API_KEY"):
        print("\n✅ 检测到API密钥，将运行LLM相关测试")
        suite.addTests(loader.loadTestsFromTestCase(TestLLMInterface))
        suite.addTests(loader.loadTestsFromTestCase(TestRewardDesignAgent))
    else:
        print("\n⚠️ 未检测到API密钥，跳过LLM相关测试")
    
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
        print("\n✅ 所有测试通过！")
    else:
        print("\n❌ 部分测试失败，请检查输出")
    
    print("=" * 80)
    
    return result


if __name__ == "__main__":
    # 运行测试
    result = run_all_tests()
    
    # 返回退出码
    sys.exit(0 if result.wasSuccessful() else 1)
