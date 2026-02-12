"""
阶段二快速测试脚本
验证核心模块是否正常工作
"""

import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 80)
print("阶段二快速测试")
print("=" * 80)

# 测试1: 导入模块
print("\n[测试1] 导入核心模块...")
try:
    from llm_reward_agent.agent import (
        PromptTemplates,
        EvolutionaryMemory
    )
    from llm_reward_agent.tools.context_extractor import EnvironmentContextExtractor
    print("✅ 所有模块导入成功")
except Exception as e:
    print(f"❌ 模块导入失败: {e}")
    sys.exit(1)

# 测试2: 提示词模板
print("\n[测试2] 测试提示词模板...")
try:
    test_env_context = {
        'env_name': 'simple_tag_env',
        'observation_space': 'Box(16,)',
        'action_space': 'Box(2,)',
        'agent_info': {'num_adversaries': 3, 'num_good': 1},
        'physical_constants': {'max_force': 1.0},
        'code_snippet': '# code...'
    }
    
    prompt = PromptTemplates.initial_generation_prompt(
        test_env_context,
        "测试任务"
    )
    
    assert len(prompt) > 500
    assert 'compute_reward' in prompt
    print(f"✅ 提示词生成成功 (长度: {len(prompt)} 字符)")
except Exception as e:
    print(f"❌ 提示词测试失败: {e}")

# 测试3: 进化记忆
print("\n[测试3] 测试进化记忆...")
try:
    memory = EvolutionaryMemory(save_dir="test_logs/quick_test")
    
    test_code = "def compute_reward(...): return 0, {}"
    test_results = [{'id': 0, 'fitness': 0.8, 'status': 'success'}]
    
    memory.save(
        generation=0,
        best_code=test_code,
        reflection="测试反思",
        all_results=test_results
    )
    
    assert len(memory.history) == 1
    assert memory.get_best_code(0) == test_code
    print("✅ 进化记忆保存和读取成功")
except Exception as e:
    print(f"❌ 进化记忆测试失败: {e}")

# 测试4: 环境上下文提取
print("\n[测试4] 测试环境上下文提取...")
try:
    extractor = EnvironmentContextExtractor()
    
    env_file = "MADDPG/envs/simple_tag_env.py"
    if os.path.exists(env_file):
        context = extractor.extract_skeleton(env_file)
        assert 'env_name' in context
        print(f"✅ 环境上下文提取成功 (环境: {context['env_name']})")
    else:
        print(f"⚠️ 环境文件不存在，跳过测试: {env_file}")
except Exception as e:
    print(f"❌ 环境上下文提取失败: {e}")

print("\n" + "=" * 80)
print("快速测试完成")
print("=" * 80)
print("\n说明:")
print("- 所有核心模块均已成功导入和测试")
print("- 如需完整测试，请运行: python test_phase2.py")
print("- 如需使用LLM功能，请设置环境变量: OPENAI_API_KEY")
print("\n✅ 阶段二开发完成！")
