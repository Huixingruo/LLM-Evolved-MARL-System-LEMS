"""
LEMS系统演示脚本
展示完整的奖励函数进化流程

Author: LEMS Project
Date: 2026-02-03
Version: 1.0

使用方法:
    python demo_evolution.py
"""

import os
import sys
import time

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def print_section(title: str):
    """打印章节标题"""
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80 + "\n")


def demo_environment_setup():
    """演示：环境准备检查"""
    print_section("步骤1: 环境检查")
    
    # 检查Python版本
    print(f"[检查] Python版本: {sys.version}")
    
    # 检查必要模块
    required_modules = ['numpy', 'yaml', 'matplotlib']
    optional_modules = ['openai']
    
    print("\n[检查] 必要模块:")
    for module in required_modules:
        try:
            __import__(module)
            print(f"  [OK] {module}")
        except ImportError:
            print(f"  [FAIL] {module} - 请运行: pip install {module}")
    
    print("\n[检查] 可选模块（LLM功能需要）:")
    for module in optional_modules:
        try:
            __import__(module)
            print(f"  [OK] {module}")
        except ImportError:
            print(f"  [WARN] {module} - LLM功能将不可用")
    
    # 检查文件
    print("\n[检查] 核心文件:")
    core_files = [
        'MADDPG/envs/simple_tag_env.py',
        'MADDPG/envs/reward_function.py',
        'llm_reward_agent/config/llm_config.yaml',
        'run_evolution.py'
    ]
    
    for file in core_files:
        if os.path.exists(file):
            print(f"  [OK] {file}")
        else:
            print(f"  [FAIL] {file} - 文件缺失")
    
    print("\n[OK] 环境检查完成")


def demo_context_extraction():
    """演示：环境上下文提取"""
    print_section("步骤2: 环境上下文提取")
    
    try:
        from llm_reward_agent.tools import EnvironmentContextExtractor
        
        extractor = EnvironmentContextExtractor()
        
        env_file = "MADDPG/envs/simple_tag_env.py"
        if not os.path.exists(env_file):
            print(f"[SKIP] 环境文件不存在: {env_file}")
            return
        
        print(f"正在提取环境上下文: {env_file}")
        context = extractor.extract_skeleton(env_file)
        
        print(f"\n[提取结果]")
        print(f"  环境名称: {context.get('env_name', '未知')}")
        print(f"  观测空间: {context.get('observation_space', '未知')}")
        print(f"  动作空间: {context.get('action_space', '未知')}")
        print(f"  智能体数量: {context.get('agent_info', {})}")
        print(f"  物理常量: {list(context.get('physical_constants', {}).keys())}")
        
        # 估算Token
        formatted = extractor.format_for_llm(context)
        tokens = extractor.estimate_token_count(formatted)
        print(f"  Token估算: ~{tokens}")
        
        print("\n[OK] 上下文提取成功")
    
    except Exception as e:
        print(f"[FAIL] 上下文提取失败: {e}")


def demo_prompt_generation():
    """演示：提示词生成"""
    print_section("步骤3: 提示词生成")
    
    try:
        from llm_reward_agent.agent import PromptTemplates
        
        # 构造测试上下文
        test_context = {
            'env_name': 'simple_tag_env',
            'observation_space': 'Box(16,)',
            'action_space': 'Box(2,)',
            'agent_info': {'num_adversaries': 3, 'num_good': 1},
            'physical_constants': {'max_force': 1.0, 'capture_threshold': 0.5},
            'code_snippet': '# 环境代码片段...'
        }
        
        task_desc = "3个追捕者围捕1个目标"
        
        # 生成初始提示词
        print("[生成] 初始提示词（第0代）...")
        prompt = PromptTemplates.initial_generation_prompt(test_context, task_desc)
        
        print(f"  提示词长度: {len(prompt)} 字符")
        print(f"  前300字符:\n{prompt[:300]}...")
        
        # 生成进化提示词
        print("\n[生成] 进化提示词（后续代）...")
        parent_code = "def compute_reward(...): return 0, {}"
        reflection = "需要增加距离奖励..."
        
        evo_prompt = PromptTemplates.evolution_prompt(
            test_context, task_desc, parent_code, reflection, n_candidates=4
        )
        
        print(f"  提示词长度: {len(evo_prompt)} 字符")
        
        print("\n[OK] 提示词生成成功")
    
    except Exception as e:
        print(f"[FAIL] 提示词生成失败: {e}")


def demo_sandbox_creation():
    """演示：沙盒创建"""
    print_section("步骤4: 沙盒创建")
    
    try:
        from llm_reward_agent.tools import SandboxManager
        
        manager = SandboxManager(base_dir="demo_sandbox")
        
        # 创建测试代码
        test_code = """
import numpy as np

def compute_reward(agent_name, observation, global_state, actions, world):
    \"\"\"演示奖励函数\"\"\"
    components = {}
    
    if global_state.get('is_adversary', False):
        # 简单的距离奖励
        agent_idx = int(agent_name.split('_')[1])
        agent_pos = global_state['agent_positions'][agent_idx]
        prey_pos = global_state['prey_position']
        dist = np.linalg.norm(agent_pos - prey_pos)
        components['distance_reward'] = -0.1 * dist
    else:
        components['escape_reward'] = 0.1
    
    components['boundary_penalty'] = 0.0
    total_reward = sum(components.values())
    return total_reward, components
"""
        
        print("[创建] 2个演示沙盒...")
        sandboxes = manager.create_sandboxes(generation=0, codes=[test_code, test_code])
        
        print(f"\n[结果]")
        for i, sandbox in enumerate(sandboxes):
            info = manager.get_sandbox_info(sandbox)
            print(f"  沙盒{i}: {info['size_mb']:.2f}MB, {info['files_count']}个文件")
        
        # 清理
        print("\n[清理] 删除演示沙盒...")
        manager.cleanup_all()
        
        print("\n[OK] 沙盒创建演示完成")
    
    except Exception as e:
        print(f"[FAIL] 沙盒创建失败: {e}")


def demo_memory_management():
    """演示：进化记忆管理"""
    print_section("步骤5: 进化记忆管理")
    
    try:
        from llm_reward_agent.agent import EvolutionaryMemory
        
        # 创建记忆实例
        memory = EvolutionaryMemory(save_dir="demo_memory")
        
        # 模拟保存3代
        print("[模拟] 保存3代进化记录...")
        for gen in range(3):
            test_code = f"# 第{gen}代代码"
            test_reflection = f"第{gen}代反思：需要改进..."
            test_results = [
                {'id': i, 'fitness': 0.7 + gen * 0.05 + i * 0.01, 'status': 'success'}
                for i in range(4)
            ]
            
            memory.save(
                generation=gen,
                best_code=test_code,
                reflection=test_reflection,
                all_results=test_results
            )
        
        # 获取历史最优
        best_ever = memory.get_best_ever()
        print(f"\n[结果]")
        print(f"  历史最优: 第{best_ever['generation']}代")
        print(f"  Fitness: {best_ever['best_fitness']:.4f}")
        
        # 获取fitness历史
        fitness_history = memory.get_fitness_history()
        print(f"  进化曲线: {[f'{f:.4f}' for f in fitness_history]}")
        
        # 清理
        import shutil
        if os.path.exists("demo_memory"):
            shutil.rmtree("demo_memory")
        
        print("\n[OK] 记忆管理演示完成")
    
    except Exception as e:
        print(f"[FAIL] 记忆管理失败: {e}")


def demo_full_workflow():
    """演示：完整工作流（不含LLM调用）"""
    print_section("步骤6: 完整工作流演示")
    
    print("[演示说明]")
    print("这将展示完整的进化流程（使用模拟数据，不调用LLM API）")
    print("真实使用请运行: python run_evolution.py")
    
    try:
        from llm_reward_agent.agent import RewardDesignAgent
        
        config_path = "llm_reward_agent/config/llm_config.yaml"
        
        if not os.path.exists(config_path):
            print(f"[SKIP] 配置文件不存在: {config_path}")
            return
        
        print("\n[初始化] 创建Agent...")
        # 注意：这里会尝试初始化LLM，如果没有API密钥会失败
        # 但这只是演示，不影响理解流程
        
        print("""
完整流程如下:

1. 初始化Agent
   agent = RewardDesignAgent(config_path)
   agent.initialize(env_file, task_description)

2. 运行进化循环
   for generation in range(N):
       result = agent.step(generation, use_real_training=True)
       # result包含: best_code, best_fitness, reflection

3. 导出结果
   best_ever = agent.memory.get_best_ever()
   save_to_file(best_ever['best_code'])

4. 生成可视化
   plotter.generate_all_plots()
""")
        
        print("[OK] 工作流演示完成")
    
    except Exception as e:
        print(f"[INFO] Agent初始化需要API密钥")
        print(f"       完整演示请参考: run_evolution.py")


def main():
    """主函数"""
    print("=" * 80)
    print(" LEMS系统功能演示")
    print(" LLM-driven Evolution of Multi-Agent Reward System")
    print("=" * 80)
    print("\n这个演示将展示LEMS系统的各个核心功能")
    print("运行时间约1分钟，不会调用LLM API或执行训练")
    
    input("\n按Enter键开始演示...")
    
    # 运行各个演示
    demo_environment_setup()
    time.sleep(1)
    
    demo_context_extraction()
    time.sleep(1)
    
    demo_prompt_generation()
    time.sleep(1)
    
    demo_sandbox_creation()
    time.sleep(1)
    
    demo_memory_management()
    time.sleep(1)
    
    demo_full_workflow()
    
    # 总结
    print_section("演示完成")
    
    print("LEMS系统的核心功能已全部展示！\n")
    
    print("下一步操作：\n")
    print("1. 设置API密钥:")
    print("   $env:OPENAI_API_KEY=\"your_key\"  # Windows")
    print("   export OPENAI_API_KEY=\"your_key\"  # Linux/Mac\n")
    
    print("2. 运行快速测试（模拟训练，5分钟）:")
    print("   python run_evolution.py --num_generations 3 --no-real-training\n")
    
    print("3. 运行真实训练（30分钟）:")
    print("   python run_evolution.py --num_generations 5 --episode_num 100\n")
    
    print("4. 生成可视化图表:")
    print("   python visualization/evolution_plot.py\n")
    
    print("5. 查看文档:")
    print("   - PHASE4_QUICK_START.md - 快速开始")
    print("   - PHASE4_DOCUMENTATION.md - 完整文档")
    print("   - IMPLEMENTATION_PLAN.md - 总体计划\n")
    
    print("=" * 80)
    print("感谢使用LEMS系统！")
    print("=" * 80)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n演示被用户中断")
    except Exception as e:
        print(f"\n演示过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
