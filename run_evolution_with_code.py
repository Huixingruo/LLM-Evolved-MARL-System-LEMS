"""
进化主流程脚本（带初始代码版本）
运行完整的奖励函数进化循环，第零代使用人工设计的代码

Author: LEMS Project
Date: 2026-02-26
Version: 1.1

运行示例:
    python run_evolution_with_code.py --num_generations 5 --episode_num 100
    python run_evolution_with_code.py --initial_code MADDPG/envs/reward_function.py
"""

import argparse
import os
import sys
import time
from datetime import datetime, timedelta

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from llm_reward_agent.agent import RewardDesignAgent


def print_banner():
    """打印欢迎横幅"""
    print("=" * 80)
    print("LEMS - LLM驱动的多智能体强化学习奖励函数进化系统")
    print("LLM-driven Evolution of Multi-Agent Reward System")
    print("（带初始代码版本 - 第零代使用人工设计代码）")
    print("=" * 80)
    print()


def print_generation_header(generation: int, total: int, is_initial: bool = False):
    """打印代次标题"""
    print("\n" + "=" * 80)
    if is_initial:
        print(f"第 0/{total} 代 - 使用人工设计代码")
        print(f"Generation 0 (Initial Code) of {total}")
    else:
        print(f"第 {generation + 1}/{total} 代进化")
        print(f"Generation {generation + 1} of {total}")
    print("=" * 80)


def print_generation_result(result: dict):
    """打印代次结果"""
    print("\n" + "-" * 80)
    print("本代最优结果:")
    print("-" * 80)
    print(f"  Fitness: {result['best_fitness']:.4f}")

    # 打印前200字符的反思
    reflection = result.get('reflection', '')
    if reflection:
        print(f"\n  反思摘要:")
        print(f"  {reflection[:200]}...")

    # 打印所有候选的fitness
    all_results = result.get('all_results', [])
    if all_results:
        print(f"\n  所有候选:")
        for r in all_results:
            status = r.get('status', 'unknown')
            fitness = r.get('fitness', 0)
            candidate_id = r.get('id', 'unknown')
            print(f"    {candidate_id}: {status:8s} Fitness={fitness:.4f}")

    print("-" * 80)


def save_generation_summary(generation: int, result: dict, save_dir: str):
    """保存代次摘要"""
    summary_file = os.path.join(save_dir, f"generation_{generation:03d}_summary.txt")

    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write(f"Generation {generation} Summary\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Fitness: {result['best_fitness']:.4f}\n\n")

        f.write("Best Code:\n")
        f.write("-" * 80 + "\n")
        f.write(result['best_code'])
        f.write("\n" + "-" * 80 + "\n\n")

        f.write("Reflection:\n")
        f.write("-" * 80 + "\n")
        f.write(result.get('reflection', ''))
        f.write("\n" + "-" * 80 + "\n")

    print(f"  [保存] 摘要已保存到: {summary_file}")


def print_final_summary(agent: RewardDesignAgent, total_time: float):
    """打印最终摘要"""
    print("\n" + "=" * 80)
    print("进化完成！Evolution Complete!")
    print("=" * 80)

    # 获取历史最优
    best_ever = agent.memory.get_best_ever()

    print(f"\n[最优结果]")
    print(f"  出现在第 {best_ever['generation']} 代")
    print(f"  Fitness: {best_ever['best_fitness']:.4f}")

    # 获取fitness历史
    fitness_history = agent.memory.get_fitness_history()
    print(f"\n[进化历史]")
    for i, fitness in enumerate(fitness_history):
        marker = " <-- 最优" if i == best_ever['generation'] else ""
        print(f"  第{i}代: {fitness:.4f}{marker}")

    # 打印耗时
    duration = str(timedelta(seconds=int(total_time)))
    print(f"\n[总耗时]")
    print(f"  {duration} ({total_time:.1f}秒)")

    print("\n" + "=" * 80)


def save_final_code(best_code: str, save_path: str):
    """保存最终代码"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(best_code)

    print(f"  [保存] 最优奖励函数已保存到: {save_path}")


def run_initial_generation(agent: RewardDesignAgent, initial_code: str,
                          generation: int, use_real_training: bool) -> dict:
    """
    运行第零代（使用预定义的初始代码）

    Args:
        agent: RewardDesignAgent实例
        initial_code: 初始代码字符串
        generation: 代数
        use_real_training: 是否使用真实训练

    Returns:
        dict: 训练结果
    """
    print(f"\n{'=' * 80}")
    print("🚀 第零代: 使用人工设计代码进行训练")
    print(f"{'=' * 80}")

    # 将初始代码复制4份作为4个候选
    num_candidates = 4
    codes = [initial_code] * num_candidates
    print(f"  [配置] 第零代生成 {num_candidates} 个候选（均使用人工设计代码）")

    # 调用仿真工具训练
    if use_real_training:
        print(f"\n{'=' * 80}")
        print("🚀 开始并行训练（真实训练）")
        print(f"{'=' * 80}")

        # 使用真实的并行训练
        from llm_reward_agent.tools.simulation_tool import SimulationTool

        # 从配置读取训练参数
        training_config = agent.config.get('training', {})

        simulator = SimulationTool(
            base_dir=agent.config['logging']['save_dir'],
            max_workers=training_config.get('parallel_workers', 4),
            timeout=training_config.get('timeout', 10800),
            episode_num=training_config.get('episode_num', 100),
            use_gpu=training_config.get('use_gpu', True)
        )

        results = simulator.run_parallel(codes, generation)
    else:
        print(f"\n{'=' * 80}")
        print("🚀 开始并行训练（模拟模式）")
        print(f"{'=' * 80}")

        # 使用模拟数据（用于快速测试）
        results = agent._simulate_training(codes)

    # 分析结果
    best_code, reflection = agent.analyze_results(results)

    # 计算fitness
    valid_results = [r for r in results if r.get('status') == 'success']
    if valid_results:
        best_fitness = max(r.get('fitness', 0) for r in valid_results)
    else:
        best_fitness = 0.0

    # 更新记忆
    agent.memory.save(
        generation=generation,
        best_code=best_code,
        reflection=reflection,
        all_results=results
    )

    return {
        'generation': generation,
        'best_code': best_code,
        'best_fitness': best_fitness,
        'reflection': reflection,
        'all_results': results
    }


def load_initial_code(file_path: str) -> str:
    """
    加载初始代码文件

    Args:
        file_path: 代码文件路径

    Returns:
        str: 代码内容
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        code = f.read()

    # 检查代码是否包含必要的函数
    if 'def compute_reward' not in code:
        raise ValueError(f"初始代码文件不包含 compute_reward 函数: {file_path}")

    print(f"  [加载] 已加载初始代码: {file_path}")
    return code


def main(args):
    """主函数"""
    # 打印横幅
    print_banner()

    # 打印配置信息
    print("[配置信息]")
    print(f"  配置文件: {args.config}")
    print(f"  初始代码: {args.initial_code}")
    print(f"  进化代数: {args.num_generations}")
    print(f"  训练模式: {'真实训练' if args.use_real_training else '模拟训练'}")
    if args.use_real_training:
        print(f"  训练回合数: {args.episode_num}")
        print(f"  并行数: {args.max_workers}")
    print()

    # 检查配置文件
    if not os.path.exists(args.config):
        print(f"[错误] 配置文件不存在: {args.config}")
        sys.exit(1)

    # 检查环境文件
    if not os.path.exists(args.env_file):
        print(f"[错误] 环境文件不存在: {args.env_file}")
        sys.exit(1)

    # 检查初始代码文件
    if not os.path.exists(args.initial_code):
        print(f"[错误] 初始代码文件不存在: {args.initial_code}")
        sys.exit(1)

    # 创建保存目录
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)
    print(f"[保存目录] {os.path.abspath(save_dir)}")
    print()

    # 加载初始代码
    print("[步骤0/3] 加载初始代码...")
    try:
        initial_code = load_initial_code(args.initial_code)
        print("[OK] 初始代码加载成功")
    except Exception as e:
        print(f"[错误] 初始代码加载失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    print()

    # 1. 初始化Agent
    print("[步骤1/3] 初始化智能体...")
    try:
        agent = RewardDesignAgent(config_path=args.config)

        # 如果使用真实训练，更新配置
        if args.use_real_training:
            agent.config['training']['episode_num'] = args.episode_num
            agent.config['training']['parallel_workers'] = args.max_workers

        agent.initialize(
            env_file_path=args.env_file,
            task_description=args.task_description
        )
        print("[OK] 智能体初始化成功")
    except Exception as e:
        print(f"[错误] 智能体初始化失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # 2. 进化循环
    print(f"\n[步骤2/3] 开始进化循环 ({args.num_generations} 代)...")

    start_time = time.time()

    # 第零代：使用人工设计的代码
    print_generation_header(0, args.num_generations, is_initial=True)

    try:
        result = run_initial_generation(
            agent=agent,
            initial_code=initial_code,
            generation=0,
            use_real_training=args.use_real_training
        )

        # 打印结果
        print_generation_result(result)

        # 保存摘要
        save_generation_summary(0, result, save_dir)

    except KeyboardInterrupt:
        print("\n\n[中断] 用户中断进化循环")
        return

    except Exception as e:
        print(f"\n[错误] 第零代训练失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # 后续代：使用LLM进化
    for generation in range(1, args.num_generations):
        try:
            # 打印代次标题
            print_generation_header(generation, args.num_generations)

            # 执行一代进化
            result = agent.step(
                generation=generation,
                use_real_training=args.use_real_training
            )

            # 打印结果
            print_generation_result(result)

            # 保存摘要
            save_generation_summary(generation, result, save_dir)

            # 保存进化曲线图（如果有matplotlib）
            if generation > 0:
                try:
                    agent.memory.plot_evolution_curve(
                        save_path=os.path.join(save_dir, "evolution_curve.png")
                    )
                except:
                    pass  # 忽略绘图错误

        except KeyboardInterrupt:
            print("\n\n[中断] 用户中断进化循环")
            break

        except Exception as e:
            print(f"\n[错误] 第{generation}代进化失败: {e}")
            import traceback
            traceback.print_exc()

            # 询问是否继续
            if generation < args.num_generations - 1:
                choice = input("\n是否继续下一代？(y/n): ").strip().lower()
                if choice != 'y':
                    break

    total_time = time.time() - start_time

    # 3. 输出最终结果
    print("\n[步骤3/3] 导出最终结果...")

    try:
        # 打印摘要
        print_final_summary(agent, total_time)

        # 获取最优代码
        best_ever = agent.memory.get_best_ever()

        # 保存最优代码
        final_code_path = os.path.join(save_dir, "reward_function_best.py")
        save_final_code(best_ever['best_code'], final_code_path)

        # 可选：复制到MADDPG目录
        if args.copy_to_maddpg:
            maddpg_path = "MADDPG/envs/reward_function_evolved.py"
            save_final_code(best_ever['best_code'], maddpg_path)

        # 导出完整摘要
        summary_path = os.path.join(save_dir, "evolution_summary.txt")
        agent.memory.export_summary(filepath=summary_path)
        print(f"  [保存] 完整摘要已保存到: {summary_path}")

        print("\n[完成] 所有结果已保存！")

    except Exception as e:
        print(f"\n[错误] 导出结果失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="LEMS - 奖励函数进化系统（带初始代码版本）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用默认初始代码（人工设计的V3.1版本）
  python run_evolution_with_code.py --num_generations 3

  # 使用真实训练（5代，每代100回合）
  python run_evolution_with_code.py --num_generations 5 --episode_num 100

  # 指定自定义初始代码
  python run_evolution_with_code.py --initial_code my_reward.py --num_generations 5
        """
    )

    # 基础参数
    parser.add_argument(
        '--config',
        type=str,
        default='llm_reward_agent/config/llm_config.yaml',
        help='LLM配置文件路径'
    )

    parser.add_argument(
        '--initial_code',
        type=str,
        default='MADDPG/envs/reward_function.py',
        help='初始代码文件路径（人工设计的奖励函数）'
    )

    parser.add_argument(
        '--num_generations',
        type=int,
        default=3,
        help='进化代数（默认: 5）'
    )

    # 训练参数
    parser.add_argument(
        '--use_real_training',
        dest='use_real_training',
        action='store_true',

        help='使用真实训练（默认）'
    )

    parser.add_argument(
        '--no-real-training',
        dest='use_real_training',
        action='store_false',
        help='使用模拟训练（快速测试）'
    )

    parser.set_defaults(use_real_training=True)

    parser.add_argument(
        '--episode_num',
        type=int,
        default=3000,
        help='每个候选的训练回合数（默认: 100）'
    )

    parser.add_argument(
        '--max_workers',
        type=int,
        default=4,
        help='并行训练的最大进程数（默认: 4）'
    )

    # 环境参数
    parser.add_argument(
        '--env_file',
        type=str,
        default='MADDPG/envs/simple_tag_env.py',
        help='环境文件路径'
    )

    parser.add_argument(
        '--task_description',
        type=str,
        default="""任务：3个追捕智能体协同围捕1个逃逸目标。
要求：
1. 追捕者需要接近并包围目标
2. 追捕者之间避免碰撞
3. 形成均匀的包围圈
4. 在尽可能短的时间内完成围捕
""",
        help='任务描述'
    )

    # 保存参数
    parser.add_argument(
        '--save_dir',
        type=str,
        default='experiments/evolution_run',
        help='结果保存目录'
    )

    parser.add_argument(
        '--copy_to_maddpg',
        action='store_true',
        help='将最优代码复制到MADDPG目录'
    )

    args = parser.parse_args()

    # 运行主函数
    try:
        main(args)
    except KeyboardInterrupt:
        print("\n\n程序被用户中断")
        sys.exit(0)
    except Exception as e:
        print(f"\n程序异常退出: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
