"""
进化主流程脚本
运行完整的奖励函数进化循环

支持从人工设计的初始代码开始进化（第零代），
也支持从零开始（Zero-Shot）。

Author: LEMS Project
Date: 2026-02-03
Version: 2.0 (合并初始代码版本)

运行示例:
    # 从零开始（Zero-Shot），3代，模拟训练
    python run_evolution.py --num_generations 3 --no-real-training

    # 从人工设计的初始代码开始，5代，真实训练
    python run_evolution.py --initial_code MADDPG/envs/reward_function.py --num_generations 5 --episode_num 100
"""

import argparse
import os
import sys
import time
from datetime import datetime, timedelta

# 强制允许OpenMP库重复加载，解决Windows下OMP: Error #15
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# ============================================================
# 【关键修复】在导入任何可能使用matplotlib的模块之前，
# 必须设置matplotlib的后端为非交互式后端(Agg)，避免tkinter错误
# 错误: RuntimeError: main thread is not in main loop
# 原因: Windows下matplotlib可能默认使用tkinter后端，
#       而tkinter要求在主线程中运行
# ============================================================
os.environ['MPLBACKEND'] = 'Agg'
os.environ['TK_SILENCE_IGNORE'] = '1'  # 抑制tkinter警告

# 立即设置matplotlib后端（在导入matplotlib相关模块之前）
import matplotlib
matplotlib.use('Agg', force=True)
# ============================================================

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from llm_reward_agent.agent import RewardDesignAgent


def print_banner():
    """打印欢迎横幅"""
    print("=" * 80)
    print("LEMS - LLM驱动的多智能体强化学习奖励函数进化系统")
    print("LLM-driven Evolution of Multi-Agent Reward System")
    print("=" * 80)
    print()


def print_generation_header(generation: int, total: int, is_initial: bool = False):
    """打印代次标题"""
    print("\n" + "=" * 80)
    if is_initial:
        print(f"第 {generation}/{total} 代 - 使用 {'人工设计代码' if generation == 0 else '进化策略'}")
        print(f"Generation {generation} of {total}")
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

    reflection = result.get('reflection', '')
    if reflection:
        print(f"\n  反思摘要:")
        print(f"  {reflection[:200]}...")

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


def print_final_summary(agent, total_time: float):
    """打印最终摘要"""
    print("\n" + "=" * 80)
    print("进化完成！Evolution Complete!")
    print("=" * 80)

    # 物理拦截空记忆
    if not agent.memory.history:
        print("\n⚠️ 警告：进化记忆为空，系统在生成任何有效数据前已中断，无数据可汇总。")
        print("\n" + "=" * 80)
        return

    best_ever = agent.memory.get_best_ever()

    print(f"\n[最优结果]")
    print(f"  出现在第 {best_ever['generation']} 代")
    print(f"  Fitness: {best_ever['best_fitness']:.4f}")

    fitness_history = agent.memory.get_fitness_history()
    print(f"\n[进化历史]")
    for i, fitness in enumerate(fitness_history):
        marker = " <-- 最优" if i == best_ever['generation'] else ""
        print(f"  第{i}代: {fitness:.4f}{marker}")

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


def load_initial_code(file_path: str) -> str:
    """
    加载初始代码文件。

    Args:
        file_path: 代码文件路径

    Returns:
        str: 代码内容
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        code = f.read()

    if 'def compute_reward' not in code:
        raise ValueError(f"初始代码文件不包含 compute_reward 函数: {file_path}")

    print(f"  [加载] 已加载初始代码: {file_path}")
    return code


def run_initial_generation(agent: RewardDesignAgent, initial_code: str,
                          generation: int, use_real_training: bool) -> dict:
    """
    运行第零代（使用预定义的初始代码）。
    将初始代码复制为4个候选进行训练。

    Args:
        agent: RewardDesignAgent实例
        initial_code: 初始代码字符串
        generation: 代数（通常为0）
        use_real_training: 是否使用真实训练

    Returns:
        dict: 训练结果
    """
    print(f"\n{'=' * 80}")
    print("🚀 第零代: 使用人工设计代码进行训练")
    print(f"{'=' * 80}")

    num_candidates = 4
    codes = [initial_code] * num_candidates
    print(f"  [配置] 第零代生成 {num_candidates} 个候选（均使用人工设计代码）")

    if use_real_training:
        print(f"\n{'=' * 80}")
        print("🚀 开始并行训练（真实训练）")
        print(f"{'=' * 80}")

        from llm_reward_agent.tools.simulation_tool import SimulationTool

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

        results = agent._simulate_training(codes)

    # 【重点修改】：接收三个返回值
    best_code, reflection, best_fitness = agent.analyze_results(results)

    # 【重点修改】：将选定的 fitness 显式传递给 memory
    agent.memory.save(
        generation=generation,
        best_code=best_code,
        reflection=reflection,
        all_results=results,
        selected_fitness=best_fitness
    )

    return {
        'generation': generation,
        'best_code': best_code,
        'best_fitness': best_fitness,
        'reflection': reflection,
        'all_results': results
    }


def main(args):
    """主函数"""
    print_banner()

    print("[配置信息]")
    print(f"  配置文件: {args.config}")
    print(f"  进化代数: {args.num_generations}")
    print(f"  训练模式: {'真实训练' if args.use_real_training else '模拟训练'}")
    if args.use_real_training:
        print(f"  训练回合数: {args.episode_num}")
        print(f"  并行数: {args.max_workers}")
    print()

    if not os.path.exists(args.config):
        print(f"[错误] 配置文件不存在: {args.config}")
        sys.exit(1)

    if not os.path.exists(args.env_file):
        print(f"[错误] 环境文件不存在: {args.env_file}")
        sys.exit(1)

    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)
    print(f"[保存目录] {os.path.abspath(save_dir)}")
    print()

    # 1. 初始化Agent
    print("[步骤1/3] 初始化智能体...")
    try:
        agent = RewardDesignAgent(config_path=args.config)

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

    # ============================================================
    # 【新增】恢复训练逻辑：加载已有的进化历史
    # ============================================================
    resume_from_generation = None
    effective_archive_dir = os.path.join(
        agent.config['logging']['save_dir'],
        'evolution_archive'
    )  # 默认的保存目录

    if args.resume:
        print(f"\n[恢复模式] 尝试加载进化历史...")
        archive_dir = args.archive_dir

        if not os.path.exists(archive_dir):
            print(f"[错误] 进化记录目录不存在: {archive_dir}")
            print(f"   请确保目录存在，或不使用 --resume 参数从头开始")
            sys.exit(1)

        # 检查是否有历史记录
        metadata_path = os.path.join(archive_dir, "metadata.json")
        if not os.path.exists(metadata_path):
            print(f"[错误] 进化记录目录中没有 metadata.json 文件")
            print(f"   请确保 {archive_dir} 是有效的进化记录目录")
            sys.exit(1)

        # 加载历史记录
        try:
            # 临时修改 agent.memory 的 save_dir 以便加载
            original_save_dir = agent.memory.save_dir
            agent.memory.save_dir = archive_dir
            agent.memory.load_from_disk()
            agent.memory.save_dir = original_save_dir

            existing_generations = len(agent.memory.history)
            if existing_generations == 0:
                print(f"[警告] 进化记录目录为空，将从头开始训练")
                resume_from_generation = None
                # 即使目录为空，也使用用户指定的目录保存新记录
                effective_archive_dir = archive_dir
                agent.memory.save_dir = archive_dir
            else:
                last_generation = existing_generations - 1
                best_ever = agent.memory.get_best_ever()
                print(f"✅ 成功加载 {existing_generations} 代进化历史")
                print(f"   最后一代: 第 {last_generation} 代")
                print(f"   历史最优: 第 {best_ever['generation']} 代, Fitness={best_ever['best_fitness']:.4f}")

                # 确定起始代数（从最后一代的下一代开始）
                resume_from_generation = existing_generations
                print(f"   计划继续训练: 第 {resume_from_generation} 代 -> 第 {args.num_generations - 1} 代")

                # 【关键修复】继续保存到用户指定的目录
                effective_archive_dir = archive_dir
                agent.memory.save_dir = archive_dir
                print(f"   新记录将保存到: {archive_dir}")
        except Exception as e:
            print(f"[错误] 加载进化历史失败: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

        print()
    # ============================================================

    # 2. 进化循环
    print(f"\n[步骤2/3] 开始进化循环 ({args.num_generations} 代)...")

    start_time = time.time()

    # ============================================================
    # 【修改】整合恢复训练的起始代数逻辑
    # ============================================================
    if args.resume and resume_from_generation is not None:
        # 恢复模式：已经有历史，从指定代数继续
        if resume_from_generation >= args.num_generations:
            print(f"\n[错误] 恢复训练的起始代数({resume_from_generation}) >= 目标代数({args.num_generations})")
            print(f"   请增大 --num_generations 参数以继续训练")
            sys.exit(1)

        print(f"\n[恢复模式] 从第 {resume_from_generation} 代继续进化...")
        first_generation = resume_from_generation

        # 显示当前进化历史摘要
        fitness_history = agent.memory.get_fitness_history()
        print(f"\n[已完成的进化历史]")
        for i, fitness in enumerate(fitness_history):
            marker = " ⭐" if i == agent.memory.metadata.get('best_generation', -1) else ""
            print(f"  第{i}代: {fitness:.4f}{marker}")
        print()

    else:
        # 正常模式：从头开始训练
        # 是否使用初始代码
        has_initial_code = args.initial_code and os.path.exists(args.initial_code)

        if has_initial_code:
            # 第零代：使用人工设计的代码
            initial_code = load_initial_code(args.initial_code)

            print_generation_header(0, args.num_generations, is_initial=True)

            try:
                result = run_initial_generation(
                    agent=agent,
                    initial_code=initial_code,
                    generation=0,
                    use_real_training=args.use_real_training
                )

                print_generation_result(result)
                save_generation_summary(0, result, save_dir)

            except KeyboardInterrupt:
                print("\n\n[中断] 用户中断进化循环")
                return

            except Exception as e:
                print(f"\n[错误] 第零代训练失败: {e}")
                import traceback
                traceback.print_exc()
                sys.exit(1)

            first_generation = 1
        else:
            first_generation = 0
    # ============================================================

    # 后续代：使用LLM进化
    for generation in range(first_generation, args.num_generations):
        try:
            print_generation_header(generation, args.num_generations)

            result = agent.step(
                generation=generation,
                use_real_training=args.use_real_training
            )

            print_generation_result(result)
            save_generation_summary(generation, result, save_dir)

            if generation > 0:
                try:
                    agent.memory.plot_evolution_curve(
                        save_path=os.path.join(save_dir, "evolution_curve.png")
                    )
                except Exception:
                    pass

        except KeyboardInterrupt:
            print("\n\n[中断] 用户中断进化循环")
            break

        except Exception as e:
            print(f"\n[错误] 第{generation}代进化失败: {e}")
            import traceback
            traceback.print_exc()

            if generation < args.num_generations - 1:
                choice = input("\n是否继续下一代？(y/n): ").strip().lower()
                if choice != 'y':
                    break

    total_time = time.time() - start_time

    # 3. 输出最终结果
    print("\n[步骤3/3] 导出最终结果...")

    try:
        print_final_summary(agent, total_time)

        # 如果连history都没有，直接跳过保存文件的逻辑
        if agent.memory.history:
            best_ever = agent.memory.get_best_ever()

            final_code_path = os.path.join(save_dir, "reward_function_best.py")
            save_final_code(best_ever['best_code'], final_code_path)

            if args.copy_to_maddpg:
                maddpg_path = "MADDPG/envs/reward_function_evolved.py"
                save_final_code(best_ever['best_code'], maddpg_path)

            summary_path = os.path.join(save_dir, "evolution_summary.txt")
            agent.memory.export_summary(filepath=summary_path)
            print(f"  [保存] 完整摘要已保存到: {summary_path}")

            print("\n[完成] 所有结果已保存！")
        else:
            print("\n[中止] 因无有效数据，已跳过文件导出。")

    except Exception as e:
        print(f"\n[错误] 导出结果失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="LEMS - 奖励函数进化系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 从零开始（Zero-Shot），3代，模拟训练（快速测试）
  python run_evolution.py --num_generations 3 --no-real-training

  # 从人工设计的初始代码开始，5代，真实训练
  python run_evolution.py --initial_code MADDPG/envs/reward_function.py --num_generations 5

  # 完整进化（10代，每代200回合）
  python run_evolution.py --initial_code MADDPG/envs/reward_function.py --num_generations 10 --episode_num 200 --max_workers 4

  # 从第2代继续训练到第5代
  python run_evolution.py --resume --archive_dir experiments/evolution_archive --num_generations 5

  # 查看已完成的进化历史
  python run_evolution.py --resume --archive_dir experiments/evolution_archive --num_generations 3
        """
    )

    parser.add_argument(
        '--config',
        type=str,
        default='llm_reward_agent/config/llm_config.yaml',
        help='LLM配置文件路径'
    )

    parser.add_argument(
        '--initial_code',
        type=str,
        default=None,
        help='初始代码文件路径（可选，不指定则从零开始）'
    )

    parser.add_argument(
        '--num_generations',
        type=int,
        default=5,
        help='进化代数（默认: 3）'
    )

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
        help='每个候选的训练回合数（默认: 3000）'
    )

    parser.add_argument(
        '--max_workers',
        type=int,
        default=4,
        help='并行训练的最大进程数（默认: 4）'
    )

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

    parser.add_argument(
        '--resume',
        action='store_true',
        help='从已有进化记录继续训练（需要配合 --archive_dir 使用）'
    )

    parser.add_argument(
        '--archive_dir',
        type=str,
        default='experiments/evolution_archive',
        help='进化记录目录（用于恢复训练或加载历史记录）'
    )

    args = parser.parse_args()

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
