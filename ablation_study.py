"""
LEMS 消融实验脚本
分析各核心模块的贡献：AIR、DREAM、多维适应度评估、收敛过滤、降级选拔

Author: LEMS Project
Date: 2026-05-10
Version: 1.0

运行示例:
    # 运行完整消融实验
    python ablation_study.py --num_generations 5 --episode_num 100

    # 只运行特定变体
    python ablation_study.py --variants full no_air no_dream --num_generations 3

    # 使用模拟训练快速测试
    python ablation_study.py --num_generations 3 --no-real-training
"""

import argparse
import os
import sys
import json
import time
import copy
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from pathlib import Path

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from llm_reward_agent.agent import RewardDesignAgent
from llm_reward_agent.agent.reward_design_agent import RewardDesignAgent
from llm_reward_agent.tools.simulation_tool import SimulationTool


class AblationVariant:
    """消融实验变体基类"""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.results = []

    def modify_config(self, config: dict) -> dict:
        """修改配置以禁用特定模块"""
        return config

    def modify_agent_behavior(self, agent: RewardDesignAgent):
        """修改agent行为以禁用特定模块"""
        pass

    def get_generation_result(self, agent: RewardDesignAgent, generation: int,
                              use_real_training: bool) -> dict:
        """获取一代进化结果，可覆盖默认行为"""
        return agent.step(generation=generation, use_real_training=use_real_training)


class FullSystem(AblationVariant):
    """完整LEMS系统（基线）"""

    def __init__(self):
        super().__init__("full", "完整LEMS系统（AIR + DREAM + 多维适应度 + 收敛过滤 + 降级选拔）")


class NoAIR(AblationVariant):
    """禁用AIR模块：不使用CoT两阶段管线，直接生成"""

    def __init__(self):
        super().__init__("no_air", "禁用AIR模块（无CoT两阶段分析）")

    def modify_agent_behavior(self, agent: RewardDesignAgent):
        """跳过阶段一的CoT分析"""
        # 标记跳过CoT分析
        agent._skip_cot_analysis = True

        # 覆盖step方法，跳过generation==0时的CoT分析
        original_step = agent.step

        def modified_step(generation: int, use_real_training: bool = True) -> dict:
            if generation == 0:
                # 直接生成，不进行CoT分析
                agent.cot_analysis_result = None
                # 使用标准prompt而非CoT prompt
                prompt = agent.prompt_builder.initial_generation_prompt(
                    agent.task_description,
                    agent.env_context
                )
                raw_outputs = agent.llm.generate(
                    prompt=prompt,
                    n=agent.config['generation']['num_candidates'],
                    temperature=agent.config['generation']['temperature'],
                    max_tokens=agent.config['generation']['max_tokens']
                )
                # 解析并验证代码
                candidates = []
                for i, output in enumerate(raw_outputs):
                    code = agent._extract_code(output)
                    if code and agent._validate_reward_function(code):
                        candidates.append(code)

                # 如果有效候选不足，用默认代码填充
                while len(candidates) < agent.config['generation']['num_candidates']:
                    candidates.append(agent._get_default_reward_code())

                # 训练并评估
                if use_real_training:
                    simulator = SimulationTool(
                        base_dir=agent.config['logging']['save_dir'],
                        max_workers=agent.config['training']['parallel_workers'],
                        timeout=agent.config['training']['timeout'],
                        episode_num=agent.config['training']['episode_num'],
                        use_gpu=agent.config['training']['use_gpu']
                    )
                    results = simulator.run_parallel(candidates, generation)
                else:
                    results = agent._simulate_training(candidates)

                # 分析结果
                best_code, reflection, best_fitness = agent.analyze_results(results)

                # 保存到记忆
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
            else:
                # 后续代使用正常流程
                return original_step(generation, use_real_training)

        agent.step = modified_step


class NoDream(AblationVariant):
    """禁用DREAM模块：使用静态算子分配替代自适应分配"""

    def __init__(self):
        super().__init__("no_dream", "禁用DREAM模块（静态算子分配）")

    def modify_agent_behavior(self, agent: RewardDesignAgent):
        """强制使用静态循环分配，忽略LLM的算子选择"""
        original_step = agent.step

        def modified_step(generation: int, use_real_training: bool = True) -> dict:
            if generation > 0:
                # 保存原始的变异方法
                original_mutate = agent._apply_dream_mutation

                # 覆盖变异方法，使用静态分配
                def static_mutation(parent_code: str, reflection: str,
                                    candidate_idx: int, n_candidates: int = 4) -> str:
                    # 静态循环分配算子
                    operators = ['F1', 'F2', 'F3', 'L1']
                    assigned_operator = operators[candidate_idx % len(operators)]

                    # 直接使用分配的算子，不解析reflection
                    return original_mutate(parent_code, reflection, candidate_idx,
                                           n_candidates, force_operator=assigned_operator)

                agent._apply_dream_mutation = static_mutation

            return original_step(generation, use_real_training)

        agent.step = modified_step


class NoConvergenceFilter(AblationVariant):
    """禁用收敛过滤：不进行三重收敛判别"""

    def __init__(self):
        super().__init__("no_convergence_filter", "禁用收敛过滤（仅使用单维适应度）")

    def modify_agent_behavior(self, agent: RewardDesignAgent):
        """跳过收敛过滤，直接取最高fitness"""
        original_analyze = agent.analyze_results

        def modified_analyze(results: list) -> Tuple[str, str, float]:
            """简化分析：直接取fitness最高的候选"""
            best_result = max(results, key=lambda x: x.get('fitness', -float('inf')))
            best_code = best_result.get('code', '')
            best_fitness = best_result.get('fitness', 0.0)

            # 生成简化的reflection
            reflection = f"选择fitness最高的候选: {best_fitness:.4f}"

            return best_code, reflection, best_fitness

        agent.analyze_results = modified_analyze


class NoDegradationSelection(AblationVariant):
    """禁用降级选拔：不进行降级处理，未收敛候选直接被淘汰"""

    def __init__(self):
        super().__init__("no_degradation_selection", "禁用降级选拔（未收敛候选被淘汰）")

    def modify_agent_behavior(self, agent: RewardDesignAgent):
        """修改分析逻辑：只选择收敛的候选"""
        original_analyze = agent.analyze_results

        def modified_analyze(results: list) -> Tuple[str, str, float]:
            """严格分析：只选择收敛的候选"""
            converged_results = [r for r in results if r.get('is_converged', False)]

            if converged_results:
                # 有收敛候选，取最高fitness
                best_result = max(converged_results, key=lambda x: x.get('fitness', -float('inf')))
                reflection = "从收敛候选中选择最高fitness"
            else:
                # 无收敛候选，取最高fitness但标记警告
                best_result = max(results, key=lambda x: x.get('fitness', -float('inf')))
                reflection = "警告：无收敛候选，选择fitness最高的候选"

            best_code = best_result.get('code', '')
            best_fitness = best_result.get('fitness', 0.0)

            return best_code, reflection, best_fitness

        agent.analyze_results = modified_analyze


class SingleDimensionFitness(AblationVariant):
    """单维适应度：只使用成功率作为fitness"""

    def __init__(self):
        super().__init__("single_fitness", "单维适应度（仅使用成功率）")

    def modify_config(self, config: dict) -> dict:
        """修改fitness权重，只保留成功率"""
        config = copy.deepcopy(config)
        config['fitness']['weights'] = {
            'success_rate': 1.0,
            'capture_time': 0.0,
            'formation_quality': 0.0,
            'collision_penalty': 0.0
        }
        return config


class NoAIRNoDream(AblationVariant):
    """同时禁用AIR和DREAM"""

    def __init__(self):
        super().__init__("no_air_no_dream", "禁用AIR和DREAM模块")

    def modify_agent_behavior(self, agent: RewardDesignAgent):
        """同时禁用AIR和DREAM"""
        NoAIR().modify_agent_behavior(agent)
        NoDream().modify_agent_behavior(agent)


class NoAIRNoFitness(AblationVariant):
    """同时禁用AIR和多维适应度"""

    def __init__(self):
        super().__init__("no_air_no_fitness", "禁用AIR和多维适应度")

    def modify_agent_behavior(self, agent: RewardDesignAgent):
        """同时禁用AIR和使用单维适应度"""
        NoAIR().modify_agent_behavior(agent)

    def modify_config(self, config: dict) -> dict:
        """使用单维适应度"""
        return SingleDimensionFitness().modify_config(config)


class NoDreamNoFitness(AblationVariant):
    """同时禁用DREAM和多维适应度"""

    def __init__(self):
        super().__init__("no_dream_no_fitness", "禁用DREAM和多维适应度")

    def modify_agent_behavior(self, agent: RewardDesignAgent):
        """禁用DREAM"""
        NoDream().modify_agent_behavior(agent)

    def modify_config(self, config: dict) -> dict:
        """使用单维适应度"""
        return SingleDimensionFitness().modify_config(config)


class AblationStudy:
    """消融实验管理器"""

    # 所有可用变体
    AVAILABLE_VARIANTS = {
        'full': (FullSystem, "完整LEMS系统（AIR + DREAM + 多维适应度 + 收敛过滤 + 降级选拔）"),
        'no_air': (NoAIR, "禁用AIR模块（无CoT两阶段分析）"),
        'no_dream': (NoDream, "禁用DREAM模块（静态算子分配）"),
        'no_convergence_filter': (NoConvergenceFilter, "禁用收敛过滤（仅使用单维适应度）"),
        'no_degradation_selection': (NoDegradationSelection, "禁用降级选拔（未收敛候选被淘汰）"),
        'single_fitness': (SingleDimensionFitness, "单维适应度（仅使用成功率）"),
        'no_air_no_dream': (NoAIRNoDream, "禁用AIR和DREAM模块"),
        'no_air_no_fitness': (NoAIRNoFitness, "禁用AIR和多维适应度"),
        'no_dream_no_fitness': (NoDreamNoFitness, "禁用DREAM和多维适应度")
    }

    def __init__(self, config_path: str = 'llm_reward_agent/config/llm_config.yaml'):
        self.config_path = config_path
        self.variants = {}
        self.results = {}

        # 注册所有变体
        self._register_variants()

    def _register_variants(self):
        """注册所有消融实验变体"""
        for name, (variant_class, _) in self.AVAILABLE_VARIANTS.items():
            self.variants[name] = variant_class()

    def run_variant(self, variant_name: str, num_generations: int,
                    episode_num: int = 100, use_real_training: bool = True,
                    env_file: str = 'MADDPG/envs/simple_tag_env.py',
                    task_description: str = None) -> dict:
        """运行单个消融实验变体"""
        if variant_name not in self.variants:
            raise ValueError(f"未知变体: {variant_name}，可用变体: {list(self.variants.keys())}")

        variant = self.variants[variant_name]
        print(f"\n{'='*80}")
        print(f"运行消融实验变体: {variant.name}")
        print(f"说明: {variant.description}")
        print(f"{'='*80}")

        # 加载基础配置
        import yaml
        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        # 修改配置
        config = variant.modify_config(config)

        # 临时保存修改后的配置
        temp_config_path = f'/tmp/ablation_config_{variant_name}.yaml'
        with open(temp_config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

        # 创建agent
        agent = RewardDesignAgent(config_path=temp_config_path)

        # 修改agent行为
        variant.modify_agent_behavior(agent)

        # 设置训练参数
        if use_real_training:
            agent.config['training']['episode_num'] = episode_num
            agent.config['training']['parallel_workers'] = 4

        # 初始化agent
        if task_description is None:
            task_description = """任务：3个追捕智能体协同围捕1个逃逸目标。
要求：
1. 追捕者需要接近并包围目标
2. 追捕者之间避免碰撞
3. 形成均匀的包围圈
4. 在尽可能短的时间内完成围捕
"""

        agent.initialize(
            env_file_path=env_file,
            task_description=task_description
        )

        # 运行进化
        start_time = time.time()
        variant_results = []

        for generation in range(num_generations):
            print(f"\n--- {variant.name} - 第 {generation+1}/{num_generations} 代 ---")

            try:
                result = variant.get_generation_result(
                    agent, generation, use_real_training
                )
                variant_results.append(result)

                print(f"  最优Fitness: {result.get('best_fitness', 0):.4f}")

            except Exception as e:
                print(f"  错误: {e}")
                import traceback
                traceback.print_exc()
                # 记录失败结果
                variant_results.append({
                    'generation': generation,
                    'best_fitness': 0.0,
                    'error': str(e)
                })

        total_time = time.time() - start_time

        # 计算统计信息
        fitness_values = [r.get('best_fitness', 0) for r in variant_results]
        best_fitness = max(fitness_values) if fitness_values else 0
        final_fitness = fitness_values[-1] if fitness_values else 0
        avg_fitness = np.mean(fitness_values) if fitness_values else 0

        result_summary = {
            'variant': variant_name,
            'description': variant.description,
            'num_generations': num_generations,
            'total_time': total_time,
            'best_fitness': best_fitness,
            'final_fitness': final_fitness,
            'avg_fitness': avg_fitness,
            'fitness_history': fitness_values,
            'generation_results': variant_results
        }

        self.results[variant_name] = result_summary
        return result_summary

    def run_full_study(self, num_generations: int, episode_num: int = 100,
                       use_real_training: bool = True, variants: List[str] = None):
        """运行完整消融实验"""
        if variants is None:
            variants = list(self.variants.keys())

        print("\n" + "="*80)
        print("LEMS 消融实验")
        print("="*80)
        print(f"变体数量: {len(variants)}")
        print(f"进化代数: {num_generations}")
        print(f"训练模式: {'真实训练' if use_real_training else '模拟训练'}")
        print("="*80)

        study_start_time = time.time()

        for variant_name in variants:
            try:
                self.run_variant(
                    variant_name=variant_name,
                    num_generations=num_generations,
                    episode_num=episode_num,
                    use_real_training=use_real_training
                )
            except Exception as e:
                print(f"\n变体 {variant_name} 运行失败: {e}")
                import traceback
                traceback.print_exc()

        total_time = time.time() - study_start_time

        # 保存结果
        self.save_results()

        # 打印总结
        self.print_summary()

        return self.results

    def save_results(self, output_dir: str = 'experiments/ablation_study'):
        """保存实验结果"""
        os.makedirs(output_dir, exist_ok=True)

        # 保存详细结果
        results_file = os.path.join(output_dir, 'ablation_results.json')
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False, default=str)

        print(f"\n详细结果已保存到: {results_file}")

        # 保存摘要表格
        summary_file = os.path.join(output_dir, 'ablation_summary.txt')
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("LEMS 消融实验结果摘要\n")
            f.write("="*80 + "\n\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # 表格头
            f.write(f"{'变体':<25} {'最优Fitness':<15} {'最终Fitness':<15} {'平均Fitness':<15} {'耗时(秒)':<12}\n")
            f.write("-"*82 + "\n")

            # 按最优fitness排序
            sorted_variants = sorted(
                self.results.items(),
                key=lambda x: x[1].get('best_fitness', 0),
                reverse=True
            )

            for variant_name, result in sorted_variants:
                f.write(f"{variant_name:<25} "
                       f"{result.get('best_fitness', 0):<15.4f} "
                       f"{result.get('final_fitness', 0):<15.4f} "
                       f"{result.get('avg_fitness', 0):<15.4f} "
                       f"{result.get('total_time', 0):<12.1f}\n")

            f.write("\n" + "="*80 + "\n")

            # 各模块贡献分析
            f.write("\n各模块贡献分析\n")
            f.write("-"*80 + "\n")

            full_result = self.results.get('full', {})
            full_best = full_result.get('best_fitness', 0)

            if full_best > 0:
                f.write(f"\n完整系统最优Fitness: {full_best:.4f}\n\n")
                f.write("禁用各模块后的性能下降:\n")

                contributions = []
                for variant_name, result in self.results.items():
                    if variant_name != 'full':
                        variant_best = result.get('best_fitness', 0)
                        degradation = (full_best - variant_best) / full_best * 100
                        contributions.append((variant_name, result.get('description', ''), degradation, variant_best))

                contributions.sort(key=lambda x: x[2], reverse=True)

                for variant_name, description, degradation, variant_best in contributions:
                    f.write(f"  {variant_name:<25} 下降 {degradation:>6.2f}% "
                           f"(Fitness: {variant_best:.4f})\n")
                    f.write(f"    说明: {description}\n")

        print(f"摘要已保存到: {summary_file}")

    def print_summary(self):
        """打印实验结果摘要"""
        print("\n" + "="*80)
        print("消融实验结果摘要")
        print("="*80)

        # 按最优fitness排序
        sorted_variants = sorted(
            self.results.items(),
            key=lambda x: x[1].get('best_fitness', 0),
            reverse=True
        )

        print(f"\n{'变体':<25} {'最优Fitness':<15} {'最终Fitness':<15}")
        print("-"*55)

        for variant_name, result in sorted_variants:
            print(f"{variant_name:<25} "
                  f"{result.get('best_fitness', 0):<15.4f} "
                  f"{result.get('final_fitness', 0):<15.4f}")

        # 计算各模块贡献
        full_result = self.results.get('full', {})
        full_best = full_result.get('best_fitness', 0)

        if full_best > 0:
            print(f"\n各模块贡献分析:")
            print(f"完整系统最优Fitness: {full_best:.4f}\n")

            for variant_name, result in self.results.items():
                if variant_name != 'full':
                    variant_best = result.get('best_fitness', 0)
                    degradation = (full_best - variant_best) / full_best * 100
                    print(f"  {variant_name:<25} 性能下降: {degradation:>6.2f}%")

        print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(
        description="LEMS 消融实验",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 运行完整消融实验
  python ablation_study.py --num_generations 5 --episode_num 100

  # 只运行特定变体
  python ablation_study.py --variants full no_air no_dream --num_generations 3

  # 使用模拟训练快速测试
  python ablation_study.py --num_generations 3 --no-real-training

  # 查看可用变体
  python ablation_study.py --list-variants
        """
    )

    parser.add_argument('--config', type=str,
                        default='llm_reward_agent/config/llm_config.yaml',
                        help='LLM配置文件路径')

    parser.add_argument('--num_generations', type=int, default=5,
                        help='进化代数（默认: 5）')

    parser.add_argument('--use_real_training', dest='use_real_training',
                        action='store_true', help='使用真实训练（默认）')

    parser.add_argument('--no-real-training', dest='use_real_training',
                        action='store_false', help='使用模拟训练（快速测试）')

    parser.set_defaults(use_real_training=True)

    parser.add_argument('--episode_num', type=int, default=100,
                        help='每个候选的训练回合数（默认: 100）')

    parser.add_argument('--env_file', type=str,
                        default='MADDPG/envs/simple_tag_env.py',
                        help='环境文件路径')

    parser.add_argument('--variants', nargs='+', type=str,
                        default=None,
                        help='要运行的变体列表（默认: 全部）')

    parser.add_argument('--list-variants', action='store_true',
                        help='列出所有可用变体')

    parser.add_argument('--output_dir', type=str,
                        default='experiments/ablation_study',
                        help='结果输出目录')

    args = parser.parse_args()

    # 列出变体
    if args.list_variants:
        study = AblationStudy(config_path=args.config)
        print("\n可用的消融实验变体:")
        print("-"*60)
        for name, variant in study.variants.items():
            print(f"  {name:<25} {variant.description}")
        print("\n使用 --variants 参数指定要运行的变体")
        return

    # 运行实验
    study = AblationStudy(config_path=args.config)

    try:
        results = study.run_full_study(
            num_generations=args.num_generations,
            episode_num=args.episode_num,
            use_real_training=args.use_real_training,
            variants=args.variants
        )

        # 保存结果
        study.save_results(output_dir=args.output_dir)

        print("\n消融实验完成！")

    except KeyboardInterrupt:
        print("\n\n用户中断实验")
        # 保存已完成的结果
        if study.results:
            study.save_results(output_dir=args.output_dir)
            print("已保存部分结果")

    except Exception as e:
        print(f"\n实验失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
