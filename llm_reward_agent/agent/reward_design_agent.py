"""
奖励函数设计智能体
整合LLM、记忆等模块，实现完整的奖励函数进化流程

Author: LEMS Project
Date: 2026-02-03
Version: 1.0
"""

import os
import re
import ast
import yaml
from typing import List, Dict, Tuple, Optional

from .llm_interface import LLMInterface
from .prompt_templates import PromptTemplates, PREDEFINED_ENV_CONTEXT, PREDEFINED_TASK_DESCRIPTION
from .memory import EvolutionaryMemory


class RewardDesignAgent:
    """奖励函数设计智能体"""
    
    def __init__(self, config_path: str = "llm_reward_agent/config/llm_config.yaml"):
        """
        初始化智能体
        
        Args:
            config_path: 配置文件路径
        """
        # 加载配置
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        print("=" * 80)
        print("初始化奖励函数设计智能体")
        print("=" * 80)
        
        # 初始化组件
        print("\n[1/4] 初始化LLM接口...")
        self.llm = LLMInterface(
            provider=self.config['llm']['provider'],
            model_name=self.config['llm']['model'],
            api_key=self._get_api_key(),
            base_url=self.config['llm'].get('base_url'),
            timeout=self.config['llm'].get('timeout', 120),
            max_retries=self.config['llm'].get('max_retries', 3)
        )
        
        print("[2/4] 初始化记忆管理...")
        save_dir = os.path.join(
            self.config['logging']['save_dir'],
            'evolution_archive'
        )
        self.memory = EvolutionaryMemory(save_dir=save_dir)
        
        print("[3/4] 初始化提示词模板...")
        self.prompt_builder = PromptTemplates()
        
        # 状态变量
        self.env_context = PREDEFINED_ENV_CONTEXT  # 使用预定义的环境上下文
        self.task_description = PREDEFINED_TASK_DESCRIPTION  # 使用预定义的任务描述
        self.current_generation = 0
        
        print("\n✅ 智能体初始化完成！(使用预定义环境上下文)")
        print("=" * 80)
    
    def _get_api_key(self) -> Optional[str]:
        """获取API密钥（优先从环境变量读取）"""
        provider = self.config['llm'].get('provider', 'openai')

        # 首先尝试从环境变量读取
        env_var_map = {
            'openai': 'OPENAI_API_KEY',
            'anthropic': 'ANTHROPIC_API_KEY',
            'zhipu': 'ZHIPU_API_KEY',
            'deepseek': 'DEEPSEEK_API_KEY',
        }
        env_var = env_var_map.get(provider.lower(), 'OPENAI_API_KEY')
        api_key = os.getenv(env_var)

        if api_key:
            return api_key

        # 其次尝试从配置文件读取（支持 ${ENV_VAR} 格式）
        config_key = self.config['llm'].get('api_key')
        if config_key:
            if config_key.startswith('${') and config_key.endswith('}'):
                # 从环境变量读取
                env_var = config_key[2:-1]  # 去掉 ${ 和 }
                api_key = os.getenv(env_var)
            else:
                # 直接使用配置文件中的key（不推荐，仅向后兼容）
                api_key = config_key

        return api_key
    
    def initialize(self, env_file_path: str = None, task_description: str = None):
        """
        初始化智能体（使用预定义环境上下文，无需动态提取）
        
        Args:
            env_file_path: 环境文件路径（可选，已使用预定义上下文）
            task_description: 任务描述（可选，默认为 PREDEFINED_TASK_DESCRIPTION）
        """
        print("\n" + "=" * 80)
        print("初始化任务环境")
        print("=" * 80)
        
        # 使用预定义的环境上下文
        self.env_context = PREDEFINED_ENV_CONTEXT
        
        # 如果提供了自定义任务描述，则使用它
        if task_description:
            self.task_description = task_description
        else:
            self.task_description = PREDEFINED_TASK_DESCRIPTION
        
        # 打印关键信息
        print(f"✅ 环境名称: {self.env_context.get('env_name', 'simple_tag_env')}")
        print(f"✅ 智能体数量: {self.env_context.get('agent_info', {})} 个")
        print(f"✅ 物理常量: {self.env_context.get('physical_constants', {})}")
        print(f"✅ 任务描述: {self.task_description[:100]}...")
        
        # 打印Token估算
        prompt = self.prompt_builder.initial_generation_prompt_with_predefined_context(
            self.task_description,
            self.env_context
        )
        token_estimate = len(prompt) // 4
        print(f"✅ 提示词Token估算: ~{token_estimate}")
        
        print("=" * 80)
    
    def generate_candidates(self, generation: int) -> List[str]:
        """
        生成候选奖励函数代码（使用预定义环境上下文）
        
        Args:
            generation: 当前代数
        
        Returns:
            List[str]: 候选代码列表
        """
        print(f"\n{'=' * 80}")
        print(f"🤖 第 {generation} 代: 生成候选奖励函数")
        print(f"{'=' * 80}")
        
        max_retries = 3
        min_valid_codes = 2  # 至少需要2个有效候选
        
        for attempt in range(max_retries):
            try:
                if generation == 0:
                    # 第一代：Zero-Shot生成（使用预定义上下文）
                    print(f"📝 使用Zero-Shot策略生成... (尝试 {attempt + 1}/{max_retries})")
                    prompt = self.prompt_builder.initial_generation_prompt_with_predefined_context(
                        # self.task_description,
                        # self.env_context
                    )
                    
                    # 生成多个候选
                    n_candidates = self.config['generation']['num_candidates']
                    raw_outputs = self.llm.generate(
                        prompt=prompt,
                        n=n_candidates,
                        temperature=self.config['generation']['temperature'],
                        max_tokens=self.config['generation']['max_tokens'],
                        system_message=self.prompt_builder.SYSTEM_MESSAGE
                    )
                    
                    # 解析代码
                    codes = self._parse_code_blocks(raw_outputs)
                
                else:
                    # 后续代：基于父本进化（使用预定义上下文）
                    print(f"🧬 使用进化策略生成... (尝试 {attempt + 1}/{max_retries})")
                    parent_code = self.memory.get_best_code(generation - 1)
                    reflection = self.memory.get_reflection(generation - 1)

                    prompt = self.prompt_builder.evolution_prompt_with_predefined_context(
                        # self.task_description,
                        parent_code,
                        reflection,
                        # n_candidates=self.config['generation']['num_candidates'],
                        # env_context=self.env_context
                    )

                    raw_outputs = self.llm.generate(
                        prompt=prompt,
                        n=1,  
                        temperature=self.config['generation']['temperature'],
                        max_tokens=self.config['generation']['max_tokens'],
                        system_message=self.prompt_builder.SYSTEM_MESSAGE
                    )

                    # 合并所有变体
                    all_codes = []
                    for raw_output in raw_outputs:
                        variants = self._parse_variants(raw_output)
                        all_codes.extend(variants)
                    codes = all_codes
                
                # 语法检查和过滤
                valid_codes = []
                for i, code in enumerate(codes):
                    if self._syntax_check(code):
                        valid_codes.append(code)
                        print(f"  ✅ 候选 {i}: 语法检查通过")
                    else:
                        print(f"  ❌ 候选 {i}: 语法错误，已跳过")
                
                # 检查是否有足够的有效代码
                if len(valid_codes) >= min_valid_codes:
                    print(f"\n✅ 成功生成 {len(valid_codes)} 个有效候选")
                    return valid_codes
                else:
                    print(f"\n⚠️ 有效代码不足（{len(valid_codes)}/{min_valid_codes}），重新生成...")
            
            except Exception as e:
                print(f"\n❌ 生成失败: {e}")
                if attempt < max_retries - 1:
                    print(f"   重试中...")
        
        # 所有尝试都失败，使用后备方案
        print("\n⚠️ 所有尝试都失败，使用后备方案...")
        return self._get_fallback_codes(generation)
    
    def _get_fallback_codes(self, generation: int) -> List[str]:
        """
        获取后备代码
        
        Args:
            generation: 当前代数
        
        Returns:
            List[str]: 后备代码列表
        """
        fallback_codes = []
        
        # 方案1: 使用上一代的最优代码
        if generation > 0:
            try:
                parent_code = self.memory.get_best_code(generation - 1)
                fallback_codes.append(parent_code)
                print("  📋 使用上一代最优代码作为后备")
            except:
                pass
        
        # 方案2: 使用人工基准代码
        if len(fallback_codes) == 0:
            try:
                baseline_code = self._get_human_baseline()
                fallback_codes.append(baseline_code)
                print("  📋 使用人工基准代码作为后备")
            except:
                pass
        
        
        return fallback_codes
    
    def _get_human_baseline(self) -> str:
        """
        获取人工设计的基准奖励函数
        
        Returns:
            str: 基准代码
        """
        baseline_path = "MADDPG/envs/reward_function.py"
        
        if os.path.exists(baseline_path):
            with open(baseline_path, 'r', encoding='utf-8') as f:
                return f.read()
        else:
            raise FileNotFoundError(f"基准文件不存在: {baseline_path}")
    
    def analyze_results(self, results: List[Dict]) -> Tuple[str, str]:
        """
        分析训练结果，生成反思
        
        Args:
            results: 训练结果列表
        
        Returns:
            Tuple[str, str]: (最优代码, 反思内容)
        """
        print(f"\n{'=' * 80}")
        print("🔍 分析训练结果")
        print(f"{'=' * 80}")
        
        # 1. 找到最优代码
        valid_results = [r for r in results if r.get('status') == 'success']
        
        if not valid_results:
            print("⚠️ 所有候选都训练失败，使用后备方案...")
            # 返回第一个候选和默认反思
            return results[0]['code'], "所有候选训练失败，需要检查环境或训练配置。"
        
        best_result = max(valid_results, key=lambda x: x.get('fitness', -float('inf')))
        
        print(f"✅ 最优候选: {best_result['id']}")
        print(f"   Fitness: {best_result['fitness']:.4f}")
        
        # 2. 构建日志摘要
        logs_summary = self._format_logs(valid_results)
        
        # 3. 调用LLM生成Reflection
        print("\n🤔 LLM正在生成反思...")
        prompt = self.prompt_builder.reflection_prompt(logs_summary)
        reflection = self.llm.analyze(
            prompt=prompt,
            temperature=self.config['reflection']['temperature'],
            max_tokens=self.config['reflection']['max_tokens']
        )
        
        print(f"\n📊 反思内容（前200字符）:")
        print(reflection[:200] + "..." if len(reflection) > 200 else reflection)
        
        return best_result['code'], reflection
    
    def step(self, generation: int, use_real_training: bool = True) -> Dict:
        """
        执行一代进化（核心流程）
        
        Args:
            generation: 代数
            use_real_training: 是否使用真实训练（False则使用模拟数据）
        
        Returns:
            Dict: 本代结果
        """
        self.current_generation = generation
        
        # 1. 生成候选代码
        codes = self.generate_candidates(generation)
        
        # 2. 调用仿真工具训练
        if use_real_training:
            print(f"\n{'=' * 80}")
            print("🚀 开始并行训练（真实训练）")
            print(f"{'=' * 80}")
            
            # 使用真实的并行训练
            from ..tools.simulation_tool import SimulationTool
            
            # 从配置读取训练参数
            training_config = self.config.get('training', {})
            
            simulator = SimulationTool(
                base_dir=self.config['logging']['save_dir'],
                max_workers=training_config.get('parallel_workers', 4),
                timeout=training_config.get('timeout', 1200),
                episode_num=training_config.get('episode_num', 100)
            )
            
            results = simulator.run_parallel(codes, generation)
        else:
            print(f"\n{'=' * 80}")
            print("🚀 开始并行训练（模拟模式）")
            print(f"{'=' * 80}")
            
            # 使用模拟数据（用于快速测试）
            results = self._simulate_training(codes)
        
        # 3. 分析结果
        best_code, reflection = self.analyze_results(results)
        
        # 4. 计算fitness
        valid_results = [r for r in results if r.get('status') == 'success']
        if valid_results:
            best_fitness = max(r.get('fitness', 0) for r in valid_results)
        else:
            best_fitness = 0.0
        
        # 5. 更新记忆
        self.memory.save(
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
    
    def _parse_code_blocks(self, raw_outputs: List[str]) -> List[str]:
        """从LLM输出中提取Python代码块"""
        codes = []
        
        for output in raw_outputs:
            # 匹配 ```python ... ```
            matches = re.findall(r'```python\n(.*?)\n```', output, re.DOTALL)
            
            if matches:
                # 取第一个匹配
                codes.append(matches[0].strip())
            else:
                # 尝试匹配没有语言标记的代码块
                matches = re.findall(r'```\n(.*?)\n```', output, re.DOTALL)
                if matches:
                    codes.append(matches[0].strip())
                else:
                    # 如果都没有，尝试直接使用整个输出
                    if 'def compute_reward' in output:
                        codes.append(output.strip())
        
        return codes
    
    def _parse_variants(self, raw_output: str) -> List[str]:
        """解析包含多个变体的输出"""
        # 按 # === VARIANT X === 分割
        variants = re.split(r'#\s*===\s*VARIANT\s+\d+\s*===', raw_output)
        
        # 过滤掉空字符串
        variants = [v.strip() for v in variants if v.strip()]
        
        # 提取每个变体中的代码
        codes = []
        for variant in variants:
            # 尝试提取代码块
            matches = re.findall(r'```python\n(.*?)\n```', variant, re.DOTALL)
            if matches:
                codes.append(matches[0].strip())
            elif 'def compute_reward' in variant:
                # 直接使用文本
                codes.append(variant.strip())
        
        # 如果解析失败，尝试按代码块分割
        if len(codes) == 0:
            codes = self._parse_code_blocks([raw_output])
        
        return codes
    
    def _syntax_check(self, code: str) -> bool:
        """检查代码语法是否正确"""
        try:
            ast.parse(code)
            return True
        except SyntaxError as e:
            print(f"  语法错误: {e}")
            return False
    
    def _format_logs(self, results: List[Dict]) -> str:
        """格式化日志为自然语言"""
        summary = f"# 本代 {len(results)} 个候选的训练结果\n\n"
        
        for result in results:
            i = result.get('id', '未知')
            summary += f"## Candidate {i}\n"
            
            if result.get('status') == 'success':
                metrics = result.get('metrics', {})
                summary += f"- **Fitness**: {result.get('fitness', 0):.4f}\n"
                summary += f"- **成功率**: {metrics.get('success_rate', 0):.2%}\n"
                summary += f"- **平均捕获时间**: {metrics.get('avg_capture_time', 0):.1f} steps\n"
                
                # 奖励分量
                reward_comps = metrics.get('reward_components', {})
                if reward_comps:
                    summary += "- **奖励分量统计**:\n"
                    for key, val in reward_comps.items():
                        if isinstance(val, dict):
                            summary += f"  * {key}: mean={val.get('mean', 0):.4f}, std={val.get('std', 0):.4f}\n"
                        else:
                            summary += f"  * {key}: {val:.4f}\n"
                
                # 协同指标
                collab_metrics = metrics.get('collaboration_metrics', {})
                if collab_metrics:
                    summary += "- **协同指标**:\n"
                    for key, val in collab_metrics.items():
                        if isinstance(val, dict):
                            summary += f"  * {key}: mean={val.get('mean', 0):.4f}\n"
                        else:
                            summary += f"  * {key}: {val:.4f}\n"
            else:
                summary += f"- **状态**: {result.get('status', '未知')}\n"
                summary += f"- **错误**: {result.get('error', '未知')}\n"
            
            summary += "\n"
        
        return summary
    
    def _simulate_training(self, codes: List[str]) -> List[Dict]:
        """模拟训练结果（用于测试，阶段三将替换为真实训练）"""
        import random
        
        print("⚠️ 使用模拟训练数据（阶段三将替换为真实训练）")
        
        results = []
        for i, code in enumerate(codes):
            # 模拟训练结果
            success_rate = random.uniform(0.6, 0.9)
            avg_time = random.uniform(40, 60)
            
            # 计算fitness
            fitness = success_rate - 0.001 * avg_time
            
            result = {
                'id': i,
                'code': code,
                'status': 'success',
                'fitness': fitness,
                'metrics': {
                    'success_rate': success_rate,
                    'avg_capture_time': avg_time,
                    'reward_components': {
                        'distance_reward': {'mean': random.uniform(-2, -0.5), 'std': 0.3},
                        'collision_penalty': {'mean': random.uniform(-1, -0.1), 'std': 0.2},
                        'formation_reward': {'mean': random.uniform(0, 1), 'std': 0.15}
                    },
                    'collaboration_metrics': {
                        'encirclement_angle_std': {'mean': random.uniform(0.2, 0.5)},
                        'min_agent_distance': {'mean': random.uniform(0.2, 0.4)}
                    }
                }
            }
            
            results.append(result)
            print(f"  候选 {i}: Fitness={fitness:.4f}, 成功率={success_rate:.2%}")
        
        return results


# ========================================
# 测试代码
# ========================================
if __name__ == "__main__":
    print("测试奖励函数设计智能体（使用预定义环境上下文）...")
    
    # 注意：测试前需要设置API密钥环境变量
    # export OPENAI_API_KEY=your_key
    
    try:
        # 1. 初始化智能体（自动使用预定义上下文）
        agent = RewardDesignAgent(config_path="llm_reward_agent/config/llm_config.yaml")
        
        # 2. 初始化任务（可选，如果不提供则使用预定义任务描述）
        agent.initialize(
            env_file_path=None,  # 不再需要
            task_description=None  # 可选，如果不提供则使用 PREDEFINED_TASK_DESCRIPTION
        )
        
        # 或者直接使用默认初始化（已在 __init__ 中完成）
        # agent.initialize()
        
        # 3. 运行一代进化（使用模拟训练）
        print("\n" + "=" * 80)
        print("开始测试进化流程")
        print("=" * 80)
        
        result = agent.step(generation=0, use_real_training=False)  # 使用模拟训练
        
        print("\n" + "=" * 80)
        print("第0代进化完成")
        print("=" * 80)
        print(f"最优Fitness: {result['best_fitness']:.4f}")
        print(f"反思: {result['reflection'][:200]}...")
        
        # 4. 运行第二代（测试进化）
        result = agent.step(generation=1, use_real_training=False)
        
        print("\n" + "=" * 80)
        print("第1代进化完成")
        print("=" * 80)
        print(f"最优Fitness: {result['best_fitness']:.4f}")
        
        # 5. 导出摘要
        summary = agent.memory.export_summary()
        print("\n" + summary)
        
        print("\n✅ 奖励函数设计智能体测试完成！")
    
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        print("\n提示：")
        print("1. 请确保已设置API密钥环境变量")
        print("2. 如果不想调用真实LLM，可以使用 use_real_training=False")
        print("3. 环境上下文已预定义在 prompt_templates.py 中")
