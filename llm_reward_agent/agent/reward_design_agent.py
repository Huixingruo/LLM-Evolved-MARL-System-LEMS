"""
奖励函数设计智能体
整合LLM、记忆等模块，实现完整的奖励函数进化流程

Author: LEMS Project
Date: 2026-02-03
Version: 1.0
"""
# python -m llm_reward_agent.agent.reward_design_agent

import os
import re
import ast
import yaml
import concurrent.futures
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
        self.cot_analysis_result = None  # CoT思维链分析结果（两阶段管线用）

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

    def _save_llm_interaction(self, generation: int, phase: str, prompt: str, response: str, candidate_id: str = None):
        """
        强制拦截并保存大模型的交互记录
        目录结构: save_dir/llm_interactions/generation_X/
        """
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # 构建独立存储目录
        log_dir = os.path.join(
            self.config['logging']['save_dir'],
            'llm_interactions',
            f'generation_{generation}'
        )
        os.makedirs(log_dir, exist_ok=True)

        # 组装文件名
        filename_parts = [timestamp, phase]
        if candidate_id is not None:
            filename_parts.append(f"candidate_{candidate_id}")
        filename = "_".join(filename_parts) + ".md"  # 使用Markdown格式便于阅读代码

        filepath = os.path.join(log_dir, filename)

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"# LLM Interaction Log\n\n")
            f.write(f"- **Generation**: {generation}\n")
            f.write(f"- **Phase**: {phase}\n")
            if candidate_id is not None:
                f.write(f"- **Candidate Info**: {candidate_id}\n")
            f.write(f"- **Timestamp**: {timestamp}\n")
            f.write(f"\n{'='*80}\n## Prompt (Sent to LLM)\n{'='*80}\n\n")
            f.write(f"```text\n{prompt}\n```\n")
            f.write(f"\n{'='*80}\n## Response (From LLM)\n{'='*80}\n\n")
            f.write(f"```text\n{response}\n```\n")
    
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

        # 每次初始化重置CoT缓存（确保新一代从头开始）
        self.cot_analysis_result = None

        # 打印关键信息
        print(f"✅ 环境名称: {self.env_context.get('env_name', 'simple_tag_env')}")
        print(f"✅ 智能体数量: {self.env_context.get('agent_info', {})} 个")
        print(f"✅ 物理常量: {self.env_context.get('physical_constants', {})}")
        print(f"✅ 任务描述: {self.task_description[:100]}...")

        print("=" * 80)
    
    def generate_candidates(self, generation: int) -> List[str]:
        """
        生成候选奖励函数代码（引入CoT两阶段管线）

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
        n_candidates = self.config['generation']['num_candidates']

        for attempt in range(max_retries):
            try:
                raw_outputs = []

                if generation == 0:
                    # ==========================================
                    # 阶段一：强制执行CoT环境解析（仅执行一次并缓存）
                    # ==========================================
                    if not self.cot_analysis_result:
                        print(f"🔍 [阶段一] 执行CoT环境与任务结构分析...")
                        analysis_prompt = self.prompt_builder.cot_analysis_prompt(
                            self.task_description, self.env_context
                        )
                        # 低温度采样保证逻辑严密性
                        self.cot_analysis_result = self.llm.analyze(
                            prompt=analysis_prompt,
                            temperature=0.3,
                            max_tokens=10000
                        )
                        # 【新增埋点1】保存CoT分析记录
                        self._save_llm_interaction(generation, "Phase1_CoT_Analysis", analysis_prompt, self.cot_analysis_result)
                        print("   ✅ CoT环境解析完成，已建立MDP表征先验。")

                    # ==========================================
                    # 阶段二：基于CoT先验并行生成候选代码
                    # ==========================================
                    print(f"📝 [阶段二] 基于先验并行生成代码... (尝试 {attempt + 1}/{max_retries})")
                    prompt = self.prompt_builder.initial_generation_prompt_with_cot(
                        self.task_description,
                        self.env_context,
                        self.cot_analysis_result
                    )

                    # 依赖底层多线程实现并发，适当提高温度增加搜索宽度
                    raw_outputs = self.llm.generate(
                        prompt=prompt,
                        n=n_candidates,
                        temperature=min(1.0, self.config['generation']['temperature'] + 0.2),
                        max_tokens=self.config['generation']['max_tokens'],
                        system_message=self.prompt_builder.SYSTEM_MESSAGE
                    )

                    # 【新增埋点2】拆解并发返回的数组，按候选ID逐个保存
                    for idx, out in enumerate(raw_outputs):
                        self._save_llm_interaction(generation, "Phase2_Initial_Generation", prompt, out, candidate_id=str(idx))

                else:
                    # ==========================================
                    # 阶段三：基于 DREAM 算子执行自适应并行突变
                    # ==========================================
                    from collections import Counter

                    print(f"🧬 执行 DREAM 自适应变异... (尝试 {attempt + 1}/{max_retries})")

                    try:
                        parent_code = self.memory.get_best_code(generation - 1)
                        reflection = self.memory.get_reflection(generation - 1)
                    except Exception as e:
                        print(f"⚠️ 无法获取父代数据，回退到 Zero-Shot: {e}")
                        prompt = self.prompt_builder.initial_generation_prompt_with_predefined_context(
                            self.task_description,
                            self.env_context
                        )
                        raw_outputs = self.llm.generate(
                            prompt=prompt,
                            n=n_candidates,
                            temperature=self.config['generation']['temperature'],
                            max_tokens=self.config['generation']['max_tokens'],
                            system_message=self.prompt_builder.SYSTEM_MESSAGE
                        )
                        continue  # 直接进入下一轮重试，避免访问未定义的 reflection

                    # ---------------------------------------------------
                    # 核心解析逻辑：从反思日志中提取算子分配并执行基数校验
                    # ---------------------------------------------------
                    assigned_operators = []
                    # 使用正则匹配 Candidate X: F1 格式
                    matches = re.findall(r'Candidate \d+:\s*(F1|F2|F3|L1)', reflection, re.IGNORECASE)

                    if len(matches) >= n_candidates:
                        assigned_operators = [m.upper() for m in matches[:n_candidates]]
                        # 基数约束校验：每种算子最多2次
                        counts = Counter(assigned_operators)
                        is_valid = all(v <= 2 for v in counts.values())

                        if is_valid:
                            print(f"   ✅ 成功解析自适应算子: {assigned_operators} (约束校验通过)")
                        else:
                            print(f"   ⚠️ LLM违反基数约束 {dict(counts)}，触发强制纠正。")
                            assigned_operators = []  # 触发兜底��制
                    else:
                        print(f"   ⚠️ 未能在反思中解析到规范的算子分配，触发兜底分配。")

                    # 兜底机制：恢复标准波束搜索
                    if not assigned_operators:
                        operators = ['F1', 'F2', 'F3', 'L1']
                        assigned_operators = [operators[i % 4] for i in range(n_candidates)]
                        print(f"   🔧 启用标准正交分配: {assigned_operators}")
                    # ---------------------------------------------------

                    def _call_evoleap(op_type: str, worker_idx: int) -> str:
                        """单线程执行单一突变算子"""
                        prompt = self.prompt_builder.evoleap_prompt(
                            operator_type=op_type,
                            task_description=self.task_description,
                            parent_code=parent_code,
                            reflection=reflection,
                            env_context=self.env_context
                        )
                        res = self.llm.generate(
                            prompt=prompt,
                            n=1,
                            temperature=self.config['generation']['temperature'],
                            max_tokens=self.config['generation']['max_tokens'],
                            system_message=self.prompt_builder.SYSTEM_MESSAGE
                        )
                        response_text = res[0] if res else ""

                        # 落盘拦截
                        if hasattr(self, '_save_llm_interaction'):
                            self._save_llm_interaction(
                                generation, "Phase3_DREAM", prompt, response_text,
                                candidate_id=f"worker{worker_idx}_{op_type}"
                            )
                        return response_text

                    print(f"   启动 {n_candidates} 个并发自适应变异线程...")

                    # 并发执行
                    raw_outputs = []
                    with concurrent.futures.ThreadPoolExecutor(max_workers=n_candidates) as executor:
                        future_to_op = {executor.submit(_call_evoleap, op, idx): op for idx, op in enumerate(assigned_operators)}
                        for future in concurrent.futures.as_completed(future_to_op):
                            op = future_to_op[future]
                            try:
                                result = future.result()
                                if result:
                                    raw_outputs.append(result)
                            except Exception as exc:
                                print(f"   ❌ 算子 {op} 执行出错: {exc}")

                        # 若并发获取结果过少，抛出异常进入重试
                        if len(raw_outputs) < min_valid_codes:
                            raise RuntimeError(f"DREAM自适应变异失败，仅获取到 {len(raw_outputs)} 份代码")

                # 统一解析与语法检查
                codes = self._parse_code_blocks(raw_outputs)
                valid_codes = []
                for i, code in enumerate(codes):
                    if self._syntax_check(code):
                        valid_codes.append(code)
                        print(f"  ✅ 候选 {i}: 语法检查通过")
                    else:
                        print(f"  ❌ 候选 {i}: 语法错误，已跳过")

                if len(valid_codes) >= min_valid_codes:
                    print(f"\n✅ 成功生成 {len(valid_codes)} 个有效候选")
                    return valid_codes[:n_candidates]
                else:
                    print(f"\n⚠️ 有效代码不足（{len(valid_codes)}/{min_valid_codes}），重新生成...")

            except Exception as e:
                print(f"\n❌ 生成失败: {e}")
                import traceback
                traceback.print_exc()
                if attempt < max_retries - 1:
                    print(f"   重试中...")

        print("\n⚠️ 所有尝试都失败，触发后备方案...")
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
    
    def analyze_results(self, results: List[Dict]) -> Tuple[str, str, float]:
        """
        分析训练结果并选择最优模板（实装降级选拔机制）

        Args:
            results: 训练结果列表

        Returns:
            Tuple[str, str, float]: (最优代码, 反思内容, 过滤后的best_fitness)
        """
        print(f"\n{'=' * 80}")
        print("🔍 评估过滤与最优策略选拔")
        print(f"{'=' * 80}")

        valid_results = [r for r in results if r.get('status') == 'success']
        if not valid_results:
            print("⚠️ 灾难性错误：所有候选执行崩溃，使用后备方案...")
            return results[0]['code'], "所有候选训练时发生运行时错误，请检查代码基本逻辑。", 0.0

        # =========================================================
        # 降级选拔算法 (Degradation Selection Algorithm)
        # =========================================================
        best_result = None

        # 优先级1：完全通过拦截器 (f_mean ∩ f_std ∩ f_slope)
        fully_converged = [
            r for r in valid_results
            if r.get('metrics', {}).get('convergence_status', {}).get('is_converged', False)
        ]

        if fully_converged:
            print("✅ 选拔优先级 1：存在完全收敛的奖励函数代码")
            best_result = max(fully_converged, key=lambda x: x.get('fitness', -float('inf')))
        else:
            # 优先级2：放宽条件，放弃对波动方差(f_std)的要求，只要趋势上升(f_mean ∩ f_slope)
            trend_converged = [
                r for r in valid_results
                if r.get('metrics', {}).get('convergence_status', {}).get('f_mean', False)
                and r.get('metrics', {}).get('convergence_status', {}).get('f_slope', False)
            ]
            if trend_converged:
                print("⚠️ 选拔优先级 2：无完全收敛代码，降级选取趋势上升的代码")
                best_result = max(trend_converged, key=lambda x: x.get('fitness', -float('inf')))
            else:
                # 优先级3：彻底失败，取最高适应度（盲目相信Max）
                print("🚨 选拔优先级 3：本代候选全军覆没(均未收敛)，降级选取绝对适应度最高者进行突变")
                best_result = max(valid_results, key=lambda x: x.get('fitness', -float('inf')))

        conv_stats = best_result.get('metrics', {}).get('convergence_status', {})
        print(f"👑 最优候选 ID: {best_result['id']}")
        print(f"   标量Fitness: {best_result['fitness']:.4f}")
        print(f"   收敛性诊断: {conv_stats.get('details', 'N/A')}")
        print(f"   完全收敛状态: {'✅' if conv_stats.get('is_converged') else '❌'}")

        logs_summary = self._format_logs([best_result])

        # 3. 调用LLM生成Reflection（带重试机制）
        print("\n🤔 LLM正在生成反思...")
        prompt = self.prompt_builder.reflection_prompt(logs_summary)

        max_retries = 3
        reflection = None
        last_error = None

        for attempt in range(max_retries):
            try:
                print(f"   尝试 {attempt + 1}/{max_retries}...")
                reflection = self.llm.analyze(
                    prompt=prompt,
                    temperature=self.config['reflection']['temperature'],
                    max_tokens=self.config['reflection']['max_tokens']
                )

                # 检查返回是否有效
                if reflection and reflection.strip():
                    print(f"   ✅ 反思生成成功")
                    # 【新增埋点4】反思生成成功后保存
                    self._save_llm_interaction(self.current_generation, "Phase4_Reflection", prompt, reflection, candidate_id="best_code_diagnosis")
                    break
                else:
                    print(f"   ⚠️ 返回为空，继续重试...")
                    reflection = None

            except Exception as e:
                last_error = e
                print(f"   ❌ 第 {attempt + 1} 次失败: {e}")

        # 4. 如果重试全部失败，保存提示词到文件
        if not reflection or not reflection.strip():
            print(f"\n❌ 反思生成全部失败（{max_retries}次）")
            print("   正在保存提示词到文件...")

            # 生成文件名（包含时间戳和代数）
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            save_filename = f"failed_reflection_prompt_g{self.current_generation}_{timestamp}.txt"
            save_path = os.path.join(self.config['logging']['save_dir'], save_filename)

            # 确保目录存在
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            # 保存提示词内容
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write(f"反思生成失败 - 第 {self.current_generation} 代\n")
                f.write(f"时间: {datetime.datetime.now().isoformat()}\n")
                f.write(f"重试次数: {max_retries}\n")
                f.write(f"最后错误: {last_error}\n")
                f.write("=" * 80 + "\n\n")
                f.write("【发送给LLM的完整提示词】\n")
                f.write("=" * 80 + "\n")
                f.write(prompt)
                f.write("\n\n【训练结果摘要】\n")
                f.write("=" * 80 + "\n")
                f.write(logs_summary)

            print(f"   ✅ 已保存到: {save_path}")

            # 不再使用默认反思，而是抛出异常向上传递
            raise RuntimeError(
                f"反思生成失败（已重试{max_retries}次）\n"
                f"最后错误: {last_error}\n"
                f"提示词已保存到: {save_path}"
            )

        print(f"\n📊 反思内容（前200字符）:")
        print(reflection[:200] + "..." if len(reflection) > 200 else reflection)

        # 【重点修改】：必须返回 best_result['fitness']，而不能让外部去重新瞎算 max()
        return best_result['code'], reflection, best_result['fitness']
    
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
            results = self._simulate_training(codes)
        
        # 3. 分析结果 【重点修改：接收三个返回值】
        best_code, reflection, best_fitness = self.analyze_results(results)

        # 4. 更新记忆 【重点修改：显式传入 selected_fitness】
        self.memory.save(
            generation=generation,
            best_code=best_code,
            reflection=reflection,
            all_results=results,
            selected_fitness=best_fitness
        )
        
        return {
            'generation': generation,
            'best_code': best_code,
            'best_fitness': best_fitness,  # 这里记录的是降级选拔后的真实性能
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
        """模拟训练结果（注入收敛特征用于测试）"""
        import random

        print("⚠️ 使用模拟训练数据")

        results = []
        for i, code in enumerate(codes):
            success_rate = random.uniform(0.6, 0.9)
            avg_time = random.uniform(40, 60)
            fitness = success_rate - 0.001 * avg_time

            is_converged = random.choice([True, False])
            f_mean = is_converged or random.choice([True, False])

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
                    },
                    'convergence_status': {
                        'f_mean': f_mean,
                        'f_std': is_converged,
                        'f_slope': f_mean,
                        'is_converged': is_converged,
                        'details': f"Mocked data - Converged: {is_converged}"
                    }
                }
            }

            results.append(result)
            print(f"  候选 {i}: Fitness={fitness:.4f}, 成功率={success_rate:.2%}, "
                  f"收敛={is_converged}")

        return results
