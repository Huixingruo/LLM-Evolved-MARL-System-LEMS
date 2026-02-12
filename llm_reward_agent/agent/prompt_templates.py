"""
提示词模板库
包含LLM生成奖励函数所需的所有提示词模板

Author: LEMS Project
Date: 2026-02-03
Version: 1.0
"""

from typing import Dict, Any


class PromptTemplates:
    """LLM提示词模板库"""
    
    # 系统提示词
    SYSTEM_MESSAGE = """你是一位专业的强化学习奖励工程师，擅长为多智能体强化学习任务设计奖励函数。

你的任务是根据环境信息和任务描述，设计Python代码形式的奖励函数。

关键要求：
1. 代码必须符合PEP8规范
2. 只输出Python代码，不要有任何额外解释
3. 考虑多智能体协同的特殊性（避免碰撞、形成队形等）
4. 奖励分量要可解释、可调试
"""
    
    @staticmethod
    def initial_generation_prompt(env_context: Dict, task_description: str) -> str:
        """
        第一代生成提示词（Zero-Shot）
        
        Args:
            env_context: 环境上下文信息
            task_description: 任务描述
        
        Returns:
            str: 完整提示词
        """
        prompt = f"""你是一位专业的强化学习奖励工程师。请根据以下信息设计奖励函数。

# 任务描述
{task_description}

# 环境信息
- **环境名称**: {env_context.get('env_name', '未知')}
- **观测空间**: {env_context.get('observation_space', '未知')}
- **动作空间**: {env_context.get('action_space', '未知')}

## 智能体信息
"""
        
        # 添加智能体信息
        agent_info = env_context.get('agent_info', {})
        for key, val in agent_info.items():
            prompt += f"- {key}: {val}\n"
        
        prompt += "\n## 物理参数\n"
        
        # 添加物理常量
        constants = env_context.get('physical_constants', {})
        for key, val in constants.items():
            prompt += f"- {key}: {val}\n"
        
        # 添加代码片段（如果有）
        code_snippet = env_context.get('code_snippet', '')
        if code_snippet:
            prompt += f"""
## 环境代码片段（关键部分）
```python
{code_snippet}
```
"""
        
        # 添加函数签名和要求
        prompt += """
# 要求

请实现 `compute_reward()` 函数，函数签名如下：

```python
def compute_reward(agent_name, observation, global_state, actions, world):
    \"\"\"
    可插拔的奖励函数接口
    
    Args:
        agent_name (str): 当前智能体名称 (e.g., "adversary_0", "agent_0")
        observation (np.ndarray): 当前智能体的观测向量
        global_state (dict): 全局状态信息
            {
                'agent_positions': np.ndarray,       # shape: (n_agents, 2)
                'agent_velocities': np.ndarray,      # shape: (n_agents, 2)
                'prey_position': np.ndarray,         # shape: (2,)
                'prey_velocity': np.ndarray,         # shape: (2,)
                'distances_to_prey': np.ndarray,     # shape: (n_adversaries,)
                'inter_agent_distances': np.ndarray, # shape: (n_agents, n_agents)
                'is_adversary': bool,                # 当前智能体是否为追捕者
                'adversary_indices': list,           # 所有追捕者的索引
                'prey_indices': list,                # 所有逃跑者的索引
                'world_size': float,                 # 世界大小
                'capture_threshold': float,          # 围捕阈值
            }
        actions (dict): 所有智能体的动作 {agent_name: action_vector}
        world (World): PettingZoo的World对象
    
    Returns:
        reward (float): 标量奖励值
        components (dict): 奖励分量字典（用于日志分析）
    \"\"\"
    # 你的代码实现
    pass
```

## 设计要点

1. **奖励分量必须是字典**，包含各个奖励分量（用于日志分析），例如：
   ```python
   components = {
       'distance_reward': ..., 
       'collision_penalty': ..., 
       'formation_reward': ...,
       'boundary_penalty': ...
   }
   ```

2. **只设计追捕者的奖励函数**（逃跑者的奖励函数已固定）：
   - 在函数内部通过 `if global_state['is_adversary']` 判断当前是否为追捕者
   - 如果不是追捕者（逃跑者），返回0或固定的小奖励

3. **追捕者奖励函数设计要点**：
   - ✅ 鼓励智能体接近并围捕目标
   - ✅ 惩罚智能体之间的碰撞
   - ✅ 奖励形成均匀的包围圈
   - ✅ 惩罚越界行为
   - ✅ 适当惩罚能耗（可选）

4. **只输出Python代码**，不要有任何额外解释。代码需要符合PEP8规范。

5. **必须导入numpy**: 在代码开头加上 `import numpy as np`

# 输出格式

只输出完整的Python代码，使用以下格式：

```python
import numpy as np

def compute_reward(agent_name, observation, global_state, actions, world):
    components = {{}}
    
    # 你的代码实现
    # ...
    
    total_reward = sum(components.values())
    return total_reward, components
```

**注意**：不要输出任何解释性文字，只输出代码块。
"""
        
        return prompt
    
    @staticmethod
    def evolution_prompt(env_context: Dict, 
                        task_description: str, 
                        parent_code: str, 
                        reflection: str,
                        n_candidates: int = 4) -> str:
        """
        进化生成提示词（基于上一代的改进）
        
        Args:
            env_context: 环境上下文信息
            task_description: 任务描述
            parent_code: 上一代最优代码
            reflection: 上一代的训练反思
            n_candidates: 需要生成的候选数量
        
        Returns:
            str: 完整提示词
        """
        prompt = f"""你是一位专业的强化学习奖励工程师。基于上一代的训练反馈，改进奖励函数。

# 任务描述
{task_description}

# 环境信息（简要）
- 观测空间: {env_context.get('observation_space', '未知')}
- 动作空间: {env_context.get('action_space', '未知')}
- 智能体数量: {env_context.get('agent_info', {}).get('num_adversaries', '未知')} 个追捕者, {env_context.get('agent_info', {}).get('num_good', '未知')} 个逃跑者

# 上一代最优代码

```python
{parent_code}
```

# 训练反馈与反思

{reflection}

# 改进要求

基于上述反思，对代码进行修改，生成 **{n_candidates} 个不同的变体（Mutation）**。

每个变体应该：
1. 基于上一代代码进行改进
2. 针对反思中提到的问题进行调整：
   - 调整权重系数（增大/减小某些分量的权重）
   - 增加/删除奖励项
   - 改变函数形式（线性 → 指数 / 分段函数 / 势场函数等）
3. 每个变体应有明显差异（不要只是微调权重）
4. 保持代码的可读性和可解释性

## 变体生成策略建议

- **变体0**: 保守改进（微调权重，保持结构）
- **变体1**: 激进改进（大幅调整权重或函数形式）
- **变体2**: 添加新的奖励分量
- **变体3**: 简化奖励函数（删除不重要的分量）

# 输出格式

输出 {n_candidates} 个完整的Python代码块，每个代码块之间用注释 `# === VARIANT {{i}} ===` 分隔。

示例：

```python
# === VARIANT 0 ===
import numpy as np

def compute_reward(agent_name, observation, global_state, actions, world):
    components = {{}}
    # ... 变体0的实现 ...
    total_reward = sum(components.values())
    return total_reward, components

# === VARIANT 1 ===
import numpy as np

def compute_reward(agent_name, observation, global_state, actions, world):
    components = {{}}
    # ... 变体1的实现 ...
    total_reward = sum(components.values())
    return total_reward, components

# === VARIANT 2 ===
import numpy as np

def compute_reward(agent_name, observation, global_state, actions, world):
    components = {{}}
    # ... 变体2的实现 ...
    total_reward = sum(components.values())
    return total_reward, components

# === VARIANT 3 ===
import numpy as np

def compute_reward(agent_name, observation, global_state, actions, world):
    components = {{}}
    # ... 变体3的实现 ...
    total_reward = sum(components.values())
    return total_reward, components
```

**注意**：
1. 只输出代码，不要有任何额外解释
2. 每个变体必须是完整的、可独立运行的函数
3. 确保代码语法正确，符合PEP8规范
"""
        
        return prompt
    
    @staticmethod
    def reflection_prompt(training_logs: str) -> str:
        """
        生成反思提示词（Reward Reflection）
        
        Args:
            training_logs: 格式化后的训练日志
        
        Returns:
            str: 反思提示词
        """
        prompt = f"""你是一位专业的强化学习研究员。请分析以下训练日志，总结奖励函数的表现。

# 训练日志

{training_logs}

# 分析要求

请从以下四个维度进行分析：

## 1. 奖励分量诊断

- 哪些分量起到了作用（数值非零且有方差）？
- 哪些分量失效了（一直为0或数值异常）？
- 各分量的数值范围是否合理？
- 各分量的权重配比是否平衡？

## 2. 任务性能分析

- 成功率是否达标（目标：>0.8）？
- 捕获时间是否合理（越短越好）？
- 是否存在明显的失败模式？
- 训练是否收敛（从日志趋势判断）？

## 3. 协同行为评估

- 智能体是否学会了均匀包围（从角度标准差判断）？
- 是否出现碰撞或扎堆现象（从最小距离判断）？
- 队形质量是否提高？
- 是否出现"搭便车"现象（某些智能体不工作）？

## 4. 改进建议（针对下一代）

基于以上分析，提出具体的改进措施：

- 需要增加哪些奖励项？（用于引导缺失的行为）
- 需要删除哪些奖励项？（失效或干扰的分量）
- 需要调整哪些权重系数？（增大/减小具体数值）
- 需要改变哪些函数形式？（线性→非线性，添加阈值等）

# 输出格式

使用自然语言，分点总结，清晰明了。总字数控制在500字以内。

示例输出：

**奖励分量诊断**：
- distance_reward工作正常，均值-1.23，引导智能体接近目标
- collision_penalty基本失效，均值接近0，说明智能体很少碰撞或惩罚不够明显
- formation_reward方差很小，可能权重过低或触发条件过严

**任务性能分析**：
- 成功率75%，尚未达标，需要进一步优化
- 平均捕获时间48步，处于中等水平
- 主要失败模式：智能体在圈外徘徊，不敢进圈

**协同行为评估**：
- 角度标准差0.34，包围不够均匀，存在扎堆现象
- 最小智能体距离0.25，接近碰撞阈值，需要更强的排斥力
- 队形质量0.56，有改进空间

**改进建议**：
1. 增大进圈奖励的权重（当前-2.0 → -3.0），鼓励更激进的接近
2. 降低角度排斥的权重（当前5.0 → 2.0），避免过度排斥导致不敢进圈
3. 添加"全员进圈"的额外奖励，强化协同
4. 调整碰撞惩罚的阈值（当前0.2 → 0.3），给予更多容错空间
"""
        
        return prompt
    
    @staticmethod
    def code_fix_prompt(broken_code: str, error_message: str) -> str:
        """
        代码修复提示词
        
        Args:
            broken_code: 有问题的代码
            error_message: 错误信息
        
        Returns:
            str: 修复提示词
        """
        prompt = f"""请修复以下Python代码中的错误。

# 错误的代码

```python
{broken_code}
```

# 错误信息

```
{error_message}
```

# 要求

1. 修复代码中的语法错误或逻辑错误
2. 保持代码的原有功能和逻辑
3. 确保符合函数签名要求
4. 只输出修复后的完整代码，不要有任何解释

# 输出格式

```python
import numpy as np

def compute_reward(agent_name, observation, global_state, actions, world):
    # 修复后的代码
    pass
```
"""
        return prompt


# ========================================
# 测试代码
# ========================================
if __name__ == "__main__":
    print("测试提示词模板...")
    
    # 构造测试环境上下文
    test_env_context = {
        'env_name': 'simple_tag_env',
        'observation_space': 'Box(16,)',
        'action_space': 'Box(2,)',
        'agent_info': {
            'num_adversaries': 3,
            'num_good': 1,
            'num_obstacles': 0
        },
        'physical_constants': {
            'max_force': 1.0,
            'capture_threshold': 0.5,
            'world_size': 2.5
        },
        'code_snippet': '# 环境代码片段...\n# ...'
    }
    
    test_task_description = """
任务：3个追捕智能体协同围捕1个逃逸目标。

要求：
1. 追捕者需要接近并包围目标
2. 追捕者之间避免碰撞
3. 形成均匀的包围圈
4. 在尽可能短的时间内完成围捕

注意：逃跑者的奖励函数已固定，不需要重新设计。
只需设计追捕者的奖励函数。
    """
    
    # 测试初始生成提示词
    print("\n=== 测试初始生成提示词 ===")
    initial_prompt = PromptTemplates.initial_generation_prompt(
        test_env_context,
        test_task_description
    )
    print(f"提示词长度: {len(initial_prompt)} 字符")
    print(f"前500字符:\n{initial_prompt[:500]}...")
    
    # 测试进化提示词
    print("\n=== 测试进化提示词 ===")
    test_parent_code = """
import numpy as np

def compute_reward(agent_name, observation, global_state, actions, world):
    components = {}
    dist = np.linalg.norm(global_state['agent_positions'][0] - global_state['prey_position'])
    components['distance_reward'] = -dist
    total_reward = sum(components.values())
    return total_reward, components
"""
    
    test_reflection = """
成功率较低（60%），主要问题是智能体扎堆。
建议：增加角度排斥力，鼓励均匀分布。
"""
    
    evolution_prompt = PromptTemplates.evolution_prompt(
        test_env_context,
        test_task_description,
        test_parent_code,
        test_reflection,
        n_candidates=4
    )
    print(f"提示词长度: {len(evolution_prompt)} 字符")
    print(f"前500字符:\n{evolution_prompt[:500]}...")
    
    # 测试反思提示词
    print("\n=== 测试反思提示词 ===")
    test_logs = """
Candidate 0: 成功率 0.75, 平均时间 48 步
  - distance_reward: -1.23 ± 0.45
  - collision_penalty: -0.05 ± 0.12
  - formation_reward: 0.34 ± 0.08

Candidate 1: 成功率 0.68, 平均时间 52 步
  - distance_reward: -1.45 ± 0.52
  - collision_penalty: -0.15 ± 0.25
  - formation_reward: 0.21 ± 0.06
"""
    
    reflection_prompt = PromptTemplates.reflection_prompt(test_logs)
    print(f"提示词长度: {len(reflection_prompt)} 字符")
    print(f"前500字符:\n{reflection_prompt[:500]}...")
    
    print("\n✅ 提示词模板测试完成！")
