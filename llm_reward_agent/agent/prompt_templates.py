"""
提示词模板库
包含LLM生成奖励函数所需的所有提示词模板
"""

from typing import Dict, Any


# ========================================
# 预定义的环境上下文（从 env_context_output.txt 提取）
# ========================================

PREDEFINED_ENV_CONTEXT = {
    "env_name": "simple_tag_env_no_obstacles",
    "agent_info": {
        "num_adversaries": 3,
        "num_good": 1,
        "num_obstacles": 0
    },
    "code_snippet": '''
import numpy as np

class CoreEnvLogic:
    """
    环境核心逻辑
    用于辅助设计 Reward Function
    """
    def __init__(self):
        # 核心物理常量
        self.world_size = 2.5          # 地图范围 (-2.5, 2.5)
        self.max_force = 1.0             # 动作最大值
        self.capture_threshold = 0.5     # 围捕判定阈值 (world_size * 0.2)
        
        # 智能体参数
        # size: 碰撞体积半径
        # max_speed: 逃跑者 1.3 > 追捕者 1.0
        self.adversary_params = {'size': 0.075, 'max_speed': 1.0}
        self.agent_params = {'size': 0.050, 'max_speed': 1.3}

    def is_collision(self, agent1_pos, agent1_size, agent2_pos, agent2_size):
        """碰撞检测：欧氏距离 < 半径之和"""
        delta_pos = agent1_pos - agent2_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1_size + agent2_size
        return dist < dist_min

    def _build_global_state(self, agent, world):
        """
        【重要】传给 compute_reward 的 global_state 结构
        """
        all_agents = world.agents
        adversaries = [a for a in all_agents if a.adversary]
        preys = [a for a in all_agents if not a.adversary]
        
        agent_positions = np.array([a.state.p_pos for a in all_agents])
        agent_velocities = np.array([a.state.p_vel for a in all_agents])
        prey_pos = preys[0].state.p_pos if preys else np.zeros(2)
        prey_vel = preys[0].state.p_vel if preys else np.zeros(2)
        
        # 每个追捕者到猎物的距离
        distances_to_prey = np.array([np.linalg.norm(adv.state.p_pos - prey_pos) for adv in adversaries])
        
        # 智能体间距离矩阵 (用于防撞)
        n_agents = len(all_agents)
        inter_agent_distances = np.zeros((n_agents, n_agents))
        for i in range(n_agents):
            for j in range(n_agents):
                inter_agent_distances[i][j] = np.linalg.norm(agent_positions[i] - agent_positions[j])

        return {
            'agent_positions': agent_positions,
            'agent_velocities': agent_velocities,
            'prey_position': prey_pos,
            'prey_velocity': prey_vel,
            'distances_to_prey': distances_to_prey,
            'inter_agent_distances': inter_agent_distances,
            'is_adversary': agent.adversary,
            'world_size': self.world_size,
            'capture_threshold': self.capture_threshold
        }

    def observation(self, agent, world):
        """
        【重要】观测向量结构
        Return: np.concatenate([norm_self_vel] + [norm_self_pos] + other_pos + other_vel)
        """
        # 自身状态
        norm_self_vel = agent.state.p_vel / agent.max_speed
        norm_self_pos = agent.state.p_pos / self.world_size
        
        # 其他智能体相对位置
        other_pos = []
        other_vel = []
        for other in world.agents:
            if other is agent: continue
            rel_pos = (other.state.p_pos - agent.state.p_pos) / self.world_size
            other_pos.append(rel_pos)
            if not other.adversary:
                other_vel.append(other.state.p_vel / other.max_speed)
        
        return np.concatenate([norm_self_vel] + [norm_self_pos] + other_pos + other_vel)
'''}


# 预定义的任务描述
PREDEFINED_TASK_DESCRIPTION = """设计追捕者(adversary)的奖励函数：
- **目标**: 3个追捕者协同围捕1个逃跑者
- **成功条件**: 所有追捕者进入逃跑者的capture_threshold范围内
- **关键**: 设计协作策略，使追捕者形成包围圈

逃跑者的奖励函数已固定，不需要重新设计。"""


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
        actions (dict): 所有智能体的动作 {{agent_name: action_vector}}
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
   - 如果不是追捕者（逃跑者），返回0

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
    # 使用预定义环境上下文的方法
    # ========================================
    
    @staticmethod
    def initial_generation_prompt_with_predefined_context(
        task_description: str = PREDEFINED_TASK_DESCRIPTION,
        env_context: Dict = None
    ) -> str:
        """
        第一代生成提示词（使用预定义环境上下文）
        
        Args:
            task_description: 任务描述（默认为 PREDEFINED_TASK_DESCRIPTION）
            env_context: 环境上下文（默认为 PREDEFINED_ENV_CONTEXT）
        
        Returns:
            str: 完整提示词
        """
        if env_context is None:
            env_context = PREDEFINED_ENV_CONTEXT
        
        agent_info = env_context.get('agent_info', {})
        code_snippet = env_context.get('code_snippet', '')
        
        prompt = f"""你是一位专业的强化学习奖励工程师。请根据以下信息设计奖励函数。

# 任务描述
{task_description}

# 环境配置

## 智能体信息
- 追捕者 (adversary): {agent_info.get('num_adversaries', 3)} 个
- 逃跑者 (agent): {agent_info.get('num_good', 1)} 个
- 障碍物 (obstacles): {agent_info.get('num_obstacles', 0)} 个


## 核心环境逻辑

```python
{code_snippet}
```

# 要求

请实现 compute_reward 函数，函数签名如下：

```
def compute_reward(agent_name, observation, global_state, actions, world):
    \"\"\"
    可插拔的奖励函数接口
    
    Args:
        agent_name (str): 当前智能体名称 (e.g., "adversary_0", "agent_0")
        observation (np.ndarray): 当前智能体的观测向量
        global_state (dict): 全局状态信息
            - 'agent_positions': np.ndarray,       # shape: (n_agents, 2)
            - 'agent_velocities': np.ndarray,      # shape: (n_agents, 2)
            - 'prey_position': np.ndarray,         # shape: (2,)
            - 'prey_velocity': np.ndarray,         # shape: (2,)
            - 'distances_to_prey': np.ndarray,     # shape: (n_adversaries,)
            - 'inter_agent_distances': np.ndarray, # shape: (n_agents, n_agents)
            - 'is_adversary': bool,                # 当前智能体是否为追捕者
            - 'adversary_indices': list,           # 所有追捕者的索引
            - 'prey_indices': list,                # 所有逃跑者的索引
            - 'world_size': float,                 # 世界大小
            - 'capture_threshold': float,          # 围捕阈值
        actions (dict): 所有智能体的动作 {{agent_name: action_vector}}
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
   components = {{
       'distance_reward': ..., 
       'collision_penalty': ..., 
       'formation_reward': ...,
       'boundary_penalty': ...
   }}
   ```

2. **只设计追捕者的奖励函数**（逃跑者的奖励函数已固定）：
   - 在函数内部通过 `if global_state['is_adversary']` 判断当前是否为追捕者
   - 如果不是追捕者（逃跑者），返回0或固定的小奖励

3. **追捕者奖励函数设计要点**：
   - 鼓励智能体接近并围捕目标
   - 惩罚智能体之间的碰撞
   - 奖励形成均匀的包围圈
   - 惩罚越界行为
   - 适当惩罚能耗（可选）

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

注意：不要输出任何解释性文字，只输出代码块。
"""
        
        return prompt
    
    @staticmethod
    def evolution_prompt_with_predefined_context(
        task_description: str = PREDEFINED_TASK_DESCRIPTION,
        parent_code: str = "",
        reflection: str = "",
        env_context: Dict = None
    ) -> str:
        """
        进化生成提示词（使用预定义环境上下文）

        Args:
            task_description: 任务描述（默认为 PREDEFINED_TASK_DESCRIPTION）
            parent_code: 上一代最优代码
            reflection: 上一代的训练反思
            env_context: 环境上下文（默认为 PREDEFINED_ENV_CONTEXT）

        Returns:
            str: 完整提示词
        """
        if env_context is None:
            env_context = PREDEFINED_ENV_CONTEXT

        # 获取环境代码片段
        code_snippet = env_context.get('code_snippet', '')
        agent_info = env_context.get('agent_info', {})

        prompt = f"""你是一位专业的强化学习奖励工程师。基于上一代的训练反馈，改进奖励函数。

# 任务描述
{task_description}

# 环境配置

## 核心环境逻辑
请严格遵守以下环境定义中的变量名称：

```python
{code_snippet}
```

# 上一代最优代码

```python
{parent_code}
```

# 训练反馈与反思

{reflection}

# 改进要求

基于上述反思，请生成 1 个 改进后的奖励函数代码。

要求：
1. 针对性优化：直接解决反思中提到的问题（如权重不合理、缺少某项奖励）。
2. 正确性：确保使用的变量在 global_state 或 observation 中真实存在。
3. 探索性：你可以调整权重、修改计算公式或增加新的奖励项。

# 输出格式

只输出 1 个 完整的 Python 函数代码，不要包含任何 Markdown 标记（如 ```python），也不要包含任何解释文字。

```python
import numpy as np

def compute_reward(agent_name, observation, global_state, actions, world):
    components = {{}}
    # 你的代码实现
    # ...
    total_reward = sum(components.values())
    return total_reward, components
```
"""

        return prompt


# ========================================
# 测试代码
# ========================================
if __name__ == "__main__":
    print("测试提示词模板...")
    
    # 测试使用预定义上下文的方法
    print("\n=== 测试使用预定义上下文的方法 ===")
    predefined_prompt = PromptTemplates.initial_generation_prompt_with_predefined_context()
    print(f"提示词长度: {len(predefined_prompt)} 字符")
    print(f"前800字符:\n{predefined_prompt[:800]}...")
    print("\n" + "="*80)
    
    # 测试进化提示词
    test_parent_code = """
import numpy as np

def compute_reward(agent_name, observation, global_state, actions, world):
    components = {{}}
    dist = np.linalg.norm(global_state['agent_positions'][0] - global_state['prey_position'])
    components['distance_reward'] = -dist
    total_reward = sum(components.values())
    return total_reward, components
"""
    
    test_reflection = """
成功率较低（60%），主要问题是智能体扎堆。
建议：增加角度排斥力，鼓励均匀分布。
"""
    
    print("\n=== 测试进化提示词（使用预定义上下文）===")
    evolution_prompt = PromptTemplates.evolution_prompt_with_predefined_context(
        parent_code=test_parent_code,
        reflection=test_reflection,
        n_candidates=4
    )
    print(f"提示词长度: {len(evolution_prompt)} 字符")
    print(f"前800字符:\n{evolution_prompt[:800]}...")
    
    print("\n" + "="*80)
    print("✅ 预定义上下文提示词测试完成！")
    print("="*80)
