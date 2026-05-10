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
import gymnasium
from gymnasium.utils import EzPickle

from pettingzoo.mpe._mpe_utils.core import Agent, Landmark, World
from pettingzoo.mpe._mpe_utils.scenario import BaseScenario
from pettingzoo.mpe._mpe_utils.simple_env import SimpleEnv, make_env
from pettingzoo.utils.conversions import parallel_wrapper_fn

from .custom_agents_dynamics import CustomWorld
from . import reward_function  # 可插拔的奖励函数（仅包含追捕者奖励）

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
    def reflection_prompt(training_logs: str) -> str:
        """
        DREAM 模块：带自适应算子分配的诊断反思提示词
        强制 LLM 在输出诊断后，严格按规定格式输出下一代的算子分配方案。
        """
        prompt = f"""你是一位严谨的强化学习数据分析师。请分析以下训练日志，给出诊断报告，并为下一代的4个并行候选制定进化算子。

# 训练日志
{training_logs}

# 可用变异算子说明（含适用场景）

## F1 (分支扩充): 增加新的缺失奖励/惩罚机制
适用场景：
- 诊断发现存在明显的行为缺陷，但现有奖励分量中找不到任何相关引导（如缺少防碰撞惩罚、缺少包围鼓励、缺少边界约束等）
- 需要引入全新的协同机制但又不想破坏现有有效设计
- 现有奖励分量覆盖维度不足（如缺少速度引导、缺少朝向引导等）

## F2 (分量重构): 重写当前失效或起反作用的奖励分量（改变其数学表达形式）
适用场景：
- 某分量统计均值长期为负或接近零，说明该设计在训练中未产生预期引导
- 某分量的设计在训练中产生了反向效果（如鼓励包围反而导致分散）
- 需要调整计算公式（线性→指数/双曲/高斯混合等）来改变稀疏性
- 需要改变分量的稀疏特性（如将密集奖励改为稀疏里程碑奖励）

## F3 (平衡微调): 仅修改现有各分量的权重系数
适用场景：
- 各分量设计本身基本合理，但权重配比失调（如成功奖励远小于失败惩罚）
- 希望增加探索性（降低某些惩罚权重）或强化某类行为（提升某分量权重）
- 训练中出现某种行为但需要微调强度而非改变方向
- 只想保守调整，不想引入过多风险

## L1 (范式跃迁): 彻底推翻重写
适用场景：
- 现有奖励设计从根本上是错误的（如完全忽略了围捕任务的核心目标）
- 多次迭代（F1/F2/F3）后仍无法收敛，说明当前范式有系统性问题
- 需要尝试完全不同的设计思路（如从局部贪婪改为全局势场）
- 历史积累的修补导致代码结构混乱，需要从零重建

# 任务要求
1. **病理诊断**：详细分析哪些分量起主导作用，哪些失效或起反作用，以及是否存在协同缺陷。
2. **算子分配**：基于诊断，为下一代的4个候选独立分配算子。
   - **硬性约束**：【每种算子最多只能被选择 2 次】。

# 算子分配示例

## 示例一：F2+F3 为主（各选2次）
```
[算子分配]
Candidate 0: F2
Candidate 1: F3
Candidate 2: F2
Candidate 3: F3
```
适用：当需要同时重构失效分量（F2）和调整权重（F3）时，让各候选侧重不同方向。

## 示例二：F1/F2/F3/L1 各1次
```
[算子分配]
Candidate 0: F1
Candidate 1: F2
Candidate 2: F3
Candidate 3: L1
```
适用：需要探索不同方向时，4种算子各选一个，最大限度保持多样性。

## 示例三：F1出现2次 + F2+F3各1次
```
[算子分配]
Candidate 0: F1
Candidate 1: F2
Candidate 2: F1
Candidate 3: F3
```
适用：当诊断指出缺少关键机制（F1），同时需要适度重构（F2）和微调（F3）时。

## 示例四：L1出现2次 + F2+F3各1次
```
[算子分配]
Candidate 0: L1
Candidate 1: F2
Candidate 2: L1
Candidate 3: F3
```
适用：当当前范式存在根本性问题时，激进候选选择L1重写，同时保留F2/F3保守调整。

## 示例五：F1+F2+F3各1次 + 某算子重复
```
[算子分配]
Candidate 0: F1
Candidate 1: F2
Candidate 2: F3
Candidate 3: F2
```
适用：允许同一种算子（如F2）在不同候选中产生不同变体，增加局部搜索深度。

# 强制输出格式（严格遵守，不要有多余文字）
[病理诊断]
(你的详细诊断内容)

[算子分配]
Candidate 0: <F1/F2/F3/L1>
Candidate 1: <F1/F2/F3/L1>
Candidate 2: <F1/F2/F3/L1>
Candidate 3: <F1/F2/F3/L1>"""
        return prompt

    # ========================================
    # 使用预定义环境上下文的方法
    # ========================================

    @staticmethod
    def cot_analysis_prompt(task_description: str, env_context: Dict) -> str:
        """
        阶段一：思维链（CoT）环境解析提示词
        强制LLM在生成代码前，对MDP模型和多智能体交互机理进行物理意义上的降维与对齐。
        """
        agent_info = env_context.get('agent_info', {})
        code_snippet = env_context.get('code_snippet', '')

        prompt = f"""你是一位资深多智能体强化学习（MARL）专家。在设计奖励函数前，你必须先对环境模型（Dec-POMDP）进行严格的代码级诊断。

# 任务描述
{task_description}

# 环境信息
追捕者数量: {agent_info.get('num_adversaries', 3)}, 逃跑者数量: {agent_info.get('num_good', 1)}

核心逻辑代码：
```python
{code_snippet}
```

# 分析任务（必须按要求分步作答）

请深呼吸，逐步回答以下五个维度的问题，以建立准确的环境状态表征：

## 实现细节 (Implementation Details)
- 环境代码使用了哪些依赖包？
- 是否引入了外部未定义的变量？

## 环境结构 (Environment Structure)
- 状态空间 (Global State): global_state 字典中包含了哪些维度的信息？它们的物理意义是什么？
- 观测空间 (Observation): 智能体的局部观测向量是如何拼接的？包含了哪些相对信息？

## 智能体交互 (Agent Interactions)
- 捕食者与猎物在物理属性（如 max_speed, size）上有何异同？
- 环境中判定"成功捕获（Collision）"的数学和物理条件是什么？

## 任务相关信息 (Task-relevant Information)
- 围捕任务的核心目标与哪些变量直接挂钩（正相关/负相关）？
- 怎样的空间拓扑（如智能体间的相对距离分布）代表形成了高质量的"包围圈"？

## 任务相关信息 (Task-relevant Information)
- 围捕任务的核心目标与哪些变量直接挂钩（正相关/负相关）？
- 怎样的空间拓扑（如智能体间的相对距离分布）代表形成了高质量的"包围圈"？

## API边界隔离 (API Boundaries)
- **重要**：上方代码片段中的 `CoreEnvLogic` 类仅是**文档伪代码**，用于辅助理解物理概念，**绝非**可以在运行时被实例化的真实类。
- 在真正的 `compute_reward` 函数中，**禁止**调用 `CoreEnvLogic()` 或访问 `world.logic`、`world.adversary_params` 等不存在的属性。
- 如果需要物理常量（如 `size`、`max_speed`、`world_size` 等），必须在函数内部以**局部变量**的形式硬编码声明，例如：`adv_size = 0.075`。

请仅输出上述五个维度的详细分析报告，**绝对不要编写任何奖励函数代码**。
"""
        return prompt

    @staticmethod
    def initial_generation_prompt_with_cot(
        task_description: str,
        env_context: Dict,
        cot_analysis_result: str
    ) -> str:
        """
        阶段二：基于CoT先验的独立生成提示词
        """
        code_snippet = env_context.get('code_snippet', '')

        prompt = f"""你是一位专业的强化学习奖励工程师。请基于之前的环境诊断分析，编写符合要求的奖励函数。

# 任务描述
{task_description}

# 先验环境诊断分析
{cot_analysis_result}

# 环境代码参考
```python
{code_snippet}
```

# 接口规范要求

请实现 compute_reward 函数，严格遵守以下签名与返回格式：

```python
import numpy as np

def compute_reward(agent_name, observation, global_state, actions, world):
    # 必须通过 global_state['is_adversary'] 过滤逃跑者
    if not global_state['is_adversary']:
        return 0.0, {{}}

    components = {{}}

    # 你的核心逻辑实现 (必须包含距离引导、防碰撞、包围队形等分量)
    # components['distance_reward'] = ...

    total_reward = sum(components.values())
    return total_reward, components
```

# 关键约束
- 奖励分量字典 components 必须保留，以便后续日志做信用分配分析
- 确保所使用的字典键值在 global_state 中真实存在
- **禁止实例化伪类**：绝对禁止在代码中写出 `CoreEnvLogic()`！上方的代码片段仅是背景文档，运行时环境中根本不存在这个类。
- **禁止虚构属性**：绝对禁止调用 `world.logic`、`world.adversary_params` 等不存在的属性。
- **物理常量硬编码**：如果需要使用物理参数（如智能体的 `size=0.075`、地图大小 `world_size=2.5` 等），必须直接在 `compute_reward` 函数内部以局部变量的形式硬编码声明（例如：`adv_size = 0.075`）。
- 只允许输出1个Python代码块，**严禁包含任何解释性文字或Markdown说明**，直接以 ```python 开头
"""
        return prompt

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
    def evoleap_prompt(
        operator_type: str,
        task_description: str,
        parent_code: str,
        reflection: str,
        env_context: Dict
    ) -> str:
        """
        EvoLeap 强约束四向变异算子

        Args:
            operator_type: 变异算子类型 ('F1', 'F2', 'F3', 'L1')
            task_description: 任务描述
            parent_code: 上一代最优代码
            reflection: 客观诊断反馈
            env_context: 环境上下文

        Returns:
            str: 完整提示词
        """
        code_snippet = env_context.get('code_snippet', '')

        # 定义四向变异策略
        strategies = {
            'F1': '【Reward Branch Augmentation (分支扩充)】\n请完全保留原代码的现有逻辑和权重，新增一个（且仅新增一个）奖励或惩罚分量，用于解决诊断报告中缺失的协同行为引导。',
            'F2': '【Reward Component Reconstruction (失效分量重构)】\n请定位诊断报告中指出的"失效"或"起反作用"的分量。不要直接删除它们，而是重构其数学逻辑（例如：将线性惩罚改为指数惩罚、引入平滑阈值或改变距离函数的计算方式）。保持其他有效分量不变。',
            'F3': '【Reward Equilibrium Tuning (平衡微调)】\n绝对不要增加或删除现有的逻辑分支！请严格保持代码拓扑不变，仅根据诊断报告，修改各奖励分量的权重系数（增大/减小）。',
            'L1': '【Reward Paradigm Leap (范式跃迁)】\n彻底抛弃原代码的设计思路！请从零开始构建一个全新的奖励函数（例如尝试全局势场、相对距离极坐标系等与原先完全不同的视角）。'
        }

        constraint = strategies.get(operator_type, strategies['F3'])

        prompt = f"""你是一位专业的强化学习奖励工程师。请基于上一代的诊断反馈执行特定的变异操作。

# 环境基座
{code_snippet}

# 上一代最优代码
```python
{parent_code}
```

# 客观诊断反馈
{reflection}

# 强制变异指令
{constraint}

# 接口规范
请实现 compute_reward 函数，保持接口签名不变：

```python
import numpy as np

def compute_reward(agent_name, observation, global_state, actions, world):
    if not global_state['is_adversary']: return 0.0, {{}}
    components = {{}}
    # [根据变异指令修改这里的核心逻辑]
    total_reward = sum(components.values())
    return total_reward, components
```

只允许输出1个Python代码块，严禁任何解释性文字，直接以 ```python 开头。

# 致命红线约束 (Anti-Hallucination Guardrails)
- **禁止引入外部依赖或未定义的类**：绝不可实例化 `CoreEnvLogic()` 或尝试访问 `world.logic`、`world.adversary_params` 等不存在属性。
- **物理常量硬编码**：若要新增依赖于物理常量的逻辑（如体积、速度），必须直接在函数内写死数值常量（如 `adv_size = 0.075`）。
- **严格保持接口签名**：必须保留 `components` 字典收集机制并返回 `total_reward, components`。
- 只允许输出1个修改后的Python代码块，严禁任何解释性文字，直接以 ```python 开头。"""
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

如果反思中指出需要**完全重构**奖励函数（即当前奖励函数存在根本性问题），请遵循以下指导：

**重构策略**：
- 从零开始设计全新的奖励函数
- 考虑使用不同的奖励机制（如势场函数、启发式规则、距离变换等）
- 可以参考环境的核心逻辑（见上文代码片段）设计更直观的奖励
- 保持函数签名不变，但内部逻辑可以完全重写

**如果不需要重构**，则基于父代码进行参数调整和优化。

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

# 致命红线约束 (Anti-Hallucination Guardrails)
- **禁止实例化伪类**：绝对禁止在代码中写出 `CoreEnvLogic()`！上方的代码片段仅是背景文档，运行时环境中根本不存在这个类。
- **禁止虚构属性**：绝对禁止调用 `world.logic`、`world.adversary_params` 等不存在的属性。
- **物理常量硬编码**：如果需要使用物理参数（如智能体的 `size=0.075`、地图大小 `world_size=2.5` 等），必须直接在函数内部以局部变量的形式硬编码声明。
"""

        return prompt
