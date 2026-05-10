# LLM Interaction Log

- **Generation**: 0
- **Phase**: Phase1_CoT_Analysis
- **Timestamp**: 20260414_121850

================================================================================
## Prompt (Sent to LLM)
================================================================================

```text
你是一位资深多智能体强化学习（MARL）专家。在设计奖励函数前，你必须先对环境模型（Dec-POMDP）进行严格的代码级诊断。

# 任务描述
任务：3个追捕智能体协同围捕1个逃逸目标。
要求：
1. 追捕者需要接近并包围目标
2. 追捕者之间避免碰撞
3. 形成均匀的包围圈
4. 在尽可能短的时间内完成围捕


# 环境信息
追捕者数量: 3, 逃跑者数量: 1

核心逻辑代码：
```python

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

```

================================================================================
## Response (From LLM)
================================================================================

```text
下面是对给定环境的代码级诊断，从五个维度逐条分析，并显式遵守“只分析、不写奖励函数代码”的约束。

---

## 一、实现细节 (Implementation Details)

### 1. 依赖包与框架结构

从代码可以看出，环境基于 PettingZoo 的 MPE（Multi-Agent Particle Environment）子框架构建，依赖如下模块：

1. 标准与通用依赖
   - `numpy as np`：向量运算、距离计算、数组拼接等。
   - `gymnasium` 与 `gymnasium.utils.EzPickle`：环境接口与序列化工具（未在片段中直接使用，但通常用于环境包装和注册）。

2. PettingZoo / MPE 相关依赖
   - `from pettingzoo.mpe._mpe_utils.core import Agent, Landmark, World`  
     提供 MPE 的基础类：智能体、地标、世界（物理仿真容器）。
   - `from pettingzoo.mpe._mpe_utils.scenario import BaseScenario`  
     场景脚手架，通常定义 `make_world`, `reset_world`, `reward`, `observation` 等接口。
   - `from pettingzoo.mpe._mpe_utils.simple_env import SimpleEnv, make_env`  
     将 scenario 封装成 PettingZoo 风格环境的工具函数/基类。
   - `from pettingzoo.utils.conversions import parallel_wrapper_fn`  
     用于将环境转换为并行 API 版本。

3. 自定义模块
   - `from .custom_agents_dynamics import CustomWorld`  
     自定义世界类，可能覆盖/扩展 `World`（例如添加特定动力学、阻尼等）。
   - `from . import reward_function`  
     插拔式奖励函数模块。这里非常关键：奖励函数在这个模块内实现，且当前给出的 `CoreEnvLogic` 仅是文档辅助，不可在真实代码中被调用。

### 2. 外部未定义变量 / 不可用对象

在当前片段中：

- `CoreEnvLogic` 内部只使用自身属性：
  - `self.world_size`
  - `self.max_force`
  - `self.capture_threshold`
  - `self.adversary_params`
  - `self.agent_params`
- 所使用的其他对象均为函数参数或局部变量：
  - `agent`, `world`, `all_agents`, `adversaries`, `preys` 等。
- 未引用任何在片段外未声明的变量名。

但需要强调 API 边界问题：

- `CoreEnvLogic` 类本身在说明中被标注为“文档伪代码”（不可在运行时实例化）。因此在真实的 `compute_reward` 实现中：
  - 不能写 `logic = CoreEnvLogic()` ；
  - 不能假设存在 `world.logic`、`world.adversary_params`、`world_size` 等属性。
- 如果在别处写 `compute_reward(agent, world)`，你必须在函数体内部用**硬编码局部变量**来重新定义必要的常量，例如：
  - `world_size = 2.5`
  - `capture_threshold = 0.5`
  - `adv_size = 0.075`, `prey_size = 0.05`
  - `adv_max_speed = 1.0`, `prey_max_speed = 1.3`

结论：片段内部没有“未定义变量”问题，但如果读者误以为 `CoreEnvLogic` 是真实可用类、进而在 reward 中调用它，就会越界使用“文档伪代码”，这在设计时被明确禁止，属于 API 使用错误。

---

## 二、环境结构 (Environment Structure)

### 1. 全局状态 (Global State) 结构与物理意义

`_build_global_state(self, agent, world)` 返回一个字典 `global_state`，包含：

```python
{
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
```

逐项解释：

1. `agent_positions: np.ndarray shape=(N, 2)`
   - 所有智能体（追捕者 + 逃跑者）的绝对位置（世界坐标系）。
   - 假设：`world.agents` 的顺序固定（例如先 adversaries 再 preys，或反之），否则还需额外标记索引含义。
   - 物理意义：每个智能体的 2D 平面位置 `[x, y]`，单位是“世界坐标”，范围约在 `(-world_size, world_size)` 之间，即 `(-2.5, 2.5)`。

2. `agent_velocities: np.ndarray shape=(N, 2)`
   - 所有智能体的线速度向量 `[vx, vy]`（绝对值，与是否归一无关；此处没有显式归一化）。
   - 物理意义：反映每个智能体的运动方向和速度大小，有助于设计基于速度的惩罚或协调机制。

3. `prey_position: np.ndarray shape=(2,)`
   - 单个逃跑者的绝对位置。
   - 若环境中无猎物（理论上不符合任务设定，这里仅做防御性处理），则返回 `np.zeros(2)`。
   - 物理意义：围捕目标的当前位置。

4. `prey_velocity: np.ndarray shape=(2,)`
   - 单个逃跑者的速度向量。
   - 若无猎物，则为 `[0, 0]`。
   - 物理意义：逃者运动方向与速度，用于衡量追捕者相对速度关系。

5. `distances_to_prey: np.ndarray shape=(N_adv,)`
   - 对所有追捕者 `adv`：`||adv.state.p_pos - prey_pos||` 的欧氏距离。
   - 物理意义：追捕者-猎物距离向量，是衡量“接近/远离目标”的核心变量之一。
   - 对本任务：距离越小，越接近完成围捕；但仅“接近”不足以构成“包围”。

6. `inter_agent_distances: np.ndarray shape=(N, N)`
   - `inter_agent_distances[i][j] = ||agent_positions[i] - agent_positions[j]||`。
   - 全体智能体之间的 pairwise 距离矩阵（包含对称元素和对角线为 0）。
   - 物理意义：
     - 用来做防撞（同阵营之间），避免追捕者相互碰撞。
     - 也可用于判断空间拓扑结构（围成环形、集中在一侧、分布是否均匀等）。

7. `is_adversary: bool`
   - 对当前 `agent`，标记其是否为追捕者。
   - 物理意义：可在 compute_reward 对“局部 agent”决定采用哪种奖励形式（追捕者 vs 逃跑者）。

8. `world_size: float`
   - 地图的半边长：`2.5`。
   - 物理意义：世界边界约为 `[-2.5, 2.5] × [-2.5, 2.5]`。

9. `capture_threshold: float`
   - 围捕判定的距离阈值，给定为 `0.5`（注释里：`world_size * 0.2`）。
   - 物理意义：通常作为“追捕者离猎物距离 < capture_threshold”视为接近或抓捕的条件之一。
   - 注意：这个阈值不等同于碰撞半径，而是一个“任务级距离门槛”。

总结：`global_state` 中已经包含了实现围捕任务统计指标所需的关键变量：全体位置、速度、追捕者-猎物距离、智能体间距离、世界尺度和围捕距离阈值。

### 2. 局部观测 (Observation) 结构与相对信息

`observation(self, agent, world)` 返回一个 1D numpy 向量，构造逻辑：

```python
# 自身
norm_self_vel = agent.state.p_vel / agent.max_speed    # shape=(2,)
norm_self_pos = agent.state.p_pos / self.world_size    # shape=(2,)

other_pos = []
other_vel = []
for other in world.agents:
    if other is agent:
        continue
    rel_pos = (other.state.p_pos - agent.state.p_pos) / self.world_size
    other_pos.append(rel_pos)
    if not other.adversary:
        other_vel.append(other.state.p_vel / other.max_speed)

obs = np.concatenate([norm_self_vel] + [norm_self_pos] + other_pos + other_vel)
```

关键点：

1. 自身信息（自中心坐标系 + 归一化）
   - `norm_self_vel`：自身速度除以自身 `max_speed` → 范围约为 `[-1, 1]`。
   - `norm_self_pos`：自身位置除以 `world_size` → 范围约为 `[-1, 1]`。
   - 物理意义：
     - 智能体知道自己在场地中的相对位置（相对于世界边界）。
     - 知道自己速度相对于最大速度的比例（利于策略泛化）。

2. 其他智能体的相对位置
   - 对于所有 `other in world.agents` 且 `other is not agent`：
     - `rel_pos = (other.p_pos - agent.p_pos) / world_size`。
     - 在观测中是**从当前 agent 看其它 agent 的向量**，以自身为原点。
   - 物理意义：
     - 所有其他智能体（包括追捕者与逃跑者）的相对位置都可见。
     - 训练时，追捕者可以通过这些 relative vectors 来协调形成包围结构。
   - 注意：`other_pos` 中没有区分“谁是猎物，谁是队友”，仅有相对位置信息；但 from PettingZoo 设定，一般 agents 的顺序是固定的，策略网络可能学到这种区分，或者场景其他部分会显式编码。

3. 其他智能体的速度（只包含非追捕者，即猎物的速度）
   - `if not other.adversary: other_vel.append(other.state.p_vel / other.max_speed)`
   - 在本任务中：有 3 个 adversaries（追捕者）和 1 个 prey（逃跑者）。
   - 因此：
     - 对于一个“追捕者”观测：
       - `other_pos` 里有 3 个元素（2 个队友 + 1 个猎物） → 6 维。
       - `other_vel` 里仅有 1 个元素（猎物速度） → 2 维。
     - 对于“猎物”观测：
       - `other_pos` 中有 3 个追捕者的位置 → 6 维。
       - 因为 `not other.adversary` 对于追捕者为 False，所以 `other_vel` 为空 → 0 维。
   - 物理意义：
     - 追捕者可以看到逃跑者的速度（相对于逃跑者最大速度），便于预测其未来位置。
     - 猎物（如果需要）看不到追捕者的速度，只能根据其相对位置进行逃避。

4. 观测向量维度（以当前任务 3 追捕 + 1 猎物为例，以每类 agent 分别统计）

- 对**单个追捕者**的观测：
  - 自己速度：2
  - 自己位置：2
  - 其他 3 个智能体相对位置：3 × 2 = 6
  - 猎物速度：1 × 2 = 2
  - 总维度：2 + 2 + 6 + 2 = 12

- 对**猎物**的观测：
  - 自己速度：2
  - 自己位置：2
  - 其他 3 个追捕者相对位置：3 × 2 = 6
  - 没有其他非 adversary，因此无 `other_vel`
  - 总维度：2 + 2 + 6 = 10

总结：局部观测采用自中心、归一化的连续空间，包含：
- 自身状态（位置/速度）；
- 所有其他智能体的相对位置；
- 仅逃跑者的速度信息（对于追捕者）。

---

## 三、智能体交互 (Agent Interactions)

### 1. 物理属性比较：追捕者 vs 逃跑者

在 `__init__` 中明确定义：

```python
self.adversary_params = {'size': 0.075, 'max_speed': 1.0}
self.agent_params = {'size': 0.050, 'max_speed': 1.3}
```

从任务描述可以推断：

- adversary = 追捕者（pursuers）
- agent (non-adversary) = 逃跑者（prey）

对比：

1. 体积大小 (`size`)
   - 追捕者 size：`0.075`
   - 逃跑者 size：`0.050`
   - 物理意义：
     - 追捕者的“碰撞半径”更大，更容易与其他体发生碰撞。
     - 在同样距离下，追捕者更容易撞到彼此或撞到猎物。

2. 最大速度 (`max_speed`)
   - 追捕者 max_speed：`1.0`
   - 逃跑者 max_speed：`1.3`
   - 物理意义：
     - 逃跑者速度更快，若追捕者仅追尾，很难直接追上。
     - 围捕要成功，必须利用多智能体协同进行侧翼包抄/包围，而非单点追击。

这与典型的“围捕实验”设定一致：单个追捕者速度劣势，多智能体通过协作来弥补速度差距。

### 2. 成功捕获（Collision）的条件

`is_collision` 函数定义如下：

```python
def is_collision(self, agent1_pos, agent1_size, agent2_pos, agent2_size):
    """碰撞检测：欧氏距离 < 半径之和"""
    delta_pos = agent1_pos - agent2_pos
    dist = np.sqrt(np.sum(np.square(delta_pos)))
    dist_min = agent1_size + agent2_size
    return dist < dist_min
```

数学表达式：

- 设两个智能体位置为 \(p_1, p_2 \in \mathbb{R}^2\)，大小（半径）为 \(r_1, r_2\)
- 距离 \(d = \|p_1 - p_2\|_2\)
- 最小允许距离 \(d_{\min} = r_1 + r_2\)
- 碰撞条件为：
  \[
  d < d_{\min} = r_1 + r_2
  \]

物理意义：

- 这是真正的“物理碰撞”检测，完全基于几何形状（圆形/半径）。
- 这与“围捕判定”中的 `capture_threshold = 0.5` 不同：
  - `is_collision`：指两个实体的物理边界重叠。
  - `capture_threshold`：任务级距离门槛，可能是“接近包围”的判定，而不要求边界相交。

在围捕任务中可能有两种“成功”定义：

1. 严格碰撞式：追捕者与猎物发生物理碰撞（`is_collision=True`）。
2. 几何包围式：多个追捕者在猎物周围的距离和角度结构满足某个约束（即形成包围圈），即便没有真正撞到猎物。

当前片段只明确了“碰撞检测”的计算方式；围捕判定逻辑需要结合 `distances_to_prey` 和 `capture_threshold`，以及空间布局来定义。

---

## 四、任务相关信息 (Task-relevant Information)

这里明确：仅分析哪些变量对任务有用/有关系，**不**给出奖励函数实现。

### 1. 围捕任务核心目标与变量的关系

任务要求：

1. 追捕者需要接近并包围目标
2. 追捕者之间避免碰撞
3. 形成均匀的包围圈
4. 在尽可能短的时间内完成围捕

与 `global_state` 和其他可用信息对应关系：

1. “接近目标”  
   直接相关变量：
   - `distances_to_prey`（正相关/负相关关系）：
     - 距离越小 → 接近目标程度越高（应正向评价/负向惩罚距离）。
   - `capture_threshold`：
     - 可以用来定义“成功接近”的 hard condition（例如所有追捕者与猎物都在该阈值内）。
   还可以间接依赖：
   - 绝对或相对位置 `agent_positions`, `prey_position`。

2. “包围目标”
   包围不是仅距离小，而是空间拓扑结构。相关变量：
   - `agent_positions` 与 `prey_position`：
     - 可以计算：
       - 各追捕者在以猎物为原点的极坐标角度分布；
       - 是否覆盖 360° 或某个角度区间达到均匀分布；
   - `distances_to_prey`：
     - 可保证包围时各追捕者距离猎物在一定区间内（既不过远也不过近）。

3. “追捕者之间避免碰撞”
   直接相关变量：
   - `inter_agent_distances`：
     - 任意两个追捕者的距离 > (size1 + size2) → 无碰撞；
     - 可针对同阵营组合 (adversary, adversary) 进行安全距离检查。
   - 物理常量：追捕者 size = 0.075 → 追捕者之间安全距离下限约为 0.15（不含缓冲项）。

4. “形成均匀的包围圈”
   主要涉及空间拓扑和角度均匀性（不只是距离均匀）：
   - `agent_positions` 和 `prey_position`：
     - 对追捕者 i，定义向量 \(v_i = p_i - p_\text{prey}\)，可以计算其极角 \(\theta_i\)。
     - 三个 \(\theta_i\) 若能接近等间隔（约 120° 相隔），则构成高质量包围圈。
   - `distances_to_prey`：
     - 三个追捕者到猎物的距离应相近，避免某个追捕者过近或过远导致不均匀。
   - `inter_agent_distances`：
     - 追捕者之间距离也会在一定范围，配合角度分布，形成一个大致“圆环”。

5. “尽可能短时间完成围捕”
   与时间或步骤数直接相关的变量在片段中没有显式给出，例如步数计数 `t` 不在 `global_state` 中。
   在环境主逻辑中通常有：
   - `env.step_count` 或 `world.steps` 之类整体时间步。
   片段里没有，但可以通过外层环境（非 `CoreEnvLogic`）访问。  
   若仅在 reward 函数内部考虑，可通过：
   - 一旦满足“围捕成功”条件，即结束（或给一次性高奖励），以鼓励更短路径；
   - 也可以在每步加入常数时间负成本（不需要额外变量）。

总结：为围捕设计奖励时，`distances_to_prey`、`inter_agent_distances`、`agent_positions` 与 `prey_position` 是最关键的几类变量；`capture_threshold` 是一个天然的几何判定参考。

### 2. 代表“高质量包围圈”的空间拓扑结构

为了描述“高质量包围圈”，至少要满足以下几类几何性质：

1. 目标处于追捕者形成的“中心区域”
   - 在 2D 平面中，三个追捕者构成一个三角形：
     - 将猎物的位置用“点在多边形内部”检测（例如 barycentric coordinates）：
       - 如果猎物处于三角形内部，则说明包围是封闭的；
       - 如果在三角形外部（例如三追捕者基本在同一侧），则只是“围聚而不包围”。
   - 对应变量：
     - `agent_positions`（追捕者 index 子集）
     - `prey_position`

2. 追捕者到猎物的距离相对接近（半径均匀）
   - 设三追捕者与猎物距离为 \(d_1, d_2, d_3\)，可测：
     - 平均距离 \(\bar{d} = (d_1 + d_2 + d_3)/3\)
     - 方差 \(\text{Var}(d_i)\) 或 max-min：
       \[
       \max_i d_i - \min_i d_i
       \]
     - 方差小 → 三者距离相近，包围圈近似圆形而非长条。
   - 对应变量：
     - `distances_to_prey`（仅追捕者）

3. 追捕者在围绕猎物的角度上均匀分布
   - 对每个追捕者 i：
     - 计算向量 \(v_i = p_i - p_\text{prey}\)
     - 角度 \(\theta_i = \text{atan2}(v_i^y, v_i^x)\)
   - 将三个角度排序：\(\theta_{(1)} \le \theta_{(2)} \le \theta_{(3)}\)；
   - 计算相邻角差（环形考虑）：
     - \(\Delta_1 = \theta_{(2)} - \theta_{(1)}\)
     - \(\Delta_2 = \theta_{(3)} - \theta_{(2)}\)
     - \(\Delta_3 = 2\pi - (\theta_{(3)} - \theta_{(1)})\)
   - 理想均匀包围：\(\Delta_1 \approx \Delta_2 \approx \Delta_3 \approx \frac{2\pi}{3}\)。
   - 对应变量：
     - `agent_positions`（追捕者部分）
     - `prey_position`

4. 追捕者之间保持安全距离（避免互撞）
   - 三追捕者间的距离：
     - `inter_agent_distances[i][j]`，i, j ∈ {追捕者 index}
   - 条件：
     - \(d_{ij} > r_i + r_j\)（碰撞半径之和）  
     - 实务中可设置更大的安全下限 \(d_{ij} > r_i + r_j + \epsilon\)，避免边缘碰撞。
   - 高质量包围圈：追捕者之间相距不过近且不过远：
     - 彼此距离在某个合理区间 [d_min_safe, d_max_reasonable]。

5. 捕获区域紧凑且包围不松散
   - 三追捕者构成的三角形面积适中，不至于围得太远：
     - 可基于 `agent_positions` 计算三角形面积：
       - 面积过大 → 猎物有较大逃脱空间；
       - 面积过小 → 追捕者过于集中，有可能撞到一起。
   - 结合猎物位置判断：猎物距三角形边界的最小距离不宜过大。

综合来看，“高质量包围圈”的空间拓扑特征可以用以下语义描述：

- 猎物位于三追捕者形成的三角形内部；
- 三个追捕者到猎物的距离接近一个中等的半径（距离不大于 capture_threshold 的某个倍数）；
- 三个追捕者相对于猎物的方位角接近等间隔（约 120° 差距）；
- 三个追捕者之间距离既大于碰撞半径和，又不过分远离；
- 三角形面积与半径匹配，形成紧凑、封闭的围捕区域。

所有这些几何特征都可以由 `agent_positions`、`prey_position` 和 `inter_agent_distances` 派生得到。

---

## 五、API 边界隔离 (API Boundaries)

这一部分是约束性很强的实现边界说明：

1. `CoreEnvLogic` 是**文档伪代码**
   - 它的作用仅是解释“环境物理、状态构造、观测结构”，不是实际运行组件。
   - 在真实项目中不会由环境实例自动创建和挂载：
     - 没有 `world.logic`；
     - 没有 `env.logic`；
     - 不能在 reward 中写 `logic = CoreEnvLogic()` 并调用其方法。

2. 在真实的 `compute_reward` 中的限制
   - 禁止：
     - `CoreEnvLogic()` 实例化；
     - 使用 `world.logic`、`world.adversary_params`、`world.agent_params`、`self.world_size` 等伪属性。
   - 必须：
     - 所有物理常量在 `compute_reward` 内部**以局部常量**形式硬编码，例如：
       - `world_size = 2.5`
       - `capture_threshold = 0.5`
       - `adv_size = 0.075`
       - `prey_size = 0.050`
       - `adv_max_speed = 1.0`
       - `prey_max_speed = 1.3`
   - 与 `global_state` 的关系：
     - 若 `compute_reward` 的接口是 `(agent, world)`，你可以在里面重新构造类似 `_build_global_state` 的状态结构，但不能依赖 `CoreEnvLogic`。
     - 若接口是 `compute_reward(agent, world, global_state)`，则只能使用传入的 `global_state` 中已有字段混合这些局部常量。

3. 回顾“不要写奖励函数”要求
   - 当前任务中，只能对环境结构、变量含义和任务需求做分析；
   - 不允许给出任何 reward 函数的代码实现或伪实现；
   - 但可以基于这些变量说明：哪些信息是 reward 设计时必须关注的（上面已经说明）。

---

总结：

- 环境基于 PettingZoo MPE，`CoreEnvLogic` 是一个**纯说明性类**，解释了全局状态构造、局部观测结构、碰撞检测和关键物理常数。
- 全局状态中包含：
  - 所有智能体的位置和速度；
  - 单一逃跑者的位置和速度；
  - 追捕者到猎物的距离；
  - 智能体间的距离矩阵；
  - 世界大小与围捕阈值。
- 局部观测采用自中心、归一化形式，包含：
  - 自身位置、速度；
  - 所有其他智能体的相对位置；
  - 对于追捕者，还包含猎物的速度。
- 追捕者与猎物在 size 与 max_speed 上物理属性不同：追捕者更大更慢，猎物更小更快，促使利用多智能体协作完成围捕。
- “高质量包围圈”可用猎物是否位于追捕者三角形内部、追捕者到猎物距离均衡、追捕者角度均匀分布以及互相安全距离等几何指标来刻画。
- 在实现 reward 时必须遵守 API 边界：不能调用 `CoreEnvLogic`，所有物理常量必须在函数内部显式声明为局部变量。
```
