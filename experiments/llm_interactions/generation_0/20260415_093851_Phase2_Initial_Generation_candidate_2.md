# LLM Interaction Log

- **Generation**: 0
- **Phase**: Phase2_Initial_Generation
- **Candidate Info**: 2
- **Timestamp**: 20260415_093851

================================================================================
## Prompt (Sent to LLM)
================================================================================

```text
你是一位专业的强化学习奖励工程师。请基于之前的环境诊断分析，编写符合要求的奖励函数。

# 任务描述
任务：3个追捕智能体协同围捕1个逃逸目标。
要求：
1. 追捕者需要接近并包围目标
2. 追捕者之间避免碰撞
3. 形成均匀的包围圈
4. 在尽可能短的时间内完成围捕


# 先验环境诊断分析
下面按要求，从五个维度对这段 Dec-POMDP 环境核心逻辑做代码级诊断说明（不涉及任何奖励函数实现）。

---

## 一、实现细节 (Implementation Details)

### 1. 使用了哪些依赖包？

在代码顶部：

```python
import numpy as np
import gymnasium
from gymnasium.utils import EzPickle

from pettingzoo.mpe._mpe_utils.core import Agent, Landmark, World
from pettingzoo.mpe._mpe_utils.scenario import BaseScenario
from pettingzoo.mpe._mpe_utils.simple_env import SimpleEnv, make_env
from pettingzoo.utils.conversions import parallel_wrapper_fn

from .custom_agents_dynamics import CustomWorld
from . import reward_function
```

依赖可分为几类：

1. 通用数值与 RL 库
   - `numpy`：用于向量运算、距离计算、数组构造（如 `np.linalg.norm`, `np.sqrt`, `np.square`, `np.zeros`, `np.array`）。
   - `gymnasium` 与 `gymnasium.utils.EzPickle`：用于环境封装、序列化与与 Gym 生态的兼容。

2. PettingZoo MPE 框架相关
   - `pettingzoo.mpe._mpe_utils.core` 中的 `Agent, Landmark, World`：多智能体物理世界和实体基类。
   - `pettingzoo.mpe._mpe_utils.scenario.BaseScenario`：MPE 下场景定义的基类。
   - `pettingzoo.mpe._mpe_utils.simple_env.SimpleEnv, make_env`：将场景封装为 PettingZoo 环境的工具。
   - `pettingzoo.utils.conversions.parallel_wrapper_fn`：用于生成 parallel API 版本的封装器。

3. 工程内模块
   - `from .custom_agents_dynamics import CustomWorld`：自定义世界物理或智能体动力学（继承/扩展 `World` 的版本），用来替换标准 MPE 的物理。
   - `from . import reward_function`：一个可插拔的奖励函数模块，内部应包含 `compute_reward` 等逻辑。

`CoreEnvLogic` 本身是一个“文档伪代码类”，用来说明环境的物理设定与状态/观测构造，并不一定在真实运行中被实例化。

### 2. 是否引入了外部未定义的变量？

在给出的 `CoreEnvLogic` 内部：

- 使用的所有标量常量都在 `__init__` 中定义：
  - `self.world_size`
  - `self.max_force`
  - `self.capture_threshold`
  - `self.adversary_params`
  - `self.agent_params`

- 在方法中使用的变量：
  - `is_collision(self, agent1_pos, agent1_size, agent2_pos, agent2_size)`  
    - 所有参数均来自函数形参，无外部未定义变量。
  - `_build_global_state(self, agent, world)`  
    - 使用 `world.agents`、`a.adversary`、`a.state.p_pos`、`a.state.p_vel`、`agent.adversary` 等属性。  
    - 这些是标准 MPE `World` 与 `Agent` 结构中通常存在的字段，因此在 MPE 框架假设下是已定义的。
    - 使用 `self.world_size`, `self.capture_threshold` 是在 `__init__` 已定义。
  - `observation(self, agent, world)`  
    - 使用 `agent.state.p_vel`, `agent.state.p_pos`, `agent.max_speed`；  
    - `world_size` 来自 `self.world_size`；  
    - 遍历 `world.agents`，访问 `other.state.p_pos`, `other.state.p_vel`, `other.max_speed`, `other.adversary`。  
    - 同样是 MPE 生态中定义的标准属性。

在本代码片段内部，没有使用未定义的 Python 变量。  
需要注意的是：**`CoreEnvLogic` 自身在真实环境代码中可能根本不会被实例化**，它只是一个“参考实现”，真正的环境逻辑和 reward 应在其他文件（如 `reward_function.py` 或 scenario 类）中实现。

---

## 二、环境结构 (Environment Structure)

### 1. 状态空间 (Global State)：`_build_global_state`

`_build_global_state` 函数构建了传给 `compute_reward` 的 `global_state` 字典，其结构为：

```python
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
```

逐项解释其物理意义：

1. `agent_positions: np.ndarray, shape = (N, 2)`
   - 对应 `world.agents` 中所有智能体（追捕者 + 逃逸者）的二维位置向量（一般是 x, y）。
   - 单位：与整个世界坐标系统一（范围大致在 [-world_size, world_size]）。

2. `agent_velocities: np.ndarray, shape = (N, 2)`
   - 所有智能体的二维速度向量 `(vx, vy)`，未归一化。
   - 反映每个智能体当前的运动方向与速率（单位速度）。

3. `prey_position: np.ndarray, shape = (2,)`
   - 逃逸目标（猎物）的当前二维位置。
   - 若 `preys` 为空则为 `np.zeros(2)`，理论上正常任务中有1只猎物。

4. `prey_velocity: np.ndarray, shape = (2,)`
   - 猎物的当前二维速度。为空时为 `np.zeros(2)`（容错）。

5. `distances_to_prey: np.ndarray, shape = (#adversaries,)`
   - 每个追捕智能体到猎物的欧氏距离：
     ```python
     distances_to_prey[i] = || adv_i.pos - prey_pos ||
     ```
   - 这是围捕任务最直接的“靠近程度”指标。

6. `inter_agent_distances: np.ndarray, shape = (N, N)`
   - 智能体间所有两两欧氏距离：
     ```python
     inter_agent_distances[i][j] = ||agent_positions[i] - agent_positions[j]||
     ```
   - 对角线 `i == j` 情况下为 0。
   - 可用于：
     - 追捕者之间碰撞检测（与 `size` 门限比较）。
     - 评价队形是否均匀、是否挤在一起等。

7. `is_adversary: bool`
   - 当前“被计算奖励的”智能体是否为追捕者。
   - 通常 reward 函数会根据该标志区分：只对 adversary 给围捕奖励，对 prey 可能给逃跑奖励或不给奖励。

8. `world_size: float`
   - 地图尺度：世界坐标范围在 `(-world_size, world_size)`，即约 [-2.5, 2.5]^2。
   - 可用于将距离归一化（例如除以 world_size），使 reward 的尺度更稳定。

9. `capture_threshold: float`
   - 围捕判定阈值，约为 `world_size * 0.2 = 0.5`。
   - 可以理解为“围捕半径”，所有追捕者距离猎物小于该阈值且满足某些空间约束（例如分布在猎物周围，而不是同侧）时，可视为成功包围。

这一 `global_state` 是一个典型的“全局可见”的状态集合，适合集中式训练的 value function 或 centralized critic 使用。

---

### 2. 观测空间 (Observation)：`observation(self, agent, world)`

观测向量通过以下方式拼接：

```python
# 自身状态（均归一化）
norm_self_vel = agent.state.p_vel / agent.max_speed     # shape: (2,)
norm_self_pos = agent.state.p_pos / self.world_size     # shape: (2,)

# 其他智能体
other_pos = []
other_vel = []
for other in world.agents:
    if other is agent: 
        continue
    rel_pos = (other.state.p_pos - agent.state.p_pos) / self.world_size
    other_pos.append(rel_pos)
    if not other.adversary:
        other_vel.append(other.state.p_vel / other.max_speed)

return np.concatenate([norm_self_vel] + [norm_self_pos] + other_pos + other_vel)
```

细节与物理含义：

1. 自身速度（归一化）
   - `norm_self_vel = p_vel / max_speed`
   - 将速度除以自身最大速度，为 [-1, 1] 左右的无量纲向量。
   - 反映当前运动方向与相对速度大小。

2. 自身位置（归一化）
   - `norm_self_pos = p_pos / world_size`
   - 把绝对位置信息归一化到 [-1, 1] 范围附近。
   - 表示在整个地图中的位置，而非相对某个目标的偏移。

3. 其他智能体相对位置（归一化）
   - 对每个 `other != agent`：
     ```python
     rel_pos = (other.pos - self.pos) / world_size
     ```
   - 这是 **以当前智能体为原点** 的相对坐标，并归一化。
   - 对当前追捕者来说，这包含：
     - 其他追捕者的位置相对自己：用于协同与防碰撞。
     - 猎物的位置相对自己：用于追踪与围捕定位。

4. 猎物的速度（归一化）
   - 对 `other` 中非对手（即 prey）：
     ```python
     other_vel.append(other.state.p_vel / other.max_speed)
     ```
   - 即：观测中包含猎物的归一化速度向量。
   - 注意：  
     - 追捕者之间的速度互相 **不包含** 在观测中（只对 `not other.adversary` 的目标记录速度）。
     - 因此每个追捕者能看到：
       - 自己的速度和位置；
       - 所有其他智能体的位置（追捕者 + 猎物）；
       - 猎物的速度（通常只有一个猎物）。

5. 拼接顺序
   - 向量顺序为：
     1. `norm_self_vel` (2)
     2. `norm_self_pos` (2)
     3. 所有 `other_pos`（依 `world.agents` 的遍历顺序），每个 2 维
     4. 所有 `other_vel`（仅猎物），每个 2 维

从信息结构看：

- 每个智能体拥有的观测是**局部但富信息的**：
  - 绝对位置信息（相对世界边界）。
  - 相对位置（自身坐标系中其它体的位置）。
  - 猎物的速度。
- 不存在 `global_state` 中那种完整的距离矩阵、世界常量等信息。

---

## 三、智能体交互 (Agent Interactions)

### 1. 捕食者与猎物在物理属性上的差异

在 `CoreEnvLogic.__init__` 中：

```python
self.adversary_params = {'size': 0.075, 'max_speed': 1.0}
self.agent_params     = {'size': 0.050, 'max_speed': 1.3}
```

结合上下文命名习惯：

- adversaries = 追捕者（捕食者）
- agents / preys = 猎物（逃跑者）

差异如下：

1. 大小 `size`（碰撞半径）
   - 追捕者（adversary）：`size = 0.075`
   - 猎物（prey）：`size = 0.050`
   - 追捕者半径更大，意味着：
     - 更容易发生碰撞（无论是追捕者-追捕者还是追捕者-猎物）。
     - 在空间中需要更大间距避免互相碰撞。

2. 最大速度 `max_speed`
   - 追捕者：`max_speed = 1.0`
   - 猎物：`max_speed = 1.3`
   - 猎物更快（1.3 > 1.0），这让围捕变成典型的“慢追快”协同任务：
     - 单个追捕者很难追上猎物。
     - 必须依赖协同包围（多方向夹击、限制活动空间）才能完成任务。

这两个物理属性直接影响你的 reward 设计和策略学习目标：任务本质上是“缓慢的、体积较大的追捕者，通过协作与位置优势，围堵更快、更灵活的猎物”。

### 2. “成功捕获（Collision）”的数学与物理条件

`is_collision` 函数定义了碰撞检测规则：

```python
def is_collision(self, agent1_pos, agent1_size, agent2_pos, agent2_size):
    delta_pos = agent1_pos - agent2_pos
    dist = np.sqrt(np.sum(np.square(delta_pos)))
    dist_min = agent1_size + agent2_size
    return dist < dist_min
```

说明：

1. 数学定义：
   - 计算两实体中心点的欧氏距离：
     \[
     d = \sqrt{ (x_1 - x_2)^2 + (y_1 - y_2)^2 }
     \]
   - 计算“碰撞阈值”（两半径和）：
     \[
     d_{\min} = r_1 + r_2
     \]
   - 判断条件：
     \[
     d < d_{\min} \implies \text{发生碰撞}
     \]

2. 物理意义：
   - 每个 agent 用 `size` 表示其“物理实体半径”。
   - 两个实体的“身体”如果存在重叠（中心距离小于半径和），则认为发生碰撞。
   - 在本任务中：
     - 追捕者-猎物碰撞可代表“抓住/接触”目标。
     - 追捕者-追捕者碰撞代表“队友相撞”，应当被惩罚或避免。

注意：  
`capture_threshold` （0.5）是“围捕判定半径”，
而 `is_collision` 使用的是“物理碰撞半径 = size1 + size2”。  
围捕判定和物理碰撞是两层概念，可以分开使用：

- 碰撞：几何体接触/重叠。
- 围捕：追捕者分布在猎物周围且距离都在某个捕获半径以内（可能不需要真的接触）。

---

## 四、任务相关信息 (Task-relevant Information)

### 1. 围捕任务的核心目标与哪些变量直接相关？

结合任务描述 4 点要求和 `global_state`/observations 中可用的变量，粗略列出**直接相关（可用来构造指标）的变量**：

1. “追捕者需要接近并包围目标”
   - 直接相关变量：
     - `distances_to_prey`（追捕者到猎物的欧氏距离）  
       - 距离越小，越接近捕获状态 → 与“接近”正相关（奖励接近或惩罚远离）。
     - `agent_positions` + `prey_position`  
       - 可计算每个追捕者与猎物的相对位置向量，用于判断是否“分布在猎物周围”。

2. “追捕者之间避免碰撞”
   - 直接相关变量：
     - `inter_agent_distances`  
       - 距离过小（小于 `size_i + size_j`）即发生碰撞。
       - 可以对小于某安全阈值的距离施加惩罚。
     - （逻辑上）需知道追捕者的 `size`，在奖励函数中需本地定义常量，例如 `adv_size = 0.075`。

3. “形成均匀的包围圈”
   - 直接相关变量：
     - `agent_positions` 与 `prey_position`：
       - 可将猎物位置作为圆心，将追捕者位置转成极坐标，检查：
         - 各追捕者到猎物的距离是否接近某期望半径（半径均匀）。
         - 各追捕者极角间隔是否接近等分角（角度均匀）。
     - `distances_to_prey`：
       - 如果三名追捕者与猎物的距离相近，则 “半径均匀”的一部分条件得到满足。

4. “在尽可能短的时间内完成围捕”
   - 直接相关变量：
     - 时间步/episode step（不在代码片段中，但通常环境中有 `timestep` 或 `elapsed_steps`）。
     - `capture_threshold`：满足围捕条件的空间阈值，一旦达到可终止 episode。
   - reward 层面通常通过“每步时间成本”或“尽早完成”的奖励结构实现，但这已经超出本问题范围。

概括：  
核心目标主要与以下变量正/负相关：

- 正相关（值越小越好 → 奖励接近，惩罚远离）：
  - `distances_to_prey`（靠近猎物）
  - 追捕者与猎物的角度分布均匀度指标（由 `agent_positions` 和 `prey_position` 推导）
  - 追捕者之间距离与期望队形距离的接近程度

- 负相关（值越大越差 → 惩罚）：
  - 超出安全阈值的小 `inter_agent_distances`（代表拥挤甚至碰撞）
  - 时间步数（长时间未完成围捕）

### 2. 怎样的空间拓扑代表“高质量的包围圈”？

从几何结构上，高质量“包围圈”一般具备以下空间特征（可基于 `agent_positions` 和 `prey_position` 定义）：

假设：
- 猎物位置：\( p \)
- 三个追捕者位置：\( a_1, a_2, a_3 \)

1. **猎物在几何中心（被包围）**
   - 理想情况下，猎物位于追捕者形成的多边形内部。
   - 对 3 个追捕者即三角形：
     - 猎物应位于三角形 \( \triangle a_1 a_2 a_3 \) 的内部或足够接近中心。
   - 实现方法（reward 中可以这样用，但这里仅描述几何）：
     - 用点在三角形内部的判定（如重心坐标法、同向叉积等）检查 `prey_position` 是否在三角形内部。

2. **均匀角度分布（环绕，而非集中在一侧）**
   - 将猎物位置作为极坐标原点：
     - 对每个 `a_i`，计算向量 \( v_i = a_i - p \)，再计算其极角 \( \theta_i \in (-\pi, \pi] \)。
   - 将角度排序后，三个相邻角度间隔应接近 \( 2\pi/3 \)：
     - \(\Delta\theta_1 \approx \Delta\theta_2 \approx \Delta\theta_3 \approx 120^\circ\)。
   - 若所有追捕者角度集中在某一半平面上（比如都在猎物左侧），则不能算真正的包围，只是“集群追击”。

3. **半径接近：追捕者到猎物的距离相近**
   - 对三个距离 \( d_i = ||a_i - p|| \)：
     - 希望 \( d_1 \approx d_2 \approx d_3 \)，方差较小。
   - 同时 \( d_i \) 应在合理区间：
     - 足够小：保证猎物运动空间被限制（比如都小于 `capture_threshold`）。
     - 不能太小：避免追捕者-猎物、追捕者-追捕者的物理碰撞（也涉及 `size` 和碰撞检测）。

4. **追捕者之间互不碰撞，具有安全间距**
   - 三个追捕者之间两两距离：
     \[
     d_{ij} = ||a_i - a_j||, \quad i \neq j
     \]
   - 需满足：
     - \( d_{ij} > r_i + r_j \)（避免碰撞），其中 `r_i = r_j = adv_size = 0.075`。
     - 理想队形下：
       - 若三追捕者在猎物周围形成正三角形，可以期望：
         \[
         d_{12} \approx d_{23} \approx d_{31}
         \]
       - 这与角度均匀和半径相近相呼应。

综合上述几何条件：

高质量“包围圈”的拓扑结构特征可概括为：

- 猎物被包含在追捕者多边形内部（或非常接近多边形中心）。
- 追捕者在猎物周围近似均匀分布：三者角度近似等分 \(360^\circ\)。
- 三个追捕者到猎物的距离相似（半径均匀）并处于 `[安全下限, capture_threshold]` 的区域。
- 追捕者之间的两两距离大于碰撞阈值，并在某个“理想值”附近，以保持稳定队形。

这些拓扑指标都可以由 `global_state` 中的 `agent_positions`, `prey_position`, `distances_to_prey`, `inter_agent_distances` 推导出来。

---

## 五、API 边界隔离 (API Boundaries)

这一部分是约束条件，关系到**真正实现 `compute_reward` 时不得越界依赖**。

核心要求：

1. `CoreEnvLogic` 是文档伪代码，不是实际可用类
   - 上面展示的 `CoreEnvLogic` 仅用于说明环境中的物理常量、状态构造和观测构造的**设计意图**。
   - 在真实环境中（例如 PettingZoo 的 scenario 或 `reward_function.py` 中），**不可以**写：
     ```python
     logic = CoreEnvLogic()
     ```
     或访问：
     ```python
     world.logic
     world.adversary_params
     world.agent_params
     ```
   - 这些在真实运行的 `World` 对象上是不存在的属性。

2. `compute_reward` 内部禁止依赖 `CoreEnvLogic` 或未定义属性
   - `compute_reward` 只能使用其函数签名中的参数（例如 `agent`, `world`, 或 `global_state` 字典）来计算奖励。
   - 所有需要的**物理常量**必须在函数内部以**局部变量**方式“硬编码”，例如：
     ```python
     adv_size = 0.075
     prey_size = 0.050
     adv_max_speed = 1.0
     prey_max_speed = 1.3
     world_size = 2.5
     capture_threshold = 0.5
     ```
   - 这样做的目的：
     - 保证 `compute_reward` 与 `CoreEnvLogic` 解耦，不依赖额外状态。
     - 明确 reward 函数是一个纯函数（只依赖传入的状态与内部常量）。

3. 使用 `global_state` 时的注意事项
   - 真实 `compute_reward` 的推荐方式是：
     - 接收一个结构与 `_build_global_state` 相同/相似的字典；
     - 在 reward 函数内部仅使用该字典中出现的键（例如 `agent_positions`, `prey_position`, `distances_to_prey`, `inter_agent_distances`, `capture_threshold` 等）。
   - 不应假设 `world` 中还挂载了任何逻辑对象，如 `world.logic`、`world.params` 等。

4. 不要在奖励函数里显式调用观测函数
   - 观测 `observation(self, agent, world)` 是环境给智能体的局部观测接口；
   - reward 函数应直接基于“真实世界物理状态”（如 global state 或 `world.agents` 的原始位置/速度）工作，不应复用观测函数，因为观测通常是稀疏、对每个智能体局部的。

总结这个边界约束：

- 将 `CoreEnvLogic` 视为“设计文档”，不作为真实运行对象。
- 真正实现 reward 函数时，**只使用**：
  - 函数参数（例如 `global_state` 或 `(agent, world)`）。
  - 在函数内部显式硬编码的常量（size, max_speed, world_size, capture_threshold）。
- 禁止访问 `world.logic`、`CoreEnvLogic()` 等在运行时不存在的对象/属性。

---

以上是对该 Dec-POMDP 追捕-逃逸环境在实现细节、环境结构、智能体交互、任务相关几何特征和 API 边界隔离方面的分步诊断说明。

# 环境代码参考
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

# 接口规范要求

请实现 compute_reward 函数，严格遵守以下签名与返回格式：

```python
import numpy as np

def compute_reward(agent_name, observation, global_state, actions, world):
    # 必须通过 global_state['is_adversary'] 过滤逃跑者
    if not global_state['is_adversary']:
        return 0.0, {}

    components = {}

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

```

================================================================================
## Response (From LLM)
================================================================================

```text
```python
import numpy as np


def compute_reward(agent_name, observation, global_state, actions, world):
    # 逃跑者不参与本奖励
    if not global_state["is_adversary"]:
        return 0.0, {}

    components = {}

    # ------------------------
    # 物理与任务超参数（硬编码）
    # ------------------------
    adv_size = 0.075
    prey_size = 0.050
    world_size = 2.5
    capture_threshold = 0.5

    # 队形与碰撞相关超参数
    safe_distance_factor = 1.5  # 安全距离 = safe_distance_factor * (r_i + r_j)
    formation_radius_target = capture_threshold * 0.9
    formation_radius_tolerance = capture_threshold * 0.3
    angle_uniform_target = 2.0 * np.pi / 3.0  # 120°
    time_penalty_per_step = -0.01

    # 取出全局状态
    agent_positions = np.asarray(global_state["agent_positions"], dtype=float)
    agent_velocities = np.asarray(global_state["agent_velocities"], dtype=float)
    prey_pos = np.asarray(global_state["prey_position"], dtype=float)
    prey_vel = np.asarray(global_state["prey_velocity"], dtype=float)
    distances_to_prey = np.asarray(global_state["distances_to_prey"], dtype=float)
    inter_agent_distances = np.asarray(
        global_state["inter_agent_distances"], dtype=float
    )

    # ------------------------
    # 解析 agent 索引与角色
    # ------------------------
    num_agents = agent_positions.shape[0]
    # 假设命名规则为 "agent_i" 或 "adversary_i" 等，尝试从 world.agents 匹配
    agent_index = None
    for idx, ag in enumerate(world.agents):
        if getattr(ag, "name", None) == agent_name:
            agent_index = idx
            break

    if agent_index is None:
        # 回退：尝试根据名字中的数字索引
        try:
            agent_index = int("".join(ch for ch in agent_name if ch.isdigit()))
            if agent_index < 0 or agent_index >= num_agents:
                agent_index = 0
        except ValueError:
            agent_index = 0

    # adversary 索引集合（追捕者）
    adversary_indices = [
        idx for idx, ag in enumerate(world.agents) if getattr(ag, "adversary", False)
    ]

    # 找出当前 agent 在 adversaries 中的索引
    try:
        adv_local_index = adversary_indices.index(agent_index)
    except ValueError:
        adv_local_index = 0

    # ------------------------
    # 1. 距离引导：靠近目标 + 围捕完成奖励
    # ------------------------
    if len(distances_to_prey) > 0:
        # 当前追捕者到猎物距离
        d_self = distances_to_prey[adv_local_index]
        # 所有追捕者到猎物平均距离（协同）
        mean_d_adv = float(np.mean(distances_to_prey))

        # 归一化距离（最好 ~ 0）
        norm_d_self = d_self / world_size
        norm_d_mean = mean_d_adv / world_size

        # 鼓励靠近猎物（自我与团队两部分）
        distance_reward_self = -norm_d_self
        distance_reward_team = -norm_d_mean * 0.5

        # 额外的捕获/围捕成功奖励：所有追捕者都在捕获半径内
        all_within_capture = bool(
            np.all(distances_to_prey < capture_threshold)
        )
        capture_bonus = 0.0
        if all_within_capture:
            # 一次性高奖励（由环境终止逻辑控制次数）
            capture_bonus = 5.0

        components["distance_self"] = distance_reward_self
        components["distance_team"] = distance_reward_team
        components["capture_bonus"] = capture_bonus
    else:
        components["distance_self"] = 0.0
        components["distance_team"] = 0.0
        components["capture_bonus"] = 0.0

    # ------------------------
    # 2. 防碰撞：追捕者-追捕者 与 追捕者-逃跑者
    # ------------------------
    collision_penalty = 0.0
    near_collision_penalty = 0.0

    # 与其他追捕者的碰撞 / 近距离惩罚
    for j in adversary_indices:
        if j == agent_index:
            continue
        d_ij = inter_agent_distances[agent_index, j]
        min_dist = 2.0 * adv_size  # r_i + r_j
        safe_dist = safe_distance_factor * min_dist

        if d_ij < min_dist:
            # 硬碰撞
            collision_penalty -= 2.0
        elif d_ij < safe_dist:
            # 软惩罚：越接近越惩罚
            ratio = (safe_dist - d_ij) / (safe_dist - min_dist + 1e-8)
            near_collision_penalty -= 0.5 * ratio

    # 与猎物碰撞（可选：轻微惩罚，鼓励“包围”而不是硬撞）
    prey_index = None
    for idx, ag in enumerate(world.agents):
        if not getattr(ag, "adversary", False):
            prey_index = idx
            break
    if prey_index is not None:
        d_prey = inter_agent_distances[agent_index, prey_index]
        min_dist_ap = adv_size + prey_size
        # 如果发生物理碰撞，给一个小负奖励，让策略偏向困住而不是撞击
        if d_prey < min_dist_ap:
            collision_penalty -= 0.5

    components["collision_penalty"] = collision_penalty
    components["near_collision_penalty"] = near_collision_penalty

    # ------------------------
    # 3. 包围队形：均匀半径 + 均匀角度
    # ------------------------
    formation_radius_reward = 0.0
    formation_angle_reward = 0.0
    center_in_triangle_reward = 0.0

    if len(adversary_indices) == 3 and prey_index is not None:
        adv_positions = agent_positions[adversary_indices]  # (3, 2)
        # 半径（到猎物的距离）
        vecs = adv_positions - prey_pos[None, :]
        radii = np.linalg.norm(vecs, axis=1) + 1e-8

        # 半径均匀度：接近目标半径 & 方差小
        mean_r = float(np.mean(radii))
        var_r = float(np.var(radii))

        # 半径接近期望值奖励
        radius_dev = abs(mean_r - formation_radius_target)
        # 只在一定容差内给予正向奖励，否则奖励接近 0
        if mean_r < capture_threshold + formation_radius_tolerance:
            formation_radius_reward += -radius_dev / (capture_threshold + 1e-8)
        # 半径方差惩罚（越均匀越好）
        formation_radius_reward += -var_r / (capture_threshold ** 2 + 1e-8)

        # 角度均匀度：将猎物作为中心
        angles = np.arctan2(vecs[:, 1], vecs[:, 0])  # (-pi, pi]
        angles_sorted = np.sort(angles)
        # 环上角度间隔
        angle_diffs = np.diff(angles_sorted)
        # 加上首尾闭环差
        last_gap = (angles_sorted[0] + 2.0 * np.pi) - angles_sorted[-1]
        angle_diffs = np.concatenate([angle_diffs, [last_gap]])
        # 与理想值的偏差
        angle_dev = np.abs(angle_diffs - angle_uniform_target)
        angle_dev_mean = float(np.mean(angle_dev))
        # 归一化到 [0, 1] 左右
        formation_angle_reward = -angle_dev_mean / np.pi

        # 判定猎物是否在三追捕者三角形内部（几何包围）
        a, b, c = adv_positions[0], adv_positions[1], adv_positions[2]
        p = prey_pos

        def _same_side(p1, p2, a_, b_):
            cp1 = np.cross(b_ - a_, p1 - a_)
            cp2 = np.cross(b_ - a_, p2 - a_)
            return cp1 * cp2 >= 0.0

        inside = (
            _same_side(p, a, b, c)
            and _same_side(p, b, a, c)
            and _same_side(p, c, a, b)
        )
        if inside:
            center_in_triangle_reward = 1.0

    components["formation_radius"] = formation_radius_reward
    components["formation_angle"] = formation_angle_reward
    components["center_in_triangle"] = center_in_triangle_reward

    # ------------------------
    # 4. 时间效率：每步小幅时间惩罚
    # ------------------------
    components["time_penalty"] = time_penalty_per_step

    # ------------------------
    # 5. 平滑性与协同速度（可选小权重）
    # ------------------------
    # 鼓励各追捕者的速度方向并非完全一致，以便形成合围
    # 这里简单加一个速度方差惩罚（过于一致则略减分）
    adv_vels = agent_velocities[adversary_indices]
    if adv_vels.shape[0] > 1:
        speed_var = float(np.mean(np.var(adv_vels, axis=0)))
        components["velocity_diversity"] = speed_var * 0.05
    else:
        components["velocity_diversity"] = 0.0

    total_reward = float(sum(components.values()))
    return total_reward, components
```
```
