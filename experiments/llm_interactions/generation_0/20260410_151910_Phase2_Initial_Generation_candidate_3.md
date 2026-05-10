# LLM Interaction Log

- **Generation**: 0
- **Phase**: Phase2_Initial_Generation
- **Candidate Info**: 3
- **Timestamp**: 20260410_151910

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
下面按要求，从五个维度对这段 Dec-POMDP 环境模型做代码级诊断（不涉及任何奖励函数实现）。

---

## 一、实现细节 (Implementation Details)

### 1. 依赖的包和模块

从代码顶部可以看出，环境逻辑依赖以下库和模块：

- 标准/数值计算：
  - `numpy as np`：用于向量、矩阵计算（距离、拼接、归一化等）。
- 强化学习环境框架相关：
  - `gymnasium` 与 `gymnasium.utils.EzPickle`：典型的 RL 环境接口与序列化工具，帮助将环境包装成 Gymnasium 兼容形式。
- PettingZoo 多智能体环境框架：
  - `from pettingzoo.mpe._mpe_utils.core import Agent, Landmark, World`  
    - 提供 MPE（Multi-Agent Particle Environment）的基础元素：智能体、地标、世界。
  - `from pettingzoo.mpe._mpe_utils.scenario import BaseScenario`  
    - MPE 场景基类，一般用于定义 `make_world`, `reset_world`, `reward`, `observation` 等。
  - `from pettingzoo.mpe._mpe_utils.simple_env import SimpleEnv, make_env`  
    - 用于将自定义场景封装成可运行的 PettingZoo 环境。
  - `from pettingzoo.utils.conversions import parallel_wrapper_fn`  
    - 将 AEC 环境包装成并行环境接口。
- 自定义模块（项目内）：
  - `from .custom_agents_dynamics import CustomWorld`  
    - 自定义的 `World` 或物理动态模型，用于替换/扩展 MPE 默认动力学。
  - `from . import reward_function`  
    - 可插拔的奖励函数文件（此处说明只包含追捕者奖励），实际的 `compute_reward` 很可能定义在这里。

### 2. 是否使用了外部未定义变量

在 `CoreEnvLogic` 伪代码中：

- 明确定义了：
  - `self.world_size`, `self.max_force`, `self.capture_threshold`
  - `self.adversary_params`, `self.agent_params`
- 在方法中使用的所有变量：
  - `is_collision` 中使用 `agent1_pos`, `agent1_size`, `agent2_pos`, `agent2_size` 都是函数参数。
  - `_build_global_state(self, agent, world)` 中：
    - 使用 `world.agents`, 每个 `a.state.p_pos`, `a.state.p_vel`, `a.adversary`
      - 这些属性来自 PettingZoo MPE 的 `World` 和 `Agent`，在真正运行环境中是合法的。
    - 使用 `self.world_size`, `self.capture_threshold`，在 `__init__` 中定义。
  - `observation(self, agent, world)` 中：
    - 使用 `agent.state.p_vel`, `agent.max_speed`, `agent.state.p_pos`：来自 Agent。
    - 使用 `world.agents`：来自 World。
    - 使用 `other.state.p_pos`, `other.state.p_vel`, `other.max_speed`, `other.adversary`：也来自 Agent。
    - 使用 `self.world_size`：在 `__init__` 定义。

在本段伪代码内部，没有对未定义的变量进行直接引用，逻辑上是自洽的。

**但要注意：**

- 用户特别说明：`CoreEnvLogic` 类是“文档伪代码”，在真实环境中**不会**被实例化；也不能在 `compute_reward` 中访问 `world.logic` 之类的属性。
- 说明真正可用的参数来源是：PettingZoo 的 `world` 对象与 `agent` 对象，以及在奖励函数中**硬编码**的常量（例如 size、max_speed 等），而不是通过 `CoreEnvLogic`。

因此：伪代码内部并无未定义变量错误，但它本身不是运行时实体，不能依赖它作为 API。

---

## 二、环境结构 (Environment Structure)

### 1. 状态空间 (Global State)

`_build_global_state` 返回的是一个 `dict`，包含：

```python
{
    'agent_positions': agent_positions,        # (n_agents, 2)
    'agent_velocities': agent_velocities,      # (n_agents, 2)
    'prey_position': prey_pos,                 # (2,)
    'prey_velocity': prey_vel,                 # (2,)
    'distances_to_prey': distances_to_prey,    # (n_adversaries,)
    'inter_agent_distances': inter_agent_distances,  # (n_agents, n_agents)
    'is_adversary': agent.adversary,           # bool
    'world_size': self.world_size,             # scalar
    'capture_threshold': self.capture_threshold # scalar
}
```

逐项解释其物理意义与维度：

1. `agent_positions: np.array(shape=(n_agents, 2))`
   - 每个智能体的**绝对位置**（世界坐标系，未归一化）。
   - 维度为 2，意味着二维平面环境。
   - 对于任务：可以用于计算任意智能体之间的距离和相对方位。

2. `agent_velocities: np.array(shape=(n_agents, 2))`
   - 每个智能体的**绝对速度向量**。
   - 用于描述运动趋势（朝向、速度大小），可以推断未来位置趋势。

3. `prey_position: np.array(shape=(2,))`
   - 当前逃跑者（猎物）的绝对位置。
   - 若没有猎物，则为零向量（此处为容错处理，但在我们场景中猎物数量固定为 1）。

4. `prey_velocity: np.array(shape=(2,))`
   - 当前猎物的绝对速度向量。
   - 对追捕者来说，此变量能表征猎物的逃跑方向和速度大小。

5. `distances_to_prey: np.array(shape=(n_adversaries,))`
   - 每个追捕者到猎物的**欧氏距离**。
   - 这是围捕任务最核心的一组标量之一：
     - 趋近猎物 → 距离减小；
     - 离猎物远 → 距离增大。
   - 可以用于奖励“接近猎物”和判定围捕是否形成等。

6. `inter_agent_distances: np.array(shape=(n_agents, n_agents))`
   - 所有智能体之间的**距离矩阵**，`[i][j]` 是第 i 个和第 j 个智能体之间的欧氏距离。
   - 对角线通常为 0（自距离），矩阵对称。
   - 对任务来说：
     - 与碰撞检测相关（是否小于 size 之和）。
     - 与布局均匀性相关（各追捕者之间的距离是否接近某个理想值，是否足够分散包围猎物）。

7. `is_adversary: bool`
   - 表明当前被计算奖励/状态的 `agent` 是否为追捕者。
   - 通常用于在奖励函数中分支处理（追捕者和猎物的目标不同）。

8. `world_size: float`
   - 世界范围（例：2.5，含义是坐标范围大致在 [-2.5, 2.5]）。
   - 用于归一化位置或定义空间尺度（例如 capture_threshold 基于它）。

9. `capture_threshold: float`
   - 围捕判定的半径阈值（给出注释为 `world_size * 0.2`，即 0.5）。
   - 物理意义：如果追捕者与猎物距离在一定范围内且形成合适拓扑，则可判定为成功围捕的一个距离尺度指标。

总体上，这个 `global_state` 已经覆盖了：  
- 所有智能体的绝对位置、速度  
- 追捕者-猎物的距离  
- 智能体之间的距离拓扑  
- 世界尺度与捕获阈值  

足够用于设计多种围捕相关的奖励或评价指标。

---

### 2. 观测空间 (Observation)

`observation(self, agent, world)` 返回的是一个一维向量，由如下部分拼接：

```python
return np.concatenate(
    [norm_self_vel] +
    [norm_self_pos] +
    other_pos +
    other_vel
)
```

拆解：

1. `norm_self_vel = agent.state.p_vel / agent.max_speed`
   - 维度：2
   - 自身速度向量除以自身最大速度，得到**归一化速度**（理论上范围 ~[-1, 1]）。
   - 物理意义：反映当前加速/减速状态、运动方向，与动作空间强相关。

2. `norm_self_pos = agent.state.p_pos / self.world_size`
   - 维度：2
   - 自身位置向量除以世界尺度，得到**归一化位置**。
   - 物理意义：反映自身位于场地的什么位置（靠近边界 / 中心等）。

3. `other_pos`：对每个其他智能体（包括追捕者和猎物）：
   ```python
   rel_pos = (other.state.p_pos - agent.state.p_pos) / self.world_size
   other_pos.append(rel_pos)
   ```
   - 对每个“其他智能体”，添加一个 2 维相对位置向量。
   - 如果一共有 4 个智能体（3 追捕者 + 1 猎物），则每个 agent 看到 `3 × 2 = 6` 维相对位置。
   - 物理意义：  
     - 相对位置是局部观测的核心，用于决策追捕、协作、避碰。
     - 归一化到 [-1,1] 左右，便于学习。

4. `other_vel`：**仅对非追捕者**（即对猎物）：
   ```python
   if not other.adversary:
       other_vel.append(other.state.p_vel / other.max_speed)
   ```
   - 在当前任务中，只有 1 个猎物，因此这里**只有一个 2 维向量**：猎物的归一化速度。
   - 其他追捕者的速度信息不会出现在观测中。
   - 物理意义：
     - 追捕者可以感知猎物的运动方向与速度大小，有利于预判路径。
     - 不能直接感知队友速度（只能通过相对位置变化间接推断）。

因此，每个追捕者的观测向量结构（在 3 追捕者 + 1 猎物的场景下，为例）：

- 自身归一化速度：2 维
- 自身归一化位置：2 维
- 3 个“其他智能体”的相对位置：3 × 2 = 6 维
- 1 个猎物的归一化速度：2 维

总维度：2 + 2 + 6 + 2 = 12 维。

内容结构可以概括为：

> 自我状态（速度+位置） +  
> 所有其他智能体的相对位置 +  
> 猎物的归一化速度

这是一种**局部观测**，无显式全局拓扑信息（例如矩阵形式的全局距离），但通过相对位置已隐含环境结构。

---

## 三、智能体交互 (Agent Interactions)

### 1. 捕食者与猎物的物理属性对比

在 `CoreEnvLogic.__init__` 中定义了两类智能体的参数：

```python
self.adversary_params = {'size': 0.075, 'max_speed': 1.0}
self.agent_params     = {'size': 0.050, 'max_speed': 1.3}
```

从命名和任务叙述可推断：

- `adversary_params`：追捕者（pursuers）
- `agent_params`：猎物（prey / runner）

对比：

1. 尺寸 (`size`)
   - 追捕者：`size = 0.075`
   - 猎物：`size = 0.050`
   - 物理意义：
     - 追捕者**碰撞半径更大**，更容易发生碰撞（与猎物或彼此）。
     - 对避撞任务来说，追捕者需特别注意与其他追捕者的间距。

2. 最大速度 (`max_speed`)
   - 追捕者：`max_speed = 1.0`
   - 猎物：`max_speed = 1.3`
   - 物理意义：
     - 猎物比追捕者**更快**，直线奔跑情况下追捕者很难“追上”猎物。
     - 围捕任务需要利用协作、包围、夹击，而不仅是单个追捕者追击。
     - 若想在奖励中强调快速围捕，应充分考虑这一动力学劣势。

3. 动作约束（在伪代码中仅有世界层面的 `self.max_force = 1.0`）
   - 表示动作（力/加速度）绝对值最大值。
   - 对具体 agent 是否相同未在伪代码中细化，但一般 MPE 中会统一。

综上，猎物更快且更小；追捕者略慢且更大，增加了避撞的难度、降低了单体追击效率，进一步强调“协同围捕”的必要性。

---

### 2. “成功捕获（Collision）”的数学与物理条件

`CoreEnvLogic.is_collision` 函数定义了两智能体间的碰撞判定：

```python
def is_collision(self, agent1_pos, agent1_size, agent2_pos, agent2_size):
    delta_pos = agent1_pos - agent2_pos
    dist = np.sqrt(np.sum(np.square(delta_pos)))   # 欧氏距离
    dist_min = agent1_size + agent2_size          # 半径之和
    return dist < dist_min
```

数学条件：

- 记 `p1, p2` 为两个智能体的位置，`r1, r2` 为它们的半径（size）。
- 碰撞条件为：
  \[
  \|p_1 - p_2\|_2 < r_1 + r_2
  \]
- 即，两圆形碰撞体的中心距离小于半径之和 → 几何圆盘重叠。

物理意义：

- 智能体被建模为圆形（半径= size）。
- 当追捕者与猎物发生碰撞：
  - 从任务角度：
    - 可以视为“抓住”或“接触到”猎物的一种事件。
  - 但这里的任务特别强调“围捕”而不仅是“接触”，因此**单次碰撞不必然等价于“完成任务”**，更可能只是过程中的子事件（例如惩罚撞到猎物太粗暴，或奖励接近但配合拓扑条件）。
- 当追捕者之间发生碰撞：
  - 是需要**避免的行为**，因为要求追捕者之间避免相互碰撞。
  - 可以在奖励中给出负向激励，或在环境中施加惩罚。

---

## 四、任务相关信息 (Task-relevant Information)

你在问题中写了两次“任务相关信息”，下面统一合并回答，一并从“目标变量”和“理想拓扑”两方面剖析。

### 1. 围捕任务与哪些变量直接挂钩（正/负相关）

围捕任务要求：

1. 追捕者接近并包围目标。
2. 追捕者之间避免碰撞。
3. 形成均匀的包围圈。
4. 尽可能短时间完成围捕。

#### 与“接近并包围目标”相关的变量

- **直接相关（正向）：**
  - `distances_to_prey`（每个追捕者到猎物的距离）：
    - 距离越小 → 越接近目标 → 对任务质量越好。
    - 围捕状态常需：所有追捕者到猎物的距离都小于某个上界（比如 `capture_threshold` 的某一比例）。
  - `prey_position` 与 `agent_positions` 相对关系：
    - 通过 `agent_positions[i] - prey_position` 可以得到每个追捕者相对猎物的方向。
    - 方向分布是否均匀决定了包围是否“完整”。

- **间接相关：**
  - `prey_velocity` 与 `agent_velocities`：
    - 如果追捕者速度方向对猎物逃跑方向有“切向夹击”，更有利于围捕。
    - 例如：不同追捕者分布在猎物运动方向前后左右，而不是都在后方追。

#### 与“避免碰撞”相关的变量

- **负向变量：**
  - `inter_agent_distances` 中追捕者-追捕者对的距离：
    - 如果 \(\text{dist}_{ij} < size_i + size_j\) → 两追捕者碰撞 → 违反任务要求。
    - 即，当这部分距离过小 → 应判为不良行为（负相关）。
- **同时还要避免追捕者-猎物的“硬碰撞”**：
  - 取决于任务设计：
    - 若视为允许接触，则可不惩罚甚至奖励；
    - 若强调“包围而不是撞击”，则追捕者-猎物碰撞也应视为负向事件（可能需要另外判断）。

#### 与“均匀包围圈”相关的变量

- **关键变量：**
  - 追捕者相对于猎物的**角度分布**和**距离分布**，可由：
    - `agent_positions` 与 `prey_position` 推导：
      - 向量：`v_i = agent_i_pos - prey_pos`
      - 极坐标：`r_i = ||v_i||`, `theta_i = atan2(v_i_y, v_i_x)`
  - 均匀包围的直觉要求：
    1. 所有 `r_i` 接近某个共同半径 `r*`。
    2. `theta_i` 在 [0, 2π) 上大致均匀分布（例如相差约 2π/3）。

- **相关变量/导出指标：**
  - 从 `distances_to_prey` 可以获得 `r_i`，不含角度信息，但可测“距离一致性”（方差小）。
  - 从 `agent_positions` + `prey_position` 可以计算角度差、扇形覆盖等。

#### 与“时间最短”相关的变量

- 通常与以下变量相关：
  - **时间步 t**：环境内部的 step 计数（这段伪代码中没显式给出，但环境通常有）。
  - **是否已完成围捕**：需自行根据 `global_state` 的距离和拓扑判定。
- 从 `global_state` 本身没有显式“时间”变量，但：
  - 可以在奖励中将“未完成围捕的每一步”视为隐含时间成本（每步给一个负常数）。
  - 只需 `compute_reward` 拿到当前 step 的状态即可，不必依赖额外变量。

---

### 2. 什么样的空间拓扑代表高质量的“包围圈”

从几何角度看，“高质量包围圈”至少包括以下特征（从纯状态角度，不涉及奖励）：

1. **猎物位于包围圈内部（接近中心）**

   根据 `prey_position` 和 `agent_positions`：

   - 求每个追捕者相对于猎物的向量：
     \[
     v_i = p_i - p_{\text{prey}}
     \]
   - 一种简单直观的条件：
     - 所有 `r_i = ||v_i||` 小于某个上界 `R_max`（例如 capture_threshold 的某倍数）。
     - 同时所有追捕者的**凸包**包含猎物位置：
       - 若对追捕者位置求凸包 `Conv({p_i})`，则 `prey_position` 应位于凸包内部或非常接近中心附近。

2. **追捕者与猎物距离适中且相近**

   - `distances_to_prey` 的元素：
     - \(r_i\) 不能太大（否则不算真正围住）。
     - 也不能太小到已经碰撞（除非设计允许贴身）。
   - “适中”可理解为：
     - `r_i` 接近某个理想半径 `r*`：例如 0.3~0.5 之间（取决于 world_size 和 size）。
   - “相近”意味着 `r_i` 的方差较小：
     - 所有追捕者距离猎物相差不大，包围圈近似圆形，而非拉长的线状或偏心形状。

3. **追捕者在角度上均匀分布，避免在同一侧扎堆**

   - 从 `v_i` 可计算每个追捕者的极角 `theta_i`：
     \[
     \theta_i = \text{atan2}(v_{i,y}, v_{i,x})
     \]
   - 对 3 个追捕者的理想分布：
     - 三者角度差接近 120°，即：
       \[
       |\theta_i - \theta_j| \approx 2\pi/3 \quad (\text{mod } 2\pi)
       \]
   - 若所有追捕者都在猎物的一侧（例如角度都在 90° 范围内），则包围不完整，猎物容易从反方向逃离。

4. **追捕者之间的距离既不太近（防止碰撞），也不太远（保证封堵）**

   - 利用 `inter_agent_distances`：
     - 对追捕者-追捕者对 `(i, j)`：
       - 下界：必须大于 `size_i + size_j + margin`（安全间距）；
       - 上界：若两者之间过远，包围圈的某个方向就出现大“缺口”。
   - 通过某种形式的均匀性指标衡量：
     - 各个追捕者-追捕者距离之间的差异应尽量小。
     - 与“理想正三角形”边长接近。

5. **与逃逸方向相关的动态拓扑**

   - 从 `prey_velocity` 看逃逸方向 `u`；
   - 若投影到 `v_i` 与 `u` 的夹角大致等分（有追捕者在前方封堵，有在两侧压迫），包围更牢靠。
   - 虽然这属于动态信息，但也是“高质量围捕”的一部分结构特征。

概括起来，**高质量包围圈**的几何特征是：

- 猎物位于追捕者构成的多边形的内部；
- 所有追捕者到猎物的距离大致相同且在合理范围内；
- 追捕者在猎物周围角度分布均匀（对于 3 个追捕者，近似 120° 等分）；
- 追捕者彼此之间间距适中，不发生碰撞，又不会存在大的空档。

这些特征都可以从 `global_state` 中的 `agent_positions`, `prey_position`, `distances_to_prey`, `inter_agent_distances` 派生出来。

---

## 五、API 边界隔离 (API Boundaries)

根据你的特别说明，必须明确以下**约束与安全边界**：

1. `CoreEnvLogic` 是**文档伪代码**：
   - 它用于表达物理概念与数据结构；
   - 实际运行时**不应**实例化 `CoreEnvLogic()`；
   - 也不应在真实环境中指望有 `world.logic` 之类的属性存在。

2. 在真实的 `compute_reward` 函数中：

   - 禁止：
     - `logic = CoreEnvLogic()` 或类似调用；
     - 使用 `world.logic.world_size`、`world.adversary_params` 等在实际运行 `World` 中不存在的字段。
   - 必须：
     - 通过 PettingZoo/MPE 的标准接口获取当前状态信息：
       - 从 `world.agents` 迭代 agent；
       - 使用 `a.state.p_pos`, `a.state.p_vel`, `a.adversary`, `a.size`, `a.max_speed` 等；
       - 如果某些物理量在 `Agent` 中没有属性（例如 world_size），则在 `compute_reward` 函数内部**以常量硬编码**方式声明。

3. 硬编码物理常量的方式示例（注意：这里只说明风格，不写任何奖励逻辑）：

   ```python
   # 在 compute_reward 内部
   world_size = 2.5
   capture_threshold = 0.5  # = world_size * 0.2
   adv_size = 0.075
   prey_size = 0.050
   adv_max_speed = 1.0
   prey_max_speed = 1.3
   ```

   然后再使用这些常量配合从 `world` 中读取的实时位置、速度数据计算距离、碰撞等。

4. `CoreEnvLogic._build_global_state` 虽然列出了非常便利的结构化 `global_state` 字典，但在真实代码中：

   - 你需要在 `compute_reward` 内部**手动构造**等价或更精简的结构；
   - 或直接在 reward 计算过程中使用 PettingZoo agent 的属性（而不是依赖一个不存在的逻辑对象）。

5. `observation` 函数在真正的场景文件中通常是属于 Scenario 或 Env 的方法：

   - 你的奖励函数不应假定可以调用 `world.logic.observation(agent, world)`；
   - 如果在 reward 中需要类似观测的信息，应直接使用世界状态和 agent 属性重新计算。

总结：

> 所有任务相关的物理常量、追捕者/猎物的 size、max_speed、world_size 等，  
> 必须在 `compute_reward` 中作为局部变量显式定义，  
> 不允许通过 `CoreEnvLogic` 或任何不存在的属性间接获取。

---

以上是对环境模型（Dec-POMDP）在五个维度上的代码级诊断，确保在后续设计奖励函数时，对状态结构、物理属性、拓扑特征和 API 边界有清晰、严格的理解。

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
    # 非追捕者直接返回 0
    if not global_state["is_adversary"]:
        return 0.0, {}

    components = {}

    # ------------------------
    # 物理常量（硬编码）
    # ------------------------
    world_size = 2.5
    capture_threshold = 0.5
    adv_size = 0.075
    prey_size = 0.050

    # 队形与时间相关超参数
    target_radius = capture_threshold * 0.7
    min_radius = capture_threshold * 0.3
    max_radius = capture_threshold * 1.2
    collision_margin = 0.01
    dt_penalty = -0.01

    # 队形权重（可调）
    w_approach = 1.0
    w_ring_radius = 0.5
    w_radius_var = 0.3
    w_angle_uniform = 0.5
    w_safe_sep = 1.0
    w_prey_block = 0.2
    w_capture_bonus = 5.0
    w_capture_all_bonus = 10.0

    # ------------------------
    # 从 global_state 提取信息
    # ------------------------
    agent_positions = global_state["agent_positions"]
    agent_velocities = global_state["agent_velocities"]
    prey_pos = global_state["prey_position"]
    prey_vel = global_state["prey_velocity"]
    distances_to_prey = global_state["distances_to_prey"]
    inter_agent_distances = global_state["inter_agent_distances"]

    all_agents = world.agents
    adversaries = [a for a in all_agents if a.adversary]
    preys = [a for a in all_agents if not a.adversary]

    # 当前智能体索引（在 world.agents 中）
    agent_index = None
    for i, a in enumerate(all_agents):
        if a.name == agent_name:
            agent_index = i
            break
    if agent_index is None:
        return 0.0, {}

    agent = all_agents[agent_index]
    if not agent.adversary:
        return 0.0, {}

    # adversaries 顺序与 distances_to_prey 对应
    try:
        adv_index = adversaries.index(agent)
    except ValueError:
        return 0.0, {}

    agent_pos = agent_positions[agent_index]
    agent_vel = agent_velocities[agent_index]

    # ------------------------
    # 1) 接近猎物（距离引导）
    # ------------------------
    dist_to_prey = np.linalg.norm(agent_pos - prey_pos)
    max_dist = np.sqrt(2) * world_size
    # 距离归一化为 [0, 1]，越近越大
    approach_reward = (max_dist - dist_to_prey) / max_dist
    approach_reward = np.clip(approach_reward, 0.0, 1.0)
    components["approach_reward"] = w_approach * approach_reward

    # ------------------------
    # 2) 环形半径与均匀性（基于所有追捕者）
    # ------------------------
    if len(adversaries) > 0:
        adv_positions = np.array([a.state.p_pos for a in adversaries])
        rel_vecs = adv_positions - prey_pos
        radii = np.linalg.norm(rel_vecs, axis=1)

        # 2.1 个体半径靠近 target_radius
        radius_diff = np.abs(radii[adv_index] - target_radius)
        radius_band = max_radius - min_radius
        if radius_band > 0:
            ring_radius_reward = 1.0 - np.clip(radius_diff / radius_band, 0.0, 1.0)
        else:
            ring_radius_reward = 0.0
        components["ring_radius_reward"] = w_ring_radius * ring_radius_reward

        # 2.2 所有追捕者到猎物距离的方差越小越好
        if len(radii) > 1:
            radius_var = np.var(radii)
            # 用一个平滑映射将方差转成奖励
            radius_var_scale = (max_radius - min_radius) ** 2 + 1e-6
            radius_uniform_reward = np.exp(-radius_var / radius_var_scale)
        else:
            radius_uniform_reward = 0.0
        components["radius_uniform_reward"] = w_radius_var * radius_uniform_reward

        # 2.3 角度均匀分布奖励（仅追捕者间角度结构）
        angles = np.arctan2(rel_vecs[:, 1], rel_vecs[:, 0])
        angles_sorted = np.sort(angles)
        # 将首尾也连起来形成完整环
        angle_gaps = np.diff(angles_sorted)
        angle_gaps = np.append(angle_gaps, (angles_sorted[0] + 2.0 * np.pi) - angles_sorted[-1])
        if len(angle_gaps) > 1:
            ideal_gap = 2.0 * np.pi / len(angles)
            angle_gap_var = np.var(angle_gaps - ideal_gap)
            angle_uniform_reward = np.exp(-angle_gap_var / (np.pi ** 2 + 1e-6))
        else:
            angle_uniform_reward = 0.0
        components["angle_uniform_reward"] = w_angle_uniform * angle_uniform_reward
    else:
        components["ring_radius_reward"] = 0.0
        components["radius_uniform_reward"] = 0.0
        components["angle_uniform_reward"] = 0.0

    # ------------------------
    # 3) 防止碰撞（追捕者-追捕者）
    # ------------------------
    safe_separation_reward = 0.0
    n_agents = len(all_agents)
    agent_pair_size = 2 * adv_size
    safe_dist = agent_pair_size + collision_margin
    for j in range(n_agents):
        if j == agent_index:
            continue
        other = all_agents[j]
        if not other.adversary:
            continue
        d = inter_agent_distances[agent_index, j]
        if d <= agent_pair_size:
            # 硬碰撞，强惩罚
            safe_separation_reward -= 1.0
        else:
            # 距离接近安全距离时小惩罚
            if d < safe_dist:
                safe_separation_reward -= (safe_dist - d) / safe_dist
    components["safe_separation_reward"] = w_safe_sep * safe_separation_reward

    # ------------------------
    # 4) 阻挡猎物逃跑方向（几何封锁）
    # ------------------------
    prey_speed = np.linalg.norm(prey_vel)
    if prey_speed > 1e-6:
        prey_dir = prey_vel / prey_speed
        rel_to_prey = agent_pos - prey_pos
        rel_dist = np.linalg.norm(rel_to_prey)
        if rel_dist > 1e-6:
            rel_dir = rel_to_prey / rel_dist
            cos_block = np.dot(rel_dir, prey_dir)
            # 若追捕者位于逃逸前方，cos_block 接近 1；后方则接近 -1
            block_reward = (cos_block + 1.0) / 2.0
        else:
            block_reward = 0.0
    else:
        block_reward = 0.0
    components["prey_block_reward"] = w_prey_block * block_reward

    # ------------------------
    # 5) 围捕成功奖励（基于全局队形 + 距离）
    # ------------------------
    capture_reward = 0.0
    if len(adversaries) > 0 and len(preys) > 0:
        adv_positions = np.array([a.state.p_pos for a in adversaries])
        rel_vecs = adv_positions - prey_pos
        radii = np.linalg.norm(rel_vecs, axis=1)
        angles = np.arctan2(rel_vecs[:, 1], rel_vecs[:, 0])
        angles_sorted = np.sort(angles)
        angle_gaps = np.diff(angles_sorted)
        angle_gaps = np.append(angle_gaps, (angles_sorted[0] + 2.0 * np.pi) - angles_sorted[-1])

        all_close = np.all((radii > min_radius) & (radii < capture_threshold))
        gap_ok = len(angle_gaps) == 0 or np.max(angle_gaps) < np.pi

        if all_close and gap_ok:
            # 当前追捕者的半径在合理范围内则获得局部 capture 奖励
            if min_radius < radii[adv_index] < max_radius:
                capture_reward += w_capture_bonus

            # 更强的队形整体成功奖励（按所有追捕者平均分配）
            capture_reward += w_capture_all_bonus / max(len(adversaries), 1)
    components["capture_reward"] = capture_reward

    # ------------------------
    # 6) 时间步惩罚（鼓励尽快完成）
    # ------------------------
    components["time_penalty"] = dt_penalty

    total_reward = float(sum(components.values()))
    return total_reward, components
```
```
