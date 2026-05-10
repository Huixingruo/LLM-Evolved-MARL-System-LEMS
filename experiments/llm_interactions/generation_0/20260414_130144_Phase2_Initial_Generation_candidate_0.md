# LLM Interaction Log

- **Generation**: 0
- **Phase**: Phase2_Initial_Generation
- **Candidate Info**: 0
- **Timestamp**: 20260414_130144

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
## 一、实现细节 (Implementation Details)

### 1. 依赖包

代码片段中显式使用的依赖有：

- 标准/数值计算：
  - `numpy as np`：向量运算、距离计算、数组拼接等。
- 强化学习环境框架：
  - `gymnasium`：环境接口（Action/Observation space 等）。
  - `gymnasium.utils.EzPickle`：减少环境对象的序列化/反序列化工作。
- PettingZoo 多智能体环境框架（MPE 子环境）：
  - `pettingzoo.mpe._mpe_utils.core`：
    - `Agent`, `Landmark`, `World`：MPE 的物理世界和实体定义。
  - `pettingzoo.mpe._mpe_utils.scenario.BaseScenario`：用于描述环境场景逻辑。
  - `pettingzoo.mpe._mpe_utils.simple_env.SimpleEnv`, `make_env`：将场景包装成 PettingZoo 环境。
  - `pettingzoo.utils.conversions.parallel_wrapper_fn`：串行/并行环境转换。
- 自定义模块（本项目本地文件）：
  - `from .custom_agents_dynamics import CustomWorld`：自定义世界动力学（替换/扩展默认 `World`）。
  - `from . import reward_function`：可插拔的奖励函数模块（只定义追捕者奖励）。

注意：`CoreEnvLogic` 本身并不使用 `gymnasium`, `EzPickle`, `SimpleEnv`, `make_env`, `parallel_wrapper_fn` 等，只是和它们处于同一文件；真实环境构造逻辑应该在其他类/函数中。

### 2. 外部未定义的变量/对象

在 `CoreEnvLogic` 内部：

- 使用的对象多为函数参数或在类内定义：
  - `agent`, `world`：作为参数传入 `is_collision`, `_build_global_state`, `observation`。
  - `world.agents`：假定 `world` 已经是 MPE 的 `World`/`CustomWorld` 实例。
  - `a.adversary`, `a.state.p_pos`, `a.state.p_vel`, `a.max_speed`：假定 `Agent` 类具备这些属性（符合 MPE 规范）。
- 没有直接引用诸如 `world.logic`, `world.adversary_params`、`world.agent_params` 等属性；这些是在 `CoreEnvLogic.__init__()` 内部定义的独立字典，不属于 `world`。

但是有两个重要的“概念性陷阱”：

1. `CoreEnvLogic` 是“文档伪代码”，不是实际环境的一部分  
   - 不能在真实的 `compute_reward` 里 `CoreEnvLogic()`。
   - 不能假设运行时存在 `world.logic` 这种引用 `CoreEnvLogic` 的字段。

2. 在实际环境中：
   - 如果想用 `world_size`, `capture_threshold`, `size`, `max_speed` 等物理常量，需要在使用这些常量的函数内部 “硬编码” 成局部变量，而不是从 `CoreEnvLogic` 取值、也不能从 `world` 上读取。

总结：从 Python 语义上看，`CoreEnvLogic` 内部没有使用未定义变量；问题在于：在真实环境中，这个类本身“不会存在”，不能依赖它或它的属性。


---

## 二、环境结构 (Environment Structure)

### 1. 全局状态空间 (Global State)

`_build_global_state(self, agent, world)` 返回的 `global_state` 是一个字典，结构为：

```python
{
    'agent_positions': agent_positions,           # shape: (N, 2)
    'agent_velocities': agent_velocities,         # shape: (N, 2)
    'prey_position': prey_pos,                    # shape: (2,)
    'prey_velocity': prey_vel,                    # shape: (2,)
    'distances_to_prey': distances_to_prey,       # shape: (N_adv,)
    'inter_agent_distances': inter_agent_distances,  # shape: (N, N)
    'is_adversary': agent.adversary,              # bool (当前这个 agent 是不是追捕者)
    'world_size': self.world_size,                # float
    'capture_threshold': self.capture_threshold   # float
}
```

逐项物理意义：

1. `agent_positions : np.array(N,2)`  
   - 所有智能体（追捕者 + 逃跑者）的绝对二维坐标 `(x, y)`。
   - 坐标范围大致在 `[-world_size, world_size] = [-2.5, 2.5]` 内。
   - 用于：
     - 计算任意 pair 的距离（防撞、包围几何形状）。
     - 计算追捕者与目标的相对分布。

2. `agent_velocities : np.array(N,2)`  
   - 所有智能体的绝对速度向量 `(vx, vy)`（单位：世界坐标/时间步）。
   - 与 `max_speed` 共同约束动作的物理效果。
   - 用于：
     - 分析追捕者的追击/拦截方向。
     - 以后若考虑速度对奖励的影响（平滑控制、避免抖动等）。

3. `prey_position : np.array(2,)`  
   - 逃跑目标的绝对位置 `(x, y)`。
   - 如果没有 prey，则为零向量 `np.zeros(2)`（通常不会发生，但防御性处理）。
   - 核心任务对象：围捕/接近都围绕此点展开。

4. `prey_velocity : np.array(2,)`  
   - 逃跑者的绝对速度 `(vx, vy)`。
   - 可用于定义拦截策略、预测轨迹、评估“包围圈稳定性”（追捕者相对于猎物运动方向的布局）。

5. `distances_to_prey : np.array(N_adv,)`  
   - 对每一个追捕者 adversary `adv`，计算 `||adv.pos - prey.pos||` 的欧氏距离。
   - N_adv = 追捕者数量 (=3)。
   - 非常关键：
     - 越小 → 越接近目标，正相关于“靠近目标”这一任务要求。
     - 可用来判定是否进入捕获半径/包围半径。

6. `inter_agent_distances : np.array(N,N)`  
   - 任意两个智能体 i, j 之间的欧氏距离 `||pos_i - pos_j||`。
   - 包含追捕者-追捕者、追捕者-逃跑者、逃跑者-逃跑者（后者一般为 0 只有 1 个逃跑者）。
   - 用途：
     - 防碰撞：追捕者之间的距离过小 → 罚。
     - 阵型控制：追捕者间距尽量均匀，在一定范围形成“多边形”包围圈。

7. `is_adversary : bool`  
   - 当前调用 `_build_global_state` 的 `agent` 的类型标记：
     - `True`：追捕者
     - `False`：逃跑者
   - 对奖励函数很有用：可以只对追捕者计算奖励，或者区分不同角色的奖励策略。

8. `world_size : float`  
   - 地图边界尺度（坐标在 `[-world_size, world_size]`）。
   - 用作归一化依据/尺度判断（如判断“相对距离是否足够近”时可使用比例）

9. `capture_threshold : float`  
   - 围捕判定半径；这里设置为 `world_size * 0.2 = 0.5`。
   - 常用于：
     - 判定是否“完成围捕”（例如所有追捕者与目标的距离均小于该阈值且布局合理）。
     - 设计奖励边界：进入此阈值区域给较大正奖励等。

### 2. 观测空间 (Observation)

`observation(self, agent, world)` 返回一个一维向量，由以下部分拼接：

```python
# 1. 自身状态 (归一化)
norm_self_vel = agent.state.p_vel / agent.max_speed    # shape: (2,)
norm_self_pos = agent.state.p_pos / self.world_size    # shape: (2,)

# 2. 其他智能体的相对位置
other_pos = []
other_vel = []
for other in world.agents:
    if other is agent: continue
    rel_pos = (other.state.p_pos - agent.state.p_pos) / self.world_size
    other_pos.append(rel_pos)                          # 每个 shape: (2,)

    if not other.adversary:
        # 只记录非追捕者（即猎物）的速度，且归一化
        other_vel.append(other.state.p_vel / other.max_speed)  # shape: (2,)

# 3. 向量拼接
obs = np.concatenate(
    [norm_self_vel]          # 2 维（自身速度）
    + [norm_self_pos]        # 2 维（自身位置）
    + other_pos              # 对每个“其他智能体”的相对位置，每个 2 维
    + other_vel              # 对每个“其他猎物”的速度，每个 2 维（此处只有 1 个 prey）
)
```

对于当前任务配置（3 追捕者 + 1 逃跑者 = 4 agents）：

- 对任意一个追捕者（adversary）：
  - 其他智能体有：2 个追捕者 + 1 个猎物 → `len(other_pos) = 3`，维度 `3 * 2 = 6`。
  - 其他中的非追捕者只有 1 个猎物 → `len(other_vel) = 1`，维度 `1 * 2 = 2`。
  - 自身速度：2 维。
  - 自身位置：2 维。
  - 观测总维度：`2 (self_vel) + 2 (self_pos) + 6 (other_pos) + 2 (prey_vel) = 12`。

包含的相对信息（对一个追捕者视角）：

1. 自身绝对信息：
   - `norm_self_vel`：自身速度 / `max_speed`，范围约在 [-1, 1]，体现相对运动状态。
   - `norm_self_pos`：自身位置 / `world_size`，范围约在 [-1, 1]，体现其在地图中的绝对位置（归一化）。

2. 其他智能体相对位置：
   - 对每一个其他 agent：`(other_pos - self_pos) / world_size`，二维相对位置向量（归一化）。
   - 这包括：
     - 其他两个追捕者相对自己位置。
     - 猎物相对自己位置。
   - 物理意义：局部视角下的“队友和目标在我周围的哪一个方向、多远”。

3. 目标（猎物）的速度（只对 prey 记录）：
   - `prey_vel / prey.max_speed`。
   - 物理意义：逃跑者当前运动方向和速度大小（归一化），有助于预测目标未来位置、进行拦截。

缺失的信息（对于单 agent）：

- 没有直接给出“其他追捕者”的速度（除非它们也被当作 prey，但这里不是）。  
  → 所以队形/协作必须主要通过相对位置关系，而不是速度协调。


---

## 三、智能体交互 (Agent Interactions)

### 1. 捕食者 vs 猎物的物理属性对比

在 `CoreEnvLogic.__init__` 中给定：

```python
self.adversary_params = {'size': 0.075, 'max_speed': 1.0}
self.agent_params     = {'size': 0.050, 'max_speed': 1.3}
```

通常 MPE 中：

- `adversary` → 追捕者（这里用 `adversary_params`）。
- `agent`（非 adversary）→ 逃跑者/其他角色（这里 `agent_params` 被当成 prey 的参数）。

因此：

1. 尺寸 `size`（碰撞半径）：
   - 追捕者：`size = 0.075`
   - 逃跑者：`size = 0.050`
   - 含义：
     - 追捕者体积更大，更容易碰撞（包括相互撞到）。
     - 在捕获判定中：追捕者的“触碰范围”更大，但同样碰撞风险增加。

2. 最大速度 `max_speed`：
   - 追捕者：`max_speed = 1.0`
   - 逃跑者：`max_speed = 1.3`
   - 含义：
     - 逃跑者比追捕者更快（1.3 > 1.0），因此单追捕者难以直接追上，从而需要协同包围。
     - 强化了协作性：必须通过“抄截、包围”而非简单直线追击。

### 2. “成功捕获（Collision）” 的判定条件

`is_collision(self, agent1_pos, agent1_size, agent2_pos, agent2_size)`：

```python
delta_pos = agent1_pos - agent2_pos
dist = np.sqrt(np.sum(np.square(delta_pos)))
dist_min = agent1_size + agent2_size
return dist < dist_min
```

数学条件：

- 定义：
  - `dist = ||agent1_pos - agent2_pos||`：两者中心点之间的欧氏距离。
  - `dist_min = agent1_size + agent2_size`：两者碰撞体积的半径之和。
- 碰撞条件：
  - `dist < dist_min` 则判定发生碰撞。

物理含义：

- 将智能体看作圆形刚体：
  - `size` 即圆形碰撞体的半径。
- 当两圆心距离小于半径之和时，两个智能体在物理上重叠/接触，视为碰撞（包括追捕者-追捕者、追捕者-逃跑者）。
- 对围捕任务：
  - 追捕者-逃跑者的碰撞可能用作“抓到目标”的硬性结束条件。
  - 追捕者-追捕者的碰撞则是必须避免的（对应任务要求 2：避免碰撞）。


---

## 四、任务相关信息 (Task-relevant Information)

> 你在问题中把这一节写了两次，这里合并回答一次（内容覆盖那两组问题）。

### 1. 围捕任务核心目标与变量的正/负相关性

任务要求回顾：

1. 追捕者需要接近并包围目标  
2. 追捕者之间避免碰撞  
3. 形成均匀的包围圈  
4. 在尽可能短的时间内完成围捕

从 `global_state` 和 `observation` 中可直接利用的关键变量：

1. 与“接近目标”/“围捕目标”高度相关的变量（正/负）：

   - `distances_to_prey[i] = ||adv_i.pos - prey.pos||`
     - 与“接近目标”：
       - 距离越小 → 越接近 → 对“靠近”是负相关变量（距离小好，距离大坏）。
     - 与“确保不太近/形成圈”：
       - 也不能所有追捕者堆在一点上，过小的距离可能意味着集中在同一侧 → 包围质量差。

   - `prey_position`, `agent_positions`：
     - 可以求各追捕者与猎物的相对位置向量 `adv_i.pos - prey.pos`。
     - 可以分析每个追捕者相对于猎物的角度分布 → 与“均匀包围圈”正相关。

   - `capture_threshold`：
     - 用作“任务完成判据”的几何参数：所有追捕者都在 `distances_to_prey < capture_threshold` 的范围内，且角度覆盖均匀 → 完成围捕。

2. 与“避免碰撞”相关的变量：

   - `inter_agent_distances[i, j] = ||pos_i - pos_j||`：
     - 对追捕者-追捕者 pair：
       - 距离越小（接近 `size_i + size_j`）→ 碰撞风险越大。
       - 所以该距离要远离 `0`，与“安全性”正相关。
     - 结合 `adversary_params['size']` 可推导最小安全距离阈值。

   - `is_collision` 函数（概念上）：
     - `dist < size_i + size_j` 为实际碰撞判定。
     - 可以在奖励中强制作为严重惩罚/终局项。

3. 与“均匀包围圈”相关的变量：

   - 利用 `agent_positions` & `prey_position`：
     - 对每个追捕者计算：
       - 向量：`v_i = adv_i.pos - prey.pos`
       - 距离：`r_i = ||v_i||`
       - 角度：`theta_i = atan2(v_i.y, v_i.x)`
     - 期望状态：
       - 三个角度之间的差接近 `120°`（`2π/3`）等分。
       - 各自 `r_i` 接近某个“合适半径”（不太大也不太小，比如略大于捕获阈值的某个范围）。

   - `distances_to_prey`：
     - 均匀包围圈要求 `distances_to_prey` 三者之间差异小（方差小 / 最大差-最小差小）。

4. 与“尽可能短时间完成围捕”相关的变量：

   - 时间步本身在 `global_state` 中没直接给出，但在环境逻辑中是已知的（例如 step count）。
   - 在没有时间变量暴露的前提下，可以通过：
     - 对围捕完成给予一次性大额奖励；
     - 对每一步持续施加小的时间惩罚（常数负奖励）；
     - 从而代理学习到“快点结束比较好”。

   - `distances_to_prey` 及其变化率：
     - 每步距离缩短的程度可以用来衡量“接近进度” → 可作为 shaping 信息。

### 2. 代表高质量“包围圈”的空间拓扑特征

高质量的包围圈至少满足：

1. **几何中心/半径结构**：
   - 以猎物位置 `prey_position` 为圆心。
   - 三个追捕者大致分布在环状区域内（略大于捕获阈值的半径 r）：
     - `distances_to_prey[i] ≈ r`（接近某一目标半径）。
   - 这一层对应“形成圈”而不是“大范围散开”。

2. **角度均匀性（环绕覆盖）**：
   - 记：
     - `v_i = adv_i.pos - prey.pos`
     - `theta_i = atan2(v_i.y, v_i.x)`，排序后的角度为 `θ_(1) ≤ θ_(2) ≤ θ_(3)`。
   - 相邻角度差：
     - `Δ1 = θ_(2) - θ_(1)`
     - `Δ2 = θ_(3) - θ_(2)`
     - `Δ3 = (θ_(1) + 2π) - θ_(3)`（闭合环）
   - 高质量包围圈：
     - `Δ1 ≈ Δ2 ≈ Δ3 ≈ 2π/3`（120° 均分）。
     - 即追捕者围绕目标形成近似正三角形/等分圆周。

3. **半径均匀性（距目标相似）**：
   - `r_i = ||adv_i.pos - prey.pos||`。
   - 高质量包围圈：
     - 极差小：`max(r_i) - min(r_i)` 很小 → 三个追捕者在距离目标上大致处于同一环上。
     - 半径值合理：`r_i` 不应太大（否则圈太远容易让猎物逃出），也不应太小（否则容易与猎物直接碰撞，不是“围住”而是“抓住”）。

4. **不相互碰撞、间距适中**：
   - 对任意追捕者 pair (i, j)：
     - `inter_agent_distances[i, j] > size_i + size_j + margin`，留有安全 margin。
   - 同时也不能太大：
     - 如果三个追捕者离得太远，则可能出现“圈很薄、猎物可从间隙逃跑”的情况。
   - 因此：
     - `inter_agent_distances` 在某个合理区间内（例如 `[d_min_safe, d_max_reasonable]`）。

5. **猎物“被封闭”**（更高阶拓扑视角）：
   - 三个追捕者的位置构成的三角形包含猎物位置：
     - 从几何上可以用“点是否在三角形内部”的测试。
   - 这是“真正围住”的强条件：
     - 即使角度不够完全均匀，只要猎物在三角形内部，理论上也算包围成功。
   - 上述判定可以通过 `agent_positions` 与 `prey_position` 实现。

综合来看，优质包围圈可以用以下指标评价（都基于 `agent_positions` & `prey_position`）：

- 猎物到三追捕者的最小距离 ≤ 目标半径需求且最大距离不超过一定阈值。
- 三追捕者相对于猎物的角度分布接近均匀等分。
- 三追捕者之间两两间距在安全区间。
- 猎物位于追捕者构成的多边形（此处是三角形）内部或很接近其内部区域。


---

## 五、API 边界隔离 (API Boundaries)

围绕你特别强调的约束，总结如下：

1. `CoreEnvLogic` 是“文档伪代码”，**不能**在实际环境中直接使用：
   - 不能在 `compute_reward` 中写 `logic = CoreEnvLogic()`。
   - 不能访问任何基于这个类构建的属性，如 `world.logic`、`world.adversary_params` 等。

2. 物理常量的使用方式：

   - 如果在 `compute_reward` 或其他运行时函数中需要使用：
     - `world_size = 2.5`
     - `capture_threshold = 0.5`（`world_size * 0.2`）
     - `adv_size = 0.075`
     - `prey_size = 0.050`
     - `adv_max_speed = 1.0`
     - `prey_max_speed = 1.3`
   - 必须在该函数内部“硬编码”为局部变量，例如：

     ```python
     def compute_reward(agent, world):
         # 在函数内部显式定义
         world_size = 2.5
         capture_threshold = 0.5
         adv_size = 0.075
         prey_size = 0.050
         adv_max_speed = 1.0
         prey_max_speed = 1.3
         ...
     ```

   - 不能从：
     - `CoreEnvLogic` 实例、类属性中读；
     - `world.adversary_params`, `world.agent_params`, `world.world_size` 等假想属性中读。

3. 可以依赖的环境信息：

   - 在 `compute_reward(agent, world)` 中，**合理依赖** MPE/CustomWorld 提供的标准内容：
     - `world.agents` 列表。
     - 每个 `agent` 的：
       - `agent.adversary`（布尔标记）。
       - `agent.state.p_pos`, `agent.state.p_vel`。
       - `agent.size`, `agent.max_speed`（如果自定义世界类中有设定这些字段；否则仍需硬编码期望值）。
   - 但这些依赖应符合真实环境实现，而不是从伪代码中“推断出”不存在的字段。

4. 状态结构建议：

   - `global_state` 的结构可以被视为“设计参考”：在 `compute_reward` 中可以手动重建类似信息，而不是调用 `_build_global_state`。
   - 即：
     - 通过 `world.agents` 自行构造：
       - 所有 `agent_positions`
       - `prey_position`
       - `distances_to_prey`
       - `inter_agent_distances`
     - 并在 `compute_reward` 内使用这些中间变量进行奖励计算。

5. 本回答的边界确认：

   - 本回答仅对环境结构和物理量进行了诊断分析；
   - 没有编写或建议任何具体的奖励函数实现代码；
   - 所有对奖励的提及均为概念性关系（正/负相关），未以代码形式出现。

如果你需要下一步，我可以在完全遵守上述 API 限制的前提下，先给出奖励设计的数学表达式和分解思路，再在你确认后才写出实际代码。

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
    # 非追捕者奖励恒为 0
    if not global_state["is_adversary"]:
        return 0.0, {}

    components = {}

    # ---------------------------
    # 物理常量（硬编码）
    # ---------------------------
    world_size = 2.5
    capture_threshold = 0.5
    adv_size = 0.075
    prey_size = 0.050

    # 形状与索引辅助
    agent_positions = np.array(global_state["agent_positions"])
    prey_pos = np.array(global_state["prey_position"])
    distances_to_prey = np.array(global_state["distances_to_prey"])
    inter_agent_distances = np.array(global_state["inter_agent_distances"])

    # 追捕者与猎物数量
    n_agents = agent_positions.shape[0]
    # 在 MPE 中 adversary 一般在列表前部；这里我们通过距离数组大小推断追捕者数量
    n_adv = len(distances_to_prey)

    # 找到当前 agent 在 world.agents 中的索引
    agent_index = None
    for idx, a in enumerate(world.agents):
        if a.name == agent_name:
            agent_index = idx
            break
    if agent_index is None:
        return 0.0, components

    agent = world.agents[agent_index]

    # ---------------------------
    # 基本几何量构造
    # ---------------------------
    # 追捕者索引列表（假定为前 n_adv 个）
    adversary_indices = list(range(n_adv))

    # 当前追捕者在 adversary 索引中的位置（若不在其中，则直接返回 0 奖励）
    if agent_index not in adversary_indices:
        return 0.0, components
    local_adv_idx = adversary_indices.index(agent_index)

    # 当前追捕者与猎物的距离
    d_ap = np.linalg.norm(agent.state.p_pos - prey_pos)

    # ---------------------------
    # 1. 距离引导奖励（靠近并保持合适半径）
    # ---------------------------
    # 目标半径：略小于 capture_threshold，鼓励形成稳定包围圈而不是贴脸碰撞
    target_radius = capture_threshold * 0.9
    # 当前追捕者到猎物距离
    distance_error = d_ap - target_radius
    # 使用负的绝对误差作为 shaping，使距离逼近 target_radius
    distance_reward = -abs(distance_error)
    # 额外的全局靠近进度奖励：使用所有追捕者平均距离
    mean_dist_to_prey = float(np.mean(distances_to_prey)) if n_adv > 0 else d_ap
    global_distance_reward = -mean_dist_to_prey

    components["distance_shaping_self"] = 0.7 * distance_reward
    components["distance_shaping_global"] = 0.3 * global_distance_reward

    # ---------------------------
    # 2. 防碰撞奖励（追捕者-追捕者）
    # ---------------------------
    # 安全距离 = 两追捕者半径之和再加一点 margin
    safe_margin = 0.02
    min_safe_dist = 2 * adv_size + safe_margin

    collision_penalty = 0.0
    near_collision_penalty = 0.0

    for j in adversary_indices:
        if j == agent_index:
            continue
        d_aa = inter_agent_distances[agent_index, j]
        # 严重碰撞
        if d_aa < 2 * adv_size:
            collision_penalty -= 5.0
        # 接近碰撞区域
        elif d_aa < min_safe_dist:
            near_collision_penalty -= (min_safe_dist - d_aa)

    components["collision_penalty"] = collision_penalty
    components["near_collision_penalty"] = near_collision_penalty

    # ---------------------------
    # 3. 队形奖励：均匀包围圈
    # ---------------------------
    formation_angle_reward = 0.0
    formation_radius_reward = 0.0
    containment_bonus = 0.0

    if n_adv >= 2 and n_agents >= n_adv + 1:
        # 追捕者相对于猎物的向量与角度
        adv_positions = agent_positions[adversary_indices]
        rel_vecs = adv_positions - prey_pos  # shape: (n_adv, 2)
        rel_dists = np.linalg.norm(rel_vecs, axis=1) + 1e-6
        rel_angles = np.arctan2(rel_vecs[:, 1], rel_vecs[:, 0])

        # 角度均匀性：希望相邻角度差接近 2π/n_adv
        target_delta = 2.0 * np.pi / float(n_adv)
        angles_sorted = np.sort(rel_angles)
        deltas = []
        for i in range(n_adv - 1):
            deltas.append(angles_sorted[i + 1] - angles_sorted[i])
        # 闭合差值
        deltas.append((angles_sorted[0] + 2.0 * np.pi) - angles_sorted[-1])
        deltas = np.array(deltas)
        angle_error = np.mean(np.abs(deltas - target_delta))
        # 负误差作为奖励
        formation_angle_reward = -angle_error

        # 半径均匀性：希望所有追捕者距离猎物相近
        radius_spread = np.max(rel_dists) - np.min(rel_dists)
        formation_radius_reward = -radius_spread

        # 包含性：若猎物在追捕者形成的多边形内则给额外奖励
        # 仅在 n_adv == 3 时使用三角形包含测试
        if n_adv == 3:
            p = prey_pos
            a, b, c = adv_positions

            def _sign(p1, p2, p3):
                return (p1[0] - p3[0]) * (p2[1] - p3[1]) - \
                       (p2[0] - p3[0]) * (p1[1] - p3[1])

            b1 = _sign(p, a, b) < 0.0
            b2 = _sign(p, b, c) < 0.0
            b3 = _sign(p, c, a) < 0.0
            is_inside = (b1 == b2) and (b2 == b3)
            if is_inside:
                containment_bonus = 2.0

    components["formation_angle_reward"] = 0.5 * formation_angle_reward
    components["formation_radius_reward"] = 0.5 * formation_radius_reward
    components["containment_bonus"] = containment_bonus

    # ---------------------------
    # 4. 时间惩罚（鼓励尽快完成）
    # ---------------------------
    # 每一步给一个小的时间成本
    time_penalty = -0.01
    components["time_penalty"] = time_penalty

    # ---------------------------
    # 5. 围捕完成奖励
    # ---------------------------
    capture_bonus = 0.0
    if n_adv > 0:
        # 所有追捕者都在 capture_threshold 内，且角度/半径相对合理
        all_close = bool(np.all(distances_to_prey < capture_threshold))
        if all_close:
            # 使用当前构造的 formation 指标判断质量
            quality_score = (
                -abs(formation_angle_reward) -
                abs(formation_radius_reward)
            )
            capture_bonus = 10.0 + 5.0 * quality_score
    components["capture_bonus"] = capture_bonus

    total_reward = float(sum(components.values()))
    return total_reward, components
```
```
