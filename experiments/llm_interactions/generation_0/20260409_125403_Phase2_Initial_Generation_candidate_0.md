# LLM Interaction Log

- **Generation**: 0
- **Phase**: Phase2_Initial_Generation
- **Candidate Info**: 0
- **Timestamp**: 20260409_125403

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

### 1. 使用了哪些依赖包？

代码仅显式依赖了：

- `numpy`：`import numpy as np`
  - 用于向量/矩阵运算、范数计算、拼接观测向量等。

没有使用其它外部库（如 gym、torch 等）在这段代码中被显式导入。

### 2. 是否引入了外部未定义的变量？

是的，`CoreEnvLogic` 代码显式依赖了一些在当前代码片段中**未定义**但在实际环境中理应存在的对象/属性：

1. `world`
   - 在 `_build_global_state(self, agent, world)` 和 `observation(self, agent, world)` 中被使用：
     - `world.agents`：假设世界对象有一个 `agents` 列表。
   - `world` 的结构在这里没有定义，只能推断为 Multi-Agent World。

2. `agent`
   - 假设了 `agent` 对象至少具有以下属性：
     - `agent.adversary`：布尔值，标识是否为追捕者（adversary）。
     - `agent.state.p_pos`：二维位置向量 `np.array([x, y])`。
     - `agent.state.p_vel`：二维速度向量。
     - `agent.max_speed`：该智能体的最大速度（用于归一化自己的速度）。
   - 对其它智能体 `other` 同样假设有上述属性。

3. `preys` / `adversaries` 中元素
   - `preys[0].state.p_pos`, `preys[0].state.p_vel`：
     - 假设至少有一个 prey；如果列表为空则使用 `np.zeros(2)` 占位。
   - `adversaries`：追捕者列表，元素也 assumed to have `state.p_pos`。

4. 暗含的结构/约定
   - `all_agents = world.agents`：约定所有智能体（追捕者+逃逸者）都在一个列表中。
   - `other.adversary`：在 observation 中用来筛选“非追捕者”（即猎物），并对其速度归一化。

这些外部对象不是 Python 层面的“未定义变量错误”（因为这段代码只是文档伪代码），但在真实运行时：
- 必须由上层环境（如 Multi-Agent Particle Env 的 world/agent 模块）提供。
- 严格按照这些字段命名和结构使用。


---

## 二、环境结构 (Environment Structure)

### 1. 状态空间 (Global State)

`_build_global_state` 返回一个字典，包含如下键及含义：

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

逐项解释：

1. `agent_positions : np.ndarray shape (N, 2)`
   - 含义：所有智能体的绝对位置（world 坐标系），单位是环境长度单位（范围大致在 [-2.5, 2.5]^2）。
   - 结构：`agent_positions[i] = world.agents[i].state.p_pos`。

2. `agent_velocities : np.ndarray shape (N, 2)`
   - 含义：所有智能体的绝对速度向量（world 坐标系），未做归一化。
   - 结构：`agent_velocities[i] = world.agents[i].state.p_vel`。

3. `prey_position : np.ndarray shape (2,)`
   - 含义：唯一逃逸目标（猎物）的绝对位置。
   - 若无猎物（`preys` 为空）则为 `np.zeros(2)` 作为 fallback。
   - 用途：衡量追捕者与猎物的距离、构造包围圈几何关系。

4. `prey_velocity : np.ndarray shape (2,)`
   - 含义：该猎物的绝对速度向量。
   - 若无猎物则为 `np.zeros(2)`。

5. `distances_to_prey : np.ndarray shape (N_adv,)`
   - 含义：每个追捕者到猎物的欧氏距离：
     ```python
     distances_to_prey[k] = || adversaries[k].state.p_pos - prey_pos ||
     ```
   - 物理意义：反映围捕紧密程度，是“接近目标”任务的核心距离指标。

6. `inter_agent_distances : np.ndarray shape (N, N)`
   - 含义：所有智能体两两之间的欧氏距离矩阵：
     ```python
     inter_agent_distances[i, j] = || agent_positions[i] - agent_positions[j] ||
     ```
   - 对角线 `i == j` 时为 0。
   - 物理意义：
     - 用于碰撞检测（是否小于各自 size 之和）。
     - 用于衡量追捕者之间的间隔是否均匀（包围圈均匀性）。

7. `is_adversary : bool`
   - 含义：当前调用 `_build_global_state` 的这个 `agent` 是否为追捕者。
   - 用途：
     - 在 reward 设计中可以根据身份区分：追捕者与猎物有不同回报结构。

8. `world_size : float`
   - 含义：环境的空间尺度上限（半边长），总地图范围为 `(-2.5, 2.5)`。
   - 用途：
     - 归一化位置、设定边界惩罚等。

9. `capture_threshold : float`
   - 含义：围捕判定阈值距离（= `world_size * 0.2 ≈ 0.5`）。
   - 物理意义：
     - 可以作为“追捕成功/包围成功”的空间尺度，用于 reward 中判断是否进入“包围区域”。

整体上，这个 global_state 把**绝对位置/速度**、**相对距离矩阵**、**与猎物的距离**、以及必要的全局常量全部暴露出来，适合在 `compute_reward` 里进行几何与拓扑分析。

---

### 2. 观测空间 (Observation)

`observation(self, agent, world)` 生成的是**单个智能体的局部观测向量**，定义为：

```python
# 返回 shape: (2) + (2) + 2*(N-1) + 2*(#prey_of_type) 具体取决于 adversary/prey 数量
return np.concatenate([norm_self_vel] + [norm_self_pos] + other_pos + other_vel)
```

逐步拆解：

1. 自身速度（归一化）
   ```python
   norm_self_vel = agent.state.p_vel / agent.max_speed
   ```
   - 维度：`(2,)`
   - 含义：
     - 当前智能体的速度向量，除以自身最大速度 `agent.max_speed`，得到 [-1, 1] 左右的无量纲速度。
     - 对追捕者（adversary）和猎物各自用自己的 `max_speed` 归一。

2. 自身位置（归一化）
   ```python
   norm_self_pos = agent.state.p_pos / self.world_size
   ```
   - 维度：`(2,)`
   - 含义：
     - 智能体绝对位置除以 `world_size=2.5`，可约在 [-1, 1] 范围。
     - 表示自己在地图中的绝对相对位置。

3. 其他智能体的相对位置
   ```python
   other_pos = []
   for other in world.agents:
       if other is agent: continue
       rel_pos = (other.state.p_pos - agent.state.p_pos) / self.world_size
       other_pos.append(rel_pos)
   ```
   - 对每一个其他智能体（无论是追捕者还是猎物）：
     - 计算 `other_pos_i = (p_other - p_self) / world_size`。
   - 维度：每个 `rel_pos` 是 `(2,)`，若总智能体数为 `N`，当前 agent 观测到 `N-1` 个 other，则 `other_pos` 拼接后为 `(2*(N-1),)`。
   - 含义：
     - 相对位置，归一化到 [-2, 2] 左右（通常）。
     - 这是局部坐标系下的“谁在我周围”的几何信息。

4. 其他智能体（非追捕者，即猎物）的速度
   ```python
   other_vel = []
   for other in world.agents:
       ...
       if not other.adversary:
           other_vel.append(other.state.p_vel / other.max_speed)
   ```
   - 仅对 `other.adversary == False` 的智能体（即猎物）记录速度。
   - 速度归一化方式：除以其 `other.max_speed`。
   - 维度：若世界里有 1 个猎物，则 `other_vel` 长度为 1，维度为 `(2,)`。若有多个猎物，则 `(2 * #prey,)`。
   - 含义：
     - 当前 agent 可以感知到猎物的速度方向和大小（无量纲）。

5. 最终观测向量拼接
   - 顺序为：
     1. `[norm_self_vel]` : `(2,)`
     2. `[norm_self_pos]` : `(2,)`
     3. `other_pos` : `(2*(N-1),)`
     4. `other_vel` : `(2 * #prey,)`
   - 对你这个任务（3 追捕 + 1 猎物，N=4）：
     - 对追捕者的观测（adversary agent）：
       - 自身速度 `(2,)`
       - 自身位置 `(2,)`
       - 其他 3 个智能体的相对位置 `(3 * 2 = 6,)`
       - 所有猎物（仅 1 个）的速度 `(2,)`
       - 总长度：`2 + 2 + 6 + 2 = 12`
     - 对猎物的观测：
       - 自身速度 `(2,)`
       - 自身位置 `(2,)`
       - 其他 3 个（都是追捕者）的相对位置 `(6,)`
       - `other_vel` 中只有非 adversary 才会加入，猎物视角下“其他”都是追捕者，因此 `other_vel` 为空 `(0,)`
       - 总长度：`2 + 2 + 6 = 10`

   - 重要特征：观测是**局部的**（以自身为原点的相对坐标），无角色标签（只能从相对速度/位置推断谁是猎物/追捕者，或通过行为模式区分）。


---

## 三、智能体交互 (Agent Interactions)

### 1. 物理属性对比（max_speed, size）

在 `__init__` 中给定：

```python
self.adversary_params = {'size': 0.075, 'max_speed': 1.0}
self.agent_params     = {'size': 0.050, 'max_speed': 1.3}
```

这里隐含命名约定：

- “adversary”: 追捕者（predator）
- “agent” (非 adversary): 猎物（prey）

对比：

1. 尺寸（碰撞半径 size）
   - 追捕者：`size = 0.075`
   - 猎物：`size = 0.050`
   - 解释：
     - 追捕者体积更“大”，在几何上更容易与他人发生碰撞。
     - 捕获判定时，“碰到”的距离阈值是两个 size 之和，所以追捕者更容易碰到目标，但也更容易彼此碰撞。

2. 最大速度（max_speed）
   - 追捕者：`max_speed = 1.0`
   - 猎物：`max_speed = 1.3`
   - 解释：
     - 猎物比追捕者更快（1.3 > 1.0）。
     - 围捕任务需要三名追捕者通过协同策略（如包夹/合围），而不是单纯依靠速度追上猎物。
     - 这也是一个典型的协作捕猎 MARL 场景：单个追捕者难以追上，更依赖团队围堵的几何结构。

### 2. “成功捕获（Collision）”的条件

碰撞检测函数：

```python
def is_collision(self, agent1_pos, agent1_size, agent2_pos, agent2_size):
    """碰撞检测：欧氏距离 < 半径之和"""
    delta_pos = agent1_pos - agent2_pos
    dist = np.sqrt(np.sum(np.square(delta_pos)))
    dist_min = agent1_size + agent2_size
    return dist < dist_min
```

数学条件：

- 设两个智能体的位置为 `p1, p2`，半径为 `r1, r2`，则发生碰撞当且仅当：

  \[
  \| p_1 - p_2 \|_2 < r_1 + r_2
  \]

- 对当前任务最关键的情况：
  - 追捕者与猎物碰撞：
    - `r_adv = 0.075`
    - `r_prey = 0.050`
    - 判定条件：`||p_adv - p_prey|| < 0.125`
  - 追捕者之间碰撞：
    - `r_adv = 0.075`（两者相同）
    - 判定条件：`||p_adv_i - p_adv_j|| < 0.15`

物理含义：

- 一旦追捕者与猎物距离小于 0.125，就可判定为“捕获碰撞”。
- 对追捕者之间，相距不足 0.15 视为“互撞”（应在 reward 中惩罚）。
- 捕获仅由**几何距离**决定，不直接与 `capture_threshold (0.5)` 绑定；
  - `capture_threshold` 更像是“包围成功”的空间尺度（如都在某半径内），而不是几何碰撞阈值。

---

## 四、任务相关信息 (Task-relevant Information)

### 1. 围捕任务核心目标与变量的关系

任务要求：

1. 追捕者需要接近并包围目标
2. 追捕者之间避免碰撞
3. 形成均匀的包围圈
4. 在尽可能短的时间内完成围捕

结合 global_state 中的变量，核心目标与变量间的关系可以分解如下：

#### (1) 接近目标 & 快速围捕

主要相关变量：

- `distances_to_prey`：每个追捕者到猎物的距离
- 衍生变量：`min(distances_to_prey)`, `mean(distances_to_prey)`, `max(distances_to_prey)`
- `capture_threshold`：可用于定义“足够接近”的半径（如所有追捕者进入该半径内）
- 时间因素：在 reward 中通过每步惩罚时间消耗或对早期成功给予更大回报实现（不在当前代码中体现，但在设计中必须考虑）

符号关系：

- 与接近目标目标**负相关**的量：
  - `distances_to_prey[k]` 越小越好 ⇒ 惩罚距离或奖励距离减小。
- 时间最小化：
  - 间接：通过在每步施加固定负奖励（时间惩罚），促使智能体更快完成围捕。
  - 直接：在捕获成功瞬间给非常大的终止奖励，使策略偏向早收敛。

#### (2) 避免碰撞

主要相关变量：

- `inter_agent_distances`：智能体间两两距离矩阵
- 智能体尺寸：`adv_size = 0.075`, `prey_size = 0.050`

符号关系：

- 与“安全”的**正相关量**：追捕者之间的距离应大于 `r_i + r_j`。
- 与“应惩罚的事件”**直接相关**：
  - 对任意两个追捕者 i, j：
    - 若 `inter_agent_distances[i, j] < r_adv_i + r_adv_j` ⇒ 发生碰撞 ⇒ 大负奖励。
  - 对追捕者与猎物，此碰撞可以同时代表“捕获成功”和“错误的硬碰撞”，如何解读需根据任务定义：
    - 如果任务希望“柔性包围（距离阈值内即可）而不是物理撞上”，则需要区分几何包围成功 vs 真正碰撞。

#### (3) 均匀包围圈的形成

主要相关变量：

- `agent_positions`
- `prey_position`
- `inter_agent_distances`（尤其是追捕者之间）

可以构造如下几何量：

- 每个追捕者到猎物的向量：
  \[
  v_k = p_{adv,k} - p_{prey}
  \]
- 角度分布：
  - 计算 `angle_k = atan2(v_k_y, v_k_x)`，排序后看相邻角度差是否接近 `2π/3`。
- 距离分布：
  - `distances_to_prey` 理想情况下在一个小范围内波动，并彼此接近。
- 追捕者之间的距离：
  - 相邻追捕者的距离在同一环带上尽量相近，避免聚集成一堆。

符号关系（高质量包围圈）：

- **半径一致性**：所有追捕者距离猎物的距离接近某目标半径 `R`，且 `R` 小于 `capture_threshold`。
  - 即：`var(distances_to_prey)` 尽量小（均匀），`mean(distances_to_prey) < capture_threshold`。
- **角度均匀性**：三个追捕者在猎物周围的极角相差接近 `120°`。
  - 相邻角度差：接近 `2π/3`。
- **避免彼此过近**：追捕者之间的最小距离不小于某安全阈（略大于 `2*adv_size`）。

#### (4) 时间最短化

在当前代码中尚未显式有时间变量，但在 reward 中可通过如下方式引入时间：

- 使用每步固定时间惩罚：`r_t += -c`（c>0）。
- 或在成功围捕时给予终止奖励，训练中会自然偏向早结束。

与状态变量的直接关系较弱，时间更是通过“episode step count”体现。


### 2. 什么样的空间拓扑代表高质量“包围圈”？

高质量的包围圈可以从几何/拓扑的角度描述为以下条件组合（对 3 个追捕者）：

1. **猎物被包含在一个小半径的“捕获区域”内**：
   - 对所有追捕者 k：
     \[
     d_k = \| p_{adv,k} - p_{prey} \| \le R_{capture}
     \]
   - 通常取 `R_capture ≤ capture_threshold = 0.5`。
   - 可以要求 `mean(d_k)` 远小于 `capture_threshold` 以保证紧密围捕。

2. **追捕者在猎物周围角度基本均匀分布**：
   - 对每个追捕者：
     - 计算相对向量：`v_k = p_{adv,k} - p_{prey}`；
     - 计算极角：`θ_k = atan2(v_k_y, v_k_x)`；
   - 排序后，3 个角之间的差应接近 `2π/3`：
     \[
     |\Delta θ_k - 2π/3| \text{ 都很小}
     \]
   - 当角度分布均匀时，猎物在任何方向上的“逃离空间”都被不同追捕者封锁。

3. **追捕者之间距离适中且近似相等**：
   - 追捕者之间的三条边形成一个近似等边三角形：
     - 三个 pairwise distance：
       \[
       d_{12}, d_{23}, d_{31}
       \]
       满足：
       - 相互之间差异很小 (`|d_ij - d_kl|` 小)；
       - 各自在一个合理范围内：大于安全距离 `> 2*adv_size`，小于一个最大半径（避免离得太散，从而留空隙）。

4. **猎物处于追捕者形成的多边形内部**：
   - 在严格几何上，可以测试猎物位置是否在追捕者位置形成的三角形内部：
     - 利用点在三角形内的 barycentric test 或向量叉积符号一致性。
   - 这从拓扑上保证猎物确实被“包在里面”，而不是只是在一边被三人挤在同一侧。

若仅使用 global_state 中已有的变量：

- 可以通过 `agent_positions` + `prey_position` 实现上述所有几何条件；
- 用 `distances_to_prey`、`inter_agent_distances` 的组合可以构造：
  - 半径一致性：`var(distances_to_prey)`
  - 环周边均匀度（非严格，但可近似）：`var` of some function of `inter_agent_distances` among adversaries。


---

## 五、API 边界隔离 (API Boundaries)

根据你的说明，必须严格遵守以下约束：

1. `CoreEnvLogic` 是文档伪代码，不是运行时真实类：
   - 在真正的 `compute_reward` 中：
     - **禁止**：`logic = CoreEnvLogic()` 或任何 `CoreEnvLogic()` 实例化调用。
     - **禁止**：假定 `world.logic` 或 `world.adversary_params` 存在。
   - 这些类/属性只用于概念说明，不可依赖。

2. 需要的物理常量必须在 `compute_reward` 内部**硬编码为局部变量**：
   - 例如：
     ```python
     world_size = 2.5
     adv_size = 0.075
     adv_max_speed = 1.0
     prey_size = 0.050
     prey_max_speed = 1.3
     capture_threshold = 0.5  # 或 world_size * 0.2
     ```
   - 不应从 `CoreEnvLogic` 或 `world` 中读取这些常量。

3. 对 global_state 的假设：
   - `compute_reward(agent, world, global_state)` 中：
     - 假定 `global_state` 已由类似 `_build_global_state` 的逻辑构造好。
     - 只能使用传入的 `global_state` 字典键：
       - `'agent_positions'`, `'agent_velocities'`, `'prey_position'`, `'prey_velocity'`,
         `'distances_to_prey'`, `'inter_agent_distances'`, `'is_adversary'`, `'world_size'`, `'capture_threshold'`。
   - 不应该在 reward 函数中去访问 `world.logic.capture_threshold` 之类的字段。

4. 在 reward 设计时，所有用到的常量（例如判定碰撞、防撞阈值、包围半径、时间惩罚系数、角度均匀性权重等）都应该是：
   - 在 `compute_reward` 函数顶部显式声明的**局部标量/常量**；
   - 与环境构造代码解耦，确保 reward 计算仅依赖 `global_state` 的“观测/状态”部分和自身的常数。

5. 当前回答仅做诊断与结构分析：
   - 按要求：**不编写任何奖励函数代码**；
   - 上述对变量和空间拓扑的分析只是为未来 reward 设计做准备，不涉及具体 reward 实现。

---

总结：  
- 代码只依赖 `numpy`，但强依赖外部的 `world` 和 `agent` 结构。  
- global_state 提供全局几何与物理信息；observation 提供局部相对位置/速度信息。  
- 追捕者比猎物更大但更慢，碰撞通过“欧氏距离 < 半径和”判定。  
- 围捕质量与 `distances_to_prey`、`inter_agent_distances`、以及追捕者相对猎物的角度分布直接相关。  
- 真正实现 reward 时必须在函数内部硬编码物理常量，不能实例化 `CoreEnvLogic` 或使用不存在的 `world` 属性。

# 环境代码参考
```python

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
    # 仅对追捕者计算奖励
    if not global_state['is_adversary']:
        return 0.0, {}

    components = {}

    # -----------------------
    # 物理与任务超参数（硬编码）
    # -----------------------
    world_size = 2.5
    adv_size = 0.075
    prey_size = 0.050
    capture_threshold = 0.5

    # 包围圈几何目标
    desired_ring_radius = 0.35
    ring_radius_tolerance = 0.20

    # 安全距离相关
    adv_adv_safe_margin = 0.02
    adv_prey_collision_margin = 0.0

    # 奖励权重
    w_distance = 1.0
    w_ring_radius = 1.0
    w_angle_uniform = 1.0
    w_triangle_containment = 1.0
    w_collision_penalty = 5.0
    w_time_penalty = 0.01
    w_success_bonus = 10.0

    # --------------------------------
    # 从 global_state 解析基础变量
    # --------------------------------
    agent_positions = global_state['agent_positions']  # shape (N, 2)
    prey_position = global_state['prey_position']      # shape (2,)
    inter_agent_distances = global_state['inter_agent_distances']
    distances_to_prey = global_state['distances_to_prey']  # adversaries only

    # 追捕者索引（假定 world.agents 与 global_state 顺序一致）
    adversary_indices = [i for i, a in enumerate(world.agents) if a.adversary]
    prey_indices = [i for i, a in enumerate(world.agents) if not a.adversary]

    if len(prey_indices) == 0 or len(adversary_indices) == 0:
        return 0.0, {}

    prey_idx = prey_indices[0]

    # 当前追捕者在 world.agents 中的索引
    try:
        this_adv_idx = [i for i, a in enumerate(world.agents)
                        if a.name == agent_name][0]
    except (AttributeError, IndexError):
        # 若找不到对应索引，返回零奖励
        return 0.0, {}

    # 追捕者在 global_state['distances_to_prey'] 中的索引
    try:
        adv_idx_in_list = adversary_indices.index(this_adv_idx)
    except ValueError:
        return 0.0, {}

    # -----------------------
    # 距离引导：接近猎物
    # -----------------------
    # 使用所有追捕者到猎物距离的平均值作为全局协同目标
    mean_dist_to_prey = np.mean(distances_to_prey)
    # 归一化距离（防止尺度过大），world_size 为最大距离尺度
    norm_mean_dist = mean_dist_to_prey / world_size
    distance_reward = -norm_mean_dist
    components['distance_reward'] = w_distance * distance_reward

    # -----------------------
    # 包围圈：半径一致性 + 均匀包围
    # -----------------------
    # 只在有 3 个追捕者时使用完整几何约束
    ring_radius_reward = 0.0
    angle_uniform_reward = 0.0
    triangle_containment_reward = 0.0

    if len(adversary_indices) == 3:
        adv_positions = agent_positions[adversary_indices]  # (3, 2)
        prey_pos = prey_position

        # 半径一致性：所有追捕者到猎物的距离接近 desired_ring_radius
        radii = np.linalg.norm(adv_positions - prey_pos, axis=1)
        ring_radius_error = np.abs(radii - desired_ring_radius)
        # 将误差裁剪在 [0, ring_radius_tolerance] 内并归一化
        norm_radius_error = np.clip(ring_radius_error / ring_radius_tolerance,
                                    0.0, 1.0)
        # 取平均误差作为整体度量
        mean_norm_radius_error = np.mean(norm_radius_error)
        ring_radius_reward = -mean_norm_radius_error
        components['ring_radius_reward'] = w_ring_radius * ring_radius_reward

        # 角度均匀性：3 个追捕者在猎物周围的角度差接近 2π/3
        rel_vecs = adv_positions - prey_pos
        angles = np.arctan2(rel_vecs[:, 1], rel_vecs[:, 0])
        # 排序后计算相邻角度差，并考虑首尾闭合
        angles_sorted = np.sort(angles)
        diffs = np.diff(angles_sorted)
        last_gap = (angles_sorted[0] + 2.0 * np.pi) - angles_sorted[-1]
        gaps = np.concatenate([diffs, [last_gap]])  # 共 3 个 gap
        ideal_gap = 2.0 * np.pi / 3.0
        angle_errors = np.abs(gaps - ideal_gap)
        # 归一化：最大合理误差设为 ideal_gap，本身即 120°
        norm_angle_errors = np.clip(angle_errors / ideal_gap, 0.0, 1.0)
        mean_norm_angle_error = np.mean(norm_angle_errors)
        angle_uniform_reward = -mean_norm_angle_error
        components['angle_uniform_reward'] = \
            w_angle_uniform * angle_uniform_reward

        # 三角形包含性：猎物在追捕者构成三角形内部
        p0, p1, p2 = adv_positions[0], adv_positions[1], adv_positions[2]
        v0 = p2 - p0
        v1 = p1 - p0
        v2 = prey_pos - p0

        dot00 = np.dot(v0, v0)
        dot01 = np.dot(v0, v1)
        dot02 = np.dot(v0, v2)
        dot11 = np.dot(v1, v1)
        dot12 = np.dot(v1, v2)
        denom = dot00 * dot11 - dot01 * dot01

        if denom > 1e-8:
            inv_denom = 1.0 / denom
            u = (dot11 * dot02 - dot01 * dot12) * inv_denom
            v = (dot00 * dot12 - dot01 * dot02) * inv_denom
            inside = (u >= 0.0) and (v >= 0.0) and (u + v <= 1.0)
        else:
            inside = False

        triangle_containment_reward = 1.0 if inside else 0.0
        components['triangle_containment_reward'] = \
            w_triangle_containment * triangle_containment_reward
    else:
        components['ring_radius_reward'] = 0.0
        components['angle_uniform_reward'] = 0.0
        components['triangle_containment_reward'] = 0.0

    # -----------------------
    # 防碰撞：追捕者-追捕者
    # -----------------------
    collision_penalty = 0.0
    adv_adv_collision_radius = 2.0 * adv_size + adv_adv_safe_margin

    for i in adversary_indices:
        for j in adversary_indices:
            if i >= j:
                continue
            if inter_agent_distances[i, j] < adv_adv_collision_radius:
                collision_penalty -= 1.0

    # 追捕者-猎物硬碰撞（可以视为不期望的撞击，仍惩罚）
    adv_prey_collision_radius = adv_size + prey_size + adv_prey_collision_margin
    for i in adversary_indices:
        if inter_agent_distances[i, prey_idx] < adv_prey_collision_radius:
            collision_penalty -= 1.0

    components['collision_penalty'] = w_collision_penalty * collision_penalty

    # -----------------------
    # 时间惩罚：鼓励尽快完成任务
    # -----------------------
    time_penalty = -w_time_penalty
    components['time_penalty'] = time_penalty

    # -----------------------
    # 成功围捕 Bonus（所有追捕者在 capture_threshold 内）
    # -----------------------
    all_within_capture = (np.all(distances_to_prey <= capture_threshold) and
                          len(adversary_indices) > 0)

    success_bonus = 0.0
    if all_within_capture:
        success_bonus = w_success_bonus

    components['success_bonus'] = success_bonus

    total_reward = sum(components.values())
    return float(total_reward), components
```
```
