# LLM Interaction Log

- **Generation**: 0
- **Phase**: Phase2_Initial_Generation
- **Candidate Info**: 3
- **Timestamp**: 20260409_123822

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

当前代码只显式依赖了一个包：

- `numpy`，导入方式为 `import numpy as np`  
  主要用途：
  - 向量与矩阵运算（`np.array`, `np.sqrt`, `np.sum`, `np.square`, `np.linalg.norm`）
  - 构建距离矩阵（`np.zeros((n_agents, n_agents))`）
  - 观测向量拼接（`np.concatenate`）

其他所有使用到的类型和名称（如 `world`, `agent`, `agent.state` 等）都来自于环境框架本身（例如类似 MPE/MADDPG 中的 `World`, `Agent`），在当前代码片段中未定义。

### 2. 外部未定义变量 / 结构

在该核心逻辑类中，有若干名称依赖于外部环境框架：

1. `world`
   - 预期是一个包含多个智能体的环境对象。
   - 被假设拥有属性：
     - `world.agents`: 可迭代的智能体列表（追捕者 + 逃跑者）。

2. `agent`
   - 预期是单个智能体对象。
   - 被假设拥有属性：
     - `agent.adversary`: bool，区分追捕者（True）和逃跑者（False）。
     - `agent.state.p_pos`: 位置向量，形状应为 `(2,)`（二维平面位置）。
     - `agent.state.p_vel`: 速度向量，形状应为 `(2,)`。
     - `agent.max_speed`: 最大速度标量。
   - 注意：`agent.max_speed` 是从外部注入的，虽然 `CoreEnvLogic` 中有 `self.adversary_params` / `self.agent_params`，但此处并没有把这两组参数直接绑定到 `agent` 对象上，所以真正用的是外部框架里 agent 的 `max_speed`。

3. `world.agents` 中元素的约定
   - 代码中使用：
     - `adversaries = [a for a in all_agents if a.adversary]`
     - `preys = [a for a in all_agents if not a.adversary]`
   - 表示所有智能体都必须有 `adversary` 属性，否则会出错。

4. 维度假设
   - 所有 `state.p_pos`、`state.p_vel` 被假设为二维向量（2D 环境）。
   - `agent.max_speed`、`world_size`、`capture_threshold` 等为标量。

5. 参数未直接使用的部分
   - `self.adversary_params` 和 `self.agent_params` 定义了 `size` 与 `max_speed`，但在给出的片段中仅 `size` 用于碰撞检测时传入 `is_collision` 的参数；`max_speed` 没有从这里直接作用到 `agent`，说明真正的智能体物理参数是在别处注入/同步的。

结论：  
除 `numpy` 外，其余类型和变量（`world`, `agent`, `agent.state` 等）均来自外部环境实现，必须在框架中保证这些属性存在且形状/类型符合假设。


---

## 二、环境结构 (Environment Structure)

### 1. 状态空间 (Global State) 的内容与物理意义

`_build_global_state` 返回的字典包含以下键：

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

逐项解析：

1. `agent_positions`
   - 类型：`np.array`，形状约为 `(n_agents, 2)`。
   - 含义：所有智能体的**绝对位置**（包含 3 个追捕者 + 1 个逃跑者等）。
   - 物理意义：
     - 在以原点为中心、边界为 `[-world_size, world_size]` 的二维平面上的坐标。
     - 对任务而言，这是全局空间拓扑的基础：可用于判断围捕形状、角度分布等。

2. `agent_velocities`
   - 类型：`np.array`，形状约为 `(n_agents, 2)`。
   - 含义：所有智能体的**绝对速度**向量。
   - 物理意义：
     - 速度方向和大小（未归一化），用于分析运动趋势、加减速、捕猎逼近动态等。

3. `prey_position`
   - 类型：`np.array`，形状 `(2,)`。
   - 含义：逃跑者（猎物）当前位置。
   - 物理意义：
     - 围捕的空间中心（围捕圆心）。
     - 计算与追捕者的相对位置、距离、夹角等的基准点。

4. `prey_velocity`
   - 类型：`np.array`，形状 `(2,)`。
   - 含义：逃跑者的当前速度向量。
   - 物理意义：
     - 表征逃跑方向和速度，任务中可用于设计预测/拦截型策略（例如先走到前方堵截）。

5. `distances_to_prey`
   - 类型：`np.array`，形状约为 `(n_adversaries,)`。
   - 含义：**每个追捕者到猎物的欧氏距离**。
   - 物理意义：
     - 距离越小，说明追捕者越接近目标。
     - 可用于判断是否接近围捕阈值 `capture_threshold`。
     - 也是衡量“收缩围捕圈”的直接指标。

6. `inter_agent_distances`
   - 类型：`np.array`，形状 `(n_agents, n_agents)`。
   - 含义：所有智能体之间的**成对欧氏距离矩阵**。
   - 物理意义：
     - 对角线为 0（自己到自己的距离）。
     - 追捕者–追捕者之间距离：用于碰撞安全、形成一定间距的包围圈。
     - 追捕者–逃跑者之间距离：是 `distances_to_prey` 的冗余信息，但更完整（包含所有组合）。
     - 可以从中推导出队形结构（如是否均匀分布、是否挤在同一侧等）。

7. `is_adversary`
   - 类型：`bool`。
   - 含义：当前调用 `_build_global_state` 的 `agent` 是否为追捕者。
   - 物理意义：
     - 让 reward 函数或上层逻辑区分“这个 global_state 是给谁用的”（追捕者视角 / 逃跑者视角）。
     - 在集中式训练但去中心化执行的 MARL 中，可能会用到此标识来做角色区分。

8. `world_size`
   - 类型：`float`，值为 `2.5`。
   - 含义：环境边界尺度，地图范围为 `(-2.5, 2.5)`。
   - 物理意义：
     - 位置坐标的绝对尺度参考。
     - 有助于按比例设计阈值（如 `capture_threshold = world_size * 0.2`）。

9. `capture_threshold`
   - 类型：`float`，值为 `0.5`。
   - 含义：围捕判定的距离尺度阈值（相当于捕获半径）。
   - 物理意义：
     - 表示“追捕者到猎物的距离小于多少，可以认为已经围捕成功的几何区域尺度”。
     - 对围捕成功条件、终止判定或奖励高峰区域有直接关联。

整体来看：  
`global_state` 提供了**完整、集中式的几何信息与基本物理量**（位置 / 速度 / 距离矩阵），足够支撑复杂的队形、包围圈质量和安全距离等指标的计算。

---

### 2. 观测空间 (Observation) 的结构与相对信息

`observation(self, agent, world)` 返回的观测向量为：

```python
np.concatenate([norm_self_vel] + [norm_self_pos] + other_pos + other_vel)
```

按顺序拆解：

1. `norm_self_vel`
   - 定义：`agent.state.p_vel / agent.max_speed`
   - 形状：`(2,)`
   - 含义：
     - 智能体自身的速度向量，按其最大速度归一化到 [-1, 1] 左右范围。
   - 物理意义：
     - 当前机动状态（走向哪里、走多快），与自身能力范围对比。

2. `norm_self_pos`
   - 定义：`agent.state.p_pos / self.world_size`
   - 形状：`(2,)`
   - 含义：
     - 智能体自身在地图中的绝对位置，按世界边界归一化到 [-1, 1]。
   - 物理意义：
     - 表征“我在整个地图的哪个位置”（靠近边界 / 靠近中心）。

3. `other_pos`
   - 构造过程：
     ```python
     other_pos = []
     for other in world.agents:
         if other is agent: continue
         rel_pos = (other.state.p_pos - agent.state.p_pos) / self.world_size
         other_pos.append(rel_pos)
     ```
   - 每个 `rel_pos` ：形状 `(2,)`。
   - 最终 `other_pos` 是一个长度为 `n_agents - 1` 的列表，每个元素是**相对位置**。
   - 含义：
     - **所有其他智能体相对于当前智能体的相对位置**，除以 `world_size` 归一化。
   - 物理意义：
     - 这是当前智能体所能感知的关键局部几何信息：
       - 追捕者之间的相对位置 → 队形、自身在队形中的相对位置。
       - 对逃跑者的相对位置 → 朝向目标、距离目标。
     - 注意：不同类型（追捕者 / 逃跑者）的相对位置没有区分标签，排序依赖于 `world.agents` 的遍历顺序。

4. `other_vel`
   - 构造过程：
     ```python
     other_vel = []
     for other in world.agents:
         if other is agent: continue
         ...
         if not other.adversary:
             other_vel.append(other.state.p_vel / other.max_speed)
     ```
   - 只对 `not other.adversary`（即逃跑者）收集速度。
   - 在当前任务设定中：只有 1 个逃跑者，因此：
     - 对任意追捕者来说：
       - `other_pos` 中包含：2 个其他追捕者 + 1 个逃跑者的相对位置，共 3×2 维。
       - `other_vel` 中只包含：1 个逃跑者的速度，共 2 维。
   - 含义：
     - 逃跑者的速度向量，归一化到 [-1, 1]。
   - 物理意义：
     - 提供目标的运动趋势信息，方便追捕者进行预测性移动（拦截、切角）。

5. 拼接顺序总结（以“当前智能体是追捕者”为例）：

   - `norm_self_vel`：2 维
   - `norm_self_pos`：2 维
   - `other_pos`：3 个其他智能体 × 2 维 = 6 维
     - 其中包括：
       - 2 个追捕者的相对位置
       - 1 个逃跑者的相对位置
   - `other_vel`：逃跑者速度 2 维

   总维度：`2 + 2 + 6 + 2 = 12` 维（在 3 追捕 + 1 逃跑的设定下）。

   观测向量结构可以抽象为：
   - 自身状态（4 维：速度 + 位置）
   - 其他所有智能体的相对位置（6 维）
   - 逃跑者的相对速度（2 维）

整体而言：  
观测是**局部化视角 + 部分全局信息**：
- 自己的位置和速度（全局绝对坐标，归一化）。
- 相对位置信息（对所有其他智能体）。
- 目标（逃跑者）的速度信息（全局速度，归一化但未转换为相对速度）。


---

## 三、智能体交互 (Agent Interactions)

### 1. 捕食者与猎物物理属性对比（max_speed, size）

在 `__init__` 中给出默认参数：

```python
self.world_size = 2.5
self.max_force = 1.0
self.capture_threshold = 0.5

self.adversary_params = {'size': 0.075, 'max_speed': 1.0}
self.agent_params     = {'size': 0.050, 'max_speed': 1.3}
```

含义与对比：

1. 大小 (`size`)
   - 追捕者（adversary）：`size = 0.075`
   - 逃跑者（agent / prey）：`size = 0.050`
   - 物理意义：
     - `size` 是碰撞体积半径，用于碰撞检测。
     - 追捕者的碰撞体积更大 → 更容易与其他追捕者相互碰撞，需要更严格的避碰策略。
     - 对“围捕成功”的几何判定也有作用（接近猎物时的安全距离）。

2. 最大速度 (`max_speed`)
   - 追捕者：`1.0`
   - 逃跑者：`1.3`
   - 物理意义：
     - 逃跑者比追捕者更快（1.3 倍），理论上可以逃离直线追逐。
     - 要确保能围捕成功，追捕者必须采用协同策略（夹击、包围），而不是单兵直线追击。

3. 力 / 动作限制 (`max_force`)
   - 所有智能体共用：`self.max_force = 1.0`
   - 物理意义：
     - 控制输入的最大幅度，实际最大加速度或速度变化的上限。
     - 在该片段中未直接用在逻辑中，但一般用于剪裁 agent 的动作空间。

小结：  
逃跑者更小、更快；追捕者更大、更慢，但有数量优势（3 vs 1）。这强制 MARL 策略从“协同围捕”角度解决问题，而不是个体能力压制。

### 2. 碰撞 / 成功捕获的数学与物理条件

`is_collision` 函数：

```python
def is_collision(self, agent1_pos, agent1_size, agent2_pos, agent2_size):
    delta_pos = agent1_pos - agent2_pos
    dist = np.sqrt(np.sum(np.square(delta_pos)))
    dist_min = agent1_size + agent2_size
    return dist < dist_min
```

数学形式：

- 实际距离：
  \[
  d = \left\| \mathbf{p}_1 - \mathbf{p}_2 \right\|_2
  \]
- 碰撞阈值：
  \[
  d_{\min} = r_1 + r_2
  \]
  其中 \(r_1 = \text{agent1\_size}\)，\(r_2 = \text{agent2\_size}\)。

碰撞/捕获条件：

- 当 \(d < d_{\min}\) 时，判定两者发生**碰撞**（或接触/重叠）。
- 物理意义：
  - 把智能体视为二维平面中的圆形实体，半径分别为 `size`。
  - 两个圆心的距离小于半径和 → 两个圆重叠/接触 → 碰撞。

在围捕任务中的含义：

- 追捕者–逃跑者之间满足上述条件可以被解释为：
  - “成功捕获猎物”事件（视具体任务定义，有的环境需要额外逻辑，如多个追捕者同时接触）。
- 追捕者–追捕者之间满足上述条件：
  - “追捕者之间碰撞”事件，需要在奖励函数中给予惩罚，以避免相互撞击。

此外还有一个**围捕尺度阈值**：

- `capture_threshold = 0.5`
  - 这是相对于 world_size 的尺度（= 2.5 × 0.2），不等同于 `size` 的碰撞阈值。
  - 更可能用来定义一类**软围捕区域**：
    - 比如：追捕者只要各自与猎物距离 < capture_threshold，即认为其已经进入“包围圈区域”。

因此，环境中有两种距离相关的“成功/状态”概念：

1. 严格碰撞：基于 `size` 的 `is_collision` → 几何接触。
2. 围捕阈值：基于 `capture_threshold` → 进入某个距离范围内的包围区域。


---

## 四、任务相关信息 (Task-relevant Information)

### 1. 围捕任务核心目标与变量的关联

任务目标回顾：

1. 追捕者需要接近并包围目标；
2. 追捕者之间避免碰撞；
3. 形成均匀的包围圈；
4. 在尽可能短的时间内完成围捕。

对应到当前 global_state/observation 中的变量，可以识别出直接相关的部分：

#### （1）接近并包围目标 —— 距离相关（正/负相关）

相关变量：

- `distances_to_prey`（追捕者到猎物的距离数组）
- `agent_positions` 与 `prey_position`（可再计算距离与角度）

关系：

- 距离越小 → 越接近目标 → 围捕质量越高。
- 所以：
  - 与任务目标**负相关**的指标：`distances_to_prey`（越小越好）。
  - 与任务目标**正相关**的指标：`-distances_to_prey` 或其函数。

此外，如果任务定义为“所有追捕者都进入 capture_threshold 内为成功”，则：

- 一个关键子目标是：  
  - `max(distances_to_prey)` < `capture_threshold`
- 这使得 `max(distances_to_prey)` 成为整个队伍围捕程度的核心尺度。

#### （2）追捕者之间避免碰撞 —— 追捕者–追捕者距离

相关变量：

- `inter_agent_distances`（全体智能体间距离矩阵）
- 或从中筛出追捕者–追捕者子矩阵：`dist(A_i, A_j)`  

关系：

- 当 `dist(A_i, A_j) < size_i + size_j` 时发生碰撞。
- 安全目标要求：
  - 所有追捕者之间保持 `dist(A_i, A_j) > size_i + size_j`。
- 因此：
  - 与任务目标**正相关**的指标：  
    - `min_{i≠j} dist(A_i, A_j)`（越大越安全；但过大又不利于形成紧密包围）。
  - 与任务目标**负相关**的指标：
    - 任何形式的碰撞指示（`is_collision` 的结果），或  
    - `max(0, (size_i + size_j) - dist(A_i, A_j))` （穿透量）。

#### （3）形成均匀的包围圈 —— 角度分布 & 半径一致性

相关变量：

- `agent_positions`
- `prey_position`
- 派生量：
  - 每个追捕者相对于猎物的极坐标表示（半径 \(r_i\), 角度 \(\theta_i\)）。

目标特征：

1. **半径一致性**：
   - 对每个追捕者 \(i\)：
     \[
     r_i = \|\mathbf{p}_{A_i} - \mathbf{p}_{P}\|
     \]
   - 理想情况下：
     - 所有 \(r_i\) 接近同一个值 \(r^*\)（比如略大于 `capture_threshold`）。
   - 相关指标：
     - 半径方差：\(\text{Var}(r_i)\) 应该越小越好 → 与均匀包围正相关的是 `-Var(r_i)`。

2. **角度均匀分布**：
   - 极角：
     \[
     \theta_i = \mathrm{atan2}(y_i - y_P, x_i - x_P)
     \]
   - 对 3 个追捕者来说，理想角度差值为：
     - 相邻追捕者间角度差 ≈ \(2\pi / 3\)。
   - 相关指标：
     - 排好序的 \(\theta_i\)，计算相邻差值 \(\Delta\theta_i\)，与 \(2\pi / 3\) 的偏差。
     - 方差或者偏差平方和：越小越接近理想均匀分布。

因此，与“均匀包围圈”直接相关的变量是：

- 从 `agent_positions` 和 `prey_position` 计算得到的：
  - `r_i`（追捕者到猎物的距离）及其分布特征（均值/方差）。
  - `θ_i`（相对角度）及其差分均匀程度。

#### （4）时间最优 —— 收敛速度 / 步数相关

在给出的片段中：

- 没有显式时间步计数变量（如 `t` 或 `step_count`）。
- 但环境通常会在更上层维护一个 step 计数器。

逻辑上：

- “尽可能短的时间内完成围捕”本质是：
  - 在**更少的时间步数**内达到围捕成功条件。
  - 或者，在固定步长内让围捕质量指标尽快达到高值。

从现有变量可间接使用：

- 从 `distances_to_prey` 的变化率 / 递减速度来度量“逼近速度”。
- 从 `r_i` → `capture_threshold` 的收敛速度。

在 reward 设计或任务评估时通常会：

- 每个时间步施加一个小负值（时间成本），或者
- 使用折扣因子使得更早实现围捕得到更高的总回报。

### 2. 代表高质量“包围圈”的空间拓扑结构

从几何角度，3 个追捕者围捕 1 个逃逸目标，在二维平面上的“高质量包围圈”可归纳为如下特征：

1. **目标居于包围圈中心附近**
   - 追捕者的位置基本围绕 `prey_position` 对称分布。
   - 空间上，猎物位于追捕者构成的三角形内部或非常接近内部位置。
   - 定量判断：
     - 猎物在三角形 `A1A2A3` 内：
       - 可通过重心坐标 / 方向一致性判断。
     - 或至少满足：各追捕者到猎物的距离均小，并且猎物难以从某个方向“无阻逃出”。

2. **三者与猎物的距离接近（半径一致性）**
   - 所有追捕者与猎物的距离 \(r_i\) 接近同一个值：
     - \( r_1 \approx r_2 \approx r_3 \approx r^* \)
   - 这个 \(r^*\) 通常与 `capture_threshold` 相关：
     - 良好的“包围圈”可以定义为：
       - \(r_i\) 既小于某个上界（足够接近），
       - 又不至于低于碰撞上限（避免与猎物发生不希望的硬碰撞，如果需要）。

3. **角度均匀分布（围绕猎物 2π 均匀分割）**
   - 三个追捕者从猎物视角看，分别位于三个大致相隔 \(120^\circ\) 的方向上：
     - \(\theta_2 - \theta_1 \approx 2\pi/3\)
     - \(\theta_3 - \theta_2 \approx 2\pi/3\)
   - 这样，猎物在任意方向逃跑都会迅速接触到至少一个追捕者的威胁范围。

4. **追捕者之间保持安全距离**
   - 任意两个追捕者之间距离：
     \[
     d(A_i, A_j) > size_i + size_j + \delta_{\text{safe}}
     \]
   - 即使三者收缩包围圈，也不会彼此撞到。
   - 结合目标 (2)，这意味着：
     - 在圆周上均匀分布的三点之间的弦长（追捕者–追捕者距离）要足够大。

5. **动态稳定性**
   - 不仅在单一时刻呈现良好队形，还要能在猎物移动时维持这个队形结构：
     - 三个追捕者相对于猎物的角度变化缓慢；
     - 半径可随时间缓慢收缩，但保持大致的一致性和角度均匀性。

从 `global_state` 中可用来描述该拓扑的关键量：

- `agent_positions` 与 `prey_position` → 计算：
  - `r_i = ||p_Ai - p_P||`
  - `θ_i = atan2(…)`
- `inter_agent_distances` → 确保：
  - 追捕者–追捕者间距足够大，避免碰撞。
- `distances_to_prey` → 简化为半径向量 `[r_1, r_2, r_3]`。

综合起来：  
一个高质量的“包围圈” =

- 所有 `distances_to_prey` 都小（圈足够紧）；
- `distances_to_prey` 之间差异小（半径一致性好）；
- 追捕者之间的相对角度接近 120° 间隔（围绕猎物均匀）；
- `inter_agent_distances` 保证没有碰撞且保持合理间距；
- 这种配置越早出现（更少步数达成），越符合“迅速围捕”的任务要求。


---

以上是基于当前 Dec-POMDP 环境核心逻辑的**代码级**状态/观测/物理交互与任务结构分析，未涉及任何奖励函数实现。

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
- 只允许输出1个Python代码块，**严禁包含任何解释性文字或Markdown说明**，直接以 ```python 开头

```

================================================================================
## Response (From LLM)
================================================================================

```text
```python
import numpy as np


def compute_reward(agent_name, observation, global_state, actions, world):
    # 逃跑者无奖励
    if not global_state['is_adversary']:
        return 0.0, {}

    components = {}

    # --------------------
    # 基本提取与索引
    # --------------------
    agent_positions = global_state['agent_positions']
    agent_velocities = global_state['agent_velocities']
    prey_pos = global_state['prey_position']
    prey_vel = global_state['prey_velocity']
    distances_to_prey = global_state['distances_to_prey']
    inter_agent_distances = global_state['inter_agent_distances']
    world_size = global_state['world_size']
    capture_threshold = global_state['capture_threshold']

    # 假设 world.agents 顺序中，agent_name 唯一标识一个追捕者
    all_agents = world.agents
    adversaries = [a for a in all_agents if a.adversary]
    preys = [a for a in all_agents if not a.adversary]

    # 追捕者数量（期望为 3）
    n_adv = len(adversaries)

    # 为了从 agent_name 找到索引
    # 同时假设 global_state['distances_to_prey'] 与 adversaries 顺序一致
    adv_indices = []
    for idx, agent in enumerate(all_agents):
        if agent.adversary:
            adv_indices.append(idx)

    # 找到当前智能体在 all_agents 中的索引
    agent_index = None
    for i, agent in enumerate(all_agents):
        if getattr(agent, 'name', None) == agent_name:
            agent_index = i
            break

    if agent_index is None:
        # 找不到该智能体，返回 0 奖励，方便调试
        return 0.0, {}

    # 找到当前智能体在追捕者列表中的局部索引
    if all_agents[agent_index].adversary:
        local_adv_index = adv_indices.index(agent_index)
    else:
        # 如果意外不是追捕者，则无奖励
        return 0.0, {}

    # --------------------
    # 1. 接近目标距离奖励
    # --------------------
    # 对自身到猎物的距离做负奖励，并对整个队伍距离做协同项
    self_dist_to_prey = distances_to_prey[local_adv_index]
    mean_team_dist_to_prey = np.mean(distances_to_prey) if n_adv > 0 else self_dist_to_prey

    # 超参数
    w_self_approach = 1.0
    w_team_approach = 0.5

    # 自身距离奖励（越近越好）
    self_dist_reward = -self_dist_to_prey / world_size
    # 队伍距离奖励（鼓励整体收缩包围圈）
    team_dist_reward = -mean_team_dist_to_prey / world_size

    components['self_distance_reward'] = w_self_approach * self_dist_reward
    components['team_distance_reward'] = w_team_approach * team_dist_reward

    # --------------------
    # 2. 捕获/进入围捕区域奖励
    # --------------------
    # 所有追捕者是否都在 capture_threshold 内
    capture_mask = distances_to_prey <= capture_threshold
    all_in_capture_zone = bool(np.all(capture_mask)) if n_adv > 0 else False

    # 超参数
    w_capture_all = 5.0
    w_capture_self = 1.0

    # 自身进入捕获圈奖励
    in_zone_self = 1.0 if self_dist_to_prey <= capture_threshold else 0.0
    components['self_in_capture_zone'] = w_capture_self * in_zone_self

    # 所有追捕者都在捕获圈内时给予额外全局奖励
    capture_all_bonus = 1.0 if all_in_capture_zone else 0.0
    components['team_full_capture_bonus'] = w_capture_all * capture_all_bonus

    # --------------------
    # 3. 防碰撞惩罚（追捕者-追捕者 + 追捕者-逃跑者）
    # --------------------
    # 利用全局 inter_agent_distances 和 size 信息
    adv_size = getattr(world, 'adversary_size', None)
    prey_size = getattr(world, 'prey_size', None)

    # 当外部环境未提供 size 时，使用 CoreEnvLogic 中的缺省值作为回退
    if adv_size is None:
        adv_size = 0.075
    if prey_size is None:
        prey_size = 0.050

    # 超参数
    w_collision = 5.0
    w_near_collision = 1.0
    safety_margin = 0.02

    # 追捕者-追捕者碰撞与接近惩罚
    self_agent_pos = agent_positions[agent_index]
    collision_penalty = 0.0
    near_collision_penalty = 0.0

    for j, other in enumerate(all_agents):
        if j == agent_index:
            continue
        # 追捕者-追捕者
        if other.adversary:
            other_pos = agent_positions[j]
            dist_aa = np.linalg.norm(self_agent_pos - other_pos)
            min_safe = 2 * adv_size
            if dist_aa < min_safe:
                collision_penalty -= 1.0
            elif dist_aa < min_safe + safety_margin:
                # 软惩罚：过近但未碰撞
                near_collision_penalty -= (min_safe + safety_margin - dist_aa) / safety_margin

    # 追捕者-逃跑者碰撞（如果不希望硬碰撞，可以单独处理，这里仍视为惩罚）
    if preys:
        prey = preys[0]
        prey_index = all_agents.index(prey)
        prey_pos = agent_positions[prey_index]
        dist_ap = np.linalg.norm(self_agent_pos - prey_pos)
        min_safe_ap = adv_size + prey_size
        if dist_ap < min_safe_ap:
            # 如果环境认为这是“成功捕获”，可以改为正奖励；
            # 此处仅作为物理碰撞的惩罚项（不与捕获奖励混淆）
            collision_penalty -= 1.0

    components['collision_penalty'] = w_collision * collision_penalty
    components['near_collision_penalty'] = w_near_collision * near_collision_penalty

    # --------------------
    # 4. 队形：均匀包围圈（半径一致性 + 角度均匀）
    # --------------------
    formation_radius_reward = 0.0
    formation_angle_reward = 0.0

    if n_adv >= 2 and preys:
        prey = preys[0]
        prey_index = all_agents.index(prey)
        prey_pos = agent_positions[prey_index]

        # 追捕者相对猎物位置
        adv_positions = np.array([agent_positions[idx] for idx in adv_indices])
        rel_adv = adv_positions - prey_pos  # shape (n_adv, 2)

        # 半径与角度
        radii = np.linalg.norm(rel_adv, axis=1) + 1e-8
        angles = np.arctan2(rel_adv[:, 1], rel_adv[:, 0])

        # ---- 半径一致性 ----
        mean_r = np.mean(radii)
        radius_var = np.mean((radii - mean_r) ** 2)
        # 将方差归一化到 [0, 1] 附近，使用 world_size 做尺度
        norm_radius_var = radius_var / (world_size ** 2)
        formation_radius_reward = -norm_radius_var

        # ---- 角度均匀性 ----
        # 将角度排序后计算相邻差值，考虑环状 2π
        sorted_angles = np.sort(angles)
        diffs = np.diff(sorted_angles)
        # 加上首尾闭环差值
        last_diff = (sorted_angles[0] + 2 * np.pi) - sorted_angles[-1]
        angle_diffs = np.concatenate([diffs, [last_diff]])

        ideal_diff = 2 * np.pi / n_adv
        angle_diff_var = np.mean((angle_diffs - ideal_diff) ** 2)
        # 归一化，最大偏差不会超过 π
        norm_angle_diff_var = angle_diff_var / (np.pi ** 2)
        formation_angle_reward = -norm_angle_diff_var

    # 超参数
    w_formation_radius = 0.5
    w_formation_angle = 1.0

    components['formation_radius_reward'] = w_formation_radius * formation_radius_reward
    components['formation_angle_reward'] = w_formation_angle * formation_angle_reward

    # --------------------
    # 5. 时间效率：步长惩罚 + 围捕后静态保持奖励
    # --------------------
    # 每一步给小的时间惩罚，鼓励尽快完成
    step_penalty = -0.01
    components['step_penalty'] = step_penalty

    # 当完成全队围捕后，鼓励保持稳定包围圈（奖励略高，用于稳定策略）
    w_hold_capture = 0.1
    hold_capture_reward = 0.0
    if all_in_capture_zone:
        # 鼓励保持位置稳定：速度越小越好
        # 使用自身速度与队伍速度的负范数
        self_vel = agent_velocities[agent_index]
        team_adv_vels = np.array(
            [agent_velocities[idx] for idx in adv_indices]
        )
        self_speed = np.linalg.norm(self_vel)
        mean_team_speed = np.mean(np.linalg.norm(team_adv_vels, axis=1))
        hold_capture_reward = -0.5 * self_speed - 0.5 * mean_team_speed

    components['hold_capture_reward'] = w_hold_capture * hold_capture_reward

    # --------------------
    # 6. 动作正则（可选，防止剧烈动作）
    # --------------------
    # 若 actions 是 dict 或 list，尽可能找到当前智能体动作并加以 L2 正则
    action_reg = 0.0
    w_action_reg = 0.001

    if isinstance(actions, dict):
        if agent_name in actions:
            act = np.array(actions[agent_name])
            action_reg = -np.linalg.norm(act)
    elif isinstance(actions, (list, tuple)):
        # 假设顺序与 all_agents 一致
        if 0 <= agent_index < len(actions):
            act = np.array(actions[agent_index])
            action_reg = -np.linalg.norm(act)

    components['action_regularization'] = w_action_reg * action_reg

    total_reward = float(sum(components.values()))
    return total_reward, components
```
```
