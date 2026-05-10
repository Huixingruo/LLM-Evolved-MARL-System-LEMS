# LLM Interaction Log

- **Generation**: 0
- **Phase**: Phase2_Initial_Generation
- **Candidate Info**: 1
- **Timestamp**: 20260409_123510

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

1. **依赖包**

   代码只显式依赖了一个外部包：
   ```python
   import numpy as np
   ```
   用途：
   - 向量/矩阵运算：`np.array`, `np.zeros`
   - 距离计算：`np.sqrt`, `np.sum`, `np.square`, `np.linalg.norm`
   - 拼接/归一化：在 `observation` 中对向量做除法、拼接

   其它使用的对象（如 `world`, `agent`）来自外部环境框架（很像 MPE / PettingZoo 风格），但在本文件中没有再导入其他 Python 包。

2. **外部未定义变量 / 结构依赖**

   代码本身依赖多个**在本文件未定义**、但在上层环境中应当已经定义好的类或字段：

   - `world` 对象：
     - `world.agents`: 一个包含所有智能体的列表
   - `agent` / `other` / `adv` / `prey` 对象需要满足：
     - `agent.adversary`: `bool`，标记是否是追捕者（True）或逃逸者（False）
     - `agent.max_speed`: 当前智能体最大速度（用于归一化、物理约束）
     - `agent.state`: 有位置和速度子结构
       - `agent.state.p_pos`: np.array, 形如 `(2,)` 的二维坐标
       - `agent.state.p_vel`: np.array, 形如 `(2,)` 的二维速度
   - 在 `CoreEnvLogic.__init__` 中：
     - `self.adversary_params` / `self.agent_params` 中的 `size` / `max_speed` 会被外部环境用于实际创建智能体（但这里没有直接使用 `size`，仅在 `is_collision` 中通过参数传入）。

   小结：  
   - 外部框架必须提供 `world`、`agent`、`agent.state` 的数据结构，否则本文件无法单独运行。  
   - 但从本片段看，没有出现“完全未引用即使用”的变量名（如拼写错误），逻辑上相对自洽。


---

## 二、环境结构 (Environment Structure)

### 1. 状态空间 (Global State) 的构成与物理意义

`_build_global_state` 返回字典 `global_state`，用于 reward 计算。结构为：

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

逐项说明：

1. `agent_positions: np.array(shape=(N, 2))`
   - 内容：所有智能体（追捕者 + 逃逸者）的**绝对位置**，二维平面坐标。
   - 物理意义：
     - 每行对应一个智能体在 2D 连续空间中的位置 `(x, y)`。
     - 空间范围受 `world_size` 约束，即坐标大致在 `(-2.5, 2.5)` 之间（不过未必严格，因为物理引擎可能允许越界再处理）。
   - 对任务的作用：
     - 直接用于判断围捕是否成功（追捕者是否接近猎物、是否围住猎物）。
     - 是计算相对位置、角度、队形结构的基础。

2. `agent_velocities: np.array(shape=(N, 2))`
   - 内容：所有智能体的**绝对速度向量**。
   - 物理意义：
     - 每行对应一个智能体当前的速度 `(vx, vy)`。
     - 数值范围受各自 `max_speed` 限制。
   - 对任务的作用：
     - 可用于惩罚乱动、鼓励减速稳定围捕。
     - 可分析追捕者是否在“追击”方向上运动（相对猎物的位置 + 速度）。

3. `prey_position: np.array(shape=(2,))`
   - 内容：唯一逃逸者（猎物）的**位置**。
   - 物理意义：
     - 提供任务中心目标的空间位置。
   - 对任务的作用：
     - 所有围捕目标都围绕此点展开（距离、夹角、包围半径等）。

4. `prey_velocity: np.array(shape=(2,))`
   - 内容：猎物速度向量。
   - 物理意义：
     - 决定猎物逃跑方向和速度大小。
   - 对任务的作用：
     - 可用于预测逃跑方向、布防位置。
     - 可帮助设计奖励鼓励提前拦截而不是只盯当前距离。

5. `distances_to_prey: np.array(shape=(#adversaries,))`
   - 内容：每个追捕者到猎物的**欧氏距离**：
     ```python
     np.linalg.norm(adv.state.p_pos - prey_pos)
     ```
   - 物理意义：
     - 标量距离（无符号），衡量接近程度。
   - 对任务的作用：
     - 越小表示追捕者越接近猎物，是“接近并包围”的一阶关键指标。
     - 可与 `capture_threshold` 对比判断是否接近捕获状态。

6. `inter_agent_distances: np.array(shape=(N, N))`
   - 内容：所有智能体之间的**两两距离矩阵**：
     ```python
     inter_agent_distances[i][j] = || p_i - p_j ||
     ```
   - 物理意义：
     - 主对角线为 0（到自己的距离）。
     - 对称矩阵：`d(i, j) = d(j, i)`。
   - 对任务的作用：
     - 追捕者-追捕者间距离：用于防撞（不能太近）。
     - 追捕者-猎物距离：与 `distances_to_prey` 部分重叠，但矩阵格式更通用，可直接推断拓扑结构，做队形分析。

7. `is_adversary: bool`
   - 内容：当前“调用 reward 的这个 agent”是否为追捕者。
   - 物理意义：
     - 区分角色，以便设计不同的 reward（如对猎物给“逃得远”奖励，对追捕者给“抓得快”奖励）。
   - 对任务的作用：
     - 在团队追捕任务中，相当于提供角色标签，可以做**角色依赖的奖励塑形**。

8. `world_size: float`
   - 内容：世界边界尺度，这里是 `2.5`。
   - 物理意义：
     - 地图坐标范围近似为 `(-world_size, world_size)`。
   - 对任务的作用：
     - 用于归一化距离 / 定义“边缘惩罚”等。
     - 可判断猎物是否被逼到墙角。

9. `capture_threshold: float`
   - 内容：围捕判定的距离阈值 `world_size * 0.2 = 0.5`。
   - 物理意义：
     - 表征“多近算是围捕完成”的距离标尺。
   - 对任务的作用：
     - 按此距离可定义“捕获成功条件”或“捕获区域”。
     - 可以在奖励中区分“接近状态”和“确实捕获”。

---

### 2. 观测空间 (Observation) 的构造与相对信息

`observation(agent, world)` 返回一个局部观测向量：

```python
return np.concatenate([norm_self_vel] + [norm_self_pos] + other_pos + other_vel)
```

构成步骤：

1. **自身信息（self-related）**

   ```python
   norm_self_vel = agent.state.p_vel / agent.max_speed
   norm_self_pos = agent.state.p_pos / self.world_size
   ```

   - `norm_self_vel ∈ R^2`
     - 自身速度向量按自身 `max_speed` 归一化到大致 `[-1, 1]` 范围。
     - 提供当前运动方向与相对速度大小。
   - `norm_self_pos ∈ R^2`
     - 自身绝对位置按 `world_size` 归一化到大致 `(-1, 1)` 范围。
     - 提供自己在整体地图上的位置（非相对）。

   到此为止，局部观测已包含：  
   - 自身绝对位置信息  
   - 自身速度（方向和幅度）

2. **其他智能体的相对位置信息**

   ```python
   other_pos = []
   other_vel = []
   for other in world.agents:
       if other is agent: continue
       rel_pos = (other.state.p_pos - agent.state.p_pos) / self.world_size
       other_pos.append(rel_pos)
       if not other.adversary:
           other_vel.append(other.state.p_vel / other.max_speed)
   ```

   - 对所有 `other != agent`：
     - `rel_pos = (p_other - p_self) / world_size ∈ R^2`
       - 是从当前智能体视角看，其他智能体的**归一化相对位置**。
       - 符号含义：  
         - `rel_pos = (x_other - x_self, y_other - y_self)/world_size`  
         - 正负表示相对方向（右/左、上/下）。
   - 被追加进 `other_pos` 列表，维度是：  
     - 若总共有 `N` 个智能体，则 `other_pos` 包含 `N-1` 个二维向量，总维度 `2*(N-1)`。

3. **其他智能体的速度信息（仅猎物）**

   - 只有在 `if not other.adversary` 条件下才追加速度：
     ```python
     other_vel.append(other.state.p_vel / other.max_speed)
     ```
   - 这意味着：
     - 对于当前视角的智能体，无论当前是追捕者还是逃逸者：
       - 它能看到所有其它智能体的**相对位置**；
       - 但它只能看到**非 adversary（通常是猎物）**的速度。
   - 在当前任务设定中：3 追捕 + 1 逃逸：
     - 对任意一个追捕者：
       - `other_pos` 中有 3 个向量（2 个追捕队友 + 1 个猎物），共 `3×2 = 6` 维。
       - `other_vel` 只包含猎物速度 1 个向量 `2` 维。
     - 对逃逸者（若也使用同一观测函数）：
       - `other_pos` 中有 3 个追捕者的位置向量，共 `6` 维。
       - `other_vel` 中没有任何元素（因为它看不到 adversary 的速度）。

4. **最终观测向量的拼接顺序与总维度**

   拼接顺序：
   ```python
   [norm_self_vel] + [norm_self_pos] + other_pos + other_vel
   ```

   对 3 追捕 + 1 逃逸情形，**以一个追捕者为例**：
   - `norm_self_vel`: 2 维
   - `norm_self_pos`: 2 维
   - `other_pos`: (3 个智能体) × 2 维 = 6 维
   - `other_vel`: (1 个猎物) × 2 维 = 2 维
   - 总维度：`2 + 2 + 6 + 2 = 12` 维

   信息属性总结：

   - **绝对信息**：
     - 自身绝对位置（归一化）
   - **相对信息**：
     - 所有其他智能体相对自身的位置：提供队形、包围的相对构型信息。
   - **部分速度信息**：
     - 自身速度
     - 猎物速度（相对的绝对速度；队友速度不可见）

   对围捕任务来说，这样的局部观测：
   - 足以知道“我在哪里，目标在哪里，队友在哪里，目标往哪里跑”；
   - 但缺少队友速度，可能会导致在协同比例控制（如同时靠拢、同时停下）上依赖隐式协调或通信机制。


---

## 三、智能体交互 (Agent Interactions)

### 1. 追捕者与猎物的物理属性差异

在 `__init__` 中：

```python
self.world_size = 2.5
self.max_force = 1.0
self.capture_threshold = 0.5

self.adversary_params = {'size': 0.075, 'max_speed': 1.0}
self.agent_params = {'size': 0.050, 'max_speed': 1.3}
```

根据注释与命名，可推断：

- **adversary**：为追捕者（pursuers）
  - `size = 0.075` （碰撞半径较大）
  - `max_speed = 1.0` （速度较慢）

- **agent**（或 prey）：为逃逸者（evader）
  - `size = 0.050` （碰撞半径较小）
  - `max_speed = 1.3` （速度较快）

物理含义与策略影响：

1. **最大速度 (max_speed)**：
   - 猎物速度更快 (`1.3 > 1.0`)，在无协作的情况下更容易逃离追捕者。
   - 对追捕者来说，单体能力不足以追上猎物，需要**多智能体配合**（夹击、包围等），与任务目标高度一致。

2. **碰撞半径 (size)**：
   - 追捕者半径更大 (`0.075 > 0.050`)：
     - 更容易彼此发生碰撞，需要保持合理间隔。
     - 在对猎物包围时，“可有效覆盖空间”的范围更大，有利于形成封锁圈。
   - 猎物半径小：
     - 在几何上更容易通过缝隙“钻出去”，对包围结构提出更高要求。

3. **世界约束与力限制**：
   - `max_force = 1.0` 表征动作最大值（推力/加速度上限），对所有智能体生效。
   - 实际能达到的速度还由 `max_speed` 限制，因此猎物在动力学上有优势。

整体上，这组参数天然形成一个**速度更快的“小目标”被速度较慢但身体更大的“群体追捕”的典型围捕场景**。

### 2. “成功捕获（Collision）”的数学与物理条件

碰撞检测函数：

```python
def is_collision(self, agent1_pos, agent1_size, agent2_pos, agent2_size):
    delta_pos = agent1_pos - agent2_pos
    dist = np.sqrt(np.sum(np.square(delta_pos)))
    dist_min = agent1_size + agent2_size
    return dist < dist_min
```

数学条件：

- 给定两个智能体的**位置向量** `p1`, `p2` 和**碰撞半径** `r1`, `r2`，定义：
  - `dist = ||p1 - p2||_2`
  - `dist_min = r1 + r2`
- 若 `dist < dist_min`，即：
  \[
  \sqrt{(x_1 - x_2)^2 + (y_1 - y_2)^2} < r_1 + r_2
  \]
  则判定发生碰撞。

物理意义：

- 智能体被视为**圆形刚体**，半径为 `size`。
- 当两个圆的边界重叠时（中心距小于半径和），认为它们发生了接触/碰撞。
- 用于：
  - 追捕者与猎物碰撞：可作为“直接抓到”条件的一部分；
  - 追捕者间碰撞：可用于惩罚以鼓励队形保持、避免拥堵。

结合 `capture_threshold = 0.5`：

- `capture_threshold` 是针对“围捕区域”的尺度，而 `is_collision` 是针对**点对点物理接触**的判断；二者可以是两种不同层级的捕获定义：
  - `is_collision`: 极靠近 / 物理接触。
  - `capture_threshold`: 不一定要求接触，但要求在一定距离范围内形成包围。


---

## 四、任务相关信息 (Task-relevant Information)

### 1. 围捕任务核心目标与变量的关系

任务要求：

1. 追捕者需要接近并包围目标  
2. 追捕者之间避免碰撞  
3. 形成均匀的包围圈  
4. 尽可能短时间内完成围捕

从 `global_state` 和 `observation` 中，直接与这些目标相关的变量（正相关 / 负相关）如下：

1. **接近目标**

   关键变量：
   - `distances_to_prey`（每个追捕者到猎物的距离）
   - `inter_agent_distances` 中追捕者-猎物的元素
   - `prey_position` 与 `agent_positions` 之差

   关系：
   - 追捕者越接近猎物，`distances_to_prey` 的值越小。
   - 与目标“接近”是**负相关**：  
     - **接近程度 ∝ -distance_to_prey**

2. **包围目标（几何意义的“围捕”）**

   关键变量：
   - `agent_positions`（全部追捕者和猎物的绝对位置）
   - 特别是：追捕者相对于猎物的**方位角 / 相对位置**：
     - 可以由 `(agent_i_pos - prey_pos)` 计算极坐标角度。
   - `inter_agent_distances`（追捕者-追捕者之间的距离）

   包围的几何直观：  
   - 猎物位于某个中心位置，追捕者分布在其周围不同方向上，且猎物难以从某个方向“穿出去”。

   与目标相关的关系：
   - 包围“闭合度”：  
     - 若追捕者均匀分布在猎物周围，且与猎物距离小于某阈值，则围捕完成度高。
   - 需要同时考虑：
     - 追捕者与猎物的**平均距离**（越小越好）。
     - 追捕者相对于猎物的**角度覆盖范围**（覆盖 360° 越均匀越好）。

3. **追捕者之间避免碰撞**

   关键变量：
   - `inter_agent_distances` 中追捕者-追捕者子矩阵。
   - 可能在更底层（不在此代码）使用 `is_collision` 配合追捕者 `size`。

   关系：
   - 追捕者两两距离越小，发生碰撞的风险越高。
   - 安全性与距离**正相关**，即：
     - **安全性 ∝ min(inter_agent_distances between adversaries)**

4. **形成均匀的包围圈**

   关键变量：
   - `agent_positions`, `prey_position` → 计算追捕者相对猎物的：
     - 距离：`r_i = ||p_adv_i - p_prey||`
     - 角度：`θ_i = atan2(y_adv_i - y_prey, x_adv_i - x_prey)`
   - 间接还会看：追捕者-追捕者的距离分布。

   均匀包围的核心要素：
   - **半径一致性**：各个追捕者与猎物的距离接近某个共同值：
     - “半径差异”越小，越接近理想圆环。
   - **角度均匀性**：追捕者在角度上大致呈 120° 间隔（针对 3 个追捕者）：
     - 相邻角度差接近 `2π/3`。

   与目标的关系：
   - 高质量“包围圈”与以下指标正相关：
     - 距离方差小：`Var(r_i)` 小；
     - 角度间距偏差小：`|Δθ_i - 2π/3|` 小；
   - 与“半径太大”负相关：半径太大则猎物仍然容易逃脱。

5. **尽可能短时间内完成围捕**

   关键变量：
   - 时间步计数 `t`（不在此代码中，但在环境循环中存在）。
   - 距离与队形的收敛速度：
     - `distances_to_prey` 下降的速率；
     - 包围圈质量指标（半径一致性 + 角度均匀性）的收敛速度。

   关系：
   - 若在较少时间步内，使：
     - `distances_to_prey` 达到一定阈值，
     - 队形满足“均匀包围圈”的几何条件，
   则表示高效完成任务。

   这里的代码虽然未直接提供时间相关变量，但**奖励函数设计时可以在外部环境层引入时间折扣或每步惩罚**，从而对“时间”施加压力。

---

### 2. 代表高质量“包围圈”的空间拓扑特征

在 3 追捕 + 1 猎物的场景里，一个“理想的包围圈”可从以下几何特征来刻画：

1. **猎物位于包围中心附近**

   - 猎物位置 `prey_position` 大致位于追捕者构成多边形（此处为三角形）的**内部**。
   - 等价几何判断：
     - 计算以三个追捕者为顶点的三角形；
     - 判断猎物是否在此三角形内部：
       - 可通过重心坐标、面积法或射线法等进行判定。
   - 若猎物在三角形外，则仍有“开口方向”，表示包围不完善。

2. **追捕者与猎物距离小且相近（近似同一半径）**

   - 每个追捕者到猎物的距离：
     - `r_i = distances_to_prey[i]`
   - 特征：
     - `max(r_i)` 不大（整体包围圈紧凑）。
     - `Var(r_i)` 或 `max(r_i) - min(r_i)` 很小（半径差异不大）。
   - 直观意义：
     - 所有追捕者距离猎物相近，像一个“等半径”的圆环，而不是一个远一个近。

3. **追捕者在猎物周围的角度分布均匀**

   - 对每个追捕者计算相对猎物的方位角：
     \[
     \theta_i = \text{atan2}(y_{adv_i} - y_{prey},\ x_{adv_i} - x_{prey})
     \]
   - 排序后，使相邻角度差：
     - 对 3 个追捕者，理想状态为：
       \[
       \Delta\theta_i \approx \frac{2\pi}{3}
       \]
   - 均匀性指标：
     - `Std(Δθ_i)` 小；
     - 或 `sum(|Δθ_i - 2π/3|)` 小。

4. **追捕者之间的距离合适：既不太近（防撞），也不太远（避免破洞）**

   - 从 `inter_agent_distances` 取追捕者之间的距离 `d_ij`：
     - 防撞要求：`d_ij` 大于 `size_i + size_j` 一个安全边界；
     - 包围紧凑要求：`d_ij` 又不能太大，否则猎物容易从间隙逃脱。
   - 对 3 追捕的理想状态：
     - 三个追捕者大致形成一个边长相近的三角形：
       - `|d_12 - d_23|`, `|d_23 - d_31|` 等差值较小；
       - 边长大小与 `r_i` （包围半径）成合理比例。

5. **整体拓扑总结**

   高质量的“包围圈”可抽象为以下空间拓扑特征：

   - 拓扑结构：
     - 猎物位于由追捕者围成的闭合多边形内部；
   - 度量结构：
     - 追捕者到猎物的距离近似相等且较小；
     - 追捕者之间的角度间隔接近均匀划分；
     - 追捕者之间距离在一个合理区间：  
       - 大于碰撞阈值的适度倍数（避免碰撞），  
       - 小于某上界（防止留下“逃逸通道”）。

   所有这些特征都可以根据当前给定的 `global_state` 中的 `agent_positions`, `prey_position`, `distances_to_prey`, `inter_agent_distances` 来计算和评估。


---

以上分析仅聚焦于环境模型（Dec-POMDP）的状态/观测结构与物理交互机理，为之后设计奖励函数奠定基础，没有涉及任何奖励函数的具体实现。

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
    # 必须通过 global_state['is_adversary'] 过滤逃跑者
    if not global_state['is_adversary']:
        return 0.0, {}

    components = {}

    # --------------------
    # 配置系数（可调）
    # --------------------
    w_approach = 1.0          # 接近猎物奖励权重
    w_capture_bonus = 5.0     # 完成围捕一次性奖励
    w_time_penalty = 0.01     # 每步时间惩罚
    w_collision_penalty = 2.0  # 追捕者间碰撞惩罚
    w_collision_near_penalty = 0.2  # 追捕者间距离过近惩罚
    w_radius_uniform = 0.5    # 半径一致性奖励权重
    w_angle_uniform = 0.5     # 角度均匀性奖励权重
    w_center_in_triangle = 0.5  # 猎物位于三角形中心奖励权重
    w_compact_circle = 0.3    # 包围圈紧凑奖励权重

    # 几何参数
    safe_distance_factor = 1.5  # 安全距离 = factor * (r1 + r2)
    compact_radius_factor = 1.5  # 紧凑半径上界 = factor * capture_threshold

    # 从 global_state 提取信息
    agent_positions = global_state['agent_positions']
    prey_position = global_state['prey_position']
    distances_to_prey = global_state['distances_to_prey']
    inter_agent_distances = global_state['inter_agent_distances']
    world_size = global_state['world_size']
    capture_threshold = global_state['capture_threshold']

    # 找到当前智能体索引与追捕者索引
    all_agents = world.agents
    adversaries = [a for a in all_agents if a.adversary]
    preys = [a for a in all_agents if not a.adversary]

    if len(adversaries) != 3 or len(preys) != 1:
        # 非期望场景：给零奖励并返回空组件，防止下标错误
        return 0.0, {}

    prey = preys[0]
    prey_idx = all_agents.index(prey)

    # 当前 agent 及其索引
    current_agent = None
    for a in all_agents:
        if a.name == agent_name:
            current_agent = a
            break

    if current_agent is None or not current_agent.adversary:
        return 0.0, {}

    agent_idx = all_agents.index(current_agent)
    adv_indices = [all_agents.index(a) for a in adversaries]
    self_adv_local_idx = adversaries.index(current_agent)

    # 追捕者到猎物距离数组（与 adversaries 对齐）
    r_all = np.array(distances_to_prey, dtype=np.float32)
    r_self = r_all[self_adv_local_idx]

    # --------------------
    # 1. 接近猎物奖励
    # --------------------
    max_possible_dist = np.sqrt(2) * world_size
    normalized_dist = np.clip(r_self / max_possible_dist, 0.0, 1.0)
    distance_reward = -normalized_dist
    components['distance_reward'] = w_approach * distance_reward

    # --------------------
    # 2. 捕获/完成围捕奖励
    #    条件：所有追捕者均在 capture_threshold 内
    # --------------------
    in_capture_zone = r_all < capture_threshold
    if np.all(in_capture_zone):
        capture_reward = 1.0
    else:
        capture_reward = 0.0
    components['capture_reward'] = w_capture_bonus * capture_reward

    # --------------------
    # 3. 时间惩罚（鼓励尽快完成）
    # --------------------
    time_penalty = -1.0
    components['time_penalty'] = w_time_penalty * time_penalty

    # --------------------
    # 4. 追捕者间防碰撞与距离过近惩罚
    # --------------------
    collision_penalty = 0.0
    near_collision_penalty = 0.0

    adv_sizes = [a.size for a in adversaries]
    self_size = current_agent.size

    for other_adv in adversaries:
        if other_adv is current_agent:
            continue
        other_idx = all_agents.index(other_adv)
        dist = inter_agent_distances[agent_idx, other_idx]
        other_size = other_adv.size
        min_dist = self_size + other_size
        safe_dist = safe_distance_factor * min_dist

        if dist < min_dist:
            collision_penalty -= 1.0
        elif dist < safe_dist:
            # 距离越接近 min_dist，惩罚越大
            ratio = (safe_dist - dist) / (safe_dist - min_dist)
            ratio = np.clip(ratio, 0.0, 1.0)
            near_collision_penalty -= ratio

    components['collision_penalty'] = w_collision_penalty * collision_penalty
    components['near_collision_penalty'] = (
        w_collision_near_penalty * near_collision_penalty
    )

    # --------------------
    # 5. 队形：均匀包围圈
    #    需要 3 个追捕者的位置和猎物位置
    # --------------------
    adv_positions = np.array([a.state.p_pos for a in adversaries], dtype=np.float32)
    prey_pos = prey_position.astype(np.float32)

    # 5.1 半径一致性：追捕者到猎物距离的方差越小越好
    r_mean = np.mean(r_all)
    r_var = np.var(r_all)
    max_reasonable_radius = compact_radius_factor * capture_threshold
    if r_mean > 1e-6:
        normalized_r_var = r_var / (max_reasonable_radius ** 2 + 1e-6)
    else:
        normalized_r_var = 0.0
    normalized_r_var = np.clip(normalized_r_var, 0.0, 1.0)
    radius_uniform_reward = -normalized_r_var
    components['radius_uniform_reward'] = w_radius_uniform * radius_uniform_reward

    # 5.2 角度均匀性：3 个追捕者相对猎物的角度差接近 120 度
    rel_vecs = adv_positions - prey_pos
    angles = np.arctan2(rel_vecs[:, 1], rel_vecs[:, 0])
    angles = np.sort(angles)

    # 角度差（考虑环绕 2π）
    angle_diffs = np.empty(3, dtype=np.float32)
    angle_diffs[0] = angles[1] - angles[0]
    angle_diffs[1] = angles[2] - angles[1]
    angle_diffs[2] = (angles[0] + 2 * np.pi) - angles[2]

    ideal_diff = 2.0 * np.pi / 3.0
    angle_diff_err = np.mean(np.abs(angle_diffs - ideal_diff))
    # 归一化到 [0, 1]
    max_angle_err = np.pi  # 最大误差设为 pi
    normalized_angle_err = np.clip(angle_diff_err / max_angle_err, 0.0, 1.0)
    angle_uniform_reward = -normalized_angle_err
    components['angle_uniform_reward'] = w_angle_uniform * angle_uniform_reward

    # 5.3 猎物是否在追捕者三角形内部（闭合包围）
    # 使用重心坐标法判断点在三角形内
    def point_in_triangle(p, a, b, c):
        v0 = c - a
        v1 = b - a
        v2 = p - a

        dot00 = np.dot(v0, v0)
        dot01 = np.dot(v0, v1)
        dot02 = np.dot(v0, v2)
        dot11 = np.dot(v1, v1)
        dot12 = np.dot(v1, v2)

        denom = dot00 * dot11 - dot01 * dot01
        if np.abs(denom) < 1e-8:
            return False

        inv_denom = 1.0 / denom
        u = (dot11 * dot02 - dot01 * dot12) * inv_denom
        v = (dot00 * dot12 - dot01 * dot02) * inv_denom

        return (u >= 0) and (v >= 0) and (u + v <= 1)

    a_pos = adv_positions[0]
    b_pos = adv_positions[1]
    c_pos = adv_positions[2]
    in_center = point_in_triangle(prey_pos, a_pos, b_pos, c_pos)
    center_reward = 1.0 if in_center else 0.0
    components['center_reward'] = w_center_in_triangle * center_reward

    # 5.4 包围圈紧凑度：平均半径需要足够小
    compact_radius = capture_threshold
    compact_ratio = r_mean / (compact_radius * compact_radius_factor + 1e-6)
    compact_ratio = np.clip(compact_ratio, 0.0, 2.0)
    compact_reward = -compact_ratio
    components['compact_reward'] = w_compact_circle * compact_reward

    total_reward = sum(components.values())
    return float(total_reward), components
```
```
