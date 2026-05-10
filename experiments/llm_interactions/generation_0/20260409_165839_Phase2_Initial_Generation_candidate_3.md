# LLM Interaction Log

- **Generation**: 0
- **Phase**: Phase2_Initial_Generation
- **Candidate Info**: 3
- **Timestamp**: 20260409_165839

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
下面是对给定 Dec-POMDP 环境的代码级诊断，从五个维度逐条分析。整个回答只用于帮助理解环境结构，为后续设计奖励函数做准备，不包含任何奖励函数代码。

---

## 一、实现细节 (Implementation Details)

### 1. 使用到的依赖包

从代码片段可以看到主要依赖如下：

- 标准数值与科学计算：
  - `numpy as np`  
    - 用于向量运算（如位置差、距离矩阵）、数组构造、拼接观测向量等。

- 环境接口相关：
  - `gymnasium`  
    - 新版的 Gym 接口库，用于环境规范（如 Space 定义、step/reset 等）。  
    - 这里虽然没有直接出现 Spaces 的定义，但可以推断环境整体是 gymnasium 风格。

- 工具类：
  - `from gymnasium.utils import EzPickle`  
    - 常用于包装环境类，使其可被 `pickle`（方便并行/保存）。

- PettingZoo MPE（Multi-Agent Particle Environment）框架：
  - `from pettingzoo.mpe._mpe_utils.core import Agent, Landmark, World`  
    - `Agent`: 智能体基本结构，含 `state`（位置、速度）、`adversary` 标志、`max_speed`、`size` 等。
    - `Landmark`: 通常为静态物体（本片段中没有使用，但属于标准 MPE 组件）。
    - `World`: 物理世界容器，包含 `agents` 列表等。
  - `from pettingzoo.mpe._mpe_utils.scenario import BaseScenario`  
    - 用于定义场景（初始化世界、重置、奖励、观测等）。
  - `from pettingzoo.mpe._mpe_utils.simple_env import SimpleEnv, make_env`  
    - 一般用于把场景包装成标准 PettingZoo环境（AECEnv/parallel env）。
  - `from pettingzoo.utils.conversions import parallel_wrapper_fn`  
    - 将 AEC 环境包装为 parallel 环境接口。

- 自定义模块（工程内依赖）：
  - `from .custom_agents_dynamics import CustomWorld`  
    - 自定义的 World 类，可能覆盖 MPE 默认物理（如加速度/阻尼/碰撞处理）。此处没有展开，但 compute_reward 会间接依赖其状态。
  - `from . import reward_function`  
    - 可插拔的奖励函数模块（只包含追捕者奖励）。  
    - 重要：我们的奖励设计将在这个模块中实现，但在当前任务中只分析状态与观测，不编写任何奖励代码。

### 2. 是否引入外部未定义变量

在 `CoreEnvLogic` 片段内部使用的变量来源情况：

- 明确在 `__init__` 中定义的属性：
  - `self.world_size`
  - `self.max_force`
  - `self.capture_threshold`
  - `self.adversary_params`
  - `self.agent_params`

- 在方法中使用的对象与字段：
  - `agent.state.p_pos`, `agent.state.p_vel`
  - `agent.max_speed`, `agent.adversary`
  - `world.agents`（一个 `Agent` 列表）
  - 对于逃跑者、追捕者的 `size` 实际上在 `is_collision` 中是通过参数显式传入的：`agent1_size`, `agent2_size`。

在这段伪代码内部，没有直接引用未定义的变量或属性；所有使用到的物理参数（size, max_speed）要么通过 `self.adversary_params/self.agent_params`，要么在函数参数中给出。

**但需要注意：**

- 文档中明确指出：`CoreEnvLogic` 是“文档伪代码”，真实运行时不会实例化此类，也不会存在诸如 `world.logic` 这样的属性。  
- 真正用于训练时的 `compute_reward` 不允许访问：
  - `CoreEnvLogic()`
  - `world.logic`
  - `world.adversary_params` / `world.agent_params` / `world.world_size` 等此类字段。
- 所需的物理常量（size、max_speed、world_size、capture_threshold 等）必须在 `compute_reward` 内部硬编码为局部变量。

因此，从逻辑上讲，当前伪代码**没有“未定义变量”的错误**，但如果在真实 `compute_reward` 中照抄这里的访问方式，就会出现“访问不存在属性/类”的运行时错误。这就是 API 边界问题（在第五部分详细说）。

---

## 二、环境结构 (Environment Structure)

### 1. 全局状态 (Global State) 结构与物理意义

`_build_global_state(self, agent, world)` 返回的是一个字典 `global_state`，其中包含：

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

1. **`agent_positions`**: `np.array` 形状为 `(N, 2)`  
   - 内容：`[a.state.p_pos for a in all_agents]`  
   - 物理意义：所有智能体（追捕者 + 逃跑者）的**绝对位置**坐标（x, y），单位为世界坐标。  
   - 对围捕任务而言，这是**完整空间拓扑信息**的基础，可用于计算任何相对距离、角度、几何配置。

2. **`agent_velocities`**: `np.array` 形状为 `(N, 2)`  
   - 内容：`[a.state.p_vel for a in all_agents]`  
   - 物理意义：所有智能体的**绝对速度向量**（vx, vy）。  
   - 对任务来说，可以用来判断追捕者是否在积极接近目标（速度指向目标）或是否围绕目标做环绕运动。

3. **`prey_position` (`prey_pos`)**: `np.array` 形状为 `(2,)`  
   - 物理意义：唯一逃跑者的绝对位置。如果 `preys` 为空，退化为零向量（但是正常任务应始终存在 1 个逃跑者）。  
   - 是围捕几何中心的关键：包围圈应围绕这个点形成。

4. **`prey_velocity` (`prey_vel`)**: `np.array` 形状为 `(2,)`  
   - 物理意义：逃跑者的绝对速度向量。  
   - 对高阶奖励设计可以用于：鼓励对逃跑方向的预判、从逃跑方向两侧包夹等，不过这是奖励层面的选择。

5. **`distances_to_prey`**: `np.array` 形状为 `(num_adversaries,)`  
   - 内容：`np.linalg.norm(adv.state.p_pos - prey_pos)` 对每个追捕者计算。  
   - 物理意义：每个追捕者到猎物的欧氏距离。  
   - 对任务目标的直接相关性：
     - **负相关**：距离越小，说明追捕者越接近目标，更接近成功包围或捕获。
     - 这是最基础的追捕进度指标。

6. **`inter_agent_distances`**: `np.array` 形状为 `(N, N)`  
   - 内容：所有智能体之间的欧氏距离矩阵：  
     `inter_agent_distances[i][j] = || agent_positions[i] - agent_positions[j] ||`  
   - 物理意义：全体智能体之间的距离关系图。
   - 对任务的关键作用：
     - 用于判断追捕者之间是否距离过近（碰撞风险/违反间隔约束）。
     - 也可用于判断追捕者之间是否形成“均匀分布”环绕目标（相邻追捕者间的角度/弦长是否接近）。

7. **`is_adversary`**: `agent.adversary` （布尔值）  
   - 物理意义：标识当前正在为之计算奖励的智能体是否为追捕者。  
   - 在奖励函数中可以用来：
     - 若环境只对追捕者提供奖励，可在这里分支；不过通常在外层已经按 agent 类型分别调用不同的 reward 逻辑。

8. **`world_size`**: 标量 `self.world_size = 2.5`  
   - 物理意义：地图坐标范围为 `(-world_size, world_size)`，即二维正方形边界 [-2.5, 2.5]^2。  
   - 用途：
     - 归一化位置或距离（例如 `dist / world_size`）。
     - 判断是否贴边、是否追到角落等。

9. **`capture_threshold`**: 标量 `self.capture_threshold = 0.5`  
   - 注释中：`# capture_threshold = world_size * 0.2`，对当前 world_size=2.5 确实为 0.5。  
   - 物理意义：围捕判定的距离尺度（但在这段代码中仅作为全局常量传出，并未在 `_build_global_state` 内部使用）。  
   - 在奖励设计/成功终止条件中可用于定义“包围圈半径”或“允许误差”。

### 2. 局部观测空间 (Observation) 结构与相对信息

`observation(self, agent, world)` 返回一个 `np.ndarray`，构造方式为：

```python
# 自身状态（归一化）
norm_self_vel = agent.state.p_vel / agent.max_speed
norm_self_pos = agent.state.p_pos / self.world_size

# 其他智能体相对位置(归一化) & 猎物速度
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

拆解：

1. **自身速度 `norm_self_vel`**: 形状 `(2,)`  
   - `agent.state.p_vel / agent.max_speed`  
   - 物理意义：
     - 当前智能体速度向量按最大速度归一化，分量范围约在 [-1, 1]。
     - 描述“朝向/速度大小”。

2. **自身位置 `norm_self_pos`**: 形状 `(2,)`  
   - `agent.state.p_pos / self.world_size`  
   - 物理意义：
     - 绝对位置归一化到 [-1, 1] 范围内。
     - 不受局部视野限制：智能体知道自己在世界中的全局坐标（即这是一个**位置全局可见**的设置）。

3. **其他智能体相对位置 `other_pos`**: 列表中每个元素形状 `(2,)`  
   - 对每个 `other in world.agents`（除自身）：
     - `rel_pos = (other.state.p_pos - agent.state.p_pos) / self.world_size`
   - 物理意义：
     - 所有其他智能体相对于**自身**的位置（相对坐标），再以 world_size 归一化。
     - 追捕者与逃跑者的相对位置信息都在其中，不区分身份。

4. **其他智能体的速度 `other_vel`**: 仅包含**逃跑者**的速度  
   - 对 `other`，如果 `not other.adversary`（即猎物）：
     - `other.state.p_vel / other.max_speed`
   - 物理意义：
     - 只暴露逃跑者（猎物）的归一化速度向量。
     - 不暴露其他追捕者的速度（追捕者之间不知道彼此速度，只知道相对位置）。

5. 最终观测向量拼接顺序（对每个智能体）：

   ```
   [ norm_self_vel (2),
     norm_self_pos (2),
     rel_pos_of(agent 1 to agent i),
     rel_pos_of(agent 2 to agent i),
     ...,
     prey_velocity (2) ]
   ```

   在你给定任务中，智能体总数是 4（3 追捕 + 1 逃跑），因此：

   - 对任一追捕者：
     - 自身速度: 2
     - 自身位置: 2
     - 其他 3 个智能体的相对位置: 3 × 2 = 6
     - 猎物速度: 2
     - 合计维度：2 + 2 + 6 + 2 = **12 维观测**（前提：按照默认顺序，`world.agents` 中三个追捕 + 一个猎物；`other_vel` 只添加猎物的速度一次）。

   - 对逃跑者：
     - 自身速度: 2
     - 自身位置: 2
     - 其他 3 个追捕者的相对位置: 3 × 2 = 6
     - `other_vel` 对其他追捕者不添加速度（因为都是 adversary），因此没有其他速度项。
     - 合计维度：2 + 2 + 6 = **10 维观测**。

   这说明：**不同类型智能体的观测维度不完全相同**（追捕者多一个“猎物速度”项），这在 MARL 中是允许的，只要在环境定义时处理好不同 agent 的 observation_space。

---

## 三、智能体交互 (Agent Interactions)

### 1. 追捕者 vs 猎物的物理属性差异

在 `__init__` 中：

```python
self.adversary_params = {'size': 0.075, 'max_speed': 1.0}
self.agent_params = {'size': 0.050, 'max_speed': 1.3}
```

结合任务描述：

- 追捕者（adversaries）：
  - `size = 0.075`
    - 碰撞半径较大，意味着：
      - 彼此更容易碰撞（距离阈值更大）。
      - 对猎物的“覆盖面积”更大，可能更容易锁定目标的逃逸区域。
  - `max_speed = 1.0`
    - 速度较慢；追捕者单体在长距离追赶上吃亏，但通过协同围堵可以弥补。

- 逃跑者（prey / agent_params）：
  - `size = 0.050`
    - 碰撞半径较小，不易被“撞上”，更灵活。
  - `max_speed = 1.3`
    - 明显快于追捕者，单个追捕者追上它会很难，需要三者协作制造封锁圈。

**物理直觉**：  
这是典型的“多慢追快”结构，强调**协同围捕**而非单向追逐。逃跑者更小更快，追捕者更大更慢。

### 2. 成功捕获（Collision）的数学与物理条件

`is_collision` 函数如下：

```python
def is_collision(self, agent1_pos, agent1_size, agent2_pos, agent2_size):
    """碰撞检测：欧氏距离 < 半径之和"""
    delta_pos = agent1_pos - agent2_pos
    dist = np.sqrt(np.sum(np.square(delta_pos)))
    dist_min = agent1_size + agent2_size
    return dist < dist_min
```

解释：

- **数学条件**：  
  设两智能体位置分别为 `p1`, `p2`，大小（半径）为 `r1`, `r2`，则：

  \[
  \text{collision} \iff \|p_1 - p_2\|_2 < r_1 + r_2
  \]

- 对本任务中的几种交互：

  - 追捕者-逃跑者碰撞：
    - 距离阈值约为 `0.075 + 0.050 = 0.125`。
  - 追捕者-追捕者碰撞：
    - 距离阈值约为 `0.075 + 0.075 = 0.150`。
  - 从世界尺度来看，世界半边长为 2.5，这些阈值相对世界范围很小。

- **物理意义**：
  - 若追捕者和逃跑者距离小于两者半径之和，视作“接触”——通常可以视作“抓住”或“碰撞到”猎物。
  - 若追捕者与追捕者之间距离小于阈值，则发生队友碰撞，是负面事件（应在奖励中惩罚）。

注意：  
- 上述 `is_collision` 只是一个**工具函数**，实际环境里是否使用它来终止 episode / 给奖励，取决于外层逻辑。
- 此处尚无“围捕完成”的定义，`capture_threshold`（0.5）比碰撞距离大得多，更像是“包围圈半径尺度”而不是“硬碰撞”。

---

## 四、任务相关信息 (Task-relevant Information)

（你在问题中这部分写了两遍，我在这里一次性回答，不重复。）

### 1. 围捕任务的核心目标与哪些变量直接相关？

任务目标列表：

1. 追捕者需要接近并包围目标  
2. 追捕者之间避免碰撞  
3. 形成均匀的包围圈  
4. 尽可能短时间完成围捕

从 `global_state` 和 `observation` 中已有变量出发，可以对应如下：

#### (1) 接近并包围目标 —— 主要与以下变量相关

- `distances_to_prey`（追捕者→猎物距离数组）：
  - 距离越小，越接近捕获。
  - 多个追捕者距离同时减小时，意味着集体收缩包围圈。
  - 对任务目标而言：对每个追捕者，**距离与成功围捕程度负相关**。

- `prey_position` 与 `agent_positions`：
  - 可以构造更多几何指标：
    - 每个追捕者到猎物的向量 `p_adv - p_prey`。
    - 追捕者在猎物周围的角度分布（极坐标角度）。
  - 一般会在奖励/终止条件中用这些信息判定“已形成围捕”而不仅仅是靠一个碰撞。

#### (2) 追捕者之间避免碰撞 —— 主要与以下变量相关

- `inter_agent_distances`：
  - 追捕者之间任意两两距离中若有值 < `size_i + size_j` 则发生碰撞。
  - 即：对追捕者 i, j：
    - `inter_agent_distances[i][j]` 与 (0) 应该有一个**下界约束**，避免太近。
  - 对目标而言：**追捕者间距离与“安全性”正相关**（过近则不安全，过远则可能漏网）。

- `adversary_params['size']` / `agent_params['size']`（在实际 reward 中需硬编码）：
  - 决定碰撞阈值：越小越不易撞；在当前设定中追捕者之间更容易“撞到”。

#### (3) 形成均匀的包围圈 —— 与以下变量高度相关

- `agent_positions` + `prey_position`：
  - 可通过这些构造：
    - 对每个追捕者：
      - 相对向量 `v_i = p_adv_i - p_prey`
      - 对应的极角 `θ_i = atan2(v_i.y, v_i.x)`
      - 对应的半径 `r_i = ||v_i||`
    - 均匀包围圈的典型特征：
      - 所有 `r_i` 大致相等（等半径），即追捕者距离猎物差不多。
      - `θ_i` 在 [0, 2π) 内近似等间隔，比如三个追捕者约为 0°, 120°, 240°。
  - 用 `distances_to_prey` 可以控制 `r_i` 一致性（半径均匀性）。
  - 用 `agent_positions` / `inter_agent_distances` 可以间接约束角度差（相邻追捕者在猎物附近的相对弦长 / 角度）。

- `capture_threshold`：
  - 可被用于定义“包围半径”的目标值，例如希望 `r_i ≈ capture_threshold`。
  - 太小接近碰撞，太大包围不紧；0.5 是世界尺度 2.5 的 20%，是一个中等大小的围捕圈半径。

#### (4) 尽可能短时间完成围捕 —— 时间相关信息

虽然当前代码片段没有显式提供时间步 `t` 或剩余步数，但在整个环境中通常还会有：

- episode step 计数（外部可见）。
- training loop 中的 `done` / episode length。

与现有变量的间接关系：

- 若在 reward 中只使用 `global_state` 和新增的时间步变量，可以：
  - 对每一时间步施加时间惩罚（-1 per step）以鼓励更快完成。
  - 或将“达成包围圈时的奖励”不随时间衰减/或者随时间衰减，从而鼓励更短路径。

总结变量与任务目标的直接关系：

- **正相关（需要增大或趋于某目标值）：**
  - 追捕者与猎物之间角度均匀分布（由 `agent_positions` + `prey_position` 派生）。
  - 追捕者-追捕者间距离保持在“适中范围”（既不太近也不太远）。
  - `distances_to_prey` 接近某目标半径，比如 `capture_threshold`，不是简单越小越好（否则就是简单逼近，而不是环绕）。

- **负相关（需要减小）：**
  - `distances_to_prey`（在“未形成包围圈”阶段，越小越接近捕获）。
  - 追捕者-追捕者过近，`inter_agent_distances` 低于某阈值表示危险。
  - 节点到猎物的半径标准差（半径不均匀时标准差大，需减小）。

### 2. 表征高质量“包围圈”的空间拓扑

在 3 追捕 + 1 逃跑的 2D 平面中，高质量包围圈具有以下空间特征——这些特征可以全部从 `agent_positions` 和 `prey_position` 中计算出来：

1. **猎物是几何中心**  
   - 逃跑者的位置 `p_prey` 接近追捕者位置的几何中心：
     \[
     \bar{p}_{adv} = \frac{1}{3}\sum_{i=1}^3 p_{adv,i}
     \]
     \[
     \|\bar{p}_{adv} - p_{\text{prey}}\| \approx 0
     \]
   - 即追捕者在猎物周围近似对称分布，而不是集中在某一侧。

2. **每个追捕者与猎物距离相近（半径均匀）**  
   - 定义 `r_i = ||p_{adv,i} - p_{prey}||`，希望：
     - 所有 `r_i` 接近某目标半径 `r_target`（如 ≈ capture_threshold）。
     - 半径的标准差 `std(r_1, r_2, r_3)` 较小。
   - 若半径均匀，则包围圈不会一侧太近、一侧太远，有利于堵截。

3. **角度均匀分布（环绕角度间隔接近 120°）**  
   - 定义相对向量 `v_i = p_{adv,i} - p_{prey}`，极角：
     \[
     \theta_i = \text{atan2}(v_i^y, v_i^x)
     \]
   - 对排序后的角度 \(\theta_{(1)} < \theta_{(2)} < \theta_{(3)}\)，希望：
     - 相邻差值约为 \(2\pi/3\)（120°），即：
       \[
       \theta_{(2)} - \theta_{(1)} \approx \frac{2\pi}{3},\quad
       \theta_{(3)} - \theta_{(2)} \approx \frac{2\pi}{3},\quad
       (\theta_{(1)} + 2\pi) - \theta_{(3)} \approx \frac{2\pi}{3}
       \]
   - 角度间隔的均匀性是衡量包围圈“均匀”的核心指标。

4. **追捕者之间的距离不小于碰撞阈值，且不过分远离**  
   - 对三追捕者 i, j，距离 `d_ij = ||p_i - p_j||` 应满足：
     - `d_ij > size_i + size_j`（避免碰撞）。
     - 同时，若 `d_ij` 过大，会在包围圈留下空隙，让猎物可以逃出。理想情况下，它与半径和角度一起决定：
       \[
       d_{ij} \approx 2 r \sin(\Delta\theta/2)
       \]
     - 在均匀包围中，`d_ij` 也会呈现一定的均衡分布。

5. **逃跑者被限制在包围圈内且相对静止 / 难以突破**（更加高阶的性质）
   - 逃跑者的速度方向可能与某个追捕者形成对峙；在稳定的高质量包围圈中：
     - 逃跑者的有效逃逸方向被各追捕者阻断：每一个可行的逃跑方向都指向某个追捕者。
   - 这在 reward 中可以通过逃跑者速度与各追捕者相对位置的角度关系来刻画，但这是更复杂的设计。

综上：  
**高质量包围圈 = 半径均匀 + 角度均匀 + 避免追捕者间碰撞 + 猎物接近追捕者几何中心**。  
所有这些拓扑特征都可以从 `global_state` 中的 `agent_positions` 与 `prey_position` 推导出来（不需要额外信息）。

---

## 五、API 边界隔离 (API Boundaries)

这部分非常关键，关系到奖励函数实现时不出错。

### 1. `CoreEnvLogic` 是文档伪代码，不会在运行时出现

代码中：

```python
class CoreEnvLogic:
    """
    环境核心逻辑
    用于辅助设计 Reward Function
    """
    ...
```

并且任务说明明确指出：

> 上方代码片段中的 `CoreEnvLogic` 类仅是文档伪代码，用于辅助理解物理概念，绝非可以在运行时被实例化的真实类。

这意味着：

- 在真正的环境实现中（尤其是 `reward_function.compute_reward(...)` 内）：
  - **不能** `logic = CoreEnvLogic()`，
  - **不能** 假设有 `world.logic`、`world.adversary_params`、`world.agent_params`、`world.world_size` 等字段存在。

`CoreEnvLogic` 的作用只是向你展示：

- 物理含义（如 `world_size`、`capture_threshold`、`size`、`max_speed` 等）。
- 全局状态 `global_state` 可能包含的字段及其含义。
- 单个 agent 的观测向量如何构造。

### 2. 在 `compute_reward` 中获取物理常量的正确方式

任务说明要求：

> 在真正的 `compute_reward` 函数中，禁止调用 `CoreEnvLogic()` 或访问 `world.logic`、`world.adversary_params` 等不存在的属性。  
> 如果需要物理常量（如 `size`、`max_speed`、`world_size` 等），必须在函数内部以局部变量的形式硬编码声明，例如：`adv_size = 0.075`。

因此，在真实的 `compute_reward` 内部：

- 若需要追捕者的大小、速度上限、世界尺寸、围捕阈值等：
  - 必须直接在函数开头定义为局部常量，例如：
    - `world_size = 2.5`
    - `capture_threshold = 0.5`
    - `adv_size = 0.075`
    - `prey_size = 0.050`
    - `adv_max_speed = 1.0`
    - `prey_max_speed = 1.3`
- 不允许写类似：
  - `world_size = world.world_size`
  - `adv_size = world.adversary_params['size']`
  - `logic = CoreEnvLogic()` 然后去访问 `logic.world_size` 等。

这样做的原因：

- 实际运行时的环境类可能和这里伪代码不一致，不一定包含这些属性。
- `reward_function` 模块设计为**只依赖传入的 `agent` 和 `world` 的标准 MPE 接口**和**本地硬编码常量**。
- 这样 reward 函数就不会因环境内部结构变化（如换成 `CustomWorld`）而失效。

### 3. 允许访问的对象与字段

即使不能访问 `world.logic` 等，也可以使用标准的 MPE 接口中常见字段：

- `world.agents`（智能体列表）
- `agent.state.p_pos`, `agent.state.p_vel`  
- `agent.adversary`（布尔）  
- `agent.size`, `agent.max_speed`（如果在 `CustomWorld` 中有设置，且对所有智能体一致；但为了稳定性，仍建议用硬编码常量）

以及，`_build_global_state` 已经构造好了 `global_state` 字典（在真实代码中，reward 函数很可能就是基于类似结构调用的），因此在 `compute_reward` 中**优先使用 `global_state` 提供的字段**，而不是自己到 `world` 里再挖。

---

总结一下五个维度的诊断结论：

1. **实现细节**：  
   - 使用 numpy、gymnasium 以及 PettingZoo MPE 基础组件，外加工程内 `CustomWorld` 和 `reward_function`，没有内部未定义变量，但 `CoreEnvLogic` 不是真实可用类。

2. **环境结构**：  
   - `global_state` 提供了所有智能体的绝对位置、速度、猎物位置/速度、追捕者到猎物距离、智能体间距离矩阵、以及用于尺度归一化的 `world_size` 和 `capture_threshold`。  
   - 局部观测包括自身速度、位置（归一化）、其他智能体相对位置以及猎物速度，因此每个追捕者基本拥有“全局位置信息 + 局部相对信息”。

3. **智能体交互**：  
   - 追捕者比猎物大而慢（size=0.075, max_speed=1.0），猎物更小更快（size=0.050, max_speed=1.3），适合研究“慢群追快单”的协同围捕。  
   - 碰撞条件是欧氏距离小于半径和，对追捕者-猎物和追捕者-追捕者都是统一规则。

4. **任务相关信息**：  
   - 围捕与 `distances_to_prey`、`inter_agent_distances`、`agent_positions` 与 `prey_position` 的几何关系密切相关。  
   - 高质量包围圈表现为：追捕者在猎物周围角度均匀、半径相近、互不碰撞且猎物接近几何中心。

5. **API 边界**：  
   - `CoreEnvLogic` 仅是文档示意，不可在真正环境中实例化或访问。  
   - 在 `compute_reward` 中必须硬编码物理常量、仅依赖传入的 `agent`、`world` 及 `global_state`，不能用 `world.logic` 或其他未保证存在的属性。

以上分析为后续奖励函数设计提供了清晰的状态与拓扑信息基础。后续若需要，我们可以在此基础上进一步抽象出“围捕成功”的几何判定逻辑，但不在本回答中实现任何奖励代码。

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
    # 非追捕者零奖励
    if not global_state["is_adversary"]:
        return 0.0, {}

    components = {}

    # -----------------------
    # 物理与任务常量（硬编码）
    # -----------------------
    world_size = 2.5
    capture_threshold = 0.5

    adv_size = 0.075
    prey_size = 0.050

    adv_max_speed = 1.0
    prey_max_speed = 1.3

    # 目标包围半径（略小于捕获阈值，兼顾稳定与安全距离）
    target_radius = capture_threshold * 0.9

    # 形状控制系数
    eps = 1e-6

    # -----------------------
    # 从 global_state 中解析全局信息
    # -----------------------
    agent_positions = global_state["agent_positions"]
    agent_velocities = global_state["agent_velocities"]
    prey_pos = global_state["prey_position"]
    prey_vel = global_state["prey_velocity"]
    distances_to_prey = global_state["distances_to_prey"]
    inter_agent_distances = global_state["inter_agent_distances"]

    # 根据命名约定解析索引：
    # 假设 agent_name 为 "adversary_0", "adversary_1", "adversary_2"
    # world.agents 中 adversaries 在前，顺序与命名一致
    try:
        adv_index = int(agent_name.split("_")[-1])
    except Exception:
        # 若解析失败，退化为第一个追捕者索引
        adv_index = 0

    all_agents = world.agents
    adversaries = [a for a in all_agents if a.adversary]
    preys = [a for a in all_agents if not a.adversary]

    num_adversaries = len(adversaries)
    num_agents = len(all_agents)

    # 构造索引映射：全局索引 -> adversary 局部索引
    adv_global_indices = []
    for i, ag in enumerate(all_agents):
        if ag.adversary:
            adv_global_indices.append(i)

    # 当前追捕者在 world.agents 中的全局索引
    if adv_index < len(adv_global_indices):
        global_adv_index = adv_global_indices[adv_index]
    else:
        global_adv_index = adv_global_indices[0]

    # 猎物全局索引（假设单一猎物）
    prey_global_index = None
    for i, ag in enumerate(all_agents):
        if not ag.adversary:
            prey_global_index = i
            break

    # -----------------------
    # 1. 距离引导：鼓励接近目标 & 停留在合理半径
    # -----------------------
    # 当前追捕者到猎物的距离
    if adv_index < len(distances_to_prey):
        dist_to_prey = float(distances_to_prey[adv_index])
    else:
        # 兜底：直接从位置计算
        adv_pos = agent_positions[global_adv_index]
        dist_to_prey = float(np.linalg.norm(adv_pos - prey_pos))

    # 基础接近奖励：距离越小越好
    max_dist = np.sqrt(2.0) * world_size
    normalized_dist = np.clip(dist_to_prey / (max_dist + eps), 0.0, 1.0)
    distance_reward = 1.0 - normalized_dist

    # 额外：鼓励靠近目标包围半径 target_radius
    radius_error = abs(dist_to_prey - target_radius)
    radius_error_norm = np.clip(radius_error / (max_dist + eps), 0.0, 1.0)
    ring_shaping_reward = 1.0 - radius_error_norm

    components["distance_progress"] = 0.5 * distance_reward
    components["radius_alignment"] = 0.5 * ring_shaping_reward

    # -----------------------
    # 2. 追捕者-追捕者防碰撞与间距惩罚
    # -----------------------
    collision_penalty = 0.0
    spacing_penalty = 0.0

    # 最小安全间距（略大于碰撞阈值）
    min_safe_dist = 1.2 * (2 * adv_size)

    for j, other in enumerate(all_agents):
        if j == global_adv_index:
            continue
        if not other.adversary:
            continue

        d = inter_agent_distances[global_adv_index, j]

        # 碰撞检测
        if d < 2 * adv_size:
            collision_penalty -= 1.0

        # 距离过近的惩罚
        if d < min_safe_dist:
            spacing_penalty -= (min_safe_dist - d) / (min_safe_dist + eps)

    components["collision_penalty"] = collision_penalty
    components["spacing_penalty"] = spacing_penalty

    # -----------------------
    # 3. 队形奖励：均匀环绕（半径一致 + 角度均匀）
    # -----------------------
    formation_radius_reward = 0.0
    formation_angle_reward = 0.0

    if num_adversaries >= 2 and prey_global_index is not None:
        # 计算所有追捕者相对猎物的半径与角度
        adv_rel_vectors = []
        adv_radii = []
        adv_angles = []

        for idx, adv in enumerate(adversaries):
            g_idx = adv_global_indices[idx]
            rel = agent_positions[g_idx] - prey_pos
            r = np.linalg.norm(rel)
            theta = np.arctan2(rel[1], rel[0])
            adv_rel_vectors.append(rel)
            adv_radii.append(r)
            adv_angles.append(theta)

        adv_radii = np.array(adv_radii, dtype=float)
        adv_angles = np.array(adv_angles, dtype=float)

        # 半径一致性：标准差越小越好
        radius_std = float(np.std(adv_radii))
        max_possible_radius_std = max_dist
        radius_std_norm = np.clip(
            radius_std / (max_possible_radius_std + eps), 0.0, 1.0
        )
        formation_radius_reward = 1.0 - radius_std_norm

        # 角度均匀性：差分接近均匀分布
        # 排序角度并计算环上的间隔
        angles_sorted = np.sort(adv_angles)
        angle_diffs = np.diff(angles_sorted)
        # 闭环间隔
        last_gap = (angles_sorted[0] + 2 * np.pi) - angles_sorted[-1]
        angle_diffs = np.concatenate([angle_diffs, [last_gap]])

        # 理想间隔
        ideal_gap = 2 * np.pi / num_adversaries
        angle_gap_error = np.abs(angle_diffs - ideal_gap)
        angle_gap_error_mean = float(np.mean(angle_gap_error))
        max_gap_error = np.pi  # 最大误差不超过 180 度
        angle_gap_error_norm = np.clip(
            angle_gap_error_mean / (max_gap_error + eps), 0.0, 1.0
        )
        formation_angle_reward = 1.0 - angle_gap_error_norm

    components["formation_radius"] = 0.5 * formation_radius_reward
    components["formation_angle"] = 0.5 * formation_angle_reward

    # -----------------------
    # 4. 围捕完成奖励：多追捕者同时围住猎物
    # -----------------------
    capture_bonus = 0.0
    if (
        num_adversaries == 3
        and prey_global_index is not None
        and len(preys) == 1
    ):
        # 条件1：所有追捕者都在一定半径内（收缩完成）
        all_close = np.all(distances_to_prey < capture_threshold)

        # 条件2：角度覆盖良好（最小角间隔不太小）
        if num_adversaries >= 3:
            # 复用上面计算的角度
            adv_rel_vectors = []
            adv_angles = []
            for idx, adv in enumerate(adversaries):
                g_idx = adv_global_indices[idx]
                rel = agent_positions[g_idx] - prey_pos
                theta = np.arctan2(rel[1], rel[0])
                adv_rel_vectors.append(rel)
                adv_angles.append(theta)
            adv_angles = np.array(adv_angles, dtype=float)
            angles_sorted = np.sort(adv_angles)
            angle_diffs = np.diff(angles_sorted)
            last_gap = (angles_sorted[0] + 2 * np.pi) - angles_sorted[-1]
            angle_diffs = np.concatenate([angle_diffs, [last_gap]])

            min_gap = float(np.min(angle_diffs))
            # 至少覆盖 90 度间隔
            angle_good = min_gap > (np.pi / 2.0)
        else:
            angle_good = False

        # 条件3：猎物靠近追捕者几何中心
        adv_positions = np.array(
            [agent_positions[g_idx] for g_idx in adv_global_indices],
            dtype=float,
        )
        center_adv = np.mean(adv_positions, axis=0)
        prey_center_dist = float(np.linalg.norm(center_adv - prey_pos))
        center_good = prey_center_dist < (capture_threshold * 0.5)

        if all_close and angle_good and center_good:
            # 高额一次性奖励，用于鼓励尽快完成围捕
            capture_bonus = 5.0

    components["capture_bonus"] = capture_bonus

    # -----------------------
    # 5. 时间惩罚：鼓励快速结束
    # -----------------------
    # 每一步小的负奖励
    time_penalty = -0.01
    components["time_penalty"] = time_penalty

    total_reward = float(sum(components.values()))
    return total_reward, components
```
```
