# LLM Interaction Log

- **Generation**: 0
- **Phase**: Phase2_Initial_Generation
- **Candidate Info**: 1
- **Timestamp**: 20260414_121239

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
下面按要求，从五个维度对这段环境逻辑做代码级诊断与任务结构分析（不写任何奖励函数代码）。

---

## 一、实现细节 (Implementation Details)

### 1. 使用的依赖包

从代码顶部的 import 可以明确看到：

- 标准库 / 科学计算
  - `import numpy as np`  
    - 用于向量/矩阵运算、范数计算、数组拼接等。

- 强化学习 / 环境接口
  - `import gymnasium`  
    - 新版 Gym 接口，用于环境封装（虽然在该片段中只引入未直接使用）。
  - `from gymnasium.utils import EzPickle`  
    - EzPickle 通常用于支持环境的序列化/反序列化（示例代码中未继续使用，但属于环境框架的一部分）。

- PettingZoo 多智能体环境工具
  - `from pettingzoo.mpe._mpe_utils.core import Agent, Landmark, World`  
    - MPE（Multiagent Particle Environment）核心类型，定义物理世界、智能体等。
  - `from pettingzoo.mpe._mpe_utils.scenario import BaseScenario`  
    - 用于自定义场景逻辑（init_world, reset, reward, observation 等）。
  - `from pettingzoo.mpe._mpe_utils.simple_env import SimpleEnv, make_env`  
    - 将场景封装成 PettingZoo 环境的工具。
  - `from pettingzoo.utils.conversions import parallel_wrapper_fn`  
    - 用于把环境转换为并行 API（parallel_env）版本。

- 本项目自定义模块
  - `from .custom_agents_dynamics import CustomWorld`  
    - 自定义的 World 类，可能对默认的 MPE 物理做了修改（例如动力学、加速度等）。
  - `from . import reward_function`  
    - 可插拔的奖励模块（只包含追捕者奖励），供真正的 `compute_reward` 使用。

### 2. 是否引入了外部未定义变量

在该片段中，`CoreEnvLogic` 内部使用的所有变量均是：

- 来自函数形参：
  - `agent`, `world`（在 `_build_global_state` 与 `observation` 内）
- 或类内固定常量：
  - `self.world_size`
  - `self.max_force`
  - `self.capture_threshold`
  - `self.adversary_params`
  - `self.agent_params`
- 或局部计算产生的变量：
  - 如 `all_agents`, `adversaries`, `preys`, `agent_positions`, `agent_velocities` 等。

代码中没有对未定义的外部变量（如 `world.logic`、`world.adversary_params` 等）进行访问；这些正是后面 API 边界要明确禁止在真正 reward 实现中使用的。

需要特别注意的是：

- 这里的 `CoreEnvLogic` 被明确声明为**文档伪代码**，不会在真实运行时被实例化，因此：
  - 真实的 `compute_reward` 中不能假设存在 `CoreEnvLogic` 实例，也不能使用其中的属性。
  - 所有需要的物理常数必须在 reward 函数内部以局部变量硬编码，例如：
    - `world_size = 2.5`
    - `adv_size = 0.075`
    - `prey_size = 0.050`
    - `capture_threshold = 0.5`

---

## 二、环境结构 (Environment Structure)

### 1. 状态空间 (Global State) —— `_build_global_state`

`_build_global_state` 返回的是传给 `compute_reward` 的 `global_state` 字典，其结构为：

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

逐项解释其数值结构和物理意义：

1. `agent_positions: np.ndarray`  
   - 形状：`(n_agents, 2)`  
   - 含义：环境中所有智能体（追捕者 + 逃跑者）的**绝对位置**，在世界坐标系下（未归一化），单位是 MPE 的位置单位。  
   - 注意：地图范围为 `(-world_size, world_size) = (-2.5, 2.5)`，因此位置坐标分量在 `[-2.5, 2.5]` 左右。

2. `agent_velocities: np.ndarray`  
   - 形状：`(n_agents, 2)`  
   - 含义：所有智能体的**绝对速度向量**，同样在世界坐标系下，单位是位置单位/时间步。  
   - 不同智能体的最大速度不同（追捕者 1.0，逃跑者 1.3），但这里未归一化。

3. `prey_position: np.ndarray`  
   - 形状：`(2,)`  
   - 含义：单个逃跑者的绝对位置 `p_pos`（若没有则为 `np.zeros(2)` 占位）。  
   - 由于场景中明确只有 1 个逃跑者，这里就是那个目标的位置。

4. `prey_velocity: np.ndarray`  
   - 形状：`(2,)`  
   - 含义：逃跑者的绝对速度向量（若没有则 `np.zeros(2)`）。

5. `distances_to_prey: np.ndarray`  
   - 形状：`(n_adversaries,)`，即本任务中为 `(3,)`。  
   - 含义：**每个追捕者到逃跑者的欧氏距离**：
     \[
     d_i = \|\text{pos}_i^{\text{adv}} - \text{pos}^{\text{prey}}\|_2
     \]
   - 这是围捕任务中最核心的几何量之一，直接反映“靠近/远离”程度。

6. `inter_agent_distances: np.ndarray`  
   - 形状：`(n_agents, n_agents)`  
   - 含义：所有智能体之间的**两两欧氏距离矩阵**：
     \[
     D_{ij} = \|\text{pos}_i - \text{pos}_j\|_2
     \]
   - 对角元素 `D_{ii}` 为 0。  
   - 用于：
     - 追捕者之间防碰撞（需要与 size 结合）
     - 分析包围圈的几何结构：追捕者之间的相对距离分布等。

7. `is_adversary: bool`  
   - 对当前调用 `_build_global_state` 的 `agent` 而言，表示它是否为追捕者（`True`）或逃跑者（`False`）。  
   - 在奖励计算中通常只对追捕者使用该全局状态，但设计 reward 时可以根据该标志区分角色（如为逃跑者计算不同目标）。

8. `world_size: float`  
   - 数值：`2.5`  
   - 含义：环境世界边长的一半，位置坐标范围为 `(-2.5, 2.5)`。  
   - 可以用于规范化、定义相对距离阈值（比如 capture_threshold 是 `world_size * 0.2`）。

9. `capture_threshold: float`  
   - 数值：`0.5` （已写死为 0.5，在注释中标为 `world_size * 0.2`）。  
   - 含义：围捕判定阈值，通常会用来判断“是否足够靠近/包围目标”，而不是单点碰撞（碰撞靠 size 判定）。  
   - 也可以用于奖励中设置分段奖励，如进入捕获半径等。

总结：  
`global_state` 提供的是**全局绝对几何信息**（位置、速度、距离矩阵），方便在 reward 中设计协作性指标（围捕、均匀分布、防碰撞）。它不是单个 agent 的观测，而是一个全知视角。

---

### 2. 观测空间 (Observation) —— `observation(self, agent, world)`

返回的局部观测向量是一个 `np.concatenate` 结果：

```python
return np.concatenate(
    [norm_self_vel] + [norm_self_pos] + other_pos + other_vel
)
```

**构成步骤：**

1. 自身状态（归一化）：
   - `norm_self_vel = agent.state.p_vel / agent.max_speed`
     - 形状：`(2,)`
     - 含义：自身速度向量，除以自身的 `max_speed` 后得到[-1,1] 左右的无量纲速度。追捕者和逃跑者各用自己的 max_speed 归一化。
   - `norm_self_pos = agent.state.p_pos / self.world_size`
     - 形状：`(2,)`
     - 含义：自身位置，除以 `world_size=2.5`，映射到大致 `[-1,1]` 范围。

2. 其他智能体的**相对位置**：
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
   - 对于每一个 `other != agent`：
     - `rel_pos = (other.pos - self.pos) / world_size`
       - 形状：`(2,)`
       - 含义：**相对位置**（从自身看向其它体的向量），再除以世界尺度。  
       - 对追捕者而言，会观察到：
         - 另外两个追捕者的相对位置
         - 逃跑者的相对位置  
       - 对逃跑者而言，会观察到三个追捕者的相对位置。
   - `other_vel` 只收集**非追捕者**的速度（即逃跑者的速度）：
     - 在当前任务设定下：
       - 追捕者的观测中会包含逃跑者的速度（`prey_vel / prey.max_speed`）
       - 逃跑者的观测中不会看到追捕者的速度（只看到他们的相对位置）

**拼接顺序：**

以一个追捕者为例，场景中共有 3 追捕 + 1 逃跑，则：

- `norm_self_vel`：2 维
- `norm_self_pos`：2 维
- `other_pos`：
  - 2 个其他追捕者 + 1 个逃跑者 ⇒ 3 × 2 = 6 维
- `other_vel`：
  - 仅 1 个逃跑者 ⇒ 2 维（归一化速度）

总维度（追捕者）：  
\[ 2 + 2 + 6 + 2 = 12 \]

总体上，局部观测向量包含：

- 自身的速度、位置（归一化）
- 所有其它智能体的**相对位置**
- 对逃跑者而言：只看到追捕者相对位置，不看到其速度
- 对追捕者而言：看到逃跑者的位置 + 速度，以及其它追捕者的位置（但不含追捕者速度）

这体现了一个典型的局部部分可观测设置：追捕者对目标的动态信息较丰富（含速度），用于预测运动；对队友只知道相对位置，促进基于几何结构的协调。

---

## 三、智能体交互 (Agent Interactions)

### 1. 捕食者 vs 猎物的物理属性差异

在 `CoreEnvLogic.__init__` 中：

```python
self.adversary_params = {'size': 0.075, 'max_speed': 1.0}
self.agent_params = {'size': 0.050, 'max_speed': 1.3}
```

此处命名略有混淆，但结合说明：

- adversary（追捕者）：
  - `size = 0.075`
  - `max_speed = 1.0`
- agent（逃跑者）：
  - `size = 0.050`
  - `max_speed = 1.3`

物理上的差异：

1. **速度上**：
   - 逃跑者的最大速度 `1.3` > 追捕者 `1.0`
   - 含义：逃跑者更灵活、更难被直接追上，要求追捕者通过协作形成包围，而不是单纯逐一追击。

2. **体积上**：
   - 追捕者半径 `0.075` > 逃跑者半径 `0.050`
   - 含义：
     - 追捕者彼此更容易发生碰撞（因体积更大）
     - 对于同等几何配置，追捕者-逃跑者碰撞条件更容易触发（因为两者半径之和较大）

综合来看，任务设计通过：
- 让猎物更快
- 让追捕者更“笨重”（大 size，速度较低）
强化了“多智能体协同围捕”而不是“一对一追逐”。

### 2. 碰撞判定条件（成功捕获的基本几何条件）

碰撞函数：

```python
def is_collision(self, agent1_pos, agent1_size, agent2_pos, agent2_size):
    """碰撞检测：欧氏距离 < 半径之和"""
    delta_pos = agent1_pos - agent2_pos
    dist = np.sqrt(np.sum(np.square(delta_pos)))
    dist_min = agent1_size + agent2_size
    return dist < dist_min
```

数学条件：

\[
\text{collision}(1,2) = \left( \|p_1 - p_2\|_2 < r_1 + r_2 \right)
\]

- `p_1, p_2`：两个智能体的二维位置向量
- `r_1, r_2`：各自的 `size`（半径）
- `dist_min = r_1 + r_2` 是两个圆刚好接触时的中心距离
- 当实际距离 `dist` 小于这个阈值时，判定为碰撞（即两个圆盘重叠）

在任务语义层面：

- 对“追捕者-逃跑者”组合，若满足上述条件，则可被解释为“成功接触”或“抓到”; 具体是否等同于完整围捕成功，要看上层任务定义（可能还需要多个追捕者协同或在 capture_threshold 内）。
- 对“追捕者-追捕者”组合，碰撞是要避免的（通常会在奖励中给予惩罚或在物理引擎中产生反弹）。

注意：  
`capture_threshold` (0.5) 是一个**额外的半径**概念，和碰撞判定（size-based）是两层概念：

- `is_collision`：几何碰撞（物理接触）
- `capture_threshold`：离目标一定范围内，表示进入“围捕区域”（即便尚未几何接触）

---

## 四、任务相关信息 (Task-relevant Information)

### 1. 围捕任务核心目标与变量的关系

根据任务要求：

1. 追捕者需要接近并包围目标
2. 追捕者之间避免碰撞
3. 形成均匀的包围圈
4. 在尽可能短的时间内完成围捕

尽管此处只做分析不写 reward，但可以指出哪些变量与目标**直接相关**（正/负相关）：

#### 与“接近并包围目标”相关

- `distances_to_prey`（全局状态）  
  - 是每个追捕者到逃跑者的欧氏距离  
  - 越小 ⇒ 追捕者越接近目标（正相关：距离越小越好）
- `capture_threshold`  
  - 提供一个期望的距离尺度  
  - 例如，期望所有追捕者距离目标小于该阈值时可认为完成“包围/捕获”状态（状态判定、奖励构造的关键常数）
- `prey_position`, `agent_positions`  
  - 通过位置可以计算：
    - 追捕者质心相对于逃跑者的位置
    - 追捕者相对逃跑者的角度分布（极坐标表示）

#### 与“避免追捕者之间碰撞”相关

- `inter_agent_distances`  
  - 包含所有 agent 间的距离
  - 与 `adversary size` 结合，可判断追捕者-追捕者的潜在碰撞：
    \[
    D_{ij} < r_i + r_j \Rightarrow \text{相互碰撞}
    \]
  - 因此追捕者之间不应该让 `inter_agent_distances` 过小（负相关：距离过小不好）。

#### 与“均匀包围圈”相关

需要从 `agent_positions` 与 `prey_position` 推出若干几何特征：

- 各追捕者到逃跑者的距离：
  - 如果都在一个包围圈上，**这些距离应该接近**（小方差）
- 各追捕者相对于逃跑者的极角：
  - 理想均匀分布：3 个追捕者在 360° 上间隔 ≈ 120°；
  - 即相邻追捕者在相对于逃跑者角度上应形成近似等分。
- 追捕者的重心应接近逃跑者位置：
  - 若三者围成的多边形包含逃跑者且重心与逃跑者接近，可以作为“包围中心一致”的指标。

这些都依赖于：

- `agent_positions`（特别是 adversaries 的位置）
- `prey_position`

#### 与“短时间内完成围捕”相关

- 时间步（由外部环境控制，不在 `CoreEnvLogic` 里，但 reward 设计时可以使用 step count）
- 状态中的距离收敛速度：
  - `distances_to_prey` 随时间递减的快慢
- 一旦满足某种“围捕完成条件”，后续时间步内可能给零/负奖励，鼓励早完成。

总的来说：

- 正相关（越大越好）：
  - 追捕者与目标的适当接近程度（在合理半径内，而不是无限逼近导致碰撞）——通常体现为“进入捕获区域”的次数或保持时间。
  - 追捕者间的距离适度大于安全距离，且在一定范围内均匀分布。
- 负相关（越小越好）：
  - `distances_to_prey`（在没捕获前）
  - 捕获前的时间步数
  - `inter_agent_distances` 接近或低于碰撞阈值的情况

### 2. 高质量“包围圈”的空间拓扑结构

理论上，一个高质量/高对称性的包围圈（3 追捕者，一逃跑者）通常满足以下几何性质：

1. **逃跑者大致在追捕者多边形的内部**  
   - 3 个追捕者的三角形包含逃跑者的位置；
   - 从逃跑者视角，三个追捕者分布在不同方位角上，覆盖全向。

2. **追捕者与逃跑者距离相近，形成环状**  
   - 设追捕者位置为 \(A_1,A_2,A_3\)，逃跑者为 \(P\)，则：
     \[
     d_i = \|A_i - P\|
     \]
   - 在高质量包围圈中：
     - 所有 \(d_i\) 在某个理想半径 \(r^*\) 附近，方差较小；
     - 且 \(r^*\) 不大于 `capture_threshold`，代表已靠得足够近。

3. **追捕者在角度上基本等分围绕目标**  
   - 考虑向量 \(\overrightarrow{PA_i} = A_i - P\) 的极角 \(\theta_i\)（在 [-π, π)）；
   - 将角度排序，得到 \(\theta_{(1)}, \theta_{(2)}, \theta_{(3)}\)；
   - 理想状态下相邻角度差应 ~ \(120^\circ = 2\pi/3\)：
     \[
     |\theta_{(2)} - \theta_{(1)}| \approx 2\pi/3, \quad
     |\theta_{(3)} - \theta_{(2)}| \approx 2\pi/3, \quad
     \left|(\theta_{(1)} + 2\pi) - \theta_{(3)}\right| \approx 2\pi/3
     \]
   - 角度差方差越小，代表围绕目标的均匀性越好。

4. **追捕者之间保持安全距离，避免碰撞**  
   - 三个追捕者间距离：
     \[
     D_{ij} = \|A_i - A_j\|_2
     \]
   - 至少要满足：
     \[
     D_{ij} > r_i + r_j = 0.075 + 0.075 = 0.15
     \]
   - 在理想三角形包围形态中，这些边长应：
     - 不太小（避免碰撞、保持可机动空间）
     - 不太大（防止目标在空隙中逃出）
     - 且三条边比较接近（接近等边三角形）

5. **整体重心与目标对齐**（可选的质量指标）  
   - 追捕者重心：
     \[
     C = \frac{A_1 + A_2 + A_3}{3}
     \]
   - 高质量圈中 \(C\) 应接近 \(P\)，即 \(\|C-P\|\) 较小。  
   - 表示包围圈对目标居中，而不是偏向一侧。

这些拓扑特征都可以从 `agent_positions` 和 `prey_position` 计算得到，`global_state` 提供了充分的信息来构造上述几何指标。

---

## 五、API 边界隔离 (API Boundaries)

关于 `CoreEnvLogic` 与真正环境/奖励函数实现之间的边界，需要严格注意以下几点：

1. `CoreEnvLogic` 是**文档伪代码**：
   - 它只是为说明物理常量、观测结构和全局状态结构而存在；
   - 在实际运行的环境中，它不会被实例化，也不应该被引用。
   - 即在真实代码中不能写出：
     ```python
     logic = CoreEnvLogic()
     ```
     或在 `world` 上访问 `world.logic`。

2. 在真正的 `compute_reward` 中：

   - **禁止**调用 `CoreEnvLogic()`。
   - **禁止**访问诸如：
     - `world.logic`
     - `world.adversary_params`
     - `world.agent_params`
     - 以及任何未在实际 World/Agent 类中定义的“虚构属性”。

3. 所有需要的物理常量必须在 `compute_reward` 内部显式声明为**局部变量**（硬编码），例如：

   ```python
   world_size = 2.5
   capture_threshold = 0.5
   adv_size = 0.075         # 追捕者半径
   prey_size = 0.050        # 逃跑者半径
   adv_max_speed = 1.0
   prey_max_speed = 1.3
   ```

   这样做的好处是：

   - 保证 reward 逻辑与环境实现之间是**弱耦合**的：reward 不依赖任何“隐藏的” World 属性；
   - 便于后续审计和复现：所有决定任务约束的数值在 reward 文件内部一目了然。

4. `compute_reward` 仅能依赖它显式接收的参数：
   - 比如如果调用形式是 `compute_reward(agent, world, global_state)`，则：
     - `global_state` 应类似 `_build_global_state` 产生的结构；
     - 任何超出此接口的数据都不应被隐式访问。
   - 若 `global_state` 不包含某些信息（如 step 数），则要么：
     - 在外部以额外参数传入；
     - 要么在 reward 内部放弃使用该信息。

5. 未来在真正设计 reward 函数时，需严格遵守：
   - 用 `global_state` 中已有的几何/拓扑信息（位置、距离矩阵等）构造目标；
   - 用函数内部硬编码的物理常数（size, max_speed, world_size, capture_threshold）设定阈值与尺度；
   - 不在 reward 中做环境结构修改（不调用物理更新、不改 agent state）。

---

以上分析确保：

- 我们对全局状态与局部观测的维度和物理含义有清晰认知；
- 理解了追捕者/逃跑者在动力学上的差异与碰撞条件；
- 明确了哪些变量可被用来刻画“靠近、包围、均匀、防碰撞、快速完成”等任务目标；
- 严格区分了说明性 `CoreEnvLogic` 与实际可用 API 的边界，为下一步设计奖励函数打下干净的接口前提。

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
    # 非追捕者不奖励
    if not global_state["is_adversary"]:
        return 0.0, {}

    components = {}

    # -----------------------
    # 物理与任务超参数（硬编码）
    # -----------------------
    world_size = 2.5
    capture_threshold = 0.5
    adv_size = 0.075
    prey_size = 0.050

    # 距离相关
    w_approach = 1.0
    w_capture_bonus = 3.0

    # 队形相关
    w_radial_uniform = 0.8
    w_angular_uniform = 1.2
    w_centering = 0.5

    # 防碰撞相关
    w_collision_penalty = 2.0
    w_proximity_penalty = 0.5
    safety_margin = 0.02

    # 时间惩罚（鼓励尽快完成）
    w_time_step = 0.01

    # -----------------------
    # 状态解析
    # -----------------------
    agent_positions = global_state["agent_positions"]
    agent_velocities = global_state["agent_velocities"]
    prey_position = global_state["prey_position"]
    inter_agent_distances = global_state["inter_agent_distances"]
    distances_to_prey = global_state["distances_to_prey"]

    n_agents = agent_positions.shape[0]

    # 识别追捕者索引（依赖 size，环境中 3 个追捕者 + 1 个逃跑者）
    # 由于 compute_reward 是从某个 agent 调用，需从 agent_name 推出自身索引
    # 假定 world.agents 按与 agent_positions 一致的顺序排列
    agent_index = None
    for idx, a in enumerate(world.agents):
        if a.name == agent_name:
            agent_index = idx
            break
    if agent_index is None:
        return 0.0, {}

    # 根据 size 将追捕者与逃跑者区分开（与环境设定一致）
    sizes = []
    for a in world.agents:
        # 根据设定：adversary size = 0.075, prey size = 0.050
        # 这里不能访问 a.size（接口未保证），改用位置索引：
        # 默认前三个是 adversary，最后一个是 prey 的情况在 MPE 中常见，
        # 但为更加稳健，还是用 global_state['distances_to_prey'] 的长度判断：
        sizes.append(adv_size)

    # 假设: 3 个追捕者 + 1 个猎物，猎物是唯一非 adversary
    # 使用 global_state['distances_to_prey'] 长度推断 adversary 数量
    n_adversaries = len(distances_to_prey)
    adversary_indices = list(range(n_adversaries))
    prey_indices = [i for i in range(n_agents) if i not in adversary_indices]
    if len(prey_indices) > 0:
        prey_index = prey_indices[0]
    else:
        prey_index = None

    # 当前追捕者在 adversary_indices 中的本地索引
    if agent_index not in adversary_indices:
        return 0.0, {}
    local_adv_index = adversary_indices.index(agent_index)

    agent_pos = agent_positions[agent_index]
    agent_vel = agent_velocities[agent_index]

    # -----------------------
    # 1. 接近目标 + 捕获激励
    # -----------------------
    dist_to_prey = np.linalg.norm(agent_pos - prey_position)
    # 归一化距离（相对于 world_size），越小越好
    norm_dist = dist_to_prey / world_size
    distance_reward = -norm_dist * w_approach

    # 捕获完成 bonus：一旦所有追捕者进入 capture_threshold 内给额外奖励
    capture_bonus = 0.0
    if n_adversaries > 0:
        if np.all(distances_to_prey <= capture_threshold):
            capture_bonus = w_capture_bonus

    components["distance_reward"] = distance_reward
    components["capture_bonus"] = capture_bonus

    # -----------------------
    # 2. 防碰撞与安全间距
    # -----------------------
    collision_penalty = 0.0
    proximity_penalty = 0.0
    min_safe_dist = 2.0 * adv_size + safety_margin

    for other_idx in adversary_indices:
        if other_idx == agent_index:
            continue
        d = inter_agent_distances[agent_index, other_idx]
        # 碰撞（严格小于半径之和）
        if d < 2.0 * adv_size:
            # 越深度碰撞惩罚越大
            overlap = max(0.0, 2.0 * adv_size - d)
            collision_penalty -= w_collision_penalty * (1.0 + overlap / (2.0 * adv_size))
        # 未碰撞但过近（潜在危险）
        elif d < min_safe_dist:
            proximity_penalty -= w_proximity_penalty * (min_safe_dist - d) / min_safe_dist

    components["collision_penalty"] = collision_penalty
    components["proximity_penalty"] = proximity_penalty

    # -----------------------
    # 3. 包围队形（半径均匀 + 角度均匀 + 中心对齐）
    # -----------------------
    radial_uniform_reward = 0.0
    angular_uniform_reward = 0.0
    centering_reward = 0.0

    if prey_index is not None and n_adversaries >= 2:
        adv_positions = agent_positions[adversary_indices]

        # 3.1 半径均匀：所有追捕者到猎物的距离方差越小越好
        all_dists = np.linalg.norm(adv_positions - prey_position, axis=1)
        mean_dist = np.mean(all_dists)
        if mean_dist > 1e-6:
            radial_var = np.var(all_dists / (mean_dist + 1e-6))
            radial_uniform_reward = -radial_var * w_radial_uniform
        else:
            radial_uniform_reward = 0.0

        # 3.2 角度均匀：追捕者相对于猎物的极角应近似等间距
        rel_vecs = adv_positions - prey_position
        angles = np.arctan2(rel_vecs[:, 1], rel_vecs[:, 0])
        # 排序角度
        sort_idx = np.argsort(angles)
        angles_sorted = angles[sort_idx]

        # 角度差（考虑环绕）
        diffs = np.diff(angles_sorted)
        # 闭合环的最后一个差值
        last_diff = (angles_sorted[0] + 2 * np.pi) - angles_sorted[-1]
        angle_diffs = np.concatenate([diffs, np.array([last_diff])])

        # 理想差值（等分圆）
        ideal_diff = 2 * np.pi / n_adversaries
        angle_diff_var = np.var(angle_diffs / ideal_diff)
        angular_uniform_reward = -angle_diff_var * w_angular_uniform

        # 3.3 重心对齐：追捕者质心应接近猎物
        center = np.mean(adv_positions, axis=0)
        center_dist = np.linalg.norm(center - prey_position)
        # 相对于 capture_threshold 归一化
        centering_reward = -w_centering * (center_dist / (capture_threshold + 1e-6))

    components["radial_uniform_reward"] = radial_uniform_reward
    components["angular_uniform_reward"] = angular_uniform_reward
    components["centering_reward"] = centering_reward

    # -----------------------
    # 4. 时间惩罚
    # -----------------------
    # 注意：此处无法直接获取 step 数，采用常数小惩罚鼓励早结束
    time_penalty = -w_time_step
    components["time_penalty"] = time_penalty

    total_reward = float(sum(components.values()))
    return total_reward, components
```
```
