# LLM Interaction Log

- **Generation**: 0
- **Phase**: Phase2_Initial_Generation
- **Candidate Info**: 1
- **Timestamp**: 20260411_133541

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

### 1. 使用到的依赖包

从代码顶部的 import 可以看出，这个环境依赖：

1. **数值与矩阵运算**
   - `numpy as np`  
     用于向量运算（位置、速度、距离、拼接观测等）。

2. **强化学习 / 环境接口**
   - `gymnasium`  
     与 Gym 兼容的环境接口（空间定义、step/reset 协议等）。

3. **PettingZoo 多智能体框架**
   - `pettingzoo.mpe._mpe_utils.core`  
     - `Agent`：MPE 中的主体类，包含 `adversary` 标记、`max_speed`、`state.p_pos/p_vel` 等属性。
     - `Landmark`：标记物（此任务里虽未直接使用，仍被引入）。
     - `World`：包含所有 agents、landmarks、全局物理属性的世界对象。
   - `pettingzoo.mpe._mpe_utils.scenario.BaseScenario`  
     用于定义 scenario（make_world、reset_world、reward、observation 等）。
   - `pettingzoo.mpe._mpe_utils.simple_env`  
     - `SimpleEnv`, `make_env`：用于从 scenario 创建 PettingZoo 环境。
   - `pettingzoo.utils.conversions.parallel_wrapper_fn`  
     用于把环境封装成并行 API 的 wrapper。

4. **本项目内部模块**
   - `from .custom_agents_dynamics import CustomWorld`  
     自定义的 World 类（可能重写或扩展了物理/动力学），这里仅被 import 未在片段中使用。
   - `from . import reward_function`  
     可插拔的奖励函数模块（实际 compute_reward 实现应在这里；目前只知道“仅包含追捕者奖励”）。

5. **工具**
   - `from gymnasium.utils import EzPickle`  
     一般用于环境的序列化 / 反序列化（便于并行化、复现），片段中暂未使用。

### 2. 是否引入了外部未定义变量

在 `CoreEnvLogic` 内部：

- 使用的所有符号都来自：
  - `numpy` (`np.sqrt`, `np.sum`, `np.square`, `np.linalg.norm`, `np.array`, `np.zeros`)
  - 方法入参 `agent`, `world`, `agent1_pos`, `agent1_size`, `agent2_pos`, `agent2_size`
  - 类属性 `self.world_size`, `self.capture_threshold`
- 这一段中 **不存在** 未定义的局部变量或引用错误：  
  - `world.agents` 来自 PettingZoo MPE 的 `World` 类标准属性。  
  - `agent.adversary`、`agent.state.p_pos`、`agent.state.p_vel`、`agent.max_speed` 都是 MPE Agent 标准字段。
  - 过滤 `preys = [a for a in all_agents if not a.adversary]` 是典型的 MPE 区分 adversary / good agent 方式。

唯一需要注意的是你在说明中特别强调：

> `CoreEnvLogic` 是“文档伪代码”，运行时不应被实例化，也不能假设 `world.logic` 等属性存在。

这一点是 API 设计约束，而非语法/变量未定义问题。因此从“代码内部”的角度看，`CoreEnvLogic` 自身没有未定义变量，但**不能**在真实环境或 `compute_reward` 中被直接调用或依赖。


---

## 二、环境结构 (Environment Structure)

### 1. 状态空间 (Global State) 的构成与物理意义

`_build_global_state` 返回一个字典：

```python
{
    'agent_positions': agent_positions,        # shape: (N, 2)
    'agent_velocities': agent_velocities,      # shape: (N, 2)
    'prey_position': prey_pos,                 # shape: (2,)
    'prey_velocity': prey_vel,                 # shape: (2,)
    'distances_to_prey': distances_to_prey,    # shape: (N_adv,)
    'inter_agent_distances': inter_agent_distances, # shape: (N, N)
    'is_adversary': agent.adversary,           # bool (当前被计算奖励的 agent 是否为追捕者)
    'world_size': self.world_size,             # float, 2.5
    'capture_threshold': self.capture_threshold # float, 0.5
}
```

具体解释：

1. **`agent_positions` (N × 2)**  
   - 每个智能体在 2D 世界中的 **绝对位置** (x, y)。  
   - N = 所有智能体数量 = 追捕者 + 逃跑者。  
   - 用于计算：
     - 追捕者与猎物的距离
     - 智能体之间距离、几何结构（围捕形状）
     - 是否出界等（若在其它地方使用）

2. **`agent_velocities` (N × 2)**  
   - 每个智能体在 2D 世界中的 **绝对速度向量** (vx, vy)。  
   - 用于分析：
     - 当前运动方向、是否朝向/远离猎物
     - 追捕者是否在尝试收缩包围圈或分散

3. **`prey_position` (2,)**  
   - 猎物（逃跑者）的绝对位置 (x, y)。  
   - 是构建与任务相关的所有“相对位置”与“包围形状”的核心参考点。

4. **`prey_velocity` (2,)**  
   - 猎物的速度向量 (vx, vy)。  
   - 反映猎物运动趋势（逃跑方向），对于设计预测性/前瞻性的捕获策略很关键（奖励中可以奖励拦截点）。

5. **`distances_to_prey` (N_adv,)**  
   - 每个追捕者到猎物的欧氏距离。  
   - N_adv = 追捕者数量（这里是 3）。  
   - 物理意义：
     - 捕获难易程度（越小代表越接近，越可能形成包围）
     - 可做为“接近猎物”的直接奖励项或 shaping 信号。

6. **`inter_agent_distances` (N × N)**  
   - 所有智能体间的 **两两欧氏距离矩阵**。  
   - 对角线元素理论上为 0（自身到自身的距离）。  
   - 对追捕者之间的关系尤为重要：
     - 避免碰撞（距离 < 碰撞阈值）
     - 包围圈的均匀性：3 个追捕者之间距离是否相近  
     - 是否在猎物附近形成某种“拓扑结构”（等边三角形等）

7. **`is_adversary` (bool)**  
   - 当前正在计算奖励的这个智能体是否为追捕者。  
   - 在奖励函数中可以用来区分：
     - 追捕者奖励 vs 逃跑者奖励（虽然当前奖励函数模块只给追捕者）  
     - 或直接忽略非追捕者。

8. **`world_size` (float = 2.5)**  
   - 世界边长的一半：地图为 [-2.5, 2.5] × [-2.5, 2.5]。  
   - 物理意义：
     - 标准化位置/距离
     - 相对围捕尺度（capture_threshold 相对于地图大小）

9. **`capture_threshold` (float = 0.5)**  
   - 表示围捕判定的一个空间尺度（world_size 的 0.2）。  
   - 更准确的物理含义需要结合你实际判定逻辑：  
     - 可能是“所有追捕者到猎物的距离都小于该值”  
     - 或“包围圈半径 ≈ 该值”，等等。  
   - 这里仅提供了一个 **标称尺度**，具体抓捕判定尚未在该片段中定义（`is_collision` 是 pairwise 碰撞判断，与“围捕成功”是两个不同层级的概念）。


### 2. 观测空间 (Observation) 的构成与物理意义

`observation(self, agent, world)` 返回的观测向量结构：

```python
# 自身状态（归一化）
norm_self_vel = agent.state.p_vel / agent.max_speed      # shape: (2,)
norm_self_pos = agent.state.p_pos / self.world_size      # shape: (2,)

# 其他智能体相对位置
other_pos = []
other_vel = []
for other in world.agents:
    if other is agent: continue
    rel_pos = (other.state.p_pos - agent.state.p_pos) / self.world_size
    other_pos.append(rel_pos)
    if not other.adversary:
        other_vel.append(other.state.p_vel / other.max_speed)

obs = np.concatenate([norm_self_vel] + [norm_self_pos] + other_pos + other_vel)
```

对 3 追捕者 + 1 逃跑者情形：

- 对一个给定智能体（无论是追捕还是逃跑），`world.agents` 中共有 4 个，去掉自己后有 3 个“其他智能体”。

1. **自身信息**
   - `norm_self_vel`：归一化后的自身速度向量，维度 2  
     - $ \frac{p\_vel}{\text{max\_speed}} $，范围大致在 [-1, 1]。  
     - 表示当前运动方向和相对速度大小。
   - `norm_self_pos`：归一化后的自身位置向量，维度 2  
     - $ \frac{p\_pos}{\text{world\_size}} $，范围大致在 [-1, 1]。  
     - 表示自身在整个地图中的相对位置。

2. **其他智能体的相对位置 (other_pos)**
   - 对每一个 `other`（除了自己），计算：
     - `rel_pos = (other.p_pos - self.p_pos) / world_size`，维度 2  
   - 对于 4-agent world，每个智能体看到 3 个其他体，因此：
     - `len(other_pos) = 3`, 拼成 6 维向量。
   - 物理意义：
     - 表示其他智能体相对自己的空间关系，无需关注绝对坐标。  
     - 用于：
       - 避碰：保持与队友一定距离
       - 围捕形成某种相对几何形状
       - 对猎物，则是追捕者相对猎物的方位信息（反之亦然）

3. **其他智能体的速度 (other_vel)**  
   - 只有当 `other` 是 **非 adversary** 才会添加其速度；在本任务设定中：
     - adversary = 追捕者  
     - not adversary = 猎物（逃跑者）
   - 因此，对于一个**追捕者**的观测：
     - 它会看到所有其他智能体的相对位置（3 个），
     - 但在 `other_vel` 里只会包含 **1 个猎物** 的归一化速度向量 (2 维)。  
   - 对于猎物的观测（如果也用同一函数）：
     - 它会看到 3 个追捕者的相对位置，但 `other_vel` 将为空（因为其它都是 adversary）。

4. **整体维度（对追捕者）**

- 自身速度：2  
- 自身位置：2  
- 其他 3 个体的相对位置：3 × 2 = 6  
- 猎物速度：2  
- 总维度：2 + 2 + 6 + 2 = **12 维**

物理含义总结：  
每个追捕者的局部观测是一个 **自中心（egocentric）** 表征：  
- 自身状态（速度/位置），  
- 所有其他体的相对方位，  
- 逃跑目标的速度。  

注意：**没有直接包含** 全局拓扑信息，如所有追捕者之间的绝对角度分布，需要 agent 通过相对位置推断。


---

## 三、智能体交互 (Agent Interactions)

### 1. 捕食者与猎物在物理属性上的差异

根据 `__init__` 中的参数设置：

```python
self.adversary_params = {'size': 0.075, 'max_speed': 1.0}
self.agent_params     = {'size': 0.050, 'max_speed': 1.3}
```

约定：

- adversary = 追捕者（捕食者）
- agent (在 MPE 里往往叫 good agent) = 猎物（逃跑者）

对比：

1. **size（碰撞半径）**
   - 追捕者：`size = 0.075`
   - 逃跑者：`size = 0.050`
   - 含义：
     - 追捕者的碰撞体积更大，更容易在与猎物或队友接触中触发碰撞。
     - 对避碰任务而言，追捕者之间需要保持更大距离。  

2. **max_speed（最大速度）**
   - 追捕者：`max_speed = 1.0`
   - 逃跑者：`max_speed = 1.3`
   - 含义：
     - 逃跑者速度更快，理论上可轻易甩开个体追捕者。
     - 围捕必须依赖 **多智能体协同**：三角包围、切断逃跑路线等，而不是单个追捕者追逐。

这种设计明确强化了“协作围捕”的必要性：  
- 单个追捕者无法持续跟上猎物。  
- 三个追捕者必须在空间上合作形成包围网。

### 2. 环境中“成功捕获（Collision）”的数学和物理条件

片段中定义的碰撞检测函数：

```python
def is_collision(self, agent1_pos, agent1_size, agent2_pos, agent2_size):
    """碰撞检测：欧氏距离 < 半径之和"""
    delta_pos = agent1_pos - agent2_pos
    dist = np.sqrt(np.sum(np.square(delta_pos)))
    dist_min = agent1_size + agent2_size
    return dist < dist_min
```

数学条件：

- 设两个智能体的位置分别为 $p\_1, p\_2 \in \mathbb{R}^2$，半径为 $r\_1, r\_2$。
- 碰撞当且仅当：
  \[
  \|p\_1 - p\_2\|_2 < r\_1 + r\_2
  \]
- 即两圆的圆心距离小于半径之和。

物理意义：

- 智能体被视为二维平面上的硬圆盘。
- 当两个圆盘重叠（有交集）即视为发生碰撞。  
- 对不同组合：
  - 追捕者 vs 追捕者：队友间不应发生碰撞（应严惩）。  
  - 追捕者 vs 猎物：可以视为“抓到”猎物，是否作为“任务完成条件”要看上层逻辑（此片段未给出完整终止条件）。
- 注意：`capture_threshold` 与碰撞阈值不同：
  - 碰撞阈值 = `r_1 + r_2`，只考虑两体接触。  
  - 围捕判定阈值通常是“多个追捕者围成的圈”与猎物距离关系，不是单对单碰撞。


---

## 四、任务相关信息 (Task-relevant Information)

> 注意：你在问题里重复写了两次“任务相关信息”小节，我这里合并成一个综合回答。

围捕任务的目标包括：  
1. 接近并包围目标  
2. 避免追捕者之间碰撞  
3. 形成均匀的包围圈  
4. 尽可能短时间内完成围捕  

### 1. 围捕核心目标与变量的正 / 负相关关系

1. **接近并包围目标**

   - 关键变量：
     - `distances_to_prey`（每个追捕者到猎物的距离）
     - `agent_positions` / `prey_position` (用于复现/推导更多几何量)

   - 关系：
     - 任务目标：**追捕者尽可能接近猎物**  
       ⇒ `distances_to_prey` 越小越好（负相关）。  
     - 多个追捕者中最远者到猎物距离：`max(distances_to_prey)` 也需要小，以形成有效包围，而不是只靠单个追捕者靠近。

2. **避免追捕者之间碰撞**

   - 关键变量：
     - `inter_agent_distances` 中 adversary-adversary 部分
     - 各追捕者的 `size`（硬编码常量：0.075）

   - 关系：
     - 任务目标：追捕者之间距离 **不能低于** 碰撞阈值  
       ⇒ 对任意两追捕者，$d_{ij} \ge r_i + r_j = 0.075 + 0.075$。  
     - 距离越接近该阈值，越危险；如果距离小于阈值，则发生碰撞，应强烈惩罚。  
     - 因此：
       - 与 **碰撞概率或碰撞次数** 负相关；
       - 与 **最小队友间距离** 正相关（但过大也不利于围捕，需要在“安全但紧密”之间权衡）。

3. **形成均匀的包围圈**

   - 关键变量：
     - `agent_positions` 与 `prey_position`  
     - 可以派生出：
       - 每个追捕者与猎物的半径：`dist_i = ||p_adv_i - p_prey||`
       - 每个追捕者相对于猎物的极角：`theta_i = atan2(y_i - y_prey, x_i - x_prey)`
       - 追捕者之间的 pairwise distance：`inter_agent_distances[i,j]`

   - 理想拓扑（后面会详细展开）：
     - 所有追捕者到猎物的距离相近（半径近似相等）。  
     - 三个追捕者在猎物周围的角度差约为 120°，形成近似等边三角形环绕猎物。  
   - 关系：
     - 半径方差 (variance of `distances_to_prey`) 越小越好（负相关）。  
     - 角度分布偏离均匀（例如与 {0°,120°,240°} 的偏差）越小越好（负相关）。  
     - 追捕者之间 pairwise distance 的方差越小越好（负相关），同时总体尺度不能太大（否则圈太松）。

4. **在尽可能短的时间内完成围捕**

   - 关键变量：
     - 时间步数 / 累积 step 数（此处未在 `CoreEnvLogic` 中直接给出，但在环境主循环中必然存在）  
     - 辅助变量：
       - `capture_threshold`
       - 是否满足围捕几何条件（需要由 `agent_positions` / `distances_to_prey` 衍生）

   - 关系：
     - 每一步没有完成围捕就是时间消耗，可以视为一个时间惩罚：step_count ↑ → 回报 ↓。  
     - 早完成围捕 ⇒ 较少时间惩罚。  
     - 可能与“达到围捕条件的时间步 t_capture”负相关。

综合来看：

- **正相关**（值大更好）的方向：
  - 队友间最小距离（但不过大，超过一定值会影响包围质量）
  - 包围稳定度（围捕成功持续时间）
- **负相关**（值小更好）的方向：
  - `distances_to_prey`（尤其是最大值）
  - 队友间碰撞发生次数或碰撞距离不足
  - 半径差（追捕者到猎物距离的方差）
  - 角度分布不均匀性
  - 完成围捕所需的时间步数

### 2. 代表高质量“包围圈”的空间拓扑

为了描述“高质量包围圈”，需要从几何上定义 **位置分布模式**。以下是在 3 追捕者围 1 猎物情境下的一种合理表征方式。

设：

- 猎物位置：$p\_0 \in \mathbb{R}^2$
- 三个追捕者位置：$p\_1, p\_2, p\_3 \in \mathbb{R}^2$

定义：

1. **半径与半径均匀性**

   - 各追捕者到猎物的距离：
     \[
     r_i = \|p_i - p_0\|, \quad i = 1,2,3
     \]
   - 期望：
     - 所有 $r_i$ 接近某个目标半径 $R$（例如接近 `capture_threshold`，既不太远也不至于立即碰撞）。
     - 半径方差：
       \[
       \mathrm{Var}(r_1, r_2, r_3) \approx 0
       \]
     - 这代表三者在猎物周围距离类似，形成一个“大致圆环”。

2. **角度与角度均匀性（扇形间隔均匀）**

   - 以猎物为原点，计算每个追捕者的极角：
     \[
     \theta_i = \mathrm{atan2}(y_i - y_0,\; x_i - x_0), \quad i=1,2,3
     \]
   - 对角度排序后，邻接角差为：
     \[
     \Delta\theta_i = \theta_{(i+1)} - \theta_{(i)}, \quad \text{并考虑环绕 } 2\pi
     \]
   - 理想情况：
     - 三个角度差都接近 $2\pi/3 = 120^\circ$。  
     - 角度差的方差越小，说明追捕者在猎物周围分布越均匀。

3. **追捕者之间的 pairwise distance**

   - 追捕者–追捕者之间距离：
     \[
     d_{ij} = \|p_i - p_j\|, \; 1 \le i < j \le 3
     \]

   对于理想的等边三角形包围：

   - 三个边长 $d_{12}, d_{23}, d_{31}$ 应接近：
     \[
     d \approx 2 R \sin(60^\circ) = \sqrt{3} R
     \]
   - 且三者之间的方差要小。

4. **猎物是否“在圈内”**

   - 对 3 点构成的三角形，判断猎物点是否在三角形内部或附近：  
     - 如果猎物在三角形内部（或非常接近），说明形成了真正的包围，而不是三者站在猎物同一侧的排队形。  
   - 从拓扑角度说：
     - 猎物应位于追捕者凸包的内部或至少接近其中心区域。

综合几何条件：  

一个高质量的 3-对-1 包围圈通常满足：

1. 所有追捕者到猎物的距离接近某个合适半径 $R$，且 $R$ 不太大（能控制猎物）也不太小（避免碰撞）。
2. 三个追捕者在猎物周围的角度大致相隔 120°，在极坐标上均匀分布。
3. 三个追捕者间的距离接近等边三角形边长，方差较小。
4. 猎物位置位于三个追捕者形成的三角形内部（或非常靠近其几何中心）。
5. 整体结构在一段时间内保持稳定（不是瞬时经过）。

这些条件都可以由 `global_state` 中的 `agent_positions`, `prey_position`, `distances_to_prey`, `inter_agent_distances` 派生得到，但需要在奖励函数里显式计算（注意：这些计算要在 `compute_reward` 内通过局部硬编码的常量完成，不能用 `CoreEnvLogic` 实例）。


---

## 五、API 边界隔离 (API Boundaries)

基于你的说明，必须严格区分“文档伪代码”和“实际可用 API”。

1. `CoreEnvLogic` 的定位

   - 这是一个 **说明性/文档用** 类，目的是帮助理解：
     - 世界空间尺度 (`world_size`)
     - agent 参数 (`size`, `max_speed`)
     - 观测构建方式 (`observation`)
     - 可用于奖励函数的全局状态构造 `_build_global_state`
   - 在真正的环境实现中：
     - 该类 **不保证存在**，也不可在运行时 `CoreEnvLogic()`。  
     - `world` 对象中也不会有 `world.logic` 这种属性。

2. 在真实的 `compute_reward` 内的限制

   - 禁止：
     - `logic = CoreEnvLogic()`
     - `world.logic.world_size`
     - `world.adversary_params`、`world.agent_params` 等假想属性。
   - 允许 / 必须：
     - 直接从 `world` 和 `agent` 的 MPE 标准属性中取得 **动态状态**（如 `agent.state.p_pos`, `agent.state.p_vel` 等）。
     - 所有 **物理常量**（`size`, `max_speed`, `world_size`, `capture_threshold`, `max_force` 等）必须在 `compute_reward` 函数内部以 **局部硬编码变量** 出现，例如：

       ```python
       world_size = 2.5
       capture_threshold = 0.5
       adv_size = 0.075
       prey_size = 0.050
       adv_max_speed = 1.0
       prey_max_speed = 1.3
       ```

3. 对 `_build_global_state` 的使用方式

   - `_build_global_state` 的实现是合理的，但仅可作“结构参考”。  
   - 在真实 `compute_reward` 中，如果想用同样的信息：
     - 需要手动从 `world.agents` 重新构造部分或全部字段。
     - 不能写成 `global_state = world.logic._build_global_state(agent, world)`，因为 `world.logic` 不存在。
   - 可以遵循同样的逻辑：  
     - 构建 `agent_positions`, `agent_velocities`, `distances_to_prey`, `inter_agent_distances` 等，但完全在 `compute_reward` 内，用局部变量。

4. 对 `observation` 的使用

   - `observation(self, agent, world)` 展示了**观测结构**，在环境真正的 `Scenario.observation` 中会有类似实现。  
   - 奖励函数 **不应该反向调用** 这个 `CoreEnvLogic.observation`。  
   - 如果奖励设计需要用到与观测类似的结构，也应直接从 `world` 和 `agent` 的低层状态构造。

总结这部分：

- `CoreEnvLogic` = 只用于“理解与推理的接口说明书”，不是运行时代码入口。
- `compute_reward` 内所有对物理常量的引用必须是**硬编码局部变量**。  
- 对全局状态的构造和拓扑计算必须直接基于 `world.agents`、`agent.state` 等底层对象来重建，**不可跨 API 边界**依赖这些说明性类或其属性。  

以上分析仅限于环境建模与信息结构，不包含任何奖励函数代码实现。

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

    # ------------------------
    # 物理与任务常量（硬编码）
    # ------------------------
    world_size = 2.5
    capture_threshold = 0.5
    adv_size = 0.075
    prey_size = 0.050
    collision_margin = 0.02  # 安全缓冲，用于软惩罚
    time_penalty = -0.01

    # 目标围捕半径（略大于碰撞范围，略小于 capture_threshold）
    target_radius = 0.35
    min_effective_radius = 0.15  # 太小意味着几乎碰撞
    max_effective_radius = capture_threshold

    # 权重
    w_approach = 1.0
    w_group_approach = 0.5
    w_radius_uniform = 0.5
    w_angle_uniform = 0.7
    w_enclose_bonus = 1.5
    w_collision = 2.0
    w_separation_soft = 0.3
    w_time = 1.0

    # ------------------------
    # 基本状态提取
    # ------------------------
    agent_positions = global_state["agent_positions"]
    agent_velocities = global_state["agent_velocities"]
    prey_pos = global_state["prey_position"]
    inter_agent_distances = global_state["inter_agent_distances"]
    distances_to_prey_all = global_state["distances_to_prey"]

    # 识别追捕者索引和当前 agent 索引
    num_agents = agent_positions.shape[0]
    # 在 MPE 中通常 adversary 在前面，但这里从 world 反查以保证一致
    adversary_indices = []
    prey_index = None
    for idx, a in enumerate(world.agents):
        if a.adversary:
            adversary_indices.append(idx)
        else:
            prey_index = idx

    # 当前 agent 索引
    agent_index = None
    for idx, a in enumerate(world.agents):
        if a.name == agent_name:
            agent_index = idx
            break

    if agent_index is None or prey_index is None or len(adversary_indices) == 0:
        return 0.0, {}

    # 追捕者相对猎物的距离（重新计算以确保索引一致）
    adv_positions = agent_positions[adversary_indices]
    dists_adv_to_prey = np.linalg.norm(adv_positions - prey_pos, axis=1)
    self_dist_to_prey = np.linalg.norm(
        agent_positions[agent_index] - prey_pos
    )

    # ------------------------
    # 1) 接近目标 & 团队收缩奖励
    # ------------------------
    # 自身接近：对距离使用负形势，归一化到 world_size
    self_approach_reward = -self_dist_to_prey / world_size

    # 团队最远追捕者距离：鼓励所有追捕者一起收缩圈
    max_team_dist = np.max(dists_adv_to_prey) if dists_adv_to_prey.size > 0 else 0.0
    group_approach_reward = -max_team_dist / world_size

    components["self_approach"] = w_approach * self_approach_reward
    components["group_approach"] = w_group_approach * group_approach_reward

    # ------------------------
    # 2) 防碰撞：硬惩罚 + 软间距
    # ------------------------
    collision_penalty = 0.0
    separation_soft_penalty = 0.0

    # 与其他追捕者的距离
    for j in adversary_indices:
        if j == agent_index:
            continue
        d = inter_agent_distances[agent_index, j]
        # 硬碰撞阈值
        hard_threshold = 2 * adv_size
        soft_threshold = hard_threshold + collision_margin

        # 硬碰撞
        if d < hard_threshold:
            collision_penalty -= 1.0

        # 软间距：当距离接近硬阈值时施加斥力
        if d < soft_threshold:
            # 线性衰减的惩罚，越近惩罚越大
            separation_soft_penalty -= (soft_threshold - d) / soft_threshold

    components["collision"] = w_collision * collision_penalty
    components["separation_soft"] = w_separation_soft * separation_soft_penalty

    # ------------------------
    # 3) 包围队形：半径均匀 + 角度均匀 + 包围判定
    # ------------------------
    formation_radius_reward = 0.0
    formation_angle_reward = 0.0
    enclose_bonus = 0.0

    if len(adversary_indices) == 3:
        # 半径均匀性：接近 target_radius 且方差小
        radii = dists_adv_to_prey
        mean_radius = np.mean(radii)
        radius_var = np.var(radii)

        # 距离目标半径的偏差（越小越好）
        radius_deviation = np.abs(mean_radius - target_radius)

        # 只在有效半径区间内奖励形状，避免过近或过远
        if min_effective_radius <= mean_radius <= max_effective_radius:
            # 标准化到 [0,1] 范围，防止过大数值
            radius_uniform_term = -radius_var / (max_effective_radius**2 + 1e-6)
            radius_target_term = -radius_deviation / (max_effective_radius + 1e-6)
            formation_radius_reward = radius_uniform_term + 0.5 * radius_target_term
        else:
            formation_radius_reward = -0.1  # 半径区间不合适时轻微惩罚

        # 角度均匀性：以猎物为原点
        rel_positions = adv_positions - prey_pos
        angles = np.arctan2(rel_positions[:, 1], rel_positions[:, 0])
        angles = np.sort(angles)

        # 计算环形角度差（考虑 2π 回绕）
        diffs = np.diff(angles)
        last_gap = (angles[0] + 2 * np.pi) - angles[-1]
        angle_gaps = np.concatenate([diffs, np.array([last_gap])])

        ideal_gap = 2 * np.pi / 3.0
        gap_dev = angle_gaps - ideal_gap
        angle_var = np.mean(gap_dev**2)

        # 标准化角度方差（最大可能偏差约 pi）
        formation_angle_reward = -angle_var / (np.pi**2 + 1e-6)

        # 包围判定：猎物在追捕者三角形内部
        # 使用重心坐标法判断点在三角形内
        p0 = prey_pos
        p1, p2, p3 = adv_positions

        def point_in_triangle(pt, a, b, c):
            v0 = c - a
            v1 = b - a
            v2 = pt - a
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
            return (u >= 0.0) and (v >= 0.0) and (u + v <= 1.0)

        inside = point_in_triangle(p0, p1, p2, p3)

        # 若猎物被包含且半径适中，则加额外奖励
        if inside and min_effective_radius <= mean_radius <= max_effective_radius:
            enclose_bonus = 1.0
            # 若同时团队半径都小于 capture_threshold，视为高质量围捕
            if np.all(radii < capture_threshold):
                enclose_bonus += 1.0

    components["formation_radius"] = w_radius_uniform * formation_radius_reward
    components["formation_angle"] = w_angle_uniform * formation_angle_reward
    components["enclose_bonus"] = w_enclose_bonus * enclose_bonus

    # ------------------------
    # 4) 时间惩罚：鼓励尽快完成围捕
    # ------------------------
    components["time_penalty"] = w_time * time_penalty

    total_reward = float(sum(components.values()))
    return total_reward, components
```
```
