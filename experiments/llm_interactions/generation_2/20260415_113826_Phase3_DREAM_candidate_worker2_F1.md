# LLM Interaction Log

- **Generation**: 2
- **Phase**: Phase3_DREAM
- **Candidate Info**: worker2_F1
- **Timestamp**: 20260415_113826

================================================================================
## Prompt (Sent to LLM)
================================================================================

```text
你是一位专业的强化学习奖励工程师。请基于上一代的诊断反馈执行特定的变异操作。

# 环境基座

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


# 上一代最优代码
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
    time_penalty_per_step = -0.002  # 降低时间惩罚权重

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
    agent_index = None
    for idx, ag in enumerate(world.agents):
        if getattr(ag, "name", None) == agent_name:
            agent_index = idx
            break

    if agent_index is None:
        try:
            agent_index = int("".join(ch for ch in agent_name if ch.isdigit()))
            if agent_index < 0 or agent_index >= num_agents:
                agent_index = 0
        except ValueError:
            agent_index = 0

    adversary_indices = [
        idx for idx, ag in enumerate(world.agents) if getattr(ag, "adversary", False)
    ]

    try:
        adv_local_index = adversary_indices.index(agent_index)
    except ValueError:
        adv_local_index = 0

    # ------------------------
    # 1. 距离引导：靠近目标 + 围捕完成奖励
    # ------------------------
    if len(distances_to_prey) > 0:
        d_self = distances_to_prey[adv_local_index]
        mean_d_adv = float(np.mean(distances_to_prey))

        norm_d_self = d_self / world_size
        norm_d_mean = mean_d_adv / world_size

        # 放大靠近猎物的正向影响，同时降低团队项惩罚
        distance_reward_self = -0.8 * norm_d_self
        distance_reward_team = -0.2 * norm_d_mean

        all_within_capture = bool(
            np.all(distances_to_prey < capture_threshold)
        )
        capture_bonus = 0.0
        if all_within_capture:
            # 显著放大捕获奖励
            capture_bonus = 15.0

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

    for j in adversary_indices:
        if j == agent_index:
            continue
        d_ij = inter_agent_distances[agent_index, j]
        min_dist = 2.0 * adv_size
        safe_dist = safe_distance_factor * min_dist

        if d_ij < min_dist:
            # 略微减弱硬碰撞惩罚，避免过度保守
            collision_penalty -= 1.0
        elif d_ij < safe_dist:
            ratio = (safe_dist - d_ij) / (safe_dist - min_dist + 1e-8)
            near_collision_penalty -= 0.3 * ratio

    prey_index = None
    for idx, ag in enumerate(world.agents):
        if not getattr(ag, "adversary", False):
            prey_index = idx
            break
    if prey_index is not None:
        d_prey = inter_agent_distances[agent_index, prey_index]
        min_dist_ap = adv_size + prey_size
        if d_prey < min_dist_ap:
            # 保留轻微惩罚
            collision_penalty -= 0.3

    components["collision_penalty"] = collision_penalty
    components["near_collision_penalty"] = near_collision_penalty

    # ------------------------
    # 3. 包围队形：均匀半径 + 均匀角度
    # ------------------------
    formation_radius_reward = 0.0
    formation_angle_reward = 0.0
    center_in_triangle_reward = 0.0

    if len(adversary_indices) == 3 and prey_index is not None:
        adv_positions = agent_positions[adversary_indices]
        vecs = adv_positions - prey_pos[None, :]
        radii = np.linalg.norm(vecs, axis=1) + 1e-8

        mean_r = float(np.mean(radii))
        var_r = float(np.var(radii))

        radius_dev = abs(mean_r - formation_radius_target)
        if mean_r < capture_threshold + formation_radius_tolerance:
            # 整体减弱半径惩罚
            formation_radius_reward += -0.4 * radius_dev / (capture_threshold + 1e-8)
        formation_radius_reward += -0.2 * var_r / (capture_threshold ** 2 + 1e-8)

        angles = np.arctan2(vecs[:, 1], vecs[:, 0])
        angles_sorted = np.sort(angles)
        angle_diffs = np.diff(angles_sorted)
        last_gap = (angles_sorted[0] + 2.0 * np.pi) - angles_sorted[-1]
        angle_diffs = np.concatenate([angle_diffs, [last_gap]])
        angle_dev = np.abs(angle_diffs - angle_uniform_target)
        angle_dev_mean = float(np.mean(angle_dev))
        # 降低角度惩罚权重
        formation_angle_reward = -0.3 * angle_dev_mean / np.pi

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
            # 明显放大奖励，引导形成有效包围
            center_in_triangle_reward = 3.0

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
    adv_vels = agent_velocities[adversary_indices]
    if adv_vels.shape[0] > 1:
        speed_var = float(np.mean(np.var(adv_vels, axis=0)))
        # 略微放大多样性正向项
        components["velocity_diversity"] = speed_var * 0.1
    else:
        components["velocity_diversity"] = 0.0

    total_reward = float(sum(components.values()))
    return total_reward, components
```

# 客观诊断反馈
[病理诊断]
1. 整体表现与任务达成度
- Fitness = 0.0020，成功率仅 1.47%，平均捕获时间 99.4 steps（接近 episode 长度上限），说明：
  - 捕获大多发生在接近超时的尾部，非高效围捕；
  - 策略整体仍处于“偶然成功”阶段，奖励设计对策略塑形较弱。
- 捕获相关分量：
  - capture_penalty: mean = -10.0（常数），说明失败结局强惩罚，但没有对应的、足够强的正向成功信号抵消。
  - capture_bonus: mean = 0.0022 且 std = 0.1822，整体期望接近 0，说明成功太稀疏，对训练几乎没有驱动力。
  - time_penalty: mean = -0.002，线性时间惩罚，相对 -10 的失败惩罚非常弱，无法强力推动尽快围捕。

  结论：当前范式是“强失败惩罚 + 极稀疏成功奖励”，整体倾向“保守避免失败”，而不是“主动高效围捕”。

2. 各奖励分量主导性与失效分析
从均值绝对值看主导项（按 |mean| 排序）：
- capture_penalty: -10.0000（终局主导项）
- center_in_triangle: +1.4262
- escape_reward: -0.5380
- distance_self: -0.6331
- formation_radius: -0.4224
- boundary_penalty: -0.3021
- distance_team: -0.1583
- formation_angle: -0.0995
- velocity_diversity: +0.0079
- time_penalty: -0.0020
- capture_bonus: +0.0022
- near_collision_penalty / collision_penalty: 约 0

2.1 明显主导、方向大致合理但可能配比失衡的分量
- capture_penalty = -10：
  - 强制区分成功/失败是必要的，但在成功率极低的早期，过大失败惩罚可能让策略更偏向“拖延”而不是“尝试围捕”。
  - 与 capture_bonus (≈0 均值) 形成严重不平衡：失败强烈可见，成功基本“看不见”。

- center_in_triangle: mean = +1.4262, std = 1.4982：
  - 这是当前少数显著为正的主导项，说明奖励在强烈鼓励“猎物在三角形中心（被包围）”。
  - 配合协同指标：
    - formation_quality: 0.2889（有一定阵型结构）
    - encirclement_angle_std: 1.1205（角度分布离散度较低，表明有一定围捕结构倾向）
  - 但成功率仍低，表明“中心在三角形内”这一几何条件经常被短暂满足，但未转化为稳定捕获事件（成功检测 / 终局条件或捕获半径与奖励几何结构不完全匹配）。

- distance_self: mean = -0.6331：
  - 惩罚个体与猎物的距离，鼓励逼近。该分量相对大且为负，说明当前策略普遍“围得不够紧”或奖励设计使得“靠近”的收益不足以克服其他惩罚（如 escape_reward）。
  - 与 avg_distance_to_prey ≈ 1.9941 相互印证：从协同指标看，平均距离偏大，未进入紧密围捕区。

- escape_reward: mean = -0.5380：
  - 名称“escape_reward”但为负，实质是惩罚猎物逃脱或向外突破。
  - 与 center_in_triangle 相互作用：一边奖励围住（center_in_triangle），一边惩罚逃脱（escape_reward），方向上是一致的。但该惩罚的均值幅度（0.538）与 distance_self / formation_radius 在同一量级，可能造成：
    - “只要维持一个大致包围，不让逃脱”比“积极收缩捕获”更容易，导致策略收缩乏力。

- formation_radius: mean = -0.4224：
  - 惩罚队形半径偏大，鼓励紧密队形。与 distance_self/center_in_triangle 共同塑形为“围成圈 + 缩小半径”的范式。
  - 但 min_agent_distance: mean = 1.9610，说明多机器人之间仍保持较大安全距离，队形不够紧凑，可能是碰撞/near_collision惩罚与formation_radius之间的平衡导致“安全大圈”。

- boundary_penalty: mean = -0.3021, std = 1.6024：
  - 有一定强度且波动较大，说明 agent 较频繁接近或触碰边界。
  - 推测：在追逐中，部分 agent 会被奖励驱动冲向边界方向（例如为从背后包抄），但边界惩罚过重又拉回，形成“拉扯”，降低策略稳定性和效率。

2.2 明显弱效或失效的分量
- capture_bonus: mean = 0.0022, std = 0.1822，成功率 1.47%：
  - 从期望看该项几乎为 0，说明大部分 episode 不触发或触发次数极少。
  - 相对 capture_penalty 的 -10，其期望贡献可忽略不计，典型“稀疏且弱”的成功信号，训练几乎依赖负向信号（失败/逃脱/距离）塑形。
  - 结论：设计方向对，但强度和触发机制在当前阶段“基本失效”。

- time_penalty: mean = -0.0020：
  - 绝对值极小，且是一步一步线性扣分，覆盖整个 episode 约 100 steps 也只有 -0.2 左右，远小于一次失败的 -10。
  - 对“加快捕获”的引导力度非常有限，难以形成显著“时间压力”。

- velocity_diversity: mean = +0.0079, std = 0.0057：
  - 数值很小，变化范围也极窄，相对于其他分量几乎不起作用。
  - 说明当前权重设计下，多样性激励对策略几乎没有塑形；当前策略可能找到某种“安全、保守”的速度模式即可满足其他主要分量，不需要刻意丰富动作用法。

- near_collision_penalty: mean ≈ 0, std = 0.0011
- collision_penalty: mean ≈ 0, std = 0.0015
  - 两者期望几乎为 0，且方差极小，说明：
    - 要么碰撞事件极少（可能仍然合理，说明边距很安全），
    - 要么惩罚权重过小，相比其他项可以被完全忽略，从而不能细致调整最靠近时的动作。
  - 在 min_agent_distance ≈ 1.96 的前提下，更可能是：策略通过“保持大间隔”来规避潜在碰撞，而不是学会贴身高效围捕；此时碰撞惩罚虽存在，对最终策略的“边界精细控制”基本失效。

2.3 协同指标与围捕策略结构分析
- encirclement_angle_std: mean = 1.1205：
  - 偏低说明围捕角度分布相对集中，有一定“环绕”结构，agents 不完全在同一侧。
- min_agent_distance: mean = 1.9610：
  - agent 彼此间保持较大安全距离，结合 near_collision/collision 几乎为 0，可以推断：
    - 当前策略选择“宽松大圈包围”，避免相互干扰，但这会限制最终收缩效率。
- avg_distance_to_prey: 1.9941：
  - 与 min_agent_distance 同量级，说明围捕圈与猎物距离不算小，基本是远距离围困而非紧逼。
- formation_quality: mean = 0.2889：
  - 中等偏低，说明阵型有一定结构，但不稳定、不精细。

综合协同诊断：
- 当前策略倾向于形成一个“相对稀疏的大圈”包住猎物，让其不易逃脱，而不是迅速缩圈完成捕获。
- 奖励范式整体在鼓励“稳定包围态”，而非“冲击捕获态”；成功事件稀疏，时间压力弱，捕获奖励弱，导致训练重心偏向消极防御，而非积极进攻。
- 没有明显“缺失机制”：边界约束、队形、包围、距离、自身间距、速度多样性等维度都已覆盖，但关键主导项（center_in_triangle + escape_reward + distance_self + capture_penalty）构成的整体势场，对“完成终结动作”的驱动不足。

2.4 是否存在反向效果 / 设计范式问题
- 没有典型意义上的“反向奖励”（即设计本身违背任务目标），但存在以下范式问题：
  1) 成功信号极稀疏且极弱：capture_bonus 完全压不住 capture_penalty，导致策略更关注“如何不失败”而非“如何成功”。
  2) 时间因素几乎不重要：time_penalty 极小，使得“拖到超时失败”与“早早失败”在期望上接近，不形成足够快收缩的动力。
  3) 包围态过度稳定：center_in_triangle + escape_reward + formation_radius 会共同鼓励“有圈就好，保持别让跑掉”，而不是“进一步缩圈、冒一定风险完成捕获”。
  4) 碰撞/近碰惩罚太弱，间接鼓励“保持大间距”而不是“在小距离下精细控制”，简化了策略，但降低最终捕获效率。

- 这是一个“范式偏防守、成功过稀疏”的系统性问题，但尚未完全不可挽回：各分量方向基本合理，只是配比与形状需要较大调整。

结论性判断：
- 奖励设计不需要立刻 L1 全盘推翻；更适合通过：
  - F2：重构 capture_bonus / time_penalty / center_in_triangle 等关键分量的数学形式（使成功更密集、更渐进；时间压力更非线性显性；强化收缩阶段）。
  - F3：重新平衡 capture_penalty vs capture_bonus vs distance_self/formation_radius/escape_reward/boundary_penalty 等权重，缓和“只会保守控场”的范式。
- F1 引入全新分量可以作为探索分支（例如显式“收缩半径奖励”或“多阶段里程碑奖励”），但现有维度已经较丰富，不是首要矛盾。
- L1 可作为一个激进候选，用于尝试完全不同范式（如基于势场的逐步缩圈 + 分阶段奖励），以防当前范式存在结构性上限。

[算子分配]
Candidate 0: F2
Candidate 1: F3
Candidate 2: F1
Candidate 3: L1

# 强制变异指令
【Reward Branch Augmentation (分支扩充)】
请完全保留原代码的现有逻辑和权重，新增一个（且仅新增一个）奖励或惩罚分量，用于解决诊断报告中缺失的协同行为引导。

# 接口规范
请实现 compute_reward 函数，保持接口签名不变：

```python
import numpy as np

def compute_reward(agent_name, observation, global_state, actions, world):
    if not global_state['is_adversary']: return 0.0, {}
    components = {}
    # [根据变异指令修改这里的核心逻辑]
    total_reward = sum(components.values())
    return total_reward, components
```

只允许输出1个Python代码块，严禁任何解释性文字，直接以 ```python 开头。

# 致命红线约束 (Anti-Hallucination Guardrails)
- **禁止引入外部依赖或未定义的类**：绝不可实例化 `CoreEnvLogic()` 或尝试访问 `world.logic`、`world.adversary_params` 等不存在属性。
- **物理常量硬编码**：若要新增依赖于物理常量的逻辑（如体积、速度），必须直接在函数内写死数值常量（如 `adv_size = 0.075`）。
- **严格保持接口签名**：必须保留 `components` 字典收集机制并返回 `total_reward, components`。
- 只允许输出1个修改后的Python代码块，严禁任何解释性文字，直接以 ```python 开头。
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
    time_penalty_per_step = -0.002  # 降低时间惩罚权重

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
    agent_index = None
    for idx, ag in enumerate(world.agents):
        if getattr(ag, "name", None) == agent_name:
            agent_index = idx
            break

    if agent_index is None:
        try:
            agent_index = int("".join(ch for ch in agent_name if ch.isdigit()))
            if agent_index < 0 or agent_index >= num_agents:
                agent_index = 0
        except ValueError:
            agent_index = 0

    adversary_indices = [
        idx for idx, ag in enumerate(world.agents) if getattr(ag, "adversary", False)
    ]

    try:
        adv_local_index = adversary_indices.index(agent_index)
    except ValueError:
        adv_local_index = 0

    # ------------------------
    # 1. 距离引导：靠近目标 + 围捕完成奖励
    # ------------------------
    if len(distances_to_prey) > 0:
        d_self = distances_to_prey[adv_local_index]
        mean_d_adv = float(np.mean(distances_to_prey))

        norm_d_self = d_self / world_size
        norm_d_mean = mean_d_adv / world_size

        # 放大靠近猎物的正向影响，同时降低团队项惩罚
        distance_reward_self = -0.8 * norm_d_self
        distance_reward_team = -0.2 * norm_d_mean

        all_within_capture = bool(
            np.all(distances_to_prey < capture_threshold)
        )
        capture_bonus = 0.0
        if all_within_capture:
            # 显著放大捕获奖励
            capture_bonus = 15.0

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

    for j in adversary_indices:
        if j == agent_index:
            continue
        d_ij = inter_agent_distances[agent_index, j]
        min_dist = 2.0 * adv_size
        safe_dist = safe_distance_factor * min_dist

        if d_ij < min_dist:
            # 略微减弱硬碰撞惩罚，避免过度保守
            collision_penalty -= 1.0
        elif d_ij < safe_dist:
            ratio = (safe_dist - d_ij) / (safe_dist - min_dist + 1e-8)
            near_collision_penalty -= 0.3 * ratio

    prey_index = None    # noqa: F841
    for idx, ag in enumerate(world.agents):
        if not getattr(ag, "adversary", False):
            prey_index = idx
            break
    if prey_index is not None:
        d_prey = inter_agent_distances[agent_index, prey_index]
        min_dist_ap = adv_size + prey_size
        if d_prey < min_dist_ap:
            # 保留轻微惩罚
            collision_penalty -= 0.3

    components["collision_penalty"] = collision_penalty
    components["near_collision_penalty"] = near_collision_penalty

    # ------------------------
    # 3. 包围队形：均匀半径 + 均匀角度
    # ------------------------
    formation_radius_reward = 0.0
    formation_angle_reward = 0.0
    center_in_triangle_reward = 0.0

    if len(adversary_indices) == 3 and prey_index is not None:
        adv_positions = agent_positions[adversary_indices]
        vecs = adv_positions - prey_pos[None, :]
        radii = np.linalg.norm(vecs, axis=1) + 1e-8

        mean_r = float(np.mean(radii))
        var_r = float(np.var(radii))

        radius_dev = abs(mean_r - formation_radius_target)
        if mean_r < capture_threshold + formation_radius_tolerance:
            # 整体减弱半径惩罚
            formation_radius_reward += -0.4 * radius_dev / (capture_threshold + 1e-8)
        formation_radius_reward += -0.2 * var_r / (capture_threshold ** 2 + 1e-8)

        angles = np.arctan2(vecs[:, 1], vecs[:, 0])
        angles_sorted = np.sort(angles)
        angle_diffs = np.diff(angles_sorted)
        last_gap = (angles_sorted[0] + 2.0 * np.pi) - angles_sorted[-1]
        angle_diffs = np.concatenate([angle_diffs, [last_gap]])
        angle_dev = np.abs(angle_diffs - angle_uniform_target)
        angle_dev_mean = float(np.mean(angle_dev))
        # 降低角度惩罚权重
        formation_angle_reward = -0.3 * angle_dev_mean / np.pi

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
            # 明显放大奖励，引导形成有效包围
            center_in_triangle_reward = 3.0

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
    adv_vels = agent_velocities[adversary_indices]
    if adv_vels.shape[0] > 1:
        speed_var = float(np.mean(np.var(adv_vels, axis=0)))
        # 略微放大多样性正向项
        components["velocity_diversity"] = speed_var * 0.1
    else:
        components["velocity_diversity"] = 0.0

    # ------------------------
    # 6. 新增：协同收缩奖励（鼓励在包围态下统一向内收缩）
    # ------------------------
    cooperative_shrink_reward = 0.0
    if len(adversary_indices) == 3:
        adv_positions = agent_positions[adversary_indices]
        adv_vels = agent_velocities[adversary_indices]
        vecs_to_prey = prey_pos[None, :] - adv_positions
        dist_to_prey = np.linalg.norm(vecs_to_prey, axis=1) + 1e-8
        unit_inward = vecs_to_prey / dist_to_prey[:, None]
        # 每个追捕者速度在“向内”方向上的分量
        inward_speed = np.sum(adv_vels * unit_inward, axis=1)
        mean_inward_speed = float(np.mean(inward_speed))
        # 仅在已经基本包围（所有追捕者不在同一半平面）时给予奖励
        centered_vecs = adv_positions - prey_pos[None, :]
        angles = np.arctan2(centered_vecs[:, 1], centered_vecs[:, 0])
        angles_sorted = np.sort(angles)
        angle_diffs = np.diff(angles_sorted)
        last_gap = (angles_sorted[0] + 2.0 * np.pi) - angles_sorted[-1]
        angle_diffs = np.concatenate([angle_diffs, [last_gap]])
        max_gap = float(np.max(angle_diffs))
        # 大间隙小于 200° 视为“有一定环绕结构”
        if max_gap < (200.0 / 180.0 * np.pi):
            cooperative_shrink_reward = 0.05 * mean_inward_speed

    components["cooperative_shrink"] = cooperative_shrink_reward

    total_reward = float(sum(components.values()))
    return total_reward, components
```
```
