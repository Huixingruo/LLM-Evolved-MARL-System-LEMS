# LLM Interaction Log

- **Generation**: 1
- **Phase**: Phase3_DREAM
- **Candidate Info**: worker1_F3
- **Timestamp**: 20260415_103800

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

# 客观诊断反馈
[病理诊断]
1. 总体表现与任务完成度
- Fitness 仅 0.0011，成功率 3.27%，说明当前策略基本“偶然”成功，围捕能力非常弱。
- 平均捕获时间接近 100 steps（看起来接近最大时长），多数回合是拖到超时而非高效收敛围捕。

2. 奖励分量主导性与失效分析  
从均值绝对值看主导负向项：  
- capture_penalty: mean = -10.0000（极大常数项）
- formation_radius: mean = -1.4712
- distance_self: mean = -0.7293  
其次负向但强度较小：
- formation_angle: -0.5908
- distance_team: -0.3646
- boundary_penalty: -0.3827
- escape_reward: -0.4963
- time_penalty: -0.0100（但为常驻密集惩罚）

正向/弱向项：
- center_in_triangle: +0.0562
- velocity_diversity: +0.0030
- capture_bonus: +0.0017（极其稀疏且均值远小于失败惩罚）
- near_collision_penalty, collision_penalty 几乎为 0，说明碰撞惩罚极少触发或权重极小。

结论：  
- 当前训练主体由强大失败惩罚 capture_penalty 和一系列几何约束负奖励主导，稀疏的成功奖励（capture_bonus）在总体上几乎没有存在感，策略在优化“避免损失”而非“实现捕获”。
- capture_penalty = -10 与 capture_bonus 均值 0.0017 的量级对比极度失衡：成功一次的长期回报几乎无法弥补一次失败的惩罚，导致策略倾向于“拖延不死”和“守恒形态”而不是积极逼近猎物、尝试围捕。

3. 各关键分量逐个诊断  
(1) capture_penalty / capture_bonus / time_penalty（任务核心相关）  
- capture_penalty: 固定 -10，std=0 ，说明几乎每个 episode 都吃到相同的大惩罚（大概率是“没抓到”时一次性扣分），且该项完全压倒其他所有项。
- capture_bonus: mean=0.0017, std=0.0909，说明偶尔有较大正奖励，但在 sample 级平均后非常小，稀疏+量级太低。
- time_penalty: -0.0100, std=0.0000，为恒定步长惩罚。由于几乎所有 episode 都拖到最大步数，time_penalty 实质成为“固定额外失败惩罚”，增加了“拖延”的成本，但在 capture_penalty = -10 的背景下并不是主导因素，而是进一步降低了整体回报，使策略在任何情况下都很难获得“正向经验”。

初步判断：
- 当前“成败机制”严重单边：强失败、弱成功，且配合长时超时，导致价值函数学习非常困难。
- 该部分需要通过 F2（重构奖励形态，例如分段、阶段里程碑）和/或 F3（权重再平衡）进行重点调整。

(2) formation_radius / distance_self / distance_team / formation_angle（队形与相对位置）  
- formation_radius: mean=-1.4712, std=2.7010，绝对值大且波动大，说明该分量在训练中占比较大，但可能设计过于苛刻：在当前低水平策略下，队形半径约束很难同时满足，导致长期持续负奖励。
- distance_self: mean=-0.7293, std=0.4310，鼓励队员相互接近，但均值明显偏负，说明代理之间距离经常不在期望区间；若阈值过严，可能导致两极化：过近→near_collision / collision 惩罚介入，过远→distance_self持续惩罚。
- distance_team: mean=-0.3646, std=0.1783，倾向于让队伍整体靠近某参考点（可能是目标或队伍中心）。负值说明普遍不达标。
- formation_angle: mean=-0.5908, std=0.1818，说明当前代理在角度分布上与期望的“包围角度”偏差较大，持续受罚。

协同指标支撑：
- formation_quality: mean=0.2240（偏低），encirclement_angle_std: 1.9962（较大），说明几乎没有稳定的包围角结构，角度分布离散。
- avg_distance_to_prey: 1.8247，min_agent_distance: 0.9462 —— 代理既不够靠近猎物，也没有形成紧密小队，整体呈“中距离、松散队形”。

综合判断：
- 队形相关奖励全部为负且无明显“边界内奖励区”，更像是“持续惩罚状态”，在代理尚未具备基本追踪能力时，过多几何约束容易把梯度埋在噪声里。
- 若这些分量设计为线性惩罚+固定期望距离/角度，很可能在实际轨迹分布上大多数状态都落在“强惩罚区”，等价于给策略施加多重束缚，却没有清晰的“通往成功的梯度路径”。

这类分量适合：
- 用 F2 调整数学形态（引入容忍区、分段函数、软阈值、非线性衰减），减少对策略初期探索的过度打击。
- 用 F3 适当减权，在当前阶段让“抓到猎物”信号更突出，队形约束退居为次要 shaping。

(3) boundary_penalty / escape_reward  
- boundary_penalty: mean=-0.3827, std=1.8090，方差极大，说明在边界相关事件上时而强罚，时而无事。当前均值偏负但权重不算最主导。
- escape_reward: mean=-0.4963, std=0.6586，从命名看本应是“阻止猎物逃脱”相关，但为负值，可能设计为“逃脱惩罚”（越接近逃脱越负）。当前均值略大于 distance_team/distance_self 等，说明猎物的逃逸问题较显著。

问题：
- 若 escape_reward 仅在猎物快逃脱/已逃脱时给巨大负数，而平时为零，则与 capture_penalty 一样是“硬失败惩罚”，没有引导如何防止逃脱，只是在失败时再扣一刀。
- 两者都偏负、且形态可能类似（硬惩罚），加重了“失败总值为极大负”的局面。

(4) collision_penalty / near_collision_penalty / velocity_diversity  
- near_collision_penalty: mean ≈ 0, std=0.0025，collision_penalty: mean ≈ 0, std=0.0005 —— 几乎不生效，说明当前策略远离碰撞状态，或者阈值/权重过低，代理不会因为碰撞问题受到明显约束。
- velocity_diversity: mean=0.0030, std=0.0026，中度正值，但绝对值很小。该分量在总奖励中权重极弱，对策略几乎无导向意义。

结论：
- 当前模型压根没发展出容易碰撞的积极进攻行为，所以碰撞相关惩罚没有发挥作用。此处不需要 L1，大概率是权重/触发条件问题（F2/F3）。

(5) center_in_triangle  
- mean=0.0562, std=0.2303，为少数正向 shaping 信号之一，意在鼓励猎物位于捕食者三角形内部。
- 但是其均值远小于一大堆负向分量，虽方向正确，但量级不足，难以驱动围捕策略形成。

该分量值得保留但需提升引导强度（F3 适合）或优化公式（F2：例如当猎物从三角形外→边界→内部时给强烈跃迁奖励）。

4. 协同缺陷总结  
- 目标层面：  
  * “成功捕获”信号非常稀疏且相对不重要；“失败/逃脱”信号巨大且集中在 episode 末尾，导致 value 学习接近“无望的负值背景”。
  * time_penalty 进一步降低所有轨迹的回报，使得任何探索路径都很难显著优于“随机拖时间失败”。

- 几何/队形层面：  
  * 多个持续负的几何分量在策略尚未学会追踪时就施加，形成一堆“无处不在的惩罚”，没有清晰的“爬坡方向”。
  * formation_quality 明显偏低，encirclement_angle_std 高，说明这些惩罚没有形成良好的“包围势场”，反而更像噪声源。

- 协作行为层面：  
  * min_agent_distance 较大（0.9462），avg_distance_to_prey = 1.8247，整体呈“远离猎物，队形也不紧凑”的防御性配置，与“强失败惩罚 + 队形惩罚”组合一致：代理宁可远离问题区域，也不愿尝试围捕。

综合判断：  
- 当前奖励范式虽未“根本错误”（仍有捕获、队形、边界等关键元素），但其权重配比和数学形态严重失衡，呈现：
  * 失败惩罚过大 + 成功奖励过弱 + 几何约束负值密集 → 导致策略普遍悲观、少进攻。
- 适合的策略：
  * 主线不必立即 L1 全推翻，但需要“深入重构若干分量（F2）+ 全局权重再平衡（F3）”，并保留一条更激进 L1 试验支路。
  * 新机制（F1）需求不算最迫切，因为核心维度并不缺（捕获、队形、边界、协同都已有），但可以设置一个 F1 分支有针对性补充“阶段性捕获/围堵成功的里程碑奖励”，以弥补当前全靠结局大惩罚/微奖励的格局。

5. 对四类算子的需求评估  
- F1：需求中等偏弱。当前主要问题不是“缺少维度”，而是现有分量用不好。但可以用一次 F1 测试“中间阶段”的协同机制（如构建局部围堵成功的奖励）。
- F2：需求高。capture_penalty、escape_reward、formation_radius、distance_self 等表现出“常压负值+缺乏渐进结构”，很适合改为分段/非线性形式。
- F3：需求高。需要明显增大捕获与 center_in_triangle 等积极分量权重，削弱全局失败惩罚与过强几何惩罚。
- L1：需求中等。虽然还未多代尝试，但目前范式的“巨大负值背景”可能从根本上不利于训练，需要至少一个激进候选重写成“潜在场 + 分阶段目标”的新范式以做对照。

因此下一代 4 个候选建议：以 F2/F3 为主，辅以 F1 和 L1 各一个，保证多样性和探索广度。

[算子分配]
Candidate 0: F2
Candidate 1: F3
Candidate 2: F1
Candidate 3: L1

# 强制变异指令
【Reward Equilibrium Tuning (平衡微调)】
绝对不要增加或删除现有的逻辑分支！请严格保持代码拓扑不变，仅根据诊断报告，修改各奖励分量的权重系数（增大/减小）。

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
```
