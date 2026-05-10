# LLM Interaction Log

- **Generation**: 1
- **Phase**: Phase3_EvoLeap
- **Candidate Info**: worker2_F3
- **Timestamp**: 20260409_175545

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
    # 非追捕者零奖励
    if not global_state['is_adversary']:
        return 0.0, {}

    components = {}

    # -----------------------
    # 硬编码物理与任务常量
    # -----------------------
    world_size = 2.5
    capture_threshold = 0.5

    adv_size = 0.075
    prey_size = 0.050

    collision_margin = 0.01
    desired_radius = capture_threshold
    time_penalty = -0.01

    # 角度与半径均匀性系数
    radius_tolerance = 0.05
    angle_target_step = 2.0 * np.pi / 3.0

    # 奖励权重
    w_approach = 1.0
    w_capture_bonus = 10.0
    w_radius_uniformity = 0.5
    w_angle_uniformity = 0.7
    w_centering = 0.3
    w_collision = -5.0
    w_min_separation = -0.5

    # -----------------------
    # 提取全局状态
    # -----------------------
    agent_positions = global_state['agent_positions']
    prey_pos = global_state['prey_position']
    distances_to_prey = global_state['distances_to_prey']
    inter_agent_distances = global_state['inter_agent_distances']

    # world.agents 顺序假设：3 adversaries + 1 prey
    # 找到当前追捕者索引
    all_agents = world.agents
    adversary_indices = [i for i, a in enumerate(all_agents) if a.adversary]
    prey_indices = [i for i, a in enumerate(all_agents) if not a.adversary]

    if len(adversary_indices) != 3 or len(prey_indices) != 1:
        # 结构异常时保守返回时间惩罚
        components['time_penalty'] = time_penalty
        return sum(components.values()), components

    prey_index = prey_indices[0]

    try:
        current_agent_index = [i for i, a in enumerate(all_agents)
                               if a.name == agent_name][0]
    except (IndexError, AttributeError):
        # 回退：按 adversary 顺序映射，保证索引有效
        # 这里假设 agent_name 为 "adversary_X"
        try:
            idx = int(str(agent_name).split('_')[-1])
            current_agent_index = adversary_indices[idx]
        except Exception:
            current_agent_index = adversary_indices[0]

    # 当前智能体在 adversary 列表中的索引（用于 distances_to_prey）
    try:
        adv_local_index = adversary_indices.index(current_agent_index)
    except ValueError:
        # 若不在 adversary_indices 中，视为无效追捕者
        components['time_penalty'] = time_penalty
        return sum(components.values()), components

    # -----------------------
    # 1. 接近目标的距离奖励
    # -----------------------
    # 使用距离的负值作为基础接近奖励，并在 capture_threshold 内鼓励靠近 desired_radius
    current_dist = distances_to_prey[adv_local_index]

    # 距离引导：越接近越好
    distance_reward = -current_dist / world_size

    # 在 capture_threshold 内，对接近 desired_radius 的行为给额外奖励
    if current_dist < capture_threshold:
        radius_alignment = -abs(current_dist - desired_radius) / max(
            radius_tolerance, desired_radius
        )
    else:
        radius_alignment = 0.0

    components['distance_reward'] = w_approach * distance_reward
    components['radius_alignment_reward'] = w_radius_uniformity * radius_alignment

    # -----------------------
    # 2. 包围圈几何形状（半径均匀 + 角度均匀 + 中心对齐）
    # -----------------------
    adv_positions = agent_positions[adversary_indices]
    prey_position = prey_pos

    rel_vectors = adv_positions - prey_position  # shape (3, 2)
    radii = np.linalg.norm(rel_vectors, axis=1)

    # 半径均匀性：标准差越小越好
    radius_std = np.std(radii)
    radius_uniform_reward = -radius_std / max(desired_radius, 1e-6)

    # 角度均匀性：三个追捕者相对于猎物的极角
    angles = np.arctan2(rel_vectors[:, 1], rel_vectors[:, 0])
    angles_sorted = np.sort(angles)
    angle_diffs = np.diff(angles_sorted)
    # 闭合环上的最后一个差值
    last_diff = (angles_sorted[0] + 2.0 * np.pi) - angles_sorted[-1]
    angle_diffs = np.concatenate([angle_diffs, [last_diff]])

    # 目标角度间隔为 2π/3，惩罚偏离
    angle_diff_deviation = np.std(angle_diffs - angle_target_step)
    angle_uniform_reward = -angle_diff_deviation / np.pi

    # 猎物相对于追捕者几何中心的偏差
    adv_center = np.mean(adv_positions, axis=0)
    centering_error = np.linalg.norm(adv_center - prey_position)
    centering_reward = -centering_error / world_size

    components['radius_uniform_reward'] = w_radius_uniformity * radius_uniform_reward
    components['angle_uniform_reward'] = w_angle_uniformity * angle_uniform_reward
    components['centering_reward'] = w_centering * centering_reward

    # -----------------------
    # 3. 防碰撞与最小安全距离
    # -----------------------
    collision_penalty = 0.0
    min_separation_penalty = 0.0

    # 追捕者之间的碰撞检测和安全距离约束
    for i in adversary_indices:
        for j in adversary_indices:
            if i >= j:
                continue
            dist_ij = inter_agent_distances[i, j]
            # 碰撞阈值
            collision_threshold = 2.0 * adv_size
            safe_threshold = collision_threshold + collision_margin

            if dist_ij < collision_threshold:
                collision_penalty += 1.0
            elif dist_ij < safe_threshold:
                # 在安全边界附近轻微惩罚，以鼓励保持一定间隔
                min_separation_penalty += (safe_threshold - dist_ij) / safe_threshold

    components['collision_penalty'] = w_collision * collision_penalty
    components['min_separation_penalty'] = w_min_separation * min_separation_penalty

    # -----------------------
    # 4. 围捕完成奖励（多追捕者共同形成稳定包围圈）
    # -----------------------
    # 简单几何判定：
    # 1) 所有追捕者在 capture_threshold 附近
    # 2) 半径和角度均匀性良好
    # 3) 追捕者之间没有碰撞
    # 4) 猎物在追捕者几何中心附近
    all_in_capture_band = np.all(
        np.logical_and(
            radii > desired_radius - 2 * radius_tolerance,
            radii < desired_radius + 2 * radius_tolerance,
        )
    )
    good_radius_uniform = radius_std < radius_tolerance
    good_angle_uniform = angle_diff_deviation < (np.pi / 12.0)  # ~15 度
    no_collisions = collision_penalty == 0.0
    good_centering = centering_error < (desired_radius / 2.0)

    if (
        all_in_capture_band
        and good_radius_uniform
        and good_angle_uniform
        and no_collisions
        and good_centering
    ):
        capture_reward = w_capture_bonus
    else:
        capture_reward = 0.0

    components['capture_reward'] = capture_reward

    # -----------------------
    # 5. 时间损耗惩罚
    # -----------------------
    components['time_penalty'] = time_penalty

    total_reward = float(sum(components.values()))
    return total_reward, components
```

# 客观诊断反馈
1. 分量贡献度  
- 完全为0：min_separation_penalty（均值与方差皆为0），collision_penalty基本不起作用（均值≈0）。  
- 方差极大：boundary_penalty std=2.18、escape_reward std=0.79、radius_uniform_reward std=0.40，说明外圈/逃跑相关反馈强烈波动，容易引发策略震荡。  
- 主导作用：capture_penalty恒为-10，对失败局面强压制；distance/escape/boundary/radius_uniform是主要 shaping 源，但普遍偏负，鼓励较保守而非积极包抄。

2. 任务瓶颈  
- 成功率≈48%，平均捕获时间81.5步，说明能捕到但效率低，且接近“半随机”水平。  
- 明显迹象：强负的escape_reward与boundary_penalty使策略可能偏向“不过分接近、避免极端姿态”，容易停在中等距离徘徊，缺乏果断收缩，疑似陷在“安全试探而非积极围捕”的局部最优。

3. 协同缺陷  
- encirclement_angle_std≈1.86、formation_quality仅0.27，说明包围形态很松散，角度分布不均，包围圈难以闭合。  
- avg_distance_to_prey=1.51、min_agent_distance≈0.95，群体既未形成紧密圈，也无明显扎堆碰撞（collision基本为0）。  
- radius_uniform_reward与radius_alignment_reward偏负，暗示部分个体半径脱队、未按队形收缩，有“个别人乱跑/搭便车”倾向，整体缺乏同步推进的协同收缩能力。

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
    # 非追捕者零奖励
    if not global_state['is_adversary']:
        return 0.0, {}

    components = {}

    # -----------------------
    # 硬编码物理与任务常量
    # -----------------------
    world_size = 2.5
    capture_threshold = 0.5

    adv_size = 0.075
    prey_size = 0.050

    collision_margin = 0.01
    desired_radius = capture_threshold
    time_penalty = -0.005

    # 角度与半径均匀性系数
    radius_tolerance = 0.05
    angle_target_step = 2.0 * np.pi / 3.0

    # 奖励权重（根据诊断进行平衡微调）
    w_approach = 1.2
    w_capture_bonus = 12.0
    w_radius_uniformity = 0.8
    w_angle_uniformity = 1.2
    w_centering = 0.7
    w_collision = -4.0
    w_min_separation = -0.1

    # -----------------------
    # 提取全局状态
    # -----------------------
    agent_positions = global_state['agent_positions']
    prey_pos = global_state['prey_position']
    distances_to_prey = global_state['distances_to_prey']
    inter_agent_distances = global_state['inter_agent_distances']

    # world.agents 顺序假设：3 adversaries + 1 prey
    # 找到当前追捕者索引
    all_agents = world.agents
    adversary_indices = [i for i, a in enumerate(all_agents) if a.adversary]
    prey_indices = [i for i, a in enumerate(all_agents) if not a.adversary]

    if len(adversary_indices) != 3 or len(prey_indices) != 1:
        # 结构异常时保守返回时间惩罚
        components['time_penalty'] = time_penalty
        return sum(components.values()), components

    prey_index = prey_indices[0]

    try:
        current_agent_index = [i for i, a in enumerate(all_agents)
                               if a.name == agent_name][0]
    except (IndexError, AttributeError):
        # 回退：按 adversary 顺序映射，保证索引有效
        # 这里假设 agent_name 为 "adversary_X"
        try:
            idx = int(str(agent_name).split('_')[-1])
            current_agent_index = adversary_indices[idx]
        except Exception:
            current_agent_index = adversary_indices[0]

    # 当前智能体在 adversary 列表中的索引（用于 distances_to_prey）
    try:
        adv_local_index = adversary_indices.index(current_agent_index)
    except ValueError:
        # 若不在 adversary_indices 中，视为无效追捕者
        components['time_penalty'] = time_penalty
        return sum(components.values()), components

    # -----------------------
    # 1. 接近目标的距离奖励
    # -----------------------
    # 使用距离的负值作为基础接近奖励，并在 capture_threshold 内鼓励靠近 desired_radius
    current_dist = distances_to_prey[adv_local_index]

    # 距离引导：越接近越好
    distance_reward = -current_dist / world_size

    # 在 capture_threshold 内，对接近 desired_radius 的行为给额外奖励
    if current_dist < capture_threshold:
        radius_alignment = -abs(current_dist - desired_radius) / max(
            radius_tolerance, desired_radius
        )
    else:
        radius_alignment = 0.0

    components['distance_reward'] = w_approach * distance_reward
    components['radius_alignment_reward'] = 0.5 * w_radius_uniformity * radius_alignment

    # -----------------------
    # 2. 包围圈几何形状（半径均匀 + 角度均匀 + 中心对齐）
    # -----------------------
    adv_positions = agent_positions[adversary_indices]
    prey_position = prey_pos

    rel_vectors = adv_positions - prey_position  # shape (3, 2)
    radii = np.linalg.norm(rel_vectors, axis=1)

    # 半径均匀性：标准差越小越好
    radius_std = np.std(radii)
    radius_uniform_reward = -radius_std / max(desired_radius, 1e-6)

    # 角度均匀性：三个追捕者相对于猎物的极角
    angles = np.arctan2(rel_vectors[:, 1], rel_vectors[:, 0])
    angles_sorted = np.sort(angles)
    angle_diffs = np.diff(angles_sorted)
    # 闭合环上的最后一个差值
    last_diff = (angles_sorted[0] + 2.0 * np.pi) - angles_sorted[-1]
    angle_diffs = np.concatenate([angle_diffs, [last_diff]])

    # 目标角度间隔为 2π/3，惩罚偏离
    angle_diff_deviation = np.std(angle_diffs - angle_target_step)
    angle_uniform_reward = -angle_diff_deviation / np.pi

    # 猎物相对于追捕者几何中心的偏差
    adv_center = np.mean(adv_positions, axis=0)
    centering_error = np.linalg.norm(adv_center - prey_position)
    centering_reward = -centering_error / world_size

    components['radius_uniform_reward'] = w_radius_uniformity * radius_uniform_reward
    components['angle_uniform_reward'] = w_angle_uniformity * angle_uniform_reward
    components['centering_reward'] = w_centering * centering_reward

    # -----------------------
    # 3. 防碰撞与最小安全距离
    # -----------------------
    collision_penalty = 0.0
    min_separation_penalty = 0.0

    # 追捕者之间的碰撞检测和安全距离约束
    for i in adversary_indices:
        for j in adversary_indices:
            if i >= j:
                continue
            dist_ij = inter_agent_distances[i, j]
            # 碰撞阈值
            collision_threshold = 2.0 * adv_size
            safe_threshold = collision_threshold + collision_margin

            if dist_ij < collision_threshold:
                collision_penalty += 1.0
            elif dist_ij < safe_threshold:
                # 在安全边界附近轻微惩罚，以鼓励保持一定间隔
                min_separation_penalty += (safe_threshold - dist_ij) / safe_threshold

    components['collision_penalty'] = w_collision * collision_penalty
    components['min_separation_penalty'] = w_min_separation * min_separation_penalty

    # -----------------------
    # 4. 围捕完成奖励（多追捕者共同形成稳定包围圈）
    # -----------------------
    # 简单几何判定：
    # 1) 所有追捕者在 capture_threshold 附近
    # 2) 半径和角度均匀性良好
    # 3) 追捕者之间没有碰撞
    # 4) 猎物在追捕者几何中心附近
    all_in_capture_band = np.all(
        np.logical_and(
            radii > desired_radius - 2 * radius_tolerance,
            radii < desired_radius + 2 * radius_tolerance,
        )
    )
    good_radius_uniform = radius_std < radius_tolerance
    good_angle_uniform = angle_diff_deviation < (np.pi / 12.0)  # ~15 度
    no_collisions = collision_penalty == 0.0
    good_centering = centering_error < (desired_radius / 2.0)

    if (
        all_in_capture_band
        and good_radius_uniform
        and good_angle_uniform
        and no_collisions
        and good_centering
    ):
        capture_reward = w_capture_bonus
    else:
        capture_reward = 0.0

    components['capture_reward'] = capture_reward

    # -----------------------
    # 5. 时间损耗惩罚
    # -----------------------
    components['time_penalty'] = time_penalty

    total_reward = float(sum(components.values()))
    return total_reward, components
```
```
