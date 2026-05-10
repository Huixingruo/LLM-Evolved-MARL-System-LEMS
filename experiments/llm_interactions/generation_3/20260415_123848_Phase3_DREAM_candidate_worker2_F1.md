# LLM Interaction Log

- **Generation**: 3
- **Phase**: Phase3_DREAM
- **Candidate Info**: worker2_F1
- **Timestamp**: 20260415_123848

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
    if not global_state["is_adversary"]:
        return 0.0, {}

    components = {}

    # ------------------------
    # 物理与任务常量（硬编码）
    # ------------------------
    adv_size = 0.075
    prey_size = 0.050
    world_size = 2.5
    capture_threshold = 0.5

    # 全局势场权重（可调）
    w_radial_shrink = 2.0        # 全局环形收缩势场
    w_gap_closure = 3.0          # 角度间隙填补
    w_alignment = 0.5            # 与理想方向对齐
    w_capture_dense = 10.0       # 紧密多体捕获奖励
    w_team_focus = 1.5           # 全局平均半径缩减
    w_centrality = 1.0           # 团队质心靠近猎物
    w_time_pressure = -0.01      # 时间压力（线性）

    # 安全与稳定
    w_collision_hard = -2.0
    w_near_collision_soft = -0.5
    safe_distance_factor = 1.3

    # 解析全局状态
    agent_positions = np.asarray(global_state["agent_positions"], dtype=float)
    agent_velocities = np.asarray(global_state["agent_velocities"], dtype=float)
    prey_pos = np.asarray(global_state["prey_position"], dtype=float)
    prey_vel = np.asarray(global_state["prey_velocity"], dtype=float)
    inter_agent_distances = np.asarray(
        global_state["inter_agent_distances"], dtype=float
    )

    num_agents = agent_positions.shape[0]

    # 找到当前 agent 的索引
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

    # adversary 与 prey 索引
    adversary_indices = [
        idx for idx, ag in enumerate(world.agents) if getattr(ag, "adversary", False)
    ]
    prey_index = None
    for idx, ag in enumerate(world.agents):
        if not getattr(ag, "adversary", False):
            prey_index = idx
            break

    if prey_index is None or len(adversary_indices) == 0:
        return 0.0, {}

    # 本追捕者在 adversary 列表中的局部索引
    try:
        adv_local_index = adversary_indices.index(agent_index)
    except ValueError:
        adv_local_index = 0

    adv_positions = agent_positions[adversary_indices]
    adv_vels = agent_velocities[adversary_indices]

    # ------------------------
    # 极坐标视角：相对猎物的半径与角度
    # ------------------------
    rel_vecs = adv_positions - prey_pos[None, :]
    radii = np.linalg.norm(rel_vecs, axis=1) + 1e-8
    angles = np.arctan2(rel_vecs[:, 1], rel_vecs[:, 0])

    # 当前 agent 极坐标
    r_self = radii[adv_local_index]
    theta_self = angles[adv_local_index]
    vel_self = adv_vels[adv_local_index]

    # ------------------------
    # 1. 全局收缩势场（环形收缩 + 团队平均半径）
    #    目标：所有追捕者逐步将半径缩小到 capture_threshold 附近
    # ------------------------
    # 单体势场：鼓励向捕获环 r = capture_threshold 靠拢
    # 采用分段势场：外层强收缩，内层温和
    if r_self > capture_threshold:
        radial_potential = -(r_self - capture_threshold) / world_size
    else:
        radial_potential = -0.3 * (capture_threshold - r_self) / world_size

    components["radial_shrink_self"] = w_radial_shrink * radial_potential

    # 团队平均半径势场
    mean_radius = float(np.mean(radii))
    team_potential = -(mean_radius - capture_threshold) / world_size
    components["radial_shrink_team"] = w_team_focus * team_potential

    # ------------------------
    # 2. 角度势场：填补包围角间隙，而非固定均匀角度
    #    思路：每个追捕者感知局部最大角间隙，并被吸引去填补
    # ------------------------
    k = len(adversary_indices)
    angle_gap_reward = 0.0
    if k >= 2:
        # 排序所有追捕者角度
        sort_idx = np.argsort(angles)
        angles_sorted = angles[sort_idx]

        # 计算环上的角度间隙
        diffs = np.diff(angles_sorted)
        last_gap = (angles_sorted[0] + 2.0 * np.pi) - angles_sorted[-1]
        gaps = np.concatenate([diffs, [last_gap]])
        max_gap_idx = int(np.argmax(gaps))
        max_gap = gaps[max_gap_idx]

        # 最大间隙中心角
        start_angle = angles_sorted[max_gap_idx]
        gap_center_angle = start_angle + max_gap / 2.0
        # 映射到 [-pi, pi]
        gap_center_angle = (gap_center_angle + np.pi) % (2.0 * np.pi) - np.pi

        # 当前 agent 与 gap center 的角度差（在环上最近距离）
        d_theta = theta_self - gap_center_angle
        d_theta = (d_theta + np.pi) % (2.0 * np.pi) - np.pi

        # 奖励靠近 gap center 的行为（只对属于最大间隙两端附近的 agent 有明显意义）
        # 使用 cos 形势场：cos(0)=1, cos(pi)= -1
        angle_alignment = np.cos(d_theta)
        # 只在 gap 比较大时施加势场
        gap_scale = max_gap / np.pi  # gap ~ pi 时该项接近 1
        angle_gap_reward = w_gap_closure * angle_alignment * gap_scale

    components["angle_gap_closure"] = angle_gap_reward

    # ------------------------
    # 3. 速度对齐势场：沿理想方向（指向 gap center 或径向内收）
    # ------------------------
    # 理想方向：在捕获圈外时以径向内收为主，在圈内时以角度填补为主
    ideal_dir = np.zeros(2, dtype=float)
    if r_self > capture_threshold * 1.2:
        # 更远时以朝向猎物的径向收缩为主
        ideal_dir = -rel_vecs[adv_local_index] / r_self
    else:
        # 接近捕获圈时以朝向 gap center 的环向方向为主
        gap_tangent = np.array(
            [-np.sin(theta_self), np.cos(theta_self)], dtype=float
        )
        # 确定 gap center 相对当前角度是在正向还是反向
        # d_theta 已在 [-pi, pi]
        if "d_theta" in locals():
            direction_sign = np.sign(-d_theta) if abs(d_theta) > 1e-3 else 0.0
            ideal_dir = gap_tangent * direction_sign
        else:
            ideal_dir = -rel_vecs[adv_local_index] / r_self

    speed_self = np.linalg.norm(vel_self) + 1e-8
    if speed_self > 1e-3:
        dir_self = vel_self / speed_self
        alignment = float(np.dot(dir_self, ideal_dir))
        components["velocity_alignment"] = w_alignment * alignment
    else:
        components["velocity_alignment"] = 0.0

    # ------------------------
    # 4. 多体捕获势场：当多名追捕者同时进入小半径且包围较密时给强奖励（密集但非完全稀疏）
    # ------------------------
    # 定义一个“捕获环带”，半径 <= capture_threshold
    inside_mask = radii <= capture_threshold
    num_inside = int(np.sum(inside_mask))

    dense_capture_reward = 0.0
    if num_inside >= 2:
        # 环带内的平均半径越小、数量越多、角度分布越均匀，奖励越大
        inner_radii = radii[inside_mask]
        inner_angles = angles[inside_mask]

        # 半径缩紧奖励
        tightness = 1.0 - np.mean(inner_radii) / (capture_threshold + 1e-8)
        tightness = max(0.0, tightness)

        # 角度覆盖度（最小间隙越小越好，但不要求完美均匀）
        inner_sorted = np.sort(inner_angles)
        inner_diffs = np.diff(inner_sorted)
        last_inner_gap = (inner_sorted[0] + 2.0 * np.pi) - inner_sorted[-1]
        inner_gaps = np.concatenate([inner_diffs, [last_inner_gap]])
        max_inner_gap = float(np.max(inner_gaps))
        # 覆盖度：1 - max_gap / (2π)
        coverage = 1.0 - max_inner_gap / (2.0 * np.pi)
        coverage = np.clip(coverage, 0.0, 1.0)

        # 数量因子
        count_factor = (num_inside - 1) / max(len(adversary_indices) - 1, 1)

        dense_capture_score = tightness * coverage * count_factor
        dense_capture_reward = w_capture_dense * dense_capture_score

    components["dense_capture"] = dense_capture_reward

    # ------------------------
    # 5. 团队几何中心势场：团队质心靠近猎物
    # ------------------------
    centroid = np.mean(adv_positions, axis=0)
    centroid_dist = np.linalg.norm(centroid - prey_pos)
    centroid_potential = -centroid_dist / world_size
    components["centroid_centrality"] = w_centrality * centroid_potential

    # ------------------------
    # 6. 安全势场：防止追捕者之间与猎物发生硬碰撞，同时允许适度接近
    # ------------------------
    collision_penalty = 0.0
    near_collision_penalty = 0.0

    # adversary-adversary
    for j in adversary_indices:
        if j == agent_index:
            continue
        d_ij = inter_agent_distances[agent_index, j]
        min_dist = 2.0 * adv_size
        safe_dist = safe_distance_factor * min_dist

        if d_ij < min_dist:
            collision_penalty += w_collision_hard
        elif d_ij < safe_dist:
            ratio = (safe_dist - d_ij) / (safe_dist - min_dist + 1e-8)
            near_collision_penalty += w_near_collision_soft * ratio

    # adversary-prey
    d_prey = inter_agent_distances[agent_index, prey_index]
    min_dist_ap = adv_size + prey_size
    if d_prey < min_dist_ap:
        collision_penalty += 0.5 * w_collision_hard

    components["collision_penalty"] = collision_penalty
    components["near_collision_penalty"] = near_collision_penalty

    # ------------------------
    # 7. 时间压力势场：与 episode 长度解耦的线性时间惩罚
    # ------------------------
    components["time_pressure"] = w_time_pressure

    total_reward = float(sum(components.values()))
    return total_reward, components
```

# 客观诊断反馈
[病理诊断]
1. 整体表现概览  
- Fitness=0.3277，对应成功率32.27%，平均捕获时间88.5步，说明策略能在部分局面完成围捕，但效率低且不稳定。  
- 捕获失败惩罚固定为 -10（capture_penalty mean=-10），在成功率仅32%的情况下，失败样本对总回报有强烈负向主导作用。

2. 各奖励分量主导性分析  
按“对策略形成的长期压力”和“数值规模”排序：

(1) capture_penalty（核心主导，极强约束）  
- mean=-10.0，std=0：  
  - 每个episode结局几乎必然触发一个固定的 -10 惩罚（大概率为“未能及时捕获”或“逃脱”）。  
  - 在成功率只有 32% 时，失败样本主导了整体梯度方向，容易形成“避免极端失败”而非“积极寻求成功”的保守策略。  
  - 这一分量目前是绝对主导项，且是**纯负向终局信号**，极可能压制了中途的正向引导（如 dense_capture、velocity_alignment）。

(2) escape_reward（次级全程负压，可能设计为“被猎物拉远/逃脱趋势”）  
- mean=-1.0970, std=0.7506：  
  - 一直为负，且量级仅次于 capture_penalty（按episode平均算），说明在大部分时间里，策略都被判定为“在向失败状态演化”。  
  - 可能的解释：  
    - 若设计为“猎物逃离成功”的过程性惩罚，则当前策略经常让猎物获得空间优势；  
    - 若设计方向本意是鼓励压缩猎物逃逸空间，但公式可能过于偏向惩罚端（例如距离稍大就给予强惩罚），导致训练中长期遭受负梯度。  
  - 目前看，该分量**未起到正向引导收缩空间的作用，而是几乎全程压制**。

(3) radial_shrink_self / radial_shrink_team（围捕收缩相关，但方向偏负）  
- radial_shrink_self: mean=-0.6696, std=0.7583  
- radial_shrink_team: mean=-0.4817, std=0.4817  
  - 两个都明显为负，说明在当前策略下，设计的“半径收缩指标”经常落在惩罚区间：  
    - 要么队形经常向外扩散（而非向猎物收缩）；  
    - 要么奖励公式对“未能显著收缩”给了较重惩罚。  
  - 与 escape_reward 一起形成**多重重叠的负向收缩压力**：  
    - 但从协同指标看：avg_distance_to_prey=1.3044、min_agent_distance=1.0010 并不特别小，说明并没有实质形成紧凑包围。  
    - 当前策略可能倾向于“谨慎靠近、保持距离、不真正关门”，以避免潜在风险（失败惩罚过大），反而长期陷在“既没围好也没彻底放弃”的中间态，被持续扣分。

(4) centroid_centrality（位置结构相关，但总体负向）  
- mean=-0.3261, std=0.3140  
  - 说明当前队形相对于理想“环绕猎物”的中心配置偏差较大，且长期偏负。  
  - 与 formation_quality=0.3118（偏低）一致：队形质量不足，多个个体空间分布不理想。  
  - 该分量有一定梯度信息，但当前策略没有很好地改善这一指标，可能是被更强烈的终局惩罚（capture_penalty）和逃逸惩罚（escape_reward）压制。

(5) angle_gap_closure（角度包围进展，明显负向且高方差）  
- mean=-1.8928, std=1.8304  
  - 均值强烈为负，方差也大，说明：  
    - 多数时间角度包围指标在“开口大”的区间（被惩罚）；  
    - 少数时间有接近或改善包围，但不足以让整体均值转正。  
  - 这对“形成连续封锁弧线”极为关键，但当前公式很可能只在极窄的高质量包围区间给奖励，而大部分状态都是显著惩罚 → 导致策略**学到“不要强行去关门”，因为一旦角度微调失败就掉入大惩罚区**。  
  - 这是一个明显“设计目标正确但公式/区间设置不当”的典型 F2 对象。

(6) velocity_alignment（方向协同，少数正向信号）  
- mean=0.1740, std=0.3310  
  - 为少数正均值的密集奖励之一，说明：  
    - 策略在朝向/速度方向上有一定程度的同步或对猎物预测，能提供持续的少量正反馈。  
  - 然而量级远小于 capture_penalty 和 escape_reward 等负项 → 正向引导被严重稀释。  
  - 本身设计方向合理，暂不必重构公式，更建议通过 F3 提升权重，放大其正引导。

(7) dense_capture（邻近包围密集度奖励，偏弱但方向正确）  
- mean=0.0646, std=0.1780  
  - 为正但较小，说明偶尔能进入“半围捕状态”，但频率不高，且奖励不足以导向稳定策略。  
  - 这是任务核心目标（“包围并捕获”）的近端信号之一，当前权重显然偏低，容易被终局失败惩罚淹没。  
  - 倾向于通过 F3 上调其权重；如果未来多代仍然偏低，再考虑用 F2 改为更具阶段性的“里程碑奖励”。

(8) time_pressure（时间约束，固定微小惩罚）  
- mean=-0.0100, std=0.0000  
  - 近似为每步固定小惩罚，对不同策略区分度极低，几乎是一个“常数偏移”，难以对学习产生结构性导向。  
  - 在平均捕获时间88.5步情况下，总惩罚约 -0.9，不算小，但因几乎无方差，主要作用是“鼓励更快结束episode”，而非引导具体行为。  
  - 若想用其控制节奏，建议将其从纯线性/常数改成随阶段/距离变化的非线性函数（F2 候选），提供更有区分度的信号。

(9) near_collision_penalty / collision_penalty（防碰撞机制）  
- collision_penalty: mean=0.0000, std=0  
- near_collision_penalty: mean=-0.0000, std=0.0006  
  - 基本不发生碰撞或近碰，或权重极低、触发条件稀疏。  
  - 当前策略并未在“过度聚拢”方向上过度探索（formation_quality 不高、avg_distance_to_prey 也不算极小），因此这一机制暂时处于基本失效状态。  
  - 不构成当前性能瓶颈，不是首要优化对象。

(10) boundary_penalty（边界相关）  
- mean=-0.2889, std=1.4779  
  - 有显著负均值与高方差，说明：  
    - 环境边界被频繁触及或靠近；  
    - 部分episode中边界影响极大（惩罚大），导致行为在边界附近高度不稳定。  
  - 若环境是有限场景，这表明当前策略经常把猎物/自己逼近边缘而陷入“边界-捕获”之间的冲突：既要追，又怕越界。  
  - 该分量可能在某些局面与“合理包围”相冲突（例如理想围捕位置靠近边界却被惩罚），建议重构为更柔和、方向性更强的势场型约束（F2）。

3. 协同指标与行为模式诊断  
- encirclement_angle_std: mean=1.3492  
  - 角度分布的标准差较大，说明个体围绕猎物的角度分布不均衡，有明显“侧翼空档”。  
  - 与 angle_gap_closure 的大幅负值相呼应：目前策略缺乏系统性的“封口”和“堵截”行为，更像是几名追踪者从近似同一侧施压。

- min_agent_distance: mean=1.0010  
  - 个体之间保持中等距离，没有过度挤在一起，也没有形成紧密包围圈。  
  - 在缺乏强正向的包围密集奖励和强协同结构奖励时，策略更倾向于“安全距离并行追踪”，不主动压缩队形。  

- avg_distance_to_prey: mean=1.3044  
  - 与稠密捕获奖励偏弱相匹配：距离既不算近，也不足以真正形成压迫态势。  
  - 再结合成功率32%，推断：  
    - 成功发生在“偶然合围”或“边界帮助”情境，而非稳定可复现的围捕策略。

- formation_quality: mean=0.3118  
  - 队形质量偏低，说明多智能体间缺少显式协同引导。  
  - 当前奖励中虽有角度、半径、中心性等设计，但大都呈负值主导，且互相叠加，导致策略更多在“避免高惩罚区域”而非“朝一个清晰的协同形态去优化”。

4. 病理总结  
- 主要问题是**“负向终局惩罚 + 多个持续负向过程性指标”共同主导**，正向协同与围捕奖励（velocity_alignment、dense_capture）太弱，只能提供零星修补。  
- 核心围捕相关分量（angle_gap_closure、radial_shrink_*、centroid_centrality）目标方向是对的，但**公式区间/刻画方式可能过于苛刻或线性，导致大部分状态落在惩罚区**，形成“学不到路径，只知道哪里不能去”的局面。  
- 协同结构（formation_quality、encirclement_angle_std）显示：  
  - 个体在同侧追逐、缺乏多角度合围；  
  - 没有形成稳定的“半环 + 封锁口”的分工协同。  
- 目前范式**并非完全错误**：有速度对齐、角度包围、半径收缩、边界约束等关键要素，属于“设计要素齐全但配比与数学形态失衡”的状态；还不需要立即 L1 推翻重写，但必须针对若干分量做结构性重构（F2）与权重平衡（F3），同时可以增加一些更直接的“团队协同结构奖励”（F1）以补足缺失的高层协同引导。

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
    if not global_state["is_adversary"]:
        return 0.0, {}

    components = {}

    # ------------------------
    # 物理与任务常量（硬编码）
    # ------------------------
    adv_size = 0.075
    prey_size = 0.050
    world_size = 2.5
    capture_threshold = 0.5

    # 全局势场权重（可调）
    w_radial_shrink = 2.0        # 全局环形收缩势场
    w_gap_closure = 3.0          # 角度间隙填补
    w_alignment = 0.5            # 与理想方向对齐
    w_capture_dense = 10.0       # 紧密多体捕获奖励
    w_team_focus = 1.5           # 全局平均半径缩减
    w_centrality = 1.0           # 团队质心靠近猎物
    w_time_pressure = -0.01      # 时间压力（线性）

    # 安全与稳定
    w_collision_hard = -2.0
    w_near_collision_soft = -0.5
    safe_distance_factor = 1.3

    # ------------------------
    # 新增：显式协同结构奖励（多扇面包围度）
    # ------------------------
    w_sector_coverage = 2.0
    sector_min_adv = 1
    n_sectors = 6

    # 解析全局状态
    agent_positions = np.asarray(global_state["agent_positions"], dtype=float)
    agent_velocities = np.asarray(global_state["agent_velocities"], dtype=float)
    prey_pos = np.asarray(global_state["prey_position"], dtype=float)
    prey_vel = np.asarray(global_state["prey_velocity"], dtype=float)
    inter_agent_distances = np.asarray(
        global_state["inter_agent_distances"], dtype=float
    )

    num_agents = agent_positions.shape[0]

    # 找到当前 agent 的索引
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

    # adversary 与 prey 索引
    adversary_indices = [
        idx for idx, ag in enumerate(world.agents) if getattr(ag, "adversary", False)
    ]
    prey_index = None
    for idx, ag in enumerate(world.agents):
        if not getattr(ag, "adversary", False):
            prey_index = idx
            break

    if prey_index is None or len(adversary_indices) == 0:
        return 0.0, {}

    # 本追捕者在 adversary 列表中的局部索引
    try:
        adv_local_index = adversary_indices.index(agent_index)
    except ValueError:
        adv_local_index = 0

    adv_positions = agent_positions[adversary_indices]
    adv_vels = agent_velocities[adversary_indices]

    # ------------------------
    # 极坐标视角：相对猎物的半径与角度
    # ------------------------
    rel_vecs = adv_positions - prey_pos[None, :]
    radii = np.linalg.norm(rel_vecs, axis=1) + 1e-8
    angles = np.arctan2(rel_vecs[:, 1], rel_vecs[:, 0])

    # 当前 agent 极坐标
    r_self = radii[adv_local_index]
    theta_self = angles[adv_local_index]
    vel_self = adv_vels[adv_local_index]

    # ------------------------
    # 1. 全局收缩势场（环形收缩 + 团队平均半径）
    #    目标：所有追捕者逐步将半径缩小到 capture_threshold 附近
    # ------------------------
    if r_self > capture_threshold:
        radial_potential = -(r_self - capture_threshold) / world_size
    else:
        radial_potential = -0.3 * (capture_threshold - r_self) / world_size

    components["radial_shrink_self"] = w_radial_shrink * radial_potential

    mean_radius = float(np.mean(radii))
    team_potential = -(mean_radius - capture_threshold) / world_size
    components["radial_shrink_team"] = w_team_focus * team_potential

    # ------------------------
    # 2. 角度势场：填补包围角间隙，而非固定均匀角度
    # ------------------------
    k = len(adversary_indices)
    angle_gap_reward = 0.0
    if k >= 2:
        sort_idx = np.argsort(angles)
        angles_sorted = angles[sort_idx]

        diffs = np.diff(angles_sorted)
        last_gap = (angles_sorted[0] + 2.0 * np.pi) - angles_sorted[-1]
        gaps = np.concatenate([diffs, [last_gap]])
        max_gap_idx = int(np.argmax(gaps))
        max_gap = gaps[max_gap_idx]

        start_angle = angles_sorted[max_gap_idx]
        gap_center_angle = start_angle + max_gap / 2.0
        gap_center_angle = (gap_center_angle + np.pi) % (2.0 * np.pi) - np.pi

        d_theta = theta_self - gap_center_angle
        d_theta = (d_theta + np.pi) % (2.0 * np.pi) - np.pi

        angle_alignment = np.cos(d_theta)
        gap_scale = max_gap / np.pi
        angle_gap_reward = w_gap_closure * angle_alignment * gap_scale

    components["angle_gap_closure"] = angle_gap_reward

    # ------------------------
    # 3. 速度对齐势场：沿理想方向（指向 gap center 或径向内收）
    # ------------------------
    ideal_dir = np.zeros(2, dtype=float)
    if r_self > capture_threshold * 1.2:
        ideal_dir = -rel_vecs[adv_local_index] / r_self
    else:
        gap_tangent = np.array(
            [-np.sin(theta_self), np.cos(theta_self)], dtype=float
        )
        if "d_theta" in locals():
            direction_sign = np.sign(-d_theta) if abs(d_theta) > 1e-3 else 0.0
            ideal_dir = gap_tangent * direction_sign
        else:
            ideal_dir = -rel_vecs[adv_local_index] / r_self

    speed_self = np.linalg.norm(vel_self) + 1e-8
    if speed_self > 1e-3:
        dir_self = vel_self / speed_self
        alignment = float(np.dot(dir_self, ideal_dir))
        components["velocity_alignment"] = w_alignment * alignment
    else:
        components["velocity_alignment"] = 0.0

    # ------------------------
    # 4. 多体捕获势场：当多名追捕者同时进入小半径且包围较密时给强奖励
    # ------------------------
    inside_mask = radii <= capture_threshold
    num_inside = int(np.sum(inside_mask))

    dense_capture_reward = 0.0
    if num_inside >= 2:
        inner_radii = radii[inside_mask]
        inner_angles = angles[inside_mask]

        tightness = 1.0 - np.mean(inner_radii) / (capture_threshold + 1e-8)
        tightness = max(0.0, tightness)

        inner_sorted = np.sort(inner_angles)
        inner_diffs = np.diff(inner_sorted)
        last_inner_gap = (inner_sorted[0] + 2.0 * np.pi) - inner_sorted[-1]
        inner_gaps = np.concatenate([inner_diffs, [last_inner_gap]])
        max_inner_gap = float(np.max(inner_gaps))
        coverage = 1.0 - max_inner_gap / (2.0 * np.pi)
        coverage = np.clip(coverage, 0.0, 1.0)

        count_factor = (num_inside - 1) / max(len(adversary_indices) - 1, 1)

        dense_capture_score = tightness * coverage * count_factor
        dense_capture_reward = w_capture_dense * dense_capture_score

    components["dense_capture"] = dense_capture_reward

    # ------------------------
    # 5. 团队几何中心势场：团队质心靠近猎物
    # ------------------------
    centroid = np.mean(adv_positions, axis=0)
    centroid_dist = np.linalg.norm(centroid - prey_pos)
    centroid_potential = -centroid_dist / world_size
    components["centroid_centrality"] = w_centrality * centroid_potential

    # ------------------------
    # 6. 安全势场：防止追捕者之间与猎物发生硬碰撞
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
            collision_penalty += w_collision_hard
        elif d_ij < safe_dist:
            ratio = (safe_dist - d_ij) / (safe_dist - min_dist + 1e-8)
            near_collision_penalty += w_near_collision_soft * ratio

    d_prey = inter_agent_distances[agent_index, prey_index]
    min_dist_ap = adv_size + prey_size
    if d_prey < min_dist_ap:
        collision_penalty += 0.5 * w_collision_hard

    components["collision_penalty"] = collision_penalty
    components["near_collision_penalty"] = near_collision_penalty

    # ------------------------
    # 7. 时间压力势场
    # ------------------------
    components["time_pressure"] = w_time_pressure

    # ------------------------
    # 新增分量：多扇面角度覆盖协同奖励
    # ------------------------
    sector_reward = 0.0
    if k >= 2:
        two_pi = 2.0 * np.pi
        sector_size = two_pi / float(n_sectors)
        sector_counts = np.zeros(n_sectors, dtype=float)

        # 将所有追捕者的角度映射到 [0, 2π)
        norm_angles = (angles + two_pi) % two_pi
        sector_indices = np.floor(norm_angles / sector_size).astype(int)
        sector_indices = np.clip(sector_indices, 0, n_sectors - 1)

        for s_idx in sector_indices:
            sector_counts[s_idx] += 1.0

        # 统计覆盖度：至少有 sector_min_adv 个追捕者的扇面视为“覆盖”
        covered = (sector_counts >= sector_min_adv).astype(float)
        coverage_ratio = float(np.mean(covered))

        # 在适度半径区间内时才强化该奖励，避免在过远时乱铺开
        radius_gate = np.clip((r_self - capture_threshold * 0.5) /
                              (world_size - capture_threshold * 0.5 + 1e-8),
                              0.0, 1.0)
        # 越靠近猎物（但未特别近）越重视角度协同
        gated_coverage = coverage_ratio * radius_gate

        sector_reward = w_sector_coverage * gated_coverage

    components["sector_coverage"] = sector_reward

    total_reward = float(sum(components.values()))
    return total_reward, components
```
```
