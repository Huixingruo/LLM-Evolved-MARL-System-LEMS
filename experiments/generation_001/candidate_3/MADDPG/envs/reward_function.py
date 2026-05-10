import numpy as np


def compute_reward(agent_name, observation, global_state, actions, world):
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

    # 重新标定的队形参数（引入“舒适区”和软阈）
    formation_radius_target = capture_threshold * 0.9
    formation_radius_tolerance_inner = capture_threshold * 0.2
    formation_radius_tolerance_outer = capture_threshold * 0.5

    angle_uniform_target = 2.0 * np.pi / 3.0  # 120°
    angle_tolerance = np.pi / 6.0  # 30°

    # 时间惩罚重构：非线性递增，前期探索更宽松
    base_time_penalty = -0.002

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

    prey_index = None
    for idx, ag in enumerate(world.agents):
        if not getattr(ag, "adversary", False):
            prey_index = idx
            break

    # ------------------------
    # 1. 距离引导：靠近目标 + 围捕完成奖励
    #    重构：将线性惩罚改为有“舒适区”的非线性势场
    # ------------------------
    if len(distances_to_prey) > 0:
        d_self = distances_to_prey[adv_local_index]
        mean_d_adv = float(np.mean(distances_to_prey))

        # 自身距离：在 capture_threshold 附近给出平缓的盆形势场
        def _shaped_distance_potential(d, d_target, d_soft):
            if d <= d_target:
                # 过近：轻微惩罚，避免硬撞，但不要过强
                diff = d_target - d
                return -0.1 * (diff / (d_target + 1e-8)) ** 2
            if d <= d_target + d_soft:
                # 舒适下降区：给出主要正向梯度
                diff = d - d_target
                return 1.0 * np.exp(-diff / (d_soft + 1e-8))
            # 远距离衰减，避免巨大负值背景
            diff = d - (d_target + d_soft)
            return 0.5 * np.exp(-diff / (2.0 * world_size))

        distance_reward_self = _shaped_distance_potential(
            d_self, capture_threshold, world_size * 0.4
        )
        distance_reward_team = 0.5 * _shaped_distance_potential(
            mean_d_adv, capture_threshold, world_size * 0.4
        )

        # 捕获成功奖励：柔和提升且不产生极大负背景
        all_within_capture = bool(np.all(distances_to_prey < capture_threshold))
        capture_bonus = 0.0
        if all_within_capture:
            # 使用可叠加但有限的成功奖励
            capture_bonus = 3.0

        components["distance_self"] = distance_reward_self
        components["distance_team"] = distance_reward_team
        components["capture_bonus"] = capture_bonus
    else:
        components["distance_self"] = 0.0
        components["distance_team"] = 0.0
        components["capture_bonus"] = 0.0

    # ------------------------
    # 2. 防碰撞：追捕者-追捕者 与 追捕者-逃跑者
    #    逻辑保留，仅稍作平滑
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
            # 仍然保持硬碰撞惩罚
            diff = min_dist - d_ij
            collision_penalty -= 2.0 * (1.0 + diff / (min_dist + 1e-8))
        elif d_ij < safe_dist:
            ratio = (safe_dist - d_ij) / (safe_dist - min_dist + 1e-8)
            # 使用平方形态，使轻微靠近时惩罚更柔和
            near_collision_penalty -= 0.5 * (ratio**2)

    if prey_index is not None:
        d_prey = inter_agent_distances[agent_index, prey_index]
        min_dist_ap = adv_size + prey_size
        if d_prey < min_dist_ap:
            diff_ap = min_dist_ap - d_prey
            collision_penalty -= 0.5 * (1.0 + diff_ap / (min_dist_ap + 1e-8))

    components["collision_penalty"] = collision_penalty
    components["near_collision_penalty"] = near_collision_penalty

    # ------------------------
    # 3. 包围队形：均匀半径 + 均匀角度
    #    重构：从“长期负惩罚”改为“舒适区+缓和惩罚”的分段形式
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

        # 半径中心偏差：在 [target - inner, target + inner] 视为舒适区
        if abs(mean_r - formation_radius_target) <= formation_radius_tolerance_inner:
            center_term = 0.3
        elif abs(mean_r - formation_radius_target) <= formation_radius_tolerance_outer:
            dev = abs(mean_r - formation_radius_target)
            scale = formation_radius_tolerance_outer - formation_radius_tolerance_inner
            center_term = 0.3 * np.exp(-(dev - formation_radius_tolerance_inner) /
                                       (scale + 1e-8))
        else:
            # 使用缓和的负惩罚，而非大幅线性惩罚
            dev = abs(mean_r - formation_radius_target)
            center_term = -0.3 * (1.0 - np.exp(-dev / (world_size + 1e-8)))

        # 半径方差：期望较小，采用 sqrt+soft 形式
        std_r = np.sqrt(max(var_r, 0.0))
        if std_r <= formation_radius_tolerance_inner * 0.5:
            var_term = 0.2
        else:
            var_term = -0.2 * (1.0 - np.exp(-std_r /
                                            (formation_radius_tolerance_outer + 1e-8)))

        formation_radius_reward = center_term + var_term

        # 角度均匀度重构：容忍区 + 软惩罚
        angles = np.arctan2(vecs[:, 1], vecs[:, 0])
        angles_sorted = np.sort(angles)
        angle_diffs = np.diff(angles_sorted)
        last_gap = (angles_sorted[0] + 2.0 * np.pi) - angles_sorted[-1]
        angle_diffs = np.concatenate([angle_diffs, [last_gap]])
        angle_dev = np.abs(angle_diffs - angle_uniform_target)
        angle_dev_mean = float(np.mean(angle_dev))

        if angle_dev_mean <= angle_tolerance:
            formation_angle_reward = 0.25
        else:
            # 使用指数衰减惩罚，避免长期大负值
            formation_angle_reward = -0.25 * (
                1.0
                - np.exp(-(angle_dev_mean - angle_tolerance) / (np.pi + 1e-8))
            )

        # 判定猎物是否在三角形内部：保持结构，但给予更柔和的奖励
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
            center_in_triangle_reward = 0.8

    components["formation_radius"] = formation_radius_reward
    components["formation_angle"] = formation_angle_reward
    components["center_in_triangle"] = center_in_triangle_reward

    # ------------------------
    # 4. 时间效率：重构为“早期几乎无惩罚，后期非线性加重”
    # ------------------------
    # 尝试从 world 中获取步数，如失败则退化为常数
    try:
        current_step = getattr(world, "step_count", None)
    except Exception:
        current_step = None

    if current_step is None:
        # 无法获取步数时，使用温和常数
        time_penalty = base_time_penalty
    else:
        # 使用 sigmoid 形式，在前 40% 步数惩罚很弱，之后迅速变重
        max_steps = getattr(world, "max_steps", 100)
        progress = float(current_step) / float(max_steps + 1e-8)
        scale = 6.0
        shift = 0.4
        factor = 1.0 / (1.0 + np.exp(-scale * (progress - shift)))
        time_penalty = base_time_penalty * (0.2 + 0.8 * factor)

    components["time_penalty"] = time_penalty

    # ------------------------
    # 5. 平滑性与协同速度：保留结构但略微柔化
    # ------------------------
    adv_vels = agent_velocities[adversary_indices]
    if adv_vels.shape[0] > 1:
        speed_var = float(np.mean(np.var(adv_vels, axis=0)))
        # 使用小权重并做 sqrt 压缩，避免成为噪声源
        components["velocity_diversity"] = 0.05 * np.sqrt(max(speed_var, 0.0))
    else:
        components["velocity_diversity"] = 0.0

    total_reward = float(sum(components.values()))
    return total_reward, components