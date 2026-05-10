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
    safe_distance_factor = 1.5
    formation_radius_target = capture_threshold * 0.9
    formation_radius_tolerance = capture_threshold * 0.3
    angle_uniform_target = 2.0 * np.pi / 3.0  # 120°

    # 失效分量重构相关超参数
    # 重构 time_penalty: 非线性时间压力（基于离散 "阶段" 的里程碑式惩罚）
    base_time_penalty = -0.001
    late_stage_extra_penalty = -0.004
    final_stage_extra_penalty = -0.008

    # 重构 capture_bonus: 渐进式 + 终局式成功奖励
    base_capture_bonus = 4.0
    full_capture_bonus = 10.0

    # 重构速度多样性（由“多样性”改为“协同速度一致性”）
    velocity_alignment_weight = 0.15

    # 重构碰撞与近碰权重（更细腻但限制区间内放大）
    collision_penalty_weight = -1.5
    near_collision_max_penalty = -0.6

    # 从 global_state 取出状态
    agent_positions = np.asarray(global_state["agent_positions"], dtype=float)
    agent_velocities = np.asarray(global_state["agent_velocities"], dtype=float)
    prey_pos = np.asarray(global_state["prey_position"], dtype=float)
    prey_vel = np.asarray(global_state["prey_velocity"], dtype=float)
    distances_to_prey = np.asarray(global_state["distances_to_prey"], dtype=float)
    inter_agent_distances = np.asarray(
        global_state["inter_agent_distances"], dtype=float
    )

    num_agents = agent_positions.shape[0]

    # ------------------------
    # 解析 agent 索引与角色
    # ------------------------
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
    # 1. 距离引导：靠近目标 + 围捕完成奖励（capture_bonus 重构）
    # ------------------------
    if len(distances_to_prey) > 0:
        d_self = distances_to_prey[adv_local_index]
        mean_d_adv = float(np.mean(distances_to_prey))

        norm_d_self = d_self / world_size
        norm_d_mean = mean_d_adv / world_size

        distance_reward_self = -0.8 * norm_d_self
        distance_reward_team = -0.2 * norm_d_mean

        # 渐进式成功奖励：当所有追捕者进入不同半径区间时分段奖励
        all_within_loose = bool(
            np.all(distances_to_prey < capture_threshold * 1.5)
        )
        all_within_tight = bool(
            np.all(distances_to_prey < capture_threshold)
        )

        capture_bonus = 0.0
        if all_within_loose:
            # 软捕获阶段：进入较小区域开始给奖励
            tightness_factor = np.clip(
                (capture_threshold * 1.5 - mean_d_adv)
                / (capture_threshold * 1.5 + 1e-8),
                0.0,
                1.0,
            )
            capture_bonus += base_capture_bonus * tightness_factor

        if all_within_tight:
            # 真正捕获：强化终局奖励
            capture_bonus += full_capture_bonus

        components["distance_self"] = distance_reward_self
        components["distance_team"] = distance_reward_team
        components["capture_bonus"] = capture_bonus
    else:
        components["distance_self"] = 0.0
        components["distance_team"] = 0.0
        components["capture_bonus"] = 0.0

    # ------------------------
    # 2. 防碰撞：追捕者-追捕者 与 追捕者-逃跑者（重构）
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
            # 硬碰撞：指数放大惩罚
            penetration = max(min_dist - d_ij, 0.0) / (min_dist + 1e-8)
            collision_penalty += collision_penalty_weight * (
                1.0 + 2.0 * np.exp(5.0 * penetration)
            )
        elif d_ij < safe_dist:
            # 近碰：在安全区内，距离越近惩罚越大，但上界有限
            ratio = (safe_dist - d_ij) / (safe_dist - min_dist + 1e-8)
            smooth_ratio = ratio**2
            near_collision_penalty += near_collision_max_penalty * smooth_ratio

    prey_index = None
    for idx, ag in enumerate(world.agents):
        if not getattr(ag, "adversary", False):
            prey_index = idx
            break
    if prey_index is not None:
        d_prey = inter_agent_distances[agent_index, prey_index]
        min_dist_ap = adv_size + prey_size
        if d_prey < min_dist_ap:
            # 轻微惩罚，带有平滑穿透度
            penetration = max(min_dist_ap - d_prey, 0.0) / (
                min_dist_ap + 1e-8
            )
            collision_penalty += -0.3 * (1.0 + 2.0 * penetration**2)

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
            formation_radius_reward += -0.4 * radius_dev / (capture_threshold + 1e-8)
        formation_radius_reward += -0.2 * var_r / (capture_threshold**2 + 1e-8)

        angles = np.arctan2(vecs[:, 1], vecs[:, 0])
        angles_sorted = np.sort(angles)
        angle_diffs = np.diff(angles_sorted)
        last_gap = (angles_sorted[0] + 2.0 * np.pi) - angles_sorted[-1]
        angle_diffs = np.concatenate([angle_diffs, [last_gap]])
        angle_dev = np.abs(angle_diffs - angle_uniform_target)
        angle_dev_mean = float(np.mean(angle_dev))
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
            center_in_triangle_reward = 3.0

    components["formation_radius"] = formation_radius_reward
    components["formation_angle"] = formation_angle_reward
    components["center_in_triangle"] = center_in_triangle_reward

    # ------------------------
    # 4. 时间效率：重构 time_penalty（分段非线性时间压力）
    # ------------------------
    # 使用 world.step_count 若存在，否则退化为常数惩罚
    step_count = getattr(world, "step_count", None)
    if step_count is None:
        time_penalty = base_time_penalty
    else:
        # 假设最大长度约 100 steps，分三阶段强化惩罚
        normalized_t = np.clip(step_count / 100.0, 0.0, 1.0)
        time_penalty = base_time_penalty

        if normalized_t > 0.33:
            # 中期：适度增加惩罚
            time_penalty += late_stage_extra_penalty * (normalized_t - 0.33) / 0.67
        if normalized_t > 0.66:
            # 末期：进一步增加，形成明显时间压力
            time_penalty += final_stage_extra_penalty * (normalized_t - 0.66) / 0.34

    components["time_penalty"] = time_penalty

    # ------------------------
    # 5. 速度协同（重构 velocity_diversity -> velocity_alignment）
    # ------------------------
    adv_vels = agent_velocities[adversary_indices]
    if adv_vels.shape[0] > 1:
        # 计算速度方向的一致性：越一致奖励越大
        speeds = np.linalg.norm(adv_vels, axis=1, keepdims=True) + 1e-8
        dirs = adv_vels / speeds

        mean_dir = np.mean(dirs, axis=0, keepdims=True)
        mean_dir_norm = np.linalg.norm(mean_dir) + 1e-8
        mean_dir_unit = mean_dir / mean_dir_norm

        cos_sims = np.clip(np.sum(dirs * mean_dir_unit, axis=1), -1.0, 1.0)
        alignment = float(np.mean(cos_sims))
        alignment_reward = velocity_alignment_weight * alignment
        components["velocity_diversity"] = alignment_reward
    else:
        components["velocity_diversity"] = 0.0

    total_reward = float(sum(components.values()))
    return total_reward, components