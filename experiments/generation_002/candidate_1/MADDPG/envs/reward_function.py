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