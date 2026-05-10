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

    # 全局势场权重（保留有效项权重）
    w_radial_shrink = 1.0
    w_gap_closure = 1.2         # ↑ 强化角度闭合正向引导
    w_alignment = 1.5
    w_capture_dense = 25.0
    w_team_focus = 0.8
    w_centrality = 0.7
    w_time_pressure = -0.003

    # 安全与稳定
    w_collision_hard = -2.0
    w_near_collision_soft = -0.5
    safe_distance_factor = 1.3

    # ------------------------
    # 解析全局状态
    # ------------------------
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

    # ========================
    # 1. 重构径向收缩分量
    # ========================
    # 使用 sigmoidal 形状的正向“靠近 capture_threshold”奖励
    # 目标半径区间 [capture_threshold, 2*capture_threshold] 给高奖励，
    # 过远或过近都衰减，避免长期负压。
    ideal_r = capture_threshold * 1.5
    max_r = world_size
    # 归一化半径误差
    radial_error = (r_self - ideal_r) / (max_r - ideal_r + 1e-8)
    # 平滑钟形奖励：r 在 ideal_r 附近时接近 1，远离时衰减至 0
    radial_shrink_self_score = np.exp(-4.0 * radial_error**2)
    components["radial_shrink_self"] = w_radial_shrink * radial_shrink_self_score

    # 团队平均半径：鼓励团队整体处在合理收缩带，但不再线性惩罚
    mean_radius = float(np.mean(radii))
    team_radial_error = (mean_radius - ideal_r) / (max_r - ideal_r + 1e-8)
    radial_shrink_team_score = np.exp(-3.0 * team_radial_error**2)
    components["radial_shrink_team"] = w_team_focus * radial_shrink_team_score

    # ========================
    # 2. 重构角度 gap 闭合分量
    # ========================
    # 显式鼓励角度均匀覆盖：gap 越均匀，奖励越高；
    # 同时鼓励当前 agent 向最大 gap 中心靠拢，且不再长期给负值。
    k = len(adversary_indices)
    angle_gap_reward = 0.0
    if k >= 2:
        sort_idx = np.argsort(angles)
        angles_sorted = angles[sort_idx]

        diffs = np.diff(angles_sorted)
        last_gap = (angles_sorted[0] + 2.0 * np.pi) - angles_sorted[-1]
        gaps = np.concatenate([diffs, [last_gap]])
        max_gap_idx = int(np.argmax(gaps))
        max_gap = float(gaps[max_gap_idx])

        # 全局覆盖均匀度奖励：若最大 gap 接近 2π/k，则奖励更高
        target_gap = 2.0 * np.pi / k
        gap_error = (max_gap - target_gap) / (2.0 * np.pi)
        uniformity_score = np.exp(-6.0 * gap_error**2)
        # 当前 agent 向最大 gap 中心靠拢奖励
        start_angle = angles_sorted[max_gap_idx]
        gap_center_angle = start_angle + max_gap / 2.0
        gap_center_angle = (gap_center_angle + np.pi) % (2.0 * np.pi) - np.pi

        d_theta = theta_self - gap_center_angle
        d_theta = (d_theta + np.pi) % (2.0 * np.pi) - np.pi

        # 将角度偏差映射到 [0,1]，0 为完美居中
        local_centering_score = np.exp(-4.0 * (d_theta / np.pi) ** 2)

        # 综合得分：既要整体均匀，又要本体靠近 gap 中心
        angle_gap_score = uniformity_score * local_centering_score
        angle_gap_reward = w_gap_closure * angle_gap_score

    components["angle_gap_closure"] = angle_gap_reward

    # ========================
    # 3. 速度对齐势场（保持原逻辑）
    # ========================
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

    # ========================
    # 4. 多体捕获势场（保持原逻辑）
    # ========================
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

    # ========================
    # 5. 重构团队几何中心分量
    # ========================
    # 由线性负惩罚改为“靠近给奖励，过远衰减”的正向势场，
    # 同时在适度距离内增强梯度。
    centroid = np.mean(adv_positions, axis=0)
    centroid_dist = np.linalg.norm(centroid - prey_pos)
    # 归一化距离
    norm_cd = centroid_dist / (world_size + 1e-8)
    # 钟形奖励：质心距离在 0 附近最高，随距离增加指数衰减
    centroid_score = np.exp(-3.0 * norm_cd**2)
    components["centroid_centrality"] = w_centrality * centroid_score

    # ========================
    # 6. 安全势场：重构软碰撞与紧凑性
    # ========================
    collision_penalty = 0.0
    near_collision_penalty = 0.0

    # 安全区间：鼓励在安全距离带内相互靠近，避免过疏散
    min_dist = 2.0 * adv_size
    safe_dist = safe_distance_factor * min_dist
    tight_dist = 0.8 * safe_dist  # 希望在 [min_dist, tight_dist] 形成紧凑但安全队形
    compact_bonus = 0.0
    compact_weight = 0.3  # 正向紧凑奖励权重

    for j in adversary_indices:
        if j == agent_index:
            continue
        d_ij = inter_agent_distances[agent_index, j]
        if d_ij < min_dist:
            # 硬碰撞：保持强负惩罚
            collision_penalty += w_collision_hard
        elif d_ij < safe_dist:
            # 软碰撞：使用平滑二次惩罚，距离越接近 min_dist 惩罚越大
            ratio = (safe_dist - d_ij) / (safe_dist - min_dist + 1e-8)
            near_collision_penalty += w_near_collision_soft * (ratio**2)
        elif min_dist < d_ij <= tight_dist:
            # 安全紧凑带：给予正向奖励，鼓励形成紧密包围
            # 映射到 [0,1]，d_ij 越靠近 min_dist，奖励越高
            compact_ratio = (tight_dist - d_ij) / (tight_dist - min_dist + 1e-8)
            compact_bonus += compact_weight * (compact_ratio**2)

    # 与猎物的硬碰撞维持原逻辑
    d_prey = inter_agent_distances[agent_index, prey_index]
    min_dist_ap = adv_size + prey_size
    if d_prey < min_dist_ap:
        collision_penalty += 0.5 * w_collision_hard

    components["collision_penalty"] = collision_penalty
    components["near_collision_penalty"] = near_collision_penalty + compact_bonus

    # ========================
    # 7. 时间压力势场（保持原逻辑）
    # ========================
    components["time_pressure"] = w_time_pressure

    total_reward = float(sum(components.values()))
    return total_reward, components