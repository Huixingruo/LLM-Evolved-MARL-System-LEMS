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