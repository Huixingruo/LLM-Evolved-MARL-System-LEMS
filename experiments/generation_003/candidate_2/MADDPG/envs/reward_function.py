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
    w_radial_shrink = 2.0        # 全局环形收缩势场（重构：带宽容区的非线性势场）
    w_gap_closure = 3.0          # 角度间隙填补（重构：宽松正区，削弱负区）
    w_alignment = 0.5            # 与理想方向对齐
    w_capture_dense = 10.0       # 紧密多体捕获奖励
    w_team_focus = 1.5           # 全局平均半径缩减（重构：相对改善型）
    w_centrality = 1.0           # 团队质心靠近猎物（重构：相对改善型）
    w_time_pressure = -0.01      # 时间压力（重构：随距离衰减）

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
    # 1. 全局收缩势场（重构：带宽容区的非线性势场）
    #    目标：允许中等距离区间近似零梯度，靠近目标圈时给予正向奖励，
    #    离得很远时给轻度惩罚，避免长期大负值主导。
    # ------------------------
    target_r = capture_threshold
    tolerance_band = 0.3  # 宽容带宽度
    far_scale = world_size

    if r_self > target_r + tolerance_band:
        # 远离目标圈：轻微负势场（使用平滑函数）
        x = (r_self - (target_r + tolerance_band)) / far_scale
        radial_potential = -x / (1.0 + x)  # [-1, 0) 区间，渐近而非线性爆炸
    elif r_self < target_r - tolerance_band:
        # 进入内侧过深：轻微负势场，避免过度挤进
        x = ((target_r - tolerance_band) - r_self) / far_scale
        radial_potential = -0.3 * x / (1.0 + x)
    else:
        # 在 [target_r - tol, target_r + tol] 内：按靠近目标圈给正奖励
        # 映射到 [-1, 1] 再成正
        center_offset = abs(r_self - target_r) / (tolerance_band + 1e-8)
        radial_potential = 1.0 - center_offset  # 距离越近越接近 1

    components["radial_shrink_self"] = w_radial_shrink * radial_potential

    # 团队平均半径势场（重构：鼓励相对改善，而非固定线性惩罚）
    mean_radius = float(np.mean(radii))
    # 以目标圈加宽容带为参考
    ref_radius = target_r + tolerance_band
    if mean_radius > ref_radius:
        # 只对明显大于参考半径的情况给轻度惩罚，且采用缓和非线性
        x_team = (mean_radius - ref_radius) / far_scale
        team_potential = -x_team / (1.0 + x_team)
    else:
        # 当平均半径已经不大时，给予正向鼓励但做饱和
        x_team = (ref_radius - mean_radius) / far_scale
        team_potential = x_team / (1.0 + x_team)

    components["radial_shrink_team"] = w_team_focus * team_potential

    # ------------------------
    # 2. 角度势场：填补包围角间隙（重构：宽松正区，削弱负区）
    # ------------------------
    k = len(adversary_indices)
    angle_gap_reward = 0.0
    d_theta_for_alignment = None
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
        gap_center_angle = (gap_center_angle + np.pi) % (2.0 * np.pi) - np.pi

        # 当前 agent 与 gap center 的角度差（在环上最近距离）
        d_theta = theta_self - gap_center_angle
        d_theta = (d_theta + np.pi) % (2.0 * np.pi) - np.pi
        d_theta_for_alignment = d_theta

        # 重构逻辑：
        # - 使用宽松高斯帽鼓励 |d_theta| 小于一定阈值
        # - 对远离 gap center 的区域几乎不惩罚，仅轻度负值
        sigma = np.pi / 3.0  # 宽一些的角度带
        pos_part = np.exp(-0.5 * (d_theta / sigma) ** 2)

        # 负区平滑处理：只在 gap 非常大时对极端反方向给轻微负值
        # 归一化 gap 大小
        gap_scale = np.clip(max_gap / (2.0 * np.pi), 0.0, 1.0)
        # 对 |d_theta| > pi/2 的区域给小负值
        large_misalign = np.maximum(0.0, abs(d_theta) - np.pi / 2.0)
        neg_part = -0.3 * gap_scale * (large_misalign / np.pi)

        angle_gap_reward = w_gap_closure * (pos_part + neg_part)

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
        if d_theta_for_alignment is not None:
            direction_sign = np.sign(-d_theta_for_alignment) \
                if abs(d_theta_for_alignment) > 1e-3 else 0.0
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
    # 5. 团队几何中心势场：团队质心靠近猎物（重构：相对改善型）
    # ------------------------
    centroid = np.mean(adv_positions, axis=0)
    centroid_dist = np.linalg.norm(centroid - prey_pos)
    # 使用相对改善形式：相对世界对角线距离归一化后映射到 [-1, 1]
    max_dist = np.sqrt(2.0) * world_size
    norm_d = np.clip(centroid_dist / (max_dist + 1e-8), 0.0, 1.0)
    centroid_potential = 1.0 - 2.0 * norm_d  # 0 附近 -> -1 到 1
    components["centroid_centrality"] = w_centrality * centroid_potential

    # ------------------------
    # 6. 安全势场：防止追捕者之间与猎物发生硬碰撞，同时允许适度接近
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
    # 7. 时间压力势场（重构：随距离衰减的时间惩罚）
    #    - 若已接近猎物，则时间惩罚减弱，避免压制终局微调。
    # ------------------------
    self_pos = agent_positions[agent_index]
    dist_to_prey = np.linalg.norm(self_pos - prey_pos)
    dist_factor = np.clip(dist_to_prey / world_size, 0.0, 1.0)
    time_pressure_scaled = w_time_pressure * (0.5 + 0.5 * dist_factor)
    components["time_pressure"] = time_pressure_scaled

    total_reward = float(sum(components.values()))
    return total_reward, components