import numpy as np


def compute_reward(agent_name, observation, global_state, actions, world):
    if not global_state["is_adversary"]:
        return 0.0, {}

    components = {}

    # ------------------------
    # 物理与环境常量（硬编码）
    # ------------------------
    adv_size = 0.075
    prey_size = 0.050
    world_size = 2.5
    capture_radius = 0.5  # 捕获判定目标半径
    soft_capture_radius = 0.8  # 软捕获区（鼓励收缩）
    min_encircle_radius = 0.3  # 真正紧密包围区

    # 安全距离
    adv_min_dist = 2.0 * adv_size
    adv_safe_dist = 1.3 * adv_min_dist
    ap_min_dist = adv_size + prey_size

    # ------------------------
    # 奖励权重（新范式：分层势场）
    # ------------------------
    # 层级：1) 逼近与控制猎物；2) 多角度封锁；3) 环形协同；4) 安全与边界约束
    w_approach = 2.5          # 逼近势场（半径压缩）
    w_ring_shaping = 2.0      # 环带内半径一致性
    w_angle_cover = 3.0       # 全局角度覆盖度
    w_local_sector = 2.5      # 局部扇区填充
    w_front_back = 1.5        # 前后夹击结构
    w_soft_capture = 6.0      # 软捕获区多体聚集
    w_tight_capture = 12.0    # 真正紧密包围
    w_velocity_flow = 1.5     # 团队流形对齐
    w_centroid_orbit = 1.0    # 质心环绕/靠近
    w_boundary_soft = -1.0    # 软边界惩罚
    w_collision_hard = -3.0   # 硬碰撞惩罚
    w_collision_soft = -0.8   # 近碰惩罚
    w_time_shaping = -0.003   # 温和时间压力（每步）

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

    # 当前追捕者在 adversary 列表中的局部索引
    try:
        adv_local_index = adversary_indices.index(agent_index)
    except ValueError:
        adv_local_index = 0

    adv_positions = agent_positions[adversary_indices]
    adv_vels = agent_velocities[adversary_indices]

    # ------------------------
    # 相对猎物的极坐标系（全局势场基础）
    # ------------------------
    rel_vecs = adv_positions - prey_pos[None, :]
    radii = np.linalg.norm(rel_vecs, axis=1) + 1e-8
    angles = np.arctan2(rel_vecs[:, 1], rel_vecs[:, 0])

    r_self = radii[adv_local_index]
    theta_self = angles[adv_local_index]
    vel_self = adv_vels[adv_local_index]

    # ------------------------
    # 1. 逼近势场：半径压缩 + 软捕获区分层
    # ------------------------
    # 1.1 基础逼近：鼓励半径尽量 <= soft_capture_radius
    # 使用平滑函数：在大范围内仍有梯度，逐渐增强
    # f(r) = 1 / (1 + (r / soft_capture_radius)^2)
    approach_score = 1.0 / (1.0 + (r_self / soft_capture_radius) ** 2)
    components["approach_prey"] = w_approach * approach_score

    # 1.2 环形半径一致性（团队层）：鼓励所有追捕者在相似半径上形成“控制圈”
    mean_radius = float(np.mean(radii))
    radius_spread = float(np.std(radii))
    # 半径一致性：半径标准差越小越好，归一化后转成 [0,1]
    # 假设合理标准差上界约为 soft_capture_radius / 2
    spread_norm = np.clip(radius_spread / (soft_capture_radius / 2.0 + 1e-8), 0.0, 1.0)
    ring_coherence = 1.0 - spread_norm
    # 同时偏向团队平均半径 <= soft_capture_radius
    ring_radius_factor = 1.0 / (1.0 + (mean_radius / soft_capture_radius) ** 2)
    ring_shaping_score = ring_coherence * ring_radius_factor
    components["ring_shaping"] = w_ring_shaping * ring_shaping_score

    # ------------------------
    # 2. 全局角度覆盖势场：多角度封锁
    # ------------------------
    k = len(adversary_indices)
    angle_cover_reward = 0.0
    local_sector_reward = 0.0
    front_back_reward = 0.0

    if k >= 2:
        # 2.1 全局覆盖度：希望追捕者在环上分散覆盖，减少大间隙
        sort_idx = np.argsort(angles)
        angles_sorted = angles[sort_idx]
        diffs = np.diff(angles_sorted)
        last_gap = (angles_sorted[0] + 2.0 * np.pi) - angles_sorted[-1]
        gaps = np.concatenate([diffs, [last_gap]])
        max_gap = float(np.max(gaps))

        # 覆盖度：1 - max_gap / (2π)，并对“接近均匀”状态额外放大
        base_coverage = 1.0 - max_gap / (2.0 * np.pi)
        base_coverage = np.clip(base_coverage, 0.0, 1.0)
        # 使用凸函数加强高覆盖区域的奖励
        angle_cover_score = base_coverage ** 2
        angle_cover_reward = w_angle_cover * angle_cover_score

        # 2.2 局部扇区填充：每个追捕者检测自身附近是否形成“扇区控制”
        # 设定局部扇区宽度，例如 60 度
        sector_width = np.pi / 3.0
        # 计算相对当前 agent 的角度差
        d_thetas = angles - theta_self
        d_thetas = (d_thetas + np.pi) % (2.0 * np.pi) - np.pi
        # 所在本扇区内的队友数量
        in_sector = np.logical_and(d_thetas > -sector_width / 2.0,
                                   d_thetas < sector_width / 2.0)
        num_in_sector = int(np.sum(in_sector))
        # 我自己必然在扇区中，所以关注是否还有 1~2 个协同者
        # 使用一个平滑函数：理想数量为 2~3
        ideal_min = 2.0
        ideal_max = 3.0
        if num_in_sector <= 1:
            sector_score = 0.0
        elif num_in_sector <= ideal_max:
            sector_score = (num_in_sector - 1) / (ideal_max - 1 + 1e-8)
        else:
            # 人太多反而拥挤，缓慢衰减
            over = num_in_sector - ideal_max
            sector_score = 1.0 / (1.0 + over ** 2)

        local_sector_reward = w_local_sector * sector_score

        # 2.3 前后夹击结构：存在“对面”追捕者时给予额外奖励
        # 找到角度与我相差约 π (±45°) 的个体
        opposite_threshold = np.pi / 4.0
        opposite_mask = np.abs(np.abs(d_thetas) - np.pi) < opposite_threshold
        has_opposite = bool(np.any(opposite_mask))
        front_back_score = 1.0 if has_opposite else 0.0
        front_back_reward = w_front_back * front_back_score

    components["angle_cover_global"] = angle_cover_reward
    components["local_sector_control"] = local_sector_reward
    components["front_back_structure"] = front_back_reward

    # ------------------------
    # 3. 捕获层次势场：软捕获与紧密包围
    # ------------------------
    # 3.1 软捕获区：在软捕获半径内的追捕者越多越好，并且角度分布不要极度偏一侧
    inside_soft_mask = radii <= soft_capture_radius
    num_soft_inside = int(np.sum(inside_soft_mask))
    soft_capture_reward = 0.0

    if num_soft_inside >= 2:
        inner_angles = angles[inside_soft_mask]
        inner_sorted = np.sort(inner_angles)
        inner_diffs = np.diff(inner_sorted)
        last_inner_gap = (inner_sorted[0] + 2.0 * np.pi) - inner_sorted[-1]
        inner_gaps = np.concatenate([inner_diffs, [last_inner_gap]])
        max_inner_gap = float(np.max(inner_gaps))
        coverage_soft = 1.0 - max_inner_gap / (2.0 * np.pi)
        coverage_soft = np.clip(coverage_soft, 0.0, 1.0)

        # 数量系数：相对总追捕者数量
        count_factor_soft = (num_soft_inside - 1) / max(len(adversary_indices) - 1, 1)
        # 自身是否在软捕获区内：更直接奖励
        self_inside_soft = float(inside_soft_mask[adv_local_index])
        soft_capture_score = coverage_soft * count_factor_soft * self_inside_soft
        soft_capture_reward = w_soft_capture * soft_capture_score

    components["soft_capture_layer"] = soft_capture_reward

    # 3.2 紧密包围：极小半径内多体聚集 + 良好角度覆盖
    inside_tight_mask = radii <= min_encircle_radius
    num_tight_inside = int(np.sum(inside_tight_mask))
    tight_capture_reward = 0.0

    if num_tight_inside >= 2:
        tight_radii = radii[inside_tight_mask]
        tight_angles = angles[inside_tight_mask]

        tight_sorted = np.sort(tight_angles)
        tight_diffs = np.diff(tight_sorted)
        last_tight_gap = (tight_sorted[0] + 2.0 * np.pi) - tight_sorted[-1]
        tight_gaps = np.concatenate([tight_diffs, [last_tight_gap]])
        max_tight_gap = float(np.max(tight_gaps))
        tight_coverage = 1.0 - max_tight_gap / (2.0 * np.pi)
        tight_coverage = np.clip(tight_coverage, 0.0, 1.0)

        # 半径极度靠内的紧凑度
        tightness = 1.0 - np.mean(tight_radii) / (min_encircle_radius + 1e-8)
        tightness = np.clip(tightness, 0.0, 1.0)

        count_factor_tight = (num_tight_inside - 1) / max(len(adversary_indices) - 1, 1)
        self_inside_tight = float(inside_tight_mask[adv_local_index])

        tight_capture_score = (
            tight_coverage * tightness * count_factor_tight * self_inside_tight
        )
        tight_capture_reward = w_tight_capture * tight_capture_score

    components["tight_capture_layer"] = tight_capture_reward

    # ------------------------
    # 4. 团队流场与质心势场：整体运动结构
    # ------------------------
    # 4.1 团队流形对齐：鼓励速度方向在局部与“理想轨道流”一致
    # 理想流场：在远端径向接近，在 capture_radius 附近沿切向绕行（环绕控制）
    ideal_dir = np.zeros(2, dtype=float)
    rel_self = rel_vecs[adv_local_index]
    r = r_self

    if r > capture_radius:
        # 远端：径向向内靠近
        ideal_dir = -rel_self / (np.linalg.norm(rel_self) + 1e-8)
    else:
        # 捕获圈附近：沿切向构成环绕流
        tangential = np.array([-rel_self[1], rel_self[0]], dtype=float)
        norm_t = np.linalg.norm(tangential) + 1e-8
        ideal_dir = tangential / norm_t

    speed_self = np.linalg.norm(vel_self) + 1e-8
    if speed_self > 1e-3:
        dir_self = vel_self / speed_self
        flow_alignment_score = float(np.dot(dir_self, ideal_dir))
        # 归一化到 [0,1]
        flow_alignment_score = (flow_alignment_score + 1.0) / 2.0
    else:
        flow_alignment_score = 0.0

    components["velocity_flow_alignment"] = w_velocity_flow * flow_alignment_score

    # 4.2 质心势场：团队质心既要靠近猎物，又偏向形成环绕
    centroid = np.mean(adv_positions, axis=0)
    centroid_vec = centroid - prey_pos
    centroid_dist = np.linalg.norm(centroid_vec) + 1e-8
    # 靠近猎物中心的奖励
    centroid_approach_score = 1.0 / (1.0 + (centroid_dist / soft_capture_radius) ** 2)
    # 环绕结构：质心不应“穿过”猎物太远，可以设质心在 capture_radius 左右更好
    # 使用高斯形势场围绕 capture_radius
    target_c = capture_radius
    sigma_c = capture_radius / 2.0
    centering = np.exp(-0.5 * ((centroid_dist - target_c) / (sigma_c + 1e-8)) ** 2)

    centroid_orbit_score = centroid_approach_score * centering
    components["centroid_orbit"] = w_centroid_orbit * centroid_orbit_score

    # ------------------------
    # 5. 安全与边界势场
    # ------------------------
    collision_penalty = 0.0
    near_collision_penalty = 0.0

    # 5.1 追捕者之间的安全距离
    for j in adversary_indices:
        if j == agent_index:
            continue
        d_ij = inter_agent_distances[agent_index, j]
        if d_ij < adv_min_dist:
            collision_penalty += w_collision_hard
        elif d_ij < adv_safe_dist:
            ratio = (adv_safe_dist - d_ij) / (adv_safe_dist - adv_min_dist + 1e-8)
            near_collision_penalty += w_collision_soft * ratio

    # 5.2 追捕者与猎物的碰撞惩罚（鼓励围住而不是撞上）
    d_prey = inter_agent_distances[agent_index, prey_index]
    if d_prey < ap_min_dist:
        collision_penalty += 0.5 * w_collision_hard

    components["collision_penalty"] = collision_penalty
    components["near_collision_penalty"] = near_collision_penalty

    # 5.3 边界势场：靠近边界时给软惩罚，鼓励在场内围捕
    pos_self = agent_positions[agent_index]
    # 计算离四个边界的距离
    margin = world_size - np.abs(pos_self)
    # 使用最近边界距离
    min_margin = float(np.min(margin))
    # 当 min_margin < 某阈值时惩罚，阈值设置为 world_size * 0.3
    boundary_thresh = world_size * 0.3
    if min_margin < boundary_thresh:
        # 惩罚随接近边界非线性增长
        boundary_ratio = 1.0 - min_margin / (boundary_thresh + 1e-8)
        boundary_penalty = w_boundary_soft * (boundary_ratio ** 2)
    else:
        boundary_penalty = 0.0

    components["boundary_penalty"] = boundary_penalty

    # ------------------------
    # 6. 时间形势：轻量时间压力，避免无意义拖延
    # ------------------------
    components["time_shaping"] = w_time_shaping

    total_reward = float(sum(components.values()))
    return total_reward, components