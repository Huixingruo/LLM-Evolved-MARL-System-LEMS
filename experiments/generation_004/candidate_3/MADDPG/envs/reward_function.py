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
    capture_radius = 0.5

    # 队形与协同权重（全新范式）
    w_ring_radius = 2.0         # 理想环半径势场
    w_ring_compact = 1.5        # 环上邻居距离紧凑度
    w_encirclement = 3.0        # 包围角覆盖与均匀度
    w_radial_inward = 1.0       # 向内协同收缩
    w_focus = 1.0               # 中心聚焦（团队到猎物）
    w_capture = 10.0            # 真正多体包围捕获奖励
    w_safe_compact = 0.8        # 安全前提下的紧凑协同
    w_boundary = -0.5           # 远离边界惩罚
    w_energy = -0.02            # 动作能量惩罚（轻微）
    w_time = -0.001             # 轻微时间压力

    # 安全与碰撞
    w_collision_hard = -3.0
    w_near_collision_soft = -0.8
    safety_factor = 1.3

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
    adv_velocities = agent_velocities[adversary_indices]
    k = len(adversary_indices)

    self_pos = agent_positions[agent_index]
    self_vel = agent_velocities[agent_index]

    # ------------------------
    # 相对猎物坐标系（全新视角）
    # ------------------------
    rel_vecs = adv_positions - prey_pos[None, :]
    radii = np.linalg.norm(rel_vecs, axis=1) + 1e-8
    angles = np.arctan2(rel_vecs[:, 1], rel_vecs[:, 0])

    r_self = radii[adv_local_index]
    theta_self = angles[adv_local_index]

    # ------------------------
    # 1. 环形势场：鼓励形成围绕猎物的“安全环”
    # ------------------------
    # 理想环半径为捕获半径稍外侧
    ideal_ring_radius = capture_radius * 1.2
    ring_radius_error = np.abs(r_self - ideal_ring_radius) / world_size
    ring_radius_reward = -ring_radius_error
    components["ring_radius_self"] = w_ring_radius * ring_radius_reward

    # 团队平均环半径接近理想值
    mean_radius = float(np.mean(radii))
    mean_radius_error = np.abs(mean_radius - ideal_ring_radius) / world_size
    ring_radius_team_reward = -mean_radius_error
    components["ring_radius_team"] = w_ring_radius * ring_radius_team_reward

    # ------------------------
    # 2. 环上紧凑度：在安全距离内鼓励追捕者彼此更紧凑
    # ------------------------
    safe_min_dist = safety_factor * (2.0 * adv_size)
    # 计算当前 agent 与其最近的其他追捕者距离
    min_dist_to_teammate = np.inf
    for j in adversary_indices:
        if j == agent_index:
            continue
        d_ij = inter_agent_distances[agent_index, j]
        if d_ij < min_dist_to_teammate:
            min_dist_to_teammate = d_ij

    if np.isfinite(min_dist_to_teammate):
        if min_dist_to_teammate >= safe_min_dist:
            # 距离过大，线性鼓励靠近到安全边缘
            # 将 [safe_min_dist, world_size] 映射到 [0, -1]
            far_ratio = (min_dist_to_teammate - safe_min_dist) / (
                world_size - safe_min_dist + 1e-8
            )
            compact_reward = -np.clip(far_ratio, 0.0, 1.0)
        else:
            # 已在安全阈内，不再鼓励继续靠近（避免挤压）
            compact_reward = 0.0
    else:
        compact_reward = 0.0

    components["ring_compact_self"] = w_ring_compact * compact_reward

    # ------------------------
    # 3. 包围角分布势场：覆盖度 + 均匀度
    # ------------------------
    encirclement_reward = 0.0
    uniformity_reward = 0.0
    coverage_reward = 0.0

    if k >= 2:
        sort_idx = np.argsort(angles)
        angles_sorted = angles[sort_idx]
        diffs = np.diff(angles_sorted)
        last_gap = (angles_sorted[0] + 2.0 * np.pi) - angles_sorted[-1]
        gaps = np.concatenate([diffs, [last_gap]])
        max_gap = float(np.max(gaps))

        # 覆盖度：gap 越小越好
        coverage = 1.0 - max_gap / (2.0 * np.pi)
        coverage = np.clip(coverage, 0.0, 1.0)
        coverage_reward = coverage

        # 均匀度：gap 与均匀 gap 的偏差
        uniform_gap = 2.0 * np.pi / float(k)
        gap_deviation = np.mean(np.abs(gaps - uniform_gap))
        # 将 [0, π] -> [1, 0]
        uniformity = 1.0 - gap_deviation / np.pi
        uniformity = np.clip(uniformity, 0.0, 1.0)
        uniformity_reward = uniformity

        # 加权综合包围质量
        encirclement_reward = 0.5 * coverage + 0.5 * uniformity

    components["encirclement_quality"] = w_encirclement * encirclement_reward
    components["encirclement_uniformity"] = w_encirclement * uniformity_reward
    components["encirclement_coverage"] = w_encirclement * coverage_reward

    # ------------------------
    # 4. 协同径向收缩：整体向猎物收网而不过度压缩
    # ------------------------
    # 只要半径大于捕获半径，就鼓励向内缩小半径
    radial_inward_reward = 0.0
    if r_self > capture_radius:
        radial_inward_reward = (capture_radius - r_self) / world_size
    else:
        # 已在捕获半径内，鼓励稍微保持，不再继续强压缩
        radial_inward_reward = 0.2 * (capture_radius - r_self) / world_size

    components["radial_inward_self"] = w_radial_inward * radial_inward_reward

    # 团队平均收缩趋势
    mean_radial_inward = np.mean(
        (capture_radius - radii) / world_size
    )
    components["radial_inward_team"] = w_radial_inward * mean_radial_inward

    # ------------------------
    # 5. 团队聚焦：追捕者质心靠近猎物
    # ------------------------
    centroid = np.mean(adv_positions, axis=0)
    centroid_dist = np.linalg.norm(centroid - prey_pos)
    centroid_focus_reward = -centroid_dist / world_size
    components["team_focus"] = w_focus * centroid_focus_reward

    # ------------------------
    # 6. 多体捕获：在小半径形成高覆盖包围时强奖励
    # ------------------------
    inside_mask = radii <= capture_radius
    num_inside = int(np.sum(inside_mask))
    capture_reward = 0.0

    if num_inside >= 2:
        inner_radii = radii[inside_mask]
        inner_angles = angles[inside_mask]

        tightness = 1.0 - np.mean(inner_radii) / (capture_radius + 1e-8)
        tightness = np.clip(tightness, 0.0, 1.0)

        inner_sorted = np.sort(inner_angles)
        inner_diffs = np.diff(inner_sorted)
        last_inner_gap = (inner_sorted[0] + 2.0 * np.pi) - inner_sorted[-1]
        inner_gaps = np.concatenate([inner_diffs, [last_inner_gap]])
        max_inner_gap = float(np.max(inner_gaps))
        inner_coverage = 1.0 - max_inner_gap / (2.0 * np.pi)
        inner_coverage = np.clip(inner_coverage, 0.0, 1.0)

        participation = (num_inside - 1) / max(k - 1, 1)
        capture_score = tightness * inner_coverage * participation
        capture_reward = capture_score

    components["multi_agent_capture"] = w_capture * capture_reward

    # ------------------------
    # 7. 碰撞与安全：硬约束 + 近距离软惩罚
    # ------------------------
    collision_penalty = 0.0
    near_collision_penalty = 0.0

    min_dist = 2.0 * adv_size
    safe_dist = safety_factor * min_dist

    for j in adversary_indices:
        if j == agent_index:
            continue
        d_ij = inter_agent_distances[agent_index, j]
        if d_ij < min_dist:
            collision_penalty += w_collision_hard
        elif d_ij < safe_dist:
            ratio = (safe_dist - d_ij) / (safe_dist - min_dist + 1e-8)
            near_collision_penalty += w_near_collision_soft * ratio

    # 与猎物直接硬碰撞
    d_prey = inter_agent_distances[agent_index, prey_index]
    min_dist_ap = adv_size + prey_size
    if d_prey < min_dist_ap:
        collision_penalty += 0.5 * w_collision_hard

    components["collision_penalty"] = collision_penalty
    components["near_collision_penalty"] = near_collision_penalty

    # ------------------------
    # 8. 边界势场：鼓励远离地图边缘
    # ------------------------
    # 将坐标归一化到 [-1,1]，距边界越近惩罚越大
    norm_pos = self_pos / world_size
    margin = 1.0 - np.clip(np.abs(norm_pos), 0.0, 1.0)
    # margin 越小越靠边界
    boundary_penalty = -np.mean(1.0 - margin)
    components["boundary_penalty"] = w_boundary * boundary_penalty

    # ------------------------
    # 9. 能量与动作平滑惩罚（仅依赖当前动作范数）
    # ------------------------
    action_vec = np.asarray(actions.get(agent_name, np.zeros(2)), dtype=float)
    action_norm = np.linalg.norm(action_vec)
    energy_penalty = -action_norm
    components["energy_penalty"] = w_energy * energy_penalty

    # ------------------------
    # 10. 轻微时间压力（每步固定小惩罚）
    # ------------------------
    components["time_penalty"] = w_time

    total_reward = float(sum(components.values()))
    return total_reward, components