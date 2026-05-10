import numpy as np


def compute_reward(agent_name, observation, global_state, actions, world):
    if not global_state["is_adversary"]:
        return 0.0, {}

    components = {}

    # ------------------------
    # 势场超参数（全新范式）
    # ------------------------
    world_size = 2.5
    capture_radius = 0.5
    adv_size = 0.075
    prey_size = 0.050

    # 势场形状参数
    # 强正：包围成功与近距离“势能井”
    w_capture = 8.0
    w_close_potential = 3.0
    # 中等：协同势场
    w_ring = 1.5
    w_angle_spread = 1.0
    # 轻微：防碰撞与时间
    w_collision = 2.0
    w_near_collision = 0.5
    w_time = -0.002
    # 轻微：速度引导
    w_radial_speed = 0.5
    w_tangential_speed = 0.3

    # ------------------------
    # 解析全局信息
    # ------------------------
    agent_positions = np.asarray(global_state["agent_positions"], dtype=float)
    agent_velocities = np.asarray(global_state["agent_velocities"], dtype=float)
    prey_pos = np.asarray(global_state["prey_position"], dtype=float)
    prey_vel = np.asarray(global_state["prey_velocity"], dtype=float)
    inter_agent_distances = np.asarray(
        global_state["inter_agent_distances"], dtype=float
    )

    num_agents = agent_positions.shape[0]

    # 寻找当前 agent 的全局索引
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

    # adversaries & prey index
    adversary_indices = [
        idx for idx, ag in enumerate(world.agents) if getattr(ag, "adversary", False)
    ]
    prey_index = None
    for idx, ag in enumerate(world.agents):
        if not getattr(ag, "adversary", False):
            prey_index = idx
            break

    # 当前 agent 在 adversaries 中的局部索引
    try:
        adv_local_index = adversary_indices.index(agent_index)
    except ValueError:
        adv_local_index = 0

    # 若没有猎物，奖励全部为 0
    if prey_index is None:
        components["no_prey"] = 0.0
        total_reward = float(sum(components.values()))
        return total_reward, components

    # ------------------------
    # 1. 目标势场：径向高斯井 + 捕获势阱
    # ------------------------
    adv_positions = agent_positions[adversary_indices]
    self_pos = agent_positions[agent_index]
    self_vel = agent_velocities[agent_index]

    vec_self = self_pos - prey_pos
    dist_self = np.linalg.norm(vec_self) + 1e-8

    # 多层径向势场：
    # - 中距离：向 capture_radius 收缩
    # - 近距离：进入捕获势阱
    target_radius = capture_radius * 0.9
    sigma_far = world_size * 0.7
    sigma_mid = capture_radius * 1.0
    sigma_near = capture_radius * 0.4

    def _gaussian_potential(r, r0, sigma):
        return np.exp(-0.5 * ((r - r0) / (sigma + 1e-8)) ** 2)

    # 在目标环附近获得较大正值
    pot_mid = _gaussian_potential(dist_self, target_radius, sigma_mid)
    # 在更靠近中心的位置也依然有正值，帮助最终逼近
    pot_near = _gaussian_potential(dist_self, 0.0, sigma_near)
    # 远距离弱势场，引导向猎物靠近
    pot_far = _gaussian_potential(dist_self, 0.0, sigma_far)

    close_potential = w_close_potential * (0.3 * pot_far + 0.4 * pot_mid + 0.3 * pot_near)

    # 捕获判定：所有追捕者都在 capture_radius 内
    dists_all = np.linalg.norm(adv_positions - prey_pos[None, :], axis=1)
    all_within = bool(np.all(dists_all < capture_radius))
    capture_signal = w_capture if all_within else 0.0

    components["close_potential"] = float(close_potential)
    components["capture_signal"] = float(capture_signal)

    # ------------------------
    # 2. 环形协同势场：半径分布与角度分布
    # ------------------------
    ring_reward = 0.0
    angle_spread_reward = 0.0
    if len(adversary_indices) >= 2:
        # 极坐标表示
        vecs_all = adv_positions - prey_pos[None, :]
        radii = np.linalg.norm(vecs_all, axis=1) + 1e-8
        angles = np.arctan2(vecs_all[:, 1], vecs_all[:, 0])

        # 半径：鼓励所有追捕者分布在环带 [0.5 * target_radius, 1.5 * target_radius] 内
        # 使用软势：在环带内奖励较高，外侧快速衰减
        ring_center = target_radius
        ring_width = capture_radius * 0.6

        ring_pots = _gaussian_potential(radii, ring_center, ring_width)
        ring_reward = w_ring * float(np.mean(ring_pots))

        # 角度：鼓励在圆周上分散开，而不是聚成团
        angles_sorted = np.sort(angles)
        diffs = np.diff(angles_sorted)
        last_gap = (angles_sorted[0] + 2.0 * np.pi) - angles_sorted[-1]
        diffs = np.concatenate([diffs, [last_gap]])

        # 理想间隔：均匀分布
        ideal_gap = 2.0 * np.pi / len(adversary_indices)
        gap_dev = np.abs(diffs - ideal_gap)
        # 用软势而非线性惩罚：间隔接近 ideal_gap 时获得奖励
        gap_sigma = np.pi / len(adversary_indices)
        gap_pots = np.exp(-0.5 * (gap_dev / (gap_sigma + 1e-8)) ** 2)
        angle_spread_reward = w_angle_spread * float(np.mean(gap_pots))

    components["ring_field"] = float(ring_reward)
    components["angle_spread_field"] = float(angle_spread_reward)

    # ------------------------
    # 3. 速度极坐标引导：径向收缩 + 切向包围
    # ------------------------
    radial_speed_reward = 0.0
    tangential_speed_reward = 0.0

    # 基于极坐标分解当前速度
    radial_dir = vec_self / (dist_self + 1e-8)
    tangential_dir = np.array([-radial_dir[1], radial_dir[0]])

    v_r = float(np.dot(self_vel, radial_dir))       # 向内为负，向外为正
    v_t = float(np.dot(self_vel, tangential_dir))   # 顺时针 / 逆时针

    # 径向：鼓励在远距离时 v_r 指向猎物（负），靠近目标环时减弱
    radial_scale = np.clip(dist_self / world_size, 0.0, 1.0)
    radial_speed_reward = w_radial_speed * (-v_r) * radial_scale

    # 切向：在接近目标环时鼓励切向移动（绕猎物转圈），远距离时减弱
    tangential_scale = _gaussian_potential(dist_self, target_radius, capture_radius)
    tangential_speed_reward = w_tangential_speed * abs(v_t) * tangential_scale

    components["radial_speed"] = float(radial_speed_reward)
    components["tangential_speed"] = float(tangential_speed_reward)

    # ------------------------
    # 4. 防碰撞势场：硬核碰撞 + 软斥力
    # ------------------------
    collision_penalty = 0.0
    near_collision_penalty = 0.0

    # 追捕者 - 追捕者
    for j in adversary_indices:
        if j == agent_index:
            continue
        d_ij = inter_agent_distances[agent_index, j]
        min_dist = 2.0 * adv_size
        soft_zone = 1.8 * min_dist

        if d_ij < min_dist:
            # 强斥力
            overlap = max(min_dist - d_ij, 0.0) / (min_dist + 1e-8)
            collision_penalty -= w_collision * (1.0 + 2.0 * overlap)
        elif d_ij < soft_zone:
            # 软势斥力
            margin = (soft_zone - d_ij) / (soft_zone - min_dist + 1e-8)
            near_collision_penalty -= w_near_collision * margin

    # 追捕者 - 猎物：轻微斥力，鼓励包围但不撞击
    d_ap = inter_agent_distances[agent_index, prey_index]
    min_ap = adv_size + prey_size
    soft_ap = 1.5 * min_ap
    if d_ap < min_ap:
        overlap = max(min_ap - d_ap, 0.0) / (min_ap + 1e-8)
        collision_penalty -= 0.5 * w_collision * (1.0 + overlap)
    elif d_ap < soft_ap:
        margin = (soft_ap - d_ap) / (soft_ap - min_ap + 1e-8)
        near_collision_penalty -= 0.25 * w_near_collision * margin

    components["collision_penalty"] = float(collision_penalty)
    components["near_collision_penalty"] = float(near_collision_penalty)

    # ------------------------
    # 5. 简单时间势场：轻微向早期捕获倾斜
    # ------------------------
    components["time_penalty"] = float(w_time)

    # ------------------------
    # 6. 环境边界斥力（势场视角）
    # ------------------------
    boundary_penalty = 0.0
    margin = 0.1 * world_size
    for dim in range(2):
        coord = self_pos[dim]
        # 距离边界的距离
        dist_min = min(coord + world_size, world_size - coord)
        if dist_min < margin:
            # 软斥力，越靠近边界惩罚越大
            factor = (margin - dist_min) / (margin + 1e-8)
            boundary_penalty -= 0.5 * factor
    components["boundary_field"] = float(boundary_penalty)

    total_reward = float(sum(components.values()))
    return total_reward, components