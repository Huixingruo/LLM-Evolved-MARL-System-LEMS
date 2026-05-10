import numpy as np


def compute_reward(agent_name, observation, global_state, actions, world):
    # 非追捕者直接返回 0
    if not global_state["is_adversary"]:
        return 0.0, {}

    components = {}

    # -----------------------------
    # 基础物理与任务常量（硬编码）
    # -----------------------------
    adv_size = 0.075
    prey_size = 0.050
    world_size = 2.5
    capture_threshold = 0.5

    # 追捕者数量与索引假设：前 3 个为追捕者，第 4 个为逃逸者
    # 若具体环境索引不同，可在集成时调整
    agent_positions = global_state["agent_positions"]
    prey_position = global_state["prey_position"]
    inter_agent_distances = global_state["inter_agent_distances"]
    distances_to_prey = global_state["distances_to_prey"]

    num_agents = agent_positions.shape[0]
    num_adversaries = distances_to_prey.shape[0]

    # 当前追捕者索引解析（假设 agent_name 形如 "adversary_0", "adversary_1"...）
    # 找不到时退化为第一个 adversary
    adv_index = 0
    if isinstance(agent_name, str) and "adversary" in agent_name:
        try:
            adv_index = int(agent_name.split("_")[-1])
        except (ValueError, IndexError):
            adv_index = 0
    adv_index = max(0, min(adv_index, num_adversaries - 1))

    # -----------------------------
    # 1) 接近目标（距离引导）
    # -----------------------------
    # 使用所有追捕者到猎物距离的平均值作为团队靠近度指标
    mean_dist_to_prey = float(np.mean(distances_to_prey)) if num_adversaries > 0 else 0.0
    # 归一化到 [0, 1] 左右，距离越小奖励越大
    # 距离尺度最大约为 world_size * sqrt(2) ≈ 3.5
    norm_dist = mean_dist_to_prey / (world_size * np.sqrt(2.0) + 1e-6)
    distance_reward = -norm_dist
    components["distance_reward"] = distance_reward

    # -----------------------------
    # 2) 防碰撞惩罚（追捕者之间）
    # -----------------------------
    # 对任意两追捕者，如果距离小于安全半径则施加惩罚
    # 安全距离略大于物理碰撞半径
    collision_margin = 0.02
    min_safe_dist = 2 * adv_size + collision_margin

    collision_penalty = 0.0
    crowded_penalty = 0.0
    for i in range(num_adversaries):
        for j in range(i + 1, num_adversaries):
            d_ij = inter_agent_distances[i, j]
            if d_ij < 2 * adv_size:
                # 真正几何碰撞（强惩罚）
                collision_penalty -= 1.0
            elif d_ij < min_safe_dist:
                # 过于接近但尚未碰撞（软惩罚）
                crowded_penalty -= (min_safe_dist - d_ij) / (min_safe_dist - 2 * adv_size + 1e-6)

    components["collision_penalty"] = collision_penalty
    components["crowded_penalty"] = crowded_penalty

    # -----------------------------
    # 3) 包围几何结构：角度均匀 + 半径均匀
    # -----------------------------
    surround_angle_reward = 0.0
    surround_radius_reward = 0.0
    inside_triangle_reward = 0.0

    if num_adversaries == 3:
        adv_positions = agent_positions[:num_adversaries]
        rel_vecs = adv_positions - prey_position  # shape: (3, 2)

        # 半径与角度
        radii = np.linalg.norm(rel_vecs, axis=1)
        angles = np.arctan2(rel_vecs[:, 1], rel_vecs[:, 0])  # [-pi, pi]

        # 半径均匀：方差越小越好
        radius_var = float(np.var(radii))
        # 距离期望半径：接近 capture_threshold 的 0.6~1.0 区间
        target_radius = capture_threshold * 0.8
        radius_deviation = np.mean(np.abs(radii - target_radius))
        # 将方差和偏差组合为惩罚
        # 归一化尺度：最大可能半径约 world_size * sqrt(2)
        radius_scale = world_size * np.sqrt(2.0)
        radius_uniformity_penalty = (
            0.5 * (radius_var / (radius_scale ** 2 + 1e-6))
            + 0.5 * (radius_deviation / (radius_scale + 1e-6))
        )
        surround_radius_reward = -radius_uniformity_penalty

        # 角度均匀：应接近 120° 等分
        sorted_angles = np.sort(angles)
        # 补上环绕间隔
        angle_gaps = np.diff(sorted_angles)
        last_gap = (sorted_angles[0] + 2 * np.pi) - sorted_angles[-1]
        angle_gaps = np.concatenate([angle_gaps, np.array([last_gap])])
        # 理想间隔
        ideal_gap = 2 * np.pi / 3.0
        angle_gap_dev = np.mean(np.abs(angle_gaps - ideal_gap))
        # 归一化到 [0, 1]
        angle_uniformity_penalty = angle_gap_dev / (np.pi + 1e-6)
        surround_angle_reward = -angle_uniformity_penalty

        # 判断猎物是否在三追捕者构成的三角形内部
        a = adv_positions[0]
        b = adv_positions[1]
        c = adv_positions[2]
        p = prey_position

        def _sign(p1, p2, p3):
            return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])

        d1 = _sign(p, a, b)
        d2 = _sign(p, b, c)
        d3 = _sign(p, c, a)
        has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
        has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
        is_inside = not (has_neg and has_pos)

        if is_inside:
            # 若猎物在三角形内，给予正奖励；否则 0
            inside_triangle_reward = 0.5

    components["surround_radius_reward"] = surround_radius_reward
    components["surround_angle_reward"] = surround_angle_reward
    components["inside_triangle_reward"] = inside_triangle_reward

    # -----------------------------
    # 4) 完成围捕 / 终局奖励（团队）
    # -----------------------------
    # 条件示例：所有追捕者距离猎物都小于 capture_threshold 且猎物在三角形内部
    capture_bonus = 0.0
    if num_adversaries == 3:
        all_close = bool(np.all(distances_to_prey < capture_threshold))
        if all_close and inside_triangle_reward > 0.0:
            capture_bonus = 2.0
    components["capture_bonus"] = capture_bonus

    # -----------------------------
    # 5) 时间惩罚（鼓励快速完成）
    # -----------------------------
    # 每个时间步给一个小的负常数，鼓励尽快结束
    time_penalty = -0.01
    components["time_penalty"] = time_penalty

    total_reward = float(sum(components.values()))
    return total_reward, components