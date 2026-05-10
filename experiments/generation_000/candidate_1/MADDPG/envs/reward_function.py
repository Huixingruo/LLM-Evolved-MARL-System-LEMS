import numpy as np


def compute_reward(agent_name, observation, global_state, actions, world):
    # 非追捕者不参与奖励
    if not global_state['is_adversary']:
        return 0.0, {}

    components = {}

    # -----------------------
    # 物理与任务常量（硬编码）
    # -----------------------
    adv_size = 0.075
    prey_size = 0.050
    world_size = 2.5
    capture_threshold = global_state.get('capture_threshold', 0.5)

    # 队形与行为权重（可调）
    w_approach = 1.0
    w_capture_bonus = 5.0
    w_spread = 0.5
    w_angle_uniform = 0.75
    w_radius_uniform = 0.5
    w_collision = -5.0
    w_too_close_penalty = -0.5
    w_time_penalty = -0.01

    # 理想围捕半径（略小于 capture_threshold，留出缓冲避免碰撞）
    ideal_capture_radius = 0.8 * capture_threshold

    # 追捕者之间的最小安全距离（略大于碰撞半径）
    min_safety_factor = 1.2
    min_safe_dist = min_safety_factor * (2.0 * adv_size)

    # -----------------------
    # 从 global_state 解析信息
    # -----------------------
    agent_positions = global_state['agent_positions']
    agent_velocities = global_state['agent_velocities']
    prey_pos = global_state['prey_position']
    prey_vel = global_state['prey_velocity']
    distances_to_prey = global_state['distances_to_prey']
    inter_agent_distances = global_state['inter_agent_distances']

    # 假设 world.agents 的前若干为追捕者，且数量与 distances_to_prey 一致
    all_agents = list(world.agents)
    adversary_indices = [i for i, a in enumerate(all_agents) if getattr(a, 'adversary', False)]
    prey_indices = [i for i, a in enumerate(all_agents) if not getattr(a, 'adversary', False)]

    # 找到当前 agent 在 world.agents 中的索引及其在追捕者列表中的索引
    try:
        agent_index = [i for i, a in enumerate(all_agents) if a.name == agent_name][0]
    except IndexError:
        # 找不到则不给奖励（防御式处理）
        return 0.0, {}

    try:
        adv_local_index = adversary_indices.index(agent_index)
    except ValueError:
        # 不是追捕者（冗余保护，与 is_adversary 一致）
        return 0.0, {}

    agent_pos = agent_positions[agent_index]
    agent_vel = agent_velocities[agent_index]

    # -----------------------
    # 1. 靠近猎物（距离引导）
    # -----------------------
    # 当前追捕者与猎物的距离（从 distances_to_prey 中取）
    if adv_local_index < len(distances_to_prey):
        dist_to_prey = distances_to_prey[adv_local_index]
    else:
        # 容错：若长度不符，则直接计算
        dist_to_prey = float(np.linalg.norm(agent_pos - prey_pos))

    # 归一化距离（0 ~ 1），避免尺度过大
    max_possible_dist = np.sqrt(2.0) * world_size
    norm_dist = np.clip(dist_to_prey / max_possible_dist, 0.0, 1.0)

    # 鼓励靠近：距离越小奖励越大
    distance_reward = w_approach * (1.0 - norm_dist)
    components['distance_reward'] = float(distance_reward)

    # 捕获完成强奖励（在捕获半径内的奖励峰值）
    if dist_to_prey < capture_threshold:
        capture_bonus = w_capture_bonus * (1.0 - dist_to_prey / capture_threshold)
    else:
        capture_bonus = 0.0
    components['capture_bonus'] = float(capture_bonus)

    # -----------------------
    # 2. 追捕者之间防碰撞
    # -----------------------
    # 对当前追捕者与其它追捕者的距离做安全约束
    collision_penalty = 0.0
    crowding_penalty = 0.0
    for other_adv_idx in adversary_indices:
        if other_adv_idx == agent_index:
            continue
        d = inter_agent_distances[agent_index][other_adv_idx]

        # 真碰撞：距离小于物理半径之和
        collision_dist = 2.0 * adv_size
        if d < collision_dist:
            collision_penalty += w_collision

        # 过于接近但尚未碰撞：额外轻微惩罚，鼓励保持安全间距
        if d < min_safe_dist:
            ratio = (min_safe_dist - d) / max(min_safe_dist, 1e-6)
            crowding_penalty += w_too_close_penalty * ratio

    components['collision_penalty'] = float(collision_penalty)
    components['crowding_penalty'] = float(crowding_penalty)

    # -----------------------
    # 3. 队形：均匀包围（角度 + 半径）
    # -----------------------
    # 仅在存在三个追捕者时才进行完整队形奖励
    formation_angle_reward = 0.0
    formation_radius_reward = 0.0
    formation_spread_reward = 0.0

    if len(adversary_indices) == 3 and len(prey_indices) > 0:
        # 所有追捕者相对猎物的位置
        adv_positions = agent_positions[adversary_indices]
        rel_pos = adv_positions - prey_pos

        # 半径（与猎物的距离）
        radii = np.linalg.norm(rel_pos, axis=1) + 1e-8

        # 极角（-pi, pi]
        angles = np.arctan2(rel_pos[:, 1], rel_pos[:, 0])
        angles_sorted = np.sort(angles)

        # 计算角度间隔（考虑 2π 周期）
        angle_diffs = np.diff(angles_sorted)
        wrap_diff = (2.0 * np.pi - (angles_sorted[-1] - angles_sorted[0]))
        angle_diffs = np.concatenate([angle_diffs, [wrap_diff]])

        # 理想角度间隔为 2π/3
        ideal_angle = 2.0 * np.pi / 3.0
        # 使用方差衡量均匀度（标准化）
        angle_var = np.var(angle_diffs)
        max_angle_var = (np.pi ** 2)
        angle_uniform_score = 1.0 - np.clip(angle_var / max_angle_var, 0.0, 1.0)

        formation_angle_reward = w_angle_uniform * angle_uniform_score

        # 半径均匀：三个追捕者距离猎物的半径相近
        radius_var = np.var(radii)
        max_radius_var = (capture_threshold ** 2)
        radius_uniform_score = 1.0 - np.clip(radius_var / max_radius_var, 0.0, 1.0)
        formation_radius_reward = w_radius_uniform * radius_uniform_score

        # 整体半径靠近理想捕获半径
        mean_radius = float(np.mean(radii))
        diff_to_ideal = abs(mean_radius - ideal_capture_radius)
        # 归一化到 [0,1]，差距越小得分越高
        max_diff = capture_threshold
        spread_score = 1.0 - np.clip(diff_to_ideal / max_diff, 0.0, 1.0)
        formation_spread_reward = w_spread * spread_score

    components['formation_angle_reward'] = float(formation_angle_reward)
    components['formation_radius_reward'] = float(formation_radius_reward)
    components['formation_spread_reward'] = float(formation_spread_reward)

    # -----------------------
    # 4. 时间惩罚（鼓励尽快完成）
    # -----------------------
    time_penalty = w_time_penalty
    components['time_penalty'] = float(time_penalty)

    # -----------------------
    # 5. 轻微速度正则（避免过度抖动，可选）
    # -----------------------
    # 惩罚过大加速度/速度抖动可通过 actions 完成，这里简单用速度幅度
    speed = float(np.linalg.norm(agent_vel))
    max_speed_adv = 1.0
    norm_speed = np.clip(speed / max_speed_adv, 0.0, 2.0)
    w_speed_smooth = -0.01
    speed_penalty = w_speed_smooth * (norm_speed ** 2)
    components['speed_penalty'] = float(speed_penalty)

    # -----------------------
    # 6. 总奖励
    # -----------------------
    total_reward = float(sum(components.values()))
    return total_reward, components