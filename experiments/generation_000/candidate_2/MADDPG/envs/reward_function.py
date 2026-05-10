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
    time_penalty_per_step = -0.01

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
    # 假设命名规则为 "agent_i" 或 "adversary_i" 等，尝试从 world.agents 匹配
    agent_index = None
    for idx, ag in enumerate(world.agents):
        if getattr(ag, "name", None) == agent_name:
            agent_index = idx
            break

    if agent_index is None:
        # 回退：尝试根据名字中的数字索引
        try:
            agent_index = int("".join(ch for ch in agent_name if ch.isdigit()))
            if agent_index < 0 or agent_index >= num_agents:
                agent_index = 0
        except ValueError:
            agent_index = 0

    # adversary 索引集合（追捕者）
    adversary_indices = [
        idx for idx, ag in enumerate(world.agents) if getattr(ag, "adversary", False)
    ]

    # 找出当前 agent 在 adversaries 中的索引
    try:
        adv_local_index = adversary_indices.index(agent_index)
    except ValueError:
        adv_local_index = 0

    # ------------------------
    # 1. 距离引导：靠近目标 + 围捕完成奖励
    # ------------------------
    if len(distances_to_prey) > 0:
        # 当前追捕者到猎物距离
        d_self = distances_to_prey[adv_local_index]
        # 所有追捕者到猎物平均距离（协同）
        mean_d_adv = float(np.mean(distances_to_prey))

        # 归一化距离（最好 ~ 0）
        norm_d_self = d_self / world_size
        norm_d_mean = mean_d_adv / world_size

        # 鼓励靠近猎物（自我与团队两部分）
        distance_reward_self = -norm_d_self
        distance_reward_team = -norm_d_mean * 0.5

        # 额外的捕获/围捕成功奖励：所有追捕者都在捕获半径内
        all_within_capture = bool(
            np.all(distances_to_prey < capture_threshold)
        )
        capture_bonus = 0.0
        if all_within_capture:
            # 一次性高奖励（由环境终止逻辑控制次数）
            capture_bonus = 5.0

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

    # 与其他追捕者的碰撞 / 近距离惩罚
    for j in adversary_indices:
        if j == agent_index:
            continue
        d_ij = inter_agent_distances[agent_index, j]
        min_dist = 2.0 * adv_size  # r_i + r_j
        safe_dist = safe_distance_factor * min_dist

        if d_ij < min_dist:
            # 硬碰撞
            collision_penalty -= 2.0
        elif d_ij < safe_dist:
            # 软惩罚：越接近越惩罚
            ratio = (safe_dist - d_ij) / (safe_dist - min_dist + 1e-8)
            near_collision_penalty -= 0.5 * ratio

    # 与猎物碰撞（可选：轻微惩罚，鼓励“包围”而不是硬撞）
    prey_index = None
    for idx, ag in enumerate(world.agents):
        if not getattr(ag, "adversary", False):
            prey_index = idx
            break
    if prey_index is not None:
        d_prey = inter_agent_distances[agent_index, prey_index]
        min_dist_ap = adv_size + prey_size
        # 如果发生物理碰撞，给一个小负奖励，让策略偏向困住而不是撞击
        if d_prey < min_dist_ap:
            collision_penalty -= 0.5

    components["collision_penalty"] = collision_penalty
    components["near_collision_penalty"] = near_collision_penalty

    # ------------------------
    # 3. 包围队形：均匀半径 + 均匀角度
    # ------------------------
    formation_radius_reward = 0.0
    formation_angle_reward = 0.0
    center_in_triangle_reward = 0.0

    if len(adversary_indices) == 3 and prey_index is not None:
        adv_positions = agent_positions[adversary_indices]  # (3, 2)
        # 半径（到猎物的距离）
        vecs = adv_positions - prey_pos[None, :]
        radii = np.linalg.norm(vecs, axis=1) + 1e-8

        # 半径均匀度：接近目标半径 & 方差小
        mean_r = float(np.mean(radii))
        var_r = float(np.var(radii))

        # 半径接近期望值奖励
        radius_dev = abs(mean_r - formation_radius_target)
        # 只在一定容差内给予正向奖励，否则奖励接近 0
        if mean_r < capture_threshold + formation_radius_tolerance:
            formation_radius_reward += -radius_dev / (capture_threshold + 1e-8)
        # 半径方差惩罚（越均匀越好）
        formation_radius_reward += -var_r / (capture_threshold ** 2 + 1e-8)

        # 角度均匀度：将猎物作为中心
        angles = np.arctan2(vecs[:, 1], vecs[:, 0])  # (-pi, pi]
        angles_sorted = np.sort(angles)
        # 环上角度间隔
        angle_diffs = np.diff(angles_sorted)
        # 加上首尾闭环差
        last_gap = (angles_sorted[0] + 2.0 * np.pi) - angles_sorted[-1]
        angle_diffs = np.concatenate([angle_diffs, [last_gap]])
        # 与理想值的偏差
        angle_dev = np.abs(angle_diffs - angle_uniform_target)
        angle_dev_mean = float(np.mean(angle_dev))
        # 归一化到 [0, 1] 左右
        formation_angle_reward = -angle_dev_mean / np.pi

        # 判定猎物是否在三追捕者三角形内部（几何包围）
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
            center_in_triangle_reward = 1.0

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
    # 鼓励各追捕者的速度方向并非完全一致，以便形成合围
    # 这里简单加一个速度方差惩罚（过于一致则略减分）
    adv_vels = agent_velocities[adversary_indices]
    if adv_vels.shape[0] > 1:
        speed_var = float(np.mean(np.var(adv_vels, axis=0)))
        components["velocity_diversity"] = speed_var * 0.05
    else:
        components["velocity_diversity"] = 0.0

    total_reward = float(sum(components.values()))
    return total_reward, components