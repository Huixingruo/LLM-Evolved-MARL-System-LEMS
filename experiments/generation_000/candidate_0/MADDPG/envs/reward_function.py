import numpy as np


def compute_reward(agent_name, observation, global_state, actions, world):
    # 非追捕者不参与奖励
    if not global_state["is_adversary"]:
        return 0.0, {}

    components = {}

    # ----------------------
    # 物理与任务常量（硬编码）
    # ----------------------
    adv_size = 0.075
    prey_size = 0.050
    world_size = 2.5
    capture_threshold = 0.5

    # 包围圈相关几何常量
    expected_angle_gap = 2.0 * np.pi / 3.0  # 120°
    # 期望围捕半径（略小于 capture_threshold，避免贴脸碰撞）
    preferred_radius = capture_threshold * 0.8

    # 距离奖励缩放
    distance_scale = 1.0
    capture_bonus = 10.0

    # 队形角度与半径奖励缩放
    radius_var_scale = 1.0
    angle_var_scale = 1.0
    inside_triangle_bonus = 2.0

    # 防碰撞惩罚缩放
    collision_penalty = 5.0
    near_collision_penalty = 1.0
    safety_margin = 0.02  # 比碰撞半径略大一点视为危险距离

    # 时间惩罚（鼓励尽快完成）
    time_penalty = 0.01

    # ----------------------
    # 从 global_state 解析
    # ----------------------
    agent_positions = global_state["agent_positions"]
    prey_pos = global_state["prey_position"]
    distances_to_prey = global_state["distances_to_prey"]
    inter_agent_distances = global_state["inter_agent_distances"]

    # 根据约定：前若干个 adversary，后面是 prey
    # 这里假定只有一个 prey
    num_agents = agent_positions.shape[0]

    # 找到各类索引
    # 在 MPE 中通常 adversary 有标志，但 global_state 不给出逐 agent 标志，只给当前 agent 是否 adversary。
    # 我们从距离向量长度推断 adversary 个数 = len(distances_to_prey)
    num_adversaries = len(distances_to_prey)
    adversary_indices = list(range(num_adversaries))
    prey_index = num_adversaries  # 假设单个猎物，紧随其后

    # 当前 agent 索引（根据名字末尾数字推断，如 'adversary_0'）
    # 若解析失败则退化为第一个 adversary
    try:
        agent_idx = int(agent_name.split("_")[-1])
    except Exception:
        agent_idx = 0
    agent_idx = max(0, min(agent_idx, num_adversaries - 1))

    agent_pos = agent_positions[agent_idx]
    prey_position = prey_pos

    # ----------------------
    # 距离引导：靠近并捕获目标
    # ----------------------
    # 当前追捕者到猎物距离
    agent_to_prey_dist = np.linalg.norm(agent_pos - prey_position)

    # 1) 距离惩罚（越近越好）
    # 归一化距离到 [0, 1~]，给负奖励
    norm_dist = agent_to_prey_dist / world_size
    components["distance_reward"] = -distance_scale * norm_dist

    # 2) 围捕成功奖励：所有追捕者都在 capture_threshold 内
    all_within_capture = bool(
        np.all(distances_to_prey <= capture_threshold)
    )
    if all_within_capture:
        components["capture_bonus"] = capture_bonus
    else:
        components["capture_bonus"] = 0.0

    # ----------------------
    # 包围队形：半径均匀 + 角度均匀 + 猎物在三角形内部
    # ----------------------
    # 仅在存在足够多的 adversary 且存在 prey 时计算
    if num_adversaries >= 3 and num_agents > prey_index:
        adv_positions = agent_positions[adversary_indices]

        # 半径均匀：追捕者到猎物距离的方差 + 与期望半径的偏差
        radii = np.linalg.norm(adv_positions - prey_position, axis=1)
        if np.all(radii > 1e-6):
            radius_var = np.var(radii)
            mean_radius = np.mean(radii)
            # 惩罚半径差异和偏离期望半径
            radius_penalty = radius_var + (mean_radius - preferred_radius) ** 2
            components["radius_uniformity"] = -radius_var_scale * radius_penalty
        else:
            components["radius_uniformity"] = 0.0

        # 角度均匀：相邻极角间距接近 2pi/3
        rel_vecs = adv_positions - prey_position
        angles = np.arctan2(rel_vecs[:, 1], rel_vecs[:, 0])
        # 归一化到 [0, 2pi)
        angles = (angles + 2.0 * np.pi) % (2.0 * np.pi)
        angles_sorted = np.sort(angles)
        # 环绕角度差
        angle_gaps = np.diff(angles_sorted, append=angles_sorted[0] + 2.0 * np.pi)
        angle_gap_var = np.var(angle_gaps)
        # 惩罚角间距方差及与期望值偏离
        angle_gap_mean = np.mean(angle_gaps)
        angle_penalty = angle_gap_var + (angle_gap_mean - expected_angle_gap) ** 2
        components["angle_uniformity"] = -angle_var_scale * angle_penalty

        # 猎物是否在三角形内部（Barycentric 方法）
        # 使用前三个追捕者形成三角形
        a, b, c = adv_positions[0], adv_positions[1], adv_positions[2]
        v0 = c - a
        v1 = b - a
        v2 = prey_position - a

        denom = v0[0] * v1[1] - v1[0] * v0[1]
        if abs(denom) > 1e-8:
            inv_denom = 1.0 / denom
            u = (v2[0] * v1[1] - v1[0] * v2[1]) * inv_denom
            v = (v0[0] * v2[1] - v2[0] * v0[1]) * inv_denom
            w = 1.0 - u - v
            inside = (
                (u >= 0.0) and (v >= 0.0) and (w >= 0.0)
            )
        else:
            # 面积过小视为未成形三角形
            inside = False

        components["prey_inside_triangle"] = inside_triangle_bonus if inside else 0.0
    else:
        components["radius_uniformity"] = 0.0
        components["angle_uniformity"] = 0.0
        components["prey_inside_triangle"] = 0.0

    # ----------------------
    # 防碰撞：追捕者-追捕者间距 & 追捕者-猎物距离（避免硬碰撞）
    # ----------------------
    # 1) 追捕者-追捕者碰撞与近距离惩罚
    min_safe_dist = 2.0 * adv_size + safety_margin
    agent_collision_penalty = 0.0
    near_collision_cost = 0.0

    for other_idx in adversary_indices:
        if other_idx == agent_idx:
            continue
        d = inter_agent_distances[agent_idx, other_idx]
        if d < (2.0 * adv_size):  # 碰撞
            agent_collision_penalty -= collision_penalty
        elif d < min_safe_dist:
            # 近距离，惩罚随距离缩小增加
            near_collision_cost -= near_collision_penalty * (min_safe_dist - d) / min_safe_dist

    components["collision_penalty"] = agent_collision_penalty
    components["near_collision_penalty"] = near_collision_cost

    # 2) 追捕者-猎物硬碰撞惩罚（避免全员贴脸冲撞）
    # 对当前追捕者单独判断
    prey_collision_dist = adv_size + prey_size
    if agent_to_prey_dist < prey_collision_dist:
        components["prey_collision_penalty"] = -collision_penalty * 0.5
    else:
        components["prey_collision_penalty"] = 0.0

    # ----------------------
    # 时间惩罚（每步小负奖励，鼓励尽快完成）
    # ----------------------
    components["time_penalty"] = -time_penalty

    total_reward = float(sum(components.values()))
    return total_reward, components