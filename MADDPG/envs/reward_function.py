"""
可插拔的奖励函数模块 - 追捕者奖励函数
LLM将生成并替换此模块中的compute_reward函数
"""

import numpy as np


def compute_reward(agent_name, observation, global_state, actions, world):
    components = {}
    reward = _adversary_reward(agent_name, global_state, components)
    return reward, components


def _adversary_reward(agent_name, global_state, components):
    """
    追捕者奖励函数 V3.1 (Tuned for Stability)

    奖励分量：
    - elastic_ring_reward: 弹性环势场奖励
    - distance_penalty: 距离惩罚（在捕获范围内）
    - angle_penalty: 角度排斥惩罚
    - collision_penalty: 碰撞惩罚
    - global_cooperation_reward: 全局协作奖励
    - boundary_penalty: 边界惩罚

    设计要点：
    1. 使用弹性环势场：鼓励追捕者进入捕获圈
    2. 动态角度排斥：避免追捕者聚在一起
    3. 物理避撞：防止智能体碰撞
    4. 全局协作：奖励成功围捕
    """
    rew = 0

    # 提取信息
    world_size = global_state['world_size']
    capture_threshold = global_state['capture_threshold']
    ideal_radius = capture_threshold * 0.8
    cosine_threshold = 0.7

    agent_positions = global_state['agent_positions']
    adversary_indices = global_state['adversary_indices']
    prey_indices = global_state['prey_indices']

    # 获取当前智能体索引
    agent_idx = int(agent_name.split('_')[1])
    agent_pos = agent_positions[agent_idx]

    # 获取目标（猎物）
    prey_pos = global_state['prey_position']

    # 获取所有追捕者位置
    adversary_positions = agent_positions[adversary_indices]

    # --- 1. 弹性环势场 ---
    rel_vec = agent_pos - prey_pos
    dist = np.sqrt(np.sum(np.square(rel_vec)))

    if dist > capture_threshold:
        # 远距离：增加吸引力
        elastic_ring_reward = -3.0 * (dist - capture_threshold)
        rew += elastic_ring_reward
        components['elastic_ring_reward'] = elastic_ring_reward
    else:
        # 捕获范围内
        in_range_reward = 2.0
        rew += in_range_reward
        components['in_range_reward'] = in_range_reward

        # 距离惩罚
        distance_penalty = -0.5 * abs(dist - ideal_radius)
        rew += distance_penalty
        components['distance_penalty'] = distance_penalty

    # --- 2. 动态角度排斥 ---
    angle_penalty_total = 0.0
    if dist < capture_threshold * 2.0:
        agent_vec = rel_vec / (dist + 1e-6)

        for other_idx in adversary_indices:
            if other_idx == agent_idx:
                continue

            other_pos = agent_positions[other_idx]
            other_rel_vec = other_pos - prey_pos
            other_dist = np.sqrt(np.sum(np.square(other_rel_vec)))

            if other_dist < capture_threshold * 2.0:
                other_vec = other_rel_vec / (other_dist + 1e-6)
                cosine_sim = np.dot(agent_vec, other_vec)

                if cosine_sim > cosine_threshold:
                    proximity_factor = np.exp(-1.0 * min(dist, other_dist))
                    angle_penalty = 2.0 * (cosine_sim - cosine_threshold) * proximity_factor
                    angle_penalty_total += angle_penalty

    rew -= angle_penalty_total
    components['angle_penalty'] = -angle_penalty_total

    # --- 3. 物理避撞 ---
    collision_penalty_total = 0.0
    for other_idx in adversary_indices:
        if other_idx == agent_idx:
            continue

        other_pos = agent_positions[other_idx]
        delta_pos = agent_pos - other_pos
        dist_adv = np.sqrt(np.sum(np.square(delta_pos)))

        agent_size = world_size * 0.1 * 0.6
        dist_min = agent_size * 2 + 0.05

        if dist_adv < dist_min:
            collision_penalty = 10.0 * (dist_min - dist_adv)
            collision_penalty_total += collision_penalty

    rew -= collision_penalty_total
    components['collision_penalty'] = -collision_penalty_total

    # --- 4. 全局协作奖励 ---
    rel_vectors = [agent_positions[idx] - prey_pos for idx in adversary_indices]
    distances = [np.linalg.norm(vec) for vec in rel_vectors]

    global_cooperation_reward = 0.0
    if all(d < capture_threshold for d in distances):
        # 所有追捕者都在捕获范围内
        global_cooperation_reward += 1.0

        # 检查角度分布均匀性
        if len(adversary_indices) >= 3:
            angles = [np.arctan2(vec[1], vec[0]) for vec in rel_vectors]
            angles.sort()
            angle_diffs = []
            for i in range(len(angles)):
                next_idx = (i + 1) % len(angles)
                diff = angles[next_idx] - angles[i]
                if diff < 0:
                    diff += 2 * np.pi
                angle_diffs.append(diff)

            max_gap = max(angle_diffs) if len(angle_diffs) > 0 else 2 * np.pi
            if max_gap < np.pi:
                global_cooperation_reward += 4.0

    rew += global_cooperation_reward
    components['global_cooperation_reward'] = global_cooperation_reward

    # --- 5. 边界惩罚 ---
    boundary_penalty_total = 0.0
    for p in range(2):
        x = abs(agent_pos[p])
        boundary_penalty = _calculate_bound_penalty(x, world_size)
        boundary_penalty_total += boundary_penalty

    rew -= boundary_penalty_total
    components['boundary_penalty'] = -boundary_penalty_total

    return rew


def _calculate_bound_penalty(x, world_size):
    """
    计算边界惩罚

    Args:
        x: 坐标值的绝对值
        world_size: 世界大小

    Returns:
        float: 边界惩罚值
    """
    boundary_start = world_size * 0.96
    full_boundary = world_size

    if x < boundary_start:
        return 0
    if x < full_boundary:
        return (x - boundary_start) * 10
    return min(np.exp(2 * x - 2 * full_boundary), 10)

