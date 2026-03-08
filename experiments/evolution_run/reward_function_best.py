import numpy as np


def compute_reward(agent_name, observation, global_state, actions, world):
    components = {}

    # Parameters from global state
    world_size = global_state.get("world_size", 2.5)
    capture_threshold = global_state.get("capture_threshold", 0.5)
    agent_positions = global_state["agent_positions"]
    agent_velocities = global_state["agent_velocities"]
    prey_pos = global_state["prey_position"]
    is_adversary = global_state["is_adversary"]
    inter_agent_distances = global_state["inter_agent_distances"]

    # Optional indices (if provided upstream)
    adversary_indices = global_state.get(
        "adversary_indices",
        [i for i, a in enumerate(world.agents) if a.adversary],
    )
    prey_indices = global_state.get(
        "prey_indices",
        [i for i, a in enumerate(world.agents) if not a.adversary],
    )

    # Default zero reward for non-adversary (prey)
    if not is_adversary:
        components["neutral_reward"] = 0.0
        return 0.0, components

    # Identify current agent index
    name_to_index = {a.name: i for i, a in enumerate(world.agents)}
    if agent_name not in name_to_index:
        components["invalid_agent_penalty"] = -1.0
        return -1.0, components
    self_idx = name_to_index[agent_name]

    self_pos = agent_positions[self_idx]
    self_vel = agent_velocities[self_idx]

    # Distance to prey
    dist_to_prey = np.linalg.norm(self_pos - prey_pos)

    # 1. Nonlinear distance shaping: stronger gradient near prey,
    # weaker penalty when far, to avoid always-large negatives
    # Normalize distance by world size
    dist_norm = dist_to_prey / world_size
    # Use smooth bounded shaping: reward in approx [-1, 0]
    # Closer distance -> value closer to 0, far -> ~-1
    distance_shaping = -dist_norm / (dist_norm + 1.0 + 1e-8)
    components["distance_reward"] = 0.7 * distance_shaping

    # 2. Capture bonus: stronger terminal positive reward
    capture_bonus = 0.0
    if dist_to_prey < capture_threshold:
        # Stronger than previous generation
        capture_bonus = 3.0
    components["capture_bonus"] = capture_bonus

    # 3. Team capture synergy: big bonus if all adversaries are close
    team_capture_bonus = 0.0
    if adversary_indices:
        adv_dists = [
            np.linalg.norm(agent_positions[i] - prey_pos)
            for i in adversary_indices
        ]
        # All within threshold -> strong team success signal
        if all(d < capture_threshold for d in adv_dists):
            team_capture_bonus = 6.0
    components["team_capture_bonus"] = team_capture_bonus

    # 4. Collision penalty between adversaries
    # Keep but with moderate strength; activate when very close
    collision_penalty = 0.0
    min_safe_dist = 0.15
    for i in adversary_indices:
        if i == self_idx:
            continue
        d = inter_agent_distances[self_idx, i]
        if d < min_safe_dist:
            collision_penalty -= (min_safe_dist - d) * 3.0
    components["collision_penalty"] = collision_penalty

    # 5. Formation reward: encourage evenly spaced encirclement
    formation_reward = 0.0
    if len(adversary_indices) >= 3:
        adv_positions = np.array(
            [agent_positions[i] for i in adversary_indices]
        )
        rel_positions = adv_positions - prey_pos
        angles = np.arctan2(rel_positions[:, 1], rel_positions[:, 0])
        angles = np.mod(angles, 2 * np.pi)
        angles_sorted = np.sort(angles)
        gaps = np.diff(
            np.concatenate(
                [angles_sorted, angles_sorted[:1] + 2 * np.pi]
            )
        )
        ideal_gap = 2 * np.pi / len(adversary_indices)
        gap_var = np.var(gaps / (ideal_gap + 1e-8))

        # Basic shaping: variance penalty
        base_formation = -gap_var

        # Extra team bonus when formation very good
        extra_bonus = 0.0
        if gap_var < 0.1:
            extra_bonus = 1.0

        formation_reward = 0.3 * base_formation + extra_bonus
    components["formation_reward"] = formation_reward

    # 6. Radial alignment reward: agents on similar radius when close
    radial_reward = 0.0
    if len(adversary_indices) >= 3:
        adv_dists = np.array(
            [
                np.linalg.norm(agent_positions[i] - prey_pos)
                for i in adversary_indices
            ]
        )
        mean_r = np.mean(adv_dists)
        if mean_r > 1e-3:
            norm_dists = adv_dists / (mean_r + 1e-8)
            radial_var = np.var(norm_dists)
            base_radial = -radial_var

            # Extra reward when all are within a ring around prey
            # Ring roughly at capture_threshold to 1.5 * capture_threshold
            in_ring = np.logical_and(
                adv_dists > 0.7 * capture_threshold,
                adv_dists < 1.5 * capture_threshold,
            )
            ring_team_bonus = 0.0
            if np.all(in_ring):
                ring_team_bonus = 1.5

            radial_reward = 0.4 * base_radial + ring_team_bonus
    components["radial_reward"] = radial_reward

    # 7. Boundary penalty: smoother and slightly weaker
    boundary_margin = 0.1 * world_size
    x, y = self_pos
    overflow_x = max(0.0, abs(x) - (world_size - boundary_margin))
    overflow_y = max(0.0, abs(y) - (world_size - boundary_margin))
    raw_boundary_penalty = overflow_x + overflow_y
    # Clip maximum penalty per step
    boundary_penalty = -min(raw_boundary_penalty * 1.5, 2.0)
    components["boundary_penalty"] = boundary_penalty

    # 8. Energy penalty: slightly weaker to encourage activity
    action = actions.get(agent_name, np.zeros_like(self_vel))
    action = np.asarray(action)
    energy_penalty = -0.03 * float(np.linalg.norm(action))
    components["energy_penalty"] = energy_penalty

    # 9. Velocity alignment: encourage moving toward prey
    vel_alignment_reward = 0.0
    if np.linalg.norm(self_vel) > 1e-6 and dist_to_prey > 1e-6:
        dir_to_prey = (prey_pos - self_pos) / dist_to_prey
        vel_dir = self_vel / (np.linalg.norm(self_vel) + 1e-8)
        cos_sim = np.dot(dir_to_prey, vel_dir)
        vel_alignment_reward = 0.15 * cos_sim
    components["velocity_alignment_reward"] = vel_alignment_reward

    # 10. Group cohesion: moderate, to keep group together
    cohesion_reward = 0.0
    if len(adversary_indices) >= 2:
        adv_positions = np.array(
            [agent_positions[i] for i in adversary_indices]
        )
        centroid = np.mean(adv_positions, axis=0)
        dists_to_centroid = np.linalg.norm(
            adv_positions - centroid, axis=1
        )
        spread = np.mean(dists_to_centroid) / world_size
        cohesion_reward = -0.08 * spread
    components["cohesion_reward"] = cohesion_reward

    # 11. Time efficiency: small step penalty to encourage faster capture
    # (shaped as small constant negative reward each step)
    time_penalty = -0.005
    components["time_penalty"] = time_penalty

    total_reward = float(sum(components.values()))
    return total_reward, components