from environments.box2d_salp.domain import SalpChainEnv
import numpy as np
import pickle
import os
import pygame


def dict_to_agent_actions(action_dict):
    """
    Convert dictionary of action components to agent-wise action array.

    Args:
        action_dict: Dictionary with action component names as keys and numpy arrays as values.
                    Each array has shape (n_agents,) or (n_agents, component_dim).
                    Example: {
                        'movement': np.array([[0.5, 0.3], [0.1, 0.2], [0.8, 0.4]]),  # (n_agents, 2)
                        'attach': np.array([1, 0, 1]),  # (n_agents,)
                        'detach': np.array([0, 1, 0])   # (n_agents,)
                    }

    Returns:
        np.ndarray: Array of shape (n_agents, total_action_dim) where each row is an agent's
                   complete action vector with movement first.
                   Example output shape: (3, 4) for the example above
                   Format: [movement_x, movement_y, attach, detach]
    """
    if not action_dict:
        raise ValueError("Action dictionary is empty")

    # Get number of agents from first component
    first_component = next(iter(action_dict.values()))
    n_agents = len(first_component)

    # Define the correct component order: movement should come first
    component_order = ["movement", "attach", "detach"]

    # Verify all expected components are present
    for key in component_order:
        if key not in action_dict:
            raise KeyError(
                f"Required action component '{key}' not found in action dictionary"
            )

    # Build action vectors for each agent
    agent_actions = []

    for agent_idx in range(n_agents):
        agent_action = []

        # Iterate through action components in the specified order
        for key in component_order:
            component = action_dict[key]

            # Handle both 1D and 2D components
            if component.ndim == 1:
                # Scalar component (e.g., attach, detach)
                agent_action.append(component[agent_idx])
            else:
                # Vector component (e.g., movement)
                agent_action.extend(component[agent_idx])

        agent_actions.append(agent_action)

    return np.array(agent_actions)


def get_target_seeking_action(env):
    """Generate actions that move each agent towards the nearest target"""

    # Extract target positions
    target_positions = np.array([(target.x, target.y) for target in env.target_areas])

    # Get current agent positions
    agent_positions = np.array(
        [[agent.position.x, agent.position.y] for agent in env.agents]
    )

    def find_nearest_target(agent_pos):
        distances = np.linalg.norm(target_positions - agent_pos, axis=1)
        nearest_idx = np.argmin(distances)
        return target_positions[nearest_idx], distances[nearest_idx]

    def calculate_movement_towards(from_pos, to_pos, max_force=1.0):
        direction = to_pos - from_pos
        distance = np.linalg.norm(direction)

        if distance < 0.01:
            return np.array([0.0, 0.0])

        normalized_direction = direction / distance

        # Scale force based on distance (stronger when further)
        return normalized_direction * max_force

    # Calculate movements toward nearest targets
    movements = []
    for i in range(env.n_agents):
        nearest_target, distance = find_nearest_target(agent_positions[i])
        movement = calculate_movement_towards(
            agent_positions[i], nearest_target, max_force=1.0
        )
        movements.append(movement)

    return np.array(movements)


def get_target_seeking_hybrid_action(env):
    """Generate actions that move each agent towards the nearest target"""

    # Get target positions from environment info
    if not hasattr(env, "target_areas") or not env.target_areas:
        # If no targets available, return random movement
        return {
            "movement": np.random.uniform(-0.5, 0.5, size=(env.n_agents, 2)),
            "attach": np.zeros(env.n_agents, dtype=np.int8),
            "detach": np.zeros(env.n_agents, dtype=np.int8),
        }

    # Extract target positions
    target_positions = np.array([(target.x, target.y) for target in env.target_areas])

    # Get current agent positions
    agent_positions = np.array(
        [[agent.position.x, agent.position.y] for agent in env.agents]
    )

    def find_nearest_target(agent_pos):
        distances = np.linalg.norm(target_positions - agent_pos, axis=1)
        nearest_idx = np.argmin(distances)
        return target_positions[nearest_idx], distances[nearest_idx]

    def calculate_movement_towards(from_pos, to_pos, max_force=1.0):
        direction = to_pos - from_pos
        distance = np.linalg.norm(direction)

        if distance < 0.01:
            return np.array([0.0, 0.0])

        normalized_direction = direction / distance

        # Scale force based on distance (stronger when further)
        force_scale = min(1.0, distance / 10.0)
        return normalized_direction * max_force * force_scale

    # Calculate movements toward nearest targets
    movements = []
    for i in range(env.n_agents):
        nearest_target, distance = find_nearest_target(agent_positions[i])
        movement = calculate_movement_towards(
            agent_positions[i], nearest_target, max_force=0.8
        )
        movements.append(movement)

    return {
        "movement": np.array(movements),
        "attach": np.zeros(env.n_agents, dtype=np.int8),  # Keep links open
        "detach": np.zeros(env.n_agents, dtype=np.int8),  # Don't detach
    }


def get_nearest_agent_action(env):
    """Generate actions that move each agent towards its nearest neighbor"""

    def find_nearest_agent(agent_idx, agent_positions):
        current_pos = agent_positions[agent_idx]
        min_distance = float("inf")
        nearest_idx = None

        for i, pos in enumerate(agent_positions):
            if i != agent_idx:
                distance = np.linalg.norm(current_pos - pos)
                if distance < min_distance:
                    min_distance = distance
                    nearest_idx = i

        return nearest_idx, min_distance

    def calculate_movement_towards(from_pos, to_pos, max_force=1.0):
        direction = to_pos - from_pos
        distance = np.linalg.norm(direction)

        if distance < 0.01:
            return np.array([0.0, 0.0])

        normalized_direction = direction / distance
        return normalized_direction * max_force

    # Get current agent positions
    agent_positions = np.array(
        [[agent.position.x, agent.position.y] for agent in env.agents]
    )

    # Calculate movements
    movements = []
    for i in range(env.n_agents):
        nearest_idx, distance = find_nearest_agent(i, agent_positions)

        if nearest_idx is not None:
            movement = calculate_movement_towards(
                agent_positions[i],
                agent_positions[nearest_idx],
                max_force=0.8,
            )
        else:
            movement = np.array([0.0, 0.0])

        movements.append(movement)

    return {
        "movement": np.array(movements),
        "attach": np.ones(env.n_agents, dtype=np.int8),
        "detach": np.zeros(env.n_agents, dtype=np.int8),
    }


def get_scatter_action(env):
    """Generate actions that move each agent away from all other agents and detach"""

    def calculate_repulsion_force(
        agent_pos, other_positions, max_force=1.0, repulsion_radius=5.0
    ):
        total_repulsion = np.array([0.0, 0.0])

        for other_pos in other_positions:
            relative_pos = agent_pos - other_pos
            distance = np.linalg.norm(relative_pos)

            if distance < 0.01:
                angle = np.random.uniform(0, 2 * np.pi)
                repulsion_direction = np.array([np.cos(angle), np.sin(angle)])
                total_repulsion += repulsion_direction
            else:
                repulsion_direction = relative_pos / distance

                if distance < repulsion_radius:
                    repulsion_strength = (
                        repulsion_radius - distance
                    ) / repulsion_radius
                    total_repulsion += repulsion_direction * repulsion_strength

        total_magnitude = np.linalg.norm(total_repulsion)
        if total_magnitude > 0:
            total_repulsion = (total_repulsion / total_magnitude) * max_force

        return total_repulsion

    agent_positions = np.array(
        [[agent.position.x, agent.position.y] for agent in env.agents]
    )

    movements = []
    for i in range(env.n_agents):
        other_positions = np.concatenate(
            [agent_positions[:i], agent_positions[i + 1 :]]
        )

        if len(other_positions) > 0:
            movement = calculate_repulsion_force(
                agent_positions[i], other_positions, max_force=0.8, repulsion_radius=4.0
            )
        else:
            movement = np.array([0.0, 0.0])

        movements.append(movement)

    return {
        "movement": np.array(movements),
        "attach": np.zeros(env.n_agents, dtype=np.int8),
        "detach": np.ones(env.n_agents, dtype=np.float32),
    }


def biased_sample(env, zero_prob=1.0):
    """Sample from action space with biased attach"""
    action = env.action_space.sample()
    random_values = np.random.random(env.n_agents)
    action["attach"] = (random_values > zero_prob).astype(np.int8)
    return action


def check_keyboard_input():
    """Check for keyboard input to switch action modes"""
    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_n:  # 'N' for nearest
                return "nearest"
            elif event.key == pygame.K_s:  # 'S' for scatter
                return "scatter"
            elif event.key == pygame.K_r:  # 'R' for random
                return "random"
            elif event.key == pygame.K_t:  # 'T' for target
                return "target"
            elif event.key == pygame.K_q:  # 'Q' for quit
                return "quit"
        elif event.type == pygame.QUIT:
            return "quit"
    return None


# Initialize environment
env = SalpChainEnv(render_mode="human", n_agents=2)

# Action mode control
action_mode = "nearest"  # Start with nearest-agent behavior

print("=" * 50)
print("CONTROLS:")
print("  N - Switch to Nearest Agent behavior")
print("  S - Switch to Scatter behavior")
print("  R - Switch to Random behavior")
print("  T - Switch to Target Seeking behavior")
print("  H - Switch to Target Seeking Hybrid behavior")
print("  Q - Quit simulation")
print("=" * 50)
print(f"Starting with '{action_mode.upper()}' action mode")

info_record = []
episode_record = {"observation": [], "reward": [], "info": []}

steps = 2000
episodes = 1
try:
    for episode in range(episodes):
        obs, _ = env.reset()

        for step in range(steps):
            # Check for keyboard input to switch modes
            key_input = check_keyboard_input()

            if key_input == "nearest":
                action_mode = "nearest"
                print(f"\n>>> Switched to NEAREST AGENT mode at step {step} <<<")
            elif key_input == "scatter":
                action_mode = "scatter"
                print(f"\n>>> Switched to SCATTER mode at step {step} <<<")
            elif key_input == "random":
                action_mode = "random"
                print(f"\n>>> Switched to RANDOM mode at step {step} <<<")
            elif key_input == "target":
                action_mode = "target"
                print(f"\n>>> Switched to TARGET SEEKING mode at step {step} <<<")
            elif key_input == "quit":
                print("\n>>> Quitting simulation <<<")
                raise KeyboardInterrupt

            # Choose action based on current mode
            if action_mode == "nearest":
                action = get_nearest_agent_action(env)
            elif action_mode == "scatter":
                action = get_scatter_action(env)
            elif action_mode == "target":
                action = get_target_seeking_hybrid_action(env)
            else:  # random mode
                action = biased_sample(env, zero_prob=0.7)

            act_array = dict_to_agent_actions(action)

            obs, reward, terminated, truncated, info = env.step(act_array)

            episode_record["observation"].append(obs)
            episode_record["reward"].append(reward)
            episode_record["info"].append(info)

            env.render()

            # Display current mode and step info
            if step % 25 == 0:
                mode_display = f"[{action_mode.upper()}]"
                print(
                    f"Step={step:3d} {mode_display:10s} Reward={reward} Reward Map={info["individual_rewards"]} Observation={obs}"
                )

            if terminated:
                print("TERMINATED")
                break

        info_record.append(episode_record)
        episode_record = {"observation": [], "reward": [], "info": []}

        print(f"Completed Episode {episode} in {action_mode.upper()} mode")

except KeyboardInterrupt:
    print("\nSimulation interrupted by user")

finally:
    env.close()

    # Save results
    output_dir = os.path.dirname(os.path.abspath(__file__))
    pickle_path = os.path.join(output_dir, "rollout_info.pkl")

    with open(pickle_path, "wb") as f:
        pickle.dump(info_record, f)

    print(f"Episode info saved to {pickle_path}")
