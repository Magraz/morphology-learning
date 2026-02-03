import pickle
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.patches import Circle
from matplotlib.collections import LineCollection
from learning.environments.box2d_salp.testing.utils import plot_agent_overlays


def load_rollout_data(file_path):
    """Load rollout data from pickle file"""
    with open(file_path, "rb") as f:
        rollout_data = pickle.load(f)
    return rollout_data


def plot_rewards(rollout_data, save_path):
    """Plot rewards over time for each episode"""
    plt.figure(figsize=(12, 6))

    # For each episode
    for episode_idx, episode_data in enumerate(rollout_data):
        rewards = episode_data["reward"]
        steps = range(len(rewards))
        plt.plot(steps, rewards, label=f"Episode {episode_idx+1}")

    plt.xlabel("Step")
    plt.ylabel("Reward")
    plt.title("Rewards over Time")
    plt.legend()
    plt.grid(True, alpha=0.3)

    path = os.path.join(os.path.dirname(save_path), "rollout_rewards.png")
    plt.savefig(path, dpi=300)
    plt.close()


def plot_agent_trajectories(rollout_data):
    """Plot agent trajectories for each episode"""
    # Extract world bounds from the first info
    world_width = 80  # Default based on SalpChainEnv
    world_height = 60  # Default based on SalpChainEnv

    # Create a figure for each episode
    for episode_idx, episode_data in enumerate(rollout_data):
        infos = episode_data["info"]

        plt.figure(figsize=(12, 9))
        ax = plt.gca()

        # Setup plot bounds
        ax.set_xlim(0, world_width)
        ax.set_ylim(0, world_height)
        ax.set_aspect("equal")
        ax.set_title(f"Agent Trajectories - Episode {episode_idx+1}")
        ax.set_xlabel("X Position")
        ax.set_ylabel("Y Position")

        # For colorful trajectories
        cmap = plt.get_cmap("viridis")

        # Extract agent trajectories
        trajectories = {}

        for step, info in enumerate(infos):
            if "agent_positions" in info:
                agents = info["agent_positions"]
                for i, agent in enumerate(agents):
                    if i not in trajectories:
                        trajectories[i] = []
                    trajectories[i].append((agent["x"], agent["y"]))

        # Plot each agent's trajectory
        for agent_idx, trajectory in trajectories.items():
            color = cmap(agent_idx / max(1, len(trajectories) - 1))

            # Convert to numpy array for easier processing
            traj_array = np.array(trajectory)

            # Plot the agent's path with gradient color for time
            points = traj_array.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            norm = plt.Normalize(0, len(segments) - 1)
            lc = LineCollection(segments, cmap=cmap, norm=norm, alpha=0.7, linewidth=2)
            lc.set_array(np.arange(len(segments)))
            ax.add_collection(lc)

            # Mark start and end positions
            ax.plot(
                traj_array[0, 0],
                traj_array[0, 1],
                "o",
                color=color,
                markersize=8,
                label=f"Agent {agent_idx} Start",
            )
            ax.plot(
                traj_array[-1, 0],
                traj_array[-1, 1],
                "s",
                color=color,
                markersize=8,
                label=f"Agent {agent_idx} End",
            )

        # Plot target positions from the first info that has them
        for info in infos:
            if "target_positions" in info and info["target_positions"]:
                targets = info["target_positions"]
                for target in targets:
                    circle = Circle(
                        (target["x"], target["y"]),
                        target["radius"],
                        fill=True,
                        alpha=0.2,
                        color="red",
                    )
                    ax.add_patch(circle)
                    ax.text(
                        target["x"],
                        target["y"],
                        f"Req: {target['requirement']}",
                        ha="center",
                        va="center",
                        fontweight="bold",
                    )
                break

        plt.grid(True, alpha=0.3)
        plt.savefig(f"rollout_trajectory_episode_{episode_idx+1}.png", dpi=300)
        plt.close()


def plot_agent_positions_over_time(rollout_data):
    """Plot x,y positions of agents over time for each episode"""
    for episode_idx, episode_data in enumerate(rollout_data):
        infos = episode_data["info"]

        # Extract agent positions over time
        agent_positions = {}

        for step, info in enumerate(infos):
            if "agent_positions" in info:
                agents = info["agent_positions"]
                for i, agent in enumerate(agents):
                    if i not in agent_positions:
                        agent_positions[i] = {"x": [], "y": [], "steps": []}
                    agent_positions[i]["x"].append(agent["x"])
                    agent_positions[i]["y"].append(agent["y"])
                    agent_positions[i]["steps"].append(step)

        # Create separate plots for x and y positions
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

        # Plot x positions
        for agent_idx, data in agent_positions.items():
            ax1.plot(data["steps"], data["x"], label=f"Agent {agent_idx}")

        ax1.set_ylabel("X Position")
        ax1.set_title(f"Agent X Positions Over Time - Episode {episode_idx+1}")
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # Plot y positions
        for agent_idx, data in agent_positions.items():
            ax2.plot(data["steps"], data["y"], label=f"Agent {agent_idx}")

        ax2.set_xlabel("Step")
        ax2.set_ylabel("Y Position")
        ax2.set_title(f"Agent Y Positions Over Time - Episode {episode_idx+1}")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"rollout_positions_episode_{episode_idx+1}.png", dpi=300)
        plt.close()


def plot_individual_rewards(rollout_data, save_path):
    """Plot individual agent rewards over time"""
    for episode_idx, episode_data in enumerate(rollout_data):
        infos = episode_data["info"]

        # Extract individual rewards
        individual_rewards = {}

        for step, info in enumerate(infos):
            if "individual_rewards" in info:
                rewards = info["individual_rewards"]
                for agent_idx, reward in rewards.items():
                    if agent_idx not in individual_rewards:
                        individual_rewards[agent_idx] = {"reward": [], "step": []}
                    individual_rewards[agent_idx]["reward"].append(reward)
                    individual_rewards[agent_idx]["step"].append(step)

        if not individual_rewards:
            continue

        plt.figure(figsize=(12, 6))

        for agent_idx, data in individual_rewards.items():
            plt.plot(data["step"], data["reward"], label=f"Agent {agent_idx}")

        plt.xlabel("Step")
        plt.ylabel("Reward")
        plt.title(f"Individual Agent Rewards - Episode {episode_idx+1}")
        plt.grid(True, alpha=0.3)
        plt.legend()

        path = os.path.join(
            os.path.dirname(save_path),
            f"rollout_individual_rewards_episode_{episode_idx+1}.png",
        )

        plt.savefig(path, dpi=300)
        plt.close()


def plot_observation_distributions(rollout_data, save_path):
    """
    Plot the distribution of observation values throughout each episode.
    Shows how observation distributions change over time for each agent.

    Handles 3D observation array with shape (n_steps, n_agents, obs_dim)
    """
    for episode_idx, episode_data in enumerate(rollout_data):
        # Extract observations from this episode
        observations = episode_data.get("observation", [])

        # Convert observations to numpy array
        obs_array = np.array(observations)
        obs_dim = obs_array.shape[-1]

        for i in range(obs_dim):
            plot_agent_overlays(
                obs_array,
                dim=i,
                agent_ids=[0, 1],
                max_points=800,
                save_path=save_path,
                episode=episode_idx,
            )


def gaussian_kde_1d(data: np.ndarray, grid: np.ndarray = None):
    data = np.asarray(data, dtype=float)
    data = data[np.isfinite(data)]
    n = data.size
    if n == 0:
        raise ValueError("Empty data for KDE.")
    std = np.std(data, ddof=1) if n > 1 else 1.0
    # Silverman's rule of thumb
    bandwidth = 1.06 * std * (n ** (-1 / 5))
    if bandwidth <= 0:
        bandwidth = 1e-3

    if grid is None:
        pad = 3.0 * bandwidth
        grid = np.linspace(data.min() - pad, data.max() + pad, 512)

    # Compute Gaussian KDE manually (vectorized)
    diffs = (grid[:, None] - data[None, :]) / bandwidth
    kernel_vals = np.exp(-0.5 * diffs**2) / np.sqrt(2 * np.pi)
    density = kernel_vals.mean(axis=1) / bandwidth

    plt.figure()
    plt.plot(grid, density)
    plt.title(f"Gaussian KDE (Silverman's bw ≈ {bandwidth:.3g})")
    plt.xlabel("x")
    plt.ylabel("Density")
    plt.tight_layout()
    plt.show()


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, "rollout_info.pkl")

    if not os.path.exists(file_path):
        print(f"Error: Could not find rollout data at {file_path}")
        return

    rollout_data = load_rollout_data(file_path)

    # Print summary info
    print(f"Loaded rollout data with {len(rollout_data)} episodes")
    for i, episode in enumerate(rollout_data):
        print(f"Episode {i+1}: {len(episode['reward'])} steps")

    # Generate plots
    print("Creating plots...")
    plot_rewards(rollout_data, file_path)
    # plot_agent_trajectories(rollout_data)
    # plot_agent_positions_over_time(rollout_data)
    plot_observation_distributions(rollout_data, file_path)  # Add this line

    plot_individual_rewards(rollout_data, file_path)

    print("Plots saved!")


if __name__ == "__main__":
    main()
