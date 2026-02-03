import pickle
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.patches import Circle
from matplotlib.colors import to_rgba

MAP_X = 60
MAP_Y = 60


def plot_all_target_positions(pickle_path):
    """
    Plot all target positions from all episodes on a single plot
    with different colors for each episode.
    """
    # Load the pickle file
    with open(pickle_path, "rb") as f:
        info_record = pickle.load(f)

    # Create figure and axes for a single plot
    fig, ax = plt.subplots(figsize=(12, 9))

    # Setup the plot
    ax.set_xlim(0, MAP_X)  # Match world width from environment
    ax.set_ylim(0, MAP_Y)  # Match world height from environment
    ax.set_aspect("equal")
    ax.set_title("Target Positions Across All Episodes")
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")

    # Get colormap for different episodes
    cmap = plt.get_cmap("tab10")

    # Track unique target positions to avoid excessive overlapping
    unique_targets = {}

    # Plot targets from all episodes
    for episode_idx, episode_info in enumerate(info_record):
        # Get episode color
        episode_color = cmap(episode_idx % 10)

        # Add episode to legend once
        ax.plot([], [], "o", color=episode_color, label=f"Episode {episode_idx+1}")

        # Get target positions from this episode
        targets = episode_info["info"][0]["target_positions"]

        # Plot each target
        for target in targets:
            x = target["x"]
            y = target["y"]
            radius = target["radius"]
            requirement = target["requirement"]

            # Generate a target key based on position and requirement
            target_key = f"{x:.1f}_{y:.1f}_{requirement}"

            # Skip if we've already plotted a very similar target
            skip = False
            for key in unique_targets:
                existing_x, existing_y = unique_targets[key]
                if np.sqrt((x - existing_x) ** 2 + (y - existing_y) ** 2) < 1.0:
                    skip = True
                    break

            if skip:
                continue

            unique_targets[target_key] = (x, y)

            # Draw circle for target area with episode-specific color
            # Adjust alpha for better visibility of overlapping targets
            fill_color = to_rgba(episode_color, alpha=0.3)
            edge_color = to_rgba(episode_color, alpha=0.8)

            circle = Circle(
                (x, y),
                radius,
                fill=True,
                facecolor=fill_color,
                edgecolor=edge_color,
                linewidth=2,
            )
            ax.add_patch(circle)

            # Add text for coupling requirement
            ax.text(
                x,
                y,
                f"{episode_idx+1}",
                ha="center",
                va="center",
                fontweight="bold",
                color="black",
            )

    # Add a legend
    ax.legend(title="Episode", loc="upper right")

    # Add grid for better readability
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save the plot
    save_path = os.path.join(os.path.dirname(pickle_path), "all_target_positions.png")
    plt.savefig(save_path, dpi=300)

    print(f"Plot saved to {save_path}")
    plt.show()


def plot_all_agent_positions(pickle_path):
    """
    Plot all agent positions from all episodes on a single plot
    with different colors for each episode.
    """
    # Load the pickle file
    with open(pickle_path, "rb") as f:
        info_record = pickle.load(f)

    # Create figure and axes for a single plot
    fig, ax = plt.subplots(figsize=(12, 9))

    # Setup the plot
    ax.set_xlim(0, MAP_X)  # Match world width from environment
    ax.set_ylim(0, MAP_Y)  # Match world height from environment
    ax.set_aspect("equal")
    ax.set_title("Agent Positions Across All Episodes")
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")

    # Get colormap for different episodes
    cmap = plt.get_cmap("tab10")

    # Plot agents from all episodes
    for episode_idx, episode_info in enumerate(info_record):
        # Get episode color
        episode_color = cmap(episode_idx % 10)

        # Add episode to legend once
        ax.plot([], [], "o", color=episode_color, label=f"Episode {episode_idx+1}")

        # Get agent positions from this episode
        agents = episode_info["info"][0]["agent_positions"]

        # Extract x and y coordinates
        x_coords = [agent["x"] for agent in agents]
        y_coords = [agent["y"] for agent in agents]

        # Plot agents as scatter points
        ax.scatter(
            x_coords,
            y_coords,
            color=episode_color,
            alpha=0.7,
            s=50,  # Size of markers
            edgecolors="black",
            linewidths=0.5,
        )

        # Optional: Draw a light convex hull around agents from the same episode
        # if len(x_coords) > 2:  # Need at least 3 points for a convex hull
        #     try:
        #         from scipy.spatial import ConvexHull

        #         points = np.column_stack((x_coords, y_coords))
        #         hull = ConvexHull(points)
        #         hull_vertices = np.append(hull.vertices, hull.vertices[0])
        #         hull_x = [points[vertex, 0] for vertex in hull_vertices]
        #         hull_y = [points[vertex, 1] for vertex in hull_vertices]
        #         ax.plot(hull_x, hull_y, "--", color=episode_color, alpha=0.4)
        #     except:
        #         # Skip convex hull if there's an error (e.g., collinear points)
        #         pass

    # Add a legend
    ax.legend(title="Episode", loc="upper right")

    # Draw a center point representing the world center
    world_center = (MAP_X / 2, MAP_Y / 2)  # Based on world width and height of 80
    ax.plot(world_center[0], world_center[1], "k+", markersize=12)

    # Draw a circle at min_center_radius (assumed to be 15 based on code review)
    center_radius = 15
    center_circle = Circle(
        world_center, center_radius, fill=False, linestyle="--", color="gray"
    )
    ax.add_patch(center_circle)

    # Add grid for better readability
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save the plot
    save_path = os.path.join(os.path.dirname(pickle_path), "all_agent_positions.png")
    plt.savefig(save_path, dpi=300)

    print(f"Plot saved to {save_path}")
    plt.show()


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pickle_path = os.path.join(script_dir, "rollout_info.pkl")

    # Plot both target and agent positions
    plot_all_target_positions(pickle_path)
    plot_all_agent_positions(pickle_path)
