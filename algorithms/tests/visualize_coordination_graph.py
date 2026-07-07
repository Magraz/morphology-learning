"""Visualize the per-head coordination graphs produced by the attention-GNN
critic during a rollout of a learned MAPPO policy.

Runs one deterministic episode with the trained policy and, at evenly spaced
snapshot timesteps, plots the environment frame next to the critic's per-head
coordination graphs. Each graph's nodes are the agents (placed at their world
positions so the graph lines up spatially with the frame) and the edge weights
are the symmetric attention adjacency that the GNN critic convolves over.

The coordination graph is read from the *behavior* network (``network_old``)
under ``no_grad`` via ``AttentionGNNCritic`` — exactly the encoder the value
path and the novelty bonus use, so the drawn graph is the grounded one.

Usage:
    SDL_VIDEODRIVER=dummy python -m algorithms.tests.visualize_coordination_graph \
        --model experiments/results/multi_box_push_9a_3o/cg_team_novelty/2/models/models_finished.pth \
        --config experiments/yamls/multi_box_push_9a_3o/cg_team_novelty.yaml \
        --env experiments/yamls/multi_box_push_9a_3o/_env.yaml \
        --out coordination_graphs.png

Defaults point at the cg_team_novelty trial 2 model, so a bare
``SDL_VIDEODRIVER=dummy python -m algorithms.tests.visualize_coordination_graph``
just works.
"""

import argparse
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from algorithms.create_env import make_vec_env  # noqa: E402
from algorithms.mappo.mappo import MAPPOAgent  # noqa: E402
from algorithms.mappo.types import MAPPO_Params, Model_Params  # noqa: E402
from algorithms.utils import set_global_seeds  # noqa: E402
from environments.types import EnvironmentEnum  # noqa: E402

DEFAULTS = {
    f"model": "experiments/results/multi_box_push_9a_3o/cg_team_novelty/2/models/models_finished.pth",
    f"config": "experiments/yamls/multi_box_push_9a_3o/cg_team_novelty.yaml",
    f"env": "experiments/yamls/multi_box_push_9a_3o/_env.yaml",
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=DEFAULTS["model"], help="path to models_*.pth")
    p.add_argument("--config", default=DEFAULTS["config"], help="experiment yaml")
    p.add_argument("--env", default=DEFAULTS["env"], help="_env.yaml")
    p.add_argument("--out", default="coordination_graphs.png", help="output figure")
    p.add_argument(
        "--snapshots", type=int, default=5, help="snapshot timesteps to plot"
    )
    p.add_argument(
        "--max-steps", type=int, default=0, help="cap rollout length (0=env max)"
    )
    p.add_argument("--seed", type=int, default=0, help="rollout seed")
    p.add_argument(
        "--edge-threshold",
        type=float,
        default=0.05,
        help="hide edges with adjacency weight below this",
    )
    p.add_argument(
        "--show-labels",
        action="store_true",
        help="annotate each drawn edge with its weight value",
    )
    p.add_argument("--device", default="cpu")
    return p.parse_args()


def load_configs(config_path, env_path, environment):
    with open(config_path) as f:
        exp = yaml.unsafe_load(f)
    with open(env_path) as f:
        env_config = yaml.safe_load(f)
    env_config["environment"] = environment
    params = MAPPO_Params(**exp["params"])
    model_params = Model_Params(**exp["model_params"])
    return params, model_params, env_config


def build_agent(model_path, params, model_params, env_config, device):
    """Construct the agent exactly like VecMAPPOTrainer and load the weights."""
    probe = make_vec_env(
        env_config["environment"],
        env_config["n_agents"],
        1,
        use_async=False,
        env_params=env_config,
    )
    obs_space = probe.single_observation_space
    act_space = probe.single_action_space
    observation_dim = obs_space.shape[1]
    n_agents = env_config["n_agents"]
    global_state_dim = observation_dim * n_agents
    action_dim = act_space.shape[1]
    probe.close()

    agent = MAPPOAgent(
        observation_dim,
        global_state_dim,
        action_dim,
        n_agents,
        params,
        device,
        False,  # discrete=False for the box2d continuous-control envs
        1,  # n_parallel_envs
        model_params=model_params,
    )

    ckpt = torch.load(model_path, map_location=device)
    agent.network.load_state_dict(ckpt["network"])
    agent.network_old.load_state_dict(ckpt["network"])
    agent.network_old.eval()
    agent.network.eval()
    assert (
        agent.critic_type == "gnn"
    ), f"this visualizer needs the gnn critic; got {agent.critic_type!r}"
    return agent, observation_dim, n_agents


def make_render_env(env_config):
    """Single MULTI_BOX env with render_mode='human' so we can grab pygame frames."""
    assert (
        env_config["environment"] == EnvironmentEnum.MULTI_BOX
    ), "frame capture is wired for the MULTI_BOX env"
    from environments.box2d_suite.multi_box_push import MultiBoxPushEnv

    return MultiBoxPushEnv(
        n_agents=env_config["n_agents"],
        n_objects=env_config["n_objects"],
        render_mode="human",
        reward_mode=env_config.get("reward_mode"),
    )


def grab_frame():
    import pygame

    surface = pygame.display.get_surface()
    if surface is None:
        return None
    frame = pygame.surfarray.array3d(surface)
    return np.transpose(frame, (1, 0, 2)).copy()


def coordination_adjacency(agent, obs):
    """Per-head symmetric coordination graph from the behavior critic encoder.

    Args:
        obs: (n_agents, obs_dim) numpy for one env.
    Returns:
        (n_heads, n_agents, n_agents) numpy adjacency.
    """
    with torch.no_grad():
        obs_t = torch.from_numpy(np.ascontiguousarray(obs, dtype=np.float32)).to(
            agent.device
        )
        _, adjacency = agent.network_old.critic.encoder(obs_t)  # (n_heads, N, N)
    return adjacency.cpu().numpy()


def rollout(agent, env, n_agents, seed, max_steps):
    """Deterministic episode; record frames and per-head coordination adjacency."""
    obs, _ = env.reset(seed=seed)
    env.render()

    frames, adjacencies, cum_rewards = [], [], []
    cum = 0.0
    step = 0
    cap = max_steps if max_steps > 0 else env.max_steps
    while True:
        adjacencies.append(coordination_adjacency(agent, obs))

        obs_batch = obs[np.newaxis]  # (1, n_agents, obs_dim)
        global_states = obs.reshape(1, -1)
        actions_t, _, _ = agent.get_actions_batched(
            obs_batch, global_states, deterministic=True
        )
        actions = actions_t.cpu().numpy()[0]  # (n_agents, action_dim)

        obs, reward, terminated, truncated, _ = env.step(actions)
        env.render()
        frames.append(grab_frame())
        cum += float(reward)
        cum_rewards.append(cum)

        step += 1
        if terminated or truncated or step >= cap:
            break

    env.close()
    return frames, adjacencies, cum_rewards


def _circle_layout(n):
    """Fixed node positions on a unit circle (layout independent of agent state)."""
    angles = np.linspace(np.pi / 2, np.pi / 2 + 2 * np.pi, n, endpoint=False)
    return np.stack([np.cos(angles), np.sin(angles)], axis=1)


def draw_graph(ax, adj, cmap, vmax, threshold, show_labels):
    """Draw one head's coordination graph on a fixed circular layout.

    The layout carries no meaning — node placement is constant across heads and
    timesteps so the only thing that varies is the edge set and edge weights,
    which are mapped to both color (``cmap`` over ``[0, vmax]``) and width.
    """
    n = adj.shape[0]
    pos = _circle_layout(n)
    for i in range(n):
        for j in range(i + 1, n):
            wgt = float(adj[i, j])
            if wgt < threshold:
                continue
            frac = wgt / vmax if vmax > 0 else 0.0
            ax.plot(
                [pos[i, 0], pos[j, 0]],
                [pos[i, 1], pos[j, 1]],
                color=cmap(np.clip(frac, 0.0, 1.0)),
                linewidth=0.5 + 5.0 * np.clip(frac, 0.0, 1.0),
                zorder=1,
            )
            if show_labels:
                mx, my = (pos[i] + pos[j]) / 2
                ax.text(
                    mx,
                    my,
                    f"{wgt:.2f}",
                    ha="center",
                    va="center",
                    fontsize=6,
                    color="black",
                    zorder=4,
                    bbox=dict(
                        boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.6
                    ),
                )
    ax.scatter(
        pos[:, 0],
        pos[:, 1],
        s=300,
        c="white",
        edgecolors="black",
        linewidths=1.5,
        zorder=2,
    )
    for i in range(n):
        ax.text(
            pos[i, 0],
            pos[i, 1],
            str(i),
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
            zorder=3,
        )
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


def build_figure(frames, adjacencies, cum_rewards, n_snapshots, threshold, show_labels):
    total = len(frames)
    if total == 0:
        raise RuntimeError("no frames captured — check SDL_VIDEODRIVER")
    n_heads = adjacencies[0].shape[0]
    indices = np.linspace(0, total - 1, min(n_snapshots, total), dtype=int)

    # Shared color/width scale across all heads and snapshots so edge weights
    # are directly comparable. Ignore the self-loop diagonal.
    n = adjacencies[0].shape[-1]
    off_diag = ~np.eye(n, dtype=bool)
    vmax = max(float(adjacencies[idx][:, off_diag].max()) for idx in indices)
    vmax = vmax if vmax > 0 else 1.0
    cmap = matplotlib.colormaps["viridis"]
    norm = matplotlib.colors.Normalize(vmin=0.0, vmax=vmax)

    n_cols = 1 + n_heads
    fig, axes = plt.subplots(
        len(indices),
        n_cols,
        figsize=(3.2 * n_cols, 3.4 * len(indices)),
        squeeze=False,
        constrained_layout=True,
    )

    for row, idx in enumerate(indices):
        axes[row, 0].imshow(frames[idx])
        axes[row, 0].set_ylabel(
            f"t = {idx}\nreturn = {cum_rewards[idx]:.2f}", fontsize=10
        )
        axes[row, 0].set_xticks([])
        axes[row, 0].set_yticks([])
        if row == 0:
            axes[row, 0].set_title("environment")

        for head in range(n_heads):
            ax = axes[row, 1 + head]
            draw_graph(ax, adjacencies[idx][head], cmap, vmax, threshold, show_labels)
            if row == 0:
                ax.set_title(f"head {head}")

    fig.suptitle("Attention-GNN critic coordination graphs", fontsize=14)
    sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(
        sm, ax=axes.ravel().tolist(), orientation="horizontal", fraction=0.04, aspect=50
    )
    cbar.set_label("edge weight (symmetric attention adjacency)")
    return fig


def main():
    args = parse_args()
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    set_global_seeds(args.seed)

    params, model_params, env_config = load_configs(
        args.config, args.env, EnvironmentEnum.MULTI_BOX
    )
    agent, _, n_agents = build_agent(
        args.model, params, model_params, env_config, args.device
    )

    env = make_render_env(env_config)
    frames, adjacencies, cum_rewards = rollout(
        agent, env, n_agents, args.seed, args.max_steps
    )
    print(f"rollout: {len(frames)} steps, final return {cum_rewards[-1]:.3f}")

    fig = build_figure(
        frames,
        adjacencies,
        cum_rewards,
        args.snapshots,
        args.edge_threshold,
        args.show_labels,
    )
    out = Path(args.out)
    fig.savefig(out, dpi=130)
    print(f"saved figure to {out.resolve()}")


if __name__ == "__main__":
    main()
