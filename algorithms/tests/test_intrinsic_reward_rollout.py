"""Run a single episode of MultiBoxPushEnv and log team + agent intrinsic rewards."""

import argparse
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import torch
from algorithms.mappo.hypergraph import (
    build_hypergraph,
    distance_based_hyperedges,
    object_contact_hyperedges,
)
from algorithms.mappo.intrinsic_reward import IntrinsicReward
from algorithms.mappo.networks.encoders import (
    HypergraphStateEncoder,
    LocalStateEncoder,
)
from algorithms.mappo.networks.models import MAPPONetwork
from environments.multi_box_push.domain import MultiBoxPushEnv


def load_from_checkpoint(checkpoint_path, n_agents, device="cpu"):
    """Load a MAPPONetwork and optional LocalStateEncoder from a checkpoint.

    Infers network dimensions from the saved state dict.

    Returns:
        (network, encoder, encoder_dim)
        - network: MAPPONetwork with loaded weights (eval mode)
        - encoder: LocalStateEncoder or None
        - encoder_dim: int or None
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    sd = checkpoint["network"]

    # Infer dimensions from actor weights
    share_actor = "actor.actor.0.weight" in sd
    if share_actor:
        obs_dim = sd["actor.actor.0.weight"].shape[1]
        action_dim = sd["actor.actor.4.weight"].shape[0]
        hidden_dim = sd["actor.actor.0.weight"].shape[0]
    else:
        obs_dim = sd["actors.0.actor.0.weight"].shape[1]
        action_dim = sd["actors.0.actor.4.weight"].shape[0]
        hidden_dim = sd["actors.0.actor.0.weight"].shape[0]

    # Detect critic type: MLP critic has "critic.critic.0.weight",
    # multi_hgnn critic has "critic.critics.0..." keys instead.
    critic_type = "mlp" if "critic.critic.0.weight" in sd else "multi_hgnn"
    if critic_type == "mlp":
        global_state_dim = sd["critic.critic.0.weight"].shape[1]
    else:
        global_state_dim = obs_dim * n_agents

    network = MAPPONetwork(
        observation_dim=obs_dim,
        global_state_dim=global_state_dim,
        action_dim=action_dim,
        n_agents=n_agents,
        hidden_dim=hidden_dim,
        discrete=False,
        share_actor=share_actor,
        critic_type=critic_type,
        n_hyperedge_types=2,
    )
    network.load_state_dict(sd)
    network.to(device)
    network.eval()

    # Load encoder if present
    encoder, encoder_dim = None, None
    if "local_state_encoder" in checkpoint:
        enc_sd = checkpoint["local_state_encoder"]
        enc_obs_size = enc_sd["init.weight"].shape[1]
        encoder_dim = enc_sd["fc3.weight"].shape[0]
        encoder = LocalStateEncoder(enc_obs_size, encoder_dim)
        encoder.load_state_dict(enc_sd)
        encoder.eval()

    return network, encoder, encoder_dim


def run_episode(
    env,
    n_agents,
    obs_per_agent,
    k,
    intrinsic_coef,
    encoder=None,
    encoder_dim=None,
    network=None,
    seed=42,
):
    """Run one episode and return team + agent intrinsic reward logs."""
    obs, _ = env.reset(seed=seed)

    use_encoder = encoder is not None
    feat_dim = encoder_dim if use_encoder else obs_per_agent
    team_obs_dim = n_agents * feat_dim

    team_rewarder = IntrinsicReward(obs_dim=team_obs_dim, k=k, memory_capacity=1024)
    agent_rewarders = [
        IntrinsicReward(obs_dim=feat_dim, k=k, memory_capacity=1024)
        for _ in range(n_agents)
    ]

    team_rewards_log = []
    agent_rewards_log = []
    env_rewards_log = []

    while True:
        if network is not None:
            with torch.no_grad():
                obs_tensor = torch.from_numpy(
                    np.ascontiguousarray(obs, dtype=np.float32)
                )
                agent_actions = []
                for agent_idx in range(n_agents):
                    action, _ = network.act(
                        obs_tensor[agent_idx].unsqueeze(0),
                        agent_idx,
                        deterministic=True,
                    )
                    agent_actions.append(action.squeeze(0).cpu().numpy())
                actions = np.stack(agent_actions)
        else:
            actions = env.action_space.sample()

        next_obs, env_reward, terminated, truncated, _ = env.step(actions)
        env_rewards_log.append(env_reward)

        if use_encoder:
            with torch.no_grad():
                obs_tensor = torch.from_numpy(
                    np.ascontiguousarray(next_obs, dtype=np.float32)
                )
                obs_flat = obs_tensor.reshape(n_agents, obs_per_agent)
                features = encoder.embedding(obs_flat).numpy()
        else:
            features = next_obs.astype(np.float32)  # (n_agents, obs_per_agent)

        # Team: concatenate all agent features
        team_features = features.reshape(-1)
        team_rewards_log.append(
            intrinsic_coef * team_rewarder.compute_reward(team_features)
        )
        team_rewarder.on_rollout_step(team_features)

        # Per-agent
        agent_irs = np.zeros(n_agents, dtype=np.float32)
        for agent_idx, rewarder in enumerate(agent_rewarders):
            agent_irs[agent_idx] = intrinsic_coef * rewarder.compute_reward(
                features[agent_idx]
            )
            rewarder.on_rollout_step(features[agent_idx])
        agent_rewards_log.append(agent_irs)

        obs = next_obs

        if terminated or truncated:
            break

    return (
        np.array(team_rewards_log),
        np.stack(agent_rewards_log),
        np.array(env_rewards_log),
    )


def run_episode_hg_encoder(
    env,
    n_agents,
    obs_per_agent,
    k,
    intrinsic_coef,
    hg_encoder,
    hg_encoder_dim,
    hg_threshold=1.0,
    network=None,
    seed=42,
):
    """Run one episode using a HypergraphStateEncoder for team intrinsic rewards."""
    obs, _ = env.reset(seed=seed)

    team_rewarder = IntrinsicReward(obs_dim=hg_encoder_dim, k=k, memory_capacity=1024)

    team_rewards_log = []
    env_rewards_log = []

    while True:
        if network is not None:
            with torch.no_grad():
                obs_tensor = torch.from_numpy(
                    np.ascontiguousarray(obs, dtype=np.float32)
                )
                agent_actions = []
                for agent_idx in range(n_agents):
                    action, _ = network.act(
                        obs_tensor[agent_idx].unsqueeze(0),
                        agent_idx,
                        deterministic=True,
                    )
                    agent_actions.append(action.squeeze(0).cpu().numpy())
                actions = np.stack(agent_actions)
        else:
            actions = env.action_space.sample()

        next_obs, env_reward, terminated, truncated, info = env.step(actions)
        env_rewards_log.append(env_reward)

        # Build hypergraphs: distance-based + object-contact
        obs_batch = np.expand_dims(next_obs, axis=0)  # (1, n_agents, obs_dim)
        hg_distance = build_hypergraph(
            1,
            n_agents,
            obs_batch,
            partial(distance_based_hyperedges, threshold=hg_threshold),
        )[0]
        hg_contact = build_hypergraph(
            1,
            n_agents,
            [info["agents_2_objects"]],
            object_contact_hyperedges,
        )[0]
        hg_list = [hg_distance, hg_contact]

        with torch.no_grad():
            X = torch.from_numpy(np.ascontiguousarray(next_obs, dtype=np.float32))
            team_features = hg_encoder.embedding(X, hg_list).numpy()

        team_rewards_log.append(
            intrinsic_coef * team_rewarder.compute_reward(team_features)
        )
        team_rewarder.on_rollout_step(team_features)

        obs = next_obs

        if terminated or truncated:
            break

    return np.array(team_rewards_log), np.array(env_rewards_log)


def test_intrinsic_reward_rollout(checkpoint_path=None, force_encoder=False):
    n_agents = 12
    obs_per_agent = 21
    k_values = [2, 4, 8, 16]

    env = MultiBoxPushEnv(n_agents=n_agents, n_objects=6, max_steps=1024)

    network = None
    if checkpoint_path is not None:
        network, encoder, encoder_dim = load_from_checkpoint(checkpoint_path, n_agents)
        use_encoder = encoder is not None
        if not use_encoder and force_encoder:
            encoder_dim = 32
            encoder = LocalStateEncoder(obs_per_agent, encoder_dim)
            use_encoder = True
        if use_encoder:
            if encoder.obs_size != obs_per_agent:
                raise ValueError(
                    f"Encoder obs_size ({encoder.obs_size}) doesn't match "
                    f"environment obs_per_agent ({obs_per_agent})"
                )
        print(f"Loaded checkpoint: {checkpoint_path}")
        encoder_status = (
            "loaded"
            if encoder is not None and not force_encoder
            else "random (forced)" if force_encoder and use_encoder else "none"
        )
        print(f"  Policy: active | Encoder: {encoder_status}")
    else:
        encoder_dim = 32
        use_encoder = True
        encoder = LocalStateEncoder(obs_per_agent, encoder_dim)

    # Hypergraph encoder (random, frozen)
    hg_encoder_dim = 32
    n_hyperedge_types = 2
    hg_encoder = HypergraphStateEncoder(
        n_hyperedge_types=n_hyperedge_types,
        observation_dim=obs_per_agent,
        num_outputs=hg_encoder_dim,
    )

    # Collect results per k
    results = {}
    hg_results = {}
    for k in k_values:
        team_log, agent_log, env_log = run_episode(
            env,
            n_agents,
            obs_per_agent,
            k,
            intrinsic_coef=1.0,
            encoder=encoder,
            encoder_dim=encoder_dim,
            network=network,
        )
        results[k] = (team_log, agent_log, env_log)

        hg_team_log, _ = run_episode_hg_encoder(
            env,
            n_agents,
            obs_per_agent,
            k,
            intrinsic_coef=2.0,
            hg_encoder=hg_encoder,
            hg_encoder_dim=hg_encoder_dim,
            network=network,
        )
        hg_results[k] = hg_team_log

        print(f"\nk={k}  steps={len(team_log)}")
        print(f"  Team (local)  — mean={team_log.mean():.4f}  std={team_log.std():.4f}")
        print(
            f"  Team (hgraph) — mean={hg_team_log.mean():.4f}  std={hg_team_log.std():.4f}"
        )
        for a in range(n_agents):
            col = agent_log[:, a]
            print(f"  Agent {a} — mean={col.mean():.4f}  std={col.std():.4f}")

    # --- Plots ---
    mode_label = "encoded" if use_encoder else "raw"
    fig, axes = plt.subplots(4, 1, figsize=(12, 14), sharex=True)

    # Environment reward (same across k values, so plot once from the first)
    env_log = results[k_values[0]][2]
    steps = np.arange(1, len(env_log) + 1)
    axes[0].plot(steps, env_log, linewidth=1, color="black")
    axes[0].set_ylabel("Reward")
    axes[0].set_title("Environment Reward")
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([0, 1.0])

    # Team intrinsic reward for each k
    for k in k_values:
        team_log = results[k][0]
        axes[1].plot(
            np.arange(1, len(team_log) + 1), team_log, linewidth=1, label=f"k={k}"
        )
    axes[1].set_ylabel("Intrinsic Reward")
    axes[1].set_title(f"Team Intrinsic Reward ({mode_label} obs)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Per-agent mean intrinsic reward for each k
    for k in k_values:
        agent_log = results[k][1]
        agent_mean = agent_log.mean(axis=1)  # mean across agents per step
        axes[2].plot(
            np.arange(1, len(agent_mean) + 1), agent_mean, linewidth=1, label=f"k={k}"
        )
    axes[2].set_ylabel("Intrinsic Reward")
    axes[2].set_title(
        f"Per-Agent Intrinsic Reward — mean across agents ({mode_label} obs)"
    )
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    # Team intrinsic reward using HypergraphStateEncoder
    for k in k_values:
        hg_team_log = hg_results[k]
        axes[3].plot(
            np.arange(1, len(hg_team_log) + 1), hg_team_log, linewidth=1, label=f"k={k}"
        )
    axes[3].set_xlabel("Step")
    axes[3].set_ylabel("Intrinsic Reward")
    axes[3].set_title("Team Intrinsic Reward (HypergraphStateEncoder)")
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("algorithms/tests/intrinsic_reward_rollout.png", dpi=150)
    plt.close()
    print(f"\nPlot saved to algorithms/tests/intrinsic_reward_rollout.png")

    # --- Combined overlay figure ---
    fig2, ax = plt.subplots(figsize=(12, 5))
    ax.plot(steps, env_log, linewidth=1.5, color="black", label="env reward")
    for k in k_values:
        team_log = results[k][0]
        ax.plot(
            np.arange(1, len(team_log) + 1),
            team_log,
            linewidth=1,
            linestyle="--",
            label=f"team intrinsic (k={k})",
        )
        agent_mean = results[k][1].mean(axis=1)
        ax.plot(
            np.arange(1, len(agent_mean) + 1),
            agent_mean,
            linewidth=1,
            linestyle=":",
            label=f"agent intrinsic (k={k})",
        )
        hg_team_log = hg_results[k]
        ax.plot(
            np.arange(1, len(hg_team_log) + 1),
            hg_team_log,
            linewidth=1,
            linestyle="-.",
            label=f"hg team intrinsic (k={k})",
        )
    ax.set_xlabel("Step")
    ax.set_ylabel("Reward")
    ax.set_ylim([0, 5])
    ax.set_title(f"All Rewards Overlay ({mode_label} obs)")
    ax.legend(fontsize=8, ncol=3)
    ax.grid(True, alpha=0.3)
    fig2.tight_layout()
    fig2.savefig("algorithms/tests/intrinsic_reward_overlay.png", dpi=150)
    plt.close(fig2)
    print(f"Plot saved to algorithms/tests/intrinsic_reward_overlay.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to a checkpoint saved by CheckpointIO.save_agent",
    )
    parser.add_argument(
        "--force-encoder",
        action="store_true",
        help="Use a random encoder even if the checkpoint has none",
    )
    args = parser.parse_args()
    test_intrinsic_reward_rollout(
        checkpoint_path=args.checkpoint, force_encoder=args.force_encoder
    )
