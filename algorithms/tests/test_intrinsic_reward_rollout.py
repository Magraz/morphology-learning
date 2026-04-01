"""Run a single episode of MultiBoxPushEnv and log team + agent intrinsic rewards."""

import matplotlib.pyplot as plt
import numpy as np
import torch
from algorithms.mappo.intrinsic_reward import IntrinsicReward
from algorithms.mappo.networks.encoders import LocalStateEncoder
from environments.multi_box_push.domain import MultiBoxPushEnv


def run_episode(
    env,
    n_agents,
    obs_per_agent,
    k,
    intrinsic_coef,
    encoder=None,
    encoder_dim=None,
    seed=42,
):
    """Run one episode and return team + agent intrinsic reward logs."""
    obs, _ = env.reset(seed=seed)

    use_encoder = encoder is not None
    feat_dim = encoder_dim if use_encoder else obs_per_agent
    team_obs_dim = n_agents * feat_dim

    team_rewarder = IntrinsicReward(obs_dim=team_obs_dim, k=k, memory_capacity=10_000)
    agent_rewarders = [
        IntrinsicReward(obs_dim=feat_dim, k=k, memory_capacity=10_000)
        for _ in range(n_agents)
    ]

    team_rewards_log = []
    agent_rewards_log = []

    while True:
        actions = env.action_space.sample()
        next_obs, _, terminated, truncated, _ = env.step(actions)

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

        if terminated or truncated:
            break

    return np.array(team_rewards_log), np.stack(agent_rewards_log)


def test_intrinsic_reward_rollout():
    n_agents = 5
    obs_per_agent = 21
    k_values = [2, 4, 8, 16]
    use_encoder = True

    env = MultiBoxPushEnv(n_agents=n_agents, n_objects=3, max_steps=1024)

    encoder_dim = 32
    encoder = LocalStateEncoder(obs_per_agent, encoder_dim) if use_encoder else None

    # Collect results per k
    results = {}
    for k in k_values:
        team_log, agent_log = run_episode(
            env,
            n_agents,
            obs_per_agent,
            k,
            intrinsic_coef=10.0,
            encoder=encoder,
            encoder_dim=encoder_dim,
        )
        results[k] = (team_log, agent_log)
        print(f"\nk={k}  steps={len(team_log)}")
        print(f"  Team  — mean={team_log.mean():.4f}  std={team_log.std():.4f}")
        for a in range(n_agents):
            col = agent_log[:, a]
            print(f"  Agent {a} — mean={col.mean():.4f}  std={col.std():.4f}")

    # --- Plots ---
    mode_label = "encoded" if use_encoder else "raw"
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Team intrinsic reward for each k
    for k in k_values:
        team_log = results[k][0]
        axes[0].plot(
            np.arange(1, len(team_log) + 1), team_log, linewidth=1, label=f"k={k}"
        )
    axes[0].set_ylabel("Intrinsic Reward")
    axes[0].set_title(f"Team Intrinsic Reward ({mode_label} obs)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Per-agent mean intrinsic reward for each k
    for k in k_values:
        agent_log = results[k][1]
        agent_mean = agent_log.mean(axis=1)  # mean across agents per step
        axes[1].plot(
            np.arange(1, len(agent_mean) + 1), agent_mean, linewidth=1, label=f"k={k}"
        )
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Intrinsic Reward")
    axes[1].set_title(
        f"Per-Agent Intrinsic Reward — mean across agents ({mode_label} obs)"
    )
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("algorithms/tests/intrinsic_reward_rollout.png", dpi=150)
    plt.close()
    print(f"\nPlot saved to algorithms/tests/intrinsic_reward_rollout.png")


if __name__ == "__main__":
    test_intrinsic_reward_rollout()
