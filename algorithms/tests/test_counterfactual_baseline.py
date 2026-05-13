"""Smoke test for HypergraphCache.compute_per_agent_counterfactual_values.

Loads the trained checkpoint from
experiments/results/multi_box_push_12a_6o_heavy/hgnn_shared/0, builds a
small synthetic minibatch from a rollout, calls the new per-agent
counterfactual evaluator, and asserts:

- shape is (B, n_agents);
- output is finite;
- when an agent participates in NO multi-vertex hyperedge at a given
  timestep, V_cf(i, t) equals the standard V(s_t).

Usage:
    python -m algorithms.tests.test_counterfactual_baseline
"""

import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from algorithms.mappo.hg_cache import HypergraphCache
from algorithms.mappo.hypergraph import (
    distance_based_hyperedges,
    object_contact_hyperedges,
)
from algorithms.tests._rollout_utils import load_mappo_network
from environments.box2d_suite.multi_box_push import MultiBoxPushEnv

N_AGENTS = 12
N_OBJECTS = 6
OBSERVATION_DIM = 22
ACTION_DIM = 2
HIDDEN_DIM = 168
N_HYPEREDGE_TYPES = 2
DEVICE = "cpu"
N_STEPS = 16  # number of rollout timesteps to collect for the synthetic minibatch

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "experiments/results/multi_box_push_12a_6o_heavy"
    / "hgnn_shared/0/models/models_checkpoint.pth"
)


def build_per_type_edge_lists(obs, info, n_agents):
    proximity_edges = distance_based_hyperedges(obs, n_agents, threshold=1.0)
    contact_edges = object_contact_hyperedges(info["agents_2_objects"], n_agents)
    return [proximity_edges, contact_edges]


def main():
    network = load_mappo_network(
        CHECKPOINT_PATH,
        n_agents=N_AGENTS,
        observation_dim=OBSERVATION_DIM,
        action_dim=ACTION_DIM,
        hidden_dim=HIDDEN_DIM,
        critic_type="multi_hgnn",
        n_hyperedge_types=N_HYPEREDGE_TYPES,
        device=DEVICE,
    )

    cache = HypergraphCache(n_agents=N_AGENTS, n_parallel_envs=1)

    env = MultiBoxPushEnv(
        n_agents=N_AGENTS, n_objects=N_OBJECTS, reward_mode="dense"
    )
    obs, info = env.reset(seed=42)

    # Collect a short rollout: signature ids per timestep + the global state.
    ts_to_signature_ids: list[int] = []
    global_states: list[np.ndarray] = []
    edge_lists_by_ts: list[list[list[tuple]]] = []

    with torch.no_grad():
        for _ in range(N_STEPS):
            obs_np = np.ascontiguousarray(obs, dtype=np.float32)
            edge_lists = build_per_type_edge_lists(obs_np, info, N_AGENTS)
            sig_id = cache.intern(edge_lists)
            ts_to_signature_ids.append(sig_id)
            global_states.append(obs_np.reshape(-1))
            edge_lists_by_ts.append(edge_lists)

            obs_tensor = torch.from_numpy(obs_np).to(DEVICE)
            actions_flat, _ = network.act(
                obs_tensor.reshape(N_AGENTS, -1), agent_idx=0, deterministic=True
            )
            actions = actions_flat.reshape(N_AGENTS, ACTION_DIM).cpu().numpy()
            obs, _, terminated, truncated, info = env.step(actions)
            if terminated or truncated:
                break

    env.close()

    n_ts = len(ts_to_signature_ids)
    print(f"Collected {n_ts} rollout steps")

    # Build a synthetic minibatch with B = n_ts * n_agents (all agents at all ts).
    batch_global_states = torch.tensor(np.stack(global_states), dtype=torch.float32)  # (n_ts, obs_dim*n_agents)
    # batch_ts_indices: each sample at minibatch position s gets ts = s // n_agents.
    batch_ts_indices = torch.arange(n_ts, dtype=torch.long).repeat_interleave(N_AGENTS)
    # Flatten global states to match: (B, obs_dim*n_agents) — every n_agents
    # consecutive rows share the same global state.
    batch_global_states_expanded = batch_global_states.repeat_interleave(N_AGENTS, dim=0)

    cache.clear_minibatch_cache()
    with torch.no_grad():
        cf_values = cache.compute_per_agent_counterfactual_values(
            network_critic=network.critic,
            batch_global_states=batch_global_states_expanded,
            batch_ts_indices=batch_ts_indices,
            ts_to_signature_ids=ts_to_signature_ids,
            observation_dim=OBSERVATION_DIM,
            device=DEVICE,
        )

    B = n_ts * N_AGENTS
    assert cf_values.shape == (B, N_AGENTS), (
        f"Expected shape ({B}, {N_AGENTS}), got {tuple(cf_values.shape)}"
    )
    assert torch.isfinite(cf_values).all(), "cf_values contains non-finite entries"
    print(f"cf_values shape OK: {tuple(cf_values.shape)}, all finite")

    # Reference: evaluate the standard V(s_t) per timestep using forward_batched
    # on the original (non-counterfactual) hypergraphs.
    obs_unique = batch_global_states  # (n_ts, obs_dim*n_agents)
    obs_flat = obs_unique.reshape(n_ts * N_AGENTS, OBSERVATION_DIM)
    unique_sig_ids = tuple(ts_to_signature_ids)
    n_types = len(cache.unique_edge_lists[0])
    batched_hgs = [
        cache.get_or_build_batched_by_type(
            unique_sig_ids, type_idx, DEVICE, cache_scope="minibatch"
        )
        for type_idx in range(n_types)
    ]
    with torch.no_grad():
        v_ref = network.critic.forward_batched(
            obs_flat, batched_hgs, n_ts
        ).squeeze(-1)  # (n_ts,)

    # Sanity: for each (t, agent) where agent is in no multi-vertex edge across
    # all types, V_cf(i, t) must equal V(s_t).
    n_match_checks = 0
    n_match_passed = 0
    for t, edge_lists in enumerate(edge_lists_by_ts):
        for agent_idx in range(N_AGENTS):
            in_multi = any(
                any(len(e) > 1 and agent_idx in e for e in type_edges)
                for type_edges in edge_lists
            )
            if in_multi:
                continue
            n_match_checks += 1
            cf_v = cf_values[t * N_AGENTS, agent_idx].item()
            ref_v = v_ref[t].item()
            if abs(cf_v - ref_v) < 1e-5:
                n_match_passed += 1
            else:
                print(
                    f"  MISMATCH t={t} agent={agent_idx} cf={cf_v:.6f} ref={ref_v:.6f}"
                )
    print(
        f"agents-with-no-multi-edge invariant: {n_match_passed}/{n_match_checks} "
        f"timesteps matched within 1e-5"
    )
    assert n_match_passed == n_match_checks, "counterfactual invariant violated"
    print("All assertions passed.")


if __name__ == "__main__":
    main()
