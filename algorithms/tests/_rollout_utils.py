"""Shared helpers for tests that load a trained MAPPO checkpoint and run rollouts.

Two loaders are exposed:

- ``load_mappo_network``: build ``MAPPONetwork`` from explicit hyperparameters
  (the experiment config), then load weights. Use this when the test already
  hard-codes the experiment config.
- ``load_mappo_network_inferred``: infer actor/critic dimensions from the
  saved ``state_dict`` and also restore an optional ``LocalStateEncoder``.
  Use this when the test should accept arbitrary checkpoints.

Both put the network in ``eval()`` mode on the requested device.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch

from algorithms.mappo.networks.encoders import LocalStateEncoder
from algorithms.mappo.networks.models import MAPPONetwork


def load_mappo_network(
    checkpoint_path: str | Path,
    *,
    n_agents: int,
    observation_dim: int,
    action_dim: int,
    hidden_dim: int,
    critic_type: str,
    n_hyperedge_types: int = 0,
    discrete: bool = False,
    share_actor: bool = True,
    entropy_conditioning: bool = False,
    hypergraph_mode: str = "predefined",
    critic_seq_len: int = 1,
    global_state_dim: Optional[int] = None,
    device: str = "cpu",
    verbose: bool = True,
) -> MAPPONetwork:
    """Build a MAPPONetwork with the given config, load weights, eval-mode it."""
    if global_state_dim is None:
        global_state_dim = observation_dim * n_agents

    network = MAPPONetwork(
        observation_dim=observation_dim,
        global_state_dim=global_state_dim,
        action_dim=action_dim,
        n_agents=n_agents,
        hidden_dim=hidden_dim,
        discrete=discrete,
        share_actor=share_actor,
        critic_type=critic_type,
        n_hyperedge_types=n_hyperedge_types,
        critic_seq_len=critic_seq_len,
        entropy_conditioning=entropy_conditioning,
        hypergraph_mode=hypergraph_mode,
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    network.load_state_dict(checkpoint["network"])
    network.eval()
    if verbose:
        print(f"Loaded checkpoint from {checkpoint_path}")
    return network


def load_mappo_network_inferred(
    checkpoint_path: str | Path,
    n_agents: int,
    *,
    n_hyperedge_types: int = 2,
    device: str = "cpu",
    verbose: bool = True,
) -> tuple[MAPPONetwork, Optional[LocalStateEncoder], Optional[int]]:
    """Load a MAPPONetwork by inferring its dimensions from the saved state dict.

    Also restores a ``LocalStateEncoder`` if present in the checkpoint.

    Returns ``(network, encoder, encoder_dim)`` where ``encoder`` and
    ``encoder_dim`` are ``None`` if the checkpoint contains no encoder.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    sd = checkpoint["network"]

    share_actor = "actor.actor.0.weight" in sd
    if share_actor:
        obs_dim = sd["actor.actor.0.weight"].shape[1]
        action_dim = sd["actor.actor.4.weight"].shape[0]
        hidden_dim = sd["actor.actor.0.weight"].shape[0]
    else:
        obs_dim = sd["actors.0.actor.0.weight"].shape[1]
        action_dim = sd["actors.0.actor.4.weight"].shape[0]
        hidden_dim = sd["actors.0.actor.0.weight"].shape[0]

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
        n_hyperedge_types=n_hyperedge_types,
    )
    network.load_state_dict(sd)
    network.to(device)
    network.eval()

    encoder: Optional[LocalStateEncoder] = None
    encoder_dim: Optional[int] = None
    if "local_state_encoder" in checkpoint:
        enc_sd = checkpoint["local_state_encoder"]
        enc_obs_size = enc_sd["init.weight"].shape[1]
        encoder_dim = enc_sd["fc3.weight"].shape[0]
        encoder = LocalStateEncoder(enc_obs_size, encoder_dim)
        encoder.load_state_dict(enc_sd)
        encoder.eval()

    if verbose:
        print(f"Loaded checkpoint from {checkpoint_path}")
    return network, encoder, encoder_dim


def capture_frame(env) -> Optional[np.ndarray]:
    """Grab the current pygame surface as an RGB numpy array, or None if no surface."""
    import pygame

    surface = pygame.display.get_surface()
    if surface is None:
        return None
    frame = pygame.surfarray.array3d(surface)
    return np.transpose(frame, (1, 0, 2)).copy()
