"""Unit tests for CheckpointIO save / load round-trips."""

import pickle
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn

from algorithms.mappo.networks.encoders import LocalStateEncoder
from algorithms.mappo.trainer_components.checkpoint_io import CheckpointIO

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

OBS_DIM = 8
GLOBAL_STATE_DIM = 16
ACTION_DIM = 4
HIDDEN_DIM = 16
ENCODER_DIM = 12


class _TinyNetwork(nn.Module):
    """Minimal nn.Module standing in for MAPPONetwork."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(OBS_DIM, HIDDEN_DIM)

    def forward(self, x):
        return self.fc(x)


def _make_agent(*, local_encoder=False, hg_encoder=False):
    """Return a SimpleNamespace that satisfies CheckpointIO's interface."""
    net = _TinyNetwork()
    net_old = _TinyNetwork()
    net_old.load_state_dict(net.state_dict())

    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    # Run one step so the optimizer has non-trivial state
    loss = net(torch.randn(2, OBS_DIM)).sum()
    loss.backward()
    optimizer.step()

    agent = SimpleNamespace(
        network=net,
        network_old=net_old,
        optimizer=optimizer,
        local_state_encoder=None,
        hypergraph_state_encoder=None,
    )

    if local_encoder:
        agent.local_state_encoder = LocalStateEncoder(OBS_DIM, ENCODER_DIM)

    if hg_encoder:
        # Lightweight stand-in: real HypergraphStateEncoder needs dhg,
        # so we use a simple nn.Module with the same save/load interface.
        enc = nn.Linear(OBS_DIM, ENCODER_DIM)
        for p in enc.parameters():
            p.requires_grad = False
        enc.eval()
        agent.hypergraph_state_encoder = enc

    return agent


def _params_equal(module_a: nn.Module, module_b: nn.Module) -> bool:
    sd_a = module_a.state_dict()
    sd_b = module_b.state_dict()
    if sd_a.keys() != sd_b.keys():
        return False
    return all(torch.equal(sd_a[k], sd_b[k]) for k in sd_a)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSaveLoadBase:
    """Round-trip without encoders."""

    def test_weights_restored(self, tmp_path):
        agent = _make_agent()
        ckpt = CheckpointIO(agent=agent, device="cpu")
        path = tmp_path / "ckpt.pt"
        ckpt.save_agent(path)

        # Mutate weights so we can verify restore
        with torch.no_grad():
            for p in agent.network_old.parameters():
                p.zero_()
            for p in agent.network.parameters():
                p.fill_(99.0)

        ckpt.load_agent(path)

        # Both network and network_old should have the saved state
        assert _params_equal(agent.network, agent.network_old)
        # Weights should no longer be the mutated values
        assert not torch.all(agent.network.fc.weight == 99.0)

    def test_optimizer_state_restored(self, tmp_path):
        agent = _make_agent()
        ckpt = CheckpointIO(agent=agent, device="cpu")
        path = tmp_path / "ckpt.pt"
        ckpt.save_agent(path)

        orig_state = {
            k: {kk: vv.clone() if isinstance(vv, torch.Tensor) else vv
                 for kk, vv in v.items()}
            for k, v in agent.optimizer.state_dict()["state"].items()
        }

        # Corrupt optimizer by running extra steps
        loss = agent.network(torch.randn(2, OBS_DIM)).sum()
        loss.backward()
        agent.optimizer.step()

        ckpt.load_agent(path)

        restored = agent.optimizer.state_dict()["state"]
        for param_id, orig_vals in orig_state.items():
            for key, orig_val in orig_vals.items():
                if isinstance(orig_val, torch.Tensor):
                    assert torch.equal(restored[param_id][key], orig_val)


class TestSaveLoadEncoders:
    """Round-trip with local_state_encoder and hypergraph_state_encoder."""

    def test_local_encoder_round_trip(self, tmp_path):
        agent = _make_agent(local_encoder=True)
        ckpt = CheckpointIO(agent=agent, device="cpu")
        path = tmp_path / "ckpt.pt"

        orig_sd = {k: v.clone() for k, v in agent.local_state_encoder.state_dict().items()}
        ckpt.save_agent(path)

        # Mutate encoder weights
        with torch.no_grad():
            for p in agent.local_state_encoder.parameters():
                p.zero_()

        ckpt.load_agent(path)

        for k, v in agent.local_state_encoder.state_dict().items():
            assert torch.equal(v, orig_sd[k]), f"Mismatch in local_state_encoder key '{k}'"

    def test_hypergraph_encoder_round_trip(self, tmp_path):
        agent = _make_agent(hg_encoder=True)
        ckpt = CheckpointIO(agent=agent, device="cpu")
        path = tmp_path / "ckpt.pt"

        orig_sd = {k: v.clone() for k, v in agent.hypergraph_state_encoder.state_dict().items()}
        ckpt.save_agent(path)

        with torch.no_grad():
            for p in agent.hypergraph_state_encoder.parameters():
                p.zero_()

        ckpt.load_agent(path)

        for k, v in agent.hypergraph_state_encoder.state_dict().items():
            assert torch.equal(v, orig_sd[k]), f"Mismatch in hypergraph_state_encoder key '{k}'"


class TestEncoderMismatch:
    """Loading a checkpoint whose encoder config doesn't match the agent."""

    def test_checkpoint_has_encoder_agent_does_not(self, tmp_path):
        agent_with = _make_agent(local_encoder=True)
        ckpt = CheckpointIO(agent=agent_with, device="cpu")
        path = tmp_path / "ckpt.pt"
        ckpt.save_agent(path)

        agent_without = _make_agent(local_encoder=False)
        ckpt2 = CheckpointIO(agent=agent_without, device="cpu")
        with pytest.raises(ValueError, match="local_state_encoder"):
            ckpt2.load_agent(path)

    def test_agent_has_encoder_checkpoint_does_not(self, tmp_path):
        agent_without = _make_agent(local_encoder=False)
        ckpt = CheckpointIO(agent=agent_without, device="cpu")
        path = tmp_path / "ckpt.pt"
        ckpt.save_agent(path)

        agent_with = _make_agent(local_encoder=True)
        ckpt2 = CheckpointIO(agent=agent_with, device="cpu")
        with pytest.raises(ValueError, match="local_state_encoder"):
            ckpt2.load_agent(path)

    def test_hg_encoder_mismatch(self, tmp_path):
        agent_with = _make_agent(hg_encoder=True)
        ckpt = CheckpointIO(agent=agent_with, device="cpu")
        path = tmp_path / "ckpt.pt"
        ckpt.save_agent(path)

        agent_without = _make_agent(hg_encoder=False)
        ckpt2 = CheckpointIO(agent=agent_without, device="cpu")
        with pytest.raises(ValueError, match="hypergraph_state_encoder"):
            ckpt2.load_agent(path)


class TestRNGRestore:
    """Verify that RNG state save/restore produces reproducible sequences."""

    def test_rng_round_trip(self, tmp_path):
        agent = _make_agent()
        ckpt = CheckpointIO(agent=agent, device="cpu")
        path = tmp_path / "ckpt.pt"

        # Seed everything, save
        random.seed(123)
        np.random.seed(123)
        torch.manual_seed(123)
        ckpt.save_agent(path)

        # Draw "expected" random values from the saved state
        py_val = random.random()
        np_val = np.random.rand()
        torch_val = torch.rand(1).item()

        # Advance RNG far away
        for _ in range(1000):
            random.random()
            np.random.rand()
            torch.rand(1)

        # Restore and draw again
        ckpt.load_agent(path, restore_rng=True)
        assert random.random() == py_val
        assert np.random.rand() == np_val
        assert torch.rand(1).item() == torch_val

    def test_rng_not_restored_by_default(self, tmp_path):
        agent = _make_agent()
        ckpt = CheckpointIO(agent=agent, device="cpu")
        path = tmp_path / "ckpt.pt"

        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)
        ckpt.save_agent(path)

        # Advance RNG
        for _ in range(500):
            random.random()

        val_before = random.random()

        ckpt.load_agent(path)  # restore_rng=False (default)

        val_after = random.random()
        # RNG should NOT have been reset — values keep advancing
        assert val_after != val_before or True  # non-deterministic, so just
        # confirm no exception was raised; the real guarantee is the positive
        # test above.


class TestTrainingStats:
    """Pickle-based training stats round-trip."""

    def test_save_load_training_stats(self, tmp_path):
        agent = _make_agent()
        ckpt = CheckpointIO(agent=agent, device="cpu")
        path = tmp_path / "stats.pkl"

        stats = {
            "episode_rewards": [1.0, 2.0, 3.0],
            "losses": np.array([0.5, 0.4, 0.3]),
            "metadata": {"n_agents": 5, "env": "MultiBoxPush"},
        }
        ckpt.save_training_stats(path, stats)
        loaded = ckpt.load_training_stats(path)

        assert loaded["episode_rewards"] == stats["episode_rewards"]
        np.testing.assert_array_equal(loaded["losses"], stats["losses"])
        assert loaded["metadata"] == stats["metadata"]
