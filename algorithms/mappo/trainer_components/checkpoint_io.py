import pickle
import random

import numpy as np
import torch


class CheckpointIO:
    def __init__(self, *, agent, device: str):
        self.agent = agent
        self.device = device

    def save_agent(self, path) -> None:
        """Save MAPPO agent weights, optimizer state, and RNG states."""
        checkpoint = {
            "network": self.agent.network_old.state_dict(),
            "optimizer": self.agent.optimizer.state_dict(),
            "rng_python": random.getstate(),
            "rng_numpy": np.random.get_state(),
            "rng_torch_cpu": torch.random.get_rng_state(),
            "rng_torch_cuda": (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            ),
        }
        if self.agent.local_state_encoder is not None:
            checkpoint["local_state_encoder"] = (
                self.agent.local_state_encoder.state_dict()
            )
        if self.agent.hypergraph_state_encoder is not None:
            checkpoint["hypergraph_state_encoder"] = (
                self.agent.hypergraph_state_encoder.state_dict()
            )

        torch.save(checkpoint, path)

    def load_agent(self, filepath, restore_rng: bool = False) -> None:
        checkpoint = torch.load(filepath, map_location=self.device)

        self.agent.network_old.load_state_dict(checkpoint["network"])
        self.agent.network.load_state_dict(checkpoint["network"])
        self.agent.optimizer.load_state_dict(checkpoint["optimizer"])

        has_encoder_checkpoint = "local_state_encoder" in checkpoint
        has_encoder_agent = self.agent.local_state_encoder is not None
        if has_encoder_checkpoint != has_encoder_agent:
            raise ValueError(
                "Checkpoint intrinsic encoder state does not match the current "
                "agent configuration (local_state_encoder)."
            )
        if has_encoder_agent:
            self.agent.local_state_encoder.load_state_dict(
                checkpoint["local_state_encoder"]
            )
            self.agent.local_state_encoder.eval()

        has_hg_encoder_checkpoint = "hypergraph_state_encoder" in checkpoint
        has_hg_encoder_agent = self.agent.hypergraph_state_encoder is not None
        if has_hg_encoder_checkpoint != has_hg_encoder_agent:
            raise ValueError(
                "Checkpoint intrinsic encoder state does not match the current "
                "agent configuration (hypergraph_state_encoder)."
            )
        if has_hg_encoder_agent:
            self.agent.hypergraph_state_encoder.load_state_dict(
                checkpoint["hypergraph_state_encoder"]
            )
            self.agent.hypergraph_state_encoder.eval()

        if restore_rng and "rng_python" in checkpoint:
            random.setstate(checkpoint["rng_python"])
            np.random.set_state(checkpoint["rng_numpy"])
            torch.random.set_rng_state(checkpoint["rng_torch_cpu"].cpu())
            if torch.cuda.is_available() and checkpoint["rng_torch_cuda"] is not None:
                torch.cuda.set_rng_state_all([s.cpu() for s in checkpoint["rng_torch_cuda"]])

    def load_training_stats(self, path) -> dict:
        with path.open("rb") as f:
            return pickle.load(f)

    def save_training_stats(self, path, stats: dict) -> None:
        with path.open("wb") as f:
            pickle.dump(stats, f)
