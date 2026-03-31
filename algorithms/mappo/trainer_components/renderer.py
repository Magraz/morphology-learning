import numpy as np
import torch

from algorithms.create_env import make_vec_env
from algorithms.mappo.hypergraph import (
    compute_hyperedge_structural_entropy_batch,
    compute_soft_hyperedge_structural_entropy_batch,
)
from environments.types import EnvironmentEnum


class PolicyRenderer:
    def __init__(
        self,
        *,
        agent,
        device: str,
        env_name,
        env_variant,
        n_agents: int,
        n_objects: int,
        reward_mode: str,
        discrete: bool,
        entropy_conditioning: bool,
        hypergraph_runtime,
    ):
        self.agent = agent
        self.device = device
        self.env_name = env_name
        self.env_variant = env_variant
        self.n_agents = n_agents
        self.n_objects = n_objects
        self.reward_mode = reward_mode
        self.discrete = discrete
        self.entropy_conditioning = entropy_conditioning
        self.hypergraph_runtime = hypergraph_runtime

    def render(self, capture_video: bool = False):
        """Run one episode with the current policy in a render environment."""
        render_env = self._make_render_env()

        self.agent.network_old.eval()
        episode_reward = []
        frames = [] if capture_video else None
        cum_sum = 0.0

        entropy_type_names = self.hypergraph_runtime.entropy_type_names
        n_types = len(entropy_type_names)
        entropy_type_logs = [[] for _ in range(n_types)]
        soft_entropy_type_logs = [[] for _ in range(n_types)]
        predicted_entropy_log = []

        predictor = self.agent.network_old.entropy_predictor
        if predictor is not None:
            obs_dim = self.agent.observation_dim
            obs_history = torch.zeros(1, self.n_agents, self.agent.entropy_pred_seq_len, obs_dim)

        seed = int(np.random.randint(0, 2**31))
        obs, infos = render_env.reset(seed=seed)
        current_masks = infos.get("avail_actions") if isinstance(infos, dict) else None
        if current_masks is not None:
            current_masks = current_masks[np.newaxis]

        # HYGMA mode intentionally keeps groups frozen during render.
        # Hypergraphs come from the latest discovered grouping state.
        with torch.no_grad():
            while True:
                global_states = obs.reshape(1, -1)

                render_hgs, render_sig_ids = self.hypergraph_runtime.build_inference_hypergraphs(
                    obs, infos, 1
                )
                render_entropies = (
                    self.hypergraph_runtime.compute_entropies_for_critic(render_sig_ids)
                    if self.entropy_conditioning and render_sig_ids is not None
                    else None
                )

                actions_t, _, _ = self.agent.get_actions_batched(
                    obs,
                    global_states,
                    deterministic=True,
                    action_masks=current_masks,
                    hypergraphs=render_hgs,
                    entropies=render_entropies,
                )
                actions = actions_t.cpu().numpy()

                if self.discrete:
                    if actions.ndim == 2 and actions.shape[-1] == 1:
                        actions = actions.squeeze(-1)
                    actions = actions.astype(np.int32)

                obs, reward, terminated, truncated, infos = render_env.step(actions)
                current_masks = infos.get("avail_actions") if isinstance(infos, dict) else None
                if current_masks is not None:
                    current_masks = current_masks[np.newaxis]
                render_env.render()

                if capture_video:
                    import pygame

                    surface = pygame.display.get_surface()
                    if surface is not None:
                        frame = pygame.surfarray.array3d(surface)
                        frame = np.transpose(frame, (1, 0, 2))
                        frames.append(frame.copy())

                cum_sum += float(reward)
                episode_reward.append(cum_sum)

                if render_hgs is not None:
                    entropies = compute_hyperedge_structural_entropy_batch(render_hgs)
                    soft_entropies = compute_soft_hyperedge_structural_entropy_batch(render_hgs)
                    for t in range(min(len(render_hgs), n_types)):
                        entropy_type_logs[t].append(entropies[t])
                        soft_entropy_type_logs[t].append(soft_entropies[t])

                if predictor is not None:
                    obs_tensor = torch.from_numpy(np.ascontiguousarray(obs, dtype=np.float32))
                    if obs_tensor.dim() == 2:
                        obs_tensor = obs_tensor.unsqueeze(0)
                    obs_history = torch.roll(obs_history, -1, dims=2)
                    obs_history[:, :, -1, :] = obs_tensor
                    obs_seqs_flat = obs_history.reshape(
                        self.n_agents, self.agent.entropy_pred_seq_len, -1
                    ).to(self.device)
                    agent_ids = torch.arange(self.n_agents, device=self.device)
                    pred_mean, _ = predictor.forward_batch(obs_seqs_flat, agent_ids)
                    predicted_entropy_log.append(pred_mean.cpu().numpy())

                if terminated or truncated:
                    break

        render_env.close()

        entropy_logs = {}
        for t, name in enumerate(entropy_type_names):
            entropy_logs[name] = np.array(entropy_type_logs[t])
            entropy_logs[f"soft_{name}"] = np.array(soft_entropy_type_logs[t])
        entropy_logs["predicted_per_agent"] = (
            np.array(predicted_entropy_log) if predicted_entropy_log else None
        )

        return np.array(episode_reward), entropy_logs, frames

    def _make_render_env(self):
        """Create a single env with render_mode='human' for the current env_name."""
        match self.env_name:
            case EnvironmentEnum.BOX2D_SALP:
                from environments.box2d_salp.domain import SalpChainEnv

                return SalpChainEnv(n_agents=self.n_agents, render_mode="human")

            case EnvironmentEnum.MULTI_BOX:
                render_env = make_vec_env(
                    self.env_name,
                    self.n_agents,
                    1,
                    use_async=True,
                    env_variant=self.env_variant,
                    n_objects=self.n_objects,
                    reward_mode=self.reward_mode,
                )
                render_env.envs[0].render_mode = "human"
                return render_env

            case EnvironmentEnum.MPE_SPREAD:
                from mpe2 import simple_spread_v3
                from algorithms.create_env import PettingZooToGymWrapper

                pz_env = simple_spread_v3.parallel_env(
                    N=self.n_agents,
                    local_ratio=0.5,
                    max_cycles=25,
                    continuous_actions=False,
                    dynamic_rescaling=True,
                    render_mode="human",
                )
                return PettingZooToGymWrapper(pz_env)

            case EnvironmentEnum.MPE_SIMPLE:
                from mpe2 import simple_v3
                from algorithms.create_env import PettingZooToGymWrapper

                pz_env = simple_v3.parallel_env(
                    max_cycles=25,
                    continuous_actions=False,
                    render_mode="human",
                )
                return PettingZooToGymWrapper(pz_env)

            case EnvironmentEnum.SMACLITE:
                from environments.smaclite.wrapper import SmacliteToGymWrapper

                return SmacliteToGymWrapper(map_name=self.env_variant)

            case _:
                return None
