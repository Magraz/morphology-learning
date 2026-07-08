import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.backends.backend_agg import FigureCanvasAgg

from algorithms.create_env import make_vec_env

from environments.types import EnvironmentEnum


class PolicyRenderer:
    def __init__(
        self,
        *,
        agent,
        device: str,
        env_params: dict,
        discrete: bool,
    ):
        self.agent = agent
        self.device = device
        self.env_params = env_params
        self.env_name = env_params.get("environment")
        self.env_variant = env_params.get("env_variant")
        self.n_agents = env_params.get("n_agents")
        self.n_objects = env_params.get("n_objects")
        self.reward_mode = env_params.get("reward_mode")
        self.discrete = discrete

    def render(self, capture_video: bool = False):
        """Run one episode with the current policy in a render environment.

        Returns:
            episode_reward: np.ndarray of cumulative reward per step.
            frames: list of RGB frames (if capture_video), else None.
        """
        render_env = self._make_render_env()

        self.agent.network_old.eval()
        episode_reward = []
        frames = [] if capture_video else None
        cum_sum = 0.0

        seed = int(np.random.randint(0, 2**31))
        obs, infos = render_env.reset(seed=seed)
        current_masks = infos.get("avail_actions") if isinstance(infos, dict) else None
        if current_masks is not None:
            current_masks = current_masks[np.newaxis]

        with torch.no_grad():

            while True:
                obs_batch = obs[np.newaxis] if obs.ndim == 2 else obs

                global_states = obs.reshape(1, -1)

                actions_t, _, _ = self.agent.get_actions_batched(
                    obs,
                    global_states,
                    deterministic=True,
                    action_masks=current_masks,
                )
                actions = actions_t.cpu().numpy()

                if self.discrete:
                    if actions.ndim == 2 and actions.shape[-1] == 1:
                        actions = actions.squeeze(-1)
                    actions = actions.astype(np.int32)

                obs, reward, terminated, truncated, infos = render_env.step(actions)
                current_masks = (
                    infos.get("avail_actions") if isinstance(infos, dict) else None
                )
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

                if terminated or truncated:
                    break

        render_env.close()

        return np.array(episode_reward), frames

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
                    env_params=self.env_params,
                )
                render_env.envs[0].render_mode = "human"
                return render_env

            case EnvironmentEnum.PUSH_BOX:
                render_env = make_vec_env(
                    self.env_name,
                    self.n_agents,
                    1,
                    use_async=True,
                    env_params=self.env_params,
                )
                render_env.envs[0].render_mode = "human"
                return render_env

            case EnvironmentEnum.SCATTER:
                render_env = make_vec_env(
                    self.env_name,
                    self.n_agents,
                    1,
                    use_async=True,
                )
                render_env.envs[0].render_mode = "human"
                return render_env

            case EnvironmentEnum.RENDEZVOUZ:
                render_env = make_vec_env(
                    self.env_name,
                    self.n_agents,
                    1,
                    use_async=True,
                )
                render_env.envs[0].render_mode = "human"
                return render_env

            case EnvironmentEnum.CONTACT:
                render_env = make_vec_env(
                    self.env_name,
                    self.n_agents,
                    1,
                    use_async=True,
                )
                render_env.envs[0].render_mode = "human"
                return render_env

            case EnvironmentEnum.HRL_SKILL:
                # n_envs=1 yields a SyncVectorEnv (built in-process, no
                # forkserver) so the obs is batched to (1, n_agents, obs_dim) —
                # the shape get_actions_batched expects. Flip the wrapper's base
                # env into human render mode; the box2d renderer only checks
                # render_mode at draw time, so setting it after construction is
                # enough.
                render_env = make_vec_env(
                    self.env_name,
                    self.n_agents,
                    1,
                    use_async=True,
                    env_params=self.env_params,
                )
                render_env.envs[0].base_env.render_mode = "human"
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
