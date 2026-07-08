import numpy as np
import torch


class PolicyEvaluator:
    def __init__(
        self,
        *,
        eval_env,
        agent,
        n_eval_episodes: int,
        discrete: bool,
    ):
        self.eval_env = eval_env
        self.agent = agent
        self.n_eval_episodes = n_eval_episodes
        self.discrete = discrete

    def evaluate(self) -> float:
        """Evaluate current policy using parallel episodes."""
        self.agent.network_old.eval()
        n_eps = self.n_eval_episodes

        with torch.no_grad():
            eval_seeds = [int(np.random.randint(0, 2**31)) for _ in range(n_eps)]
            obs, infos = self.eval_env.reset(seed=eval_seeds)
            current_masks = (
                infos.get("avail_actions") if isinstance(infos, dict) else None
            )
            episode_rewards = np.zeros(n_eps)
            finished = np.zeros(n_eps, dtype=bool)

            while not finished.all():
                global_states = obs.reshape(n_eps, -1)

                actions_t, _, _ = self.agent.get_actions_batched(
                    obs,
                    global_states,
                    deterministic=True,
                    action_masks=current_masks,
                )
                actions = actions_t.cpu().numpy()

                if self.discrete:
                    if actions.ndim == 3 and actions.shape[-1] == 1:
                        actions = actions.squeeze(-1)
                    actions = actions.astype(np.int32)

                obs, rewards, terminated, truncated, infos = self.eval_env.step(actions)
                current_masks = (
                    infos.get("avail_actions") if isinstance(infos, dict) else None
                )

                dones = np.logical_or(terminated, truncated)

                episode_rewards[~finished] += rewards[~finished]
                finished |= dones

        self.agent.network_old.train()
        return float(episode_rewards.mean())
