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
        entropy_conditioning: bool,
        hypergraph_runtime,
    ):
        self.eval_env = eval_env
        self.agent = agent
        self.n_eval_episodes = n_eval_episodes
        self.discrete = discrete
        self.entropy_conditioning = entropy_conditioning
        self.hypergraph_runtime = hypergraph_runtime

    def evaluate(self) -> float:
        """Evaluate current policy using parallel episodes."""
        self.agent.network_old.eval()
        n_eps = self.n_eval_episodes

        # HYGMA mode intentionally keeps groups frozen during eval.
        # Hypergraphs come from the latest discovered grouping state.
        with torch.no_grad():
            eval_seeds = [int(np.random.randint(0, 2**31)) for _ in range(n_eps)]
            obs, infos = self.eval_env.reset(seed=eval_seeds)
            current_masks = infos.get("avail_actions") if isinstance(infos, dict) else None
            episode_rewards = np.zeros(n_eps)
            finished = np.zeros(n_eps, dtype=bool)

            while not finished.all():
                global_states = obs.reshape(n_eps, -1)

                eval_hgs, eval_sig_ids = self.hypergraph_runtime.build_inference_hypergraphs(
                    obs, infos, n_eps
                )
                eval_entropies = (
                    self.hypergraph_runtime.compute_entropies_for_critic(eval_sig_ids)
                    if self.entropy_conditioning
                    else None
                )

                actions_t, _, _ = self.agent.get_actions_batched(
                    obs,
                    global_states,
                    deterministic=True,
                    action_masks=current_masks,
                    hypergraphs=eval_hgs,
                    entropies=eval_entropies,
                )
                actions = actions_t.cpu().numpy()

                if self.discrete:
                    if actions.ndim == 3 and actions.shape[-1] == 1:
                        actions = actions.squeeze(-1)
                    actions = actions.astype(np.int32)

                obs, rewards, terminated, truncated, infos = self.eval_env.step(actions)
                current_masks = infos.get("avail_actions") if isinstance(infos, dict) else None

                dones = np.logical_or(terminated, truncated)
                episode_rewards[~finished] += rewards[~finished]
                finished |= dones

        self.agent.network_old.train()
        return float(episode_rewards.mean())
