"""Trained rover plots must preserve recurrent inference and episode boundaries."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from algorithms.rmappo_jax.mappo import create_train_state, zero_hidden
from algorithms.rmappo_jax.run import RecurrentViewPolicy
from algorithms.rmappo_jax.trainer import actor_step
from algorithms.rmappo_jax.types import RMAPPOConfig
from environments.rover.config import load_rover_env
from environments.rover.rollout import rollout_view_policy


@pytest.mark.parametrize("recurrent", [False, True], ids=["mappo_mlp", "rmappo"])
def test_view_matches_functional_policy_and_resets_between_episodes(recurrent):
    with jax.default_device(jax.devices("cpu")[0]):
        env = load_rover_env("rover_2poi_c2", ["env.max_steps=6"])
        config = RMAPPOConfig(
            n_envs=1, n_steps=10, n_total_steps=10, hidden_size=8,
            use_recurrent_policy=recurrent,
        )
        ts = create_train_state(
            jax.random.PRNGKey(9), config, env.observation_dim,
            env.n_agents * env.observation_dim, env.action_dim, False, 1,
        )
        key = jax.random.PRNGKey(5)
        obs, state = env.reset(key)

        # Independent functional rollout, carrying the GRU explicitly in scan.
        def step(carry, _):
            obs, state, hidden = carry
            action, _, hidden = actor_step(
                ts.actor_ts.apply_fn, ts.actor_ts.params, obs[None], hidden,
                jnp.ones((1,)), jax.random.PRNGKey(0), False, deterministic=True,
            )
            obs, state, reward, _, _, _ = env.step(state, action[0])
            return (obs, state, hidden), (action[0], state.rovers.position, reward)

        _, (actions, positions, rewards) = jax.lax.scan(
            step, (obs, state, zero_hidden(config, (1, env.n_agents))),
            None, length=env.max_steps,
        )
        policy = RecurrentViewPolicy(ts, env, config)
        first = rollout_view_policy(env, policy, key)
        second = rollout_view_policy(env, policy, key)
        np.testing.assert_allclose(first.actions, actions, atol=1e-6)
        np.testing.assert_allclose(first.positions[1:], positions, atol=1e-6)
        np.testing.assert_allclose(first.rewards, rewards, atol=1e-6)
        np.testing.assert_array_equal(first.positions[0], state.rovers.position)
        assert first.done.tolist() == [False] * 5 + [True]
        for a, b in zip(jax.tree.leaves(first), jax.tree.leaves(second)):
            np.testing.assert_array_equal(a, b)


def test_view_stops_on_early_termination():
    env = load_rover_env("rover_2poi_c2", [
        "env.pois.radius=100", "env.pois.rate=1", "env.max_steps=10",
    ])

    class StillPolicy:
        def reset(self):
            self.calls = 0

        def __call__(self, obs):
            self.calls += 1
            return jnp.tile(jnp.array([0.0, -1.0]), (env.n_agents, 1))

    with jax.default_device(jax.devices("cpu")[0]):
        policy = StillPolicy()
        traj = rollout_view_policy(env, policy, jax.random.PRNGKey(0))
        assert policy.calls == 1
        assert traj.positions.shape[0] == 2
        assert traj.done.tolist() == [True]
        assert float(traj.episode_return) == pytest.approx(2.0)
