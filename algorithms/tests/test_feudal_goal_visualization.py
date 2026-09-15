"""Check latent geometry and recording at the trained-policy view boundary."""

from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from algorithms.feudal_mappo_jax.goal_visualization import (
    project_goal_outcomes,
    save_goal_plot,
)


def test_projection_pairs_horizon_outcomes_and_anchors_directions():
    # In 2D PCA preserves distances. Different agent/time values expose either
    # an agent-axis mixup, a one-step outcome, or plotting g as an absolute s.
    latents = np.arange(5 * 2 * 2).reshape(5, 2, 2) ** 2
    goals = np.arange(4 * 2 * 2).reshape(4, 2, 2) / 10
    targets, reached, variance = project_goal_outcomes(goals, latents, 2)
    assert targets.shape == reached.shape == (3, 2, 2)
    expected = latents[:3] + goals[:3] - latents[2:]
    np.testing.assert_allclose(np.linalg.norm(targets - reached, axis=-1),
                               np.linalg.norm(expected, axis=-1))
    # All points share the same projection, including points across agents.
    original = np.concatenate(((latents[:3] + goals[:3]).reshape(-1, 2),
                               latents[2:].reshape(-1, 2)))
    projected = np.concatenate((targets.reshape(-1, 2), reached.reshape(-1, 2)))
    np.testing.assert_allclose(
        np.linalg.norm(projected[:, None] - projected[None], axis=-1),
        np.linalg.norm(original[:, None] - original[None], axis=-1), atol=1e-10,
    )
    assert variance.sum() == pytest.approx(1)


@pytest.mark.parametrize("steps,dim,horizon", [(1, 1, 1), (3, 4, 2), (1, 4, 5)])
def test_plot_handles_collapsed_latents_and_short_episodes(tmp_path, steps, dim, horizon):
    goals = np.zeros((steps, 2, dim))
    latents = np.zeros((steps + 1, 2, dim))
    targets, reached, variance = project_goal_outcomes(goals, latents, horizon)
    assert targets.shape == reached.shape == (max(0, steps - horizon + 1), 2, 2)
    assert not np.any(variance)
    path = tmp_path / "goals.png"
    before = plt.get_fignums()
    save_goal_plot(goals, latents, horizon, path, episode=0)
    assert not path.exists()
    for agent in (1, 2):
        assert (tmp_path / f"goals_agent_{agent}.png").read_bytes().startswith(b"\x89PNG")
    assert plt.get_fignums() == before


@pytest.mark.parametrize("steps,horizon,data_pairs,indices", [
    (1000, 10, 4, [240, 490, 740, 990]),
    (1000, 10, 1, [990]),
    (10, 8, 2, [0, 2]),
    (5, 2, 20, [0, 1, 2, 3]),
    (5, 2, None, [0, 1, 2, 3]),
    (1, 5, 4, []),
])
def test_displayed_pairs_are_spread_over_episode(
    monkeypatch, tmp_path, steps, horizon, data_pairs, indices,
):
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    rng = np.random.default_rng(42)
    goals = rng.normal(size=(steps, 2, 3))
    latents = rng.normal(size=(steps + 1, 2, 3))
    targets, reached, _ = project_goal_outcomes(goals, latents, horizon)
    displayed = []
    original_scatter = Axes.scatter

    def record_scatter(ax, x, y, **kwargs):
        displayed.append(np.column_stack((x, y)))
        return original_scatter(ax, x, y, **kwargs)

    monkeypatch.setattr(Axes, "scatter", record_scatter)
    monkeypatch.setattr(Figure, "savefig", lambda *args, **kwargs: None)
    save_goal_plot(goals, latents, horizon, tmp_path / "goals.png", 0,
                   data_pairs=data_pairs)
    assert len(displayed) == 4
    for agent in range(2):
        np.testing.assert_allclose(displayed[2 * agent], reached[indices, agent])
        np.testing.assert_allclose(displayed[2 * agent + 1], targets[indices, agent])


@pytest.mark.parametrize("data_pairs", [0, -1, 2.5, True])
def test_data_pairs_requires_positive_integer(tmp_path, data_pairs):
    with pytest.raises(ValueError, match="data_pairs"):
        save_goal_plot(np.zeros((2, 1, 2)), np.zeros((3, 1, 2)), 1,
                       tmp_path / "goals.png", 0, data_pairs=data_pairs)


@pytest.mark.parametrize("kind", ["mjx", "macro", "env_renderer"])
def test_view_records_actual_manager_outputs_and_final_observation(monkeypatch, tmp_path, kind):
    import jax
    import imageio
    from algorithms.feudal_mappo_jax import goal_visualization, mappo, network
    from algorithms.feudal_mappo_jax.run import Feudal_MAPPO_JAX_Runner
    from environments.mjx_suite import renderer

    # Exercise both Python view loops, including early termination and macro
    # boundaries, without compiling physics or opening a graphics context.
    monkeypatch.setattr(jax, "jit", lambda fn: fn)

    class Env:
        n_agents = 2
        observation_dim = 3
        max_steps = 5
        discrete = True

        def reset(self, key):
            return np.zeros((2, 3)), SimpleNamespace(t=0, key=key, env_state=0)

        def step(self, state, actions):
            t = state.t + 1
            return (np.full((2, 3), t), SimpleNamespace(t=t, key=state.key, env_state=t),
                    0, t == 3, False, {"task_reward": 1.0})

        def global_state(self, state):
            return np.array([state.t + 100])

        def avail_actions(self, state):
            return np.ones((2, 2))

        def to_action_dict(self, actions):
            return {}

    env = Env()
    if kind == "macro":
        env.env = Env()
        env.macro_len = 2
        env.base_state = lambda state: state
        env._skill_actions = lambda state, skills: skills
    elif kind == "env_renderer":
        env.render_episode = lambda states, path: None

    class Manager:
        def initialize_carry(self, key, shape):
            return 0

        def apply(self, params, carry, gs, obs):
            if kind == "env_renderer":
                assert gs[0] == obs[0, 0] + 100
            return carry + 1, np.full((2, 3), carry + 1), obs * 10

    monkeypatch.setattr(mappo, "build_manager", lambda config, n_agents: Manager())
    monkeypatch.setattr(network, "sample_action", lambda *args, **kwargs: (np.zeros(2), None))
    dummy_renderer = lambda env: SimpleNamespace(render=lambda *args, **kwargs: np.zeros((2, 2, 3)))
    monkeypatch.setattr(renderer, "MJXRenderer", dummy_renderer)
    monkeypatch.setattr(renderer, "MuJoCoNativeRenderer", dummy_renderer)
    monkeypatch.setattr(imageio, "mimwrite", lambda *args, **kwargs: None)
    recorded = []
    monkeypatch.setattr(goal_visualization, "save_goal_plot",
                        lambda goals, latents, horizon, path, episode:
                        recorded.append((np.array(goals), np.array(latents), horizon, path)))

    runner = object.__new__(Feudal_MAPPO_JAX_Runner)
    runner.env = env
    runner.config = SimpleNamespace(goal_horizon=2, goal_dim=3)
    runner.dirs = {"logs": tmp_path}
    runner.rng_seed = 0
    runner._load_train_state = lambda: SimpleNamespace(
        manager_ts=SimpleNamespace(params={}),
        actor_ts=SimpleNamespace(params={}, apply_fn=lambda *args: None),
    )
    runner.view()
    assert len(recorded) == (3 if kind == "env_renderer" else 10)
    expected_times = [0, 2, 3] if kind == "macro" else [0, 1, 2, 3]
    for episode, (goals, latents, horizon, path) in enumerate(recorded):
        np.testing.assert_array_equal(latents[:, 0, 0], np.array(expected_times) * 10)
        np.testing.assert_array_equal(goals[:, 0, 0], np.arange(1, len(expected_times)))
        assert latents.shape == (len(goals) + 1, 2, 3)
        assert horizon == 2
        assert path == tmp_path / f"goals_episode_{episode}.png"
