"""Check latent geometry and recording at the trained-policy view boundary."""

from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from algorithms.feudal_mappo_jax.goal_visualization import (
    ALIGNMENT_NEGATIVE,
    ALIGNMENT_NEUTRAL,
    ALIGNMENT_POSITIVE,
    alignment_rgb,
    frame_alignment,
    goal_alignment_scores,
    goal_following_figure,
    rasterize_panel,
    project_goal_outcomes,
    save_goal_alignment_plot,
    save_goal_plot,
    summarize_task_alignment,
    save_task_alignment_episode,
    save_task_alignment_overview,
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
        if kwargs.get("marker") in ("o", "*"):
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


@pytest.mark.parametrize("horizon", [1, 3])
def test_trajectory_windows_and_arrows_follow_states(monkeypatch, tmp_path, horizon):
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from algorithms.feudal_mappo_jax.goal_visualization import _project_goal_episode

    rng = np.random.default_rng(7)
    goals = rng.normal(size=(8, 2, 2))
    latents = rng.normal(size=(9, 2, 2))
    targets, states, _ = _project_goal_episode(goals, latents, horizon)
    paths, arrows = [], []
    original_plot, original_quiver = Axes.plot, Axes.quiver

    def record_plot(ax, x, y, **kwargs):
        paths.append(np.column_stack((x, y)))
        return original_plot(ax, x, y, **kwargs)

    def record_quiver(ax, x, y, u, v, **kwargs):
        arrows.append((np.column_stack((x, y)), np.column_stack((u, v))))
        return original_quiver(ax, x, y, u, v, **kwargs)

    monkeypatch.setattr(Axes, "plot", record_plot)
    monkeypatch.setattr(Axes, "quiver", record_quiver)
    monkeypatch.setattr(Figure, "savefig", lambda *args, **kwargs: None)
    save_goal_plot(goals, latents, horizon, tmp_path / "goals.png", 0, data_pairs=2)

    assert len(paths) == 4  # Separate windows for each agent; no connecting gaps.
    assert len(arrows) == 2
    for agent in range(2):
        for pair, end in enumerate((4, 8)):
            start = end - horizon
            path = paths[2 * agent + pair]
            np.testing.assert_allclose(path, states[start:end + 1, agent])
            # In 2D, the joint PCA must preserve state steps AND goal vectors.
            np.testing.assert_allclose(
                np.linalg.norm(np.diff(path, axis=0), axis=-1),
                np.linalg.norm(np.diff(latents[start:end + 1, agent], axis=0), axis=-1),
            )
            assert np.linalg.norm(targets[start, agent] - path[0]) == pytest.approx(
                np.linalg.norm(goals[start, agent])
            )
        origins, directions = arrows[agent]
        expected_paths = paths[2 * agent:2 * agent + 2]
        np.testing.assert_allclose(origins, np.concatenate([p[:-1] for p in expected_paths]))
        np.testing.assert_allclose(origins + directions,
                                   np.concatenate([p[1:] for p in expected_paths]))


def test_alignment_labels_use_full_dimensions_and_identify_their_trajectory(monkeypatch, tmp_path):
    from matplotlib.figure import Figure

    goals = np.zeros((4, 3, 3))
    goals[:, :2, 2] = 1
    latents = np.zeros((5, 3, 3))
    # Agent 1 goes along then against the goal in the third dimension.
    latents[:, 0, 2] = [0, 1, 2, 1, 0]
    # Agent 2 goes perpendicular, then has zero net displacement.
    latents[:, 1, 0] = [0, 50, 100, 100, 100]
    # Agent 3 moves but has no goal direction.
    latents[:, 2, 1] = [0, 50, 100, 150, 200]
    _, reached, _ = project_goal_outcomes(goals, latents, 2)
    labels = []

    def capture_figure(fig, *args, **kwargs):
        labels.append([(text.get_text(), text.xy) for text in fig.axes[0].texts
                       if "cos=" in text.get_text()])

    monkeypatch.setattr(Figure, "savefig", capture_figure)
    save_goal_plot(goals, latents, 2, tmp_path / "goals.png", 0, data_pairs=2)
    expected_scores = [("+1.00", "-1.00"), ("+0.00", "N/A"), ("N/A", "N/A")]
    assert len(labels) == 3
    for agent, scores in enumerate(expected_scores):
        assert len(labels[agent]) == 2
        for pair, start in enumerate((0, 2)):
            text, position = labels[agent][pair]
            assert text == f"t={start}–{start + 2}\ncos={scores[pair]}"
            np.testing.assert_allclose(position, reached[start, agent])


@pytest.mark.parametrize("goal_space", ["latent", "position_direction"])
def test_alignment_scores_keep_time_agents_dimensions_and_worker_goals(goal_space):
    goals = np.zeros((4, 2, 3))
    goals[:, 0, 2] = 1
    goals[:, 1, 0] = 1
    latents = np.zeros((5, 2, 3))
    latents[:, 0, 2] = [0, 1, 2, 1, 0]
    latents[:, 1, 1] = [0, 10, 20, 30, 40]
    pooled = np.zeros_like(goals)
    pooled[:, 0, 2] = [1, -1, -1, 1]
    pooled[:, 1, 1] = 1

    horizon, one_step = goal_alignment_scores(
        goals, latents, pooled, 2, goal_space=goal_space,
    )
    np.testing.assert_allclose(horizon, [[1, 0], [np.nan, 0], [-1, 0], [np.nan, np.nan]])
    np.testing.assert_allclose(one_step, [[1, 1], [-1, 1], [1, 1], [-1, 1]])


def test_alignment_masks_inactive_windows_but_keeps_terminal_transition():
    goals = np.ones((4, 2, 1))
    latents = np.broadcast_to(np.arange(5)[:, None, None], (5, 2, 1))
    active = [[1, 1], [1, 1], [0, 1], [0, 1]]
    horizon, one_step = goal_alignment_scores(goals, latents, goals, 2, active=active)
    np.testing.assert_allclose(horizon[:, 0], [1, np.nan, np.nan, np.nan])
    np.testing.assert_allclose(horizon[:, 1], [1, 1, 1, np.nan])
    np.testing.assert_allclose(one_step[:, 0], [1, 1, np.nan, np.nan])
    np.testing.assert_allclose(one_step[:, 1], [1, 1, 1, 1])


def test_waypoint_alignment_uses_latches_and_live_target_direction():
    # Targets latch at x=1 then x=-0.2. Proposals at t=1,3 are never used.
    goals = np.array([[1, 0], [-1, 0], [-1, 0], [1, 0]])[:, None]
    latents = np.array([[0, 0], [0.4, 0], [0.8, 0], [0.6, 0], [0.5, 0]])[:, None]
    pooled = np.array([[1, 0], [0.6, 0], [-1, 0], [-0.8, 0]])[:, None]
    horizon, one_step = goal_alignment_scores(
        goals, latents, pooled, 2, goal_space="position_waypoint",
    )
    np.testing.assert_allclose(horizon[:, 0], [1, np.nan, 1, np.nan])
    np.testing.assert_allclose(one_step[:, 0], [1, 1, 1, 1])


@pytest.mark.parametrize("steps,horizon", [(0, 2), (1, 1), (1, 5), (4, 2)])
def test_alignment_handles_zero_vectors_and_short_episodes(tmp_path, steps, horizon):
    goals = np.ones((steps, 2, 2))
    goals[:, 0] = 0
    latents = np.zeros((steps + 1, 2, 2))
    latents[:, 0, 0] = np.arange(steps + 1)  # Moving agent, no goal.
    # Agent 2 has a goal but is stationary.
    scores = goal_alignment_scores(goals, latents, goals, horizon)
    for score in scores:
        assert score.shape == (steps, 2)
        assert np.isnan(score).all()
    before = plt.get_fignums()
    save_goal_alignment_plot(goals, latents, goals, horizon, tmp_path / "alignment.png", 0)
    for agent in (1, 2):
        assert (tmp_path / f"alignment_agent_{agent}.png").read_bytes().startswith(b"\x89PNG")
    assert plt.get_fignums() == before


@pytest.mark.parametrize("horizon", [0, -1, 1.5, True])
def test_alignment_rejects_invalid_horizon(horizon):
    with pytest.raises(ValueError, match="positive integer"):
        goal_alignment_scores(np.ones((2, 1, 2)), np.ones((3, 1, 2)),
                              np.ones((2, 1, 2)), horizon)


@pytest.mark.parametrize("field,value,match", [
    ("latents", np.zeros((2, 1, 2)), "latents"),
    ("goals", np.full((2, 1, 2), np.nan), "finite"),
    ("pooled_goals", np.zeros((2, 2)), "pooled"),
    ("pooled_goals", np.full((2, 1, 2), np.inf), "finite"),
    ("active", np.ones((2,)), "activity"),
    ("goal_space", "unknown", "goal space"),
])
def test_alignment_rejects_misaligned_or_nonfinite_inputs(field, value, match):
    args = dict(goals=np.ones((2, 1, 2)), latents=np.ones((3, 1, 2)),
                pooled_goals=np.ones((2, 1, 2)), horizon=1)
    args[field] = value
    with pytest.raises(ValueError, match=match):
        goal_alignment_scores(**args)


def test_alignment_plot_preserves_every_timestep_and_shows_both_panels(monkeypatch, tmp_path):
    from matplotlib.figure import Figure

    goals = np.ones((4, 1, 1))
    latents = np.array([0, 1, 2, 1, 0]).reshape(5, 1, 1)
    figures = []
    monkeypatch.setattr(Figure, "savefig", lambda fig, *args, **kwargs: figures.append(fig))
    before = plt.get_fignums()
    save_goal_alignment_plot(
        goals, latents, goals, 2, tmp_path / "alignment.png", 7, macro_steps=True,
    )
    assert len(figures) == 1
    horizon_ax, step_ax = figures[0].axes
    for ax in (horizon_ax, step_ax):
        np.testing.assert_array_equal(ax.lines[0].get_xdata(), [0, 1, 2, 3])
        assert ax.get_ylim() == (-1.05, 1.05)
        np.testing.assert_array_equal(ax.lines[1].get_ydata(), [0, 0])
    np.testing.assert_allclose(horizon_ax.lines[0].get_ydata(), [1, np.nan, -1, np.nan])
    np.testing.assert_allclose(step_ax.lines[0].get_ydata(), [1, 1, -1, -1])
    assert len(horizon_ax.patches) == 1  # Incomplete horizon tail.
    assert "macro decision" in step_ax.get_xlabel()
    assert "Episode 7" in figures[0]._suptitle.get_text()
    assert plt.get_fignums() == before


def test_task_alignment_windows_match_rewards_and_exclude_crossing_horizons():
    goals = np.ones((8, 2, 1))
    states = np.zeros((9, 2, 1))
    states[:, 0, 0] = [0, 1, 2, 3, 2, 1, 0, -1, -2]
    rewards = [1, 2, 3, 4, 10, 20, 30, 40]
    summary = summarize_task_alignment(
        goals, states, goals, 2, rewards, episode=3, max_windows=2,
    )
    first, second = summary["windows"]
    assert [(w["start"], w["end"]) for w in summary["windows"]] == [(0, 4), (4, 8)]
    assert [w["task_return"] for w in summary["windows"]] == [10, 100]
    assert summary["task_return"] == 110  # Team return, counted once.
    # The horizon issued at t=3 goes backwards but ends in the next window.
    np.testing.assert_allclose(first["horizon_alignment"], [1, np.nan])
    np.testing.assert_allclose(second["horizon_alignment"], [-1, np.nan])
    np.testing.assert_allclose(first["one_step_alignment"], [0.5, np.nan])
    np.testing.assert_allclose(second["one_step_alignment"], [-1, np.nan])
    # The episode summary includes every complete horizon, even across bins.
    np.testing.assert_allclose(summary["horizon_alignment"], [-1 / 3, np.nan])


def test_task_alignment_summaries_keep_agents_equally_weighted():
    goals = np.ones((4, 2, 1))
    states = np.arange(5)[:, None, None] * np.array([[[1], [-1]]])
    active = [[1, 1], [1, 0], [1, 0], [1, 0]]
    summary = summarize_task_alignment(
        goals, states, goals, 1, [1, 2, 3, 4], episode=0, active=active,
    )
    # Agent 1 has four scores; agent 2 has one. Preserve their separate means.
    np.testing.assert_allclose(summary["horizon_alignment"], [1, -1])
    np.testing.assert_allclose(summary["one_step_alignment"], [1, -1])
    assert summary["task_return"] == 10


def test_task_alignment_windows_follow_waypoint_latches_and_keep_short_tail():
    goals = np.ones((10, 1, 1))
    states = np.arange(11).reshape(11, 1, 1)
    summary = summarize_task_alignment(
        goals, states, goals, 3, np.ones(10), episode=0,
        max_windows=3, goal_space="position_waypoint",
    )
    assert [(w["start"], w["end"]) for w in summary["windows"]] == [(0, 6), (6, 10)]
    assert all(w["start"] % 3 == 0 for w in summary["windows"])
    np.testing.assert_allclose(summary["horizon_alignment"], [1])
    for window in summary["windows"]:
        np.testing.assert_allclose(window["horizon_alignment"], [1])


@pytest.mark.parametrize("steps,horizon", [(0, 2), (1, 5), (4, 2)])
def test_task_alignment_plots_handle_missing_scores_and_write_pngs(tmp_path, steps, horizon):
    goals = np.ones((steps, 2, 1))
    states = np.arange(steps + 1)[:, None, None] * np.ones((1, 2, 1))
    summary = summarize_task_alignment(
        goals, states, goals, horizon, np.ones(steps), episode=0,
        active=np.zeros((steps, 2)),  # All scores absent; returns still valid.
    )
    assert np.isnan(summary["horizon_alignment"]).all()
    assert np.isnan(summary["one_step_alignment"]).all()
    assert summary["task_return"] == steps
    before = plt.get_fignums()
    save_task_alignment_episode(summary, tmp_path / "episode.png")
    save_task_alignment_overview([summary], tmp_path / "overview.png")
    for name in ("episode.png", "overview.png"):
        assert (tmp_path / name).read_bytes().startswith(b"\x89PNG")
    assert plt.get_fignums() == before


@pytest.mark.parametrize("rewards", [[1], [[1], [2]], [1, np.nan]])
def test_task_alignment_rejects_rewards_not_matching_policy_steps(rewards):
    with pytest.raises(ValueError, match="one per policy timestep"):
        summarize_task_alignment(
            np.ones((2, 1, 1)), np.ones((3, 1, 1)), np.ones((2, 1, 1)),
            1, rewards, episode=0,
        )


def test_task_alignment_overview_pairs_returns_and_agent_means(monkeypatch, tmp_path):
    from matplotlib.collections import PathCollection
    from matplotlib.figure import Figure

    summaries = [
        dict(episode=0, steps=4, task_return=7,
             horizon_alignment=np.array([1, -1, np.nan]),
             one_step_alignment=np.array([0.25, 0.75, np.nan])),
        dict(episode=1, steps=2, task_return=-2,
             horizon_alignment=np.array([-0.5, np.nan, np.nan]),
             one_step_alignment=np.full(3, np.nan)),
    ]
    saved = []
    monkeypatch.setattr(Figure, "savefig", lambda fig, *args, **kwargs: saved.append(fig))
    before = plt.get_fignums()
    save_task_alignment_overview(summaries, tmp_path / "task_vs_alignment.png")
    assert len(saved) == 1
    task_ax, horizon_ax, step_ax = saved[0].axes
    assert [bar.get_height() for bar in task_ax.patches] == [7, -2]
    assert [bar.get_height() for bar in horizon_ax.patches] == [0, -0.5]
    assert [bar.get_height() for bar in step_ax.patches] == [0.5]
    assert any(text.get_text() == "N/A" for text in step_ax.texts)
    dots = [item for item in horizon_ax.collections if isinstance(item, PathCollection)]
    np.testing.assert_allclose(dots[0].get_offsets()[:, 1], [1, -1])
    assert task_ax.patches[0].get_facecolor() == horizon_ax.patches[0].get_facecolor()
    assert task_ax.get_shared_x_axes().joined(task_ax, step_ax)
    assert plt.get_fignums() == before


@pytest.mark.parametrize("detailed", [False, True])
@pytest.mark.parametrize("kind,goal_space", [
    ("mjx", "latent"), ("macro", "latent"), ("env_renderer", "latent"),
    ("mjx", "position_direction"), ("macro", "position_direction"),
    ("mjx", "position_waypoint"), ("macro", "position_waypoint"),
])
def test_view_records_actual_manager_outputs_and_final_observation(monkeypatch, tmp_path, kind, goal_space, detailed):
    import jax
    import imageio
    from algorithms.feudal_mappo_jax import goal_visualization, mappo, network, worker
    from algorithms.feudal_mappo_jax.run import Feudal_MAPPO_JAX_Runner
    from environments.mjx_suite import renderer

    # Exercise both Python view loops, including early termination and macro
    # boundaries, without compiling physics or opening a graphics context.
    monkeypatch.setattr(jax, "jit", lambda fn: fn)
    from matplotlib.figure import Figure
    monkeypatch.setattr(Figure, "savefig", lambda *args, **kwargs: None)
    goal_dim = 3 if goal_space == "latent" else 2

    class Env:
        n_agents = 2
        observation_dim = 3
        max_steps = 5
        # Continuous on the flat MJX path, so the goal-influence arrows run.
        discrete = kind != "mjx"

        def reset(self, key):
            return np.zeros((2, 3)), SimpleNamespace(t=0, key=key, env_state=0)

        def step(self, state, actions):
            t = state.t + 1
            info = {"task_reward": 1.0}
            if kind in ("macro", "env_renderer"):
                info["active"] = np.array([1, state.t == 0])
            return (np.full((2, 3), t), SimpleNamespace(t=t, key=state.key, env_state=t),
                    0, t == 3, False, info)

        @staticmethod
        def base_state(state):
            return state

        def goal_state(self, state):
            return np.full((2, 2), state.t)

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
            return carry + 1, np.full((2, goal_dim), carry + 1), obs * 10

    conditioned = []

    def record_binding(apply_fn, goal):
        conditioned.append(np.array(goal))
        return goal

    def goal_dependent_action(key, goal, params, obs, **kwargs):
        # A goal-sensitive stand-in policy, so the influence arrow is nonzero.
        return np.repeat(0.1 * np.asarray(goal).sum(-1, keepdims=True), 2, -1), None

    monkeypatch.setattr(worker, "bind_goal", record_binding)
    monkeypatch.setattr(mappo, "build_manager", lambda config, n_agents: Manager())
    monkeypatch.setattr(network, "sample_action", goal_dependent_action)

    def annotate(frame, draw):
        import pygame
        draw(pygame.Surface(frame.shape[1::-1]), lambda x, y: (int(x), int(y)), 1.0)
        return frame

    dummy_renderer = lambda env: SimpleNamespace(  # noqa: E731
        render=lambda *args, **kwargs: np.zeros((300, 300, 3), np.uint8),
        agent_positions=lambda state: np.full((2, 2), float(state.t)),
        annotate=annotate,
    )
    monkeypatch.setattr(renderer, "MJXRenderer", dummy_renderer)
    monkeypatch.setattr(renderer, "MuJoCoNativeRenderer", dummy_renderer)
    monkeypatch.setattr(imageio, "mimwrite", lambda *args, **kwargs: None)
    written = []

    class Writer:
        def __init__(self, path, **kwargs):
            self.path, self.frames = path, []

        def append_data(self, frame):
            self.frames.append(frame)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            written.append(self)

    monkeypatch.setattr(imageio, "get_writer", Writer)
    from algorithms.feudal_mappo_jax import goal_video
    rasters, goal_videos = [], []
    original_raster = goal_visualization.save_goal_following_raster
    original_video = goal_video.save_goal_video

    def record_raster(ring, dot, rewards, path, **kwargs):
        rasters.append((np.array(ring), np.array(rewards), path))
        return original_raster(ring, dot, rewards, path, **kwargs)

    def record_video(path, renderer_, frames, **kwargs):
        goal_videos.append((path, len(frames), kwargs))
        return original_video(path, renderer_, frames, **kwargs)

    monkeypatch.setattr(goal_visualization, "save_goal_following_raster", record_raster)
    monkeypatch.setattr(goal_video, "save_goal_video", record_video)
    recorded = []
    monkeypatch.setattr(goal_visualization, "save_goal_plot",
                        lambda goals, latents, horizon, path, episode:
                        recorded.append((np.array(goals), np.array(latents), horizon, path)))

    alignments = []

    def record_alignment(goals, latents, pooled, horizon, path, episode, **kwargs):
        alignments.append((np.array(goals), np.array(latents), np.array(pooled),
                           horizon, path, episode, kwargs))

    monkeypatch.setattr(goal_visualization, "save_goal_alignment_plot", record_alignment)
    runner = object.__new__(Feudal_MAPPO_JAX_Runner)
    runner.env = env
    # A stand-in for MAPPOConfig. Every key `view()` reads must be present:
    # a SimpleNamespace raises AttributeError rather than falling back, so a
    # missing one fails the test instead of silently taking a default path.
    # (`worker_encoder` was already missing here before `goal_space` was added —
    # these three cases have been failing at HEAD.)
    runner.config = SimpleNamespace(
        goal_horizon=2,
        goal_dim=goal_dim,
        worker_encoder="none",
        goal_space=goal_space,
        manager_latent_dim=None,
        waypoint_radius=1.0 / 3.0,
    )
    runner.dirs = {"logs": tmp_path}
    runner.rng_seed = 0
    runner._load_train_state = lambda: SimpleNamespace(
        manager_ts=SimpleNamespace(params={}),
        actor_ts=SimpleNamespace(params={}, apply_fn=lambda *args: None),
    )
    summary_inputs, episode_plots, overview_plots = [], [], []
    original_summary = goal_visualization.summarize_task_alignment

    def record_summary(goals, latents, pooled, horizon, task_rewards, **kwargs):
        summary_inputs.append((np.array(goals), np.array(latents), np.array(pooled),
                               horizon, np.array(task_rewards), kwargs))
        return original_summary(goals, latents, pooled, horizon, task_rewards, **kwargs)

    monkeypatch.setattr(goal_visualization, "summarize_task_alignment", record_summary)
    monkeypatch.setattr(goal_visualization, "save_task_alignment_episode",
                        lambda summary, path: episode_plots.append((summary, path)))
    monkeypatch.setattr(goal_visualization, "save_task_alignment_overview",
                        lambda summaries, path: overview_plots.append((list(summaries), path)))
    if detailed:
        runner.view(detailed_goal_plots=True)
    else:
        runner.view()
    episodes = 3 if kind == "env_renderer" else 10
    assert len(summary_inputs) == len(episode_plots) == episodes
    assert len(recorded) == len(alignments) == (episodes if detailed else 0)
    assert len(overview_plots) == 1
    assert overview_plots[0][1] == tmp_path / "task_vs_alignment.png"
    assert all(summary is plotted[0] for summary, plotted in
               zip(overview_plots[0][0], episode_plots))

    expected_times = [0, 2, 3] if kind == "macro" else [0, 1, 2, 3]
    steps = len(expected_times) - 1
    if kind == "mjx":
        # Every step conditions the worker twice: the real pooled goal, then the
        # ZERO goal the influence arrow is measured against.
        assert all((null == 0).all() for null in conditioned[1::2])
        conditioned = conditioned[0::2]

    # The raster is written for every kind; the goal video only where the view
    # owns the frames (not jaxmarl's visualizer).
    assert [path for *_, path in rasters] == [
        tmp_path / f"goal_following_episode_{e}.png" for e in range(episodes)]
    frames = 3  # low-level frames in every stub episode (macro: 2 + 1)
    for ring, rewards, _ in rasters:
        assert ring.shape == (frames, 2) and len(rewards) == frames
        assert np.isnan(ring[0]).all()  # no complete horizon at the first frame
    if kind == "env_renderer":
        assert goal_videos == [] and written == []
    else:
        assert [v[0] for v in goal_videos] == [
            tmp_path / f"episode_{e}_goals.mp4" for e in range(episodes)]
        assert [len(w.frames) for w in written] == [frames] * episodes
        assert all(f.shape == (300, 600, 3) for w in written for f in w.frames)
        kwargs = goal_videos[0][2]
        np.testing.assert_array_equal(
            kwargs["frame_decision"], [0, 0, 1] if kind == "macro" else [0, 1, 2])
        if kind == "mjx":
            pooled0 = np.array(conditioned[:steps])
            expected = np.clip(np.repeat(0.1 * pooled0.sum(-1, keepdims=True), 2, -1), -1, 1)
            np.testing.assert_allclose(kwargs["influence"], expected)
        else:
            assert kwargs["influence"] is None
        assert (kwargs["goal_states"] is None) == (goal_space == "latent")
    for episode, (goals, latents, pooled, horizon, task_rewards, kwargs) in enumerate(summary_inputs):
        state_scale = 10 if goal_space == "latent" else 1
        np.testing.assert_array_equal(latents[:, 0, 0], np.array(expected_times) * state_scale)
        np.testing.assert_array_equal(goals[:, 0, 0], np.arange(1, len(expected_times)))
        assert latents.shape == (len(goals) + 1, 2, goal_dim)
        np.testing.assert_array_equal(pooled, conditioned[episode * steps:(episode + 1) * steps])
        expected_pool = ([1, -5] if kind == "macro" else [1, -2, 3]) if (
            goal_space == "position_waypoint"
        ) else ([1, 3] if kind == "macro" else [1, 3, 5])
        np.testing.assert_allclose(pooled[:, 0, 0], expected_pool)
        np.testing.assert_array_equal(task_rewards, [2, 1] if kind == "macro" else [1, 1, 1])
        assert horizon == 2 and kwargs["episode"] == episode
        assert kwargs["goal_space"] == goal_space
        assert kwargs["macro_steps"] == (kind == "macro")
        if kind == "macro":
            expected_active = [[1, 0], [1, 0]]
        elif kind == "env_renderer":
            expected_active = [[1, 1], [1, 0], [1, 0]]
        else:
            expected_active = np.ones((steps, 2))
        np.testing.assert_array_equal(kwargs["active"], expected_active)
        summary, path = episode_plots[episode]
        assert summary["task_return"] == 3
        assert summary["steps"] == steps
        assert sum(w["task_return"] for w in summary["windows"]) == 3
        assert path == tmp_path / f"task_vs_alignment_episode_{episode}.png"
        if detailed:
            np.testing.assert_array_equal(recorded[episode][0], goals)
            np.testing.assert_array_equal(alignments[episode][2], pooled)
            assert recorded[episode][3] == tmp_path / f"goals_episode_{episode}.png"
            assert alignments[episode][4] == tmp_path / f"goal_alignment_episode_{episode}.png"


def _random_episode(steps=12, agents=3, dim=4, seed=0):
    rng = np.random.default_rng(seed)
    goals = rng.normal(size=(steps, agents, dim))
    latents = np.cumsum(rng.normal(size=(steps + 1, agents, dim)), axis=0)
    pooled = rng.normal(size=(steps, agents, dim))
    return goals, latents, pooled


@pytest.mark.parametrize("goal_space", ["latent", "position_direction"])
def test_frame_alignment_is_causal_and_indexes_the_scores(goal_space):
    goals, latents, pooled = _random_episode()
    horizon = 4
    scores, one_step = goal_alignment_scores(
        goals, latents, pooled, horizon, goal_space=goal_space,
    )
    ring, dot = frame_alignment(
        scores, one_step, np.arange(len(goals)), horizon, goal_space=goal_space,
    )
    assert ring.shape == dot.shape == one_step.shape
    assert np.isnan(ring[:horizon]).all() and np.isnan(dot[0]).all()
    for d in range(horizon, len(goals)):
        # cos(g_(d-c), s_d - s_(d-c)): uses nothing past the frame's own state.
        np.testing.assert_array_equal(ring[d], scores[d - horizon])
    np.testing.assert_array_equal(dot[1:], one_step[:-1])


def test_frame_alignment_holds_one_decision_across_macro_frames():
    goals, latents, pooled = _random_episode(steps=6)
    scores, one_step = goal_alignment_scores(goals, latents, pooled, 2)
    decision = np.array([0, 0, 1, 1, 1, 2, 3, 3, 4, 5])
    ring, dot = frame_alignment(scores, one_step, decision, 2)
    for f, d in enumerate(decision):
        np.testing.assert_array_equal(dot[f], one_step[d - 1] if d else np.nan)
        np.testing.assert_array_equal(ring[f], scores[d - 2] if d >= 2 else np.nan)


def test_frame_alignment_holds_the_last_completed_waypoint_latch():
    goals, latents, pooled = _random_episode(steps=10)
    horizon = 3
    scores, one_step = goal_alignment_scores(
        goals, latents, pooled, horizon, goal_space="position_waypoint",
    )
    ring, _ = frame_alignment(
        scores, one_step, np.arange(10), horizon, goal_space="position_waypoint",
    )
    assert np.isnan(ring[:3]).all()
    for d in range(3, 10):
        latch = (d // horizon) * horizon - horizon
        np.testing.assert_array_equal(ring[d], scores[latch])
        assert np.isfinite(ring[d]).all()  # never the NaN between latches


@pytest.mark.parametrize("decision", [np.array([0, 5]), np.array([-1]), np.array([0.0, 1.0])])
def test_frame_alignment_rejects_decisions_outside_the_episode(decision):
    goals, latents, pooled = _random_episode(steps=5)
    with pytest.raises(ValueError):
        frame_alignment(*goal_alignment_scores(goals, latents, pooled, 2), decision, 2)


def test_alignment_colors_share_poles_and_midpoint():
    def rgb(hex_color):
        return [int(hex_color[i:i + 2], 16) for i in (1, 3, 5)]

    colors = alignment_rgb([1.0, -1.0, 0.0, np.nan])
    np.testing.assert_array_equal(colors[0], rgb(ALIGNMENT_POSITIVE))
    np.testing.assert_array_equal(colors[1], rgb(ALIGNMENT_NEGATIVE))
    np.testing.assert_allclose(colors[2], rgb(ALIGNMENT_NEUTRAL), atol=1)
    np.testing.assert_array_equal(colors[3], colors[2])


@pytest.mark.parametrize("agents", [1, 16])
def test_panel_cursor_moves_monotonically_inside_the_heatmaps(agents):
    frames = 300
    ring = np.linspace(-1, 1, frames * agents).reshape(frames, agents)
    ring[:10] = np.nan
    fig, axes = goal_following_figure(
        ring, ring[::-1], np.zeros(frames), horizon=10, goal_space="latent",
        episode=0, size_px=(700, 700), hud_px=64,
    )
    cursor_x, (top, bottom), image = rasterize_panel(fig, axes, frames)
    assert image.shape == (700, 700, 3) and image.dtype == np.uint8
    assert np.all(np.diff(cursor_x) >= 0) and cursor_x[-1] > cursor_x[0]
    heat = axes[1].get_window_extent()
    assert heat.x0 <= cursor_x[0] and cursor_x[-1] <= heat.x1
    assert 64 <= top < bottom <= 700  # below the HUD band


def test_goal_overlay_draws_ring_dot_and_arena_marks():
    import pygame

    from algorithms.feudal_mappo_jax import goal_video

    surface = pygame.Surface((200, 200))
    surface.fill((255, 255, 255))
    to_screen = lambda x, y: (int(x), int(200 - y))  # noqa: E731 - y flipped
    goal_video.draw_goal_overlay(
        surface, to_screen, 1.0, agent_pos=np.array([[50.0, 150.0], [150.0, 50.0]]),
        agent_radius=10, ring=np.array([1.0, np.nan]), dot=np.array([-1.0, np.nan]),
        heading=np.array([[0.0, 1.0], [np.nan, np.nan]]),
    )
    center = (50, 50)
    r_ring = 10 + 2 + 4
    assert tuple(surface.get_at((center[0] + r_ring - 2, center[1]))[:3]) == tuple(
        alignment_rgb(1.0))
    assert tuple(surface.get_at(center)[:3]) == tuple(alignment_rgb(-1.0))
    # Undefined ring: a thin muted outline, no colored band.
    assert tuple(surface.get_at((150 + r_ring - 2, 150))[:3]) == (255, 255, 255)
    # Heading +y in the world points UP on screen.
    # Mid-shaft: past the ring (r=16), short of the haloed head (last 12 px).
    assert tuple(surface.get_at((50, 50 - 25))[:3]) == goal_video._GOAL


def test_world_goal_marks_map_goal_space_into_the_arena():
    from algorithms.feudal_mappo_jax.goal_video import _world_goal_marks

    to_world = lambda s: np.asarray(s) * [30.0, 10.0] + [15.0, 5.0]  # noqa: E731
    s = np.zeros((3, 1, 2))
    goals = np.array([[[1.0, 1.0]], [[0.0, -2.0]]])
    heading, none = _world_goal_marks("position_direction", s, goals, None, None, to_world)
    assert none is None
    # A unit goal-space step is anisotropic in the world: (1,1) -> (30,10).
    np.testing.assert_allclose(heading[0, 0], np.array([30.0, 10.0]) / np.hypot(30, 10))
    np.testing.assert_allclose(heading[1, 0], [0.0, -1.0])
    none, waypoint = _world_goal_marks(
        "position_waypoint", s, None, goals, 0.5, to_world,
    )
    assert none is None
    np.testing.assert_allclose(waypoint[0, 0], [15.0 + 15.0, 5.0 + 5.0])
    assert _world_goal_marks("latent", s, goals, goals, 0.5, to_world) == (None, None)


def test_goal_video_streams_every_frame_beside_the_panel(monkeypatch, tmp_path):
    import imageio
    import pygame

    from algorithms.feudal_mappo_jax import goal_video

    written = []

    class Writer:
        def __init__(self, path, **kwargs):
            self.path, self.frames = path, []

        def append_data(self, frame):
            self.frames.append(frame)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            written.append(self)

    monkeypatch.setattr(imageio, "get_writer", Writer)
    trails = []
    original = goal_video.draw_goal_overlay

    def spy(*args, trail=None, **kwargs):
        trails.append(None if trail is None else len(trail))
        return original(*args, trail=trail, **kwargs)

    monkeypatch.setattr(goal_video, "draw_goal_overlay", spy)

    def annotate(frame, draw):
        surface = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
        draw(surface, lambda x, y: (int(x), int(300 - y)), 1.0)
        return np.transpose(pygame.surfarray.array3d(surface), (1, 0, 2))

    steps, agents, horizon = 7, 2, 3
    frames = [np.full((300, 300, 3), 255, np.uint8) for _ in range(steps)]
    goals, latents, pooled = _random_episode(steps, agents, 2)
    scores, one_step = goal_alignment_scores(
        goals, latents, pooled, horizon, goal_space="position_waypoint",
    )
    ring, dot = frame_alignment(scores, one_step, np.arange(steps), horizon,
                                goal_space="position_waypoint")
    goal_video.save_goal_video(
        tmp_path / "goals.mp4", SimpleNamespace(annotate=annotate), frames,
        agent_pos=np.full((steps, agents, 2), 100.0), frame_decision=np.arange(steps),
        ring=ring, dot=dot, rewards=np.arange(steps, dtype=float), horizon=horizon,
        goal_space="position_waypoint", episode=0, agent_radius=5,
        influence=np.zeros((steps, agents, 2)), goal_states=latents, goals=goals,
        pooled_goals=pooled, waypoint_radius=1.0 / 3.0, to_world=lambda s: s * 30 + 150,
    )
    assert len(written) == 1 and written[0].path == tmp_path / "goals.mp4"
    assert len(written[0].frames) == steps
    assert all(frame.shape == (300, 600, 3) for frame in written[0].frames)
    # Trail = the path since the waypoint latched (every `horizon` steps).
    assert trails == [1, 2, 3, 1, 2, 3, 1]
