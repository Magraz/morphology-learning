"""``BaseRoverEnv``: the algorithm-independent core of the rover domain.

A 2-D, physics-free multi-rover world with a functional, ``jit``/``vmap``-able
API in the style of the MJX envs::

    obs, state = env.reset(key)
    obs, state, reward, terminated, truncated, info = env.step(state, actions)

There is no auto-reset (the caller's job) and no PRNG key in ``step``, so a
rollout is a pure function of the reset key and the actions. That serves
evolutionary algorithms (vmap a rollout over a population) and RL alike.

**Dynamics.** Each rover has a position and a heading. The action is
``a = [turn, speed]`` in ``[-1, 1]^2``::

    heading <- wrap(heading + a[0] * max_turn)
    speed    = (a[1] + 1) / 2 * max_speed          # set directly, not integrated
    position <- clip(position + speed * (cos heading, sin heading), arena)

Walls are inert: a rover pressed against one slides along it and the episode
does not end.

**Components.** What a rover sees, what POIs exist and what the team is paid
are pluggable (see ``components.py``). Specific variants either pass other
components or subclass this env (e.g. override ``_spawn_rovers`` or ``move``).
"""

import math

import jax
import jax.numpy as jnp

from environments.rover.observations import SectorDensityObservation
from environments.rover.rewards import CouplingDepletionReward
from environments.rover.state import RoverEnvState, RoverState


def wrap_angle(x):
    """Wrap radians to [-pi, pi)."""
    return (x + jnp.pi) % (2 * jnp.pi) - jnp.pi


class BaseRoverEnv:
    """Multi-rover domain with pluggable POIs, observations and reward.

    Args:
        n_agents: number of rovers.
        pois: a ``POIModel`` (required: the layout is what defines a variant).
        observation: an ``ObservationModel``; default ``SectorDensityObservation``.
        reward: a ``RewardModel``; default ``CouplingDepletionReward``.
        world_size: arena ``(width, height)``; positions live in
            ``[0, width] x [0, height]``.
        spawn_region: ``(x_min, x_max, y_min, y_max)`` rovers are spawned in,
            uniformly, with a uniform heading. Default: the whole arena.
        max_speed: distance covered per step at ``a[1] = 1``.
        max_turn: heading change (radians) per step at ``|a[0]| = 1``.
        max_steps: episode length; ``truncated`` fires at this step.
    """

    action_dim = 2
    discrete = False

    def __init__(
        self,
        n_agents,
        pois,
        observation=None,
        reward=None,
        *,
        world_size=(10.0, 10.0),
        spawn_region=None,
        max_speed=0.5,
        max_turn=math.pi / 4,
        max_steps=100,
    ):
        self.n_agents = int(n_agents)
        if self.n_agents < 1:
            raise ValueError(f"n_agents must be >= 1, got {n_agents}")
        width, height = (float(v) for v in world_size)
        if width <= 0 or height <= 0:
            raise ValueError(f"world_size must be positive, got {world_size}")
        self.world_size = (width, height)
        self.world_diagonal = math.hypot(width, height)

        if spawn_region is None:
            spawn_region = (0.0, width, 0.0, height)
        x_min, x_max, y_min, y_max = (float(v) for v in spawn_region)
        if not (0 <= x_min <= x_max <= width and 0 <= y_min <= y_max <= height):
            raise ValueError(
                f"spawn_region {spawn_region} must satisfy 0 <= min <= max <= "
                f"the arena size {self.world_size}"
            )
        self.spawn_region = (x_min, x_max, y_min, y_max)

        self.max_speed = float(max_speed)
        self.max_turn = float(max_turn)
        self.max_steps = int(max_steps)
        if self.max_speed <= 0 or self.max_turn <= 0 or self.max_steps < 1:
            raise ValueError(
                "max_speed and max_turn must be > 0 and max_steps >= 1, got "
                f"{max_speed}, {max_turn}, {max_steps}"
            )

        self.pois = pois
        self.observation = SectorDensityObservation() if observation is None else observation
        self.reward = CouplingDepletionReward() if reward is None else reward
        self.pois.validate(self)

        key = jax.random.PRNGKey(0)
        self.n_pois = int(
            jax.eval_shape(lambda k: self.pois.init(k, self), key).position.shape[0]
        )
        self.observation_dim = int(self.observation.obs_dim)
        # Abstract evaluation (no FLOPs): the declared width must be what is emitted.
        obs_shape = jax.eval_shape(lambda k: self.reset(k)[0], key).shape
        if obs_shape != (self.n_agents, self.observation_dim):
            raise ValueError(
                f"observation model declares obs_dim={self.observation_dim} but "
                f"emits {obs_shape}"
            )

    # ---------------------------------------------------------------- API

    def reset(self, key):
        k_rovers, k_pois = jax.random.split(key)
        state = RoverEnvState(
            rovers=self._spawn_rovers(k_rovers),
            pois=self.pois.init(k_pois, self),
            t=jnp.zeros((), jnp.int32),
        )
        return self.observe(state), state

    def step(self, state, actions):
        rovers, speed = self.move(state.rovers, actions)
        pois = self.pois.step(state.pois, rovers, state.t)
        out = self.reward(self, rovers, pois)
        t = state.t + 1
        terminated = jnp.asarray(out.terminated, dtype=bool)
        truncated = (t >= self.max_steps) & ~terminated
        next_state = RoverEnvState(rovers=rovers, pois=out.pois, t=t)
        reward = jnp.asarray(out.reward)
        info = {
            "task_reward": reward if reward.ndim == 0 else reward.sum(),
            "agent_positions": rovers.position,
            "agent_headings": rovers.heading,
            "agent_speeds": speed,
            "poi_remaining": out.pois.remaining,
            **out.info,
        }
        return self.observe(next_state), next_state, reward, terminated, truncated, info

    def observe(self, state):
        return self.observation(self, state)

    # ------------------------------------------------------------ dynamics

    def move(self, rovers, actions):
        """Apply ``[turn, speed]`` actions; returns ``(rovers, speed (A,))``."""
        a = jnp.clip(jnp.asarray(actions, dtype=jnp.float32), -1.0, 1.0)
        heading = wrap_angle(rovers.heading + a[:, 0] * self.max_turn)
        speed = (a[:, 1] + 1.0) * 0.5 * self.max_speed
        direction = jnp.stack([jnp.cos(heading), jnp.sin(heading)], axis=-1)
        position = jnp.clip(
            rovers.position + speed[:, None] * direction,
            0.0,
            jnp.asarray(self.world_size, dtype=jnp.float32),
        )
        return RoverState(position=position, heading=heading), speed

    def _spawn_rovers(self, key):
        k_pos, k_heading = jax.random.split(key)
        x_min, x_max, y_min, y_max = self.spawn_region
        position = jax.random.uniform(
            k_pos,
            (self.n_agents, 2),
            minval=jnp.asarray([x_min, y_min]),
            maxval=jnp.asarray([x_max, y_max]),
        )
        heading = jax.random.uniform(
            k_heading, (self.n_agents,), minval=-jnp.pi, maxval=jnp.pi
        )
        return RoverState(position=position, heading=heading)
