import gymnasium as gym
import numpy as np

from Box2D import (
    b2World,
)

from environments.box2d_suite.utils import (
    BoundaryContactListener,
)


class BaseEnv(gym.Env):
    metadata = {"render_fps": 30}

    def __init__(
        self,
        render_mode=None,
        n_agents=2,
        max_steps=512,
    ):
        super().__init__()

        self.n_agents = n_agents
        self.render_mode = render_mode

        self.world = b2World(gravity=(0, 0))
        self.time_step = 1.0 / 60.0

        # Add contact listener
        self.contact_listener = BoundaryContactListener()
        self.world.contactListener = self.contact_listener

        # Auto-scale world size based on entity count
        # Reference: 8 entities (5 agents + 3 objects) -> 30x30
        _total_entities = self.n_agents
        self.world_width = int(30 * max(1.0, _total_entities / 8) ** 0.5)
        self.world_height = self.world_width
        self.world_center_x = self.world_width // 2
        self.world_center_y = self.world_height // 2
        self.world_diagonal = np.sqrt(self.world_height**2 + self.world_width**2)
        self.boundary_thickness = 0.5

        # Add force tracking
        self.applied_forces = np.zeros((self.n_agents, 2), dtype=np.float32)
        self.force_scale = 2.0  # Scale factor for visualizing forces
        self.force_multiplier = 100.0  # Max force an agent can apply per axis
        # Per-agent normal contact force against any object, averaged over the
        # last physics step. Scatter has no objects/force listener, so this
        # stays zero, but the shared ObservationManager reads it.
        self.agent_contact_forces = np.zeros(self.n_agents, dtype=np.float32)

        # Velocity normalization constant (agents have linear damping=10.0,
        # so terminal velocity is bounded; world_width/10 keeps values ~[-1,1])
        self.velocity_norm = self.world_width / 10.0

        # Step tracking for truncation
        self.max_steps = max_steps
        self.current_step = 0

    def _init_agents(self):
        pass

    def _create_agents(self):
        pass

    def _create_boundary(self):
        pass

    def _get_observation(self):
        pass

    def _get_rewards(self):
        pass

    def _calculate_reward(self):
        pass

    def _get_info(self):
        pass

    def reset(self):
        pass

    def step(self):
        pass

    def render(self):
        pass

    def close(self):
        pass
