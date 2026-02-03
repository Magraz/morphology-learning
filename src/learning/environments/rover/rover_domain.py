#  Copyright (c) 2022-2024.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.

import typing
from typing import Dict, List

import torch
from torch import Tensor

from vmas import render_interactively
from vmas.simulator.core import Agent, Landmark, Sphere, World
from vmas.simulator.heuristic_policy import BaseHeuristicPolicy
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.utils import ScenarioUtils

if typing.TYPE_CHECKING:
    from vmas.simulator.rendering import Geom

from learning.environments.rover.sensors import SectorDensity
from learning.environments.rover.utils import COLOR_MAP


class RoverDomain(BaseScenario):
    def make_world(self, batch_dim: int, device: torch.device, **kwargs):

        self.x_semidim = kwargs.pop("x_semidim", 1)
        self.y_semidim = kwargs.pop("y_semidim", 1)

        self.viewer_zoom = kwargs.pop("viewer_zoom", 1)

        self.n_agents = kwargs.pop("n_agents", 5)
        self.agents_colors = kwargs.pop("agents_colors", [])
        self.n_targets = kwargs.pop("n_targets", 7)
        self.use_order = kwargs.pop("use_order", False)
        self.targets_positions = kwargs.pop("targets_positions", [])
        self.targets_colors = kwargs.pop("targets_colors", [])
        self.targets_orders = kwargs.pop("targets_orders", [])
        self.targets_types = kwargs.pop("targets_types", [])
        self.targets_values = torch.tensor(
            kwargs.pop("targets_values", []), device=device
        )
        self.agents_positions = kwargs.pop("agents_positions", [])

        self._min_dist_between_entities = kwargs.pop("min_dist_between_entities", 0.2)
        self._lidar_range = kwargs.pop("lidar_range", 0.35)
        self._covering_range = kwargs.pop("covering_range", 0.25)

        self._agents_per_target = kwargs.pop("agents_per_target", 2)
        self.targets_respawn = kwargs.pop("targets_respawn", False)
        self.random_spawn = kwargs.pop("random_spawn", False)

        self.covering_rew_coeff = kwargs.pop("covering_rew_coeff", 1.0)

        ScenarioUtils.check_kwargs_consumed(kwargs)

        self.agent_radius = 0.05
        self.target_radius = self.agent_radius

        self.device = device

        self.current_order_per_env = torch.zeros((batch_dim, 1), device=device)

        self.order_tensor = torch.tensor(self.targets_orders, device=device)

        self.global_rew = torch.zeros(batch_dim, device=device)
        self.covered_targets = torch.zeros((batch_dim, self.n_targets), device=device)

        # Make world
        world = World(
            batch_dim,
            device,
            x_semidim=self.x_semidim,
            y_semidim=self.y_semidim,
            substeps=2,
        )

        # Set targets
        self._targets = []
        for i in range(self.n_targets):

            target = Landmark(
                name=f"target_{i}",
                collide=False,
                shape=Sphere(radius=self.target_radius),
                color=COLOR_MAP[self.targets_colors[i]],
            )

            target.value = self.targets_values[i]
            target.type = self.targets_types[i]
            target.order = self.targets_orders[i]

            world.add_landmark(target)
            self._targets.append(target)

        # Set agents
        for i in range(self.n_agents):
            # Constraint: all agents have same action range and multiplier
            agent = Agent(
                name=f"agent_{i}",
                collide=False,
                shape=Sphere(radius=self.agent_radius),
                sensors=([SectorDensity(world, max_range=self._lidar_range)]),
                color=COLOR_MAP[self.agents_colors[i]],
            )
            agent.difference_rew = torch.zeros(batch_dim, device=device)
            world.add_agent(agent)

        return world

    def reset_world_at(self, env_index: int = None):
        placable_entities = self._targets[: self.n_targets] + self.world.agents

        if env_index is None:
            self.all_time_covered_targets = torch.full(
                (self.world.batch_dim, self.n_targets),
                False,
                device=self.world.device,
            )
        else:
            self.all_time_covered_targets[env_index] = False

        if self.random_spawn:
            ScenarioUtils.spawn_entities_randomly(
                entities=placable_entities,
                world=self.world,
                env_index=env_index,
                min_dist_between_entities=self._min_dist_between_entities,
                x_bounds=(-self.world.x_semidim, self.world.x_semidim),
                y_bounds=(-self.world.y_semidim, self.world.y_semidim),
            )

            for target in self._targets[self.n_targets :]:
                target.set_pos(self.get_outside_pos(env_index), batch_index=env_index)
        else:
            for idx, agent in enumerate(self.world.agents):
                pos = torch.ones(
                    (self.world.batch_dim, self.world.dim_p), device=self.world.device
                ) * torch.tensor(self.agents_positions[idx], device=self.world.device)
                agent.set_pos(
                    pos,
                    batch_index=env_index,
                )

            for idx, target in enumerate(self._targets):
                pos = torch.ones(
                    (self.world.batch_dim, self.world.dim_p), device=self.world.device
                ) * torch.tensor(self.targets_positions[idx], device=self.world.device)
                target.set_pos(
                    pos,
                    batch_index=env_index,
                )

    def get_order_mask(self, covered_targets: torch.Tensor) -> torch.Tensor:
        current_order_tensor = covered_targets * self.order_tensor
        return current_order_tensor == self.current_order_per_env

    def calculate_global_reward(
        self, targets_pos: torch.Tensor, agent: Agent
    ) -> torch.Tensor:

        agents_pos = torch.stack([a.state.pos for a in self.world.agents], dim=1)

        agents_targets_dists = torch.cdist(agents_pos, targets_pos)
        agents_per_target = torch.sum(
            (agents_targets_dists < self._covering_range).type(torch.int),
            dim=1,
        )

        self.covered_targets = agents_per_target >= self._agents_per_target

        # Get order mask
        if self.use_order:
            self.covered_targets &= self.get_order_mask(self.covered_targets)

        # After order has been taken into account continue
        agents_covering_targets_mask = agents_targets_dists < self._covering_range

        covered_targets_dists = agents_covering_targets_mask * agents_targets_dists

        masked_covered_targets_dists = torch.where(
            covered_targets_dists == 0, float("inf"), covered_targets_dists
        )

        min_covered_targets_dists, _ = torch.min(masked_covered_targets_dists, dim=1)

        min_covered_targets_dists = torch.clamp(min_covered_targets_dists, min=1e-2)

        min_covered_targets_dists[torch.isinf(min_covered_targets_dists)] = 0

        global_reward_spread = torch.log10(
            self.covered_targets / min_covered_targets_dists
        )

        global_reward_spread *= self.targets_values

        global_reward_spread[torch.isnan(global_reward_spread)] = 0
        global_reward_spread[torch.isinf(global_reward_spread)] = 0

        return torch.sum(
            global_reward_spread,
            dim=1,
        )

    def calculate_difference_reward(
        self, targets_pos: torch.Tensor, me: Agent
    ) -> torch.Tensor:

        global_rew_without_me = torch.zeros(
            self.world.batch_dim, device=self.world.device
        )

        me.difference_rew[:] = 0

        agents_without_me = [agent for agent in self.world.agents if agent != me]

        agents_pos_without_me = torch.stack(
            [a.state.pos for a in agents_without_me], dim=1
        )

        agents_targets_dists_without_me = torch.cdist(
            agents_pos_without_me, targets_pos
        )

        agents_per_target_without_me = torch.sum(
            (agents_targets_dists_without_me < self._covering_range).type(torch.int),
            dim=1,
        )

        covered_targets_without_me = (
            agents_per_target_without_me >= self._agents_per_target
        )

        # Get order mask
        if self.use_order:
            covered_targets_without_me &= self.get_order_mask(
                covered_targets_without_me
            )

        covered_targets_mask = agents_targets_dists_without_me < self._covering_range

        covered_targets_dists_without_me = (
            covered_targets_mask * agents_targets_dists_without_me
        )

        masked_covered_targets_dists_without_me = torch.where(
            covered_targets_dists_without_me == 0,
            float("inf"),
            covered_targets_dists_without_me,
        )

        min_covered_targets_dists_without_me, _ = torch.min(
            masked_covered_targets_dists_without_me, dim=1
        )

        min_covered_targets_dists_without_me = torch.clamp(
            min_covered_targets_dists_without_me, min=1e-2
        )

        min_covered_targets_dists_without_me[
            torch.isinf(min_covered_targets_dists_without_me)
        ] = 0

        global_reward_spread = torch.log10(
            covered_targets_without_me / min_covered_targets_dists_without_me
        )

        global_reward_spread *= self.targets_values

        global_reward_spread[torch.isnan(global_reward_spread)] = 0
        global_reward_spread[torch.isinf(global_reward_spread)] = 0

        global_rew_without_me = torch.sum(global_reward_spread, dim=1)

        return self.global_rew - global_rew_without_me

    def reward(self, agent: Agent) -> torch.Tensor:
        is_first = agent == self.world.agents[0]
        is_last = agent == self.world.agents[-1]

        if is_first:

            targets_pos = torch.stack([t.state.pos for t in self._targets], dim=1)

            # Calculate G
            self.global_rew = self.calculate_global_reward(targets_pos, agent)

            # Calculate D
            if len(self.world.agents) > 1:  # Can't calculate D with a team of 1 agent
                for me in self.world.agents:
                    me.difference_rew = self.calculate_difference_reward(
                        targets_pos, me
                    )

        if is_last:
            self.all_time_covered_targets += self.covered_targets
            self.current_order_per_env = torch.sum(
                self.all_time_covered_targets, dim=1
            ).unsqueeze(-1)

        covering_rew = torch.cat([self.global_rew, agent.difference_rew])

        return covering_rew

    def get_outside_pos(self, env_index) -> torch.Tensor:
        return torch.empty(
            (
                (1, self.world.dim_p)
                if env_index is not None
                else (self.world.batch_dim, self.world.dim_p)
            ),
            device=self.world.device,
        ).uniform_(-1000 * self.world.x_semidim, -10 * self.world.x_semidim)

    def observation(self, agent: Agent) -> torch.Tensor:

        obs = agent.sensors[0].measure()

        return obs

    def info(self, agent: Agent) -> Dict[str, Tensor]:
        info = {
            "global_reward": (self.global_rew),
            "difference_reward": (agent.difference_rew),
            "targets_covered": self.covered_targets.sum(-1),
        }
        return info

    def done(self):
        return self.all_time_covered_targets.all(dim=-1)

    def extra_render(self, env_index: int = 0) -> "List[Geom]":
        from vmas.simulator import rendering

        geoms: List[Geom] = []
        # Target ranges
        for i, target in enumerate(self._targets):
            range_circle = rendering.make_circle(self._covering_range, filled=False)
            xform = rendering.Transform()
            xform.set_translation(*target.state.pos[env_index])
            range_circle.add_attr(xform)
            range_circle.set_color(*COLOR_MAP[self.targets_colors[i]].value)
            geoms.append(range_circle)

        return geoms


class HeuristicPolicy(BaseHeuristicPolicy):
    def __init__(self, continuous_action, device, targets_positions):

        super().__init__(continuous_action)

        self.device = device
        self.theta_max = 6 * torch.pi
        self.theta_range = torch.arange(
            0, self.theta_max, step=self.theta_max / 600
        ).to(device)

        self.targets = torch.tensor(targets_positions, device=device)
        self.current_target = 0

    def compute_action(
        self,
        agent_position,
        observation: torch.Tensor,
        u_range: float,
    ) -> torch.Tensor:

        des_pos = torch.zeros((1, 2), device=observation.device).to(observation.device)

        if self.current_target >= len(self.targets):
            self.current_target = len(self.targets) - 1

        des_pos[:, 0] = self.targets[self.current_target, 0]
        des_pos[:, 1] = self.targets[self.current_target, 1]

        action = torch.clamp(
            (des_pos - agent_position),
            min=-u_range,
            max=u_range,
        )

        dist_to_target = torch.cdist(des_pos, agent_position) ** 2

        if dist_to_target[0, 0] < 0.01:
            self.current_target += 1

        return action


if __name__ == "__main__":
    render_interactively(__file__, control_two_agents=True)
