#  Copyright (c) 2022-2024.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.

from __future__ import annotations

import typing
from abc import ABC, abstractmethod
from typing import Callable, List, Tuple, Union

import torch

import vmas.simulator.core
from vmas.simulator.sensors import Sensor

if typing.TYPE_CHECKING:
    from vmas.simulator.rendering import Geom


class SectorDensity(Sensor):
    def __init__(
        self,
        world: vmas.simulator.core.World,
        angle_start: float = 0.0,
        angle_end: float = 2 * torch.pi,
        max_range: float = 1.0,
        sectors: int = 4,
    ):
        super().__init__(world)
        if (angle_start - angle_end) % (torch.pi * 2) < 1e-5:
            angles = torch.linspace(
                angle_start, angle_end, sectors + 1, device=self._world.device
            )[:sectors]
        else:
            angles = torch.linspace(
                angle_start, angle_end, sectors, device=self._world.device
            )

        self._angles = angles
        self._max_range = max_range
        self._last_measurement = None
        self._sectors = sectors

        self.target_values = torch.empty((0)).to(self._world.device)
        for l in self._world._landmarks:
            self.target_values = torch.cat(
                (self.target_values, l.value.unsqueeze(0)), dim=-1
            )

    def to(self, device: torch.device):
        self._angles = self._angles.to(device)

    def measure(self):

        measurement = torch.zeros((8)).to(self._world.device)

        all_other_agents = [
            agent for agent in self._world._agents if agent != self.agent
        ]

        dists_2_targets = torch.empty((0, self._world.batch_dim)).to(self._world.device)
        dists_2_t_angles = torch.empty((0, self._world.batch_dim)).to(
            self._world.device
        )
        dists_2_agents = torch.empty((0, self._world.batch_dim)).to(self._world.device)
        dists_2_a_angles = torch.empty((0, self._world.batch_dim)).to(
            self._world.device
        )

        for a in all_other_agents:
            dists_2_agents = torch.cat(
                (
                    dists_2_agents,
                    torch.pairwise_distance(
                        self.agent.state.pos, a.state.pos
                    ).unsqueeze(0),
                ),
                dim=0,
            )
            dif = self.agent.state.pos - a.state.pos
            dif_complex = torch.complex(dif[:, 0], dif[:, 1])
            dists_2_a_angles = torch.cat(
                (
                    dists_2_a_angles,
                    torch.remainder(
                        torch.angle(dif_complex).unsqueeze(0), 2 * torch.pi
                    ),
                ),
                dim=0,
            )

        for l in self._world._landmarks:
            dists_2_targets = torch.cat(
                (
                    dists_2_targets,
                    torch.pairwise_distance(
                        self.agent.state.pos, l.state.pos
                    ).unsqueeze(0),
                ),
                dim=0,
            )

            dif = self.agent.state.pos - l.state.pos
            dif_complex = torch.complex(dif[:, 0], dif[:, 1])

            dists_2_t_angles = torch.cat(
                (
                    dists_2_t_angles,
                    torch.remainder(
                        torch.angle(dif_complex).unsqueeze(0), 2 * torch.pi
                    ),
                ),
                dim=0,
            )

        dists_2_agents = torch.clamp(dists_2_agents, min=5e-2)
        dists_2_targets = torch.clamp(dists_2_targets, min=5e-2)

        agent_density_per_sector = torch.zeros(
            (self._sectors, self._world.batch_dim)
        ).to(self._world.device)
        target_density_per_sector = torch.zeros(
            (self._sectors, self._world.batch_dim)
        ).to(self._world.device)

        for sector in range(self._sectors):
            # If we are in last sector
            if sector + 1 == self._sectors:
                targets_in_sector = (self._angles[sector] < dists_2_t_angles) & (
                    dists_2_t_angles < (2 * torch.pi)
                )
                agents_in_sector = (self._angles[sector] < dists_2_a_angles) & (
                    dists_2_a_angles < (2 * torch.pi)
                )
            else:
                targets_in_sector = (self._angles[sector] < dists_2_t_angles) & (
                    dists_2_t_angles < self._angles[sector + 1]
                )
                agents_in_sector = (self._angles[sector] < dists_2_a_angles) & (
                    dists_2_a_angles < self._angles[sector + 1]
                )

            # Get distances within sector

            dist_2_agents_in_sector = dists_2_agents * agents_in_sector
            dist_2_targets_in_sector = dists_2_targets * targets_in_sector

            dist_2_agents_in_sector[dist_2_agents_in_sector > self._max_range] = 0
            dist_2_targets_in_sector[dist_2_targets_in_sector > self._max_range] = 0

            # Get distances greater than zero within sector
            filtered_agent_dists = 1 / dist_2_agents_in_sector
            filtered_target_dists = (
                self.target_values.unsqueeze(-1) / dist_2_targets_in_sector
            )

            filtered_agent_dists[torch.isinf(filtered_agent_dists)] = 0
            filtered_target_dists[torch.isinf(filtered_target_dists)] = 0

            # Sum distances in sector
            filtered_agent_dists_sum = torch.sum(
                filtered_agent_dists,
                dim=0,
            )
            filtered_target_dists_sum = torch.sum(
                filtered_target_dists,
                dim=0,
            )

            # Concatenate to measurements
            agent_density_per_sector[sector] += filtered_agent_dists_sum
            target_density_per_sector[sector] += filtered_target_dists_sum

        measurement = torch.transpose(
            torch.cat((agent_density_per_sector, target_density_per_sector), dim=0),
            dim0=0,
            dim1=-1,
        )

        self._last_measurement = measurement
        return measurement

    def render(self, env_index: int = 0) -> "List[Geom]":
        return []
