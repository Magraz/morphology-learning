import os
import torch


from learning.environments.types import EnvironmentParams
from learning.environments.create_env import create_env

from pathlib import Path

from pynput.keyboard import Listener
from learning.testing.manual_control import manual_control


class ManualControl:
    def __init__(
        self,
        device: str,
        batch_dir: Path,
        trials_dir: Path,
        trial_id: int,
        trial_name: str,
        video_name: str,
    ):
        # Directories
        self.device = device
        self.batch_dir = batch_dir
        self.trials_dir = trials_dir
        self.trial_name = trial_name
        self.trial_id = trial_id
        self.video_name = video_name

    def view(self, exp_config, env_config: EnvironmentParams):

        max_steps = 1000000
        n_agents = env_config.n_agents
        n_envs = 1

        env = create_env(
            self.batch_dir,
            n_envs,
            device=self.device,
            env_name=env_config.environment,
            seed=118,
            n_agents=n_agents,
        )

        mc = manual_control(n_agents)

        G_total = torch.zeros((n_envs)).to(self.device)
        G_list = []
        obs_list = []

        with Listener(on_press=mc.on_press, on_release=mc.on_release) as listener:

            listener.join(timeout=1)

            for step in range(max_steps):

                step += 1

                actions = []

                for i, agent in enumerate(env.agents):

                    # Move one agent at a time
                    if i == mc.controlled_agent:
                        cmd_action = mc.cmd_vel  # + mc.join[:]
                        action = torch.tensor(cmd_action).repeat(n_envs, 1)
                    else:
                        action = torch.tensor([0.0, 0.0]).repeat(n_envs, 1)

                    # Move all agents at the same time
                    # action = torch.tensor(mc.cmd_vel).repeat(self.n_envs, 1)

                    actions.append(action)

                obs, rews, dones, info = env.step(actions)

                obs_list.append(obs[0][0])

                G_list.append(rews[0][0])

                G_total += rews[0]

                # print("\n")
                # print(f"sin_dtheta {obs[0][:,:6]}")
                # print(f"cos_dtheta {obs[0][:,6:12]}")
                # print(f"bend_speed {obs[0][:,12:18]}")

                print(G_total)

                G = rews[0][0]

                if dones.any():
                    return

                # if any(tensor.any() for tensor in rews):
                #     print("G")
                #     print(G)

                #     # print("Total G")
                #     # print(G_total)

                #     pass

                _ = env.render(
                    mode="rgb_array",
                    agent_index_focus=None,  # Can give the camera an agent index to focus on
                    visualize_when_rgb=True,
                )
