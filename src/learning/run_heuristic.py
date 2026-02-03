#  Copyright (c) 2022-2024.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.
import os
import time
import argparse
import yaml
import torch

from vmas import make_env
from vmas.simulator.utils import save_video
from vmas.simulator.environment import Environment

from learning.environments.salp.create_env import create_env
from pynput.keyboard import Listener
from learning.testing.manual_control import manual_control
from pathlib import Path


def use_vmas_env(
    name: str,
    env: Environment = None,
    render: bool = False,
    save_render: bool = False,
    n_envs: int = 1,
    n_steps: int = 100,
    device: str = "cpu",
    visualize_render: bool = True,
    use_heuristic: bool = True,
    **kwargs,
):
    """Example function to use a vmas environment

    Args:
        device (str): Torch device to use
        render (bool): Whether to render the scenario
        save_render (bool):  Whether to save render of the scenario
        n_envs (int): Number of vectorized environments
        n_steps (int): Number of steps before returning done
        random_action (bool): Use random actions or have all agents perform the down action
        visualize_render (bool, optional): Whether to visualize the render. Defaults to ``True``.
        kwargs (dict, optional): Keyword arguments to pass to the scenario

    Returns:

    """
    assert not (save_render and not render), "To save the video you have to render it"

    frame_list = []  # For creating a gif
    init_time = time.time()
    step = 0
    n_agents = kwargs.pop("n_agents", 0)

    mc = manual_control(n_agents)

    G_total = torch.zeros((n_agents, n_envs)).to(device)
    G_list = []
    obs_list = []

    _ = env.reset()

    with Listener(on_press=mc.on_press, on_release=mc.on_release) as listener:

        listener.join(timeout=1)

        for step in range(n_steps):
            step += 1

            actions = []

            for i, agent in enumerate(env.agents):

                # Move one agent at a time
                # if i == mc.controlled_agent:
                #     cmd_action = mc.cmd_vel[:]  # + mc.join[:]
                #     action = torch.tensor(cmd_action).repeat(n_envs, 1)
                # else:
                #     action = torch.tensor([0.0, 0.0]).repeat(n_envs, 1)

                # Move all agents at the same time
                cmd_action = mc.cmd_vel[-1]  # + mc.join[:]
                action = torch.tensor(cmd_action).repeat(n_envs, 1)

                actions.append(action)

            obs, rews, dones, info = env.step(actions)

            obs_list.append(obs[0][0])

            G_list.append(torch.stack([g[:n_envs] for g in rews], dim=0)[0])

            G_total += torch.stack([g[:n_envs] for g in rews], dim=0)

            G = torch.stack([g[:n_envs] for g in rews], dim=0)

            print(obs)

            # if any(tensor.any() for tensor in rews):
            #     print("G")
            #     print(G)

            #     # print("Total G")
            #     # print(G_total)

            #     pass

            if render:
                frame = env.render(
                    mode="rgb_array",
                    agent_index_focus=None,  # Can give the camera an agent index to focus on
                    visualize_when_rgb=visualize_render,
                )
                if save_render:
                    frame_list.append(frame)

    # print("G List Agg")agents
    # G_agg = torch.sum(torch.stack(G_list), dim=0)
    # print(G_agg)
    # print("D List Agg")
    # D_agg = torch.sum(torch.stack(D_list), dim=0)
    # print(torch.transpose(D_agg, dim0=0, dim1=1))
    # print("Obs List")
    # print(obs_list)

    # print("G List")
    # print(G_list)
    # print("D List")
    # print(D_list)

    total_time = time.time() - init_time

    print(
        f"It took: {total_time}s for {n_steps} steps of {n_envs} parallel environments on device {device} "
        f"for {name} scenario."
    )

    if render and save_render:
        save_video(name, frame_list, fps=10 / env.scenario.world.dt)


if __name__ == "__main__":
    # Arg parser variables
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--batch",
        default="",
        help="Experiment batch",
        type=str,
    )
    parser.add_argument(
        "--name",
        default="",
        help="Experiment name",
        type=str,
    )
    parser.add_argument(
        "--environment",
        default="",
        help="Learning environment name",
        type=str,
    )

    parser.add_argument("--trial_id", default=0, help="Sets trial ID", type=int)

    args = vars(parser.parse_args())

    # Set base_config path
    dir_path = Path(__file__).parent

    # Set configuration folder
    batch_dir = dir_path / "experiments" / "yamls" / args["batch"]

    env_file = batch_dir / "_env.yaml"

    with open(str(env_file), "r") as file:
        env_config = yaml.safe_load(file)

    # Environment dataw
    map_size = env_config["map_size"]

    # Agent data
    n_agents = 6
    agents_positions = [poi["position"]["coordinates"] for poi in env_config["agents"]]
    lidar_range = [rover["observation_radius"] for rover in env_config["agents"]]

    # POIs data
    poi_positions = [poi["position"]["coordinates"] for poi in env_config["targets"]]
    n_envs = 1

    use_vmas_env(
        name=f"{args["batch"]}_{n_agents}a",
        env=create_env(
            batch_dir=batch_dir,
            n_envs=n_envs,
            device="cpu",
            benchmark=False,
            n_agents=n_agents,
            viewer_zoom=1.7,
        ),
        render=True,
        save_render=False,
        device="cpu",
        n_envs=n_envs,
        n_steps=100,
        # kwargs
        n_agents=n_agents,
        targets_positions=poi_positions,
        use_heuristic=False,
    )
