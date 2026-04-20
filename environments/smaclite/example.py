import time

import numpy as np

from environments.smaclite.wrapper import SmacliteToGymWrapper

RENDER = True
USE_CPP_RVO2 = False


def main():
    env = "MMM2"
    env = SmacliteToGymWrapper(env, use_cpp_rvo2=USE_CPP_RVO2)
    episode_num = 20
    total_time = 0
    total_timesteps = 0
    for i in range(episode_num):
        obs, _ = env.reset()
        if RENDER:
            env.render()
        done = False
        episode_reward = 0
        timer = time.time()
        episode_time = 0
        timestep_no = 0
        while not done and timestep_no < 200:
            avail_actions = env._get_avail_actions()
            actions = []
            for agent_idx in range(env.n_agents):
                avail_indices = np.flatnonzero(avail_actions[agent_idx]).tolist()
                actions.append(int(np.random.choice(avail_indices)))
            timer = time.time()
            obs, reward, done, truncated, info = env.step(np.asarray(actions))
            episode_time += time.time() - timer
            episode_reward += reward
            timestep_no += 1
        print(f"Total reward in episode {episode_reward}")
        print(
            f"Episode {i} took {episode_time} seconds " f"and {timestep_no} timesteps."
        )
        print(f"Average per timestep: {episode_time/timestep_no}")
        total_time += episode_time
        total_timesteps += timestep_no
    print(f"Average time per timestep: {total_time/total_timesteps}")
    env.close()


if __name__ == "__main__":
    main()
