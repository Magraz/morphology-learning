# run_5m_vs_6m.py
import numpy as np
from smacv2.env import StarCraft2Env  # SMACv2 main env


def run(map_name="5m_vs_6m", n_episodes=5, seed=0):
    env = StarCraft2Env(
        map_name=map_name,
        seed=seed,
        debug=False,
        # You can tweak these if you want:
        # step_mul=8,
        # reward_sparse=False,
        # obs_own_pos=True,
    )

    env_info = env.get_env_info()
    n_agents = env_info["n_agents"]

    print("Loaded:", map_name)
    print("env_info:", env_info)

    try:
        for ep in range(n_episodes):
            env.reset()
            terminated = False
            ep_ret = 0.0

            while not terminated:
                actions = []
                for i in range(n_agents):
                    avail = env.get_avail_agent_actions(i)  # binary mask
                    avail_ids = np.nonzero(avail)[0]  # valid action indices
                    actions.append(int(np.random.choice(avail_ids)))

                reward, terminated, info = env.step(actions)
                ep_ret += reward

            print(f"Episode {ep:03d} | return={ep_ret:.2f} | info={info}")

    finally:
        # IMPORTANT: closes the SC2 process cleanly
        env.close()


if __name__ == "__main__":
    run()
