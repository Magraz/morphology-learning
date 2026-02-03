import td3
import cem_rl

import sys
import torch


class CEMRL_Trainer:
    def __init__(
        self,
        env,
        n_agents,
        state_dim,
        action_dim=None,
        dirs=None,
        use_dict_actions=False,
        device="cpu",
        seed=None,
    ):
        self.env = env
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.seed = seed

    def train(self):
        episode_length = 500
        curr_steps, total_steps = 0, 1e6
        resample = 5
        warmup_iters = 5

        num_episodes = int(total_steps // episode_length)

        td3_alg = td3.TD3(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            max_action=1.0,
        )
        replay_buffer = td3.ReplayBuffer(
            state_dim=self.state_size, action_dim=self.action_dim
        )

        cemrl = cem_rl.CEM_RL(
            rl=td3_alg, replay_buffer=replay_buffer, env=self.env, seed=self.seed
        )

        iters = 0

        while curr_steps < total_steps:
            # if iters >= warmup_iters and iters % resample == 0:
            #     print("resampling")
            #     bd = cemrl.resample_mu()

            cemrl.create_population()

            if iters >= warmup_iters:
                steps = cemrl.evaluate_solutions(first_half=True)
                cemrl.train_rl_pols(steps)
                print("finished training rl")
                steps += cemrl.evaluate_solutions(first_half=False)

            else:
                steps = cemrl.evaluate_all_solutions()

            cemrl.update_solution()

            if iters % 10 == 0:  # every 10 iterations
                message = (
                    "i=" + str(iters) + ": " + str([float(i) for i in cemrl.rewards])
                )

            # if iters % 25 == 0:
            #     cem_rl.set_flat_params(cemrl.stateless_actor, torch.from_numpy(cemrl.mu).float())
            #     pypickle.save("pkl/cem_1M_" + sys.argv[2] + "_" + str(iters) + ".pkl", cemrl.stateless_actor)

            curr_steps += steps
            iters += 1

        # cem_rl.set_flat_params(cemrl.stateless_actor, torch.from_numpy(cemrl.mu).float())
        # pypickle.save("pkl/cem_1M_" + sys.argv[2] + ".pkl", cemrl.stateless_actor)
