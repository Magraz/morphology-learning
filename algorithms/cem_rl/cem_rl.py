import numpy as np
import random
import copy
import torch
import torch.nn as nn

import sys

np.random.seed(int(sys.argv[1]))
torch.manual_seed(int(sys.argv[1]))
random.seed(int(sys.argv[1]))

def get_flat_params(model: nn.Module) -> torch.Tensor:
    params = []
    for p in model.parameters():
        params.append(p.data.view(-1).cpu())
    return torch.cat(params, dim=0)

def set_flat_params(model: nn.Module, flat_params: torch.Tensor):
    idx = 0
    for p in model.parameters():
        numel = p.numel()
        new_vals = flat_params[idx:idx + numel]
        idx += numel
        new_vals = new_vals.view(p.shape)
        p.data.copy_(new_vals)

def rollout_actor(env, actor, replay_buffer, archive, seed, descriptor=None):
    actor = actor.to("cpu")
    replay_buffer = [replay_buffer]
    observations, info = env.reset(seed=seed)
    num_agents = env.num_agents # number of agents
    all_agents = env.possible_agents # agent string (ie. "walker_0")

    agent_idx_dict = {}

    unwrapped_env = env.single_agent_env.unwrapped

    # Extract and decode all geom names
    geom_names_bytes = unwrapped_env.model.names
    geom_names = geom_names_bytes.split(b'\x00')
    geom_names = [name.decode('utf-8') for name in geom_names]

    # Map geom IDs to names
    geom_id_to_name = {i: geom_names[i] for i in range(unwrapped_env.model.ngeom)}

    ground_geom_id = 0 #get_geom_id(ground_geom_name)
    left_foot_geom_id = 5
    right_foot_geom_id = 8

    left_contact_count = 0
    right_contact_count = 0
    

    for index, name in enumerate(all_agents):
        agent_idx_dict[name] = index

    num_steps = 0

    env_global_reward = 0.0

    experiences = [[] for _ in range(num_agents)]

    
    while env.agents:
        
        actions = {agent: (actor.forward(torch.FloatTensor(observations[agent].reshape(1, -1))).cpu().data.numpy().flatten()
                            ).clip(-1.0, 1.0).astype(np.float32) for agent in env.agents}

        old_observations = observations.copy()

        observations, rewards, terminations, truncations, infos = env.step(actions)

        env_global_reward += sum(rewards.values()) / num_agents

        contacts = unwrapped_env.data.contact

        left_in_contact = False
        right_in_contact = False

        for contact in contacts:
            geom1 = contact.geom1
            geom2 = contact.geom2
            
            # Check contact with left foot
            if (geom1 == left_foot_geom_id and geom2 == ground_geom_id) or (geom2 == left_foot_geom_id and geom1 == ground_geom_id):
                left_in_contact = True
            
            # Check contact with right foot
            if (geom1 == right_foot_geom_id and geom2 == ground_geom_id) or (geom2 == right_foot_geom_id and geom1 == ground_geom_id):
                right_in_contact = True
            
            # Early exit if both contacts are found
            if left_in_contact and right_in_contact:
                break
        
        if left_in_contact:
            left_contact_count += 1
        if right_in_contact:
            right_contact_count += 1
        
        for agent in old_observations.keys():
            done = terminations[agent] or truncations[agent]

            if(agent not in observations):
                print("Agent isn't in observation!!!") # NOTE: this should never be printed out
                print(observations)
            
            next_obs = observations[agent]
            
            experiences[agent_idx_dict[agent]].append([
                old_observations[agent],
                actions[agent],
                next_obs,
                rewards[agent],
                done
            ])
                
        #print("Timestep reward", sum(dist_rewards), dist_rewards)
        
        num_steps += 1

    bd = (round(left_contact_count / num_steps, 2), round(right_contact_count / num_steps, 2))
    
    if replay_buffer is not None:
        for agent_ind, agent_exp in enumerate(experiences):
            for i, exp in enumerate(agent_exp):
                agent_exp[i].append(bd)

                assert len(agent_exp[i]) == 6, "Each experience should have 6 elements only (rollout actor)"

                replay_buffer[agent_ind].add(
                    state=agent_exp[i][0],
                    action=agent_exp[i][1],
                    next_state=agent_exp[i][2],
                    reward=agent_exp[i][3],
                    done=agent_exp[i][4],
                    d_prime=agent_exp[i][5] if descriptor is None else descriptor,
                    d=agent_exp[i][5]
                    
                )

    if archive is not None:
        #print("BD being added is", bd, "right count", right_contact_count)
        archive.add(policy=copy.deepcopy(actor), bd=bd, fitness=env_global_reward)
        #print("Adding bd:", bd, "fitness:", env_global_reward)

    return env_global_reward, num_steps, bd

class CEM_RL():
    def __init__(self, rl, replay_buffer, env, seed):
        self.tau = 0.95
        self.damp_limit = 1e-5
        self.damp = 1e-3
        self.rl = rl
        self.replay_buffer = replay_buffer

        self.population_size = 10
        self.elite_frac = 0.5

        #  parents are the elites that are left, in this case the top half are used in the update rule
        self.parents = int(self.population_size * self.elite_frac)

        # weights are weightages to each elite's params based on how good they were (for "gradient" update)
        self.weights = np.array([np.log((self.parents + 1) / ind) for ind in range(1, self.parents + 1)])
        self.weights /= self.weights.sum()

        # Initial actor weights (mu) and diagonal covariance matrix
        self.mu = get_flat_params(self.rl.actor).numpy().flatten()
        self.num_params = self.mu.shape[0]
        self.cov = np.ones(self.num_params) * 1e-3

        # NOTE: first half of solutions will be flat np arrays (from cem), second half will be torch nn objects (from rl)
        self.solutions, self.rewards = [], [] # stores the current population (cleared everytime a new generation starts)

        self.stateless_actor = copy.deepcopy(self.rl.actor) # used only for policy evaluation (weights mutated constantly)

        self.env = env
        self.seed = seed

    def create_population(self):
        # clearing the prev gen's solutions
        self.solutions = []
        self.rewards = [None] * self.population_size

        for _ in range(self.population_size):
            sample = self.mu + np.sqrt(self.cov) * np.random.randn(self.num_params)
            self.solutions.append(sample)
        
        # just in case something goes wrong over multiple generations:
        assert len(self.solutions) == self.population_size, str("[create pop] Mismatch between true pop size=" + str(len(self.solutions)) + " and desired size=" + str(self.population_size))

    def train_rl_pols(self, num_steps):
        # Grab half of the population, train critic, and ascend the policies from the pop
        for i in range(self.population_size//2, self.population_size):
            # set the rl actor and target to the individual
            set_flat_params(self.rl.actor, torch.from_numpy(self.solutions[i]).float())
            self.rl.set_new_actor(self.rl.actor)

            curr_pol = copy.deepcopy(self.rl.actor) # the actual policy that will be updated

            for _ in range(num_steps):
                self.rl.train(replay_buffer=self.replay_buffer)

            # ascend policy
            self.solutions[i] = self.rl.apply_pg_variation(self.replay_buffer, curr_pol)
        
        assert len(self.solutions) == self.population_size, str("[train rl] Mismatch between true pop size=" + str(len(self.solutions)) + " and desired size=" + str(self.population_size))
    
    def evaluate_solutions(self, first_half):
        '''Uses RL for evaluation'''
        total_steps = 0

        if first_half:
            for i in range(self.population_size//2):

                set_flat_params(self.stateless_actor, torch.from_numpy(self.solutions[i]).float())
                curr_actor = copy.deepcopy(self.stateless_actor).to("cpu") # using this just to ensure not evaluating a prev solution

                reward, steps, _ = rollout_actor(self.env, curr_actor, self.replay_buffer, self.archive, self.seed)

                self.rewards[i] = reward
                total_steps += steps
        else:
            for index in range(self.population_size//2, self.population_size):
                if index < self.population_size - 2:
                    # evaluate rl solution with positive sample
                    reward, steps, _ = rollout_actor(self.env, self.solutions[index], self.replay_buffer, self.archive, self.seed)

                    #self.solutions[index] = get_flat_params(self.solutions[index]).numpy().flatten() # prepping for update (requires all to be flat)
                    self.rewards[index] = reward
                    total_steps += steps
                else:
                    # use a negative sample, RL evaluation
                    reward, steps, _ = rollout_actor(self.env, self.solutions[index], self.replay_buffer, self.archive, self.seed, descriptor=self.cur_bd)
                    self.rewards[index] = reward

                self.solutions[index] = get_flat_params(self.solutions[index]).numpy().flatten() # prepping for update (requires all to be flat)

        return total_steps

    def evaluate_all_solutions(self):
        '''Uses only CEM'''
        total_steps = 0
        for i in range(self.population_size):

            set_flat_params(self.stateless_actor, torch.from_numpy(self.solutions[i]).float())
            curr_actor = copy.deepcopy(self.stateless_actor) # using this just to ensure not evaluating a prev solution

            reward, steps, bd = rollout_actor(self.env, curr_actor, self.replay_buffer, self.archive, self.seed)
            self.archive.add(curr_actor, bd, reward)

            self.rewards[i] = reward
            total_steps += steps
        
        return total_steps
    
    def update_solution(self):
        scores = np.array(self.rewards)
        scores *= -1  # -1 so argsort sorts in descending
        idx_sorted = np.argsort(scores)  # sorted indices (ascending of negated -> descending original)

        old_mu = self.mu.copy()
        self.damp = self.damp * self.tau + (1 - self.tau) * self.damp_limit

        # Select top parents solutions to use in gradient calculation
        top_indices = idx_sorted[:self.parents]
        top_solutions = np.array(self.solutions)[top_indices]  # shape: (parents, num_params)

        # Update mean using weighted sum of top solutions
        self.mu = self.weights @ top_solutions #np.sum(top_solutions * weights[:, None], axis=0)

        # Calculate deviation of top solutions from old mean
        z = top_solutions - old_mu

        # Update diagonal covariance using weighted average of squared deviations
        self.cov = (1 / self.parents) * self.weights @ (z * z) + self.damp * np.ones(self.num_params)