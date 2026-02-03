import torch
from pathlib import Path

from learning.environments.types import EnvironmentParams
from learning.environments.create_env import create_env
from learning.algorithms.ppo.types import Experiment, Params
from learning.algorithms.ppo.ppo import PPO
from learning.algorithms.ppo.utils import get_state_dim, process_state

import dill

from vmas.simulator.utils import save_video


def evaluate(
    exp_config: Experiment,
    env_config: EnvironmentParams,
    device: str,
    trial_id: str,
    dirs: dict,
):

    params = Params(**exp_config.params)

    random_seeds = [56, 948, 8137, 6347, 1998]

    # Create environment to get dimension data
    dummy_env = create_env(
        dirs["batch"],
        1,
        device,
        env_config.environment,
        0,
        n_agents=env_config.n_agents,
    )

    d_action = dummy_env.action_space.spaces[0].shape[0]
    d_state = get_state_dim(
        dummy_env.observation_space.spaces[0].shape[0],
        env_config.state_representation,
        exp_config.model,
        env_config.n_agents,
    )

    # get_attention_data(
    #     exp_config,
    #     env_config,
    #     params,
    #     device,
    #     dirs,
    #     env_config.n_agents,
    #     d_state,
    #     d_action,
    # )

    get_scalability_data(
        exp_config,
        env_config,
        params,
        device,
        dirs,
        random_seeds[int(trial_id)],
        env_config.n_agents,
        d_state,
        d_action,
    )


def get_scalability_data(
    exp_config: Experiment,
    env_config: EnvironmentParams,
    params: Params,
    device: Path,
    dirs: dict,
    # Parameters for scalability experiment
    seed: int,
    n_agents: int,
    d_state: int,
    d_action: int,
    n_rollouts: int = 50,
    extra_agents: int = 64,
):
    n_agents_list = list(range(4, extra_agents + 1, 4))
    data = {n_agents: {} for n_agents in n_agents_list}

    for i, n_agents in enumerate(n_agents_list):

        # Load environment and policy
        env = create_env(
            dirs["batch"],
            n_rollouts,
            device,
            env_config.environment,
            seed,
            training=True,
            n_agents=n_agents,
        )

        learner = PPO(
            device,
            exp_config.model,
            params,
            env_config.n_agents,
            n_agents,
            n_rollouts,
            d_state,
            d_action,
        )
        learner.load(dirs["models"] / "best_model")

        # Set policy to evaluation mode
        learner.policy.eval()

        rewards = []
        distance_rewards = []
        frechet_rewards = []
        episode_count = 0
        state = env.reset()
        cumulative_rewards = torch.zeros(n_rollouts, dtype=torch.float32, device=device)
        cum_dist_rewards = torch.zeros(n_rollouts, dtype=torch.float32, device=device)
        cum_frech_rewards = torch.zeros(n_rollouts, dtype=torch.float32, device=device)
        episode_len = torch.zeros(n_rollouts, dtype=torch.int32, device=device)

        for step in range(0, params.n_max_steps_per_episode):

            action = torch.clamp(
                learner.deterministic_action(
                    process_state(
                        state,
                        env_config.state_representation,
                        exp_config.model,
                    )
                ),
                min=-1.0,
                max=1.0,
            )

            action_tensor = action.reshape(
                n_rollouts,
                n_agents,
                d_action,
            ).transpose(1, 0)

            # Turn action tensor into list of tensors with shape (n_env, action_dim)
            action_tensor_list = torch.unbind(action_tensor)

            state, reward, done, info = env.step(action_tensor_list)

            cumulative_rewards += reward[0]
            cum_frech_rewards = info[0]["frechet_rew"]
            cum_dist_rewards = info[0]["distance_rew"]

            episode_len += torch.ones(n_rollouts, dtype=torch.int32, device=device)

            # Create timeout boolean mask
            timeout = episode_len == params.n_max_steps_per_episode

            if torch.any(done) or torch.any(timeout):

                # Get done and timeout indices
                done_indices = torch.nonzero(done).flatten().tolist()
                timeout_indices = torch.nonzero(timeout).flatten().tolist()

                # Merge indices and remove duplicates
                indices = list(set(done_indices + timeout_indices))

                for idx in indices:
                    # Log data when episode is done
                    rewards.append(cumulative_rewards[idx].item())
                    distance_rewards.append(cum_dist_rewards[idx].item())
                    frechet_rewards.append(cum_frech_rewards[idx].item())

                    # Reset vars, and increase counters
                    state = env.reset_at(index=idx)
                    cumulative_rewards[idx] = 0

                    episode_count += 1

            if episode_count >= n_rollouts:
                break

        data[n_agents]["rewards"] = rewards
        data[n_agents]["dist_rewards"] = distance_rewards
        data[n_agents]["frech_rewards"] = frechet_rewards

        print(f"Done evaluating {n_agents}")

    # Store environment
    with open(dirs["logs"] / "evaluation.dat", "wb") as f:
        dill.dump(data, f)


def get_attention_data(
    exp_config: Experiment,
    env_config: EnvironmentParams,
    params: Params,
    device: str,
    dirs: dict,
    # Parameters for attention experiment
    n_agents: int,
    d_state: int,
    d_action: int,
    extra_agents: int = 64,
    seed=1998,
):

    n_agents_list = list(range(8, extra_agents + 1, 16))
    attention_dict = {}

    for i, n_agents in enumerate(n_agents_list):

        # Load environment
        env = create_env(
            dirs["batch"],
            1,
            device,
            env_config.environment,
            seed,
            training=False,
            n_agents=n_agents,
        )

        # Load PPO agent
        learner = PPO(
            device,
            exp_config.model,
            params,
            env_config.n_agents,
            n_agents,
            1,
            d_state,
            d_action,
        )
        learner.load(dirs["models"] / "best_model")

        # Set policy to evaluation mode
        learner.policy.eval()

        edge_indices = []
        attention_weights = []
        attention_over_time = {
            "Enc_L0": [],  # Encoder self-attention
            "Dec_L0": [],  # Decoder self-attention
            "Cross_L0": [],  # Cross-attention
        }
        match (exp_config.model):
            case (
                "transformer"
                | "transformer_full"
                | "transformer_encoder"
                | "transformer_decoder"
            ):
                attention_weights = learner.policy.build_attention_hooks()

        # Frame list for vide
        frames = []

        # Reset environment
        state = env.reset()

        for _ in range(0, params.n_max_steps_per_episode):

            action = torch.clamp(
                learner.deterministic_action(
                    process_state(
                        state,
                        env_config.state_representation,
                        exp_config.model,
                    )
                ),
                min=-1.0,
                max=1.0,
            )

            match (exp_config.model):
                case "gat" | "graph_transformer":
                    x = learner.policy.get_batched_graph(
                        process_state(
                            state,
                            env_config.state_representation,
                            exp_config.model,
                        )
                    )
                    _, attention_layers = learner.policy.forward_evaluation(x)

                    # Store edge indices and weights from last layer
                    # Make sure to do a deep copy
                    edge_index, attn_weight = attention_layers[-1]

                    # Store completely detached copies
                    edge_indices.append(edge_index.clone())
                    attention_weights.append(attn_weight.clone())

                case (
                    "transformer"
                    | "transformer_full"
                    | "transformer_encoder"
                    | "transformer_decoder"
                ):
                    _ = learner.policy.forward(
                        process_state(
                            state,
                            env_config.state_representation,
                            exp_config.model,
                        )
                    )

                    # Store attention weights for this timestep
                    for attn_type in attention_over_time:
                        if attn_type in attention_weights:
                            attention_over_time[attn_type].append(
                                attention_weights[attn_type].clone()
                            )

            action_tensor = action.reshape(
                1,
                n_agents,
                d_action,
            ).transpose(1, 0)

            # Turn action tensor into list of tensors with shape (n_env, action_dim)
            action_tensor_list = torch.unbind(action_tensor)

            state, _, done, _ = env.step(action_tensor_list)

            # Store frames for video
            frames.append(
                env.render(
                    mode="rgb_array",
                    agent_index_focus=None,  # Can give the camera an agent index to focus on
                    visualize_when_rgb=False,
                )
            )

            if torch.any(done):
                break

        # Save video
        save_video(
            str(dirs["videos"] / f"plots_video_{n_agents}"),
            frames,
            fps=1 / env.scenario.world.dt,
        )

        # Store environment
        attention_dict[n_agents] = {
            "edge_indices": edge_indices,
            "attention_weights": attention_weights,
            "attention_over_time": attention_over_time,
        }

    with open(dirs["logs"] / "attention.dat", "wb") as f:
        dill.dump(attention_dict, f)
