from deap import base
from deap import creator
from deap import tools
import torch
import torch.nn.functional as F

import random

from vmas.simulator.environment import Environment
from vmas.simulator.utils import save_video

from learning.algorithms.ccea.policies.mlp import MLP_Policy
from learning.algorithms.ccea.policies.gru import GRU_Policy

from learning.algorithms.ccea.selection import (
    binarySelection,
    epsilonGreedySelection,
    softmaxSelection,
)

from learning.environments.types import EnvironmentParams

from learning.algorithms.ccea.types import (
    Experiment,
    Params,
    Policy,
    Team,
    EvalInfo,
    PolicyEnum,
    SelectionEnum,
    FitnessShapingEnum,
    InitializationEnum,
    FitnessCalculationEnum,
)

from copy import deepcopy
import numpy as np
import logging

# Create and configure logger
logging.basicConfig(format="%(asctime)s %(message)s")

# Creating an object
logger = logging.getLogger()

# Setting the threshold of logger to DEBUG
logger.setLevel(logging.INFO)


class CooperativeCoevolutionaryAlgorithm:
    def __init__(
        self,
        device: str,
        env_config: EnvironmentParams,
        exp_config: Experiment,
        **kwargs,
    ):
        ccea_config = Params(**exp_config.ccea_config)
        policy_config = Policy(**ccea_config.policy_config)

        # Environment data
        self.device = device
        self.observation_size = env_config.observation_size
        self.action_size = env_config.action_size
        self.n_agents = len(env_config.agents)

        # Experiment Data
        self.n_gens_between_save = exp_config.n_gens_between_save

        # Policy
        self.output_multiplier = policy_config.output_multiplier
        self.policy_hidden_layers = policy_config.hidden_layers
        self.policy_type = policy_config.type
        self.weight_initialization = policy_config.weight_initialization

        # CCEA
        self.n_gens = ccea_config.n_gens
        self.n_steps = ccea_config.n_steps
        self.subpop_size = ccea_config.subpopulation_size
        self.n_mutants = self.subpop_size // 2
        self.selection_method = ccea_config.selection
        self.fitness_shaping_method = ccea_config.fitness_shaping
        self.fitness_calculation = ccea_config.fitness_calculation
        self.max_std_dev = ccea_config.mutation["max_std_deviation"]
        self.min_std_dev = ccea_config.mutation["min_std_deviation"]
        self.mutation_mean = ccea_config.mutation["mean"]

        self.std_dev_list = np.arange(
            start=self.max_std_dev,
            stop=self.min_std_dev,
            step=-((self.max_std_dev - self.min_std_dev) / (self.n_gens + 1)),
        )

        # Create the type of fitness we're optimizing
        creator.create("Individual", np.ndarray, fitness=0.0)

        # Now set up the toolbox
        self.toolbox = base.Toolbox()

        self.toolbox.register(
            "subpopulation",
            tools.initRepeat,
            list,
            self.createIndividual,
            n=self.subpop_size,
        )

        self.toolbox.register(
            "population",
            tools.initRepeat,
            list,
            self.toolbox.subpopulation,
            n=self.n_agents,
        )

    def createIndividual(self):
        match (self.weight_initialization):
            case InitializationEnum.KAIMING:
                temp_model = self.generateTemplateNN()
                params = temp_model.get_params()
        return creator.Individual(params[:].cpu().numpy().astype(np.float32))

    def generateTemplateNN(self):
        match (self.policy_type):

            case PolicyEnum.GRU:
                agent_nn = GRU_Policy(
                    input_size=self.observation_size,
                    hidden_size=self.policy_hidden_layers[0],
                    hidden_layers=len(self.policy_hidden_layers),
                    output_size=self.action_size,
                ).to(self.device)

            case PolicyEnum.MLP:
                agent_nn = MLP_Policy(
                    input_size=self.observation_size,
                    hidden_layers=len(self.policy_hidden_layers),
                    hidden_size=self.policy_hidden_layers[0],
                    output_size=self.action_size,
                ).to(self.device)

        return agent_nn

    def getBestAgents(self, population) -> list:
        best_agents = []

        # Get best agents
        for subpop in population:
            # Get the best N individuals
            best_ind = tools.selBest(subpop, 1)[0]
            best_agents.append(best_ind)

        return best_agents

    def formTeams(self, population, joint_policies: int) -> list[Team]:
        # Start a list of teams
        teams = []

        # For each row in the population of subpops (grabs an individual from each row in the subpops)
        for i in range(joint_policies):

            # Get agents in this row of subpopulations
            teams.append(
                Team(
                    individuals=[subpop[i] for subpop in population],
                )
            )

        return teams

    def evaluateTeams(
        self,
        env: Environment,
        teams: list[Team],
        render: bool = False,
        save_render: bool = False,
    ):
        # Set up models
        joint_policies = [
            [self.generateTemplateNN() for _ in range(self.n_agents)] for _ in teams
        ]

        # Load in the weights
        for i, team in enumerate(teams):
            for agent_nn, individual in zip(joint_policies[i], team.individuals):
                agent_nn.set_params(torch.from_numpy(individual).to(self.device))

        # Get initial observations per agent
        observations = env.reset()

        G_list = []
        frame_list = []
        D_list = []

        # Start evaluation
        for step in range(self.n_steps):

            stacked_obs = torch.stack(observations, -1)

            actions = [
                torch.empty((0, self.action_size)).to(self.device)
                for _ in range(self.n_agents)
            ]

            for observation, joint_policy in zip(stacked_obs, joint_policies):

                for i, policy in enumerate(joint_policy):
                    policy_output = policy.forward(observation[:, i])
                    actions[i] = torch.cat(
                        (
                            actions[i],
                            policy_output * self.output_multiplier,
                        ),
                        dim=0,
                    )

            observations, rewards, _, _ = env.step(actions)

            G_list.append(torch.stack([g[: len(teams)] for g in rewards], dim=0)[0])

            if self.fitness_shaping_method == FitnessShapingEnum.D:
                D_list.append(
                    torch.stack(
                        [d[len(teams) : len(teams) * 2] for d in rewards], dim=0
                    )
                )

            # Visualization
            if render:
                frame = env.render(
                    mode="rgb_array",
                    agent_index_focus=None,  # Can give the camera an agent index to focus on
                    visualize_when_rgb=True,
                )
                if save_render:
                    frame_list.append(frame)

        # Save video
        if render and save_render:
            save_video(self.video_name, frame_list, fps=1 / env.scenario.world.dt)

        # Compute team fitness
        match (self.fitness_calculation):

            case FitnessCalculationEnum.AGG:
                g_per_env = torch.sum(torch.stack(G_list), dim=0).tolist()

                if self.fitness_shaping_method == FitnessShapingEnum.D:
                    d_per_env = torch.transpose(
                        torch.sum(torch.stack(D_list), dim=0), dim0=0, dim1=1
                    ).tolist()

            case FitnessCalculationEnum.LAST:
                g_per_env = G_list[-1].tolist()

                if self.fitness_shaping_method == FitnessShapingEnum.D:
                    d_per_env = torch.transpose(D_list[-1], dim0=0, dim1=1).tolist()

        # Generate evaluation infos
        eval_infos = [
            EvalInfo(
                team=team,
                team_fitness=g_per_env[i],
                agent_fitnesses=(
                    d_per_env[i]
                    if self.fitness_shaping_method == FitnessShapingEnum.D
                    else g_per_env[i]
                ),
            )
            for i, team in enumerate(teams)
        ]

        return eval_infos

    def mutateIndividual(self, individual):

        individual += np.random.normal(
            loc=self.mutation_mean,
            scale=self.std_dev_list[self.gen],
            size=np.shape(individual),
        )

    def mutate(self, population):
        # Don't mutate the elites
        for n_individual in range(self.n_mutants):

            mutant_idx = n_individual + self.n_mutants

            for subpop in population:
                self.mutateIndividual(subpop[mutant_idx])
                subpop[mutant_idx].fitness = np.float32(0.0)

    def selectSubPopulation(self, subpopulation):

        match (self.selection_method):
            case SelectionEnum.BINARY:
                chosen_ones = binarySelection(subpopulation, tournsize=2)
            case SelectionEnum.EPSILON:
                chosen_ones = epsilonGreedySelection(
                    subpopulation, self.subpop_size // 2, epsilon=0.3
                )
            case SelectionEnum.SOFTMAX:
                chosen_ones = softmaxSelection(subpopulation, self.subpop_size // 2)
            case SelectionEnum.TOURNAMENT:
                chosen_ones = tools.selTournament(
                    subpopulation, self.subpop_size // 2, 2
                )

        offspring = chosen_ones + chosen_ones

        # Return a deepcopy so that modifying an individual that was selected does not modify every single individual
        # that came from the same selected individual
        return [deepcopy(individual) for individual in offspring]

    def select(self, population):
        # Perform a selection on that subpopulation and add it to the offspring population
        return [self.selectSubPopulation(subpop) for subpop in population]

    def shuffle(self, population):
        for subpop in population:
            random.shuffle(subpop)

    def assignFitnesses(
        self,
        eval_infos: list[EvalInfo],
    ):

        match (self.fitness_shaping_method):

            case FitnessShapingEnum.G:
                for eval_info in eval_infos:
                    for individual in eval_info.team.individuals:
                        individual.fitness = eval_info.team_fitness

            case FitnessShapingEnum.D:
                for eval_info in eval_infos:
                    for idx, individual in enumerate(
                        eval_info.team.individuals,
                    ):
                        individual.fitness = eval_info.agent_fitnesses[idx]

    def setPopulation(self, population, offspring):
        for subpop, subpop_offspring in zip(population, offspring):
            subpop[:] = subpop_offspring

    def step(self, n_gen, pop, env):

        # Set gen counter global var
        self.gen = n_gen

        # Perform selection
        offspring = self.select(pop)

        # Perform mutation
        self.mutate(offspring)

        # Shuffle subpopulations in offspring
        # to make teams random
        self.shuffle(offspring)

        # Form teams for evaluation
        teams = self.formTeams(offspring, joint_policies=self.subpop_size)

        # Evaluate each team
        eval_infos = self.evaluateTeams(env, teams)

        # Now assign fitnesses to each individual
        self.assignFitnesses(eval_infos)

        # Evaluate best team of generation
        avg_team_fitness = (
            sum([eval_info.team_fitness for eval_info in eval_infos]) / self.subpop_size
        )
        best_team_eval_info = max(eval_infos, key=lambda item: item.team_fitness)

        # Now populate the population with individuals from the offspring
        self.setPopulation(pop, offspring)

        return (
            pop,
            best_team_eval_info.team,
            best_team_eval_info.team_fitness,
            avg_team_fitness,
        )
