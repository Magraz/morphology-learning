from operator import attrgetter

import random
import numpy as np
import torch
import torch.nn.functional as F


def binarySelection(individuals, tournsize: int, fit_attr: str = "fitness"):

    # Shuffle the list randomly
    random.shuffle(individuals)

    # Create list of random pairs without repetition
    pairs_of_candidates = [
        (individuals[i], individuals[i + 1])
        for i in range(0, len(individuals) - 1, tournsize)
    ]

    chosen_ones = [
        max(candidates, key=attrgetter(fit_attr)) for candidates in pairs_of_candidates
    ]

    return chosen_ones


def epsilonGreedySelection(
    individuals,
    k: int,
    epsilon: float,
    fit_attr: str = "fitness",
):

    chosen_ones = []

    for _ in range(k):
        if np.random.choice([True, False], 1, p=[1 - epsilon, epsilon]):
            chosen_one = max(individuals, key=attrgetter(fit_attr))
        else:
            chosen_one = random.choice(individuals)

        chosen_ones.append(chosen_one)

    return chosen_ones


def softmaxSelection(
    individuals,
    k: int,
):

    chosen_ones = []

    individuals_fitnesses = [individual.fitness for individual in individuals]
    softmax_over_fitnesses = F.softmax(torch.Tensor(individuals_fitnesses), dim=0)
    selected_indexes = torch.multinomial(softmax_over_fitnesses, num_samples=k)

    for idx in selected_indexes:
        chosen_ones.append(individuals[idx])

    return chosen_ones
