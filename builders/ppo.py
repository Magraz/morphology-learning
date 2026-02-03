from algorithms.ppo.types import Experiment, Params
from environments.types import EnvironmentEnum

from dataclasses import asdict

# EXPERIMENT SETTINGS
ENVIRONMENT = EnvironmentEnum.VMAS_SALP_PASSAGE
BATCH_NAME = f"{ENVIRONMENT}_3a"
EXPERIMENTS_LIST = [
    "mlp",
    "gru",
]
# EXPERIMENTS_LIST = [
#     "mlp",
#     "transformer_full",
#     "gcn",
#     "gat",
#     "graph_transformer",
#     "gcn_full",
#     "gat_full",
#     "graph_transformer_full",
# ]
DEVICE = "cpu"

# EXPERIMENTS
experiments = []
for experiment_name in EXPERIMENTS_LIST:
    experiment = Experiment(
        device=DEVICE,
        model=experiment_name,
        params=Params(
            n_epochs=10,
            n_total_steps=2e8,
            n_minibatches=4,
            batch_size=5120,
            random_seeds=[118, 1234, 8764, 3486, 2487, 5439, 6584, 7894, 523, 69],
            eps_clip=0.2,
            grad_clip=0.5,
            gamma=0.99,
            lmbda=0.95,
            ent_coef=1e-2,
            val_coef=0.5,
            std_coef=0.0,
            lr=3e-4,
        ),
    )
    experiments.append(experiment)

EXP_DICTS = [
    {
        "batch": BATCH_NAME,
        "name": experiment.model,
        "config": asdict(experiment),
    }
    for experiment in experiments
]
