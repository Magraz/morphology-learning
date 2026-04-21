from algorithms.mappo.types import Experiment, MAPPO_Params, Model_Params
from environments.types import EnvironmentEnum

from dataclasses import asdict

# EXPERIMENT SETTINGS
ENVIRONMENT = EnvironmentEnum.MULTI_BOX
BATCH_NAME = f"{ENVIRONMENT}_test"
# EXPERIMENTS_LIST = ["mlp", "gru"]
EXPERIMENTS_LIST = ["hgnn_shared", "hgnn_shared_entropy"]
DEVICE = "cpu"

# EXPERIMENTS
experiments = []
for i, experiment_name in enumerate(EXPERIMENTS_LIST):
    experiment = Experiment(
        device=DEVICE,
        model_params=Model_Params(
            model_name="mlp",
            critic_type="multi_hgnn",
            hyperedge_fn_names=["proximity", "contact"],
            entropy_conditioning=True,
            entropy_pred_seq_len=32,
            entropy_pred_coef=0.01,
        ),
        params=MAPPO_Params(
            n_epochs=10,
            n_total_steps=1e7,
            n_minibatches=4,
            batch_size=5120,
            parameter_sharing=True,
            random_seeds=[118, 1234, 8764, 3486, 2487, 5439, 6584, 7894, 523, 69],
            eps_clip=0.2,
            grad_clip=0.5,
            gamma=0.99,
            lmbda=0.95,
            ent_coef=0.01,
            val_coef=0.8,
            std_coef=0.0,
            lr=3e-4,
        ),
    )
    experiments.append(experiment)

EXP_DICTS = [
    {
        "batch": BATCH_NAME,
        "name": EXPERIMENTS_LIST[i],
        "config": asdict(experiment),
    }
    for i, experiment in enumerate(experiments)
]
