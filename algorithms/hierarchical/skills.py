"""Loading frozen low-level skills for the hierarchical macro-action controller.

A "skill" is one of the 4 pre-trained MAPPO policies (contact / scatter /
push_box / rendezvouz). Every box2d_suite env shares ``ObservationManager``, so
each skill's actor consumes the same ``obs_dim`` local observation and emits the
same ``action_dim`` continuous force. Only their *critics* differ, which is
irrelevant here: the high-level controller only ever runs the skill *actors* for
inference. So a skill collapses to a single frozen, eval-mode ``MAPPOActor``.
"""

from pathlib import Path

import torch

from algorithms.mappo.networks.actors import MAPPOActor

# Repo root: .../algorithms/hierarchical/skills.py -> parents[2]
_REPO_ROOT = Path(__file__).resolve().parents[2]
_RESULTS_DIR = _REPO_ROOT / "experiments" / "results"

# Discrete high-level action index -> skill name. Index i selects skill
# SKILL_ORDER[i]; the matching batch holding its weights is SKILL_BATCHES[name].
SKILL_ORDER = ["contact", "scatter", "push_box", "rendezvouz"]
SKILL_BATCHES = {
    "contact": "contact_9a",
    "scatter": "scatter_9a",
    "push_box": "push_box_9a",
    "rendezvouz": "rendezvouz_9a",
}


def resolve_skill_checkpoint(
    batch: str,
    experiment: str = "mlp_shared",
    trial: str = "0",
) -> Path:
    """Path to a skill's checkpoint, preferring the finished model.

    Falls back to ``models_checkpoint.pth`` when ``models_finished.pth`` is
    absent (e.g. ``scatter_9a`` only ships the checkpoint file).
    """
    models_dir = _RESULTS_DIR / batch / experiment / trial / "models"
    finished = models_dir / "models_finished.pth"
    checkpoint = models_dir / "models_checkpoint.pth"
    if finished.exists():
        return finished
    if checkpoint.exists():
        return checkpoint
    raise FileNotFoundError(
        f"No skill checkpoint found in {models_dir} "
        f"(looked for models_finished.pth / models_checkpoint.pth)"
    )


def load_skill_actor(
    ckpt_path: Path,
    obs_dim: int = 40,
    action_dim: int = 2,
    hidden_dim: int | None = None,
    device: str = "cpu",
) -> MAPPOActor:
    """Build a frozen, eval-mode ``MAPPOActor`` from a MAPPO checkpoint.

    The shared actor's weights live in ``checkpoint["network"]`` under keys
    prefixed ``"actor."`` (e.g. ``actor.actor.0.weight``,
    ``actor.log_action_std``). Stripping that one prefix maps them exactly onto a
    fresh ``MAPPOActor`` state dict.

    The actor architecture (input/hidden/action dims) is read from the saved
    weights, not from the training yaml: the pre-trained ``mlp_shared`` actors
    use a hidden size (183) that differs from ``Model_Params.hidden_dim`` (168),
    so inferring keeps the loader correct regardless. The ``obs_dim`` /
    ``action_dim`` args are asserted against the checkpoint as a guard.
    """
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    network_sd = checkpoint["network"]

    prefix = "actor."
    actor_sd = {
        key[len(prefix):]: value
        for key, value in network_sd.items()
        if key.startswith(prefix)
    }

    # Infer architecture from the first/last linear layers of the actor MLP.
    in_features = actor_sd["actor.0.weight"].shape[1]
    inferred_hidden = actor_sd["actor.0.weight"].shape[0]
    out_features = actor_sd["actor.4.weight"].shape[0]
    if obs_dim is not None and in_features != obs_dim:
        raise ValueError(
            f"Skill {ckpt_path}: actor input dim {in_features} != expected obs_dim "
            f"{obs_dim}. Was this skill trained with entropy conditioning?"
        )
    if action_dim is not None and out_features != action_dim:
        raise ValueError(
            f"Skill {ckpt_path}: actor output dim {out_features} != expected "
            f"action_dim {action_dim}."
        )
    if hidden_dim is not None and inferred_hidden != hidden_dim:
        # Trust the checkpoint; the yaml hidden_dim is not authoritative here.
        hidden_dim = inferred_hidden
    hidden_dim = inferred_hidden

    actor = MAPPOActor(in_features, out_features, hidden_dim, discrete=False)
    actor.load_state_dict(actor_sd)
    actor.to(device)
    actor.eval()
    actor.requires_grad_(False)
    return actor


def load_skills(
    obs_dim: int = 40,
    action_dim: int = 2,
    hidden_dim: int = 168,
    device: str = "cpu",
    experiment: str = "mlp_shared",
    trial: str = "0",
    skill_batches: dict | None = None,
) -> list[MAPPOActor]:
    """Load the 4 skill actors in ``SKILL_ORDER`` (index = high-level action).

    ``skill_batches`` optionally overrides the default per-skill batch names.
    """
    batches = {**SKILL_BATCHES, **(skill_batches or {})}
    actors = []
    for name in SKILL_ORDER:
        ckpt = resolve_skill_checkpoint(batches[name], experiment, trial)
        actors.append(
            load_skill_actor(ckpt, obs_dim, action_dim, hidden_dim, device)
        )
    return actors
