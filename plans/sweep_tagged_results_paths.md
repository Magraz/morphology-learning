# Sweep-tagged results paths: `tag=` instead of a yaml per value

**Status:** planned, not implemented. The working tree is clean — nothing in this
plan has been applied.

## Context

Sweeping a hyperparameter today requires creating one `conf/model/*.yaml` per
value. That is duplicated state (the value appears in both the filename and the
body) and it does not scale — one axis costs 3 files, a 3x3 costs 9.

The cause is `train.py:_build_dispatch_args`, which derives the output directory
from Hydra **group choices** only:

```py
batch, name = choices["env"], choices["model"]
results_dir = .../ "experiments" / "results" / batch / name    # + /<trial_id>
```

Anything swept as a plain override (`-m params.intrinsic_coef=0,0.1,0.5`) is
invisible to that path, so every arm writes into one directory and clobbers the
others' `training_stats_*.pkl`. Creating a model group per value is the current
workaround, and it is why the feudal alpha ablation is blocked.

Goal: sweep any `params.*` / `model_params.*` / `env.*` key straight from the CLI
and get one results folder per arm, with **no change to any existing path**.

### Constraints found during research (these shape the design)

1. **The layout is exactly three deep** and `plotting/plot_training_stats.ipynb`
   does *no globbing* — it builds `base_path / batch / experiment / trial /
   "logs" / "training_stats_checkpoint.pkl"` from three literal name lists in
   `plotting/config.yaml`. So the sweep identity must be a **suffix on the
   `<model>` component**, never a new directory level.
2. **Hydra's `override_dirname` cannot be used raw.** Its `exclude_keys` is exact
   string match (`hydra/_internal/config_loader_impl.py:588-600`) with no prefix
   matching, so it is a deny-list. Measured on this config it renders as
   `+tag=...,algorithm=...,checkpoint=true,env=...,model=...,trial_id=0` — every
   existing run would move, and every future control knob would silently fork the
   path. (It does correctly ignore `hydra/launcher=` and `hydra.*` overrides,
   confirmed at `config_loader_impl.py:277-283`.) We need an **allow-list**, so
   the default stays today's behavior.
3. **`params.n_total_steps=... checkpoint=true` must keep resuming.** CLAUDE.md
   documents extending a finished run this way; auto-tagging that override would
   send it to a fresh empty folder. Same for `env.n_envs`, the documented
   autoscale opt-out. Both are exempt.
4. Cross-experiment consumers address producers by **bare model name** —
   `algorithms/hierarchical/skills.py:43` (`<batch>/<experiment>/<trial>/models`)
   and `algorithms/mappo/trainer_components/hypergraph_runtime.py:25-28`
   (hardcoded `"learned_affinity"`). Untagged paths stay byte-identical, so
   neither is affected.

## Design

A top-level `tag` key (default `null`) appended to the name component as
`<model>__<tag>`, plus a guard that makes forgetting it impossible.

```
# readable name, interpolating the swept value
uv run python train.py -m algorithm=feudal_mappo_jax env=mjx_16a_4o model=feudal \
    params.intrinsic_coef=0,0.1,0.5 'tag=alpha${params.intrinsic_coef}' trial_id=0,1,2
  -> experiments/results/mjx_16a_4o/feudal__alpha0.0/{0,1,2}
                                    feudal__alpha0.1/{0,1,2}
                                    feudal__alpha0.5/{0,1,2}

# machine-derived name
    params.intrinsic_coef=0,0.1,0.5 tag=auto
  -> experiments/results/mjx_16a_4o/feudal__params.intrinsic_coef=0.5/...

# nothing typed -> path unchanged
  -> experiments/results/mjx_16a_4o/feudal/0
```

Both forms were **verified against the real config** during planning: the
interpolation resolves to `alpha0.5`, and `cfg.hydra.overrides.task` yields the
raw override list the guard reads (populated per-job under the joblib launcher,
since `load_sweep_config` recomposes each job from its own task overrides).

## Changes

### 1. `conf/config.yaml` — declare the key

Add next to the other top-level knobs (`device`/`trial_id`/`view`/...). Declaring
it means `tag=...` works without the `+` prefix.

```yaml
# Sweep identity. The results path is <env>/<model>/<trial_id> and records no
# arbitrary override, so a bare `-m params.foo=1,2` sweep would write both arms
# into ONE directory and clobber each other's training_stats_*.pkl. `tag` is
# appended to the model component (<model>__<tag>) — a suffix, not a directory
# level, because plotting/plot_training_stats.ipynb reads the layout at exactly
# three deep. null (default) leaves every existing path untouched; `tag=auto`
# derives the name from the swept overrides. Sweeping a params./model_params./
# env. key with no tag is a hard error. See `_build_dispatch_args` in train.py.
#
#   -m params.intrinsic_coef=0,0.1,0.5 'tag=alpha${params.intrinsic_coef}'
tag: null
```

### 2. `train.py` — the substantive change

**(a) Module-level constants and two pure helpers**, placed after the
`OmegaConf.register_new_resolver` calls and before `_build_dispatch_args`
(needs `import re` added at the top):

```py
# --------------------------------------------------------------------------
# Sweep tagging
#
# The results path is `<env>/<model>/<trial_id>` and records no arbitrary
# override, so an override that changes the experiment but not the path makes
# two arms share a directory. `tag` fixes that (see conf/config.yaml); these
# constants decide which overrides *require* one.
#
# This is deliberately an ALLOW-list, not a deny-list. Hydra's own
# `job.override_dirname` is the latter, and it renders every override you typed
# -- `algorithm=`, `env=`, `model=`, `trial_id=`, `checkpoint=true` -- so
# adopting it would move every existing results directory and make any future
# control knob silently fork the path. With an allow-list the default stays
# exactly today's behaviour and only experiment-defining keys can tag a run.
# --------------------------------------------------------------------------
TAGGED_PREFIXES = ("params.", "model_params.", "env.")

# ...except these, which are budget/parallelism rather than arm identity.
# `params.n_total_steps=<bigger> checkpoint=true` is the documented way to EXTEND
# a finished run, so it must resolve to the ORIGINAL directory rather than a
# fresh empty one; `env.n_envs` is the documented autoscale opt-out.
UNTAGGED_OVERRIDES = frozenset({"params.n_total_steps", "env.n_envs"})

# Anything else is replaced in an auto tag. `/` would corrupt the plotting
# notebook's `plot_group.split("/", maxsplit=1)`, and an unquoted `,` splits a
# YAML flow sequence in plotting/config.yaml's `experiments:` list.
_TAG_UNSAFE = re.compile(r"[^A-Za-z0-9._=+-]")


def _override_key(override: str) -> str:
    """Config key of a raw override line, minus any `+`/`++`/`~` prefix."""
    return override.split("=", 1)[0].lstrip("+~")


def _arm_overrides(task_overrides) -> list[str]:
    """The typed overrides that define this arm rather than how it is run."""
    return sorted(
        o
        for o in task_overrides
        if _override_key(o).startswith(TAGGED_PREFIXES)
        and _override_key(o) not in UNTAGGED_OVERRIDES
    )


def _auto_tag(arm_overrides) -> str:
    """Directory-safe rendering of `_arm_overrides` (the `tag=auto` name)."""
    return "_".join(_TAG_UNSAFE.sub("-", o) for o in arm_overrides)
```

**(b) `_build_dispatch_args(cfg, choices, task_overrides=())`** — new *defaulted*
third parameter, so the existing equivalence-harness call site keeps working.
After `batch, name = choices["env"], choices["model"]`:

```py
tag = c.get("tag")
arm = _arm_overrides(task_overrides)
if tag is None and arm:
    raise RuntimeError(...)          # names the overrides + the colliding path
if tag == "auto":
    tag = _auto_tag(arm) or None     # `tag=auto` with nothing to tag -> no suffix
if tag:
    name = f"{name}__{tag}"
```

The error message must name the offending overrides, the directory they would
collide in, and the three ways out (`tag=<name>`, `tag=auto`, or adding the key
to `UNTAGGED_OVERRIDES`). Extend the docstring to describe `task_overrides` and
why the tag is a name suffix rather than a directory level.

**(c) `main`** — pass the overrides through:

```py
hc = HydraConfig.get()
_dispatch(**_build_dispatch_args(cfg, hc.runtime.choices, hc.overrides.task))
```

**(d) Module docstring** — add a sweep example line and note that a sweep
suffixes the model component.

### 3. `CLAUDE.md`

- Update the feudal intrinsic-reward bullet (it currently points at this plan and
  warns the sweep is blocked) with the working `tag=` command.
- Add a short **Sweeps (`tag=`)** subsection to the Hydra config section
  documenting: the `<model>__<tag>` layout, the `TAGGED_PREFIXES` allow-list and
  why it is an allow-list, the two `UNTAGGED_OVERRIDES` exemptions and why, the
  guard, and the fact that `view`/`evaluate`/`checkpoint` on a tagged run need
  the **same `tag=`** (pass it literally, e.g. `tag=alpha0.5`) to resolve the
  directory.

## Verification

1. **No existing path moves** — compose every `(algorithm, env, model)` combo
   currently in `conf/` through `_build_dispatch_args` with no `tag` and assert
   `results_dir` equals `experiments/results/<env>/<model>` exactly. This is the
   backward-compatibility contract; it protects every checkpoint on disk.
2. **Tag paths** — compose with `tag=alpha${params.intrinsic_coef}` and with
   `tag=auto` over `params.intrinsic_coef=0,0.1,0.5`; assert three distinct
   directories and the right `intrinsic_coef` in each `exp_dict`.
3. **Guard fires** — `params.intrinsic_coef=0.5` with no tag raises
   `RuntimeError`; `params.n_total_steps=2e8 checkpoint=true` does **not** raise
   and resolves to the untagged directory (the resume workflow).
4. **Real multirun smoke** (writes to disk, then delete the smoke dirs):
   ```
   uv run python train.py -m algorithm=feudal_mappo_jax env=mjx_16a_4o \
       model=feudal params.intrinsic_coef=0,0.5 'tag=alpha${params.intrinsic_coef}' \
       params.n_steps=8 params.n_total_steps=512 trial_id=smoke_test
   ```
   Expect `experiments/results/mjx_16a_4o/feudal__alpha{0,0.5}/smoke_test/logs/
   training_stats_finished.pkl` — two separate arms, neither in `feudal/`.
5. **Nothing else regressed** — `uv run pytest algorithms/tests/test_feudal_seams.py -q`.
6. **Plotting reads a tagged run** — point `plotting/config.yaml` at
   `experiments: ["feudal__alpha0.5"]` and confirm the notebook's first data cell
   loads it (the name has no `/` or `,`, so `plot_group` and the YAML flow
   sequence both survive). Note the legend label becomes the raw directory name
   unless a `custom_labels` entry is added.

## Rejected alternatives

- **One `conf/model/*.yaml` per value** (`feudal_a0/a01/a05`) — works and is the
  repo's existing idiom, but duplicates the value in filename and body and costs
  a file per cell of the sweep grid. This plan replaces it.
- **Always auto-tag, no `tag` key** — no flag to remember, but it breaks the
  `params.n_total_steps=<bigger> checkpoint=true` resume workflow and gives only
  machine-generated folder names.
- **Raw `hydra.job.override_dirname`** — see constraint 2; it is a deny-list and
  would relocate every existing results directory.
