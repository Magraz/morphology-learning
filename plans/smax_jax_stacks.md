# Plan: run `mappo_jax` and `feudal_mappo_jax` on SMAX

> **STATUS: implemented 2026-09-01.** Both stacks train, resume, evaluate and render on
> SMAX. See the `## SMAX (JaxMARL StarCraft)` section of `CLAUDE.md` for the shipped
> design. Deltas from the plan below, all discovered during implementation:
>
> - **The `jax.tree_*` shim IS required** (`environments/smax/_compat.py`). Phase 0
>   originally concluded it was not, on the strength of `import jaxmarl` succeeding —
>   that was too shallow a check. `jaxmarl.make(...)` and `env.reset(key)` also succeed
>   without it; only a call that walks a pytree (`step_env`, `get_avail_actions`) raises.
> - **`obs["world_state"]` already exists** in SMAX's own step output, so the adapter
>   carries it on `SMAXState` rather than calling `get_world_state` separately — the
>   global state can then never drift from the observation it is paired with.
>   72 dims at `3m` against 195 for concatenated observations.
> - **SMAX's `info` dict is empty**, so `task_reward`, `active` and `won_episode` are all
>   synthesized by the adapter.
> - `step_env` confirmed as the non-auto-reset entry point; the step counter is
>   `state.state.time`; `env.agents` is allies-only under `HeuristicEnemySMAX`.
> - The no-hook byte-identity claim was **verified** (two git worktrees, CPU stub env,
>   both stacks: identical losses, `param_sq_sum` and eval).
> - `view()` was implemented in full (not stubbed) on both stacks.

## Context

Both JAX stacks are hard-wired to the MJX suite. `algorithms/mappo_jax/run.py:101-161`
and its `feudal_mappo_jax` twin construct one of three `MultiBoxPushMJX`-family envs
and **raise on anything else**. The trainers assume a gymnax-style *functional array*
contract (`environments/mjx_suite/multi_box_push_mjx.py`): `reset(key) -> (obs, state)`,
`step(state, actions) -> (obs, state, reward, terminated, truncated, info)`, obs
`(n_agents, obs_dim)`, no auto-reset, no action masking, and a critic global state
built as `obs.reshape(n_envs, -1)`.

SMAX (JaxMARL StarCraft) is a **different shape**: a PettingZoo-parallel dict API keyed
by agent name, `step(key, state, actions)` (needs an RNG — the heuristic enemy is
stochastic), auto-reset inside `MultiAgentEnv.step`, a combined `dones["__all__"]` with
no truncation flag, legal-action masks via `get_avail_actions(state)`, and units that
die mid-episode.

A JaxMARL dict path used to exist in `mappo_jax` (commit `f074a3a`) and was deleted in
`e78638b` when the stack was rewired to MJX. It is **not** worth resurrecting: it had no
truncation bootstrap, no eval, no per-agent-reward path, no checkpointing, and none of
the feudal machinery. Instead, **adapt SMAX to the contract the trainers already have**,
and add exactly three capabilities the contract is missing (action masks, a global-state
hook, a per-agent alive mask).

`EnvironmentEnum.SMAX = "SMAX"` already exists (`environments/types.py:38`) but is dead —
nothing dispatches it.

Goal: `train.py algorithm={mappo_jax,feudal_mappo_jax} env=smax_3m model={mlp,feudal}`
trains, checkpoints, resumes and evaluates, with **every existing MJX arm byte-identical**.

---

## Phase 0 — Dependency: DONE

`uv add jaxmarl` has been run (`pyproject.toml` now carries `jaxmarl>=0.0.2`). Verified:

- `import jaxmarl` succeeds on the installed **JAX 0.10.2** — so the `jax.tree_*` shim the
  deleted path needed (commit `f074a3a`, lines 10-13) is **not required**. Do not add it.
- `jax.default_backend()` is still `gpu` (`CudaDevice(id=0)`), so the training path is
  intact. There is a **pre-existing, non-fatal** `jax_cuda13_plugin` vs `jaxlib 0.10.2`
  `ALREADY_EXISTS: PJRT_Api` warning on every import; it does not prevent GPU use. Leave it.

Still to confirm at the start of Phase 1 (version-specific, and the adapter's shape depends
on the answers): whether `env.get_world_state(state)` / an `obs["world_state"]` key
exists, whether `step_env` is the non-auto-reset entry point, and the field that holds
the step counter on `state` (`state.time` vs `state.step`). The dead demo
`environments/smax/smax_introduction.py` pins the rest of the surface:
`make("HeuristicEnemySMAX", scenario=..., action_type=, observation_type=)`,
`env.agents`, `env.num_agents`, `env.num_allies`, `env.observation_space(a).shape`,
`env.action_space(a).n`, `env.get_avail_actions(state)`, `env.step(key, state, actions)`,
`dones["__all__"]`, `SMAXVisualizer(env, state_seq).animate(save_fname=...)`.

---

## Phase 1 — The adapter: `environments/smax/smax_env.py`

New `SMAXAdapter` presenting SMAX under the MJX functional contract. **No trainer sees
a dict.** Pure, `jit`/`vmap`-able.

```python
@register_pytree_node_class          # or flax.struct.dataclass, matching EnvState
class SMAXState:
    env_state: Any    # jaxmarl SMAX State
    key: jax.Array    # carried RNG — SMAX.step needs one, the MJX contract has none
```

`SMAXState` **must be a flat pytree whose every leaf carries a leading `n_envs` axis and
survives `jnp.where`** — the collector's `_restart_done` (`trainer.py:152-169`) does
`jax.tree.map(_select, reset_state, cur_state)` with a rank-broadcast `done`. The carried
`key` (uint32 `(n_envs, 2)`) and SMAX's int/bool/float leaves all satisfy this; a Python-int
or `None` leaf would not.

**Attributes** consumed by `make_train` / `run.py` (`trainer.py:53-58,306`,
`run.py:166-167,196,387-389`):
`n_agents`, `observation_dim`, `action_dim`, `discrete=True`, `max_steps`,
`reward_mode="dense"`, `per_agent_rewards=False`, plus the two new hooks below.
Derive the dims the way the deleted path did — `env.observation_spaces[a].shape[0]`,
`env.action_spaces[a].n`, `isinstance(env.action_spaces[a], jaxmarl…spaces.Discrete)`.
⚠ `self.env.reward_mode` is read **unguarded** at `run.py:167`, before `make_train` is ever
called — a missing attribute crashes at construction, not at train time.

Control **only `env.agents`** (for `HeuristicEnemySMAX` that is the allies; the enemy team
is driven by the built-in heuristic). Fix `self.agents` once at construction and use that
one ordering for every stack/un-stack — obs, actions, rewards, masks, alive flags.

**`reset(key)`** — split into `(reset_key, carry_key)`; call `env.reset(reset_key)`;
stack the obs dict into `(n_agents, obs_dim)` in `self.agents` order (agent-major,
matching every reshape in the stack).

**`step(state, actions)`**:
1. Split the carried key.
2. Un-stack `actions` `(n_agents,)` back into a dict over `self.agents`.
3. **Call `env.step_env(key, env_state, actions)`, not `env.step`** — `MultiAgentEnv.step`
   auto-resets on done, so running it *plus* the collector's own `_restart_done` is a
   double reset, and the truncation bootstrap at `trainer.py:143-147` would value the
   post-reset obs instead of the true successor. (Confirm `step_env` is the non-auto-reset
   entry point for the installed version; if the name differs, the adapter must instead
   defeat the auto-reset by other means — this is the single highest-risk correctness
   detail in the port.) `eval_fn` also relies on stepping past a finished episode without
   NaN-ing (`trainer.py:296-297` masks by `finished`); a terminated SMAX state stepped
   with `step_env` is static, which satisfies that.
4. Split the done flag the trainer needs two separate answers from:
   `truncated = next_env_state.time >= self.max_steps`,
   `terminated = dones["__all__"] & ~truncated`.
   (The whole `mappo_jax`/`mappo_vanilla`/`mappo` truncation-bootstrap design in
   CLAUDE.md hinges on these being distinct; SMAX only gives the union.)
5. `reward` = team scalar = `rewards[self.agents[0]]` (SMAX shares it; assert this at
   construction when `use_self_play_reward=False`).
6. `info` must carry:
   - `"task_reward"` — the team scalar. **Read unconditionally at
     `trainer.py:132` and `:295`**; omitting it is a `KeyError`.
   - `"active"` — the `(n_agents,)` **alive mask, computed from the PRE-step state**
     (the transition stores the pre-step obs/action, so the mask must describe who was
     alive when they acted). Feeds `Transition.active_mask` with zero trainer changes
     (`trainer.py:174`, `mappo.py:212,242,266,285`).
   - `"won_episode"` — for the SMAX win-rate log line.

**`avail_actions(state) -> (n_agents, action_dim)`** float32 — stacked
`env.get_avail_actions(state)`, computed at the **pre-step** state.

**`global_state(state) -> (global_state_dim,)`** + `global_state_dim` —
`env.get_world_state(state)` if it exists, else the concat fallback. This is the
user-chosen "world_state hook".

**`render_episode(state_seq, path)`** — wraps `SMAXVisualizer(env, state_seq).animate()`;
`state_seq` is a list of `(key, smax_state, action_dict)` tuples.

Keep `action_type="discrete"`. SMAX's `"continuous"` mode exists but discrete + masking
is the standard benchmark and is what makes DCG-comparable numbers.

---

## Phase 2 — Three capability additions to the shared stack

All three are **inert by default**, so every MJX arm stays byte-identical. Each change
must be made **twice, identically**, in `mappo_jax/` and `feudal_mappo_jax/` (CLAUDE.md
already requires those two run.py files stay in sync; the trainer/mappo pairs are the
same near-copies).

### 2a. Action masking — reuse what already exists

The deleted JaxMARL path had this plumbing live end-to-end and is a **drop-in template**
for all four sites: `git show f074a3a:algorithms/mappo_jax/{types,trainer,mappo}.py`
(`Transition.action_mask`, the `masks_flat` reshape into the shared-actor forward, the
`mb_masks = masks_flat[mb_ids]` slice, and the `evaluate_action(..., mb_masks)` call).

`network.py` **already** has the whole masking path and it is currently dead code:
`_masked_logits` (`network.py:116-119`) and the `action_mask=None` parameter on both
`sample_action` (`:139`) and `evaluate_action` (`:168`). Nothing passes it
(`grep action_mask algorithms/*/mappo.py` → empty). Wire it, don't write it.

- `types.py` — add `Transition.action_mask` (`(n_envs, n_agents, action_dim)`).
- `trainer.py` — static flag `use_action_mask = hasattr(env, "avail_actions")` in
  `make_train`. When false, store a scalar placeholder and pass `None` (avoids
  allocating a `(T,E,N,A)` array for continuous MJX runs and keeps numerics untouched).
  Thread a `mask` argument through `_actor_forward` (`trainer.py:76-93`) — flatten it
  `(b*n_agents, action_dim)` **agent-major, exactly like the obs reshape at `:83`**.
- Call sites: `_env_step` (sampling), **and `eval_fn` (`trainer.py:284-298`)** — a
  deterministic `argmax` over unmasked logits will happily pick an illegal action, so
  eval scores would be garbage; and `view()`.
- `mappo.py:ppo_update` — flatten `mask_ts` alongside `obs_ts`/`act_ts`
  (`mappo.py:203-212`), slice `mb_mask = mask_ts[mb_ids].reshape(n_flat, action_dim)`,
  pass to `evaluate_action` (`mappo.py:256-258`). Gate on
  `use_mask = trajectory.action_mask.ndim == 4`, mirroring the existing
  `per_agent = trajectory.reward.ndim == 3` idiom (`mappo.py:177`).
- **Correctness requirement:** the mask stored in the transition must be the one used at
  sampling time, or the PPO ratio is invalid at update time. Same class of bug as the
  feudal pooled-goal storage rule in CLAUDE.md. Pin it with the ratio==1 test (Phase 4).

### 2b. Global-state hook

Replace the four hardcoded `obs.reshape(n_envs, -1)` sites with one indirection built in
`make_train`:

```python
if hasattr(env, "global_state"):
    _v_gs = jax.vmap(env.global_state)
    def _gs(obs, env_state): return _v_gs(env_state)
    gs_dim = env.global_state_dim
else:
    def _gs(obs, env_state): return obs.reshape(obs.shape[0], -1)
    gs_dim = obs_dim * n_agents
```

Sites — `mappo_jax/trainer.py`: `:109` (init dims), `:120`, `:144`, `:248`.
`feudal_mappo_jax/trainer.py`: `:168`, `:181`, `:218`, `:425`, `:509` (the manager's
eval-path global state).

Two snags, both easy to get wrong:

- `:144` computes the truncation bootstrap `next_value` from `next_obs` **before** the
  reset `lax.cond` at `:166`. With a state-derived global state it must use
  `next_env_state` *before* that cond too. Keep the ordering.
- `collect_fn` currently **discards the final env state**: `(train_state, _, last_obs, rng)`
  at `trainer.py:235`. The `last_value` bootstrap at `:248` now needs it — bind it.

The `else` branch reproduces today's behavior exactly, so no MJX arm moves.

### 2c. Dead agents

Nothing new to build: `Transition.active_mask` and its masked means in `ppo_update`
(`mappo.py:266-267, 285-287`) were built for `SyncMacroMJX`'s staggered starts and apply
verbatim. The adapter supplies `info["active"]` and `trainer.py:174`'s
`info.get("active", ones)` picks it up. Dead units are dropped from the policy loss, the
entropy loss and the per-agent critic head; all-ones for every other env keeps those
reductions exact plain means.

---

## Phase 3 — Wiring and config

**`run.py` (both stacks, identical edits):** a fourth branch before the `else: raise`
(`mappo_jax/run.py:155-161`, `feudal_mappo_jax/run.py:~157`):

```python
elif environment == EnvironmentEnum.SMAX:
    from environments.smax.smax_env import SMAXAdapter
    self.env = SMAXAdapter(
        map_name=env_config.get("env_variant", "3m"),
        smax_env_id=env_config.get("smax_env_id", "HeuristicEnemySMAX"),
        max_steps=env_config.get("max_steps", 100),
        walls_cause_death=env_config.get("walls_cause_death", True),
        use_self_play_reward=env_config.get("use_self_play_reward", False),
    )
```

- **Do not pass `n_agents`** — the SMAX scenario owns it. Read it back off the adapter.
- **Do not set `max_steps=self.params.n_steps`** the way the MJX branches do
  (`run.py:105,114,140`). SMAX's episode length is a property of the benchmark; keep it
  independent of the rollout length.
- Update the `else:` error string to list SMAX.
- `run.py:166-167` (`per_agent_rewards`) and `:196` (`self.env.reward_mode`) read those
  attributes unconditionally — the adapter supplies both.

**`view()`** (`run.py:417-535`) is deeply MJX-coupled and is the one part that cannot be
adapted, only branched around: it imports `MJXRenderer`/`MuJoCoNativeRenderer`
unconditionally at `:423`, and those read `env.world_width`, `env.sector_sensor_radius`,
`env.n_objects`, `env.objects_push_coupling_list`, `env._build_xml(...)`, `state.data.qpos`,
plus hardcoded MJX obs-slice constants. The macro path additionally probes `env.macro_len`,
`env.env`, `env.base_state` and the private `env._skill_actions`.
Branch at the top of `view()` on `hasattr(self.env, "render_episode")` and route SMAX to
`SMAXVisualizer`; the recording loop accumulates `(key, smax_state, action_dict)` tuples
(the pattern is already in `smax_introduction.py:121-132`). Deterministic-argmax action
selection here needs the avail-actions mask too (2a). Landing this as a
`NotImplementedError` first and the visualizer second is a reasonable split.

**Config groups** — `conf/env/smax_3m.yaml`, `conf/env/smax_5m_vs_6m.yaml`:

```yaml
# @package _global_
env:
  environment: SMAX          # matches EnvironmentEnum.SMAX's literal value
  env_variant: 3m            # SMAX scenario / map name
  n_envs: 64                 # LITERAL: a vmap width on one device, i.e. a
                             # hyperparameter (batch = n_steps * n_envs), NOT a
                             # core budget. Keep identical across compared arms.
  max_steps: 100
params:                      # COLUMN 0 — see below
  n_steps: 128               # >= env max_steps: collect_fn resets every env at the
                             # top of every rollout, so a shorter n_steps means the
                             # back half of every episode is never trained on
  n_total_steps: 1e7
```

⚠ Three CLAUDE.md rules that bite here:
- **`n_steps >= env.max_steps`.** `collect_fn` (`trainer.py:227-240`) resets all envs at
  the start of every rollout and scans exactly `n_steps`; CLAUDE.md records this biting
  `mjx_16a_4o_multi_goal` (`n_steps: 512` against `max_steps: 1024`, so the late-episode
  delivery bonuses were outside the training distribution). SMAX wins land at episode end,
  so this matters here for the same reason.
- **`params:` must be at column 0, not indented under `env:`.** Nothing validates the
  `env:` block, so a misindented key is silently inert — this already cost a whole
  experiment group (`mjx_16a_4o`, commit `1fbb3f3`).
- `n_envs` belongs in the env group and must be a **literal** for a vmapped JAX env
  (never `${envs_per_job:...}`, which is for subprocess box2d envs).

**`environments/types.py`** — no change; `EnvironmentEnum.SMAX` already exists. Do **not**
touch `create_env.py`; that is the Gym/subprocess path and no JAX env goes through it.

**Feudal-only notes:** `model=feudal` is mandatory (`train.py` writes
`results/<env>/<model>/`, the algorithm is not in the path, so `model=mlp` would collide
with the `mappo_jax` baseline). The worker's discrete head, `bind_goal`
(`worker.py:169-186`) and `zero_goal` all work unchanged — `bind_goal` returns the flat
actor signature, so the mask threads through `sample_action` identically. Two things to
watch on SMAX: `goal_direction_count`'s healthy baseline is `N²/(N + N(N-1)/goal_dim)`,
which changes with the scenario's agent count; and dead agents still get goals, so
`transition_cosine` / the intrinsic reward will include their frozen latents — low
priority while `intrinsic_coef: 0.0`, but note it before running an intrinsic arm.

---

## Phase 4 — Verification

**Non-regression is the load-bearing test.** MJX rollouts are not reproducible across
processes (CLAUDE.md), so prove it on a **pure-CPU stub env**, in two git worktrees, the
way the `intrinsic_coef` no-op was verified:

1. Extend the StubEnv in `algorithms/tests/test_feudal_seams.py` (or add
   `algorithms/tests/test_smax_seams.py`) with a discrete stub exposing `avail_actions`,
   `global_state` and `info["active"]`. New tests:
   - all-ones mask + concat-`global_state` path is **bit-identical** to the pre-change
     code (this is the MJX-arm safety proof);
   - a masked-out action is never sampled and never survives `argmax` in `eval_fn`;
   - **the PPO ratio is exactly 1 before any update** with masks live — the existing
     seam-test idiom, and what catches a sampling/update mask mismatch;
   - `global_state` hook absent ⇒ falls back to `obs.reshape`, dims agree;
   - `active_mask` zeros a dead agent out of the policy loss.
   ```
   uv run pytest algorithms/tests/ -q
   ```
2. Manager self-checks still pass: `uv run python -m algorithms.feudal_mappo_jax.manager`
3. MJX smoke, both stacks, a few updates each — confirm losses unchanged vs `master`:
   ```
   uv run python train.py algorithm=mappo_jax env=mjx_16a_4o model=mlp trial_id=0 \
       params.n_total_steps=65536
   ```
4. SMAX end-to-end, flat then feudal — train, then resume, then evaluate:
   ```
   uv run python train.py algorithm=mappo_jax env=smax_3m model=mlp trial_id=0
   uv run python train.py algorithm=mappo_jax env=smax_3m model=mlp trial_id=0 checkpoint=true
   uv run python train.py algorithm=mappo_jax env=smax_3m model=mlp trial_id=0 evaluate=true
   uv run python train.py algorithm=feudal_mappo_jax env=smax_3m model=feudal trial_id=0
   ```
   Sanity: illegal-action rate 0; win rate rises off ~0 on `3m`; `episode_count > 0` per
   rollout; `explained_variance` climbs.
5. `view()` writes a SMAXVisualizer gif:
   ```
   uv run python train.py algorithm=mappo_jax env=smax_3m model=mlp trial_id=0 view=true
   ```
6. Update CLAUDE.md — a `## SMAX (JaxMARL)` section covering the adapter contract, the
   `step_env`/no-auto-reset requirement, the terminated-vs-truncated split, the three new
   optional env hooks (`avail_actions`, `global_state`, `info["active"]`) and their inert
   defaults, and the jaxmarl/JAX version shim.

---

## Files touched

| File | Change |
|---|---|
| `pyproject.toml` | ✅ done — `jaxmarl>=0.0.2` added |
| `environments/smax/smax_env.py` | **new** — `SMAXAdapter` + `SMAXState` |
| `algorithms/{mappo_jax,feudal_mappo_jax}/types.py` | `Transition.action_mask` |
| `algorithms/{mappo_jax,feudal_mappo_jax}/trainer.py` | mask threading, global-state hook, keep final env state |
| `algorithms/{mappo_jax,feudal_mappo_jax}/mappo.py` | mask in `evaluate_action` minibatch path |
| `algorithms/{mappo_jax,feudal_mappo_jax}/run.py` | SMAX branch, error string, `view()` branch |
| `conf/env/smax_3m.yaml`, `conf/env/smax_5m_vs_6m.yaml` | **new** |
| `algorithms/tests/test_smax_seams.py` | **new** |
| `CLAUDE.md` | new SMAX section |

`algorithms/{mappo_jax,feudal_mappo_jax}/network.py` needs **no change** — the masking
path is already there and unused.
