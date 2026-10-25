"""Learned fork scheduler that may fork at any step of a main window, from a
per-stream budget (`intervention_scheduler: anytime | anytime_random`).

Plan: `/home/magraz/.claude/plans/i-want-to-create-breezy-wind.md` (2026-10-25).
The adaptive bandit (`intervention_bandit.py`) decided only whether to fork at a
manager decision, around a random agent with random recruits, for one window.
Here, at every primitive step of a main window (`anytime_collection.py`), a
stochastic policy picks:

* skip, or a fork around focal agent i: one categorical over `{skip, 1..N}`;
* for a fork, the teammates to teleport next to agent i: a product of
  Bernoullis conditioned on at least one recruit;
* and its length of k goal windows, `1 <= k <= kmax`. `kmax` is the stream's
  remaining budget, capped by its free slots and by the windows left before the
  episode's time limit.

Its input is every agent's manager actor input, the pairwise goal-state offsets
`pos_j - pos_i` (to choose recruits), and `[budget_left / B, slots_left /
(2 n_windows), t / max_steps, step / goal_horizon]`. It is a training-time
device only: execution and evaluation never read it.

The reward of a placed fork is its paired gain (`span_return`, computed by the
collector): the fork's discounted raw team reward minus the main rollout's over
the same span from the same state, plus the difference of their manager values
at the span's end. A skip and a failed placement earn 0. The scheduler is
trained by PPO with GAE over each stream's decisions within a rollout
(`decision_gae`, `fit_scheduler`), so its value `V(context, budget)` can learn
what unspent budget is worth. That is 0 at the rollout's end, where unspent
budget becomes ordinary main windows. `intervention_scheduler_gamma: 0` makes it
a one-step bandit. `anytime_random` samples from the same network frozen at its
initialization: focal agent uniform, each recruit with probability 1/2 (then
conditioned on one), the length uniform, and the launch rate set by
`intervention_scheduler_init_forks`.
"""

import math
from typing import NamedTuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState

from algorithms.mappo_jax.mappo import compute_gae, masked_mean_std

ANYTIME_SCHEDULERS = ("anytime", "anytime_random")
# [budget_left / B, slots_left / (2 n_windows), t / max_steps, step / goal_horizon]
N_GLOBAL = 4

# The scheduler's PPO, in one place.
N_EPOCHS = 4
N_MINIBATCHES = 4
CLIP = 0.2
VALUE_COEF = 0.5
GRAD_CLIP = 0.5
# Logit of a masked option (as `mappo_jax.network` masks actions): finite, so no
# probability or entropy term becomes NaN.
MASKED = -1e9

_gae = jax.jit(compute_gae, static_argnums=(4, 5))


def fork_budget(config) -> int:
    """B: fork windows per main stream per update (None = n_windows)."""
    if config.intervention_fork_budget is None:
        return config.worker.n_steps // config.goal_horizon
    return int(config.intervention_fork_budget)


def stream_slots(config) -> int:
    """Window slots per main stream per update, main and fork windows together."""
    return 2 * (config.worker.n_steps // config.goal_horizon)


def skip_bias(config, n_agents: int) -> float:
    """The skip logit's initial bias. With every other logit 0 a decision then
    launches with probability `p0 = init_forks / (2 n_steps)`: about
    `init_forks` launches over the `2 n_steps` steps a stream decides at when
    it never forks."""
    p0 = config.intervention_scheduler_init_forks / (2 * config.worker.n_steps)
    return math.log(n_agents * (1.0 - p0) / p0)


class Heads(NamedTuple):
    """The scheduler's outputs for one decision (any leading batch dims)."""

    skip: jax.Array  # (...,)
    focal: jax.Array  # (..., N)
    recruit: jax.Array  # (..., N, N): [i, j] the logit of recruiting j for focal i
    duration: jax.Array  # (..., N, B): per focal agent, k = 1..B
    value: jax.Array  # (...,)


class Action(NamedTuple):
    choice: jax.Array  # int: 0 skip, i + 1 a fork around agent i
    recruit: jax.Array  # (N,) bool, never the focal agent; at least one on a fork
    duration: jax.Array  # int: the fork's windows minus 1


class SchedulerNet(nn.Module):
    """Shared over agents and equivariant to their order: every head reads the
    per-agent encodings `e_i = MLP([x_i, g])` and their mean. Every output layer
    starts at zero, so at initialization the focal agent, each recruit (p = 1/2)
    and the length are uniform, and the launch rate is set by `skip_bias`."""

    max_duration: int
    hidden: int = 32
    skip_bias: float = 0.0

    @nn.compact
    def __call__(self, x, pair, glob):
        zeros = nn.initializers.zeros
        g = jnp.broadcast_to(glob[..., None, :], x.shape[:-1] + glob.shape[-1:])
        e = jnp.tanh(nn.Dense(self.hidden, name="encode_0")(jnp.concatenate([x, g], -1)))
        e = jnp.tanh(nn.Dense(self.hidden, name="encode_1")(e))  # (..., N, h)
        n = e.shape[-2]
        pooled = e.mean(axis=-2)
        team = jnp.concatenate([pooled, glob], axis=-1)
        skip = nn.Dense(
            1, kernel_init=zeros, bias_init=nn.initializers.constant(self.skip_bias),
            name="skip",
        )(team)[..., 0]
        focal = nn.Dense(1, kernel_init=zeros, name="focal")(e)[..., 0]
        e_i = jnp.broadcast_to(e[..., :, None, :], e.shape[:-2] + (n, n, e.shape[-1]))
        e_j = jnp.broadcast_to(e[..., None, :, :], e_i.shape)
        r = jnp.tanh(
            nn.Dense(self.hidden, name="recruit_0")(jnp.concatenate([e_i, e_j, pair], -1))
        )
        recruit = nn.Dense(1, kernel_init=zeros, name="recruit_1")(r)[..., 0]
        duration = nn.Dense(self.max_duration, kernel_init=zeros, name="duration")(
            jnp.concatenate([e, jnp.broadcast_to(pooled[..., None, :], e.shape)], -1)
        )
        v = jnp.tanh(nn.Dense(self.hidden, name="value_0")(team))
        value = nn.Dense(1, kernel_init=zeros, name="value_1")(v)[..., 0]
        return Heads(skip=skip, focal=focal, recruit=recruit, duration=duration, value=value)


def create_scheduler_state(
    rng, actor_dim: int, n_agents: int, max_duration: int, hidden: int, lr: float,
    bias: float,
) -> TrainState:
    net = SchedulerNet(max_duration=max_duration, hidden=hidden, skip_bias=bias)
    params = net.init(
        rng,
        jnp.zeros((n_agents, actor_dim)),
        jnp.zeros((n_agents, n_agents, 2)),
        jnp.zeros((N_GLOBAL,)),
    )
    tx = optax.chain(optax.clip_by_global_norm(GRAD_CLIP), optax.adam(lr))
    return TrainState.create(apply_fn=net.apply, params=params, tx=tx)


# ------------------------------------------------------------- the policy


def _choice_logits(h: Heads):
    return jnp.concatenate([h.skip[..., None], h.focal], axis=-1)


def _duration_logits(h: Heads, kmax):
    """`(..., N, B)` length logits per focal agent, masked to `k <= kmax`. At
    least one length stays allowed, so a non-decision's terms stay finite."""
    allowed = jnp.arange(h.duration.shape[-1]) < jnp.maximum(kmax, 1)[..., None, None]
    return jnp.where(allowed, h.duration, MASKED)


def _recruit_terms(h: Heads):
    """Per focal agent i: `log p_j` and `log(1 - p_j)` of recruiting each
    teammate j (0 at j = i, who is never recruited), the log probability that
    the unconditioned product recruits nobody, `s_i = sum_j log(1 - p_j)`, and
    the `(N, N)` teammate mask."""
    n = h.recruit.shape[-1]
    others = ~jnp.eye(n, dtype=bool)
    log_p = jnp.where(others, jax.nn.log_sigmoid(h.recruit), 0.0)
    log_q = jnp.where(others, jax.nn.log_sigmoid(-h.recruit), 0.0)
    return log_p, log_q, log_q.sum(-1), others


def _log_not_empty(s):
    """`log(1 - exp(s))`: the log probability of at least one recruit."""
    return jnp.log(jnp.maximum(-jnp.expm1(s), 1e-30))


def _focal_row(x, focal):
    """`x[..., focal, :]` for a per-focal `(..., N, D)` array."""
    return jnp.take_along_axis(x, focal[..., None, None], axis=-2)[..., 0, :]


def log_prob(h: Heads, action: Action, kmax):
    """`log pi(action | context)`, any leading batch dims: the choice's term, plus
    on a fork the recruits' (conditioned on at least one) and the length's."""
    choice_lp = jax.nn.log_softmax(_choice_logits(h))
    lp = jnp.take_along_axis(choice_lp, action.choice[..., None], -1)[..., 0]
    focal = jnp.maximum(action.choice - 1, 0)
    log_p, log_q, s, _ = _recruit_terms(h)
    rec = action.recruit.astype(jnp.float32)
    rec_lp = (rec * _focal_row(log_p, focal) + (1 - rec) * _focal_row(log_q, focal)).sum(
        -1
    ) - _log_not_empty(jnp.take_along_axis(s, focal[..., None], -1)[..., 0])
    dur_lp = jax.nn.log_softmax(_focal_row(_duration_logits(h, kmax), focal))
    dur_lp = jnp.take_along_axis(dur_lp, action.duration[..., None], -1)[..., 0]
    return lp + jnp.where(action.choice > 0, rec_lp + dur_lp, 0.0)


def entropy(h: Heads, kmax):
    """Exact entropy of the whole decision:
    `H(choice) + sum_i pi(i) [H(recruits | i) + H(length | i)]`.

    The recruits' entropy conditioned on at least one is
    `(H_u + P0 log P0) / (1 - P0) + log(1 - P0)`, with `H_u` the unconditioned
    product's entropy and `P0 = exp(s)` its probability of nobody."""
    choice_lp = jax.nn.log_softmax(_choice_logits(h))
    pi = jnp.exp(choice_lp)
    h_choice = -(pi * choice_lp).sum(-1)
    log_p, log_q, s, others = _recruit_terms(h)
    p = jnp.where(others, jnp.exp(log_p), 0.0)
    q = jnp.where(others, jnp.exp(log_q), 0.0)
    h_u = -(p * log_p + q * log_q).sum(-1)  # (..., N)
    log_z = _log_not_empty(s)
    h_rec = (h_u + jnp.exp(s) * s) / jnp.exp(log_z) + log_z
    dur_lp = jax.nn.log_softmax(_duration_logits(h, kmax))
    h_dur = -(jnp.exp(dur_lp) * dur_lp).sum(-1)  # (..., N)
    return h_choice + (pi[..., 1:] * (h_rec + h_dur)).sum(-1)


def sample(h: Heads, kmax, key):
    """One decision from unbatched heads. Returns `(Action, log_prob)`.

    The recruit set is drawn exactly from the product of Bernoullis conditioned
    on at least one recruit: first its smallest member m, with `P(m = j)`
    proportional to `p_j prod_{l < j} (1 - p_l)`, then every later teammate
    independently. Recruits and length are drawn for every decision, also a
    skip, which never uses them."""
    k_choice, k_first, k_rest, k_dur = jax.random.split(key, 4)
    choice = jax.random.categorical(k_choice, _choice_logits(h))
    focal = jnp.maximum(choice - 1, 0)
    log_p, log_q, _, others = _recruit_terms(h)
    lp_row, lq_row, mates = log_p[focal], log_q[focal], others[focal]
    before = jnp.cumsum(lq_row) - lq_row
    first = jax.random.categorical(k_first, jnp.where(mates, lp_row + before, MASKED))
    idx = jnp.arange(lp_row.shape[0])
    rest = jax.random.uniform(k_rest, lp_row.shape) < jnp.exp(lp_row)
    recruit = (idx == first) | ((idx > first) & mates & rest)
    duration = jax.random.categorical(k_dur, _duration_logits(h, kmax)[focal])
    action = Action(choice=choice, recruit=recruit, duration=duration)
    return action, log_prob(h, action, kmax)


# ------------------------------------------------------------- the reward


def continuation(active, ended):
    """`(W * H + 1,)` float over a run of W windows of H steps: entry q is 1 when
    state q belongs to the same episode as state q - 1 (entry 0, the run's first
    state, is 1). `active` `(W, H)`: each step's pre-step live flag; `ended`
    `(W,)`: the episode ended in that window (the next window starts a new
    episode, or the run stops). Entry `W * H` is the state after the run."""
    n_windows, horizon = active.shape
    flat = np.asarray(active, np.float64).reshape(-1)
    q = np.arange(1, n_windows * horizon + 1)
    inside = q % horizon != 0
    within = flat[np.minimum(q, n_windows * horizon - 1)]
    across = 1.0 - np.asarray(ended, np.float64)[q // horizon - 1]
    return np.concatenate([[1.0], np.where(inside, within, across)])


def span_return(reward, cont, value, gamma, start: int, length: int):
    """`(g, bootstrap)` of one side of a fork's comparison: the discounted raw
    reward over `length` steps from step `start`, cut at the episode's end
    (termination or time limit), and the discounted value of the state `length`
    steps later if the episode is still running there. `reward` `(T,)`; `cont`
    (`continuation`) and `value` (pre-step values, then the value after the run)
    `(T + 1,)`."""
    live = np.cumprod(np.concatenate([[1.0], cont[start + 1 : start + length + 1]]))
    disc = gamma ** np.arange(length + 1)
    g = float((disc[:length] * live[:length] * reward[start : start + length]).sum())
    return g, float(disc[length] * live[length] * value[start + length])


def decision_gae(reward, value, decided, gamma: float, lam: float):
    """GAE over each stream's decisions in time order (`mappo.compute_gae`).

    `reward`, `value`, `decided`: `(T, S)` over each stream's main steps, time
    major. Only the decided cells are steps of this process; the cells between
    two decisions are skipped. Each stream ends in a terminal of value 0 (unspent
    budget becomes main windows, the zero of the paired gain). Returns
    `(advantage, returns)`, `(T, S)`, 0 off the decisions."""
    decided = np.asarray(decided, bool)
    n_cells, n_streams = decided.shape
    order = np.argsort(~decided, axis=0, kind="stable")  # decisions first, in order
    count = decided.sum(axis=0)
    position = np.arange(n_cells)[:, None]
    real = position < count[None]
    terminal = position >= count[None] - 1
    r = np.take_along_axis(np.asarray(reward, np.float32), order, 0) * real
    v = np.take_along_axis(np.asarray(value, np.float32), order, 0) * real
    adv, _ = _gae(
        jnp.asarray(r), jnp.asarray(v), jnp.asarray(terminal, jnp.float32),
        jnp.zeros(n_streams), float(gamma), float(lam),
    )
    out = np.zeros((n_cells, n_streams), np.float32)
    np.put_along_axis(out, order, np.asarray(adv) * real, 0)
    return out, (out + np.asarray(value, np.float32)) * decided


# ------------------------------------------------------------- training


def fit_scheduler(state: TrainState, batch, rng, ent_coef: float):
    """`N_EPOCHS` PPO passes over the decided cells of a `SchedulerBatch`, in
    `N_MINIBATCHES` shuffled minibatches per epoch; the other cells carry weight
    0 and never move a parameter. Advantages are normalized over the decided
    cells. A minibatch without a decided cell takes no step (Adam would move on
    a zero gradient).

    Returns `(state, stats)`: mean policy / value loss, entropy and clip
    fraction over the steps taken, the number of steps, and the explained
    variance of the pre-update values on the returns."""
    flat = jax.tree.map(lambda x: x.reshape((-1,) + x.shape[3:]), batch)
    weight = flat.weight
    mean, std, _ = masked_mean_std(flat.advantage, weight)
    adv = jnp.where(weight > 0, (flat.advantage - mean) / (std + 1e-8), 0.0)
    _, ret_std, _ = masked_mean_std(flat.returns, weight)
    _, err_std, _ = masked_mean_std(flat.returns - flat.value, weight)
    value_ev = 1.0 - err_std**2 / jnp.maximum(ret_std**2, 1e-8)

    n_cells = weight.shape[0]
    size = -(-n_cells // N_MINIBATCHES)
    pad = size * N_MINIBATCHES - n_cells

    def loss_fn(params, idx, w):
        h = state.apply_fn(params, flat.x[idx], flat.pair[idx], flat.glob[idx])
        action = Action(flat.choice[idx], flat.recruit[idx], flat.duration[idx])
        kmax = flat.kmax[idx]
        lp = log_prob(h, action, kmax)
        ratio = jnp.exp(jnp.where(w > 0, lp - flat.log_prob[idx], 0.0))
        a = adv[idx]
        pg = -jnp.minimum(ratio * a, jnp.clip(ratio, 1 - CLIP, 1 + CLIP) * a)
        v_loss = 0.5 * (h.value - flat.returns[idx]) ** 2
        ent = entropy(h, kmax)
        n = jnp.maximum(w.sum(), 1.0)

        def mean_(x):
            return jnp.where(w > 0, x, 0.0).sum() / n

        loss = mean_(pg) + VALUE_COEF * mean_(v_loss) - ent_coef * mean_(ent)
        clipped = (jnp.abs(ratio - 1.0) > CLIP).astype(jnp.float32)
        return loss, jnp.stack([mean_(pg), mean_(v_loss), mean_(ent), mean_(clipped)])

    def epoch(carry, key):
        st, sums, steps = carry
        order = jnp.concatenate(
            [jax.random.permutation(key, n_cells), jnp.zeros((pad,), jnp.int32)]
        )
        valid = jnp.concatenate([jnp.ones(n_cells, bool), jnp.zeros(pad, bool)])

        def minibatch(carry, i):
            st, sums, steps = carry
            idx = jax.lax.dynamic_slice(order, (i * size,), (size,))
            ok = jax.lax.dynamic_slice(valid, (i * size,), (size,))
            w = weight[idx] * ok

            def step(st):
                (_, parts), grads = jax.value_and_grad(loss_fn, has_aux=True)(
                    st.params, idx, w
                )
                return st.apply_gradients(grads=grads), parts

            st, parts = jax.lax.cond(
                w.sum() > 0, step, lambda st: (st, jnp.zeros(4)), st
            )
            taken = w.sum() > 0
            return (st, sums + parts, steps + taken), None

        carry, _ = jax.lax.scan(minibatch, (st, sums, steps), jnp.arange(N_MINIBATCHES))
        return carry, None

    (state, sums, steps), _ = jax.lax.scan(
        epoch,
        (state, jnp.zeros(4), jnp.zeros((), jnp.int32)),
        jax.random.split(rng, N_EPOCHS),
    )
    means = sums / jnp.maximum(steps, 1)
    stats = {
        "scheduler_policy_loss": means[0],
        "scheduler_value_loss": means[1],
        "scheduler_entropy": means[2],
        "scheduler_clip_frac": means[3],
        "scheduler_steps": steps.astype(jnp.float32),
        "scheduler_value_explained_variance": value_ev,
    }
    return state, stats
