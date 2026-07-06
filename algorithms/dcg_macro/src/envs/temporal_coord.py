'''
Temporal Coordination Game -- a didactic matrix-game-style toy for studying whether
a (deep) coordination-graph *payoff* needs access to the temporal *phase* (remaining
macro-action duration) of its neighbours in order to coordinate in time.

It is the timing analogue of the relative-overgeneralization stag-hunt used by
Boehmer et al. (2020): coordination fails not because of partial observability but
because the optimal *joint* action is risky for a greedy per-agent learner. Here the
coordination that must be discovered is *temporal* -- agents must align (or anti-align)
the windows during which they are "active" (executing a committed macro-action).

MECHANIC
Each agent is FREE, BUSY, or COOLING.
  - A FREE agent may `idle` (stay free) or `go` (commit to a macro-action).
  - `go` draws a *random* duration d ~ U{d_min..d_max}; the agent is BUSY/active for d
    steps, then is forced to COOL (cannot go) for `cooldown` steps, then FREE again.
The cooldown is essential: it makes "just stay active forever" (or "one agent covers
everything") impossible, so agents genuinely have to *hand off in time*. A clean handoff
requires knowing *when* a neighbour will free up -- i.e. its remaining duration -- which
is the whole point of the phase / duration term.

Crucially the drawn duration is never told to the agent directly -- it is only visible
through the *phase* feature (remaining steps, normalised) and cannot be reconstructed
from the action history, because it is random. This is what makes the "does the payoff
need the duration term?" question non-circular.

REWARD (collaborative, per step; `mode` selects the coordination pattern)
Let A = number of agents active this step.
  - mode="relay" (anti-phase / turn-taking): exactly one agent active at a time.
        +1                      if A == 1                 (clean coverage)
        -collision_penalty      if A  > 1                 (temporal collision)
         0                      if A == 0                 (coverage gap -- wasted)
  - mode="sync"  (in-phase / rendezvous): agents should be active *together*.
        +1                      if A == n_agents          (fully aligned)
        -misalign_penalty       if 0 < A < n_agents       (un-aligned effort)
         0                      if A == 0
Both shapes create a relative-overgeneralization-in-time pressure: the miscoordination
penalty makes independent/greedy learners shy away from the risky-but-optimal `go`.

THE ABLATION (the whole point)
`observe_partner_phase` is the "duration term".
  - True  -> an agent sees each neighbour's remaining duration (and cooldown); the CG
             payoff can time the handoff.
  - False -> an agent sees only *that* a neighbour is busy, not for how much longer
             (random durations => the busy flag alone is uninformative). This mimics a
             macro-controller whose neighbour hidden-state is stale between the
             neighbour's own decision points -- a payoff *without* the duration term.
The headline result to look for: with a coordination graph (`cg_edges=full`) AND
`observe_partner_phase=True`, DCG solves the timing task; dropping either the edges
(VDN) or the partner phase collapses it.

INTERFACE
Implements the PyMARL ``MultiAgentEnv`` surface, so it runs on the unmodified DCG
controller/learner. No spatial/physics state -- purely the temporal commitment bookkeeping.
'''

import numpy as np

from envs.multiagentenv import MultiAgentEnv
from utils.dict2namedtuple import convert


# Discrete action ids
IDLE = 0
GO = 1


class TemporalCoordination(MultiAgentEnv):
    """ N-agent temporal coordination toy; see module docstring. """

    def __init__(self, batch_size=None, **kwargs):
        # Unpack sacred-style args into an attribute-access namedtuple
        args = kwargs
        if isinstance(args, dict):
            args = convert(args)
        self.args = args

        self.n_agents = getattr(args, "n_agents", 2)
        self.episode_limit = getattr(args, "episode_limit", 32)

        # Macro-action duration range (inclusive). Randomness forces asynchrony and
        # makes the phase feature the *only* source of remaining-duration information.
        self.d_min = getattr(args, "duration_min", 2)
        self.d_max = getattr(args, "duration_max", 6)
        assert 1 <= self.d_min <= self.d_max, "require 1 <= duration_min <= duration_max"

        # Forced rest after each macro; blocks the degenerate "always active" optima and
        # forces genuine, timed hand-offs between agents.
        self.cooldown = getattr(args, "cooldown", self.d_max)
        assert self.cooldown >= 1, "cooldown must be >= 1 (else timing is trivial)"

        # Coordination pattern and its miscoordination penalties
        self.mode = getattr(args, "mode", "relay")
        assert self.mode in ("sync", "relay"), "mode must be 'sync' or 'relay'"
        self.misalign_penalty = getattr(args, "misalign_penalty", 0.5)   # sync mode
        self.collision_penalty = getattr(args, "collision_penalty", 1.0)  # relay mode

        # Observation ablation knobs
        self.observe_own_phase = getattr(args, "observe_own_phase", True)
        self.observe_partner_phase = getattr(args, "observe_partner_phase", True)  # the "duration term"

        self.n_actions = 2  # {idle, go}

        # RNG (seed injected by PyMARL run.py via env_args when available)
        self.rng = np.random.RandomState(getattr(args, "seed", None))

        # Per-agent state: remaining macro duration (0 == not busy) and remaining cooldown.
        self.phase = np.zeros(self.n_agents, dtype=np.int64)
        self.cool = np.zeros(self.n_agents, dtype=np.int64)
        self.t = 0
        # Running diagnostics for the episode (logged through env_info at termination)
        self._good_steps = 0    # steps that scored the +1 (fully coordinated)
        self._bad_steps = 0     # steps that scored a miscoordination penalty

    # ===================== Core dynamics ============================================

    def reset(self):
        """ Resets to an all-free state and returns (obs, state). """
        self.phase[:] = 0
        self.cool[:] = 0
        self.t = 0
        self._good_steps = 0
        self._bad_steps = 0
        return self.get_obs(), self.get_state()

    def step(self, actions):
        """ Applies one step of the game. Returns (reward, terminated, info). """
        actions = np.asarray(actions).reshape(-1).astype(np.int64)

        # Only FREE agents (not busy, not cooling) may commit to a fresh macro.
        can_go = (self.phase == 0) & (self.cool == 0)
        for i in range(self.n_agents):
            if can_go[i] and actions[i] == GO:
                self.phase[i] = self.rng.randint(self.d_min, self.d_max + 1)

        # Active set is evaluated *after* new commitments this step.
        was_active = self.phase > 0
        active_count = int(was_active.sum())
        reward = self._reward(active_count)

        # Consume one step of every running macro, then advance cooldowns.
        self.phase = np.maximum(self.phase - 1, 0)
        just_finished = was_active & (self.phase == 0)
        free = self.phase == 0
        # decrement existing cooldowns for agents that are free (not busy)
        self.cool = np.where(free, np.maximum(self.cool - 1, 0), self.cool)
        # agents that just finished a macro start a fresh cooldown
        self.cool[just_finished] = self.cooldown

        self.t += 1
        terminated = self.t >= self.episode_limit
        info = {}
        if terminated:
            # Flag limit-termination so the runner treats it as truncation (bootstraps),
            # and expose episode-level coordination diagnostics for logging.
            info["episode_limit"] = True
            denom = max(self.t, 1)
            info["coordination_rate"] = self._good_steps / denom
            info["miscoordination_rate"] = self._bad_steps / denom
        return reward, terminated, info

    def _reward(self, active_count):
        """ Collaborative per-step reward; updates good/bad step diagnostics. """
        if self.mode == "sync":
            if active_count == self.n_agents:
                self._good_steps += 1
                return 1.0
            if active_count > 0:
                self._bad_steps += 1
                return -self.misalign_penalty
            return 0.0
        else:  # relay
            if active_count == 1:
                self._good_steps += 1
                return 1.0
            if active_count > 1:
                self._bad_steps += 1
                return -self.collision_penalty
            return 0.0

    # ===================== Observations / state =====================================

    def _per_agent_temporal(self, j):
        """ (phase, cooldown) of agent j, normalised. """
        return [self.phase[j] / self.d_max, self.cool[j] / self.cooldown]

    def _agent_feature_size(self):
        """ Size of the per-agent block: self + partners, respecting ablation flags. """
        self_size = 1 + (2 if self.observe_own_phase else 0)             # busy [+ phase, cool]
        partner_size = (self.n_agents - 1) * (1 + (2 if self.observe_partner_phase else 0))
        return self_size + partner_size

    def get_obs_agent(self, agent_id):
        """ Egocentric obs: own (busy[, phase, cool]) then each partner (busy[, phase, cool]). """
        feats = [1.0 if self.phase[agent_id] > 0 else 0.0]
        if self.observe_own_phase:
            feats.extend(self._per_agent_temporal(agent_id))
        for j in range(self.n_agents):
            if j == agent_id:
                continue
            feats.append(1.0 if self.phase[j] > 0 else 0.0)
            if self.observe_partner_phase:
                feats.extend(self._per_agent_temporal(j))
        return np.asarray(feats, dtype=np.float32)

    def get_obs(self):
        return [self.get_obs_agent(i) for i in range(self.n_agents)]

    def get_obs_size(self):
        return self._agent_feature_size()

    def get_state(self):
        """ Full state (for the centralised bias / duelling option): busy, phase, cool per agent. """
        state = np.zeros(3 * self.n_agents, dtype=np.float32)
        for j in range(self.n_agents):
            state[3 * j] = 1.0 if self.phase[j] > 0 else 0.0
            state[3 * j + 1], state[3 * j + 2] = self._per_agent_temporal(j)
        return state

    def get_state_size(self):
        return 3 * self.n_agents

    # ===================== Available actions ========================================

    def get_avail_agent_actions(self, agent_id):
        """ Only a FREE agent (not busy, not cooling) may `go`; everyone may `idle`. """
        if self.phase[agent_id] == 0 and self.cool[agent_id] == 0:
            return [1, 1]
        return [1, 0]

    def get_avail_actions(self):
        return [self.get_avail_agent_actions(i) for i in range(self.n_agents)]

    def get_total_actions(self):
        return self.n_actions

    # ===================== Boilerplate ==============================================

    def render(self):
        def glyph(i):
            if self.phase[i] > 0:
                return "#" * int(self.phase[i])           # busy/active
            if self.cool[i] > 0:
                return "~" * int(self.cool[i])            # cooling
            return "."                                    # free
        bar = " ".join(f"A{i}:{glyph(i)}" for i in range(self.n_agents))
        print(f"t={self.t:3d} | {bar}")

    def close(self):
        pass

    def seed(self, seed=None):
        self.rng = np.random.RandomState(seed)
        return seed

    def save_replay(self):
        pass

    def get_env_info(self):
        return {"state_shape": self.get_state_size(),
                "obs_shape": self.get_obs_size(),
                "n_actions": self.get_total_actions(),
                "n_agents": self.n_agents,
                "episode_limit": self.episode_limit}
