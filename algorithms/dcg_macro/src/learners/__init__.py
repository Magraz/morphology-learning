REGISTRY = {}

from .q_learner import QLearner
REGISTRY["q_learner"] = QLearner

from .dcg_learner import DCGLearner
REGISTRY["dcg_learner"] = DCGLearner

# NOTE: coma_learner / qtran_learner are vendored but not registered here (out
# of scope for the framework adapter). Re-add them if they are ever ported.
