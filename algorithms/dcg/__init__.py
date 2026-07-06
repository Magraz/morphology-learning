"""DCG (Deep Coordination Graph) algorithm, adapted to the Runner framework.

The upstream PyMARL project lives under ``algorithms/dcg/src`` and is reused
as-is (controller, learner, episode buffer, message passing). The framework
adapter (``run.py`` / ``trainer.py`` / ``types.py`` / ``args_builder.py``)
wraps it so DCG launches through the Hydra ``train.py`` entry point like the
other algorithms.
"""
