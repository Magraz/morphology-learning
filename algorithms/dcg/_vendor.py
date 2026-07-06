"""Make the vendored PyMARL DCG source importable.

The code under ``algorithms/dcg/src`` uses ``src``-root absolute imports
(``from modules.agents import REGISTRY``, ``from components... import ...``),
exactly as PyMARL expects when run with ``src`` on ``sys.path``. Importing this
module once puts that directory on the path so the adapter can do
``from controllers.dcg_controller import DeepCoordinationGraphMAC`` etc.

The top-level package names this exposes (``controllers``, ``learners``,
``components``, ``modules``) do not collide with anything at the repo root, so
the insertion is safe. We deliberately never import the ``envs`` / ``runners``
packages, which pull in SMAC/StarCraft and PyMARL's multiprocessing harness.
"""

import os
import sys

_SRC = os.path.join(os.path.dirname(__file__), "src")

if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
