"""Compatibility shim: restore the ``jax.tree_*`` aliases jaxmarl still calls.

jaxmarl (0.0.x) calls ``jax.tree_map`` / ``jax.tree_leaves`` / … , which JAX moved
under the ``jax.tree`` namespace and removed from the top level in 0.9.x. This repo
runs JAX 0.10.2, so those calls raise ``AttributeError`` at *runtime*.

The failure is easy to miss: ``import jaxmarl`` succeeds, and so does
``jaxmarl.make(...)`` and ``env.reset(key)``. Only a call that actually walks a pytree
— ``env.step_env(...)``, ``env.get_avail_actions(...)`` — trips it. So an import smoke
test is NOT sufficient evidence that the shim is unnecessary.

Import this module **before** importing ``jaxmarl`` anywhere::

    import environments.smax._compat  # noqa: F401  (must precede jaxmarl)
    import jaxmarl

Re-importing is free (Python caches it) and the patch is idempotent. Delete this file
if jaxmarl is ever upgraded to a release that uses the ``jax.tree`` namespace.
"""

import jax

# Every top-level alias jaxmarl reaches for that now lives under `jax.tree`.
_ALIASES = (
    "map",
    "leaves",
    "structure",
    "unflatten",
    "flatten",
    "transpose",
    "reduce",
    "all",
)

for _name in _ALIASES:
    _legacy = f"tree_{_name}"
    if not hasattr(jax, _legacy) and hasattr(jax.tree, _name):
        setattr(jax, _legacy, getattr(jax.tree, _name))

del _name, _legacy, _ALIASES
