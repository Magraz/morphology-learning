"""POI models. ``StaticPOIs`` is the fixed, config-defined layout."""

import warnings

import jax.numpy as jnp
import numpy as np

from environments.rover.components import POIModel
from environments.rover.state import POIState

_ENTRY_KEYS = {"position", "coupling", "radius", "value", "rate"}


class StaticPOIs(POIModel):
    """A fixed POI layout, identical in every episode.

    Args:
        entries: one mapping per POI with ``position: [x, y]`` and ``coupling``
            (required) and optional ``radius`` / ``value`` / ``rate``, which
            override the defaults below for that POI.
        radius: observation radius (inclusive) a rover must be within.
        value: total reward the POI can pay in one episode.
        rate: reward paid per step while its coupling is met; the POI is used
            up after ``ceil(value / rate)`` coupled steps.
    """

    def __init__(self, entries, radius=1.0, value=1.0, rate=0.05):
        entries = [dict(e) for e in entries]
        if not entries:
            raise ValueError("StaticPOIs needs at least one POI entry")
        rows = []
        for k, entry in enumerate(entries):
            unknown = set(entry) - _ENTRY_KEYS
            if unknown:
                raise ValueError(
                    f"POI {k}: unknown keys {sorted(unknown)}; "
                    f"allowed {sorted(_ENTRY_KEYS)}"
                )
            missing = {"position", "coupling"} - set(entry)
            if missing:
                raise ValueError(f"POI {k}: missing required keys {sorted(missing)}")
            coupling = entry["coupling"]
            if int(coupling) != coupling or int(coupling) < 1:
                raise ValueError(f"POI {k}: coupling must be an integer >= 1, got {coupling}")
            rows.append(
                (
                    entry["position"],
                    int(coupling),
                    float(entry.get("radius", radius)),
                    float(entry.get("value", value)),
                    float(entry.get("rate", rate)),
                )
            )
        position, coupling, radius_, value_, rate_ = zip(*rows)
        self.position = np.asarray(position, dtype=np.float32)
        if self.position.shape != (len(rows), 2):
            raise ValueError(f"every POI position must be [x, y], got {position}")
        self.coupling = np.asarray(coupling, dtype=np.int32)
        self.radius = np.asarray(radius_, dtype=np.float32)
        self.value = np.asarray(value_, dtype=np.float32)
        self.rate = np.asarray(rate_, dtype=np.float32)
        for name in ("radius", "value", "rate"):
            arr = getattr(self, name)
            if np.any(arr <= 0):
                raise ValueError(f"POI {name} must be > 0, got {arr.tolist()}")

    def init(self, key, env):
        return POIState(
            position=jnp.asarray(self.position),
            coupling=jnp.asarray(self.coupling),
            radius=jnp.asarray(self.radius),
            value=jnp.asarray(self.value),
            rate=jnp.asarray(self.rate),
            remaining=jnp.asarray(self.value),
        )

    def validate(self, env):
        width, height = env.world_size
        x, y = self.position[:, 0], self.position[:, 1]
        outside = (x < 0) | (x > width) | (y < 0) | (y > height)
        if np.any(outside):
            raise ValueError(
                f"POIs {np.flatnonzero(outside).tolist()} lie outside the "
                f"{width} x {height} arena"
            )
        unreachable = self.coupling > env.n_agents
        if np.any(unreachable):
            warnings.warn(
                f"POIs {np.flatnonzero(unreachable).tolist()} need more rovers "
                f"({self.coupling[unreachable].tolist()}) than the env has "
                f"({env.n_agents}); they can never pay.",
                stacklevel=2,
            )
