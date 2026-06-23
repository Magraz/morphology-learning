"""Lightweight, torch>=2 compatible stand-in for the subset of the ``dhg``
(DeepHypergraph) API this project actually uses.

The upstream ``dhg`` package pins ``torch<2``. The only thing the runtime
consumes from it is a thin container that turns ``(num_v, edge_list)`` into the
sparse incidence matrices ``H`` / ``H_T`` — the HGNN smoothing math itself is
reimplemented in :mod:`hypergraphs.hgnn_conv_layer`. This module reproduces that
surface so ``dhg`` can be dropped entirely.

Implemented surface (everything the runtime + tests touch):
  - ``Hypergraph(num_v, e_list, device=...)`` with ``.H``, ``.H_T``, ``.num_e``,
    ``.num_v``, ``.device``, ``.to(device)``, ``.e``, ``.draw(...)``
  - ``random.hypergraph_Gnm`` / ``random.graph_Gnm`` (demo / test helpers)

Use as a drop-in replacement::

    import hypergraphs.hg_compat as dhg

Semantics are matched against ``dhg`` 0.9.x:
  - ``H`` has shape ``(num_v, num_e)``, dtype float32, unit entries.
  - Identical hyperedges (order-independent) are merged into one, so ``num_e``
    counts *unique* edges.
  - Duplicate vertices *within* an edge accumulate (coalesced sum), matching
    dhg's sparse construction.
Edge/column ordering is irrelevant to every consumer (HGNN smoothing is
``H Hᵀ``; structural entropy is permutation-invariant over edges), so it is not
guaranteed to match dhg's ordering.
"""

from __future__ import annotations

import random as _pyrandom
from typing import Sequence

import torch

__all__ = ["Hypergraph", "random"]


class Hypergraph:
    """Minimal hypergraph container exposing dhg's incidence-matrix interface."""

    def __init__(self, num_v: int, e_list: Sequence[Sequence[int]] | None = None,
                 device: str | torch.device = "cpu", **_ignored):
        self.num_v = int(num_v)
        self._device = torch.device(device)

        # Normalize edges: drop empties and merge identical hyperedges
        # (order-independent), preserving first-appearance order.
        edges: list[tuple[int, ...]] = []
        seen: set[tuple[int, ...]] = set()
        for e in (e_list or []):
            e = tuple(int(v) for v in e)
            if not e:
                continue
            key = tuple(sorted(e))
            if key in seen:
                continue
            seen.add(key)
            edges.append(e)
        self._e_list = edges
        self._build()

    def _build(self) -> None:
        num_e = len(self._e_list)
        if num_e == 0:
            idx = torch.zeros((2, 0), dtype=torch.long)
            vals = torch.zeros((0,), dtype=torch.float32)
        else:
            rows: list[int] = []
            cols: list[int] = []
            for j, e in enumerate(self._e_list):
                for v in e:
                    rows.append(v)
                    cols.append(j)
            idx = torch.tensor([rows, cols], dtype=torch.long)
            vals = torch.ones(len(rows), dtype=torch.float32)
        H = torch.sparse_coo_tensor(idx, vals, (self.num_v, num_e)).coalesce()
        self._num_e = num_e
        self._H = H.to(self._device)
        self._H_T = self._H.transpose(0, 1).coalesce()

    # --- dhg.Hypergraph attribute surface -------------------------------
    @property
    def num_e(self) -> int:
        return self._num_e

    @property
    def H(self) -> torch.Tensor:
        return self._H

    @property
    def H_T(self) -> torch.Tensor:
        return self._H_T

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def e(self):
        """``(edge_list, weights)`` like ``dhg.Hypergraph.e`` (unit weights)."""
        return list(self._e_list), [1.0] * self._num_e

    def to(self, device: str | torch.device) -> "Hypergraph":
        self._device = torch.device(device)
        self._H = self._H.to(self._device)
        self._H_T = self._H_T.to(self._device)
        return self

    def __repr__(self) -> str:
        return (f"Hypergraph(num_v={self.num_v}, num_e={self._num_e}, "
                f"device={self._device})")

    # --- visualization (best-effort; dhg raises ValueError when empty) --
    def draw(self, v_label=None, v_color="gray", **_kwargs):
        """Render to the current pyplot figure (callers grab ``plt.gcf()``).

        Not a pixel-faithful copy of dhg's renderer — nodes sit on a fixed
        circular layout and hyperedges are drawn as translucent blobs (size>=3),
        lines (size 2) or rings (size 1). Mirrors dhg by raising ``ValueError``
        on an empty hypergraph (the renderer catches that and skips the panel).
        """
        import numpy as np
        import matplotlib.pyplot as plt

        if self._num_e == 0:
            raise ValueError("cannot draw a hypergraph with no hyperedges")

        n = max(self.num_v, 1)
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        pos = np.stack([np.cos(angles), np.sin(angles)], axis=1)

        fig, ax = plt.subplots()
        cmap = plt.get_cmap("tab10")
        for j, e in enumerate(self._e_list):
            pts = pos[list(e)]
            color = cmap(j % 10)
            if len(e) == 1:
                ax.scatter(pts[0, 0], pts[0, 1], s=700,
                           facecolors="none", edgecolors=color, linewidths=2)
            elif len(e) == 2:
                ax.plot(pts[:, 0], pts[:, 1], color=color, lw=2.5, alpha=0.6)
            else:
                center = pts.mean(axis=0)
                order = np.argsort(np.arctan2(pts[:, 1] - center[1],
                                              pts[:, 0] - center[0]))
                ax.add_patch(plt.Polygon(pts[order], closed=True,
                                         alpha=0.25, color=color))
        ax.scatter(pos[:, 0], pos[:, 1], s=220, color=v_color, zorder=3)
        if v_label is not None:
            for i, (x, y) in enumerate(pos):
                ax.text(x, y, str(v_label[i]), ha="center", va="center",
                        zorder=4, fontsize=8)
        ax.set_aspect("equal")
        ax.axis("off")
        return ax


class _Random:
    """Stand-in for ``dhg.random`` (only the demo/test helpers we use)."""

    @staticmethod
    def hypergraph_Gnm(num_v: int, num_e: int,
                       device: str | torch.device = "cpu") -> Hypergraph:
        """Random hypergraph with ``num_e`` unique hyperedges of size 2..4."""
        edges: list[tuple[int, ...]] = []
        seen: set[tuple[int, ...]] = set()
        max_k = min(4, num_v)
        guard = 0
        while len(edges) < num_e and guard < num_e * 50:
            guard += 1
            k = _pyrandom.randint(2, max(2, max_k))
            e = tuple(sorted(_pyrandom.sample(range(num_v), k)))
            if e in seen:
                continue
            seen.add(e)
            edges.append(e)
        return Hypergraph(num_v, edges, device=device)

    @staticmethod
    def graph_Gnm(num_v: int, num_e: int,
                  device: str | torch.device = "cpu") -> Hypergraph:
        """Random simple graph as a hypergraph of distinct 2-edges."""
        edges: list[tuple[int, int]] = []
        seen: set[tuple[int, int]] = set()
        guard = 0
        while len(edges) < num_e and guard < num_e * 50:
            guard += 1
            a, b = _pyrandom.sample(range(num_v), 2)
            e = (min(a, b), max(a, b))
            if e in seen:
                continue
            seen.add(e)
            edges.append(e)
        return Hypergraph(num_v, edges, device=device)


random = _Random()
