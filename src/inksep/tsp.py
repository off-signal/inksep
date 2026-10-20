"""Travel-order optimization for dot export (pen-plotter pen-up travel).

Reorders the dots so consecutive hops are short. The path is open: it does
not loop back to the start. Everything here is deterministic; with
``budget_s=None`` the result is exactly reproducible, and the time budget
only truncates work, never returning a worse order than it was given.
"""

from __future__ import annotations

import time

import numpy as np
from numpy.typing import NDArray

__all__ = ["nearest_neighbour", "open_path_length", "solve_tsp", "two_opt"]


def open_path_length(points: NDArray[np.float64], order: NDArray[np.intp]) -> float:
    """Total length of the open polyline through ``points[order]``.

    No closing edge is added; an open plotting path starts and ends at its
    endpoints.
    """
    if len(order) < 2:
        return 0.0
    pts = points[order]
    return float(np.hypot(np.diff(pts[:, 0]), np.diff(pts[:, 1])).sum())


def nearest_neighbour(
    points: NDArray[np.float64], *, start: int = 0
) -> NDArray[np.intp]:
    """Greedy open-path construction: repeatedly visit the closest unvisited point.

    Deterministic given ``points`` and ``start``. Returns a permutation of
    ``range(len(points))``.
    """
    n = len(points)
    if n == 0:
        return np.empty(0, dtype=np.intp)
    if not 0 <= start < n:
        raise ValueError(f"start must be in [0, {n}), got {start}")

    order = np.empty(n, dtype=np.intp)
    visited = np.zeros(n, dtype=bool)
    order[0] = start
    visited[start] = True
    for k in range(1, n):
        last = points[order[k - 1]]
        d = np.hypot(points[:, 0] - last[0], points[:, 1] - last[1])
        d[visited] = np.inf
        nxt = int(np.argmin(d))
        order[k] = nxt
        visited[nxt] = True
    return order


def two_opt(
    points: NDArray[np.float64],
    order: NDArray[np.intp],
    *,
    budget_s: float | None = 5.0,
    max_passes: int = 500,
) -> NDArray[np.intp]:
    """First-improvement 2-opt on an open path: reverse a middle segment
    whenever that shortens the path, sweeping until a full pass finds
    nothing. Never returns a longer order than given.

    ``budget_s`` is a wall-clock safety cap checked between passes; ``None``
    runs to convergence (fully reproducible). Returns a new order; the input
    is not modified.
    """
    n = len(order)
    if n < 4:
        return np.array(order, dtype=np.intp, copy=True)

    order = order.copy()
    t0 = time.perf_counter()
    for _ in range(max_passes):
        improved = False
        for p in range(n - 2):
            # Reversing o[p+1..q] replaces edges (o[p],o[p+1]) and
            # (o[q],o[q+1]); q tops out at n-2 so q+1 exists.
            qs = np.arange(p + 2, n - 1)
            if qs.size == 0:
                continue
            a = points[order[p]]
            b = points[order[p + 1]]
            c = points[order[qs]]
            d = points[order[qs + 1]]
            old = float(np.hypot(*(a - b))) + np.hypot(c[:, 0] - d[:, 0], c[:, 1] - d[:, 1])
            new = np.hypot(a[0] - c[:, 0], a[1] - c[:, 1]) + np.hypot(
                b[0] - d[:, 0], b[1] - d[:, 1]
            )
            delta = new - old
            k = int(np.argmin(delta))
            if delta[k] < -1e-12:
                q = int(qs[k])
                order[p + 1 : q + 1] = order[p + 1 : q + 1][::-1]
                improved = True
        if not improved:
            break
        if budget_s is not None and time.perf_counter() - t0 > budget_s:
            break
    return order


def solve_tsp(
    points: NDArray[np.float64],
    *,
    budget_s: float | None = 5.0,
    max_passes: int = 500,
) -> NDArray[np.intp]:
    """Best result of a few fixed greedy nearest-neighbour starts, then 2-opt.

    Deterministic (starts are fixed fractions of n, no RNG). The budget
    only caps time and never worsens the returned order.
    """
    n = len(points)
    if n < 2:
        return np.arange(n, dtype=np.intp)

    starts = sorted({0, n // 4, n // 2, (3 * n) // 4})
    best_order = nearest_neighbour(points, start=starts[0])
    best_len = open_path_length(points, best_order)
    for s in starts[1:]:
        cand = nearest_neighbour(points, start=s)
        cand_len = open_path_length(points, cand)
        if cand_len < best_len:
            best_len = cand_len
            best_order = cand

    return two_opt(points, best_order, budget_s=budget_s, max_passes=max_passes)
