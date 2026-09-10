"""Fixed/equality null hypotheses for independent one-parameter SMO groups."""

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import logsumexp

from ._validation import FloatArray, finite


def _components(constraints: ArrayLike, size: int) -> list[tuple[list[int], float | None]]:
    rows = finite(constraints, "constraints")
    if rows.ndim == 1:
        rows = rows[None, :]
    if rows.ndim != 2 or rows.shape[1] != 3 or not 1 <= len(rows) <= 10_000:
        raise ValueError("constraints must have 1..10000 rows of (type, index, value/index)")
    if np.any((rows[:, 0] != 1) & (rows[:, 0] != 2)):
        raise ValueError("constraint type must be 1 (fixed) or 2 (equality)")
    indices = np.r_[rows[:, 1], rows[rows[:, 0] == 2, 2]]
    if np.any((indices < 1) | (indices > size) | (indices != np.floor(indices))):
        raise ValueError("constraint indices must be integers in 1..number of groups")
    parent = list(range(size))

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for kind, index, value in rows:
        if kind == 2:
            parent[root(int(index) - 1)] = root(int(value) - 1)
    fixed: dict[int, float] = {}
    for kind, index, value in rows:
        if kind == 1:
            component = root(int(index) - 1)
            if component in fixed and fixed[component] != value:
                raise ValueError("conflicting fixed values in an equality component")
            fixed[component] = float(value)
    members: dict[int, list[int]] = {}
    for i in range(size):
        members.setdefault(root(i), []).append(i)
    return [(group, fixed.get(component)) for component, group in members.items()]


def _weighted_mean(p: FloatArray, log_weight: FloatArray) -> float:
    """Positive weighted mean without overflowing products or losing close gaps."""
    if p.min() >= 0.5 * p.max():
        normalized = np.exp(log_weight - logsumexp(log_weight))
        normalized /= normalized.sum()
        mean = p.min() + normalized @ (p - p.min())
    else:
        log_mean = logsumexp(log_weight + np.log(p)) - logsumexp(log_weight)
        with np.errstate(over="ignore", under="ignore"):
            mean = np.exp(np.clip(log_mean, np.log(p.min()), np.log(p.max())))
    return float(np.clip(mean, p.min(), p.max()))


def _group_null(
    p: FloatArray,
    log_weight: FloatArray,
    null: ArrayLike | None,
    constraints: ArrayLike | None,
) -> tuple[FloatArray, int]:
    if null is not None:
        if constraints is not None:
            raise ValueError("supply either null parameters or constraints, not both")
        return np.broadcast_to(finite(null, "null parameters"), p.shape), p.size
    if constraints is None:
        if p.size < 2:
            raise ValueError("equality testing requires at least two groups")
        return np.full_like(p, _weighted_mean(p, log_weight)), p.size - 1
    q = p.copy()
    df = 0
    for group, fixed in _components(constraints, p.size):
        if fixed is not None:
            q[group] = fixed
            df += len(group)
        elif len(group) > 1:
            q[group] = _weighted_mean(p[group], log_weight[group])
            df += len(group) - 1
    if df == 0:
        raise ValueError("constraints must impose at least one independent restriction")
    return q, df
