"""Posterior-utility and global-safety decisions for the 2010 U2OET GAO model."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from ._cdflib import _freeze
from ._validation import FloatArray
from .hierarchical_binomial import summarize_chains
from .u2oet import _real
from .u2oet_decision import _integer

_MAX_POSTERIOR_CELLS = 20_000_000


@dataclass(frozen=True)
class U2OETGAO2010Decision:
    """One original-GAO decision with posterior summaries on the dose grid."""

    mean_utility: FloatArray
    utility_sd: FloatArray
    utility_mcse: FloatArray
    mean_severe_toxicity: FloatArray
    severe_toxicity_mcse: FloatArray
    severe_toxicity_exceedance_probability: FloatArray
    exceedance_probability_mcse: FloatArray
    eligible: np.ndarray
    stopped_for_global_toxicity: bool
    minimum_exceedance_probability: float
    selected_pair: tuple[int, int] | None
    action: str
    posterior_draws: int


def _bounded_shape(value: ArrayLike, name: str) -> tuple[int, ...]:
    """Obtain bounded rectangular dimensions before materializing nested input."""
    if isinstance(value, np.ndarray):
        if value.size > _MAX_POSTERIOR_CELLS:
            raise ValueError(f"{name} exceeds the posterior cell limit")
        return value.shape
    if isinstance(value, (list, tuple)):

        def visit(item: object, depth: int) -> tuple[tuple[int, ...], int]:
            if depth == 6:
                if isinstance(item, (list, tuple, np.ndarray)):
                    raise ValueError(f"{name} must be a rectangular six-dimensional array")
                return (), 1
            if not isinstance(item, (list, tuple)) or len(item) > _MAX_POSTERIOR_CELLS:
                raise ValueError(f"{name} must be a bounded rectangular six-dimensional array")
            child_shapes = [visit(child, depth + 1) for child in item]
            if child_shapes and any(shape != child_shapes[0][0] for shape, _ in child_shapes):
                raise ValueError(f"{name} must be rectangular")
            size = sum(cells for _, cells in child_shapes)
            if size > _MAX_POSTERIOR_CELLS:
                raise ValueError(f"{name} exceeds the posterior cell limit")
            return (len(item),) + (child_shapes[0][0] if child_shapes else ()), size

        shape, _ = visit(value, 0)
        return shape
    shape_attr = getattr(value, "shape", None)
    size = getattr(value, "size", None)
    if shape_attr is None or size is None or int(size) > _MAX_POSTERIOR_CELLS:
        raise ValueError(f"{name} must expose bounded shape and size before conversion")
    try:
        return tuple(int(dimension) for dimension in shape_attr)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must have a valid six-dimensional shape") from exc


def _bounded_small_shape(
    value: ArrayLike, name: str, dimensions: int, limit: int
) -> tuple[int, ...]:
    """Inspect tiny public arrays before NumPy can expand a nested sequence."""
    if isinstance(value, np.ndarray):
        if value.size > limit:
            raise ValueError(f"{name} exceeds its {limit}-cell bound")
        return value.shape
    if isinstance(value, (list, tuple)):

        def visit(item: object, depth: int) -> tuple[int, ...]:
            if depth == 0:
                if isinstance(item, (list, tuple, np.ndarray)):
                    raise ValueError(f"{name} must be a {dimensions}-dimensional array")
                return ()
            if not isinstance(item, (list, tuple)) or len(item) > limit:
                raise ValueError(f"{name} exceeds its bounded array shape")
            children = [visit(child, depth - 1) for child in item]
            if children and any(shape != children[0] for shape in children):
                raise ValueError(f"{name} must be rectangular")
            return (len(item),) + (children[0] if children else (0,) * (depth - 1))

        return visit(value, dimensions)
    shape = getattr(value, "shape", None)
    size = getattr(value, "size", None)
    if shape is None or size is None or int(size) > limit:
        raise ValueError(f"{name} must expose bounded shape and size before conversion")
    try:
        return tuple(int(dimension) for dimension in shape)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must have a valid shape") from exc


def _grid_pair(value: tuple[int, int], shape: tuple[int, int], name: str) -> tuple[int, int]:
    if not isinstance(value, (list, tuple, np.ndarray)) or len(value) != 2:
        raise ValueError(f"{name} must contain two zero-based dose indices")
    return (
        _integer(value[0], name, 0, shape[0] - 1),
        _integer(value[1], name, 0, shape[1] - 1),
    )


def u2oet_gao2010_decision(
    posterior_joint_draws: ArrayLike,
    utility: ArrayLike,
    *,
    toxicity_limit: float,
    stopping_probability: float = 0.80,
    current_pair: tuple[int, int] | None = None,
    treated: ArrayLike | None = None,
) -> U2OETGAO2010Decision:
    """Apply the 2010 paper's utility maximum and global all-dose safety stop.

    ``posterior_joint_draws`` has axes ``(chain, draw, dose1, dose2,
    efficacy, toxicity)``. Utility is efficacy-by-toxicity. The global rule is
    strict: stop only when every dose pair has posterior probability greater
    than ``stopping_probability`` that its most-severe-toxicity probability
    exceeds ``toxicity_limit``.

    For an interim decision, ``current_pair`` and the nonnegative integer
    ``treated`` grid are required. Previously treated pairs remain candidates.
    An untried pair is eligible if it is coordinatewise no higher than the
    current pair, or is one of the three listed upward neighbors. Untried
    mixed-direction moves are excluded as a conservative Python convention:
    the paper names the three allowed upward neighbors but does not resolve a
    move that increases one agent while decreasing the other. With
    ``current_pair=None``, the decision is final and the full dose grid is
    eligible. Ties select the first dose pair in row-major grid order.
    """
    expected_shape = _bounded_shape(posterior_joint_draws, "posterior_joint_draws")
    if (
        len(expected_shape) != 6
        or not 2 <= expected_shape[0] <= 16
        or expected_shape[1] < 8
        or any(not 2 <= size <= 5 for size in expected_shape[2:4])
        or any(not 2 <= size <= 4 for size in expected_shape[4:])
    ):
        raise ValueError(
            "posterior_joint_draws must have chain, draw, dose1, dose2, efficacy, toxicity axes"
        )
    if np.iscomplexobj(posterior_joint_draws):
        raise ValueError("posterior_joint_draws must be real")
    draws = _real(posterior_joint_draws, "posterior_joint_draws")
    if draws.size > _MAX_POSTERIOR_CELLS:
        raise ValueError("posterior joint draws exceed the 20-million-cell bound")
    if np.any((draws < 0.0) | (draws > 1.0)):
        raise ValueError("posterior joint probabilities must lie in [0,1]")
    if np.any(np.abs(draws.sum(axis=(-2, -1)) - 1.0) > 2e-11):
        raise ValueError("each posterior draw and dose pair must have unit joint probability")

    efficacy_levels, toxicity_levels = expected_shape[-2:]
    utility_shape = _bounded_small_shape(utility, "utility", 2, 16)
    if utility_shape != (efficacy_levels, toxicity_levels):
        raise ValueError("utility must have efficacy-by-toxicity shape")
    u = _real(utility, "utility")
    limit = _real(toxicity_limit, "toxicity_limit")
    p_upper = _real(stopping_probability, "stopping_probability")
    if limit.ndim != 0 or not 0.0 <= float(limit) <= 1.0:
        raise ValueError("toxicity_limit must be a probability")
    if p_upper.ndim != 0 or not 0.0 <= float(p_upper) <= 1.0:
        raise ValueError("stopping_probability must be a probability")

    dose_shape = expected_shape[2:4]
    if current_pair is None:
        if treated is not None:
            raise ValueError("treated is only used for an interim decision")
        eligible = np.ones(dose_shape, dtype=bool)
    else:
        current = _grid_pair(current_pair, dose_shape, "current_pair")
        if treated is None:
            raise ValueError("treated is required for an interim decision")
        treated_shape = _bounded_small_shape(treated, "treated", 2, 25)
        if tuple(treated_shape) != dose_shape:
            raise ValueError("treated must match the dose grid")
        if isinstance(treated, np.ndarray) and treated.size > 25:
            raise ValueError("treated must match the bounded dose grid")
        counts = _real(treated, "treated")
        if np.any(counts < 0) or np.any(counts != np.floor(counts)):
            raise ValueError("treated must contain nonnegative integer counts")
        eligible = counts > 0
        for first in range(dose_shape[0]):
            for second in range(dose_shape[1]):
                if eligible[first, second]:
                    continue
                coordinatewise_deescalation = first <= current[0] and second <= current[1]
                source_upward_neighbor = (first, second) in (
                    (current[0] + 1, current[1]),
                    (current[0], current[1] + 1),
                    (current[0] + 1, current[1] + 1),
                )
                eligible[first, second] = coordinatewise_deescalation or source_upward_neighbor

    utility_scale = float(np.max(np.abs(u)))
    if not np.isfinite(utility_scale):
        raise ValueError("utility must contain only finite values")
    if utility_scale == 0.0:
        utility_scale = 1.0
    with np.errstate(over="ignore", invalid="ignore"):
        normalized_utility_draws = np.einsum(
            "cdijxy,xy->cdij", draws, u / utility_scale, optimize=True
        )
    if not np.all(np.isfinite(normalized_utility_draws)):
        raise ArithmeticError("posterior expected utility exceeds floating-point range")
    normalized_summary = summarize_chains(normalized_utility_draws)
    with np.errstate(over="ignore", invalid="ignore"):
        utility_mean = normalized_summary.mean * utility_scale
        utility_sd = normalized_summary.standard_deviation * utility_scale
        utility_mcse = normalized_summary.batch_mean_mcse * utility_scale
    if not all(np.all(np.isfinite(value)) for value in (utility_mean, utility_sd, utility_mcse)):
        raise ArithmeticError("posterior utility summaries exceed floating-point range")

    # Extended-precision summation preserves representable boundary ties when
    # efficacy-category cells partition an exactly specified severe mass.
    severe = draws[..., -1].sum(axis=-1, dtype=np.longdouble).astype(float)
    exceedance_draws = (severe > float(limit)).astype(float)
    severe_summary = summarize_chains(severe)
    exceedance_summary = summarize_chains(exceedance_draws)
    exceedance_probability = np.asarray(exceedance_summary.mean)
    stopped = bool(np.min(exceedance_probability) > float(p_upper))
    if stopped:
        selected = None
        action = "stop_all_dose_pairs_too_toxic"
    elif not np.any(eligible):
        selected = None
        action = "no_eligible_pair"
    else:
        scores = np.where(eligible, utility_mean, -np.inf)
        flat_index = int(np.argmax(scores))
        selected = (flat_index // dose_shape[1], flat_index % dose_shape[1])
        action = "final_select" if current_pair is None else "assign_next_cohort"

    eligible_readonly = np.frombuffer(eligible.tobytes(), dtype=bool).reshape(dose_shape)
    return U2OETGAO2010Decision(
        mean_utility=_freeze(utility_mean),
        utility_sd=_freeze(utility_sd),
        utility_mcse=_freeze(utility_mcse),
        mean_severe_toxicity=_freeze(severe_summary.mean),
        severe_toxicity_mcse=_freeze(severe_summary.batch_mean_mcse),
        severe_toxicity_exceedance_probability=_freeze(exceedance_probability),
        exceedance_probability_mcse=_freeze(exceedance_summary.batch_mean_mcse),
        eligible=eligible_readonly,
        stopped_for_global_toxicity=stopped,
        minimum_exceedance_probability=float(np.min(exceedance_probability)),
        selected_pair=selected,
        action=action,
        posterior_draws=int(expected_shape[0] * expected_shape[1]),
    )
