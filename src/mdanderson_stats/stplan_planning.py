"""Bounded inverse planning for the public STPLAN forward-power methods."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from inspect import Signature, signature
from math import isfinite
from types import MappingProxyType
from typing import Literal, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import brentq
from scipy.stats import poisson

from ._cdflib import _freeze as _freeze_array
from ._validation import scalar
from .stplan_case_control import stplan_case_control_power, stplan_matched_case_control_power
from .stplan_continuous import (
    stplan_exponential_one_sample_power,
    stplan_exponential_two_sample_power,
    stplan_lognormal_two_sample_power,
    stplan_normal_one_sample_power,
    stplan_normal_two_sample_power,
    stplan_welch_two_sample_power,
)
from .stplan_correlation import (
    stplan_correlation_one_sample_power,
    stplan_correlation_two_sample_power,
)
from .stplan_discrete import (
    stplan_arcsine_binomial_two_sample_power,
    stplan_binomial_k_sample_power,
    stplan_exact_binomial_power,
    stplan_exact_poisson_power,
    stplan_fisher_exact_approx_power,
    stplan_historical_binomial_power,
    stplan_median_split_power,
    stplan_responder_normal_approximation_power,
    stplan_retention_probability,
)
from .stplan_poisson import stplan_poisson_two_sample_power
from .stplan_survival import (
    stplan_censored_exponential_one_sample_power,
    stplan_george_desu_survival_power,
    stplan_historical_survival_power,
    stplan_information_survival_power,
    stplan_piecewise_survival_power,
)
from .survival_sample_size import exponential_event_probability

_MAX_EVALUATIONS = 20_000
_MAX_VECTOR_SIZE = 200_000
_MAX_CUMULATIVE_WORK = 5_000_000
_FORBIDDEN_COMPUTE = frozenset({"sides", "tail_tolerance", "continued_followup", "model_arm"})
type ForwardFunction = Callable[..., ArrayLike]


@dataclass(frozen=True)
class STPLANMethod:
    """Immutable inverse-planning capabilities for one forward method."""

    function: ForwardFunction
    parameters: tuple[str, ...]
    compute_parameters: tuple[str, ...]
    required_parameters: tuple[str, ...]
    integer_parameters: frozenset[str] = frozenset()
    tied_parameters: tuple[tuple[str, ...], ...] = ()
    group_parameters: tuple[str, ...] = ()


def _descriptor(
    function: ForwardFunction,
    *,
    integer: frozenset[str] = frozenset(),
    tied: tuple[tuple[str, ...], ...] = (),
    groups: tuple[str, ...] = (),
) -> STPLANMethod:
    sig = signature(function)
    names = tuple(sig.parameters)
    required = tuple(
        name for name, parameter in sig.parameters.items() if parameter.default is Signature.empty
    )
    numeric = tuple(name for name in names if name not in _FORBIDDEN_COMPUTE)
    return STPLANMethod(function, names, numeric, required, integer, tied, groups)


STPLAN_METHODS: Mapping[str, STPLANMethod] = MappingProxyType(
    {
        function.__name__: _descriptor(function, integer=integer, tied=tied, groups=groups)
        for function, integer, tied, groups in (
            (stplan_normal_one_sample_power, frozenset(), (), ()),
            (stplan_normal_two_sample_power, frozenset(), (("n1", "n2"),), ()),
            (stplan_welch_two_sample_power, frozenset(), (("n1", "n2"),), ()),
            (stplan_lognormal_two_sample_power, frozenset(), (("n1", "n2"),), ()),
            (stplan_exponential_one_sample_power, frozenset(), (), ()),
            (stplan_exponential_two_sample_power, frozenset(), (("n1", "n2"),), ()),
            (stplan_arcsine_binomial_two_sample_power, frozenset(), (("n1", "n2"),), ()),
            (stplan_median_split_power, frozenset(), (), ()),
            (
                stplan_historical_binomial_power,
                frozenset(),
                (("control_size", "experimental_size"),),
                (),
            ),
            (
                stplan_responder_normal_approximation_power,
                frozenset(),
                (("conservative_size", "expensive_size"),),
                (),
            ),
            (stplan_binomial_k_sample_power, frozenset(), (), ("probabilities", "sample_sizes")),
            (
                stplan_retention_probability,
                frozenset({"initial_size", "minimum_remaining"}),
                (),
                (),
            ),
            (stplan_fisher_exact_approx_power, frozenset(), (), ()),
            (stplan_exact_binomial_power, frozenset({"sample_size"}), (), ()),
            (stplan_exact_poisson_power, frozenset(), (), ()),
            (stplan_correlation_one_sample_power, frozenset(), (), ()),
            (stplan_correlation_two_sample_power, frozenset(), (("n1", "n2"),), ()),
            (stplan_case_control_power, frozenset(), (("n_cases", "n_controls"),), ()),
            (stplan_matched_case_control_power, frozenset({"n_pairs"}), (), ()),
            (stplan_poisson_two_sample_power, frozenset(), (), ()),
            (stplan_censored_exponential_one_sample_power, frozenset(), (), ()),
            (stplan_george_desu_survival_power, frozenset(), (), ()),
            (stplan_information_survival_power, frozenset(), (), ()),
            (stplan_historical_survival_power, frozenset(), (), ()),
            (stplan_piecewise_survival_power, frozenset(), (), ()),
        )
    }
)


@dataclass(frozen=True)
class STPLANSolution:
    """One scalar inverse result and its verified completed forward inputs."""

    method: str
    compute: str | tuple[str, ...]
    index: int | None
    value: float | tuple[float, ...]
    target_power: float
    achieved_power: float
    integer_search: bool
    integer_goal: str | None
    previous_value: float | None
    previous_power: float | None
    bounds: tuple[float, float]
    evaluations: int
    inputs: Mapping[str, object]


def _freeze(value: object) -> object:
    if isinstance(value, np.ndarray):
        return _freeze_array(value)
    if isinstance(value, np.generic):
        return float(value)
    if isinstance(value, tuple):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _freeze_inputs(values: Mapping[str, object]) -> Mapping[str, object]:
    return MappingProxyType({key: _freeze(value) for key, value in values.items()})


def _scalar_parameter(name: str, value: object) -> object:
    if name in ("continued_followup",):
        if not isinstance(value, (bool, np.bool_)):
            raise ValueError(f"{name} must be boolean")
        return bool(value)
    if name == "model_arm":
        if not isinstance(value, str):
            raise ValueError("model_arm must be a string")
        return value
    array = np.asarray(value)
    if array.ndim != 0 or array.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a finite numeric scalar")
    if array.dtype.kind == "b":
        raise ValueError(f"{name} must be numeric, not boolean")
    number = float(array)
    if not isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _check_scalar_design(method: str, values: Mapping[str, object]) -> dict[str, object]:
    spec = STPLAN_METHODS[method]
    result: dict[str, object] = {}
    for name, value in values.items():
        if name in spec.group_parameters:
            array = np.asarray(value)
            if array.ndim != 1 or array.size < 2 or array.size > _MAX_VECTOR_SIZE:
                raise ValueError(f"{name} must be a one-dimensional group vector of size 2..200000")
            if array.dtype.kind not in "iuf" or np.any(~np.isfinite(array.astype(float))):
                raise ValueError(f"{name} must contain finite numeric values")
            result[name] = np.array(array, dtype=float, copy=True)
        else:
            result[name] = _scalar_parameter(name, value)
    return result


def _completed_parameters(
    method: str,
    spec: STPLANMethod,
    supplied: Mapping[str, object],
    computed: tuple[str, ...],
    *,
    whole_k_total: bool = False,
) -> dict[str, object]:
    unknown = set(supplied) - set(spec.parameters)
    if unknown:
        raise ValueError(f"unsupported parameters for method: {', '.join(sorted(unknown))}")
    missing = set(spec.required_parameters) - set(supplied) - set(computed)
    if whole_k_total:
        missing.discard("sample_sizes")
    if missing:
        raise ValueError(f"missing required parameters: {', '.join(sorted(missing))}")
    return _check_scalar_design(method, supplied)


def _call_power(spec: STPLANMethod, kwargs: Mapping[str, object]) -> float:
    raw = np.asarray(spec.function(**kwargs), dtype=float)
    if raw.ndim != 0 or not np.isfinite(raw):
        raise ValueError("inverse planning requires a scalar forward-power result")
    power = float(raw)
    if not 0 <= power <= 1:
        raise ArithmeticError("forward method returned power outside [0,1]")
    return power


def _estimated_work(method: str, kwargs: Mapping[str, object]) -> int:
    """Conservative count of terms generated by a forward mixture/vector call."""
    if method == "stplan_matched_case_control_power":
        return int(cast(float, kwargs["n_pairs"])) + 1
    if method == "stplan_binomial_k_sample_power":
        return int(np.asarray(kwargs["probabilities"]).size)
    if method in (
        "stplan_poisson_two_sample_power",
        "stplan_censored_exponential_one_sample_power",
    ):
        tail = float(cast(float, kwargs.get("tail_tolerance", 1e-10)))
        if method == "stplan_poisson_two_sample_power":
            with np.errstate(over="ignore"):
                mean = cast(float, kwargs["rate1"]) * cast(float, kwargs["exposure1"]) + cast(
                    float, kwargs["rate2"]
                ) * cast(float, kwargs["exposure2"])
        else:
            alternative_mean = cast(float, kwargs["alternative_mean"])
            probability = float(
                exponential_event_probability(
                    1 / alternative_mean,
                    cast(float, kwargs["accrual_duration"]),
                    cast(float, kwargs["followup_duration"]),
                )
            )
            with np.errstate(over="ignore"):
                mean = (
                    cast(float, kwargs["accrual_rate"])
                    * cast(float, kwargs["accrual_duration"])
                    * probability
                )
        if not isfinite(mean) or mean < 0 or not 0 < tail < 0.01:
            raise ValueError("mixture inputs are outside the bounded inverse-planning domain")
        cutoff = float(np.ceil(poisson.isf(tail, mean)))
        if not isfinite(cutoff) or cutoff >= _MAX_CUMULATIVE_WORK:
            return _MAX_CUMULATIVE_WORK + 1
        return max(1, int(cutoff) + 1)
    return 1


def stplan_solve(
    method: str,
    *,
    compute: str | tuple[str, ...],
    target_power: float,
    bounds: tuple[float, float],
    parameters: Mapping[str, object],
    index: int | None = None,
    allocation_weights: ArrayLike | None = None,
    integer: bool | None = None,
    integer_goal: Literal["smallest", "largest"] | None = None,
    power_tolerance: float = 1e-8,
    max_evaluations: int = _MAX_EVALUATIONS,
) -> STPLANSolution:
    """Solve a bounded inverse design problem for an STPLAN forward method.

    ``method`` is the full public ``stplan_*`` function name. Fixed arguments
    are supplied by keyword in ``parameters``; computed parameters are omitted.
    A tuple of tied parameter names (such as ``('n1', 'n2')``) solves a shared
    value for those groups. K-sample inversion accepts ``index`` to replace one
    probability or group size, or computes a total sample size allocated in
    proportion to ``allocation_weights`` (equal weights by default).

    Integer designs are searched exhaustively and return the smallest or largest
    candidate attaining target power within the bounds, so exact discrete
    sawtooth behavior is preserved. Continuous designs solve only within the explicit bracket;
    this does not find all roots or claim a global minimum. A forward residual
    check rejects targets that a discrete jump or numerical limit cannot attain.
    Search is bounded by ``max_evaluations`` (at most 20,000 forward calls).
    """
    if not isinstance(method, str) or method not in STPLAN_METHODS:
        raise ValueError("unknown STPLAN method; use a full public stplan_* function name")
    spec = STPLAN_METHODS[method]
    if not isinstance(parameters, Mapping):
        raise ValueError("parameters must be a mapping of fixed forward-method inputs")
    if isinstance(compute, str):
        keys: tuple[str, ...] = (compute,)
        compute_value: str | tuple[str, ...] = compute
    else:
        keys = tuple(compute)
        compute_value = keys
        if not keys or len(set(keys)) != len(keys):
            raise ValueError("compute tuple must contain distinct parameter names")
        if keys not in spec.tied_parameters:
            raise ValueError("that parameter pair is not a supported shared-size computation")
    if set(keys) & _FORBIDDEN_COMPUTE or any(key not in spec.compute_parameters for key in keys):
        raise ValueError("compute must name a supported numeric design parameter")
    indexed_input = (
        index is not None and len(keys) == 1 and keys[0] in ("probabilities", "sample_sizes")
    )
    if any(key in parameters for key in keys) and not indexed_input:
        raise ValueError("computed parameters must be omitted from parameters")
    target = scalar(target_power, "target_power")
    if not 0 < target < 1:
        raise ValueError("target_power must lie strictly between 0 and 1")
    if not isinstance(bounds, tuple) or len(bounds) != 2:
        raise ValueError("bounds must be a two-value tuple")
    lower, upper = (scalar(value, "bounds") for value in bounds)
    if not lower < upper:
        raise ValueError("bounds must be finite and strictly increasing")
    tolerance = scalar(power_tolerance, "power_tolerance")
    if not 0 < tolerance < 0.1:
        raise ValueError("power_tolerance must lie strictly between 0 and 0.1")
    if isinstance(max_evaluations, bool) or not isinstance(max_evaluations, int):
        raise ValueError("max_evaluations must be an integer")
    if not 1 <= max_evaluations <= _MAX_EVALUATIONS:
        raise ValueError(f"max_evaluations must lie in 1..{_MAX_EVALUATIONS}")

    is_k_sample = method == "stplan_binomial_k_sample_power"
    whole_k_total = is_k_sample and keys == ("sample_sizes",) and index is None
    indexed_k = is_k_sample and index is not None and keys[0] in ("probabilities", "sample_sizes")
    if index is not None and not indexed_k:
        raise ValueError("index is supported only for indexed K-sample parameters")
    if whole_k_total and integer is True:
        raise ValueError(
            "whole K-sample total-size scaling is continuous; integer=True is unsupported"
        )
    if is_k_sample and keys == ("sample_sizes",) and index is None and not whole_k_total:
        raise ValueError("K-sample sizes require an index or whole-vector total-size solve")
    if is_k_sample and keys == ("probabilities",) and index is None:
        raise ValueError("K-sample probabilities require index")
    if allocation_weights is not None and not whole_k_total:
        raise ValueError("allocation_weights is only used for whole K-sample total-size solving")

    allocation: NDArray[np.float64] | None = None
    supplied = dict(parameters)
    if whole_k_total:
        if "sample_sizes" in supplied:
            raise ValueError("omit sample_sizes when solving a K-sample total size")
        if "probabilities" not in supplied:
            raise ValueError("K-sample total size requires fixed probabilities")
        raw_probs = np.asarray(supplied["probabilities"])
        if (
            raw_probs.ndim != 1
            or raw_probs.size < 2
            or raw_probs.size > _MAX_VECTOR_SIZE
            or raw_probs.dtype.kind not in "iuf"
        ):
            raise ValueError("probabilities must be a one-dimensional group vector")
        probs = np.asarray(raw_probs, dtype=float)
        raw_weights = (
            np.ones(probs.size) if allocation_weights is None else np.asarray(allocation_weights)
        )
        if (
            raw_weights.ndim != 1
            or raw_weights.size != probs.size
            or raw_weights.dtype.kind not in "iuf"
        ):
            raise ValueError("allocation_weights must be a positive vector matching group count")
        allocation = np.asarray(raw_weights, dtype=float)
        if np.any(~np.isfinite(allocation) | (allocation <= 0)):
            raise ValueError("allocation_weights must be finite and positive")
        scaled = allocation / np.max(allocation)
        allocation = scaled / np.sum(scaled)
        supplied = {key: value for key, value in supplied.items()}
        supplied["probabilities"] = probs
    elif indexed_k:
        group_value = supplied.get(keys[0])
        if group_value is None:
            raise ValueError(f"indexed K-sample solving requires fixed {keys[0]}")
        raw_vector = np.asarray(group_value)
        if (
            raw_vector.ndim != 1
            or raw_vector.size < 2
            or raw_vector.size > _MAX_VECTOR_SIZE
            or raw_vector.dtype.kind not in "iuf"
        ):
            raise ValueError(f"{keys[0]} must be a one-dimensional group vector")
        vector = np.asarray(raw_vector, dtype=float)
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < vector.size:
            raise ValueError("index is outside the K-sample group vector")

    passed = supplied
    missing_computed = () if indexed_k else keys
    fixed = _completed_parameters(
        method, spec, passed, missing_computed, whole_k_total=whole_k_total
    )

    if indexed_k:
        base_vector = np.asarray(fixed[keys[0]], dtype=float)
        if index is None or not 0 <= index < base_vector.size:
            raise ValueError("index is outside the K-sample group vector")

    intrinsic_integer = keys[0] in spec.integer_parameters and len(keys) == 1
    integer_search = intrinsic_integer if integer is None else integer
    if not isinstance(integer_search, bool):
        raise ValueError("integer must be boolean or None")
    if intrinsic_integer and not integer_search:
        raise ValueError(f"{keys[0]} requires integer attainment search")
    if integer_goal is None:
        goal: Literal["smallest", "largest"] = (
            "largest" if keys == ("minimum_remaining",) else "smallest"
        )
    elif integer_goal in ("smallest", "largest"):
        goal = integer_goal
    else:
        raise ValueError("integer_goal must be 'smallest' or 'largest'")
    if not integer_search and integer_goal is not None:
        raise ValueError("integer_goal applies only to integer attainment searches")
    if integer_search:
        if whole_k_total:
            raise ValueError("whole K-sample total-size scaling is continuous")
        if lower != np.floor(lower) or upper != np.floor(upper):
            raise ValueError("integer search bounds must be integer-valued")
        if max(abs(lower), abs(upper)) >= 2**53:
            raise ValueError("integer search bounds must be below 2**53 for exact count resolution")
        first, last = int(lower), int(upper)
        candidate_count = last - first + 1
        if candidate_count > max_evaluations:
            raise ValueError("integer candidate range exceeds max_evaluations")
    else:
        first, last = 0, 0

    def complete(candidate: float) -> tuple[dict[str, object], dict[str, object]]:
        kwargs = dict(fixed)
        if not indexed_k and not whole_k_total:
            for key in keys:
                kwargs[key] = candidate
        if whole_k_total:
            if allocation is None:
                raise RuntimeError("internal K-sample allocation was not prepared")
            kwargs["sample_sizes"] = candidate * allocation
        if indexed_k:
            vector = np.array(kwargs[keys[0]], dtype=float, copy=True)
            if index is None:
                raise RuntimeError("internal K-sample index missing")
            vector[index] = candidate
            kwargs[keys[0]] = vector
        bound = signature(spec.function).bind(**kwargs)
        bound.apply_defaults()
        call_inputs = dict(bound.arguments)
        return call_inputs, call_inputs

    cumulative_work = 0

    def evaluate(candidate: float) -> tuple[float, dict[str, object]]:
        nonlocal cumulative_work
        kwargs, reported = complete(candidate)
        work = _estimated_work(method, kwargs)
        if work > _MAX_CUMULATIVE_WORK - cumulative_work:
            raise ValueError("inverse search exceeds the five-million-term cumulative work budget")
        cumulative_work += work
        return _call_power(spec, kwargs), reported

    if integer_search:
        evaluations = 0
        previous_value: float | None = None
        previous_power: float | None = None
        candidates = range(first, last + 1) if goal == "smallest" else range(last, first - 1, -1)
        for candidate in candidates:
            achieved, reported = evaluate(float(candidate))
            evaluations += 1
            if achieved >= target:
                return STPLANSolution(
                    method,
                    compute_value,
                    index,
                    float(candidate) if len(keys) == 1 else tuple(float(candidate) for _ in keys),
                    target,
                    achieved,
                    True,
                    goal,
                    previous_value,
                    previous_power,
                    (lower, upper),
                    evaluations,
                    _freeze_inputs(reported),
                )
            previous_value, previous_power = float(candidate), achieved
        raise ValueError("no integer candidate within bounds attains target_power")

    evaluations = 0
    if whole_k_total and lower <= 0:
        raise ValueError("whole K-sample total size bounds must be positive")
    if indexed_k and keys[0] == "sample_sizes" and lower <= 0:
        raise ValueError("K-sample group-size bounds must be positive")
    if max_evaluations < 4:
        raise ValueError("continuous solves require max_evaluations >= 4")
    cache: dict[float, float] = {}

    def candidate_at(coordinate: float) -> float:
        if coordinate <= 0:
            return lower
        if coordinate >= 1:
            return upper
        if lower > 0:
            return float(np.exp(np.log(lower) + coordinate * (np.log(upper) - np.log(lower))))
        return float(lower * (1 - coordinate) + upper * coordinate)

    def residual(coordinate: float) -> float:
        nonlocal evaluations
        if coordinate in cache:
            return cache[coordinate] - target
        if evaluations >= max_evaluations:
            raise ValueError("continuous solve exceeded max_evaluations")
        candidate = candidate_at(coordinate)
        achieved, _ = evaluate(candidate)
        evaluations += 1
        cache[coordinate] = achieved
        return achieved - target

    low_residual, high_residual = residual(0.0), residual(1.0)
    if low_residual == 0:
        root = 0.0
    elif high_residual == 0:
        root = 1.0
    elif low_residual * high_residual > 0:
        raise ValueError("target power is not bracketed by the supplied bounds")
    else:
        root, result = brentq(
            residual,
            0.0,
            1.0,
            xtol=2e-13,
            rtol=8 * np.finfo(float).eps,
            maxiter=max_evaluations - 3,
            full_output=True,
        )
        if not result.converged:
            raise ArithmeticError("bracketed STPLAN solve did not converge")
    value = candidate_at(root)
    achieved, reported = evaluate(value)
    evaluations += 1
    if abs(achieved - target) > tolerance:
        raise ArithmeticError(
            f"forward residual {abs(achieved - target):.6g} exceeds power_tolerance={tolerance:g}"
        )
    return STPLANSolution(
        method,
        compute_value,
        index,
        value if len(keys) == 1 else tuple(value for _ in keys),
        target,
        achieved,
        False,
        None,
        None,
        None,
        (lower, upper),
        evaluations,
        _freeze_inputs(reported),
    )
