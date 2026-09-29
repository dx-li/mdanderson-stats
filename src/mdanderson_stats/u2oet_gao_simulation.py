"""Calendar trials using the explicit-prior generalized Aranda–Ordaz fit."""

import hashlib
import json
from dataclasses import asdict, dataclass

import numpy as np
from numpy.typing import ArrayLike

from ._validation import FloatArray
from .u2oet import _real, u2oet_standardize
from .u2oet_decision import U2OETCriteria, U2OETPosterior, _integer, u2oet_posterior
from .u2oet_gao_fit import (
    _MAX_LIKELIHOOD_EVALUATIONS,
    _MAX_RETAINED_CELLS,
    _MAX_WORK_UNITS,
    fit_u2oet_gao,
    u2oet_gao_parameter_names,
)
from .u2oet_patients import U2OETPatients, u2oet_next_patient, u2oet_patients
from .u2oet_scenario import U2OETScenario
from .u2oet_simulation import U2OETTrial, U2OETTrialDecision, _window

_MAX_TRIAL_CELLS = 25_000_000


@dataclass(frozen=True)
class U2OETGAOTrial(U2OETTrial):
    """A summary-compatible trial plus GAO work and replay metadata."""

    likelihood_evaluations: int = 0
    likelihood_work_units: int = 0
    maximum_split_rhat: float = float("nan")
    planned_arrival_times: FloatArray | None = None
    data_uniforms: FloatArray | None = None
    replay_only: bool = False


def _shape(value: ArrayLike, name: str) -> tuple[int, ...]:
    shape = getattr(value, "shape", None)
    if shape is None:
        if isinstance(value, (list, tuple)):
            nested = any(not np.isscalar(row) for row in value)
            if nested:
                if (
                    not value
                    or any(np.isscalar(row) for row in value)
                    or any(not hasattr(row, "__len__") for row in value)
                    or any(len(row) != len(value[0]) for row in value)
                ):
                    raise ValueError(f"{name} must be a rectangular array")
                return (len(value), len(value[0]))
            return (len(value),)
        return ()
    return tuple(int(size) for size in shape)


def _tape(value: ArrayLike | None, shape: tuple[int, ...], name: str) -> FloatArray | None:
    if value is None:
        return None
    actual_shape = _shape(value, name)
    if actual_shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if np.iscomplexobj(value):
        raise ValueError(f"{name} must contain real values")
    result = _real(value, name)
    if result.shape != shape or np.any((result < 0) | (result >= 1)):
        raise ValueError(f"{name} values must lie in [0,1)")
    return result.copy()


def _arrivals(value: ArrayLike | None, size: int) -> FloatArray | None:
    if value is None:
        return None
    if _shape(value, "arrival_times") != (size,) or np.iscomplexobj(value):
        raise ValueError("arrival_times must be a real vector of length max_patients")
    times = _real(value, "arrival_times")
    if (
        times.shape != (size,)
        or not np.all(np.isfinite(times))
        or times[0] != 0
        or np.any(np.diff(times) <= 0)
    ):
        raise ValueError("arrival_times must start at zero and increase strictly")
    return times.copy()


def _readonly(value: ArrayLike) -> FloatArray:
    result = np.array(value, dtype=np.float64, copy=True)
    result.flags.writeable = False
    return result


def _minimum_fit_evaluations(chains: int, warmup: int, draws: int, free: bool) -> int:
    return chains * (1 + warmup + draws if free else 1)


def _maximum_defined(values: ArrayLike) -> float:
    array = np.asarray(values, dtype=np.float64)
    defined = array[~np.isnan(array)]
    return float(np.max(defined)) if defined.size else float("nan")


def simulate_u2oet_gao_trial(
    doses1: ArrayLike,
    doses2: ArrayLike,
    scenario: U2OETScenario,
    utility: ArrayLike,
    *,
    prior_mean: ArrayLike,
    prior_sd: ArrayLike,
    initial: tuple[int, int],
    criteria: U2OETCriteria = U2OETCriteria(),
    max_patients: int = 60,
    cohort_size: int = 3,
    surplus: int | None = None,
    top: int | None = 2,
    greedy: bool = False,
    efficacy_window: ArrayLike = (42.0, 42.0),
    toxicity_window: ArrayLike = (42.0, 42.0),
    mean_interarrival: float = 20.0,
    final_scope: str = "acceptable",
    draws: int = 1000,
    warmup: int = 500,
    chains: int = 4,
    initial_parameters: ArrayLike | None = None,
    arrival_times: ArrayLike | None = None,
    data_uniforms: ArrayLike | None = None,
    max_likelihood_evaluations: int = _MAX_LIKELIHOOD_EVALUATIONS,
    max_work: int = _MAX_WORK_UNITS,
    rng: np.random.Generator,
) -> U2OETGAOTrial:
    """Run the existing U2OET calendar and allocation rules with GAO fitting.

    ``prior_mean`` and ``prior_sd`` use every coordinate returned by
    ``u2oet_gao_parameter_names``, including ``association.fisher_z``. They
    define an explicit Python prior and do not claim native prior-file parity.
    ``scenario`` supplies the joint ordinal truth at each dose pair.

    Optional ``arrival_times`` must give the full planned schedule beginning at
    zero. Optional ``data_uniforms`` has shape ``(max_patients, 4)`` with
    columns allocation, outcome category, efficacy delay and toxicity delay.
    These explicit tapes make the data path replayable independently of the
    posterior stream. If omitted, the returned trial retains generated tapes.
    The model is refit only when available complete or toxicity-only counts
    change. Cumulative fit budgets and the combined fit/history cell bound
    cover every refit rather than resetting for each calendar look.
    """
    if not isinstance(rng, np.random.Generator) or not isinstance(scenario, U2OETScenario):
        raise ValueError("require an explicit Generator and U2OETScenario")
    d1, d2 = _real(doses1, "doses1"), _real(doses2, "doses2")
    u2oet_standardize(d1)
    u2oet_standardize(d2)
    p_true = _real(scenario.joint, "scenario.joint")
    if (
        p_true.ndim != 4
        or p_true.shape[:2] != (d1.size, d2.size)
        or any(not 2 <= size <= 4 for size in p_true.shape[-2:])
        or p_true.size > 20_000_000
        or np.any((p_true < 0) | (p_true > 1))
        or np.any(np.abs(p_true.sum(axis=(-2, -1)) - 1.0) > 1e-12)
    ):
        raise ValueError("scenario joint probabilities must match the dose grid and sum to one")
    shape = p_true.shape
    utility_array = _real(utility, "utility")
    if utility_array.shape != shape[2:] or np.any(utility_array < 0):
        raise ValueError("utility must be a nonnegative efficacy-by-toxicity matrix")
    if (
        not isinstance(criteria, U2OETCriteria)
        or criteria.efficacy_level >= shape[2]
        or criteria.toxicity_level >= shape[3]
    ):
        raise ValueError("criteria must match outcome categories")
    names = u2oet_gao_parameter_names(shape[2], shape[3])
    mean, sd = _real(prior_mean, "prior_mean"), _real(prior_sd, "prior_sd")
    if (
        mean.shape != (len(names),)
        or sd.shape != mean.shape
        or not np.all(np.isfinite(mean))
        or not np.all(np.isfinite(sd))
        or np.any(sd < 0)
    ):
        raise ValueError("GAO prior arrays must match all named coordinates; SDs may be zero")
    maximum = _integer(max_patients, "max_patients", 1, 2500)
    cohort = _integer(cohort_size, "cohort_size", 1, 2500)
    surplus_value = cohort if surplus is None else _integer(surplus, "surplus", 0, 2500)
    top_value = None if top is None else _integer(top, "top", 1, d1.size * d2.size)
    if len(initial) != 2:
        raise ValueError("initial must contain two dose indices")
    first = (
        _integer(initial[0], "initial", 0, shape[0] - 1),
        _integer(initial[1], "initial", 0, shape[1] - 1),
    )
    if final_scope not in ("acceptable", "tried"):
        raise ValueError("final_scope must be acceptable or tried")
    if not isinstance(greedy, (bool, np.bool_)):
        raise ValueError("greedy must be boolean")
    draws = _integer(draws, "draws", 8, 100_000)
    warmup = _integer(warmup, "warmup", 0, 100_000)
    chains = _integer(chains, "chains", 2, 16)
    max_likelihood_evaluations = _integer(
        max_likelihood_evaluations,
        "max_likelihood_evaluations",
        1,
        _MAX_LIKELIHOOD_EVALUATIONS,
    )
    max_work = _integer(max_work, "max_work", 1, _MAX_WORK_UNITS)
    free = bool(np.any(sd > 0))
    minimum_evaluations = _minimum_fit_evaluations(chains, warmup, draws, free)
    joint_cells = int(np.prod(shape, dtype=np.int64))
    if minimum_evaluations > max_likelihood_evaluations:
        raise ValueError("minimum GAO fit evaluations exceed the cumulative trial budget")
    if minimum_evaluations * joint_cells > max_work:
        raise ValueError("minimum GAO fit work exceeds the cumulative trial budget")
    if initial_parameters is None:
        initial_values = None
    else:
        initial_shape = _shape(initial_parameters, "initial_parameters")
        if initial_shape not in ((len(names),), (chains, len(names))):
            raise ValueError("initial_parameters must be one named vector or one vector per chain")
        if np.iscomplexobj(initial_parameters):
            raise ValueError("initial_parameters must be real")
        initial_values = _real(initial_parameters, "initial_parameters").copy()
        if initial_values.shape == (len(names),):
            initial_values = np.broadcast_to(initial_values, (chains, len(names))).copy()
        if not np.all(np.isfinite(initial_values)):
            raise ValueError("initial_parameters must be finite")
        if np.any(initial_values[:, sd == 0] != mean[sd == 0]):
            raise ValueError("fixed initial coordinates must equal their prior means")
    ew, tw = (
        _window(efficacy_window, "efficacy_window"),
        _window(toxicity_window, "toxicity_window"),
    )
    gap = _real(mean_interarrival, "mean_interarrival")
    if gap.ndim or not np.isfinite(gap) or gap <= 0:
        raise ValueError("mean_interarrival must be a positive finite scalar")
    planned_arrivals = _arrivals(arrival_times, maximum)
    explicit_uniform_tape = data_uniforms is not None
    uniforms = _tape(data_uniforms, (maximum, 4), "data_uniforms")
    if planned_arrivals is not None:
        if np.any(~np.isfinite(planned_arrivals + tw[1])) or np.any(
            ~np.isfinite(planned_arrivals + ew[1])
        ):
            raise ValueError("planned outcome times exceed floating-point range")
        if (tw[0] > 0 and np.any(planned_arrivals + tw[0] <= planned_arrivals)) or (
            ew[0] > 0 and np.any(planned_arrivals + ew[0] <= planned_arrivals)
        ):
            raise ValueError(
                "positive observation delays are not representable at this calendar scale"
            )
    dose_pairs = d1.size * d2.size
    fit_cells = chains * draws * (14 * len(names) + 2 * joint_cells + 2)
    posterior_cells = chains * draws * (2 * dose_pairs + joint_cells)
    history_cells = maximum * (5 * dose_pairs + 24) + maximum * 12 + 2 * joint_cells
    tape_cells = maximum * 5 + maximum
    combined_cells = fit_cells + posterior_cells + history_cells + tape_cells
    if fit_cells > _MAX_RETAINED_CELLS or combined_cells > _MAX_TRIAL_CELLS:
        raise ValueError(
            "combined GAO fit, posterior, calendar and history storage exceeds its bound"
        )

    design_json = json.dumps(
        {
            "format_version": 1,
            "model": "gao",
            "doses1": d1.tolist(),
            "doses2": d2.tolist(),
            "scenario_joint": p_true.tolist(),
            "utility": utility_array.tolist(),
            "criteria": asdict(criteria),
            "prior_coordinate_names": names,
            "prior_mean": mean.tolist(),
            "prior_sd": sd.tolist(),
            "initial_parameters": None if initial_values is None else initial_values.tolist(),
            "initial": first,
            "max_patients": maximum,
            "cohort_size": cohort,
            "surplus": surplus_value,
            "top": top_value,
            "greedy": bool(greedy),
            "efficacy_window": ew.tolist(),
            "toxicity_window": tw.tolist(),
            "mean_interarrival": float(gap),
            "arrival_schedule": "explicit" if planned_arrivals is not None else "exponential",
            "arrival_times": None if planned_arrivals is None else planned_arrivals.tolist(),
            "uniform_tape_sha256": (
                None
                if uniforms is None
                else hashlib.sha256(np.ascontiguousarray(uniforms).tobytes()).hexdigest()
            ),
            "final_scope": final_scope,
            "draws": draws,
            "warmup": warmup,
            "chains": chains,
            "max_likelihood_evaluations": max_likelihood_evaluations,
            "max_work": max_work,
        },
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )

    seeds = rng.integers(0, 2**63, size=2, dtype=np.int64)
    data_rng, posterior_rng = (np.random.default_rng(int(seed)) for seed in seeds)
    if planned_arrivals is None:
        gaps = data_rng.exponential(float(gap), size=maximum - 1)
        planned_arrivals = np.r_[0.0, np.cumsum(gaps, dtype=np.float64)]
        if not np.all(np.isfinite(planned_arrivals)) or np.any(np.diff(planned_arrivals) <= 0):
            raise ArithmeticError("generated arrivals are not finite and strictly increasing")
    if uniforms is None:
        uniforms = data_rng.random((maximum, 4))
    uniforms = _readonly(uniforms)
    planned_arrivals = _readonly(planned_arrivals)

    records: list[list[int]] = []
    used_arrivals: list[float] = []
    outcome_times: list[tuple[float, float]] = []
    history: list[U2OETTrialDecision] = []
    cached_counts: tuple[FloatArray, FloatArray] | None = None
    posterior: U2OETPosterior | None = None
    rhat = float("nan")
    max_rhat = float("nan")
    fits = 0
    total_evaluations = 0
    total_work = 0

    def snapshot(now: float) -> U2OETPatients:
        rows = np.asarray(records, dtype=np.float64).reshape(-1, 5)
        if rows.size:
            observed = np.asarray(outcome_times, dtype=np.float64) <= now
            rows[:, 3:] = np.where(observed, rows[:, 3:], -1)
        return u2oet_patients(
            rows,
            dose_counts=(d1.size, d2.size),
            efficacy_levels=shape[2],
            toxicity_levels=shape[3],
        )

    def update(data: U2OETPatients) -> tuple[U2OETPosterior, float]:
        nonlocal posterior, cached_counts, fits, rhat, max_rhat
        nonlocal total_evaluations, total_work
        if cached_counts is None or not (
            np.array_equal(data.complete, cached_counts[0])
            and np.array_equal(data.toxicity_only, cached_counts[1])
        ):
            remaining_evaluations = max_likelihood_evaluations - total_evaluations
            remaining_work = max_work - total_work
            if (
                remaining_evaluations < minimum_evaluations
                or remaining_work < minimum_evaluations * joint_cells
            ):
                raise ArithmeticError("cumulative GAO trial fit budget exhausted")
            fit = fit_u2oet_gao(
                d1,
                d2,
                data.complete,
                toxicity_only=data.toxicity_only,
                prior_mean=mean,
                prior_sd=sd,
                draws=draws,
                warmup=warmup,
                chains=chains,
                initial=initial_values,
                rng=posterior_rng,
                max_likelihood_evaluations=remaining_evaluations,
                max_work=remaining_work,
            )
            total_evaluations += fit.likelihood_evaluations
            total_work += fit.likelihood_work_units
            posterior = u2oet_posterior(
                fit.joint.reshape(-1, *shape), utility_array, criteria=criteria
            )
            rhat = _maximum_defined(fit.parameter_summary.split_rhat)
            if np.isnan(max_rhat) or (not np.isnan(rhat) and rhat > max_rhat):
                max_rhat = rhat
            cached_counts = (data.complete, data.toxicity_only)
            fits += 1
        assert posterior is not None
        return posterior, rhat

    now = 0.0
    stopped_early = False
    for patient in range(maximum):
        now = float(planned_arrivals[patient])
        if patient == 0:
            pair = first
        else:
            data = snapshot(now)
            current, diagnostic = update(data)
            decision = u2oet_next_patient(
                current,
                data,
                cohort_size=cohort,
                max_patients=maximum,
                surplus=surplus_value,
                top=top_value,
                initial=first,
                greedy=greedy,
            )
            history.append(
                U2OETTrialDecision(
                    now,
                    int(data.complete.sum()),
                    int(data.toxicity_only.sum()),
                    data.ignored_outcomes,
                    decision.probabilities,
                    current,
                    diagnostic,
                    decision.continuing_cohort,
                    decision.reason,
                )
            )
            if not np.any(decision.probabilities):
                stopped_early = True
                break
            cumulative = np.cumsum(decision.probabilities.ravel())
            cumulative[-1] = 1.0
            chosen = int(np.searchsorted(cumulative, uniforms[patient, 0], side="right"))
            if chosen >= dose_pairs:
                raise ArithmeticError("allocation tape fell outside the normalized probabilities")
            pair = (chosen // d2.size, chosen % d2.size)
        outcome_cdf = np.cumsum(p_true[pair].ravel() / p_true[pair].sum())
        outcome_cdf[-1] = 1.0
        outcome = int(np.searchsorted(outcome_cdf, uniforms[patient, 1], side="right"))
        if outcome >= shape[2] * shape[3]:
            raise ArithmeticError("outcome tape fell outside the normalized probabilities")
        records.append(
            [patient + 1, pair[0] + 1, pair[1] + 1, outcome // shape[3], outcome % shape[3]]
        )
        used_arrivals.append(now)
        efficacy_delay = ew[0] + uniforms[patient, 2] * (ew[1] - ew[0])
        toxicity_delay = tw[0] + uniforms[patient, 3] * (tw[1] - tw[0])
        observed = (now + float(efficacy_delay), now + float(toxicity_delay))
        if not np.all(np.isfinite(observed)):
            raise ArithmeticError("outcome observation times exceed floating-point range")
        if (efficacy_delay > 0 and observed[0] <= now) or (
            toxicity_delay > 0 and observed[1] <= now
        ):
            raise ArithmeticError(
                "positive outcome delays are not representable at this calendar scale"
            )
        outcome_times.append(observed)

    stop_time = now
    analysis_time = max(stop_time, float(np.max(outcome_times)))
    final_data = snapshot(analysis_time)
    final_posterior, final_rhat = update(final_data)
    acceptable = final_posterior.acceptable.copy()
    if final_scope == "tried":
        acceptable &= final_data.treated > 0
    selected = None
    if not stopped_early and np.any(acceptable):
        best = int(np.argmax(np.where(acceptable, final_posterior.mean_utility, -np.inf)))
        selected = (best // d2.size, best % d2.size)
    return U2OETGAOTrial(
        patients=final_data,
        arrival_times=_readonly(used_arrivals),
        outcome_times=_readonly(outcome_times),
        decisions=tuple(history),
        final_posterior=final_posterior,
        final_max_split_rhat=final_rhat,
        selected=selected,
        stopped_early=stopped_early,
        stop_time=stop_time,
        analysis_time=analysis_time,
        posterior_fits=fits,
        data_seed=int(seeds[0]),
        posterior_seed=int(seeds[1]),
        final_scope=final_scope,
        design_json=design_json,
        likelihood_evaluations=total_evaluations,
        likelihood_work_units=total_work,
        maximum_split_rhat=max_rhat,
        planned_arrival_times=planned_arrivals,
        data_uniforms=uniforms,
        replay_only=explicit_uniform_tape,
    )
