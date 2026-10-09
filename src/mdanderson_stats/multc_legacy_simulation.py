"""Bounded Monte Carlo study summaries for the recovered Multc Lean timing law."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._validation import FloatArray
from .multc_core import MultcLeanDesign
from .multc_legacy_boundaries import MultcLegacyBoundaries, multc_legacy_boundaries
from .multc_legacy_duration import (
    MultcLegacyDuration,
    MultcLegacyPriorDecision,
    _real_scalar,
    _run_multc_legacy_duration,
    _stream,
)
from .multc_simulation import _scaled_mean_mcse


def _integer(value: int, name: str, minimum: int, maximum: int) -> int:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or not minimum <= value <= maximum
    ):
        raise ValueError(f"{name} must be an integer in [{minimum},{maximum}]")
    return int(value)


def _owned(values: ArrayLike) -> FloatArray:
    array = np.array(values, dtype=np.float64, copy=True)
    array.flags.writeable = False
    return array


@dataclass(frozen=True)
class MultcLegacyDurationSummary:
    """Native-defined study means/PMF, with Python Monte Carlo error estimates.

    Duration, sample size, response, toxicity and balk arrays describe every
    replicate, including prior rejections. MCSE is undefined (NaN) for one
    replicate. A prior rejection has sample-size probability one at zero;
    native 2.1 leaves its probability vector unpopulated in that special case.
    """

    trials: int
    max_subjects: int
    duration: FloatArray
    sample_size: FloatArray
    responses: FloatArray
    toxicities: FloatArray
    balks: FloatArray
    sample_size_probability: FloatArray
    mean_duration: float
    duration_mcse: float
    mean_sample_size: float
    sample_size_mcse: float
    mean_responses: float
    responses_mcse: float
    mean_toxicities: float
    toxicities_mcse: float
    mean_balks: float
    balks_mcse: float


def summarize_multc_legacy_durations(
    max_subjects: int, trials: Sequence[MultcLegacyDuration]
) -> MultcLegacyDurationSummary:
    """Summarize 1..10,000 completed explicit-variate or simulated trials."""
    cap = _integer(max_subjects, "max_subjects", 3, 1000)
    if not isinstance(trials, Sequence) or not 1 <= len(trials) <= 10_000:
        raise ValueError("trials must be a sequence of 1..10000 completed legacy trials")
    for trial in trials:
        if not isinstance(trial, MultcLegacyDuration) or not 0 <= trial.sample_size <= cap:
            raise ValueError("trial must be a completed MultcLegacyDuration within the cap")
        for field in ("responses", "toxicities", "balks"):
            value = getattr(trial, field)
            _integer(value, field, 0, 1_000_000 if field == "balks" else trial.sample_size)
        duration = np.asarray(trial.duration)
        if (
            duration.ndim != 0
            or duration.dtype.kind not in "iuf"
            or not np.isfinite(duration)
            or duration < 0
        ):
            raise ValueError("trial duration must be finite and nonnegative")
    arrays = tuple(
        _owned([getattr(trial, name) for trial in trials])
        for name in ("duration", "sample_size", "responses", "toxicities", "balks")
    )
    return _summarize_arrays(cap, arrays)


def _summarize_arrays(cap: int, arrays: Sequence[FloatArray]) -> MultcLegacyDurationSummary:
    duration, size, response, toxicity, balks = arrays
    probability = _owned(np.bincount(size.astype(np.int64), minlength=cap + 1) / size.size)
    mean_d, error_d = _scaled_mean_mcse(duration)
    mean_n, error_n = _scaled_mean_mcse(size)
    mean_r, error_r = _scaled_mean_mcse(response)
    mean_t, error_t = _scaled_mean_mcse(toxicity)
    mean_b, error_b = _scaled_mean_mcse(balks)
    return MultcLegacyDurationSummary(
        size.size,
        cap,
        duration,
        size,
        response,
        toxicity,
        balks,
        probability,
        mean_d,
        error_d,
        mean_n,
        error_n,
        mean_r,
        error_r,
        mean_t,
        error_t,
        mean_b,
        error_b,
    )


def _prior_decision(bounds: MultcLegacyBoundaries) -> MultcLegacyPriorDecision | None:
    if bounds.prior_response and bounds.prior_toxicity:
        return "prior_both"
    if bounds.prior_response:
        return "prior_response"
    if bounds.prior_toxicity:
        return "prior_toxicity"
    return None


def _random_trial(
    bounds: MultcLegacyBoundaries,
    probabilities: FloatArray,
    mean: float,
    window: float,
    seed: int,
    draws: int,
) -> MultcLegacyDuration:
    prior = _prior_decision(bounds)
    if prior is None:
        children = np.random.SeedSequence(seed).spawn(2)
        u = np.random.Generator(np.random.PCG64(children[0])).random(bounds.max_subjects)
        # The explicit native variate contract excludes zero; it has zero
        # probability under the mathematical continuous uniform distribution.
        np.maximum(u, np.nextafter(0.0, 1.0), out=u)
        exp = np.random.Generator(np.random.PCG64(children[1])).exponential(size=draws)
    else:
        u, exp = np.empty(0), np.empty(0)
    return _run_multc_legacy_duration(
        bounds.max_subjects,
        response_stop_at=() if bounds.prior_response else bounds.response_stop_at,
        nontoxicity_stop_at=() if bounds.prior_toxicity else bounds.nontoxicity_stop_at,
        joint_probabilities=probabilities,
        mean_interarrival=mean,
        response_window=window,
        uniforms=u,
        unit_exponentials=exp,
        prior_decision=prior,
    )


@dataclass(frozen=True)
class MultcLegacySimulation:
    """Captured Python PCG64 inputs/seeds and summaries, without patient ledgers."""

    boundaries: MultcLegacyBoundaries
    joint_probabilities: FloatArray
    mean_interarrival: float
    response_window: float
    seed: int
    max_unit_exponentials_per_trial: int
    trial_seeds: NDArray[np.uint64]
    summary: MultcLegacyDurationSummary

    def replay_trial(self, index: int) -> MultcLegacyDuration:
        """Recreate one patient ledger from its saved child seed (zero based)."""
        index = _integer(index, "index", 0, self.summary.trials - 1)
        return _random_trial(
            self.boundaries,
            self.joint_probabilities,
            self.mean_interarrival,
            self.response_window,
            int(self.trial_seeds[index]),
            self.max_unit_exponentials_per_trial,
        )


def simulate_multc_legacy_duration(
    design: MultcLeanDesign,
    *,
    joint_probabilities: ArrayLike,
    mean_interarrival: float,
    response_window: float,
    trials: int,
    seed: int,
    max_unit_exponentials_per_trial: int = 10_000,
    max_total_draws: int = 50_000_000,
    max_storage_bytes: int = 64_000_000,
) -> MultcLegacySimulation:
    """Run the recovered timing/boundary model with reproducible Python streams.

    Original means and the sample-size PMF are reported alongside MCSEs and
    owned replicate arrays. No patient ledgers are retained; individual trials
    can be replayed. Each trial preallocates a bounded exponential stream;
    exhaustion raises an error without returning a truncated study. Work and
    conservative storage limits are checked before RNG creation. PCG64 streams
    differ from the original Windows generator. The enrollment cap is not an
    adverse-outcome decision. Prior rejection consumes no outcome/time draws.
    """
    if not isinstance(design, MultcLeanDesign):
        raise ValueError("design must be a MultcLeanDesign")
    repetitions = _integer(trials, "trials", 1, 10_000)
    root_seed = _integer(seed, "seed", 0, 2**64 - 1)
    draws = _integer(
        max_unit_exponentials_per_trial, "max_unit_exponentials_per_trial", 1, 1_000_000
    )
    budget = _integer(max_total_draws, "max_total_draws", 1, 1_000_000_000)
    storage = _integer(max_storage_bytes, "max_storage_bytes", 1, 1_000_000_000)
    probabilities = _stream(joint_probabilities, "joint_probabilities", 4)
    if (
        probabilities.size != 4
        or np.any(probabilities < 0)
        or not np.isclose(probabilities.sum(), 1.0, rtol=0, atol=1e-12)
    ):
        raise ValueError("joint_probabilities must be four nonnegative probabilities summing to 1")
    probabilities = _owned(probabilities)
    mean = _real_scalar(mean_interarrival, "mean_interarrival")
    window = _real_scalar(response_window, "response_window")
    cap = design.max_subjects
    if repetitions * (cap + draws) > budget:
        raise ValueError("legacy simulation exceeds max_total_draws")
    # One duration ledger plus RNG buffers and immutable result arrays. Summary
    # copying temporaries and list/object overhead receive an additional margin.
    required_storage = repetitions * 128 + cap * 1024 + draws * 10 + 65_536
    if required_storage > storage:
        raise ValueError("legacy simulation exceeds max_storage_bytes")
    bounds = multc_legacy_boundaries(design)
    rng = np.random.Generator(np.random.PCG64(root_seed))
    trial_seeds = rng.integers(0, 2**64 - 1, size=repetitions, dtype=np.uint64, endpoint=True)
    trial_seeds.flags.writeable = False
    # Retain compact numeric trial records only. A full ledger is discarded
    # immediately after its contribution to the five summary arrays.
    records = np.empty((repetitions, 5), dtype=np.float64)
    for index, child_seed in enumerate(trial_seeds):
        trial = _random_trial(bounds, probabilities, mean, window, int(child_seed), draws)
        records[index] = (
            trial.duration,
            trial.sample_size,
            trial.responses,
            trial.toxicities,
            trial.balks,
        )
        del trial
    arrays = tuple(_owned(records[:, column]) for column in range(5))
    summary = _summarize_arrays(cap, arrays)
    return MultcLegacySimulation(
        bounds, probabilities, mean, window, root_seed, draws, trial_seeds, summary
    )
