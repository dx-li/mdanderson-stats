"""BOP2-DC monitoring and operating characteristics for a Normal endpoint.

The continuous-endpoint model follows §2.1.2 of the BOP2-DC paper: a Normal
mean with a Normal-Inverse-Gamma prior. Prior and truth parameters are explicit;
no application defaults or native continuous-endpoint executable parity are
inferred here.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import t as student_t

from ._validation import FloatArray, scalar
from .boin import _owned
from .bop2_dc_survival_trial import _mean, _replay_seed, _sample_mcse
from .normal_updating import NormalInverseGamma, NormalSample

_MAX_SUBJECTS = 1000
_MAX_SIMULATION_TRIALS = 100_000
_MAX_SIMULATION_PATIENT_CELLS = 1_000_000
_MAX_LOOK_WORK = 30_000_000
_DECISIONS = ("stop_no_go", "final_go", "final_consider", "final_no_go")


def _observation_vector(value: ArrayLike, maximum: int, *, exact: bool = False) -> FloatArray:
    """Validate a bounded one-dimensional real vector before numeric copying."""
    shape = getattr(value, "shape", None)
    if shape is not None:
        if len(shape) != 1:
            raise ValueError("observations must be a one-dimensional vector")
        size = int(shape[0])
        if not 1 <= size <= maximum or (exact and size != maximum):
            raise ValueError("observations have an invalid length")
        if np.iscomplexobj(value):
            raise ValueError("observations must be real")
        raw = np.asarray(value)
        if raw.dtype.kind == "O" and any(not np.isscalar(item) for item in raw):
            raise ValueError("observations must be a one-dimensional vector")
    else:
        if not isinstance(value, (list, tuple)):
            raise ValueError("observations must be a one-dimensional vector")
        size = len(value)
        if not 1 <= size <= maximum or (exact and size != maximum):
            raise ValueError("observations have an invalid length")
        if any(not np.isscalar(item) for item in value):
            raise ValueError("observations must be a one-dimensional vector")
        if any(np.iscomplexobj(item) for item in value):
            raise ValueError("observations must be real")
    values = np.asarray(value, dtype=np.float64)
    if values.shape != (size,) or np.any(~np.isfinite(values)):
        raise ValueError("observations must be a finite one-dimensional vector")
    return values


def _real_scalar(value: float, name: str) -> float:
    if np.iscomplexobj(value):
        raise ValueError(f"{name} must be real")
    return scalar(value, name)


@dataclass(frozen=True)
class BOP2DCNormalState:
    """Posterior summary and dual-criterion decision at one analysis."""

    sample_size: int
    posterior_location_centered: float
    location_offset: float
    posterior_df: float
    posterior_scale: float
    posterior_lrv: float
    posterior_cmv: float
    decision: str

    @property
    def posterior_location(self) -> float:
        """Absolute mean; at very large offsets, use centered value and offset separately."""
        value = self.posterior_location_centered + self.location_offset
        if not np.isfinite(value):
            raise ArithmeticError("absolute posterior location is not representable")
        return value


@dataclass(frozen=True)
class BOP2DCNormalDesign:
    """Single-arm Normal endpoint design with explicit Normal-Inverse-Gamma prior."""

    max_subjects: int
    theta_lrv: float
    theta_cmv: float
    lambda_lrv: float
    lambda_cmv: float
    gamma_lrv: float
    gamma_cmv: float
    prior_mean: float
    prior_precision: float
    prior_shape: float
    prior_scale: float
    looks: NDArray[np.int64]

    def monitor(self, observations: ArrayLike) -> BOP2DCNormalState:
        """Evaluate complete continuous observations at or between configured looks."""
        values = _observation_vector(observations, self.max_subjects)
        n = int(values.size)
        offset = float(values[0])
        with np.errstate(over="ignore", invalid="ignore"):
            centered_values = values - offset
            centered_prior_mean = self.prior_mean - offset
            centered_lrv = self.theta_lrv - offset
            centered_cmv = self.theta_cmv - offset
        if any(
            np.any(~np.isfinite(item))
            for item in (centered_values, centered_prior_mean, centered_lrv, centered_cmv)
        ):
            raise ArithmeticError("Normal endpoint centering is not representable")
        sample = NormalSample.from_data(centered_values)
        posterior = NormalInverseGamma(
            centered_prior_mean, self.prior_precision, self.prior_shape, self.prior_scale
        ).update(sample)
        location = float(posterior.location)
        df = float(2 * posterior.shape)
        scale = float(posterior.mean_scale)
        probability_lrv = float(student_t.sf(centered_lrv, df, loc=location, scale=scale))
        probability_cmv = float(student_t.sf(centered_cmv, df, loc=location, scale=scale))
        if not (np.isfinite(probability_lrv) and np.isfinite(probability_cmv)):
            raise ArithmeticError("Normal posterior tail probability is not representable")
        decision = "continue"
        if n == self.max_subjects:
            go = probability_lrv > self.lambda_lrv and probability_cmv > self.lambda_cmv
            no_go = probability_lrv < self.lambda_lrv and probability_cmv < self.lambda_cmv
            decision = "final_go" if go else "final_no_go" if no_go else "final_consider"
        elif n in self.looks[:-1]:
            cutoff_lrv = self.lambda_lrv * (n / self.max_subjects) ** self.gamma_lrv
            cutoff_cmv = self.lambda_cmv * (n / self.max_subjects) ** self.gamma_cmv
            if probability_lrv < cutoff_lrv and probability_cmv < cutoff_cmv:
                decision = "stop_no_go"
        return BOP2DCNormalState(
            n, location, offset, df, scale, probability_lrv, probability_cmv, decision
        )


@dataclass(frozen=True)
class BOP2DCNormalTrial:
    """Compact complete-outcome replay with states only at reached looks."""

    states: tuple[BOP2DCNormalState, ...]
    enrolled: int
    decision: str


@dataclass(frozen=True)
class BOP2DCNormalSimulation:
    """Bounded Monte Carlo operating characteristics for a Normal truth."""

    trials: int
    decision_labels: tuple[str, ...]
    decision_count: NDArray[np.int64]
    decision_probability: FloatArray
    decision_mcse: FloatArray
    mean_enrollment: float
    enrollment_mcse: float
    sample_size: NDArray[np.int64]
    decision: NDArray[np.str_]
    rng_seed: int


def bop2_dc_normal_design(
    max_subjects: int,
    theta_lrv: float,
    theta_cmv: float,
    *,
    prior_mean: float,
    prior_precision: float,
    prior_shape: float,
    prior_scale: float,
    lambda_lrv: float = 0.9,
    lambda_cmv: float = 0.5,
    gamma_lrv: float = 0.5,
    gamma_cmv: float = 0.5,
    looks: ArrayLike | None = None,
    min_subjects: int = 10,
    cohort_size: int = 5,
) -> BOP2DCNormalDesign:
    """Construct a Normal endpoint design using explicit NIG prior parameters.

    `prior_mean`, `prior_precision`, `prior_shape`, and `prior_scale` map to the
    paper's `(theta0, n0, a2, b2)`. All prior parameters must be positive except
    the location, which may be any finite real value.
    """
    n_value = scalar(max_subjects, "max_subjects")
    n = int(n_value)
    if n_value != n or not 1 <= n <= _MAX_SUBJECTS:
        raise ValueError(f"max_subjects must be an integer in [1,{_MAX_SUBJECTS}]")
    low, high = _real_scalar(theta_lrv, "theta_lrv"), _real_scalar(theta_cmv, "theta_cmv")
    if not low < high:
        raise ValueError("require theta_lrv < theta_cmv")
    ll, lc = _real_scalar(lambda_lrv, "lambda_lrv"), _real_scalar(lambda_cmv, "lambda_cmv")
    gl, gc = _real_scalar(gamma_lrv, "gamma_lrv"), _real_scalar(gamma_cmv, "gamma_cmv")
    if not 0 < ll < 1 or not 0 < lc < 1 or not 0 <= gl <= 1 or not 0 <= gc <= 1:
        raise ValueError("lambda values must lie in (0,1) and gamma values in [0,1]")
    m0 = _real_scalar(prior_mean, "prior_mean")
    k0, a0, b0 = (
        _real_scalar(prior_precision, "prior_precision"),
        _real_scalar(prior_shape, "prior_shape"),
        _real_scalar(prior_scale, "prior_scale"),
    )
    if min(k0, a0, b0) <= 0:
        raise ValueError("prior_precision, prior_shape and prior_scale must be positive")
    if looks is None:
        first_v, step_v = scalar(min_subjects, "min_subjects"), scalar(cohort_size, "cohort_size")
        first, step = int(first_v), int(step_v)
        if first_v != first or step_v != step or not 1 <= first <= n or step < 1:
            raise ValueError("invalid min_subjects/cohort_size schedule")
        schedule = np.unique(np.r_[np.arange(first, n, step), n]).astype(np.int64)
    else:
        shape = getattr(looks, "shape", None)
        if shape is None:
            if not isinstance(looks, (list, tuple)):
                raise ValueError("looks must be a one-dimensional schedule")
            if len(looks) == 0 or len(looks) > n:
                raise ValueError("looks must be a nonempty schedule no longer than max_subjects")
            if any(not np.isscalar(item) for item in looks):
                raise ValueError("looks must be a one-dimensional schedule")
        elif len(shape) != 1 or int(shape[0]) == 0 or int(shape[0]) > n:
            raise ValueError("looks must be a nonempty schedule no longer than max_subjects")
        if np.iscomplexobj(looks):
            raise ValueError("looks must be real")
        raw = np.asarray(looks)
        if raw.ndim != 1 or raw.size == 0 or raw.size > n:
            raise ValueError("looks must be a nonempty one-dimensional schedule")
        if np.iscomplexobj(raw):
            raise ValueError("looks must be real")
        numeric = np.asarray(raw, dtype=np.float64)
        if np.any(~np.isfinite(numeric)) or np.any(numeric != np.floor(numeric)):
            raise ValueError("looks must be finite integers")
        schedule = numeric.astype(np.int64)
        if (
            np.any(schedule < 1)
            or np.any(schedule > n)
            or np.any(np.diff(schedule) <= 0)
            or schedule[-1] != n
        ):
            raise ValueError("looks must increase strictly and end at max_subjects")
    if ll * (schedule[0] / n) ** gl == 0 or lc * (schedule[0] / n) ** gc == 0:
        raise ArithmeticError("interim cutoff underflows; increase cutoff scales")
    return BOP2DCNormalDesign(n, low, high, ll, lc, gl, gc, m0, k0, a0, b0, _owned(schedule))


def run_bop2_dc_normal_trial(
    design: BOP2DCNormalDesign, observations: ArrayLike
) -> BOP2DCNormalTrial:
    """Replay a fully observed outcome vector, stopping at the first interim no-go."""
    if not isinstance(design, BOP2DCNormalDesign):
        raise ValueError("design must be a BOP2DCNormalDesign")
    values = _observation_vector(observations, design.max_subjects, exact=True)
    states: list[BOP2DCNormalState] = []
    for raw_n in design.looks:
        state = design.monitor(values[: int(raw_n)])
        states.append(state)
        if state.decision != "continue":
            break
    return BOP2DCNormalTrial(tuple(states), states[-1].sample_size, states[-1].decision)


def simulate_bop2_dc_normal(
    design: BOP2DCNormalDesign,
    true_mean: float,
    true_sd: float,
    *,
    n_trials: int = 10_000,
    rng: np.random.Generator | int | None = None,
) -> BOP2DCNormalSimulation:
    """Simulate a complete-data Normal trial with serial independent trials.

    The returned seed recreates the full call. No accrual-time process is
    modeled; outcomes are complete and available at each configured analysis.
    """
    if not isinstance(design, BOP2DCNormalDesign):
        raise ValueError("design must be a BOP2DCNormalDesign")
    trial_value = _real_scalar(n_trials, "n_trials")
    trials = int(trial_value)
    if trial_value != trials or not 1 <= trials <= _MAX_SIMULATION_TRIALS:
        raise ValueError(f"n_trials must be an integer in [1,{_MAX_SIMULATION_TRIALS}]")
    mean, sd = _real_scalar(true_mean, "true_mean"), _real_scalar(true_sd, "true_sd")
    if sd <= 0:
        raise ValueError("true_sd must be positive")
    n = design.max_subjects
    if trials * n > _MAX_SIMULATION_PATIENT_CELLS:
        raise ValueError("simulation exceeds the patient-work budget")
    look_work = trials * int(np.sum(design.looks, dtype=np.int64))
    if look_work > _MAX_LOOK_WORK:
        raise ValueError("simulation exceeds the repeated-look work budget")
    seed = _replay_seed(rng)
    generator = np.random.default_rng(seed)
    sizes = np.empty(trials, dtype=np.int64)
    decisions = np.empty(trials, dtype="U16")
    for i in range(trials):
        outcomes = generator.normal(mean, sd, size=n)
        if np.any(~np.isfinite(outcomes)):
            raise ArithmeticError("simulated Normal outcomes are not finite")
        trial = run_bop2_dc_normal_trial(design, outcomes)
        sizes[i], decisions[i] = trial.enrolled, trial.decision
    counts = np.asarray([np.count_nonzero(decisions == label) for label in _DECISIONS])
    probability = counts.astype(np.float64) / trials
    return BOP2DCNormalSimulation(
        trials,
        _DECISIONS,
        _owned(counts),
        _owned(probability),
        _owned(np.sqrt(probability * (1 - probability) / trials)),
        _mean(sizes.astype(np.float64)),
        _sample_mcse(sizes.astype(np.float64)),
        _owned(sizes),
        _owned(decisions),
        seed,
    )
