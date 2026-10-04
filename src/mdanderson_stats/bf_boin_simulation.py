"""Calendar-time BF-BOIN simulation with asynchronous assessment."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import log_expit, logsumexp

from ._validation import FloatArray, count, finite
from .bard_response import BARDResponseModel, bard_response_probabilities
from .bf_boin import BFBOINDesign
from .boin import _owned


@dataclass(frozen=True)
class BFBOINSimulation:
    patients: NDArray[np.int64]
    toxicities: NDArray[np.int64]
    assigned: NDArray[np.int64]
    selected_dose: NDArray[np.int64]
    stop_reason: tuple[str, ...]
    selection_probability: FloatArray
    selection_mcse: FloatArray
    mean_patients: FloatArray
    mean_toxicities: FloatArray
    mean_assigned: FloatArray
    assigned_history: tuple[NDArray[np.int64], ...]
    arrival_history: tuple[FloatArray, ...]
    assessment_history: tuple[FloatArray, ...]
    dlt_history: tuple[NDArray[np.bool_], ...]
    response_history: tuple[NDArray[np.bool_], ...]
    backfill_history: tuple[NDArray[np.bool_], ...]
    escalation_end: FloatArray
    trial_duration: FloatArray
    expansion_stop_reason: tuple[str, ...]
    expansion_patients: NDArray[np.int64]
    expansion_end: FloatArray
    grade2_history: tuple[NDArray[np.bool_], ...] | None = None
    grade2_assessment_history: tuple[FloatArray, ...] | None = None
    titration_stop_reason: tuple[str, ...] | None = None
    titration_patients: NDArray[np.int64] | None = None
    titration_grade2: NDArray[np.int64] | None = None
    titration_end: FloatArray | None = None
    profile_index_history: tuple[NDArray[np.int64], ...] | None = None
    factor_history: tuple[NDArray[np.int64], ...] | None = None
    response_probability_history: tuple[FloatArray, ...] | None = None
    sampled_response_probability_history: tuple[FloatArray, ...] | None = None
    eliminated_history: tuple[NDArray[np.bool_], ...] | None = None

    @property
    def final_assessments(self) -> tuple[FloatArray, ...]:
        """Compatibility alias for per-patient DLT assessment times."""
        return self.assessment_history


@dataclass
class _PatientRecord:
    dose: int
    arrival: float
    dlt_assessment: float
    response_assessment: float
    dlt: bool
    response: bool
    grade2: bool = False
    grade2_assessment: float = np.inf
    titration: bool = False
    dlt_seen: bool = False
    response_seen: bool = False
    grade2_seen: bool = False
    backfill: bool = False
    profile_index: int | None = None
    response_probability: float | None = None
    sampled_response_probability: float | None = None


def _weibull_endpoint(probability: float, window: float) -> tuple[float, float]:
    """Calibrate Weibull F(window)=p and F(window/2)=p/2."""
    if probability == 0:
        return 1.0, np.inf
    if probability >= 1:
        return 1.0, 0.0
    a = -np.log1p(-probability)
    b = -np.log1p(-probability / 2)
    shape = np.log(a / b) / np.log(2.0)
    scale = np.exp(np.log(window) - np.log(a) / shape)
    if not np.isfinite(shape) or not np.isfinite(scale) or scale <= 0:
        raise ValueError("true_toxicity and dlt_window exceed finite Weibull calibration range")
    return float(shape), float(scale)


def _positive(value: float, name: str) -> float:
    result = float(np.asarray(value, dtype=float))
    if not np.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _bard_conditional_response_probability(
    response_probability: float,
    toxicity_probability: float,
    joint_probability: float | None,
    dlt: bool,
) -> float:
    """Return the response probability conditional on the generated DLT status."""
    if joint_probability is None:
        return response_probability
    if dlt:
        if toxicity_probability == 0:
            raise ArithmeticError("observed DLT under zero true toxicity probability")
        result = float(np.longdouble(joint_probability) / np.longdouble(toxicity_probability))
    else:
        if toxicity_probability == 1:
            raise ArithmeticError("no DLT under unit true toxicity probability")
        result = float(
            (np.longdouble(response_probability) - np.longdouble(joint_probability))
            / (1 - np.longdouble(toxicity_probability))
        )
    if not np.isfinite(result) or not 0 <= result <= 1:
        raise ArithmeticError("joint endpoint truth produced an invalid conditional probability")
    return result


def _validate_bard_response_truth(
    response_model: BARDResponseModel | None,
    true_response: FloatArray,
    true_toxicity: FloatArray,
    joint_toxicity_response_probability: ArrayLike | None,
) -> tuple[NDArray[np.int64] | None, FloatArray | None, FloatArray | None, FloatArray | None]:
    """Validate/copy a response model and optional joint endpoint probabilities."""
    if response_model is None:
        if joint_toxicity_response_probability is not None:
            raise ValueError("joint endpoint truth requires response_model")
        return None, None, None, None
    if not isinstance(response_model, BARDResponseModel):
        raise TypeError("response_model must be a BARDResponseModel")
    fields = (
        response_model.population_response,
        response_model.factor_profiles,
        response_model.profile_probabilities,
        response_model.response_odds_ratios,
        response_model.intercepts,
        response_model.conditional_probabilities,
        response_model.marginal_residuals,
    )
    if any(not isinstance(value, np.ndarray) for value in fields):
        raise ValueError(
            "response_model fields must be NumPy arrays created by bard_response_model"
        )
    target0, profiles0, weights0, odds0, intercepts0, conditional0, residuals0 = fields
    doses = true_response.size
    if (
        target0.ndim != 1
        or target0.shape != (doses,)
        or profiles0.ndim != 2
        or not 1 <= profiles0.shape[1] <= 5
        or not 1 <= profiles0.shape[0] <= 100_000
        or profiles0.shape[0] * profiles0.shape[1] > 500_000
        or weights0.shape != (profiles0.shape[0],)
        or odds0.ndim != 2
        or odds0.shape[0] != profiles0.shape[1]
        or not 1 <= odds0.shape[1] <= 20
        or intercepts0.shape != (doses,)
        or conditional0.shape != (doses, profiles0.shape[0])
        or residuals0.shape != (doses,)
        or doses * profiles0.shape[0] > 1_000_000
    ):
        raise ValueError("response_model has invalid or unsupported dimensions")
    if (
        target0.dtype.kind not in "iuf"
        or profiles0.dtype.kind not in "iu"
        or weights0.dtype.kind not in "iuf"
        or odds0.dtype.kind not in "iuf"
        or intercepts0.dtype.kind not in "iuf"
        or conditional0.dtype.kind not in "iuf"
        or residuals0.dtype.kind not in "iuf"
    ):
        raise ValueError("response_model fields must contain real numeric arrays")
    target = np.asarray(target0, dtype=np.float64)
    profiles = np.asarray(profiles0, dtype=np.int64)
    weights = np.asarray(weights0, dtype=np.float64)
    odds = np.asarray(odds0, dtype=np.float64)
    intercepts = np.asarray(intercepts0, dtype=np.float64)
    conditional = np.asarray(conditional0, dtype=np.float64)
    residuals = np.asarray(residuals0, dtype=np.float64)
    weight_total = float(np.sum(weights, dtype=np.float64))
    if (
        not np.isfinite(target).all()
        or np.any((target < 0) | (target > 1))
        or not np.array_equal(target, true_response)
        or not np.isfinite(weights).all()
        or np.any(weights < 0)
        or not np.any(weights > 0)
        or not np.isfinite(weight_total)
        or abs(weight_total - 1.0) > 1e-12
        or np.any(profiles < 1)
        or np.any(profiles > odds.shape[1])
        or not np.isfinite(odds).all()
        or np.any(odds <= 0)
        or np.any(odds[:, 0] != 1.0)
        or np.any(np.isnan(intercepts))
        or not np.isfinite(conditional).all()
        or np.any((conditional < 0) | (conditional > 1))
        or not np.isfinite(residuals).all()
    ):
        raise ValueError("response_model contains invalid values or mismatched response margins")
    expected = bard_response_probabilities(intercepts, profiles, odds)
    if not np.allclose(conditional, expected, rtol=2e-12, atol=0.0):
        raise ValueError("response_model conditional probabilities do not match its intercepts")
    # Use only recomputed probabilities from here onward. A tolerated serialized
    # table is evidence to validate, never the source of endpoint bounds.
    conditional = np.asarray(expected)
    expected_residuals = expected @ weights - target
    if not np.allclose(residuals, expected_residuals, rtol=8e-15, atol=0.0):
        raise ValueError("response_model marginal residuals do not match its inputs")
    offsets = np.zeros(profiles.shape[0], dtype=np.float64)
    log_odds = np.log(odds)
    for factor in range(profiles.shape[1]):
        offsets += log_odds[factor, profiles[:, factor] - 1]
    active = weights > 0
    log_weights = np.log(weights[active])
    for dose, margin in enumerate(target):
        if margin == 0:
            if not np.isneginf(intercepts[dose]) or np.any(expected[dose] != 0):
                raise ValueError("zero response margin requires a zero-probability model")
            continue
        if margin == 1:
            if not np.isposinf(intercepts[dose]) or np.any(expected[dose] != 1):
                raise ValueError("unit response margin requires a unit-probability model")
            continue
        linear = intercepts[dose] + offsets[active]
        if margin <= 0.5:
            observed_log_tail = float(logsumexp(log_weights + log_expit(linear)))
            target_log_tail = float(np.log(margin))
        else:
            observed_log_tail = float(logsumexp(log_weights + log_expit(-linear)))
            target_log_tail = float(np.log1p(-margin))
        if not np.isfinite(observed_log_tail) or abs(observed_log_tail - target_log_tail) > 5e-11:
            raise ValueError("response_model intercepts do not calibrate the supplied margins")

    joint: FloatArray | None = None
    if joint_toxicity_response_probability is not None:
        raw = joint_toxicity_response_probability
        expected_shape = conditional.shape
        if isinstance(raw, np.ndarray):
            if raw.ndim != 2 or raw.shape != expected_shape or raw.dtype.kind not in "iuf":
                raise ValueError(
                    "joint_toxicity_response_probability must have dose-by-profile shape"
                )
        elif isinstance(raw, (list, tuple)):
            if len(raw) != doses:
                raise ValueError(
                    "joint_toxicity_response_probability must have dose-by-profile shape"
                )
            for row in raw:
                if not isinstance(row, (list, tuple, np.ndarray)) or getattr(row, "ndim", 1) != 1:
                    raise ValueError("joint endpoint probabilities must be rectangular rows")
                if len(row) != profiles.shape[0] or any(
                    isinstance(value, (list, tuple, np.ndarray, complex, bool, np.bool_))
                    for value in row
                ):
                    raise ValueError("joint endpoint probabilities must be real scalar rows")
        else:
            raise ValueError("joint endpoint probabilities must be a bounded matrix")
        with np.errstate(over="ignore", invalid="ignore"):
            joint_array = np.asarray(raw, dtype=np.float64)
        if not np.isfinite(joint_array).all() or np.any((joint_array < 0) | (joint_array > 1)):
            raise ValueError("joint endpoint probabilities must lie in [0,1]")
        # Compute the positive lower tail as e - (1 - t), avoiding cancellation
        # in t + e - 1 when either event is rare.
        upper_event = np.maximum(true_toxicity[:, None], conditional)
        lower = np.maximum(
            0.0,
            np.minimum(true_toxicity[:, None], conditional) - (1.0 - upper_event),
        )
        upper = np.minimum(true_toxicity[:, None], conditional)
        tolerance = 4.0 * np.maximum.reduce(
            (np.abs(np.spacing(joint_array)), np.abs(np.spacing(lower)), np.abs(np.spacing(upper)))
        )
        if np.any(joint_array < lower - tolerance) or np.any(joint_array > upper + tolerance):
            raise ValueError("joint endpoint probability violates the Frechet bounds")
        joint = np.array(np.minimum(np.maximum(joint_array, lower), upper), copy=True)
    return (
        np.array(profiles, copy=True),
        np.array(weights, copy=True),
        np.array(expected, copy=True),
        joint,
    )


def simulate_bf_boin(
    design: BFBOINDesign,
    true_toxicity: ArrayLike,
    true_response: ArrayLike,
    *,
    cohorts: ArrayLike = 10,
    cohort_size: int = 3,
    trials: int = 1000,
    start_dose: int = 1,
    accrual_rate: float = 1.0,
    dlt_window: float = 1.0,
    arrival_distribution: str = "uniform",
    expand_after_escalation: bool = False,
    accelerated_titration: bool = False,
    titration_cap: int | None = None,
    true_grade2: ArrayLike | None = None,
    grade2_assessment_delay: float | None = None,
    response_model: BARDResponseModel | None = None,
    joint_toxicity_response_probability: ArrayLike | None = None,
    rng: int | np.random.Generator | None = None,
) -> BFBOINSimulation:
    """Run primary-mode BF-BOIN with separate DLT and response observation.

    Responses are sampled at enrollment and observed at arrival plus the DLT
    window. DLT times are calibrated Weibull endpoints and observed at the
    earlier event time or window. Arrivals use one persistent renewal schedule;
    the first arrival is at time zero.

    With ``expand_after_escalation=True``, a completed escalation (including
    the optional precision stop) is followed by fixed-dose expansion one level
    below the last dose actually treated for escalation. Expansion uses
    ordinary BF-BOIN backfill eligibility and ends at the target dose's
    assigned-patient cap or when it closes for toxicity. This asynchronous
    calendar continuation is an explicit Python policy; the source specifies
    the target dose and stopping conditions, but not calendar timing.

    ``accelerated_titration=True`` implements Guide Remarks 2 before the
    ordinary cohort sequence. It enrolls one patient per dose and prohibits
    backfill during this prelude. ``true_grade2`` gives the per-dose grade-2
    probability conditional on no DLT; the generated severity categories are
    mutually exclusive. Grade-2 outcomes are observed at enrollment plus
    ``grade2_assessment_delay``, a fixed Python timing policy because the guide
    does not specify grade-2 assessment timing. ``titration_cap=None`` means
    the highest dose. ``cohorts`` counts ordinary full cohorts, including a
    top-up cohort after a trigger or highest-dose cap; earlier singleton visits
    are additional. Reaching a lower cap without a trigger starts ordinary
    cohorts at the next dose.

    If ``response_model`` is supplied, each enrollment samples a factor profile
    from its explicit joint distribution and response from that dose/profile
    probability. ``true_response`` must match the model's dose margins. Without
    ``joint_toxicity_response_probability``, toxicity and response are
    conditionally independent given dose and profile. If supplied, it gives
    ``P(DLT and response | dose, profile)`` and response is sampled conditional
    on the generated DLT status. Profile and response probabilities are
    retained in enrollment order. The default no-model branch keeps the legacy
    random draws unchanged.
    """
    if not isinstance(design, BFBOINDesign):
        raise ValueError("design must be a BFBOINDesign")
    toxicity, response = (
        finite(true_toxicity, "true_toxicity"),
        finite(true_response, "true_response"),
    )
    if (
        toxicity.ndim != 1
        or not 2 <= toxicity.size <= 100
        or response.shape != toxicity.shape
        or np.any((toxicity < 0) | (toxicity > 1))
        or np.any((response < 0) | (response > 1))
    ):
        raise ValueError("true_toxicity and true_response must match with probabilities in [0,1]")
    model_profiles, model_weights, model_response, joint_response = _validate_bard_response_truth(
        response_model,
        response,
        toxicity,
        joint_toxicity_response_probability,
    )
    sizes = count([cohort_size, trials, start_dose], "simulation sizes")
    if sizes.shape != (3,) or np.any(sizes < 1):
        raise ValueError("cohort_size, trials and start_dose must be positive integers")
    size, repetitions, start = map(int, sizes)
    if repetitions > 1_000_000 or start > toxicity.size:
        raise ValueError("require at most 1000000 trials and a valid start_dose")
    cohort_counts = np.broadcast_to(count(cohorts, "cohorts"), (repetitions,))
    if np.any(cohort_counts < 1):
        raise ValueError("cohorts must contain positive integers")
    rate, window = _positive(accrual_rate, "accrual_rate"), _positive(dlt_window, "dlt_window")
    if arrival_distribution not in ("uniform", "exponential"):
        raise ValueError("arrival_distribution must be 'uniform' or 'exponential'")
    if not isinstance(expand_after_escalation, (bool, np.bool_)):
        raise ValueError("expand_after_escalation must be Boolean")
    if not isinstance(accelerated_titration, (bool, np.bool_)):
        raise ValueError("accelerated_titration must be Boolean")
    if accelerated_titration:
        if titration_cap is not None and np.iscomplexobj(np.asarray(titration_cap)):
            raise ValueError("titration_cap must be a real integer dose level")
        cap_values = count(
            toxicity.size if titration_cap is None else titration_cap,
            "titration_cap",
        )
        if cap_values.shape != () or not start <= int(cap_values) <= toxicity.size:
            raise ValueError("titration_cap must be between start_dose and the highest dose")
        cap = int(cap_values)
        if true_grade2 is None or grade2_assessment_delay is None:
            raise ValueError(
                "accelerated titration requires true_grade2 and grade2_assessment_delay"
            )
        if np.iscomplexobj(np.asarray(true_grade2)):
            raise ValueError("true_grade2 must be real-valued")
        if np.iscomplexobj(np.asarray(grade2_assessment_delay)):
            raise ValueError("grade2_assessment_delay must be real-valued")
        grade2 = finite(true_grade2, "true_grade2")
        delay = _positive(grade2_assessment_delay, "grade2_assessment_delay")
        if grade2.shape != toxicity.shape or np.any((grade2 < 0) | (grade2 > 1)):
            raise ValueError("true_grade2 must match doses and contain probabilities in [0,1]")
    else:
        if (
            titration_cap is not None
            or true_grade2 is not None
            or grade2_assessment_delay is not None
        ):
            raise ValueError("titration settings require accelerated_titration=True")
        cap, grade2, delay = toxicity.size, np.zeros_like(toxicity), np.inf
    # The singletons before a trigger/cap can be additional to the ordinary
    # cohort budget. Reserve the entire possible staircase plus one cohort.
    titration_allowance = cap - start + 1 + size if accelerated_titration else 0
    bound = repetitions * (
        int(np.max(cohort_counts)) * size + toxicity.size * design.n_cap + titration_allowance
    )
    if bound > 100_000:
        raise ValueError("requested trials and retained patient records exceed 100000")
    max_records = int(np.max(cohort_counts)) * size + toxicity.size * design.n_cap
    if max_records + titration_allowance > 1_000:
        raise ValueError("a trial may retain at most 1000 patient records")
    if model_profiles is not None:
        profile_history_cells = bound * (model_profiles.shape[1] + 3)
        if profile_history_cells > 1_000_000:
            raise ValueError("response-profile histories exceed the 1000000-cell retention limit")
    generator = np.random.default_rng(rng)
    endpoints = tuple(_weibull_endpoint(float(p), window) for p in toxicity)
    patients = np.zeros((repetitions, toxicity.size), dtype=np.int64)
    toxicities = np.zeros_like(patients)
    assigned = np.zeros_like(patients)
    selected: NDArray[np.int64] = np.zeros(repetitions, dtype=np.int64)
    reasons, dose_histories, arrival_histories = [], [], []
    assessment_histories, dlt_histories, response_histories, backfill_histories = [], [], [], []
    expansion_reasons: list[str] = []
    expansion_counts = np.zeros(repetitions, dtype=np.int64)
    expansion_ends = np.full(repetitions, np.nan)
    escalation_ends, durations = np.zeros(repetitions), np.zeros(repetitions)
    grade2_histories: list[NDArray[np.bool_]] = []
    grade2_assessment_histories: list[FloatArray] = []
    titration_reasons: list[str] = []
    titration_counts = np.zeros(repetitions, dtype=np.int64)
    titration_grade2_counts = np.zeros(repetitions, dtype=np.int64)
    titration_ends = np.full(repetitions, np.nan)
    profile_index_histories: list[NDArray[np.int64]] = []
    factor_histories: list[NDArray[np.int64]] = []
    response_probability_histories: list[FloatArray] = []
    sampled_response_probability_histories: list[FloatArray] = []
    eliminated_histories: list[NDArray[np.bool_]] = []

    for trial in range(repetitions):
        evaluated = np.zeros(toxicity.size, dtype=np.int64)
        observed_dlt = np.zeros_like(evaluated)
        activity, excluded = np.zeros(toxicity.size, bool), np.zeros(toxicity.size, bool)
        # dose, arrival, dlt assessment, response assessment, dlt, response,
        # dlt_seen, response_seen, backfill
        records: list[_PatientRecord] = []
        titration_grade2_seen = 0
        next_arrival, clock, dose = 0.0, 0.0, start
        last_escalation_dose: int | None = None
        reason = "max_cohorts"
        arrival_steps = 0
        arrival_limit = 100_000

        def advance_arrival() -> float:
            nonlocal next_arrival, arrival_steps
            if arrival_steps >= arrival_limit:
                raise RuntimeError("BF-BOIN calendar arrival limit (100000) exceeded")
            arrival = next_arrival
            gap = (
                generator.uniform(0, 2 / rate)
                if arrival_distribution == "uniform"
                else generator.exponential(1 / rate)
            )
            next_arrival = arrival + gap
            if not np.isfinite(next_arrival) or next_arrival <= arrival:
                raise RuntimeError("BF-BOIN calendar failed to advance at an arrival")
            arrival_steps += 1
            return arrival

        def observe(until: float) -> None:
            nonlocal titration_grade2_seen
            for r in records:
                if not r.dlt_seen and r.dlt_assessment <= until:
                    evaluated[r.dose] += 1
                    observed_dlt[r.dose] += int(r.dlt)
                    r.dlt_seen = True
                if not r.response_seen and r.response_assessment <= until:
                    activity[r.dose] |= r.response
                    r.response_seen = True
                if not r.grade2_seen and r.grade2_assessment <= until:
                    r.grade2_seen = True
                    if r.titration:
                        titration_grade2_seen += int(r.grade2)

        def enroll(
            j: int,
            when: float,
            backfill: bool,
            *,
            titration: bool = False,
        ) -> int:
            if not np.isfinite(when + window):
                raise RuntimeError("BF-BOIN assessment time exceeds floating-point range")
            shape, scale = endpoints[j]
            dlt_time = (
                np.inf
                if np.isinf(scale)
                else (window / 2 if scale == 0 else scale * generator.weibull(shape))
            )
            dlt_value = dlt_time <= window
            if model_profiles is None:
                # Keep the original one-draw response path exactly when the
                # optional profile model is absent.
                response_value = bool(generator.random() < response[j])
                profile_index = None
                response_probability = None
                sampled_response_probability = None
            else:
                assert model_weights is not None and model_response is not None
                profile_index = int(generator.choice(model_weights.size, p=model_weights))
                response_probability = float(model_response[j, profile_index])
                sampled_response_probability = _bard_conditional_response_probability(
                    response_probability,
                    float(toxicity[j]),
                    None if joint_response is None else float(joint_response[j, profile_index]),
                    bool(dlt_value),
                )
                response_value = bool(generator.random() < sampled_response_probability)
            grade2_value = bool(
                accelerated_titration and dlt_time > window and generator.random() < grade2[j]
            )
            grade2_time = when + delay if accelerated_titration else np.inf
            if accelerated_titration and not np.isfinite(grade2_time):
                raise RuntimeError("BF-BOIN grade-2 assessment time exceeds floating-point range")
            records.append(
                _PatientRecord(
                    j,
                    when,
                    when + min(dlt_time, window),
                    when + window,
                    dlt_time <= window,
                    response_value,
                    grade2_value,
                    grade2_time,
                    titration,
                    backfill=backfill,
                    profile_index=profile_index,
                    response_probability=response_probability,
                    sampled_response_probability=sampled_response_probability,
                )
            )
            assigned[trial, j] += 1
            return len(records) - 1

        initial_cohort: list[int] = []
        titration_reason = "not_requested"
        if accelerated_titration:
            while True:
                clock = advance_arrival()
                observe(clock)
                current_patient = enroll(dose - 1, clock, False, titration=True)
                titration_counts[trial] += 1
                last_escalation_dose = dose

                # The default highest-dose cap ends the singleton prelude on
                # enrollment; the first full cohort is topped up immediately.
                if dose == cap == toxicity.size:
                    titration_reason = "highest_cap"
                    initial_cohort = [current_patient]
                    break

                record = records[current_patient]
                while not (record.dlt_seen and record.grade2_seen):
                    next_assessment = min(
                        record.dlt_assessment if not record.dlt_seen else np.inf,
                        record.grade2_assessment if not record.grade2_seen else np.inf,
                    )
                    if next_arrival < next_assessment:
                        # Consume but do not enroll arrivals while the
                        # singleton is under assessment: titration has no
                        # backfill or parallel patients.
                        advance_arrival()
                        continue
                    clock = next_assessment
                    observe(clock)
                    if record.dlt_seen and record.dlt:
                        titration_reason = "first_dlt"
                        initial_cohort = [current_patient]
                        break
                    if titration_grade2_seen >= 2:
                        titration_reason = "second_grade2"
                        initial_cohort = [current_patient]
                        break
                if initial_cohort:
                    break
                if dose == cap:
                    titration_reason = "lower_cap_no_trigger"
                    dose = cap + 1
                    break
                dose += 1
            titration_grade2_counts[trial] = titration_grade2_seen
            titration_ends[trial] = clock

        for cohort_index in range(int(cohort_counts[trial])):
            current: list[int] = initial_cohort.copy() if cohort_index == 0 else []
            while len(current) < size:
                clock = advance_arrival()
                observe(clock)
                current.append(enroll(dose - 1, clock, False))
                last_escalation_dose = dose
            while not all(records[i].dlt_seen for i in current):
                next_dlt = min(
                    records[i].dlt_assessment for i in current if not records[i].dlt_seen
                )
                if next_arrival < next_dlt:
                    clock = advance_arrival()
                    observe(clock)
                    eligibility = design.backfill_eligibility(
                        evaluated,
                        observed_dlt,
                        assigned[trial],
                        dose,
                        response_observed=activity,
                        eliminated=excluded,
                    )
                    if eligibility.dose is not None:
                        enroll(eligibility.dose - 1, clock, True)
                    # An ineligible arrival is referred away, but still
                    # advances the same renewal process.
                else:
                    clock = next_dlt
                    observe(clock)
            decision = design.next_dose(
                evaluated,
                observed_dlt,
                assigned[trial],
                dose,
                backfilled=np.array(
                    [any(r.dose == j and r.backfill for r in records) for j in range(toxicity.size)]
                ),
                eliminated=excluded,
                response_observed=activity,
            )
            excluded = np.array(decision.eliminated, copy=True)
            if decision.next_dose is None:
                reason = decision.action
                break
            dose = decision.next_dose
        escalation_ends[trial] = clock

        expansion_reason = "not_requested"
        if expand_after_escalation:
            expansion_ends[trial] = clock
            if reason == "stop_safety":
                expansion_reason = "safety_stopped"
            elif last_escalation_dose is None or last_escalation_dose <= 1:
                expansion_reason = "no_lower_dose"
            else:
                expansion_dose = last_escalation_dose - 1
                expansion_index = expansion_dose - 1
                expansion_reason = "activity_unavailable"
                observe(clock)
                while True:
                    if assigned[trial, expansion_index] >= design.n_cap:
                        expansion_reason = "assigned_cap"
                        break

                    eligibility = design.backfill_eligibility(
                        evaluated,
                        observed_dlt,
                        assigned[trial],
                        last_escalation_dose,
                        response_observed=activity,
                        eliminated=excluded,
                    )
                    if eligibility.closed[expansion_index]:
                        expansion_reason = "toxicity_closed"
                        break

                    next_observation = min(
                        (
                            time
                            for record in records
                            for time, seen in (
                                (record.dlt_assessment, record.dlt_seen),
                                (record.response_assessment, record.response_seen),
                            )
                            if not seen
                        ),
                        default=np.inf,
                    )
                    if not eligibility.eligible[expansion_index]:
                        if not np.isfinite(next_observation):
                            expansion_reason = "activity_unavailable"
                            break
                        while next_arrival < next_observation:
                            clock = advance_arrival()
                        clock = next_observation
                        observe(clock)
                        continue

                    if next_arrival < next_observation:
                        clock = advance_arrival()
                        observe(clock)
                        if assigned[trial, expansion_index] >= design.n_cap:
                            expansion_reason = "assigned_cap"
                            break
                        eligibility = design.backfill_eligibility(
                            evaluated,
                            observed_dlt,
                            assigned[trial],
                            last_escalation_dose,
                            response_observed=activity,
                            eliminated=excluded,
                        )
                        if eligibility.closed[expansion_index]:
                            expansion_reason = "toxicity_closed"
                            break
                        if eligibility.eligible[expansion_index]:
                            enroll(expansion_index, clock, True)
                            expansion_counts[trial] += 1
                        continue

                    if not np.isfinite(next_observation):
                        raise RuntimeError("eligible expansion has no finite calendar event")
                    clock = next_observation
                    observe(clock)

                expansion_ends[trial] = clock
        expansion_reasons.append(expansion_reason)

        observe(np.inf)
        patients[trial], toxicities[trial] = evaluated, observed_dlt
        durations[trial] = max(
            (
                max(r.response_assessment, r.grade2_assessment)
                if accelerated_titration
                else r.response_assessment
                for r in records
            ),
            default=clock,
        )
        selection = design.select_mtd(evaluated, observed_dlt, eliminated=excluded)
        selected[trial] = 0 if selection.dose is None else selection.dose
        reasons.append(reason)
        dose_histories.append(_owned(np.asarray([r.dose + 1 for r in records], dtype=np.int64)))
        arrival_histories.append(_owned(np.asarray([r.arrival for r in records])))
        assessment_histories.append(_owned(np.asarray([r.dlt_assessment for r in records])))
        dlt_histories.append(_owned(np.asarray([r.dlt for r in records])))
        response_histories.append(_owned(np.asarray([r.response for r in records])))
        backfill_histories.append(_owned(np.asarray([r.backfill for r in records])))
        if model_profiles is not None:
            profile_indices = np.asarray([r.profile_index for r in records], dtype=np.int64)
            profile_index_histories.append(_owned(profile_indices))
            factor_histories.append(_owned(model_profiles[profile_indices]))
            response_probability_histories.append(
                _owned(np.asarray([r.response_probability for r in records], dtype=np.float64))
            )
            sampled_response_probability_histories.append(
                _owned(
                    np.asarray([r.sampled_response_probability for r in records], dtype=np.float64)
                )
            )
            eliminated_histories.append(_owned(selection.eliminated))
        if accelerated_titration:
            grade2_histories.append(_owned(np.asarray([r.grade2 for r in records])))
            grade2_assessment_histories.append(
                _owned(np.asarray([r.grade2_assessment for r in records]))
            )
        titration_reasons.append(titration_reason)
    frequency = np.bincount(selected, minlength=toxicity.size + 1) / repetitions
    return BFBOINSimulation(
        _owned(patients),
        _owned(toxicities),
        _owned(assigned),
        _owned(selected),
        tuple(reasons),
        _owned(frequency),
        _owned(np.sqrt(frequency * (1 - frequency) / repetitions)),
        _owned(patients.mean(0)),
        _owned(toxicities.mean(0)),
        _owned(assigned.mean(0)),
        tuple(dose_histories),
        tuple(arrival_histories),
        tuple(assessment_histories),
        tuple(dlt_histories),
        tuple(response_histories),
        tuple(backfill_histories),
        _owned(escalation_ends),
        _owned(durations),
        tuple(expansion_reasons),
        _owned(expansion_counts),
        _owned(expansion_ends),
        tuple(grade2_histories) if accelerated_titration else None,
        tuple(grade2_assessment_histories) if accelerated_titration else None,
        tuple(titration_reasons) if accelerated_titration else None,
        _owned(titration_counts) if accelerated_titration else None,
        _owned(titration_grade2_counts) if accelerated_titration else None,
        _owned(titration_ends) if accelerated_titration else None,
        tuple(profile_index_histories) if model_profiles is not None else None,
        tuple(factor_histories) if model_profiles is not None else None,
        tuple(response_probability_histories) if model_profiles is not None else None,
        tuple(sampled_response_probability_histories) if model_profiles is not None else None,
        tuple(eliminated_histories) if model_profiles is not None else None,
    )
