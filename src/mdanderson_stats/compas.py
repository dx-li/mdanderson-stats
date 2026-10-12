"""Explicitly configured ComPAS hierarchy recovered from PNOC022 v3.0.

This profile is not the original app's unspecified default configuration or a
complete PNOC022 platform simulator. See docs/compas.md for source ambiguities.
"""

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral, Real

from scipy.special import logsumexp

from ._compas_quadrature import compas_group_integral

COMPAS_SHRINKAGE_WEIGHTS = (0.999, 0.75, 0.5, 0.25, 0.001)
COMPAS_PROFILE = "pnoc022_explicit_hierarchy_2023"
_PNOC_VARIANCES = {
    "cohorts_1_2": (0.00012, 0.042, 0.126, 0.379, 126.1),
    "cohort_3": (0.00007, 0.023, 0.068, 0.204, 67.96),
}
_LIMITS = (
    "Borrowing groups and model prior probabilities are explicit caller choices; original "
    "defaults are unrecovered.",
    "Group means have independent Normal(0, variance=1000000) priors. Arm logits are "
    "conditionally independent given group means and candidate variance.",
    "MAP model selection, not model averaging. Numerical error estimates are not posterior "
    "credible intervals.",
    "Strict .1 futility and .95 graduation rules follow the protocol equations; cohort-3 "
    "prose also mentions .8.",
    "Completed binary outcomes only; pending outcomes must not be counted as failures.",
    "No changing-arm retention, replacement ordering, final selection, minimization "
    "allocation or optional cross-cohort borrowing is specified here.",
    "Independent mathematical implementation, not exact deployed-app parity or clinical "
    "validation.",
)


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number")
    try:
        number = float(value)
    except OverflowError as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def compas_variances(maximum_n: int, alternative_response: float) -> tuple[float, ...]:
    """Calibrate five variances using design maximum N and alternative rate.

    These are fixed design inputs, not interim N or an observed response rate.
    The printed PNOC tuples are separately available and contain discrepancies.
    """
    if (
        isinstance(maximum_n, bool)
        or not isinstance(maximum_n, Integral)
        or not 1 <= int(maximum_n) <= 1000000
    ):
        raise ValueError("maximum_n must be an integer from 1 to 1000000")
    p = _number(alternative_response, "alternative_response")
    if not 0 < p < 1:
        raise ValueError("alternative_response must lie strictly between 0 and 1")
    denominator = int(maximum_n) * p * (1 - p)
    if any(denominator * w == 0 for w in COMPAS_SHRINKAGE_WEIGHTS):
        raise ValueError("variance calibration denominator underflows")
    result = tuple((1 - w) / (denominator * w) for w in COMPAS_SHRINKAGE_WEIGHTS)
    if not all(math.isfinite(v) and v > 0 for v in result):
        raise ValueError("calibrated variances are not finite positive numbers")
    return result


def compas_pnoc022_variances(cohort: str) -> tuple[float, ...]:
    """Return the literal reported tuple, without correcting conflicting prose.

    cohorts_1_2 tracks N=33,p=.60 rather than sizing p=.65. cohort_3 approximately
    tracks N=70,p=.70, not stated maximum N=42; this does not establish intent.
    """
    if not isinstance(cohort, str) or cohort not in _PNOC_VARIANCES:
        raise ValueError("cohort must be cohorts_1_2 or cohort_3")
    return _PNOC_VARIANCES[cohort]


def compas_decisions(probabilities: Sequence[float], *, equality_policy: str) -> tuple[str, ...]:
    """Strict protocol rules with a required choice for exact .1/.95 equality."""
    if equality_policy not in ("hold", "continue"):
        raise ValueError("equality_policy must be hold or continue")
    try:
        values = tuple(_number(v, "probability") for v in probabilities)
    except TypeError as exc:
        raise ValueError("probabilities must be a nonempty sequence") from exc
    if not 1 <= len(values) <= 12 or any(not 0 <= v <= 1 for v in values):
        raise ValueError("provide 1 to 12 probabilities in [0,1]")
    return tuple(
        "drop"
        if p < 0.1
        else "graduate"
        if p > 0.95
        else equality_policy
        if p in (0.1, 0.95)
        else "continue"
        for p in values
    )


@dataclass(frozen=True)
class ComPASStudy:
    """One fully specified interim fit; groups/model priors have no inferred defaults.

    Numerical scope: 1–12 arms, at most6 per group, completed N<=100 per arm,
    five variances in [1e-6,1000]. These are implementation limits, not protocol
    restrictions. Ordering identifies arms consistently across saved reports.
    """

    completed: tuple[int, ...]
    successes: tuple[int, ...]
    groups: tuple[str, ...]
    variances: tuple[float, ...]
    model_prior: tuple[float, ...]
    null_response: float
    equality_policy: str
    quadrature_tolerance: float = 1e-7

    def __post_init__(self) -> None:
        for name in ("completed", "successes", "groups", "variances", "model_prior"):
            value = getattr(self, name)
            if isinstance(value, (str, bytes, dict)):
                raise ValueError(f"{name} must be a sequence")
            try:
                object.__setattr__(self, name, tuple(value))
            except TypeError as exc:
                raise ValueError(f"{name} must be a sequence") from exc
        if (
            not 1 <= len(self.completed) <= 12
            or len(self.successes) != len(self.completed)
            or len(self.groups) != len(self.completed)
        ):
            raise ValueError("completed, successes and groups must have equal lengths from 1 to 12")
        for name in ("completed", "successes"):
            values = getattr(self, name)
            if any(
                isinstance(v, bool) or not isinstance(v, Integral) or not 0 <= int(v) <= 100
                for v in values
            ):
                raise ValueError(f"{name} must contain integer counts from 0 to 100")
            object.__setattr__(self, name, tuple(int(v) for v in values))
        if any(y > n for y, n in zip(self.successes, self.completed)):
            raise ValueError("successes cannot exceed completed outcomes")
        if any(not isinstance(g, str) or not g.strip() or len(g) > 64 for g in self.groups):
            raise ValueError("group identifiers must be nonempty strings of at most64 characters")
        if any(self.groups.count(g) > 6 for g in self.groups):
            raise ValueError("at most6 arms per borrowing group are supported")
        for name in ("variances", "model_prior"):
            values = tuple(_number(v, name) for v in getattr(self, name))
            if len(values) != 5:
                raise ValueError(f"{name} must contain exactly five values")
            object.__setattr__(self, name, values)
        if any(not 1e-6 <= v <= 1000 for v in self.variances):
            raise ValueError("supported candidate variances are in [1e-6,1000]")
        if any(q < 0 for q in self.model_prior) or abs(math.fsum(self.model_prior) - 1) > 1e-12:
            raise ValueError("model_prior must be nonnegative and sum to1")
        p = _number(self.null_response, "null_response")
        tol = _number(self.quadrature_tolerance, "quadrature_tolerance")
        if not 0 < p < 1 or not 1e-10 <= tol <= 1e-4:
            raise ValueError("null_response must be in (0,1), quadrature_tolerance in [1e-10,1e-4]")
        if self.equality_policy not in ("hold", "continue"):
            raise ValueError("equality_policy must be hold or continue")
        object.__setattr__(self, "null_response", p)
        object.__setattr__(self, "quadrature_tolerance", tol)

    def to_json(self) -> str:
        return (
            json.dumps(
                {
                    "schema_version": 1,
                    "profile": COMPAS_PROFILE,
                    **{name: getattr(self, name) for name in self.__dataclass_fields__},
                },
                indent=2,
                allow_nan=False,
            )
            + "\n"
        )

    @classmethod
    def from_json(cls, text: str) -> "ComPASStudy":
        if not isinstance(text, str) or len(text.encode("utf-8")) > 32768:
            raise ValueError("study must be JSON text of at most32768 bytes")

        def unique(pairs: list[tuple[str, object]]) -> dict[str, object]:
            values: dict[str, object] = {}
            for key, value in pairs:
                if key in values:
                    raise ValueError(f"duplicate JSON field: {key}")
                values[key] = value
            return values

        try:
            data = json.loads(text, object_pairs_hook=unique)
        except (json.JSONDecodeError, RecursionError) as exc:
            raise ValueError("invalid ComPAS JSON") from exc
        if (
            not isinstance(data, dict)
            or set(data) != {"schema_version", "profile", *cls.__dataclass_fields__}
            or type(data["schema_version"]) is not int
            or data["schema_version"] != 1
            or data["profile"] != COMPAS_PROFILE
        ):
            raise ValueError("expected ComPAS explicit-hierarchy study schema version1")
        return cls(**{name: data[name] for name in cls.__dataclass_fields__})

    def fit(self) -> "ComPASFit":
        return fit_compas(self)


@dataclass(frozen=True)
class ComPASFit:
    study: ComPASStudy
    log_evidence: tuple[float, ...]
    model_probabilities: tuple[float, ...]
    model_probability_bounds: tuple[tuple[float, float], ...]
    tail_probabilities: tuple[tuple[float, ...], ...]
    tail_error_estimates: tuple[tuple[float, ...], ...]
    evidence_relative_error_estimates: tuple[float, ...]
    selected_model: int | None
    decisions: tuple[str, ...]
    evaluations: tuple[int, ...]

    def to_json(self) -> str:
        report = {
            "report_schema_version": 1,
            "study": json.loads(self.study.to_json()),
            **{name: getattr(self, name) for name in self.__dataclass_fields__ if name != "study"},
            "model_index_convention": "zero-based; null means no numerically separated maximum",
            "source": "PNOC022 v3.0, 2023-01-26, pp83-84",
            "protocol_sha256": "fa69432cb8e68abec0b206a3e0f753c7cb2b0e40827651a99fc18411e5c37e80",
            "limitations": _LIMITS,
        }
        return json.dumps(report, indent=2, allow_nan=False) + "\n"


def fit_compas(study: ComPASStudy) -> ComPASFit:
    """Fit all five explicit candidates and apply only numerically resolved decisions.

    Any integration failure raises ArithmeticError. A model tie or a decision
    interval crossing a cutoff returns numerical_hold, distinct from a clinical
    stop or futility decision. Bounds use quadrature error estimates, not a proof.
    """
    if not isinstance(study, ComPASStudy):
        raise ValueError("study must be a validated ComPASStudy")
    groups = [
        tuple(i for i, label in enumerate(study.groups) if label == group)
        for group in dict.fromkeys(study.groups)
    ]
    evidence, tails, errors, relative_errors, evaluations = [], [], [], [], []
    for variance in study.variances:
        logz, relative, work = 0.0, 0.0, 0
        probabilities, error = [0.0] * len(study.completed), [0.0] * len(study.completed)
        for indices in groups:
            z, p, e, pe, count = compas_group_integral(
                tuple(study.completed[i] for i in indices),
                tuple(study.successes[i] for i in indices),
                variance,
                study.null_response,
                study.quadrature_tolerance / (2 * len(groups)),
            )
            logz += z
            relative += e
            work += count
            for i, value, err in zip(indices, p, pe):
                probabilities[i], error[i] = value, err
        evidence.append(logz)
        relative_errors.append(math.expm1(relative))
        tails.append(tuple(probabilities))
        errors.append(tuple(error))
        evaluations.append(work)
    logweights = [z + math.log(q) if q else -math.inf for z, q in zip(evidence, study.model_prior)]
    normalizer = float(logsumexp(logweights))
    posterior = tuple(math.exp(z - normalizer) for z in logweights)
    lower = [p * max(0.0, 1 - e) for p, e in zip(posterior, relative_errors)]
    upper = [p * (1 + e) for p, e in zip(posterior, relative_errors)]
    bounds = tuple(
        (
            lo / (lo + sum(upper[j] for j in range(5) if j != i)),
            hi / (hi + sum(lower[j] for j in range(5) if j != i)),
        )
        for i, (lo, hi) in enumerate(zip(lower, upper))
    )
    selected = next(
        (i for i in range(5) if bounds[i][0] > max(bounds[j][1] for j in range(5) if j != i)), None
    )
    decisions = ["numerical_hold"] * len(study.completed)
    if selected is not None:
        exact = compas_decisions(tails[selected], equality_policy=study.equality_policy)
        for i, (probability, uncertainty) in enumerate(zip(tails[selected], errors[selected])):
            if (
                not any(
                    probability - uncertainty <= cutoff <= probability + uncertainty
                    for cutoff in (0.1, 0.95)
                )
                or uncertainty == 0
            ):
                decisions[i] = exact[i]
    return ComPASFit(
        study,
        tuple(evidence),
        posterior,
        bounds,
        tuple(tails),
        tuple(errors),
        tuple(relative_errors),
        selected,
        tuple(decisions),
        tuple(evaluations),
    )
