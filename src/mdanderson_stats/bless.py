"""BLESS survival using the rounded 2021 published tables, on their exact grid.

Independent implementation of Molina et al., doi:10.1016/j.chest.2021.03.059,
e-Appendix 1 / e-Tables 1–8. Not a reconstruction of the deployed daily fit.
"""

import csv
import io
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal, localcontext
from numbers import Integral, Real
from types import MappingProxyType

BLESS_PROFILE = "published_tables_2021"
BLESS_TIME_DAYS = (0, 30, 60, 90, 120, 150, 180)
_SOURCE = "https://doi.org/10.1016/j.chest.2021.03.059"
_SUPPLEMENT_SHA256 = "089787cdb4ae270ff645523e0bb1fab70f95fc467acc3c1bbf3ce5547e2b9744"
_BASELINES = {
    "lung": (1.0, 0.933, 0.879, 0.849, 0.800, 0.791, 0.768),
    "breast": (1.0, 0.882, 0.796, 0.705, 0.646, 0.601, 0.582),
    "hematologic": (1.0, 0.801, 0.707, 0.593, 0.520, 0.467, 0.461),
    "gcmm": (1.0, 0.763, 0.623, 0.510, 0.421, 0.376, 0.352),
}
_KEYS = {
    "lung": {
        "ecog",
        "surgery_within_30_days",
        "bilateral",
        "benign_cytology",
        "ldh_u_l",
        "protein_g_dl",
    },
    "breast": {"ecog", "benign_cytology", "nlr", "ldh_u_l", "protein_g_dl"},
    "hematologic": {
        "ecog",
        "high_clinical_suspicion_pneumonia",
        "bilateral",
        "benign_cytology",
        "protein_g_dl",
    },
    "gcmm": {"cancer", "ecog", "benign_cytology", "nlr", "bilateral", "ldh_u_l", "protein_g_dl"},
}
_WARNINGS = {
    "lung": (
        "Use equation/table centers LDH/100=7.19 and protein=3.7; "
        "earlier prose gives 7.31 and 3.8.",
    ),
    "breast": (
        "Use equation/table NLR center 5.398; baseline footnote gives 5.389.",
        "Printed example survival conflicts with the printed coefficients and baseline "
        "beyond ordinary rounding; results follow the tables.",
    ),
    "hematologic": (
        "Rounded coefficients give example LP -0.502 rather than printed -0.49662; "
        "full-precision coefficients are unknown.",
    ),
    "gcmm": (
        "Published example requires bilateral=1 although its prose omits laterality.",
        "Age has no coefficient; LDH is divided by 100 as in the equation and e-Table 7.",
    ),
}
_LIMITS = (
    "Rounded published parameters; only days 0/30/60/90/120/150/180 from first thoracentesis.",
    "No interpolation, extrapolation, individual confidence intervals or deployed-app parity.",
    "Mathematical implementation is not clinical validation.",
)


def _inputs(model: str, predictors: Mapping[str, object]) -> Mapping[str, object]:
    if not isinstance(model, str) or model not in _KEYS:
        raise ValueError("model must be lung, breast, hematologic or gcmm")
    if not isinstance(predictors, Mapping) or set(predictors) != _KEYS[model]:
        raise ValueError(f"{model} requires exactly these predictors: {sorted(_KEYS[model])}")
    normalized: dict[str, object] = {}
    for key, value in predictors.items():
        if key == "cancer":
            if not isinstance(value, str) or value not in ("lung", "breast", "hematologic"):
                raise ValueError("cancer must be lung, breast or hematologic")
            normalized[key] = value
        elif key == "ecog":
            if (
                isinstance(value, bool)
                or not isinstance(value, Integral)
                or not 0 <= int(value) <= 4
            ):
                raise ValueError("ecog must be an integer from 0 to 4")
            normalized[key] = int(value)
        elif key in ("ldh_u_l", "protein_g_dl", "nlr"):
            if isinstance(value, bool) or not isinstance(value, Real):
                raise ValueError(f"{key} must be a finite nonnegative number")
            try:
                number = float(value)
            except OverflowError as exc:
                raise ValueError(f"{key} must be representable as a finite number") from exc
            if not math.isfinite(number) or number < 0:
                raise ValueError(f"{key} must be a finite nonnegative number")
            normalized[key] = number
        else:
            if not isinstance(value, (bool, Integral)) or value not in (0, 1):
                raise ValueError(f"{key} must be a boolean or integer 0/1")
            normalized[key] = int(value)
    return MappingProxyType(dict(sorted(normalized.items())))


def _times(values: Sequence[int]) -> tuple[int, ...]:
    try:
        times = tuple(values)
    except TypeError as exc:
        raise ValueError(
            "time_days must be a nonempty increasing subset of the published grid"
        ) from exc
    if (
        not times
        or any(
            isinstance(t, bool) or not isinstance(t, Integral) or t not in BLESS_TIME_DAYS
            for t in times
        )
        or any(b <= a for a, b in zip(times, times[1:]))
    ):
        raise ValueError("time_days must be a nonempty increasing subset of the published grid")
    return tuple(int(t) for t in times)


def _eta(model: str, x: Mapping[str, object]) -> float:
    # Printed coefficients are decimal quantities. Evaluating their products in
    # binary64 before fsum can erase centering when large opposing terms cancel.
    # 400 digits cover every normalized finite binary64 input's integer part,
    # the printed decimal factors/centers, and substantial cancellation guard.
    with localcontext() as context:
        context.prec = 400
        return _eta_decimal(model, x)


def _eta_decimal(model: str, x: Mapping[str, object]) -> float:
    v = {
        k: Decimal.from_float(float(value))  # type: ignore[arg-type]
        for k, value in x.items()
        if k != "cancer"
    }
    ecog = int(v["ecog"])
    benign, protein = (v["benign_cytology"], v["protein_g_dl"])
    terms: tuple[Decimal, ...]
    if model == "lung":
        terms = (
            (Decimal("0.0"), Decimal("0.51"), Decimal("1.32"), Decimal("1.23"), Decimal("1.49"))[
                ecog
            ],
            Decimal("0.806") * v["surgery_within_30_days"],
            Decimal("0.812") * v["bilateral"],
            -Decimal("0.621") * benign,
            Decimal("0.018") * (v["ldh_u_l"] / 100 - Decimal("7.19")),
            -Decimal("0.558") * (protein - Decimal("3.7")),
        )
    elif model == "breast":
        terms = (
            Decimal("0.648") * (ecog >= 2),
            -Decimal("0.987") * benign,
            Decimal("0.054") * (v["nlr"] - Decimal("5.398")),
            Decimal("0.011") * (v["ldh_u_l"] / 100 - Decimal("6.91")),
            -Decimal("0.379") * (protein - Decimal("3.8")),
        )
    elif model == "hematologic":
        terms = (
            Decimal("0.84") * (ecog >= 2),
            Decimal("0.5") * v["high_clinical_suspicion_pneumonia"],
            -Decimal("0.53") * v["bilateral"],
            -Decimal("0.73") * benign,
            -Decimal("0.41") * (protein - Decimal("2.8")),
        )
    else:
        cancer = str(x["cancer"])
        nlr, ldh, prot, unilateral, bilateral = {
            "lung": (
                Decimal("0.009"),
                Decimal("0.016"),
                -Decimal("0.497"),
                Decimal("0.0"),
                Decimal("0.825"),
            ),
            "hematologic": (
                Decimal("0.002"),
                -Decimal("0.001"),
                -Decimal("0.42"),
                Decimal("0.556"),
                -Decimal("0.026"),
            ),
            "breast": (
                Decimal("0.051"),
                Decimal("0.008"),
                -Decimal("0.449"),
                Decimal("0.407"),
                -Decimal("0.047"),
            ),
        }[cancer]
        terms = (
            (
                -Decimal("1.485"),
                -Decimal("0.631"),
                Decimal("0.0"),
                -Decimal("0.033"),
                Decimal("0.269"),
            )[ecog],
            -Decimal("0.649") * benign,
            nlr * (v["nlr"] - Decimal("6.11")),
            ldh * (v["ldh_u_l"] / 100 - Decimal("5.89")),
            prot * (protein - Decimal("3.4")),
            bilateral if v["bilateral"] else unilateral,
        )
    try:
        eta = float(sum(terms, Decimal(0)))
    except OverflowError as exc:
        raise ValueError("linear predictor exceeds finite floating-point range") from exc
    if not math.isfinite(eta):
        raise ValueError("linear predictor exceeds finite floating-point range")
    return eta


@dataclass(frozen=True)
class BLESSPublishedStudy:
    """Validated, saved input for one published-grid prediction.

    Cytology: benign=1, malignant=0. LDH: pleural fluid U/L; protein: pleural
    fluid g/dL; NLR: blood neutrophil/lymphocyte ratio. No missing-value imputation.
    Continuous inputs have mathematical bounds only, not validated clinical ranges.
    """

    model: str
    predictors: Mapping[str, object]
    time_days: tuple[int, ...] = BLESS_TIME_DAYS

    def __post_init__(self) -> None:
        object.__setattr__(self, "predictors", _inputs(self.model, self.predictors))
        object.__setattr__(self, "time_days", _times(self.time_days))

    def to_json(self) -> str:
        return (
            json.dumps(
                {
                    "schema_version": 1,
                    "profile": BLESS_PROFILE,
                    "model": self.model,
                    "predictors": dict(self.predictors),
                    "time_days": self.time_days,
                },
                indent=2,
                allow_nan=False,
            )
            + "\n"
        )

    @classmethod
    def from_json(cls, text: str) -> "BLESSPublishedStudy":
        """Read a strict versioned input; reject duplicate/unknown fields."""
        if not isinstance(text, str) or len(text.encode("utf-8")) > 16384:
            raise ValueError("study must be JSON text of at most 16384 bytes")

        def unique(pairs: list[tuple[str, object]]) -> dict[str, object]:
            result: dict[str, object] = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError(f"duplicate JSON field: {key}")
                result[key] = value
            return result

        try:
            data = json.loads(text, object_pairs_hook=unique)
        except (json.JSONDecodeError, RecursionError) as exc:
            raise ValueError("invalid study JSON") from exc
        if (
            not isinstance(data, dict)
            or set(data) != {"schema_version", "profile", "model", "predictors", "time_days"}
            or type(data["schema_version"]) is not int
            or data["schema_version"] != 1
            or data["profile"] != BLESS_PROFILE
            or not isinstance(data["time_days"], list)
        ):
            raise ValueError("expected BLESS published_tables_2021 study schema version 1")
        return cls(data["model"], data["predictors"], tuple(data["time_days"]))

    def predict(self) -> "BLESSPublishedPrediction":
        eta = _eta(self.model, self.predictors)
        baseline = tuple(_BASELINES[self.model][BLESS_TIME_DAYS.index(t)] for t in self.time_days)
        survival = []
        for s in baseline:
            if s == 1:
                survival.append(1.0)
            else:
                log_hazard = eta + math.log(-math.log(s))
                # exp(-H) already rounds to zero for H > 746. Avoid exp(eta)
                # overflow while retaining accurate finite, near-one results.
                survival.append(
                    0.0 if log_hazard > math.log(746.0) else math.exp(-math.exp(log_hazard))
                )
        return BLESSPublishedPrediction(self, eta, baseline, tuple(survival))


@dataclass(frozen=True)
class BLESSPublishedPrediction:
    study: BLESSPublishedStudy
    linear_predictor: float
    reference_survival: tuple[float, ...]
    survival: tuple[float, ...]

    @property
    def source_warnings(self) -> tuple[str, ...]:
        return _WARNINGS[self.study.model]

    def to_json(self) -> str:
        """Self-contained report with input, precision, source choices and limits."""
        report = {
            "report_schema_version": 1,
            "study": json.loads(self.study.to_json()),
            "linear_predictor": self.linear_predictor,
            "reference_survival": self.reference_survival,
            "survival": self.survival,
            "source": _SOURCE,
            "supplement_sha256": _SUPPLEMENT_SHA256,
            "source_warnings": self.source_warnings,
            "limitations": _LIMITS,
        }
        return json.dumps(report, indent=2, allow_nan=False) + "\n"

    def to_csv(self) -> str:
        """Curve export; use to_json for the complete inputs and source audit."""
        stream = io.StringIO(newline="")
        writer = csv.writer(stream)
        writer.writerow(
            ("profile", "model", "day", "reference_survival", "survival", "linear_predictor")
        )
        for day, baseline, survivor in zip(
            self.study.time_days, self.reference_survival, self.survival
        ):
            writer.writerow(
                (BLESS_PROFILE, self.study.model, day, baseline, survivor, self.linear_predictor)
            )
        return stream.getvalue()


def bless_published_prediction(
    model: str, predictors: Mapping[str, object], *, time_days: Sequence[int] = BLESS_TIME_DAYS
) -> BLESSPublishedPrediction:
    """Predict on a subset of the seven published days, with explicit precision limits."""
    return BLESSPublishedStudy(model, predictors, _times(time_days)).predict()
