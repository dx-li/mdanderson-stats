"""Portable inputs and captured-result reports for the general Multc99 workflow.

Adapted workflow retains the source's noncommercial terms; see the bundled
Multc99 notice. These are community JSON/HTML files, not native menu files.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from html import escape
from pathlib import Path
from typing import Any

import numpy as np

from .multc99 import Multc99Event, multc99_design
from .multc99_boundaries import multc99_with_boundaries
from .multc99_randomized import Multc99RandomizedSimulation, simulate_multc99_randomized
from .multc99_trial import Multc99Simulation, simulate_multc99
from .multc_study import _atomic_write

_MAX_INPUT_BYTES = 4 * 1024 * 1024


def _row(values: list[object], *, header: bool = False) -> str:
    tag = "th" if header else "td"
    return "<tr>" + "".join(f"<{tag}>{escape(str(v))}</{tag}>" for v in values) + "</tr>"


@dataclass(frozen=True)
class Multc99StudyReport:
    title: str
    inputs_json: str
    results: Multc99Simulation | Multc99RandomizedSimulation

    def to_html(self) -> str:
        design = self.results.design
        if isinstance(self.results, Multc99Simulation):
            result = self.results
            body = f"<p>Early-stop probability: {result.early_stop_probability:.6g} "
            body += f"(MCSE {result.early_stop_mcse:.6g}); "
            body += f"mean sample size: {result.mean_sample_size:.6g} "
            body += f"(MCSE {result.sample_size_mcse:.6g}).</p>"
            body += "<p>Empirical sample-size quantiles (10, 25, 50, 75, 90%): "
            body += escape(str(result.sample_size_quantiles.tolist())) + ".</p>"
            body += "<table>" + _row(["Event hit pattern", "Trials", "Probability"], header=True)
            for pattern, count in result.hit_patterns:
                labels = ("none", "lower", "upper", "lower and upper")
                body += _row(
                    [", ".join(labels[i] for i in pattern), count, f"{count / result.trials:.6g}"]
                )
            body += "</table><p>Pattern positions follow the event order below.</p>"
        else:
            randomized = self.results
            body = "<table>" + _row(
                [
                    "Arm",
                    "Selection probability",
                    "Selection MCSE",
                    "Termination probability",
                    "Mean enrolled",
                ],
                header=True,
            )
            body += _row(
                [
                    "None",
                    f"{randomized.selection_probability[0]:.6g}",
                    f"{randomized.selection_mcse[0]:.6g}",
                    "—",
                    "—",
                ]
            )
            for j in range(randomized.arm_probabilities.shape[0]):
                body += _row(
                    [
                        j,
                        f"{randomized.selection_probability[j + 1]:.6g}",
                        f"{randomized.selection_mcse[j + 1]:.6g}",
                        f"{randomized.termination_probability[j]:.6g}",
                        f"{np.mean(randomized.arm_sample_sizes[:, j]):.6g}",
                    ]
                )
            body += "</table><p>Arm indices are zero-based; "
            body += "the design cap is a total assignment-slot budget.</p>"
        body += "<h2>Events and boundaries</h2>"
        for j, event in enumerate(design.events):
            body += f"<h3>{escape(event.name)}</h3><p>Numerator mask: {event.definition}; "
            body += f"conditioning mask: {event.conditioning_definition or 'all categories'}. "
            body += "Boundary index is the conditioning count; "
            body += "-1 and n+1 denote unattainable stops.</p>"
            body += "<table>" + _row(["n", "Lower stop maximum", "Upper stop minimum"], header=True)
            for n in range(design.max_subjects + 1):
                body += _row([n, design.lower_bounds[j, n], design.upper_bounds[j, n]])
            body += "</table>"
        return (
            "<!doctype html><html lang='en'><meta charset='utf-8'><title>"
            + escape(self.title)
            + "</title><style>body{font-family:system-ui;max-width:1000px;margin:2em auto}"
            "table{border-collapse:collapse}td,th{border:1px solid #aaa;padding:.35em}"
            "pre{white-space:pre-wrap;overflow-wrap:anywhere}</style><body><h1>"
            + escape(self.title)
            + "</h1><p>Independent Python Multc99 study. "
            "Monte Carlo errors describe simulation error.</p>"
            + body
            + "<h2>Complete captured inputs</h2><pre>"
            + escape(self.inputs_json)
            + "</pre></body></html>"
        )

    def write_html(self, path: str | Path) -> Path:
        return _atomic_write(path, self.to_html())

    def write_inputs(self, path: str | Path) -> Path:
        return _atomic_write(path, self.inputs_json)


def multc99_study_report(
    results: Multc99Simulation | Multc99RandomizedSimulation,
    *,
    title: str = "Multc99 Python study",
) -> Multc99StudyReport:
    """Capture all actual inputs, seed and settings from a completed simulation."""
    if not isinstance(results, (Multc99Simulation, Multc99RandomizedSimulation)):
        raise TypeError("results must be a completed Multc99 simulation")
    if not isinstance(title, str) or not title.strip() or len(title) > 200:
        raise ValueError("title must be nonempty text of at most 200 characters")
    design = results.design
    specification = dict(
        experimental_prior=design.experimental_prior.tolist(),
        historical_prior=design.historical_prior.tolist(),
        historical_weights=design.historical_weights.tolist(),
        events=[asdict(e) for e in design.events],
        max_subjects=design.max_subjects,
        min_subjects=design.min_subjects,
        cohort_size=design.cohort_size,
        manual_boundaries=(
            dict(lower=design.lower_bounds.tolist(), upper=design.upper_bounds.tolist())
            if design.boundary_origin == "manual"
            else None
        ),
    )
    settings: dict[str, Any] = dict(trials=results.trials, seed=results.seed)
    if isinstance(results, Multc99Simulation):
        mode = "single"
        settings.update(
            elementary_probabilities=results.elementary_probabilities.tolist(),
            monitoring_period=results.monitoring_period,
            accrual_rate=results.accrual_rate,
            response_window=results.response_window,
        )
    else:
        mode = "randomized"
        settings.update(
            arm_probabilities=results.arm_probabilities.tolist(),
            target_event=results.target_event,
            maximize=results.maximize,
            reassign=results.reassign,
        )
    inputs = dict(
        schema_version=1,
        numpy_version=np.__version__,
        title=title,
        mode=mode,
        design=specification,
        simulation=settings,
    )
    return Multc99StudyReport(title, json.dumps(inputs, indent=2, allow_nan=False) + "\n", results)


def _unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def _nonfinite(value: str) -> None:
    raise ValueError(f"nonfinite JSON value: {value}")


def replay_multc99_study(path: str | Path) -> Multc99StudyReport:
    """Read bounded versioned inputs and repeat the captured serial study.

    Exact RNG replay requires the recorded NumPy version. Version mismatches
    raise rather than implying that an upgraded sampler reproduces old draws.
    """
    with Path(path).open("rb") as stream:
        data = stream.read(_MAX_INPUT_BYTES + 1)
    if len(data) > _MAX_INPUT_BYTES:
        raise ValueError("Multc99 input file exceeds 4 MiB")
    try:
        inputs = json.loads(data, object_pairs_hook=_unique, parse_constant=_nonfinite)
        required = {"schema_version", "numpy_version", "title", "mode", "design", "simulation"}
        if not isinstance(inputs, dict) or set(inputs) != required:
            raise ValueError("invalid Multc99 input schema")
        if type(inputs["schema_version"]) is not int or inputs["schema_version"] != 1:
            raise ValueError("unsupported Multc99 schema_version")
        if inputs["numpy_version"] != np.__version__:
            raise ValueError("exact replay requires the recorded NumPy version")
        if inputs["mode"] not in ("single", "randomized"):
            raise ValueError("unsupported Multc99 study mode")
        design_fields = {
            "experimental_prior",
            "historical_prior",
            "historical_weights",
            "events",
            "max_subjects",
            "min_subjects",
            "cohort_size",
            "manual_boundaries",
        }
        specification = inputs["design"]
        if not isinstance(specification, dict) or set(specification) != design_fields:
            raise ValueError("invalid Multc99 design fields")
        if (
            not isinstance(specification["events"], list)
            or not 1 <= len(specification["events"]) <= 32
        ):
            raise ValueError("invalid Multc99 events")
        event_fields = {field.name for field in fields(Multc99Event)}
        if any(not isinstance(e, dict) or set(e) != event_fields for e in specification["events"]):
            raise ValueError("invalid Multc99 event fields")
        specification["events"] = [Multc99Event(**e) for e in specification["events"]]
        manual = specification.pop("manual_boundaries")
        design = multc99_design(**specification)
        if manual is not None:
            if not isinstance(manual, dict) or set(manual) != {"lower", "upper"}:
                raise ValueError("invalid manual boundary fields")
            design = multc99_with_boundaries(design, manual["lower"], manual["upper"])
        settings = inputs["simulation"]
        simulation_fields = (
            {
                "trials",
                "seed",
                "elementary_probabilities",
                "monitoring_period",
                "accrual_rate",
                "response_window",
            }
            if inputs["mode"] == "single"
            else {"trials", "seed", "arm_probabilities", "target_event", "maximize", "reassign"}
        )
        if not isinstance(settings, dict) or set(settings) != simulation_fields:
            raise ValueError("invalid Multc99 simulation fields")
        result = (
            simulate_multc99(design, **settings)
            if inputs["mode"] == "single"
            else simulate_multc99_randomized(design, **settings)
        )
        return multc99_study_report(result, title=inputs["title"])
    except (TypeError, KeyError, UnicodeError, OverflowError, RecursionError) as error:
        raise ValueError("invalid Multc99 study inputs") from error
