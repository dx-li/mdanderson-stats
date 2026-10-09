"""Captured-input, replayable HTML reports for the explicit iBOIN workflow."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike

from .iboin import IBOINDesign
from .iboin_simulation import IBOINOperatingCharacteristics, simulate_iboin
from .iboin_trial import _conduct_settings

_MAX_INPUT_BYTES = 4 * 1024 * 1024
_DESIGN_FIELDS = (
    "skeleton",
    "prior_ess",
    "target",
    "safe_probability",
    "toxic_probability",
    "elimination_probability",
    "robust_prior",
    "extra_safe",
    "safety_offset",
    "early_stop_patients",
)
_SIMULATION_FIELDS = {
    "grade2_probability",
    "dlt_probability",
    "cohort_size",
    "max_patients",
    "repetitions",
    "prior_mode",
    "isotonic_weights",
    "trial_seeds",
    "starting_dose",
    "titration",
    "titration_cap",
    "eligible_doses",
    "tie_policy",
    "enforce_deescalation_boundary",
    "max_total_work",
    "max_total_storage_bytes",
}


def _plain(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write(path: str | Path, text: str) -> Path:
    destination = Path(path)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=destination.parent, delete=False
        ) as handle:
            temporary = handle.name
            handle.write(text)
        os.replace(temporary, destination)
    finally:
        if temporary is not None and os.path.exists(temporary):
            os.unlink(temporary)
    return destination


def _estimate(mean: float, mcse: float) -> str:
    error = f"{mcse:.6g}" if np.isfinite(mcse) else "unavailable (one repetition)"
    return f"{mean:.6g} (MCSE {error})"


@dataclass(frozen=True)
class IBOINSimulationReport:
    """Simulation results and complete portable inputs, including exact trial seeds."""

    title: str
    inputs_json: str
    results: IBOINOperatingCharacteristics

    def to_html(self) -> str:
        inputs = json.loads(self.inputs_json)
        design, simulation = inputs["design"], inputs["simulation"]
        oc = self.results
        rows = []
        for j, (skeleton, original, effective) in enumerate(
            zip(design["skeleton"], design["prior_ess"], inputs["effective_prior_ess"], strict=True)
        ):
            cells = [
                str(j + 1),
                str(skeleton),
                str(original),
                str(effective),
                str(simulation["grade2_probability"][j]),
                str(simulation["dlt_probability"][j]),
                _estimate(oc.selection_probability[j + 1], oc.selection_mcse[j + 1]),
                _estimate(oc.mean_patients_by_dose[j], oc.mean_patients_mcse[j]),
                _estimate(oc.mean_dlt_by_dose[j], oc.mean_dlt_mcse[j]),
                _estimate(oc.mean_grade2_by_dose[j], oc.mean_grade2_mcse[j]),
            ]
            rows.append("<tr>" + "".join(f"<td>{escape(cell)}</td>" for cell in cells) + "</tr>")
        headings = (
            "Dose",
            "Historical toxicity",
            "Original ESS",
            "Effective ESS",
            "True grade 2",
            "True DLT",
            "Selection probability",
            "Mean patients",
            "Mean DLT",
            "Mean grade 2",
        )
        stops = "".join(
            f"<li>{escape(label)}: {escape(_estimate(float(prob), float(error)))}</li>"
            for label, prob, error in zip(
                oc.stop_reason_labels, oc.stop_reason_probability, oc.stop_reason_mcse, strict=True
            )
        )
        return (
            '<!doctype html><html lang="en"><meta charset="utf-8">'
            f"<title>{escape(self.title)}</title><body><h1>{escape(self.title)}</h1>"
            "<p>Complete-outcome Python simulation. Grade-2 and DLT outcomes are mutually "
            "exclusive. Historical priors affect assignment; safety uses a uniform Beta(1,1) "
            "prior. Native isotonic weights, ties and final-prior linkage remain unverified. "
            "The captured inputs specify the Python policies used here.</p>"
            f"<p>Repetitions: {oc.repetitions}. Target: {design['target']}.</p>"
            "<table><thead><tr>"
            + "".join(f"<th>{escape(label)}</th>" for label in headings)
            + "</tr></thead><tbody>"
            + "".join(rows)
            + "</tbody></table>"
            + "<p>No MTD selected: "
            + escape(_estimate(oc.selection_probability[0], oc.selection_mcse[0]))
            + ".</p>"
            + "<p>Mean total enrollment: "
            + escape(_estimate(oc.mean_total_patients, oc.total_patients_mcse))
            + ". "
            + "Enrollment quantiles (10%, 50%, 90%): "
            + escape(", ".join(f"{value:.6g}" for value in oc.total_patient_quantiles))
            + ".</p>"
            + "<p>Probabilities and count means use all repetitions as their denominator. "
            "MCSE describes Monte Carlo uncertainty, not a clinical confidence interval.</p>"
            + "<h2>Stopping probabilities</h2><ul>"
            + stops
            + "</ul>"
            + "<h2>Complete inputs and replay seeds</h2><p>Seeds are exact integers; preserve "
            "them with an integer-capable JSON reader. Record the package source revision "
            "separately for reproducible analyses.</p><pre>"
            + escape(self.inputs_json)
            + "</pre></body></html>"
        )

    def write_html(self, path: str | Path) -> Path:
        """Atomically save a self-contained HTML report; parent must exist."""
        return _write(path, self.to_html())

    def write_inputs(self, path: str | Path) -> Path:
        """Atomically save standard JSON inputs suitable for replay_iboin_report."""
        return _write(path, self.inputs_json + "\n")


def simulate_iboin_report(
    design: IBOINDesign,
    grade2_probability: ArrayLike,
    dlt_probability: ArrayLike,
    *,
    cohort_size: int,
    max_patients: int,
    repetitions: int,
    prior_mode: Literal["none", "original", "effective"],
    isotonic_weights: Literal["patients", "effective", "equal"] | ArrayLike,
    seed: int | None = None,
    trial_seeds: ArrayLike | None = None,
    starting_dose: int = 1,
    titration: bool = True,
    titration_cap: int | None = None,
    eligible_doses: ArrayLike | None = None,
    tie_policy: Literal["lowest", "highest"] = "lowest",
    enforce_deescalation_boundary: bool = False,
    max_total_work: int = 1_000_000,
    max_total_storage_bytes: int = 128 * 1024 * 1024,
    title: str = "iBOIN simulation report",
) -> IBOINSimulationReport:
    """Run the existing bounded simulator and capture all effective settings."""
    if (
        not isinstance(title, str)
        or not 1 <= len(title) <= 256
        or any(ord(char) < 32 or ord(char) == 127 for char in title)
    ):
        raise ValueError("title must be a nonempty string of at most 256 printable characters")
    if not isinstance(design, IBOINDesign):
        raise TypeError("design must be an IBOINDesign")
    design_settings = {name: _plain(getattr(design, name)) for name in _DESIGN_FIELDS}
    snapshot = IBOINDesign(**design_settings)
    grade2 = np.array(grade2_probability, copy=True)
    dlt = np.array(dlt_probability, copy=True)
    weights = (
        isotonic_weights
        if isinstance(isotonic_weights, str)
        else np.array(isotonic_weights, copy=True)
    )
    eligible = None if eligible_doses is None else np.array(eligible_doses, copy=True)
    _, cohort, start, use_titration, cap, maximum = _conduct_settings(
        snapshot, cohort_size, starting_dose, titration, titration_cap, max_patients
    )
    settings: dict[str, Any] = dict(
        grade2_probability=grade2,
        dlt_probability=dlt,
        cohort_size=cohort,
        max_patients=maximum,
        repetitions=repetitions,
        prior_mode=prior_mode,
        isotonic_weights=weights,
        starting_dose=start,
        titration=use_titration,
        titration_cap=cap if use_titration else None,
        eligible_doses=eligible,
        tie_policy=tie_policy,
        enforce_deescalation_boundary=enforce_deescalation_boundary,
        max_total_work=max_total_work,
        max_total_storage_bytes=max_total_storage_bytes,
    )
    results = simulate_iboin(snapshot, seed=seed, trial_seeds=trial_seeds, **settings)
    settings["trial_seeds"] = results.trial_seeds
    inputs = dict(
        format_version=1,
        title=title,
        design=design_settings,
        effective_prior_ess=snapshot.effective_prior_ess.tolist(),
        simulation={name: _plain(value) for name, value in settings.items()},
    )
    encoded = json.dumps(inputs, allow_nan=False, indent=2)
    return IBOINSimulationReport(title, encoded, results)


def replay_iboin_report(inputs_json: str) -> IBOINSimulationReport:
    """Validate versioned captured inputs and rerun with exact per-trial seeds."""
    if (
        not isinstance(inputs_json, str)
        or len(inputs_json) > _MAX_INPUT_BYTES
        or len(inputs_json.encode("utf-8")) > _MAX_INPUT_BYTES
    ):
        raise ValueError("report inputs must be a JSON string of at most 4 MiB")

    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def reject_constant(value: str) -> Any:
        raise ValueError(f"nonfinite JSON constant: {value}")

    try:
        inputs = json.loads(inputs_json, object_pairs_hook=unique, parse_constant=reject_constant)
    except (json.JSONDecodeError, RecursionError) as exc:
        raise ValueError("invalid iBOIN report JSON") from exc
    if (
        not isinstance(inputs, dict)
        or set(inputs) != {"format_version", "title", "design", "effective_prior_ess", "simulation"}
        or type(inputs["format_version"]) is not int
        or inputs["format_version"] != 1
    ):
        raise ValueError("unsupported iBOIN report input schema")
    design, simulation = inputs["design"], inputs["simulation"]
    if (
        not isinstance(design, dict)
        or set(design) != set(_DESIGN_FIELDS)
        or not isinstance(simulation, dict)
        or set(simulation) != _SIMULATION_FIELDS
    ):
        raise ValueError("incomplete or unknown design/simulation fields")
    snapshot = IBOINDesign(**design)
    if inputs["effective_prior_ess"] != snapshot.effective_prior_ess.tolist():
        raise ValueError("effective prior ESS does not match the captured design")
    seeds = simulation["trial_seeds"]
    if (
        not isinstance(seeds, list)
        or not 1 <= len(seeds) <= 100_000
        or any(type(value) is not int or not 0 <= value < 2**64 for value in seeds)
    ):
        raise ValueError("trial_seeds must contain 1..100000 unsigned 64-bit integers")
    # Explicit dtype avoids NumPy promoting mixed small/large Python integers to
    # float64 and losing the exact seed bits before the simulator sees them.
    simulation["trial_seeds"] = np.array(seeds, dtype=np.uint64)
    return simulate_iboin_report(snapshot, title=inputs["title"], **simulation)
