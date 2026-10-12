"""Self-contained, escaped EffTox workflow reports with inference qualifications."""

from html import escape

from .efftox_desktop import EffToxDesktopStudy
from .efftox_desktop_controller import EffToxDesktopRecommendation
from .efftox_desktop_inference import EffToxDesktopFitDiagnostic, EffToxDesktopInference
from .efftox_desktop_simulation import EffToxDesktopSimulation


def _table(headers: tuple[str, ...], rows: list[tuple[object, ...]]) -> str:
    return (
        "<table><thead><tr>"
        + "".join(f"<th>{escape(h)}</th>" for h in headers)
        + (
            "</tr></thead><tbody>"
            + "".join(
                "<tr>" + "".join(f"<td>{escape(str(v))}</td>" for v in row) + "</tr>"
                for row in rows
            )
            + "</tbody></table>"
        )
    )


def _page(title: str, body: str, study: EffToxDesktopStudy) -> str:
    return (
        '<!doctype html><html lang="en"><meta charset="utf-8">'
        f"<title>{escape(title)}</title><body><h1>{escape(title)}</h1>"
        "<p>Current EffTox controller profile with independent Python inference. "
        "Source-compatible control flow, numerical accuracy and clinical validity "
        "are separate claims. Native posterior integrator and RNG identity are not asserted.</p>"
        + body
        + "<h2>All saved input settings and patient records</h2><pre>"
        + escape(study.xml)
        + "</pre></body></html>"
    )


def _diagnostics(records: tuple[EffToxDesktopFitDiagnostic, ...]) -> str:
    if not records:
        return "<p>Inference diagnostics were not supplied by the analysis service.</p>"
    return (
        "<h2>Every posterior analysis</h2><p>R-hat and MCSE are descriptive diagnostics, "
        "not convergence certificates. Nonfinite values remain visible.</p>"
        + _table(
            (
                "Observed patients",
                "Selected dose index",
                "Likelihood evaluations",
                "Maximum split R-hat",
                "Maximum batch-mean MCSE",
            ),
            [
                (
                    r.observed_patients,
                    r.selected_dose_index,
                    r.likelihood_evaluations,
                    r.max_split_rhat,
                    r.max_batch_mean_mcse,
                )
                for r in records
            ],
        )
    )


def efftox_desktop_conduct_html(
    study: EffToxDesktopStudy,
    recommendation: EffToxDesktopRecommendation,
    *,
    diagnostics: tuple[EffToxDesktopFitDiagnostic, ...] = (),
    inference: EffToxDesktopInference | None = None,
) -> str:
    """Render controller evidence, analyzed prefix, pending bounds and history."""
    settings: tuple[tuple[str, str], ...] = ()
    if inference is not None:
        inference.check_study(study)
        if diagnostics:
            raise ValueError("supply inference or diagnostics, not both")
        diagnostics = inference.diagnostics
        settings = inference.configuration
    body = _table(
        ("Controller field", "Value"),
        [
            ("Status", recommendation.status),
            ("Dose index (zero based)", recommendation.dose_index),
            ("Analyzed patients", recommendation.analyzed_patients),
            ("Pending in analyzed prefix", recommendation.pending_patients),
            ("Possible completed count matrices", recommendation.completion_count),
            ("Analyses performed", recommendation.analyses),
            ("Capacity safeguard applied", recommendation.capacity_guard),
            ("Next cohort", recommendation.next_cohort),
        ],
    )
    body += (
        _table(("Inference setting", "Value"), list(settings))
        if settings
        else ("<p>Inference settings were not supplied.</p>")
    )
    baseline = recommendation.baseline
    if baseline is not None:
        body += "<h2>Observed-prefix baseline dose diagnostics</h2>" + _table(
            (
                "Dose",
                "Mean efficacy",
                "Mean toxicity",
                "Efficacy tail",
                "Toxicity tail",
                "Utility",
                "Admissible",
            ),
            [
                (
                    dose,
                    baseline.mean_efficacy[i]
                    if baseline.mean_efficacy is not None
                    else "not supplied",
                    baseline.mean_toxicity[i]
                    if baseline.mean_toxicity is not None
                    else "not supplied",
                    baseline.efficacy_tail_probability[i],
                    baseline.toxicity_tail_probability[i],
                    baseline.utility[i],
                    bool(baseline.admissible[i]),
                )
                for i, dose in enumerate(study.doses)
            ],
        )
    body += "<h2>Patient history</h2>" + _table(
        (
            "ID",
            "Date identifier",
            "Calculated dose index",
            "Administered dose index",
            "Native outcome code",
            "Cohort",
        ),
        [
            (
                p.patient_id,
                p.begin_date,
                p.calculated_dose_index,
                p.administered_dose_index,
                p.outcome,
                p.cohort,
            )
            for p in study.patients
        ],
    )
    return _page("EffTox trial conduct", body + _diagnostics(diagnostics), study)


def efftox_desktop_simulation_html(result: EffToxDesktopSimulation) -> str:
    """Render true utility, conditional Rselect, allocation and prior diagnostics."""
    scenario = result.study.scenarios[result.scenario_index]
    body = _table(
        ("Simulation field", "Value"),
        [
            ("Scenario index", result.scenario_index),
            ("Trials", len(result.selected_dose_index)),
            ("Outcome seed", result.seed),
            ("No-selection probability", result.no_selection_probability),
            (
                "Rselect conditional on selection",
                "undefined: constant true utility" if result.rselect is None else result.rselect,
            ),
            *list(result.inference_settings),
        ],
    )
    body += "<h2>Dose results</h2>" + _table(
        (
            "Dose",
            "True efficacy",
            "True toxicity",
            "True utility",
            "Selection probability",
            "Selection MCSE",
            "Mean patients",
        ),
        [
            (
                dose,
                scenario.efficacy[i],
                scenario.toxicity[i],
                result.true_utility[i],
                result.selection_probability[i],
                result.selection_mcse[i],
                result.mean_patients_per_dose[i],
            )
            for i, dose in enumerate(result.study.doses)
        ],
    )
    prior = result.prior_decision
    means = result.inference_diagnostics[0] if result.inference_diagnostics else None
    body += "<h2>Prior-only diagnostics</h2>" + _table(
        ("Dose", "Mean efficacy", "Mean toxicity", "Efficacy tail", "Toxicity tail", "Utility"),
        [
            (
                dose,
                means.mean_efficacy[i] if means else "not supplied",
                means.mean_toxicity[i] if means else "not supplied",
                prior.efficacy_tail_probability[i],
                prior.toxicity_tail_probability[i],
                prior.utility[i],
            )
            for i, dose in enumerate(result.study.doses)
        ],
    )
    return _page(
        "EffTox simulation", body + _diagnostics(result.inference_diagnostics), result.study
    )
