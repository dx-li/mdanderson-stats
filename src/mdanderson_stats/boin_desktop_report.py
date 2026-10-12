"""Independent, escaped BOIN desktop conduct reports with saved-study replay."""

import hashlib
from html import escape

import numpy as np

from .boin_desktop import BOINDesktopStudy, read_boin_desktop_project
from .boin_desktop_controller import BOINDesktopRecommendation


def _table(headers, rows):
    head = "".join(f"<th>{escape(str(h))}</th>" for h in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{escape(str(v))}</td>" for v in row) + "</tr>" for row in rows
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def _number(value):
    return "—" if not np.isfinite(value) else f"{value:.6g}"


def boin_desktop_conduct_html(
    study: BOINDesktopStudy, recommendation: BOINDesktopRecommendation
) -> str:
    """Return standalone HTML plus native-marker XML for reopening the study.

    A report cannot silently pair a recommendation with different saved inputs.
    Pending outcomes are displayed separately from completed no-DLT observations.
    """
    if recommendation.study_sha256 != hashlib.sha256(study.xml.encode()).hexdigest():
        raise ValueError("recommendation does not match the saved BOIN study")
    r = recommendation
    n, y, pending = study.counts()
    parts = [
        "<!doctype html><html><head><meta charset='utf-8'><title>BOIN conduct</title>",
        "<style>body{font:16px sans-serif;max-width:1100px;margin:2em auto}"
        "table{border-collapse:collapse}th,td{border:1px solid #aaa;padding:.4em}"
        "pre{white-space:pre-wrap;overflow-wrap:anywhere}</style></head><body>",
        "<h1>BOIN conduct</h1>",
        _table(
            ["Status", "Cohort", "Next dose", "Reason", "Pending"],
            [(r.status, r.cohort, r.dose or "—", r.reason, r.pending_patients)],
        ),
        "<h2>Analysis settings</h2>",
        _table(
            ["Mode", "Target", "Capacity profile", "Selection profile", "Seed"],
            [(study.mode, study.target, r.capacity_profile, r.selection_profile, r.seed)],
        ),
        "<h2>Administered outcomes</h2>",
    ]
    rows: list[tuple[object, ...]] = []
    for a, b in np.ndindex(study.shape):
        completed = n[a, b] - pending[a, b]
        rows.append(
            (
                (a + 1, b + 1),
                n[a, b],
                completed,
                y[a, b],
                pending[a, b],
                _number(y[a, b] / completed) if completed else "—",
            )
        )
    parts.append(
        _table(["Dose", "Enrolled", "Completed", "DLT", "Pending", "Completed DLT fraction"], rows)
    )
    parts.append("<h2>Patient history</h2>")
    parts.append(
        _table(
            ["ID", "Fingerprint", "Cohort", "Dose", "Outcome", "Entered", "Updated"],
            [
                (
                    p.patient_id,
                    p.fingerprint,
                    p.cohort,
                    p.dose,
                    ("No DLT", "DLT", "Pending")[p.outcome],
                    p.date_entered,
                    p.date_updated,
                )
                for p in study.patients
            ],
        )
    )
    if r.selection is not None:
        selected = r.selection.contour if study.mode == "contour" else r.selection.dose
        parts.append(
            f"<h2>Final selection</h2><p>{escape(str(selected or 'No dose selected'))}</p>"
        )
        details = r.selection_details
        if details is None:
            parts.append(
                "<p>Custom selection service: interval and compatibility "
                "diagnostics were not supplied.</p>"
            )
        else:
            if details.selected_eliminated:
                parts.append(
                    "<p><strong>Legacy compatibility result: the selected dose is "
                    "eliminated by the safety rule.</strong></p>"
                )
            rows = []
            for a, b in np.ndindex(study.shape):
                rows.append(
                    (
                        (a + 1, b + 1),
                        bool(r.selection.eliminated[a, b]),
                        _number(details.reported_mean[a, b]),
                        _number(details.lower[a, b]) if details.lower is not None else "—",
                        _number(details.upper[a, b]) if details.upper is not None else "—",
                        _number(r.selection.safety_overdose_probability[a, b]),
                    )
                )
            parts.append(
                _table(
                    [
                        "Dose",
                        "Eliminated",
                        "Isotonic mean",
                        "Display lower",
                        "Display upper",
                        "Overdose probability",
                    ],
                    rows,
                )
            )
            parts.append(
                "<p>Display limits are rounded isotonic transforms of marginal beta quantiles; "
                "they are not simultaneous credible bands. Selection scores within 1e-10 "
                "use deterministic tie ordering.</p>"
            )
    parts.append(
        "<details><summary>Complete saved inputs</summary><pre>"
        + escape(study.xml)
        + "</pre></details>"
    )
    parts.append("</body></html>")
    return study.to_html("".join(parts))


def boin_desktop_simulation_html(batch) -> str:
    """Report actual and source-reported outcomes for every scenario and batch.

    Results carry their own saved study; mixing unrelated studies is rejected.
    Legacy nominal binomial errors do not account for cross-replication state.
    """
    if not batch.results:
        raise ValueError("simulation report requires at least one result")
    study = batch.results[0].study
    if read_boin_desktop_project(study.xml) != study:
        raise ValueError("simulation inputs do not match the retained saved XML")
    profile = batch.results[0].profile
    method = batch.results[0].method
    if any(r.study != study or r.profile != profile or r.method != method for r in batch.results):
        raise ValueError("simulation report requires one saved study, profile and method")
    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>{escape(method)} simulation</title>",
        "<style>body{font:16px sans-serif;margin:2em}table{border-collapse:collapse}"
        "th,td{border:1px solid #aaa;padding:.4em}pre{white-space:pre-wrap}</style>"
        f"</head><body><h1>{escape(method)} saved-study simulation</h1>",
        f"<p>Profile: {escape(profile)}. Master seed: {batch.master_seed}.</p>",
        "<p>Python random streams are reproducible; "
        "native R random-stream identity is not claimed. "
        "Monte Carlo error describes simulation precision, not clinical validity.</p>",
    ]
    if profile == "desktop_legacy" and study.mode in ("combination", "contour"):
        parts.append(
            "<p><strong>Legacy compatibility can report phantom patients, discard observed "
            "outcomes, or select eliminated or unobserved doses. Single-MTD state defects "
            "persist within each batch and restart between batches. Nominal binomial errors "
            "do not account for this dependence and are not "
            "validated uncertainty estimates.</strong></p>"
        )
    if profile == "desktop_legacy" and study.mode == "tite" and method == "BOIN":
        parts.append(
            "<p><strong>Legacy TITE compatibility can stop using future unobserved outcomes "
            "and report a duration earlier than its actual decision clock. These are explicit "
            "compatibility defects, not defaults of the safe analysis.</strong></p>"
        )
    if method.startswith("3+3 with"):
        parts.append(
            "<p>Expansion adds patients at the fixed dose-finding selection up to the "
            "planned study maximum; it never shrinks a larger dose-finding sample or "
            "revises the selection after expansion outcomes.</p>"
        )
    if method == "Rolling Six" and profile == "desktop_legacy":
        parts.append(
            "<p>Legacy Rolling Six can reassign an eliminated dose and report duration "
            "earlier than its decision clock. Runs exceeding six patients per dose "
            "raise an explicit error; failed replications are not discarded.</p>"
        )
    if study.mode == "tite":
        parts.append("<h2>Calendar diagnostics</h2>")
        parts.append(
            _table(
                [
                    "Scenario",
                    "Seed",
                    "Arrival-gap budget",
                    "Mean reported duration (months)",
                    "Mean actual final time (months)",
                    "Mean extra checks",
                    "Future-outcome stops",
                    "Unsafe assignments",
                ],
                [
                    (
                        r.scenario_index,
                        r.seed,
                        r.calendar_gap_budget,
                        _number(np.mean(r.reported_duration)),
                        _number(np.mean(r.actual_duration)),
                        _number(np.mean(r.suspension_checks)),
                        int(np.sum(r.future_stop)),
                        int(np.sum(r.unsafe_assignment_count))
                        if r.unsafe_assignment_count is not None
                        else 0,
                    )
                    for r in batch.results
                ],
            )
        )
        parts.append(
            "<p>Desktop enrollment starts at zero. Pending decisions advance on additional "
            "arrival gaps. Time is in months; one month is "
            "30.436666666666667 days. Replaying a batch requires its seed "
            "and arrival-gap budget.</p>"
        )
    parts.append("<h2>Replay and discrepancies</h2>")
    parts.append(
        _table(
            [
                "Scenario",
                "Seed",
                "Trials",
                "Reported patients",
                "Actual patients",
                "Reported DLT",
                "Actual DLT",
                "Trials selecting eliminated",
                "Trials selecting unobserved",
                "Stops",
            ],
            [
                (
                    r.scenario_index,
                    r.seed,
                    len(r.patients),
                    int(r.patients.sum()),
                    int(r.actual_patients.sum()),
                    int(r.toxicities.sum()),
                    int(r.actual_toxicities.sum()),
                    int(r.selected_eliminated.sum()),
                    int(np.any(r.selected & (r.actual_patients == 0), axis=(1, 2)).sum()),
                    "; ".join(
                        f"{reason}: {r.stop_reasons.count(reason)}"
                        for reason in sorted(set(r.stop_reasons))
                    ),
                )
                for r in batch.results
            ],
        )
    )
    for index in dict.fromkeys(r.scenario_index for r in batch.results):
        scenario = study.scenarios[index]
        result = batch.aggregate(index)
        parts.append(f"<h2>Scenario {index}: {escape(scenario.name)}</h2>")
        parts.append(
            f"<p>{result['trials']} trials. Selection probabilities are per dose; "
            "a contour may select several doses in one trial.</p>"
        )
        rows = []
        for a, b in np.ndindex(study.shape):
            rows.append(
                (
                    (a + 1, b + 1),
                    scenario.probabilities[a][b],
                    *(
                        _number(result[key][a, b])
                        for key in (
                            "selection_probability",
                            "selection_mcse",
                            "mean_patients",
                            "mean_actual_patients",
                            "mean_toxicities",
                            "mean_actual_toxicities",
                        )
                    ),
                )
            )
        parts.append(
            _table(
                [
                    "Dose",
                    "True DLT probability",
                    "Selection probability",
                    "Nominal binomial MCSE",
                    "Mean reported patients",
                    "Mean actual patients",
                    "Mean reported DLT",
                    "Mean actual DLT",
                ],
                rows,
            )
        )
    parts.append(
        "<details><summary>Complete saved inputs</summary><pre>"
        + escape(study.xml)
        + "</pre></details></body></html>"
    )
    return study.to_html("".join(parts))


def boin_desktop_workflow_html(workflow) -> str:
    """Combine method-specific reports in one reopenable saved-study document."""
    primary = boin_desktop_simulation_html(workflow.primary)
    if workflow.comparator is None:
        return primary
    first = workflow.primary.results[0]
    if any(
        r.study != first.study or r.profile != first.profile for r in workflow.comparator.results
    ):
        raise ValueError("comparison report requires matching saved study and profile")
    comparison = boin_desktop_simulation_html(workflow.comparator)
    body = comparison.split("<body>", 1)[1].split("</body>", 1)[0]
    return primary.replace("</body>", "<hr>" + body + "</body>", 1)
