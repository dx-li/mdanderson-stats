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
    if any(r.study != study or r.profile != profile for r in batch.results):
        raise ValueError("simulation report requires one saved study and one profile")
    parts = [
        "<!doctype html><html><head><meta charset='utf-8'><title>BOIN simulation</title>",
        "<style>body{font:16px sans-serif;margin:2em}table{border-collapse:collapse}"
        "th,td{border:1px solid #aaa;padding:.4em}pre{white-space:pre-wrap}</style>"
        "</head><body><h1>BOIN saved-study simulation</h1>",
        f"<p>Profile: {escape(profile)}. Master seed: {batch.master_seed}.</p>",
        "<p>Python random streams are reproducible; "
        "native R random-stream identity is not claimed. "
        "Monte Carlo error describes simulation precision, not clinical validity.</p>",
    ]
    if profile == "desktop_legacy":
        parts.append(
            "<p><strong>Legacy compatibility can report phantom patients, discard observed "
            "outcomes, or select eliminated or unobserved doses. Single-MTD state defects "
            "persist within each batch and restart between batches. Nominal binomial errors "
            "do not account for this dependence and are not "
            "validated uncertainty estimates.</strong></p>"
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
