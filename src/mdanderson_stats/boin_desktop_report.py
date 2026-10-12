"""Independent, escaped BOIN desktop conduct reports with saved-study replay."""

import hashlib
from html import escape

import numpy as np

from .boin_desktop import BOINDesktopStudy, read_boin_desktop_project
from .boin_desktop_controller import BOINDesktopRecommendation
from .boin_desktop_summary import summarize_boin_desktop


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
        summary = summarize_boin_desktop(batch, index)
        parts.append(f"<h2>Scenario {index}: {escape(scenario.name)}</h2>")
        parts.append(
            f"<p>{result['trials']} trials. Selection probabilities are per dose; "
            "a contour may select several doses in one trial.</p>"
        )
        summary_rows = [
            ("Mean reported enrollment", summary.mean_reported_patients),
            ("Mean actual enrollment", summary.mean_actual_patients),
            ("Mean actual DLT", summary.mean_actual_toxicities),
            ("No selection probability", summary.no_selection_probability),
        ]
        for label, value in (
            ("Recorded safety-stop probability", summary.recorded_safety_stop_probability),
            (
                "Exact-target selection probability (native console)",
                summary.exact_target_selection_probability,
            ),
            ("Actual fraction at exact target", summary.actual_fraction_at_exact_target),
            (
                "Native console fraction at exact target",
                summary.native_console_fraction_at_exact_target,
            ),
            (
                "Nearest-target selection probability (returned R definition)",
                summary.closest_target_selection_probability,
            ),
            ("Actual fraction at nearest target", summary.actual_fraction_at_closest_target),
            (
                "Native exact reference-contour selection probability",
                summary.native_contour_selection_probability,
            ),
            (
                "Actual fraction at reference contour points",
                summary.actual_fraction_at_reference_points,
            ),
            ("Actual incomparable fraction", summary.actual_incomparable_fraction),
            (
                "Conventional duration approximation (months)",
                summary.conventional_duration_approximation,
            ),
        ):
            if value is not None:
                summary_rows.append((label, value))
        parts.append(
            _table(
                ["Scenario summary", "Value"],
                [(label, _number(value)) for label, value in summary_rows],
            )
        )
        if summary.reference_contour:
            parts.append("<p>Reference contour: " + escape(str(summary.reference_contour)) + "</p>")
        for label, regions in (
            ("Reported native geometric fractions", summary.native_reported_contour_regions),
            ("Actual native geometric fractions", summary.native_actual_contour_regions),
        ):
            if regions is not None:
                parts.append(
                    _table(
                        [label, "Residual (includes incomparable)", "Above", "Below"],
                        [("Fraction", *(_number(v) for v in regions))],
                    )
                )
        parts.extend("<p>" + escape(note) + "</p>" for note in summary.qualifications)
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


def boin_desktop_pending_table_html(study, table, *, language="en") -> str:
    """Render exact stored thresholds, a decision flow and saved-input replay."""
    if language not in ("en", "zh"):
        raise ValueError("language must be en or zh")
    if table.study_sha256 != hashlib.sha256(study.xml.encode()).hexdigest():
        raise ValueError("pending table does not match saved study")
    if read_boin_desktop_project(study.xml) != study:
        raise ValueError("saved inputs do not match retained XML")

    def text(english, chinese):
        return english if language == "en" else chinese

    title = text("Desktop TITE pending-outcome table", "桌面 TITE 待观察结局表")
    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        f"<title>{title}</title><style>body{{font:16px sans-serif;margin:2em}}"
        "table{border-collapse:collapse}td,th{border:1px solid #aaa;padding:.4em}"
        "pre{white-space:pre-wrap}</style></head><body>",
        f"<h1>{title}</h1>",
        f"<p>{escape(table.profile)}; {'WSTFT' if table.weighted else 'STFT'}.</p>",
    ]
    if table.profile == "desktop_table_legacy":
        parts.append(
            "<p><strong>"
            + text(
                "Explicit native-table compatibility: two-decimal thresholds, strict integer "
                "de-escalation and no simulation run-in override. Native table behavior can "
                "differ from the calendar and can mark elimination without meeting posterior "
                "safety. This is not the safe simulation rule.",
                "显式原生表兼容模式：阈值保留两位小数，整数降量采用严格边界，不采用模拟导入覆盖规则。"
                "原生表可能与日历不同，并可能在未满足后验安全阈值时标记排除。此表不是安全模拟规则。",
            )
            + "</strong></p>"
        )
    parts.append(
        "<p>"
        + text(
            "At the lowest dose, apply the lowest-stop flag first. At other doses, elimination "
            "excludes that and higher doses. Otherwise use the fixed action, or escalate at "
            "STFT/WSTFT ≥ the upper threshold and de-escalate at STFT/WSTFT ≤ the lower "
            "threshold; stay between thresholds. A dash means that action is unavailable. "
            "Pending totals range from zero to the pending count. Unrounded thresholds are "
            "printed with round-trip precision. Dose exclusions and convergence still apply.",
            "最低剂量先应用最低剂量停止标记；其他剂量的排除同时排除更高剂量。否则执行固定动作，"
            "或 STFT/WSTFT 不低于升量阈值时升量、不高于降量阈值时降量，中间维持。横线表示无该动作。"
            "待观察总随访量介于零与待观察人数之间。未舍入阈值按可还原精度显示。仍需应用剂量排除及收敛规则。",
        )
        + "</p>"
    )
    steps = [
        text("Observed safety", "已观察安全性"),
        text("Observed DLT rule", "已观察 DLT 规则"),
        text("Pending cap", "待观察比例上限"),
        text("Imputed toxicity", "插补毒性"),
    ]
    diagram = ['<svg viewBox="0 0 1000 90" role="img" aria-label="Decision order">']
    for i, label in enumerate(steps):
        x = 10 + 250 * i
        diagram.append(
            f'<rect x="{x}" y="10" width="225" height="45" rx="5" fill="#eef3fa" stroke="#345"/>'
        )
        diagram.append(
            f'<text x="{x + 112}" y="38" text-anchor="middle" font-size="17">{escape(label)}</text>'
        )
        if i < 3:
            diagram.append(
                f'<path d="M{x + 225} 32 h20 m-5 -5 l5 5 -5 5" fill="none" stroke="#345"/>'
            )
    diagram.append("</svg>")
    parts.append("".join(diagram))
    parts.append(
        _table(
            [
                text("Patients", "例数"),
                text("DLT", "DLT"),
                text("Pending", "待观察"),
                text("Fixed action", "固定动作"),
                text("Escalate ≥", "升量 ≥"),
                text("De-escalate ≤", "降量 ≤"),
                text("Lowest stop", "最低剂量停止"),
                text("Posterior elimination", "后验排除"),
            ],
            [
                (
                    r.patients,
                    r.toxicities,
                    r.pending,
                    r.action or "—",
                    repr(r.escalate_at_least) if r.escalate_at_least is not None else "—",
                    repr(r.deescalate_at_most) if r.deescalate_at_most is not None else "—",
                    r.lowest_stop,
                    r.safety_eliminated,
                )
                for r in table.rows
            ],
        )
    )
    parts.append(
        "<details><summary>"
        + text("Complete saved inputs", "完整保存输入")
        + "</summary><pre>"
        + escape(study.xml)
        + "</pre></details></body></html>"
    )
    return study.to_html("".join(parts))
