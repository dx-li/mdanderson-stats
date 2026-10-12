"""Captured-input DSF conduct/simulation reports and portable posterior draws."""

import json
from dataclasses import asdict, dataclass
from html import escape

import numpy as np

from .dose_schedule_desktop_kernel import _profile
from .dose_schedule_desktop_study import (
    _MAX_INPUT,
    DoseScheduleDesktopStudy,
    _unique,
    read_dose_schedule_desktop_study,
)
from .dose_schedule_desktop_workflow import (
    DoseScheduleDesktopAnalysis,
    DoseScheduleDesktopSampler,
    DoseScheduleDesktopSimulation,
    DoseScheduleDesktopTrial,
    _seed,
)


def _table(headers, rows):
    def row(values, tag):
        return "<tr>" + "".join(f"<{tag}>{escape(str(v))}</{tag}>" for v in values) + "</tr>"

    return "<table>" + row(headers, "th") + "".join(row(v, "td") for v in rows) + "</table>"


def _inputs(study):
    values = asdict(study)
    rows = [
        (name, value) for name, value in values.items() if name not in ("scenarios", "patients")
    ]
    return "<h2>Captured study parameters</h2>" + _table(("Parameter", "Value"), rows)


def _document(result, body, kind):
    settings = {
        "kind": kind,
        "profile": result.profile,
        "seed": result.seed,
        "sampler": asdict(result.sampler),
    }
    if kind == "analysis":
        settings["final"] = result.final
    return (
        '<!doctype html><html lang="en"><meta charset="utf-8">'
        "<title>Dose Schedule Finder report</title>"
        "<style>body{font:16px system-ui;margin:2rem}table{border-collapse:collapse;margin:1rem 0}"
        "td,th{padding:.4rem;border:1px solid #aaa}"
        "svg{max-width:600px;width:100%;display:block}</style><body>"
        "<h1>Dose Schedule Finder</h1>"
        f"<p>Profile: {escape(result.profile)}; seed {result.seed}. Python posterior sampler: "
        f"{result.sampler.chains} chains, {result.sampler.draws} retained draws per chain, "
        f"{result.sampler.warmup} warmup iterations.</p>"
        "<p>Source compatibility, mathematical correctness "
        "and clinical validity are separate claims. "
        "Native integration and native random-stream equivalence are not claimed. "
        "Inspect posterior "
        "precision as well as simulation Monte Carlo errors.</p>"
        "<p>Legacy profile retains cutoff equality, propagated dose exclusions, final no-skip and "
        "fixed 14-day low-grade qualification. Safe profile evaluates each regimen independently "
        "and requires completed follow-up/adjudication for final analysis.</p>"
        + _inputs(result.study)
        + body
        + '<script id="dsf-study" type="application/json">'
        + result.study.to_json().replace("<", "\\u003c")
        + "</script>"
        + '<script id="dsf-run" type="application/json">'
        + json.dumps(settings, allow_nan=False)
        + "</script></body></html>"
    )


def _precision(result):
    return (
        f"<p>Maximum split R-hat: {result.maximum_rhat:.6g}; "
        f"maximum posterior risk MCSE: {result.maximum_risk_mcse:.6g}; "
        f"maximum overdose-indicator MCSE: {result.maximum_tail_mcse:.6g}. "
        "Nonfinite diagnostics and short chains do not establish precision.</p>"
    )


def _density(values, label, kind):
    density, edges = np.histogram(values, bins=np.linspace(0, 1, 31), density=True)
    height = max(float(np.max(density)), 1.0)
    rectangles = []
    for i, value in enumerate(density):
        y = 150 - float(value) / height * 140
        rectangles.append(
            f'<rect x="{i * 16 + 10}" y="{y:g}" width="15" height="{150 - y:g}" fill="#2865a8"/>'
        )
    return (
        f"<figure><figcaption>{escape(label)}: {kind} risk density</figcaption>"
        '<svg viewBox="0 0 510 180" role="img" aria-label="Risk density">'
        + "".join(rectangles)
        + '<text x="10" y="175">0</text><text x="480" y="175">1</text></svg></figure>'
    )


def dose_schedule_desktop_analysis_report(result: DoseScheduleDesktopAnalysis) -> str:
    """Standalone conduct/prior report with regimen densities and saved inputs."""
    if not isinstance(result, DoseScheduleDesktopAnalysis):
        raise ValueError("result must be DoseScheduleDesktopAnalysis")
    decision = result.decision
    body = (
        f"<h2>{'Final' if result.final else 'Interim'} analysis "
        f"at day {result.study.study_time:g}</h2>"
        f"<p>Action: {escape(decision.action)}; "
        f"selected zero-based pair: {escape(str(decision.pair))}. "
        f"Follow-up ready: {result.final_ready}; "
        f"pending patient indices: {result.pending_patients}.</p>" + _precision(result)
    )
    rows = []
    for dose, schedule in np.ndindex(decision.mean_risk.shape):
        values = result.fit.regimen_risk[:, :, dose, schedule].ravel()
        rows.append(
            (
                dose + 1,
                schedule + 1,
                f"{decision.mean_risk[dose, schedule]:.8g}",
                f"{np.std(values, ddof=1):.8g}",
                f"{decision.overdose_probability[dose, schedule]:.8g}",
                bool(decision.acceptable[dose, schedule]),
                bool(decision.eligible[dose, schedule]),
            )
        )
    body += _table(
        (
            "Dose",
            "Schedule",
            "Mean risk",
            "SD",
            "Overdose probability",
            "Safety screen",
            "Eligible",
        ),
        rows,
    )
    body += "<h2>Actual treatment ledger</h2>" + _table(
        (
            "Patient",
            "Entry day",
            "Assigned dose/schedule (1-based)",
            "Outcome code",
            "Outcome/censor time",
            "Actual day:dose (1-based)",
        ),
        [
            (
                i + 1,
                p.entry_time,
                (p.assigned_dose + 1, p.assigned_schedule + 1),
                p.outcome,
                p.outcome_time,
                ", ".join(
                    f"{t:g}:{d + 1}"
                    for t, d in zip(p.administration_times, p.dose_indices, strict=True)
                ),
            )
            for i, p in enumerate(result.study.patients)
        ],
    )
    body += f"<h2>{'Prior' if result.fit.direct_prior else 'Posterior'} distributions</h2>"
    for dose, schedule in np.ndindex(decision.mean_risk.shape):
        body += _density(
            result.fit.regimen_risk[:, :, dose, schedule].ravel(),
            f"Dose {dose + 1}, schedule {schedule + 1}",
            "prior" if result.fit.direct_prior else "posterior",
        )
    return _document(result, body, "analysis")


def dose_schedule_desktop_simulation_report(result: DoseScheduleDesktopSimulation) -> str:
    """All scenarios, selection/enrollment precision, toxicity and duration."""
    if not isinstance(result, DoseScheduleDesktopSimulation):
        raise ValueError("result must be DoseScheduleDesktopSimulation")
    body = f"<h2>Simulation: {result.study.repetitions} trials per scenario</h2>" + _precision(
        result
    )
    body += "<p>Event generation uses exponential arrivals and the saved event distribution, with "
    body += "an explicit 0.01-day minimum event time. "
    body += "Every generated low-grade event eventually qualifies; "
    body += "these are simulation assumptions, not clinical adjudications. "
    body += "Pooled toxicity is zero where no patients were allocated.</p>"
    for index, truth in enumerate(result.study.scenarios):
        body += f"<h3>Scenario {index + 1}</h3>"
        rows = []
        mean_toxicities = result.pooled_toxicity[index] * result.mean_enrollment[index]
        for dose, schedule in np.ndindex(np.shape(truth)):
            flat = dose * result.study.schedule_count + schedule
            rows.append(
                (
                    dose + 1,
                    schedule + 1,
                    truth[dose][schedule],
                    f"{result.selection_probability[index, flat]:.8g}",
                    f"{result.selection_mcse[index, flat]:.8g}",
                    f"{result.mean_enrollment[index, dose, schedule]:.8g}",
                    f"{result.enrollment_mcse[index, dose, schedule]:.8g}",
                    f"{result.pooled_toxicity[index, dose, schedule]:.8g}",
                    f"{mean_toxicities[dose, schedule]:.8g}",
                )
            )
        body += _table(
            (
                "Dose",
                "Schedule",
                "Truth",
                "Selection probability",
                "Selection MCSE",
                "Mean enrollment",
                "Enrollment MCSE",
                "Pooled toxicity",
                "Mean toxicities",
            ),
            rows,
        )
        body += (
            f"<p>Mean total enrollment: {np.sum(result.mean_enrollment[index]):.8g}; "
            f"mean total toxicities: {np.sum(mean_toxicities):.8g}.</p>"
            f"<p>No selection: {result.selection_probability[index, -1]:.8g} "
            f"(MCSE {result.selection_mcse[index, -1]:.8g}); "
            f"early stop: {result.early_stop_probability[index]:.8g}; "
            f"pending at final analysis: {result.pending_final_probability[index]:.8g}; "
            f"mean duration: {result.mean_duration[index]:.8g} days "
            f"(MCSE {result.duration_mcse[index]:.8g}).</p>"
        )
    return _document(result, body, "simulation")


def dose_schedule_desktop_distribution_json(result: DoseScheduleDesktopAnalysis) -> str:
    """Export named log-parameter and risk draws with exact analysis context."""
    if not isinstance(result, DoseScheduleDesktopAnalysis):
        raise ValueError("result must be DoseScheduleDesktopAnalysis")
    return (
        json.dumps(
            {
                "schema": "mdanderson-stats/dose-schedule-draws/v1",
                "study": json.loads(result.study.to_json()),
                "profile": result.profile,
                "seed": result.seed,
                "sampler": asdict(result.sampler),
                "final": result.final,
                "parameter_names": result.fit.parameter_names,
                "log_parameters": result.fit.log_parameters.tolist(),
                "regimen_risk": result.fit.regimen_risk.tolist(),
            },
            allow_nan=False,
            separators=(",", ":"),
        )
        + "\n"
    )


@dataclass(frozen=True)
class DoseScheduleDesktopSavedReport:
    study: DoseScheduleDesktopStudy
    profile: str
    seed: int
    sampler: DoseScheduleDesktopSampler
    kind: str
    final: bool


def read_dose_schedule_desktop_report(text: str) -> DoseScheduleDesktopSavedReport:
    """Restore inputs and explicit execution settings without executing HTML."""
    if not isinstance(text, str) or len(text.encode("utf-8")) > _MAX_INPUT:
        raise ValueError("report exceeds bounded size")

    def extract(name):
        marker = f'<script id="{name}" type="application/json">'
        if text.count(marker) != 1:
            raise ValueError("missing or duplicate captured data")
        return text.split(marker, 1)[1].split("</script>", 1)[0]

    study = read_dose_schedule_desktop_study(extract("dsf-study"))
    settings = json.loads(extract("dsf-run"), object_pairs_hook=_unique)
    if settings.get("kind") not in ("analysis", "simulation"):
        raise ValueError("unknown report kind")
    required = {"kind", "profile", "seed", "sampler"} | (
        {"final"} if settings["kind"] == "analysis" else set()
    )
    if set(settings) != required or not isinstance(settings.get("final", False), bool):
        raise ValueError("invalid report execution fields")
    return DoseScheduleDesktopSavedReport(
        study,
        _profile(settings["profile"]),
        _seed(settings["seed"]),
        DoseScheduleDesktopSampler(**settings["sampler"]),
        settings["kind"],
        settings.get("final", False),
    )


def dose_schedule_desktop_trial_json(result: DoseScheduleDesktopTrial) -> str:
    """Export detailed trial history and supplied tapes without retaining a batch.

    Use as a streaming simulation callback to write one scenario/trial at a time.
    Nonfinite precision diagnostics are explicitly represented by null; posterior
    draws are available separately through the final-analysis distribution export.
    """
    if not isinstance(result, DoseScheduleDesktopTrial):
        raise ValueError("result must be DoseScheduleDesktopTrial")

    def clean(value):
        if isinstance(value, np.ndarray):
            return clean(value.tolist())
        if isinstance(value, np.generic):
            return clean(value.item())
        if isinstance(value, float) and not np.isfinite(value):
            return None
        if isinstance(value, dict):
            return {k: clean(v) for k, v in value.items()}
        if isinstance(value, (tuple, list)):
            return [clean(v) for v in value]
        return value

    payload = {
        "schema": "mdanderson-stats/dose-schedule-trial/v1",
        "study": json.loads(result.study.to_json()),
        "profile": result.analysis.profile,
        "truth": result.truth,
        "sampler": asdict(result.sampler),
        "max_evaluations": result.max_evaluations,
        "max_work": result.max_work,
        "arrival_uniforms": result.arrival_uniforms,
        "event_uniforms": result.event_uniforms,
        "grade_uniforms": result.grade_uniforms,
        "posterior_seeds": result.posterior_seeds,
        "assignments": result.assignments,
        "patients": [asdict(p) for p in result.analysis.study.patients],
        "decisions": [asdict(d) for d in result.decisions],
        "final_decision": asdict(result.analysis.decision),
        "duration": result.duration,
        "final_analysis_time": result.analysis.study.study_time,
        "pending_final_patients": result.analysis.pending_patients,
        "stopped_early": result.stopped_early,
        "maximum_rhat": result.maximum_rhat,
        "maximum_risk_mcse": result.maximum_risk_mcse,
        "maximum_tail_mcse": result.maximum_tail_mcse,
        "undefined_diagnostic_encoding": "null",
    }
    return json.dumps(clean(payload), allow_nan=False, separators=(",", ":")) + "\n"
