"""Reopenable mTPI monitoring and scenario reports with captured inputs."""

import json
from dataclasses import dataclass
from html import escape

import numpy as np
from scipy.special import betainc

from .boin import _owned
from .mtpi import MTPITable
from .mtpi_desktop import MTPIDesktopTableStudy, _integer, _unique, read_mtpi_desktop_project
from .mtpi_desktop_simulation import MTPIDesktopSimulation


def mtpi_desktop_monitoring_table(study, *, profile="safe"):
    """Return E/S/D with U for overdose; legacy strict UPM ties remain blank."""
    if not isinstance(study, MTPIDesktopTableStudy):
        raise ValueError("study must be MTPIDesktopTableStudy")
    if profile not in ("safe", "desktop_legacy"):
        raise ValueError("profile must be safe or desktop_legacy")
    design = study.design()
    table = design.decision_table(study.maximum_patients)
    if profile == "safe":
        return table
    action = table.action.copy()
    for n in range(1, study.maximum_patients + 1):
        y = np.arange(n + 1)
        left = betainc(y + 1, n - y + 1, design.lower)
        right = betainc(y + 1, n - y + 1, design.upper)
        upm = np.stack(
            (
                left / design.lower,
                (right - left) / (study.epsilon_lower + study.epsilon_upper),
                (1 - right) / (1 - design.upper),
            ),
            axis=-1,
        )
        ties = np.sum(upm == upm.max(axis=-1, keepdims=True), axis=-1) > 1
        unsafe = 1 - betainc(y + 1, n - y + 1, design.target) > 0.95
        action[y, n - 1] = np.char.add(
            np.array(["E", "S", "D"])[upm.argmax(axis=-1)], np.where(unsafe, "U", "")
        )
        action[y[ties], n - 1] = ""
        action[y[unsafe], n - 1] = "DU"
    return MTPITable(table.patients, table.toxicities, _owned(action))


def _table(headers, rows):
    def row(values, tag):
        return "<tr>" + "".join(f"<{tag}>{escape(str(v))}</{tag}>" for v in values) + "</tr>"

    return "<table>" + row(headers, "th") + "".join(row(r, "td") for r in rows) + "</table>"


def _document(study, profile, body, seed=None):
    captured = study.to_json().replace("<", "\\u003c")
    return (
        '<!doctype html><html lang="en"><meta charset="utf-8"><title>mTPI report</title>'
        "<style>body{font:16px system-ui;margin:2rem}table{border-collapse:collapse;"
        "margin:1rem 0}td,th{border:1px solid #aaa;padding:.35rem}</style><body>"
        f"<h1>mTPI report</h1><p>Profile: {escape(profile)}.</p>"
        "<p>Source compatibility, mathematical correctness and clinical validity "
        "are separate claims. Simulation results do not establish clinical validity.</p>"
        + body
        + '<script id="mtpi-study" type="application/json">'
        + captured
        + '</script><script id="mtpi-run" type="application/json">'
        + json.dumps({"profile": profile, "seed": seed})
        + "</script></body></html>"
    )


def mtpi_desktop_table_report(study, *, profile="safe"):
    """Create a standalone monitoring report; saved inputs can be reopened."""
    table = mtpi_desktop_monitoring_table(study, profile=profile)
    body = (
        f"<p>Target {study.target:g}; equivalence interval "
        f"({study.target - study.epsilon_lower:g}, {study.target + study.epsilon_upper:g}). "
        "Uniform Beta(1,1) prior. E: escalate; S: stay; D: de-escalate; "
        "U: posterior overdose probability exceeds 0.95. This count table does "
        "not replace trial-level safety rules. Legacy exact UPM ties are blank.</p>"
    )
    body += _table(
        ["DLTs / patients", *table.patients],
        [[int(y), *table.action[i]] for i, y in enumerate(table.toxicities)],
    )
    return _document(study, profile, body)


def mtpi_desktop_simulation_report(result):
    """Render all scenarios, MCSEs, actual enrollment and qualified legacy rules."""
    if not isinstance(result, MTPIDesktopSimulation):
        raise ValueError("result must be MTPIDesktopSimulation")
    study = result.study
    body = (
        f"<p>Target {study.target}; "
        f"lower/upper margins {study.epsilon_lower}/{study.epsilon_upper}; "
        f"{study.trials} trials per scenario; seed {result.seed}; "
        f"planned maximum {study.maximum_patients}; actual capacity {result.actual_capacity}; "
        f"cohort size {study.cohort_size}; starting dose {study.starting_dose}; "
        f"{result.uniforms_consumed} uniforms consumed.</p>"
        "<p>Selection MCSE uses the binomial estimate; enrollment MCSE uses "
        "sample standard deviation (undefined for one trial). Pooled toxicity "
        "is total DLTs divided by total enrollment, not a mean of trial ratios.</p>"
    )
    if result.profile == "desktop_legacy":
        body += (
            "<p>Legacy compatibility preserves Jeffreys-prior stop rules, the "
            "next-dose beta shape omitting toxicity subtraction, carried top-dose "
            "indicator, overwritten exclusion history, whole-cohort overflow, "
            "and XLL final ranking. This profile is not the clinical default.</p>"
        )
    else:
        body += (
            "<p>Paper controller: persistent exclusions, uniform-prior safety, "
            "final cohort clipped to capacity.</p>"
        )
    for i, truth in enumerate(study.scenarios):
        body += f"<h2>Scenario {i + 1}</h2>"
        rows = []
        for j, probability in enumerate(truth):
            rows.append(
                [
                    j + 1,
                    probability,
                    result.selection_probability[i, j],
                    result.selection_mcse[i, j],
                    result.mean_patients[i, j],
                    result.mean_patients_mcse[i, j],
                ]
            )
        rows.append(
            [
                "No selection",
                "",
                result.selection_probability[i, -1],
                result.selection_mcse[i, -1],
                "",
                "",
            ]
        )
        body += _table(
            [
                "Dose",
                "True toxicity",
                "Selection",
                "Selection MCSE",
                "Mean patients",
                "Enrollment MCSE",
            ],
            rows,
        )
        body += _table(
            [
                "Mean total patients",
                "Pooled toxicity",
                "Safety stop",
                "Underdose stop",
                "Capacity reached",
                "Any assignment unsafe under paper posterior",
            ],
            [
                [
                    result.mean_total_patients[i],
                    result.pooled_toxicity_rate[i],
                    *result.stopping_probability[i],
                    result.unsafe_assignment_probability[i],
                ]
            ],
        )
    return _document(study, result.profile, body, result.seed)


@dataclass(frozen=True)
class MTPIDesktopReportInputs:
    """Reopened study plus the profile and seed needed to reproduce its report."""

    study: object
    profile: str
    seed: int | None


def read_mtpi_desktop_report(text):
    """Recover captured scientific inputs and execution settings from our report."""
    study = read_mtpi_desktop_project(text)
    marker = '<script id="mtpi-run" type="application/json">'
    if text.count(marker) != 1:
        raise ValueError("require one captured execution-settings envelope")
    try:
        settings = json.loads(
            text.split(marker, 1)[1].split("</script>", 1)[0], object_pairs_hook=_unique
        )
    except json.JSONDecodeError as exc:
        raise ValueError("invalid captured execution settings") from exc
    if not isinstance(settings, dict) or set(settings) != {"profile", "seed"}:
        raise ValueError("invalid captured execution fields")
    profile, seed = settings["profile"], settings["seed"]
    if profile not in ("safe", "desktop_legacy"):
        raise ValueError("unsupported captured profile")
    if isinstance(study, MTPIDesktopTableStudy):
        if seed is not None:
            raise ValueError("monitoring report must not have a random seed")
    else:
        seed = _integer(seed, "seed", 0, 2**32 - 1)
    return MTPIDesktopReportInputs(study, profile, seed)
