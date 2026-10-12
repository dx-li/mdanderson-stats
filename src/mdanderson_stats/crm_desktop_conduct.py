"""CRM Suite patient conduct with explicit dates and independent inference.

Original patient censoring, within-cohort DA disposition and observed raw-rate
routing are preserved. BMA pending outcomes use exhaustive bounded completions;
native completion-bound and adaptive-MCMC shortcuts are not substituted.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import date
from html import escape
from math import sqrt

import numpy as np

from .bmacrm import fit_bmacrm
from .bmacrm_decision import BMACRMDecision, bmacrm_decision
from .bmacrm_lookahead import BMACRMLookAhead, bmacrm_lookahead
from .crm_calendar import CRMCalendarDecision, crm_calendar_decision
from .crm_desktop import _design_html, _envelope, _integer, _number, _text, read_crm_desktop_html
from .crm_desktop_history import (
    CRMDesktopHistory,
    CRMDesktopPatient,
    _settings,
    _whole,
    crm_desktop_snapshot,
)
from .dacrm import DACRMPosterior
from .dacrm_priors import dacrm_trimester_prior


def _incomplete_cohort(cohorts: tuple[int, ...], size: int) -> bool:
    if size == 1 or not cohorts:
        return False
    count = 0
    for value in reversed(cohorts):
        if value != cohorts[-1]:
            break
        count += 1
    return count < size


def _cohort_plan(cohorts: tuple[int, ...], size: int, action: str) -> tuple[int | None, int | None]:
    if action not in ("start", "treat"):
        return None, None
    within = _incomplete_cohort(cohorts, size)
    cohort = 1 if not cohorts else cohorts[-1] + (0 if within else 1)
    if not within:
        return cohort, None
    first = len(cohorts) - 1
    while first and cohorts[first - 1] == cohorts[-1]:
        first -= 1
    return cohort, 0 if cohort == 1 else first


@dataclass(frozen=True)
class CRMDesktopRecommendation:
    history: CRMDesktopHistory
    decision_date: date
    action: str
    dose: int | None
    can_add_patient: bool
    within_cohort: bool
    raw_rate_limited: bool
    analysis: CRMCalendarDecision
    explanation: str
    seed: int
    draws: int
    warmup: int
    chains: int
    cohort: int | None = None
    cohort_entry_analysis: CRMCalendarDecision | None = None

    @property
    def evaluations(self) -> int:
        return self.analysis.evaluations + (
            self.cohort_entry_analysis.evaluations if self.cohort_entry_analysis else 0
        )

    def add_patient(self, *, patient_id: int, fingerprint: int) -> CRMDesktopHistory:
        """Add the recommended patient with a pending, same-day assessment."""
        if not self.can_add_patient or self.dose is None or self.cohort is None:
            raise ValueError("this recommendation does not allow patient addition")
        patient = CRMDesktopPatient(
            patient_id,
            fingerprint,
            self.cohort,
            self.dose,
            self.decision_date,
            self.decision_date,
            -1,
            self.history.study.dose_labels[self.dose],
        )
        return self.history.with_patients(self.history.patients + (patient,))

    def to_html(self) -> str:
        """Capture patient inputs, dates, inference settings and recommendation."""
        posterior = self.analysis.posterior
        body = [
            "<h1>CRM Suite patient conduct: Python inference</h1>",
            "<p>Original day-resolution observations; independent posterior inference. "
            "BMA look-ahead uses bounded exhaustive completions. Native stochastic "
            "streams and adaptive stopping shortcuts are not reproduced.</p>",
            f"<p>Decision date: {self.decision_date}; date order: {self.history.date_order}; "
            f"action: {escape(self.action)}; zero-based dose: {self.dose}; "
            f"can add patient: {self.can_add_patient}.</p>",
            f"<p>{escape(self.explanation)}</p>",
            f"<p>Seed: {self.seed}; draws: {self.draws}; warmup: {self.warmup}; "
            f"chains: {self.chains}; posterior evaluations: {self.evaluations}.</p>",
            "<h2>Current analysis</h2><table><tr><th>Dose</th><th>Treated</th><th>Observed</th>"
            "<th>DLTs</th><th>Pending</th><th>Posterior mean</th></tr>",
        ]
        body.insert(1, _design_html(self.history.study))
        snap = self.analysis.snapshot
        for i, label in enumerate(self.history.study.dose_labels):
            values = [
                label,
                snap.treated_counts[i],
                snap.observed_counts[i],
                snap.toxicities[i],
                snap.pending_counts[i],
                posterior.dose_mean[i],
            ]
            body.append("<tr>" + "".join(f"<td>{escape(str(v))}</td>" for v in values) + "</tr>")
        body.append("</table>")
        if isinstance(posterior, DACRMPosterior):
            body.append(
                "<p>Parameter split R-hat: "
                + escape(str(posterior.parameter_summary.split_rhat))
                + "; parameter MCSE: "
                + escape(str(posterior.parameter_summary.batch_mean_mcse))
                + "</p>"
            )
        if self.cohort_entry_analysis is not None:
            entry = self.cohort_entry_analysis
            body.append(
                "<h2>Cohort-entry analysis</h2><p>This analysis determines the "
                "within-cohort recommendation after the current safety check.</p>"
            )
            body.append(
                "<table><tr><th>Dose</th><th>Observed</th><th>DLTs</th>"
                "<th>Pending</th><th>Posterior mean</th></tr>"
            )
            for i, label in enumerate(self.history.study.dose_labels):
                entry_values = (
                    label,
                    entry.snapshot.observed_counts[i],
                    entry.snapshot.toxicities[i],
                    entry.snapshot.pending_counts[i],
                    entry.posterior.dose_mean[i],
                )
                body.append(
                    "<tr>" + "".join(f"<td>{escape(str(v))}</td>" for v in entry_values) + "</tr>"
                )
            body.append("</table>")
            if isinstance(entry.posterior, DACRMPosterior):
                body.append(
                    "<p>Cohort-entry parameter split R-hat: "
                    + escape(str(entry.posterior.parameter_summary.split_rhat))
                    + "; parameter MCSE: "
                    + escape(str(entry.posterior.parameter_summary.batch_mean_mcse))
                    + "</p>"
                )
        xml = read_crm_desktop_html(self.history.to_html()).xml
        return _envelope("\n".join(body), xml)


def _recommend_crm_desktop(
    history: CRMDesktopHistory,
    *,
    at: date,
    seed: int,
    draws: int = 2000,
    warmup: int = 1000,
    chains: int = 2,
    max_completions: int = 128,
    max_evaluations: int = 2_000_000,
    replay_cohort: bool = True,
) -> CRMDesktopRecommendation:
    """Recommend from a saved patient history without extrapolating assessments.

    Historical/future-date analysis is available, but adding a patient is allowed
    only on/after every retained treatment and assessment date. At capacity,
    pending DA studies also use complete-outcome BMA look-ahead for final MTD
    selection. Resource exhaustion is an explicit wait or exception, not a guess.
    """
    _whole(seed, "seed", 0, 2**31 - 1)
    _whole(max_evaluations, "max_evaluations", 1, 2_000_000)
    _whole(max_completions, "max_completions", 1, 1024)
    snap = crm_desktop_snapshot(history, at=at)
    model = ET.fromstring(history.study.xml).find("CRMSimulatorData/Model")
    assert model is not None
    for key, expected in [
        ("PriorsAre_P", "1"),
        ("DoseEscalationRule", "eStrict"),
        ("MinPatientsToDeclareMTD", "3"),
    ]:
        if _text(model, key) != expected:
            raise ValueError(f"unsupported native {key}; refusing another policy")
    window, size, maximum = _settings(history.study)
    cohorts = tuple(history.patients[int(i)].cohort for i in snap.row_indices)
    within = _incomplete_cohort(cohorts, size)
    final = len(cohorts) >= maximum
    target = _number(model, "TargetToxicity")
    cutoff = _number(model, "UpperLimitOnProbToxAtLowestDose")
    start = _integer(model, "StartingDoseIndex", 0, len(history.study.dose_labels) - 1)
    current = int(snap.doses[-1]) if snap.doses.size else None
    pending = bool(np.any(snap.pending_counts))
    da = history.study.data_augmentation and pending and not final
    if da:
        prior = dacrm_trimester_prior(
            window,
            [
                float(_text(v, "."))
                for v in model.findall("ToxicityOccurrencePriorProbabilities/double")
            ],
            dispersion=2,
            endpoint_policy="native_desktop",
        )
        analysis = crm_calendar_decision(
            snap,
            history.study.skeletons,
            target=target,
            method="dacrm",
            da_prior=prior,
            current_dose=current,
            starting_dose=start,
            safety_cutoff=cutoff,
            minimum_observed=_integer(model, "MinimumNumberRequiredToEscalate", 0, 200),
            rng=np.random.default_rng(seed),
            draws=draws,
            warmup=warmup,
            chains=chains,
            max_evaluations=max_evaluations,
            max_completions=max_completions,
        )
    else:
        posterior = fit_bmacrm(
            history.study.skeletons,
            snap.toxicities,
            snap.observed_counts,
            target=target,
            prior_sd=sqrt(2),
            max_evaluations=min(200_000, max_evaluations),
        )
        used = posterior.evaluations
        decision: BMACRMDecision | BMACRMLookAhead
        if pending:
            if used >= max_evaluations:
                raise RuntimeError(
                    "desktop conduct exhausted its evaluation budget before look-ahead"
                )
            decision = bmacrm_lookahead(
                posterior,
                snap.pending_counts,
                current_dose=current,
                starting_dose=start,
                safety_cutoff=cutoff,
                final=final,
                apply_raw_rate=False,
                max_completions=max_completions,
                max_evaluations=max_evaluations - used,
            )
            used += decision.evaluations
        else:
            decision = bmacrm_decision(
                posterior,
                current_dose=current,
                starting_dose=start,
                safety_cutoff=cutoff,
                final=final,
                apply_raw_rate=False,
            )
        analysis = CRMCalendarDecision(
            snap, posterior, decision, "lookahead" if pending else "complete", used
        )
    action: str = analysis.decision.action
    dose = analysis.decision.dose
    explanation = getattr(
        analysis.decision,
        "explanation",
        "Bounded complete-outcome look-ahead: " + getattr(analysis.decision, "reason", ""),
    )
    limited = bool(getattr(analysis.decision, "raw_rate_limited", False))
    if da and within:
        limited = False
        # Original strict, pending, within-cohort branch retains the last dose;
        # only the DA safety flag can change treat to wait in this branch.
        safety = float(analysis.posterior.overdose_probability[0])
        action = "wait" if cutoff < 1 and safety > cutoff else "treat"
        dose = None if action == "wait" else current
        explanation = (
            "Wait for pending outcomes after a within-cohort DA safety signal."
            if action == "wait"
            else "Continue the current cohort at its retained dose."
        )
    if action in ("treat", "start") and not within and current is not None and dose is not None:
        for j in range(current, dose):
            if snap.observed_counts[j] and snap.toxicities[j] / snap.observed_counts[j] > target:
                dose, limited = j, True
                explanation = "Cap escalation at the first observed raw toxicity rate above target."
                break
    cohort, first = _cohort_plan(cohorts, size, action)
    entry_analysis = None
    if first is not None and replay_cohort:
        index = int(snap.row_indices[first])
        entry_date = history.patients[index].treatment_date
        before = history.with_patients(history.patients[:index])
        remaining = max_evaluations - analysis.evaluations
        if remaining < 1:
            raise RuntimeError("desktop conduct exhausted its cohort-entry replay budget")
        entry = _recommend_crm_desktop(
            before,
            at=entry_date,
            seed=int(np.random.SeedSequence(seed).generate_state(1)[0] % (2**31)),
            draws=draws,
            warmup=warmup,
            chains=chains,
            max_completions=max_completions,
            max_evaluations=remaining,
            replay_cohort=False,
        )
        action = "treat" if entry.action == "start" else entry.action
        dose = entry.dose
        entry_analysis = entry.analysis
        limited = entry.raw_rate_limited
        explanation = (
            "Current analysis permits treatment; use the recalculated cohort-entry "
            f"decision at {entry_date} from the corrected preceding history. " + entry.explanation
        )
    latest = max((p.assessment_date for p in history.patients), default=at)
    can_add = action in ("start", "treat") and at >= latest and len(history.patients) < maximum
    return CRMDesktopRecommendation(
        history,
        at,
        action,
        dose,
        can_add,
        within,
        limited,
        analysis,
        explanation,
        seed,
        draws,
        warmup,
        chains,
        cohort,
        entry_analysis,
    )


def recommend_crm_desktop(
    history: CRMDesktopHistory,
    *,
    at: date,
    seed: int,
    draws: int = 2000,
    warmup: int = 1000,
    chains: int = 2,
    max_completions: int = 128,
    max_evaluations: int = 2_000_000,
) -> CRMDesktopRecommendation:
    """Analyze saved patient data with explicit day-resolution observations.

    Current safety/wait decisions take precedence. If treatment is allowed within
    an incomplete cohort, recalculate its entry decision at its first treatment
    date using only preceding patients, matching the original managed wrapper.
    Independent exhaustive completion checks replace native bound shortcuts.
    """
    return _recommend_crm_desktop(
        history,
        at=at,
        seed=seed,
        draws=draws,
        warmup=warmup,
        chains=chains,
        max_completions=max_completions,
        max_evaluations=max_evaluations,
    )
