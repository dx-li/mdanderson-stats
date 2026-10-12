"""Saved PRT conduct analysis with explicit paper/native profiles and quality holds."""

from __future__ import annotations

import csv
import io
import json
from dataclasses import asdict, dataclass
from typing import cast

import numpy as np

from .hierarchical_binomial import summarize_chains
from .prt import prt_decision, prt_predictive_risk
from .prt_fit import fit_prt_model, prt_isotonic_projection
from .prt_legacy import (
    prt_legacy_beta_moments,
    prt_legacy_conduct_decision,
    prt_legacy_predictive,
    prt_legacy_projection,
)
from .prt_study import PRTStudy, _integer, _number, prt_study_counts


@dataclass(frozen=True)
class PRTConductResult:
    study: PRTStudy
    profile: str
    status: str
    action: str
    dose: int | None
    rule: str
    provisional_action: str | None
    provisional_dose: int | None
    posterior_mean: tuple[float, ...]
    posterior_exceedance: tuple[float, ...]
    predictive_negligible: tuple[float | None, ...]
    predictive_excessive: tuple[float | None, ...]
    beta_alpha: tuple[float, ...]
    beta_beta: tuple[float, ...]
    assigned: tuple[int, ...]
    assessed: tuple[int, ...]
    events: tuple[int, ...]
    max_split_rhat: float | None
    max_risk_mcse: float | None
    maximum_condition_number: float | None
    chain_decisions_agree: bool | None
    likelihood_evaluations: int | None
    max_rhat: float
    max_mcse: float
    max_likelihood_evaluations: int
    rng_state_json: str
    warnings: tuple[str, ...]

    def to_json(self) -> str:
        data = asdict(self)
        data["study"] = json.loads(self.study.to_json())
        data["rng_state"] = json.loads(data.pop("rng_state_json"))
        return (
            json.dumps(
                {"schema": "mdanderson_stats.prt.conduct", "version": 1, **data},
                indent=2,
                allow_nan=False,
            )
            + "\n"
        )

    def to_csv(self) -> str:
        """Dose summary; metadata/action are retained in the accompanying JSON."""
        stream = io.StringIO(newline="")
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(
            [
                "dose_zero_based",
                "assigned",
                "assessed",
                "events",
                "posterior_mean",
                "posterior_exceedance",
                "predictive_negligible",
                "predictive_excessive",
                "beta_alpha",
                "beta_beta",
                "status",
                "profile",
                "action",
                "selected_dose_zero_based",
                "rule",
            ]
        )
        for k in range(self.study.doses):
            writer.writerow(
                [
                    k,
                    self.assigned[k],
                    self.assessed[k],
                    self.events[k],
                    *[
                        v[k] if v else ""
                        for v in (
                            self.posterior_mean,
                            self.posterior_exceedance,
                            self.predictive_negligible,
                            self.predictive_excessive,
                            self.beta_alpha,
                            self.beta_beta,
                        )
                    ],
                    self.status,
                    self.profile,
                    self.action,
                    self.dose,
                    self.rule,
                ]
            )
        return stream.getvalue()


def _evaluate(study: PRTStudy, raw: np.ndarray, profile: str):
    """One full or individual-chain estimate; no sampling or profile fallback."""
    _, _, assigned, assessed, _ = prt_study_counts(study)
    k = study.patients[-1].dose
    if profile == "paper_2008":
        projection = prt_isotonic_projection(raw)
        projected = projection.probability
        condition = projection.maximum_condition_number
    else:
        legacy = prt_legacy_projection(raw[:, :1])
        if not legacy.valid_probabilities:
            raise ArithmeticError("native total-risk projection exceeds one")
        projected = legacy.transformed
        condition = legacy.maximum_condition_number
    total = projected[:, 0]
    mean, xi = total.mean(axis=0), (total > study.target).mean(axis=0)
    moments = prt_legacy_beta_moments(mean, total.std(axis=0))
    if not np.all(moments.valid_moments):
        raise ArithmeticError(
            "posterior moments do not define proper beta distributions; no fallback applied"
        )
    pn: list[float | None] = [None] * study.doses
    pe: list[float | None] = [None] * study.doses
    relevant = range(study.doses) if profile == "paper_2008" else range(k, min(k + 2, study.doses))
    for dose in relevant:
        pending = [
            p
            for p in study.patients
            if p.dose == dose and p.censored and p.observed_interval < study.intervals
        ]
        if not pending:
            pn[dose], pe[dose] = (1.0, 0.0) if profile == "paper_2008" else (0.0, 0.0)
            continue
        if profile == "paper_2008":
            p = np.column_stack(
                [projected[:, patient.observed_interval, dose] for patient in pending]
            )
            risk = prt_predictive_risk(
                total[:, dose],
                p,
                target=study.target,
                negligible_cutoff=study.negligible_cutoff,
                excessive_cutoff=study.excessive_cutoff,
            )
            pn[dose], pe[dose] = risk.decision_negligible, risk.decision_excessive
        else:
            p = np.array([raw[:, patient.observed_interval, dose].mean() for patient in pending])
            risk_native = prt_legacy_predictive(
                float(moments.alpha[dose]),
                float(moments.beta[dose]),
                p,
                target=study.target,
                negligible_cutoff=study.negligible_cutoff,
                excessive_cutoff=study.excessive_cutoff,
            )
            pn[dose], pe[dose] = risk_native.negligible, risk_native.excessive
    warnings: tuple[str, ...]
    if profile == "paper_2008":
        decision = prt_decision(
            xi,
            np.asarray(pn, dtype=float),
            np.asarray(pe, dtype=float),
            assigned - assessed,
            k,
            epsilon=study.epsilon,
            negligible_cutoff=study.negligible_cutoff,
            excessive_cutoff=study.excessive_cutoff,
        )
        action, selected, rule, warnings = decision.action, decision.dose, decision.rule, ()
    else:
        native = prt_legacy_conduct_decision(
            xi,
            assigned,
            assessed,
            k,
            cast(float, pn[k]),
            cast(float, pe[k]),
            cast(float, pe[k + 1]) if k + 1 < study.doses else 0.0,
            epsilon=study.epsilon,
            negligible_cutoff=study.negligible_cutoff,
            excessive_cutoff=study.excessive_cutoff,
        )
        action, rule, warnings = native.action, native.rule, native.warnings
        selected = native.source_dose if action in ("stay", "escalate") else None
    return total, mean, xi, pn, pe, moments, action, selected, rule, warnings, condition


def analyze_prt_conduct(
    study: PRTStudy,
    *,
    rng: np.random.Generator,
    profile: str = "paper_2008",
    max_rhat: float = 1.05,
    max_mcse: float = 0.02,
    max_likelihood_evaluations: int = 200000,
) -> PRTConductResult:
    """Analyze a nonempty saved conduct history; keep invalid/unstable results on hold.

    The current dose is the last patient's dose. The legacy profile changes only
    the recovered projection/prediction/controller arithmetic, not the independent
    posterior sampler's RNG. Individual-chain reanalysis checks sensitivity to
    covariance, moments and cutoffs; it is not a confidence guarantee.
    """
    if not isinstance(study, PRTStudy) or study.max_patients is not None or not study.patients:
        raise ValueError("require a nonempty PRT conduct study")
    if profile not in ("paper_2008", "legacy_1_0"):
        raise ValueError("profile must be paper_2008 or legacy_1_0")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy Generator")
    max_rhat, max_mcse = _number(max_rhat, "max_rhat"), _number(max_mcse, "max_mcse")
    if not 1 < max_rhat <= 1.1 or not 0 < max_mcse <= 0.05:
        raise ValueError("require max_rhat in (1,1.1] and max_mcse in (0,.05]")
    cap = _integer(max_likelihood_evaluations, "max_likelihood_evaluations", 1, 20000000)
    s, e, assigned, assessed, pending = prt_study_counts(study)
    cells = study.chains * study.draws * study.intervals * study.doses
    work = (
        2
        * study.chains
        * study.draws
        * (study.intervals * study.doses**3 + sum((int(x) + 1) ** 2 for x in pending.sum(axis=0)))
    )
    sampling_work = study.chains * (study.draws + study.warmup) * study.intervals * study.doses
    likelihood_work = cap * study.intervals * study.doses
    if 8 * cells > 8000000 or work + sampling_work + likelihood_work > 100000000:
        raise ValueError("conduct fit/prediction exceeds bounded cell/work budget")
    if profile == "legacy_1_0" and np.any(pending.sum(axis=0) > 10):
        raise ValueError("legacy conduct supports at most10 pending patients per dose")
    state = json.dumps(
        rng.bit_generator.state, default=lambda x: x.tolist(), allow_nan=False, sort_keys=True
    )
    warnings = list(study.source_notes)
    if profile == "legacy_1_0":
        warnings.append(
            "Explicit legacy arithmetic; native RNG and clinical validity are not claimed."
        )
    diagnostics: dict = dict(
        study=study,
        profile=profile,
        status="numerical_hold",
        action="hold",
        dose=None,
        rule="",
        provisional_action=None,
        provisional_dose=None,
        posterior_mean=(),
        posterior_exceedance=(),
        predictive_negligible=(),
        predictive_excessive=(),
        beta_alpha=(),
        beta_beta=(),
        assigned=tuple(map(int, assigned)),
        assessed=tuple(map(int, assessed)),
        events=tuple(map(int, e.sum(axis=0))),
        max_split_rhat=None,
        max_risk_mcse=None,
        maximum_condition_number=None,
        chain_decisions_agree=None,
        likelihood_evaluations=None,
        max_rhat=max_rhat,
        max_mcse=max_mcse,
        max_likelihood_evaluations=cap,
        rng_state_json=state,
    )
    try:
        fit = fit_prt_model(
            s,
            e,
            prior_mean=study.prior_mean,
            prior_variance=study.prior_variance,
            draws=study.draws,
            warmup=study.warmup,
            chains=study.chains,
            rng=rng,
            max_likelihood_evaluations=cap,
        )
        diagnostics["likelihood_evaluations"] = fit.likelihood_evaluations
        raw_rhat = np.concatenate(
            [fit.beta_summary.split_rhat.ravel(), fit.risk_summary.split_rhat.ravel()]
        )
        raw_mcse = fit.risk_summary.batch_mean_mcse
        if np.all(np.isfinite(raw_rhat)):
            diagnostics["max_split_rhat"] = float(raw_rhat.max())
        if np.all(np.isfinite(raw_mcse)):
            diagnostics["max_risk_mcse"] = float(raw_mcse.max())
        raw = fit.conditional_toxicity[..., :-1, :]
        result = _evaluate(study, raw.reshape(-1, study.intervals, study.doses), profile)
        total, mean, xi, pn, pe, moments, action, selected, rule, branch_warnings, condition = (
            result
        )
        warnings.extend(branch_warnings)
        diagnostics.update(
            posterior_mean=tuple(map(float, mean)),
            posterior_exceedance=tuple(map(float, xi)),
            predictive_negligible=tuple(pn),
            predictive_excessive=tuple(pe),
            beta_alpha=tuple(map(float, moments.alpha)),
            beta_beta=tuple(map(float, moments.beta)),
            provisional_action=action,
            provisional_dose=selected,
            rule=rule,
            maximum_condition_number=condition,
        )
        transformed = summarize_chains(total.reshape(study.chains, study.draws, study.doses))
        xi_summary = summarize_chains(
            (total > study.target).reshape(study.chains, study.draws, study.doses).astype(float)
        )
        rhat = np.concatenate(
            [
                fit.beta_summary.split_rhat.ravel(),
                fit.risk_summary.split_rhat.ravel(),
                transformed.split_rhat.ravel(),
            ]
        )
        mcse = np.concatenate(
            [
                fit.risk_summary.batch_mean_mcse.ravel(),
                transformed.batch_mean_mcse.ravel(),
                xi_summary.batch_mean_mcse.ravel(),
            ]
        )
        if np.all(np.isfinite(rhat)):
            diagnostics["max_split_rhat"] = float(rhat.max())
        if np.all(np.isfinite(mcse)):
            diagnostics["max_risk_mcse"] = float(mcse.max())
        stable = True
        for chain in raw:
            try:
                check = _evaluate(study, chain, profile)
                stable &= (check[6], check[7], check[8]) == (action, selected, rule)
            except (ValueError, ArithmeticError):
                stable = False
        diagnostics["chain_decisions_agree"] = bool(stable)
        if not np.all(np.isfinite(rhat)) or np.max(rhat) > max_rhat:
            warnings.append("Split-Rhat is undefined or exceeds the configured limit.")
        elif not np.all(np.isfinite(mcse)) or np.max(mcse) > max_mcse:
            warnings.append("Risk/exceedance MCSE is undefined or exceeds the configured limit.")
        elif not stable:
            warnings.append("Individual-chain projections/decisions disagree or are invalid.")
        elif branch_warnings or action.startswith("legacy_"):
            warnings.append(
                "Native controller output is incomplete or invalid; no actionable dose."
            )
        else:
            diagnostics.update(status="computed", action=action, dose=selected)
    except (ArithmeticError, ValueError) as exc:
        warnings.append(str(exc))
    warnings.append(
        "Sampling diagnostics and chain agreement are screening checks, "
        "not convergence or clinical guarantees."
    )
    return PRTConductResult(**diagnostics, warnings=tuple(warnings))
