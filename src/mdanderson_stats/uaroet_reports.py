"""Portable UAROET assignment reports and bounded streaming posterior exports."""

import csv
import io
import json
from typing import TextIO

import numpy as np

from .uaroet import _integer
from .uaroet_desktop_conduct import UAROETDesktopDecision
from .uaroet_desktop_study import UAROETDesktopStudy
from .uaroet_fit import UAROETFit

_STATUS = {
    1: "eligible",
    -1: "futile",
    -2: "toxic",
    -3: "utility_gap",
    -4: "low_best_probability",
    -5: "no_skip",
}


class UAROETDistributionWriter:
    """CSV callback for fit/recommend on_chunk; dose, chain and draw indices are1-based.

    Writes every marginal category and utility without accumulating all posterior
    grids. A partial stream after any error is diagnostic output only. Call
    validate_complete after the returned fit to establish complete export coverage.
    This is a documented Python CSV schema, not native text-file byte identity.
    """

    def __init__(self, stream: TextIO, study: UAROETDesktopStudy):
        if not isinstance(study, UAROETDesktopStudy) or not callable(
            getattr(stream, "write", None)
        ):
            raise ValueError("require a text stream and UAROETDesktopStudy")
        self.study = study
        self.writer = csv.writer(stream, lineterminator="\n")
        self.next_draw: dict[int, int] = {}
        self.rows_written = 0
        self.writer.writerow(
            [
                "chain_1based",
                "draw_1based",
                "dose_1based",
                *(f"efficacy_p{k}" for k in range(study.efficacy_levels)),
                *(f"toxicity_p{k}" for k in range(study.toxicity_levels)),
                "utility",
            ]
        )

    def __call__(self, chain: int, first_draw: int, fit: UAROETFit) -> None:
        c = _integer(chain, "chain", 0, 19)
        first = _integer(first_draw, "first_draw", 0, 1_000_000)
        expected = (self.study.dose_count, self.study.efficacy_levels, self.study.toxicity_levels)
        if (
            not isinstance(fit, UAROETFit)
            or fit.joint.ndim != 5
            or fit.joint.shape[0] != 1
            or fit.joint.shape[2:] != expected
        ):
            raise ValueError("chunk must contain one chain on the study grid")
        if first != self.next_draw.get(c, 0):
            raise ValueError("posterior chunks must be contiguous and nonduplicated per chain")
        j = fit.joint[0]
        e, t = j.sum(-1), j.sum(-2)
        utility = np.einsum("ndet,et->nd", j, self.study.utility)
        for draw in range(len(j)):
            for dose in range(self.study.dose_count):
                self.writer.writerow(
                    [
                        c + 1,
                        first + draw + 1,
                        dose + 1,
                        *e[draw, dose],
                        *t[draw, dose],
                        utility[draw, dose],
                    ]
                )
                self.rows_written += 1
        self.next_draw[c] = first + len(j)

    def validate_complete(self, *, chains: int, draws_per_chain: int) -> None:
        count = _integer(chains, "chains", 1, 20)
        draws = _integer(draws_per_chain, "draws_per_chain", 8, 1_000_000)
        if self.next_draw != {c: draws for c in range(count)}:
            raise ValueError("distribution stream does not cover the complete fit")


def _finite_list(array: np.ndarray) -> object:
    """Represent unavailable/infinite diagnostics as null; report their status separately."""
    if array.ndim == 0:
        return float(array) if np.isfinite(array) else None
    return [_finite_list(x) for x in array]


def uaroet_assignment_report(decision: UAROETDesktopDecision) -> str:
    """Versioned JSON with exact input study,1-based reported doses and quality evidence."""
    if not isinstance(decision, UAROETDesktopDecision):
        raise ValueError("decision must be a UAROETDesktopDecision")
    s = decision.study
    counts = s.counts()
    fit = decision.posterior
    posterior = None
    if fit is not None:
        posterior = dict(
            mean_joint=fit.mean_joint.tolist(),
            mean_parameters=fit.mean_parameters.tolist(),
            mean_association=fit.mean_association,
            mean_utility=fit.utility_samples.mean((0, 1)).tolist(),
            probability_best=fit.probability_best.tolist(),
            toxicity_exceedance=fit.toxicity_exceedance.tolist(),
            efficacy_shortfall=fit.efficacy_shortfall.tolist(),
            precision_met=fit.precision_met,
            target=fit.target,
            monitored_doses_1based=[d + 1 for d in fit.monitored_doses],
            draws_per_chain=fit.draws_per_chain,
            chains=fit.utility_samples.shape[0],
            utility_mcse_ratio=_finite_list(fit.utility_mcse_ratio),
            utility_mcse_ratio_defined=np.isfinite(fit.utility_mcse_ratio).tolist(),
            utility_split_rhat=_finite_list(fit.utility_split_rhat),
            utility_rhat_status=[
                "unavailable" if np.isnan(x) else "divergent" if np.isinf(x) else "finite"
                for x in fit.utility_split_rhat
            ],
            work_evaluations=fit.work_evaluations,
        )
    result = dict(
        schema="mdanderson-stats.uaroet-assignment",
        version=1,
        input_study=json.loads(s.to_json()),
        profile=decision.profile,
        action=decision.action,
        reason=decision.reason,
        final=decision.final,
        selected_dose_1based=None if decision.selected is None else decision.selected + 1,
        status=list(decision.status),
        status_labels=[_STATUS[x] for x in decision.status],
        allocation_weights=decision.allocation_weights.tolist(),
        utility_cutoff=decision.utility_cutoff,
        assigned=counts.assigned.tolist(),
        complete=counts.complete.tolist(),
        likelihood_used=counts.likelihood.tolist(),
        posterior=posterior,
        native_rng_or_report_byte_identity=False,
        clinical_validation=False,
    )
    return json.dumps(result, indent=2, allow_nan=False) + "\n"


def uaroet_assignment_csv(decision: UAROETDesktopDecision) -> str:
    """Per-dose assignment summary; unevaluated posterior quantities are empty cells."""
    if not isinstance(decision, UAROETDesktopDecision):
        raise ValueError("decision must be a UAROETDesktopDecision")
    stream = io.StringIO()
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(
        [
            "dose_1based",
            "status",
            "assigned",
            "complete",
            "likelihood_used",
            "mean_utility",
            "probability_best",
            "toxicity_exceedance",
            "efficacy_shortfall",
            "allocation_probability",
        ]
    )
    counts = decision.study.counts()
    fit = decision.posterior
    for d in range(decision.study.dose_count):
        quantities = (
            ["", "", "", ""]
            if fit is None
            else [
                fit.utility_samples[:, :, d].mean(),
                fit.probability_best[d],
                fit.toxicity_exceedance[d],
                fit.efficacy_shortfall[d],
            ]
        )
        writer.writerow(
            [
                d + 1,
                _STATUS[decision.status[d]] if decision.status else "not_evaluated",
                counts.assigned[d],
                counts.complete[d],
                counts.likelihood[d],
                *quantities,
                decision.allocation_weights[d],
            ]
        )
    return stream.getvalue()
