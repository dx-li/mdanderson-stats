"""Independent scrambled-Sobol posterior check and complete DSF workflow exercise."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.special import logsumexp, ndtri
from scipy.stats import qmc

from mdanderson_stats import (
    DoseScheduleDesktopSampler,
    DoseScheduleDesktopStudy,
    DoseSchedulePatient,
    fit_dose_schedule,
    simulate_dose_schedule_desktop,
)


def independent_risk(times, area, peak, tail, t):
    """Direct triangular integrals, independently vectorized over QMC samples."""
    x = np.maximum(t - np.asarray(times), 0)[None, None, :]
    a = area[:, :, None]
    b = peak[:, :, None]
    c = tail[:, :, None]
    fraction = np.where(
        x <= b, x * x / (b * (b + c)), 1 - np.maximum(b + c - x, 0) ** 2 / (c * (b + c))
    )
    return -np.expm1(-np.sum(a * fraction, axis=-1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=32)
    args = parser.parse_args()
    study = DoseScheduleDesktopStudy(
        (0.15, 0.3),
        (4, 6),
        (10, 12),
        2,
        3,
        14,
        28,
        0.3,
        0.4,
        0.8,
        maximum_patients=4,
        repetitions=args.trials,
        nu_area=3,
        nu_time=3,
        scenarios=(((0.05, 0.1), (0.15, 0.25)), ((0.2, 0.3), (0.4, 0.5)), ((0.6, 0.7), (0.8, 0.9))),
        low_grade_fraction=0.3,
        low_grade_delay=14,
    )
    prior = study.prior()
    histories = [
        DoseSchedulePatient(12, False, [0, 1, 2], [0, 0, 0]),
        DoseSchedulePatient(20, True, [0, 1, 2, 14, 15, 16], [1] * 6),
    ]
    references = []
    for seed in range(8):
        normal = ndtri(qmc.Sobol(6, scramble=True, seed=7500 + seed).random_base2(15))
        physical = np.exp((prior.mean + prior.sd * normal).reshape(-1, 2, 3))
        area = np.cumsum(physical[:, :, 0], axis=1)
        peak = physical[:, :, 1]
        tail = physical[:, :, 2]
        risk0 = independent_risk([0, 1, 2], area, peak, tail, 12)[:, 0]
        risk1 = independent_risk([0, 1, 2, 14, 15, 16], area, peak, tail, 20)[:, 1]
        elapsed = (20 - np.array([0, 1, 2, 14, 15, 16]))[None, :]
        b = peak[:, 1, None]
        c = tail[:, 1, None]
        a = area[:, 1, None]
        hazard = np.sum(
            np.where(
                elapsed <= b,
                2 * a * elapsed / (b * (b + c)),
                2 * a * np.maximum(b + c - elapsed, 0) / (c * (b + c)),
            ),
            axis=1,
        )
        with np.errstate(divide="ignore"):
            log_weight = np.log1p(-risk0) + np.log1p(-risk1) + np.log(hazard)
        weights = np.exp(log_weight - logsumexp(log_weight))
        risks = np.stack(
            [independent_risk(times, area, peak, tail, 28) for times in study.schedules()], axis=-1
        )
        references.append(np.sum(weights[:, None, None] * risks, axis=0))
    reference = np.mean(references, axis=0)
    reference_mcse = np.std(references, axis=0, ddof=1) / np.sqrt(len(references))
    fit = fit_dose_schedule(
        histories,
        prior,
        study.schedules(),
        28,
        draws=2000,
        warmup=1000,
        chains=4,
        rng=np.random.default_rng(750),
        max_evaluations=200000,
        max_work=20000000,
    )
    error = abs(fit.risk_summary.mean - reference)
    tolerance = 6 * np.sqrt(fit.risk_summary.batch_mean_mcse**2 + reference_mcse**2) + 0.001
    assert np.all(error < tolerance), (error, tolerance)
    assert np.max(fit.risk_summary.split_rhat) < 1.1
    results = []
    for profile in ("safe", "desktop_legacy"):
        result = simulate_dose_schedule_desktop(
            study,
            seed=751,
            profile=profile,
            sampler=DoseScheduleDesktopSampler(draws=32, warmup=16, chains=2),
        )
        assert np.allclose(result.selection_probability.sum(axis=1), 1)
        assert np.all(result.mean_enrollment.sum(axis=(1, 2)) <= study.maximum_patients)
        if profile == "safe":
            assert np.all(result.pending_final_probability == 0)
        results.append(
            {
                "profile": profile,
                "trials": args.trials * len(study.scenarios),
                "maximum_selection_mcse": float(np.max(result.selection_mcse)),
                "maximum_posterior_rhat": result.maximum_rhat,
                "maximum_posterior_risk_mcse": result.maximum_risk_mcse,
                "maximum_posterior_tail_mcse": result.maximum_tail_mcse,
                "pending_final_probability": result.pending_final_probability.tolist(),
                "likelihood_evaluations": result.likelihood_evaluations,
                "work_units": result.work_units,
            }
        )
    payload = {
        "independent_Sobol_points": 8 * 2**15,
        "posterior_mean": fit.risk_summary.mean.tolist(),
        "independent_reference_mean": reference.tolist(),
        "reference_mcse": reference_mcse.tolist(),
        "absolute_difference": error.tolist(),
        "six_joint_mcse_plus_001": tolerance.tolist(),
        "maximum_posterior_rhat": float(np.max(fit.risk_summary.split_rhat)),
        "workflow_checks": results,
        "qualification": (
            "Short-chain workflow simulations exercise complete operations; "
            "their posterior diagnostics do not establish clinical precision. "
            "Independent posterior accuracy is checked separately "
            "against scrambled Sobol integration."
        ),
    }
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(json.dumps(payload))


if __name__ == "__main__":
    main()
