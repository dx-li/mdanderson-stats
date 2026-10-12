"""Reopen a synthetic study and fit its disjoint observations with short demo chains."""

import numpy as np

from mdanderson_stats import UAROETDesktopStudy, fit_uaroet, read_uaroet_desktop_study

study = UAROETDesktopStudy(
    dose_count=2,
    efficacy_levels=2,
    toxicity_levels=2,
    prior_mean=np.array([0, 0.3, -1, 0.3]),
    prior_sd=np.array([0.7, 0.4, 0.7, 0.4]),
    utility=np.array([[30, 0], [100, 50]]),
    trial_parameters=dict(
        NumPatients=30,
        CohortSize_First=3,
        CohortSize_NewDose=1,
        CohortSize_OldDose=1,
        DoseStart=1,
        LowestUnacceptableToxLevel=1,
        MaxProbTox=0.2,
        UpperAcceptablePiToxCutoff=0.8,
        LowestAcceptableEffLevel=1,
        MinProbEff=0.2,
        UpperAcceptablePiEffCutoff=1,
        MinProbUtilityIsHighest=0.1,
        MinUtilityGoodSet=50,
        UseHybridGreedyArandAssignment=1,
        NumPatientSurplusBeforeRandomize=1,
        RandomizeByMeanUtility=1,
        InitialUtilityCutoffToRandomize=100,
        FinalUtilityCutoffToRandomize=0,
        NumTimesToShrinkUtilityCutoff=10,
        MinDaysToEfficacy=1,
        MaxDaysToEfficacy=42,
        MinDaysToToxicity=1,
        MaxDaysToToxicity=42,
        MeanDaysBetweenArrivals=45,
    ),
    simulation_parameters={"NumChains": 4, "BurnInSamples": 1000},
    patients=((1, 0, 1, 0), (2, 1, -1, 1), (3, 1, 1, -1)),
)
study = read_uaroet_desktop_study(study.to_json())
counts = study.counts()
assert counts.assigned.sum() == 3
assert counts.complete.sum() == 1
assert counts.likelihood.sum() == 2
fit = fit_uaroet(
    counts.joint,
    toxicity_only_counts=counts.toxicity_only,
    prior_mean=study.prior_mean,
    prior_sd=study.prior_sd,
    association=0,
    draws=64,
    warmup=32,
    chains=2,
    rng=np.random.default_rng(92),
)
assert not fit.direct_prior
print(
    "Assigned / complete / likelihood counts:", counts.assigned, counts.complete, counts.likelihood
)
print("Short demo chains; not target-precision evidence:", fit.parameter_summary.split_rhat)
