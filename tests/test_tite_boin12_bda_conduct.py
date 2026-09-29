import numpy as np

from mdanderson_stats.boin12 import BOIN12Design
from mdanderson_stats.tite_boin12 import tite_boin12_decision
from mdanderson_stats.tite_boin12_bda import tite_boin12_bda_decision


def test_bda_conduct_reduces_to_al_movement_when_all_endpoints_are_observed():
    design = BOIN12Design(0.35, 0.25, utilities=(100.0, 30.0, 65.0, 0.0))
    inputs = (
        [1, 1, 2, 2, 2],
        [0, 0, 0, 1, 0],
        [1, 0, 1, 1, 0],
        [1.0, 1.0, 1.0, 0.2, 1.0],
        [1.0, 2.0, 0.2, 0.3, 2.0],
    )
    common = dict(
        toxicity_window=1.0,
        efficacy_window=2.0,
        n_doses=2,
        current_dose=1,
    )
    al = tite_boin12_decision(design, *inputs, **common)
    bda = tite_boin12_bda_decision(
        design,
        *inputs,
        **common,
        prior_concentrations=[1.2, 0.8, 0.3, 0.7],
        rng=np.random.default_rng(117),
        draws=24,
        warmup=8,
        chains=2,
    )
    assert (bda.action, bda.next_dose) == (al.action, al.next_dose)
    assert bda.posterior is not None
    np.testing.assert_allclose(bda.imputed_toxicity_rate, [0.0, 1 / 3])


def test_bda_current_dose_pending_gate_is_strict_and_precedes_sampling():
    design = BOIN12Design(0.35, 0.25)
    rng = np.random.default_rng(202)
    before = rng.bit_generator.state
    result = tite_boin12_bda_decision(
        design,
        [1, 1],
        [-1, -1],
        [1, 0],
        [0.2, 0.3],
        [0.2, 2.0],
        toxicity_window=1.0,
        efficacy_window=2.0,
        n_doses=1,
        current_dose=1,
        prior_concentrations=[0.5, 0.25, 0.125, 0.125],
        rng=rng,
        draws=20,
        warmup=0,
        chains=2,
    )
    assert result.action == "suspend_pending"
    assert result.posterior is None
    assert rng.bit_generator.state == before

    at_half = tite_boin12_bda_decision(
        design,
        [1, 1],
        [0, -1],
        [1, 0],
        [1.0, 0.0],
        [2.0, 2.0],
        toxicity_window=1.0,
        efficacy_window=2.0,
        n_doses=1,
        current_dose=1,
        prior_concentrations=[0.5, 0.25, 0.125, 0.125],
        rng=np.random.default_rng(203),
        draws=20,
        warmup=0,
        chains=2,
    )
    assert at_half.action != "suspend_pending"
    assert at_half.posterior is not None


def test_bda_does_not_apply_al_zero_ess_guard_to_other_doses():
    design = BOIN12Design(0.35, 0.25)
    result = tite_boin12_bda_decision(
        design,
        [1, 1, 2],
        [0, 0, -1],
        [1, 1, 1],
        [1.0, 1.0, 0.0],
        [2.0, 2.0, 2.0],
        toxicity_window=1.0,
        efficacy_window=2.0,
        n_doses=2,
        current_dose=1,
        prior_concentrations=[0.5, 0.25, 0.125, 0.125],
        rng=np.random.default_rng(221),
        draws=24,
        warmup=8,
        chains=2,
    )
    assert result.action != "suspend_no_information"
    assert result.posterior is not None


def test_bda_reuses_explicit_python_runin_and_precision_policies():
    run_in = tite_boin12_bda_decision(
        BOIN12Design(0.25, 0.25, toxicity_cutoff=0.95, efficacy_cutoff=0.95),
        [2, 2, 2],
        [1, 1, 0],
        [1, 1, 1],
        [0.2, 0.3, 1.0],
        [0.5, 0.7, 2.0],
        toxicity_window=1.0,
        efficacy_window=2.0,
        n_doses=2,
        current_dose=2,
        prior_concentrations=[0.5, 0.25, 0.125, 0.125],
        rng=np.random.default_rng(610),
        draws=24,
        warmup=8,
        chains=2,
        run_in_3plus3=True,
    )
    assert (run_in.action, run_in.next_dose) == ("deescalate", 1)

    precision = tite_boin12_bda_decision(
        BOIN12Design(0.35, 0.25, early_stop_patients=2),
        [1, 1],
        [0, 0],
        [1, 1],
        [1.0, 1.0],
        [2.0, 2.0],
        toxicity_window=1.0,
        efficacy_window=2.0,
        n_doses=1,
        current_dose=1,
        prior_concentrations=[0.5, 0.25, 0.125, 0.125],
        rng=np.random.default_rng(611),
        draws=20,
        warmup=0,
        chains=2,
    )
    assert precision.action == "stop_precision"
