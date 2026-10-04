import numpy as np
import pytest

from mdanderson_stats.bop2_dc import bop2_dc_design
from mdanderson_stats.bop2_dc_categorical import bop2_dc_categorical_design
from mdanderson_stats.bop2_dc_categorical_simulation import simulate_bop2_dc_categorical
from mdanderson_stats.bop2_dc_normal import bop2_dc_normal_design
from mdanderson_stats.bop2_dc_paired import bop2_dc_paired_design
from mdanderson_stats.bop2_dc_randomized_binary import bop2_dc_randomized_binary_design
from mdanderson_stats.bop2_dc_randomized_normal import bop2_dc_randomized_normal_design
from mdanderson_stats.bop2_dc_randomized_paired import bop2_dc_randomized_paired_design
from mdanderson_stats.bop2_dc_randomized_survival import bop2_dc_randomized_survival_design
from mdanderson_stats.bop2_dc_report import (
    BOP2DCCategoricalScenario,
    BOP2DCNormalScenario,
    BOP2DCRandomizedNormalScenario,
    BOP2DCRandomizedPairedScenario,
    BOP2DCRandomizedSurvivalScenario,
    BOP2DCSurvivalScenario,
    bop2_dc_binary_report,
    bop2_dc_categorical_report,
    bop2_dc_normal_report,
    bop2_dc_paired_report,
    bop2_dc_randomized_binary_report,
    bop2_dc_randomized_normal_report,
    bop2_dc_randomized_paired_report,
    bop2_dc_randomized_survival_report,
    bop2_dc_survival_report,
)
from mdanderson_stats.bop2_dc_survival import bop2_dc_survival_design


def test_binary_report_is_an_adapter_of_exact_oc_and_keeps_three_final_actions():
    design = bop2_dc_design(
        4,
        lrv=0.2,
        cmv=0.6,
        prior=(1, 1),
        looks=[2, 4],
        lambda_lrv=0.8,
        lambda_cmv=0.5,
        gamma_lrv=0.5,
        gamma_cmv=0.5,
    )
    report = bop2_dc_binary_report(design, [("truth <safe>", 0.4)])
    expected = design.operating_characteristics([0.4])
    case = report.cases[0]

    assert [action.name for action in case.actions] == [
        "stop_no_go",
        "final_go",
        "final_consider",
        "final_no_go",
    ]
    assert case.actions[0].probability == expected.stop_no_go[0].sum()
    assert [action.probability for action in case.actions[1:]] == [
        expected.final_go[0],
        expected.final_consider[0],
        expected.final_no_go[0],
    ]
    assert "truth &lt;safe&gt;" in report.to_html()
    assert "Native app report controls" in report.to_html()


def test_categorical_report_preserves_core_seeded_summary_and_denominators():
    design = bop2_dc_categorical_design(
        4,
        [[1, 0], [0, 1]],
        combination="any",
        directions=("greater", "less"),
        lrv=(0.1, 0.9),
        cmv=(0.3, 0.7),
        prior=(1, 1),
        looks=(2, 4),
    )
    truth = (0.45, 0.55)
    expected = simulate_bop2_dc_categorical(design, truth, n_trials=8, rng=728)
    report = bop2_dc_categorical_report(
        design, [BOP2DCCategoricalScenario("named truth", truth, seed=728, n_trials=8)]
    )
    case = report.cases[0]

    assert [action.name for action in case.actions] == list(expected.terminal_names)
    assert [action.count for action in case.actions] == expected.terminal_counts.tolist()
    assert [
        action.probability for action in case.actions
    ] == expected.terminal_probabilities.tolist()
    assert [action.mcse for action in case.actions] == expected.terminal_mcse.tolist()
    assert case.replay_seeds == tuple(map(int, expected.trial_seeds))
    assert case.look_summaries[0].reached == expected.look_reached_counts[0]
    html = report.to_html()
    assert "category 0: endpoint 1=yes, endpoint 2=no" in html
    assert "all simulated trials" in html
    assert str(int(expected.trial_seeds[0])) in html

    with pytest.raises(ValueError, match="explicit nonnegative integer seed"):
        bop2_dc_categorical_report(
            design,
            [
                BOP2DCCategoricalScenario(
                    "bad seed", truth, seed=np.random.default_rng(), n_trials=8
                )
            ],
        )


def test_all_remaining_report_adapters_construct_from_core_designs():
    normal = bop2_dc_normal_design(
        2,
        0.0,
        0.5,
        prior_mean=0.0,
        prior_precision=1.0,
        prior_shape=2.0,
        prior_scale=1.0,
        looks=[2],
    )
    survival = bop2_dc_survival_design(2, 1.0, 2.0, prior_shape=2.0, prior_scale=1.0, looks=[2])
    paired = bop2_dc_paired_design(2, "efficacy_toxicity", [0.2, 0.8], [0.4, 0.6], looks=[2])
    randomized_binary = bop2_dc_randomized_binary_design(
        2,
        0.0,
        0.2,
        control_prior=(1, 1),
        treatment_prior=(1, 1),
        arm_assignments=[0, 1],
        looks=[2],
    )
    randomized_normal = bop2_dc_randomized_normal_design(
        2,
        0.0,
        0.2,
        control_prior=(0.0, 1, 2, 1),
            treatment_prior=(0.0, 1, 2, 1),
            arm_assignments=[0, 1],
            looks=[2],
        )
    randomized_survival = bop2_dc_randomized_survival_design(
        2,
        -1.0,
        0.0,
        control_prior=(2, 1),
        treatment_prior=(2, 1),
        arm_assignments=[0, 1],
        looks=[2],
    )
    randomized_paired = bop2_dc_randomized_paired_design(
        2,
        "multiple_efficacy",
        [0.0, 0.0],
        [0.2, 0.2],
        control_prior=(1, 1, 1, 1),
        treatment_prior=(1, 1, 1, 1),
        arm_assignments=[0, 1],
        looks=[2],
    )
    categorical_randomized = bop2_dc_categorical_design(
        2,
        [[1, 0]],
        combination="any",
        directions="greater",
        lrv=0.0,
        cmv=0.2,
        prior=(1, 1),
        control_prior=(1, 1),
        arm_assignments=[0, 1],
        looks=[2],
    )

    reports = (
        bop2_dc_normal_report(normal, [BOP2DCNormalScenario("n", 0.2, 0.5, 1, 2)]),
        bop2_dc_survival_report(survival, [BOP2DCSurvivalScenario("s", 2.0, 1.0, 2.0, 1, 3)]),
        bop2_dc_paired_report(paired, [("efftox", [0.1, 0.2, 0.3, 0.4])]),
        bop2_dc_randomized_binary_report(randomized_binary, [("rb", 0.2, 0.4)]),
        bop2_dc_randomized_normal_report(
            randomized_normal, [BOP2DCRandomizedNormalScenario("rn", 0, 0.5, 0.2, 0.5, 4, 1)]
        ),
        bop2_dc_randomized_survival_report(
            randomized_survival,
            [BOP2DCRandomizedSurvivalScenario("rs", 2, 3, 1, 2, 5, 1)],
        ),
        bop2_dc_randomized_paired_report(
            randomized_paired,
            [
                BOP2DCRandomizedPairedScenario(
                    "rp", (0.2, 0.3, 0.1, 0.4), (0.3, 0.2, 0.2, 0.3), 6, 1
                )
            ],
        ),
        bop2_dc_categorical_report(
            categorical_randomized,
            [BOP2DCCategoricalScenario("rc", ((0.2, 0.8), (0.3, 0.7)), 7, 1)],
        ),
    )
    assert all(len(report.cases) == 1 for report in reports)
    assert (
        "efficacy and toxicity"
        in bop2_dc_paired_report(paired, [("efftox", [0.1, 0.2, 0.3, 0.4])]).cases[0].truth[0][0]
    )
