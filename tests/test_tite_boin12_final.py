import numpy as np
import pytest

from mdanderson_stats.boin12 import BOIN12Design
from mdanderson_stats.tite_boin12 import tite_boin12_select_obd


def test_final_selection_reduces_exactly_to_boin12_on_complete_data():
    design = BOIN12Design(0.35, 0.25, utilities=(100.0, 30.0, 65.0, 0.0))
    doses = np.repeat([1, 2, 3], 4)
    toxicity = np.asarray([0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0])
    efficacy = np.asarray([1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0])
    toxicity_followup = np.where(toxicity == 1, 0.5, 1.0)
    efficacy_followup = np.where(efficacy == 1, 0.75, 2.0)
    expected = design.select_obd(
        [4, 4, 4],
        np.bincount(doses[toxicity == 1], minlength=4)[1:],
        np.bincount(doses[efficacy == 1], minlength=4)[1:],
        efficacy_without_toxicity=np.bincount(
            doses[(toxicity == 0) & (efficacy == 1)], minlength=4
        )[1:],
    )
    result = tite_boin12_select_obd(
        design,
        doses,
        toxicity,
        efficacy,
        toxicity_followup,
        efficacy_followup,
        toxicity_window=1.0,
        efficacy_window=2.0,
        n_doses=3,
    )
    assert result.obd == expected.obd
    assert result.mtd == expected.mtd
    np.testing.assert_array_equal(result.admissible, expected.admissible)
    np.testing.assert_allclose(result.isotonic_toxicity, expected.isotonic_toxicity)
    np.testing.assert_allclose(
        result.posterior.utility_probability, expected.posterior.utility_probability
    )
    later_look = tite_boin12_select_obd(
        design,
        doses,
        toxicity,
        efficacy,
        np.full(doses.size, 10.0),
        np.full(doses.size, 20.0),
        toxicity_window=1.0,
        efficacy_window=2.0,
        n_doses=3,
    )
    assert later_look.obd == result.obd
    assert later_look.mtd == result.mtd


def test_final_selection_rejects_pending_or_inconsistent_histories():
    design = BOIN12Design(0.35, 0.25)
    common = dict(
        toxicity_window=1.0,
        efficacy_window=2.0,
        n_doses=1,
    )
    with pytest.raises(ValueError, match="fully observed"):
        tite_boin12_select_obd(design, [1], [-1], [1], [0.5], [0.5], **common)
    with pytest.raises(ValueError, match="completed non-event"):
        tite_boin12_select_obd(design, [1], [0], [0], [0.5], [2.0], **common)
    with pytest.raises(ValueError, match="at least one patient"):
        tite_boin12_select_obd(design, [], [], [], [], [], **common)
