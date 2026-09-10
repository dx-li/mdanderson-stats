from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

from mdanderson_stats import (
    read_u2oet_doses,
    read_u2oet_scenario,
    read_u2oet_utility,
    u2oet_scenario,
)


def test_gaussian_rectangles_match_independent_r_and_near_singular_limits():
    reference = np.loadtxt(
        Path(__file__).parent / "fixtures/u2oet-gaussian-reference.csv", delimiter=",", skiprows=1
    )
    e = np.broadcast_to([0.1, 0.2, 0.3, 0.4], (2, 2, 4))
    t = np.broadcast_to([0.4, 0.3, 0.2, 0.1], (2, 2, 4))
    for rho in (0.1, 0.75, -0.6):
        scenario = u2oet_scenario(e, t, association=rho)
        expected = reference[reference[:, 0] == rho, -1].reshape(4, 4)
        assert_allclose(scenario.joint[0, 0], expected, atol=2e-12, rtol=0)
        assert_allclose(scenario.joint.sum(axis=-1), e, atol=2e-12)
        assert_allclose(scenario.joint.sum(axis=-2), t, atol=2e-12)
        assert not scenario.joint.flags.writeable
    half = np.full((2, 2, 2), 0.5)
    for rho in (0, 1, -1, 0.999999999999, -0.999999999999, 1e-310):
        scenario = u2oet_scenario(half, half, association=rho)
        diagonal = 0.25 + np.arcsin(rho) / (2 * np.pi)
        assert_allclose(
            scenario.joint[0, 0],
            [[diagonal, 0.5 - diagonal], [0.5 - diagonal, diagonal]],
            atol=2e-12,
            rtol=0,
        )
    sparse = u2oet_scenario(np.broadcast_to([0, 1, 0], (2, 2, 3)), half, association=0.8)
    assert_allclose(sparse.joint[:, :, 1], half, atol=2e-12)
    assert np.all(sparse.joint[:, :, [0, 2]] == 0)


def test_native_file_indices_missing_cells_and_utility_orientation(tmp_path):
    scenario = tmp_path / "scenario.txt"
    # Deliberately agent-2-major order, as in the vendor examples.
    lines = ["0.1"]
    for j in (1, 2):
        for i in (1, 2):
            lines.append(f"{i} {j} .25 .75 .6 .4")
    scenario.write_text("\n".join(lines))
    loaded = read_u2oet_scenario(scenario, dose_counts=(2, 2), efficacy_levels=2, toxicity_levels=2)
    assert loaded.association == 0.1 and loaded.joint.shape == (2, 2, 2, 2)
    assert_allclose(loaded.efficacy, np.broadcast_to([0.25, 0.75], (2, 2, 2)))
    scenario.write_text("\n".join(lines[1:]))
    independent = read_u2oet_scenario(
        scenario, dose_counts=(2, 2), efficacy_levels=2, toxicity_levels=2
    )
    assert_allclose(
        independent.joint, independent.efficacy[:, :, :, None] * independent.toxicity[:, :, None, :]
    )
    scenario.write_text("\n".join(lines[:-1]))
    with pytest.raises(ValueError, match="every dose"):
        read_u2oet_scenario(scenario, dose_counts=(2, 2), efficacy_levels=2, toxicity_levels=2)
    utility = tmp_path / "utility.txt"
    utility.write_text("0 1 0\n1 1 40\n0 0 10\n1 0 100\n")
    assert_allclose(
        read_u2oet_utility(utility, efficacy_levels=2, toxicity_levels=2), [[10, 0], [100, 40]]
    )
    doses = tmp_path / "doses.txt"
    doses.write_text("4 5 6\n40 60 80\n")
    a, b = read_u2oet_doses(doses)
    assert_allclose(a, [4, 5, 6])
    assert_allclose(b, [40, 60, 80])
