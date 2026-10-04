import numpy as np
import pytest
from numpy.testing import assert_allclose

from mdanderson_stats.asypow import AsymptoticPower
from mdanderson_stats.asypow_smo import SMOPower


def test_lr_vector_inversion_matches_independent_reference_and_forward_power():
    design = AsymptoticPower(0.72, 2, np.array([0.0]))
    targets = np.array([0.7, 0.8, 0.9])
    expected = np.array([10.6969090729914, 13.3815123166254, 17.5749111657766])
    sizes = design.sample_size(targets, 0.05)

    assert isinstance(design.sample_size(), float)
    assert isinstance(sizes, np.ndarray) and not sizes.flags.writeable
    assert_allclose(sizes, expected, rtol=1e-12)
    assert_allclose(design.power(sizes, 0.05), targets, atol=2e-13)
    assert_allclose(
        design.sample_size(0.8, [0.01, 0.05, 0.1]),
        [design.sample_size(0.8, alpha) for alpha in [0.01, 0.05, 0.1]],
    )
    assert design.sample_size(0.8, 0.05) == pytest.approx(design.sample_size([0.8], [0.05])[0])
    assert_allclose(design.sample_size([0.7, 0.8]), design.sample_size([0.7, 0.8], 0.05))


def test_smo_vector_inversion_preserves_df_correction_and_matches_reference():
    design = SMOPower(1.25, 4, np.array([0.0]), True)
    alphas = np.array([0.1, 0.05, 0.01])
    targets = np.full(3, 0.85)
    expected = np.array([8.84007848642612, 10.7379004125920, 14.7728825440243]) + 4 / 1.25

    corrected = design.sample_size(targets, alphas)
    uncorrected = SMOPower(1.25, 4, np.array([0.0]), False).sample_size(targets, alphas)
    lr = AsymptoticPower(1.25, 4, np.array([0.0])).sample_size(targets, alphas)

    assert_allclose(corrected, expected, rtol=1e-12)
    assert_allclose(uncorrected, lr, rtol=1e-13)
    assert_allclose(design.power(corrected, alphas), targets, atol=2e-13)


def test_vector_target_shapes_limits_and_unattainable_cases():
    design = AsymptoticPower(0.72, 2, np.array([0.0]))
    with pytest.raises(ValueError, match="equal lengths"):
        design.sample_size([0.8, 0.9], [0.05, 0.1, 0.2])
    with pytest.raises(ValueError, match="one-dimensional"):
        design.sample_size([[0.8, 0.9]], 0.05)
    with pytest.raises(ValueError, match="1..10000"):
        design.sample_size(np.full(10_001, 0.8), 0.05)
    with pytest.raises(ValueError, match="real"):
        design.sample_size(np.array([0.8 + 0j]), 0.05)
    with pytest.raises(ValueError, match="unattainable"):
        AsymptoticPower(0.0, 1, np.array([0.0])).sample_size([0.7, 0.8], 0.05)
    with pytest.raises(ValueError, match="SMO inversion"):
        SMOPower(1.25, 4, np.array([0.0]), True).sample_size([0.05, 0.8], 0.05)
    tiny = SMOPower(1e-320, 1, np.array([0.0]), False)
    with pytest.raises(ArithmeticError, match="not representable"):
        tiny.sample_size(0.8, 0.05)
