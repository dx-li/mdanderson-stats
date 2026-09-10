from decimal import Decimal, localcontext

import numpy as np
import pytest
from numpy.testing import assert_allclose

from mdanderson_stats import (
    asypow_smo_binomial,
    asypow_smo_multinomial,
    asypow_smo_ordinal,
)


def test_native_categorical_models_and_inversions():
    p = np.array([[0.2, 0.3], [0.4, 0.1]])
    multi = asypow_smo_multinomial(p, group_size=[1, 3])
    ordinal = asypow_smo_ordinal(p.cumsum(axis=1), group_size=[1, 3])
    for x in [multi, ordinal]:
        assert x.degrees_of_freedom == 2
        assert_allclose(x.divergence_per_observation, 0.06730956764893814, rtol=1e-13)
        assert_allclose(x.power(100), 0.48063004441969115, atol=1e-13)
        assert_allclose(x.sample_size(), 172.85341853111393, rtol=1e-12)
        assert_allclose(x.power(x.sample_size()), 0.8, atol=1e-13)
        assert not x.null_parameters.flags.writeable
    assert_allclose(multi.null_parameters, [[0.35, 0.15]] * 2, atol=1e-15)
    assert_allclose(ordinal.null_parameters, [[0.35, 0.5]] * 2, atol=1e-15)
    fixed = asypow_smo_multinomial(p, null_probabilities=[0.25, 0.25], group_size=[1, 3])
    # Original ordinal code incorrectly rejects this fully specified null.
    fixed_ord = asypow_smo_ordinal(p.cumsum(axis=1), null_cumulative=[0.25, 0.5], group_size=[1, 3])
    for x in [fixed, fixed_ord]:
        assert x.degrees_of_freedom == 4
        assert_allclose(x.divergence_per_observation, 0.14959244615398992, rtol=1e-13)
        assert_allclose(x.power(100, x.significance(100)), 0.8, atol=1e-13)


def test_binary_reduction_and_allocation_invariance():
    p = np.array([0.4, 0.3])
    for correction in [False, True]:
        binary = asypow_smo_binomial(p, group_size=[10, 9], subtract_df=correction)
        for model in [asypow_smo_multinomial, asypow_smo_ordinal]:
            x = model(p[:, None], group_size=[1e301, 9e300], subtract_df=correction)
            assert_allclose(x.power(100), binary.power(100), rtol=1e-12)
            assert_allclose(x.sample_size(), binary.sample_size(), rtol=1e-12)
            same = model([[0.2, 0.4]] * 3, group_size=[1, 2, 7])
            assert same.divergence_per_observation == 0


def test_close_and_rare_categories_against_decimal_likelihood():
    for p, q in [([0.2 + 1e-10, 0.3 - 1e-10], [0.2, 0.3]), ([1e-300, 0.3], [2e-300, 0.3])]:
        with localcontext() as ctx:
            ctx.prec = 340
            pd = [Decimal(x) for x in p]
            qd = [Decimal(x) for x in q]
            pd.append(1 - sum(pd))
            qd.append(1 - sum(qd))
            reference = float(2 * sum(a * (a / b).ln() for a, b in zip(pd, qd, strict=True)))
        x = asypow_smo_multinomial(p, null_probabilities=q)
        assert_allclose(x.divergence_per_observation, reference, rtol=1e-12, atol=0)
    with pytest.raises(ValueError, match="sum to less"):
        asypow_smo_multinomial([0.5, 0.5], null_probabilities=[0.2, 0.3])
    with pytest.raises(ValueError, match="increase strictly"):
        asypow_smo_ordinal([0.2, 0.2], null_cumulative=[0.3, 0.5])
