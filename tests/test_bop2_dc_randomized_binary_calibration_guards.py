import pytest

from mdanderson_stats.bop2_dc_randomized_binary_optimization import (
    optimize_bop2_dc_randomized_binary,
)


def test_later_grid_candidate_cutoff_underflow_is_rejected_before_tables() -> None:
    with pytest.raises(ArithmeticError, match="cutoff underflows"):
        optimize_bop2_dc_randomized_binary(
            4,
            0.0,
            0.2,
            (0.2, 0.2),
            (0.2, 0.6),
            control_prior=(1.0, 1.0),
            treatment_prior=(1.0, 1.0),
            arm_assignments=(0, 1, 0, 1),
            looks=(2, 4),
            lambda_lrv_grid=(0.5, 5e-324),
            lambda_cmv_grid=(0.5,),
            gamma_lrv_grid=(1.0,),
            gamma_cmv_grid=(0.5,),
        )
