"""End-to-end reproduction of the SEQBIN manual's alternative-probability table."""

import numpy as np
from numpy.testing import assert_allclose

from mdanderson_stats import SeqBinStudySpecification


def test_manual_nearest_calibration_and_probability_table():
    # seqbin_doc.txt: N=50, Jeffreys prior, null .2, nearest level to .05.
    study = SeqBinStudySpecification(
        50,
        prior=[0.5, 0.5],
        significance=0.05,
        selection="nearest",
        legacy_bounds=True,
        probabilities=np.arange(5, 14) / 20,
    ).run()
    p = study.properties
    assert_allclose(
        p.rejection_probability[1:],
        [0.1612, 0.3758, 0.6400, 0.8483, 0.9552, 0.9911, 0.9989, 0.9999, 1.0],
        atol=5e-5,
        rtol=0,
    )
    assert_allclose(
        p.expected_subjects[1:],
        [45.13, 39.52, 31.98, 24.28, 17.98, 13.49, 10.47, 8.43, 7.0],
        atol=0.005,
        rtol=0,
    )
    assert_allclose(
        p.expected_subjects_quit_high[1:],
        [19.81, 22.11, 21.84, 19.69, 16.48, 13.16, 10.43, 8.43, 7.0],
        atol=0.005,
        rtol=0,
    )
    table = study.boundary_table(compact=True)
    assert_allclose(table.rows[-1, 8], 0.05060, atol=5e-6, rtol=0)
    assert_allclose(p.rejection_probability[0], table.rows[-1, 8], atol=1e-15)
    # The printed null summary (.0151) is inconsistent with its boundary table
    # due to the original compaction bug; do not reproduce that corrupted value.
    assert p.rejection_probability[0] > 0.05
    assert study.calibration.chosen.significance == p.rejection_probability[0]
