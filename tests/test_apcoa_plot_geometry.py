import csv
from pathlib import Path

import numpy as np
import pytest

from mdanderson_stats.apcoa import adjusted_pcoa
from mdanderson_stats.apcoa_plot import _profile_medoid, adjusted_pcoa_plot_geometry

_FIXTURES = Path(__file__).parent / "fixtures"


def _csv(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _reference_fit():
    rows = _csv(_FIXTURES / "apcoa-overlay-input.csv")
    features = np.array([[float(row[f"X{i}"]) for i in range(1, 4)] for row in rows])
    groups = np.array([row["group"] for row in rows])
    covariate = np.array([float(row["covariate"]) for row in rows])[:, None]
    distance = np.sqrt(((features[:, None, :] - features[None, :, :]) ** 2).sum(axis=2))
    return adjusted_pcoa(distance, covariate, components=2), groups


def test_native_overlay_reference_coordinates_medoids_and_ellipse_vertices():
    fit, groups = _reference_fit()
    expected_coordinates = _csv(_FIXTURES / "apcoa-overlay-coordinates.csv")
    for row in expected_coordinates:
        coordinates = (
            fit.original.coordinates if row["panel"] == "original" else fit.adjusted.coordinates
        )
        index = int(row["index"]) - 1
        np.testing.assert_allclose(
            coordinates[index, :2], [float(row["x"]), float(row["y"])], atol=2e-12
        )

    geometry = adjusted_pcoa_plot_geometry(
        fit, groups, show_ellipses=True, show_medoid_connectors=True
    )
    expected_summary = _csv(_FIXTURES / "apcoa-overlay-summary.csv")
    expected_vertices = _csv(_FIXTURES / "apcoa-overlay-vertices.csv")
    by_key = {(row["panel"], row["group"]): row for row in expected_summary}
    vertices_by_key: dict[tuple[str, str], list[list[float]]] = {}
    for row in expected_vertices:
        vertices_by_key.setdefault((row["panel"], row["group"]), []).append(
            [float(row["x"]), float(row["y"])]
        )
    for group in geometry.groups:
        for panel, ellipse, medoid in (
            ("original", group.original_ellipse, group.original_medoid_index),
            ("adjusted", group.adjusted_ellipse, group.adjusted_medoid_index),
        ):
            summary = by_key[(panel, group.label)]
            assert medoid == int(summary["medoid"]) - 1
            assert ellipse is not None
            assert not ellipse.flags.writeable
            np.testing.assert_allclose(
                ellipse,
                np.asarray(vertices_by_key[(panel, group.label)]),
                rtol=2e-12,
                atol=2e-12,
            )


def test_pam_k_one_build_tie_cases_match_reference():
    for row in _csv(_FIXTURES / "apcoa-medoid-ties.csv"):
        size = int(row["n"])
        if row["kind"] == "constant":
            profile = np.zeros((size, size))
        else:
            profile = np.abs(np.arange(size)[:, None] - np.arange(size)[None, :])
        assert _profile_medoid(profile) == int(row["medoid"]) - 1


def test_ellipse_rank_one_collapses_and_rank_zero_fails():
    from mdanderson_stats.apcoa_plot import _ellipse

    line = np.column_stack((np.arange(3.0), 2 * np.arange(3.0)))
    polygon, rank = _ellipse(line, "line")
    assert rank == 1
    assert polygon.shape == (52, 2)
    np.testing.assert_allclose(polygon[:, 1], 2 * polygon[:, 0], atol=1e-14)
    with pytest.raises(ValueError, match="zero covariance"):
        _ellipse(np.ones((3, 2)), "constant")


def test_ellipse_stays_finite_for_tight_clusters_and_extreme_units():
    from mdanderson_stats.apcoa_plot import _ellipse

    tight = np.array(
        [
            [1e12, 1e12],
            [1e12 + 0.001, 1e12],
            [1e12, 1e12 + 0.002],
            [1e12 + 0.002, 1e12 + 0.002],
        ]
    )
    tight_polygon, rank = _ellipse(tight, "tight")
    assert rank == 2
    assert np.isfinite(tight_polygon).all()
    assert np.ptp(tight_polygon[:, 0]) > 0
    baseline = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    reference, _ = _ellipse(baseline, "unit")
    for scale in (1e-150, 1e150):
        scaled, _ = _ellipse(baseline * scale, "scaled")
        np.testing.assert_allclose(scaled / scale, reference, rtol=3e-14, atol=3e-14)


def test_original_distance_recovery_respects_tiny_gram_scale():
    from mdanderson_stats.apcoa_plot import _original_distance_profile

    points = np.array([[0.0, 0.0], [1e-10, 0.0], [0.0, 2e-10]])
    centered = points - points.mean(axis=0)
    gram = centered @ centered.T
    distances, clipped = _original_distance_profile(gram, np.arange(len(points)))
    expected = np.sqrt(((points[:, None, :] - points[None, :, :]) ** 2).sum(axis=2))
    np.testing.assert_allclose(distances, expected, rtol=2e-15, atol=1e-25)
    assert clipped == 0
    with pytest.raises(ArithmeticError, match="could not be recovered"):
        _original_distance_profile(1e-20 * np.array([[1.0, 2.0], [2.0, 1.0]]), np.arange(2))
