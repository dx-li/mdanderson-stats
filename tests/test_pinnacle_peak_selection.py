import csv

import numpy as np
import pytest

from mdanderson_stats.pinnacle_peak_selection import (
    PinnaclePeakSelection,
    quantify_selected_peaks,
    write_pinnacle_selection_csv,
)


def test_selection_add_remove_preserves_exact_coordinates_and_rejects_duplicates():
    selection = PinnaclePeakSelection.from_coordinates(
        [[2, 3], [1, 1]], np.array([5, 6]), region=(1, 4, 1, 5)
    )
    np.testing.assert_array_equal(selection.coordinates, [[2, 3], [1, 1]])
    assert not selection.coordinates.flags.writeable

    added = selection.add_peak((3, 4))
    np.testing.assert_array_equal(added.coordinates, [[2, 3], [1, 1], [3, 4]])
    removed = added.remove_peak((1, 1))
    np.testing.assert_array_equal(removed.coordinates, [[2, 3], [3, 4]])
    with pytest.raises(ValueError, match="duplicate"):
        selection.add_peak((2, 3))
    with pytest.raises(ValueError, match="not present"):
        selection.remove_peak((1, 2))


def test_manual_quantification_uses_supplied_order_and_existing_measurement_rules():
    first = np.full((5, 6), 2.0)
    second = np.full((5, 6), 4.0)
    first[2, 3], first[1, 1] = 10.0, 8.0
    second[2, 3], second[1, 1] = 14.0, 12.0
    selection = PinnaclePeakSelection.from_coordinates(
        [[2, 3], [1, 1]], (5, 6), region=(1, 4, 1, 5)
    )
    result = quantify_selected_peaks(
        iter((first, second)),
        selection,
        peak_radius=0,
        background="local_minimum",
        background_radius=1,
        normalization="none",
    )
    np.testing.assert_array_equal(result.quantification.coordinates, [[2, 3], [1, 1]])
    np.testing.assert_array_equal(result.quantification.raw, [[10, 8], [14, 12]])
    np.testing.assert_array_equal(result.quantification.background, [[2, 2], [4, 4]])
    np.testing.assert_array_equal(result.quantification.corrected, [[8, 6], [10, 8]])
    np.testing.assert_array_equal(result.quantification.normalized, [[8, 6], [10, 8]])
    with pytest.raises(ValueError, match="dimensions do not match"):
        quantify_selected_peaks((np.ones((6, 6)),), selection, normalization="none")


def test_csv_export_records_long_form_measurements_and_effective_settings(tmp_path):
    image = np.full((4, 5), 2.0)
    image[1, 2] = 9.0
    image[3, 4] = 7.0
    selection = PinnaclePeakSelection.from_coordinates([[3, 4], [1, 2]], image.shape)
    analysis = quantify_selected_peaks(
        (image,),
        selection,
        peak_radius=0,
        background="none",
        normalization="none",
    )
    destination = tmp_path / "quantification.csv"
    assert write_pinnacle_selection_csv(analysis, destination) == destination
    with destination.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert [(int(row["image_index"]), int(row["peak_index"])) for row in rows] == [(0, 0), (0, 1)]
    assert [(int(row["row"]), int(row["column"])) for row in rows] == [(3, 4), (1, 2)]
    assert [float(row["raw"]) for row in rows] == [7.0, 9.0]
    assert [row["normalization"] for row in rows] == ["none", "none"]
    assert [row["peak_radius"] for row in rows] == ["0", "0"]


def test_selection_validates_region_bounds_duplicates_and_integer_coordinates():
    with pytest.raises(ValueError, match="within the image region"):
        PinnaclePeakSelection.from_coordinates([[0, 1]], (3, 4), region=(1, 3, 0, 4))
    with pytest.raises(ValueError, match="duplicate"):
        PinnaclePeakSelection.from_coordinates([[1, 1], [1, 1]], (3, 4))
    with pytest.raises(ValueError, match="integer"):
        PinnaclePeakSelection.from_coordinates([[1.0, 1.0]], (3, 4))
