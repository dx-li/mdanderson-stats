"""Replay the pinned 25-page manual's worked examples through the application."""

import io
import math

import numpy as np
import pytest

from mdanderson_stats import run_stattab


def run(text):
    output, report = io.StringIO(), io.StringIO()
    result = run_stattab(io.StringIO(text), output, report_stream=report)
    assert result.reason == "exit"
    assert result.last_result is not None
    return result, output.getvalue(), report.getvalue()


def t9_cdf(t):
    # Independent angular integral of the t9 density; composite Simpson rule.
    theta = math.atan(t / 3)
    count = 2048
    step = theta / count
    total = math.fsum(
        (1 if i in (0, count) else 2 if i % 2 == 0 else 4) * math.cos(i * step) ** 8
        for i in range(count + 1)
    )
    return 0.5 + 128 / (35 * math.pi) * step / 3 * total


def test_manual_t_pvalue_help_and_report_pages_7_to_9():
    result, output, report = run("11\nhelp\n1.97 9 ? .\n\n0\n")
    expected = t9_cdf(1.97)
    values = result.last_result.parameters
    assert result.completed == 1 and result.rejected == 0
    assert values["cum"] == pytest.approx(expected, abs=2e-14)
    assert values["cum"] == pytest.approx(0.959829, abs=5.1e-7)
    assert result.last_result.values[-1] == pytest.approx(2 * (1 - expected), abs=4e-14)
    assert "Request: 1.97 9 ? ." in report
    assert "two_sided_p" in output and "two_sided_p" in report


def test_manual_inverse_t_table_pages_10_to_12():
    # Commentary mentions '=', although its displayed request repeats 9.
    result, _, _ = run("11\n1.97 9 ? .\n? = t .\n2\n0.1 0.9 8\n8\n\n0\n")
    values = result.last_result.parameters
    targets = np.linspace(0.1, 0.9, 9)
    np.testing.assert_allclose(values["cum"], targets, atol=2e-16)
    np.testing.assert_allclose([t9_cdf(t) for t in values["t"]], targets, atol=3e-14)
    printed = [
        -1.383029,
        -0.883404,
        -0.543480,
        -0.260955,
        0,
        0.260955,
        0.543480,
        0.883404,
        1.383029,
    ]
    np.testing.assert_allclose(values["t"], printed, rtol=0, atol=5.1e-7)
    assert result.completed == 2 and result.rejected == 0


def test_manual_poisson_inverse_and_integer_neighbors_pages_13_to_14():
    result, _, report = run("10\n? 10 t .\n2\n0.2 0.8 6\n8\n\n0\n")
    values = result.last_result.parameters
    printed = [6.798297, 7.724082, 8.543104, 9.331353, 10.141095, 11.030752, 12.101946]
    np.testing.assert_allclose(values["s"], printed, rtol=0, atol=2e-6)
    for neighbor in result.last_result.neighbors:
        assert np.all(neighbor.valid)
        for row in neighbor.values:
            c, cc, mean, count = row
            expected = math.exp(-10) * math.fsum(
                10**k / math.factorial(k) for k in range(int(count) + 1)
            )
            assert mean == 10
            assert c == pytest.approx(expected, abs=3e-14)
            assert cc == pytest.approx(1 - expected, abs=3e-14)
    assert report.count("floor_plus_one") == 7
    assert result.completed == 1 and result.rejected == 0


def test_manual_unattainable_binomial_target_page_15():
    result, output, _ = run("2\n0 5 0.5 . ? .\n? 5 0.5 . 0.01 .\n\n0\n")
    assert result.completed == 1 and result.rejected == 1
    assert "Invalid request:" in output
    # A failed request must not publish the source's misleading boundary answer.
    assert result.last_result.parameters["cum"] == pytest.approx(1 / 32)
    assert result.last_result.computed == ("cum", "ccum")


def test_manual_binomial_table_pages_16_to_17():
    result, _, _ = run("2\nt 10 0.5 . ? .\n2\n0 10 10\n8\n\n0\n")
    expected = np.array([math.comb(10, k) / 1024 for k in range(11)])
    np.testing.assert_allclose(result.last_result.values[:, -1], expected, rtol=3e-14)
    np.testing.assert_allclose(result.last_result.parameters["cum"], expected.cumsum(), atol=2e-15)
    assert result.completed == 1 and result.rejected == 0


def test_manual_normal_height_example_page_19():
    result, _, _ = run("9\n85 70 7.3 ? .\n\n0\n")
    upper = math.erfc((85 - 70) / (7.3 * math.sqrt(2))) / 2
    assert result.last_result.parameters["ccum"] == pytest.approx(upper, rel=2e-14)
    assert result.last_result.parameters["cum"] == pytest.approx(0.98, abs=0.005)
