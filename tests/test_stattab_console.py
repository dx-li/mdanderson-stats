"""Complete console flows, source list actions, reporting and bounded failures."""

import io

import numpy as np
import pytest
from test_stattab_reference import CASES
from test_stattab_results import INPUTS

from mdanderson_stats import (
    STATTAB_DISTRIBUTIONS,
    CDFConsoleError,
    format_stattab_result,
    run_stattab,
    stattab_help,
    stattab_solve,
)


def run(text, **kwargs):
    source, output, report = io.StringIO(text), io.StringIO(), io.StringIO()
    result = run_stattab(source, output, report_stream=report, **kwargs)
    assert not source.closed and not output.closed and not report.closed
    return result, output.getvalue(), report.getvalue()


@pytest.mark.parametrize("name", INPUTS)
def test_each_distribution_from_menu_to_forward_result_and_exit(name):
    menu = STATTAB_DISTRIBUTIONS[name].menu
    line = CASES[name + "_which_1"]["input"].splitlines()[3]
    outcome, output, report = run(f"{menu}\n{line}\n\n0\n")
    assert outcome.reason == "exit" and outcome.completed == 1 and outcome.rejected == 0
    assert outcome.selected_distribution is None
    np.testing.assert_array_equal(
        outcome.last_result.values, stattab_solve(name, **INPUTS[name]).values
    )
    assert format_stattab_result(outcome.last_result) in output
    assert format_stattab_result(outcome.last_result) in report
    assert "Request: " + line in report
    assert "Input order: " + " ".join(STATTAB_DISTRIBUTIONS[name].parameters) in output


@pytest.mark.parametrize("name", INPUTS)
def test_help_for_every_distribution_names_all_supported_groups(name):
    help_text = stattab_help(name)
    for group in STATTAB_DISTRIBUTIONS[name].groups:
        assert "/".join(group) in help_text
    assert "cum is the lower tail" in help_text and "HELP" in help_text
    assert "Input order:" in help_text


def test_help_explains_gamma_zero_boundaries_and_f_df_limit():
    assert "Rate is not scale" in stattab_help("gamma")
    assert "unit mass at zero failures" in stattab_help("neg_binomial")
    assert "zero at positive counts" in stattab_help("poisson")
    assert "does not invert F degrees" in stattab_help("f")
    assert "multiple roots" in stattab_help("nc_t")
    assert "0 Exit" in stattab_help()
    with pytest.raises(ValueError):
        stattab_help("bad")


def test_reuse_help_error_recovery_and_distribution_change():
    text = "10\n2 3 ? .\nHELP\n? = = .\n0 -1 ? .\n? = = .\n\n9\n= = = ? .\n1 0 1 ? .\n\n0\n"
    outcome, output, report = run(text)
    assert outcome.completed == 4 and outcome.rejected == 2
    assert outcome.last_result.distribution == "normal"
    assert output.count("Invalid request:") == 2
    assert "no previous value" in output
    assert "Poisson" in output and "floor_plus_one" in report
    assert "1 0 1 ? ." in report


def test_table_then_scalar_reuses_last_row_without_stale_table_mode():
    text = "9\nT 0 1 ? .\n1\n3\n0 1 2\n8\n3 = = ? .\n\n0\n"
    outcome, output, report = run(text)
    assert outcome.completed == 2 and outcome.rejected == 0
    assert outcome.last_result.parameters["x"] == 3
    assert output.count("List for x") == 2
    assert "normal: computed cum/ccum" in report


def test_all_eight_list_actions_and_pagination_in_actual_application():
    # Add 3,1,2,2; linear 4,5; logarithmic 1,10; sort unique gives 1,2,3,4,5,10.
    # Print (one page), delete position 2, delete positions 2..3; retain 1,5,10.
    editor = "1\n4\n3 1 2 2\n2\n4 5 1\n3\n1 10 1\n7\n4\n\n5\n2\n6\n2 3\n8\n"
    outcome, output, _ = run("9\nT 0 1 ? .\n" + editor + "\n0\n")
    assert outcome.completed == 1
    np.testing.assert_array_equal(outcome.last_result.parameters["x"], [1, 5, 10])
    assert "Press the Return or Enter key" in output
    assert "1.000000e+01" in output


def test_rejected_list_action_preserves_successful_edits():
    editor = "1\n2\n0 1\n2\n2 3 1\n8\n"
    outcome, output, _ = run("9\nT 0 1 ? .\n" + editor + "\n0\n", max_table_size=3)
    np.testing.assert_array_equal(outcome.last_result.parameters["x"], [0, 1])
    assert "List capacity exceeded" in output


def test_empty_table_is_a_successful_empty_report():
    outcome, output, _ = run("9\nT 0 1 ? .\n8\n\n0\n")
    assert outcome.completed == 1 and outcome.last_result.values.shape == (0, 6)
    assert "(empty table)" in output


@pytest.mark.parametrize(
    "text,completed,selected",
    [
        ("", 0, None),
        ("9\n", 0, "normal"),
        ("10\n2 3 ? .\n", 1, "poisson"),
    ],
)
def test_eof_is_a_clean_outcome(text, completed, selected):
    outcome, output, report = run(text)
    assert outcome.reason == "eof" and outcome.completed == completed
    assert outcome.selected_distribution == selected and outcome.pending_table is None
    assert output.endswith("End of input.\n") and report.endswith("End of input.\n")


def test_eof_during_list_editing_returns_immutable_partial_list():
    outcome, _, _ = run("9\n1 0 1 ? .\nT = = ? .\n1\n2\n4 5\n")
    assert outcome.reason == "eof" and outcome.completed == 1
    assert outcome.last_result.parameters["x"] == 1
    np.testing.assert_array_equal(outcome.pending_table, [4, 5])
    with pytest.raises(ValueError):
        outcome.pending_table.setflags(write=True)


def test_report_alias_does_not_duplicate_results_and_stream_errors_propagate():
    stream = io.StringIO()
    run_stattab(io.StringIO("9\n1 0 1 ? .\n\n0\n"), stream, report_stream=stream)
    assert stream.getvalue().count("normal: computed cum/ccum") == 1

    class Broken(io.StringIO):
        def write(self, text):
            raise OSError("disk full")

    with pytest.raises(OSError, match="disk full"):
        run_stattab(io.StringIO("0\n"), io.StringIO(), report_stream=Broken())


@pytest.mark.parametrize(
    "kwargs,text",
    [
        (dict(max_steps=1), "9\n"),
        (dict(max_list_actions=1), "9\nT 0 1 ? .\n1\n1\n0\n"),
        (dict(max_line_length=4), "9\n1 0 1 ? .\n"),
    ],
)
def test_console_limits_fail_explicitly(kwargs, text):
    with pytest.raises(CDFConsoleError):
        run(text, **kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(max_steps=0),
        dict(max_table_size=0),
        dict(precision=18),
        dict(precision=True),
        dict(page_size=-1),
        dict(max_output=0),
        dict(max_list_actions=0),
    ],
)
def test_invalid_console_configuration(kwargs):
    with pytest.raises(ValueError):
        run("0\n", **kwargs)


def test_formatter_preserves_small_probabilities_and_labels_neighbors():
    value = stattab_solve("normal", x=9, mean=0, sd=1)
    rendered = format_stattab_result(value, precision=17)
    numbers = rendered.splitlines()[2].split()[2:]
    np.testing.assert_array_equal([float(x) for x in numbers], value.values)
    assert float(numbers[-1]) > 0
    inverse = stattab_solve("binomial", compute="s", n=0, pr=0.5, cum=1)
    rendered = format_stattab_result(inverse)
    assert "floor_plus_one: unavailable" in rendered
    assert "floor " in rendered and "nan" not in rendered.lower()


def test_formatter_flattens_batches_and_keeps_neighbor_indices():
    forward = stattab_solve("binomial", s=[[2.4], [1.4]], n=[2.5, 4], pr=0.5)
    result = stattab_solve(
        "binomial", compute="s", n=[2.5, 4], pr=0.5, cum=forward.parameters["cum"]
    )
    lines = format_stattab_result(result).splitlines()[2:]
    assert len(lines) == 12
    for i in range(4):
        assert all(line.split()[0] == str(i + 1) for line in lines[i * 3 : (i + 1) * 3])
    assert "1 floor_plus_one: unavailable" in lines[2]
    assert "unavailable" not in lines[-1]


def test_formatter_resource_limits_apply_before_returning_output():
    value = stattab_solve("poisson", s=[0, 1], mean=3)
    for kwargs in [dict(max_rows=1), dict(max_output=5), dict(precision=0), dict(precision=18)]:
        with pytest.raises(ValueError):
            format_stattab_result(value, **kwargs)
    text = format_stattab_result(value)
    assert format_stattab_result(value, max_output=len(text)) == text
    with pytest.raises(ValueError):
        format_stattab_result(value, max_output=len(text) - 1)
    inverse = stattab_solve("poisson", compute="s", mean=3, cum=0.5)
    with pytest.raises(ValueError):
        format_stattab_result(inverse, max_rows=2)
    with pytest.raises(ValueError):
        run("9\n1 0 1 ? .\n", max_output=5)


def test_module_entry_point_runs_a_complete_session():
    import subprocess
    import sys

    process = subprocess.run(
        [sys.executable, "-m", "mdanderson_stats.stattab"],
        input="n\n9\n1 0 1 ? .\n\n0\n",
        text=True,
        capture_output=True,
        timeout=10,
    )
    assert process.returncode == 0 and not process.stderr
    assert "normal: computed cum/ccum" in process.stdout
    process = subprocess.run(
        [sys.executable, "-m", "mdanderson_stats.stattab"],
        input="n\n99\n99\n99\n",
        text=True,
        capture_output=True,
        timeout=10,
    )
    assert process.returncode != 0 and "Traceback" not in process.stderr
