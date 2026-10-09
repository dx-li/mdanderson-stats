import json
import subprocess
import sys
from pathlib import Path

import pytest

from mdanderson_stats.flecs90 import FLECS90SyntaxError, translate_flecs90, translate_flecs90_file

_FIXTURES = json.loads((Path(__file__).parent / "fixtures/flecs90-reference.json").read_text())


@pytest.mark.parametrize(
    "case",
    _FIXTURES,
    ids=lambda c: c["name"] + "-" + "".join(str(int(v)) for v in c["options"].values()),
)
def test_native_source_translations(case):
    result = translate_flecs90(case["source"], **case["options"])
    assert result.fortran_source == case["fortran_source"]
    assert len(result.source_line_numbers) == len(result.fortran_source.splitlines())
    assert all(1 <= line <= len(case["source"].splitlines()) for line in result.source_line_numbers)


@pytest.mark.parametrize(
    "body, message",
    [
        ("FIN", "FIN has no matching"),
        ("EXIT", "nothing to EXIT"),
        ("CYCLE", "nothing to CYCLE"),
        ("REVERT", "nothing to REVERT"),
        ("ELSE", "ELSE has no matching"),
        ("ENDIF", "ENDIF has no matching"),
        ("IF X", "missing parenthesized"),
        ("IF (X", "unbalanced"),
        ("TO NOHYPHEN", "hyphenated"),
        ("(X) A=1", "INSTANCE has no matching"),
        ("WHEN (X) A=1\n      A=2", "WHEN requires ELSE"),
        ("CONDITIONAL\n      (OTHERWISE) A=1\n      (X) A=2", "expected INSTANCE"),
        ("IF (X)", "unfinished before END"),
    ],
)
def test_invalid_structures_have_line_diagnostics(body, message):
    source = "      PROGRAM DEMO\n      " + body + "\n      END\n"
    with pytest.raises(FLECS90SyntaxError, match=message) as info:
        translate_flecs90(source)
    assert info.value.line_number >= 2


def test_quotes_keep_whitespace_parentheses_and_doubled_quote_escapes():
    source = """      PROGRAM DEMO
      IF (INDEX(')  it''s (', ')').GT.0) PRINT *, 'two  spaces'
      END
"""
    output = translate_flecs90(source).fortran_source
    assert "INDEX(')  it''s (', ')')" in output
    assert "'two  spaces'" in output


def test_finished_loop_exit_targets_that_loop_not_its_parent():
    source = (
        "      PROGRAM DEMO\n      LOOP\n      DO (I=1,3) EXIT\n      EXIT\n      FIN\n      END\n"
    )
    output = translate_flecs90(source).fortran_source
    assert "      EXIT FLECS_LOOP_2\n      END DO FLECS_LOOP_2" in output
    assert "      EXIT FLECS_LOOP_1\n      END DO FLECS_LOOP_1" in output


def test_native_fortran_if_else_and_flecs_nested_structure_order():
    output = translate_flecs90(
        "      IF (X) THEN\n      A=1\n      ELSE\n      A=2\n      ENDIF\n      END\n"
    ).fortran_source
    assert "      ELSE\n" in output and "      END IF\n" in output
    with pytest.raises(FLECS90SyntaxError, match="FIN has no matching"):
        translate_flecs90("      LOOP\n      IF (X) THEN\n      FIN\n      END\n")


def test_empty_conditional_does_not_emit_an_unmatched_end_if():
    output = translate_flecs90("      CONDITIONAL\n      FIN\n      END\n").fortran_source
    assert output == "      END\n"


def test_numbered_long_generated_lines_wrap_and_map_to_source():
    condition = "X.GT.0 .AND. " * 18 + "X.LT.2"
    chunks = [condition[i : i + 55] for i in range(0, len(condition), 55)]
    source = (
        "      IF ("
        + chunks[0]
        + "\n"
        + "".join("     +" + part + "\n" for part in chunks[1:])
        + "     +) A=1\n      END\n"
    )
    result = translate_flecs90(source, line_numbers=True)
    lines = result.fortran_source.splitlines()
    assert len(lines) > 2
    assert all(len(line) == 78 for line in lines)
    assert lines[1].startswith("     +")
    assert all(line[73:] == "00001" for line in lines[:-1])
    assert result.source_line_numbers[:-1] == (1,) * (len(lines) - 1)


def test_columns_after_72_are_sequence_fields_and_do_not_change_translation():
    source = "      IF (X) A=1".ljust(72) + "12345678\n      END\n"
    assert translate_flecs90(source).fortran_source == "      IF (X) A=1\n      END\n"


def test_input_limits_and_option_types():
    for source in ["x" * (4 * 1024 * 1024 + 1), "\x00", "é"]:
        with pytest.raises(ValueError):
            translate_flecs90(source)
    with pytest.raises(ValueError, match="boolean"):
        translate_flecs90("      END", line_numbers=1)
    with pytest.raises(ValueError, match="99999"):
        translate_flecs90("\n" * 100000)
    with pytest.raises(FLECS90SyntaxError, match="nesting"):
        translate_flecs90("      LOOP\n" * 513 + "      END\n")
    with pytest.raises(FLECS90SyntaxError, match="missing END"):
        translate_flecs90("      A=1\n")
    with pytest.raises(FLECS90SyntaxError, match="continuation"):
        translate_flecs90("     +A=1\n      END\n")


def test_file_workflow_preserves_input_and_existing_outputs(tmp_path):
    source = tmp_path / "demo.flx"
    original = "      LOOP\n      EXIT\n      FIN\n      END\n"
    source.write_text(original)
    result = translate_flecs90_file(source)
    output = source.with_suffix(".f")
    assert output.read_text() == result.fortran_source
    assert source.read_text() == original
    with pytest.raises(FileExistsError):
        translate_flecs90_file(source)
    with pytest.raises(ValueError, match="differ"):
        translate_flecs90_file(source, source, overwrite=True)
    output.write_text("existing")
    source.write_text("      FIN\n      END\n")
    with pytest.raises(FLECS90SyntaxError):
        translate_flecs90_file(source, overwrite=True)
    assert output.read_text() == "existing"
    assert {p.name for p in tmp_path.iterdir()} == {"demo.flx", "demo.f"}


def test_cli_check_only_translation_and_nonzero_errors(tmp_path):
    path = tmp_path / "demo.flx"
    path.write_text("      LOOP\n      EXIT\n      FIN\n      END\n")
    command = [sys.executable, "-m", "mdanderson_stats.flecs90_cli"]
    checked = subprocess.run(command + ["-f", str(path)], capture_output=True, text=True)
    assert checked.returncode == 0 and not path.with_suffix(".f").exists()
    translated = subprocess.run(
        command + ["-c", "-n", "-s", str(path)], capture_output=True, text=True
    )
    assert translated.returncode == 0
    assert path.with_suffix(".f").read_text().splitlines()[0].endswith("00001")
    path.write_text("      FIN\n      END\n")
    failed = subprocess.run(command + ["-f", str(path)], capture_output=True, text=True)
    assert failed.returncode == 1 and "FIN has no matching" in failed.stderr
