"""Compare template interpretation with original Fortran readers, not Python parsing."""

import hashlib
import itertools
import json
import subprocess
from pathlib import Path

DRIVER = """program reference
use genetics_mod, only: read_genetic_parameters
use ascertain_mod, only: read_ascertain_parameters, set_ascertain=>set_module_variables
use statistics_mod, only: read_statistics_parameters, set_statistics=>set_module_variables
use param_io
use user_interface_mod, only: input_unit
implicit none
integer i
open(unit=20,file='input.txt',status='old')
input_unit=20
call read_genetic_parameters
call set_ascertain
call read_ascertain_parameters
call set_statistics
call read_statistics_parameters
write(*,*) 'BEGIN_REFERENCE'
do i=1,23
write(*,'(ES26.17E3)') get_parameter(i)
end do
end program reference
"""


def main():
    source = Path("research/raw/TDTASP/source/source/tdtasp_1.1/source").resolve()
    work = Path("research/raw/reference/tdtasp-template").resolve()
    work.mkdir(parents=True, exist_ok=True)
    names = [
        "print_it",
        "parser_mod",
        "user_interface_mod",
        "get_numbers_mod",
        "param_defn_mod",
        "param_io",
        "offspring_mod",
        "families_mod",
        "tail_poisson_mod",
        "ascertain_mod",
        "zero_finder",
        "cdf_aux_mod",
        "beta_gamma_mod",
        "cdf_beta_mod",
        "cdf_binomial_mod",
        "cdf_normal_mod",
        "power_bin1_mod",
        "bin1_ss_pow_mod",
        "normal_integrate",
        "power_bin1_vn_mod",
        "bin1_vn_ss_pow_mod",
        "genetics_mod",
        "statistics_mod",
    ]
    paths = [source / f"{name}.f90" for name in names]
    compile_paths = []
    adaptations = {}
    for path in paths:
        if path.stem in ("ascertain_mod", "statistics_mod"):
            stage = "ascertain" if path.stem == "ascertain_mod" else "statistics"
            anchor = f"      PUBLIC :: perform_{stage}_step"
            text = path.read_text()
            if text.count(anchor) != 1:
                raise RuntimeError("unexpected accessibility declaration")
            addition = f"\n      PUBLIC :: read_{stage}_parameters, set_module_variables"
            adapted = work / path.name
            adapted.write_text(text.replace(anchor, anchor + addition))
            compile_paths.append(adapted)
            adaptations[path.name] = addition.strip()
        else:
            compile_paths.append(path)
    driver = work / "driver.f90"
    driver.write_text(DRIVER)
    executable = work / "reference"
    command = [
        "gfortran",
        "-O0",
        "-ffp-contract=off",
        "-fcheck=all",
        *map(str, compile_paths),
        str(driver),
        "-o",
        str(executable),
    ]
    subprocess.run(command, cwd=work, check=True)
    manual = source.parent / "doc/tdtasp_1.1_doc.tex"
    excerpt = manual.read_text().split(r"\part{Filled in Data Form}", 1)[1]
    excerpt = excerpt.split(r"\begin{verbatim}", 1)[1].split(r"\end{verbatim}", 1)[0]
    base = "\n".join(line for line in excerpt.splitlines() if line.strip() and "#" not in line)
    cases = []
    cases.append(("manual_filled_form", base))
    for test, method, sampling, eligibility, calculation in itertools.product(
        ("tdt", "asp"),
        ("direct", "diseq"),
        ("family", "individual"),
        ("random", "one", "both"),
        ("s", "p"),
    ):
        replacements = {
            "tdt_or_asp": test,
            "how_enter_popn_freq": method,
            "family_or_individual": sampling,
            "parent_hetero_requirement": eligibility,
            "which_calc": calculation,
            "sample_size": "2E2",
            "mininum_n_affected": "2",
        }
        lines = []
        for line in base.splitlines():
            name = line.split("=", 1)[0].strip()
            lines.append(f"{name} = {replacements[name]}" if name in replacements else line)
        cases.append(
            ("_".join((test, method, sampling, eligibility, calculation)), "\n".join(lines))
        )
    output = []
    for name, text in cases:
        (work / "input.txt").write_text(text + "\n")
        completed = subprocess.run(
            [str(executable)], cwd=work, text=True, capture_output=True, check=True, timeout=30
        )
        if "BEGIN_REFERENCE" not in completed.stdout:
            raise RuntimeError(f"native parser failed for {name}: {completed.stdout}")
        values = [float(x) for x in completed.stdout.split("BEGIN_REFERENCE", 1)[1].split()]
        if len(values) != 23:
            raise RuntimeError("unexpected native parameter count")
        output.append({"name": name, "template": text + "\n", "parameters": values})
    archive = Path("research/raw/TDTASP/TDTASP  _V1.tar.gz")
    fixture = {
        "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        "manual_sha256": hashlib.sha256(manual.read_bytes()).hexdigest(),
        "compiler": subprocess.check_output(["gfortran", "--version"], text=True).splitlines()[0],
        "command": command,
        "driver": DRIVER,
        "adaptations": adaptations,
        "parameter_order": [
            "DD",
            "Dd",
            "dd",
            "theta",
            "AD",
            "Ad",
            "BD",
            "Bd",
            "delta",
            "p",
            "m",
            "mean_offspring",
            "sides",
            "alpha",
            "calculation",
            "power",
            "families",
            "test",
            "frequency_method",
            "sampling",
            "all_affected",
            "minimum_affected",
            "eligibility",
        ],
        "notes": (
            "Reader bodies unchanged; private readers/setters exposed. "
            "Inactive parameters may remain -1."
        ),
        "cases": output,
    }
    target = Path("tests/fixtures/tdtasp_template.json")
    target.write_text(json.dumps(fixture, indent=2) + "\n")
    print(f"Recorded {len(output)} original-reader cases in {target}")


if __name__ == "__main__":
    main()
