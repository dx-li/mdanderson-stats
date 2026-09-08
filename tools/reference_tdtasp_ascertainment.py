"""Validate TDTASP selection with original routines exposed to a test driver."""

import hashlib
import json
import subprocess
from pathlib import Path

DRIVER = """program reference
use ascertain_mod
use families_mod
use param_io
implicit none
real(8) p(4),pen(3),theta,weight(256),selection,answer,mean
integer i,t,s,e,k,a
read(*,*) p,pen,theta,mean,t,s,e,k,a
call put_parameter(c1_ix,p(1))
call put_parameter(c2_ix,p(2))
call put_parameter(c3_ix,p(3))
call put_parameter(c4_ix,p(4))
call put_parameter(alpha_ix,pen(1))
call put_parameter(beta_ix,pen(2))
call put_parameter(gamma_ix,pen(3))
call put_parameter(theta_ix,theta)
call put_parameter(test_ix,real(t,8))
call families
lambda=mean
is_tdt=t==1
family_ascertain=s==1
min_n_hetero=e
min_n_affected=k
all_affected_included=a==1
call calc_exp_n_given_k(k)
call calc_ascertain(weight,selection)
call ascertain_properties(weight,answer)
write(*,*) 'BEGIN_REFERENCE'
write(*,'(5ES26.17E3)') selection,answer,p_father_hetero,p_mother_hetero,overall_exp_n_g_k
do i=1,256
write(*,'(3ES26.17E3)') weight(i),fam_exp_n_g_k(i),fam_ss(i)
end do
end program reference
"""


def main():
    source = Path("research/raw/TDTASP/source/source/tdtasp_1.1/source").resolve()
    work = Path("research/raw/reference/tdtasp-ascertainment").resolve()
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
    ]
    paths = [source / f"{name}.f90" for name in names]
    adapted = work / "ascertain_mod.f90"
    original = paths[-1].read_text()
    anchor = "      PUBLIC :: perform_ascertain_step"
    if original.count(anchor) != 1:
        raise RuntimeError("unexpected source accessibility declaration")
    adapted.write_text(
        original.replace(
            anchor,
            anchor
            + """
      PUBLIC :: lambda,is_tdt,family_ascertain,min_n_hetero,min_n_affected
      PUBLIC :: all_affected_included,calc_exp_n_given_k,calc_ascertain
      PUBLIC :: ascertain_properties,p_father_hetero,p_mother_hetero""",
        )
    )
    driver = work / "driver.f90"
    driver.write_text(DRIVER)
    executable = work / "reference"
    command = [
        "gfortran",
        "-O0",
        "-ffp-contract=off",
        "-fcheck=all",
        *map(str, paths[:-1]),
        str(adapted),
        str(driver),
        "-o",
        str(executable),
    ]
    subprocess.run(command, cwd=work, check=True)
    cases = []
    for test, k, all_affected in [("tdt", 1, False), ("tdt", 2, True), ("asp", 2, False)]:
        for sampling in ["family", "individual"]:
            for eligibility in ["father", "one", "both"]:
                p, pen, theta, mean = [0.3, 0.2, 0.1, 0.4], [0.8, 0.5, 0.2], 0.1, 4.0
                values = [
                    *p,
                    *pen,
                    theta,
                    mean,
                    1 if test == "tdt" else 2,
                    1 if sampling == "family" else 2,
                    ["father", "one", "both"].index(eligibility),
                    k,
                    int(all_affected),
                ]
                output = subprocess.check_output(
                    [str(executable)],
                    cwd=work,
                    input=" ".join(map(str, values)) + "\n",
                    text=True,
                    timeout=30,
                )
                rows = [
                    [float(x) for x in line.split()]
                    for line in output.split("BEGIN_REFERENCE\n")[1].splitlines()
                ]
                if len(rows) != 257:
                    raise RuntimeError("unexpected native output")
                cases.append(
                    {
                        "haplotype_frequencies": p,
                        "penetrance": pen,
                        "recombination": theta,
                        "mean_offspring": mean,
                        "test": test,
                        "sampling": sampling,
                        "eligibility": eligibility,
                        "minimum_affected": k,
                        "all_affected": all_affected,
                        "summary": rows[0],
                        "families": rows[1:],
                    }
                )
    fixture = {
        "provenance": {
            "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
            "compiler": subprocess.check_output(["gfortran", "--version"], text=True).splitlines()[
                0
            ],
            "build_command": command,
            "driver": DRIVER,
            "adaptation": "Expose private state/routines using PUBLIC declarations only; "
            "all numerical routine bodies unchanged. Run one study per process.",
            "summary_columns": [
                "selection_probability",
                "answer_probability",
                "father_heterozygous",
                "mother_heterozygous",
                "population_average_truncated_mean",
            ],
            "family_columns": ["selected_probability", "truncated_mean_affected", "contributions"],
            "archive_url": "https://biostatistics.mdanderson.org/SoftwareDownload/SoftwareFiles/TDTASP/TDTASP%20%20_V1.tar.gz",
            "archive_sha256": hashlib.sha256(
                Path("research/raw/TDTASP/TDTASP  _V1.tar.gz").read_bytes()
            ).hexdigest(),
        },
        "cases": cases,
    }
    Path("tests/fixtures/tdtasp_ascertainment.json").write_text(
        json.dumps(fixture, indent=2) + "\n"
    )
    print(f"Wrote {len(cases)} native ascertainment studies")


if __name__ == "__main__":
    main()
