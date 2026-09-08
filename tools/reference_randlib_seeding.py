"""Validate RANDLIB phrase hashes and resulting generator streams against native code."""

import hashlib
import json
import subprocess
from pathlib import Path


def main():
    root = Path("research/raw/RANDLIB/source").resolve()
    work = Path("research/raw/reference/randlib-seeding").resolve()
    work.mkdir(parents=True, exist_ok=True)
    cases, provenance = [], {}
    for language in ["f77", "f95", "c"]:
        directory = work / language
        directory.mkdir(exist_ok=True)
        exe = directory / "reference"
        if language == "c":
            folder = root / "source/randlib.c/src"
            selected = [folder / name for name in ["com.c", "randlib.c", "linpack.c"]]
            driver = directory / "driver.c"
            driver.write_text("""#include "randlib.h"
#include <stdio.h>
#include <string.h>
int main(void){
long a,b,x,y,g,anti,z;int i;char phrase[512];
if(scanf("%ld %ld\\n",&g,&anti)!=2)return 1;
if(!fgets(phrase,sizeof phrase,stdin))phrase[0]='\\0';
phrase[strcspn(phrase,"\\n")]='\\0';
phrtsd(phrase,&a,&b);printf("%ld %ld\\n",a,b);
setall(a,b);gscgn(1,&g);setant(anti);
for(i=0;i<10;i++){z=ignlgi();getsd(&x,&y);printf("%ld %ld %ld\\n",z,x,y);}
return 0;}
""")
            # Read the numeric line separately so whitespace in phrases is preserved.
            driver.write_text(
                driver.read_text()
                .replace('scanf("%ld %ld\\n",&g,&anti)', 'scanf("%ld %ld",&g,&anti)')
                .replace("if(!fgets(phrase", "getchar();\nif(!fgets(phrase")
            )
            flags = ["-O0", "-std=gnu89", "-ffp-contract=off"]
            compiler = "cc"
            command = [
                compiler,
                *flags,
                "-I",
                str(folder),
                *map(str, selected),
                str(driver),
                "-lm",
                "-o",
                str(exe),
            ]
            adaptation = "No source modification"
        else:
            prefix = "program reference\n"
            if language == "f77":
                folder = root / "source/randlib.f/src"
                selected = [
                    folder / f"{name}.f"
                    for name in [
                        "getcgn",
                        "getsd",
                        "ignlgi",
                        "initgn",
                        "inrgcm",
                        "mltmod",
                        "qrgnin",
                        "setall",
                        "setant",
                        "phrtsd",
                        "lennob",
                    ]
                ]
                declarations = "integer ignlgi\nexternal ignlgi\n"
                build_sources = selected
                adaptation = "No source modification"
            else:
                folder = root / "RANDLIB90/source/randlib90/source"
                selected = [folder / "ecuyer_cote_mod.f90", folder / "user_set_generator.f90"]
                exposed = directory / "user_set_generator.f90"
                exposed.write_text(
                    selected[1]
                    .read_text()
                    .replace("PUBLIC :: set_seeds", "PUBLIC :: set_seeds, phrase_to_seed")
                )
                build_sources = [selected[0], exposed]
                prefix += """use ecuyer_cote_mod, only: setall=>set_all_seeds, &
setcgn=>set_current_generator,setant=>set_antithetic,getsd=>get_seeds, &
ignlgi=>random_large_integer
use user_set_generator, only: phrtsd=>phrase_to_seed
"""
                declarations = ""
                adaptation = (
                    "Exposes private phrase_to_seed for measurement; routine bodies unchanged"
                )
            driver = directory / "driver.f90"
            driver.write_text(
                prefix
                + "implicit none\n"
                + declarations
                + """integer a,b,x,y,g,anti,z,i
character(512) phrase
read(*,*)g,anti
read(*,'(A)')phrase
call phrtsd(phrase,a,b)
write(*,'(2I16)')a,b
call setall(a,b)
call setcgn(g)
call setant(anti==1)
do i=1,10
 z=ignlgi()
 call getsd(x,y)
 write(*,'(3I16)')z,x,y
enddo
end program
"""
            )
            flags = ["-O0", "-std=legacy", "-ffixed-line-length-none"]
            compiler = "gfortran"
            command = [compiler, *flags, *map(str, build_sources), str(driver), "-o", str(exe)]
        subprocess.run(command, cwd=directory, check=True, capture_output=True)
        provenance[language] = dict(
            compiler=subprocess.check_output([compiler, "--version"], text=True).splitlines()[0],
            flags=flags,
            adaptation=adaptation,
            source_sha256={
                str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in selected
            },
        )
        phrases = [
            "",
            " ",
            "abc",
            "abc  ",
            "A sample phrase!",
            "000000.000",
            "235959.999",
            "123456.789",
            "a" * 80,
            "a" * 81,
            "".join(map(chr, range(32, 127))),
            "\tabc",
        ]
        c_table = (
            "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*()_+[];:'"
            + '\\"'
            + "<>?,./"
        )
        phrases.extend(["/", "\\", '"', ".,/"])
        for phrase in phrases:
            if language == "c" and any(
                character not in c_table for character in phrase.rstrip(" ")
            ):
                continue  # Original C lookup reads out of bounds for these characters.
            for stream in [1, 32]:
                for antithetic in [False, True]:
                    output = subprocess.check_output(
                        [str(exe)],
                        input=f"{stream} {int(antithetic)}\n{phrase}\n",
                        text=True,
                        timeout=10,
                    )
                    rows = [
                        list(map(int, line.split())) for line in output.splitlines() if line.strip()
                    ]
                    if len(rows) != 11:
                        raise ValueError("expected seeds and ten raw draws")
                    cases.append(
                        dict(
                            language=language,
                            phrase=phrase,
                            stream=stream,
                            antithetic=antithetic,
                            seed=rows[0],
                            values=[row[0] for row in rows[1:]],
                            states=[row[1:] for row in rows[1:]],
                        )
                    )
    Path("tests/fixtures/randlib_seeding.json").write_text(
        json.dumps(dict(provenance=provenance, cases=cases), indent=2) + "\n"
    )
    print(f"Generated {len(cases)} native phrase/stream cases")


if __name__ == "__main__":
    main()
