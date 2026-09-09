"""Pin SPPCR's archive and collect native fits without adopting undefined outputs."""

import hashlib
import json
import re
import subprocess
import tempfile
import zipfile
from pathlib import Path

ARCHIVE = Path("research/raw/SPPCR/SPPCR   _V1zip.zip")
SHA256 = "98e0937e26a5209421144c20178a2bd9c77d3baad7a3b28606d0f0f6555e2c98"
DRIVER = """program probe
use structures_mod
use one_data_set_mod, only: fit_one, modify_data
implicit none
integer :: i,j
read(*,*) n_dna,n_allele
call initialize
read(*,*) dna
read(*,*) n_well
read(*,*) progenitor
read(*,*) seen(:,:,data)
do j=1,n_allele
 unseen(:,j,data)=n_well-seen(:,j,data)
end do
call modify_data(seen(:,:,data),unseen(:,:,data),seen(:,:,shrink),unseen(:,:,shrink))
do j=1,n_allele
 write(*,'(A,I4,100ES25.16E3)') 'adjusted_seen ',j,seen(:,j,shrink)
end do
call fit_one(.false.,.false.,-1)
do j=1,n_allele
 write(*,'(A,I4,100ES25.16E3)') 'mu ',j,mu(j,data,init_est:sd)
 write(*,'(A,I4,100ES25.16E3)') 'frequency ',j,freq(j,data,untrans,est:sd)
end do
write(*,'(A,100ES25.16E3)') 'calibration ',calibrate(data,est:sd)
write(*,'(A,100ES25.16E3)') 'mutant ',mutant(data,untrans,est:sd)
end program
"""
CASES = {
    "single_level": ([1], [20], [[6, 10]], [1, 1]),
    "scaled_dna": ([0.25], [20], [[6, 10]], [1, 1]),
    "heterozygous": ([1], [20], [[6, 10, 2]], [1, 2]),
    "two_levels_exact_model": ([1, 2], [100, 100], [[20, 50], [36, 75]], [1, 1]),
    "partly_saturated": ([1, 2], [20, 20], [[10, 5], [20, 10]], [1, 1]),
    "negative_adjustment": ([1, 2], [20, 20], [[0, 5], [20, 10]], [1, 1]),
    "fully_saturated": ([1, 2], [20, 20], [[20, 5], [20, 10]], [1, 1]),
    "never_seen": ([1, 2], [20, 20], [[0, 5], [0, 10]], [1, 1]),
}


def main():
    if hashlib.sha256(ARCHIVE.read_bytes()).hexdigest() != SHA256:
        raise RuntimeError("SPPCR archive changed")
    with zipfile.ZipFile(ARCHIVE) as z:
        if z.testzip() is not None or len(set(z.namelist())) != len(z.namelist()):
            raise RuntimeError("Corrupt or duplicate archive member")
        contents = {i.filename: z.read(i) for i in z.infolist() if not i.is_dir()}
    root = "sppcr/source/"
    order = re.findall(r"-c (\w+\.f90)", contents[root + "compile.sppcr"].decode())
    if len(order) != 27 or set(order) != {Path(n).name for n in contents if n.endswith(".f90")}:
        raise RuntimeError("Source/build membership changed")
    flags = ["gfortran", "-std=legacy", "-O0", "-ffp-contract=off", "-fcheck=all"]
    cases = {}
    with tempfile.TemporaryDirectory(prefix="sppcr-reference-") as tmp:
        w = Path(tmp)
        for n in order:
            (w / n).write_bytes(contents[root + n])
        (w / "probe.f90").write_text(DRIVER)
        compiled = subprocess.run(
            [*flags, "-c", *order], cwd=w, check=True, capture_output=True, text=True, timeout=60
        )
        subprocess.run(
            [*flags, *[str(Path(n).with_suffix(".o")) for n in order], "-o", "sppcr"],
            cwd=w,
            check=True,
            capture_output=True,
            timeout=60,
        )
        subprocess.run(
            [
                *flags,
                "probe.f90",
                *[str(Path(n).with_suffix(".o")) for n in order if n != "sppcr.f90"],
                "-o",
                "probe",
            ],
            cwd=w,
            check=True,
            capture_output=True,
            timeout=60,
        )
        menu = subprocess.run(
            [str(w / "sppcr")], input="\n0\n", text=True, capture_output=True, timeout=3
        )
        for name, (dna, wells, seen, progenitor) in CASES.items():
            text = f"{len(dna)} {len(seen[0])}\n"
            for row in [
                dna,
                wells,
                progenitor,
                [seen[i][j] for j in range(len(seen[0])) for i in range(len(dna))],
            ]:
                text += " ".join(map(str, row)) + "\n"
            run = subprocess.run(
                [str(w / "probe")], input=text, cwd=w, text=True, capture_output=True, timeout=3
            )
            cases[name] = dict(
                dna=dna,
                wells=wells,
                seen=seen,
                progenitor=progenitor,
                stdin=text,
                stdout=run.stdout,
                stderr=run.stderr,
                returncode=run.returncode,
            )
    rows = []
    for name, data in sorted(contents.items()):
        path = Path(name)
        if path.suffix == ".f90":
            role = "source"
        elif path.name in {"compile.sppcr", "Makefile"}:
            role = "build"
        elif path.suffix in {".exe", ".hqx"}:
            role = "historical_binary"
        elif "/docs/" in name or "/EXAMPLE/" in name or name == "sppcr/readme":
            role = "foreign_sogs_documentation_or_example"
        else:
            role = "acquisition_build_or_legal_document"
        rows.append(
            dict(path=name, size=len(data), sha256=hashlib.sha256(data).hexdigest(), role=role)
        )
    notice = Path("notices/mdanderson-sppcr-Legal.doc.txt")
    if notice.exists() and notice.read_bytes() != contents["sppcr/Legal.doc"]:
        raise RuntimeError("Existing notice differs")
    notice.write_bytes(contents["sppcr/Legal.doc"])
    report = dict(
        archive_url="https://biostatistics.mdanderson.org/SoftwareDownload/SoftwareFiles/SPPCR/"
        "SPPCR%20%20%20_V1zip.zip",
        archive_sha256=SHA256,
        catalog_id=26,
        status="native_reference_evidence",
        compiler=subprocess.check_output(["gfortran", "--version"], text=True).splitlines()[0],
        flags=flags,
        source_order=order,
        compile_diagnostics=compiled.stderr,
        members=rows,
        driver=DRIVER,
        startup=dict(returncode=menu.returncode, stdout=menu.stdout, stderr=menu.stderr),
        cases=cases,
        limits="Successful compilation and exit are not correctness proofs. Foreign SOGS "
        "examples and invalid native fits must not be used as SPPCR correctness oracles.",
    )
    Path("tests/fixtures/sppcr.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        f"Inventoried {len(rows)} files; compiled 27 sources; "
        f"ran startup and {len(cases)} fit probes"
    )


if __name__ == "__main__":
    main()
