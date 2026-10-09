"""Reproduce FLECS90 translation fixtures using the pinned public-domain C source.

Run from the repository root. Requires GCC; no native code is bundled at runtime.
--check verifies existing fixtures without replacing them. Input programs are
read from the committed fixtures, but expected translations come only from C.
"""

import argparse
import hashlib
import itertools
import json
import subprocess
import tarfile
from pathlib import Path
from urllib.request import urlopen

_URL = (
    "https://biostatistics.mdanderson.org/SoftwareDownload/SoftwareFiles/FLECS90/FLECS90_V1.tar.gz"
)
_SHA256 = "82eb73ef0b6e8e27c850fae8a9d1fc5b7b8e1a4505a729e0476267e9efc39c29"
_RAW = Path("research/raw/flecs90")
_FIXTURE = Path("tests/fixtures/flecs90-reference.json")
_FLAGS = ["-std=c99", "-fcommon", "-O0"]


def compile_reference() -> Path:
    _RAW.mkdir(parents=True, exist_ok=True)
    archive = _RAW / "FLECS90_V1.tar.gz"
    if not archive.exists():
        with urlopen(_URL, timeout=60) as response:
            body = response.read(1024 * 1024 + 1)
        if len(body) > 1024 * 1024 or hashlib.sha256(body).hexdigest() != _SHA256:
            raise ValueError("FLECS90 archive integrity check failed")
        archive.write_bytes(body)
    if hashlib.sha256(archive.read_bytes()).hexdigest() != _SHA256:
        raise ValueError("FLECS90 cached archive integrity check failed")
    build = _RAW / "reference-build"
    build.mkdir(exist_ok=True)
    with tarfile.open(archive, "r:gz") as bundle:
        for name in ("flecs90.h", "getopt.c", "flecs90.c", "util.c", "work.c"):
            member = bundle.getmember("source/" + name)
            if not member.isfile() or member.size > 128 * 1024:
                raise ValueError("unexpected source archive member")
            handle = bundle.extractfile(member)
            assert handle is not None
            original = handle.read().decode("ascii")
            # Modern stdio declares getline with a different signature; modern
            # GCC also defaults to -fno-common for the legacy header's globals.
            text = original.replace("getline(", "flecs_getline(")
            if name == "work.c":
                # Remove undefined pointer/free and branch-flag initialization.
                text = text.replace("char *flecs_in;", "char *flecs_in=NULL;")
                text = text.replace("int qotherwise;", "int qotherwise=0;")
            (build / name).write_text(text)
    executable = build / "flecs90"
    subprocess.run(
        [
            "gcc",
            *_FLAGS,
            "-o",
            str(executable),
            *(str(build / name) for name in ("getopt.c", "flecs90.c", "util.c", "work.c")),
        ],
        check=True,
    )
    return executable.resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    existing = json.loads(_FIXTURE.read_text())
    sources = {case["name"]: case["source"] for case in existing}
    executable = compile_reference()
    cases = _RAW / "cases"
    cases.mkdir(exist_ok=True)
    collected = []
    for index, (name, source) in enumerate(sources.items()):
        stem = f"reference-{index}"
        (cases / (stem + ".flx")).write_text(source)
        for flags in itertools.product((False, True), repeat=4):
            options = dict(
                zip(
                    ("echo_comments", "line_numbers", "select_case", "label_loops"),
                    flags,
                    strict=True,
                )
            )
            command = [str(executable)]
            for enabled, option in zip(
                (flags[0], flags[1], flags[2], not flags[3]), ("-c", "-n", "-s", "-l"), strict=True
            ):
                if enabled:
                    command.append(option)
            command.append(stem)
            result = subprocess.run(command, cwd=cases, text=True, capture_output=True, timeout=5)
            if result.returncode or result.stderr:
                raise RuntimeError(f"C reference failed for {name}: {result.stderr}")
            output = (cases / (stem + ".f")).read_text()
            collected.append(dict(name=name, source=source, options=options, fortran_source=output))
    if args.check:
        if collected != existing:
            raise ValueError("native translations differ from committed fixtures")
    else:
        _FIXTURE.write_text(json.dumps(collected, indent=2) + "\n")
    print(
        f"{len(collected)} native translation references {'verified' if args.check else 'recorded'}"
    )


if __name__ == "__main__":
    main()
