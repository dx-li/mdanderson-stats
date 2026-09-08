"""Record the unmodified BMPVPB/BMURPB decision loop using native MIXBET."""

import json
from pathlib import Path

from reference_multi import build
from reference_numerics import run


def main():
    executable = build()
    previous = json.loads(Path("tests/fixtures/beta_mixture.json").read_text())
    cases = []
    for evaluation in previous["evaluations"]:
        model = evaluation["model"]
        for sequence in ("rank", "input"):
            original = evaluation["pvalues"]
            order = (
                sorted(range(len(original)), key=original.__getitem__)
                if sequence == "rank"
                else list(range(len(original)))
            )
            values = [original[i] for i in order]
            for alpha in (0.05, 0.5, 1.0):
                k = len(model["weights"])
                data = f"21 {len(values)} {alpha} 0\n"
                data += " ".join(map(str, values)) + "\n"
                data += f"{k} {model['null_weight']}\n"
                if k:
                    for key in ("weights", "a", "b"):
                        data += " ".join(map(str, model[key])) + "\n"
                output = [row.split() for row in run(executable, data).splitlines()]
                scores, reject = [0.0] * len(values), [False] * len(values)
                for i, row in zip(order, output, strict=True):
                    scores[i], reject[i] = float(row[0]), bool(int(row[1]))
                cases.append(
                    {
                        "pvalues": original,
                        "model": model,
                        "sequence": sequence,
                        "alpha": alpha,
                        "scores": scores,
                        "reject": reject,
                    }
                )
    result = {
        "provenance": {
            k: v
            for k, v in previous.items()
            if k not in ("evaluations", "starts", "em_fits", "ml_fits")
        },
        "notes": "Unchanged original desktop BMPVPB decision loop, verified equal to "
        "BMURPB after local variable/label renaming. Original MIXBET and INITLN. "
        "Report/input UI replaced by a numerical harness.",
        "cases": cases,
    }
    Path("tests/fixtures/beta_testing.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    print(f"Recorded {len(cases)} native desktop beta decision cases")


if __name__ == "__main__":
    main()
