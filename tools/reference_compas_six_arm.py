"""Independent six-dimensional marginal-prior check for a reported outer-quadrature concern."""

import json
from pathlib import Path

from reference_compas_three_arm import evidence

values = [
    dict(order=order, log_evidence=evidence([100] * 6, [50] * 6, 1000.0, order))
    for order in (8, 10, 12)
]
output = dict(
    completed=[100] * 6,
    successes=[50] * 6,
    variance=1000.0,
    method=(
        "Laplace-centered six-dimensional marginal-prior Gaussian quadrature, "
        "without a response threshold"
    ),
    refinements=values,
)
Path("tests/fixtures/compas-six-arm-reference.json").write_text(json.dumps(output, indent=2) + "\n")
print(json.dumps(output, indent=2))
