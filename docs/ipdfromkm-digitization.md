# IPDfromKM image digitization

`load_km_image`, `pick_km_curve`, and `digitize_km_points` cover the manual
image-to-coordinate step used before IPD reconstruction. This workflow asks the
user to calibrate the plotted axes and select curve points. It does not trace a
curve automatically, infer risk-table values, or recover original patient data.

```python
from mdanderson_stats import (
    digitize_km_points,
    km_axis_calibration,
    load_km_image,
    plot_ipdfromkm_diagnostics,
    plot_km_digitization,
    prepare_km_coordinates,
    reconstruct_ipd,
)

image = load_km_image("published-curve.png")
calibration = km_axis_calibration(
    image,
    x_pixel_anchors=[80, 720],  # left, right pixel centers
    x_values=[0, 60],  # corresponding time values
    y_pixel_anchors=[540, 80],  # lower, upper survival pixel rows
    y_values=[0, 1],
    time_unit="months",
)
curve = digitize_km_points([[80, 80], [210, 92], [350, 140], [510, 220], [720, 390]], calibration)
prepared = curve.prepare()
ipd = reconstruct_ipd(
    prepared.time,
    prepared.survival,
    patients=120,
    risk_time=[0, 12, 24, 36, 48],
    at_risk=[120, 91, 65, 40, 18],
)
plot_km_digitization(image, calibration, curve).figure.savefig("digitized.png")
plot_ipdfromkm_diagnostics(prepared, ipd, time_unit=calibration.time_unit)[0].figure.savefig(
    "reconstruction-check.png"
)
```

Pixel coordinates use the image convention: `(0, 0)` is the center of the
upper-left pixel, x increases rightward and y increases downward. Supply
left-to-right time anchors and lower-to-upper survival anchors; their pixel-row
order is therefore usually reversed. The Python transform is linear between
these anchors and accepts only curve points inside the calibrated rectangle.
Native R applies the linear transform to selected clicks without this inside-
the-axes restriction, so the Python check is an explicit safety constraint.
Use actual time units and survival probabilities in the anchor values.

`pick_km_curve` provides mouse-based anchor and curve selection when Matplotlib
has an interactive backend. Click four anchors in the documented order, then
click curve points from left to right; middle-click or press Enter to end the
curve early. A maximum of 512 points is accepted. The picker is optional and
requires the `plot` extra; image loading requires the `image` extra. Raster
files are limited to 100 MB encoded and 10 million decoded pixels, and animated
or multipage images are rejected. Larger images can be cropped externally
before loading.

The click sequence is kept in `DigitizedKMCurve.pixel_points`, `.time`, and
`.survival`. Call `.prepare()` explicitly to apply the package's separately
documented native coordinate-cleaning rules; inspect its omission and correction
counts before reconstruction. Calibration and outputs record the source image
name and time-unit label. Keep the original bitmap and click data with any
analysis if exact reviewability matters.

The diagnostic plot overlays the supplied prepared curve with the reconstructed
IPD Kaplan–Meier curve over its full event-time step grid, followed by
coordinate differences and (when available) risk-table counts. Since
`ReconstructedIPD` does not retain the original survival inputs, the caller is
responsible for pairing it with the matching prepared curve; matching time
coordinates alone cannot establish that relationship. No confidence ribbon is
drawn. The native ±5 RMSE ribbon is a visual tolerance, not a statistical
confidence interval.

This supplies the recovered functional manual image/calibration/plot workflow
without claiming native graphical styling, file-format byte identity, automatic
tracing, or GUI parity. Compatibility with the legacy application's remaining
UI and export behavior has not been verified.
