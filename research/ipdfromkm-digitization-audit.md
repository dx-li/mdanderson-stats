# IPDfromKM digitization source crosswalk

The implementation is based on cached CRAN source for `IPDfromKM` 0.1.10,
source revision `16ea3e163b8ad409e51e035154c52803dcb1c28b` (see the repository's
existing [IPDfromKM source record](../docs/ipdfromkm-sources.json)). The cached
source is retained outside this change; no upstream source code is redistributed.

`R/getpoints.R` defines the interactive acquisition contract: read a plotted
image, ask the user to mark left/right time-axis and lower/upper survival-axis
anchors, then click KM curve coordinates. The curve is collected from the
image; no source-defined automatic trace detector or pixel-classification
algorithm is present. The function accepts no more than 512 manually selected
curve points and computes x and y linear mappings from their separate anchor
pairs. Its comments and prompt labels disagree on x-anchor names, so the Python
API names them by visual role and explicitly records the required order.

`R/plot.getKM.R` compares the reconstructed Kaplan–Meier estimate to the
digitized curve, plots their difference, and conditionally compares risk counts
when multiple risk-table rows are supplied. It adds a ±5 RMSE visual envelope;
the code does not define that envelope as a calibrated confidence procedure.
The Python comparison uses the existing Kaplan–Meier summary and reconstructed
risk values, displays the full reconstructed step curve, and omits the
potentially misleading envelope.

The Python boundary is explicit: users supply image-axis anchor values and
clicked points, then separately invoke `prepare_km_coordinates` and
`reconstruct_ipd`. Python restricts clicked points to the anchor rectangle,
labels time units, caps encoded and decoded image size before decoding, and
keeps the original click sequence. These are documented usability/resource
policies; pixel-identical native UI behavior is not claimed. Optional Pillow and
Matplotlib imports remain lazy.

The diagnostic renderer cannot verify that a caller pairs a reconstruction
with the original prepared survival vector because `ReconstructedIPD` does not
store that vector. It checks matching coordinates and labels this as a caller
responsibility rather than mutating the statistical result model for plotting.
