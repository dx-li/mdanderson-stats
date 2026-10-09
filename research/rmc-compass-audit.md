# RMC-COMPASS public model recovery — October 8, 2026

The public [institutional app](https://biostatistics.mdanderson.org/shinyapps/RMC-COMPASS/)
was inspected with system Chromium 151 through Playwright, retaining HTTPS
verification. Its “Model Details and Coefficients” panel rendered successfully.
The environment's existing proxy CA was imported into Chromium's trust store;
no certificate-verification bypass was used. No private endpoint, patient
data, fitted-model download or author correspondence was accessed.

The live app identifies PID 1189, V1.0.2.0, updated September 15, 2026. Its
panel supplies the log-normal AFT distribution, sigma 0.8334, intercept
5.7273, uniform shrinkage 0.895 and four unshrunken coefficients. The formula
expressly uses shrunken coefficients. Time is in months, confirmed against
the survival table and median rather than assumed from the intercept.
The complete public equation is transcribed in the Python guide.

The input panel reports NLR development range 0.84–22, corrected calcium
8.3–11.2, ECOG development maximum 3 and disease-site maximum 7. It accepts
ECOG 4 and eleven selectable sites. Fresh outputs confirm that continuous
inputs are clamped and ECOG/site extrapolation is retained. The raw calcium
widget responds to albumin changes on both sides of 4 g/dL, but the Python
API accepts already corrected calcium: a complete raw-input/validation
contract was not recovered. It similarly accepts an already computed NLR.

The committed fixture contains eleven independent native responses, each
with five displayed survival probabilities, a displayed median and a risk
group. Numeric changes were allowed to debounce and finish before clicking
Calculate Prognosis. Except for the default observation, capture waited for
the rendered prediction summary to change before recording the table. Earlier
captures made before debouncing produced stale predictions; **those were
excluded** from the fixture. The raw verified captures remain ignored under
`research/raw/rmc-compass/verified-ui-captures.json`; their SHA-256 is pinned
in the source record and fixture. Shiny's untyped numeric input-value lookup
returns null because numeric bindings use typed names; it is not interpreted
as the calculator accepting missing values. Requested changes and rendered
computed-covariate text establish the input profiles used here.

The default rounded equation gives median 21.248 months whereas the app
displays 21.3. Thus a full-precision parity claim is disproved even before
examining uncertainty. The Python API is explicitly named
`rmc_compass_reported_prediction`. Its envelopes propagate the displayed
parameter precision conditional on effective inputs; they are not statistical
confidence intervals. All 55 native probability rounding intervals and eleven
median rounding intervals intersect those envelopes; all eleven risk groups
agree. Clamping references are observed consistency checks, not proof of the
unrounded native clamp limits. Tests also cover zero time, output ordering,
immutable arrays, horizon restrictions, visible clamping/extrapolation and
classification sensitivity near a cutoff. All 29 focused tests passed.

No fitted covariance, exact parameter vector or uncertainty algorithm was
exposed by the inspected panel. Native intervals cannot be reconstructed
reliably from a few rounded output intervals. Entry 174 moves from pending
to partial, with exact fit, uncertainty, raw-input rules and complete native
output workflows still open. No public-panel HTML/artwork is redistributed.
