# aPCoA functional workflow coverage review

This review asks whether the cached sources expose a further reproducible
analysis step that is missing from Python, rather than requiring exact Shiny
appearance, R object serialization, or undocumented file bytes.

## Source-to-Python workflow

| User step | Python path | Evidence and boundary |
| --- | --- | --- |
| Load a labeled distance matrix and metadata | `read_apcoa_distance_csv`, `read_apcoa_metadata_csv` | `docs/apcoa-inputs.md` documents bounded CSV/TSV formats, uniqueness, and exact sample-ID alignment. |
| Select adjustment covariates and a separate display group | `prepare_apcoa_input` | Numeric columns and explicit treatment-coded categorical columns are aligned by ID; `main_group` is retained separately. The caller chooses the categorical reference and whether to include an intercept. |
| Compute the original and adjusted ordinations | `PreparedAPCoAInput.fit` and `adjusted_pcoa` | The cached `R/aPCoA.R` constructs `model.frame`/`model.matrix`, rank-reduces the design, removes its intercept column, and applies the projected centered-distance equation. Python implements the matrix equation with an explicit encoding/rank policy and stable SVD, not arbitrary R formula evaluation. |
| Inspect and display grouped ordinations | `plot_adjusted_pcoa`, `adjusted_pcoa_plot_geometry` | The existing plot provides original/adjusted panels; optional ellipse and medoid geometry is separately source-audited in `research/apcoa-plot-geometry-audit.md`. Group labels remain attached to original sample order. |
| Save a plot | Returned Matplotlib axes | The plotting API returns the figure axes; callers can use Matplotlib's ordinary save operation. The package does not claim native image styling or bytes. |

The equation and validated fixtures are documented in `docs/apcoa.md`; labeled
input preparation is documented in `docs/apcoa-inputs.md`. Those guides cite
the unmodified aPCoA 1.3 comparison and independent overlay references. No
separate permutation test, hypothesis test, or additional model output is
defined by the cached method source or static app page.

## Remaining source boundaries

The cached static Shiny page exposes metadata and distance file uploads, main
and adjustment covariate outputs, categorical/ellipse/center controls, a plot
action, and download links named for example inputs, plots, and “aPCoA plot
information” (`research/raw/aPCoA/app.html`, around lines 45–140). The cached
page does not include server code or a payload/schema for those downloads. The
Python package can save a returned plot through Matplotlib, but reproducing an
unknown result file would invent its contents.

Likewise, the original R call uses the host R implementation's formula,
contrasts, `model.matrix`, and QR pivot/rank conventions. The Python input
adapter instead requires already selected covariates and explicit categorical
levels/reference. That is a documented caller choice, not an unimplemented
matrix equation. Exact formula/contrast parity is a real unresolved
convention; no broader formula interpreter was added without a bounded,
specified compatibility contract.

Under the functional Python workflow criterion, the audited source-supported
path is available from aligned files through calculation and grouped plotting.
No additional mathematical calculation or fully specified export workflow was
identified. This is a coverage review, not a claim of exact Shiny/R interface
parity and not a catalog-status change.
