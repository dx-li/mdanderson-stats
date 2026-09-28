# MDS-HOPE source status

Catalog entry 171 remains pending. Inspected September 28, 2026.

The indexed [official app](https://biostatistics.mdanderson.org/shinyapps/MDS-HOPE/)
describes six-group risk stratification for patients with MDS treated with
hypomethylating agents. Its visible inputs include age, marrow blasts,
hemoglobin, platelets, absolute neutrophil count, TP53 allelic status,
PTPN11/KRAS/SF3B1/EZH2 mutations and karyotype text. Mutation menus include
an unspecified status; its interpretation is not established by the form.
Visible outputs are a patient summary, stratification result and plot.

The primary publication is Chien et al.,
[Performance of molecular scoring systems in hypomethylating agent-treated
myelodysplastic neoplasms](https://www.nature.com/articles/s41375-026-02895-5),
Leukemia 40, 841–844 (2026), published March 6, 2026. Figure 1 names MDS-HOPE
and reference 12 links the same six-group calculator, establishing its relation
to the deployed tool. Accessible methods describe Cox proportional-hazards
modeling and derived scores and refer to Supplemental Equation S1 for the
model equation. The main article preview does not supply deployable parameters.

The publisher's exact **Supplemental Material (download DOCX)** link is
[41375_2026_2895_MOESM1_ESM.docx](https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41375-026-02895-5/MediaObjects/41375_2026_2895_MOESM1_ESM.docx).
The reader returned a cache miss for that document; no local supplement was
retrieved. An alternate text extractor was unavailable because its account
quota was exhausted. The failed retrieval does not establish that the
supplement lacks the model.

Implementation still needs the exact score equation and coefficients,
transformations/input coding, cytogenetic parsing rules, unspecified-mutation
handling, six-group cutpoints and the plotted estimand/time scale. Absolute
survival predictions would additionally require the model's baseline survival.
No coefficients, cutpoints or clinical risks have been estimated from the
input form, and no deployed prediction was reproduced. The exact supplement
is the next source to inspect when accessible; a generic Cox fitter would
not implement this clinical calculator.
