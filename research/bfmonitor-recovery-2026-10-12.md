# BFMonitor143 public-paper recovery continuation

The implemented explicit iMOM shape interface remains scientifically separate
from the app's ESS input. Existing docs/bfmonitor.md records paper k=1.24 versus
app display k=1.25 at ESS=5, null 0.2 and alternative mode 0.4. A rounded default
is not enough to recover arbitrary calibration or a reference-prior convention.

Primary source identification was extended to PMC9308506, manuscript identifier
NIHMS1820126 and publisher supporting file pst2139-sup-0001-Supinfo.pdf. The
publisher lists that Appendix S1; listing is not successful content recovery.
Direct PMC requests returned a browser challenge, direct Europe PMC XML returned
Forbidden in this environment, and PDF fetches failed. No challenge was bypassed.

An Exa search reviewed five candidate results in one calibration-source workstream.
The PMC and publisher fetches returned abstract/bibliographic material, not the
needed equations. They were not treated as full text. The final supplement fetch
failed with an Exa connection error; no retry, authentication or settings change
was made. This does not establish that the public manuscript has no equations.
The next evidence requirement remains its explicit ESS/reference-prior equations
and independently checked search/rounding rules. No fitted calibration curve is
inferred from the live default, and catalog143 is not promoted.

Primary URLs:
- https://pmc.ncbi.nlm.nih.gov/articles/PMC9308506/
- https://onlinelibrary.wiley.com/doi/10.1002/pst.2139
- https://biostatistics.mdanderson.org/shinyapps/BFMonitor/
- Existing guide and BioC provenance: docs/bfmonitor-sources.json.

Work advances to CondiS157's recoverable supplied-sample saved workflow while
this source boundary remains explicit. No author contact or remote GitHub retry.
