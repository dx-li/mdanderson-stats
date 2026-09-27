# Next uncovered method: Pinnacle

Entry 95 is pending. The indexed [official page](https://biostatistics.mdanderson.org/SoftwareDownload/SingleSoftware/Index/95)
identifies version 1.2.6, updated August 27, 2014. The catalog records
`Pinnacle_V1.2.6_v.zip`; the native archive has not been inspected in this pass.

The official [detailed manual](https://biostatistics.mdanderson.org/SoftwareDownload/SoftwareFiles/Pinnacle/Pinnacle%20detailed%20demo%20and%20specifications.pdf)
is readable. It describes detection on an average of already aligned gel
images, followed by local intensity quantification, background correction and
normalization. Image registration is an upstream requirement, not a native
Pinnacle feature. The manual documents configurable Daubechies filter lengths,
thresholds, regions, spot neighborhoods and local/global background estimates.

The [2008 primary article](https://pmc.ncbi.nlm.nih.gov/articles/PMC2662725/)
specifies an undecimated wavelet transform, hard thresholding and a noise
estimate from finest-level coefficient MAD divided by 0.6745. Its defaults
differ from the later GUI manual; do not silently equate the two versions.
The [2016 chapter](https://pmc.ncbi.nlm.nih.gov/articles/PMC5512556/) provides a
second readable account of the later settings and may recover equations that
the PDF text extractor omitted.

Next: inspect the complete method and available source/archive terms; resolve
wavelet boundary/level conventions and exact neighborhood definitions. Search
the package for reusable image, quantile and normalization primitives. Plan
bounded, preferably streamed image handling before implementation: avoid
materializing a large gel stack or multiple wavelet copies after the user's
OOM report. Do not replace the specified undecimated denoiser with a generic
blur. No coverage or executable-equivalence claim is made yet.

The [publisher's full 2008 article](https://academic.oup.com/bioinformatics/article/24/4/529/206532)
is accessible even though both live PMC pages returned browser challenges.
It specifies cross-direction local maxima, the 75th intensity percentile,
square proximity suppression (default radius 2), maximum-intensity spot
quantification in a matching square, local-minimum background within radius
100, and normalization by each gel's mean pinnacle intensity. Native tie,
boundary and suppression-order conventions still need an explicit treatment.
The later manual adds quantile background and image-volume normalization;
these should remain distinguishable settings. It also identifies patent
US 8,031,925 in its acknowledgements; source/archive terms have not yet been
inspected, and no native code has been copied.

There is no wavelet/gel implementation or wavelet dependency in the current
package. The institutional [Rice Wavelet Toolbox page](https://www.ece.rice.edu/dsp/software/rwt.shtml)
documents the 2.4 release used by the paper. Rice's [source repository](https://github.com/ricedsp/rwt)
retains older distributions in `dist`, plus forward/inverse redundant transforms,
denoising and Daubechies filter routines. This is a primary-source lead for
resolving transform conventions and building a small independent reference;
do not infer that current version 3.0 defaults equal the paper's version 2.4.
