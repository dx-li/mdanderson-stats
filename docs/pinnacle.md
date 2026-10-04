# Pinnacle

Pinnacle detects and quantifies protein spots in a set of already aligned
two-dimensional gel images. This implementation follows the
[2008 article](https://academic.oup.com/bioinformatics/article/24/4/529/206532)
and the MD Anderson
[detailed manual](https://biostatistics.mdanderson.org/SoftwareDownload/SoftwareFiles/Pinnacle/Pinnacle%20detailed%20demo%20and%20specifications.pdf).
Image registration is an upstream requirement.

## Python workflow

Supply a callable that returns the same aligned gels on each pass. This lets a
file-backed application read one image at a time. The following small example
generates three gels and analyzes a 24-by-24 region:

```python
import numpy as np
from mdanderson_stats import run_pinnacle

row, col = np.indices((32, 32))
base = (
    2.0
    + 25 * np.exp(-((row - 10) ** 2 + (col - 11) ** 2) / 4)
    + 18 * np.exp(-((row - 21) ** 2 + (col - 20) ** 2) / 6)
)


def gels():
    for scale in (0.8, 1.0, 1.2):
        yield scale * base + 0.1


result = run_pinnacle(
    gels,
    region=(4, 28, 4, 28),
    levels=3,  # 24 is divisible by 2**3.
    peak_radius=1,
    background_radius=3,
)
assert result.image_count == 3
assert result.average_image.shape == (24, 24)
assert result.denoising.origin == (4, 4)
assert result.quantification.normalized.shape == (3, len(result.peaks.coordinates))
np.testing.assert_allclose(result.quantification.normalized.mean(axis=1), 1)
```

`average_image` and `denoising.image` are cropped images. Peak coordinates and
the denoising origin refer to the original images. Quantification matrices have
one row per gel and one column per detected peak, in the returned peak order.
The workflow checks that both passes have identical ordered pixel data, counts
and dimensions using a streaming digest; it rejects changing input.

### Reading aligned TIFF gels

`PinnacleTiffSource` is a replayable TIFF input for file-backed analyses. It
preflights every selected TIFF frame before decoding any pixels, then opens,
decodes, copies and closes one file at a time on each pass. Install Pillow
through the optional `image` extra (`python -m pip install '.[image]'` from
the repository checkout); this adapter supports Pillow >=12.3,<13.

```python
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image

from mdanderson_stats import PinnacleTiffSource, run_pinnacle

with TemporaryDirectory() as temporary_directory:
    tiff_paths = []
    for index, height in enumerate((18, 20, 22)):
        pixels = np.full((16, 16), 2 + index, dtype=np.uint16)
        pixels[7, 8] = height
        path = Path(temporary_directory) / f"gel-{index}.tif"
        Image.fromarray(pixels).save(path, format="TIFF")
        tiff_paths.append(path)

    tiff_source = PinnacleTiffSource(tiff_paths)
    tiff_result = run_pinnacle(
        tiff_source,
        filter_length=2,
        levels=2,
        peak_radius=0,
        background="none",
        normalization="none",
    )
    assert len(tiff_source.image_ids) == tiff_result.image_count == 3
    assert tiff_result.average_image.shape == (16, 16)
```

The order supplied in `paths` is retained. `image_ids` contains each resolved
path and its zero-based selected frame, so the identity used by the pipeline's
two-pass pixel digest is inspectable. The ordered header records are available
in `frame_metadata`. The source accepts 2–200 selected TIFF frames and limits
each image to 4,194,304 pixels. Its default 512 MiB
`max_decode_bytes` preflight uses a conservative 48-byte-per-pixel workspace
estimate for decoder buffers, array conversion and overlap with the previous
streamed image. This is a preflight estimate, not a bound on Pillow's internal
decoder or process-wide memory.

Only single-sample grayscale integer or 32-bit floating-point TIFFs supported
by Pillow >=12.3,<13 are accepted. Pixels must be finite and nonnegative;
values are not rescaled. RGB, palette, alpha and other multichannel images are
rejected rather than converted. Pillow applies TIFF Orientation
metadata while loading; returned arrays therefore use Pillow's normalized
top-left orientation, with dimensions measured after that orientation. The
TIFF photometric tag is retained as metadata; for 8-bit WhiteIsZero files the
source reverses Pillow's decoder inversion to preserve the stored numerical
samples. No additional photometric inversion or display-oriented conversion
is applied.

Byte order is honored for supported samples; Pillow 12.3 does not support
every TIFF combination. In particular, its decoder rejects big-endian
WhiteIsZero unsigned 16-bit files, unsigned big-endian 32-bit TIFFs are not
mapped by its scalar decoder, and non-native-order signed integer pages that
use libtiff are rejected because their decoder rawmodes are not normalized.

By default, each file must contain exactly one frame. For multipage inputs,
provide one explicit selector per path, for example
`PinnacleTiffSource([stack_path, stack_path], frame_indices=[0, 1])`. Repeated
paths can select different pages. `None` requires a single-frame file, while
an integer selects that zero-based page; selectors are bounded below 200 and
included in each image ID. The source does not silently choose a page or treat
an unselected multipage file as a stack. It does not cache decoded frames, and
each repeated pass reopens the files; `run_pinnacle` verifies that the selected
ordered pixels still match.

The stages are also available separately through `pinnacle_mean_image`,
`pinnacle_denoise`, `pinnacle_detect_peaks` and `pinnacle_quantify`.
`pinnacle_daubechies_filter`, `pinnacle_rdwt` and `pinnacle_irdwt` expose the
wavelet calculations for inspection or other analyses.

## Method

The pixelwise average across gels supplies the common detection image.
An undecimated Daubechies wavelet transform removes small detail coefficients
and reconstructs a denoised average. Peaks must exceed an intensity quantile
and be local maxima in both horizontal and vertical directions. Nearby
candidates are suppressed in favor of brighter ones.

For each detected location and each gel, quantification takes the largest
intensity in a surrounding square. A local minimum or local/global quantile
can estimate the background. Normalization can divide corrected peak values
by their gel's mean or sum of corrected peaks, or by the raw image volume in
the selected region. Background correction and normalization can also be
disabled explicitly. Negative corrected values are retained; an unusable
normalization denominator raises an error.

## Individual-gel denoising and rectangular backgrounds

`background_radius=(row_radius, column_radius)` selects separate background
window sizes. For example, `(2, 5)` uses up to five rows and eleven columns,
clipped to the selected region. This applies to local minimum and local
quantile backgrounds. A scalar keeps the original square-window behavior.

Individual-gel denoising is optional and off by default, as in the later manual.
Use an explicit `PinnacleDenoiseSettings` object to select filter, threshold
and noise conventions. Continuing the example above:

```python
from mdanderson_stats import PinnacleDenoiseSettings

settings = PinnacleDenoiseSettings(
    filter_length=6,
    threshold_multiplier=3.6,
    convention="rwt",
    levels=3,
    max_work_bytes=64 * 1024 * 1024,
)
individual = run_pinnacle(
    gels,
    region=(4, 28, 4, 28),
    levels=3,
    peak_radius=1,
    background_radius=(2, 5),
    normalization="image_volume",
    quantification_denoising=settings,
)
np.testing.assert_array_equal(individual.peaks.coordinates, result.peaks.coordinates)
np.testing.assert_allclose(
    individual.quantification.normalization_factors,
    [gel[4:28, 4:28].sum() for gel in gels()],
)
assert individual.quantification.denoising_thresholds.shape == (3,)
```

Detection still uses the denoised average of raw gels. Each individual gel's
noise is estimated separately, then its reconstruction supplies peak and
background measurements. Signed reconstructed values are retained. Image-volume
normalization uses the original raw image region; peak-based normalization uses
the corrected measured peaks. Raw images also remain the basis of the replay
identity check. This processing order is an explicit Python convention;
the manual does not establish executable-level ordering for every option.

The example selects the manual's filter length and threshold multiplier with
the separately documented Rice noise convention. This is a specified analysis,
not a claim that the hidden executable uses that exact noise convention.
Direct `pinnacle_quantify` calls accept the same object as `denoising=settings`.
Results retain settings and per-gel noise/threshold summaries without retaining
the reconstructed images.

## Reproducible conventions

The 2008 paper uses a Daubechies filter with four vanishing moments (length
eight) and threshold multiplier two. The later GUI manual specifies filter
length six and multiplier 3.6. These are different configurations.

The wavelet transform uses periodic boundaries and requires both dimensions
to be divisible by `2**levels`. No padding or resampling is implicit. The
default level count follows the original denoiser's floor of the base-two
logarithm of the smaller dimension; some rectangular dimensions therefore
require a smaller explicit level count.
The lower-level `pinnacle_rdwt` transform instead defaults to the largest
power of two dividing both dimensions, following the transform routine itself.

The paper and original Rice toolbox describe different threshold conventions.
The paper setting uses a median-centered absolute deviation divided by .6745,
interpreting the article's MAD wording, and retains coefficients equal to the
threshold. The Rice setting uses the median of absolute coefficients about
zero divided by .67 and discards equality. Both estimate noise from the finest
diagonal detail image and preserve the final low-pass image. The exact
configuration inside the Pinnacle executable has not been verified.

Python peak coordinates are zero-based `(row, column)`. A region is
`(row_start, row_stop, column_start, column_stop)` with excluded stop indices.
Local windows are clipped to that region. A detection candidate must be
strictly above the intensity quantile and at least as high as its existing
four orthogonal neighbors. Candidates are processed by descending intensity,
then row and column. Greedy square suppression can retain separated points
on a broad plateau. These edge, equality and tie choices are explicit Python
conventions, rather than verified native application behavior.

## Memory and validation

The pixel mean and per-gel quantification consume images sequentially. The
complete workflow requires replayable input so it can make separate mean
and quantification passes without caching all gels. Inputs must remain aligned
and have the same dimensions. Raw images must be finite and nonnegative;
wavelet reconstruction can have signed values.

Wavelet calls check a conservative `(3*levels + 16)` full-image-buffer estimate
before allocating the coefficient pyramid. `max_work_bytes` defaults to 512 MiB
and is limited to 1 GiB. The standard 1024-by-1024, ten-level transform fits the
default estimate. This bounds the algorithm's working arrays, not the Python
runtime or images already held by the caller. Images are limited to 4,194,304
pixels and each input pass to 200 gels. Quantification separately bounds pixel
work and the combined size of its four image-by-peak result matrices.
Individual-gel denoising also accounts for retained pipeline images and result
matrices. Its effective budget is the smaller of the pipeline limit and the
settings limit; results report those effective settings.

Independent base-R calculations check filter coefficients, mean images,
cropped peak detection, backgrounds and normalization. The original Rice
Wavelet Toolbox 2.4 C transforms provide forward/inverse and denoising
references. See the [audit](../research/pinnacle-audit.md) for completed checks
and resource measurements.

This product includes software developed by Rice University, Houston, Texas
and its contributors. The complete conditions are in the
[preserved license](../notices/rice-wavelet-LICENSE.txt).

Unsupported TIFF encoding combinations, native project-file ingestion,
interactive peak editing and native report equivalence remain open. No original
Pinnacle executable, source archive or article PDF is bundled.
