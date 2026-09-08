"""Explicit beta prior conventions for SEQBIN study inputs."""

from ._validation import scalar


def seqbin_prior(
    mean: float, effective_subjects: float, *, conversion: str = "exact"
) -> tuple[float, float]:
    """Convert a beta mean and total shape mass to positive shape parameters.

    exact uses a=mean*effective_subjects and b=(1-mean)*effective_subjects.
    legacy reproduces SEQBIN's a=mean*(effective_subjects+1)-0.5,
    b=effective_subjects-a, which generally changes the actual beta mean.
    Use (0.5,0.5) for the source's noninformative prior or (1,1) for uniform.
    """
    mu, size = scalar(mean, "mean"), scalar(effective_subjects, "effective_subjects")
    if not 0 < mu < 1 or size <= 0:
        raise ValueError("Require mean in (0,1) and positive effective_subjects")
    if conversion == "exact":
        a, b = mu * size, (1 - mu) * size
    elif conversion == "legacy":
        a = mu * (size + 1) - 0.5
        b = size - a
    else:
        raise ValueError("conversion must be exact or legacy")
    if a <= 0 or b <= 0:
        raise ValueError("The requested prior produces a nonpositive beta shape")
    return a, b
