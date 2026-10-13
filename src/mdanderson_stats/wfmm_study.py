"""Versioned Python WFMM studies, qualified execution and saved summaries."""

from __future__ import annotations

import csv
import hashlib
import io
import json
from dataclasses import dataclass, fields, replace

import numpy as np
from numpy.typing import ArrayLike

from ._cdflib import _freeze
from .hierarchical_binomial import summarize_chains
from .wfmm_basis import WFMMBasis, _setting, wfmm_basis, wfmm_transform
from .wfmm_boundary import wfmm_extended_wavelet_basis, wfmm_named_wavelet_basis
from .wfmm_compression import TiePolicy, WFMMCompression, wfmm_compress_coefficients
from .wfmm_covariance import WFMMCovarianceSummary, wfmm_summarize_covariance
from .wfmm_hyperprior import WFMMShrinkageHyperprior, _hyperprior_arrays
from .wfmm_initialization import WFMMInitialization, initialize_wfmm
from .wfmm_model import WFMMCoefficientFit, _strata, fit_wfmm_coefficients
from .wfmm_pca import wfmm_pca_basis
from .wfmm_posterior import WFMMPosteriorSummary, wfmm_summarize
from .wfmm_prediction import wfmm_predict_coefficients
from .wfmm_selection import wfmm_pass_filter, wfmm_restore_coefficients
from .wfmm_spatial import wfmm_spatial_wavelet_basis

_MAX_JSON_BYTES = 16_000_000


def _matrix(value, name, rows=None, columns=None):
    raw = np.asarray(value)
    if (
        raw.ndim != 2
        or raw.dtype.kind not in "iuf"
        or raw.size == 0
        or raw.size > 2_000_000
        or (rows is not None and raw.shape[0] != rows)
        or (columns is not None and raw.shape[1] != columns)
        or not np.all(np.isfinite(raw))
    ):
        raise ValueError(f"{name} must be a bounded finite real matrix with matching dimensions")
    with np.errstate(over="ignore", invalid="ignore"):
        converted = np.asarray(raw, dtype=float)
    if not np.all(np.isfinite(converted)):
        raise ValueError(f"{name} is not representable as float64")
    return _freeze(converted)


def _json_value(value):
    if isinstance(value, WFMMShrinkageHyperprior):
        return {f.name: _json_value(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, (tuple, list)):
        return [_json_value(x) for x in value]
    if isinstance(value, dict):
        return {k: _json_value(v) for k, v in value.items()}
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


@dataclass(frozen=True)
class WFMMStudy:
    """Saved observed curves and explicit controls for the stable Python workflow.

    The JSON schema is not a native MAT-file format. Initializer quality holds
    stop execution before MCMC; there is no automatic compatibility fallback.
    Predictions combine fixed effects, retained existing random levels, optional
    new random levels and optional residual noise. Effect contrasts apply only
    to fixed-effect summaries; time contrasts also apply to random/predictive
    summaries. Covariance functions retain the original time grid.
    """

    curves: ArrayLike
    fixed_design: ArrayLike
    random_design: ArrayLike | None = None
    random_strata: ArrayLike | None = None
    residual_strata: ArrayLike | None = None
    transform: str = "wavelet"
    analysis_matrix: ArrayLike | None = None
    synthesis_matrix: ArrayLike | None = None
    levels: int = 1
    filter_length: int = 2
    alpha_pc: float = 0.99
    pc_partition_base: float = 10.0
    highpass: int = 0
    lowpass: int = 0
    alpha_energy: float = 1.0
    compression_votes: int = 0
    compression_ties: TiePolicy = "reject_split"
    draws: int = 1000
    warmup: int = 500
    chains: int = 2
    seed: int = 0
    min_variance: float = 1e-6
    delta_omega: float = 1e-4
    proposal_multiplier: float = 1.5
    moment_iterations: int = 100
    optimizer_iterations: int = 200
    empirical_bayes_iterations: int = 1000
    prediction_design: ArrayLike | None = None
    prediction_residual: bool = False
    prediction_strata: ArrayLike | None = None
    boundary: str = "periodic"
    wavelet_name: str | None = None
    spatial_shape: tuple[int, ...] | None = None
    spatial_layout: str = "pyramid"
    axis_levels: tuple[int, ...] | None = None
    axis_wavelets: tuple[str, ...] | None = None
    axis_boundaries: tuple[str, ...] | None = None
    summarize_random_effects: bool = False
    include_covariance: bool = False
    effect_contrast: ArrayLike | None = None
    time_contrast: ArrayLike | None = None
    confidence: float = 0.95
    quantiles: ArrayLike = (0.025, 0.5, 0.975)
    effect_sizes: ArrayLike = (0.0,)
    prediction_existing_random_design: ArrayLike | None = None
    prediction_new_random_design: ArrayLike | None = None
    prediction_new_random_strata: ArrayLike | None = None
    thin: int = 1
    shrinkage_hyperprior: WFMMShrinkageHyperprior | None = None

    def __post_init__(self):
        if self.shrinkage_hyperprior is not None and not isinstance(
            self.shrinkage_hyperprior, WFMMShrinkageHyperprior
        ):
            raise ValueError("shrinkage_hyperprior must be a WFMMShrinkageHyperprior")
        y = _matrix(self.curves, "curves")
        n, t = y.shape
        if not 2 <= n <= 500 or not 2 <= t <= 4096:
            raise ValueError("study curve dimensions exceed supported bounds")
        x = _matrix(self.fixed_design, "fixed_design", rows=n)
        if not 1 <= x.shape[1] <= min(n, 100):
            raise ValueError("fixed design has an unsupported column count")
        z = (
            None
            if self.random_design is None
            else _matrix(self.random_design, "random_design", rows=n)
        )
        if z is not None and z.shape[1] > 500:
            raise ValueError("random design exceeds 500 columns")
        if z is None and self.random_strata is not None:
            raise ValueError("random_strata requires random_design")
        object.__setattr__(self, "curves", y)
        object.__setattr__(self, "fixed_design", x)
        object.__setattr__(self, "random_design", z)
        if z is not None:
            labels = _strata(self.random_strata, z.shape[1], "random_strata")
            object.__setattr__(self, "random_strata", tuple(int(i) for i in labels))
        labels = _strata(self.residual_strata, n, "residual_strata")
        object.__setattr__(self, "residual_strata", tuple(int(i) for i in labels))
        if self.transform not in (
            "identity",
            "wavelet",
            "extended_wavelet",
            "spatial_wavelet",
            "custom",
            "PC",
            "PCw",
            "wPC",
        ):
            raise ValueError("unsupported study transform")
        if self.boundary not in ("periodic", "zero", "reflection"):
            raise ValueError("unsupported boundary convention")
        if (
            self.transform not in ("extended_wavelet", "spatial_wavelet")
            and self.boundary != "periodic"
        ):
            raise ValueError("nonperiodic boundary requires extended_wavelet")
        if self.transform == "spatial_wavelet":
            if self.filter_length != 2:
                raise ValueError("spatial_wavelet uses named wavelets and default filter_length")
            for axis, scalar, default in (
                ("axis_levels", "levels", 1),
                ("axis_wavelets", "wavelet_name", None),
                ("axis_boundaries", "boundary", "periodic"),
            ):
                if getattr(self, axis) is not None and getattr(self, scalar) != default:
                    raise ValueError("axis settings conflict with scalar settings")
            basis = _spatial_basis(self)
            if basis.time_count != t:
                raise ValueError("spatial_shape product must equal curve columns")
            object.__setattr__(self, "spatial_shape", basis.spatial_shape)
            for field in ("axis_levels", "axis_wavelets", "axis_boundaries"):
                value = getattr(self, field)
                if value is not None:
                    if not isinstance(value, (tuple, list)):
                        raise ValueError(field + " must be an axis sequence")
                    object.__setattr__(self, field, tuple(value))
        elif (
            self.spatial_shape is not None
            or self.spatial_layout != "pyramid"
            or any(
                value is not None
                for value in (self.axis_levels, self.axis_wavelets, self.axis_boundaries)
            )
        ):
            raise ValueError("spatial settings require spatial_wavelet")
        if self.wavelet_name is not None and self.transform != "spatial_wavelet":
            if self.transform != "extended_wavelet" or self.filter_length != 2:
                raise ValueError(
                    "wavelet_name requires extended_wavelet and the default filter_length"
                )
            wfmm_named_wavelet_basis(
                t, wavelet=self.wavelet_name, levels=self.levels, boundary=self.boundary
            )
        elif self.transform == "extended_wavelet":
            wfmm_extended_wavelet_basis(
                t, levels=self.levels, filter_length=self.filter_length, boundary=self.boundary
            )
        if self.transform == "custom":
            for name in ("analysis_matrix", "synthesis_matrix"):
                value = _matrix(getattr(self, name), name, rows=t, columns=t)
                object.__setattr__(self, name, value)
        elif self.analysis_matrix is not None or self.synthesis_matrix is not None:
            raise ValueError("matrix pairs require transform=custom")
        for name, low, high in (
            ("levels", 1, 12),
            ("filter_length", 2, 20),
            ("highpass", 0, 12),
            ("lowpass", 0, 12),
            ("compression_votes", 0, n),
            ("draws", 8, 5000),
            ("warmup", 0, 5000),
            ("thin", 1, 100),
            ("chains", 2, 4),
            ("seed", 0, 2**63 - 1),
            ("moment_iterations", 1, 5000),
            ("optimizer_iterations", 1, 5000),
            ("empirical_bayes_iterations", 1, 5000),
        ):
            object.__setattr__(self, name, _setting(getattr(self, name), name, low, high))
        if self.transform in ("wavelet", "PCw", "wPC") and t % (2**self.levels):
            raise ValueError("2**levels must divide the number of time points")
        if self.filter_length % 2:
            raise ValueError("filter_length must be even")
        if self.transform not in ("wavelet", "extended_wavelet", "spatial_wavelet") and (
            self.highpass or self.lowpass
        ):
            raise ValueError("saved pass flags currently require the wavelet transform")
        if (
            self.transform == "spatial_wavelet"
            and self.spatial_layout == "rectangular"
            and (self.highpass or self.lowpass)
        ):
            raise ValueError("rectangular spatial pass filtering is not yet supported")
        if self.transform == "spatial_wavelet" and self.spatial_layout == "pyramid":
            if (
                self.highpass > basis.levels
                or self.lowpass > basis.levels
                or (self.highpass and self.highpass + self.lowpass >= basis.levels)
            ):
                raise ValueError("spatial pass flags retain no coefficients or exceed levels")
        if self.compression_ties not in ("reject_split", "original_index"):
            raise ValueError("unsupported compression tie policy")
        for name, lo, hi, strict in (
            ("alpha_pc", 0, 1, False),
            ("alpha_energy", 0, 1, True),
            ("pc_partition_base", 1, 1e100, True),
            ("min_variance", 0, 1e100, True),
            ("delta_omega", 0, 1e100, True),
            ("proposal_multiplier", 0, 1e100, True),
        ):
            raw = np.asarray(getattr(self, name))
            if (
                raw.ndim != 0
                or raw.dtype.kind not in "iuf"
                or not np.isfinite(raw)
                or float(raw) > hi
                or (float(raw) <= lo if strict else float(raw) < lo)
            ):
                raise ValueError(f"invalid {name}")
            object.__setattr__(self, name, float(raw))
        if type(self.include_covariance) is not bool:
            raise ValueError("include_covariance must be boolean")
        if type(self.summarize_random_effects) is not bool:
            raise ValueError("summarize_random_effects must be boolean")
        if self.summarize_random_effects and (z is None or z.shape[1] > 128):
            raise ValueError("random-effect summaries require 1..128 fitted levels")
        if self.effect_contrast is not None:
            contrast = _matrix(self.effect_contrast, "effect_contrast", columns=x.shape[1])
            if contrast.shape[0] > 128:
                raise ValueError("effect_contrast exceeds 128 output effects")
            object.__setattr__(self, "effect_contrast", contrast)
        if self.time_contrast is not None:
            contrast = _matrix(self.time_contrast, "time_contrast", rows=t)
            if contrast.shape[1] > t:
                raise ValueError("time_contrast cannot expand the time dimension")
            object.__setattr__(self, "time_contrast", contrast)
        raw_confidence = np.asarray(self.confidence)
        if (
            raw_confidence.ndim != 0
            or raw_confidence.dtype.kind not in "iuf"
            or not np.isfinite(raw_confidence)
            or not 0 < float(raw_confidence) < 1
        ):
            raise ValueError("confidence must lie strictly between zero and one")
        object.__setattr__(self, "confidence", float(raw_confidence))
        for name in ("quantiles", "effect_sizes"):
            raw = np.asarray(getattr(self, name))
            if name == "effect_sizes" and raw.ndim == 0:
                raw = raw.reshape(1)
            if (
                raw.ndim != 1
                or raw.dtype.kind not in "iuf"
                or not 1 <= raw.size <= 128
                or not np.all(np.isfinite(raw))
            ):
                raise ValueError(f"{name} must be a bounded finite real vector")
            if name == "quantiles":
                if np.any((raw <= 0) | (raw >= 1)) or np.any(np.diff(raw) <= 0):
                    raise ValueError("quantiles must increase strictly within (0,1)")
            elif np.any(raw < 0):
                raise ValueError("effect_sizes must be nonnegative")
            object.__setattr__(self, name, tuple(float(value) for value in raw))
        if type(self.prediction_residual) is not bool:
            raise ValueError("prediction_residual must be boolean")
        if self.prediction_design is None:
            if (
                self.prediction_strata is not None
                or self.prediction_residual
                or self.prediction_existing_random_design is not None
                or self.prediction_new_random_design is not None
                or self.prediction_new_random_strata is not None
            ):
                raise ValueError("prediction controls require prediction_design")
        else:
            design = _matrix(self.prediction_design, "prediction_design", columns=x.shape[1])
            if design.shape[0] > 128:
                raise ValueError("prediction summary exceeds 128 rows")
            object.__setattr__(self, "prediction_design", design)
            for name in ("prediction_existing_random_design", "prediction_new_random_design"):
                value = getattr(self, name)
                if value is not None:
                    if z is None:
                        raise ValueError("random prediction requires a fitted random design")
                    matrix = _matrix(
                        value,
                        name,
                        rows=design.shape[0],
                        columns=z.shape[1] if name == "prediction_existing_random_design" else None,
                    )
                    if matrix.shape[1] > 500:
                        raise ValueError("random prediction exceeds 500 level columns")
                    object.__setattr__(self, name, matrix)
            if self.prediction_new_random_design is not None:
                new_labels = np.asarray(self.prediction_new_random_strata)
                count = np.asarray(self.prediction_new_random_design).shape[1]
                groups = int(np.max(np.asarray(self.random_strata))) + 1
                if (
                    new_labels.shape != (count,)
                    or new_labels.dtype.kind not in "iu"
                    or np.any(new_labels < 0)
                    or np.any(new_labels >= groups)
                ):
                    raise ValueError("prediction_new_random_strata must index fitted random strata")
                object.__setattr__(
                    self, "prediction_new_random_strata", tuple(int(i) for i in new_labels)
                )
            elif self.prediction_new_random_strata is not None:
                raise ValueError("new random strata require a new random design")
            if self.prediction_residual:
                lab = np.asarray(self.prediction_strata)
                groups = max(labels) + 1
                if (
                    lab.shape != (design.shape[0],)
                    or lab.dtype.kind not in "iu"
                    or np.any(lab < 0)
                    or np.any(lab >= groups)
                ):
                    raise ValueError("prediction_strata must index fitted residual strata")
                object.__setattr__(self, "prediction_strata", tuple(int(i) for i in lab))
            elif self.prediction_strata is not None:
                raise ValueError("prediction_strata is only used with residual prediction")

    def to_json(self) -> str:
        data = {f.name: _json_value(getattr(self, f.name)) for f in fields(self)}
        data.update(schema="mdanderson_stats.wfmm.study", version=1)
        text = json.dumps(data, sort_keys=True, allow_nan=False, separators=(",", ":")) + "\n"
        if len(text.encode()) > _MAX_JSON_BYTES:
            raise ValueError("saved study exceeds 16 MB")
        return text


def _spatial_basis(study):
    return wfmm_spatial_wavelet_basis(
        study.spatial_shape,
        levels=study.levels if study.axis_levels is None else study.axis_levels,
        wavelets=("db1" if study.wavelet_name is None else study.wavelet_name)
        if study.axis_wavelets is None
        else study.axis_wavelets,
        boundaries=study.boundary if study.axis_boundaries is None else study.axis_boundaries,
        layout=study.spatial_layout,
    )


def read_wfmm_study(text: str) -> WFMMStudy:
    """Read complete version1 JSON with duplicate/unknown-field rejection."""
    if not isinstance(text, str) or len(text.encode()) > _MAX_JSON_BYTES:
        raise ValueError("study JSON must be text no larger than 16 MB")

    def unique(pairs):
        out = {}
        for key, value in pairs:
            if key in out:
                raise ValueError("duplicate JSON field")
            out[key] = value
        return out

    def invalid_constant(value):
        raise ValueError(f"nonfinite JSON constant: {value}")

    try:
        data = json.loads(text, object_pairs_hook=unique, parse_constant=invalid_constant)
        if isinstance(data, dict):
            # Version 1 studies saved before extended boundaries used periodic defaults.
            for name, default in {
                "thin": 1,
                "shrinkage_hyperprior": None,
                "boundary": "periodic",
                "wavelet_name": None,
                "spatial_shape": None,
                "spatial_layout": "pyramid",
                "axis_levels": None,
                "axis_wavelets": None,
                "axis_boundaries": None,
                "summarize_random_effects": False,
                "include_covariance": False,
                "effect_contrast": None,
                "time_contrast": None,
                "confidence": 0.95,
                "quantiles": [0.025, 0.5, 0.975],
                "effect_sizes": [0.0],
                "prediction_existing_random_design": None,
                "prediction_new_random_design": None,
                "prediction_new_random_strata": None,
            }.items():
                data.setdefault(name, default)
        if (
            not isinstance(data, dict)
            or data.pop("schema", None) != "mdanderson_stats.wfmm.study"
            or type(data.get("version")) is not int
            or data.pop("version") != 1
            or set(data) != {f.name for f in fields(WFMMStudy)}
        ):
            raise ValueError("unsupported WFMM study schema/version/fields")
        if data["shrinkage_hyperprior"] is not None:
            hyper = data["shrinkage_hyperprior"]
            if not isinstance(hyper, dict) or set(hyper) != {
                f.name for f in fields(WFMMShrinkageHyperprior)
            }:
                raise ValueError("invalid shrinkage hyperprior fields")
            data["shrinkage_hyperprior"] = WFMMShrinkageHyperprior(**hyper)
        return WFMMStudy(**data)
    except (TypeError, OverflowError, RecursionError) as exc:
        raise ValueError("invalid WFMM study JSON") from exc


@dataclass(frozen=True)
class WFMMStudyResult:
    study: WFMMStudy
    basis: WFMMBasis
    compression: WFMMCompression
    initialization: WFMMInitialization
    fit: WFMMCoefficientFit | None
    summary: WFMMPosteriorSummary | None
    covariance: WFMMCovarianceSummary | None
    prediction: WFMMPosteriorSummary | None
    random_effects: WFMMPosteriorSummary | None = None

    def report_json(self) -> str:
        """Save summaries and diagnostics; null diagnostic values are explicitly unavailable."""
        data = dict(
            schema="mdanderson_stats.wfmm.report",
            version=1,
            study_sha256=hashlib.sha256(self.study.to_json().encode()).hexdigest(),
            status="held" if self.fit is None else "sampled",
            initializer_holds=self.initialization.holds,
            rng="NumPy PCG64; SeedSequence.spawn(2): fit, prediction",
            thinning=self.study.thin,
            retained_draws_per_chain=0 if self.fit is None else self.study.draws,
            iterations_per_chain=(
                None if self.fit is None else self.study.warmup + self.study.draws * self.study.thin
            ),
            seed=self.study.seed,
            numpy_version=np.__version__,
            retained_indices=self.compression.selection.retained_indices,
            compression_energy_fraction=self.compression.energy_fraction,
            posterior_precision="not_certified",
            clinical_validity="not_established",
            null_diagnostic_meaning="unavailable, including constant-chain R-hat",
        )
        if self.fit is not None:
            fit = self.fit
            variance = np.concatenate((fit.random_variances, fit.residual_variances), axis=2)
            variance_summary = summarize_chains(variance)
            data.update(
                coefficient_rhat=fit.summary.split_rhat,
                coefficient_mcse=fit.summary.batch_mean_mcse,
                variance_rhat=variance_summary.split_rhat,
                variance_mcse=variance_summary.batch_mean_mcse,
                variance_acceptance=fit.variance_acceptance,
                variance_proposals=fit.variance_proposals,
                likelihood_evaluations=fit.likelihood_evaluations,
            )
            if fit.inclusion_probabilities is not None and fit.slab_variances is not None:
                hyper_report: dict = dict(
                    partition_labels=fit.hyperparameter_partitions,
                    update_mask=fit.hyperparameter_update_mask,
                    prior_parameterization="explicit_beta_and_inverse_gamma_shape_scale",
                )
                for name, samples in (
                    ("pi", fit.inclusion_probabilities),
                    ("tau", fit.slab_variances),
                ):
                    stats = summarize_chains(samples)
                    hyper_report[name] = dict(
                        mean=np.mean(samples, axis=(0, 1)),
                        quantiles=np.quantile(
                            samples, np.asarray(self.study.quantiles, dtype=float), axis=(0, 1)
                        ),
                        quantile_probabilities=self.study.quantiles,
                        rhat=stats.split_rhat,
                        mcse=stats.batch_mean_mcse,
                    )
                data["shrinkage_hyperparameters"] = hyper_report
            for name, summary in (
                ("curves", self.summary),
                ("prediction", self.prediction),
                ("random_effects", self.random_effects),
            ):
                if summary is not None:
                    data[name] = dict(
                        mean=summary.mean,
                        sd=summary.standard_deviation,
                        quantiles=summary.quantiles,
                        quantile_probabilities=summary.quantile_probabilities,
                        simultaneous_lower=summary.simultaneous_lower,
                        simultaneous_upper=summary.simultaneous_upper,
                        simbas=summary.simbas_probability,
                        effect_sizes=summary.effect_sizes,
                        effect_size_probability=summary.effect_size_probability,
                        sign_tail_score=summary.sign_tail_score,
                        confidence=summary.confidence,
                        effect_contrast=summary.effect_contrast,
                        time_contrast=summary.time_contrast,
                    )
            if self.covariance is not None:
                data["variance_functions"] = dict(
                    mean=self.covariance.variance_function_mean,
                    sd=self.covariance.variance_function_standard_deviation,
                    quantiles=self.covariance.variance_function_quantiles,
                    quantile_probabilities=self.covariance.quantile_probabilities,
                    coordinate_system="original_time_grid",
                    covariance_mean=self.covariance.covariance_mean,
                    correlation_from_mean_variance=self.covariance.correlation_from_mean_variance,
                )
        text = json.dumps(_json_value(data), sort_keys=True, allow_nan=False) + "\n"
        if len(text.encode()) > _MAX_JSON_BYTES:
            raise ValueError("report exceeds 16 MB")
        return text

    def curve_csv(self) -> str:
        if self.summary is None:
            raise ValueError("held studies have no posterior curve report")
        output = io.StringIO(newline="")
        writer = csv.writer(output, lineterminator="\n")
        writer.writerow(
            (
                "effect",
                "region_index" if self.study.time_contrast is not None else "time_index",
                "mean",
                "sd",
                "simultaneous_lower",
                "simultaneous_upper",
            )
        )
        for effect, row in enumerate(self.summary.mean):
            for time, value in enumerate(row):
                writer.writerow(
                    (
                        effect,
                        time,
                        float(value),
                        float(self.summary.standard_deviation[effect, time]),
                        float(self.summary.simultaneous_lower[effect, time]),
                        float(self.summary.simultaneous_upper[effect, time]),
                    )
                )
        return output.getvalue()


def run_wfmm_study(study: WFMMStudy) -> WFMMStudyResult:
    """Transform, initialize, sample and summarize, stopping on initializer holds."""
    if not isinstance(study, WFMMStudy):
        raise ValueError("study must be a WFMMStudy")
    t = np.asarray(study.curves).shape[1]
    if study.transform == "identity":
        basis = wfmm_basis(t, transform="identity")
    elif study.transform == "spatial_wavelet":
        basis = _spatial_basis(study)
    elif study.transform == "custom":
        basis = wfmm_basis(
            t,
            transform="custom",
            analysis_matrix=study.analysis_matrix,
            synthesis_matrix=study.synthesis_matrix,
        )
    elif study.transform == "extended_wavelet" and study.wavelet_name is not None:
        basis = wfmm_named_wavelet_basis(
            t, wavelet=study.wavelet_name, levels=study.levels, boundary=study.boundary
        )
    elif study.transform == "extended_wavelet":
        basis = wfmm_extended_wavelet_basis(
            t, levels=study.levels, filter_length=study.filter_length, boundary=study.boundary
        )
    elif study.transform == "wavelet":
        basis = wfmm_basis(t, levels=study.levels, filter_length=study.filter_length)
    else:
        wave = (
            None
            if study.transform == "PC"
            else wfmm_basis(t, levels=study.levels, filter_length=study.filter_length)
        )
        basis = wfmm_pca_basis(
            study.curves,
            kind=study.transform,
            alpha=study.alpha_pc,
            partition_base=study.pc_partition_base,
            wavelet_basis=wave,
        ).basis
    transformed = wfmm_transform(study.curves, basis).coefficients
    eligible = None
    if study.transform in ("wavelet", "extended_wavelet") or (
        study.transform == "spatial_wavelet" and study.spatial_layout == "pyramid"
    ):
        eligible = wfmm_pass_filter(
            transformed, basis, highpass=study.highpass, lowpass=study.lowpass
        ).retained_indices
    compression = wfmm_compress_coefficients(
        transformed,
        basis,
        alpha=study.alpha_energy,
        t=study.compression_votes,
        tie_policy=study.compression_ties,
        eligible_indices=eligible,
    )
    selection = compression.selection
    initialization = initialize_wfmm(
        selection.coefficients,
        study.fixed_design,
        study.random_design,
        random_strata=study.random_strata,
        residual_strata=study.residual_strata,
        coefficient_partition=selection.coefficient_partition,
        min_variance=study.min_variance,
        delta_omega=study.delta_omega,
        proposal_multiplier=study.proposal_multiplier,
        moment_iterations=study.moment_iterations,
        optimizer_iterations=study.optimizer_iterations,
        empirical_bayes_iterations=study.empirical_bayes_iterations,
    )
    if not initialization.ready_for_sampling:
        return WFMMStudyResult(study, basis, compression, initialization, None, None, None, None)
    # Zero restoration and curve summaries have bounds separate from sampler K.
    restored_cells = (
        study.chains
        * study.draws
        * t
        * max(
            np.asarray(study.fixed_design).shape[1],
            0 if study.effect_contrast is None else np.asarray(study.effect_contrast).shape[0],
            0 if not study.summarize_random_effects else np.asarray(study.random_design).shape[1],
            initialization.conditional.proposal_sd.shape[0],
            0 if study.prediction_design is None else np.asarray(study.prediction_design).shape[0],
        )
    )
    if restored_cells > 2_000_000:
        raise ValueError("study restored draws exceed the 2000000-cell limit")
    state = initialization.conditional
    h = state.random_variance.shape[0]
    sampler_prior = initialization.prior
    if study.shrinkage_hyperprior is not None:
        labels, groups = np.unique(selection.coefficient_partition, return_inverse=True)
        a, b, shape, scale, mask = _hyperprior_arrays(
            study.shrinkage_hyperprior, (np.asarray(study.fixed_design).shape[1], len(labels))
        )
        # Explicit hyperpriors initialize active shared parameters at the Beta
        # mean and inverse-gamma mode; retain empirical-Bayes values when masked.
        pi = np.asarray(sampler_prior.inclusion_probability).copy()
        tau = np.asarray(sampler_prior.slab_variance).copy()
        pi = np.where(mask[:, groups], (a / (a + b))[:, groups], pi)
        tau = np.where(mask[:, groups], (scale / (shape + 1))[:, groups], tau)
        sampler_prior = replace(sampler_prior, inclusion_probability=pi, slab_variance=tau)
    seeds = np.random.SeedSequence(study.seed).spawn(2)
    fit = fit_wfmm_coefficients(
        selection.coefficients,
        study.fixed_design,
        study.random_design,
        prior=sampler_prior,
        random_variance=state.random_variance,
        residual_variance=state.residual_variance,
        random_strata=study.random_strata,
        residual_strata=study.residual_strata,
        coefficient_partition=selection.coefficient_partition,
        coefficient_scale=selection.coefficient_scale,
        proposal_sd=(state.proposal_sd[:h], state.proposal_sd[h:]),
        draws=study.draws,
        warmup=study.warmup,
        thin=study.thin,
        shrinkage_hyperprior=study.shrinkage_hyperprior,
        chains=study.chains,
        sample_random_effects=study.random_design is not None,
        rng=np.random.Generator(np.random.PCG64(seeds[0])),
    )
    summary = wfmm_summarize(
        wfmm_restore_coefficients(fit.coefficients, selection),
        basis,
        effect_contrast=study.effect_contrast,
        time_contrast=study.time_contrast,
        confidence=study.confidence,
        quantiles=study.quantiles,
        effect_sizes=study.effect_sizes,
    )
    random_summary = None
    if study.summarize_random_effects and fit.random_effects is not None:
        random_summary = wfmm_summarize(
            wfmm_restore_coefficients(fit.random_effects, selection),
            basis,
            time_contrast=study.time_contrast,
            confidence=study.confidence,
            quantiles=study.quantiles,
            effect_sizes=study.effect_sizes,
        )
    variances = np.concatenate((fit.random_variances, fit.residual_variances), axis=2)
    covariance = wfmm_summarize_covariance(
        wfmm_restore_coefficients(variances, selection),
        basis,
        quantiles=study.quantiles,
        include_covariance=study.include_covariance,
    )
    prediction = None
    if study.prediction_design is not None:
        predicted = wfmm_predict_coefficients(
            fit,
            study.prediction_design,
            existing_random_design=study.prediction_existing_random_design,
            new_random_design=study.prediction_new_random_design,
            new_random_strata=study.prediction_new_random_strata,
            include_residual=study.prediction_residual,
            residual_strata=study.prediction_strata,
            rng=np.random.Generator(np.random.PCG64(seeds[1])),
        )
        prediction = wfmm_summarize(
            wfmm_restore_coefficients(predicted, selection),
            basis,
            time_contrast=study.time_contrast,
            confidence=study.confidence,
            quantiles=study.quantiles,
            effect_sizes=study.effect_sizes,
        )
    return WFMMStudyResult(
        study,
        basis,
        compression,
        initialization,
        fit,
        summary,
        covariance,
        prediction,
        random_summary,
    )
