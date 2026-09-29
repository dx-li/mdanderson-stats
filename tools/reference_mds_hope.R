# Independent arithmetic fixtures for the published MDS-HOPE Eq. S1.
# Run with: Rscript tools/reference_mds_hope.R tests/fixtures/mds-hope
# All profiles, including the ordinal cytogenetic-score value, are synthetic.
# The synthetic score values do not assert a source-derived cytogenetic mapping.
# z_mean and z_sd below are arbitrary test constants, NOT recovered calibration.

out_dir <- if (length(commandArgs(trailingOnly = TRUE))) {
  commandArgs(trailingOnly = TRUE)[[1L]]
} else {
  "."
}
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

# Preserve adjacent floating-point cutoff cases during CSV round trips.
write_exact_csv <- function(data, path, row.names = FALSE) {
  data[] <- lapply(data, function(x) {
    if (is.numeric(x)) {
      out <- sprintf("%.17g", x)
      out[is.na(x) & !is.nan(x)] <- NA_character_
      out
    } else x
  })
  write.csv(data, path, row.names = row.names)
}

# Inputs are supplied in original clinical units. Eq. S1 defines transformed
# anemia=-Hb, thrombocytopenia=-platelets/10, neutropenia=-ANC; after substitution
# the original-unit terms are +.067*ANC, -.160*Hb, and -.0021*platelets.
coeff <- c(
  age_years = 0.025,
  anc_10e9_l = 0.067,
  hemoglobin_g_dl = -0.160,
  platelets_10e9_l = -0.0021,
  marrow_blast_percent = 0.051,
  cytogenetic_score_synthetic_ordinal = 0.284,
  SF3B1 = -0.294,
  EZH2 = 0.347,
  TP53 = 0.345,
  KRAS = 0.681,
  PTPN11 = 1.094
)

eta <- function(d) {
  stopifnot(all(names(coeff) %in% names(d)))
  as.numeric(as.matrix(d[, names(coeff), drop = FALSE]) %*% coeff)
}

# Synthetic, clinically plausible reference values; genes absent/wild-type.
# Cytogenetic score is an unlabelled explicit ordinal numeric test input.
base <- data.frame(
  age_years = 70, anc_10e9_l = 1.13, hemoglobin_g_dl = 9.1,
  platelets_10e9_l = 79, marrow_blast_percent = 7,
  cytogenetic_score_synthetic_ordinal = 3,
  SF3B1 = 0, EZH2 = 0, TP53 = 0, KRAS = 0, PTPN11 = 0
)
base_eta <- eta(base)

# One-predictor contrasts, each differing from the reference in only one field.
# Numeric increments are chosen for transparent interpretation, not source claims.
contrast_field <- c(
  age_years = "age_years", anc_10e9_l = "anc_10e9_l",
  hemoglobin_g_dl = "hemoglobin_g_dl", platelets_10e9_l = "platelets_10e9_l",
  marrow_blast_percent = "marrow_blast_percent",
  cytogenetic_score_synthetic_ordinal = "cytogenetic_score_synthetic_ordinal",
  SF3B1 = "SF3B1", EZH2 = "EZH2", TP53 = "TP53", KRAS = "KRAS", PTPN11 = "PTPN11"
)
contrast_to <- c(
  age_years = 80, anc_10e9_l = 2.13, hemoglobin_g_dl = 8.1,
  platelets_10e9_l = 69, marrow_blast_percent = 8,
  cytogenetic_score_synthetic_ordinal = 4,
  SF3B1 = 1, EZH2 = 1, TP53 = 1, KRAS = 1, PTPN11 = 1
)
one_predictor <- do.call(rbind, lapply(names(contrast_field), function(nm) {
  field <- contrast_field[[nm]]
  x <- base
  x[[field]] <- contrast_to[[nm]]
  e <- eta(x)
  data.frame(
    predictor = nm,
    reference_value = base[[field]],
    contrast_value = x[[field]],
    reference_eta = base_eta,
    contrast_eta = e,
    delta_log_hazard_ratio = e - base_eta,
    hazard_ratio_vs_reference = exp(e - base_eta),
    stringsAsFactors = FALSE
  )
}))
write_exact_csv(one_predictor, file.path(out_dir, "mds_hope_one_predictor_contrasts.csv"), row.names = FALSE)

# Explicit gene-only profiles relative to the same synthetic baseline.
gene_profiles <- data.frame(
  profile = c("reference_no_mutations", "SF3B1_only", "EZH2_only", "TP53_single_hit",
              "TP53_multi_hit", "KRAS_only", "PTPN11_only", "combined_adverse_genes"),
  SF3B1 = c(0, 1, 0, 0, 0, 0, 0, 0),
  EZH2 = c(0, 0, 1, 0, 0, 0, 0, 1),
  TP53 = c(0, 0, 0, 1, 2, 0, 0, 2),
  KRAS = c(0, 0, 0, 0, 0, 1, 0, 1),
  PTPN11 = c(0, 0, 0, 0, 0, 0, 1, 1),
  stringsAsFactors = FALSE
)
profile_data <- base[rep(1L, nrow(gene_profiles)), , drop = FALSE]
for (g in c("SF3B1", "EZH2", "TP53", "KRAS", "PTPN11")) profile_data[[g]] <- gene_profiles[[g]]
profile_data$eta <- eta(profile_data)
gene_profiles$eta <- profile_data$eta
gene_profiles$delta_log_hazard_ratio <- gene_profiles$eta - base_eta
gene_profiles$hazard_ratio_vs_reference <- exp(gene_profiles$delta_log_hazard_ratio)
write_exact_csv(gene_profiles, file.path(out_dir, "mds_hope_gene_profiles.csv"), row.names = FALSE)

# Explicit contrasts against the reference profile: each reported logHR is
# eta(profile)-eta(reference), HR=exp(logHR). No baseline survival is implied.
reference_profiles <- data.frame(
  profile = c("reference_no_mutations", "SF3B1_only", "TP53_multi_hit", "KRAS_and_PTPN11"),
  SF3B1 = c(0, 1, 0, 0), EZH2 = c(0, 0, 0, 0),
  TP53 = c(0, 0, 2, 0), KRAS = c(0, 0, 0, 1), PTPN11 = c(0, 0, 0, 1)
)
ref_data <- base[rep(1L, nrow(reference_profiles)), , drop = FALSE]
for (g in c("SF3B1", "EZH2", "TP53", "KRAS", "PTPN11")) ref_data[[g]] <- reference_profiles[[g]]
reference_profiles$eta <- eta(ref_data)
reference_profiles$delta_log_hazard_ratio_vs_reference <- reference_profiles$eta - reference_profiles$eta[1L]
reference_profiles$hazard_ratio_vs_reference <- exp(reference_profiles$delta_log_hazard_ratio_vs_reference)
write_exact_csv(reference_profiles, file.path(out_dir, "mds_hope_reference_profile_hazard_ratios.csv"), row.names = FALSE)

# Boundary fixtures for standardized score classification only. Constants below
# are arbitrary test constants and must never be represented as the study mean/SD.
z_mean <- 0.25
z_sd <- 2.0
cuts <- c(-1.5, -0.5, 0, 0.5, 1.5)
# Each cut appears below, exactly at, and above the cutoff. Small multiples of
# machine epsilon make distinct finite doubles without relying on extra packages.
cut_cases <- do.call(rbind, lapply(cuts, function(x) {
  delta <- 2 * .Machine$double.eps * max(1, abs(x))
  data.frame(
    threshold = x,
    boundary_position = c("just_below", "exact_cut", "just_above"),
    z = c(x - delta, x, x + delta),
    stringsAsFactors = FALSE
  )
}))
z_values <- c(-2, cut_cases$z, 2)
boundary_position <- c("finite_below_min", cut_cases$boundary_position, "finite_above_max")
threshold_label <- c(NA_real_, cut_cases$threshold, NA_real_)
classify <- function(z) {
  ifelse(z <= -1.5, "very_low",
  ifelse(z <= -0.5, "low",
  ifelse(z <= 0, "intermediate_low",
  ifelse(z <= 0.5, "intermediate_high",
  ifelse(z <= 1.5, "high", "very_high")))))
}
boundary <- data.frame(
  case = sprintf("z_case_%02d", seq_along(z_values)),
  standardized_score_test_value = z_values,
  boundary_position = boundary_position,
  threshold = threshold_label,
  expected_six_group = classify(z_values),
  arbitrary_test_mean = z_mean,
  arbitrary_test_sd = z_sd,
  corresponding_synthetic_raw_eta = z_mean + z_sd * z_values,
  stringsAsFactors = FALSE
)
write_exact_csv(boundary, file.path(out_dir, "mds_hope_six_group_boundaries.csv"), row.names = FALSE)

# The published five-group alternative collapses intermediate-high and high
# into intermediate. Retain the six-group fixture's .5 neighbors to check that
# .5 does not create a boundary in this alternative.
classify_five <- function(z) {
  ifelse(z <= -1.5, "very_low",
  ifelse(z <= -0.5, "low",
  ifelse(z <= 0, "intermediate_low",
  ifelse(z <= 1.5, "intermediate", "very_high"))))
}
five_boundary <- boundary
five_boundary$expected_six_group <- NULL
five_boundary$expected_five_group <- classify_five(z_values)
write_exact_csv(five_boundary, file.path(out_dir, "mds_hope_five_group_boundaries.csv"), row.names = FALSE)

# Non-finite standardized values are input-validation cases, not risk groups.
nonfinite <- data.frame(
  case = c("negative_infinity", "positive_infinity", "not_a_number"),
  standardized_score_test_value = c(-Inf, Inf, NaN),
  expected = "reject_nonfinite_score",
  stringsAsFactors = FALSE
)
write_exact_csv(nonfinite, file.path(out_dir, "mds_hope_nonfinite_score_rejections.csv"), row.names = FALSE)
