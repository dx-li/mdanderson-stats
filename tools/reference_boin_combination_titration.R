#!/usr/bin/env Rscript

# Execute the original BOIN 2.7.2 accelerated-titration and first-cohort
# expressions, with deterministic outcome/direction draws. No packages needed.
# Cached upstream code is a read-only reference and is not redistributed.
# Run from the repository root; the native cache is research/raw/BOINComb.
# Reference file SHA-256:
# 0bcd01f550cece9b7caae5b6070c259475cd6298f46d8d0d5b06f5b41a610814

source("research/raw/BOINComb/R/get.oc.comb.R")

statement <- function(expressions, predicate) {
  found <- Filter(predicate, as.list(expressions)[-1L])
  stopifnot(length(found) == 1L)
  found[[1L]]
}
is_head <- function(x, head) is.call(x) && identical(x[[1L]], as.name(head))
inner_assignment <- statement(body(get.oc.comb), function(x) {
  is_head(x, "<-") && identical(x[[2L]], as.name("get.oc.comb.boin"))
})
inner <- eval(inner_assignment[[3L]])
trial <- statement(body(inner), function(x) {
  is_head(x, "for") && identical(x[[2L]], as.name("trial"))
})[[4L]]
titration_expression <- statement(trial, function(x) {
  is_head(x, "if") && identical(x[[2L]], as.name("titration"))
})
cohort_loop <- statement(trial, function(x) {
  is_head(x, "for") && identical(x[[2L]], as.name("pp"))
})
first_cohort_expression <- as.list(cohort_loop[[4L]])[[2L]]
stopifnot(is_head(first_cohort_expression, "if"))

encode <- function(x) paste(as.integer(t(x)), collapse = ";")
reference <- function(name, shape, start = c(1L, 1L), directions = 0L,
                      toxic_cell = NULL, cohort = 3L) {
  e <- new.env(parent = baseenv())
  e$p.true <- matrix(0, shape[1L], shape[2L])
  if (!is.null(toxic_cell)) e$p.true[matrix(toxic_cell, nrow = 1L)] <- 1
  e$n <- e$y <- matrix(0, shape[1L], shape[2L])
  e$d <- start
  e$titration <- cohort > 1L
  e$cohortsize <- cohort
  e$ft <- TRUE
  position <- 0L
  e$runif <- function(n) rep(0.5, n)
  e$rbinom <- function(n, size, prob) {
    stopifnot(n == 1L, size == 1L, all(prob == 0.5))
    position <<- position + 1L
    directions[(position - 1L) %% length(directions) + 1L]
  }
  eval(titration_expression, e)
  titration_n <- e$n
  titration_y <- e$y
  endpoint <- e$d
  eval(first_cohort_expression, e)
  data.frame(case = name, rows = shape[1L], columns = shape[2L],
             start_a = start[1L], start_b = start[2L], cohort_size = cohort,
             directions = paste(directions, collapse = ";"),
             true_toxicity = encode(e$p.true),
             titration_patients = encode(titration_n),
             titration_toxicities = encode(titration_y),
             endpoint_a = endpoint[1L], endpoint_b = endpoint[2L],
             first_cohort_patients = encode(e$n),
             first_cohort_toxicities = encode(e$y),
             stringsAsFactors = FALSE)
}

cases <- list(
  reference("no_dlt_row_first", c(2L, 3L)),
  reference("no_dlt_column_first", c(2L, 3L), directions = 1L),
  reference("dlt_at_start", c(2L, 3L), toxic_cell = c(1L, 1L)),
  reference("dlt_interior", c(3L, 3L), directions = c(0L, 1L),
            toxic_cell = c(2L, 2L)),
  reference("forced_upper_edge", c(2L, 3L), start = c(2L, 1L)),
  reference("forced_right_edge", c(2L, 3L), start = c(1L, 3L)),
  reference("start_at_upper_right", c(2L, 3L), start = c(2L, 3L)),
  reference("upper_right_dlt", c(2L, 3L), toxic_cell = c(2L, 3L)),
  reference("single_patient_disables_titration", c(2L, 3L), cohort = 1L)
)
write.csv(do.call(rbind, cases),
          "tests/fixtures/boin-combination-titration.csv", row.names = FALSE)
