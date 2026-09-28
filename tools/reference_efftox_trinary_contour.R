# Independent base-R reference for the 2006 EffTox trinary Lp contour.
# Solve the eliminated intercept equation on log(p), excluding its spurious
# p=0 root; no Python code or native EffTox executable is used.
options(digits = 17)

known_shape <- function(name, e0, p, toxicity_intercept) {
  a <- 1 - e0
  th <- (a^(-p) + toxicity_intercept^(-p))^(-1 / p)
  eh <- 1 - th
  em <- (e0 + eh) / 2
  tm <- toxicity_intercept * (1 - ((1 - em) / a)^p)^(1 / p)
  list(name = name, e0 = e0, em = em, tm = tm, eh = eh, th = th,
       known_p = p, known_t = toxicity_intercept)
}

cases <- list(
  list(name = "stroke_targets", e0 = .45, em = .55, tm = .10,
       eh = .84, th = .16, known_p = NA_real_, known_t = NA_real_),
  known_shape("linear", .4, 1, .3),
  known_shape("elliptical", .4, 2, .4),
  known_shape("concave_intercept_above_one", .4, .5, 2)
)

rows <- lapply(cases, function(case) {
  a <- 1 - case$e0
  b <- 1 - case$em
  c <- case$th
  d <- case$tm
  aa <- -log(b / a)
  cc <- -log(c / a)
  dd <- -log(d / c)
  root <- function(log_p) {
    p <- exp(log_p)
    log(-expm1(-aa * p)) - log(-expm1(-cc * p)) + dd * p
  }
  p <- exp(uniroot(root, c(-20, 20), tol = 1e-13)$root)
  # Compute the intercept from the middle point, independent of the
  # hypotenuse-point expression used to derive the scalar root equation.
  t_star <- d / (-expm1(-aa * p))^(1 / p)
  if (is.finite(case$known_p)) {
    stopifnot(abs(p - case$known_p) < 1e-11,
              abs(t_star - case$known_t) < 1e-11)
  }
  e <- c(case$e0, case$em, case$eh, 1, 0, 0, .25, .5, .75,
         (1 + case$em) / 2, .9, .1)
  t <- c(0, case$tm, case$th, 0, 0, 1, .25, .5, .2,
         case$tm / 2, .1, .8)
  utility <- 1 - (((1 - e) / a)^p + (t / t_star)^p)^(1 / p)
  stopifnot(max(abs(utility[1:3])) < 1e-12,
            abs(utility[4] - 1) < 1e-14,
            abs(utility[10] - .5) < 1e-12,
            all(e + t <= 1 + 1e-14))
  data.frame(case = case$name, e0 = case$e0, em = case$em,
             tm = case$tm, eh = case$eh, th = case$th,
             shape = p, toxicity_intercept = t_star,
             efficacy = e, toxicity = t, utility = utility)
})
write.csv(do.call(rbind, rows), "tests/fixtures/efftox-trinary-contour.csv",
          row.names = FALSE)
cat("Wrote 48 trinary contour values, including an intercept above one.\n")
