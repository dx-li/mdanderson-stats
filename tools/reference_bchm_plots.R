# Independent BCHM display references: unchanged boa.hpd and R 4.4.1 density.
# No BCHM/JAGS package installation or posterior sampling is required.
source_path <- "research/raw/BCHM/BCHM_src/R/utils.R"
expressions <- parse(source_path)
is_hpd <- vapply(expressions, function(expr) {
    is.call(expr) && identical(expr[[1L]], as.name("<-")) &&
        identical(expr[[2L]], as.name("boa.hpd"))
}, logical(1L))
stopifnot(sum(is_hpd) == 1L)
native <- new.env(parent = baseenv())
eval(expressions[[which(is_hpd)]], envir = native)

cases <- list(
    ordinary = c(.02, .05, .11, .16, .22, .29, .35, .45, .61, .8),
    tied = c(.01, .01, .01, .1, .1, .2, .2, .2, .9, .95),
    short = c(.2, .8),
    zero = rep(0, 10),
    constant = rep(.4, 10),
    one = rep(1, 10)
)
inputs <- intervals <- curves <- list()
for (name in names(cases)) {
    samples <- cases[[name]]
    inputs[[name]] <- data.frame(case = name, index = seq_along(samples), sample = samples)
    intervals[[name]] <- do.call(rbind, lapply(c(.5, .8, .95), function(level) {
        hpd <- native$boa.hpd(samples, 1 - level)
        data.frame(case = name, level = level, lower = hpd$lower, upper = hpd$upper)
    }))
    value <- density(samples, kernel = "gaussian", bw = "nrd0", n = 512,
                     cut = 3, old.coords = FALSE)
    # The exact Gaussian mixture evaluates the same kernel/bandwidth without
    # R's FFT binning/interpolation; both are retained to expose that difference.
    direct <- vapply(value$x, function(at) mean(dnorm(at, samples, value$bw)), 0.0)
    curves[[name]] <- data.frame(case = name, x = value$x, bandwidth = value$bw,
                                native_fft = value$y, direct_gaussian = direct)
}
write.csv(do.call(rbind, inputs), "tests/fixtures/bchm-plot-input.csv", row.names = FALSE)
write.csv(do.call(rbind, intervals), "tests/fixtures/bchm-plot-hpd.csv", row.names = FALSE)
write.csv(do.call(rbind, curves), "tests/fixtures/bchm-plot-density.csv", row.names = FALSE)
cat(R.version.string, "\n", length(cases), "cases, 18 HPD intervals, 3072 density points\n")
