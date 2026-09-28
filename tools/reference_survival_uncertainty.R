# Independent pointwise survival summaries from supplied parameter draws.
# Input coordinates are original-unit intercept, two slopes, then log-scale
# (ordinary AFT), log-sigma/Q (Prentice), or log-shape/log-k (Stacy).
# This uses base-R distributions and type-7 quantiles, not the Python kernel.
# Gaussian sampling is validated separately; identical supplied draws avoid
# attributing random-stream differences to prediction/quantile calculations.
options(warn=2, digits=17)
args <- commandArgs(trailingOnly=TRUE)
if (length(args) != 2L) stop("supply input-draw CSV and output-summary CSV")
draws <- read.csv(args[1], stringsAsFactors=FALSE, check.names=FALSE)
required <- c("case", "distribution", "draw", "beta0", "beta1", "beta2",
              "log_sigma", "Q", "log_shape", "log_k")
stopifnot(all(required %in% names(draws)))
profiles <- rbind(c(-.7,.2), c(.4,-.3), c(1.2,.7))
times <- c(0,.01,.2,.7,1,2,4,10,100,1e8)

survival <- function(parameters, profile, time) {
  if (time == 0) return(rep(1, nrow(parameters)))
  mu <- parameters$beta0 + profile[1]*parameters$beta1 + profile[2]*parameters$beta2
  family <- unique(parameters$distribution)
  stopifnot(length(family)==1L)
  if (family == "weibull") {
    pweibull(time, shape=exp(-parameters$log_sigma), scale=exp(mu), lower.tail=FALSE)
  } else if (family == "lognormal") {
    plnorm(time, meanlog=mu, sdlog=exp(parameters$log_sigma), lower.tail=FALSE)
  } else if (family == "loglogistic") {
    plogis(log(time), location=mu, scale=exp(parameters$log_sigma), lower.tail=FALSE)
  } else if (family == "prentice") {
    q <- parameters$Q
    z <- (log(time)-mu)/exp(parameters$log_sigma)
    answer <- numeric(length(q))
    normal <- q == 0
    answer[normal] <- pnorm(z[normal], lower.tail=FALSE)
    positive <- q > 0
    negative <- q < 0
    # Keep the independent gamma identity away from its cancellation-prone
    # normal limit; that limit has dedicated native-kernel checks elsewhere.
    stopifnot(all(abs(q[!normal]) >= 0.005))
    for (selection in list(positive,negative)) {
      if (any(selection)) {
        a <- 1/q[selection]^2
        argument <- exp(q[selection]*z[selection]) * a
        answer[selection] <- pgamma(argument, shape=a,
                                   lower.tail=all(q[selection]<0))
      }
    }
    answer
  } else if (family == "stacy") {
    argument <- exp(exp(parameters$log_shape)*(log(time)-mu))
    pgamma(argument, shape=exp(parameters$log_k), lower.tail=FALSE)
  } else stop(paste("unsupported reference distribution", family))
}

result <- list()
for (case in unique(draws$case)) {
  parameters <- draws[draws$case == case,]
  stopifnot(nrow(parameters)>1L, !anyDuplicated(parameters$draw))
  for (profile in seq_len(nrow(profiles))) for (time in times) {
    values <- survival(parameters, profiles[profile,], time)
    stopifnot(all(is.finite(values)), all(values >= 0 & values <= 1))
    limits <- quantile(values, c(.025,.975), type=7, names=FALSE)
    result[[length(result)+1L]] <- data.frame(
      case=case, profile=profile, time=time, draws=length(values),
      mean=mean(values), sd=sd(values), lower=limits[1], upper=limits[2])
  }
}
write.csv(do.call(rbind,result), args[2], row.names=FALSE)
cat("Wrote", length(result), "pointwise reference rows from",
    length(unique(draws$case)), "parameter-draw cases.\n")
