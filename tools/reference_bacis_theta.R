# Independent half-normal posterior references for BaCIS model1 theta.
# Component probabilities come from the existing base-R classification audit.
options(digits=17)
classification <- read.csv('tests/fixtures/bacis-classification.csv')
z <- c(-Inf, -8, -2, -1e-12, 0, 1e-12, 2, 8, Inf)
# Direct integration avoids subtracting nearly equal normal CDFs near zero.
middle <- vapply(abs(z), function(x) {
  if(is.infinite(x)) return(1)
  if(x == 0) return(0)
  2*integrate(dnorm, 0, x, abs.tol=0, rel.tol=2e-12)$value
}, 0.)
curves <- list()
moments <- list()
for(i in seq_len(nrow(classification))) for(precision in c(.001, 1, 25)) {
  x <- classification[i,]
  wlo <- x$p_low
  whi <- x$p_high
  scale <- 1/sqrt(precision)
  theta <- z*scale
  density <- 2*ifelse(z < 0, wlo, whi)*dnorm(z)/scale
  cdf <- ifelse(z < 0, 2*wlo*pnorm(z), wlo+whi*middle)
  survival <- ifelse(z < 0, whi+wlo*middle,
                    2*whi*pnorm(z, lower.tail=FALSE))
  curves[[length(curves)+1]] <- data.frame(
    case=x$case, group=x$group, latent_precision=precision,
    standardized_theta=z, theta=theta, low_probability=wlo,
    high_probability=whi, density=density, cdf=cdf, survival=survival)
  mean <- (whi-wlo)*scale*sqrt(2/pi)
  variance <- (1-(2/pi)*(whi-wlo)^2)/precision
  moments[[length(moments)+1]] <- data.frame(
    case=x$case, group=x$group, latent_precision=precision,
    low_probability=wlo, high_probability=whi, mean=mean,
    variance=variance, second_moment=1/precision)
}
curves <- do.call(rbind, curves)
moments <- do.call(rbind, moments)
stopifnot(max(abs(curves$cdf+curves$survival-1)) < 1e-13,
          all(curves$density >= 0), all(moments$variance > 0))
write.csv(curves, 'tests/fixtures/bacis-theta-curves.csv', row.names=FALSE)
write.csv(moments, 'tests/fixtures/bacis-theta-moments.csv', row.names=FALSE)
cat(nrow(curves), 'posterior curve rows and', nrow(moments), 'moment rows generated.\n')
