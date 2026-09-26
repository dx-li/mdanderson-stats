# Independent base-R references for the ToxFinder six-parameter model.
# Run from the repository root. No third-party R packages are required.
options(digits=17)

# Official case-study guide pp12-13 and simulation guide pp20-21.
scenarios <- list(
  standard=c(.4,8,.4,8,1,.1), concave=c(.4,8,.4,8,1,.04),
  convex=c(.4,8,.4,8,1,.15), skewed=c(.4,8,.45,1,7.5,.375),
  additive=c(1,4,1,8,0,1), interaction=c(0,1,0,1,1,.5))
doses <- rbind(c(0,0),c(.5,0),c(0,.5),c(.12,.12),c(.25,.25),
               c(.4,.4),c(.52,.52),c(.8,.3),c(1,1),c(1.5,.8))
prob <- function(x,t) {
  q <- t[1]*x[1]^t[2]+t[3]*x[2]^t[4]+
    t[5]*(x[1]^t[2]*x[2]^t[4])^t[6]
  q/(1+q)
}
rows <- list()
for(name in names(scenarios)) for(i in seq_len(nrow(doses))) {
  t <- scenarios[[name]]; x <- doses[i,]
  rows[[length(rows)+1]] <- data.frame(scenario=name,x1=x[1],x2=x[2],
    alpha1=t[1],beta1=t[2],alpha2=t[3],beta2=t[4],alpha3=t[5],beta3=t[6],
    probability=prob(x,t))
}
write.csv(do.call(rbind,rows),'tests/fixtures/toxfinder-surfaces.csv',row.names=FALSE)

# Nonconjugate scalar posterior: alpha1~Gamma(shape=2,scale=.25),
# beta1=2 and (alpha2,beta2,alpha3,beta3)=(.4,3,1,.5) fixed.
# Historical single-agent observations at x1=(.5,1,1.5), x2=0,
# y=(0,1,3), n=(2,4,4). Integrate over log(alpha1), including Jacobian.
x <- c(.5,1,1.5); y <- c(0,1,3); n <- c(2,4,4)
log1pexp <- function(v) pmax(v,0)+log1p(exp(-abs(v)))
logdensity <- function(z) {
  vapply(z,function(zi) {
    if(zi>700) return(-Inf)
    eta <- zi+2*log(x)
    2*zi-exp(zi)/.25-sum(y*log1pexp(-eta)+(n-y)*log1pexp(eta))
  },numeric(1))
}
mode <- optimize(function(z) -logdensity(z),c(-15,5))$minimum
shift <- logdensity(mode)
weight <- function(z) exp(logdensity(z)-shift)
integral <- function(f) integrate(function(z) weight(z)*f(z),-Inf,Inf,
  rel.tol=1e-10,abs.tol=1e-12,subdivisions=300)$value
normalizer <- integral(function(z) rep(1,length(z)))
# exp(z) factors are combined in log space before integration to avoid 0*Inf.
moment <- function(k) integrate(function(z) exp(logdensity(z)-shift+k*z),
  -Inf,Inf,rel.tol=1e-10,abs.tol=1e-12,subdivisions=300)$value/normalizer
mean <- moment(1); second <- moment(2)
rows <- data.frame(quantity=c('alpha1_mean','alpha1_sd','log_alpha1_mean','log_alpha1_sd'),
  value=c(mean,sqrt(second-mean^2),integral(function(z) z)/normalizer,
    sqrt(integral(function(z) z^2)/normalizer-(integral(function(z) z)/normalizer)^2)))
write.csv(rows,'tests/fixtures/toxfinder-posterior.csv',row.names=FALSE)

# Prior with beta3 mean .05, variance3 in the paper: log draws remain finite
# even when exponentiation underflows. These are analytic log-gamma moments.
shape <- .05^2/3; scale <- 3/.05
write.csv(data.frame(mean=digamma(shape)+log(scale),sd=sqrt(trigamma(shape))),
  'tests/fixtures/toxfinder-log-prior.csv',row.names=FALSE)
