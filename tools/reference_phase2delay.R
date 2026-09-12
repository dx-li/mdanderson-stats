# Independent, deterministic references for the delayed binary MI posterior.
# Gamma priors use rates: E(lambda_j | lambda_previous) = lambda_previous.
# These integrals do not simulate a Markov chain or imputed observations.

# One interval: one event at .2, one complete non-event at 1, two pending
# non-events at .25 and .75. Prior Gamma(2, rate 2/.4), posterior Gamma(3,7.2).
laplace <- function(t) (7.2/(7.2+t))^3
weights1 <- c(laplace(1), laplace(.25)+laplace(.75)-2*laplace(1),
              1-laplace(.25)-laplace(.75)+laplace(1))
stopifnot(all(weights1 >= 0), abs(sum(weights1)-1)<1e-14)

# Two intervals [0,.5), [.5,1]: event .25, complete non-event 1,
# pending non-event .5. Counts (1,0), exposures (1.25,.5).
# Prior c=(2,3), lambda0=.4. Integrate lambda2 out analytically, then
# integrate lambda1 numerically. Conditional lambda2 ~ Gamma(3,.5+3/lambda1).
kernel <- function(x) x^2*exp(-6.25*x)*(3/(3+.5*x))^3
integral <- function(f) integrate(f,0,Inf,rel.tol=1e-11,abs.tol=1e-13)$value
normalizer <- integral(kernel)
pending_no_event <- integral(function(x)kernel(x)*((3+.5*x)/(3+x))^3)/normalizer
weights2 <- c(pending_no_event,1-pending_no_event)
mean1 <- integral(function(x)x*kernel(x))/normalizer
mean2 <- integral(function(x)3*x/(3+.5*x)*kernel(x))/normalizer
rows <- list()
for (case in 1:2) for (endpoint in c('response','toxicity','progression')) {
  n <- if(case==1)4 else 3
  weights <- if(case==1)weights1 else weights2
  k <- seq_along(weights)-1
  tails <- pbeta(.4,.3+1+k,.7+(n-1-k),lower.tail=(endpoint=='response'))
  rows[[length(rows)+1]] <- data.frame(case=case,endpoint=endpoint,
    probability=sum(weights*tails),
    pending_event_mean=if(case==1)sum(k*weights)/2 else weights[2],
    hazard_mean_1=if(case==1)3/7.2 else mean1,
    hazard_mean_2=if(case==1)NA else mean2)
}
write.csv(do.call(rbind,rows),'tests/fixtures/phase2delay-reference.csv',row.names=FALSE)
