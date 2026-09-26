# Independent enumeration of all four two-patient CRM outcome histories.
# Base R quadrature only; no Python calls or original software code.
# Run from the repository root with Rscript tools/reference_crm_trial.R.
# Safety is disabled. At this sample size every possible final dose has fewer
# than three subjects and no lower fallback qualifies, so retain the candidate.
options(digits=17)
skeleton <- c(.1,.3)
target <- .25
posterior_mean <- function(n,y) {
  density <- function(a) vapply(a,function(x) {
    q <- skeleton^exp(x)
    prod(dbinom(y,n,q)) * dnorm(x,sd=sqrt(2))
  },numeric(1))
  evidence <- integrate(density,-Inf,Inf,rel.tol=1e-10,abs.tol=1e-13)$value
  vapply(seq_along(skeleton),function(j) {
    integrate(function(a) skeleton[j]^exp(a)*density(a),-Inf,Inf,
              rel.tol=1e-10,abs.tol=1e-13)$value/evidence
  },numeric(1))
}
rows <- list()
for (first in 0:1) for (second in 0:1) {
  n <- c(1,0); y <- c(first,0)
  means <- posterior_mean(n,y)
  next_dose <- which.min(abs(means-target))
  if (next_dose>1 && y[1]/n[1]>target) next_dose <- 1
  n[next_dose] <- n[next_dose]+1
  y[next_dose] <- y[next_dose]+second
  means <- posterior_mean(n,y)
  selected <- which.min(abs(means-target))
  rows[[length(rows)+1]] <- data.frame(first=first,second=second,
    next_dose=next_dose-1,selected_dose=selected-1,
    n0=n[1],n1=n[2],y0=y[1],y1=y[2],mean0=means[1],mean1=means[2])
}
write.csv(do.call(rbind,rows),'tests/fixtures/crm-trial-reference.csv',row.names=FALSE)
