# Independent base-R references for the documented single-outcome bCRM model.
# Run from the repository root; no R packages or original program code needed.
# Skeleton from bCRM Test Results V1.R3.M0 (Goodman example). Counts below are
# deliberately chosen audit cases, not claimed native simulation output.
options(digits=17)
cases <- list(
  prior=list(s=c(.05,.10,.20,.35,.50,.70),a=3,L=0,U=1,
             y=rep(0,6),n=rep(0,6),bounds=c(0,3)),
  goodman=list(s=c(.05,.10,.20,.35,.50,.70),a=3,L=0,U=1,
               y=c(0,0,1,2,0,0),n=c(2,4,6,4,0,0),bounds=c(0,3)),
  bounded=list(s=c(.12,.28,.62),a=-3,L=.1,U=.8,
               y=c(1,2,3),n=c(4,6,4),bounds=c(.25,2.75)),
  concentrated=list(s=c(.05,.10,.20),a=3,L=0,U=1,
                    y=c(0,0,2000),n=c(0,0,10000),bounds=c(0,3)),
  boundary=list(s=c(.05,.10,.20),a=3,L=0,U=1,
                y=c(10000,0,0),n=c(10000,0,0),bounds=c(0,3)))
rows <- list()
for(name in names(cases)) {
  c <- cases[[name]]
  x <- log(c$s-c$L)-log(c$U-c$s)-c$a
  prob <- function(b,j) c$L+(c$U-c$L)*plogis(c$a+b*x[j])
  ll <- function(b) vapply(b,function(v) {
    p <- c$L+(c$U-c$L)*plogis(c$a+v*x)
    sum(dbinom(c$y,c$n,p,log=TRUE)-lchoose(c$n,c$y))
  },numeric(1))
  lo <- c$bounds[1]; hi <- c$bounds[2]
  mode <- optimize(function(b) -ll(b),c(lo,hi),tol=1e-12)$minimum
  shift <- max(ll(c(lo,mode,hi)))
  weight <- function(b) exp(ll(b)-shift)
  # Explicit mode and fine endpoint splits protect the boundary audit case.
  breaks <- sort(unique(c(lo,hi,mode,lo+(hi-lo)*c(1e-6,1e-4,.01,.1,.5,.9))))
  integral <- function(f,upper=hi) {
    cuts <- sort(unique(c(breaks[breaks<upper],upper)))
    sum(vapply(seq_len(length(cuts)-1),function(i)
      integrate(function(b) weight(b)*f(b),cuts[i],cuts[i+1],
                rel.tol=1e-10,abs.tol=1e-14,subdivisions=200)$value,numeric(1)))
  }
  z <- integral(function(b) rep(1,length(b)))
  mean <- integral(function(b) b)/z
  sd <- sqrt(integral(function(b) (b-mean)^2)/z)
  ci <- vapply(c(.025,.975),function(p)
    uniroot(function(b) integral(function(v) rep(1,length(v)),b)/z-p,
            c(lo,hi),tol=1e-12)$root,numeric(1))
  for(j in seq_along(x)) {
    pi <- sort(prob(ci,j))
    rows[[length(rows)+1]] <- data.frame(
      scenario=name,dose=j-1,skeleton=c$s[j],alpha=c$a,lower=c$L,upper=c$U,
      events=c$y[j],subjects=c$n[j],prior_lower=lo,prior_upper=hi,
      x=x[j],beta_mean=mean,beta_sd=sd,beta_low=ci[1],beta_high=ci[2],
      dose_mean=integral(function(b) prob(b,j))/z,
      dose_low=pi[1],dose_high=pi[2],plugin_probability=prob(mean,j),
      log_evidence=shift+log(z/(hi-lo)))
  }
}
write.csv(do.call(rbind,rows),'tests/fixtures/bcrm-posterior.csv',row.names=FALSE)
