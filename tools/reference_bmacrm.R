# Independent base-R integration of the BMA-CRM power model. No original code,
# Python calls, R packages, or native simulation output are used.
# Run from the repository root with Rscript tools/reference_bmacrm.R.
options(digits=17)
guide <- rbind(c(.10,.21,.24,.30,.45), c(.15,.26,.29,.35,.50),
               c(.20,.31,.34,.40,.55))
cases <- list(
  prior=list(p=guide, y=rep(0,5), n=rep(0,5)),
  mixed=list(p=guide, y=c(0,1,2,0,0), n=c(3,6,6,0,0)),
  weighted=list(p=guide, y=c(0,1,2,0,0), n=c(3,6,6,0,0), w=c(2,0,3)),
  all_toxic=list(p=guide, y=c(3,3,0,0,0), n=c(3,3,0,0,0)),
  no_toxic=list(p=guide, y=rep(0,5), n=c(3,3,3,3,3)),
  single=list(p=guide[1,,drop=FALSE], y=c(0,1,2,0,0), n=c(3,6,6,0,0)),
  concentrated=list(p=guide, y=c(0,0,3000,0,0), n=c(0,0,10000,0,0)),
  boundary=list(p=guide, y=c(10000,0,0,0,0), n=c(10000,0,0,0,0)),
  extreme_skeleton=list(p=rbind(c(1e-6,.5,.999999),c(1e-5,.6,.99999)),
                        y=c(0,2,3), n=c(3,6,3), sd=.75),
  rescued_prior=list(p=rbind(c(.01,.99),c(.49,.51)),y=c(0,5000),
                    n=c(5000,5000),w=c(1e-300,1e300)))
rows <- list()
for (name in names(cases)) {
  case <- cases[[name]]
  sd <- if (is.null(case$sd)) sqrt(2) else case$sd
  target <- .3
  K <- nrow(case$p); J <- ncol(case$p)
  raw_prior <- if (is.null(case$w)) rep(1,K) else case$w
  log_prior <- log(raw_prior)
  log_prior <- log_prior-max(log_prior)-log(sum(exp(log_prior-max(log_prior))))
  prior <- exp(log_prior)
  fits <- lapply(seq_len(K), function(k) {
    p <- case$p[k,]
    prob <- function(a,j) exp(log(p[j])*exp(a))
    # dbinom includes the binomial coefficients, unlike a likelihood kernel.
    lp <- function(a) vapply(a, function(v)
      sum(dbinom(case$y,case$n,exp(log(p)*exp(v)),log=TRUE)) +
        dnorm(v,0,sd,log=TRUE), numeric(1))
    objective <- function(a) {
      value <- -lp(a)
      # Binomial densities can be exactly zero in remote tails.
      if (is.infinite(value) && value>0) .Machine$double.xmax else value
    }
    mode <- optimize(objective,c(-40*sd,40*sd),tol=1e-10)$minimum
    shift <- lp(mode)
    weight <- function(a) exp(lp(a)-shift)
    # Work directly in alpha, splitting around the mode at several scales.
    # Infinite endpoints retain the full normal prior, including tail mass.
    cuts <- sort(unique(c(-Inf,mode+sd*c(-12,-4,-1,-.1,-.01,-.001,
                                          0,.001,.01,.1,1,4,12),Inf)))
    integral <- function(f, lower=-Inf, upper=Inf) {
      parts <- c(lower,cuts[cuts>lower & cuts<upper],upper)
      sum(vapply(seq_len(length(parts)-1), function(i)
        integrate(function(a) weight(a)*f(a),parts[i],parts[i+1],
                  abs.tol=1e-13,rel.tol=1e-10,subdivisions=200)$value,numeric(1)))
    }
    one <- function(a) rep(1,length(a))
    z <- integral(one)
    mean <- integral(function(a) a)/z
    variance <- integral(function(a) (a-mean)^2)/z
    dose_mean <- vapply(seq_len(J),function(j) integral(function(a) prob(a,j))/z,
                       numeric(1))
    overdose <- vapply(seq_len(J),function(j) {
      threshold <- log(-log(target))-log(-log(p[j]))
      if (threshold<=mode) integral(one,upper=threshold)/z else
        1-integral(one,lower=threshold)/z
    },numeric(1))
    list(log_evidence=shift+log(z),mean=mean,sd=sqrt(variance),
         dose_mean=dose_mean,overdose=overdose)
  })
  log_weights <- vapply(fits,function(f) f$log_evidence,numeric(1))+log_prior
  post <- exp(log_weights-max(log_weights)); post <- post/sum(post)
  bma_mean <- Reduce(`+`,Map(function(f,w) w*f$dose_mean,fits,post))
  bma_overdose <- Reduce(`+`,Map(function(f,w) w*f$overdose,fits,post))
  for (k in seq_len(K)) for (j in seq_len(J)) {
    f <- fits[[k]]
    rows[[length(rows)+1]] <- data.frame(
      scenario=name,model=k-1,dose=j-1,skeleton=case$p[k,j],
      events=case$y[j],subjects=case$n[j],target=target,prior_sd=sd,
      raw_prior_weight=raw_prior[k],prior_weight=prior[k],posterior_weight=post[k],
      model_log_evidence=f$log_evidence,alpha_mean=f$mean,alpha_sd=f$sd,
      model_dose_mean=f$dose_mean[j],model_overdose=f$overdose[j],
      dose_mean=bma_mean[j],overdose=bma_overdose[j])
  }
}
write.csv(do.call(rbind,rows),'tests/fixtures/bmacrm-posterior.csv',row.names=FALSE)
