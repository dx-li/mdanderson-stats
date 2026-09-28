# Bounded references from pinned dfcrm and the original BayesESS derivative.
# Only Bernoulli draws are injected; dfcrm inference and allocation are unchanged.
options(warn=2, digits=17)
native <- new.env()
sys.source("research/raw/dfcrm/dfcrm.R", envir=native)
bayesess <- new.env()
sys.source("research/raw/BayesESS/source/R/internal.R", envir=bayesess)
# Extract the author's nested derivative definition without running simulations.
definition <- body(bayesess$essCRM)
derivative <- definition[[which(vapply(as.list(definition), function(x) {
  is.call(x) && identical(x[[1]], as.name("<-")) &&
    identical(x[[2]], as.name("getDiffCRM"))
}, logical(1)))]]
eval(derivative, envir=bayesess)

M <- 12L
replications <- 3L
set.seed(1542026)
uniforms <- matrix(runif(M*replications), nrow=replications, byrow=TRUE)
cases <- list(
  ordinary=list(truth=c(.08,.18,.30,.45), prior=c(.05,.15,.30,.50),
                target=.25, sd=sqrt(1.34)),
  all_safe=list(truth=rep(0,4), prior=c(.05,.15,.30,.50),
                target=.25, sd=sqrt(1.34)),
  all_toxic=list(truth=rep(1,4), prior=c(.05,.15,.30,.50),
                 target=.25, sd=sqrt(1.34)))
settings <- histories <- summaries <- paths <- list()
for (case in names(cases)) {
  spec <- cases[[case]]
  settings[[length(settings)+1L]] <- data.frame(case=case,
    dose=seq_along(spec$prior), truth=spec$truth, skeleton=spec$prior,
    target=spec$target, beta_sd=spec$sd)
  information <- numeric(replications)
  for (r in seq_len(replications)) {
    position <- 0L
    native$rbinom <- function(n,size,prob) {
      stopifnot(n==1L,size==1L)
      position <<- position+1L
      as.integer(uniforms[r,position] < prob)
    }
    result <- native$onetrial(spec$truth,spec$prior,spec$target,n=M,x0=1,
      mcohort=1,restrict=TRUE,method="bayes",model="empiric",
      scale=spec$sd,seed=r)
    stopifnot(position==M)
    second <- bayesess$getDiffCRM(spec$prior[result$level],result$tox)
    information[r] <- -sum(second)
    histories[[length(histories)+1L]] <- data.frame(case=case,replicate=r,
      patient=seq_len(M),uniform=uniforms[r,],dose=result$level,
      dlt=result$tox,beta_before=result$beta.hat,
      likelihood_second_derivative=second)
    summaries[[length(summaries)+1L]] <- data.frame(case=case,replicate=r,
      final_beta=result$final.est,final_variance=result$post.var,
      final_dose=result$MTD,full_information=information[r])
  }
  # This is the exact expectation over all size-m subsets of each realized trial.
  mean_information <- (0:M)/M*mean(information)
  gap <- 1/spec$sd^2-mean_information
  source_grid <- approx(0:M,gap,method="linear")
  grid_ess <- source_grid$x[which.min(abs(source_grid$y))]
  root <- 1/spec$sd^2/(mean(information)/M)
  paths[[length(paths)+1L]] <- data.frame(case=case,patients=0:M,
    prior_information=1/spec$sd^2,mean_subset_information=mean_information,
    gap=gap,native_grid_ess=grid_ess,
    continuous_ess=if(root<=M)root else NA_real_)
}
write.csv(do.call(rbind,settings),"tests/fixtures/crm-prior-ess-settings.csv",row.names=FALSE)
write.csv(do.call(rbind,histories),"tests/fixtures/crm-prior-ess-histories.csv",row.names=FALSE)
write.csv(do.call(rbind,summaries),"tests/fixtures/crm-prior-ess-summary.csv",row.names=FALSE)
write.csv(do.call(rbind,paths),"tests/fixtures/crm-prior-ess-paths.csv",row.names=FALSE,na="NA")

# Native crm integrates mean/second-moment numerators on [-10,10], but its
# denominator on the whole real line. Compare both contracts independently.
posterior <- list()
for (sd in c(sqrt(1.34),4)) {
  for (tox in list(c(0,0,0),c(0,1,0),c(1,1,1))) {
    levels <- c(1,2,2)
    prior <- c(.05,.15,.30,.50)
    result <- native$crm(prior,.25,tox,levels,scale=sd)
    x <- prior[levels]
    weights <- rep(1,length(x))
    den <- integrate(native$crmh,-Inf,Inf,x,tox,weights,sd,
      abs.tol=0,rel.tol=1e-10)$value
    mean <- integrate(native$crmht,-Inf,Inf,x,tox,weights,sd,
      abs.tol=0,rel.tol=1e-10)$value/den
    second <- integrate(native$crmht2,-Inf,Inf,x,tox,weights,sd,
      abs.tol=0,rel.tol=1e-10)$value/den
    posterior[[length(posterior)+1L]] <- data.frame(beta_sd=sd,
      outcomes=paste0(tox,collapse=""),native_mean=result$estimate,
      native_variance=result$post.var,full_mean=mean,
      full_variance=second-mean^2)
  }
}
write.csv(do.call(rbind,posterior),"tests/fixtures/crm-prior-ess-posterior.csv",row.names=FALSE)
cat("Wrote 108 adaptive patient rows, nine trial endpoints, 39 subset paths and six posterior references.\n")
