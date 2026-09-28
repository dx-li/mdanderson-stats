# Original author's R calculation with fixed covariate inputs and exit tracing.
# No compiled helpers, simulation packages or installed BayesESS are required.
options(warn=2, digits=17)
source_path <- "research/raw/BayesESS/source/R/internal.R"
environment <- new.env()
sys.source(source_path, envir=environment)
environment$capture <- new.env()
invisible(trace("ESS_RegressionCalc", exit=quote({
  capture$whole <- ESS
  capture$paths <- cbind(Dqm.out,Dqm.out.s1,Dqm.out.s2)
  capture$prior <- c(sum_Dp,sum_Dp.s1,sum_Dp.s2)
  capture$epsilon <- c(sum_Dq0,sum_Dq0.s1,sum_Dq0.s2)
}), print=FALSE, where=environment))
set.seed(80154)
replicates <- 16L
maximum <- 32L
covariates <- matrix(runif(replicates*maximum*2,-1,1),ncol=2)
# Native code draws all ten potential covariates, even when only two are used.
padded <- cbind(covariates,matrix(0,nrow(covariates),8))
stream <- as.vector(t(padded))
position <- 0L
environment$runif <- function(n,min,max) {
  stopifnot(n==1,min==-1,max==1)
  position <<- position+1L
  stream[position]
}
input <- data.frame(replicate=rep(seq_len(replicates),each=maximum),
                    patient=rep(seq_len(maximum),replicates),
                    x1=covariates[,1],x2=covariates[,2])
write.csv(input,"tests/fixtures/regression-ess-simulation-covariates.csv",row.names=FALSE)
cases <- list(
  normal=list(model=1,priors=list(c(1,.2,1),c(2,2,1),c(1,-.4,2),c(2,3,2))),
  logistic=list(model=2,priors=list(c(1,-.4,1),c(1,.7,2),c(2,2,1))),
  flat_logistic=list(model=2,priors=list(c(1,0,1),c(1,0,1),c(1,0,1))),
  unreached_logistic=list(model=2,priors=list(c(1,10,1),c(1,0,1),c(1,0,1)))
)
prior_rows <- output <- paths <- list()
for(case in names(cases)) {
  settings <- cases[[case]]
  np <- length(settings$priors)
  priors <- c(settings$priors,rep(list(c(1,0,1000)),12-np))
  subset1 <- c(rep(1,3),rep(0,9))
  subset2 <- if(settings$model==1)c(0,0,0,1,rep(0,8)) else c(0,1,1,rep(0,9))
  args <- c(list(Reg_model=settings$model,Num_cov=2),
            setNames(priors,paste0("Prior_",0:11)),
            list(M=maximum,NumSims=replicates,theta_sub1=subset1,theta_sub2=subset2))
  position <- 0L
  result <- do.call(environment$ESS_RegressionCalc,args)
  stopifnot(position==length(stream))
  for(j in seq_len(np)) {
    prior <- priors[[j]]
    prior_rows[[length(prior_rows)+1L]] <- data.frame(case=case,
      model=if(settings$model==1)"normal" else "logistic",parameter=j,
      family=if(prior[1]==1)"normal" else "gamma",parameter1=prior[2],
      parameter2=prior[3]) # Normal variance; gamma RATE, as in original R.
  }
  found <- c(environment$capture$whole,result$ESSsubvector1,result$ESSsubvector2)
  for(j in 1:3) {
    scope <- c("whole","coefficients","precision_or_slopes")[j]
    output[[length(output)+1L]] <- data.frame(case=case,scope=scope,ess=found[j])
    paths[[length(paths)+1L]] <- data.frame(case=case,scope=scope,patients=0:maximum,
      prior_information=environment$capture$prior[j],
      epsilon_information=environment$capture$epsilon[j],
      mean_posterior_information=environment$capture$paths[,j])
  }
}
write.csv(do.call(rbind,prior_rows),"tests/fixtures/regression-ess-simulation-priors.csv",row.names=FALSE)
write.csv(do.call(rbind,output),"tests/fixtures/regression-ess-simulation-ess.csv",row.names=FALSE,na="NA")
write.csv(do.call(rbind,paths),"tests/fixtures/regression-ess-simulation-paths.csv",row.names=FALSE)
cat("Wrote four original-R cases, 12 ESS results and 396 path rows.\n")
