# Run unchanged institutional CI-IIV2 functions with synthetic inputs.
# First argument: checksum-verified extracted source folder. Second: output dir.
args <- commandArgs(trailingOnly=TRUE)
source_dir <- args[1]; out <- args[2]
dir.create(out, recursive=TRUE, showWarnings=FALSE)
env <- new.env()
sys.source(file.path(source_dir, 'CI_IIV2.SSC'), envir=env)
for (case in 1:2) {
  ratio <- if(case==1) 2 else 0.7
  doses <- list(seq(.1,3,length.out=5), seq(.1,6,length.out=6), seq(.5,4.5,length.out=4))
  truth <- list(c(0,-1), c(log(2),-1), c(2*log(1.5),-2))
  responses <- lapply(1:3,function(j) {
    x <- doses[[j]]
    noise <- .05*case*cos(seq_along(x)*(j+.7))
    plogis(truth[[j]][1]+truth[[j]][2]*log(x)+noise)
  })
  d1<-doses[[1]];d2<-doses[[2]];d12<-doses[[3]]
  e1<-responses[[1]];e2<-responses[[2]];e12<-responses[[3]]
  grid <- c(.2,.5,.8,e12)
  delta <- env$CI.delta(d1,e1,d2,e2,d12,e12,ratio,grid)
  known <- env$CI.known.effect(d1,e1,d2,e2,d12/(1+ratio),d12*ratio/(1+ratio),e12)
  normals <- list()
  env$rnorm <- function(n,mean=0,sd=1) {
    values <- stats::rnorm(n,mean=mean,sd=sd)
    normals[[length(normals)+1]] <<- values
    values
  }
  samples <- 31
  mc <- env$CI.simulation(d1,e1,d2,e2,d12,e12,ratio,grid,N.iter=samples,my.seed=299+case)
  z <- do.call(rbind,normals)
  fits <- lapply(1:3,function(j) lm(qlogis(responses[[j]])~log(doses[[j]])))
  draws <- matrix(0,samples,6)
  for(j in 1:3) {
    fit <- fits[[j]]; v <- vcov(fit); b <- coef(fit)
    correlation <- v[1,2]/sqrt(v[1,1]*v[2,2])
    draws[,2*j-1] <- b[1]+sqrt(v[1,1])*(correlation*z[,2*j-1]+sqrt(1-correlation^2)*z[,2*j])
    draws[,2*j] <- b[2]+sqrt(v[2,2])*z[,2*j-1]
  }
  write.csv(draws,file.path(out,paste0('draws-',case,'.csv')),row.names=FALSE)
  inputs <- do.call(rbind,lapply(1:3,function(j) data.frame(curve=j,dose=doses[[j]],effect=responses[[j]])))
  write.csv(inputs,file.path(out,paste0('inputs-',case,'.csv')),row.names=FALSE)
  inverse <- lapply(fits,function(f) exp((qlogis(grid)-coef(f)[1])/coef(f)[2]))
  inverse_variances <- lapply(1:3,function(j) {
    gradients <- cbind(1,log(inverse[[j]]))/coef(fits[[j]])[2]
    rowSums((gradients %*% vcov(fits[[j]]))*gradients)
  })
  variance_corrected <- (inverse[[3]]/inverse[[1]]/(1+ratio))^2*inverse_variances[[1]] +
    (inverse[[3]]/inverse[[2]]*ratio/(1+ratio))^2*inverse_variances[[2]] +
    delta$ii^2*inverse_variances[[3]]
  width_corrected <- qt(.975,sum(lengths(doses))-6)*sqrt(variance_corrected)/delta$ii
  reference <- data.frame(effect=grid,ratio=ratio,index=delta$ii,delta_lower=delta$ii.low,
             delta_upper=delta$ii.up,corrected_lower=delta$ii*exp(-width_corrected),
             corrected_upper=delta$ii*exp(width_corrected),mc_sd=mc$std,mc_lower=mc$ii.low,mc_upper=mc$ii.up)
  write.csv(reference,file.path(out,paste0('reference-',case,'.csv')),row.names=FALSE)
  write.csv(data.frame(effect=e12,index=known$ii,lower=known$ii.low,upper=known$ii.up),
             file.path(out,paste0('observed-',case,'.csv')),row.names=FALSE)
}
write.csv(data.frame(effect=seq(.1,.95,.02),index=env$Solve.II(seq(.1,.95,.02),-1,-1,-2,1,2,1.5,2)),
          file.path(out,'source-truth.csv'),row.names=FALSE)
cat('Executed original delta, pooled-error and Monte Carlo functions for two synthetic designs\n')
