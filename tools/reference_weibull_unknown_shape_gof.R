# Independent joint Weibull posterior integration under a Gaussian prior on
# (log shape, log scale). Tensor Gauss-Legendre quadrature uses base-R densities.
options(warn=2,digits=17)
legendre <- function(n,width) {
  j <- seq_len(n-1L)
  off <- j/sqrt(4*j*j-1)
  matrix <- matrix(0,n,n)
  matrix[cbind(j,j+1L)] <- off
  matrix[cbind(j+1L,j)] <- off
  eig <- eigen(matrix,symmetric=TRUE)
  list(node=width*eig$values,weight=2*width*eig$vectors[1,]^2)
}
cases <- list(
  rising_hazard=list(times=c(.5,1,1.5,2.2,3),mean=log(c(1.5,2)),
    sd=c(.4,.7),correlation=.2),
  falling_hazard=list(times=c(.1,.7,2,8),mean=log(c(.8,1.8)),
    sd=c(.5,1),correlation=-.35)
)
integrate_case <- function(x,rule) {
  pairs <- expand.grid(i=seq_along(rule$node),j=seq_along(rule$node))
  z <- cbind(rule$node[pairs$i],rule$node[pairs$j])
  covariance <- outer(x$sd,x$sd)*matrix(c(1,x$correlation,x$correlation,1),2,2)
  theta <- sweep(z%*%chol(covariance),2,x$mean,'+')
  shape <- exp(theta[,1])
  scale <- exp(theta[,2])
  likelihood <- rowSums(vapply(x$times,function(t) {
    # Use the guide's log-density equation directly: R's density has
    # nonfinite intermediate powers in remote tails of this quadrature.
    log_hazard <- shape*(log(t)-theta[,2])
    representable <- log_hazard<=log(.Machine$double.xmax)
    value <- rep(-Inf,nrow(theta))
    value[representable] <- theta[representable,1]-log(t)+
      log_hazard[representable]-exp(log_hazard[representable])
    value
  },numeric(nrow(theta))))
  weight <- exp(likelihood-max(likelihood))*dnorm(z[,1])*dnorm(z[,2])*
    rule$weight[pairs$i]*rule$weight[pairs$j]
  weight <- weight/sum(weight)
  average <- function(v) sum(weight*v)
  mean <- colSums(theta*weight)
  centered <- sweep(theta,2,mean,'-')
  c(mean_log_shape=mean[1],mean_log_scale=mean[2],
    variance_log_shape=average(centered[,1]^2),variance_log_scale=average(centered[,2]^2),
    covariance=average(centered[,1]*centered[,2]),
    mean_shape=average(shape),mean_scale=average(scale),
    setNames(vapply(x$times,function(t) average(pweibull(t,shape=shape,scale=scale)),numeric(1)),
      paste0('mean_cdf.',seq_along(x$times)-1L)))
}
coarse <- legendre(181L,8)
fine <- legendre(241L,10)
output <- input <- list()
maximum_difference <- 0
for (name in names(cases)) {
  x <- cases[[name]]
  first <- integrate_case(x,coarse)
  second <- integrate_case(x,fine)
  difference <- max(abs(first-second))
  stopifnot(difference<2e-8)
  maximum_difference <- max(maximum_difference,difference)
  output[[name]] <- data.frame(case=name,metric=names(second),value=unname(second))
  input[[name]] <- data.frame(case=name,index=seq_along(x$times)-1L,time=x$times,
    prior_log_shape=x$mean[1],prior_log_scale=x$mean[2],
    prior_shape_sd=x$sd[1],prior_scale_sd=x$sd[2],prior_correlation=x$correlation)
}
write_exact <- function(rows,path) {
  data <- do.call(rbind,rows)
  for (name in names(data)) if (is.numeric(data[[name]])) data[[name]] <- sprintf('%.17g',data[[name]])
  write.csv(data,path,row.names=FALSE)
}
write_exact(input,'tests/fixtures/weibull-unknown-shape-inputs.csv')
write_exact(output,'tests/fixtures/weibull-unknown-shape-reference.csv')
cat(sprintf('Two joint posterior references; quadrature/domain discrepancy %.9g.\n',maximum_difference))
