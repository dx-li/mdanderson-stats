# Independent quadrature reference for unknown-shape Weibull right-censor fits.
# Base R only; tensor Gauss-Legendre integration is over standardized Gaussian
# prior coordinates (log shape, log scale).
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
  mixed_zero_censor=list(
    times=c(0,.6,1.4,2.7),event=c(FALSE,TRUE,FALSE,TRUE),
    mean=log(c(1.4,1.8)),sd=c(.45,.65),correlation=.22),
  all_censored=list(
    times=c(.5,1.5,3.0),event=rep(FALSE,3),
    mean=log(c(1.1,2.4)),sd=c(.55,.8),correlation=-.3)
)
integrate_case <- function(x,rule) {
  pairs <- expand.grid(i=seq_along(rule$node),j=seq_along(rule$node))
  z <- cbind(rule$node[pairs$i],rule$node[pairs$j])
  covariance <- outer(x$sd,x$sd)*
    matrix(c(1,x$correlation,x$correlation,1),2,2)
  theta <- sweep(z%*%chol(covariance),2,x$mean,'+')
  shape <- exp(theta[,1])
  log_likelihood <- numeric(nrow(theta))
  for (i in seq_along(x$times)) {
    t <- x$times[i]
    if (t == 0 && !x$event[i]) next
    log_ratio <- log(t)-theta[,2]
    log_hazard <- shape*log_ratio
    finite_hazard <- log_hazard<=log(.Machine$double.xmax)
    hazard <- rep(Inf,nrow(theta))
    hazard[finite_hazard] <- exp(log_hazard[finite_hazard])
    term <- -hazard
    if (x$event[i]) {
      term[finite_hazard] <- theta[finite_hazard,1]-log(t)+
        log_hazard[finite_hazard]-hazard[finite_hazard]
    }
    log_likelihood <- log_likelihood+term
  }
  weight <- exp(log_likelihood-max(log_likelihood))*dnorm(z[,1])*dnorm(z[,2])*
    rule$weight[pairs$i]*rule$weight[pairs$j]
  weight <- weight/sum(weight)
  average <- function(v) sum(weight*v)
  mean <- colSums(theta*weight)
  centered <- sweep(theta,2,mean,'-')
  c(mean_log_shape=mean[1],mean_log_scale=mean[2],
    variance_log_shape=average(centered[,1]^2),
    variance_log_scale=average(centered[,2]^2),
    covariance=average(centered[,1]*centered[,2]),
    mean_shape=average(shape),mean_scale=average(exp(theta[,2])),
    setNames(vapply(x$times,function(t) {
      if (t==0) return(0)
      average(-expm1(-exp(shape*(log(t)-theta[,2]))))
    },numeric(1)),paste0('mean_cdf.',seq_along(x$times)-1L)))
}
coarse <- legendre(161L,8)
fine <- legendre(201L,10)
output <- input <- list()
maximum_difference <- 0
for (name in names(cases)) {
  x <- cases[[name]]
  first <- integrate_case(x,coarse)
  second <- integrate_case(x,fine)
  difference <- max(abs(first-second))
  stopifnot(difference<3e-8)
  maximum_difference <- max(maximum_difference,difference)
  output[[name]] <- data.frame(case=name,metric=names(second),value=unname(second))
  input[[name]] <- data.frame(case=name,index=seq_along(x$times)-1L,
    time=x$times,event=x$event,prior_log_shape=x$mean[1],
    prior_log_scale=x$mean[2],prior_shape_sd=x$sd[1],
    prior_scale_sd=x$sd[2],prior_correlation=x$correlation)
}
write_exact <- function(rows,path) {
  data <- do.call(rbind,rows)
  for (name in names(data)) if (is.numeric(data[[name]]))
    data[[name]] <- sprintf('%.17g',data[[name]])
  write.csv(data,path,row.names=FALSE)
}
write_exact(input,'tests/fixtures/weibull-unknown-shape-censor-inputs.csv')
write_exact(output,'tests/fixtures/weibull-unknown-shape-censor-reference.csv')
cat(sprintf('Two censored posterior references; quadrature/domain discrepancy %.9g.\n',
  maximum_difference))
