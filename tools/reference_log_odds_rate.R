# Independent base-R reference for Shen-Thall's generalized odds-rate model.
# It checks the sourced density/survival/CDF and an explicit proper Gaussian
# posterior on log(shape), log(scale), log(c). The prior is not a BCSTTE claim.
options(warn=2, digits=17)

legendre <- function(n, lower=-8, upper=8) {
  index <- seq_len(n-1L)
  off <- index / sqrt(4*index*index-1)
  jacobi <- matrix(0, n, n)
  jacobi[cbind(seq_len(n-1L), 2:n)] <- off
  jacobi[cbind(2:n, seq_len(n-1L))] <- off
  eig <- eigen(jacobi, symmetric=TRUE)
  ord <- order(eig$values)
  list(x=(lower+upper)/2+(upper-lower)/2*eig$values[ord],
       w=(upper-lower)*eig$vectors[1L,ord]^2)
}
softplus <- function(x) pmax(x,0)+log1p(exp(-abs(x)))

components <- function(relative_log_times, log_shape, centered_log_scale, log_c) {
  shape <- exp(log_shape)
  log_ratio <- relative_log_times-centered_log_scale
  log_power <- shape*log_ratio
  log_product <- log_c+log_power
  small <- log_product < -36
  log_q <- log_product
  log_q[small] <- log_power[small]
  regular <- !small
  log_q[regular] <- log(log1p(exp(log_product[regular])))-log_c
  q <- exp(log_q)
  log_one_plus_product <- softplus(log_product)
  log_survival <- -q
  log_density_centered <- log_shape-centered_log_scale+(shape-1)*log_ratio-
    q-log_one_plus_product
  cdf <- -expm1(-q)
  list(log_density_centered=log_density_centered,
       log_survival=log_survival,cdf=cdf)
}

# A small five-time design, with a right-censoring pattern requiring both
# event-density and survivor contributions. The 3x3 covariance is correlated.
times <- c(.45,.8,1.3,2.1,3.4)
relative_log_times <- log(times)-log(times[1])
prior_mean <- log(c(1.6,1.2,.7))
prior_covariance <- matrix(c(.13,.035,-.018,
                             .035,.19,.04,
                             -.018,.04,.16),3,3,byrow=TRUE)
stopifnot(min(eigen(prior_covariance,symmetric=TRUE,only.values=TRUE)$values)>0)
prior_upper <- chol(prior_covariance)
stopifnot(max(abs(crossprod(prior_upper)-prior_covariance)) < 1e-14)
cases <- list(complete=rep(1,5),right_censored=c(1,0,1,1,0))

posterior_reference <- function(events,nquad) {
  q <- legendre(nquad)
  grid <- expand.grid(z1=q$x,z2=q$x,z3=q$x)
  standardized <- as.matrix(grid)
  # R's chol returns upper U with t(U) %*% U = Sigma. Row vectors multiply U.
  log_coordinates <- sweep(standardized %*% prior_upper,2,
                           prior_mean,'+')
  prior_log_weight <- dnorm(grid$z1,log=TRUE)+dnorm(grid$z2,log=TRUE)+
    dnorm(grid$z3,log=TRUE)+log(q$w[match(grid$z1,q$x)])+
    log(q$w[match(grid$z2,q$x)])+log(q$w[match(grid$z3,q$x)])
  log_likelihood <- vapply(seq_len(nrow(log_coordinates)),function(i) {
    parts <- components(relative_log_times,log_coordinates[i,1],
                        log_coordinates[i,2],log_coordinates[i,3])
    sum(ifelse(events==1,parts$log_density_centered-log(times[1]),parts$log_survival))
  },numeric(1))
  log_weight <- prior_log_weight+log_likelihood
  weight <- exp(log_weight-max(log_weight))
  weight <- weight/sum(weight)
  cdf_draws <- vapply(seq_len(nrow(log_coordinates)),function(i) {
    components(relative_log_times,log_coordinates[i,1],
               log_coordinates[i,2],log_coordinates[i,3])$cdf
  },numeric(length(times)))
  means <- colSums(log_coordinates*weight)
  centered <- sweep(log_coordinates,2,means,'-')
  covariance <- crossprod(centered*sqrt(weight))
  c(mean_log_shape=means[1],mean_log_scale=means[2],mean_log_c=means[3],
    var_log_shape=covariance[1,1],var_log_scale=covariance[2,2],
    var_log_c=covariance[3,3],cov_log_shape_scale=covariance[1,2],
    cov_log_shape_c=covariance[1,3],cov_log_scale_c=covariance[2,3],
    mean_shape=sum(weight*exp(log_coordinates[,1])),
    mean_scale=sum(weight*exp(log_coordinates[,2])),
    mean_c=sum(weight*exp(log_coordinates[,3])),
    setNames(as.vector(cdf_draws %*% weight),paste0('posterior_mean_cdf_',seq_along(times))))
}

out <- list()
for (case_name in names(cases)) for (nquad in c(25L,35L,45L)) {
  summary <- posterior_reference(cases[[case_name]],nquad)
  out[[length(out)+1L]] <- data.frame(case=case_name,nquad=nquad,
    metric=names(summary),value=unname(summary))
}
reference <- do.call(rbind,out)
reference$value <- sprintf('%.17g',reference$value)
args <- commandArgs(trailingOnly=TRUE)
outdir <- if (length(args)) args[[1]] else getwd()
if (!dir.exists(outdir)) stop('output directory does not exist')
write.csv(reference,file.path(outdir,'log-odds-rate-reference.csv'),row.names=FALSE)

# Deterministic distribution identities, separately from the posterior grid.
identity_times <- c(.3,1,2.7)
identity <- list()
for (t in identity_times) {
  direct <- components(log(t),log(1.8),log(1.1),log(0.65))
  identity[[length(identity)+1L]] <- data.frame(time=t,
    log_density_centered=direct$log_density_centered,
    log_survival=direct$log_survival,cdf=direct$cdf,
    survival_plus_cdf=exp(direct$log_survival)+direct$cdf)
}
identity <- do.call(rbind,identity)
for (column in names(identity)) identity[[column]] <- sprintf('%.17g',identity[[column]])
write.csv(identity,file.path(outdir,'log-odds-rate-identities.csv'),row.names=FALSE)
cat('Generated independent generalized odds-rate references.\n')
