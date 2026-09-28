# Source-only intccr 3.0.4 numerical references. No package installation.
# Run sequentially from the repository root under an external time limit.
# Native optimizer outputs are compatibility references, not certified optima.
options(warn=2, digits=17)
expected <- c(
  "intccr/R/ciregic.R"="e1f2f6e29d84e6f520e0482adfdcc0a48c8f9b90",
  "intccr/R/bssmle.R"="d122c3ac22a446bfb8071b720c9df962de188ebe",
  "intccr/R/Surv2.R"="c32f52d7c8c275cda51dd23e9868d9d7e2615fd9",
  "intccr/R/bsderivs.R"="e468f3fd071e490b6e57f94ffc490776ab56752a",
  "intccr/R/naive_b.R"="bb46420a92a19bd634f2db6d69e5edb0d6677248",
  "intccr/R/bssmle_lse.R"="416b818d2e826169e20f917990193736011fa012",
  "alabama/R/constrOptim.nl.R"="05c93764ceb3eacafd1d2cc06ad8d6b4f7015cf1",
  "numDeriv/R/numDeriv.R"="394d5f1db1d644fd1b44a2e70f0294ce04fcf50d",
  "numDeriv/R/num2Deriv.R"="9adbafd4d39dc95d2d0b69283a5fc571bf1e497c")
for (name in names(expected)) {
  actual <- system2("git", c("hash-object", shQuote(file.path("research/raw",name))), stdout=TRUE)
  stopifnot(identical(unname(actual),unname(expected[name])))
}
library(splines)
for(name in c("numDeriv/R/numDeriv.R", "numDeriv/R/num2Deriv.R",
              "alabama/R/constrOptim.nl.R", "intccr/R/Surv2.R",
              "intccr/R/bsderivs.R", "intccr/R/naive_b.R",
              "intccr/R/bssmle.R", "intccr/R/bssmle_lse.R", "intccr/R/ciregic.R")) {
  source(file.path("research/raw",name))
}
# Change only package dispatch to the source-loaded, unchanged optimizer.
# Capture model closures to compare the supplied derivatives independently.
native_optimizer <- constrOptim.nl
captured <- NULL
derivative_mode <- "native"
reference_optimizer <- function(...) {
  captured <<- list(...)
  args <- list(...)
  if (derivative_mode == "finite_difference") {
    args$hin.jac <- function(theta) jacobian(captured$hin,theta)
    args$heq.jac <- function(theta) jacobian(captured$heq,theta)
  }
  if (derivative_mode == "analytic_corrected") {
    model <- environment(args$hin)
    m <- get("n",model); p <- get("q",model)
    corners <- as.matrix(get("comb",model))
    alpha <- c(get("a1",model),get("a2",model))
    link_derivative <- function(eta,a) {
      if(a==0) exp(eta-exp(eta))
      else exp(eta-(1+1/a)*log1p(a*exp(eta)))
    }
    constraint_jacobian <- function(theta,equality=FALSE) {
      result <- matrix(0,2*(m-1)+nrow(corners),length(theta))
      if(!equality) result[seq_len(2*(m-1)),] <-
        captured$hin.jac(theta)[seq_len(2*(m-1)),]
      rows <- 2*(m-1)+seq_len(nrow(corners))
      for(cause in 1:2) {
        control <- (cause-1)*m+if(equality)1 else m
        beta <- 2*m+(cause-1)*p+seq_len(p)
        value <- -link_derivative(theta[control]+as.vector(corners%*%theta[beta]),alpha[cause])
        result[rows,control] <- value
        result[rows,beta] <- corners*value
      }
      result
    }
    args$hin.jac <- function(theta) constraint_jacobian(theta,FALSE)
    args$heq.jac <- function(theta) constraint_jacobian(theta,TRUE)
    point <- args$par
    point[1:m] <- seq(-6,-2,length.out=m)
    point[(m+1):(2*m)] <- seq(-7,-2.5,length.out=m)
    point[(2*m+1):length(point)] <- c(.15,-.05,-.1,.08)
    stopifnot(max(abs(args$hin.jac(point)-jacobian(args$hin,point)))<1e-7,
              max(abs(args$heq.jac(point)-jacobian(args$heq,point)))<1e-7)
  }
  do.call(native_optimizer,args)
}
redirect <- function(expr) {
  if (identical(expr,quote(alabama::constrOptim.nl))) return(quote(reference_optimizer))
  if (is.call(expr)) return(as.call(lapply(as.list(expr),redirect)))
  expr
}
body(bssmle) <- redirect(body(bssmle))

id <- 1:120
x1 <- cos(id*.71)+id/200
x2 <- sin(id*1.31)-id/250
u1 <- ((id*37)%%127+.5)/127
u2 <- ((id*53)%%131+.5)/131
t1 <- -log(u1)/(.3*exp(.4*x1+.2*x2))
t2 <- -log(u2)/(.25*exp(-.2*x1+.1*x2))
latent <- pmin(t1,t2)
cause <- ifelse(t1<t2,1,2)
width <- .3+.013*(id%%11)
v <- floor(latent/width)*width
u <- v+width
censor <- 2+.023*(id%%13)
right <- u>censor
v[right] <- censor[right]
u[right] <- Inf
cause[right] <- 0
data <- data.frame(v=v,u=u,event=cause,x1=x1,x2=x2)
out <- "research/raw/interval-competing-reference"
dir.create(out,showWarnings=FALSE)
write.csv(data,file.path(out,"inputs.csv"),row.names=FALSE)
write.csv(data,"tests/fixtures/interval-competing-risk-inputs.csv",row.names=FALSE)
cases <- list(fine_gray=c(0,0),odds=c(1,1),mixed=c(0,1))
parameter_tables <- covariance_tables <- curve_tables <- list()
reports <- list(reference=list(
  intccr="252644c0d347a663ea5d7bef88fa2ab04f114b1e",
  alabama="3dd535fac47afe823162a4755c3a1faddef6c566",
  numDeriv="54dc4181ec0543a95a2cf7a5e3c483ab0a109750"),
  native=list(),corrected_constraint_jacobians=list(),
  notes=paste("Native convergence is not a certificate of an optimum.",
              "Ordinary objective scores are not constrained KKT residuals."))
for (name in names(cases)) {
  cat("START",name,"\n")
  before <- proc.time()[3]
  fit <- bssmle(Surv2(v,u,event=event)~x1+x2,data,alpha=cases[[name]],k=.5)
  stopifnot(!is.null(captured))
  # Verify derivatives at an interior evaluation point, even if fit fails.
  point <- captured$par
  q <- 2
  m <- (length(point)-2*q)/2
  point[1:m] <- seq(-6,-2,length.out=m)
  point[(m+1):(2*m)] <- seq(-7,-2.5,length.out=m)
  point[(2*m+1):(2*m+2*q)] <- c(.15,-.05,-.1,.08)
  gradient_error <- max(abs(captured$gr(point)-grad(captured$fn,point)))
  inequality_error <- max(abs(captured$hin.jac(point)-jacobian(captured$hin,point)))
  equality_error <- max(abs(captured$heq.jac(point)-jacobian(captured$heq,point)))
  report <- list(case=name,alpha=unname(cases[[name]]),convergence=fit$convergence,
    elapsed=proc.time()[3]-before,log_likelihood=fit$loglikelihood,
    gradient_error=gradient_error,inequality_jacobian_error=inequality_error,
    equality_jacobian_error=equality_error)
  if (all(is.finite(fit$beta))) {
    report$minimum_inequality <- min(captured$hin(fit$beta))
    report$maximum_equality <- max(abs(captured$heq(fit$beta)))
    report$first_baseline_controls <- fit$beta[c(1,m+1)]
    report$maximum_objective_score <- max(abs(captured$gr(fit$beta)))
  }
  print(report)
  saveRDS(list(fit=fit,report=report),file.path(out,paste0(name,".rds")))
  reports$native[[length(reports$native)+1L]] <- report
  if (all(is.finite(fit$beta))) {
    parameter_tables[[name]] <- data.frame(case=name,
      parameter=seq_along(fit$beta),value=fit$beta)
    write.csv(data.frame(parameter=seq_along(fit$beta),value=fit$beta),
              file.path(out,paste0(name,"-parameters.csv")),row.names=FALSE)
    covariance <- bssmle_lse(fit)
    stopifnot(all(is.finite(covariance)))
    covariance_tables[[name]] <- data.frame(case=name,
      row=rep(seq_len(nrow(covariance)),ncol(covariance)),
      column=rep(seq_len(ncol(covariance)),each=nrow(covariance)),
      value=as.vector(covariance))
    write.csv(covariance,file.path(out,paste0(name,"-covariance.csv")),row.names=FALSE)
    object <- list(coefficients=tail(fit$beta,4),gamma=head(fit$beta,-4),
                   Bv=fit$Bv,tms=fit$tms,alpha=fit$alpha)
    curves <- lapply(list(c(0,0),c(-.5,.25),c(.5,-.25)),function(profile) {
      pred <- predict.ciregic(object,profile,seq(fit$tms[1],fit$tms[2],length.out=31))
      cbind(x1=profile[1],x2=profile[2],pred)
    })
    write.csv(do.call(rbind,curves),file.path(out,paste0(name,"-curves.csv")),row.names=FALSE)
    curve_tables[[name]] <- data.frame(case=name,do.call(rbind,curves))
  }
}
for (kind in c("parameters","covariance","curves")) {
  tables <- switch(kind,parameters=parameter_tables,covariance=covariance_tables,
                    curves=curve_tables)
  write.csv(do.call(rbind,tables),paste0("tests/fixtures/interval-competing-risk-",kind,".csv"),
            row.names=FALSE)
}

# Same model and native optimizer, independently evaluated constraint Jacobians.
# Keep these outputs distinct from the unchanged native-derivative results.
derivative_mode <- "analytic_corrected"
for (name in names(cases)) {
  cat("START corrected",name,"\n")
  before <- proc.time()[3]
  fit <- bssmle(Surv2(v,u,event=event)~x1+x2,data,alpha=cases[[name]],k=.5)
  report <- list(case=name,convergence=fit$convergence,
                 elapsed=proc.time()[3]-before,log_likelihood=fit$loglikelihood)
  if (all(is.finite(fit$beta))) {
    report$minimum_inequality <- min(captured$hin(fit$beta))
    report$maximum_equality <- max(abs(captured$heq(fit$beta)))
    report$maximum_objective_score <- max(abs(captured$gr(fit$beta)))
  }
  print(report)
  saveRDS(list(fit=fit,report=report),file.path(out,paste0(name,"-corrected.rds")))
  reports$corrected_constraint_jacobians[[length(reports$corrected_constraint_jacobians)+1L]] <- report
}
jsonlite::write_json(reports,"tests/fixtures/interval-competing-risk-diagnostics.json",
                     pretty=TRUE,auto_unbox=TRUE,digits=17)
