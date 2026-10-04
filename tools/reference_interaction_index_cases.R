# Independent base-R reference for Lee & Kong (2009) CI of interaction index.
# Inputs are transcribed from cached BioC paper Table 2 and Table 3.
# Source: https://doi.org/10.1198/sbr.2009.0001, Section 4.2.
# No project Python implementation is called here.
args <- commandArgs(trailingOnly = TRUE)
out <- if (length(args)) args[[1]] else file.path(getwd(), "interaction-reference.csv")

fit_me <- function(dose, response, case, curve) {
  fit <- lm(qlogis(response) ~ log(dose))
  b <- unname(coef(fit))
  v <- vcov(fit)
  list(case=case, curve=curve, fit=fit, beta0=b[1], beta1=b[2],
       se0=sqrt(v[1,1]), se1=sqrt(v[2,2]), sigma=summary(fit)$sigma,
       Dm=exp(-b[1]/b[2]), covariance=v, n=length(dose),
       residual_variance=deviance(fit)/df.residual(fit))
}
logit <- function(y) qlogis(y)
invlogdose <- function(f, y) (logit(y)-f$beta0)/f$beta1

# Observed-combination Eq. (6)-(9), using the explicit residual-df pooled
# transformed-error variance convention documented by the Python API.
observed <- function(fits, dose, y, case, row) {
  z <- vapply(fits, invlogdose, numeric(1), y=y)
  terms <- log(dose) - z
  li <- max(terms) + log(sum(exp(terms-max(terms))))
  share <- exp(terms-li)
  variance <- 0
  for (i in seq_along(fits)) {
    g <- share[i]/fits[[i]]$beta1
    grad <- c(g, g*z[i])
    variance <- variance + drop(t(grad) %*% fits[[i]]$covariance %*% grad)
  }
  dfs <- vapply(fits, function(x) x$n-2, numeric(1))
  pooled <- sum(dfs * vapply(fits, `[[`, numeric(1), "residual_variance"))/sum(dfs)
  resp_grad <- -sum(share/vapply(fits, `[[`, numeric(1), "beta1"))
  variance <- variance + (resp_grad*sqrt(pooled))^2
  df <- sum(dfs)
  se <- sqrt(variance)
  width <- qt(.975, df)*se
  data.frame(case=case, kind="observed", row=row, response=y,
             log_index=li, log_se=se, lower=exp(li-width), upper=exp(li+width),
             df=df, pooled_logit_variance=pooled)
}

ray <- function(marginal, combo, proportion, y, case) {
  p <- proportion/sum(proportion)
  zi <- vapply(marginal, invlogdose, numeric(1), y=y)
  zc <- invlogdose(combo,y)
  terms <- log(p)+zc-zi
  li <- max(terms)+log(sum(exp(terms-max(terms))))
  share <- exp(terms-li)
  variance <- 0
  for(i in seq_along(marginal)) {
    g <- share[i]/marginal[[i]]$beta1
    grad <- c(g,g*zi[i])
    variance <- variance+drop(t(grad)%*%marginal[[i]]$covariance%*%grad)
  }
  gc <- -1/combo$beta1
  gcgrad <- c(gc,gc*zc)
  variance <- variance+drop(t(gcgrad)%*%combo$covariance%*%gcgrad)
  df <- sum(vapply(c(marginal,list(combo)),function(x)x$n-2,numeric(1)))
  se <- sqrt(variance)
  width <- qt(.975,df)*se
  data.frame(case=case,kind="ray",row=NA_integer_,response=y,
             log_index=li,log_se=se,lower=exp(li-width),upper=exp(li+width),df=df,
             pooled_logit_variance=NA_real_)
}

# Table 2: fractional survival is the modeled response, hence negative slopes.
t2 <- list(
  fit_me(c(.1,.5,1,2,4), c(.6701,.6289,.5577,.4550,.3755), "sch66336_4hpr", "sch66336"),
  fit_me(c(.1,.5,1,2), c(.7666,.5833,.5706,.4934), "sch66336_4hpr", "4hpr"),
  fit_me(c(.2,1,2,4), c(.6539,.4919,.3551,.2341), "sch66336_4hpr", "combination_total_dose")
)
combo_doses2 <- c(.1,.5,1,2)
combo_effects2 <- c(.6539,.4919,.3551,.2341)
obs2 <- lapply(seq_len(4), function(i) observed(t2[1:2], rep(combo_doses2[i],2),
                  combo_effects2[i], "sch66336_4hpr", i))
# Along equal-concentration ray, combination fit is against d1+d2; y below is
# fractional survival corresponding to plotted fractional inhibition 0.25,.5,.75.
ray2 <- do.call(rbind,lapply(c(.25,.5,.75),function(e) ray(t2[1:2],t2[[3]],c(1,1),1-e,"sch66336_4hpr")))

# Table 3: fractional inhibition is the modeled response. Combination dose is
# total dose x, split 17.4:1 (component proportions sum to 18.4).
t3 <- list(
  fit_me(c(8.7,17.4,26.1,34.8,43.5), c(.132,.267,.411,.476,.548), "phenanthroline_adp", "o_phenanthroline"),
  fit_me(c(.5,1,1.5,2,2.5), c(.175,.400,.492,.542,.592), "phenanthroline_adp", "adp"),
  fit_me(c(9.2,18.4,27.6,36.8,46.0), c(.507,.769,.872,.919,.944), "phenanthroline_adp", "combination_total_dose")
)
combo_total3 <- c(9.2,18.4,27.6,36.8,46.0)
combo_effects3 <- c(.507,.769,.872,.919,.944)
obs3 <- lapply(seq_along(combo_total3), function(i) {
  observed(t3[1:2], combo_total3[i]*c(17.4,1)/18.4, combo_effects3[i],
           "phenanthroline_adp", i)
})
ray3 <- do.call(rbind,lapply(c(.25,.5,.75),function(e) ray(t3[1:2],t3[[3]],c(17.4,1),e,"phenanthroline_adp")))

fit_rows <- do.call(rbind,lapply(c(t2,t3),function(f) data.frame(
  case=f$case,kind="fit",row=NA_integer_,curve=f$curve,beta0=f$beta0,se_beta0=f$se0,
  beta1=f$beta1,se_beta1=f$se1,Dm=f$Dm,sigma=f$sigma,n=f$n,df=f$n-2)))
results <- rbind(do.call(rbind,obs2),do.call(rbind,obs3),ray2,ray3)
write.csv(fit_rows,file=sub("\\.csv$","-fits.csv",out),row.names=FALSE,na="")
write.csv(results,out,row.names=FALSE,na="")
