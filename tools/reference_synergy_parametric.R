# Run with R --vanilla --slave -f this-file --args ORIGINAL-SOURCE-DIRECTORY.
# Original institutional listings are external comparison inputs, not vendored.
args <- commandArgs(trailingOnly=TRUE)
if (length(args) != 1L) stop("Supply the extracted, verified SYNERGY source directory")
root <- args[[1]]
files <- c("greco90.ssc", "machado94.SSC", "plummer90.SSC", "carter88.SSC")
md5 <- c("ffdb7761dca940611ad084514dfe300c", "83023c50378cf325c1c56494a9403ccf",
         "7f7134b141694fb4d753d16659b45fff", "04979fe2239887176802b69a5152130c")
stopifnot(unname(tools::md5sum(file.path(root,files))) == md5)
for (file in files) source(file.path(root,file))
models <- c("greco", "machado", "plummer", "carter")
# Disable only the plotting callbacks; fitting/initialization kernels are unchanged.
for (model in models) assign(paste0(model,".plot"),function(...) NULL)
parameters <- list(greco=c(-.7,-1.3,1.8,.9,1.2), machado=c(-.7,-1.3,1.8,.9,.6),
                   plummer=c(.3,-.8,.2,.4,.6), carter=c(.8,-.4,-.6,-.08))
predict_native <- function(model,theta,d1,d2) {
  do.call(get(paste0(model,".model")),c(list(d1,d2),as.list(theta)))
}
grid <- expand.grid(dose1=c(0,.1,.3,1,3,10),dose2=c(0,.2,1,4))
grid <- grid[grid$dose1+grid$dose2>0,,drop=FALSE]
kernel_rows <- list()
for (model in models) {
  variants <- list(parameters[[model]])
  if (model == "greco") variants <- c(variants,list(c(-.7,-1.3,1.8,.9,-.1),
                                                   c(-1,-1,1.8,.9,1.2)))
  for (variant in seq_along(variants)) {
    theta <- variants[[variant]]
    logits <- predict_native(model,theta,grid$dose1,grid$dose2)
    for (i in seq_len(nrow(grid))) {
      padded <- c(theta,rep(NA,5-length(theta)))
      kernel_rows[[length(kernel_rows)+1L]] <- data.frame(
        model=model,variant=variant,p1=padded[1],p2=padded[2],p3=padded[3],
        p4=padded[4],p5=padded[5],dose1=grid$dose1[i],dose2=grid$dose2[i],
        logit_response=logits[i],response=plogis(logits[i]))
    }
  }
}
write.csv(do.call(rbind,kernel_rows),"tests/fixtures/synergy-parametric-kernels.csv",row.names=FALSE)
paper <- read.table(file.path(root,"nl22B2.txt"),header=TRUE)
inputs <- list(); fits <- list(); covariance <- list()
for (model in models) {
  d <- c(.1,.3,1,3,10)
  combinations <- expand.grid(dose1=c(.2,1,4),dose2=c(.2,1,4))
  d1 <- c(d,rep(0,5),combinations$dose1)
  d2 <- c(rep(0,5),d,combinations$dose2)
  logits <- predict_native(model,parameters[[model]],d1,d2) + .04*cos(seq_along(d1)*1.7)
  synthetic <- data.frame(drug1=c(0,d1),drug2=c(0,d2),response=c(.95,plogis(logits)))
  cases <- c("paper","synthetic")
  if (model == "greco") {
    cases <- c(cases,"antagonistic")
    anti_logits <- predict_native(model,c(parameters[[model]][1:4],-.1),d1,d2) +
                   .04*cos(seq_along(d1)*1.7)
    antagonistic <- data.frame(drug1=c(0,d1),drug2=c(0,d2),response=c(.95,plogis(anti_logits)))
  }
  for (case in cases) {
    data <- if (case == "paper") paper else if (case == "antagonistic") antagonistic else synthetic
    names(data) <- c("drug1","drug2","response")
    output <- get(paste0("drug.combination.",model))(data)
    fit <- output[[model]]
    info <- summary(fit,correlation=TRUE)
    theta <- coef(fit)
    initial <- output[[1]]
    inputs[[length(inputs)+1L]] <- data.frame(model=model,case=case,row=seq_len(nrow(data)),data)
    for (i in seq_along(theta)) {
      fits[[length(fits)+1L]] <- data.frame(model=model,case=case,parameter=names(theta)[i],
        initial=initial[i],estimate=theta[i],standard_error=info$parameters[i,2],
        t_statistic=info$parameters[i,3],p_value=info$parameters[i,4],
        residual_standard_error=info$sigma,sse=deviance(fit),df=df.residual(fit))
      for (j in seq_along(theta)) covariance[[length(covariance)+1L]] <- data.frame(
        model=model,case=case,i=i,j=j,covariance=vcov(fit)[i,j],correlation=info$correlation[i,j])
    }
  }
}
write.csv(do.call(rbind,inputs),"tests/fixtures/synergy-parametric-inputs.csv",row.names=FALSE)
write.csv(do.call(rbind,fits),"tests/fixtures/synergy-parametric-fits.csv",row.names=FALSE)
write.csv(do.call(rbind,covariance),"tests/fixtures/synergy-parametric-covariance.csv",row.names=FALSE)
cat("Verified original sources and wrote 138 kernels and nine native fits\n")
