# Original desktop single-MTD conduct with adverse edited histories.
# Original program remains an external, hash-checked input.
options(warn=2,digits=17)
args<-commandArgs(TRUE);stopifnot(length(args)==2)
files<-c('get.boundary.R','next.comb.R')
expected<-c('4c67779cb7735c845f9b490e3d2ed57b','8cd06fdadf896bf0f283bb803cbbfc41')
stopifnot(all(unname(tools::md5sum(file.path(args[1],files)))==expected))
native<-new.env(parent=globalenv())
for(file in files) sys.source(file.path(args[1],file),envir=native)
native$runif<-function(n,...) rep(0,n)
encode<-function(x) paste(as.vector(t(x)),collapse=';')
records<-list();set.seed(812)
for(case in 1:120) {
 n<-matrix(sample(c(0L,3L,6L,12L),6,replace=TRUE),2,3)
 y<-matrix(rbinom(6,as.vector(n),.35),2,3)
 dose<-c(sample(1:2,1),sample(1:3,1));n[dose[1],dose[2]]<-max(3,n[dose[1],dose[2]])
 for(safe in c(FALSE,TRUE)) for(runin in c(FALSE,TRUE)) {
  result<-NULL
  invisible(capture.output(result<-native$next.comb(.25,n,y,dose,n.earlystop=12,
       extrasafe=safe,boDoThreePlusThreeRunIn=runin)))
  stopifnot(length(result)==2)
  records[[length(records)+1L]]<-data.frame(extra_safe=safe,runin=runin,
    patients=encode(n),toxicities=encode(y),current_a=dose[1],current_b=dose[2],
    next_a=result[1],next_b=result[2])
 }
}
write.csv(do.call(rbind,records),args[2],row.names=FALSE)
