# Original desktop single-MTD interactive R, external source only.
# Usage: R --vanilla --slave -f this_file --args ORIGINAL_R_DIRECTORY OUTPUT_CSV
options(warn=2,digits=17)
args<-commandArgs(TRUE);stopifnot(length(args)==2)
files<-c('get.boundary.R','next.comb.R')
expected<-c('4c67779cb7735c845f9b490e3d2ed57b','8cd06fdadf896bf0f283bb803cbbfc41')
stopifnot(all(unname(tools::md5sum(file.path(args[1],files)))==expected))
native<-new.env(parent=globalenv())
for(file in files) sys.source(file.path(args[1],file),envir=native)
native$runif<-function(n,...) rep(0,n)
encode<-function(x) paste(as.vector(t(x)),collapse=';')
records<-list()
for(target in c(.25,.3)) for(toxic in c(1.4*target,.5)) for(runin in c(FALSE,TRUE)) for(safe in c(FALSE,TRUE)) {
 for(i in 1:2) for(j in 1:3) for(size in c(3L,6L,12L)) for(y in 0:size) {
  n<-tox<-matrix(0L,2,3);n[i,j]<-size;tox[i,j]<-y
  result<-NULL
  invisible(capture.output(result<-native$next.comb(target,n,tox,c(i,j),n.earlystop=12,
     extrasafe=safe,p.tox=toxic,boDoThreePlusThreeRunIn=runin)))
  stopifnot(length(result)==2)
  records[[length(records)+1L]]<-data.frame(target=target,toxic_probability=toxic,runin=runin,extra_safe=safe,
    patients=encode(n),toxicities=encode(tox),current_a=i,current_b=j,
    next_a=result[1],next_b=result[2])
 }
}
write.csv(do.call(rbind,records),args[2],row.names=FALSE)
