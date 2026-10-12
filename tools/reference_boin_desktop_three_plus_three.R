# Original desktop3+3 wrapper with controlled binomial service.
# Usage: R --vanilla --slave -f this_file --args ORIGINAL_R_DIR OUTPUT_CSV
args<-commandArgs(TRUE);stopifnot(length(args)==2)
path<-file.path(args[1],'get.oc.3plus3.R')
stopifnot(unname(tools::md5sum(path))=='0d93129ca684fd5277d2255af086d774')
native<-new.env(parent=globalenv());sys.source(path,envir=native)
native$rbinom<-function(n,size,prob) {stopifnot(n==1);sum(rep(c(.1,.5,.9),length.out=size)<prob)}
encode<-function(x) paste(x,collapse=';');rows<-list()
for(cap in c(NA,6L,18L,30L)) for(truth in c('gradient','all','none')) {
 p<-switch(truth,gradient=c(.05,.15,.25,.55,.75),all=rep(.95,5),none=rep(.01,5))
 result<-NULL
 invisible(capture.output({result<-native$get.oc.3plus3(.25,p,nmax=cap,ntrial=2,seed=317)}))
 rows[[length(rows)+1L]]<-data.frame(cap=cap,truth=truth,patients=encode(result$nptsdose),
  toxicities=encode(result$ntoxdose),selection=encode(result$selpercent))
}
write.csv(do.call(rbind,rows),args[2],row.names=FALSE,na='')
