# Original single-agent desktop wrapper with deterministic uniform host service.
# Usage: R --vanilla --slave -f this_file --args ORIGINAL_R_DIR OUTPUT_CSV
args<-commandArgs(TRUE);stopifnot(length(args)==2)
files<-c('get.boundary.BOIN.R','select.mtd.shiny.R','get.oc.shiny.R')
stopifnot(all(unname(tools::md5sum(file.path(args[1],files)))==
 c('03e4d09b025ac32fea2fe6037a6d55f8','2dba478c6426095633e3cb432f0f5661','75bafec883cb37416bb453eb7896a2e9')))
native<-new.env(parent=globalenv());for(f in files) sys.source(file.path(args[1],f),envir=native)
native$file.exists<-function(...) FALSE
native$runif<-function(n,...) rep(c(.1,.5,.9),length.out=n)
encode<-function(x) paste(x,collapse=';');rows<-list()
for(target in c(.25,.3)) for(titration in c(FALSE,TRUE)) for(runin in c(FALSE,TRUE)) {
 for(safe in c(FALSE,TRUE)) for(start in c(1L,2L)) for(cap in c(1L,3L)) {
  for(truth in c('gradient','all','none')) {
   p<-switch(truth,gradient=c(.05,.15,target,.55,.75),all=rep(.95,5),none=rep(.01,5))
   result<-NULL;error<-''
   invisible(capture.output(tryCatch({result<-native$get.oc.shiny(target,p,6L,3L,
    n.earlystop=12L,titration=titration,startdose=start,extrasafe=safe,ntrial=2L,
    RunIn=runin,titration.cap=cap,seed=137)},error=function(e) error<<-conditionMessage(e))))
   rows[[length(rows)+1L]]<-data.frame(target=target,titration=titration,runin=runin,
    extra_safe=safe,start=start,cap=cap,truth=truth,patients=encode(result$nptsdose),
    toxicities=encode(result$ntoxdose),selection=encode(result$selpercent),error=error)
  }
 }
}
write.csv(do.call(rbind,rows),args[2],row.names=FALSE,na='')
