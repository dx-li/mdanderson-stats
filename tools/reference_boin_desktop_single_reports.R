# Run original selectors unchanged; output synthetic numerical references only.
# Usage: R --vanilla --slave -f this_file --args ORIGINAL_R_DIR OUTPUT_CSV
options(warn=2,digits=17)
args<-commandArgs(TRUE);stopifnot(length(args)==2)
files<-c('select.mtd.shiny.R','select.mtd.tite.boin.R')
stopifnot(all(unname(tools::md5sum(file.path(args[1],files)))==
 c('2dba478c6426095633e3cb432f0f5661','8cee6b3372ddf8362e0ab4cc189f4a43')))
native<-new.env(parent=globalenv())
for(f in files) sys.source(file.path(args[1],f),envir=native)
set.seed(7531);records<-list();encode<-function(v) paste(v,collapse=';')
for(i in 1:80) {
 n<-sample(c(0L,3L,6L,12L,24L),5,replace=TRUE)
 y<-rbinom(5,n,c(.03,.12,.25,.4,.65))
 for(target in c(.2,.25,.3)) for(safe in c(FALSE,TRUE)) for(tite in c(FALSE,TRUE)) {
  value<-if(tite) native$select.mtd.tite.boin(target,n,y,extrasafe=safe,verbose=FALSE)$MTD else
   native$select.mtd.shiny(target,n,y,extrasafe=safe,print=FALSE)
  report<-new.env(parent=emptyenv())
  invisible(capture.output(if(tite) native$select.mtd.tite.boin(target,n,y,extrasafe=safe,
      verbose=TRUE,selectMTDOutput=report) else
      native$select.mtd.shiny(target,n,y,extrasafe=safe,print=TRUE,selectMTDOutput=report)))
  records[[length(records)+1L]]<-data.frame(patients=encode(n),toxicities=encode(y),
    target=target,extra_safe=safe,tite=tite,selected=value,
    doses=encode(report$doseLevels),means=encode(report$estToxProb),
    lower=encode(report$lowerCI),upper=encode(report$upperCI),overdose=encode(report$probTox))
 }
}
write.csv(do.call(rbind,records),args[2],row.names=FALSE)
