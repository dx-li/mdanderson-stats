# Execute the active original desktop conduct boundary routine, external only.
# Usage: R --vanilla --slave -f this_file --args ORIGINAL_R OUTPUT_CSV
options(warn=2,digits=17)
args<-commandArgs(TRUE)
stopifnot(length(args)==2,
          unname(tools::md5sum(args[1]))=='4c67779cb7735c845f9b490e3d2ed57b')
native<-new.env(parent=globalenv());sys.source(args[1],envir=native)
rows<-list()
for(target in c(.2,.25,.3)) for(enabled in c(FALSE,TRUE)) for(safe in c(FALSE,TRUE)) {
 b<-native$get.boundary(target,ncohort=18,cohortsize=1,n.earlystop=18,
      print=FALSE,boDoThreePlusThreeRunIn=enabled,extrasafe=safe)
 for(i in seq_len(ncol(b))) rows[[length(rows)+1L]]<-data.frame(
   target=target,runin=enabled,extra_safe=safe,n=i,
   escalate=b[2,i],deescalate=b[3,i],eliminate=b[4,i])
}
write.csv(do.call(rbind,rows),args[2],row.names=FALSE,na='')
