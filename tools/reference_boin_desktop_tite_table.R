# Active desktop pending-state table; original source remains external.
# Managed GetBoundaryTITE always requests stft=1, not the inactive TFT branch.
options(warn=2,digits=17)
args<-commandArgs(TRUE);stopifnot(length(args)==2)
path<-file.path(args[1],'get.boundary.tite.R')
stopifnot(unname(tools::md5sum(path))=='19803f88bc78c49cc9fe7104c0258c94')
native<-new.env(parent=globalenv());sys.source(path,envir=native)
rows<-list();case<-0L
for(target in c(.25,.3)) for(toxic in c(1.4*target,1-target)) for(pending in c(.5,.65)) for(size in c(1L,3L))
for(informative in c(FALSE,TRUE)) for(safe in c(FALSE,TRUE)) {
 case<-case+1L
 result<-native$get.boundary.tite(target,12L/size,size,n.earlystop=12,maxpen=pending,
   maxt=3,p.tox=toxic,prior.p=if(informative) c(.1,.2,.7) else NA,extrasafe=safe,stft=1,
   print=FALSE,output=FALSE)
 stopifnot(is.list(result))
 tab<-as.data.frame(result$boundary_tab,stringsAsFactors=FALSE)
 names(tab)<-c('patients','toxicities','pending','escalate','stay','deescalate')
 rows[[length(rows)+1L]]<-cbind(data.frame(case=case,target=target,toxic_probability=toxic,max_pending=pending,
   cohort_size=size,informative=informative,extra_safe=safe),tab)
}
write.csv(do.call(rbind,rows),args[2],row.names=FALSE)
