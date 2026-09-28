# Pinned native adaptive-splitter algorithm, with Rmath special functions.
# Run at repository root with Rcpp available and MAKEFLAGS=-j1. No installation.
# RNG, proposal, median splitting, sign correction and probability assembly
# bodies are unchanged. Boost digamma/trigamma are replaced by Rmath functions;
# these fixtures do NOT establish bitwise Boost or whole-package parity.
options(warn=2,digits=17)
base <- 'research/raw/EasyCellType/Bioc-3.18/fgsea/src'
hashes <- c('util.h'='eb23ea993ba2984473b2a61193144cf6',
  'util.cpp'='59d839aad00a7075327c798197c1477f',
  'esCalculation.h'='f94b9538f4866e2d094210ec772b92a3',
  'esCalculation.cpp'='5269032620bab0b25130cff24d2a6a9d',
  'fgseaMultilevelSupplement.h'='f7ea83cc7fd02f2cf7e3bfa2344bfaad',
  'fgseaMultilevelSupplement.cpp'='e13d31e0d92b2ce5a1e71f28b2c2b646',
  'fgseaMultilevel.cpp'='f069c267ef5ac4f8656e5040488644d3')
parts <- lapply(names(hashes),function(file) {
  path <- file.path(base,file)
  stopifnot(unname(tools::md5sum(path))==hashes[[file]])
  lines <- readLines(path,warn=FALSE)
  # Source bodies are flattened into one translation unit; their local header
  # declarations are already included in the ordered parts above.
  lines <- lines[!grepl('^#include "',lines) &
                 !grepl('^#include <boost/',lines) & lines!='#pragma once']
  lines <- gsub('boost::math::digamma','R::digamma',lines,fixed=TRUE)
  lines <- gsub('boost::math::trigamma','R::trigamma',lines,fixed=TRUE)
  if(file=='fgseaMultilevel.cpp') {
    marker <- which(grepl('^DataFrame fgseaMultilevelCpp',lines))
    stopifnot(length(marker)==1L)
    lines <- append(lines,'// [[Rcpp::export]]',after=marker-1L)
  }
  lines
})
Rcpp::sourceCpp(code=paste(c('#include <Rcpp.h>','using namespace Rcpp;',
  unlist(parts,use.names=FALSE)),collapse='\n'),showOutput=FALSE)

ranks <- abs(c(6.3,5.9,4.2,2.8,1.1,.7,-.25,-.8,-1.6,-2.4,-4.3,-7.7))
targets <- c(1,.8,-1,-.8)
rows <- list()
for(seed in 1:12) for(signed in c(FALSE,TRUE)) {
  result <- fgseaMultilevelCpp(targets,ranks,3L,101L,seed,1e-10,signed)
  rows[[length(rows)+1L]] <- data.frame(seed=seed,set_size=3L,sample_size=101L,
    eps=1e-10,one_sided=signed,ES=targets,cppMPval=result$cppMPval,
    cppIsCpGeHalf=result$cppIsCpGeHalf)
}
out <- do.call(rbind,rows)
stopifnot(all(is.finite(out$cppMPval)),all(out$cppMPval>0),all(out$cppMPval<=1))
write.csv(out,'tests/fixtures/easycelltype-gsea-splitter-tails.csv',row.names=FALSE)
cat('Generated',nrow(out),'bounded adaptive-tail estimates from native algorithm; Rmath special functions.\n')
