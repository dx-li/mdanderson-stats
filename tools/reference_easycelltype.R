# Original EasyCellType Fisher results without installing the single-cell stack.
# The ignored author source is pinned in research/easycelltype-next-audit.md.
# No third-party source is copied into this repository.
options(warn=2, digits=17)
source_path <- "research/raw/EasyCellType/R/test_fisher.R"
stopifnot(unname(tools::md5sum(source_path)) == "910f26b5923026ce3c0cf71b950068ef")
native <- new.env(parent=globalenv())
sys.source(source_path, envir=native)
stopifnot(is.function(native$test_fisher))

queries <- list()
references <- list()
results <- list()
summaries <- list()
run_case <- function(name, genes, scores, associations) {
  query <- data.frame(gene=genes, cluster=rep(name, length(genes)), score=scores)
  reference <- data.frame(celltype=associations[[1]], entrezid=associations[[2]])
  output <- native$test_fisher(query, reference, names(query))
  queries[[length(queries)+1L]] <<- data.frame(case=name, query)
  references[[length(references)+1L]] <<- data.frame(case=name, reference)
  summaries[[length(summaries)+1L]] <<- data.frame(
    case=name, query_rows=nrow(query), reference_rows=nrow(reference),
    reference_types=length(unique(reference$celltype)), tested_types=nrow(output))
  if(nrow(output)) {
    # This base-R ordering is the original Fisher process_results expression;
    # dplyr's surrounding data-frame assembly is not needed to verify ranking.
    ranked <- order(-output$p_adjust, abs(output$score), decreasing=TRUE, na.last=TRUE)
    output$rank <- match(seq_len(nrow(output)), ranked)
    output$method <- ifelse(output$rank==1L, "hard_fisher",
      ifelse(output$rank<=5L, "soft_fisher", "outside_top_five"))
    results[[length(results)+1L]] <<- data.frame(case=name, output)
  }
}
mixed <- list(c(rep("A",2),rep("B",3),rep("C",4),rep("D",11)),
  c("g1","g2","g2","g3","g4","g1","g4","g5","g6",paste0("g",8:18)))
run_case("mixed",c("g1","g2","g3","g7"),c(1,-3,5,2),mixed)
run_case("duplicate_query",c("g1","g1","g2","g2","g3"),c(1,5,-3,3,9),mixed)
duplicate_ref <- list(c("A","A","A","B","B",rep("C",9)),
  c("g1","g1","g2","g2","g3",paste0("z",1:9)))
run_case("duplicate_reference",c("g2","g1","g3"),c(-6,2,7),duplicate_ref)
run_case("exact_ties",c("g2","g1","g3"),c(-2,4,1),
  list(c(rep("First",2),rep("Second",2),rep("Other",10)),
       c("g1","g2","g1","g2",paste0("z",1:10))))
run_case("score_tiebreak",c("g1","g2"),c(2,-9),
  list(c("First","Second",rep("Other",10)),c("g1","g2",paste0("z",1:10))))
run_case("no_overlap",c("unknown1","unknown2"),c(2,-9),mixed)
run_case("equal_gate",c("g1","g2"),c(2,-9),list(c("A","A"),c("g1","g2")))
run_case("top_five",paste0("g",1:7),seq_len(7),
  list(c(paste0("Type",1:7),rep("Background",70)),c(paste0("g",1:7),paste0("z",1:70))))
run_case("small_tail",paste0("g",1:20),seq_len(20),
  list(c(rep("Enriched",20),rep("Background",1000)),
       c(paste0("g",1:20),paste0("z",1:1000))))

write.csv(do.call(rbind,queries),"tests/fixtures/easycelltype-queries.csv",row.names=FALSE)
write.csv(do.call(rbind,references),"tests/fixtures/easycelltype-reference.csv",row.names=FALSE)
write.csv(do.call(rbind,results),"tests/fixtures/easycelltype-results.csv",row.names=FALSE)
write.csv(do.call(rbind,summaries),"tests/fixtures/easycelltype-cases.csv",row.names=FALSE)
adjustment <- p.adjust(c(.01,NA,.04),method="BH")
stopifnot(isTRUE(all.equal(adjustment,c(.02,NA,.04))))
cat("Verified",length(summaries),"source cases and BH removal of NA hypotheses.\n")
