# Export the pinned EasyCellType 1.5.4 reference tables without its dependencies.
# Input: author R/sysdata.rda, Git blob 788349b5140d932a76b988441ffb01e250cb64e5.
# Verify the input SHA-256 before running:
# 845023954bf3fb6d7426bd7544e095cb7306b42eec5912f9733f27daac8be77b.
# Exports support local files and the pinned bundled-reference conversion.
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2L) stop("Supply sysdata.rda and a CSV output directory")
source <- new.env(parent = emptyenv())
load(args[[1L]], envir = source)
expected <- c(cellmarker = 49149L, clustermole = 172003L, panglao = 15067L)
stopifnot(setequal(ls(source), paste0(names(expected), "_db")))
dir.create(args[[2L]], recursive = TRUE, showWarnings = FALSE)
for (database in names(expected)) {
    value <- source[[paste0(database, "_db")]]
    stopifnot(is.data.frame(value), nrow(value) == expected[[database]])
    stopifnot(identical(names(value), c("celltype", "spe", "organ", "entrezid")))
    stopifnot(!anyNA(value), all(value$spe %in% c("Human", "Mouse")))
    # Plain data.frame removes tibble classes; columns and row order stay intact.
    value <- as.data.frame(value, stringsAsFactors = FALSE)
    for (column in names(value)) value[[column]] <- as.character(value[[column]])
    path <- file.path(args[[2L]], paste0(database, ".csv"))
    write.csv(value, path, row.names = FALSE, fileEncoding = "UTF-8", na = "")
    # Separate R-filtered rows anchor the Python species/tissue selection.
    for (species in c("Human", "Mouse")) {
        selected <- value[value$spe == species, , drop = FALSE]
        stopifnot(nrow(selected) > 0L)
        first_tissue <- selected$organ[[1L]]
        selected <- selected[selected$organ == first_tissue, , drop = FALSE]
        name <- paste(database, species, "first-tissue", sep = "-")
        write.csv(selected, file.path(args[[2L]], paste0(name, ".csv")),
                  row.names = FALSE, fileEncoding = "UTF-8", na = "")
    }
    cat(database, nrow(value), "rows exported\n")
}
