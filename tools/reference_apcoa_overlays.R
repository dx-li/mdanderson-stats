# aPCoA 1.3 overlay references using cluster::pam and unchanged car::ellipse.
# car source: cran/car b6f80103057cfc3dff208cc2a68525c72d61a0ca (3.0-12).
# No car installation or graphical device is needed.
expr <- parse("research/raw/aPCoA/car-3.0-12-Ellipse.R")
selected <- vapply(expr, function(x) {
    is.call(x) && identical(x[[1L]], as.name("<-")) &&
        identical(x[[2L]], as.name("ellipse"))
}, logical(1L))
stopifnot(sum(selected) == 1L)
native <- new.env(parent = globalenv())
eval(expr[[which(selected)]], native)

index <- 1:14
features <- cbind(sin(index * .7) + index * .05, cos(index * .4), ((index * 7) %% 11)/5)
X <- matrix(index %% 3 + 1, ncol = 1)
groups <- rep(c("A", "B"), each = 7)
D <- as.matrix(dist(features))
J <- diag(length(index)) - 1/length(index)
G <- J %*% (-D^2/2) %*% J
H <- X %*% solve(t(X) %*% X) %*% t(X)
E <- (diag(length(index)) - H) %*% G %*% (diag(length(index)) - H)
write.csv(data.frame(index, group = groups, covariate = X[, 1], features),
          "tests/fixtures/apcoa-overlay-input.csv", row.names = FALSE)

coordinates <- summaries <- vertices <- list()
for (panel in c("original", "adjusted")) {
    gram <- if (panel == "original") G else E
    eig <- eigen(gram, symmetric = TRUE)
    scores <- sweep(eig$vectors[, 1:2, drop = FALSE], 2, sqrt(eig$values[1:2]), "*")
    # Match the Python display sign convention; distinct eigenvalues avoid rotations.
    for (axis in 1:2) {
        pivot <- which.max(abs(scores[, axis]))
        if (scores[pivot, axis] < 0) scores[, axis] <- -scores[, axis]
    }
    coordinates[[panel]] <- data.frame(panel, index, x = scores[, 1], y = scores[, 2])
    for (group in unique(groups)) {
        rows <- which(groups == group)
        profiles <- if (panel == "original") D[rows, rows] else E[rows, rows]
        medoid <- rows[cluster::pam(profiles, 1)$id.med]
        moments <- cov.wt(scores[rows, , drop = FALSE])
        radius <- sqrt(2 * qf(.95, 2, length(rows) - 1))
        shape <- moments$cov
        curve <- native$ellipse(moments$center, shape, radius, draw = FALSE, col = "black")
        key <- paste(panel, group)
        summaries[[key]] <- data.frame(panel, group, medoid,
            center_x = moments$center[1], center_y = moments$center[2],
            cov_xx = shape[1, 1], cov_xy = shape[1, 2], cov_yy = shape[2, 2], radius)
        vertices[[key]] <- data.frame(panel, group, vertex = seq_len(nrow(curve)),
                                      x = curve[, 1], y = curve[, 2])
    }
}
write.csv(do.call(rbind, coordinates), "tests/fixtures/apcoa-overlay-coordinates.csv", row.names = FALSE)
write.csv(do.call(rbind, summaries), "tests/fixtures/apcoa-overlay-summary.csv", row.names = FALSE)
write.csv(do.call(rbind, vertices), "tests/fixtures/apcoa-overlay-vertices.csv", row.names = FALSE)

ties <- list()
for (n in 2:6) {
    for (kind in c("constant", "path")) {
        profiles <- if (kind == "constant") matrix(0, n, n) else abs(outer(1:n, 1:n, "-"))
        ties[[paste(kind, n)]] <- data.frame(kind, n, medoid = cluster::pam(profiles, 1)$id.med)
    }
}
write.csv(do.call(rbind, ties), "tests/fixtures/apcoa-medoid-ties.csv", row.names = FALSE)
cat(R.version.string, "; cluster", as.character(packageVersion("cluster")), "\n")
cat("4 group/panel geometries; 208 ellipse vertices; 10 medoid tie cases\n")
