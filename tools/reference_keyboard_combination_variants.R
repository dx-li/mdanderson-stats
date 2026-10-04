#!/usr/bin/env Rscript

# Independent base-R references for KeyboardComb key2/key3/key4 movement
# probabilities. Equations follow the cached combination paper §§2–3.

args <- commandArgs(trailingOnly = TRUE)
out_dir <- if (length(args)) args[[1]] else getwd()
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

new_case <- function(name, move, current, edits, excluded = character()) {
  patients <- matrix(0, nrow = 3, ncol = 3)
  toxicities <- matrix(0, nrow = 3, ncol = 3)
  for (edit in edits) {
    i <- edit[[1]]
    j <- edit[[2]]
    patients[i, j] <- edit[[3]]
    toxicities[i, j] <- edit[[4]]
  }
  mask <- matrix(FALSE, nrow = 3, ncol = 3)
  for (dose in excluded) {
    ij <- as.integer(strsplit(dose, ",", fixed = TRUE)[[1]])
    mask[ij[[1]], ij[[2]]] <- TRUE
  }
  list(name = name, move = move, current = current, patients = patients,
       toxicities = toxicities, excluded = mask)
}

cases <- list(
  new_case(
    "escalate_diagonal_competition", "escalate", c(1, 1),
    list(c(1, 1, 1, 0), c(2, 1, 10, 3), c(2, 2, 100, 30))
  ),
  new_case(
    "deescalate_diagonal_competition", "deescalate", c(2, 2),
    list(c(2, 2, 1, 1), c(1, 2, 10, 0), c(2, 1, 10, 0), c(1, 1, 100, 30))
  ),
  new_case(
    "masked_escalation", "escalate", c(1, 1),
    list(c(1, 1, 1, 0), c(1, 2, 0, 0), c(2, 2, 100, 30)), excluded = "2,1"
  ),
  new_case(
    "boundary_escalation", "escalate", c(1, 3),
    list(c(1, 3, 1, 0), c(2, 3, 10, 3))
  ),
  new_case(
    "uniform_axial_tie", "escalate", c(1, 1),
    list(c(1, 1, 1, 0))
  )
)

target <- 0.30
lower <- 0.25
upper <- 0.35
algorithm_candidates <- list(
  key2 = list(escalate = rbind(c(1, 0), c(0, 1)), deescalate = rbind(c(-1, 0), c(0, -1), c(-1, -1))),
  key3 = list(escalate = rbind(c(1, 0), c(0, 1), c(1, 1)), deescalate = rbind(c(-1, 0), c(0, -1), c(-1, -1))),
  key4 = list(escalate = rbind(c(1, 0), c(0, 1)), deescalate = rbind(c(-1, 0), c(0, -1)))
)

candidate_rows <- list()
decision_rows <- list()
config_rows <- list()
for (z in cases) {
  current <- z$current
  config_rows[[length(config_rows) + 1L]] <- data.frame(
    case = z$name, movement = z$move,
    current_a = current[[1]], current_b = current[[2]],
    target = target, lower = lower, upper = upper,
    patients_row_major = paste(as.vector(t(z$patients)), collapse = ";"),
    toxicities_row_major = paste(as.vector(t(z$toxicities)), collapse = ";"),
    excluded_row_major = paste(as.integer(as.vector(t(z$excluded))), collapse = ";"),
    stringsAsFactors = FALSE
  )
  for (algorithm in names(algorithm_candidates)) {
    offsets <- algorithm_candidates[[algorithm]][[z$move]]
    raw_candidates <- sweep(offsets, 2, current, "+")
    inside <- raw_candidates[, 1] >= 1 & raw_candidates[, 1] <= 3 &
      raw_candidates[, 2] >= 1 & raw_candidates[, 2] <= 3
    doses <- raw_candidates[inside, , drop = FALSE]
    if (nrow(doses)) {
      allowed <- !vapply(seq_len(nrow(doses)), function(i) {
        z$excluded[doses[i, 1], doses[i, 2]]
      }, logical(1))
      doses <- doses[allowed, , drop = FALSE]
    }
    masses <- if (nrow(doses)) vapply(seq_len(nrow(doses)), function(i) {
      a <- z$toxicities[doses[i, 1], doses[i, 2]] + 1
      b <- z$patients[doses[i, 1], doses[i, 2]] - z$toxicities[doses[i, 1], doses[i, 2]] + 1
      pbeta(upper, a, b) - pbeta(lower, a, b)
    }, numeric(1)) else numeric()
    probabilities <- numeric(length(masses))
    if (length(masses)) {
      if (algorithm == "key4") {
        probabilities <- masses / sum(masses)
      } else {
        winners <- masses == max(masses)
        probabilities[winners] <- 1 / sum(winners)
      }
      for (i in seq_along(masses)) {
        dose_text <- paste(doses[i, ], collapse = ",")
        candidate_rows[[length(candidate_rows) + 1L]] <- data.frame(
          case = z$name, algorithm = algorithm, movement = z$move,
          candidate_a = doses[i, 1], candidate_b = doses[i, 2],
          posterior_a = z$toxicities[doses[i, 1], doses[i, 2]] + 1,
          posterior_b = z$patients[doses[i, 1], doses[i, 2]] - z$toxicities[doses[i, 1], doses[i, 2]] + 1,
          raw_target_key_mass = masses[i], selection_probability = probabilities[i],
          stringsAsFactors = FALSE
        )
      }
    }
    selected <- which(probabilities > 0)
    decision_rows[[length(decision_rows) + 1L]] <- data.frame(
      case = z$name, algorithm = algorithm, movement = z$move,
      eligible_candidates = paste(apply(doses, 1, paste, collapse = ","), collapse = ";"),
      selected_candidates = paste(apply(doses[selected, , drop = FALSE], 1, paste, collapse = ","), collapse = ";"),
      candidate_count = nrow(doses),
      stringsAsFactors = FALSE
    )
  }
}

write.csv(do.call(rbind, config_rows), file.path(out_dir, "keyboard-combination-variants-config.csv"), row.names = FALSE)
write.csv(do.call(rbind, candidate_rows), file.path(out_dir, "keyboard-combination-variants-masses.csv"), row.names = FALSE)
write.csv(do.call(rbind, decision_rows), file.path(out_dir, "keyboard-combination-variants-decisions.csv"), row.names = FALSE)
