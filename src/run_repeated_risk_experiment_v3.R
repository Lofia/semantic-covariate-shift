# run_repeated_risk_experiment.R
#
# Repeated target-risk estimation under the FIXED semantic covariate-shift DGP.
#
# Fixed across replicates:
#   - 2000-example finite target population
#   - semantic score S(T) = P_pilot(Business | T)
#   - fixed pilot classifier and its per-example loss
#   - b = 12
#   - source n = 1958
#   - target-X n = 2000
#
# Randomized within each replicate:
#   - IID target-population indices
#   - randomized empirical-PIT jitter
#   - Bernoulli source selection
#   - independent target-X sample
#
# Methods:
#   Naive, Oracle, uLSIF, RuLSIF, KLIEP, KMM, MM
#
# Default:
#   B = 20
#
# Run:
#   Rscript src/run_repeated_risk_experiment.R
#
# Later extend to B=100 without losing the first 20:
#   B=100 Rscript src/run_repeated_risk_experiment.R
#
# The script resumes from an existing replicate_results.csv.

suppressPackageStartupMessages({
  library(densityratio)
  library(densratio)
})

source("MM_function.R")

# ============================================================
# 1. Settings
# ============================================================

BASE_DIR <- "data/cov_shift_semantic_business_exp_b12"

LOOKUP_FILE <- file.path(
  BASE_DIR,
  "target_loss_lookup.csv"
)

OUTPUT_DIR <- file.path(
  BASE_DIR,
  "repeated_risk"
)

REPLICATE_FILE <- file.path(
  OUTPUT_DIR,
  "replicate_results_v3.csv"
)

SUMMARY_FILE <- file.path(
  OUTPUT_DIR,
  "summary_results.csv"
)

dir.create(
  OUTPUT_DIR,
  recursive = TRUE,
  showWarnings = FALSE
)

B <- as.integer(
  Sys.getenv(
    "B",
    "20"
  )
)

BASE_SEED <- as.integer(
  Sys.getenv(
    "BASE_SEED",
    "20260825"
  )
)

B_SHIFT <- 12

N_SOURCE <- 1958
N_TARGET_X <- 2000

NCENTERS <- 40
RULSIF_ALPHA <- 0.1

MM_ALPHA <- 0.7
MM_EPSILON <- 0.2

if (
  is.na(B)
  || B < 1
) {
  stop("B must be a positive integer.")
}


# ============================================================
# 2. Load fixed finite target population + loss lookup
# ============================================================

lookup <- read.csv(
  LOOKUP_FILE
)

required_columns <- c(
  "pool_index",
  "semantic_score",
  "cross_entropy",
  "correct"
)

if (
  !all(
    required_columns
    %in%
    names(lookup)
  )
) {
  stop(
    "target_loss_lookup.csv is missing required columns."
  )
}

lookup <- lookup[
  order(
    lookup$pool_index
  ),
]

row.names(
  lookup
) <- NULL

N_POP <- nrow(
  lookup
)

if (
  !identical(
    as.integer(
      lookup$pool_index
    ),
    0:(N_POP - 1)
  )
) {
  stop(
    "pool_index must be exactly 0,...,N-1."
  )
}

scores <- as.numeric(
  lookup$semantic_score
)

loss_lookup <- as.numeric(
  lookup$cross_entropy
)

correct_lookup <- as.numeric(
  lookup$correct
)

TRUE_TARGET_CE <- mean(
  loss_lookup
)

TRUE_TARGET_ACCURACY <- mean(
  correct_lookup
)

cat("============================================================\n")
cat("Fixed target population\n")
cat("============================================================\n")
cat("N population:", N_POP, "\n")
cat("Exact target CE:", TRUE_TARGET_CE, "\n")
cat("Exact target accuracy:", TRUE_TARGET_ACCURACY, "\n")
cat("Requested B:", B, "\n\n")


# ============================================================
# 3. Randomized empirical PIT setup
# ============================================================

unique_scores <- sort(
  unique(
    scores
  )
)

score_group <- match(
  scores,
  unique_scores
)

score_counts <- tabulate(
  score_group,
  nbins = length(
    unique_scores
  )
)

score_mass <- (
  score_counts
  /
  N_POP
)

score_lower <- (
  c(
    0,
    cumsum(
      score_mass
    )[
      -length(
        score_mass
      )
    ]
  )
)

lower_by_row <- score_lower[
  score_group
]

mass_by_row <- score_mass[
  score_group
]


draw_target_x <- function(
    n
) {

  # R indices are 1,...,N_POP.
  index <- sample.int(
    N_POP,
    size = n,
    replace = TRUE
  )

  jitter <- runif(
    n
  )

  u <- (
    lower_by_row[
      index
    ]
    +
    jitter
    *
    mass_by_row[
      index
    ]
  )

  # Protect against numerical log(0).
  u <- pmin(
    pmax(
      u,
      .Machine$double.eps
    ),
    1
    -
    .Machine$double.eps
  )

  x <- -log1p(
    -u
  )

  list(
    index = index,
    x = x
  )
}


selection_probability <- function(
    x,
    b = B_SHIFT
) {

  0.2 +
    0.8 *
    (10 * x + 1) /
    (10 * x + 1 + b)
}


TRUE_Z <- integrate(
  function(x) {
    selection_probability(
      x,
      B_SHIFT
    ) *
      dexp(
        x,
        rate = 1
      )
  },
  lower = 0,
  upper = Inf,
  subdivisions = 500L
)$value


# ============================================================
# 4. Draw exactly N_SOURCE IID observations from selected source
# ============================================================

draw_source <- function(
    n
) {

  accepted_index <- integer(0)
  accepted_x <- numeric(0)

  while (
    length(
      accepted_x
    )
    < n
  ) {

    remaining <- (
      n
      -
      length(
        accepted_x
      )
    )

    # Oversample enough candidates to avoid many while-loop iterations.
    batch_n <- max(
      200,
      ceiling(
        remaining
        /
        TRUE_Z
        *
        1.20
      )
    )

    candidate <- draw_target_x(
      batch_n
    )

    p <- selection_probability(
      candidate$x,
      B_SHIFT
    )

    keep <- (
      runif(
        batch_n
      )
      <
      p
    )

    if (
      any(
        keep
      )
    ) {

      accepted_index <- c(
        accepted_index,
        candidate$index[
          keep
        ]
      )

      accepted_x <- c(
        accepted_x,
        candidate$x[
          keep
        ]
      )
    }
  }

  # Truncating an accepted IID sequence at the first n preserves IID g draws.
  accepted_index <- accepted_index[
    seq_len(
      n
    )
  ]

  accepted_x <- accepted_x[
    seq_len(
      n
    )
  ]

  source_loss <- loss_lookup[
    accepted_index
  ]

  source_correct <- correct_lookup[
    accepted_index
  ]

  oracle_weight <- (
    TRUE_Z
    /
    selection_probability(
      accepted_x,
      B_SHIFT
    )
  )

  list(
    index = accepted_index,
    x = accepted_x,
    loss = source_loss,
    correct = source_correct,
    oracle_weight = oracle_weight
  )
}


# ============================================================
# 5. Weight/risk helpers
# ============================================================

normalize_mean_one <- function(
    w
) {
  w / mean(w)
}


weight_cv2 <- function(
    w
) {
  var(
    w
  ) /
    mean(
      w
    )^2
}


weight_ess <- function(
    w
) {
  sum(
    w
  )^2 /
    sum(
      w^2
    )
}


validate_weight <- function(
    w,
    n_expected
) {

  w <- as.numeric(
    w
  )

  if (
    length(
      w
    )
    !=
    n_expected
  ) {
    stop(
      "Wrong number of weights."
    )
  }

  if (
    any(
      !is.finite(
        w
      )
    )
  ) {
    stop(
      "Non-finite weights."
    )
  }

  if (
    any(
      w < 0
    )
  ) {
    stop(
      "Negative weights."
    )
  }

  if (
    sum(
      w
    )
    <=
    0
  ) {
    stop(
      "All weights are zero."
    )
  }

  w
}


risk_row <- function(
    replicate_id,
    method,
    source_loss,
    source_correct,
    weight = NULL,
    oracle_weight = NULL
) {

  n <- length(
    source_loss
  )

  if (
    is.null(
      weight
    )
  ) {

    estimate_ce <- mean(
      source_loss
    )

    estimate_accuracy <- mean(
      source_correct
    )

    return(
      data.frame(
        replicate = replicate_id,
        method = method,
        target_ce_estimate = estimate_ce,
        signed_error = estimate_ce - TRUE_TARGET_CE,
        absolute_error = abs(
          estimate_ce - TRUE_TARGET_CE
        ),
        squared_error = (
          estimate_ce - TRUE_TARGET_CE
        )^2,
        accuracy_estimate = estimate_accuracy,
        accuracy_absolute_error = abs(
          estimate_accuracy
          -
          TRUE_TARGET_ACCURACY
        ),
        mean_weight = 1,
        min_weight = 1,
        max_weight = 1,
        cv2 = 0,
        ess = n,
        weight_mse_vs_oracle = NA_real_,
        weight_corr_vs_oracle = NA_real_,
        status = "ok"
      )
    )
  }

  weight <- validate_weight(
    weight,
    n
  )

  estimate_ce <- (
    sum(
      weight
      *
      source_loss
    )
    /
    sum(
      weight
    )
  )

  estimate_accuracy <- (
    sum(
      weight
      *
      source_correct
    )
    /
    sum(
      weight
    )
  )

  w_norm <- normalize_mean_one(
    weight
  )

  if (
    !is.null(
      oracle_weight
    )
  ) {

    oracle_norm <- normalize_mean_one(
      oracle_weight
    )

    mse_w <- mean(
      (
        w_norm
        -
        oracle_norm
      )^2
    )

    corr_w <- suppressWarnings(
      cor(
        w_norm,
        oracle_norm
      )
    )

  } else {

    mse_w <- NA_real_
    corr_w <- NA_real_
  }

  data.frame(
    replicate = replicate_id,
    method = method,
    target_ce_estimate = estimate_ce,
    signed_error = estimate_ce - TRUE_TARGET_CE,
    absolute_error = abs(
      estimate_ce - TRUE_TARGET_CE
    ),
    squared_error = (
      estimate_ce - TRUE_TARGET_CE
    )^2,
    accuracy_estimate = estimate_accuracy,
    accuracy_absolute_error = abs(
      estimate_accuracy
      -
      TRUE_TARGET_ACCURACY
    ),
    mean_weight = mean(
      weight
    ),
    min_weight = min(
      weight
    ),
    max_weight = max(
      weight
    ),
    cv2 = weight_cv2(
      weight
    ),
    ess = weight_ess(
      weight
    ),
    weight_mse_vs_oracle = mse_w,
    weight_corr_vs_oracle = corr_w,
    status = "ok"
  )
}


failed_row <- function(
    replicate_id,
    method,
    message
) {

  data.frame(
    replicate = replicate_id,
    method = method,
    target_ce_estimate = NA_real_,
    signed_error = NA_real_,
    absolute_error = NA_real_,
    squared_error = NA_real_,
    accuracy_estimate = NA_real_,
    accuracy_absolute_error = NA_real_,
    mean_weight = NA_real_,
    min_weight = NA_real_,
    max_weight = NA_real_,
    cv2 = NA_real_,
    ess = NA_real_,
    weight_mse_vs_oracle = NA_real_,
    weight_corr_vs_oracle = NA_real_,
    status = paste0(
      "ERROR: ",
      message
    )
  )
}


# ============================================================
# 6. Density-ratio estimators
# ============================================================

fit_ulsif_weights <- function(
    x_source,
    x_target
) {

  fit <- densityratio::ulsif(
    data.frame(
      x = x_target
    ),
    data.frame(
      x = x_source
    ),
    ncenters = NCENTERS
  )

  predict(
    fit,
    newdata = data.frame(
      x = x_source
    )
  )
}


fit_rulsif_weights <- function(
    x_source,
    x_target
) {

  fit <- densratio::densratio(
    x_target,
    x_source,
    method = "RuLSIF",
    alpha = RULSIF_ALPHA,
    kernel_num = NCENTERS,
    verbose = FALSE
  )

  r_alpha <- as.numeric(
    fit$compute_density_ratio(
      x_source
    )
  )

  denominator <- (
    1
    -
    RULSIF_ALPHA
    *
    r_alpha
  )

  if (
    any(
      denominator <= 0
    )
  ) {
    stop(
      "RuLSIF conversion denominator <= 0."
    )
  }

  (
    (1 - RULSIF_ALPHA)
    *
    r_alpha
    /
    denominator
  )
}


fit_kliep_weights <- function(
    x_source,
    x_target
) {

  fit <- densityratio::kliep(
    data.frame(
      x = x_target
    ),
    data.frame(
      x = x_source
    ),
    ncenters = NCENTERS,
    progressbar = FALSE
  )

  predict(
    fit,
    newdata = data.frame(
      x = x_source
    )
  )
}


fit_kmm_weights <- function(
    x_source,
    x_target
) {

  fit <- densityratio::kmm(
    data.frame(
      x = x_target
    ),
    data.frame(
      x = x_source
    ),
    ncenters = NCENTERS
  )

  predict(
    fit,
    newdata = data.frame(
      x = x_source
    )
  )
}


fit_mm_weights <- function(
    x_source,
    x_target
) {

  fit <- MM(
    x_source,
    x_target,
    setting = "exp",
    alpha = MM_ALPHA,
    epsilon = MM_EPSILON
  )

  if (
    is.null(
      fit$v
    )
  ) {
    stop(
      "MM result has no $v."
    )
  }

  v_hat_ordered <- as.numeric(
    fit$v
  )

  if (
    length(
      v_hat_ordered
    )
    !=
    length(
      x_source
    )
  ) {
    stop(
      "MM $v has wrong length."
    )
  }

  rank_index <- as.integer(
    rank(
      x_source,
      ties.method = "first"
    )
  )

  p_hat <- (
    0.8
    *
    v_hat_ordered[
      rank_index
    ]
    +
    0.2
  )

  if (
    any(
      !is.finite(
        p_hat
      )
    )
    ||
    any(
      p_hat <= 0
    )
  ) {
    stop(
      "Invalid MM estimated selection probabilities."
    )
  }

  1 / p_hat
}


# ============================================================
# 7. One replicate
# ============================================================

run_one_replicate <- function(
    replicate_id
) {

  cat(
    "\n============================================================\n"
  )

  cat(
    "Replicate",
    replicate_id,
    "of",
    B,
    "\n"
  )

  cat(
    "============================================================\n"
  )

  # Each replicate is reproducible independently.
  set.seed(
    BASE_SEED
    +
    replicate_id
  )

  source <- draw_source(
    N_SOURCE
  )

  target <- draw_target_x(
    N_TARGET_X
  )

  cat(
    "Source X mean:",
    mean(
      source$x
    ),
    "| Target X mean:",
    mean(
      target$x
    ),
    "\n"
  )

  rows <- list()

  rows[["Naive source"]] <- risk_row(
      replicate_id,
      "Naive source",
      source$loss,
      source$correct
    )

  rows[["Oracle"]] <- risk_row(
      replicate_id,
      "Oracle",
      source$loss,
      source$correct,
      weight = source$oracle_weight,
      oracle_weight = source$oracle_weight
    )

  method_specs <- list(

    uLSIF = list(
      fun = fit_ulsif_weights,
      seed_offset = 1000
    ),

    RuLSIF = list(
      fun = fit_rulsif_weights,
      seed_offset = 2000
    ),

    KLIEP = list(
      fun = fit_kliep_weights,
      seed_offset = 3000
    ),

    KMM = list(
      fun = fit_kmm_weights,
      seed_offset = 4000
    ),

    MM = list(
      fun = fit_mm_weights,
      seed_offset = 5000
    )
  )

  for (
    method_name
    in
    names(
      method_specs
    )
  ) {

    cat(
      "  ",
      method_name,
      "... "
    )

    spec <- method_specs[[method_name]]

    result <- tryCatch(

      {

        # Fix estimator-internal randomness separately.
        set.seed(
          BASE_SEED
          +
          spec$seed_offset
          +
          replicate_id
        )

        w <- spec$fun(
          source$x,
          target$x
        )

        w <- validate_weight(
          w,
          N_SOURCE
        )

        ans <- risk_row(
          replicate_id,
          method_name,
          source$loss,
          source$correct,
          weight = w,
          oracle_weight = source$oracle_weight
        )

        cat(
          "CE =",
          sprintf(
            "%.6f",
            ans$target_ce_estimate
          ),
          "\n"
        )

        ans
      },

      error = function(e) {

        cat(
          "FAILED:",
          conditionMessage(
            e
          ),
          "\n"
        )

        failed_row(
          replicate_id,
          method_name,
          conditionMessage(
            e
          )
        )
      }
    )

    rows[[method_name]] <- result
  }

  do.call(
    rbind,
    rows
  )
}


# ============================================================
# 8. Resume logic
# ============================================================

METHOD_NAMES <- c(
  "Naive source",
  "Oracle",
  "uLSIF",
  "RuLSIF",
  "KLIEP",
  "KMM",
  "MM"
)

completed_replicates <- integer(
  0
)

existing <- NULL

if (
  file.exists(
    REPLICATE_FILE
  )
) {

  existing <- read.csv(
    REPLICATE_FILE,
    stringsAsFactors = FALSE
  )

  # A replicate counts as complete only if all method rows are present.
  replicate_counts <- table(
    existing$replicate
  )

  completed_replicates <- as.integer(
    names(
      replicate_counts[
        replicate_counts
        >=
        length(
          METHOD_NAMES
        )
      ]
    )
  )

  cat(
    "Existing complete replicates:",
    length(
      completed_replicates
    ),
    "\n"
  )
}


for (
  replicate_id
  in
  seq_len(
    B
  )
) {

  if (
    replicate_id
    %in%
    completed_replicates
  ) {

    cat(
      "Skipping completed replicate",
      replicate_id,
      "\n"
    )

    next
  }

  result <- run_one_replicate(
    replicate_id
  )

  if (
    file.exists(
      REPLICATE_FILE
    )
  ) {

    write.table(
      result,
      REPLICATE_FILE,
      sep = ",",
      row.names = FALSE,
      col.names = FALSE,
      append = TRUE,
      quote = TRUE
    )

  } else {

    write.csv(
      result,
      REPLICATE_FILE,
      row.names = FALSE
    )
  }
}


# ============================================================
# 9. Summarize requested replicates 1,...,B
# ============================================================

all_results <- read.csv(
  REPLICATE_FILE,
  stringsAsFactors = FALSE
)

all_results <- all_results[
  all_results$replicate
  %in%
  seq_len(
    B
  ),
]

summarize_method <- function(
    method_name
) {

  dat <- all_results[
    all_results$method
    ==
    method_name,
  ]

  valid <- (
    dat$status
    ==
    "ok"
    &
    is.finite(
      dat$target_ce_estimate
    )
  )

  dat_valid <- dat[
    valid,
  ]

  if (
    nrow(
      dat_valid
    )
    ==
    0
  ) {

    return(
      data.frame(
        method = method_name,
        valid_B = 0,
        mean_target_ce = NA_real_,
        bias = NA_real_,
        sd_target_ce = NA_real_,
        mae = NA_real_,
        rmse = NA_real_,
        mean_accuracy = NA_real_,
        mean_accuracy_abs_error = NA_real_,
        mean_weight_mse = NA_real_,
        mean_weight_corr = NA_real_,
        mean_cv2 = NA_real_,
        mean_ess = NA_real_
      )
    )
  }

  mean_or_na <- function(
      x
  ) {

    if (
      all(
        is.na(
          x
        )
      )
    ) {
      return(
        NA_real_
      )
    }

    mean(
      x,
      na.rm = TRUE
    )
  }

  data.frame(
    method = method_name,

    valid_B = nrow(
      dat_valid
    ),

    mean_target_ce = mean(
      dat_valid$target_ce_estimate
    ),

    bias = mean(
      dat_valid$signed_error
    ),

    sd_target_ce = if (
      nrow(
        dat_valid
      )
      >
      1
    ) {
      sd(
        dat_valid$target_ce_estimate
      )
    } else {
      NA_real_
    },

    mae = mean(
      dat_valid$absolute_error
    ),

    rmse = sqrt(
      mean(
        dat_valid$squared_error
      )
    ),

    mean_accuracy = mean(
      dat_valid$accuracy_estimate
    ),

    mean_accuracy_abs_error = mean(
      dat_valid$accuracy_absolute_error
    ),

    mean_weight_mse = mean_or_na(
      dat_valid$weight_mse_vs_oracle
    ),

    mean_weight_corr = mean_or_na(
      dat_valid$weight_corr_vs_oracle
    ),

    mean_cv2 = mean_or_na(
      dat_valid$cv2
    ),

    mean_ess = mean_or_na(
      dat_valid$ess
    )
  )
}


summary_results <- do.call(
  rbind,
  lapply(
    METHOD_NAMES,
    summarize_method
  )
)

write.csv(
  summary_results,
  SUMMARY_FILE,
  row.names = FALSE
)

cat("\n============================================================\n")
cat("Repeated-experiment summary\n")
cat("============================================================\n")

print(
  summary_results,
  row.names = FALSE,
  digits = 6
)

cat(
  "\nReplicate-level results:",
  REPLICATE_FILE,
  "\n"
)

cat(
  "Summary:",
  SUMMARY_FILE,
  "\n"
)
