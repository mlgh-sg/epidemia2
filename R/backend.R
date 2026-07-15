# CmdStanR fitting backend and posterior-draws representation
#
# This file replaces the parts of epidemia that used to talk to rstan directly.
# Models are fit with CmdStanR; posterior draws are represented with the
# `posterior` package and wrapped in a small S3 class (`epimodel_draws`) that
# exposes the same as.matrix()/as.array()/names()/summary() interface the rest
# of the package expects from what used to be an rstan `stanfit` object.

# ---------------------------------------------------------------------------
# Argument translation (rstanarm/rstan-style -> CmdStanR)
# ---------------------------------------------------------------------------

# Number of cores to use for running chains in parallel. Running chains
# concurrently is the single cheapest speed-up and is on by default.
default_parallel_chains <- function(chains) {
  cores <- getOption("mc.cores", NULL)
  if (is.null(cores)) {
    cores <- tryCatch(parallel::detectCores(logical = FALSE),
                      error = function(e) 1L)
    if (is.na(cores) || cores < 1L) cores <- 1L
  }
  max(1L, min(as.integer(chains), as.integer(cores)))
}

# Translate the user's `...` (historically forwarded to rstan::sampling) into
# the argument list for CmdStanModel$sample().
#
# Recognised rstan-style names are mapped; `iter` (total, rstan convention) is
# split into warmup + sampling. `control = list(adapt_delta=, max_treedepth=)`
# is unpacked. `init_r` (set internally by epim) maps to CmdStanR's `init`.
translate_sample_args <- function(args) {
  a <- args
  control <- a$control %ORifNULL% list()

  chains  <- a$chains  %ORifNULL% 4L
  iter    <- a$iter    %ORifNULL% 2000L
  warmup  <- a$warmup  %ORifNULL% floor(iter / 2)
  sampling <- max(1L, as.integer(iter - warmup))

  init <- a$init %ORifNULL% a$init_r %ORifNULL% 1e-6

  out <- list(
    chains          = as.integer(chains),
    parallel_chains = a$cores %ORifNULL% default_parallel_chains(chains),
    iter_warmup     = as.integer(warmup),
    iter_sampling   = as.integer(sampling),
    thin            = a$thin %ORifNULL% 1L,
    seed            = a$seed,
    init            = init,
    refresh         = a$refresh,
    adapt_delta     = control$adapt_delta,
    max_treedepth   = control$max_treedepth,
    show_messages   = a$show_messages %ORifNULL% TRUE
  )
  # drop NULLs so CmdStanR uses its own defaults
  out[!vapply(out, is.null, logical(1))]
}

# Translate `...` into arguments for CmdStanModel$variational() (ADVI), used for
# algorithm = "meanfield" / "fullrank".
translate_vb_args <- function(args, algorithm) {
  a <- args
  out <- list(
    algorithm      = algorithm,
    iter           = a$iter %ORifNULL% 10000L,
    tol_rel_obj    = a$tol_rel_obj,
    seed           = a$seed,
    init           = a$init %ORifNULL% a$init_r %ORifNULL% 1e-6,
    output_samples = a$output_samples %ORifNULL% 1000L,
    refresh        = a$refresh
  )
  out[!vapply(out, is.null, logical(1))]
}

# Keep only the numeric Stan-data elements of the standata list. CmdStanR
# serialises the data to JSON and cannot handle the character / nested-list
# fields (e.g. `groups`, `rt_prior_info`) that epidemia stores alongside the
# real Stan data; CmdStan simply ignores any extra numeric variables it does
# not need.
#
# Ragged / array-of-vector Stan variables (such as `pvecs`, the infection-to-
# observation distributions, declared `array[R] vector[NS]`) are represented in
# R as a list of numeric vectors; these must be kept, as write_stan_json()
# serialises them correctly. Lists that contain any non-numeric element (the
# prior-info bookkeeping) are dropped.
clean_standata <- function(sdat, data_vars = NULL) {
  is_stan_data <- function(x) {
    if (is.logical(x) || is.numeric(x)) return(TRUE)
    if (is.list(x)) {
      return(length(x) > 0 &&
             all(vapply(x, function(e) is.numeric(e) || is.logical(e),
                        logical(1))))
    }
    FALSE
  }
  keep <- vapply(sdat, is_stan_data, logical(1))
  out <- sdat[keep]

  # Restrict to exactly the variables the Stan program declares as data. This
  # both avoids serialising R-only bookkeeping fields and prevents CmdStanR from
  # rejecting NA-valued housekeeping variables it would otherwise never use.
  if (!is.null(data_vars)) {
    out <- out[names(out) %in% data_vars]
  }

  # logicals -> integers (top level and within lists)
  out <- lapply(out, function(x) {
    if (is.logical(x)) return(x * 1L)
    if (is.list(x)) return(lapply(x, function(e) if (is.logical(e)) e * 1L else e))
    x
  })
  out
}

# Names of the data-block variables declared by a compiled CmdStanModel.
model_data_vars <- function(mod) {
  names(mod$variables()$data)
}

# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------

# Fit the base model with CmdStanR. Returns the raw CmdStanFit object.
#
# @param sdat The (full) standata list.
# @param algorithm One of "sampling", "meanfield", "fullrank".
# @param sampling_args The user's `...` list.
fit_cmdstan <- function(sdat, algorithm, sampling_args) {
  mod  <- epidemia_stan_model("epidemia_base")
  data <- clean_standata(sdat, model_data_vars(mod))

  if (algorithm == "sampling") {
    args <- translate_sample_args(sampling_args)
    args$data <- data
    fit <- do.call(mod$sample, args)
  } else {
    args <- translate_vb_args(sampling_args, algorithm)
    args$data <- data
    fit <- do.call(mod$variational, args)
  }
  fit
}

# ---------------------------------------------------------------------------
# Posterior-draws representation
# ---------------------------------------------------------------------------

# Container names (in pars()/new_names() order) that are actually present in the
# fit output with a non-zero number of elements. Zero-length Stan containers are
# not emitted by CmdStan, so intersecting with the fit's variables reproduces
# the same flat parameter set the positional renaming in new_names() expects.
present_pars <- function(fit, monitor) {
  avail <- fit$metadata()$stan_variables
  monitor[monitor %in% avail]
}

# Build the human-named posterior draws for a fitted model.
#
# Returns an `epimodel_draws` object holding a posterior draws_array whose
# variables have been renamed to epidemia's human-readable names, plus the raw
# (Stan-named) draws and the original flat names (needed by posterior_sims).
#
# theta_L parameters (Cholesky/scale parameterisation of the random-effects
# covariance) are transformed to the covariance-matrix entries Sigma[...] here,
# matching the previous rstan behaviour.
build_draws <- function(fit, sdat, rt, obs, data, monitor,
                        algorithm = "sampling") {
  mon <- present_pars(fit, monitor)

  # raw draws for exactly the monitored containers, in order, plus lp__
  raw <- fit$draws(variables = c(mon, "lp__"))
  raw <- posterior::as_draws_array(raw)

  # transform theta_L -> Sigma if group-specific covariance is present
  if (sdat$len_theta_L && "theta_L" %in% mon) {
    raw <- transform_theta_L_draws(raw, rt$group$cnms)
  }

  flat <- posterior::variables(raw)
  human <- new_names(sdat, rt, obs, fit, data)

  if (length(flat) != length(human)) {
    stop("Internal error: mismatch between monitored parameters (",
         length(flat), ") and generated names (", length(human), ").",
         call. = FALSE)
  }

  orig_names <- flat                       # Stan-side flat names (for gqs)
  named <- raw
  posterior::variables(named) <- human

  structure(
    list(
      draws      = named,       # posterior draws_array, human names
      raw_draws  = raw,         # posterior draws_array, Stan names
      orig_names = orig_names,
      algorithm  = algorithm
    ),
    class = "epimodel_draws"
  )
}

# Replace the theta_L columns of a draws_array with the corresponding
# covariance-matrix entries Sigma[...], computed per draw with lme4::mkVarCorr.
transform_theta_L_draws <- function(draws, cnms) {
  vars <- posterior::variables(draws)
  idx <- grep("^theta_L(\\[|$)", vars)
  if (!length(idx)) return(draws)

  nc  <- vapply(cnms, length, integer(1))
  nms <- names(cnms)

  # theta_L draws as [iteration, chain, param]
  thetas <- draws[, , idx, drop = FALSE]
  ta <- posterior::as_draws_array(thetas)
  arr <- as.array(ta)  # dims: iter x chain x nparam

  niter  <- dim(arr)[1]
  nchain <- dim(arr)[2]

  # for each (iter, chain) build the covariance and pull lower-tri (incl diag)
  sig <- apply(arr, c(1, 2), function(theta) {
    Sigma <- lme4::mkVarCorr(sc = 1, cnms = cnms, nc = nc, theta = theta,
                             nms = nms)
    unlist(lapply(Sigma, function(x) x[lower.tri(x, diag = TRUE)]))
  })
  # sig has dims: nSigma x iter x chain ; move to iter x chain x nSigma
  sig <- aperm(sig, c(2, 3, 1))

  arr[, , seq_len(dim(sig)[3])] <- sig
  draws[, , idx] <- posterior::as_draws_array(arr)
  draws
}

# ---------------------------------------------------------------------------
# S3 methods for the epimodel_draws wrapper (stand in for a stanfit object)
# ---------------------------------------------------------------------------

#' @export
as.matrix.epimodel_draws <- function(x, ...) {
  dm <- posterior::as_draws_matrix(x$draws)
  # return a *plain* base matrix. Keeping the `draws_matrix` S3/S4 classes leaks
  # into downstream matrix algebra (e.g. `draws_matrix %*% <sparse Matrix>` in
  # the random-effect / random-walk linear predictors) and sends S4 dispatch
  # into infinite recursion.
  out <- matrix(as.numeric(dm), nrow = nrow(dm), ncol = ncol(dm))
  dimnames(out) <- list(NULL, posterior::variables(dm))
  out
}

#' @export
as.array.epimodel_draws <- function(x, ...) {
  da <- posterior::as_draws_array(x$draws)
  # plain base array with dims [iterations, chains, parameters]
  out <- array(as.numeric(da), dim = dim(da))
  dimnames(out) <- list(iterations = NULL,
                        chains = dimnames(da)[[2]],
                        parameters = posterior::variables(da))
  out
}

#' @export
names.epimodel_draws <- function(x) {
  posterior::variables(x$draws)
}

#' @export
dim.epimodel_draws <- function(x) {
  m <- posterior::as_draws_matrix(x$draws)
  dim(m)
}

#' @export
summary.epimodel_draws <- function(object, probs = c(0.5), ...) {
  list(summary = epidemia_summary(object$draws, probs = probs,
                                  sampling = object$algorithm == "sampling"))
}

# Number of posterior draws (across all chains).
ndraws_epimodel_draws <- function(x) {
  posterior::ndraws(posterior::as_draws_matrix(x$draws))
}

# ---------------------------------------------------------------------------
# Summary matrix (rstan::summary(...)$summary lookalike)
# ---------------------------------------------------------------------------

# Produce a summary matrix with the same layout rstan::summary()$summary
# returned: rownames = parameter names, columns = mean, se_mean, sd, the
# requested quantiles (named e.g. "2.5%"), and (for MCMC) n_eff and Rhat.
epidemia_summary <- function(draws, probs = c(0.5), sampling = TRUE) {
  draws <- posterior::as_draws_array(draws)
  vars  <- posterior::variables(draws)

  qfun <- function(x) stats::quantile(x, probs = probs, names = FALSE,
                                      na.rm = TRUE)

  s <- posterior::summarise_draws(
    draws,
    mean = ~mean(.x, na.rm = TRUE),
    sd   = ~stats::sd(.x, na.rm = TRUE),
    mcse_mean = posterior::mcse_mean,
    quantiles = qfun,
    ess_bulk = posterior::ess_bulk,
    rhat = posterior::rhat
  )

  qnames <- paste0(formatC(probs * 100, format = "g"), "%")
  qmat <- as.matrix(s[, grep("^quantiles", colnames(s)), drop = FALSE])
  colnames(qmat) <- qnames

  out <- cbind(
    mean    = s$mean,
    se_mean = s$mcse_mean,
    sd      = s$sd,
    qmat,
    n_eff   = s$ess_bulk,
    Rhat    = s$rhat
  )
  rownames(out) <- vars

  if (!sampling) {
    # variational fits have no meaningful convergence diagnostics
    out[, "n_eff"] <- NA_real_
    out[, "Rhat"]  <- NA_real_
  }
  out
}

# ---------------------------------------------------------------------------
# Sparse-matrix parts (replacement for rstan::extract_sparse_parts)
# ---------------------------------------------------------------------------

# Extract the compressed-sparse-row components of a matrix, matching the output
# of rstan::extract_sparse_parts(): a list with `w` (non-zero values), `v`
# (1-based column indices) and `u` (1-based row-start pointers, length nrow+1).
extract_sparse_parts <- function(A) {
  A <- methods::as(methods::as(A, "CsparseMatrix"), "RsparseMatrix")
  list(
    w = A@x,
    v = A@j + 1L,
    u = A@p + 1L
  )
}
