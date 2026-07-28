# If a is NULL (and Inf, respectively) return b, otherwise just return a
# @param a,b Objects
`%ORifNULL%` <- function(a, b) {
  if (is.null(a)) b else a
}


# Return names of the last dimension in a matrix/array (e.g. colnames if matrix)
#
# @param x A matrix or array
last_dimnames <- function(x) {
  ndim <- length(dim(x))
  dimnames(x)[[ndim]]
}

used.sampling <- function(x) {
  x$algorithm == "sampling"
}
used.variational <- function(x) {
  x$algorithm %in% c("meanfield", "fullrank")
}

# Maybe broadcast 
#
# @param x A vector or scalar.
# @param n Number of replications to possibly make. 
# @return If \code{x} has no length the \code{0} replicated \code{n} times is
#   returned. If \code{x} has length 1, the \code{x} replicated \code{n} times
#   is returned. Otherwise \code{x} itself is returned.
maybe_broadcast <- function(x, n) {
  if (!length(x)) {
    rep(0, times = n)
  } else if (length(x) == 1L) {
    rep(x, times = n)
  } else {
    x
  }
}

# Check and set scale parameters for priors
#
# @param scale Value of scale parameter (can be NULL).
# @param default Default value to use if \code{scale} is NULL.
# @param link String naming the link function or NULL.
# @return If a probit link is being used, \code{scale} (or \code{default} if
#   \code{scale} is NULL) is scaled by \code{dnorm(0) / dlogis(0)}. Otherwise
#   either \code{scale} or \code{default} is returned.
set_prior_scale <- function(scale, default, link) {
  stopifnot(is.numeric(default), is.character(link) || is.null(link))
  if (is.null(scale)) 
    scale <- default
  if (isTRUE(link == "probit"))
    scale <- scale * dnorm(0) / dlogis(0)
  
  return(scale)
}

make_stan_summary <- function(stanfit) {
  levs <- c(0.5, 0.8, 0.95)
  qq <- (1 - levs) / 2
  probs <- sort(c(0.5, qq, 1 - qq))
  summary(stanfit, probs = probs)$summary
}

check_reTrms <- function(reTrms) {
  stopifnot(is.list(reTrms))
  nms <- names(reTrms$cnms)
  dupes <- duplicated(nms)
  for (i in which(dupes)) {
    original <- reTrms$cnms[[nms[i]]]
    dupe <- reTrms$cnms[[i]]
    overlap <- dupe %in% original
    if (any(overlap))
      stop("epidemia does not permit formulas with duplicate group-specific terms.\n", 
           "In this case ", nms[i], " is used as a grouping factor multiple times and\n",
           dupe[overlap], " is included multiple times.\n", 
           "Consider using || or -1 in your formulas to prevent this from happening.", call.=FALSE)
  }
  return(invisible(NULL))
}


validate_rhat <- function(x) {
  stopifnot(is.numeric(x), !is.list(x), !is.array(x))
  if (any(x < 0, na.rm = TRUE)) {
    stop("All 'rhat' values must be positive.", .call = FALSE)
  }
  x
}


validate_neff_ratio <- function(x) {
  stopifnot(is.numeric(x), !is.list(x), !is.array(x))
  if (any(x < 0, na.rm = TRUE)) {
    stop("All neff ratios must be positive.", .call = FALSE)
  }
  x
}

# Consistent error message to use when something is only available for 
# models fit using MCMC
#
# @param what An optional message to prepend to the default message.
STOP_sampling_only <- function(what) {
  msg <- "only available for models fit using MCMC (algorithm='sampling')."
  if (!missing(what)) 
    msg <- paste(what, msg)
  stop(msg, call. = FALSE)
}

make_glmerControl <- function(..., ignore_lhs = FALSE, ignore_x_scale = FALSE) {
  lme4::glmerControl(check.nlev.gtreq.5 = "ignore",
                     check.nlev.gtr.1 = "stop",
                     check.nobs.vs.rankZ = "ignore",
                     check.nobs.vs.nlev = "ignore",
                     check.nobs.vs.nRE = "ignore", 
                     check.formula.LHS = if (ignore_lhs) "ignore" else "stop",
                     check.scaleX = if (ignore_x_scale) "ignore" else "warning",
                     ...)  
}


check_rhats <- function(rhats, threshold = 1.1, check_lp = FALSE) {
  if (!check_lp)
    rhats <- rhats[!names(rhats) %in% c("lp__", "log-posterior")]
  
  if (any(rhats > threshold, na.rm = TRUE))
    warning("Markov chains did not converge! Do not analyze results!",
            call. = FALSE, noBreaks. = TRUE)
}

# Warn at fit time about HMC problems, the way CmdStan does on the console.
#
# CmdStan's own warnings are console output: they vanish when the output scrolls
# or when a saved fit is reloaded. Re-raising them as R warnings means they
# survive `suppressMessages()`-heavy scripts and knitr chunks that hide messages
# but not warnings, and points at sampler_diagnostics() for the detail.
check_hmc_diagnostics <- function(diagnostics) {
  if (is.null(diagnostics)) return(invisible(NULL))
  pc <- diagnostics$per_chain
  ndiv <- sum(pc$divergent)
  ntd  <- sum(pc$max_treedepth)

  if (ndiv > 0) {
    warning(ndiv, " divergent transition", if (ndiv == 1) "" else "s",
            " after warmup. These bias the posterior and are not fixed by ",
            "drawing more samples: raise adapt_delta or reparameterise. ",
            "See sampler_diagnostics().",
            call. = FALSE, noBreaks. = TRUE)
  }
  if (ntd > 0) {
    warning(ntd, " iteration", if (ntd == 1) "" else "s",
            " saturated the maximum tree depth. This costs efficiency rather ",
            "than correctness: raise max_treedepth. See sampler_diagnostics().",
            call. = FALSE, noBreaks. = TRUE)
  }
  if (any(pc$ebfmi < 0.2, na.rm = TRUE)) {
    warning("E-BFMI of ", format(min(pc$ebfmi), digits = 2), " is below 0.2, ",
            "suggesting the sampler is not exploring the energy distribution ",
            "well. See sampler_diagnostics().",
            call. = FALSE, noBreaks. = TRUE)
  }
  invisible(NULL)
}


#' Get posterior sample size from a fitted model
#'
#' The total number of posterior draws retained across all chains, which is what
#' \code{spaghetti_*} plots use to decide how many paths they may draw.
#'
#' @param object An object of class \code{epimodel}
#' @return A single integer: the number of posterior draws, pooled over chains.
#' @export
posterior_sample_size <- function(object) {
  UseMethod("posterior_sample_size", object)
}

#' @rdname plot_linpred
#' @export
posterior_sample_size.epimodel <- function(object) {
  ndraws_epimodel_draws(object$stanfit)
}

#' Get a list of all observation types used in a model
#'
#' The response name of each \code{epiobs} model that was fitted -- for example
#' \code{c("deaths", "cases")} for a joint fit. These are the values the
#' \code{type} argument of \code{posterior_predict}, \code{plot_obs} and
#' \code{evaluate_forecast} accepts.
#'
#' @param object An object of class \code{epimodel}.
#' @return A character vector of observation type names, in the order the models
#'   were passed to \code{\link{epim}}.
#' @export
all_obs_types <- function(object) {
  UseMethod("all_obs_types", object)
}

#' @rdname plot_linpred
#' @export
all_obs_types.epimodel <- function(object) {
  return(sapply(object$obs, function(x) .get_obs(formula(x))))
}


# Methods from rstanarm for creating a linear predictor
#
# Offset removed
#
# @param beta, x A vector or matrix or parameter estimates.
# @param x Predictor matrix.
# @param offset Optional offset vector.
# @return A vector or matrix.
linear_predictor <- function(beta, x) {
  UseMethod("linear_predictor")
}

# expects beta a vector of point estimates (not a sample)
linear_predictor.default <- function(beta, x) {
  return(as.vector(if (NCOL(x) == 1L) x * beta else x %*% beta))
}

linear_predictor.matrix <- function(beta, x) {
  return(as.matrix(beta) %*% Matrix::t(x))
}

# From loo package
nlist <- function (...) 
{
  m <- match.call()
  out <- list(...)
  no_names <- is.null(names(out))
  has_name <- if (no_names) 
    FALSE
  else nzchar(names(out))
  if (all(has_name)) 
    return(out)
  nms <- as.character(m)[-1L]
  if (no_names) {
    names(out) <- nms
  }
  else {
    names(out)[!has_name] <- nms[!has_name]
  }
  return(out)
}
