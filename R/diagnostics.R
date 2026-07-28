#' Sampler diagnostics for a fitted model
#'
#' Reports the HMC diagnostics CmdStan produces during sampling -- divergent
#' transitions, iterations that saturated the maximum tree depth, and E-BFMI --
#' together with the worst \eqn{\hat{R}} and lowest effective sample size across
#' all parameters.
#'
#' CmdStan prints these warnings to the console as it samples, but they are not
#' part of the draws, so they disappear once the console output scrolls away or
#' the fitted object is saved to disk. \code{epim()} copies them onto the fitted
#' model so a fit can be checked long after it was run.
#'
#' The mirror of this function in the Python port is
#' \code{epidemia.sampler_diagnostics()}, which reports the same quantities from
#' nutpie's sample statistics.
#'
#' @param object A fitted model of class \code{epimodel}.
#' @param ... Unused.
#'
#' @return An object of class \code{epidemia_diagnostics}: a list with
#'   \code{per_chain} (a data frame with one row per chain, giving the number of
#'   \code{divergent} transitions, the number of iterations at
#'   \code{max_treedepth}, and \code{ebfmi}), the number of sampling iterations
#'   per chain, and the worst \eqn{\hat{R}} / lowest effective sample size.
#'   Printing it summarises the totals and flags anything that warrants
#'   attention. Returns \code{NULL} with a message for variational fits, which
#'   have no NUTS diagnostics.
#'
#' @details
#' What the numbers mean:
#' \itemize{
#'   \item \strong{Divergent transitions} indicate the sampler could not follow
#'     the posterior's curvature. Even a handful biases the result, and they are
#'     not fixed by drawing more samples. Raise \code{adapt_delta}, or
#'     reparameterise.
#'   \item \strong{Max treedepth} is an efficiency problem rather than a
#'     correctness one: the sampler was still making progress when it hit the
#'     limit. Raise \code{max_treedepth}.
#'   \item \strong{E-BFMI} below roughly 0.2 suggests the momentum resampling is
#'     not exploring the energy distribution, often a sign of heavy tails.
#' }
#'
#' @seealso \code{\link{epim}}
#' @examples
#' \dontrun{
#' fm <- epim(...)
#' sampler_diagnostics(fm)
#' }
#' @export
sampler_diagnostics <- function(object, ...) {
  if (!is.epimodel(object)) {
    stop("'object' must be a fitted model returned by epim().", call. = FALSE)
  }
  diag <- object$stanfit$diagnostics
  if (is.null(diag)) {
    # Two quite different reasons for an absence, and saying the wrong one is
    # worse than saying nothing: a variational fit never had NUTS diagnostics,
    # whereas a sampled fit from before epidemia started retaining them did have
    # them and threw them away.
    if (identical(object$algorithm, "sampling")) {
      message("No NUTS diagnostics stored. This model was fitted before ",
              "epidemia retained them; refit to get them. CmdStan did report ",
              "them on the console at the time.")
    } else {
      message("No NUTS diagnostics: this model was fitted with algorithm = '",
              object$algorithm, "', which is not a Hamiltonian sampler and ",
              "does not produce them.")
    }
    return(invisible(NULL))
  }

  # Worst R-hat / lowest ESS, computed from the draws rather than stored, so
  # they stay correct if the draws are ever subset.
  s <- posterior::summarise_draws(object$stanfit$draws,
                                  rhat = posterior::rhat,
                                  ess_bulk = posterior::ess_bulk,
                                  ess_tail = posterior::ess_tail)
  s <- s[is.finite(s$rhat), , drop = FALSE]

  diag$worst_rhat      <- if (nrow(s)) max(s$rhat) else NA_real_
  diag$worst_rhat_par  <- if (nrow(s)) s$variable[which.max(s$rhat)] else NA_character_
  diag$min_ess_bulk    <- if (nrow(s)) min(s$ess_bulk) else NA_real_
  diag$min_ess_bulk_par <- if (nrow(s)) s$variable[which.min(s$ess_bulk)] else NA_character_
  diag$min_ess_tail    <- if (nrow(s)) min(s$ess_tail) else NA_real_
  diag
}

#' @rdname sampler_diagnostics
#' @param x An \code{epidemia_diagnostics} object.
#' @export
print.epidemia_diagnostics <- function(x, ...) {
  pc <- x$per_chain
  nchain <- nrow(pc)
  total <- if (is.na(x$iter_sampling)) NA_integer_ else nchain * x$iter_sampling

  cat("Sampler diagnostics\n")
  if (!is.na(total)) {
    cat(sprintf("%d chains x %d post-warmup draws = %d\n\n",
                nchain, x$iter_sampling, total))
  } else {
    cat(sprintf("%d chains\n\n", nchain))
  }
  print(pc, row.names = FALSE)

  pct <- function(n) if (is.na(total) || total == 0) "" else
    sprintf(" (%.1f%%)", 100 * n / total)
  ndiv <- sum(pc$divergent)
  ntd  <- sum(pc$max_treedepth)
  cat(sprintf("\nDivergent transitions: %d%s\n", ndiv, pct(ndiv)))
  cat(sprintf("Hit max treedepth:     %d%s\n", ntd, pct(ntd)))
  cat(sprintf("Lowest E-BFMI:         %.2f\n", min(pc$ebfmi)))
  if (!is.null(x$worst_rhat) && !is.na(x$worst_rhat)) {
    cat(sprintf("Worst R-hat:           %.3f  (%s)\n", x$worst_rhat, x$worst_rhat_par))
    cat(sprintf("Lowest bulk ESS:       %.0f  (%s)\n", x$min_ess_bulk, x$min_ess_bulk_par))
    cat(sprintf("Lowest tail ESS:       %.0f\n", x$min_ess_tail))
  }

  # Flag anything worth acting on. Thresholds follow the Stan warning guide.
  problems <- character(0)
  if (ndiv > 0) {
    problems <- c(problems, sprintf(
      "%d divergent transition%s. These bias the posterior and more draws will not help; raise adapt_delta above its current value or reparameterise.",
      ndiv, if (ndiv == 1) "" else "s"))
  }
  if (ntd > 0) {
    problems <- c(problems, sprintf(
      "%d iteration%s saturated max_treedepth. This costs efficiency rather than correctness; raise max_treedepth.",
      ntd, if (ntd == 1) "" else "s"))
  }
  if (min(pc$ebfmi) < 0.2) {
    problems <- c(problems, sprintf(
      "E-BFMI of %.2f is below 0.2, suggesting the sampler is not exploring the energy distribution well.",
      min(pc$ebfmi)))
  }
  # Compare the value as printed. Warning that "1.010 exceeds 1.01" because the
  # unrounded value is 1.0104 reads as a contradiction and trains people to
  # ignore the warning.
  if (!is.null(x$worst_rhat) && !is.na(x$worst_rhat) && round(x$worst_rhat, 3) > 1.01) {
    problems <- c(problems, sprintf(
      "R-hat of %.3f for %s exceeds 1.01, so the chains have not mixed.",
      x$worst_rhat, x$worst_rhat_par))
  }
  if (!is.null(x$min_ess_bulk) && !is.na(x$min_ess_bulk) && x$min_ess_bulk < 100 * nchain) {
    problems <- c(problems, sprintf(
      "Bulk ESS of %.0f for %s is %.0f per chain, below the 100 per chain that keeps posterior summaries stable.",
      x$min_ess_bulk, x$min_ess_bulk_par, x$min_ess_bulk / nchain))
  }

  if (length(problems)) {
    cat("\nWarnings:\n")
    for (p in problems) cat(strwrap(paste("*", p), exdent = 2), sep = "\n")
  } else {
    cat("\nNo problems detected.\n")
  }
  invisible(x)
}
