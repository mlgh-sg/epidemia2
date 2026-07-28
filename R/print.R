#' Print fitted model details
#' 
#' Prints estimated regression parameters, and other model parameters.
#' Similar to printing of \code{rstan::stanreg} objects.
#' 
#' @templateVar epimodelArg x
#' @template args-epimodel-object
#' @param digits Number of decimal places to print.
#' @param ... Not used.
#' @export
#' @return No return value.
print.epimodel <- function(x, digits=1, ...) {

  mixed <- is.mixed(x)
  mat <- as.matrix(x)
  nms <- setdiff(rownames(x$stan_summary), "log-posterior")

  # remove group effects
  if (mixed) 
    nms <- setdiff(nms, grep("^R\\|b\\[", nms, value = TRUE))

  coef_mat <- mat[, nms, drop = FALSE]
  estimates <- .median_and_madsd(coef_mat)

  if (mixed) {
    estimates <- estimates[!grepl("^R\\|Sigma\\[", rownames(estimates)),, drop=FALSE]
  }

  inf_pars <- c("seeds", "seeds_aux", "inf_aux", "rm_noise")
  inf_pars <- paste(paste0("^", inf_pars), collapse="|")
  inf_pars <- grepl(inf_pars, rownames(estimates))
  estimates_reg <- estimates[!inf_pars,, drop=FALSE]
  estimates_inf <- estimates[inf_pars,, drop=FALSE]

  cat("\nRt regression parameters:\n")
  cat("==========")
  cat("\ncoefficients:\n")
  nms <- grep("^R\\|", rownames(estimates_reg), value=T)
  mat <- estimates_reg[nms,,drop=FALSE]
  if(length(mat))
    .printfr(mat, digits)

  if (mixed) {
    cat("\nError terms:\n")
    print(lme4::VarCorr(x), digits = digits + 1)
    cat("\nNum. levels:", 
        paste(names(ngrps(x)), unname(ngrps(x)), collapse = ", "), "\n")
  }

  for(obs in x$obs) {
  nme <- .get_obs(formula(obs))
  cat("\n", nme, " regression parameters:\n")
  cat("==========")
  cat("\ncoefficients:\n")
  nms <- grep(paste0("^", nme, "\\|"), rownames(estimates_reg), value=T)
  mat <- estimates_reg[nms,,drop=FALSE]
  if (length(mat))
    .printfr(mat, digits)

  
} 
  cat("\nInfection model parameters:\n")
  cat("==========\n")
  .printfr(estimates_inf, digits)

  print_diagnostic_footer(x)

   invisible(x)
}

# One line at the foot of print.epimodel when the sampler had trouble.
#
# Printing a model is the moment someone is about to read numbers off it, so a
# sampling problem should be visible here rather than only on request. Stays
# silent for a clean fit -- a footer that always appears stops being read.
print_diagnostic_footer <- function(x) {
  diag <- x$stanfit$diagnostics
  if (is.null(diag)) return(invisible(NULL))
  pc <- diag$per_chain
  bad <- c(
    if (sum(pc$divergent)) paste0(sum(pc$divergent), " divergent transitions"),
    if (sum(pc$max_treedepth)) paste0(sum(pc$max_treedepth), " iterations at max treedepth"),
    if (any(pc$ebfmi < 0.2, na.rm = TRUE)) "low E-BFMI"
  )
  if (length(bad)) {
    cat("\nSampler warnings: ", paste(bad, collapse = ", "),
        ". See sampler_diagnostics().\n", sep = "")
  }
  invisible(NULL)
}

