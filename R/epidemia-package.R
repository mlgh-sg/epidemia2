#' Flexible Epidemic Modeling with epidemia
#' 
#' @description The \pkg{epidemia} package allows researchers to flexibly 
#'      specify and fit Bayesian epidemiological models in the style of 
#'      \insertCite{Flaxman2020;textual}{epidemia}. The 
#'      package leverages R's formula interface to parameterize the reproduction rate 
#'      in terms of covariates, and allows pooling of parameters. 
#'      The design of the package has been inspired by, and borrowed from, the \pkg{rstanarm}
#'      package \insertCite{goodrich_2020}{epidemia}.
#'      \pkg{epidemia} uses \pkg{cmdstanr} as the backend for fitting the models, and the
#'      \pkg{posterior} package to represent posterior draws.
#'      The primary model fitting function in \pkg{epidemia} is \code{\link[epidemia]{epim}}.
#'
#' @docType package
#' @name epidemia-package
#' @aliases epidemia
#' @import methods
#' @import stats
#' @import rstantools
#' @importFrom lme4 ngrps
#' @importFrom Rdpack reprompt
#' @importFrom utils tail
#' @importFrom magrittr %>%
#' @importFrom rlang .data
#' @references
#' \insertAllCited()
#'
NULL

#' @export 
lme4::ngrps

#' @export
rstantools::posterior_predict

#' @export
rstantools::prior_summary
