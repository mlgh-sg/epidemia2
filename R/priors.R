# Prior distributions for epidemia
#
# These prior-constructor functions are vendored from rstanarm (Goodrich,
# Gabry, Ali and Brilleman) so that epidemia does not depend on rstanarm at
# run time. They are thin builders that return a named list describing a prior;
# the list is parsed internally by epim() and translated into the integer
# `prior_dist` codes understood by the Stan program. Behaviour is identical to
# the corresponding rstanarm functions. rstanarm is licensed GPL-3 (the same as
# epidemia) and its authors are credited as contributors in DESCRIPTION.
#
# The set of prior *families* that can actually be used is fixed by the Stan
# model (see ok_dists / ok_int_dists / ok_aux_dists / ok_cov_dists in
# utilities.R); these constructors are the R front end to that set. Users retain
# full control over the hyperparameters of each family.

#' Prior distributions and options
#'
#' @description These functions specify prior distributions for the modelling
#' functions \code{\link{epirt}} and \code{\link{epiobs}} (and, for some, in
#' \code{\link{epiinf}}). They construct lightweight lists that are interpreted
#' internally by \code{\link{epim}}. The functions and their arguments mirror
#' the prior helpers in \pkg{rstanarm}, and are provided directly by
#' \pkg{epidemia} so that \pkg{rstanarm} is not required.
#'
#' @details
#' The functions return a named list that epidemia parses into the Stan
#' program's prior representation. Which distributions are permitted depends on
#' the role of the parameter:
#' \itemize{
#'   \item \strong{Regression coefficients} (\code{prior} in \code{epirt}
#'     and \code{epiobs}): \code{normal}, \code{student_t}, \code{cauchy},
#'     \code{hs}, \code{hs_plus}, \code{laplace}, \code{lasso},
#'     \code{product_normal}, and the epidemia-specific
#'     \code{\link{shifted_gamma}}.
#'   \item \strong{Intercepts} (\code{prior_intercept}): \code{normal},
#'     \code{student_t}, \code{cauchy}.
#'   \item \strong{Auxiliary parameters} (\code{prior_aux}): \code{normal},
#'     \code{student_t}, \code{cauchy}, \code{exponential}.
#'   \item \strong{Covariance of group-specific terms}
#'     (\code{prior_covariance}): \code{decov}, \code{lkj}.
#' }
#'
#' @param location Prior location. For \code{normal} and \code{student_t} (and
#'   so \code{cauchy}) this is the prior mean. Defaults to \code{0}.
#' @param scale Prior scale. A positive number (or \code{NULL} to use a sensible
#'   internal default, in which case the scale may be rescaled if
#'   \code{autoscale = TRUE}).
#' @param df,df1,df2 Prior degrees of freedom. For \code{student_t} a single
#'   positive number; for \code{product_normal} an integer \eqn{\ge 1} giving
#'   the number of normal factors; for \code{hs_plus}, \code{df1} and
#'   \code{df2} are the degrees of freedom for the local shrinkage parameters.
#' @param autoscale If \code{TRUE}, the scale is adjusted automatically
#'   according to the scale of the predictors. See the priors vignette and
#'   \code{\link{prior_summary}}.
#' @param rate Prior rate for the \code{exponential} distribution (a positive
#'   number). The scale is the reciprocal of the rate.
#' @param global_df,global_scale,slab_df,slab_scale Hyperparameters for the
#'   regularised horseshoe priors \code{hs} and \code{hs_plus}. See Piironen and
#'   Vehtari (2017).
#' @param regularization Exponent for an LKJ prior on the correlation matrix.
#' @param concentration Concentration parameter for a symmetric Dirichlet
#'   distribution over the relative variances of the group-specific terms
#'   (\code{decov}).
#' @param shape Shape parameter for the Gamma prior on the standard deviation of
#'   group-specific terms (\code{decov}).
#'
#' @return A named list to be used internally by \code{\link{epim}}.
#'
#' @name priors
#' @seealso \code{\link{shifted_gamma}}, \code{\link{hexp}},
#'   \code{\link{prior_summary}}
#'
#' @references
#' Piironen, J., and Vehtari, A. (2017). Sparsity information and regularization
#' in the horseshoe and other shrinkage priors. \emph{Electronic Journal of
#' Statistics}. 11(2), 5018-5051.
NULL

#' @rdname priors
#' @export
normal <- function(location = 0, scale = NULL, autoscale = FALSE) {
  validate_parameter_value(scale)
  nlist(dist = "normal", df = NA, location, scale, autoscale)
}

#' @rdname priors
#' @export
student_t <- function(df = 1, location = 0, scale = NULL, autoscale = FALSE) {
  validate_parameter_value(scale)
  validate_parameter_value(df)
  nlist(dist = "t", df, location, scale, autoscale)
}

#' @rdname priors
#' @export
cauchy <- function(location = 0, scale = NULL, autoscale = FALSE) {
  student_t(df = 1, location = location, scale = scale, autoscale)
}

#' @rdname priors
#' @export
exponential <- function(rate = 1, autoscale = FALSE) {
  stopifnot(length(rate) == 1)
  validate_parameter_value(rate)
  nlist(dist = "exponential", df = NA, location = NA, scale = 1 / rate,
        autoscale)
}

#' @rdname priors
#' @export
laplace <- function(location = 0, scale = NULL, autoscale = FALSE) {
  nlist(dist = "laplace", df = NA, location, scale, autoscale)
}

#' @rdname priors
#' @export
lasso <- function(df = 1, location = 0, scale = NULL, autoscale = FALSE) {
  nlist(dist = "lasso", df, location, scale, autoscale)
}

#' @rdname priors
#' @export
hs <- function(df = 1, global_df = 1, global_scale = 0.01, slab_df = 4,
               slab_scale = 2.5) {
  validate_parameter_value(df)
  validate_parameter_value(global_df)
  validate_parameter_value(global_scale)
  validate_parameter_value(slab_df)
  validate_parameter_value(slab_scale)
  nlist(dist = "hs", df, location = 0, scale = 1, global_df,
        global_scale, slab_df, slab_scale)
}

#' @rdname priors
#' @export
hs_plus <- function(df1 = 1, df2 = 1, global_df = 1, global_scale = 0.01,
                    slab_df = 4, slab_scale = 2.5) {
  validate_parameter_value(df1)
  validate_parameter_value(df2)
  validate_parameter_value(global_df)
  validate_parameter_value(global_scale)
  validate_parameter_value(slab_df)
  validate_parameter_value(slab_scale)
  # dist_shape is 2 * df1
  nlist(dist = "hs_plus", df = df1, location = 0, scale = df2,
        global_df, global_scale, slab_df, slab_scale)
}

#' @rdname priors
#' @export
product_normal <- function(df = 2, location = 0, scale = 1) {
  validate_parameter_value(df)
  stopifnot(all(df >= 1), all(df == as.integer(df)))
  validate_parameter_value(scale)
  nlist(dist = "product_normal", df, location, scale)
}

#' @rdname priors
#' @export
lkj <- function(regularization = 1, scale = 10, df = 1, autoscale = TRUE) {
  validate_parameter_value(regularization)
  validate_parameter_value(scale)
  validate_parameter_value(df)
  nlist(dist = "lkj", regularization, scale, df, autoscale)
}

#' @rdname priors
#' @export
decov <- function(regularization = 1, concentration = 1, shape = 1,
                  scale = 1) {
  validate_parameter_value(regularization)
  validate_parameter_value(concentration)
  validate_parameter_value(shape)
  validate_parameter_value(scale)
  nlist(dist = "decov", regularization, concentration, shape, scale)
}
