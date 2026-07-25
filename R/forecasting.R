#' Posterior model evaluations
#'
#' Calculate daily error using one of three metrics, and also return coverage
#' of credible intervals. Uses continuous ranked probability
#' score (CRPS), mean absolute error and median absolute error.
#'
#' @inherit plot_obs params
#' @param newdata  If provided, the original \code{data} used
#'  in \code{object} is overridden. Useful for forecasting
#' @param metrics A string or character vector specifying the plotted
#'  forecast error metrics. One of \code{NULL}, \code{"crps"},
#'  \code{"mean_abs_error"}
#' @return A named list with dataframes giving metrics and coverage.
#' @export
evaluate_forecast <-
  function(object,
           newdata = NULL,
           type,
           groups = NULL,
           metrics = NULL,
           levels = c(50, 95)) {
    if (is.null(type)) {
      stop("must specify an observation type")
    }
    alltypes <- sapply(object$obs, function(x) .get_obs(formula(x)))
    # `which(type %in% alltypes)` reversed the operands: `type` is a single
    # name, so `type %in% alltypes` is a length-1 logical and `which()` of it is
    # 1 for ANY modelled type. Every series was therefore scored against the
    # FIRST series' observations -- on a deaths+cases model, asking for "cases"
    # compared case predictions to death counts.
    w <- which(alltypes == type)
    if (length(w) == 0) {
      stop(paste0("obs does not contain any observations
    for type '", type, "'"), call. = FALSE)
    }

    ok_metrics <- c("crps", "mean_abs_error", "median_abs_error")
    metrics <- metrics %ORifNULL% ok_metrics
    if (any(!(metrics %in% ok_metrics))) {
      stop("Unrecognised metrics. Allowed metrics include ",
        paste(ok_metrics, collapse = ", "),
        call. = FALSE
      )
    }
    levels <- check_levels(levels)

    # process data
    groups <- groups %ORifNULL% object$groups
    # simulate from posterior predictive
    obs <- posterior_predict(
      object = object,
      types = type,
      newdata = newdata
    )

    obs <- gr_subset(obs, groups)

    if (is.null(newdata)) {
      data <- object$data
      data <- data[data$group %in% groups, ]
    } else {
      check_data(newdata, object$rt, object$inf, object$obs, object$groups)
      data <- parse_data(newdata, object$rt, object$inf, object$obs, object$groups)
      # `obs` was restricted to `groups` above, so the observed outcomes must be
      # too. Without this, a `groups` subset combined with `newdata` fed all
      # groups' observations into daily_error() against one group's predictions
      # ("replacement has N rows, data has M").
      data <- data[data$group %in% groups, ]
    }

    # get observed outcomes
    obj <- epiobs_(object$obs[[w]], data)
    y <- get_obs(obj)

    # Rows coded -1 are forecast placeholders, not observations -- epiobs_()
    # documents them as such ("Must either be positive, NA, or coded -1 (for
    # forecasting)") and the multiple-observations tutorial builds `newdata`
    # that way. Scoring them treats the truth as -1, which inflates the error
    # and collapses coverage. Drop them from the predictions and the outcomes
    # together, so the two stay aligned.
    keep <- !is.na(y) & y >= 0
    if (!all(keep)) {
      obs$group <- obs$group[keep]
      obs$time <- obs$time[keep]
      obs$draws <- obs$draws[, keep, drop = FALSE]
      y <- y[keep]
    }

    return(list(
      error = daily_error(obs, metrics, y), 
      coverage = daily_coverage(obs, levels, y))
      )
  }

#' Coverage of posterior credible intervals
#'
#' @inherit evaluate_forecast
#' @return A dataframe indicating whether observations fall within the
#'  specified credible intervals
#' @export
posterior_coverage <-
  function(object,
           type,
           newdata = NULL,
           groups = NULL,
           levels = c(50, 95)) {
    out <- evaluate_forecast(
      object = object,
      type = type,
      newdata = newdata,
      groups = groups,
      levels = levels
    )
    return(out$coverage)
}

#' CRPS, Mean Absolute Error, Median Absolute Error
#' 
#' @inherit evaluate_forecast
#' @return A dataframe giving forecast error for each metric and observation
#' @export
posterior_metrics <-
  function(object,
           type,
           newdata = NULL,
           groups = NULL,
           metrics = NULL) {
    out <- evaluate_forecast(
      object = object,
      type = type,
      newdata = newdata,
      groups = groups,
      metrics = metrics
    )
    return(out$error)
  }

#' Plot coverage probability of posterior credible intervals
#'
#' Plots histograms showing empirical coverage of credible intervals
#' specified using 'levels'. Can bucket by time period, by group, by
#' whether the observation is new (not used in fitting).
#'
#' @inherit evaluate_forecast params
#' @inherit plot_obs params
#' @param period Buckets computed empirical probabilities into time periods
#' if specified.
#' @param by_group Plot coverage for each group individually
#' @param by_unseen Plot coverage separately for seen and unseen observations.
#' Observations are 'seen' if they were used for fitting.
#' @export
#' @return A \code{ggplot} object. 
plot_coverage <-
  function(object,
           type,
           newdata = NULL,
           groups = NULL,
           levels = c(50, 95),
           period = NULL,
           by_group = FALSE,
           by_unseen = FALSE) {
    
    groups <- groups %ORifNULL% object$groups
    cov <- posterior_coverage(
      object = object,
      type = type,
      groups = groups,
      newdata = newdata,
      levels = levels
    )
    
    if (!is.null(period)) {
      cov$period <- cut(cov$date, period)
    }
    
    cols <- c(
      "tag",
      if (!is.null(period)) "period",
      if (by_group) "group",
      if (by_unseen) "unseen"
    )
    
    if (by_unseen) { # need to check which observations are new
      data <- object$data
      data <- data[data$group %in% groups, c("group", "date", type)]
      data <- data %>% dplyr::rename("DUMMY" = type)
      cov <- dplyr::left_join(cov, data, by = c("group", "date"))
      cov <- cov %>% dplyr::rename("unseen" = .data$DUMMY)
      w <- is.na(cov$unseen)
      cov$unseen[w] <- "Unseen"
      cov$unseen[!w] <- "Seen"
    }
    
    df <- cov %>%
      dplyr::group_by_at(cols) %>%
      dplyr::summarise(value = mean(.data$in_ci))
    
    if (is.null(period)) {
      p <- ggplot2::ggplot(
        df,
        ggplot2::aes(x = .data$tag, y = .data$value, fill = .data$tag)
      ) +
        ggplot2::labs(
          y = "Mean Coverage",
          x = "Credible Interval"
        )
    } else {
      p <- ggplot2::ggplot(
        df,
        ggplot2::aes(x = period, y = .data$value, fill = .data$tag)
      ) +
        ggplot2::labs(
          y = "Mean Coverage",
          x = "period"
        )
    }
    
    # general formatting
    p <- p + ggplot2::geom_bar(
      stat = "identity",
      position = "dodge"
    ) +
      ggplot2::scale_y_continuous(
        labels = scales::percent_format(),
        minor_breaks = seq(0, 1, 0.05),
        breaks = seq(0, 1, 0.1)
      ) +
      hrbrthemes::theme_ipsum() +
      ggplot2::theme(
        axis.text.x = ggplot2::element_text(angle = 50, vjust = 0.5)
      )
    
    if ("group" %in% cols && "unseen" %in% cols) {
      p <- p + ggplot2::facet_grid(ggplot2::vars(.data$group), ggplot2::vars(.data$unseen))
    } else if ("group" %in% cols) {
      p <- p + ggplot2::facet_wrap(~group)
    } else if ("unseen" %in% cols) {
      p <- p + ggplot2::facet_wrap(~unseen)
    }
    
    p <- p +
      ggplot2::scale_fill_manual(
        name = "Fill",
        # The `tag` factor these fills are matched against was built from
        # check_levels(), which sorts ascending. Sort here too, or a caller
        # passing levels out of order (e.g. c(95, 50)) gets the alpha values
        # attached to the wrong credible intervals.
        values = ggplot2::alpha(
          "deepskyblue4",
          rev(sort(levels)) / 100
        )
      )
    
    return(p)
  }


#' Plot CRPS, Median/Mean Absolute Error
#'
#' Plots various metrics for evaluating probabilistic forecasts by group.
#'
#' @inherit evaluate_forecast params
#' @inherit plot_rt return
#' @export
plot_metrics <-
  function(object,
           groups = NULL,
           type,
           metrics = NULL,
           newdata = NULL) {
    groups <- groups %ORifNULL% object$groups
    
    df <- posterior_metrics(
      object = object,
      type = type,
      groups = groups,
      newdata = newdata,
      metrics = metrics
    )
    metrics <- colnames(df)[colnames(df) %in% c("crps", "mean_abs_error", "median_abs_error")]
    
    df <- df %>%
      tidyr::pivot_longer(
        c(dplyr::all_of(metrics)),
        names_to = "metric",
        values_to = "value"
      )
    
    data <- object$data
    data <- data[data$group %in% groups, c("group", "date", type)]
    data <- data %>% dplyr::rename("DUMMY" = type)
    df <- dplyr::left_join(df, data, by = c("group", "date"))
    df <- df %>% dplyr::rename("unseen" = .data$DUMMY)
    w <- is.na(df$unseen)
    df$unseen[w] <- "Unseen"
    df$unseen[!w] <- "Seen"
    
    p <- ggplot2::ggplot(
      df,
      ggplot2::aes(
        x = .data$date,
        y = .data$value,
        linetype = .data$metric,
        color = .data$unseen
      )
    ) +
      ggplot2::geom_line(alpha = 0.7, linewidth = 0.8) +
      ggplot2::facet_wrap(
        ~group,
        scales = "free_y"
      ) +
      ggplot2::labs(
        y = "Value",
        x = "Date",
        linetype = "Metric"
      ) +
      hrbrthemes::theme_ipsum() +
      ggplot2::theme(legend.position = "right")
    
    p <- p + ggplot2::scale_color_manual(
      values = c("coral4", "darkslategray4")
    )
    
    return(p)
  }

daily_error <- function(obs, metrics, y) {
  draws <- obs$draws
  mat <- (abs(sweep(t(draws), 1, y)))
  out <- data.frame(
    group = obs$group,
    date = obs$time
  )
  if ("crps" %in% metrics)
    out$crps <- crps(y, t(draws))

  if ("mean_abs_error" %in% metrics)
    out$mean_abs_error <- rowMeans(mat)
  
  if ("median_abs_error" %in% metrics)
    out$median_abs_error <- apply(mat, 1, median)

  return(out)
}

daily_coverage <- function(obs, levels, y) {
  f <- function(level) {
    qtl <- get_quantiles(obs, level)
    out <- data.frame(
      group = obs$group,
      date = qtl$date,
      tag = qtl$tag[1],
      in_ci = (qtl$lower <= y) * (y <= qtl$upper)
    )
  return(out)
  }
  dfs <- lapply(levels, f)
  return(do.call(rbind, dfs))
}

# Continuous ranked probability score.
#
# `dat` is [observation, draw]: each ROW is the predictive sample for one
# observation, and each observation must be scored against its own predictive
# distribution.
#
# This previously pooled the entire matrix into a single empirical distribution
# (`sort(dat)` over every date and group at once) and scored every observation
# against that marginal. The result was not a forecast score at all: with a
# point-mass predictive it returned 4.44 where the answer is 0, and on
# well-calibrated draws it overstated the score by one to two orders of
# magnitude. Every early day with an observed count of zero also received the
# same non-zero value, which is what made the bug visible.
crps <- function(y, dat) {
  dat <- as.matrix(dat)
  vapply(seq_along(y), function(i) crps_sample(y[i], dat[i, ]), numeric(1))
}

# CRPS of a single observation `s` against a single predictive sample `x`, using
# the standard sorted-sample estimator.
crps_sample <- function(s, x) {
  x <- sort(x)
  n <- length(x)
  c_1n <- 1 / n
  a <- seq.int(0.5 * c_1n, 1 - 0.5 * c_1n, length.out = n)
  2 * c_1n * sum(((s < x) - a) * (x - s))
}