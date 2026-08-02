#' Covid-19 data for European countries
#' 
#' Contains a dataframe with recorded daily deaths from Covid-19 in 11 European countries up until 05/05/2020.
#' The dataframe includes variables representing different non-pharmaceutical interventions implemented by the 
#' countries considered. The data matches that used in \insertCite{Flaxman2020;textual}{epidemia}. Also 
#' includes empirical distributions for the serial interval and the time from infection to death.
#' 
#' @format A named list. The fields are:
#' \describe{
#'  \item{data}{A data frame giving indicators of certain non-pharmaceutical interventions in each country, along with death data and populations.
#'  The earliest date for each country in the dataframe is exactly 30 days before 10 cumulative deaths were observed in the country.}
#'  \item{inf2death}{A numeric vector giving the time from infection to death:
#'  infection-to-onset \eqn{\sim} Gamma(mean 5.1, cv 0.86) plus onset-to-death
#'  \eqn{\sim} Gamma(mean 18.8, cv 0.45), as in
#'  \insertCite{Flaxman2020;textual}{epidemia}. Entry \eqn{k} carries the mass
#'  in \eqn{(k-1, k]}, so the kernel places no mass at lag zero and its mean lag
#'  is 24.4 days, half a day above the distribution's own 23.9. This matches the
#'  convention of \code{si} and epidemia's lag-1-first convolution, in which
#'  entry 1 weights infections one day earlier. Note this is NOT how
#'  \insertCite{Flaxman2020;textual}{epidemia} discretises it -- the paper uses
#'  the midpoint rule -- so the reproduction vignette builds its own kernel.
#'  Regenerate with \code{data-raw/inf2death.R}.}
#'  \item{si}{The serial interval of covid-19 assumed in \insertCite{Flaxman2020;textual}{epidemia}.}
#' }
#' @references
#' \insertAllCited{}
"EuropeCovid"



#' Covid-19 data for European countries
#' 
#' Similar to `EuropeCovid`, with the following exceptions. Daily death data is obtained from the WHO COVID-19 Explorer as of 05/01/2021. This differs 
#' from the data used in \insertCite{Flaxman2020;textual}{epidemia}, because counts were updated retrospectively by the WHO as new information came 
#' to light. Daily case data is also included from the same source. This data runs from 03/01/2020 until 30/06/2020.
#' 
#' @format A named list. The fields are:
#' \describe{
#'  \item{data}{A data frame giving indicators of certain non-pharmaceutical interventions in each country, along with death data and populations.}
#'  \item{inf2death}{A numeric vector giving the time from infection to death:
#'  infection-to-onset \eqn{\sim} Gamma(mean 5.1, cv 0.86) plus onset-to-death
#'  \eqn{\sim} Gamma(mean 18.8, cv 0.45), as in
#'  \insertCite{Flaxman2020;textual}{epidemia}. Entry \eqn{k} carries the mass
#'  in \eqn{(k-1, k]}, so the kernel places no mass at lag zero and its mean lag
#'  is 24.4 days, half a day above the distribution's own 23.9. This matches the
#'  convention of \code{si} and epidemia's lag-1-first convolution, in which
#'  entry 1 weights infections one day earlier. Note this is NOT how
#'  \insertCite{Flaxman2020;textual}{epidemia} discretises it -- the paper uses
#'  the midpoint rule -- so the reproduction vignette builds its own kernel.
#'  Regenerate with \code{data-raw/inf2death.R}.}
#'  \item{si}{The serial interval of covid-19 assumed in \insertCite{Flaxman2020;textual}{epidemia}.}
#' }
#' @references
#' \insertAllCited{}
"EuropeCovid2"


#' Covid-19 Case Counts for England
#' 
#' Contains case counts of SARS-CoV-2 in England from 30/01/2020 until 30/05/2021. Case counts correspond to 
#' 'New Cases by Specimen Date', as defined by Public Health England. The data was downloaded from 
#' \insertCite{PHE;textual}{epidemia} on 01/06/2021. Case counts in the last few days of May may be 
#' underreported as not all cases have been counted as of the download date.
#'
#' @format A dataframe with three columns, `date`, `region` and `cases`. Each row gives case counts for a given date in England.
#' 
#' @references
#' \insertAllCited{}
"EnglandNewCases"

#' SGTF-split case counts for England, autumn 2020 to January 2021
#'
#' Daily PCR case counts for England split by S-gene target failure (SGTF), the
#' data behind \insertCite{Volz2021;textual}{epidemia}'s estimate of the
#' transmissibility advantage of SARS-CoV-2 lineage B.1.1.7.
#'
#' Routine testing in England used a three-target assay. B.1.1.7 carries a
#' deletion that makes the S-gene target fail while the other two still
#' amplify, so SGTF acts as a proxy for the lineage without sequencing every
#' sample. Each area-day therefore splits into S-gene negative (B.1.1.7) and
#' S-gene positive (everything else) counts, already adjusted for testing
#' effort by the original authors.
#'
#' @format A list with five elements:
#' \describe{
#'  \item{data}{A data frame of 5,880 rows: \code{date}, \code{area},
#'  \code{corrected_positive} (non-B.1.1.7), \code{corrected_negative}
#'  (B.1.1.7) and \code{epiweek}, for 49 areas over 120 days
#'  (2020-09-26 to 2021-01-23).}
#'  \item{pop}{Populations for 42 of those areas. The remaining seven are NHS
#'  England \emph{regions} -- aggregates of the others -- which the source does
#'  not size, so they cannot be used with \code{epiinf(pop_adjust = TRUE)}.}
#'  \item{iar}{England-wide daily infection ascertainment rate.}
#'  \item{iar_sd}{Standard deviation of the ascertainment rate, used as the
#'  prior scale on the observation coefficient.}
#'  \item{i2o}{Infection-to-observation kernel. The observations are weekly
#'  case totals attached to a daily series, so a daily delay distribution is
#'  spread over seven offsets; it therefore sums to 7, not 1.}
#' }
#'
#' @details The counts are aggregates. The raw SGSS line-list behind them is
#' disclosure-controlled -- counts below five are suppressed at source -- so the
#' raw-to-aggregate step cannot be reproduced outside PHE. Regenerate the
#' aggregates with \code{data-raw/england-b117.R}, which pulls them from
#' \code{mrc-ide/sarscov2-b.1.1.7} at tag \code{v1.0} (MIT licensed).
#'
#' @references
#'  \insertAllCited{}
"EnglandB117"
