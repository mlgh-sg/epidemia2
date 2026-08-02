## Build the EnglandB117 dataset: SGTF-split case counts for England, used to
## estimate the transmissibility advantage of SARS-CoV-2 lineage B.1.1.7.
##
## SOURCE  mrc-ide/sarscov2-b.1.1.7 at tag v1.0 (MIT licensed), the code and data
##         released with
##
##           Volz et al. (2021), "Assessing transmissibility of SARS-CoV-2
##           lineage B.1.1.7 in England", Nature 593, 266-269.
##           https://www.nature.com/articles/s41586-021-03470-x
##
## WHAT THE COUNTS ARE. Routine PCR testing in England used a three-target
## assay. B.1.1.7 carries a deletion that makes the S-gene target fail while the
## other two still amplify, so "S-gene target failure" (SGTF) acts as a proxy
## for the lineage without sequencing every sample. Each area-day therefore
## splits into `corrected_negative` (S-gene negative -> B.1.1.7) and
## `corrected_positive` (S-gene positive -> everything else), already adjusted
## for testing effort by the authors' pipeline.
##
## WHAT IS NOT REPRODUCIBLE HERE. These are aggregates. The raw SGSS line-list
## behind them is disclosure-controlled -- the authors' own processing script
## (src/total-cases-SGSS.R) contains
##     df$neg[df$neg < 5] = NA   # "for disclosure purposes according to PHE"
## so counts below five are suppressed upstream and the raw -> aggregate step
## cannot be re-run outside PHE. This script starts from the published
## aggregates, which is as far back as anyone outside PHE can go.
##
##     Rscript data-raw/england-b117.R

stopifnot(file.exists("DESCRIPTION"))

tag  <- "v1.0"
base <- sprintf("https://raw.githubusercontent.com/mrc-ide/sarscov2-b.1.1.7/%s", tag)

fetch <- function(path) {
  dest <- file.path(tempdir(), basename(path))
  if (!file.exists(dest)) {
    message("downloading ", path)
    utils::download.file(file.path(base, path), dest, quiet = TRUE, mode = "wb")
  }
  dest
}

sgtf <- readRDS(fetch("data/sgtf_transmission_data.rds"))
rates <- readRDS(fetch("data/i2o_rates.rds"))
pops <- utils::read.csv(fetch("data/stp_population.csv"), stringsAsFactors = FALSE)

## ---- case counts --------------------------------------------------------
counts <- as.data.frame(sgtf)[, c("date", "area", "corrected_positive",
                                  "corrected_negative", "epiweek")]
counts$date <- as.Date(counts$date)
counts <- counts[order(counts$area, counts$date), ]
rownames(counts) <- NULL

## ---- population ---------------------------------------------------------
pops <- pops[pops$subgroup == "All", c("AREA", "Y2018")]
names(pops) <- c("area", "pop")
pops <- pops[pops$area %in% unique(counts$area), ]
## 49 areas but only 42 populations: the extra seven are NHS England REGIONS
## (aggregates of STPs), which stp_population.csv does not size. They are kept
## because their case counts are still useful for description, but they cannot
## be fitted with pop_adjust. Documented on the dataset rather than dropped.
regions <- setdiff(unique(counts$area), pops$area)
message(sprintf("%d areas have populations; %d are NHS regions without one: %s",
                nrow(pops), length(regions), paste(regions, collapse = ", ")))

## ---- infection ascertainment rate ---------------------------------------
## One England-wide daily series, plus a single standard deviation that becomes
## the prior scale on the observation coefficient.
iar <- as.data.frame(rates[rates$type == "IAR", c("date", "value")])
names(iar) <- c("date", "iar")
iar$date <- as.Date(iar$date)
iar <- iar[order(iar$date), ]
rownames(iar) <- NULL
iar_sd <- as.numeric(rates$value[rates$type == "IAR_sd"][1])

## ---- infection-to-observation kernel ------------------------------------
## The observations are WEEKLY case totals attached to a daily series, so the
## authors spread a daily infection-to-case delay over seven offsets
## (models/joint_model.R):
##     i2o2week <- function(i2o)
##       rowSums(sapply(0:6, function(k) c(rep(0,k), i2o, rep(0,6-k))))
## The daily kernel is flat over days 4-13 after infection. The result
## deliberately sums to 7, not 1: each observation covers seven days of cases.
i2o_daily <- c(0, 0, 0, rep(1 / 10, 10))
i2o <- rowSums(sapply(0:6, function(k) c(rep(0, k), i2o_daily, rep(0, 6 - k))))
stopifnot(abs(sum(i2o) - 7) < 1e-12)

## ---- the paper's published estimates, for checking against ---------------
## transmission_output_joint.rds carries the fitted per-area, per-week
## multiplicative advantage; time_varying_advantage.rds is the England-wide
## series with credible intervals, pooled over every area. Shipping both means
## a tutorial can check itself against the paper rather than against a number
## copied by hand.
joint <- readRDS(fetch("data/transmission_output_joint.rds"))
joint <- as.data.frame(joint)
joint$geometry <- NULL
published <- data.frame(
  area    = as.character(joint$name),
  epiweek = as.integer(sub("^Week ", "", as.character(joint$epiweek))),
  ratio   = as.numeric(joint$Ratio),
  rt_b117 = as.numeric(joint[["R(S-)"]]),
  rt_other = as.numeric(joint[["R(S+)"]]),
  stringsAsFactors = FALSE
)
published <- published[order(published$area, published$epiweek), ]
rownames(published) <- NULL

eng <- as.data.frame(readRDS(fetch("data/time_varying_advantage.rds")))
published_england <- data.frame(
  epiweek = as.integer(eng$week),
  median  = as.numeric(eng[["50%"]]),
  lower   = as.numeric(eng[["2.5%"]]),
  upper   = as.numeric(eng[["97.5%"]]),
  stringsAsFactors = FALSE
)

EnglandB117 <- list(
  data              = counts,
  pop               = pops,
  iar               = iar,
  iar_sd            = iar_sd,
  i2o               = i2o,
  published         = published,
  published_england = published_england
)

message(sprintf("areas %d, days/area %d, %s .. %s",
                length(unique(counts$area)),
                as.integer(median(table(counts$area))),
                min(counts$date), max(counts$date)))
message(sprintf("iar_sd %.6f, i2o length %d summing to %.0f",
                iar_sd, length(i2o), sum(i2o)))

save(EnglandB117, file = "data/EnglandB117.RData", compress = "bzip2")
message("wrote data/EnglandB117.RData")

## ---- and the Python port's copies ---------------------------------------
d <- file.path("python", "src", "epidemia", "data_files")
utils::write.csv(counts, file.path(d, "england_b117.csv"), row.names = FALSE)
utils::write.csv(pops, file.path(d, "england_b117_pop.csv"), row.names = FALSE)
utils::write.csv(iar, file.path(d, "england_b117_iar.csv"), row.names = FALSE)
utils::write.csv(data.frame(i2o = i2o), file.path(d, "england_b117_i2o.csv"),
                 row.names = FALSE)
utils::write.csv(data.frame(iar_sd = iar_sd),
                 file.path(d, "england_b117_iar_sd.csv"), row.names = FALSE)
utils::write.csv(published, file.path(d, "england_b117_published.csv"),
                 row.names = FALSE)
utils::write.csv(published_england,
                 file.path(d, "england_b117_published_england.csv"),
                 row.names = FALSE)
message("wrote the Python data_files copies")
