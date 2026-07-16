#!/usr/bin/env Rscript
# Generate llms-full.txt for the R package: the curated llms.txt overview plus
# the README, every vignette, and a function reference, concatenated for LLM
# ingestion (mirrors python/docs/llms-full.txt on the Python side).
#
# The curated root llms.txt is hand-maintained (the source of truth for the
# "how to USE the package" overview). This script only (re)builds llms-full.txt
# from it plus the docs; it never overwrites llms.txt.
#
# Run from the package root:  Rscript data-raw/build_llms.R
# The pkgdown workflow runs this and copies llms.txt + llms-full.txt into the
# built site, so they are served at <site>/llms.txt and <site>/llms-full.txt.

SITE <- "https://mlgh-sg.com/epidemia2"

read_file <- function(p) paste(readLines(p, warn = FALSE), collapse = "\n")

# Return (title, body) for a vignette, dropping any leading YAML front matter.
split_rmd <- function(path) {
  lines <- readLines(path, warn = FALSE)
  title <- sub("\\.Rmd$", "", basename(path))
  if (length(lines) && grepl("^---\\s*$", lines[1])) {
    close <- which(grepl("^---\\s*$", lines))[2]
    yaml <- lines[2:(close - 1)]
    t <- grep("^title:", yaml, value = TRUE)
    if (length(t)) title <- trimws(gsub('^title:\\s*|["\']', "", t[1]))
    body <- if (close < length(lines)) lines[(close + 1):length(lines)] else character(0)
  } else {
    body <- lines
  }
  list(title = title, body = paste(body, collapse = "\n"))
}

# Plain text of a top-level Rd tag (e.g. "\\title", "\\name").
rd_tag_text <- function(rd, tag) {
  tags <- vapply(rd, function(x) attr(x, "Rd_tag"), character(1))
  idx <- which(tags == tag)
  if (!length(idx)) return("")
  txt <- paste(rapply(rd[idx], as.character, how = "unlist"), collapse = "")
  trimws(gsub("\\s+", " ", txt))
}

# --- function reference index from man/*.Rd -------------------------------
rd_files <- list.files("man", pattern = "\\.Rd$", full.names = TRUE)
entries <- vapply(rd_files, function(f) {
  rd <- tools::parse_Rd(f)
  name <- rd_tag_text(rd, "\\name")
  title <- rd_tag_text(rd, "\\title")
  if (!nzchar(name) || grepl("-package$", name)) return(NA_character_)
  sprintf("- `%s`: %s", name, title)
}, character(1))
reference <- sort(entries[!is.na(entries)])

# --- vignette order (concepts first, then tutorials) ----------------------
vig_order <- c(
  "install", "model-introduction", "model-description", "model-implementation",
  "model-schematic", "priors", "partial-pooling", "multiple-obs",
  "flu", "europe-covid"
)
vig_files <- list.files("vignettes", pattern = "\\.Rmd$", full.names = TRUE)
names(vig_files) <- sub("\\.Rmd$", "", basename(vig_files))
vig_files <- vig_files[c(intersect(vig_order, names(vig_files)),
                         setdiff(names(vig_files), vig_order))]
vignettes <- lapply(vig_files, split_rmd)

# --- assemble llms-full.txt -----------------------------------------------
sep <- function(name) sprintf("\n\n<!-- ===== %s ===== -->\n\n", name)
parts <- c(
  "# epidemia (R) — full documentation\n",
  sprintf("> Concatenation of the epidemia R docs for LLM ingestion. Source: %s/\n", SITE),
  "> Curated overview (llms.txt) first, then the README, every article, and the\n",
  "> function reference.\n\n---\n",
  sep("llms.txt (curated overview)"), read_file("llms.txt"),
  sep("README.md"), read_file("README.md")
)
for (i in seq_along(vignettes)) {
  v <- vignettes[[i]]
  parts <- c(parts, sep(paste0("article: ", names(vig_files)[i])),
             sprintf("# %s\n\n", v$title), v$body)
}
parts <- c(parts, sep("function reference"),
           "# Function reference\n\n", paste(reference, collapse = "\n"), "\n")
writeLines(paste(parts, collapse = ""), "llms-full.txt")

cat(sprintf("wrote llms-full.txt (%d articles, %d reference entries, %d bytes)\n",
            length(vignettes), length(reference), file.size("llms-full.txt")))
