# Common development tasks for the epidemia R package.
# See AGENTS.md for what each area of the codebase does.

RSCRIPT := Rscript

.PHONY: help setup test test-slow document check tutorials tutorials-clean compile docs clean

help:
	@echo "setup            restore the renv library and pinned CmdStan"
	@echo "test             run the testthat suite (fast tier)"
	@echo "test-slow        run the suite including the slow fitting tests"
	@echo "document         regenerate NAMESPACE and man/ with roxygen2"
	@echo "check            R CMD check"
	@echo "compile          precompile both Stan programs into the user cache"
	@echo "tutorials        re-bake the precomputed tutorial vignettes"
	@echo "tutorials-clean  drop the knitr cache first, forcing genuine re-fits"
	@echo "docs             build the pkgdown site"

setup:
	$(RSCRIPT) tools/setup.R

# load_all() rather than library(): the package is not installed into the renv
# project library, and we want the tests to run against the working tree anyway.
# NOT_CRAN is what devtools::test() sets; without it every skip_on_cran() test
# silently skips, which is most of the suite.
test:
	NOT_CRAN=true $(RSCRIPT) -e 'pkgload::load_all(".", quiet = TRUE); testthat::test_dir("tests/testthat")'

# EPIDEMIA_SLOW_TESTS gates the heavier fits; see tests/testthat/helper-epidemia.R.
test-slow:
	NOT_CRAN=true EPIDEMIA_SLOW_TESTS=true $(RSCRIPT) -e 'pkgload::load_all(".", quiet = TRUE); testthat::test_dir("tests/testthat")'

document:
	$(RSCRIPT) -e 'roxygen2::roxygenise()'

check:
	$(RSCRIPT) -e 'devtools::check(document = FALSE, args = "--no-manual")'

compile:
	$(RSCRIPT) -e 'pkgload::load_all("."); compile_epidemia(quiet = FALSE)'

tutorials:
	$(RSCRIPT) vignettes/precompute.R

# knitr replays cache=TRUE chunks, so a re-bake proves nothing unless the cache
# goes first. Use this when verifying that the tutorials still fit.
tutorials-clean:
	rm -rf vignettes/cache
	$(RSCRIPT) vignettes/precompute.R

docs:
	$(RSCRIPT) -e 'pkgdown::build_site()'

clean:
	rm -rf vignettes/cache vignettes/*_files ..Rcheck
