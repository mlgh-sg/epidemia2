# Handover — state at `v1.1.0`

Written so a new session can pick up without rediscovering any of it. For how to
*use* the packages see `llms.txt` (R) and `python/docs/llms.txt` (Python); for
working *on* the source see `AGENTS.md`. This file records what was done, what
was learned the hard way, and what is still open.

---

## Where things stand

`v1.1.0` (R) / `0.2.0` (Python port) is tagged and pushed. R-CMD-check green on
both `main` and the tag; both docs stamps report fresh.

**Two paper reproductions, each checked against published values rather than
asserted.**

*B.1.1.7* — Volz et al. (2021), *Nature* 593:266-269, one area (Kent and Medway):

| week | published | R | Python |
|---|---|---|---|
| 45 | 1.68 | 1.68 | 1.68 |
| 47 | 1.94 | 1.93 | 1.94 |
| 52 | 1.46 | 1.45 | 1.45 |

mean \|R − Python\| **0.0036**; both within **0.006** of published. ~15 min/fit.

*Flaxman et al. (2020)* — lockdown **79.3%** [72.5, 84.1] (R) and **79.2%**
[73.4, 83.4] (Python) against the paper's 81% [75, 87].

**Also in 1.1.0:** the `EnglandB117` dataset (ships the paper's own estimates so
tutorials self-check); `fixed_effects` for random-only covariates; a ~1-second
docs-freshness check; the corrected `inf2death` kernel; `europe-covid` retired
with a redirect to `b117`.

---

## Traps that cost real time. Read before touching this code.

Each of these looks correct, produces no error, and gives wrong answers.

**1. R reports random-walk *increments*; Python stores cumulative *levels*.**
The same arithmetic is right in one port and wrong in the other. Treating R's as
levels gave correlation 0.44 with the published B.1.1.7 series where the correct
reading gives 0.99. **Derive quantities from `posterior_rt()` / the posterior
`Rt`, not from coefficients** — for a two-group model the ratio of the groups'
`R_t` cancels any shared walk and depends on no parameterisation. That is what
the paper's own code does.

**2. `epiweek()` / `isocalendar().week` restart at 1 each January.** The B.1.1.7
data numbers weeks continuously to 56, so the last four weeks were silently
relabelled 1, 2, 3 and dropped from the comparison. Use the dataset's own
`epiweek` column.

**3. knitr caches a chunk's *error* and replays it.** A "successful" re-bake can
be replaying a failure from a previous run — the numbers simply do not move.
`vignettes/precompute.R --force` now drops the whole cache; a per-vignette
pattern is unreliable because chunk labels need not start with the vignette's
name. Always confirm numbers actually changed.

**4. Sampler settings do not transfer between models.** `adaptation="low_rank"`
is the Python default and wins on collinear-covariate posteriors (1.7× Stan's
ESS/second on the NPI model), but Flaxman needed `"diag"` (R-hat 1.17 → 1.03).
`maxdepth` above nutpie's 10 is often required, but not always affordable —
see below.

**5. Correlated region effects need enough groups.** `correlated=True` on 11
regions *plus* a per-region random walk did not identify: R-hat **1.620** on
`Sigma_chol[0]`, bulk ESS **7**, E-BFMI 0.11. Raising `maxdepth` to 14 did not
rescue it (extrapolated to 15+ hours and still saturated). The independent
`||` form at identical settings gave R-hat 1.020 and ESS 269. **Check
`Sigma_chol` before trusting anything downstream of it.**

**6. A wrong turn worth not repeating.** `R/autocor.R` looks buggy — `paste0`
renders `NA` as the string `"NA"`, so an `rw(time=)` column with `NA` gains a
spurious level. It is *not* a bug: `parse_all_terms` deliberately keeps that
column and moves it past the `sum(ntime)` columns Stan consumes, so those days
contribute zero. "Fixing" it broke the build and `test-obs-autocor.R`.

The pattern across all of these: **two independent implementations disagreeing
was a reliable signal that something was wrong, but the fault was in how the
results were being read, not in the code being compared.** Suspect your own
extraction before the package.

---

## Delay kernels

Both `gen` and `i2o` are **lag-1-first**: entry 1 weights infections one day
back. Entry *k* carries the mass in **(k-1, k]**, so a kernel's mean lag sits
half a day above its distribution's mean and no mass falls at lag 0.

`inf2death` was wrong until 1.1.0 — discretised a day early, 1.5 days out of step
with `si`, with mass at lag zero. Mean lag 22.9 → **24.4**. Regenerate with
`data-raw/inf2death.R` (deterministic convolution, no Monte Carlo, writes the R
object and the Python CSVs from one definition).

**Reproductions must build their own kernel.** Flaxman discretises at the
midpoint (`F(k+1/2) − F(k−1/2)`, mean lag 23.9); the B.1.1.7 model's `i2o` sums
to **7, not 1**, because its observations are weekly totals on a daily series
(`epiobs` warns; the warning is expected).

---

## Docs: how the pipeline works

Tutorials ship **pre-baked** so the site builds without CmdStan, which means
published numbers can drift from the code. Three pieces keep that honest:

```
make docs-check        # ~1s: hashes the inputs, no fitting
make tutorials-clean   # re-bake R vignettes (drops the knitr cache)
make tutorials-python  # re-bake Python notebooks
make docs-stamp        # record the fingerprint after a FULL bake
```

`tools/docs-stamp.sh` hashes git-tracked *contents* of everything that can move a
fitted number, so it is identical on every machine. Both precompute scripts write
the stamp automatically after a full bake and **refuse to after a partial one** —
a stamp for a subset would overstate freshness.

`docs-freshness.yaml` runs the check on every push and *warns* rather than fails,
because stale docs are normal after a modelling change and the fix is a local
re-bake. `refresh-vignettes.yaml` is manual + weekly only, one job per tutorial:
a single job could not fit five tutorials into GitHub's six-hour ceiling and was
killed at 360 minutes having produced nothing. **Baking belongs on a developer
machine, not a shared runner.**

LLM docs: `llms.txt` / `python/docs/llms.txt` are hand-curated; `llms-full.txt`
on both sides is generated (`data-raw/build_llms.R`, `python/scripts/build_llms_full.py`)
and must be regenerated after tutorials change.

**Fit costs** (idle machine): b117 ~15 min · flaxman ~50 min · multiple-obs ~2 min
· multilevel-multi-obs ~75 min · flu ~1 min. The per-notebook timeout default is
4 hours; 1 hour was too tight.

---

## Still open

- **`multilevel-multi-obs` worst R-hat is 1.020**, just above the 1.01
  threshold. Everything else is clean (0 divergences, 0 saturation, E-BFMI 0.80).
- **`refresh-vignettes` cannot open PRs** without *Settings → Actions → General →
  Workflow permissions → allow Actions to create and approve pull requests*
  (and the same at org level). It pushes `auto/refresh-vignettes` regardless, so
  the output is not lost.
- **Only Kent is fitted** in the B.1.1.7 tutorials. The published estimate
  combines separate per-area fits over all 42 sized areas; the tutorials say so
  and give the route.
