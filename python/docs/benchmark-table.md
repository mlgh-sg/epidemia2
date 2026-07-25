| Engine | Compile (s) | Sample (s) | min ESS | ESS/s | Divergences | max R-hat |
|---|---:|---:|---:|---:|---:|---:|
| R / cmdstanr | 0.1 | 974.6 | 88 | 0.09 | None | 1.044 |
| Python / nutpie (diag) | 20.5 | 963.1 | 27 | 0.03 | 27 | 1.100 |
| Python / nutpie (low_rank) | 20.2 | 752.0 | 114 | 0.15 | 10 | 1.030 |

4 chains x (500 tune + 500 draws), seed 12345. Model: Europe/COVID multilevel, 11 countries, 5 NPIs, deaths.

Run on arm, 10 cores, macOS-26.5.2-arm64-arm-64bit. Python 3.12.9; pymc 5.28.5, nutpie 0.16.11, numba 0.66.0, numpy 2.4.6, pytensor 2.38.2, arviz 0.23.4

min ESS is the smallest bulk effective sample size across the five NPI effects; ESS/s divides it by sampling wall-clock. Compile time is excluded from ESS/s because Stan caches its executable across runs while nutpie re-compiles each time.
