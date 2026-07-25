"""Reproduce the 1918 influenza example (the R package's basic tutorial).

Run it with the console script::

    epidemia-flu

or as a module::

    python -m epidemia.examples.flu
"""

from __future__ import annotations

import argparse

import numpy as np

import epidemia as epi


def build():
    """Return the observed series ``y`` and an :class:`~epidemia.EpiConfig`."""
    d = epi.flu1918()
    # The first observation is explained by earlier (seeded) infections, so we
    # offset by one day and mark the very first day as unobserved.
    y = np.concatenate([[np.nan], d.incidence]).astype(float)
    config = epi.EpiConfig(
        gen=d.generation,             # renewal generation kernel
        i2o=np.repeat(0.25, 4),       # observed over the 4 days after infection
        seed_days=6,
        link="log",
        family="neg_binom",           # overdispersion absorbs day-to-day noise (as in R)
        rw_prior_scale=0.1,           # daily random walk on log R_t
        intercept_loc=float(np.log(2.0)),  # R0 prior mean ~ 2
        intercept_scale=0.2,
        seed_prior_mean=10.0,
    )
    return y, config


def main(argv=None):
    parser = argparse.ArgumentParser(description="Fit the 1918 flu renewal model.")
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--tune", type=int, default=1000)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--backend", default="numba", choices=["numba", "jax"],
                        help="log-density backend (jax enables GPU on Linux)")
    parser.add_argument("--adaptation", default="diag",
                        choices=["diag", "low_rank", "flow"])
    parser.add_argument("--save", default=None, help="path prefix to save plots (PNG)")
    args = parser.parse_args(argv)

    print(f"epidemia {epi.__version__} | nutpie backend: {args.backend}")

    y, config = build()
    print(f"Fitting flu model with nutpie "
          f"(draws={args.draws}, tune={args.tune}, chains={args.chains}, "
          f"adaptation={args.adaptation}) ...")
    import time
    t0 = time.time()
    idata = epi.fit(y, config, draws=args.draws, tune=args.tune, chains=args.chains,
                    seed=args.seed, backend=args.backend, adaptation=args.adaptation)
    print(f"done in {time.time() - t0:.1f}s")

    import arviz as az
    summ = az.summary(idata, var_names=["intercept", "rw_scale", "seed"])
    print(summ.to_string())
    rt = np.asarray(idata.posterior["Rt"]).reshape(-1, idata.posterior["Rt"].shape[-1])
    med = np.median(rt, axis=0)
    print(f"R_t: peak median = {med.max():.2f} at day {int(med.argmax())}; "
          f"final = {med[-1]:.2f}")

    if args.save:
        # The plot functions save themselves, so name the files here rather than
        # calling .save() again -- that would write each figure twice, once to
        # the requested path and once to the default figure_dir().
        epi.plots.plot_rt(idata, save=f"{args.save}_rt")
        epi.plots.plot_obs(idata, observed=y, save=f"{args.save}_obs")
        epi.plots.plot_infections(idata, save=f"{args.save}_infections")
    else:
        for p in (epi.plots.plot_rt, epi.plots.plot_infections):
            p(idata, save=False)
        epi.plots.plot_obs(idata, observed=y, save=False)
    return idata


if __name__ == "__main__":
    main()
