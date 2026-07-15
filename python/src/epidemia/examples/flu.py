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
    parser.add_argument("--sampler", default="nutpie",
                        choices=["nutpie", "numpyro", "blackjax"])
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--tune", type=int, default=1000)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--save", default=None, help="path prefix to save plots (PNG)")
    args = parser.parse_args(argv)

    import jax
    print(f"epidemia {epi.__version__} | JAX backend: {jax.default_backend()} "
          f"| devices: {jax.devices()}")

    y, config = build()
    print(f"Fitting flu model with sampler={args.sampler} "
          f"(draws={args.draws}, tune={args.tune}, chains={args.chains}) ...")
    import time
    t0 = time.time()
    idata = epi.fit(y, config, sampler=args.sampler, draws=args.draws,
                    tune=args.tune, chains=args.chains, seed=args.seed)
    print(f"done in {time.time() - t0:.1f}s")

    import arviz as az
    summ = az.summary(idata, var_names=["intercept", "rw_scale", "seed"])
    print(summ.to_string())
    rt = np.asarray(idata.posterior["Rt"]).reshape(-1, idata.posterior["Rt"].shape[-1])
    med = np.median(rt, axis=0)
    print(f"R_t: peak median = {med.max():.2f} at day {int(med.argmax())}; "
          f"final = {med[-1]:.2f}")

    if args.save:
        epi.plots.plot_rt(idata).save(f"{args.save}_rt.png", verbose=False)
        epi.plots.plot_obs(idata, observed=y).save(f"{args.save}_obs.png", verbose=False)
        epi.plots.plot_infections(idata).save(f"{args.save}_infections.png", verbose=False)
        print(f"saved plots to {args.save}_*.png")
    return idata


if __name__ == "__main__":
    main()
