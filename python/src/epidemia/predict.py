"""Forecasting and posterior *predictive* sampling.

Mirrors R's ``posterior_predict(newdata=)`` / ``posterior_infections()`` /
``posterior_rt()``. Those work by handing the fitted parameter draws to Stan's
standalone generated quantities (see ``R/posterior_sims.R``), which re-runs the
renewal recursion over the *new* time span. Nothing is re-fitted: conditional on
a draw of the parameters the latent series are a **deterministic** function of
them, so the same thing here is plain forward simulation in NumPy, vectorised
over the draw axis.

Two things this adds over :mod:`epidemia.renewal`:

* the susceptibility ("population") adjustment of R's Stan model, which matters
  as soon as a forecast runs long enough to deplete the susceptible pool;
* :func:`posterior_predict`, which *draws* from the observation family rather
  than returning its mean. Banding the posterior of the mean -- what the plots
  do today -- gives intervals that exclude observation noise and are therefore
  far too narrow to compare against data.

Kernels keep the package-wide lag-1-first convention: ``gen[s-1]`` weights
infections ``s`` days ago and ``i2o[k-1]`` weights infections ``k`` days before
the observation. See :mod:`epidemia.renewal` for why.
"""

from __future__ import annotations

import numpy as np

from .renewal import renewal_infections

__all__ = ["simulate", "expected_observations", "posterior_predict"]

#: Observation families supported by :func:`posterior_predict`, and what ``aux``
#: means for each. ``quasi_poisson``/``log_normal`` are here for parity with R's
#: ``posterior_predict.epimodel``; the Python models currently fit the first three.
FAMILIES = ("poisson", "neg_binom", "quasi_poisson", "normal", "log_normal")


def simulate(Rt, gen, seed, seed_days, n_days, pop=None, susceptible0=None):
    """Forward-simulate infections from draws of ``R_t`` and the seed.

    Without ``pop`` this is the plain renewal recursion of
    :func:`epidemia.renewal.renewal_infections` with a constant seed over the
    first ``seed_days`` days. With ``pop`` it additionally applies the
    susceptibility adjustment exactly as ``inst/stan/tparameters/gen_infections.stan``
    does::

        i'_t    = R_t * sum_{s>=1} i_{t-s} gen_s
        i_t     = S_t * (1 - exp(-i'_t / pop))
        S_{t+1} = S_t - i_t

    Note that the recursion feeds back the *adjusted* infections (Stan overwrites
    its ``infections`` matrix in place before computing the next day's load), and
    that the adjustment is applied to the seeding days too -- there ``i'_t`` is
    the seed rather than ``R_t`` times the load. Both details matter for parity.

    Parameters
    ----------
    Rt : array (..., n_days) or scalar
        Reproduction number. Leading axes are treated as draws; values during
        the seeding window are ignored. May be longer than ``n_days`` (extra
        days are dropped), e.g. when a fit is extended by a forecast horizon.
    gen : array (L,)
        Generation kernel, lag-1-first.
    seed : scalar or array (...)
        Seeded infections, held constant over the first ``seed_days`` days.
    seed_days : int
        Length of the seeding window (R's ``N0``).
    n_days : int
        Number of days to simulate, including the seeding window.
    pop : scalar or array (...), optional
        Population size. If ``None`` no susceptibility adjustment is made.
    susceptible0 : scalar or array (...), optional
        Susceptibles on the first modelled day; defaults to ``pop``. This is R's
        ``pops[m] * S0[m]``. Ignored when ``pop`` is ``None``.

    Returns
    -------
    infections : array (..., n_days)
        Latent daily infections. The leading shape is the broadcast of the
        leading shapes of ``Rt``, ``seed``, ``pop`` and ``susceptible0``; a
        1-D ``Rt`` with scalar parameters gives a 1-D result.

    Notes
    -----
    Vaccination is not modelled (R's ``veps``/``vacc``), so the susceptible
    series is recoverable from the output as
    ``susceptible0 - cumsum(infections)``.
    """
    gen = np.asarray(gen, dtype=float)
    seed_days = int(seed_days)
    n_days = int(n_days)
    if seed_days < 1:
        raise ValueError("seed_days must be >= 1")
    if n_days < seed_days:
        raise ValueError(f"n_days ({n_days}) must be >= seed_days ({seed_days})")

    R = np.asarray(Rt, dtype=float)
    if R.ndim == 0:
        R = np.broadcast_to(R, (n_days,))
    if R.shape[-1] < n_days:
        raise ValueError(
            f"Rt covers {R.shape[-1]} days but {n_days} were requested"
        )
    R = R[..., :n_days]

    seed_a = np.asarray(seed, dtype=float)
    batch = np.broadcast_shapes(R.shape[:-1], seed_a.shape)
    if pop is not None:
        batch = np.broadcast_shapes(batch, np.shape(pop), np.shape(susceptible0))

    # Single draw, no depletion: hand straight to the reference implementation
    # rather than keeping a second copy of the same recursion in sync with it.
    if pop is None and batch == ():
        return renewal_infections(R, np.full(seed_days, float(seed_a)), gen)

    n_draws = int(np.prod(batch, dtype=int))
    R2 = np.broadcast_to(R, batch + (n_days,)).reshape(n_draws, n_days)
    seeds = np.broadcast_to(seed_a, batch).reshape(n_draws)

    L = gen.shape[0]
    infections = np.empty((n_draws, n_days))
    infections[:, :seed_days] = seeds[:, None]

    if pop is None:
        # Same time-varying linear filter as renewal_infections, one step at a
        # time (R_t feeds back, so there is no convolution form), but with the
        # per-step dot product batched into a single matvec over draws.
        for t in range(seed_days, n_days):
            lo = max(0, t - L)
            window = infections[:, lo:t][:, ::-1]
            infections[:, t] = R2[:, t] * (window @ gen[: t - lo])
        return infections.reshape(batch + (n_days,))

    pops = np.broadcast_to(np.asarray(pop, dtype=float), batch).reshape(n_draws)
    if np.any(pops <= 0):
        raise ValueError("pop must be positive")
    if susceptible0 is None:
        susc = pops.astype(float, copy=True)
    else:
        s0 = np.asarray(susceptible0, dtype=float)
        susc = np.broadcast_to(s0, batch).reshape(n_draws).astype(float, copy=True)

    for t in range(n_days):
        if t < seed_days:
            unadj = seeds
        else:
            lo = max(0, t - L)
            window = infections[:, lo:t][:, ::-1]
            unadj = R2[:, t] * (window @ gen[: t - lo])
        # -expm1(-x) == 1 - exp(-x), accurately for the small x of a big pop.
        i_t = susc * -np.expm1(-unadj / pops)
        infections[:, t] = i_t
        susc = susc - i_t  # i_t <= susc by construction, so susc stays >= 0

    return infections.reshape(batch + (n_days,))


def expected_observations(infections, i2o, ascertainment=1.0):
    """Expected observations, vectorised over a leading draw axis.

    The batched counterpart of :func:`epidemia.renewal.expected_observations`::

        E_t = alpha_t * sum_{k=1..K} i_{t-k} * i2o_k

    Parameters
    ----------
    infections : array (..., T)
        Latent daily infections; leading axes are draws.
    i2o : array (K,)
        Infection-to-observation delay kernel, lag-1-first: ``i2o[k-1]`` weights
        the infection ``k`` days before the observation, so lag 0 is never used.
    ascertainment : scalar or array, default 1.0
        Multiplier on the convolution -- R's ``ifr``/``iar``. Broadcast against
        the result, so a scalar, a per-day ``(T,)`` vector and a per-draw
        ``(draws, 1)`` column all work.

    Returns
    -------
    expected : array (..., T)
    """
    infections = np.asarray(infections, dtype=float)
    i2o = np.asarray(i2o, dtype=float)
    T = infections.shape[-1]

    # Unlike the renewal recursion this delay is time-invariant, so it is a true
    # convolution: accumulate K shifted copies instead of stepping through time.
    out = np.zeros_like(infections)
    for k in range(1, min(i2o.shape[0], T - 1) + 1):
        out[..., k:] += i2o[k - 1] * infections[..., : T - k]
    return np.asarray(ascertainment) * out


def posterior_predict(expected, family, aux=None, rng=None):
    """Draw from the observation model given its expected values.

    This is the piece that separates a *predictive* interval from a credible
    interval on the mean: the returned draws carry observation noise on top of
    the parameter uncertainty already present in ``expected``.

    Parameters
    ----------
    expected : array (..., T)
        Expected observations, typically one row per posterior draw (the output
        of :func:`expected_observations`).
    family : {"poisson", "neg_binom", "quasi_poisson", "normal", "log_normal"}
        Observation family, named as in R's ``epiobs``.
    aux : scalar or array, optional
        The family's auxiliary parameter, matching R and PyMC:
        ``neg_binom`` -> ``reciprocal_dispersion`` (``alpha`` of
        ``pm.NegativeBinomial(mu, alpha)``, i.e. ``size`` of ``rnbinom(mu, size)``),
        so ``var = mu + mu^2 / aux``; ``quasi_poisson`` -> ``size = mu / aux``
        as in R, so ``var = mu * (1 + aux)``;
        ``normal``/``log_normal`` -> the standard deviation. Unused (and
        ignored) for ``poisson``. A per-draw vector of shape ``(draws,)`` is
        reshaped to a column so it broadcasts over time.
    rng : numpy.random.Generator or int, optional
        Generator or seed. Defaults to a fresh unseeded generator.

    Returns
    -------
    draws : array (..., T)
        Same shape as ``expected``; integer dtype for the count families.
    """
    expected = np.asarray(expected, dtype=float)
    if family not in FAMILIES:
        raise ValueError(f"unknown family: {family!r} (expected one of {FAMILIES})")
    if not isinstance(rng, np.random.Generator):
        rng = np.random.default_rng(rng)

    if family != "poisson":
        if aux is None:
            raise ValueError(f"family {family!r} requires aux")
        aux = np.asarray(aux, dtype=float)
        # A (draws,) vector of auxiliary draws pairs with the draw axis, not the
        # time axis -- pairing it with time is a silent, plausible-looking bug.
        if aux.ndim == 1 and expected.ndim >= 2 and aux.shape[0] == expected.shape[0]:
            aux = aux.reshape(aux.shape + (1,) * (expected.ndim - 1))
        if np.any(aux <= 0):
            raise ValueError("aux must be positive")

    if family in ("poisson", "neg_binom", "quasi_poisson"):
        # E_obs carries a small positive floor in the models, but a forecast can
        # still produce -0.0 or a rounding-level negative; the samplers reject it.
        mu = np.clip(expected, 0.0, None)

    if family == "poisson":
        return rng.poisson(mu)
    if family in ("neg_binom", "quasi_poisson"):
        # numpy parameterises NB by (n successes, p); n == R's `size` == PyMC's
        # `alpha`, and p = n / (n + mu) recovers mean mu. Floor n away from 0
        # (mu == 0 with quasi_poisson) -- p is then 1 and the draw is 0 anyway.
        n = np.broadcast_to(aux, mu.shape) if family == "neg_binom" else mu / aux
        n = np.clip(n, np.finfo(float).tiny, None)
        return rng.negative_binomial(n, n / (n + mu))
    if family == "normal":
        return rng.normal(expected, aux)
    # log_normal: R draws rlnorm(meanlog = mu - sigma^2/2, sdlog = sigma), whose
    # mean is exp(mu), so `expected` is the mean on the *log* scale's exponent.
    return rng.lognormal(expected - aux**2 / 2.0, aux)
