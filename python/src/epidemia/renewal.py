"""Core renewal-process dynamics (NumPy reference implementation).

The (discrete) renewal equation propagates latent daily infections as a
self-exciting point process::

    i_t = R_t * sum_{s=1..L} i_{t-s} * g_s

where ``g`` is the generation-interval kernel and ``R_t`` the reproduction
number. Observations are linked to infections through a delay/ascertainment
convolution::

    y_t = alpha_t * sum_{k=1..K} i_{t-k} * pi_k

Both sums start at lag **1**: an infection cannot generate a secondary infection
or an observation on the day it happens. Both kernels are therefore stored
lag-1-first -- ``gen[j]`` and ``i2o[j]`` (0-indexed) weight the infection ``j+1``
days in the past. This matches the R package, whose Stan code sums
``dot_product(infections[start:t-1], tail(reverse(vec), .))`` for both kernels
and never touches lag 0, so a vector exported from R (e.g. ``EuropeCovid2$si``,
``EuropeCovid2$inf2death``) drops straight in.

These NumPy functions are used for forward/prior simulation and testing; the
model fit in :mod:`epidemia.model` implements the same recursion in PyTensor.
"""

from __future__ import annotations

import numpy as np


def renewal_infections(R, seeds, gen):
    """Propagate infections via the discrete renewal equation.

    Parameters
    ----------
    R : array (N,)
        Reproduction number on each of the ``N`` modelled days (values before
        the end of the seeding window are ignored).
    seeds : array (v,)
        Seeded infections for the first ``v`` days.
    gen : array (L,)
        Generation kernel; ``gen[s-1]`` is the weight of infections ``s`` days ago.

    Returns
    -------
    infections : array (N,)
        Latent daily infections, with the first ``v`` entries equal to ``seeds``.
    """
    R = np.asarray(R, dtype=float)
    seeds = np.asarray(seeds, dtype=float)
    gen = np.asarray(gen, dtype=float)
    L = gen.shape[0]
    v = seeds.shape[0]
    N = R.shape[0]

    # This recursion is a *time-varying* linear filter (R_t changes each step and
    # feeds back), so it has no convolution/FFT form -- it must stay sequential.
    # We keep it allocation-free: read the window straight out of `infections`
    # (views, no copies) and take one BLAS dot per step.
    infections = np.empty(N)
    infections[:v] = seeds
    for t in range(v, N):
        lo = t - L if t >= L else 0
        window = infections[lo:t][::-1]           # i_{t-1}, i_{t-2}, ... (view)
        infections[t] = R[t] * np.dot(window, gen[: t - lo])
    return infections


def infectiousness(infections, gen):
    """Total infectiousness ``sum_{s} i_{t-s} * gen_s`` (the renewal weight)."""
    infections = np.asarray(infections, dtype=float)
    gen = np.asarray(gen, dtype=float)
    N = infections.shape[0]
    shifted = np.concatenate([[0.0], infections[:-1]])  # day t uses only past
    return np.convolve(shifted, gen)[:N]


def expected_observations(infections, i2o, ascertainment=1.0):
    """Expected observations via a causal infection-to-observation convolution.

    ``i2o[k]`` weights the infection ``k+1`` days before the observation, so the
    sum runs over lags ``1..K`` and never lag 0 -- the same convention as ``gen``
    and as the R package's Stan code.
    """
    infections = np.asarray(infections, dtype=float)
    i2o = np.asarray(i2o, dtype=float)
    N = infections.shape[0]
    shifted = np.concatenate([[0.0], infections[:-1]])  # day t uses only the past
    conv = np.convolve(shifted, i2o)[:N]  # sum_k i2o[k] * i_{t-1-k}
    return np.asarray(ascertainment) * conv


def random_walk(scale, noise, intercept):
    """Non-centred random walk ``intercept + cumsum(scale * noise)``."""
    steps = np.asarray(scale) * np.asarray(noise, dtype=float)
    return np.asarray(intercept) + np.cumsum(steps)
