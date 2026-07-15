"""Core renewal-process dynamics in JAX.

The (discrete) renewal equation propagates latent daily infections as a
self-exciting point process::

    i_t = R_t * sum_{s=1..L} i_{t-s} * g_s

where ``g`` is the generation-interval PMF and ``R_t`` the reproduction number.
Observations are linked to infections through a delay/ascertainment
convolution::

    y_t = alpha_t * sum_{k=0..K-1} i_{t-k} * pi_k

All functions are pure JAX and differentiable, so they compose inside a NUTS
target.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def renewal_infections(R, seeds, gen):
    """Propagate infections via the discrete renewal equation.

    Parameters
    ----------
    R : array (N,)
        Reproduction number on each of the ``N`` modelled days. Values before
        the end of the seeding window are ignored.
    seeds : array (v,)
        Seeded infections for the first ``v`` days (parameters of the model).
    gen : array (L,)
        Generation-interval PMF; ``gen[s-1]`` is the probability that a
        secondary infection occurs ``s`` days after the primary one.

    Returns
    -------
    infections : array (N,)
        Latent daily infections, with the first ``v`` entries equal to ``seeds``.
    """
    R = jnp.asarray(R)
    seeds = jnp.asarray(seeds)
    gen = jnp.asarray(gen)
    L = gen.shape[0]
    v = seeds.shape[0]

    # Buffer of the most recent ``L`` infections, most-recent-first, so that
    # dot(buffer, gen) == sum_{s=1..L} i_{t-s} * gen[s-1].
    rev = seeds[::-1]
    if v >= L:
        buf0 = rev[:L]
    else:
        buf0 = jnp.concatenate([rev, jnp.zeros(L - v, dtype=seeds.dtype)])

    def step(buf, R_t):
        i_t = R_t * jnp.dot(buf, gen)
        new_buf = jnp.concatenate([i_t[None], buf[:-1]])
        return new_buf, i_t

    _, infections_after_seed = jax.lax.scan(step, buf0, R[v:])
    return jnp.concatenate([seeds, infections_after_seed])


def infectiousness(infections, gen):
    """Total infectiousness ``sum_{s} i_{t-s} * gen_s`` (the renewal weight)."""
    infections = jnp.asarray(infections)
    gen = jnp.asarray(gen)
    N = infections.shape[0]
    # shift by one day so day t uses only past infections
    shifted = jnp.concatenate([jnp.zeros(1, infections.dtype), infections[:-1]])
    return jnp.convolve(shifted, gen)[:N]


def expected_observations(infections, i2o, ascertainment=1.0):
    """Expected observations from infections via an infection-to-observation delay.

    Parameters
    ----------
    infections : array (N,)
        Latent daily infections.
    i2o : array (K,)
        Infection-to-observation delay distribution; ``i2o[k]`` weights
        infections ``k`` days before the observation. Need not sum to one (it
        can also encode an overall ascertainment scale).
    ascertainment : float or array (N,)
        Time-varying multiplier (e.g. an ascertainment/IFR rate).

    Returns
    -------
    y : array (N,)
        Expected observations on each day.
    """
    infections = jnp.asarray(infections)
    i2o = jnp.asarray(i2o)
    N = infections.shape[0]
    conv = jnp.convolve(infections, i2o)[:N]  # causal: sum_k i2o[k] * i_{t-k}
    return jnp.asarray(ascertainment) * conv


def random_walk(scale, noise, intercept):
    """Build a random-walk linear predictor ``intercept + cumsum(scale * noise)``.

    ``noise`` are standardised (unit-normal) increments — a *non-centred*
    parameterisation, which gives the sampler a far easier geometry than
    drawing the increments directly at the unknown ``scale``.
    """
    steps = jnp.asarray(scale) * jnp.asarray(noise)
    return jnp.asarray(intercept) + jnp.cumsum(steps)
