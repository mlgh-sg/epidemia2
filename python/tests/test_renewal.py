"""Deterministic checks of the renewal dynamics."""

import jax.numpy as jnp
import numpy as np

from epidemia.renewal import expected_observations, random_walk, renewal_infections


def test_renewal_constant_infectiousness():
    # gen puts all weight on the 1-day-ago infection, R = 1 -> infections persist
    R = jnp.ones(5)
    gen = jnp.array([1.0])
    seeds = jnp.array([2.0, 3.0])
    inf = np.asarray(renewal_infections(R, seeds, gen))
    assert np.allclose(inf[:2], [2.0, 3.0])       # seeds preserved
    assert np.allclose(inf[2:], [3.0, 3.0, 3.0])  # i_t = i_{t-1}


def test_renewal_geometric_growth():
    # R = 2 with a 1-day generation -> exact doubling
    R = jnp.full(4, 2.0)
    gen = jnp.array([1.0])
    seeds = jnp.array([1.0])
    inf = np.asarray(renewal_infections(R, seeds, gen))
    assert np.allclose(inf, [1.0, 2.0, 4.0, 8.0])


def test_expected_obs_is_causal_convolution():
    inf = jnp.array([1.0, 0.0, 0.0, 0.0])
    i2o = jnp.array([0.5, 0.3, 0.2])  # same-day, 1-day, 2-day delay
    y = np.asarray(expected_observations(inf, i2o, 1.0))
    assert np.allclose(y, [0.5, 0.3, 0.2, 0.0])


def test_ascertainment_scales_observations():
    inf = jnp.ones(4)
    i2o = jnp.array([1.0])
    y = np.asarray(expected_observations(inf, i2o, 0.1))
    assert np.allclose(y, [0.1, 0.1, 0.1, 0.1])


def test_non_centered_random_walk():
    walk = np.asarray(random_walk(0.5, jnp.array([1.0, 1.0, 1.0]), 2.0))
    assert np.allclose(walk, [2.5, 3.0, 3.5])
