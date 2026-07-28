"""Swappable prior families, mirroring the R package's prior system.

In R, ``normal()``, ``student_t()``, ``shifted_gamma()``, ``hexp()`` ... are thin
constructors that return a *description* of a prior (a named list with a
``dist`` field); ``epim()`` later translates that description into the Stan
program's prior representation. Nothing is built at construction time.

This module is the Python counterpart. Each constructor returns a small frozen
dataclass carrying the same argument names and defaults as its R sibling, so the
same model can be written in either language::

    # R                                    # Python
    normal(location = 0, scale = 0.5)      normal(location=0, scale=0.5)
    shifted_gamma(shape = 1/6, shift = .1) shifted_gamma(shape=1/6, shift=0.1)
    hexp(prior_aux = exponential(0.03))    hexp(prior_aux=exponential(0.03))

A spec is turned into an actual PyMC random variable by :func:`build` (or the
equivalent ``spec.build(...)`` method) *inside* an enclosing ``pm.Model``
context. ``positive=True`` truncates the family to the positive half line, which
is how R expresses e.g. ``prior_aux = normal(10, 5)`` on a Stan
``real<lower=0>`` dispersion parameter.

Which families are usable is fixed by the model, not by the user -- exactly as
in R, where ``epirt``/``epiobs``/``epiinf`` check the requested ``dist`` against
``ok_dists`` / ``ok_int_dists`` / ``ok_aux_dists`` / ``ok_cov_dists``. The same
sets live here as :data:`OK_DISTS` and friends, and :func:`resolve` raises rather
than silently accepting an unsupported family.

**Covariance priors.** ``decov`` and ``lkj`` describe the covariance of a vector
of correlated group-specific ("random") effects, not a scalar parameter. Their
``build`` therefore raises; use :func:`build_covariance` (or read the
hyperparameters back with ``spec.covariance_params()`` and construct the
covariance yourself). See that function for the decomposition used.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from typing import Any, ClassVar

import numpy as np

__all__ = [
    "OK_AUX_DISTS",
    "OK_COV_DISTS",
    "OK_DISTS",
    "OK_INT_DISTS",
    "CauchyPrior",
    "DecovPrior",
    "ExponentialPrior",
    "HexpPrior",
    "LKJPrior",
    "LaplacePrior",
    "NormalPrior",
    "Prior",
    "ShiftedGammaPrior",
    "StudentTPrior",
    "build",
    "build_covariance",
    "cauchy",
    "decov",
    "exponential",
    "hexp",
    "laplace",
    "lkj",
    "normal",
    "resolve",
    "shifted_gamma",
    "student_t",
]


def _validate_positive(value: Any, what: str) -> float:
    """Mirror R's ``validate_parameter_value``: scales/dfs/rates must be > 0."""
    try:
        x = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:  # e.g. a string slipped through
        raise ValueError(f"{what} should be numeric, got {value!r}") from exc
    if not np.all(np.isfinite(x)) or np.any(x <= 0):
        raise ValueError(f"{what} should be positive, got {value!r}")
    return value



#: What R resolves a ``scale = NULL`` prior to. R's constructors take
#: ``scale = NULL`` and ``set_prior_scale()`` (R/misc.R:48-56) substitutes
#: ``default_scale``, which standata_reg passes as 0.25 for coefficients,
#: intercepts and auxiliary parameters alike (R/standata_reg.R:18, 32, 80).
#: Python has no NULL to resolve, so the same number is the constructor default:
#: a bare ``normal()`` must mean N(0, 0.25) in both languages, not N(0, 1).
DEFAULT_PRIOR_SCALE = 0.25

class Prior:
    """Base class for prior specifications.

    A ``Prior`` only *describes* a prior: it stores hyperparameters and the
    family name (:attr:`dist`, the same string R's constructors use). Call
    :meth:`build` inside a ``pm.Model`` context to get a random variable.
    """

    #: Family name, matching the ``dist`` element of the R constructor's list.
    dist: ClassVar[str] = ""
    #: True when the family already has support on (0, inf), so ``positive=True``
    #: is a no-op rather than a truncation.
    positive_support: ClassVar[bool] = False
    #: R's per-family ``autoscale`` default. epidemia (unlike rstanarm) defaults
    #: this to FALSE for normal/student_t/cauchy/exponential/laplace/lasso and
    #: TRUE only for shifted_gamma -- see R/priors.R and R/additional_priors.R.
    autoscale_default: ClassVar[bool] = False

    def params(self) -> dict[str, Any]:
        """Hyperparameters as a dict (the R list, minus ``dist``)."""
        return {f.name: getattr(self, f.name) for f in fields(self)}

    def _pymc_dist(self):
        """Return ``(pymc distribution class, kwargs)`` for this family."""
        raise NotImplementedError

    def build(self, name: str, shape=None, positive: bool = False, dims=None):
        """Create the PyMC random variable inside the enclosing model context.

        Parameters
        ----------
        name : str
            Name of the variable in the model.
        shape : int | tuple | None
            Shape of the variable; ``None`` gives a scalar.
        positive : bool
            Truncate the family to ``[0, inf)``. This is how R expresses a prior
            on a Stan ``real<lower=0>`` parameter, e.g. ``prior_aux =
            normal(10, 5)`` for a negative-binomial dispersion.
        dims : str | tuple | None
            PyMC dims, if the model declares coordinates.

        Returns
        -------
        pytensor.tensor.TensorVariable
            The random variable (or, for the families that are built by
            transforming another variable, the named ``Deterministic``).
        """
        import pymc as pm

        cls, kwargs = self._pymc_dist()
        if positive and not self.positive_support:
            # An honest truncation, not a re-parameterisation: Stan's
            # `real<lower=0> x; x ~ normal(m, s);` *is* a truncated normal.
            return pm.Truncated(name, cls.dist(**kwargs), lower=0.0, shape=shape,
                                dims=dims)
        return cls(name, shape=shape, dims=dims, **kwargs)


@dataclass(frozen=True)
class NormalPrior(Prior):
    """Normal prior. See :func:`normal`."""

    location: float = 0.0
    scale: float = DEFAULT_PRIOR_SCALE
    dist: ClassVar[str] = "normal"

    def __post_init__(self):
        _validate_positive(self.scale, "scale")

    def _pymc_dist(self):
        import pymc as pm

        return pm.Normal, {"mu": self.location, "sigma": self.scale}


@dataclass(frozen=True)
class StudentTPrior(Prior):
    """Student-t prior. See :func:`student_t`."""

    df: float = 1.0
    location: float = 0.0
    scale: float = DEFAULT_PRIOR_SCALE
    #: R stores the Student-t family as "t" (rstanarm's naming); we keep that.
    dist: ClassVar[str] = "t"

    def __post_init__(self):
        _validate_positive(self.df, "df")
        _validate_positive(self.scale, "scale")

    def _pymc_dist(self):
        import pymc as pm

        return pm.StudentT, {"nu": self.df, "mu": self.location, "sigma": self.scale}


@dataclass(frozen=True)
class CauchyPrior(Prior):
    """Cauchy prior. See :func:`cauchy`."""

    location: float = 0.0
    scale: float = DEFAULT_PRIOR_SCALE
    dist: ClassVar[str] = "cauchy"

    def __post_init__(self):
        _validate_positive(self.scale, "scale")

    def _pymc_dist(self):
        import pymc as pm

        return pm.Cauchy, {"alpha": self.location, "beta": self.scale}


@dataclass(frozen=True)
class ExponentialPrior(Prior):
    """Exponential prior. See :func:`exponential`."""

    rate: float = 1.0
    dist: ClassVar[str] = "exponential"
    positive_support: ClassVar[bool] = True

    def __post_init__(self):
        _validate_positive(self.rate, "rate")

    @property
    def scale(self) -> float:
        """Reciprocal of the rate -- the field R's ``exponential()`` stores."""
        return 1.0 / self.rate

    def _pymc_dist(self):
        import pymc as pm

        return pm.Exponential, {"lam": self.rate}


@dataclass(frozen=True)
class LaplacePrior(Prior):
    """Laplace (double-exponential) prior. See :func:`laplace`."""

    location: float = 0.0
    scale: float = DEFAULT_PRIOR_SCALE
    dist: ClassVar[str] = "laplace"

    def __post_init__(self):
        _validate_positive(self.scale, "scale")

    def _pymc_dist(self):
        import pymc as pm

        return pm.Laplace, {"mu": self.location, "b": self.scale}


@dataclass(frozen=True)
class ShiftedGammaPrior(Prior):
    """Shifted gamma prior. See :func:`shifted_gamma`."""

    shape: float = 1.0
    scale: float = 1.0
    shift: float = 0.0
    #: R's ``shifted_gamma()`` reports dist = "gamma".
    dist: ClassVar[str] = "gamma"
    #: R's ``shifted_gamma(autoscale = TRUE)`` -- the one family that defaults on.
    autoscale_default: ClassVar[bool] = True

    def __post_init__(self):
        _validate_positive(self.shape, "shape")
        _validate_positive(self.scale, "scale")

    def _pymc_dist(self):
        import pymc as pm

        # Only the *unshifted, unnegated* gamma is a standard family; build()
        # applies `shift - g` on top of it.
        return pm.Gamma, {"alpha": self.shape, "beta": 1.0 / self.scale}

    def build(self, name: str, shape=None, positive: bool = False, dims=None):
        """Build ``shift - Gamma(shape, scale)``.

        The support is ``(-inf, shift]``: as in R, this puts a priori
        non-positive mass on an intervention effect. The underlying gamma is
        registered as ``f"{name}_gamma"`` and the returned (named) variable is a
        ``Deterministic``.
        """
        import pymc as pm

        if positive:
            raise ValueError(
                "shifted_gamma has support (-inf, shift]; positive=True is not "
                "meaningful for it. Use normal/student_t/cauchy/exponential for a "
                "positive parameter."
            )
        cls, kwargs = self._pymc_dist()
        g = cls(f"{name}_gamma", shape=shape, dims=dims, **kwargs)
        return pm.Deterministic(name, self.shift - g, dims=dims)


@dataclass(frozen=True)
class HexpPrior(Prior):
    """Hierarchical exponential prior on seeded infections. See :func:`hexp`."""

    prior_aux: Prior = ExponentialPrior(0.03)
    dist: ClassVar[str] = "hexp"
    positive_support: ClassVar[bool] = True

    def __post_init__(self):
        # R: check_prior(prior_aux); check_in_set(prior_aux$dist, ok_aux_dists)
        resolve(self.prior_aux, allowed=OK_AUX_DISTS, what="prior_aux")

    def _pymc_dist(self):
        raise ValueError(
            "hexp is hierarchical (tau, then seeds | tau) and has no single PyMC "
            "distribution; use its build() method."
        )

    def build(self, name: str, shape=None, positive: bool = False, dims=None):
        """Build ``tau ~ prior_aux`` and ``x | tau ~ Exponential(mean=tau)``.

        The seeds are written non-centred (``tau`` times a unit exponential),
        as R's Stan program does, which keeps the funnel geometry tractable for
        NUTS. ``tau`` is registered as ``f"{name}_tau"``, the unit exponentials
        as ``f"{name}_raw"``, and the returned variable is a ``Deterministic``
        named ``name``. ``positive`` is ignored: the result is positive by
        construction.
        """
        import pymc as pm

        # The auxiliary parameter is a *mean*, so it must be positive whatever
        # family it comes from (R declares it `real<lower=0>` in Stan).
        tau = self.prior_aux.build(f"{name}_tau", positive=True)
        raw = pm.Exponential(f"{name}_raw", 1.0, shape=shape, dims=dims)
        return pm.Deterministic(name, tau * raw, dims=dims)


class _CovariancePrior(Prior):
    """Base for priors that describe a covariance rather than a scalar."""

    def covariance_params(self) -> dict[str, Any]:
        """Hyperparameters, for a caller building the covariance itself."""
        return self.params()

    def _pymc_dist(self):
        raise ValueError(self._misuse_message())

    def _misuse_message(self) -> str:
        return (
            f"{self.dist} describes the covariance of a vector of correlated "
            "group-specific effects, not a single random variable. Use "
            "epidemia.priors.build_covariance(spec, name, n) to get a Cholesky "
            "factor, or read the hyperparameters with spec.covariance_params()."
        )

    def build(self, name: str, shape=None, positive: bool = False, dims=None):
        """Always raises: covariance priors do not build a scalar RV."""
        raise ValueError(self._misuse_message())


@dataclass(frozen=True)
class DecovPrior(_CovariancePrior):
    """Decomposition-of-covariance prior. See :func:`decov`."""

    regularization: float = 1.0
    concentration: float = 1.0
    shape: float = 1.0
    scale: float = 1.0
    dist: ClassVar[str] = "decov"

    def __post_init__(self):
        _validate_positive(self.regularization, "regularization")
        _validate_positive(self.concentration, "concentration")
        _validate_positive(self.shape, "shape")
        _validate_positive(self.scale, "scale")


@dataclass(frozen=True)
class LKJPrior(_CovariancePrior):
    """LKJ prior on a correlation matrix plus half-t scales. See :func:`lkj`."""

    regularization: float = 1.0
    scale: float = 10.0
    df: float = 1.0
    dist: ClassVar[str] = "lkj"

    def __post_init__(self):
        _validate_positive(self.regularization, "regularization")
        _validate_positive(self.scale, "scale")
        _validate_positive(self.df, "df")


# ---------------------------------------------------------------------------
# Constructors -- names, argument names and defaults mirror R/priors.R.
# ---------------------------------------------------------------------------


def normal(location: float = 0.0, scale: float = DEFAULT_PRIOR_SCALE) -> NormalPrior:
    """Normal prior with mean ``location`` and standard deviation ``scale``."""
    return NormalPrior(location=location, scale=scale)


def student_t(df: float = 1.0, location: float = 0.0,
              scale: float = DEFAULT_PRIOR_SCALE) -> StudentTPrior:
    """Student-t prior with ``df`` degrees of freedom."""
    return StudentTPrior(df=df, location=location, scale=scale)


def cauchy(location: float = 0.0, scale: float = DEFAULT_PRIOR_SCALE) -> CauchyPrior:
    """Cauchy prior.

    R implements ``cauchy()`` as ``student_t(df = 1)`` and so reports
    ``dist == "t"``; here it keeps its own name, ``"cauchy"`` (also an
    admissible family in R's ``ok_dists``), and builds a ``pm.Cauchy`` -- the
    same distribution, either way.
    """
    return CauchyPrior(location=location, scale=scale)


def exponential(rate: float = 1.0) -> ExponentialPrior:
    """Exponential prior with the given ``rate`` (mean ``1 / rate``)."""
    return ExponentialPrior(rate=rate)


def laplace(location: float = 0.0, scale: float = DEFAULT_PRIOR_SCALE) -> LaplacePrior:
    """Laplace (double-exponential) prior."""
    return LaplacePrior(location=location, scale=scale)


def hexp(prior_aux: Prior | None = None) -> HexpPrior:
    """Hierarchical exponential prior for seeded infections.

    Seeds in each population get an exponential prior whose *mean* (``tau``) is
    shared across populations and itself given ``prior_aux``. This is R's
    ``epiinf`` default, ``hexp(prior_aux = exponential(0.03))``, which gives the
    shared mean a prior mean of about 33 infections per day.
    """
    return HexpPrior(prior_aux=exponential(0.03) if prior_aux is None else prior_aux)


def shifted_gamma(shape: float = 1.0, scale: float = 1.0,
                  shift: float = 0.0) -> ShiftedGammaPrior:
    """Shifted gamma prior: ``beta = shift - Gamma(shape, scale)``.

    Used for non-pharmaceutical-intervention effects, which are a priori
    unlikely to *increase* transmission: with ``shift = 0`` the support is the
    negative half line, and a small positive ``shift`` allows a little positive
    mass (Flaxman et al., 2020).
    """
    return ShiftedGammaPrior(shape=shape, scale=scale, shift=shift)


def decov(regularization: float = 1.0, concentration: float = 1.0,
          shape: float = 1.0, scale: float = 1.0) -> DecovPrior:
    """Decomposition-of-covariance prior for correlated group-specific terms.

    Describes a covariance, so it has no scalar ``build``; pass it to
    :func:`build_covariance`.
    """
    return DecovPrior(regularization=regularization, concentration=concentration,
                      shape=shape, scale=scale)


def lkj(regularization: float = 1.0, scale: float = 10.0,
        df: float = 1.0) -> LKJPrior:
    """LKJ prior on the correlations plus half-t priors on the scales.

    As in R, ``lkj`` is provided for completeness; the covariance prior actually
    used by the model is :func:`decov` (see :data:`OK_COV_DISTS`).
    """
    return LKJPrior(regularization=regularization, scale=scale, df=df)


#: Families accepted for regression coefficients (R's ``ok_dists``). The
#: shrinkage families -- hs, hs_plus, lasso, product_normal -- are defined at the
#: bottom of this module and added there, so this set matches R's exactly.
OK_DISTS = frozenset({"gamma", "normal", "t", "cauchy", "laplace", "hexp"})
#: Families accepted for intercepts (R's ``ok_int_dists``).
OK_INT_DISTS = frozenset({"normal", "t", "cauchy"})
#: Families accepted for auxiliary parameters (R's ``ok_aux_dists``).
OK_AUX_DISTS = frozenset({"normal", "t", "cauchy", "exponential"})
#: Families accepted as a covariance prior (R's ``ok_cov_dists``).
OK_COV_DISTS = frozenset({"decov"})

# Family name -> constructor, so a caller may name a family as a string and get
# its defaults (`resolve("normal")`).
_FAMILIES: dict[str, Any] = {
    "normal": normal,
    "t": student_t,
    "student_t": student_t,
    "cauchy": cauchy,
    "exponential": exponential,
    "laplace": laplace,
    "gamma": shifted_gamma,
    "shifted_gamma": shifted_gamma,
    "hexp": hexp,
    "decov": decov,
    "lkj": lkj,
}


def resolve(spec: Prior | str | None, default: Prior | None = None, *,
            allowed=None, what: str = "prior") -> Prior:
    """Return a :class:`Prior`, falling back to ``default`` when none is given.

    Lets a caller accept either a prior spec or nothing::

        prior = resolve(config.prior_intercept, normal(0, 0.5),
                        allowed=OK_INT_DISTS, what="prior_intercept")

    Parameters
    ----------
    spec : Prior | str | None
        A prior spec, a family name (which is constructed with its defaults),
        or ``None`` to use ``default``.
    default : Prior | None
        Used when ``spec`` is ``None``.
    allowed : set of str | None
        Admissible family names, e.g. :data:`OK_AUX_DISTS`. Mirrors R's
        ``check_in_set(prior$dist, ok_*_dists)``: the set of usable families is
        fixed by the model, so an unsupported one is an error, not a silent
        fallback.
    what : str
        Name of the argument, used in error messages.

    Raises
    ------
    TypeError
        If ``spec`` is neither a :class:`Prior`, a family name nor ``None``.
    ValueError
        If no prior is available, or the family is not in ``allowed``.
    """
    if spec is None:
        spec = default
    if spec is None:
        raise ValueError(f"{what}: no prior given and no default available")
    if isinstance(spec, str):
        if spec not in _FAMILIES:
            raise ValueError(
                f"{what}: unknown prior family {spec!r}; "
                f"known families are {', '.join(sorted(_FAMILIES))}"
            )
        spec = _FAMILIES[spec]()
    if not isinstance(spec, Prior):
        raise TypeError(
            f"{what} should be a prior specification (e.g. normal(0, 1), "
            f"student_t(3, 0, 1), shifted_gamma(...)), got {type(spec).__name__}"
        )
    if allowed is not None and spec.dist not in allowed:
        raise ValueError(
            f"{what}: prior family {spec.dist!r} is not supported here; "
            f"use one of {', '.join(sorted(allowed))}"
        )
    return spec


def build(spec: Prior | str, name: str, shape=None, positive: bool = False,
          dims=None, allowed=None):
    """Create the PyMC random variable described by ``spec``.

    Must be called inside a ``pm.Model`` context. Thin wrapper around
    ``spec.build(...)`` that also accepts a family name; see
    :meth:`Prior.build` for the arguments.

    ``allowed`` restricts the family, as R does at every one of its own prior
    entry points (``epirt``, ``epiinf`` and ``handle_glm_prior``'s ``ok_dists``).
    Leaving it ``None`` accepts anything, which is what the shrinkage families
    and the covariance builders want.
    """
    return resolve(spec, allowed=allowed, what=name).build(
        name, shape=shape, positive=positive, dims=dims)


def build_covariance(spec: Prior | str, name: str, n: int):
    """Build the Cholesky factor of the covariance of ``n`` group-specific effects.

    This is the helper that :func:`decov` and :func:`lkj` point at, since those
    describe a covariance and have no scalar ``build``. Use the returned factor
    ``L`` to give a group's effects a correlated normal prior non-centred, as
    ``b = z @ L.T`` with ``z ~ Normal(0, 1)``.

    The two families decompose the covariance differently:

    ``decov``
        As in rstanarm (and hence R's Stan program): a single scale
        ``tau ~ Gamma(shape, scale)`` is split across the ``n`` effects by a
        symmetric Dirichlet, ``pi ~ Dirichlet(concentration)``, giving
        ``sd_j = tau * sqrt(n * pi_j)``; the correlations get
        ``LKJ(regularization)``. With ``n = 1`` this reduces to
        ``sd ~ Gamma(shape, scale)``, matching R's documented behaviour.
    ``lkj``
        Independent half-t scales, ``sd_j ~ HalfStudentT(df, scale)``, with
        ``LKJ(regularization)`` correlations.

    Parameters
    ----------
    spec : DecovPrior | LKJPrior | str
        The covariance prior.
    name : str
        Base name. The scales are registered as ``f"{name}_sd"``, the
        correlation matrix as ``f"{name}_corr"`` (only when ``n > 1``), and the
        Cholesky factor as ``name``.
    n : int
        Number of correlated effects in the group.

    Returns
    -------
    pytensor.tensor.TensorVariable
        Lower-triangular Cholesky factor, shape ``(n, n)``.
    """
    import pymc as pm
    import pytensor.tensor as pt

    spec = resolve(spec, allowed={"decov", "lkj"}, what=name)
    n = int(n)
    if n < 1:
        raise ValueError(f"n should be a positive integer, got {n}")

    if isinstance(spec, DecovPrior):
        tau = pm.Gamma(f"{name}_tau", alpha=spec.shape, beta=1.0 / spec.scale)
        if n == 1:
            sd = pm.Deterministic(f"{name}_sd", tau.reshape((1,)))
        else:
            pi = pm.Dirichlet(f"{name}_pi", np.full(n, float(spec.concentration)))
            sd = pm.Deterministic(f"{name}_sd", tau * pt.sqrt(n * pi))
    else:  # LKJPrior
        sd = pm.HalfStudentT(f"{name}_sd", nu=spec.df, sigma=spec.scale, shape=n)

    if n == 1:
        chol = sd.reshape((1, 1))
    else:
        corr = pm.LKJCorr(f"{name}_corr", n=n, eta=spec.regularization,
                          return_matrix=True)
        chol = sd[:, None] * pt.linalg.cholesky(corr)
    return pm.Deterministic(name, chol)


# ===========================================================================
# Shrinkage families and autoscaling
# ===========================================================================
#
# R's ``ok_dists`` also admits hs, hs_plus, lasso and product_normal for
# regression coefficients, and every scale-bearing constructor takes
# ``autoscale``. Both are added below.
#
# The shrinkage families are *hierarchical*: like ``hexp`` and
# ``shifted_gamma`` above, they have no single PyMC distribution, so
# ``_pymc_dist()`` raises and ``build()`` assembles the graph. All four are
# written non-centred (a standardised variable times its scales), because the
# funnel between a coefficient and its own local scale is exactly the geometry
# NUTS cannot traverse when written centred -- the same reason R's Stan program
# stores ``z_beta`` and reconstructs ``beta`` in ``make_beta()``.


class _ShrinkagePrior(Prior):
    """Base for the shrinkage families: hierarchical, so no single PyMC family."""

    def _pymc_dist(self):
        raise ValueError(
            f"{self.dist} is hierarchical (global and local scales, then the "
            "coefficients given those scales) and has no single PyMC "
            "distribution; use its build() method."
        )

    def _reject_positive(self, positive: bool) -> None:
        """Shrinkage priors are symmetric about their location; truncation is out.

        Truncating would silently break the non-centred parameterisation (the
        scales would no longer be the scales of the truncated variable), and R
        never applies a shrinkage prior to a ``real<lower=0>`` parameter -- they
        are only in ``ok_dists``, never in ``ok_aux_dists``.
        """
        if positive:
            raise ValueError(
                f"{self.dist} is a shrinkage prior on regression coefficients "
                "and is symmetric about its location; positive=True is not "
                "meaningful for it. Use normal/student_t/cauchy/exponential for "
                "a positive parameter."
            )


def _regularised_local_scale(name: str, lam, tau, slab_df: float,
                             slab_scale: float):
    """Slab-regularise a horseshoe's local scales (Piironen & Vehtari, 2017).

    The plain horseshoe leaves the large coefficients essentially unpenalised,
    which makes the posterior improper under a flat likelihood (e.g. separation)
    and gives NUTS heavy tails to explore. The *regularised* horseshoe multiplies
    each local scale by a slab of width ``c``, so a coefficient the data do not
    pin down is shrunk towards a ``Normal(0, slab_scale)`` instead of towards
    nothing::

        c^2 ~ InvGamma(slab_df / 2, slab_df * slab_scale^2 / 2)
        lambda_tilde^2 = c^2 lambda^2 / (c^2 + tau^2 lambda^2)

    The squared slab ``c^2`` is registered as ``f"{name}_slab"``.
    """
    import pymc as pm
    import pytensor.tensor as pt

    c2 = pm.InverseGamma(
        f"{name}_slab",
        alpha=0.5 * slab_df,
        beta=0.5 * slab_df * slab_scale ** 2,
    )
    lam2 = lam ** 2
    return pt.sqrt(c2 * lam2 / (c2 + tau ** 2 * lam2))


@dataclass(frozen=True)
class HorseshoePrior(_ShrinkagePrior):
    """Regularised horseshoe prior. See :func:`hs`."""

    df: float = 1.0
    global_df: float = 1.0
    global_scale: float = 0.01
    slab_df: float = 4.0
    slab_scale: float = 2.5
    dist: ClassVar[str] = "hs"

    def __post_init__(self):
        _validate_positive(self.df, "df")
        _validate_positive(self.global_df, "global_df")
        _validate_positive(self.global_scale, "global_scale")
        _validate_positive(self.slab_df, "slab_df")
        _validate_positive(self.slab_scale, "slab_scale")

    # R's hs() also records location = 0 and scale = 1; they are fixed by the
    # family rather than chosen, so they are properties here and stay out of
    # params().
    @property
    def location(self) -> float:
        """0 -- the horseshoe is centred at zero, as in R."""
        return 0.0

    @property
    def scale(self) -> float:
        """1 -- the overall scale is ``global_scale`` times the local scales."""
        return 1.0

    def build(self, name: str, shape=None, positive: bool = False, dims=None):
        """Build the regularised horseshoe, non-centred.

        ::

            tau    ~ HalfStudentT(global_df, global_scale)      # global
            lambda ~ HalfStudentT(df, 1)                        # local, per coef
            c^2    ~ InvGamma(slab_df/2, slab_df slab_scale^2/2)
            z      ~ Normal(0, 1)
            beta   = z * tau * lambda_tilde

        Sub-parameters are registered as ``f"{name}_global"``,
        ``f"{name}_local"``, ``f"{name}_slab"`` and ``f"{name}_z"``; the returned
        variable is a ``Deterministic`` named ``name``.
        """
        import pymc as pm

        self._reject_positive(positive)
        tau = pm.HalfStudentT(f"{name}_global", nu=self.global_df,
                              sigma=self.global_scale)
        lam = pm.HalfStudentT(f"{name}_local", nu=self.df, sigma=1.0,
                              shape=shape, dims=dims)
        # The slab is registered before z so that all the scales sit together
        # in the model's variable order, ahead of the standardised coefficients.
        lam_t = _regularised_local_scale(name, lam, tau, self.slab_df,
                                         self.slab_scale)
        z = pm.Normal(f"{name}_z", 0.0, 1.0, shape=shape, dims=dims)
        return pm.Deterministic(name, z * tau * lam_t, dims=dims)


@dataclass(frozen=True)
class HorseshoePlusPrior(_ShrinkagePrior):
    """Regularised horseshoe+ prior. See :func:`hs_plus`."""

    df1: float = 1.0
    df2: float = 1.0
    global_df: float = 1.0
    global_scale: float = 0.01
    slab_df: float = 4.0
    slab_scale: float = 2.5
    dist: ClassVar[str] = "hs_plus"

    def __post_init__(self):
        _validate_positive(self.df1, "df1")
        _validate_positive(self.df2, "df2")
        _validate_positive(self.global_df, "global_df")
        _validate_positive(self.global_scale, "global_scale")
        _validate_positive(self.slab_df, "slab_df")
        _validate_positive(self.slab_scale, "slab_scale")

    # R packs the two local degrees of freedom into the generic slots of its
    # prior list -- `df = df1`, `scale = df2` -- so that handle_glm_prior() can
    # stay generic. Expose the same two names for anyone reading R code.
    @property
    def df(self) -> float:
        """``df1`` -- the slot R's ``hs_plus()`` stores it in."""
        return self.df1

    @property
    def scale(self) -> float:
        """``df2`` -- the slot R's ``hs_plus()`` stores it in (not a scale)."""
        return self.df2

    @property
    def location(self) -> float:
        """0 -- the horseshoe+ is centred at zero, as in R."""
        return 0.0

    def build(self, name: str, shape=None, positive: bool = False, dims=None):
        """Build the regularised horseshoe+, non-centred.

        The horseshoe+ differs from the horseshoe only in the local scale, which
        is a *product* of two half-t variables::

            lambda = lambda1 * lambda2,
            lambda1 ~ HalfStudentT(df1, 1), lambda2 ~ HalfStudentT(df2, 1)

        The extra level puts even more mass near zero and even heavier tails, so
        strong signals are shrunk less than under the horseshoe. Sub-parameters
        are ``f"{name}_global"``, ``f"{name}_local1"``, ``f"{name}_local2"``,
        ``f"{name}_slab"`` and ``f"{name}_z"``.
        """
        import pymc as pm

        self._reject_positive(positive)
        tau = pm.HalfStudentT(f"{name}_global", nu=self.global_df,
                              sigma=self.global_scale)
        lam1 = pm.HalfStudentT(f"{name}_local1", nu=self.df1, sigma=1.0,
                               shape=shape, dims=dims)
        lam2 = pm.HalfStudentT(f"{name}_local2", nu=self.df2, sigma=1.0,
                               shape=shape, dims=dims)
        lam_t = _regularised_local_scale(name, lam1 * lam2, tau, self.slab_df,
                                         self.slab_scale)
        z = pm.Normal(f"{name}_z", 0.0, 1.0, shape=shape, dims=dims)
        return pm.Deterministic(name, z * tau * lam_t, dims=dims)


@dataclass(frozen=True)
class LassoPrior(_ShrinkagePrior):
    """Bayesian lasso: Laplace with an estimated global scale. See :func:`lasso`."""

    df: float = 1.0
    location: float = 0.0
    scale: float = DEFAULT_PRIOR_SCALE
    dist: ClassVar[str] = "lasso"

    def __post_init__(self):
        _validate_positive(self.df, "df")
        _validate_positive(self.scale, "scale")

    def build(self, name: str, shape=None, positive: bool = False, dims=None):
        """Build ``location + global * scale * Laplace(0, 1)``.

        This is the Bayesian lasso as rstanarm (and hence R's Stan program)
        writes it: the double-exponential's scale is not fixed at ``scale`` but
        multiplied by a shared ``global ~ ChiSquared(df)``, so the amount of
        shrinkage is estimated from the data rather than assumed. Sub-parameters
        are ``f"{name}_global"`` and ``f"{name}_z"``.
        """
        import pymc as pm

        self._reject_positive(positive)
        g = pm.ChiSquared(f"{name}_global", nu=self.df)
        z = pm.Laplace(f"{name}_z", mu=0.0, b=1.0, shape=shape, dims=dims)
        return pm.Deterministic(name, self.location + self.scale * g * z,
                                dims=dims)


@dataclass(frozen=True)
class ProductNormalPrior(_ShrinkagePrior):
    """Product of independent normals. See :func:`product_normal`."""

    num_terms: int = 2
    location: float = 0.0
    scale: float = 1.0
    #: R names this argument ``df``; see the :attr:`df` alias.
    dist: ClassVar[str] = "product_normal"

    def __post_init__(self):
        _validate_positive(self.num_terms, "num_terms")
        # R: stopifnot(all(df >= 1), all(df == as.integer(df)))
        if int(self.num_terms) != self.num_terms or self.num_terms < 1:
            raise ValueError(
                f"num_terms should be an integer >= 1, got {self.num_terms!r}"
            )
        _validate_positive(self.scale, "scale")

    @property
    def df(self) -> float:
        """``num_terms`` -- the slot R's ``product_normal()`` stores it in."""
        return self.num_terms

    def build(self, name: str, shape=None, positive: bool = False, dims=None):
        """Build ``location + scale**num_terms * z_1 * ... * z_num_terms``.

        A product of independent normals is a shrinkage prior with a spike at
        zero (any factor near zero kills the coefficient) and polynomial tails;
        with ``num_terms = 1`` it degenerates to ``Normal(location, scale)``.
        Each factor is a standard normal registered as ``f"{name}_z1"`` ...
        ``f"{name}_z{num_terms}"``; the scale is applied once per factor, as in
        R's ``make_beta()`` (``beta *= prior_scale ^ num_normals``).
        """
        import pymc as pm

        self._reject_positive(positive)
        k = int(self.num_terms)
        prod = None
        for i in range(k):
            z = pm.Normal(f"{name}_z{i + 1}", 0.0, 1.0, shape=shape, dims=dims)
            prod = z if prod is None else prod * z
        return pm.Deterministic(name, self.location + self.scale ** k * prod,
                                dims=dims)


def hs(df: float = 1.0, global_df: float = 1.0, global_scale: float = 0.01,
       slab_df: float = 4.0, slab_scale: float = 2.5) -> HorseshoePrior:
    """Regularised horseshoe prior for regression coefficients.

    Shrinks small effects hard towards zero while leaving large ones alone --
    useful when many of the covariates (e.g. NPIs) are expected to do nothing.
    ``global_scale`` controls the overall sparsity and should be set from the
    number of coefficients believed to be non-zero; ``slab_df``/``slab_scale``
    describe the ``Student-t(slab_df, 0, slab_scale)`` slab that the
    unambiguously non-zero coefficients are shrunk towards.

    Takes no ``autoscale``: as in R the family fixes ``location = 0`` and
    ``scale = 1``, and the scaling is expressed by ``global_scale`` instead.
    """
    return HorseshoePrior(df=df, global_df=global_df, global_scale=global_scale,
                          slab_df=slab_df, slab_scale=slab_scale)


def hs_plus(df1: float = 1.0, df2: float = 1.0, global_df: float = 1.0,
            global_scale: float = 0.01, slab_df: float = 4.0,
            slab_scale: float = 2.5) -> HorseshoePlusPrior:
    """Regularised horseshoe+ prior: two nested local scales.

    Like :func:`hs` but with ``lambda = lambda1 * lambda2``, which sharpens the
    spike at zero and fattens the tails -- more aggressive sparsification at the
    cost of a harder posterior geometry.
    """
    return HorseshoePlusPrior(df1=df1, df2=df2, global_df=global_df,
                              global_scale=global_scale, slab_df=slab_df,
                              slab_scale=slab_scale)


def lasso(df: float = 1.0, location: float = 0.0,
          scale: float = DEFAULT_PRIOR_SCALE) -> LassoPrior:
    """Bayesian lasso prior: Laplace with an estimated global scale.

    The double-exponential's scale is ``scale`` times a shared
    ``ChiSquared(df)`` variable, so the penalty is estimated rather than fixed.
    Unlike the frequentist lasso this does not produce exact zeros; it is a
    shrinkage prior, not a selection procedure.
    """
    return LassoPrior(df=df, location=location, scale=scale)


def product_normal(num_terms: int = 2, location: float = 0.0,
                   scale: float = 1.0) -> ProductNormalPrior:
    """Prior on a coefficient that is a product of ``num_terms`` normals.

    R calls the first argument ``df``; it is a count of factors, so it is named
    ``num_terms`` here and mirrored back as ``.df``. It must be an integer
    ``>= 1``.
    """
    return ProductNormalPrior(num_terms=num_terms, location=location,
                              scale=scale)


# ---------------------------------------------------------------------------
# Autoscaling
# ---------------------------------------------------------------------------

#: Families for which :func:`autoscale` does anything -- the ones whose ``scale``
#: is a scale *of the coefficient*, so dividing it by the predictor's scale keeps
#: the prior invariant to the covariate's units. Mirrors R, where ``autoscale``
#: is an argument of normal/student_t/cauchy/laplace/lasso.
# "gamma" (shifted_gamma) belongs here: R routes it through the same
# `prior_scale <- prior$scale` branch as normal/t/cauchy (R/helpers.R:699-707)
# and then divides that scale in standata_reg(). It is also the ONE family whose
# autoscale defaults to TRUE, and the default R_t covariate prior in the
# vignettes, so leaving it out silently changed every such model.
AUTOSCALE_DISTS = frozenset({"normal", "t", "cauchy", "laplace", "lasso", "gamma"})

#: R's ``min_prior_scale`` in standata_reg(): a floor, so a wildly-scaled
#: predictor cannot collapse the prior to a point mass at zero.
MIN_PRIOR_SCALE = 1e-12



def predictor_scale(x) -> Any:
    """Scale of a predictor, by R's rule in ``standata_reg()``.

    R does not blindly use the standard deviation:

    ==================== =========================================
    unique values        scale
    ==================== =========================================
    1 (constant)         ``1`` -- nothing to rescale
    2 (a dummy/binary)   ``max(x) - min(x)``, the range
    > 2                  ``sd(x)`` (R's ``sd``, i.e. the n-1 divisor)
    ==================== =========================================

    The range is used for a binary covariate because there its coefficient is a
    difference between two groups, not a per-standard-deviation slope.

    Parameters
    ----------
    x : array_like
        A predictor, shape ``(n,)``, or a design matrix, shape ``(n, k)``, in
        which case the rule is applied column by column.

    Returns
    -------
    float or numpy.ndarray
        The scale (a scalar for a vector ``x``, one per column for a matrix).
    """
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        u = np.unique(x[np.isfinite(x)])
        if u.size <= 1:
            return 1.0
        if u.size == 2:
            return float(u[-1] - u[0])
        return float(np.std(x, ddof=1))
    if x.ndim != 2:
        raise ValueError(f"x should be 1- or 2-dimensional, got ndim={x.ndim}")
    return np.array([predictor_scale(col) for col in x.T])


def autoscale(spec: Prior | str, predictor_sd) -> Prior:
    """Rescale a prior to the units of its predictor -- R's ``autoscale = TRUE``.

    A ``normal(0, 0.5)`` prior on a coefficient means "half a unit of response
    per unit of covariate", which is a completely different statement when the
    covariate is measured in days rather than weeks. R's ``autoscale`` removes
    that dependence by dividing the prior scale by the predictor's own scale, so
    the prior is a statement about *standardised* covariates::

        scale <- max(1e-12, scale / predictor_scale(x))

    Use :func:`predictor_scale` to get ``predictor_sd`` the way R does (range for
    binary covariates, standard deviation otherwise).

    Parameters
    ----------
    spec : Prior | str
        The prior to rescale. Not mutated -- specs are frozen, so a copy is
        returned.
    predictor_sd : float or array_like
        Scale of the predictor(s), strictly positive. An array gives a per-
        coefficient scale, which the families here accept (their PyMC
        distributions broadcast over the coefficient axis).

    Returns
    -------
    Prior
        A copy of ``spec`` with ``scale`` divided by ``predictor_sd``, floored at
        :data:`MIN_PRIOR_SCALE`.

    Notes
    -----
    This is a **no-op** for every family outside :data:`AUTOSCALE_DISTS`, and the
    spec is returned unchanged -- matching R, where those families either have no
    ``autoscale`` argument (``hs``, ``hs_plus``, ``product_normal``, ``decov``)
    or carry a ``scale`` that is not a coefficient scale and is never divided in
    ``standata_reg()`` (``exponential``'s rate, ``hexp``'s hierarchy, ``lkj``'s
    half-t scale).

    The existing constructors deliberately gained no ``autoscale`` field:
    ``normal()`` and friends are frozen dataclasses whose :meth:`Prior.params`
    output is part of the tested R-parity contract, so adding a field would
    change what they report even when it is ``False``. Autoscaling is therefore
    exposed only through this function -- call it where R would have consulted
    ``prior$autoscale``, i.e. once the design matrix is known.

    Examples
    --------
    >>> autoscale(normal(0, 0.5), 2.0)
    NormalPrior(location=0, scale=0.25)
    """
    spec = resolve(spec, what="autoscale")
    if spec.dist not in AUTOSCALE_DISTS:
        # Deliberately silent: a pipeline autoscales whatever prior the user
        # supplied, and R likewise just leaves such a prior alone.
        return spec

    sd = np.asarray(predictor_sd, dtype=float)
    if not np.all(np.isfinite(sd)) or np.any(sd <= 0):
        raise ValueError(
            "predictor_sd should be positive and finite, got "
            f"{predictor_sd!r}; a constant predictor has scale 1 (see "
            "predictor_scale)"
        )
    scale = np.asarray(spec.scale, dtype=float) / sd
    # R: pmax(min_prior_scale, ...) then pmin(.Machine$double.xmax, ...)
    scale = np.clip(scale, MIN_PRIOR_SCALE, np.finfo(float).max)
    # replace() re-runs __post_init__, so the rescaled spec is validated too.
    return replace(spec, scale=float(scale) if scale.ndim == 0 else scale)


# The shrinkage families complete R's ok_dists; nothing else changes, since R
# admits them for coefficients only (never for intercepts, aux or covariance).
OK_DISTS = OK_DISTS | frozenset({"hs", "hs_plus", "lasso", "product_normal"})

_FAMILIES.update({
    "hs": hs,
    "hs_plus": hs_plus,
    "lasso": lasso,
    "product_normal": product_normal,
})

__all__ += [
    "AUTOSCALE_DISTS",
    "MIN_PRIOR_SCALE",
    "HorseshoePlusPrior",
    "HorseshoePrior",
    "LassoPrior",
    "ProductNormalPrior",
    "autoscale",
    "hs",
    "hs_plus",
    "lasso",
    "predictor_scale",
    "product_normal",
]
