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

from dataclasses import dataclass, fields
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
    scale: float = 1.0
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
    scale: float = 1.0
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
    scale: float = 1.0
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
    scale: float = 1.0
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


def normal(location: float = 0.0, scale: float = 1.0) -> NormalPrior:
    """Normal prior with mean ``location`` and standard deviation ``scale``."""
    return NormalPrior(location=location, scale=scale)


def student_t(df: float = 1.0, location: float = 0.0,
              scale: float = 1.0) -> StudentTPrior:
    """Student-t prior with ``df`` degrees of freedom."""
    return StudentTPrior(df=df, location=location, scale=scale)


def cauchy(location: float = 0.0, scale: float = 1.0) -> CauchyPrior:
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


def laplace(location: float = 0.0, scale: float = 1.0) -> LaplacePrior:
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


#: Families accepted for regression coefficients (R's ``ok_dists``, restricted to
#: those implemented here -- R additionally allows hs, hs_plus, lasso and
#: product_normal, which this port does not provide).
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
          dims=None):
    """Create the PyMC random variable described by ``spec``.

    Must be called inside a ``pm.Model`` context. Thin wrapper around
    ``spec.build(...)`` that also accepts a family name; see
    :meth:`Prior.build` for the arguments.
    """
    return resolve(spec, what=name).build(name, shape=shape, positive=positive,
                                          dims=dims)


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
