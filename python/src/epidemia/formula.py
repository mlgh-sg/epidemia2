"""R's formula mini-language for epidemia models.

In R a model is written down rather than assembled::

    R(country, date) ~ 1 + (1 + lockdown || country) + rw(time = week)

The left hand side is the ``R(group, date)`` sugar that :func:`epidemia::epirt`
requires; the right hand side is a sum of terms, each of which is a fixed
effect, a group-specific ("random") effect written ``(expr | factor)``, or an
autocorrelation term written ``rw()``. This module parses that string into a
:class:`FormulaSpec` and, through :func:`build_from_formula`, drives
:func:`epidemia.core.prepare_panel` with it.

The R semantics reproduced here:

* ``0 +`` (equivalently ``-1``) drops the intercept, leaving the group-specific
  intercepts to carry the baseline.
* ``|`` gives *correlated* group-specific terms (R's ``decov`` prior on a full
  covariance); ``||`` gives independent ones.
* ``(expr | factor)`` parses ``expr`` as its own little formula, so the
  intercept in ``(lockdown | country)`` is implicit -- that term means
  ``(1 + lockdown | country)``. Write ``(0 + lockdown | country)`` to avoid it.
* ``rw(gr = x)`` gives one walk per level of ``x``; ``rw()`` without ``gr``
  gives a single walk shared by every group. ``time`` names the column that
  indexes the walk's steps and defaults to the date column.

Parsing is deliberately string-based rather than a full expression grammar:
these formulas are small and R itself only ever deparses them (see
``R/autocor.R``, which finds ``rw()`` terms with a regular expression).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

__all__ = [
    "RandomEffect",
    "RwTerm",
    "FormulaSpec",
    "parse_formula",
    "build_from_formula",
]


# ---------------------------------------------------------------------------
# grammar fragments
# ---------------------------------------------------------------------------

_IDENT = r"[A-Za-z_.][A-Za-z0-9_.]*"
_IDENT_RE = re.compile(rf"^{_IDENT}$")
# Fixed terms may be interactions (`lockdown:country`), which R writes with `:`.
_FIXED_RE = re.compile(rf"^{_IDENT}(\s*:\s*{_IDENT})*$")
_CALL_RE = re.compile(rf"^({_IDENT})\s*\((.*)\)$", re.S)
_LHS_SUGAR_RE = re.compile(rf"^R\s*\(\s*({_IDENT})\s*,\s*({_IDENT})\s*\)$")
# `.*?` stops at the first bar, and `\|\|?` then eats a second one if present,
# so a single group of the regex distinguishes `|` from `||`.
_BAR_RE = re.compile(r"^\(\s*(?P<expr>.*?)\s*(?P<bars>\|\|?)\s*(?P<factor>[^|]*?)\s*\)$",
                     re.S)
_MINUS_ONE_RE = re.compile(r"^(?P<head>.*?)\s*-\s*1$", re.S)


@dataclass
class RandomEffect:
    """A group-specific term, R's ``(expr | factor)``.

    Attributes
    ----------
    terms : list[str]
        The terms of ``expr``. ``"1"`` marks the group-specific intercept and is
        present unless ``expr`` starts with ``0 +`` / ``-1``, because R parses
        ``expr`` into a model matrix and so adds the intercept implicitly.
    factor : str
        Column whose levels index the effects.
    correlated : bool
        ``True`` for ``|`` (a full covariance across ``terms``), ``False`` for
        ``||`` (independent effects).
    """

    terms: list[str]
    factor: str
    correlated: bool = True

    @property
    def intercept(self) -> bool:
        """Whether this term carries a group-specific intercept."""
        return "1" in self.terms

    @property
    def covariates(self) -> list[str]:
        """The non-intercept terms, i.e. the group-specific slopes."""
        return [t for t in self.terms if t != "1"]

    def __str__(self) -> str:
        bars = "|" if self.correlated else "||"
        return f"({' + '.join(self.terms)} {bars} {self.factor})"


@dataclass
class RwTerm:
    """An autocorrelation term, R's ``rw(time, gr, prior_scale)``.

    Attributes
    ----------
    time : str | None
        Column indexing the steps of the walk (e.g. an ISO week). ``None`` means
        the date column, i.e. a daily walk.
    gr : str | None
        Column whose levels each get their own independent walk. ``None`` means
        one walk shared by every group.
    prior_scale : float
        Scale of the half-normal hyperprior on the step size.
    """

    time: str | None = None
    gr: str | None = None
    prior_scale: float = 0.2

    def __str__(self) -> str:
        args = []
        if self.time is not None:
            args.append(f"time = {self.time}")
        if self.gr is not None:
            args.append(f"gr = {self.gr}")
        if self.prior_scale != 0.2:
            args.append(f"prior_scale = {self.prior_scale}")
        return f"rw({', '.join(args)})"


@dataclass
class FormulaSpec:
    """The parsed form of one model formula.

    Attributes
    ----------
    response : str
        ``"R"`` for a reproduction-number formula (see ``group``/``date``),
        otherwise the observed column, e.g. ``"deaths"``.
    group, date : str | None
        The two variables of the ``R(group, date)`` sugar. Both ``None`` for an
        observation formula such as ``deaths ~ 1``.
    intercept : bool
        ``False`` when the right hand side starts ``0 +`` (or contains ``-1``).
    fixed : list[str]
        Population-level covariate terms.
    random : list[RandomEffect]
    rw : list[RwTerm]
    text : str
        The formula as given, kept for error messages and ``__str__``.
    """

    response: str
    group: str | None = None
    date: str | None = None
    intercept: bool = True
    fixed: list[str] = field(default_factory=list)
    random: list[RandomEffect] = field(default_factory=list)
    rw: list[RwTerm] = field(default_factory=list)
    text: str = ""

    @property
    def is_rt(self) -> bool:
        """Whether this is an ``R(group, date) ~ ...`` formula."""
        return self.group is not None

    @property
    def correlated(self) -> bool:
        """Whether any group-specific term uses a single ``|``."""
        return any(r.correlated for r in self.random)

    @property
    def covariates(self) -> list[str]:
        """Every covariate column the formula needs, in order, de-duplicated.

        Fixed and group-specific slopes are pooled into one list because
        :func:`epidemia.core.build_epidemia_model` shares a single design matrix
        between the population-level ``beta`` and the region-level ``b``.
        """
        out: list[str] = []
        for name in list(self.fixed) + [t for r in self.random for t in r.covariates]:
            if name not in out:
                out.append(name)
        return out

    def __str__(self) -> str:
        return self.text or "<formula>"


# ---------------------------------------------------------------------------
# parsing
# ---------------------------------------------------------------------------


def _split_top_level(text: str, sep: str) -> list[str]:
    """Split on ``sep`` outside parentheses, keeping nested calls intact."""
    parts, depth, cur = [], 0, []
    for ch in text:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth < 0:
                raise ValueError(f"unbalanced parentheses in {text!r}")
        if ch == sep and depth == 0:
            parts.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    if depth != 0:
        raise ValueError(f"unbalanced parentheses in {text!r}")
    parts.append("".join(cur))
    return [p.strip() for p in parts]


def _parse_rw(args: str, source: str) -> RwTerm:
    """Parse the argument list of an ``rw(...)`` call.

    R does not evaluate these arguments -- ``rw()`` deparses them (see
    ``R/autocor.R``) -- so ``time`` and ``gr`` are column *names*, not values.
    """
    term = RwTerm()
    if not args.strip():
        return term

    positional = ["time", "gr", "prior_scale"]
    seen: set[str] = set()
    for i, raw in enumerate(_split_top_level(args, ",")):
        if not raw:
            raise ValueError(f"empty argument in {source!r}")
        if "=" in raw:
            key, _, value = raw.partition("=")
            key, value = key.strip(), value.strip()
        else:
            if i >= len(positional):
                raise ValueError(f"too many arguments in {source!r}")
            key, value = positional[i], raw
        if key not in positional:
            raise ValueError(
                f"unknown argument {key!r} in {source!r}; "
                f"rw() takes {positional}"
            )
        if key in seen:
            raise ValueError(f"argument {key!r} given twice in {source!r}")
        seen.add(key)

        value = value.strip().strip("'\"")
        if key == "prior_scale":
            try:
                term.prior_scale = float(value)
            except ValueError:
                raise ValueError(
                    f"prior_scale in {source!r} must be a number, got {value!r}"
                ) from None
            if term.prior_scale <= 0:
                raise ValueError(f"prior_scale in {source!r} must be positive")
        else:
            if not _IDENT_RE.match(value):
                raise ValueError(
                    f"{key} in {source!r} must be a column name, got {value!r}"
                )
            setattr(term, key, value)
    return term


def _parse_rhs(rhs: str, spec: FormulaSpec, *, nested: bool = False) -> list[str]:
    """Parse a right hand side into ``spec``; returns its plain terms.

    ``nested`` marks the inside of a ``(expr | factor)`` term, where further
    bars and ``rw()`` calls are not allowed. R's ``terms_rw()`` likewise skips
    any ``rw()`` that appears within a random-effect term.
    """
    terms: list[str] = []
    for raw in _split_top_level(rhs, "+"):
        token = raw.strip()
        if not token:
            raise ValueError(
                f"empty term in {spec.text!r}; check for a stray '+'"
            )

        # `x - 1` and `-1` both drop the intercept, so peel that off first.
        m = _MINUS_ONE_RE.match(token)
        if m:
            # Inside a bar the intercept is tracked through the term list, which
            # the caller re-reads; at the top level it lives on the spec.
            if not nested:
                spec.intercept = False
            terms = [t for t in terms if t != "1"]
            token = m.group("head").strip()
            if not token:
                continue

        if token.startswith("("):
            if nested:
                raise ValueError(
                    f"nested group-specific term {token!r} in {spec.text!r}; "
                    "'(a | b | c)' style nesting is not supported"
                )
            spec.random.append(_parse_bar(token, spec))
            continue

        if token in {"0", "-1"}:
            if not nested:
                spec.intercept = False
            terms = [t for t in terms if t != "1"]
            continue

        if token == "1":
            terms.append("1")
            continue

        call = _CALL_RE.match(token)
        if call:
            name, args = call.group(1), call.group(2)
            if name != "rw":
                raise ValueError(
                    f"unknown function {name!r} in {spec.text!r}; the only call "
                    "allowed on the right hand side is rw()"
                )
            if nested:
                raise ValueError(
                    f"rw() inside a group-specific term ({token!r}) is not "
                    "supported"
                )
            spec.rw.append(_parse_rw(args, token))
            continue

        if not _FIXED_RE.match(token):
            raise ValueError(f"cannot parse term {token!r} in {spec.text!r}")
        terms.append(token.replace(" ", ""))

    return terms


def _parse_bar(token: str, spec: FormulaSpec) -> RandomEffect:
    """Parse ``(expr | factor)`` / ``(expr || factor)``."""
    m = _BAR_RE.match(token)
    if not m:
        raise ValueError(
            f"malformed group-specific term {token!r} in {spec.text!r}; "
            "expected '(expr | factor)' or '(expr || factor)'"
        )
    factor = m.group("factor")
    if not _IDENT_RE.match(factor):
        if "/" in factor or ":" in factor:
            raise ValueError(
                f"nested or interacted grouping factor {factor!r} in {token!r} "
                "is not supported; expand it into separate terms"
            )
        raise ValueError(
            f"grouping factor {factor!r} in {token!r} must be a column name"
        )

    inner = FormulaSpec(response=spec.response, text=spec.text)
    terms = _parse_rhs(m.group("expr"), inner, nested=True)
    if inner.random or inner.rw:  # defensive: _parse_rhs rejects these already
        raise ValueError(f"cannot parse {token!r} in {spec.text!r}")
    # R parses `expr` into a model matrix, so the intercept is implicit unless
    # explicitly removed with `0 +` or `-1`.
    if "1" not in terms and _has_implicit_intercept(m.group("expr")):
        terms = ["1"] + terms
    terms = _dedup(terms)
    return RandomEffect(terms=terms, factor=factor,
                        correlated=m.group("bars") == "|")


def _has_implicit_intercept(expr: str) -> bool:
    """Whether ``expr`` keeps its implicit intercept (no ``0`` and no ``-1``)."""
    tokens = [t.strip() for t in _split_top_level(expr, "+")]
    for t in tokens:
        if t in {"0", "-1"} or _MINUS_ONE_RE.match(t):
            return False
    return True


def _dedup(names) -> list[str]:
    out: list[str] = []
    for n in names:
        if n not in out:
            out.append(n)
    return out


def parse_formula(text: str) -> FormulaSpec:
    """Parse a model formula into a :class:`FormulaSpec`.

    Parameters
    ----------
    text : str
        For example ``"R(country, date) ~ 1 + rw(time = week) + lockdown"`` or
        ``"deaths ~ 1"``.

    Returns
    -------
    FormulaSpec

    Raises
    ------
    ValueError
        If there is no ``~``, if an ``R(...)`` left hand side is malformed, if
        the right hand side calls anything other than ``rw()``, or if a
        group-specific term is duplicated (R rejects that too).

    Examples
    --------
    >>> spec = parse_formula("R(country, date) ~ 0 + (1 + lockdown || country)")
    >>> spec.intercept, spec.correlated
    (False, False)
    >>> spec.random[0].terms
    ['1', 'lockdown']
    """
    if not isinstance(text, str):
        raise TypeError(f"formula must be a string, got {type(text).__name__}")

    clean = " ".join(text.split())
    if "~" not in clean:
        raise ValueError(f"{text!r} is not a formula: no '~' found")
    lhs, sep, rhs = clean.partition("~")
    if "~" in rhs:
        raise ValueError(f"{text!r} has more than one '~'")
    lhs, rhs = lhs.strip(), rhs.strip()
    if not lhs:
        raise ValueError(
            f"{text!r} has an empty left hand side; epidemia formulas need a "
            "response, e.g. 'R(country, date) ~ 1' or 'deaths ~ 1'"
        )
    if not rhs:
        raise ValueError(f"{text!r} has an empty right hand side")

    spec = FormulaSpec(response=lhs, text=clean)

    if lhs.startswith("R(") or lhs.startswith("R ("):
        m = _LHS_SUGAR_RE.match(lhs)
        if not m:
            raise ValueError(
                f"left hand side of {clean!r} does not have the required form "
                "R(group, date)"
            )
        spec.response, spec.group, spec.date = "R", m.group(1), m.group(2)
    elif not _IDENT_RE.match(lhs):
        raise ValueError(
            f"left hand side of {clean!r} must be a column name or the "
            "R(group, date) sugar"
        )

    terms = _parse_rhs(rhs, spec)
    spec.fixed = _dedup([t for t in terms if t != "1"])

    # lme4 -- and hence epim() -- refuses duplicated group-specific terms
    # because the two would be indistinguishable in the likelihood.
    seen: set[tuple] = set()
    for r in spec.random:
        key = (tuple(r.terms), r.factor)
        if key in seen:
            raise ValueError(
                f"duplicate group-specific term {r} in {clean!r}"
            )
        seen.add(key)

    return spec


# ---------------------------------------------------------------------------
# glue to core.prepare_panel
# ---------------------------------------------------------------------------


def _expand_factors(df, names, intercept):
    """Dummy-code any non-numeric covariate column, as R's ``model.matrix`` does.

    R turns ``~ 0 + country`` into one indicator per country. ``prepare_panel``
    only reads numeric columns, so do that expansion here on a shallow copy --
    the caller's frame is never touched. Treatment contrasts: the first level is
    dropped when an intercept is present, kept when it is not.
    """
    import pandas as pd

    expanded: list[str] = []
    extra = {}
    for name in names:
        if ":" in name:
            raise ValueError(
                f"interaction term {name!r} is not supported by "
                "build_from_formula; precompute the product as a column"
            )
        if name not in df.columns:
            raise ValueError(
                f"column {name!r} from the formula is not in the data; "
                f"available columns: {sorted(df.columns)}"
            )
        col = df[name]
        if pd.api.types.is_numeric_dtype(col) or pd.api.types.is_bool_dtype(col):
            expanded.append(name)
            continue
        levels = sorted(map(str, pd.unique(col.dropna())))
        if intercept and levels:
            levels = levels[1:]           # treatment contrast: drop the baseline
        as_str = col.astype(str)
        for level in levels:
            new = f"{name}{level}"
            extra[new] = (as_str == level).astype(float)
            expanded.append(new)

    if extra:
        df = df.copy()
        for k, v in extra.items():
            df[k] = v
    return df, expanded


def build_from_formula(df, formula, responses, pop=None, seed_offset=30,
                       threshold=10, fit_until=None, **kw):
    """Build panel arrays and config keywords from a formula.

    The formula-driven counterpart of calling
    :func:`epidemia.core.prepare_panel` with a hand-written list of column
    names: it works out which columns are covariates, which column indexes the
    random walk, and which :class:`~epidemia.core.EpiModelConfig` options the
    formula implies.

    Parameters
    ----------
    df : pandas.DataFrame
        Long panel, one row per group-day.
    formula : str | FormulaSpec
        An ``R(group, date) ~ ...`` formula, or an already parsed spec.
    responses : sequence[str]
        Response columns, one per observation series, as in
        :func:`epidemia.core.prepare_panel`.
    pop : str | None
        Population column, needed for ``EpiModelConfig.pop_adjust``.
    seed_offset, threshold, fit_until
        Passed through to :func:`epidemia.core.prepare_panel`.
    **kw
        Also passed through (``threshold_on``, and ``group``/``date`` when the
        formula has no ``R(group, date)`` sugar to supply them).

    Returns
    -------
    (PanelData, dict, dict)
        The panel, the per-series ``{"y", "mask"}`` mapping, and a dict of
        :class:`~epidemia.core.EpiModelConfig` keyword arguments:
        ``intercept``, ``correlated``, and ``rw`` when the formula has an
        ``rw()`` term.

    Notes
    -----
    The model in :mod:`epidemia.core` shares one design matrix between the
    population-level effects and the group-level ones, so a covariate that
    appears only inside ``(... | group)`` still gets a population-level
    coefficient, and vice versa. The formula's *pooling* structure (``|`` vs
    ``||``, intercept or not) is honoured exactly; the split between which
    covariates are fixed and which vary is not.

    Examples
    --------
    >>> from epidemia import europe_covid2                     # doctest: +SKIP
    >>> panel, series, cfg = build_from_formula(               # doctest: +SKIP
    ...     europe_covid2().data,
    ...     "R(country, date) ~ 0 + (1 + lockdown || country) + lockdown",
    ...     responses=["deaths"], pop="pop")
    """
    from .core import RandomWalk, prepare_panel

    spec = formula if isinstance(formula, FormulaSpec) else parse_formula(formula)

    group = kw.pop("group", None) or spec.group
    date = kw.pop("date", None) or spec.date
    if group is None or date is None:
        raise ValueError(
            f"{spec} does not carry the R(group, date) sugar; pass group= and "
            "date= explicitly"
        )

    if len(spec.rw) > 1:
        raise ValueError(
            "epidemia.core supports at most one rw() term per model, "
            f"{spec} has {len(spec.rw)}"
        )
    rw_term = spec.rw[0] if spec.rw else None
    # R defaults rw(time=) to the date column implied by the formula, giving a
    # daily walk (see get_autocor_time in R/autocor.R).
    rw_by = (rw_term.time or date) if rw_term is not None else None

    df, npis = _expand_factors(df, spec.covariates, spec.intercept)

    panel, series = prepare_panel(
        df, npis=npis, responses=list(responses), group=group, date=date,
        pop=pop, seed_offset=seed_offset, threshold=threshold,
        fit_until=fit_until, rw_by=rw_by, **kw,
    )

    # A formula with no bar term has no group-level effects at all -- R builds
    # no Z matrix for it (R/standata_reg.R:112). EpiModelConfig.region_effects
    # defaults True, so leaving it unset gave a fully pooled formula
    # per-region intercepts AND slopes it never asked for.
    config_kwargs = {
        "intercept": spec.intercept,
        "correlated": spec.correlated,
        "region_effects": bool(spec.random),
    }
    if rw_term is not None:
        if rw_term.gr is not None and rw_term.gr != group:
            raise ValueError(
                f"rw(gr = {rw_term.gr}) does not match the modelled group "
                f"{group!r}; epidemia.core walks are per modelled group"
            )
        config_kwargs["rw"] = RandomWalk(
            index=panel.rw_index,
            by_region=rw_term.gr is not None,
            prior_scale=rw_term.prior_scale,
        )
    return panel, series, config_kwargs
