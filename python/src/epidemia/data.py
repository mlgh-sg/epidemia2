"""Example datasets.

``flu1918`` is the 1918 influenza pandemic in Baltimore, the same data used in
the R package's basic tutorial (originally from the EpiEstim package).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Daily case counts, Baltimore 1918 influenza pandemic (EpiEstim::Flu1918$incidence)
_FLU1918_INCIDENCE = np.array([
    5, 1, 6, 15, 2, 3, 8, 7, 2, 15, 4, 17, 4, 10, 31, 11, 13, 36, 13, 33, 17,
    15, 32, 27, 70, 58, 32, 69, 54, 80, 405, 192, 243, 204, 280, 229, 304, 265,
    196, 372, 158, 222, 141, 172, 553, 148, 95, 144, 85, 143, 87, 73, 70, 62,
    116, 44, 38, 60, 45, 60, 27, 51, 34, 22, 16, 11, 18, 11, 10, 8, 13, 3, 3, 6,
    6, 13, 5, 6, 6, 5, 5, 1, 2, 2, 3, 8, 4, 1, 2, 3, 1, 0,
], dtype=float)

# Serial-interval PMF (EpiEstim::Flu1918$si_distr): si[k] = P(serial interval = k days).
# si[0] == 0 (no same-day generation), so the renewal generation kernel drops it.
_FLU1918_SI = np.array([
    0.0, 0.233, 0.359, 0.198, 0.103, 0.053, 0.027, 0.014, 0.007, 0.003, 0.002,
    0.001,
], dtype=float)


@dataclass
class EpiData:
    """A simple container for an example epidemic series."""

    incidence: np.ndarray      # daily observed counts
    serial_interval: np.ndarray  # full SI PMF, si[k] = P(SI = k days)

    @property
    def generation(self) -> np.ndarray:
        """Renewal generation kernel: ``generation[k] = P(SI = k+1 days)``.

        This drops the (zero) same-day entry of the serial interval, so that
        ``generation`` aligns with :func:`epidemia.renewal.renewal_infections`,
        which weights the infection ``k+1`` days in the past by ``generation[k]``.
        """
        return self.serial_interval[1:]


def flu1918() -> EpiData:
    """Return the 1918 Baltimore influenza data (92 days) with its serial interval."""
    return EpiData(
        incidence=_FLU1918_INCIDENCE.copy(),
        serial_interval=_FLU1918_SI.copy(),
    )
