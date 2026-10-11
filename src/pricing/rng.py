"""Common random numbers for paired valuations.

Why this module exists
----------------------
The method (paper: Pricing Template) requires that the gold-standard parameter set and the
extracted parameter set be priced with *the same* random numbers, so that an
identical parameter set yields exactly zero price deviation and every measured
deviation is attributable to extraction error.

Two properties are needed, and neither comes for free:

**P1 - reproducibility.**  The same configuration must reproduce the same draw.

**P2 - shape stability.**  A mis-extracted number of observation dates is a
common extraction error.  A naive
``rng.standard_normal((n_paths, n_obs, n_assets))`` re-shapes the request and
shifts the whole draw sequence, so the two valuations would differ by simulation
noise on top of the genuine parameter effect.  Instead the draw is made once at
a *fixed* ``(n_paths, max_steps, max_assets)`` shape and each valuation slices
the leading block it needs.  The variate feeding asset ``j`` at step ``i`` is
then the same number for both parameter sets.

``max_steps`` and ``max_assets`` are therefore part of the paired-valuation
configuration, not of either parameter set: the caller sets them to cover both
(see ``src.eval.deviation``).

Generators
----------
``"sobol"`` (default): scrambled Sobol points mapped through the inverse normal
CDF.  Measured variance reduction against plain Monte Carlo on this payoff class
is 3x-16x on the gold-labelled notes (``experiments/08_convergence.py``), at under twice the run time.

``"iid"``: pseudo-random normals, optionally with antithetic variates.  Retained
for the convergence and variance-reduction comparison in ``experiments/08_convergence.py``.

A note on antithetic variates
----------------------------
The original plan specified antithetic variates.  On the gold-labelled notes
(``experiments/08_convergence.py``, 20 seeds at 2^17 paths) the variance ratio
``Var_plain / Var_antithetic`` is 0.85 (single underlier), 1.32 and 1.61
(at-the-money worst-of baskets) - all inside the ~[0.4, 2.5] sampling band of a
20-seed variance ratio, i.e. no reliable gain - and 3.7 only when spot sits
just above the barriers.  Antithetic sampling needs the payoff to be close to
monotone in the Brownian driver; early termination and the coupon barrier break
that.  Scrambled Sobol gives 3.2x-15.8x on the same notes.  The flag is kept so
that the paper can report this measurement.

Caveat to state in the paper
----------------------------
The slice is stable, but correlated increments are formed as ``Z @ chol(corr).T``,
so changing the *set* of underlyings changes the Cholesky factor and therefore
the correlated draws.  Coupling is exact for the case that matters - same
underlyings, differing dates, barriers or coupon rules - and approximate when
the extraction error changes the basket itself.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterator, Literal

import numpy as np
from scipy.stats import norm, qmc

Generator = Literal["sobol", "iid"]

#: Default per-filing dimension bounds.  A quarterly note observed for 16 years,
#: or a monthly note observed for 5 years, fits in 64 steps; baskets in this
#: product class rarely exceed four underlyings.  Sobol dimension is
#: ``max_steps * max_assets``, so these are deliberately not generous.
DEFAULT_MAX_STEPS: int = 64
DEFAULT_MAX_ASSETS: int = 4

MAX_STEPS = DEFAULT_MAX_STEPS
MAX_ASSETS = DEFAULT_MAX_ASSETS


def _is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


#: Blocks are cached on their full configuration.  Paired valuations of one filing
#: share that configuration by construction, so the (not cheap) draw happens once
#: per filing rather than once per valuation - with a few dozen perturbation
#: pricings per filing, that is the difference between seconds and minutes.
@lru_cache(maxsize=4)
def _make_block(
    seed: int,
    n_paths: int,
    generator: str,
    antithetic: bool,
    max_steps: int,
    max_assets: int,
) -> np.ndarray:
    shape = (n_paths, max_steps, max_assets)
    if generator == "iid":
        rng = np.random.default_rng(seed)
        if antithetic:
            half = rng.standard_normal((n_paths // 2, *shape[1:]))
            block = np.concatenate([half, -half], axis=0)
        else:
            block = rng.standard_normal(shape)
    else:
        with warnings.catch_warnings():
            # The power-of-two warning is already raised in __post_init__.
            warnings.simplefilter("ignore", UserWarning)
            points = qmc.Sobol(
                d=max_steps * max_assets, scramble=True, seed=seed
            ).random(n_paths)
        # Guard the inverse CDF against the exact 0 and 1 a scrambled point set
        # can produce.
        np.clip(points, 1e-12, 1.0 - 1e-12, out=points)
        block = norm.ppf(points).reshape(shape)

    block.flags.writeable = False
    return block


def _iter_raw_blocks(
    seed: int,
    n_paths: int,
    generator: str,
    antithetic: bool,
    max_steps: int,
    max_assets: int,
    chunk_paths: int,
) -> Iterator[np.ndarray]:
    """The rows of ``_make_block`` in consecutive chunks, never all at once.

    Both generators continue their stream across ``random`` / ``standard_normal``
    calls, so concatenating the chunks reproduces ``_make_block`` exactly (Sobol
    rows come from one scrambled sequence, not one scramble per chunk).  With
    antithetic sampling each chunk is ``[h, -h]`` for the next ``chunk_paths // 2``
    rows of the half-block; the pair structure is kept within the chunk so that
    pair averages can be formed there.
    """
    inner = (max_steps, max_assets)
    if generator == "iid":
        rng = np.random.default_rng(seed)
        total = n_paths // 2 if antithetic else n_paths
        step = chunk_paths // 2 if antithetic else chunk_paths
        done = 0
        while done < total:
            m = min(step, total - done)
            z = rng.standard_normal((m, *inner))
            yield np.concatenate([z, -z], axis=0) if antithetic else z
            done += m
    else:
        engine = qmc.Sobol(d=max_steps * max_assets, scramble=True, seed=seed)
        done = 0
        while done < n_paths:
            m = min(chunk_paths, n_paths - done)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                points = engine.random(m)
            np.clip(points, 1e-12, 1.0 - 1e-12, out=points)
            yield norm.ppf(points).reshape(m, *inner)
            done += m


@dataclass(frozen=True)
class CommonRandomNumbers:
    """A fixed-shape block of standard normal variates shared by paired pricings.

    Parameters
    ----------
    seed:
        Seed of the pseudo-random generator, or of the Sobol scramble.
    n_paths:
        Total number of paths.  Sobol point sets are balanced only at powers of
        two, so a non-power-of-two ``n_paths`` warns when ``generator="sobol"``.
    generator:
        ``"sobol"`` for scrambled quasi-Monte Carlo, ``"iid"`` for pseudo-random.
    antithetic:
        Antithetic variates; only valid with ``generator="iid"``.
    max_steps, max_assets:
        Fixed draw shape shared by the paired valuations.  Must cover the
        observation-date count and basket size of *both* parameter sets.
    """

    seed: int
    n_paths: int
    generator: Generator = "sobol"
    antithetic: bool = False
    max_steps: int = DEFAULT_MAX_STEPS
    max_assets: int = DEFAULT_MAX_ASSETS

    def __post_init__(self) -> None:
        if self.n_paths <= 0:
            raise ValueError("n_paths must be positive")
        if self.generator not in ("sobol", "iid"):
            raise ValueError(f"unknown generator {self.generator!r}")
        if self.antithetic:
            if self.generator != "iid":
                raise ValueError('antithetic variates require generator="iid"')
            if self.n_paths % 2 != 0:
                raise ValueError("n_paths must be even when antithetic=True")
        if self.max_steps <= 0 or self.max_assets <= 0:
            raise ValueError("max_steps and max_assets must be positive")
        if self.generator == "sobol" and not _is_power_of_two(self.n_paths):
            warnings.warn(
                f"n_paths={self.n_paths} is not a power of two; Sobol point sets "
                "are balanced only at powers of two, which costs accuracy. "
                f"Consider {1 << (self.n_paths - 1).bit_length()}.",
                RuntimeWarning,
                stacklevel=2,
            )

    @property
    def dimension(self) -> int:
        """Sobol dimension implied by the fixed draw shape."""
        return self.max_steps * self.max_assets

    def block(self) -> np.ndarray:
        """The ``(n_paths, max_steps, max_assets)`` normal block.

        Read-only, and shared between every call with the same configuration.
        """
        return _make_block(
            self.seed,
            self.n_paths,
            self.generator,
            self.antithetic,
            self.max_steps,
            self.max_assets,
        )

    def draw(self, n_steps: int, n_assets: int) -> np.ndarray:
        """Leading ``(n_paths, n_steps, n_assets)`` slice of the fixed block."""
        self._check_shape(n_steps, n_assets)
        return self.block()[:, :n_steps, :n_assets]

    def _check_shape(self, n_steps: int, n_assets: int) -> None:
        if n_steps > self.max_steps:
            raise ValueError(
                f"{n_steps} observation dates exceeds max_steps={self.max_steps}; "
                "raise max_steps for this filing"
            )
        if n_assets > self.max_assets:
            raise ValueError(
                f"{n_assets} underlyings exceeds max_assets={self.max_assets}; "
                "raise max_assets for this filing"
            )

    def iter_draws(
        self, n_steps: int, n_assets: int, chunk_paths: int
    ) -> Iterator[np.ndarray]:
        """``draw`` in consecutive row chunks of at most ``chunk_paths`` paths.

        For path counts whose full block does not fit in memory.  The chunks
        concatenate to exactly ``draw(n_steps, n_assets)`` - except with
        antithetic sampling, where each chunk is ``[h, -h]`` (same multiset of
        rows, pairs kept together).  Nothing is cached.
        """
        self._check_shape(n_steps, n_assets)
        if chunk_paths <= 0:
            raise ValueError("chunk_paths must be positive")
        if self.antithetic and chunk_paths % 2:
            raise ValueError("chunk_paths must be even when antithetic=True")
        for z in _iter_raw_blocks(
            self.seed,
            self.n_paths,
            self.generator,
            self.antithetic,
            self.max_steps,
            self.max_assets,
            chunk_paths,
        ):
            yield z[:, :n_steps, :n_assets]
