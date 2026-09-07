"""Choosing the collapse window W from data rather than from geometry.

The paper's negative result is that field-of-view depth divided by permitted
speed does not give the deployed 120 s window and is not the right quantity
anyway.  Refuting a derivation is only half a contribution, so this module
supplies the replacement: four estimators of W that a deployment can run on its
own trace, and the two-sided error curve they are chosen against.

The quantity every estimator works from is the inter-observation time (IOT):
the gap between consecutive observations that share an entity key.  Its
distribution is a mixture of two populations --- gaps inside one operational
event, which are short, and gaps between separate events involving the same
entity, which are long --- and the window's job is to separate them.

Estimators:

  mixture_crossover   two-component log-normal mixture fitted by EM; W is the
                      posterior crossover.  The most principled where the two
                      populations separate.
  kneedle             maximum perpendicular distance from the chord of the
                      (suppression, masking) curve.  Parameter-free.
  cost_weighted       argmin of lambda_r * FP(W) + lambda_m * FN(W).  Makes the
                      value judgement explicit and lets another deployment
                      substitute its own costs.
  spread_rule         max within-event gap plus two standard deviations, the
                      simple engineering rule.  Included because it is what a
                      practitioner reaches for first, and because on this data
                      it is wrong in an instructive direction.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Sequence

import numpy as np


def inter_observation_times(events) -> dict[str, np.ndarray]:
    """Gaps between consecutive observations sharing an entity key.

    Returns the pooled gaps and, using the ground-truth incident label, the two
    populations separately.  A deployment does not have the labels; it has only
    the pooled column, which is exactly why the mixture fit is needed.
    """
    by_entity: dict[str, list] = defaultdict(list)
    for e in events:
        by_entity[e.entity_id].append((e.ts, e.truth_incident))
    pooled, within, between = [], [], []
    for seq in by_entity.values():
        seq.sort()
        for (t0, i0), (t1, i1) in zip(seq, seq[1:]):
            gap = t1 - t0
            if gap <= 0:
                continue
            pooled.append(gap)
            (within if i0 == i1 else between).append(gap)
    return {
        "pooled": np.asarray(pooled, dtype=float),
        "within_event": np.asarray(within, dtype=float),
        "between_event": np.asarray(between, dtype=float),
    }


def fit_lognormal_mixture(gaps: np.ndarray, iters: int = 200, seed: int = 0) -> dict:
    """Two-component log-normal mixture by EM on log gaps."""
    x = np.log(np.asarray(gaps, dtype=float)[gaps > 0])
    if x.size < 20:
        raise ValueError("not enough gaps to fit a mixture")
    g = np.random.default_rng(seed)
    lo, hi = np.percentile(x, [25, 75])
    mu = np.array([lo, hi], dtype=float)
    sd = np.array([x.std() / 2 or 1.0] * 2, dtype=float)
    w = np.array([0.5, 0.5])
    for _ in range(iters):
        dens = np.stack([
            w[k] * np.exp(-0.5 * ((x - mu[k]) / sd[k]) ** 2) / (sd[k] * math.sqrt(2 * math.pi))
            for k in range(2)
        ])
        tot = dens.sum(axis=0)
        tot[tot == 0] = 1e-300
        r = dens / tot
        nk = r.sum(axis=1)
        w = nk / x.size
        mu = (r * x).sum(axis=1) / np.maximum(nk, 1e-12)
        sd = np.sqrt((r * (x - mu[:, None]) ** 2).sum(axis=1) / np.maximum(nk, 1e-12))
        sd = np.maximum(sd, 1e-3)
    order = np.argsort(mu)
    mu, sd, w = mu[order], sd[order], w[order]

    # Crossover: the gap at which the long component becomes the more probable
    # explanation.  Searched between the two component medians.  Without that
    # constraint the search degenerates whenever the long component is much
    # broader than the short one, because a broad Gaussian also dominates in
    # the far left tail, and the unconstrained argmax returns a crossover near
    # zero.  Both values are returned so the degeneracy is visible rather than
    # hidden by the constraint.
    lg = np.linspace(x.min(), x.max(), 4000)
    grid = np.exp(lg)
    p_short = w[0] * np.exp(-0.5 * ((lg - mu[0]) / sd[0]) ** 2) / sd[0]
    p_long = w[1] * np.exp(-0.5 * ((lg - mu[1]) / sd[1]) ** 2) / sd[1]
    dominant = p_long > p_short
    unconstrained = float(grid[int(np.argmax(dominant))]) if dominant.any() else float(grid[-1])
    band = (lg >= mu[0]) & (lg <= mu[1]) & dominant
    crossover = float(grid[int(np.argmax(band))]) if band.any() else None

    # A well-separated mixture has a long component no broader than a few times
    # the short one.  When it does not, the fit has absorbed the tail of the
    # short population instead of isolating the long one, and the crossover
    # should not be trusted.
    separated = bool(sd[1] < 3.0 * sd[0] and math.exp(mu[1]) > 5.0 * math.exp(mu[0]))
    return {
        "weights": w.tolist(),
        "median_short_s": float(np.exp(mu[0])),
        "median_long_s": float(np.exp(mu[1])),
        "sigma_log": sd.tolist(),
        "crossover_s": crossover,  # None when the components are not separated
        "crossover_unconstrained_s": unconstrained,
        "components_separated": separated,
    }


def kneedle(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Point of maximum perpendicular distance from the chord (Satopaa 2011)."""
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    xn = (x - x.min()) / (x.max() - x.min() or 1.0)
    yn = (y - y.min()) / (y.max() - y.min() or 1.0)
    d = yn - xn
    return float(x[int(np.argmax(d))])


def cost_weighted_window(rows, lambda_redundant: float, lambda_masked: float) -> dict:
    """argmin over the swept windows of a weighted two-sided error."""
    best = None
    for r in rows:
        cost = lambda_redundant * r["redundant_rate"] + lambda_masked * r["masked_rate"]
        if best is None or cost < best["cost"]:
            best = {"window_s": r["window_s"], "cost": cost}
    return best


def spread_rule(iot: dict) -> dict:
    """max(within-event gap) + 2 sd(within-event gap)."""
    w = iot["within_event"]
    return {
        "max_within_event_s": float(w.max()),
        "sd_within_event_s": float(w.std()),
        "window_s": float(w.max() + 2 * w.std()),
    }


def debar_wespi_window(iot: dict, quantile: float = 99.0) -> dict:
    """Twice a high quantile of within-event gaps.

    Debar and Wespi aggregate alerts that share a key inside a window set from
    the observed correlation span.  We operationalise "observed span" as the
    99th percentile of within-event gaps rather than the maximum, so that one
    outlying pair cannot set the window for the whole estate.
    """
    q = float(np.percentile(iot["within_event"], quantile))
    return {"within_event_q99_s": q, "window_s": 2.0 * q}
