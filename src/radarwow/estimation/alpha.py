"""Drop-size-adaptive alpha for R(A): Wang et al. 2019, J. Hydrometeor. 20, 985-997, doi:10.1175/JHM-D-18-0071.1
(MRMS / NEXRAD Build 19), with the formula as specified by Grams et al. 2019 (NWS ROC, AMS Annual Meeting, no DOI).

Specific attenuation from ZPHI scales with alpha = A/KDP, which depends on the drop size distribution: rain with
many small drops (tropical, atmospheric rivers) has a larger alpha than continental rain with big drops. Wang et
al. estimate it from how fast ZDR rises with Z: small drops keep ZDR low as Z grows. With K the slope of the
median ZDR versus Z between 20 and 50 dBZ (dB per dB),
    alpha = 0.04875 - 0.75 K,   bounded to [0.015 (continental), 0.04 (tropical)].
K is a slope, so a constant ZDR calibration offset does not change it.

radar-wow choices: medians in 1 dB Z bins with at least ``min_count`` gates, a straight-line fit through at least
``min_bins`` bins; gates pooled over the last ``window_min`` minutes; only rain gates with rhoHV > ``rhohv_min`` and
whatever mask the caller passes (below the melting layer, unblocked). With an adaptive alpha we use R(A) on rays
with at least 10 deg of PHIDP span instead of the S-band default of 40 deg: on KDAX 24 Oct 2021 the 40 deg rule
left R(A) almost unused, and on KTLX 5 May 2022 the change made no difference (radarwow-verify notebook 05).
"""

from __future__ import annotations

from collections import deque

import numpy as np
import pandas as pd

from radarwow.core.noise import signal_mask

ALPHA_MIN, ALPHA_MAX = 0.015, 0.04
ADAPTIVE_MIN_DPHI = 10.0


def zdr_slope(z, zdr, z_range=(20.0, 50.0), min_count: int = 50, min_bins: int = 6):
    """Slope K (dB/dB) of median ZDR vs Z in 1 dB bins; NaN if too few populated bins."""
    z, zdr = np.asarray(z, float).ravel(), np.asarray(zdr, float).ravel()
    ok = np.isfinite(z) & np.isfinite(zdr)
    z, zdr = z[ok], zdr[ok]
    edges = np.arange(z_range[0], z_range[1] + 1e-9, 1.0)
    idx = np.digitize(z, edges) - 1
    centres, meds = [], []
    for i in range(edges.size - 1):
        sel = idx == i
        if sel.sum() >= min_count:
            centres.append(edges[i] + 0.5)
            meds.append(np.median(zdr[sel]))
    if len(centres) < min_bins:
        return np.nan
    return float(np.polyfit(centres, meds, 1)[0])


def alpha_from_slope(k, lo: float = ALPHA_MIN, hi: float = ALPHA_MAX, default: float = ALPHA_MIN):
    """alpha = 0.04875 - 0.75 K, bounded; ``default`` when K is NaN."""
    if not np.isfinite(k):
        return default
    return float(np.clip(0.04875 - 0.75 * k, lo, hi))


class ZDRSlopeAlpha:
    """Rolling estimate of alpha from the volumes of the last ``window_min`` minutes.

    ``update(time, ds, mask)`` adds the rain gates of one sweep (``ds`` with DBZH, ZDR, RHOHV) and returns
    (alpha, K) from everything inside the window.
    """

    def __init__(self, window_min: float = 60.0, rhohv_min: float = 0.98, default: float = ALPHA_MIN):
        self.window_min, self.rhohv_min, self.default = window_min, rhohv_min, default
        self.buf = deque()

    def update(self, time, ds, mask=None):
        t = pd.Timestamp(time)
        ok = signal_mask(ds) & (ds["RHOHV"].values > self.rhohv_min)
        if mask is not None:
            ok &= np.asarray(mask, bool)
        self.buf.append((t, ds["DBZH"].values[ok].astype("float32"), ds["ZDR"].values[ok].astype("float32")))
        while self.buf and (t - self.buf[0][0]).total_seconds() / 60 > self.window_min:
            self.buf.popleft()
        k = zdr_slope(np.concatenate([b[1] for b in self.buf]), np.concatenate([b[2] for b in self.buf]))
        return alpha_from_slope(k, default=self.default), k
