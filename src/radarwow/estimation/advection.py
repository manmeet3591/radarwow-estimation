"""Advection-corrected rain accumulation: Anagnostou & Krajewski 1999, J. Atmos. Oceanic Technol. 16, 189-197,
doi:10.1175/1520-0426(1999)016<0189:RTRREP>2.0.CO;2, the scheme of the pysteps example ``advection_correction``.

Holding each scan's rain rate until the next scan (`accumulate`) leaves stripes where cells move several km
between scans. Instead, for each pair of consecutive scans R0 (time t0) and R1 (t1 = t0 + T):

1. Estimate the motion field V (pixels per T) from the pair with pysteps' Lucas-Kanade optical flow on
   log-transformed rain (Pulkkinen et al. 2019, GMD, doi:10.5194/gmd-12-4185-2019).
2. At sub-steps t0 + f T (f = k dt / T), interpolate along the motion: R(f) = (1 - f) R0 moved forward by f V
   + f R1 moved back by (1 - f) V (semi-Lagrangian, bilinear).
3. Add R(f) dt for every sub-step inside the accumulation window.

Optical flow needs a regular grid, so rates are first mapped from polar to a Cartesian grid (`polar_to_cartesian`).
radar-wow choices: motion from each pair alone (no temporal smoothing of V); pairs further apart than
``max_gap_min`` fall back to holding R0 for ``max_gap_min``, as in `accumulate`; missing rain (NaN) is 0 for the
motion and the interpolation, and a cell counts as valid in a sub-step only if both scans saw it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage import map_coordinates

from .vpr import ground_km


def polar_to_cartesian(da: xr.DataArray, elevation_deg: float, max_km: float = 150.0, res_km: float = 1.0) -> xr.DataArray:
    """Nearest-gate map of a polar (azimuth x range) field onto an x/y grid in km (x east, y north)."""
    x = np.arange(-max_km + res_km / 2, max_km, res_km)
    xx, yy = np.meshgrid(x, x[::-1])
    s = np.hypot(xx, yy)
    az = np.rad2deg(np.arctan2(xx, yy)) % 360.0
    azs = da["azimuth"].values % 360.0
    order = np.argsort(azs)
    a_s = azs[order]
    i = np.searchsorted(a_s, az) % a_s.size
    im = (i - 1) % a_s.size
    d_i = np.abs((az - a_s[i] + 180) % 360 - 180)
    d_m = np.abs((az - a_s[im] + 180) % 360 - 180)
    ia = order[np.where(d_m < d_i, im, i)]
    g = ground_km(da["range"].values, elevation_deg)
    j = np.clip(np.searchsorted(g, s), 0, g.size - 1)
    jm = np.clip(j - 1, 0, g.size - 1)
    j = np.where(np.abs(g[jm] - s) < np.abs(g[j] - s), jm, j)
    out = da.values[ia, j].astype(float)
    out = np.where((s <= min(max_km, g[-1])) & (np.minimum(d_i, d_m) <= 1.0), out, np.nan)
    return xr.DataArray(out.astype("float32"), dims=("y", "x"), coords={"y": x[::-1], "x": x},
                        attrs={"units": da.attrs.get("units", ""), "res_km": res_km})


def _log_rain(r, threshold=0.1, zero=-15.0):
    r = np.nan_to_num(r, nan=0.0)
    return np.where(r >= threshold, 10 * np.log10(np.maximum(r, threshold)), zero)


def motion_field(r0, r1, **lk_kwargs):
    """(2, ny, nx) motion (x, y components, pixels per scan interval, in array index order: +x right, +y down)."""
    from pysteps import motion

    lk = motion.get_method("lucaskanade")
    stack = np.stack([_log_rain(r0), _log_rain(r1)])
    kw = {"fd_kwargs": {"buffer_mask": 10}, "verbose": False, **lk_kwargs}
    return np.asarray(lk(stack, **kw))


def _shift(field, vx, vy, frac):
    """Value at each cell of ``field`` moved by frac*V (semi-Lagrangian backward lookup, bilinear)."""
    ny, nx = field.shape
    yy, xx = np.mgrid[0:ny, 0:nx].astype(float)
    return map_coordinates(field, [yy - frac * vy, xx - frac * vx], order=1, mode="constant", cval=0.0)


def advection_accumulate(scans, start, end, dt_min: float = 1.0, max_gap_min: float = 15.0, return_motion: bool = False):
    """Accumulated rain (mm) in [start, end) from ``scans`` = list of (time, rate DataArray mm/h on one y/x grid).

    Returns a Dataset with ACCUM (mm), VALID_MINUTES and, with ``return_motion``, the mean motion (km per hour).
    """
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    scans = sorted(((pd.Timestamp(t), r) for t, r in scans), key=lambda s: s[0])
    ref = scans[0][1]
    shape = ref.shape
    acc, valid = np.zeros(shape), np.zeros(shape)
    vsum, vn = np.zeros((2,) + shape), 0
    covered = 0.0
    for k, (t0, r0da) in enumerate(scans):
        r0 = r0da.values.astype(float)
        nxt = scans[k + 1] if k + 1 < len(scans) else None
        gap = (nxt[0] - t0).total_seconds() / 60 if nxt is not None else np.inf
        if nxt is None or gap > max_gap_min:  # hold, as in `accumulate`
            a, b = max(t0, start), min(t0 + pd.Timedelta(minutes=max_gap_min), end)
            if b > a:
                h = (b - a).total_seconds() / 3600
                acc += np.nan_to_num(r0) * h
                valid += np.isfinite(r0) * h * 60
                covered += h
            continue
        t1, r1 = nxt[0], nxt[1].values.astype(float)
        if t1 <= start or t0 >= end:
            continue
        v = motion_field(r0, r1)
        vsum += v * (60.0 / gap)
        vn += 1
        z0, z1 = np.nan_to_num(r0), np.nan_to_num(r1)
        both = np.isfinite(r0) & np.isfinite(r1)
        n = max(1, int(round(gap / dt_min)))
        step = gap / n
        for i in range(n):
            tm = t0 + pd.Timedelta(minutes=(i + 0.5) * step)  # sub-step centre
            if not (start <= tm < end):
                continue
            f = (i + 0.5) / n
            r = (1 - f) * _shift(z0, v[0], v[1], f) + f * _shift(z1, v[0], v[1], f - 1)
            h = step / 60
            acc += r * h
            valid += both * step
            covered += h
    window_h = (end - start).total_seconds() / 3600
    data = {"ACCUM": (("y", "x"), acc.astype("float32"), {"units": "mm"}),
            "VALID_MINUTES": (("y", "x"), valid.astype("float32"))}
    if return_motion and vn:
        data["U"] = (("y", "x"), (vsum[0] / vn * ref.attrs.get("res_km", 1.0)).astype("float32"), {"units": "km/h", "long_name": "eastward motion"})
        data["V"] = (("y", "x"), (-vsum[1] / vn * ref.attrs.get("res_km", 1.0)).astype("float32"), {"units": "km/h", "long_name": "northward motion"})
    return xr.Dataset(data, coords={k: ref[k].values for k in ref.dims},
                      attrs={"start": str(start), "end": str(end), "coverage": covered / window_h, "n_scans": len(scans),
                             "method": "advection-corrected (Anagnostou & Krajewski 1999; pysteps Lucas-Kanade)"})
