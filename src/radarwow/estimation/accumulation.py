"""Rain accumulation from a sequence of rain-rate sweeps.

Each scan's rate is held from its start time until the next scan starts (at most ``max_gap_min``),
and the part of that interval inside the accumulation window is added. This is the simplest scheme
("Eulerian, no advection"): fast-moving cells leave stripes when scans are minutes apart. Advection-
corrected accumulation is in `advection.py`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr


def to_fixed_azimuths(da: xr.DataArray, n_az: int = 720) -> xr.DataArray:
    """Nearest-ray regrid onto ``n_az`` equal azimuth bins so scans can be added gate by gate."""
    centres = (np.arange(n_az) + 0.5) * 360.0 / n_az
    az = da["azimuth"].values % 360.0
    order = np.argsort(az)
    az_s = az[order]
    idx = np.searchsorted(az_s, centres)
    lo = np.clip(idx - 1, 0, az_s.size - 1)
    hi = np.clip(idx, 0, az_s.size - 1)
    d_lo = np.abs((centres - az_s[lo] + 180) % 360 - 180)
    d_hi = np.abs((centres - az_s[hi] + 180) % 360 - 180)
    pick = np.where(d_lo <= d_hi, lo, hi)
    ok = np.minimum(d_lo, d_hi) <= 1.5 * 360.0 / n_az
    vals = da.values[order][pick]
    vals = np.where(ok[:, None], vals, np.nan)
    return xr.DataArray(vals, dims=("azimuth", "range"), coords={"azimuth": centres, "range": da["range"].values})


def accumulate(scans, start, end, max_gap_min: float = 15.0):
    """Accumulated rain (mm) in [start, end) from ``scans`` = list of (time, rate DataArray in mm/h).

    All rate arrays must share the same azimuth and range grid (use `to_fixed_azimuths`). Returns
    (accumulation, coverage) where coverage is the fraction of the window covered by scans; gates
    with NaN rate in a scan contribute nothing and are tracked in ``valid_minutes``.
    """
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    scans = sorted(scans, key=lambda s: pd.Timestamp(s[0]))
    times = [pd.Timestamp(t) for t, _ in scans]
    acc = None
    valid_min = None
    covered = 0.0
    for i, (t, rate) in enumerate(scans):
        t_next = times[i + 1] if i + 1 < len(times) else t + pd.Timedelta(minutes=max_gap_min)
        t_next = min(t_next, t + pd.Timedelta(minutes=max_gap_min))
        a, b = max(t, start), min(t_next, end)
        if b <= a:
            continue
        dt_h = (b - a).total_seconds() / 3600.0
        r = rate.values
        if acc is None:
            acc = np.zeros(r.shape)
            valid_min = np.zeros(r.shape)
        acc += np.nan_to_num(r) * dt_h
        valid_min += np.isfinite(r) * dt_h * 60
        covered += dt_h
    if acc is None:
        raise ValueError("no scans inside the window")
    window_h = (end - start).total_seconds() / 3600.0
    ref = scans[0][1]
    dims = ref.dims  # (azimuth, range) for polar scans, (y, x) for Cartesian grids
    out = xr.Dataset({"ACCUM": (dims, acc.astype("float32"), {"units": "mm"}),
                      "VALID_MINUTES": (dims, valid_min.astype("float32"))},
                     coords={d: ref[d].values for d in dims},
                     attrs={"start": str(start), "end": str(end), "coverage": covered / window_h, "n_scans": len(scans)})
    return out
