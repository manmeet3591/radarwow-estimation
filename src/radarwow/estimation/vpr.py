"""Brightband VPR correction: Zhang & Qi 2010, J. Hydrometeor. 11, 1157-1171, doi:10.1175/2010JHM1201.1
(the NMQ/MRMS apparent-VPR method). Steps as summarised in radar_wow/literature/reading_notes/14_zhangqi2010_vpr.md:

1. Convective/stratiform split by VIL: LW = 3.44e-3 Z^(4/7) g m-3, times the beam depth of each tilt, summed
   over tilts; VIL > 6.5 kg m-2 is convective.
2. Brightband-affected area (BBA): stratiform columns with composite Z >= 30 dBZ, gates between the apparent BB
   top h_t = model 0 C height + half the lowest beam's depth at the freezing level, and h_t - 5 km.
3. Apparent VPR (AVPR) per tilt: azimuthal mean of Z in the BBA at each range. QC: minimum gate count per range;
   points more than 2x the mean absolute deviation from a 15-point running mean are bad; > 40 % bad -> no
   correction for that tilt.
4. Five-parameter linear model: BB top h_t, peak h_p (the maximum below the top), bottom h_b (the minimum below
   the peak, with Z floored at 28 dBZ), slope alpha above the peak (< 0) and beta below it (> 0), least squares.
   The true VPR is flat below h_b. If the bottom is not seen (BB near the ground), a symmetric model:
   alpha = -beta, h_b = 2 h_p - h_t.
5. dBZ_corrected = dBZ_obs - dBz_a(h) in the BBA for h_b <= h <= h_t, with
   dBz_a(h) = alpha (h - h_p) + beta (h_p - h_b) above the peak and beta (h - h_b) at or below it.
   One AVPR per tilt, so beam broadening is handled implicitly.

radar-wow choices where the notes do not give a value (all parameters, documented in ``params``):
AVPR in 1 km range bins (``bin_km``), averaged in dBZ; at least ``min_count`` gates per bin; the running mean
and MAD test on those bins; the peak and bottom located on the running-mean profile, the bottom being where the descent below the peak
flattens to within ``flat_db`` of its minimum; no correction if the modelled peak excess is < 1 dB; the symmetric model only when the maximum lies within
``max_symmetric_depth_km`` of h_t (otherwise a profile with no bright band, maximum at the lowest height, would
qualify); the peak searched only between ``peak_below_fz_km`` below and ``peak_above_fz_km`` above the 0 C
height (on KTLX 5 May 2022 the unconstrained maximum of the lowest tilt's AVPR wandered from 2.4 to 4.3 km, i.e.
up to 0.8 km above the 0 C level, following horizontal rain structure rather than the bright band); beam width 1 deg.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import xarray as xr

from radarwow.core import Algorithm, Reference, register

KE_RE_M = 4.0 / 3.0 * 6371e3
DIMS = ("azimuth", "range")


def beam_height_km(r_m, elevation_deg):
    """Beam-centre height above the radar (4/3-Earth model), km."""
    el = np.deg2rad(elevation_deg)
    r = np.asarray(r_m, dtype=float)
    return (np.sqrt(r ** 2 + KE_RE_M ** 2 + 2 * r * KE_RE_M * np.sin(el)) - KE_RE_M) / 1e3


def ground_km(r_m, elevation_deg):
    """Distance along the ground (km) of a gate at slant range ``r_m``."""
    el = np.deg2rad(elevation_deg)
    r = np.asarray(r_m, dtype=float)
    h = beam_height_km(r, elevation_deg) * 1e3
    return KE_RE_M * np.arcsin(r * np.cos(el) / (KE_RE_M + h)) / 1e3


def _elevation(ds) -> float:
    """Measured (median) antenna elevation. Not ``sweep_fixed_angle``: in a damaged KTLX file a 0.9 deg sweep was
    labelled 0.48 deg, and the beam heights must follow where the antenna actually pointed."""
    if "elevation" in ds:
        return float(np.nanmedian(ds["elevation"].values))
    return float(np.asarray(ds["sweep_fixed_angle"]).ravel()[0])


def unique_tilts(dt, tol_deg: float = 0.15) -> list:
    """(elevation, sweep name) for each distinct elevation with DBZH, lowest first. Sweeps within ``tol_deg`` of
    each other are one tilt (NEXRAD split cuts, SAILS); of those the first with PHIDP (the dual-pol surveillance
    cut) is kept, else the first one."""
    names = sorted((k for k in dt.children if k.startswith("sweep")), key=lambda s: int(s.split("_")[1]))
    cand = []
    for name in names:
        ds = dt[name].to_dataset()
        if "DBZH" in ds:
            cand.append((_elevation(ds), int(name.split("_")[1]), name, "PHIDP" in ds))
    groups = []
    for c in sorted(cand):
        if groups and c[0] - groups[-1][0][0] <= tol_deg:
            groups[-1].append(c)
        else:
            groups.append([c])
    out = []
    for g in groups:
        g = sorted(g, key=lambda c: (not c[3], c[1]))
        out.append((g[0][0], g[0][2]))
    return out


class _Grid:
    """Common polar grid (n_az azimuth bins x ground-distance bins) and nearest-gate maps to and from each sweep."""

    def __init__(self, n_az: int, max_km: float, dg_km: float):
        self.n_az, self.dg = n_az, dg_km
        self.g = (np.arange(int(max_km / dg_km)) + 0.5) * dg_km
        self.az = (np.arange(n_az) + 0.5) * 360.0 / n_az

    def sweep_to_grid(self, ds, el):
        """Index arrays (az_idx [n_az], gate_idx [ng]) picking the sweep gate nearest to each grid cell; -1 = none."""
        az = ds["azimuth"].values % 360.0
        order = np.argsort(az)
        az_s = az[order]
        idx = np.searchsorted(az_s, self.az)
        lo, hi = np.clip(idx - 1, 0, az_s.size - 1), np.clip(idx, 0, az_s.size - 1)
        d_lo = np.abs((self.az - az_s[lo] + 180) % 360 - 180)
        d_hi = np.abs((self.az - az_s[hi] + 180) % 360 - 180)
        a_idx = order[np.where(d_lo <= d_hi, lo, hi)]
        a_idx = np.where(np.minimum(d_lo, d_hi) <= 1.5 * 360.0 / self.n_az + 0.5, a_idx, -1)
        g = ground_km(ds["range"].values, el)
        j = np.clip(np.searchsorted(g, self.g), 0, g.size - 1)
        jm = np.clip(j - 1, 0, g.size - 1)
        j = np.where(np.abs(g[jm] - self.g) < np.abs(g[j] - self.g), jm, j)
        ok = np.abs(g[j] - self.g) <= self.dg
        return a_idx, np.where(ok, j, -1)

    def grid_to_sweep(self, ds, el):
        """Index arrays (az_idx per ray, g_idx per gate) into the grid; -1 outside it."""
        az = ds["azimuth"].values % 360.0
        a = np.floor(az / (360.0 / self.n_az)).astype(int) % self.n_az
        g = ground_km(ds["range"].values, el)
        j = np.floor(g / self.dg).astype(int)
        return a, np.where(j < self.g.size, j, -1)


def _take(values, a_idx, g_idx):
    out = values[np.clip(a_idx, 0, None)][:, np.clip(g_idx, 0, None)]
    out = np.where((a_idx >= 0)[:, None] & (g_idx >= 0)[None, :], out, np.nan)
    return out


@dataclass
class TiltVPR:
    """Fit of the apparent VPR of one tilt (heights in km above the radar, Z in dBZ)."""

    elevation: float
    sweep: str
    corrected: bool = False
    reason: str = ""
    symmetric: bool = False
    h_top: float = np.nan
    h_peak: float = np.nan
    h_bottom: float = np.nan
    alpha: float = np.nan
    beta: float = np.nan
    z_peak: float = np.nan
    bad_fraction: float = np.nan
    profile_h: np.ndarray = field(default_factory=lambda: np.array([]), repr=False)
    profile_z: np.ndarray = field(default_factory=lambda: np.array([]), repr=False)
    profile_smooth: np.ndarray = field(default_factory=lambda: np.array([]), repr=False)

    def delta_db(self, h):
        """Apparent excess dBz_a(h) of the fitted model relative to the flat profile below h_b."""
        h = np.asarray(h, dtype=float)
        d = np.where(h > self.h_peak, self.alpha * (h - self.h_peak) + self.beta * (self.h_peak - self.h_bottom),
                     self.beta * (h - self.h_bottom))
        return np.where(h < self.h_bottom, 0.0, d)


def _running_mean(x, n):
    k = np.ones(n)
    num = np.convolve(np.nan_to_num(x), k, mode="same")
    den = np.convolve(np.isfinite(x).astype(float), k, mode="same")
    return np.where(den > 0, num / np.maximum(den, 1), np.nan)


def _lsq_slope(h, z):
    if h.size < 3 or np.ptp(h) <= 0:
        return np.nan
    return float(np.polyfit(h, z, 1)[0])


def fit_tilt(h, z, h_top, p, elevation=np.nan, sweep="", fz_km=None) -> TiltVPR:
    """Fit the Zhang & Qi linear model to one AVPR (``h`` ascending, km; ``z`` mean dBZ or NaN)."""
    t = TiltVPR(elevation, sweep, h_top=h_top, profile_h=h, profile_z=z)
    valid = np.isfinite(z)
    if valid.sum() < p["min_points"]:
        t.reason = f"only {valid.sum()} AVPR points in the BBA"
        return t
    sm = _running_mean(z, p["running_n"])
    mad = np.nanmean(np.abs(z[valid] - sm[valid]))
    bad = valid & (np.abs(z - sm) > p["mad_factor"] * mad)
    t.bad_fraction = float(bad.sum() / valid.sum())
    t.profile_smooth = sm
    if t.bad_fraction > p["max_bad_fraction"]:
        t.reason = f"{t.bad_fraction:.0%} of AVPR points flagged (> {p['max_bad_fraction']:.0%})"
        return t
    good = valid & ~bad & (h <= h_top)
    if good.sum() < p["min_points"]:
        t.reason = "too few good AVPR points below the BB top"
        return t
    smg = np.where(good, sm, np.nan)
    if fz_km is not None:  # radar-wow: the peak must lie near the 0 C level, not wherever rain is heaviest
        win = (h >= fz_km - p["peak_below_fz_km"]) & (h <= fz_km + p["peak_above_fz_km"])
        if not np.isfinite(smg[win]).any():
            t.reason = "no AVPR point in the peak search window around the 0 C level"
            return t
        ip = int(np.nanargmax(np.where(win, smg, np.nan)))
    else:
        ip = int(np.nanargmax(smg))
    t.h_peak, t.z_peak = float(h[ip]), float(smg[ip])
    # descend from the peak to the first minimum (Z floored at z_floor)
    # Descend from the peak until Z reaches the floor or rises again by more than flat_db above the lowest value
    # seen; the bottom is the highest point within flat_db of that minimum (a flat rain profile below the bright
    # band has no strict minimum, so the bottom is where the decrease ends).
    scanned, zmin, ib, stopped = [], np.inf, None, False
    for i in range(ip - 1, -1, -1):
        if not np.isfinite(smg[i]):
            continue
        if smg[i] > zmin + p["flat_db"]:
            stopped = True
            break
        scanned.append(i)
        zmin = min(zmin, smg[i])
        if smg[i] <= p["z_floor_dbz"]:
            stopped = True
            break
    if scanned:
        cand = next(i for i in scanned if smg[i] <= zmin + p["flat_db"])
        # bottom seen if the descent stopped (rise or floor) or the profile flattened before its lowest point
        if stopped or cand != scanned[-1]:
            ib = cand
    above = good & (h >= t.h_peak) & (h <= h_top)
    t.alpha = _lsq_slope(h[above], z[above])
    if not np.isfinite(t.alpha) or t.alpha >= 0:
        t.reason = f"slope above the peak is {t.alpha:.2f} dB/km (must be < 0)"
        return t
    if ib is not None and ib < ip:
        t.h_bottom = float(h[ib])
        below = good & (h >= t.h_bottom) & (h <= t.h_peak)
        t.beta = _lsq_slope(h[below], z[below])
    if ib is None or ib >= ip or not np.isfinite(t.beta) or t.beta <= 0:
        if h_top - t.h_peak > p["max_symmetric_depth_km"]:
            t.reason = (f"no BB bottom found and the maximum is {h_top - t.h_peak:.1f} km below the BB top "
                        "(not a bright band near the ground)")
            return t
        t.symmetric = True
        t.beta = -t.alpha
        t.h_bottom = 2 * t.h_peak - h_top
    if t.beta * (t.h_peak - t.h_bottom) < p["min_peak_excess_db"]:
        t.reason = f"bright-band excess {t.beta * (t.h_peak - t.h_bottom):.1f} dB below {p['min_peak_excess_db']} dB"
        return t
    t.corrected = True
    return t


@register
class ZhangQi2010VPR(Algorithm):
    name = "zhang_qi_2010_vpr"
    stage = "estimation"
    reference = Reference("J. Zhang, Y. Qi", 2010, "A real-time algorithm for the correction of brightband effects in radar-derived QPE",
                          "J. Hydrometeor. 11, 1157-1171", "10.1175/2010JHM1201.1")
    requires = ("DBZH",)
    provides = ("DBZH_VPR", "VPR_CORRECTION", "VPR_BBA", "VIL")
    notes = ("Volume algorithm: call with a DataTree and freezing_level_km (model 0 C height above the radar). "
             "Adds DBZH_VPR (corrected Z), VPR_CORRECTION (dB subtracted) and VPR_BBA (gate in the brightband-affected "
             "area) to every sweep; per-tilt fits in .fits. Gates outside the BBA keep DBZH. Ice-region correction "
             "(Qi et al. 2013) not included.")
    params = dict(freezing_level_km=None, vil_convective=6.5, z_composite_min=30.0, bba_depth_km=5.0, beamwidth_deg=1.0,
                  max_km=230.0, n_az=360, dg_km=1.0, bin_km=1.0, min_count=10, running_n=15, mad_factor=2.0,
                  max_bad_fraction=0.4, z_floor_dbz=28.0, flat_db=0.5, max_symmetric_depth_km=1.5, peak_below_fz_km=1.5, peak_above_fz_km=0.5, min_points=8, min_peak_excess_db=1.0)

    def run_sweep(self, ds, cfg):  # pragma: no cover - volume algorithm
        raise TypeError(f"{self.name} needs the whole volume (DataTree), not one sweep")

    def __call__(self, data, band=None, **params):
        if not isinstance(data, xr.DataTree):
            raise TypeError(f"{self.name} needs the whole volume (DataTree)")
        if params:
            self.params = {**self.params, **params}
        cfg = self._config(data, band)
        return self.run_volume(data, cfg)

    def run_volume(self, dt, cfg):
        p = self.params
        fz = p["freezing_level_km"]
        if fz is None or not np.isfinite(fz):
            raise ValueError(f"{self.name}: pass freezing_level_km (0 C height above the radar, km)")
        tilts = unique_tilts(dt)
        if not tilts:
            raise ValueError("no sweep with DBZH")
        grid = _Grid(p["n_az"], p["max_km"], p["dg_km"])
        bw = np.deg2rad(p["beamwidth_deg"])
        comp = np.full((grid.n_az, grid.g.size), np.nan)
        vil = np.zeros((grid.n_az, grid.g.size))
        for el, name in tilts:
            ds = dt[name].to_dataset()
            a, j = grid.sweep_to_grid(ds, el)
            z = _take(ds["DBZH"].values.astype(float), a, j)
            comp = np.fmax(comp, z)
            r_m = _take(np.broadcast_to(ds["range"].values, ds["DBZH"].shape), a, j)
            lw = 3.44e-3 * (10 ** (0.1 * np.nan_to_num(z, nan=-99.0))) ** (4.0 / 7.0)  # g m-3
            vil += np.where(np.isfinite(z), lw * r_m * bw, 0.0) / 1e3  # kg m-2
        strat = vil <= p["vil_convective"]
        col_ok = strat & (comp >= p["z_composite_min"])
        # apparent BB top: 0 C height + half the lowest beam's depth where its centre reaches the freezing level
        el0 = tilts[0][0]
        r_grid = np.linspace(0, 400e3, 4001)
        r_fz = np.interp(fz, beam_height_km(r_grid, el0), r_grid)
        h_top = fz + r_fz * np.tan(bw / 2) / 1e3
        self.h_top, self.vil, self.composite, self.grid = h_top, vil, comp, grid
        self.fits = []
        out = dt.copy()
        nb = max(1, int(round(p["bin_km"] / p["dg_km"])))
        for el, name in tilts:
            ds = dt[name].to_dataset()
            a, j = grid.grid_to_sweep(ds, el)
            col = _take(col_ok.astype(float), a, j) > 0.5
            h = np.broadcast_to(beam_height_km(ds["range"].values, el), ds["DBZH"].shape)
            z = ds["DBZH"].values.astype(float)
            bba = col & (h <= h_top) & (h >= h_top - p["bba_depth_km"]) & np.isfinite(z)
            # AVPR: azimuthal mean of dBZ in the BBA per range bin
            g = ground_km(ds["range"].values, el)
            bins = np.floor(g / (p["dg_km"] * nb)).astype(int)
            nbin = bins.max() + 1
            cnt = np.bincount(bins, weights=bba.sum(axis=0), minlength=nbin)
            sm = np.bincount(bins, weights=np.where(bba, z, 0.0).sum(axis=0), minlength=nbin)
            hb = np.bincount(bins, weights=h[0], minlength=nbin) / np.maximum(np.bincount(bins, minlength=nbin), 1)
            avpr = np.where(cnt >= p["min_count"], sm / np.maximum(cnt, 1), np.nan)
            fit = fit_tilt(hb, avpr, h_top, p, el, name, fz_km=fz)
            self.fits.append(fit)
            corr = np.zeros(z.shape)
            if fit.corrected:
                apply = bba & (h >= fit.h_bottom)
                corr = np.where(apply, fit.delta_db(h), 0.0)
            nds = out[name].to_dataset()
            nds["DBZH_VPR"] = (DIMS, (z - corr).astype("float32"), {"units": "dBZ", "long_name": "reflectivity, brightband VPR corrected",
                                                                     "reference": "Zhang & Qi 2010, doi:10.1175/2010JHM1201.1"})
            nds["VPR_CORRECTION"] = (DIMS, corr.astype("float32"), {"units": "dB", "long_name": "dB subtracted by VPR correction"})
            nds["VPR_BBA"] = (DIMS, bba, {"long_name": "gate in the brightband-affected area"})
            nds.attrs["radarwow_history"] = (nds.attrs.get("radarwow_history", "") +
                                             f"\n{self.name} ({self.reference.short()}) fz={fz:.2f} km h_top={h_top:.2f} km "
                                             f"{'corrected' if fit.corrected else 'not corrected: ' + fit.reason}").strip()
            out[name] = xr.DataTree(nds)
        return out
