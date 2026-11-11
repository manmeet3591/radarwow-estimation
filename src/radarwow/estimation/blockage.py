"""Partial beam blockage from terrain (Bech et al. 2003, J. Atmos. Oceanic Technol. 20, 845-855,
doi:10.1175/1520-0426(2003)020<0845:TSOSPW>2.0.CO;2) and the hybrid scan (Fulton et al. 1998, Wea. Forecasting
13, 377-395, doi:10.1175/1520-0434(1998)013<0377:TWRA>2.0.CO;2).

Bech et al. treat the beam cross-section as a disc of radius a = r theta/2. With y = terrain height - beam-centre
height, the blocked fraction is 0 for y <= -a, 1 for y >= a, and otherwise
    PBB = [y sqrt(a^2 - y^2) + a^2 arcsin(y/a) + pi a^2 / 2] / (pi a^2)        (their Eq. 2, as in wradlib)
Blockage is carried forward along the ray: the cumulative blockage (CBB) at a gate is the largest PBB at or before
it. Beam heights use the 4/3-Earth model (standard refraction); Bech et al. show that ducting and antenna pointing
errors (0.1 deg ~ 10 % blockage) can change this.

Corrections: NEXRAD's (Fulton et al. 1998, as summarised by Bech et al.) adds +1/+2/+3/+4 dB for 11-29/30-43/
44-55/56-60 % occultation, none below 10 %, and above 60 % a higher tilt is used. The hybrid scan here takes, at
every place, the lowest tilt with CBB below ``threshold`` (0.5, the rule used by Zhang & Qi 2010).

Terrain: Copernicus GLO-90 DEM (90 m, heights above the EGM2008 geoid) from the public AWS bucket
``copernicus-dem-90m``; missing tiles (open sea) are 0 m. Buildings are not in the DEM.
"""

from __future__ import annotations

import os

import numpy as np
import xarray as xr

from radarwow.core import Algorithm, Reference, register

from .vpr import KE_RE_M, beam_height_km, ground_km

RE_M = 6371e3
DIMS = ("azimuth", "range")


# -- terrain -----------------------------------------------------------------
def _tile_key(lat, lon):
    ns, ew = ("N" if lat >= 0 else "S"), ("E" if lon >= 0 else "W")
    name = f"Copernicus_DSM_COG_30_{ns}{abs(lat):02d}_00_{ew}{abs(lon):03d}_00_DEM"
    return f"{name}/{name}.tif"


def copernicus_dem(lat0: float, lon0: float, radius_km: float, cache_dir: str, res_deg: float = 1 / 1200) -> xr.DataArray:
    """Terrain height (m) on a regular lat/lon grid covering ``radius_km`` around (lat0, lon0)."""
    import boto3
    import rasterio
    from botocore import UNSIGNED
    from botocore.config import Config

    dlat = radius_km / 111.2 + 0.05
    dlon = radius_km / (111.2 * np.cos(np.deg2rad(lat0))) + 0.05
    lats = np.arange(lat0 - dlat, lat0 + dlat, res_deg)
    lons = np.arange(lon0 - dlon, lon0 + dlon, res_deg)
    out = np.zeros((lats.size, lons.size), dtype="float32")
    os.makedirs(cache_dir, exist_ok=True)
    s3 = None
    for la in range(int(np.floor(lats[0])), int(np.floor(lats[-1])) + 1):
        for lo in range(int(np.floor(lons[0])), int(np.floor(lons[-1])) + 1):
            key = _tile_key(la, lo)
            local = os.path.join(cache_dir, os.path.basename(key))
            if not os.path.exists(local):
                s3 = s3 or boto3.client("s3", config=Config(signature_version=UNSIGNED), region_name="eu-central-1")
                try:
                    s3.download_file("copernicus-dem-90m", key, local)
                except Exception:  # noqa: BLE001 - no tile = sea
                    open(local + ".missing", "w").close() if not os.path.exists(local + ".missing") else None
                    continue
            if not os.path.exists(local):
                continue
            with rasterio.open(local) as src:
                z = src.read(1).astype("float32")
                t = src.transform
            iy = (lats >= la) & (lats < la + 1)
            ix = (lons >= lo) & (lons < lo + 1)
            r = np.clip(((lats[iy] - t.f) / t.e).astype(int), 0, z.shape[0] - 1)
            c = np.clip(((lons[ix] - t.c) / t.a).astype(int), 0, z.shape[1] - 1)
            out[np.ix_(iy, ix)] = z[np.ix_(r, c)]
    return xr.DataArray(out, dims=("lat", "lon"), coords={"lat": lats, "lon": lons},
                        attrs={"units": "m", "source": "Copernicus GLO-90 DEM (AWS copernicus-dem-90m)"})


def gate_lonlat(range_m, azimuth_deg, elevation_deg, lat0, lon0):
    """Ground longitude and latitude of every gate (4/3-Earth path, local spherical approximation)."""
    s = ground_km(range_m, elevation_deg)[None, :] * 1e3
    az = np.deg2rad(np.asarray(azimuth_deg))[:, None]
    lat = lat0 + np.rad2deg(s * np.cos(az) / RE_M)
    lon = lon0 + np.rad2deg(s * np.sin(az) / (RE_M * np.cos(np.deg2rad(lat0))))
    return lon, lat


def terrain_at(dem: xr.DataArray, lon, lat):
    """Bilinear terrain height (m) at the given points."""
    from scipy.ndimage import map_coordinates

    la, lo = dem["lat"].values, dem["lon"].values
    i = (np.asarray(lat) - la[0]) / (la[1] - la[0])
    j = (np.asarray(lon) - lo[0]) / (lo[1] - lo[0])
    return map_coordinates(dem.values, [i, j], order=1, mode="nearest")


# -- blockage ------------------------------------------------------------------
def beam_block_fraction(terrain_m, beam_centre_m, beam_radius_m):
    """Bech et al. (2003) Eq. 2: fraction of a disc-shaped beam below the terrain."""
    a = np.asarray(beam_radius_m, dtype=float)
    y = np.asarray(terrain_m, dtype=float) - np.asarray(beam_centre_m, dtype=float)
    yc = np.clip(y, -a, a)
    with np.errstate(invalid="ignore", divide="ignore"):
        f = (yc * np.sqrt(np.maximum(a ** 2 - yc ** 2, 0)) + a ** 2 * np.arcsin(np.clip(yc / a, -1, 1)) + np.pi * a ** 2 / 2) / (np.pi * a ** 2)
    return np.clip(np.where(a > 0, f, (y > 0).astype(float)), 0.0, 1.0)


def fulton1998_correction_db(cbb):
    """NEXRAD dB added to Z for a cumulative occultation (fraction); NaN above 60 % (use a higher tilt)."""
    p = np.asarray(cbb) * 100
    return np.select([p < 11, p < 30, p < 44, p < 56, p <= 60], [0.0, 1.0, 2.0, 3.0, 4.0], np.nan)


@register
class Bech2003BeamBlockage(Algorithm):
    name = "bech_2003_beam_blockage"
    stage = "estimation"
    reference = Reference("J. Bech, B. Codina, J. Lorente, D. Bebbington", 2003,
                          "The sensitivity of single polarization weather radar beam blockage correction to variability in the "
                          "vertical refractivity gradient", "J. Atmos. Oceanic Technol. 20, 845-855",
                          "10.1175/1520-0426(2003)020<0845:TSOSPW>2.0.CO;2")
    requires = ("DBZH",)
    provides = ("PBB", "CBB", "TERRAIN", "DBZH_BB")
    notes = ("Pass dem (from copernicus_dem) and site=(lat, lon, antenna altitude m). PBB per gate, CBB cumulative along "
             "the ray; DBZH_BB = DBZH + Fulton et al. 1998 dB correction (NaN where CBB > 60 %). 4/3-Earth refraction.")
    params = dict(dem=None, site=None, beamwidth_deg=1.0)

    def run_sweep(self, ds, cfg):
        dem, site = self.params["dem"], self.params["site"]
        if dem is None or site is None:
            raise ValueError(f"{self.name}: pass dem= and site=(lat, lon, altitude_m)")
        lat0, lon0, alt = site
        el = float(np.nanmedian(ds["elevation"].values))
        r = ds["range"].values
        lon, lat = gate_lonlat(r, ds["azimuth"].values, el, lat0, lon0)
        terrain = terrain_at(dem, lon, lat)
        beam = alt + beam_height_km(r, el)[None, :] * 1e3
        a = r[None, :] * np.tan(np.deg2rad(self.params["beamwidth_deg"]) / 2)
        pbb = beam_block_fraction(terrain, np.broadcast_to(beam, terrain.shape), np.broadcast_to(a, terrain.shape))
        cbb = np.maximum.accumulate(pbb, axis=1)
        out = ds.copy()
        out["TERRAIN"] = (DIMS, terrain.astype("float32"), {"units": "m"})
        out["PBB"] = (DIMS, pbb.astype("float32"), {"long_name": "partial beam blockage fraction (Bech et al. 2003)"})
        out["CBB"] = (DIMS, cbb.astype("float32"), {"long_name": "cumulative beam blockage fraction along the ray"})
        out["DBZH_BB"] = (DIMS, (ds["DBZH"].values + fulton1998_correction_db(cbb)).astype("float32"),
                          {"units": "dBZ", "long_name": "DBZH with NEXRAD blockage correction (Fulton et al. 1998)"})
        return out


def hybrid_scan(tilts, threshold: float = 0.5):
    """Combine co-located fields from several tilts, lowest first: ``tilts`` = list of (field, cbb) arrays on one grid.

    At each cell the lowest tilt with CBB < ``threshold`` is used. Returns (field, tilt index; -1 where none qualifies).
    """
    field = np.full(np.shape(tilts[0][0]), np.nan)
    idx = np.full(field.shape, -1, dtype="int8")
    for k, (f, c) in enumerate(tilts):
        take = (idx < 0) & np.isfinite(c) & (np.asarray(c) < threshold)
        field = np.where(take, f, field)
        idx = np.where(take, k, idx)
    return field, idx
