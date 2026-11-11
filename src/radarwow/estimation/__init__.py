"""radar-wow stage 6: rain-rate estimators (R(Z), R(KDP), R(A), R(Z,ZDR), synthetic blend) brightband VPR correction and accumulation."""

from . import coefficients
from .accumulation import accumulate, to_fixed_azimuths
from .alpha import ADAPTIVE_MIN_DPHI, ZDRSlopeAlpha, alpha_from_slope, zdr_slope
from .blockage import Bech2003BeamBlockage, beam_block_fraction, copernicus_dem, fulton1998_correction_db, hybrid_scan
from .advection import advection_accumulate, motion_field, polar_to_cartesian
from .vpr import TiltVPR, ZhangQi2010VPR, fit_tilt, unique_tilts
from .rates import RainRateA, RainRateKDP, RainRateZZDR, SyntheticRainRate, ZRRelation, below_melting_layer

PLANNED: list[str] = []
__all__ = ["ADAPTIVE_MIN_DPHI", "ZDRSlopeAlpha", "alpha_from_slope", "zdr_slope", "Bech2003BeamBlockage", "beam_block_fraction", "copernicus_dem", "fulton1998_correction_db", "hybrid_scan", "accumulate", "advection_accumulate", "motion_field", "polar_to_cartesian", "to_fixed_azimuths", "RainRateA", "RainRateKDP", "RainRateZZDR", "SyntheticRainRate", "ZRRelation", "below_melting_layer", "coefficients", "TiltVPR", "ZhangQi2010VPR", "fit_tilt", "unique_tilts"]
