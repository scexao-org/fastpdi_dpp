from multiprocessing import cpu_count

import astropy.units as u
import numpy as np
from astropy.coordinates import EarthLocation

# important parameters
PIXEL_SCALE = 15.3  # mas / px
PUPIL_OFFSET = 2.7  # deg
SATSPOT_ANGLE = (45 - PUPIL_OFFSET) % 90  # deg
# Subaru location - DO NOT CHANGE!
SUBARU_LOC = EarthLocation(lat=19.825504 * u.deg, lon=-155.4760187 * u.deg)

FILTER_ANGULAR_SIZE = {
    "open": np.rad2deg(1.03e-6 / 7.79) * 3.6e6,
    "y": np.rad2deg(1.03e-6 / 7.79) * 3.6e6,
    "j": np.rad2deg(1.24e-6 / 7.79) * 3.6e6,
    "h": np.rad2deg(1.63e-6 / 7.79) * 3.6e6,
}

# limit default nproc since many operations are
# throttled by file I/O
DEFAULT_NPROC = min(cpu_count(), 8)
