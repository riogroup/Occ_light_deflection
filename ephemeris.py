import numpy as np
import astropy.units as u
import astropy.constants as const
import spiceypy as spice

from astropy.coordinates import SkyCoord, GCRS
from astropy.time import Time
from astropy.utils.exceptions import AstropyWarning


def ephem_kernel(time, target, observer):
    """Calculates the ephemeris from kernel files.
    Adapted from the SORA package

    Parameters
    ----------
    time : `str`, `astropy.time.Time`
        Reference instant to calculate ephemeris. It can be a string
        in the ISO format (yyyy-mm-dd hh:mm:ss.s) or an astropy Time object.

    target : `str`
        IAU (kernel) code of the target.

    observer : `str`, `astropy.EarthLocation`
        IAU (kernel) code of the observer, a SORA observer object, a SORA
        spacecraft object, or one of ``'geocenter'`` and ``'barycenter'``.
        String IAU codes must be present in the loaded kernels.

    kernels : `list`, `str`
        List of paths for all the kernels.

    output : `str`, optional, default='ephemeris'
        The output of data. ``ephemeris`` will output the observed position,
        while ``vector`` will output the Cartesian state vector, without
        light time correction.

    Returns
    -------
    coord : `astropy.coordinates.SkyCoord`
        ICRS coordinate of the target when ``output='ephemeris'``. Cartesian
        state vector of the target relative to the observer when
        ``output='vector'``.
    """
    time = Time(time)
    t0 = Time('J2000', scale='tdb')
    dt = (time - t0)

    # calculates vector Solar System Barycenter -> Geocenter
    state = spice.spkezr('399', dt.sec, 'J2000', 'NONE', '0')[0]
    position1 = state[0:3]
    velocity1 = state[3:6]*u.km/u.s
    position1 = SkyCoord(*position1.T * u.km, representation_type='cartesian')
    # calculates vector Geocenter -> Topocenter
    if observer != '500':
        gcrs = observer.get_gcrs(obstime=time)
        # vector Solar System Barycenter -> Topocenter
        position1 = SkyCoord(position1.cartesian + gcrs.cartesian, representation_type='cartesian')
        gcrs_vel = gcrs.cartesian.differentials['s'].d_xyz
        velocity1 = velocity1 + gcrs_vel

    delt = 0 * u.s
    while True:
        # calculates new time
        tempo = dt - delt
        # calculates vector Solar System Barycenter -> Target
        state2 = spice.spkezr(target, tempo.sec, 'J2000', 'NONE', '0')[0]
        position2 = state2[0:3]
        velocity2 = state2[3:6]*u.km/u.s
        position2 = SkyCoord(*position2.T * u.km, representation_type='cartesian')
        # calculates vector Observer -> Target considering light time
        position = position2.cartesian - position1.cartesian
        # calculates new light time
        delt = (position.norm() / const.c).decompose()
        # if difference between new and previous light time is smaller than 0.001 sec, then continue.
        if np.all(np.absolute(((dt - tempo) - delt).sec) < 0.001):
            break
    velocity = velocity2 - velocity1
    coord = SkyCoord(position, representation_type='cartesian')
    if not coord.isscalar and len(coord) == 1:
        coord = coord[0]
    return coord, velocity


def on_sky_velocity(position, velocity):
    pos = SkyCoord(position, representation_type='cartesian').spherical
    e_alpha = SkyCoord(-np.sin(pos.lon), np.cos(pos.lon), 0, representation_type='cartesian')
    e_delta = SkyCoord(-np.cos(pos.lon)*np.sin(pos.lat), -np.sin(pos.lon)*np.sin(pos.lat),np.cos(pos.lat), representation_type='cartesian')
    da_cos_dec = np.sum(e_alpha.cartesian.xyz*velocity)/pos.distance
    ddec = np.sum(e_delta.cartesian.xyz*velocity)/pos.distance
    return da_cos_dec*u.rad, ddec*u.rad