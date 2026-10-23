"""Ephemeris and angular-velocity helpers for the accompanying Io notebooks.

Load the required SPICE kernels before calling ephem_kernel and clear them
afterwards. This module neither downloads nor loads kernels. Kernel coverage
must include the target and Earth relative to the Solar System barycenter
at the requested reception and retarded epochs.

The implementation uses J2000-oriented SPICE state vectors and a Euclidean,
iterated light-time correction. Observer-centered vectors are stored in
SkyCoord objects using the default ICRS frame metadata; constructing these
objects is not a transformation to barycentric positions. Read ephem_kernel's
Notes before using the returned coordinates in a different frame workflow.
"""

import numpy as np
import astropy.units as u
import astropy.constants as const
import spiceypy as spice

from astropy.coordinates import SkyCoord, GCRS, CartesianRepresentation
from astropy.time import Time
from astropy.utils.exceptions import AstropyWarning


def ephem_kernel(
    time,
    target,
    observer,
    tol=1e-4 * u.s,
    max_iter=20,
):
    """Compute observer-to-target positions with iterated geometric light time.

    Parameters
    ----------
    time : astropy.time.Time or Time-compatible input
        Reception epoch(s). Internally converted to TDB seconds from J2000.
    target : str
        SPICE target name or NAIF ID string, e.g. '501' for Io, '599' for Jupiter
        or '10' for the Sun. The loaded kernels must cover the requested target.
    observer : {'500'} or astropy.coordinates.EarthLocation
        The exact string '500' selects the geocenter; this is a local helper
        convention, mapped internally to SPICE Earth ID '399'. Any other input
        is expected to be an EarthLocation, optionally a broadcastable grid.
    tol : astropy.units.Quantity, optional
        Light-time convergence tolerance; default 1e-4 s. Convergence requires
        the light-time change to be below tol for every element.
    max_iter : int, optional
        Maximum number of light-time iterations; default 20. Must be positive.

    Returns
    -------
    coord : astropy.coordinates.SkyCoord
        Observer-to-target position with distances in km, expressed along the
        SPICE J2000 axes. The observer is evaluated at reception and the target
        at the converged retarded epoch. A length-one first coordinate axis is
        removed by the final indexing step.
    velocity : astropy.units.Quantity
        Target velocity at emission minus observer velocity at reception, in
        km/s. Cartesian components occupy the first axis: shape (3,) for a
        scalar, or (3, ...) for an array. Unlike coord, this is not squeezed.

    Raises
    ------
    RuntimeError
        If the light-time iteration does not converge within max_iter steps.
        SPICE errors for missing kernels or unavailable coverage also propagate.

    Notes
    -----
    All SPICE calls use frame='J2000', abcorr='NONE' and observer='0' (the Solar
    System barycenter). Light time is then solved here from the Euclidean
    observer-target distance divided by c. The observer remains at reception
    throughout the iteration. This also applies when the target is a deflector:
    Jupiter and the Sun are each evaluated at their own retarded epoch.
    No stellar aberration, gravitational light deflection or relativistic
    light-time delay is added by this function.

    For an EarthLocation, get_gcrs supplies the geocentric position and velocity
    at reception, including Earth's rotation via Astropy. The code directly
    adds these to Earth's barycentric state, without an additional relativistic
    GCRS-to-BCRS transformation. Astropy's Earth-orientation/IERS configuration
    therefore matters for reproducing topocentric calculations.

    The returned velocity is the difference of two state velocities; it does
    not include the derivative of emission time with respect to reception time.
    Consequently it is not the exact time derivative of the light-time-corrected
    position. The returned SkyCoord uses default ICRS metadata as a container
    for these observer-centered vectors; no frame transformation is performed.
    """

    time = Time(time).tdb
    t0 = Time('J2000', scale='tdb')
    dt = time - t0

    def calculate_state(body, delta_time):
        """Calls spkezr with a 1-D time array and restores its shape."""

        original_shape = delta_time.shape
        et = np.asarray(delta_time.sec).ravel()
        state, _ = spice.spkezr(body, et, 'J2000', 'NONE', '0',)

        # Shape: original time dimensions + six state components
        return np.asarray(state).reshape(original_shape + (6,))

    # Solar System Barycenter -> Geocenter
    state1 = calculate_state('399', dt)
    position1 = CartesianRepresentation(
        x=state1[..., 0] * u.km,
        y=state1[..., 1] * u.km,
        z=state1[..., 2] * u.km,
    )

    # Velocities are kept internally with shape (..., 3)
    velocity1 = state1[..., 3:6] * u.km / u.s

    # Geocenter -> Topocenter
    is_geocenter = isinstance(observer, str) and observer == '500'

    if not is_geocenter:
        # This assumes "observer" is an Astropy EarthLocation object
        gcrs = observer.get_gcrs(obstime=time)

        # Solar System Barycenter -> Topocenter
        position1 = position1 + gcrs.cartesian

        # d_xyz has shape (3, ...); move the component axis to the end
        gcrs_velocity = np.moveaxis(gcrs.cartesian.differentials['s'].d_xyz, 0, -1)
        velocity1 = velocity1 + gcrs_velocity

    # Initialize the light time with the shape of the observer grid
    delt = np.zeros(position1.shape) * u.s

    for _ in range(max_iter):
        retarded_time = dt - delt

        # Solar System Barycenter -> Target at retarded time
        state2 = calculate_state(target, retarded_time)
        position2 = CartesianRepresentation(
            x=state2[..., 0] * u.km,
            y=state2[..., 1] * u.km,
            z=state2[..., 2] * u.km,
        )
        velocity2 = state2[..., 3:6] * u.km / u.s

        # Observer -> Target
        position = position2 - position1

        # Updated light time
        new_delt = (position.norm() / const.c).to(u.s)
        if np.all(np.abs(new_delt - delt) < tol):
            break
        delt = new_delt

    else:
        raise RuntimeError(
            f"Light time computations did not converge after {max_iter} iterations."
        )

    # Return velocity with Astropy's usual component-first convention:
    # (3, n_lon, n_lat)
    velocity = np.moveaxis(velocity2 - velocity1, -1, 0)

    coord = SkyCoord(position, representation_type='cartesian')

    if not coord.isscalar and len(coord) == 1:
        coord = coord[0]

    return coord, velocity


def on_sky_velocity(position, velocity):
    """Project a single relative velocity onto the local sky tangent basis.

    Parameters
    ----------
    position : astropy.coordinates.CartesianRepresentation or SkyCoord
        One observer-to-target position with length units.
    velocity : astropy.units.Quantity
        One relative Cartesian velocity, shape (3,), with length/time units,
        expressed in the same axes as position.

    Returns
    -------
    da_cos_dec, ddec : astropy.units.Quantity
        Tangential angular-velocity components (dRA/dt * cos(Dec), dDec/dt).
        With position in km and velocity in km/s the units are rad/s; convert
        explicitly to arcsec/hour for the table. The first component already
        includes cos(Dec).

    Notes
    -----
    This helper uses unqualified np.sum and is intended for scalar positions
    and single velocity vectors, as used in the parameter notebook. It does
    not preserve independent samples when given position/velocity arrays.
    When using velocities from ephem_kernel, the light-time-derivative limitation
    documented there also applies to these projected angular velocities.
    """
    pos = SkyCoord(position, representation_type='cartesian').spherical
    e_alpha = CartesianRepresentation(x = -np.sin(pos.lon),
                                      y = np.cos(pos.lon), 
                                      z = 0)
    e_delta = CartesianRepresentation(x = -np.cos(pos.lon)*np.sin(pos.lat),
                                      y = -np.sin(pos.lon)*np.sin(pos.lat),
                                      z = np.cos(pos.lat))
    da_cos_dec = np.sum(e_alpha.xyz*velocity)/pos.distance
    ddec = np.sum(e_delta.xyz*velocity)/pos.distance
    return da_cos_dec*u.rad, ddec*u.rad