"""Monopole light-deflection calculations accompanying the article's Io example.

The derivations and applicability conditions are not repeated here.
All functions adopt gamma = 1, so (1 + gamma) GM / c**2
is implemented as 2 GM / c**2. Each deflection call handles one deflecting body.

Conventions
-----------
O is the observer, A the deflector, E the ephemeris position, S a source,
and P a point in the tangent plane. Vectors use r_XY = x_Y - x_X and must
share Cartesian axes and the epoch conventions used to prepare the inputs.
The functions do not query ephemerides or apply coordinate transformations.

Supply physical inputs as Astropy quantities and vectors as Astropy Cartesian
representations (or SkyCoord for calc_kP, as documented below). Distances must
have length units and GM length**3 / time**2 units. Angular separations should
carry angular units. Deflection amplitudes and their angular derivatives are
returned as quantities equivalent to dimensionless values; angles are measured
in radians. Convert amplitudes to mas with u.dimensionless_angles().

The C estimators return lengths in (parallel, perpendicular) order, using the
directions defined in the article. They are estimates, not direct differences
between independently recomputed events. NumPy/Astropy broadcasting applies.
"""

import numpy as np
import astropy.constants as const
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SphericalRepresentation, CartesianRepresentation, SkyOffsetFrame

def delta_kpn(gm, vec_rso, vec_rao, vec_ras):
    """Return the post-Newtonian deflection vector for one deflector (Equation 9).

    Parameters
    ----------
    gm : astropy.units.Quantity
        Gravitational parameter of the deflector, with length**3 / time**2 units.
    vec_rso : astropy.coordinates.CartesianRepresentation
        Source-to-observer vector, x_O - x_S, with length units.
    vec_rao : astropy.coordinates.CartesianRepresentation
        Deflector-to-observer vector, x_O - x_A, with length units.
    vec_ras : astropy.coordinates.CartesianRepresentation
        Deflector-to-source vector, x_S - x_A, with length units.
        Consistent input vectors satisfy vec_rso = vec_rao - vec_ras.

    Returns
    -------
    astropy.coordinates.CartesianRepresentation
        Dimensionless deflection vector, using the sign convention of Equation 9.
        Sum vectors from different deflectors before taking a total norm.

    Notes
    -----
    This is the finite-distance vector expression. It does not regularize a
    singular alignment or check whether the ray intersects the deflector.
    """
    term = 2*gm/const.c**2
    num = vec_rso.cross(vec_rao.cross(vec_ras))
    den = vec_rso.norm()*vec_rao.norm()*(vec_ras.norm()*vec_rao.norm() + vec_ras.dot(vec_rao))
    delta_k = term*num/den
    return delta_k

def calc_alpha_star(gm, DA, psi_star):
    """Return the deflection amplitude for a source at infinity (Equation 10).

    Parameters
    ----------
    gm : astropy.units.Quantity
        Deflector GM, with length**3 / time**2 units.
    DA : astropy.units.Quantity
        Observer-to-deflector distance.
    psi_star : astropy.units.Quantity
        Angular separation between the star and the deflector.

    Returns
    -------
    astropy.units.Quantity
        Deflection amplitude in milliarcseconds (mas).
    """
    term1 = 2*gm/(const.c**2*DA)
    term2 = np.tan(psi_star/2)
    return (term1/term2).to(u.mas, equivalencies=u.dimensionless_angles())

def calc_alpha_E(gm, DA, DE, psi_E):
    """Return the finite-distance deflection amplitude at E (Equation 12).

    Parameters
    ----------
    gm : astropy.units.Quantity
        Deflector GM, with length**3 / time**2 units.
    DA, DE : astropy.units.Quantity
        Observer-to-deflector and observer-to-ephemeris distances, respectively.
    psi_E : astropy.units.Quantity
        Angular separation between E and the deflector.

    Returns
    -------
    astropy.units.Quantity
        Deflection amplitude at E in milliarcseconds (mas).

    Notes
    -----
    The deflector-to-E distance is computed internally from DA, DE and psi_E.
    """
    rae = np.sqrt(DA**2 + DE**2 - 2*DA*DE*np.cos(psi_E))
    term1 = 2*gm/(const.c**2*DA)
    term2 = DE*np.sin(psi_E)
    term3 = rae + DA - DE*np.cos(psi_E)
    return (term1*term2/term3).to(u.mas, equivalencies=u.dimensionless_angles())

def calc_der_star(gm, DA, psi_star):
    """Return d(alpha_star)/d(psi_star) at fixed DA (Equation 39).

    Parameters
    ----------
    gm : astropy.units.Quantity
        Deflector GM, with length**3 / time**2 units.
    DA : astropy.units.Quantity
        Observer-to-deflector distance.
    psi_star : astropy.units.Quantity
        Angular separation between the star and the deflector.

    Returns
    -------
    astropy.units.Quantity
        Signed dimensionless derivative, with both angles measured in radians.
        For 0 < psi_star < pi this derivative is negative.
    """
    term1 = 2*gm/(const.c**2*DA)
    term2 = (1 - np.cos(psi_star))
    return - term1/term2
    
def calc_der_E(gm, DA, DE, psi_E):
    """Return d(alpha_E)/d(psi_E) at fixed DA and DE (Equation 40).

    Parameters
    ----------
    gm : astropy.units.Quantity
        Deflector GM, with length**3 / time**2 units.
    DA, DE : astropy.units.Quantity
        Observer-to-deflector and observer-to-ephemeris distances, respectively.
    psi_E : astropy.units.Quantity
        Angular separation between E and the deflector.

    Returns
    -------
    astropy.units.Quantity
        Signed dimensionless derivative, with both angles measured in radians.
    """
    rae = np.sqrt(DA**2 + DE**2 - 2*DA*DE*np.cos(psi_E))
    term1 = 2*gm*DE/(const.c**2*DA)
    term2 = DA*np.cos(psi_E) - DE
    term3 = rae*(rae + DA - DE*np.cos(psi_E))
    return term1*term2/term3

def calc_CE(DE, delta_k_E, delta_k_star):
    """Return the linear correction amplitude at E (Equations 25 and 26).

    Parameters
    ----------
    DE : astropy.units.Quantity
        Observer-to-ephemeris distance.
    delta_k_E, delta_k_star : astropy.coordinates.CartesianRepresentation
        Dimensionless deflection vectors for E and the star, in the same axes.
        Supply either one body's contribution or the vector sum over bodies.

    Returns
    -------
    astropy.units.Quantity
        DE times the norm of (delta_k_E - delta_k_star), with length units.
        This is a magnitude; signed angular components are projected separately.
    """
    delta_theta = delta_k_E - delta_k_star
    return DE*delta_theta.norm()

def calc_CPE(der_E, RP, alpha_E, psi_E):
    """Estimate the residual from replacing P by E (Equation 42).

    Parameters
    ----------
    der_E : astropy.units.Quantity
        Signed dimensionless derivative returned by calc_der_E.
    RP : astropy.units.Quantity
        Tangent-plane displacement scale, with length units.
    alpha_E : astropy.units.Quantity
        Deflection amplitude at E, returned in mas by calc_alpha_E. Any
        equivalent angular unit is accepted, as is a dimensionless quantity
        interpreted as radians. Converted to numerical radians internally.
    psi_E : astropy.units.Quantity
        Angular separation between E and the deflector.

    Returns
    -------
    CPE_parallel, CPE_perpendicular : astropy.units.Quantity
        Nonnegative residual estimates, with length units, in the directions
        defined in the article. These are not signed vector components.
    """
    alpha_E_rad = alpha_E.to_value(u.rad, equivalencies=u.dimensionless_angles())
    CP_par = np.absolute(der_E*RP)
    CP_per = RP*alpha_E_rad/np.sin(psi_E)
    return CP_par, CP_per

def calc_CO(der_E, der_star, DA, DE, RO, alpha_E, psi_E, alpha_star, psi_star):
    """Estimate the residual from sharing a correction across observers (Equation 45).

    Parameters
    ----------
    der_E, der_star : astropy.units.Quantity
        Signed dimensionless angular derivatives from calc_der_E and calc_der_star.
    DA, DE : astropy.units.Quantity
        Observer-to-deflector and observer-to-ephemeris distances, respectively.
    RO : astropy.units.Quantity
        Observer-displacement scale used in the article, with length units.
    alpha_E, alpha_star : astropy.units.Quantity
        Deflection amplitudes returned in mas by calc_alpha_E and
        calc_alpha_star. Any equivalent angular units are accepted, including
        different units for the two inputs. Legacy dimensionless quantities
        are interpreted as radians. Converted to numerical radians internally.
    psi_E, psi_star : astropy.units.Quantity
        Angular separations of E and the star from the deflector, respectively.

    Returns
    -------
    CO_parallel, CO_perpendicular : astropy.units.Quantity
        Nonnegative residual estimates with length units, for one deflector.
    """
    alpha_E_rad = alpha_E.to_value(u.rad, equivalencies=u.dimensionless_angles())
    alpha_star_rad = alpha_star.to_value(u.rad, equivalencies=u.dimensionless_angles())
    
    term1 = np.absolute(der_E)
    term2 = np.absolute(der_star - der_E)*DE/DA
    CO_par = RO*(term1 + term2)
    
    term1 = alpha_E_rad/np.sin(psi_E)
    term2 = np.absolute(alpha_E_rad/np.sin(psi_E) - alpha_star_rad/np.sin(psi_star))*DE/DA
    CO_per = RO*(term1 + term2)
    
    return CO_par, CO_per

def calc_CT(der_E, der_star, WA, DE, RT, alpha_E, psi_E, alpha_star, psi_star, v_occ):
    """Estimate the residual from a time-independent correction (Equation 48).

    Parameters
    ----------
    der_E, der_star : astropy.units.Quantity
        Signed dimensionless angular derivatives from calc_der_E and calc_der_star.
    WA : astropy.units.Quantity
        Nonnegative angular-speed scale W_A, with angle/time units. In the
        parameter notebook this is the deflector's on-sky speed, with stellar
        motion neglected. It is not a signed separation-angle derivative.
    DE : astropy.units.Quantity
        Observer-to-ephemeris distance.
    RT : astropy.units.Quantity
        Track-length scale, corresponding to the interval RT / v_occ.
    alpha_E, alpha_star : astropy.units.Quantity
        Deflection amplitudes returned in mas by calc_alpha_E and
        calc_alpha_star. Any equivalent angular units are accepted, including
        different units for the two inputs. Legacy dimensionless quantities
        are interpreted as radians. Converted to numerical radians internally.
    psi_E, psi_star : astropy.units.Quantity
        Angular separations of E and the star from the deflector, respectively.
    v_occ : astropy.units.Quantity
        Positive occultation speed, with length/time units.

    Returns
    -------
    CT_parallel, CT_perpendicular : astropy.units.Quantity
        Nonnegative temporal residual estimates with length units, for one body.
    """
    alpha_E_rad = alpha_E.to_value(u.rad, equivalencies=u.dimensionless_angles())
    alpha_star_rad = alpha_star.to_value(u.rad, equivalencies=u.dimensionless_angles())
    # Strip the angular unit from W_A to form the dimensionless speed ratio.
    speed_ratio = (DE*WA/v_occ).to_value(u.dimensionless_unscaled, equivalencies=u.dimensionless_angles())
    
    term1 = np.absolute(der_E)
    term2 = np.absolute(der_star - der_E)*speed_ratio
    CT_par = RT*(term1 + term2)
    
    term1 = alpha_E_rad/np.sin(psi_E)
    term2 = np.absolute(alpha_E_rad/np.sin(psi_E) - alpha_star_rad/np.sin(psi_star))*speed_ratio
    CT_per = RT*(term1 + term2)
    
    return CT_par, CT_per

def calc_kP(gm, pos_A, pos_E, delta_k_star, f, g, delta_f_P=0*u.km, delta_g_P=0*u.km):
    """Iterate the tangent-plane correction for one deflector.

    Parameters
    ----------
    gm : astropy.units.Quantity
        Deflector GM, with length**3 / time**2 units.
    pos_A, pos_E : astropy.coordinates.SkyCoord
        Observer-to-deflector and observer-to-ephemeris positions in common
        Cartesian axes, with distances. Scalar positions or broadcastable grids.
    delta_k_star : astropy.coordinates.CartesianRepresentation
        Dimensionless stellar deflection due to this same deflector, evaluated
        for the corresponding observer(s). It is held fixed during iteration.
    f, g : astropy.units.Quantity
        Uncorrected tangent-plane coordinates, with length units. These base
        coordinates remain fixed while the correction is updated recursively.
    delta_f_P, delta_g_P : astropy.units.Quantity, optional
        Initial correction estimates, with length units; both default to zero.

    Returns
    -------
    delta_f_P_new, delta_g_P_new : astropy.units.Quantity
        Signed tangent-plane corrections in km, not the corrected coordinates.

    Notes
    -----
    The basis and DE are defined at E. Each update evaluates the deflection at P
    and projects delta_k_P - delta_k_star (Equations 2, 3, 9, 13, 19 and 20).
    Recursion continues while the largest non-NaN change in the correction over
    the input grid exceeds 1 m. There is no explicit iteration limit or NaN
    validation. The notebooks solve separately for Jupiter and the Sun, then
    sum their returned corrections.

    Implementation check required: the supplied P_dir expression uses delta_g_P
    in both coordinates, so delta_f_P does not update the f coordinate. The
    intended update described in the notebook uses (f + delta_f_P) there.
    This documentation revision preserves the original numerical implementation.
    """
    # Equation (2)
    e_f = CartesianRepresentation(x=-np.sin(pos_E.spherical.lon), y=np.cos(pos_E.spherical.lon), z=0)
    
    # Equation (3)
    e_g = CartesianRepresentation(x=-np.cos(pos_E.spherical.lon)*np.sin(pos_E.spherical.lat), y= -np.sin(pos_E.spherical.lon)*np.sin(pos_E.spherical.lat), z=np.cos(pos_E.spherical.lat))

    DE = pos_E.cartesian.norm()
    # Equation (13)
    P_dir = pos_E.cartesian + (f + delta_f_P)*e_f + (g + delta_g_P)*e_g
    
    # Equation (9)
    delta_k_P = delta_kpn(gm, vec_rso=-P_dir, vec_rao=-pos_A.cartesian, vec_ras=P_dir - pos_A.cartesian)
    
    # Equation (19)
    theta = delta_k_P - delta_k_star

    # Equation (20)
    delta_f_P_new = DE*theta.dot(e_f)
    delta_g_P_new = DE*theta.dot(e_g)
    
    if np.nanmax(np.sqrt((delta_f_P_new - delta_f_P)**2 + (delta_g_P_new - delta_g_P)**2)) > 1*u.m:
        delta_f_P_new, delta_g_P_new = calc_kP(gm, pos_A, pos_E, delta_k_star, f, g, delta_f_P_new, delta_g_P_new)

    return delta_f_P_new.to(u.km), delta_g_P_new.to(u.km)

