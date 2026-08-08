import numpy as np
import astropy.constants as const
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, SphericalRepresentation, SkyOffsetFrame

def delta_kpn(gm, vec_rso, vec_rao, vec_ras):
    term = 2*gm/const.c**2
    num = vec_rso.cross(vec_rao.cross(vec_ras))
    den = vec_rso.norm()*vec_rao.norm()*(vec_ras.norm()*vec_rao.norm() + vec_ras.dot(vec_rao))
    delta_k = term*num/den
    return delta_k

def calc_kP(gm, pos_A, pos_E, delta_k_star, theta_f=0*u.mas, theta_g=0*u.mas):
    P_dir = pos_E.spherical_offsets_by(theta_f, theta_g)
    P_dir = SkyCoord(P_dir.ra, P_dir.dec, pos_E.spherical.distance)
    delta_kp = delta_kpn(gm, vec_rso=-P_dir.cartesian, vec_rao=-pos_A.cartesian, vec_ras=P_dir.cartesian - pos_A.cartesian)
    ### 
    theta = delta_k_star - delta_kp

    e_f = SkyCoord(-np.sin(pos_E.spherical.lon), np.cos(pos_E.spherical.lon), 0, representation_type='cartesian')
    e_g = SkyCoord(-np.cos(pos_E.spherical.lon)*np.sin(pos_E.spherical.lat), -np.sin(pos_E.spherical.lon)*np.sin(pos_E.spherical.lat), np.cos(pos_E.spherical.lat), representation_type='cartesian')
    theta_f_new = np.sum(theta.xyz*e_f.cartesian.xyz, axis=0).to(u.rad, equivalencies=u.dimensionless_angles())
    theta_g_new = np.sum(theta.xyz*e_g.cartesian.xyz, axis=0).to(u.rad, equivalencies=u.dimensionless_angles())
    if np.nanmax(np.sqrt((theta_f_new - theta_f)**2 + (theta_g_new - theta_g)**2)) > 0.001*u.mas:
        return calc_kP(gm, pos_A, pos_E, delta_k_star, theta_f_new, theta_g_new)
    return theta, theta_f_new, theta_g_new

def calc_CE(DE, delta_k_E, delta_k_star):
    delta_theta = delta_k_E - delta_k_star
    return DE*delta_theta.norm()

def calc_der_E(gm, DA, DE, phi_E):
    rae = np.sqrt(DA**2 + DE**2 - 2*DA*DE*np.cos(phi_E))
    term1 = 2*gm*DE/(const.c**2*DA)
    term2 = DA*np.cos(phi_E) - DE
    term3 = rae*(rae + DA - DE*np.cos(phi_E))
    return term1*term2/term3

def calc_der_star(gm, DA, phi_star):
    term1 = 2*gm/(const.c**2*DA)
    term2 = (1 - np.cos(phi_star))
    return term1/term2

def calc_CPE(der, RP):
    return np.absolute(der*RP)

def calc_CO(der_E, der_star, DA, DE, RO):
    term1 = np.absolute(der_E*(1 - DE/DA))
    term2 = np.absolute(der_star*DE/DA)
    return RO*(term1 + term2)

def calc_CT(der_E, der_star, dot_phi_E_A, dot_phi_star_A, DE, RT, v_occ):
    term1 = DE*RT/v_occ
    term2 = np.absolute(der_E*dot_phi_E_A)
    term3 = np.absolute(der_star*dot_phi_star_A)
    return term1*(term2 + term3)