"""
NAME:
	calculate_mass_loading_factor.py

AUTHOR:
	Bronwyn Reichardt Chu
	Swinburne
	2019

EMAIL:
	<breichardtchu@swin.edu.au>

PURPOSE:
	Calculates the outflow velocity from koffee results.
	Written on MacOS Mojave 10.14.5, with Python 3.7

FUNCTIONS INCLUDED:
    calc_mass_outflow_rate
    calc_mass_loading_factor
    calc_mass_loading_factor2

MODIFICATION HISTORY:
		v.1.0 - first created September 2020

"""
import numpy as np

from astropy.cosmology import WMAP9 as cosmo
from astropy.constants import c
from astropy.constants import m_p
from astropy import units as u

from . import calculate_outflow_velocity as calc_outvel
from . import calculate_star_formation_rate as calc_sfr

import importlib
importlib.reload(calc_sfr)



def calc_mass_outflow_rate(OIII_results, OIII_error, hbeta_results, hbeta_error, statistical_results, z):
    """
    Calculates the mass outflow rate using the equation:
        M_out = (1.36m_H)/(gamma_Halpha n_e) * (v_out/R_out) * L_Halpha,broad
    To convert from Halpha to Hbeta luminosities:
        L_Halpha/L_Hbeta = 2.87
    So:
        M_out = (1.36m_H)/(gamma_Halpha n_e) * (v_out/R_out) *
                (L_Halpha,broad/L_Hbeta,broad) * L_Hbeta,broad

    Parameters
    ----------
    OIII_results : :obj:'~numpy.ndarray'
        array of outflow results from KOFFEE for OIII line.  Used to calculate
        the outflow velocity.  Should be (7, statistical_results.shape)

    OIII_error : :obj:'~numpy.ndarray'
        array of the outflow result errors from KOFFEE for OIII line

    hbeta_results : :obj:'~numpy.ndarray'
        array of outflow results from KOFFEE for Hbeta line.  Used to calculate
        the Sigma SFR.  Should be (7, statistical_results.shape)

    hbeta_error : :obj:'~numpy.ndarray'
        array of the outflow result errors from KOFFEE for Hbeta line

    statistical_results : :obj:'~numpy.ndarray'
        array of statistical results from KOFFEE.

    z : float
        redshift

    Returns
    -------
    M_out : :obj:'~numpy.ndarray'
        mass outflow rate in units of g/s

    M_out_max : :obj:'~numpy.ndarray'
        maximum mass outflow rate in units of g/s if R_min is 350pc

    M_out_min : :obj:'~numpy.ndarray'
        minimum mass outflow rate in units of g/s if R_max is 2000pc
    """
    #from Calzetti 2001 PASP 113 we have L_Halpha/L_Hbeta = 2.87
    lum_ratio_alpha_to_beta = 2.87

    #m_H is the atomic mass of Hydrogen (in kg)
    m_H = m_p

    #gamma_Halpha is the Halpha emissivity at 10^4K (in erg cm^3 s^-1)
    gamma_Halpha = 3.56*10**-25 * u.erg * u.cm**3 / u.s

    #n_e is the local electron density in the outflow
    #use the same value as Davies et al. for now
    n_e = 380 * (u.cm)**-3

    #v_out comes from the OIII line
    vel_disp, vel_disp_err, vel_diff, vel_diff_err, vel_out, vel_out_err = calc_outvel.calc_outflow_vel(OIII_results, OIII_error, statistical_results, z)
    #put in the units for the velocity
    vel_out = vel_out * u.km/u.s
    vel_out_err = vel_out_err * u.km/u.s

    #R_out is the radial extent of the outflow
    #use the 90% radius of the galaxy as the maximum radius - this is 5" or 2kpc
    R_max = 2 * 1000 * u.parsec
    #use the resolution of the spaxels as the minimum radial extent - this is ~350pc
    R_min = 350 * u.parsec
    #then use the average as R_out
    R_out = (R_max + R_min)/2

    #L_Hbeta is the luminosity of the broad line of Hbeta (we want the outflow flux)
    systemic_flux, systemic_flux_err, outflow_flux, outflow_flux_err = calc_sfr.calc_flux_from_koffee(hbeta_results, hbeta_error, statistical_results, z, outflow=True)
    #put the units in erg/s/cm^2
    outflow_flux = outflow_flux * 10**(-16) * u.erg / (u.s*(u.cm**2))
    outflow_flux_err = outflow_flux_err * 10**(-16) * u.erg / (u.s*(u.cm**2))

    #now get rid of the cm^2
    #get the Hubble constant at z=0; this is in km/Mpc/s
    H_0 = cosmo.H(0)
    #use d = cz/H0 to find the distance in cm
    dist = (c*z/H_0).decompose().to('cm')
    print('distance:', dist)
    #multiply by 4*pi*d^2 to get rid of the cm
    L_Hbeta = (outflow_flux*(4*np.pi*(dist**2))).to('erg/s')

    #do the whole calculation
    M_out_max = (1.36*m_H) / (gamma_Halpha*n_e) * (vel_out/R_max) * lum_ratio_alpha_to_beta*L_Hbeta
    M_out_min = (1.36*m_H) / (gamma_Halpha*n_e) * (vel_out/R_min) * lum_ratio_alpha_to_beta*L_Hbeta
    M_out = (1.36*m_H) / (gamma_Halpha*n_e) * (vel_out/R_out) * lum_ratio_alpha_to_beta*L_Hbeta

    #decompose the units to g/s
    M_out = M_out.to(u.g/u.s)
    M_out_max = M_out_max.to(u.g/u.s)
    M_out_min = M_out_min.to(u.g/u.s)

    return M_out, M_out_max, M_out_min


def calc_mass_loading_factor(OIII_results, OIII_error, hbeta_results, hbeta_error, hbeta_no_outflow_results, hbeta_no_outflow_error, statistical_results, z):
    """
    Calculates the mass loading factor
        eta = M_out/SFR
    Using the calc_sfr.calc_sfr_koffee and the calc_mass_outflow_rate functions

    Parameters
    ----------
    OIII_results : :obj:'~numpy.ndarray'
        array of outflow results from KOFFEE for OIII line.  Used to calculate
        the outflow velocity.  Should be (7, statistical_results.shape)

    OIII_error : :obj:'~numpy.ndarray'
        array of the outflow result errors from KOFFEE for OIII line

    hbeta_results : :obj:'~numpy.ndarray'
        array of outflow results from KOFFEE for Hbeta line.  Used to calculate
        the Sigma SFR.  Should be (7, statistical_results.shape)

    hbeta_error : :obj:'~numpy.ndarray'
        array of the outflow result errors from KOFFEE for Hbeta line

    hbeta_no_outflow_results : :obj:'~numpy.ndarray'
        array of single gaussian results from KOFFEE for Hbeta line.  Used to
        calculate the Sigma SFR.  Should be (4, statistical_results.shape)

    hbeta_no_outflow_error : :obj:'~numpy.ndarray'
        array of the single gaussian result errors from KOFFEE for Hbeta line

    statistical_results : :obj:'~numpy.ndarray'
        array of statistical results from KOFFEE.

    z : float
        redshift

    Returns
    -------
    mlf_out : :obj:'~numpy.ndarray'
        mass loading factor

    mlf_max : :obj:'~numpy.ndarray'
        maximum mass loading factor if R_min is 350pc

    mlf_min : :obj:'~numpy.ndarray'
        minimum mass loading factor if R_max is 2000pc
    """
    #calculate the mass outflow rate (in g/s)
    m_out, m_out_max, m_out_min = calc_mass_outflow_rate(OIII_results, OIII_error, hbeta_results, hbeta_error, statistical_results, z)

    #calculate the SFR (I wrote this to give the answer without units...)
    #(I should probably change that!)
    sfr, sfr_err, total_sfr, sigma_sfr, sfr_surface_density_err = calc_sfr.calc_sfr_koffee(hbeta_results, hbeta_error, hbeta_no_outflow_results, hbeta_no_outflow_error, statistical_results, z, include_extinction=False, include_outflow=False)

    #put the units back onto the sfr (M_sun/yr)
    sfr = sfr * (u.solMass/u.yr)

    #put the sfr into g/s
    sfr = sfr.to(u.g/u.s)

    #calculate mass loading factor
    mlf = m_out/sfr

    mlf_max = m_out_max/sfr
    mlf_min = m_out_min/sfr

    return mlf, mlf_max, mlf_min


def calc_mass_loading_factor2(OIII_results, OIII_error, hbeta_results, hbeta_error, hbeta_no_outflow_results, hbeta_no_outflow_error, statistical_results, z):
    """
    Calculates the mass loading factor, simplifying the M_out/SFR equation
        eta = M_out/SFR
    Using the whole equation, not the functions (to double check).
    Where
        M_out = (1.36m_H)/(gamma_Halpha n_e) * (v_out/R_out) * (L_Halpha,broad/L_Hbeta,broad)*L_Hbeta,broad
    and
        SFR = C_Halpha (L_Halpha / L_Hbeta)_0 x 10^{-0.4A_Hbeta} x L_Hbeta,narrow[erg/s]
    so:
        eta = (1.36m_H)/(gamma_Halpha n_e) * (v_out/R_out) * (L_Hbeta,broad/L_Hbeta,narrow) * (1/C_Halpha x 10^{-0.4A_Hbeta})

    Parameters
    ----------
    OIII_results : :obj:'~numpy.ndarray'
        array of outflow results from KOFFEE for OIII line.  Used to calculate
        the outflow velocity.  Should be (7, statistical_results.shape)

    OIII_error : :obj:'~numpy.ndarray'
        array of the outflow result errors from KOFFEE for OIII line

    hbeta_results : :obj:'~numpy.ndarray'
        array of outflow results from KOFFEE for Hbeta line.  Used to calculate
        the Sigma SFR.  Should be (7, statistical_results.shape)

    hbeta_error : :obj:'~numpy.ndarray'
        array of the outflow result errors from KOFFEE for Hbeta line

    hbeta_no_outflow_results : :obj:'~numpy.ndarray'
        array of single gaussian results from KOFFEE for Hbeta line.  Used to
        calculate the Sigma SFR.  Should be (4, statistical_results.shape)

    hbeta_no_outflow_error : :obj:'~numpy.ndarray'
        array of the single gaussian result errors from KOFFEE for Hbeta line

    statistical_results : :obj:'~numpy.ndarray'
        array of statistical results from KOFFEE.

    z : float
        redshift

    Returns
    -------
    mlf_out : :obj:'~numpy.ndarray'
        mass loading factor

    mlf_max : :obj:'~numpy.ndarray'
        maximum mass loading factor if R_min is 350pc

    mlf_min : :obj:'~numpy.ndarray'
        minimum mass loading factor if R_max is 2000pc
    """
    #m_H is the atomic mass of Hydrogen (in kg)
    m_H = m_p

    #gamma_Halpha is the Halpha emissivity at 10^4K (in erg cm^3 s^-1)
    gamma_Halpha = 3.56*10**-25 * u.erg * u.cm**3 / u.s

    #n_e is the local electron density in the outflow
    #use the same value as Davies et al. for now
    n_e = 380 * (u.cm)**-3
    #could be anywhere from 100-700 so use +/- 300 for error
    n_e_error = 300 * (u.cm)**-3

    #v_out comes from the OIII line
    vel_disp, vel_disp_err, vel_diff, vel_diff_err, vel_out, vel_out_err = calc_outvel.calc_outflow_vel(OIII_results, OIII_error, statistical_results, z)
    #put in the units for the velocity
    vel_out = vel_out * u.km/u.s
    vel_out_err = vel_out_err * u.km/u.s

    #R_out is the radial extent of the outflow
    #use the 90% radius of the galaxy as the maximum radius - this is 5" or 2kpc
    R_max = 2 * 1000 * u.parsec
    #use the resolution of the spaxels as the minimum radial extent - this is ~350pc
    R_min = 350 * u.parsec
    #then use the average as R_out
    R_out = (R_max + R_min)/2

    #need the luminosity of the broad and narrow lines of Hbeta
    systemic_flux, systemic_flux_err, outflow_flux, outflow_flux_err = calc_sfr.calc_flux_from_koffee(hbeta_results, hbeta_error, statistical_results, z, outflow=True)

    #don't need the units because it is a ratio - they cancel out
    #calculate the ratio
    Hbeta_broad_to_narrow = outflow_flux/systemic_flux

    #outflow_flux_err = outflow_flux_err * 10**(-16) * u.erg / (u.s*(u.cm**2))

    #define C_Halpha, using Hao et al. 2011 ApJ 741:124
    #From table 2, uses a Kroupa IMF, solar metallicity and 100Myr
    #it has units of M_sun yr^-1 erg^-1 s
    c_halpha = 10**(-41.257) * (u.solMass*u.s)/(u.yr*u.erg)

    #do the whole calculation
    mlf_max = (1.36*m_H) / (c_halpha*gamma_Halpha*n_e) * (10**(-0.4*0.0)) * (vel_out/R_max) * Hbeta_broad_to_narrow
    mlf_min = (1.36*m_H) / (c_halpha*gamma_Halpha*n_e) * (10**(-0.4*0.0)) * (vel_out/R_min) * Hbeta_broad_to_narrow
    mlf = (1.36*m_H) / (c_halpha*gamma_Halpha*n_e) * (10**(-0.4*0.0)) * (vel_out/R_out) * Hbeta_broad_to_narrow

    #decompose the units
    mlf_max = mlf_max.decompose()
    mlf_min = mlf_min.decompose()
    mlf = mlf.decompose()

    return mlf, mlf_max, mlf_min
