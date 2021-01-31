"""
NAME:
	calculate_energy_loading_factor.py

AUTHOR:
	Bronwyn Reichardt Chu
	Swinburne
	2019

EMAIL:
	<breichardtchu@swin.edu.au>

PURPOSE:
	Calculates the outflow energy loading factor from koffee results.
	Written on MacOS Mojave 10.14.5, with Python 3.7

FUNCTIONS INCLUDED:
    calc_mass_outflow_rate
    calc_mass_loading_factor
    calc_mass_loading_factor2

MODIFICATION HISTORY:
		v.1.0 - first created January 2021

"""
import numpy as np

from astropy.cosmology import WMAP9 as cosmo
from astropy.constants import c
from astropy.constants import m_p
from astropy import units as u

from . import calculate_outflow_velocity as calc_outvel
from . import calculate_star_formation_rate as calc_sfr
from . import calculate_mass_loading_factor as calc_mlf

import importlib
importlib.reload(calc_sfr)


def calc_energy_loading_factor(OIII_results, OIII_error, hbeta_results, hbeta_error, hbeta_no_outflow_results, hbeta_no_outflow_error, statistical_results, z):
    """
    Calculates the energy loading factor
        eta_E   = dotE_out/dotE_SN
                = 1/2 eta_m (v_out/707km/s)^2
    Using the calc_sfr.calc_sfr_koffee and the calc_mlf.calc_mass_loading_factor
    functions

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
    elf_out : :obj:'~numpy.ndarray'
        energy loading factor

    elf_max : :obj:'~numpy.ndarray'
        maximum energy loading factor if R_min is 350pc

    elf_min : :obj:'~numpy.ndarray'
        minimum energy loading factor if R_max is 2000pc
    """
    #calculate the mass loading factor
    mlf, mlf_max, mlf_min = calc_mlf.calc_mass_loading_factor(OIII_results, OIII_error, hbeta_results, hbeta_error, hbeta_no_outflow_results, hbeta_no_outflow_error, statistical_results, z)

    #calculate the outflow velocity
    vel_disp, vel_disp_err, vel_diff, vel_diff_err, vel_out, vel_out_err = calc_outvel.calc_outflow_vel(OIII_results, OIII_error, statistical_results, z)

    #need to put units on the outflow velocity (km/s)
    vel_out = vel_out * (u.km/u.s)
    vel_out_err = vel_out_err * (u.km/u.s)

    #put units on the 707km/s in the denominator
    denominator = 707.0 * (u.km/u.s)

    #do energy loading factor calculation
    elf = 1/2 * mlf * (vel_out/denominator)**2
    elf_max = 1/2 * mlf_max * (vel_out/denominator)**2
    elf_min = 1/2 * mlf_min * (vel_out/denominator)**2

    return elf, elf_max, elf_min
