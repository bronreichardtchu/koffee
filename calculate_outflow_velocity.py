"""
NAME:
	calculate_outflow_velocity.py

AUTHOR:
	Bronwyn Reichardt Chu
	Swinburne
	2019

EMAIL:
	<breichardtchu@swin.edu.au>

PURPOSE:
	Calculates the outflow velocity from koffee results.
	Written on MacOS Mojave 10.14.5, with Python 3.7

MODIFICATION HISTORY:
		v.1.0 - first created September 2020

"""
import numpy as np


def calc_outflow_vel(outflow_results, outflow_error, statistical_results, z):
    """
    Calculates the outflow velocity

    Parameters
    ----------
    outflow_results : :obj:'~numpy.ndarray' object
        Array containing the outflow results found in koffee fits.  This will have
        either shape [6, i, j] or [7, i, j] depending on whether a constant was included
        in the koffee fit.  Either way, the flow and galaxy parameters are in the same shape.
        [[gal_sigma, gal_mean, gal_amp, flow_sigma, flow_mean, flow_amp], i, j]
        [[gal_sigma, gal_mean, gal_amp, flow_sigma, flow_mean, flow_amp, continuum_const], i, j]

    outflow_error : :obj:'~numpy.ndarray' object
        Array containing the outflow errors found in koffee fits.  This will have
        either shape [6, i, j] or [7, i, j] depending on whether a constant was included
        in the koffee fit.  Either way, the flow and galaxy parameters are in the same shape.
        [[gal_sigma, gal_mean, gal_amp, flow_sigma, flow_mean, flow_amp], i, j]
        [[gal_sigma, gal_mean, gal_amp, flow_sigma, flow_mean, flow_amp, continuum_const], i, j]

    statistical_results : :obj:'~numpy.ndarray' object
        Array containing the statistical results from koffee.  This has 0 if no flow
        was found, 1 if a flow was found, 2 if an outflow was found using a forced
        second fit due to the blue chi square test.

    redshift : float
        The redshift of the galaxy

    Returns
    -------
    vel_diff : :obj:'~numpy.ndarray' object
        Array with the mean difference between the outflow and systemic lines in
        km/s, and np.nan where no velocity was found.

    vel_diff_err : :obj:'~numpy.ndarray' object
        Array with the error for the mean difference between the outflow and systemic
        lines in km/s, and np.nan where no velocity was found.

    vel_out : :obj:'~numpy.ndarray' object
        Array with the outflow velocities, and np.nan where no velocity was found.

    vel_out_err : :obj:'~numpy.ndarray' object
        Array with the outflow velocity errors, and np.nan where no velocity was found.
    """
    #create array to keep velocity differences in, filled with np.nan
    vel_out = np.full_like(statistical_results, np.nan, dtype=np.double)
    vel_out_err = np.full_like(statistical_results, np.nan, dtype=np.double)
    vel_diff = np.full_like(statistical_results, np.nan, dtype=np.double)
    vel_diff_err = np.full_like(statistical_results, np.nan, dtype=np.double)

    #create an outflow mask - outflows found at 1 and 2
    flow_mask = (statistical_results > 0)

    #de-redshift the results
    systemic_mean = outflow_results[1,:,:][flow_mask]/(1+z)
    flow_mean = outflow_results[4,:,:][flow_mask]/(1+z)
    flow_sigma = outflow_results[3,:,:][flow_mask]/(1+z)

    #calculate the velocity difference
    #doing c*(lam_gal-lam_out)/lam_gal
    vel_diff_calc = 299792.458*(systemic_mean - flow_mean)/systemic_mean

    #calculate the error on the velocity difference
    #do the numerator first (lam_gal-lam_out)
    num_err = np.sqrt(outflow_error[1,:,:][flow_mask]**2 + outflow_error[4,:,:][flow_mask]**2)
    #now put that into the vel_diff error
    vel_diff_calc_err = vel_diff_calc*np.sqrt((num_err/(systemic_mean-flow_mean))**2 + outflow_error[1,:,:][flow_mask]**2/systemic_mean**2)

    #now doing 2*c*flow_sigma/lam_gal + vel_diff
    v_out = 2*flow_sigma*299792.458/systemic_mean + vel_diff_calc

    #calculate the error on v_out
    v_out_err = np.sqrt((flow_sigma**2/systemic_mean**2)*((outflow_error[3,:,:][flow_mask]/flow_sigma)**2 + (outflow_error[1,:,:][flow_mask]/systemic_mean)**2) + vel_diff_calc_err**2)

    #and put it into the array
    vel_diff[flow_mask] = vel_diff_calc
    vel_diff_err[flow_mask] = vel_diff_calc_err
    vel_out[flow_mask] = v_out
    vel_out_err[flow_mask] = v_out_err

    return vel_diff, vel_diff_err, vel_out, vel_out_err
