"""
NAME:
	compare_to_models.py

AUTHOR:
	Bronwyn Reichardt Chu
	Swinburne
	2021

EMAIL:
	<breichardtchu@swin.edu.au>

PURPOSE:
	Compares the koffee results to literature models.
	Written on MacOS Mojave 10.14.5, with Python 3.7

FUNCTIONS INCLUDED:
    expected_value
    root_mean_square_deviation
    compare_outvel_to_models


MODIFICATION HISTORY:
		v.1.0 - first created April 2021

"""
import numpy as np

from . import calculate_outflow_velocity as calc_outvel
from . import calculate_star_formation_rate as calc_sfr
from . import plotting_functions as pf




def expected_value(x, model, scale_factor):
    """
    Finds the expected values for y from the given model

    Parameters
    ----------
    y : :obj:'~np.ndarray'
        the y-values to put into the model (e.g. sigma_sfr)

    model : function
        the definition for the model you need the expected value for
        (e.g. chen_et_al_2010)

    Returns
    -------
    y : :obj:'~np.ndarray'
        the expected values from the model
    """
    #create an array to save x values into
    y = np.full_like(x, np.nan, dtype=np.double)

    #iterate through the x values
    for i in np.arange(x.shape[0]):
        #calculate the expected value from the model
        expected_x_value, expected_y_value = model(x[i], x[i], scale_factor=scale_factor)

        #save in the array
        y[i] = expected_y_value[0]

    return y


def root_mean_square_deviation(data, y, model, scale_factor):
    """
    Finds the root mean square deviation of the data from model, where
    RMSD = sqrt(sum[x_exp - x_data]**2 / num_points)

    Parameters
    ----------
    data : :obj:'~np.ndarray'
        the measured values which we want to compare to the model

    y : :obj:'~np.ndarray'
        the y-values for each data point to put into the model (e.g. sigma_sfr)

    model : function
        the definition for the model you need the expected value for
        (e.g. chen_et_al_2010)

    Returns
    -------
    x : :obj:'~np.ndarray'
        the expected values from the model
    """
    #calculate the expected values
    expected = expected_value(y, model, scale_factor)

    #take the square of the difference between data and expected values
    diff_squared = (expected - data)**2

    #sum the difference squared
    diff_sum = np.nansum(diff_squared)

    #divide by the number of points
    diff_divide = diff_sum/(data.shape[0])

    #take the square root
    rmsd = np.sqrt(diff_divide)

    return rmsd


def compare_outvel_to_models(OIII_outflow_results, OIII_outflow_error, hbeta_outflow_results, hbeta_outflow_error, hbeta_no_outflow_results, hbeta_no_outflow_error, BIC_outflow, BIC_no_outflow, statistical_results, z, radius, header):
    """
    Calculates the outflow velocity and SFR surface density from the koffee results
    and compares each outflow velocity to the expected value for its SFR surface
    density from the Chen et al. and Murray et al. models.

    Parameters
    ----------
    OIII_outflow_results : :obj:'~numpy.ndarray'
        array of outflow results from KOFFEE for OIII line.  Used to calculate
        the outflow velocity.  Should be (7, statistical_results.shape)

    OIII_outflow_err : :obj:'~numpy.ndarray'
        array of the outflow result errors from KOFFEE for OIII line

    hbeta_outflow_results : :obj:'~numpy.ndarray'
        array of outflow results from KOFFEE for Hbeta line.  Used to calculate
        the Sigma SFR. Should be (7, statistical_results.shape)

    hbeta_outflow_err : :obj:'~numpy.ndarray'
        array of the outflow result errors from KOFFEE for Hbeta line

    hbeta_no_outflow_results : :obj:'~numpy.ndarray'
        array of single gaussian results from KOFFEE for Hbeta line.  Used to
        calculate the Sigma SFR.  Should be (4, statistical_results.shape)

    hbeta_no_outflow_err : :obj:'~numpy.ndarray'
        array of the single gaussian result errors from KOFFEE for Hbeta line

    BIC_outflow : :obj:'~numpy.ndarray'
        array of BIC values from the double gaussian fits, this is usually
        chi_square[1,:,:] returned from koffee

    BIC_no_outflow : :obj:'~numpy.ndarray'
        array of BIC values from the single gaussian fits, this is usually
        chi_square[0,:,:] returned from koffee

    statistical_results : :obj:'~numpy.ndarray'
        array of statistical results from KOFFEE.

    z : float
        redshift

    radius : :obj:'~numpy.ndarray'
        array of galaxy radius values

    header : FITS header object
        the header from the fits file

    Returns
    -------
    rmsd :
    """
    #calculate the outflow velocity
    vel_disp, vel_disp_err, vel_diff, vel_diff_err, vel_out, vel_out_err = calc_outvel.calc_outflow_vel(OIII_outflow_results, OIII_outflow_error, statistical_results, z)

    #calculate the sfr surface density - using just the systemic line, and including the flux line
    #don't include extinction since this was included in the continuum subtraction using ppxf
    sfr, sfr_err, total_sfr, sfr_surface_density, sfr_surface_density_err = calc_sfr.calc_sfr_koffee(hbeta_outflow_results, hbeta_outflow_error, hbeta_no_outflow_results, hbeta_no_outflow_error, statistical_results, z, header, include_extinction=False, include_outflow=False)

    #get the mask for the outflow spaxels
    flow_mask = (statistical_results>0) #& (sfr_surface_density>0.1)

    #flatten all the arrays and get rid of extra spaxels
    sig_sfr = sfr_surface_density[flow_mask]
    sig_sfr_err = sfr_surface_density_err[flow_mask]
    vel_out = vel_out[flow_mask]
    vel_out_err = vel_out_err[flow_mask]
    BIC_outflow = BIC_outflow[flow_mask]
    BIC_no_outflow = BIC_no_outflow[flow_mask]
    vel_disp = vel_disp[flow_mask]
    radius = radius[flow_mask]

    #create BIC diff
    BIC_diff = BIC_outflow - BIC_no_outflow
    BIC_diff_weak = (BIC_diff < -10) & (BIC_diff >= -30)
    BIC_diff_moderate = (BIC_diff < -30) & (BIC_diff >= -50)
    BIC_diff_strong = (BIC_diff < -50)

    #physical limits mask -
    #for the radius mask 6.1" is the 90% radius
    #also mask out the fits which lie on the lower limit of dispersion < 51km/s
    physical_mask = (radius < 6.1) & (vel_disp>51)

    #create the scale factors for the models
    chen_scale_factor = np.nanmedian(vel_out[BIC_diff_strong])/(np.nanmedian(sig_sfr[BIC_diff_strong])**0.1)
    murray_scale_factor = np.nanmedian(vel_out[BIC_diff_strong])/(np.nanmedian(sig_sfr[BIC_diff_strong])**2)

    #calculate the root mean square deviation from the models
    rmsd_chen_all = root_mean_square_deviation(vel_out, sig_sfr, pf.chen_et_al_2010, scale_factor=chen_scale_factor)
    rmsd_murray_all = root_mean_square_deviation(vel_out, sig_sfr, pf.murray_et_al_2011, scale_factor=murray_scale_factor)

    rmsd_chen_physical = root_mean_square_deviation(vel_out[physical_mask], sig_sfr[physical_mask], pf.chen_et_al_2010, scale_factor=chen_scale_factor)
    rmsd_murray_physical = root_mean_square_deviation(vel_out[physical_mask], sig_sfr[physical_mask], pf.murray_et_al_2011, scale_factor=murray_scale_factor)

    rmsd_chen_strong = root_mean_square_deviation(vel_out[BIC_diff_strong], sig_sfr[BIC_diff_strong], pf.chen_et_al_2010, scale_factor=chen_scale_factor)
    rmsd_murray_strong = root_mean_square_deviation(vel_out[BIC_diff_strong], sig_sfr[BIC_diff_strong], pf.murray_et_al_2011, scale_factor=murray_scale_factor)

    print("The RMSD of all spaxels with S/N > 20 compared to the Chen model:", rmsd_chen_all)
    print("The RMSD of all spaxels with S/N > 20 compared to the Murray model:", rmsd_murray_all)
    print('')

    print("The RMSD of physically selected spaxels compared to the Chen model:", rmsd_chen_physical)
    print("The RMSD of physically selected spaxels compared to the Murray model:", rmsd_murray_physical)
    print('')

    print("The RMSD of strong BIC spaxels compared to the Chen model:", rmsd_chen_strong)
    print("The RMSD of strong BIC spaxels compared to the Murray model:", rmsd_murray_strong)
