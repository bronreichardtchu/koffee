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

FUNCTIONS INCLUDED:
    calc_outflow_vel
    calc_save_as_fits
    binned_velocities

MODIFICATION HISTORY:
		v.1.0 - first created September 2020

"""
import math
import numpy as np
from astropy.io import fits


def calc_outflow_vel(outflow_results, outflow_error, statistical_results, z):
    """
    Calculates the outflow velocity

    Parameters
    ----------
    outflow_results : :obj:'~numpy.ndarray'
        Array containing the outflow results found in koffee fits.  This will have
        either shape [6, i, j] or [7, i, j] depending on whether a constant was
        included in the koffee fit.  Either way, the flow and galaxy parameters
        are in the same shape.
        [[gal_sigma, gal_mean, gal_amp, flow_sigma, flow_mean, flow_amp], i, j]
        [[gal_sigma, gal_mean, gal_amp, flow_sigma, flow_mean, flow_amp, continuum_const], i, j]

    outflow_error : :obj:'~numpy.ndarray'
        Array containing the outflow errors found in koffee fits.  This will have
        either shape [6, i, j] or [7, i, j] depending on whether a constant was
        included in the koffee fit.  Either way, the flow and galaxy parameters
        are in the same shape.
        [[gal_sigma, gal_mean, gal_amp, flow_sigma, flow_mean, flow_amp], i, j]
        [[gal_sigma, gal_mean, gal_amp, flow_sigma, flow_mean, flow_amp, continuum_const], i, j]

    statistical_results : :obj:'~numpy.ndarray'
        Array containing the statistical results from koffee.  This has 0 if no
        flow was found, 1 if a flow was found, 2 if an outflow was found using a
        forced second fit due to the blue chi square test.

    z : float
        The redshift of the galaxy

    Returns
    -------
    vel_disp : :obj:'~numpy.ndarray'
        Array with the dispersion of the outflow component in km/s, and np.nan
        where no velocity was found.

    vel_disp_err : :obj:'~numpy.ndarray'
        Array with the error for dispersion of the outflow component in km/s,
        and np.nan where no velocity was found.

    vel_diff : :obj:'~numpy.ndarray'
        Array with the mean difference between the outflow and systemic lines in
        km/s, and np.nan where no velocity was found.

    vel_diff_err : :obj:'~numpy.ndarray'
        Array with the error for the mean difference between the outflow and
        systemic lines in km/s, and np.nan where no velocity was found.

    vel_out : :obj:'~numpy.ndarray'
        Array with the outflow velocities in km/s, and np.nan where no velocity
        was found.

    vel_out_err : :obj:'~numpy.ndarray'
        Array with the outflow velocity errors in km/s, and np.nan where no
        velocity was found.
    """
    #create array to keep velocity differences in, filled with np.nan
    vel_out = np.full_like(statistical_results, np.nan, dtype=np.double)
    vel_out_err = np.full_like(statistical_results, 0.0, dtype=np.double)
    vel_diff = np.full_like(statistical_results, np.nan, dtype=np.double)
    vel_diff_err = np.full_like(statistical_results, 0.0, dtype=np.double)
    vel_disp = np.full_like(statistical_results, np.nan, dtype=np.double)
    vel_disp_err = np.full_like(statistical_results, 0.0, dtype=np.double)

    #create an outflow mask - outflows found at 1 and 2
    flow_mask = (statistical_results > 0)

    #de-redshift the results
    systemic_mean = outflow_results[1,:,:][flow_mask]/(1+z)
    flow_mean = outflow_results[4,:,:][flow_mask]/(1+z)
    flow_sigma = outflow_results[3,:,:][flow_mask]/(1+z)

    #calculate the velocity difference
    #doing c*(lam_gal-lam_out)/lam_gal
    vel_diff_calc = 299792.458*abs(systemic_mean - flow_mean)/systemic_mean

    #calculate the error on the velocity difference
    #do the numerator first (lam_gal-lam_out)
    num_err = np.sqrt(outflow_error[1,:,:][flow_mask]**2 + outflow_error[4,:,:][flow_mask]**2)
    #now put that into the vel_diff error
    vel_diff_calc_err = vel_diff_calc * np.sqrt((num_err/(systemic_mean-flow_mean))**2 + outflow_error[1,:,:][flow_mask]**2/systemic_mean**2)

    #calculate the dispersion
    vel_disp_calc = flow_sigma*299792.458/systemic_mean

    #calculate the error on velocity dispersion
    vel_disp_calc_err = vel_disp_calc * np.sqrt((outflow_error[3,:,:][flow_mask]/flow_sigma)**2 + (outflow_error[1,:,:][flow_mask]/systemic_mean)**2)

    #now doing 2*c*flow_sigma/lam_gal + vel_diff
    v_out = 2*vel_disp_calc + vel_diff_calc

    #calculate the error on v_out
    v_out_err = np.sqrt(vel_disp_calc_err**2 + vel_diff_calc_err**2)

    #and put it into the array
    vel_diff[flow_mask] = vel_diff_calc
    vel_diff_err[flow_mask] = vel_diff_calc_err
    vel_disp[flow_mask] = vel_disp_calc
    vel_disp_err[flow_mask] = vel_disp_calc_err
    vel_out[flow_mask] = v_out
    vel_out_err[flow_mask] = v_out_err

    return vel_disp, vel_disp_err, vel_diff, vel_diff_err, vel_out, vel_out_err


def calc_save_as_fits(outflow_results, outflow_error, statistical_results, z, header, gal_name, output_folder):
    """
    Calculates the outflow velocity and saves the results into a fits file.

    Parameters
    ----------
    outflow_results : :obj:'~numpy.ndarray'
        Array containing the outflow results found in koffee fits.  This will have
        either shape [6, i, j] or [7, i, j] depending on whether a constant was
        included in the koffee fit.  Either way, the flow and galaxy parameters
        are in the same shape.
        [[gal_sigma, gal_mean, gal_amp, flow_sigma, flow_mean, flow_amp], i, j]
        [[gal_sigma, gal_mean, gal_amp, flow_sigma, flow_mean, flow_amp, continuum_const], i, j]

    outflow_error : :obj:'~numpy.ndarray'
        Array containing the outflow errors found in koffee fits.  This will have
        either shape [6, i, j] or [7, i, j] depending on whether a constant was
        included in the koffee fit.  Either way, the flow and galaxy parameters
        are in the same shape.
        [[gal_sigma, gal_mean, gal_amp, flow_sigma, flow_mean, flow_amp], i, j]
        [[gal_sigma, gal_mean, gal_amp, flow_sigma, flow_mean, flow_amp, continuum_const], i, j]

    statistical_results : :obj:'~numpy.ndarray'
        Array containing the statistical results from koffee.  This has 0 if no
        flow was found, 1 if a flow was found, 2 if an outflow was found using a
        forced second fit due to the blue chi square test.

    z : float
        The redshift of the galaxy

    header : FITS file object
        The header of the data fits file, to be used in the new fits file

    gal_name : str
        the name of the galaxy, and whatever descriptor to be used in the saving
        (e.g. IRAS08_binned_2_by_1)

    output_folder : str
        where to save the fits file

    Returns
    -------
    A saved fits file
    """
    #calculate the outflow velocity
    vel_disp, vel_disp_err, vel_diff, vel_diff_err, vel_out, vel_out_err = calc_outflow_vel(outflow_results, outflow_error, statistical_results, z)

    #copy the header and change the keywords so it's just 2 axes
    new_header = header.copy()

    new_header['NAXIS'] = 2

    del new_header['NAXIS3']
    del new_header['CNAME3']
    del new_header['CRPIX3']
    del new_header['CRVAL3']
    del new_header['CTYPE3']
    del new_header['CUNIT3']

    try:
        del new_header['CD3_3']
    except:
        del new_header['CDELT3']

    #create HDU object for outflow velocity
    hdu = fits.PrimaryHDU(vel_out, header=new_header)
    hdu_error = fits.ImageHDU(vel_out_err, name='Error')

    #create HDU list
    hdul = fits.HDUList([hdu, hdu_error])

    #write to file
    hdul.writeto(output_folder+gal_name+'_outflow_velocity.fits')

    #create HDU object for velocity difference
    hdu = fits.PrimaryHDU(vel_diff, header=new_header)
    hdu_error = fits.ImageHDU(vel_diff_err, name='Error')

    #create HDU list
    hdul = fits.HDUList([hdu, hdu_error])

    #write to file
    hdul.writeto(output_folder+gal_name+'_outflow_velocity_difference.fits')

    #create HDU object for velocity dispersion
    hdu = fits.PrimaryHDU(vel_disp, header=new_header)
    hdu_error = fits.ImageHDU(vel_disp_err, name='Error')

    #create HDU list
    hdul = fits.HDUList([hdu, hdu_error])

    #write to file
    hdul.writeto(output_folder+gal_name+'_outflow_velocity_dispersion.fits')



def binned_velocities(out_vel, bin_size=[3,3]):
    """
    Calculates the range and median of the velocities in the binned area

    Parameters
    ----------
    data : :obj:'~numpy.ndarray'
        the data to be binned

    bin_size : int
        the number of spaxels to bin by (Default is [3,3], this will bin 3x3)

    Returns
    -------
    binned_data : :obj:'~numpy.ndarray'
        the binned data
    """
    #create empty array to put binned things in:
    #vel_out min, max, median and standard deviation
    binned_data = np.empty([4, math.ceil(out_vel.shape[0]/bin_size[0]), math.ceil(out_vel.shape[1]/bin_size[1])])

    #create counter for x direction
    start_xi = 0
    end_xi = bin_size[0]

    #iterate through x direction
    for x in np.arange(out_vel.shape[0]/bin_size[0]):
        #create counter for y direction
        start_yi = 0
        end_yi = bin_size[1]
        #iterate through y direction
        for y in np.arange(out_vel.shape[1]/bin_size[1]):

            #bin the data
            binned_data[0, int(x), int(y)] = np.nanmin(out_vel[start_xi:end_xi, start_yi:end_yi])
            binned_data[1, int(x), int(y)] = np.nanmax(out_vel[start_xi:end_xi, start_yi:end_yi])
            binned_data[2, int(x), int(y)] = np.nanmedian(out_vel[start_xi:end_xi, start_yi:end_yi])
            binned_data[3, int(x), int(y)] = np.nanstd(out_vel[start_xi:end_xi, start_yi:end_yi])

            #increase y counters
            start_yi += bin_size[1]
            end_yi += bin_size[1]

        #increase x counters
        start_xi += bin_size[0]
        end_xi += bin_size[0]

    return binned_data
