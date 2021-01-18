"""
NAME:
	koffee_plots.py

AUTHOR:
	Bronwyn Reichardt Chu
	Swinburne
	2019

EMAIL:
	<breichardtchu@swin.edu.au>

PURPOSE:
	Contains the plotting routines for prepare_cubes and KOFFEE.
	Written on MacOS Mojave 10.14.5, with Python 3.7

FUNCTIONS INCLUDED:
    plot_compare_fits
    map_of_outflows
    map_of_outflows2
    sn_cut_plot
    proposal_plot
    plot_sfr_vout
    plot_flux_vout
    plot_vel_out_radius
    plot_outflow_frequency_radius
    plot_vel_disp_vel_diff
    plot_sigma_sfr
    plot_vel_diff_sfr
    plot_vdiff_amp_ratio

MODIFICATION HISTORY:
		v.1.0 - first created October 2019

"""

import numpy as np
from datetime import date

import matplotlib.pyplot as plt
from matplotlib import gridspec

#from .display_pixels import cap_display_pixels as cdp
from . import brons_display_pixels_kcwi as bdpk
from . import prepare_cubes as pc
from . import koffee
from . import calculate_outflow_velocity as calc_outvel
from . import calculate_mass_loading_factor as calc_mlf




#-------------------------------------------------------------------------------
#PLOTTING FUNCTIONS
#-------------------------------------------------------------------------------

def plot_compare_fits(lamdas, data, spaxels, z):
    """
    Plots the normalised single and double gaussian fits for the OIII 5007 line
    using a list of spaxels.

    Parameters
    ----------
    lamdas : :obj:'~numpy.ndarray'
        wavelength vector for the data

    data : :obj:'~numpy.ndarray'
        3D data cube for the galaxy with shape [len(lamdas),:,:]

    spaxels : list
        list of spaxels to compare

    z : float
        redshift

    Returns
    -------
    A plot comparing the OIII 5007 spaxel fits from single and double gaussians
    """
    #make a mask for the emission line
    OIII_mask = (lamdas>5008.24*(1+z)-20.0) & (lamdas<5008.24*(1+z)+20.0)

    #mask the wavelength
    lam_OIII = lamdas[OIII_mask]

    #create the fine sampling array
    fine_sampling = np.linspace(min(lam_OIII), max(lam_OIII), 1000)

    #get the number of spaxels
    spaxel_num = len(spaxels)

    #create a figure
    plt.rcParams.update(pf.get_rc_params())
    fig, ax = plt.subplots(nrows=2, ncols=spaxel_num, sharex=True, sharey=True, figsize=(spaxel_num*3, 4))
    plt.subplots_adjust(wspace=0, hspace=0, left=0.08, right=0.99, top=0.95)

    #iterate through the spaxels
    for i in range(spaxel_num):
        #mask the data to get the flux
        flux = data[OIII_mask, spaxels[i][0], spaxels[i][1]]

        #fit data with single gaussian
        gmodel1, pars1 = koffee.gaussian1_const(lam_OIII, flux)
        bestfit1 = koffee.fitter(gmodel1, pars1, lam_OIII, flux, verbose=False)

        #fit the data with double gaussian
        gmodel2, pars2 = koffee.gaussian2_const(lam_OIII, flux)
        bestfit2 = koffee.fitter(gmodel2, pars2, lam_OIII, flux, verbose=False)

        #find the significance level using the BIC difference
        BIC_diff = bestfit2.bic - bestfit1.bic
        print(BIC_diff)
        if -10 > BIC_diff >= -30:
            significance_level = 'weakly likely'
        elif -30 > BIC_diff >= -50:
            significance_level = 'moderately likely'
        elif -50 > BIC_diff:
            significance_level = 'strongly likely'

        #get the value to normalise by
        max_value = np.nanmax(flux)

        #create a plotting mask
        plotting_mask = (lam_OIII>lam_OIII[25]) & (lam_OIII<lam_OIII[-25])
        plotting_mask2 = (fine_sampling>lam_OIII[25]) & (fine_sampling<lam_OIII[-25])

        #plot the fits on the figure
        ax[0,i].step(lam_OIII[plotting_mask], flux[plotting_mask]/max_value, where='mid', c='k')
        ax[0,i].plot(fine_sampling[plotting_mask2], bestfit1.eval(x=fine_sampling[plotting_mask2])/max_value, c='r', ls='--', lw=1)

        ax[1,i].step(lam_OIII[plotting_mask], flux[plotting_mask]/max_value, where='mid', c='k', label='Data')
        ax[1,i].plot(fine_sampling[plotting_mask2], bestfit2.components[0].eval(bestfit2.params, x=fine_sampling[plotting_mask2])/max_value, c='tab:blue', label='Narrow component')
        ax[1,i].plot(fine_sampling[plotting_mask2], bestfit2.components[1].eval(bestfit2.params, x=fine_sampling[plotting_mask2])/max_value, c='tab:green', label='Broad component')
        ax[1,i].plot(fine_sampling[plotting_mask2], bestfit2.eval(x=fine_sampling[plotting_mask2])/max_value, c='r', ls='--', lw=1, label='Bestfit')

        ax[1,i].set_xlabel('Wavelength($\AA$)')
        ax[0,i].set_title(significance_level, fontsize='medium')

        if i == 0:
            ax[1,i].legend(fontsize='x-small', frameon=False, loc='upper left')
            ax[0,i].set_ylabel('Normalised Flux')
            ax[1,i].set_ylabel('Normalised Flux')
            ax[0,i].set_ylim(-0.05, 0.75)
            ax[1,i].set_ylim(-0.05, 0.75)

    plt.show()


def map_of_outflows(lamdas, xx_flat, yy_flat, rad_flat, data_flat, z, outflow_results, outflow_error, statistical_results):
    """
    Plots the map of outflow velocities.

    Parameters
    ----------
    lamdas : :obj:'~numpy.ndarray'
        wavelength vector for the data

    xx_flat : :obj:'~numpy.ndarray'
        x-coordinates for the data

    yy_flat : :obj:'~numpy.ndarray'
        y-coordinates for the data (same shape as xx_flat)

    rad_flat : :obj:'~numpy.ndarray'
        galaxy radius for each spaxel (same shape as xx_flat)

    data_flat : :obj:'~numpy.ndarray'
        the data in a 2D array with shape [len(lamdas), len(xx_flat)]

    z : float
        redshift of the galaxy

    outflow_results : :obj:'~numpy.ndarray'
        array of results from KOFFEE.  Used to calculate the outflow velocity.
        Should have shape [7, :, :]

    outflow_error : :obj:'~numpy.ndarray'
        array of the outflow result errors from KOFFEE.  Same shape as outflow_results

    statistical_results : :obj:'~numpy.ndarray'
        array of statistical results from KOFFEE.  Should have same shape as the
        second two dimensions of outflow_results.

    Returns
    -------
    A map of the outflow velocities
    """
    #calcualte the outflow velocities
    vel_disp, vel_disp_err, vel_diff, vel_diff_err, vel_out, vel_out_err = calc_outvel.calc_outflow_vel(outflow_results, outflow_error, statistical_results, z)

    #create outflow mask
    flow_mask = (statistical_results>0)

    xx_flat_out = xx_flat[flow_mask.reshape(-1)]
    yy_flat_out = yy_flat[flow_mask.reshape(-1)]
    vel_out = vel_out[flow_mask]

    #create an array with zero where there are no outflows, but there is high enough S/N
    no_vel_out = np.empty_like(statistical_results)
    no_vel_out = no_vel_out[np.isfinite(outflow_results[1,:,:])]
    no_vel_out[:] = 0.0

    xx_flat_no_out = xx_flat[np.isfinite(outflow_results[1,:,:]).reshape(-1)]
    yy_flat_no_out = yy_flat[np.isfinite(outflow_results[1,:,:]).reshape(-1)]

    #make limits for the plots
    xmin, xmax = xx_flat.min(), xx_flat.max()
    ymin, ymax = yy_flat.min(), yy_flat.max()
    vmin, vmax = vel_out.min(), vel_out.max()

    #create figure and subplots
    plt.rcParams.update(pf.get_rc_params())

    #get colours from cmasher
    colours = cmr.take_cmap_colors('cmr.gem', 3, cmap_range=(0.25, 0.85), return_fmt='hex')

    #fig, (ax1, ax2) = plt.subplots(1,2, sharey=True)
    fig, ax1 = plt.subplots(1,1, constrained_layout=True)

    #get the continuum contours
    i, j = statistical_results.shape
    cont_contours1 = pf.plot_continuum_contours(lamdas, np.reshape(xx_flat, (i,j)), np.reshape(yy_flat, (i, j)), np.reshape(data_flat, (data_flat.shape[0],i,j)), z, ax1)
    #cont_contours2 = plot_continuum_contours(lamdas, np.reshape(xx_flat, (67,24)), np.reshape(yy_flat, (67, 24)), np.reshape(data_flat, (data_flat.shape[0],67,24)), z, ax2)

    #create figure of just outflows
    outflow_spax = bdpk.display_pixels(xx_flat_out, yy_flat_out, vel_out.reshape(1,-1), axes=ax1, cmap=cmr.gem)#, vmin=100, vmax=500)
    ax1.set_xlim(xmin, xmax)
    ax1.invert_xaxis()
    ax1.set_ylabel('Arcseconds')
    ax1.set_xlabel('Arcseconds')
    cbar = plt.colorbar(outflow_spax, ax=ax1, shrink=0.8)
    cbar.set_label('Maximum Outflow Velocity [km/s]')
    #cont_contours1

    #create subplot of after S/N cut
    #ax2.set_title('Including No Flow Spaxels')
    #no_outflow_spax = bdpk.display_pixels(xx_flat_no_out, yy_flat_no_out, no_vel_out.reshape(1,-1), axes=ax2, vmin=-50.0, vmax=50.0, cmap='binary')
    #outflow_spax = bdpk.display_pixels(xx_flat_out, yy_flat_out, vel_out.reshape(1,-1), axes=ax2)#, vmin=vmin, vmax=vmax)
    #ax2.set_xlim(xmin, xmax)
    #ax2.set_ylim(-7.5,7.5)
    #ax2.invert_xaxis()
    #ax2.set_ylabel('Arcseconds')
    #ax2.set_xlabel('Arcseconds')
    #cbar = plt.colorbar(outflow_spax, ax=ax2, shrink=0.8)
    #cbar.set_label('Outflow Velocity ($v_{sys}-v_{broad}$)/$v_{sys} + 2\sigma_{broad}$ [km/s]')
    #cont_contours2

    #plt.suptitle('S/N Threshold: '+str(sn))

    plt.show()


def map_of_outflows2(outflow_results, statistical_results, lamdas, xx, yy, data, z):
    """
    Plots the velocity difference between the main galaxy line and the outflow
    where there is an outflow, and 0km/s where there is no outflow (v_broad-v_narrow).
    Contours from the continuum are added on top.  This one uses a different
    plotting method to the previous function.

    Parameters
    ----------
    outflow_results : :obj:'~numpy.ndarray'
        array of results from KOFFEE.  Used to calculate the outflow velocity.
        Should have shape [7, :, :]

    statistical_results : :obj:'~numpy.ndarray'
        array of statistical results from KOFFEE.  Should have same shape as the
        second two dimensions of outflow_results.

    xx : :obj:'~numpy.ndarray'
        x-coordinates for the data (not flattened array)

    yy : :obj:'~numpy.ndarray'
        y-coordinates for the data (same shape as xx)

    rad : :obj:'~numpy.ndarray'
        galaxy radius for each spaxel (same shape as xx)

    data : :obj:'~numpy.ndarray'
        the data in a 3D array with shape [len(lamdas), xx.shape]

    z : float
        redshift of the galaxy

    Returns
    -------
    Two figures, one just of vel_out with continuum contours plotted on top, and
    the other showing the spaxels which have high enough S/N to be measured, but
    return a result of no outflow as spaxels with vel_out = 0 km/s.
    """
    #create array to keep velocity difference in
    vel_out = np.empty_like(statistical_results)

    #create outflow mask
    flow_mask = (statistical_results>0)

    #de-redshift the data first!!!
    systemic_mean = outflow_results[1,:,:][flow_mask]/(1+z)
    flow_mean = outflow_results[4,:,:][flow_mask]/(1+z)
    flow_sigma = outflow_results[3,:,:][flow_mask]/(1+z)

    #find the velocity difference
    #doing c*(lam_gal-lam_out)/lam_gal
    vel_diff = 299792.458*(systemic_mean-flow_mean)/systemic_mean

    v_out = 2*flow_sigma*299792.458/systemic_mean + vel_diff

    vel_out[~flow_mask] = np.nan
    vel_out[flow_mask] = v_out

    #create an array with zero where there are no outflows, but there is high enough S/N
    no_vel_out = np.empty_like(vel_out)
    no_vel_out[np.isnan(outflow_results[1,:,:])] = np.nan
    no_vel_out[flow_mask] = np.nan
    no_vel_out[np.isnan(outflow_results[1,:,:])][statistical_results[np.isnan(outflow_results[1,:,:])]==0] = 0.0

    #find the median of the continuum for the contours
    cont_mask = (lamdas>4600*(1+z))&(lamdas<4800*(1+z))
    cont_median = np.median(data[cont_mask,:,:], axis=0)

    xmin, xmax = xx.min(), xx.max()
    ymin, ymax = yy.min(), yy.max()

    plt.figure()
    plt.rcParams['axes.facecolor']='black'
    im = plt.pcolormesh(xx, yy, vel_out, cmap='viridis_r')
    plt.contour(yy, xx, cont_median, colors='white', linewidths=0.7, alpha=0.7, levels=(0.2,0.3,0.4,0.7,1.0,2.0,4.0))
    plt.gca().invert_xaxis()
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.colorbar(im, label=r'$\Delta v_{\rm broad-narrow}$ (km s$^{-1}$)', pad=0.01)
    plt.xlabel('Arcseconds')
    plt.ylabel('Arcseconds')
    plt.show()

    plt.figure()
    plt.rcParams['axes.facecolor']='black'
    #im = plt.pcolormesh(yy, xx, vel_diff.T, cmap='viridis_r', vmin=-150, vmax=60)
    im1 = plt.pcolormesh(xx, yy, no_vel_out, cmap='binary', vmin=0.0, vmax=50.0)
    im2 = plt.pcolormesh(xx, yy, vel_out, cmap='viridis_r')
    #plt.gca().invert_xaxis()
    plt.contour(yy, xx, cont_median, colors='white', linewidths=0.7, alpha=0.7, levels=(0.2,0.3,0.4,0.7,1.0,2.0,4.0))
    plt.colorbar(im2, label=r'$\Delta v_{\rm broad-narrow}$ (km s$^{-1}$)', pad=0.01)
    plt.xlabel('Arcseconds')
    plt.ylabel('Arcseconds')
    plt.show()



def sn_cut_plot(lamdas, xx_flat, yy_flat, rad_flat, data_flat, z, sn):
    """
    Plots the two maps before and after the S/N cut is made, so that you can
    check which pixels have been removed.

    Parameters
    ----------
    lamdas : :obj:'~numpy.ndarray'
        wavelength vector for the data

    xx_flat : :obj:'~numpy.ndarray'
        x-coordinates for the data

    yy_flat : :obj:'~numpy.ndarray'
        y-coordinates for the data (same shape as xx_flat)

    rad_flat : :obj:'~numpy.ndarray'
        galaxy radius for each spaxel (same shape as xx_flat)

    data_flat : :obj:'~numpy.ndarray'
        the data in a 2D array with shape [len(lamdas), len(xx_flat)]

    z : float
        redshift of the galaxy

    sn : float
        signal-to-noise threshold value at which to cut the data

    Returns
    -------
    A two panel figure of the galaxy before and after the S/N cut

    """
    #find the S/N at the OIII line
    cont_mask = (lamdas>4866*(1+z))&(lamdas<4900*(1+z))
    OIII_mask = (lamdas>5006*(1+z))&(lamdas<5011*(1+z))
    rms = np.sqrt(np.mean(np.square(data_flat[cont_mask,:]), axis=0))
    s_n_OIII = np.median(data_flat[OIII_mask,:], axis=0)/rms

    #make limits for the plots
    xmin, xmax = xx_flat.min(), xx_flat.max()
    ymin, ymax = yy_flat.min(), yy_flat.max()
    vmin, vmax = s_n_OIII.min(), s_n_OIII.max()

    #create figure and subplots
    plt.rcParams.update(pf.get_rc_params())
    fig, (ax1, ax2) = plt.subplots(1,2, sharey=True)

    #get the continuum contours
    cont_contours1 = pf.plot_continuum_contours(lamdas, np.reshape(xx_flat, (67,24)), np.reshape(yy_flat, (67, 24)), np.reshape(data_flat, (data_flat.shape[0],67,24)), z, ax1)
    cont_contours2 = pf.plot_continuum_contours(lamdas, np.reshape(xx_flat, (67,24)), np.reshape(yy_flat, (67, 24)), np.reshape(data_flat, (data_flat.shape[0],67,24)), z, ax2)

    #create figure of before S/N cut
    ax1.set_title('Before S/N cut')
    before_sn = bdpk.display_pixels(xx_flat, yy_flat, s_n_OIII, axes=ax1, vmin=vmin, vmax=vmax)
    ax1.set_xlim(xmin, xmax)
    ax1.set_ylim(ymin, ymax)
    ax1.invert_xaxis()
    ax1.set_ylabel('Arcseconds')
    ax1.set_xlabel('Arcseconds')
    plt.colorbar(before_sn, ax=ax1, shrink=0.8)
    cont_contours1


    #do the S/N cut
    xx_flat, yy_flat, rad_flat, data_flat, s_n_OIII = pc.sn_cut(lamdas, xx_flat, yy_flat, rad_flat, data_flat, z, sn=sn)

    #create subplot of after S/N cut
    ax2.set_title('After S/N cut')
    after_sn = bdpk.display_pixels(xx_flat, yy_flat, s_n_OIII, axes=ax2, vmin=vmin, vmax=vmax)
    ax2.set_xlim(xmin, xmax)
    ax2.set_ylim(ymin, ymax)
    ax2.invert_xaxis()
    #ax2.set_ylabel('Arcseconds')
    ax2.set_xlabel('Arcseconds')
    plt.colorbar(after_sn, ax=ax2, shrink=0.8)
    cont_contours2

    plt.suptitle('S/N Threshold: '+str(sn))

    plt.show()


def proposal_plot():
    """
    Plot for the ESO proposal - need to think about function inputs
    """
    plt.rcParams.update(pf.get_rc_params())
    fig1, ax = plt.subplots(1,2, figsize=(8,4))

    ax[0].step(halpha_vel_low_NaD[~NII_I_mask_low_NaD], halpha_data_low_NaD[~NII_I_mask_low_NaD]/halpha_norm_low_NaD, where='mid', c='k', ls='-')
    ax[1].step(halpha_vel_red_halpha[~NII_I_mask_red_halpha], halpha_data_red_halpha[~NII_I_mask_red_halpha]/halpha_norm_red_halpa, where='mid', c='k', ls='-')

    ax[0].plot(halpha_vel_low_NaD_fine, koffee.gaussian_func(halpha_lam_fine, 11384.65, 6732.04, 1.46)/halpha_norm_low_NaD, c='k', ls='--')
    ax[1].plot(halpha_vel_red_halpha_fine, koffee.gaussian_func(halpha_lam_fine, 11810.63, 6726.52, 1.93)/halpha_norm_red_halpa, c='k', ls='--')
    ax[0].plot(halpha_vel_low_NaD_fine, koffee.gaussian_func(halpha_lam_fine, 474.05, 6729.46, 3.62)/halpha_norm_low_NaD, c='k', ls=':')
    ax[1].plot(halpha_vel_red_halpha_fine, koffee.gaussian_func(halpha_lam_fine, 1573.79, 6726.52, 4.95)/halpha_norm_red_halpa, c='k', ls=':')

    ax[0].step(NaD_vel_low_NaD, -NaD_data_low_NaD/NaD_norm_low_NaD, where='mid', c='C3', ls='-')
    ax[1].step(NaD_vel_red_halpha, -NaD_data_red_halpha/NaD_norm_red_halpha, where='mid', c='C3', ls='-')

    ax[0].plot(NaD_vel_low_NaD_fine, koffee.gaussian_func(NaD_lam_fine, 271.17, 6039.96, 1.46)/NaD_norm_low_NaD, c='C3', ls='--')
    ax[1].plot(NaD_vel_red_halpha_fine, koffee.gaussian_func(NaD_lam_fine, 777.59, 6038.74, 1.72)/NaD_norm_red_halpha, c='C3', ls='--')
    ax[0].plot(NaD_vel_low_NaD_fine, koffee.gaussian_func(NaD_lam_fine, 92.39, 6038.11, 1.74)/NaD_norm_low_NaD, c='C3', ls=':')
    ax[1].plot(NaD_vel_red_halpha_fine, koffee.gaussian_func(NaD_lam_fine, 304.88, 6036.55, 3.18)/NaD_norm_red_halpha, c='C3', ls=':')

    ax[0].set_xlim(-550,0)
    ax[1].set_xlim(-600,0)
    ax[0].set_ylim(0,1.3)
    ax[1].set_ylim(0,1.3)
    ax[0].set_xlabel('Velocity (km/s)')
    ax[1].set_xlabel('Velocity (km/s)')
    ax[0].set_ylabel('PDF')
    ax[0].set(adjustable='box')
    ax[1].set(adjustable='box')

    plt.show()



def plot_sfr_vout(OIII_outflow_results, OIII_outflow_error, hbeta_outflow_results, hbeta_outflow_error, hbeta_no_outflow_results, hbeta_no_outflow_error, statistical_results, z, colour_by=None, colour_by_array=None, weighted_average=True):
    """
    Plots the SFR surface density against the outflow velocity, with Sigma_SFR
    calculated using only the narrow component.  There is an option to colour
    the points in the figure using another variable.

    Parameters
    ----------
    OIII_outflow_results : :obj:'~numpy.ndarray'
        array of outflow results from KOFFEE for OIII line.  Used to calculate
        the outflow velocity.  Should be (7, statistical_results.shape)

    OIII_outflow_error : :obj:'~numpy.ndarray'
        array of the outflow result errors from KOFFEE for OIII line

    hbeta_outflow_results : :obj:'~numpy.ndarray'
        array of outflow results from KOFFEE for Hbeta line.  Used to calculate
        the Sigma SFR.  Should be (7, statistical_results.shape)

    hbeta_outflow_error : :obj:'~numpy.ndarray'
        array of the outflow result errors from KOFFEE for Hbeta line

    hbeta_no_outflow_results : :obj:'~numpy.ndarray'
        array of single gaussian results from KOFFEE for Hbeta line.  Used to
        calculate the Sigma SFR.  Should be (4, statistical_results.shape)

    hbeta_no_outflow_error : :obj:'~numpy.ndarray'
        array of the single gaussian result errors from KOFFEE for Hbeta line

    BIC_outflow : :obj:'~numpy.ndarray'
        array of BIC values from the double gaussian fits

    BIC_no_outflow : :obj:'~numpy.ndarray'
        array of BIC values from the single gaussian fits

    statistical_results : :obj:'~numpy.ndarray'
        array of statistical results from KOFFEE.

    z : float
        redshift

    colour_by : string
        the variable used for colouring the points on the graph, used in the
        plotting labels for the colourbar (Default=None)

    colour_by_array : :obj:'~numpy.ndarray'
        the array used for colouring the points on the graph (Default=None)

    weighted_average : boolean
        whether or not to take a weighted average using the errors (Default=True)

    Returns
    -------
    A one panel figure of outflow velocity against the SFR surface density

    """
    #calculate the outflow velocity
    vel_diff, vel_diff_err, vel_out, vel_out_err = calc_outvel.calc_outflow_vel(OIII_outflow_results, OIII_outflow_error, statistical_results, z)

    #calculate the sfr surface density - using just the systemic line, and including the flux line
    #don't include extinction since this was included in the continuum subtraction using ppxf
    sfr, total_sfr, sfr_surface_density, h_beta_integral_err = calc_sfr.calc_sfr_koffee(hbeta_outflow_results, hbeta_outflow_error, hbeta_no_outflow_results, hbeta_no_outflow_error, statistical_results, z, include_extinction=False, include_outflow=False)

    #get the sfr for the outflow spaxels
    flow_mask = (statistical_results>0)

    #flatten all the arrays and get rid of extra spaxels
    sig_sfr = sfr_surface_density[flow_mask]
    sig_sfr_err = h_beta_integral_err[flow_mask]
    vel_out = vel_out[flow_mask]
    vel_out_err = vel_out_err[flow_mask]
    if colour_by is not None:
        colour_by_array = colour_by_array[flow_mask]

    #
    colour_by_array = colour_by_array[flow_mask]
    BIC_mask = (colour_by_array<-10)

    #make sure none of the errors are nan values
    vel_out_err[np.where(np.isnan(vel_out_err)==True)] = np.nanmedian(vel_out_err)

    #do the calculations for all the bins
    num_bins = 5
    min_bin = None #-0.05
    max_bin = None #0.6

    if weighted_average == False:
        bin_center, v_out_bin_medians, v_out_bin_lower_q, v_out_bin_upper_q = pf.binned_median_quantile_log(sig_sfr[BIC_mask], vel_out[BIC_mask], num_bins=num_bins, weights=None, min_bin=min_bin, max_bin=max_bin)

    elif weighted_average == True:
        bin_center, v_out_bin_medians, v_out_bin_lower_q, v_out_bin_upper_q = pf.binned_median_quantile_log(sig_sfr, vel_out, num_bins=num_bins, weights=[vel_out_err], min_bin=min_bin, max_bin=max_bin)


    print(bin_center)
    print(v_out_bin_medians)

    #calculate the r value for the median values
    r_vel_out_med, p_value_v_out = pf.pearson_correlation(bin_center, v_out_bin_medians)

    #calculate the r value for all the values
    r_vel_out, p_value_v_out = pf.pearson_correlation(sig_sfr, vel_out)
    r_vel_out_BIC, p_value_v_out_BIC = pf.pearson_correlation(sig_sfr[BIC_mask], vel_out[BIC_mask])

    #create vectors to plot the literature trends
    sfr_surface_density_chen, v_out_chen = pf.chen_et_al_2010(sig_sfr.min(), sig_sfr.max(), scale_factor=np.nanmedian(vel_out[BIC_mask])/(np.nanmedian(sig_sfr[BIC_mask])**0.1))
    sfr_surface_density_murray, v_out_murray = pf.murray_et_al_2011(sig_sfr.min(), sig_sfr.max(), scale_factor=np.nanmedian(vel_out[BIC_mask])/(np.nanmedian(sig_sfr[BIC_mask])**2))

    #plot it
    plt.rcParams.update(pf.get_rc_params())
    plt.figure(figsize=(5,4))

    #----------------
    #Including Outflow Line Plots
    #-----------------
    if colour_by is not None:
        plt.scatter(sig_sfr, vel_out, marker='o', lw=0, label='Flow spaxels; R={:.2f}'.format(r_vel_out), alpha=0.6, c=colour_by_array)
        cbar = plt.colorbar()
        cbar.ax.set_ylabel(colour_by)
        plt.errorbar(5, 150, xerr=np.nanmedian(sig_sfr_err), yerr=np.nanmedian(vel_out_err), c='k')

    elif colour_by is None:
        #plt.errorbar(sig_sfr, vel_out, xerr=sig_sfr_err, yerr=vel_out_err, marker='o', lw=0, label='Flow spaxels; R={:.2f}'.format(r_vel_out), alpha=0.4, color='tab:blue', ecolor='tab:blue', elinewidth=1)
        plt.scatter(sig_sfr[BIC_mask], vel_out[BIC_mask], marker='o', lw=0, label='Definite Flow spaxels; R={:.2f}'.format(r_vel_out_BIC), alpha=0.6, c='tab:blue')
        plt.scatter(sig_sfr[~BIC_mask], vel_out[~BIC_mask], marker='o', lw=0, label='Likely Flow spaxels; R={:.2f}'.format(r_vel_out), alpha=0.6, c='tab:pink')

        plt.errorbar(5, 150, xerr=np.nanmedian(sig_sfr_err), yerr=np.nanmedian(vel_out_err), c='k')


    plt.fill_between(bin_center, v_out_bin_lower_q, v_out_bin_upper_q, color='tab:blue', alpha=0.3)
    plt.plot(bin_center, v_out_bin_medians, marker='', color='tab:blue', lw=3.0, label='Median; R={:.2f}'.format(r_vel_out_med))
    plt.plot(sfr_surface_density_chen, v_out_chen, ':k', label='Energy driven, $v_{out} \propto \Sigma_{SFR}^{0.1}$')
    plt.plot(sfr_surface_density_murray, v_out_murray, '--k', label='Momentum driven, $v_{out} \propto \Sigma_{SFR}^{2}$')
    plt.ylim(100, 500)
    #plt.ylim(-50, 550)
    plt.xscale('log')
    lgnd = plt.legend(frameon=False, fontsize='x-small', loc='lower left')
    lgnd.legendHandles[0]._legmarker.set_markersize(4)
    plt.ylabel('Maximum Outflow Velocity [km s$^{-1}$]')
    plt.xlabel('$\Sigma_{SFR}$ [M$_\odot$ yr$^{-1}$ kpc$^{-2}$]')

    plt.tight_layout()
    plt.show()



def plot_flux_vout(OIII_outflow_results, OIII_outflow_error, flux_line_outflow_results, flux_line_outflow_error, statistical_results, radius, z):
    """
    Plots the broad-to-narrow line flux ratio against the outflow velocity, the
    outflow velocity dispersion and the outflow velocity offset

    Parameters
    ----------
    OIII_outflow_results : :obj:'~numpy.ndarray'
        array of outflow results from KOFFEE for OIII line.  Used to calculate
        the outflow velocity.  Should be (7, statistical_results.shape)

    OIII_outflow_error : :obj:'~numpy.ndarray'
        array of the outflow result errors from KOFFEE for OIII line

    flux_line_outflow_results : :obj:'~numpy.ndarray'
        array of outflow results from KOFFEE for the emission line used in the
        broad-to-narrow flux ratio.  Should be (7, statistical_results.shape)

    flux_line_outflow_error : :obj:'~numpy.ndarray'
        array of the outflow result errors from KOFFEE for the emission line used
        in the broad-to-narrow flux ratio

    statistical_results : :obj:'~numpy.ndarray'
        array of statistical results from KOFFEE.

    radius : :obj:'~numpy.ndarray'
        array of galaxy radius values

    z : float
        redshift

    Returns
    -------
    A three panel graph of velocity offset, velocity dispersion and outflow
    velocity against the flux ratio

    """
    #calculate the outflow velocity
    vel_disp, vel_disp_err, vel_diff, vel_diff_err, vel_out, vel_out_err = calc_outvel.calc_outflow_vel(OIII_outflow_results, OIII_outflow_error, statistical_results, z)

    #calculate the flux for systematic and flow gaussians
    systemic_flux, systemic_flux_err, outflow_flux, outflow_flux_err = calc_sfr.calc_flux_from_koffee(flux_line_outflow_results, flux_line_outflow_error, statistical_results, z, outflow=True)

    #make the flow mask
    flow_mask = (statistical_results>0)

    #flatten all the arrays and get rid of extra spaxels
    vel_disp = vel_disp[flow_mask]
    vel_disp_err = vel_disp_err[flow_mask]
    vel_diff = vel_diff[flow_mask]
    vel_diff_err = vel_diff_err[flow_mask]
    vel_out = vel_out[flow_mask]
    vel_out_err = vel_out_err[flow_mask]
    systemic_flux = systemic_flux[flow_mask]
    outflow_flux = outflow_flux[flow_mask]
    radius = radius[flow_mask]

    #take the log and do the flux ratio
    flux_ratio = np.log10(outflow_flux/systemic_flux)


    #calculate bins
    num_bins=5
    min_bin = None
    max_bin = None
    bin_center, v_out_bin_medians, v_out_bin_lower_q, v_out_bin_upper_q = pf.binned_median_quantile_lin(flux_ratio, vel_out, num_bins=num_bins, weights=None, min_bin=None, max_bin=None)
    bin_center, vel_diff_bin_medians, vel_diff_bin_lower_q, vel_diff_bin_upper_q = pf.binned_median_quantile_lin(flux_ratio, vel_diff, num_bins=num_bins, weights=None, min_bin=min_bin, max_bin=max_bin)
    bin_center, disp_bin_medians, disp_bin_lower_q, disp_bin_upper_q = pf.binned_median_quantile_lin(flux_ratio, vel_disp, num_bins=num_bins, weights=None, min_bin=min_bin, max_bin=max_bin)



    #fit our own trends
    #popt_vout, pcov_vout = curve_fit(pf.fitting_function, flux_ratio_bin_medians, v_out_bin_medians)
    #popt_vel_diff, pcov_vel_diff = curve_fit(pf.fitting_function, flux_ratio_bin_medians, vel_diff_bin_medians)
    #popt_disp, pcov_disp = curve_fit(pf.fitting_function, flux_ratio_bin_medians, disp_bin_medians)

    #calculate the r value for the median values
    r_vel_out, p_value_v_out = pf.pearson_correlation(bin_center, v_out_bin_medians)
    r_vel_diff, p_value_v_diff = pf.pearson_correlation(bin_center, vel_diff_bin_medians)
    r_disp, p_value_disp = pf.pearson_correlation(bin_center, disp_bin_medians)



    #plot it
    plt.rcParams.update(pf.get_rc_params())
    fig, ax = plt.subplots(nrows=1, ncols=3, sharex=True, figsize=(12,5))

    ax[0].scatter(flux_ratio, vel_out, marker='o', lw=0, label='Flow spaxels', alpha=0.6, c=radius)
    ax[0].fill_between(bin_center, v_out_bin_lower_q, v_out_bin_upper_q, color='tab:blue', alpha=0.3)
    ax[0].plot(bin_center, v_out_bin_medians, marker='', color='tab:blue', lw=3.0, label='Median; R={:.2f}'.format(r_vel_out))
    #ax[0].plot(sfr_linspace, pf.fitting_function(sfr_linspace, *popt_vout), 'r-', label='Fit: $v_{out}=%5.0f$ $\Sigma_{SFR}^{%5.2f}$' % tuple(popt_vout))
    ax[0].legend(frameon=False, fontsize='x-small', loc='lower left')
    ax[0].set_ylabel('Maximum Outflow Velocity [km s$^{-1}$]')
    ax[0].set_xlabel('Log Broad/Narrow Flux')

    ax[1].scatter(flux_ratio, vel_diff, marker='o', lw=0, alpha=0.6, c=radius)
    ax[1].fill_between(bin_center, vel_diff_bin_lower_q, vel_diff_bin_upper_q, color='tab:blue', alpha=0.3)
    ax[1].plot(bin_center, vel_diff_bin_medians, marker='', color='tab:blue', lw=3.0, label='Median; R={:.2f}'.format(r_vel_diff))
    #ax[1].plot(sfr_linspace, pf.fitting_function(sfr_linspace, *popt_vel_diff), 'r-', label='Fit: $\mu_{sys}-\mu_{flow}=%5.0f$ $\Sigma_{SFR}^{%5.2f}$' % tuple(popt_vel_diff))
    #ax[1].set_xscale('log')
    ax[1].legend(frameon=False, fontsize='x-small', loc='lower left')
    ax[1].set_ylabel('Velocity Offset [km s$^{-1}$]')
    ax[1].set_xlabel('Log Broad/Narrow Flux')


    im = ax[2].scatter(flux_ratio, vel_disp, marker='o', lw=0, alpha=0.6, c=radius)
    ax[2].fill_between(bin_center, disp_bin_lower_q, disp_bin_upper_q, color='tab:blue', alpha=0.3)
    ax[2].plot(bin_center, disp_bin_medians, marker='', color='tab:blue', lw=3.0, label='Median; R={:.2f}'.format(r_disp))
    #ax[2].plot(sfr_linspace, pf.fitting_function(sfr_linspace, *popt_disp), 'r-', label='Fit: $\sigma_{flow}=%5.0f$ $\Sigma_{SFR}^{%5.2f}$' % tuple(popt_disp))
    #ax[2].set_xscale('log')
    #ax[2].set_ylim(30,230)
    plt.colorbar(im)
    ax[2].legend(frameon=False, fontsize='x-small', loc='lower left')
    ax[2].set_ylabel('Velocity Dispersion [km s$^{-1}$]')
    ax[2].set_xlabel('Log Broad/Narrow Flux')

    plt.tight_layout()
    plt.show()



def plot_vel_out_radius(rad, OIII_outflow_results, OIII_outflow_error, statistical_results, z):
    """
    Plots the outflow velocity against galaxy radius

    Parameters
    ----------
    rad : :obj:'~numpy.ndarray'
        array of galaxy radius values

    OIII_outflow_results : :obj:'~numpy.ndarray'
        array of outflow results from KOFFEE for OIII line.  Used to calculate
        the outflow velocity.  Should be (7, statistical_results.shape)

    OIII_outflow_error : :obj:'~numpy.ndarray'
        array of the outflow result errors from KOFFEE for OIII line

    statistical_results : :obj:'~numpy.ndarray'
        array of statistical results from KOFFEE.

    z : float
        redshift

    Returns
    -------
    A figure of the outflow radius against galaxy radius

    """
    #calculate the velocity
    vel_disp, vel_disp_err, vel_diff, vel_diff_err, vel_out, vel_out_err = calc_outvel.calc_outflow_vel(OIII_outflow_results, OIII_outflow_error, statistical_results, z)

    #create the flow mask
    flow_mask = (stat_results > 0)

    #flatten arrays and get rid of non-flow spaxels
    vel_out = vel_out[flow_mask]

    #mask out the spaxels above 500km/s (they're bad fits)
    mask_500 = vel_out<500
    vel_out = vel_out[mask_500]

    #apply to the rad array
    rad_flow = rad[flow_mask]
    rad_flow = rad_flow[mask_500]

    #create the median points for the outflowing vel
    first_bin, last_bin = rad_flow.min(), 6.4
    bin_width = (last_bin-first_bin)/8
    #loop through all the bins
    bin_edges = [first_bin, first_bin+bin_width]
    rad_bin_medians = []
    v_out_bin_medians = []
    rad_bin_stdev = []
    v_out_bin_stdev = []
    while bin_edges[1] <= last_bin+bin_width-bin_width/6:
        #create the bin
        rad_bin = rad_flow[(rad_flow>=bin_edges[0])&(rad_flow<bin_edges[1])]
        v_out_bin = vel_out[(rad_flow>=bin_edges[0])&(rad_flow<bin_edges[1])]

        #find the median in the bin
        rad_median = np.nanmedian(rad_bin)
        v_out_median = np.nanmedian(v_out_bin)

        #find the standard deviation in the bin
        rad_stdev = np.nanstd(rad_bin)
        v_out_stdev = np.nanstd(v_out_bin)

        #use the stdev to cut out any points greater than 2 sigma away from the median
        if np.any(v_out_bin >= v_out_median+2*v_out_stdev) or np.any(v_out_bin <= v_out_median-2*v_out_stdev):
            v_out_median = np.nanmedian(v_out_bin[(v_out_bin>v_out_median-2*v_out_stdev)&(v_out_bin<v_out_median+2*v_out_stdev)])
            v_out_stdev = np.nanstd(v_out_bin[(v_out_bin>v_out_median-2*v_out_stdev)&(v_out_bin<v_out_median+2*v_out_stdev)])

        rad_bin_medians.append(rad_median)
        v_out_bin_medians.append(v_out_median)
        rad_bin_stdev.append(rad_stdev)
        v_out_bin_stdev.append(v_out_stdev)

        #change bin_edges
        bin_edges = [bin_edges[0]+0.5, bin_edges[1]+0.5]

    rad_bin_medians = np.array(rad_bin_medians)
    v_out_bin_medians = np.array(v_out_bin_medians)
    rad_bin_stdev = np.array(rad_bin_stdev)
    v_out_bin_stdev = np.array(v_out_bin_stdev)
    print('radius medians', rad_bin_medians)
    print('v_out medians', v_out_bin_medians)

    #plot
    plt.rcParams.update(pf.get_rc_params())
    plt.figure()
    plt.scatter(rad_flow, vel_out, alpha=0.4)
    plt.plot(rad_bin_medians, v_out_bin_medians, marker='', color='tab:blue', lw=3.0)
    plt.axvline(2.5, ls='--', c='k', lw=3)
    plt.axvline(6.4, ls='--', c='k', lw=3)
    plt.ylim(100,500)
    plt.xlim(-0.4, 7.8)
    plt.xlabel('Radius [Arcseconds]')
    plt.ylabel('Maximum Outflow Velocity [km s$^{-1}$]')
    plt.show()



def plot_outflow_frequency_radius(rad_flat, stat_results):
    """
    Plots the frequency of outflow spaxels, and non-outflow spaxels against
    radius, using the flattened arrays.

    Parameters
    ----------
    rad_flat : :obj:'~numpy.ndarray'
        flattened array of galaxy radius values

    stat_results : :obj:'~numpy.ndarray'
        array of statistical results from KOFFEE

    Returns
    -------
    A figure showing the histogram of outflow spaxels against radius
    """
    #create flow mask
    flow_mask = (stat_results > 0)

    #create no_flow mask
    no_flow_mask = (stat_results == 0)

    #create low S/N mask
    low_sn_mask = np.isnan(stat_results)

    #get the radius array for each type of spaxel
    rad_flow = rad_flat[flow_mask]
    rad_no_flow = rad_flat[no_flow_mask]
    rad_low_sn = rad_flat[low_sn_mask]

    #get total number of spaxels
    total_spaxels = stat_results.shape[0]
    print(total_spaxels)
    total_outflows = rad_flow.shape[0]
    total_no_flows = rad_no_flow.shape[0]

    #iterate through the radii and get the number of spaxels in that range
    num_flow = []
    num_no_flow = []
    num_low_sn = []
    num_spax_rad = []
    num_spax_flow_rad = []

    cum_spax = []
    cumulative_spaxels = 0

    step_range = 0.5

    for i in np.arange(0,21, step_range):
        flow_bin = rad_flow[(rad_flow>=i)&(rad_flow<(i+step_range))].shape[0]
        no_flow_bin = rad_no_flow[(rad_no_flow>=i)&(rad_no_flow<(i+step_range))].shape[0]
        low_sn_bin = rad_low_sn[(rad_low_sn>=i)&(rad_low_sn<(i+step_range))].shape[0]

        rad_bin = flow_bin+no_flow_bin+low_sn_bin
        rad_flow_bin = flow_bin + no_flow_bin

        cumulative_spaxels = cumulative_spaxels + rad_flow_bin
        cum_spax.append(cumulative_spaxels)

        num_flow.append(flow_bin)
        num_no_flow.append(no_flow_bin)
        num_low_sn.append(low_sn_bin)
        num_spax_rad.append(rad_bin)
        num_spax_flow_rad.append(rad_flow_bin)

    num_flow = np.array(num_flow)
    num_no_flow = np.array(num_no_flow)
    num_low_sn = np.array(num_low_sn)
    num_spax_rad = np.array(num_spax_rad)
    num_spax_flow_rad = np.array(num_spax_flow_rad)
    cum_spax = np.array(cum_spax)
    cum_spax = cum_spax/cum_spax[-1]

    #create the plot
    plt.rcParams.update(pf.get_rc_params())
    """
    plt.figure()
    plt.fill_between(np.arange(0,21, step_range), 0.0, cum_spax, color='grey', alpha=0.4)
    plt.plot(np.arange(0,21, step_range), num_low_sn/num_spax_rad, label='Low S/N', alpha=0.5)
    plt.plot(np.arange(0,21, step_range), num_flow/num_spax_rad, label='Outflows', alpha=0.5)
    plt.plot(np.arange(0,21, step_range), num_no_flow/num_spax_rad, label='No Outflows', alpha=0.5)
    plt.xlabel('Radius [Arcseconds]')
    plt.ylabel('Fraction of spaxels')
    plt.xlim(0,20)
    plt.legend()
    plt.show()
    """

    plt.figure()
    plt.fill_between(np.arange(0,21, step_range), 0.0, cum_spax, color='grey', alpha=0.4)
    plt.plot(np.arange(0,21, step_range), num_flow/num_spax_flow_rad, label="Outflow Spaxels", lw=3)
    plt.text(0.2, 0.91, 'Outflow Spaxels', fontsize=14, color='tab:blue')
    plt.plot(np.arange(0,21, step_range), num_no_flow/num_spax_flow_rad, label="No Flow Spaxels", lw=3)
    plt.text(0.2, 0.05, 'No Flow Spaxels', fontsize=14, color='tab:orange')
    plt.axvline(6.4, ls='--', c='k', lw=3)
    plt.axvline(2.5, ls='--', c='k', lw=3)
    plt.xlabel('Radius [Arcseconds]')
    plt.ylabel('Spaxel PDF')
    plt.xlim(-0.4, 7.8)
    #plt.legend(loc='upper right')
    plt.show()



def plot_vel_disp_vel_diff(outflow_results, outflow_error, stat_results, z):
    """
    Plots the velocity difference between systemic and flow components against
    the flow sigma

    Parameters
    ----------
    outflow_results : :obj:'~numpy.ndarray'
        array of outflow results from KOFFEE.  Used to calculate the outflow
        velocity. Should be (7, stat_results.shape)

    outflow_error : :obj:'~numpy.ndarray'
        array of the outflow result errors from KOFFEE.

    stat_results : :obj:'~numpy.ndarray'
        array of statistical results from KOFFEE

    z : float
        redshift

    Returns
    -------
    A figure of outflow velocity offset against outflow velocity dispersion
    """
    #create outflow mask
    flow_mask = (stat_results==1)

    #de-redshift the data first!!!
    systemic_mean = outflow_results[1,:][flow_mask]/(1+z)
    flow_mean = outflow_results[4,:][flow_mask]/(1+z)
    flow_sigma = outflow_results[3,:][flow_mask]/(1+z)

    #find the velocity difference
    #doing c*(lam_gal-lam_out)/lam_gal
    vel_diff = 299792.458*(systemic_mean-flow_mean)/systemic_mean

    #seperate them into out and in flows
    vel_diff_out = vel_diff[vel_diff > 20]
    vel_diff_mid = vel_diff[(vel_diff<=20)&(vel_diff>-20)]
    vel_diff_in = vel_diff[vel_diff <= -20]

    sigma_out = flow_sigma[vel_diff > 20]*299792.458/systemic_mean[vel_diff > 20]
    sigma_mid = flow_sigma[(vel_diff<=20)&(vel_diff>-20)]*299792.458/systemic_mean[(vel_diff<=20)&(vel_diff>-20)]
    sigma_in = flow_sigma[vel_diff <= -20]*299792.458/systemic_mean[vel_diff <= -20]

    #plot it
    plt.rcParams.update(pf.get_rc_params())
    plt.figure()
    #plt.errorbar(sfr_out, v_out, yerr=vel_err_out, marker='o', lw=0, label='Outflow', alpha=0.2)
    #plt.errorbar(sfr_in, abs(v_in), yerr=vel_err_in, marker='o', lw=0, label='Inflow', alpha=0.2)
    plt.plot(vel_diff_out, sigma_out, marker='o', lw=0, label='Blue shifted flow', alpha=0.4)
    plt.plot(vel_diff_in, sigma_in, marker='o', lw=0, label='Red shifted flow', alpha=0.4)
    plt.plot(vel_diff_mid, sigma_mid, marker='o', lw=0, label='Centred flow', alpha=0.4)
    #plt.xscale('log')
    plt.legend(frameon=False, fontsize='small', loc='lower left')
    plt.ylabel('Flow Dispersion ($\sigma_{out}$) [km s$^{-1}$]')
    plt.xlabel('Velocity Difference (v$_{sys}$-v$_{flow}$) [km s$^{-1}$]')
    plt.show()



def plot_sigma_sfr(sfr, outflow_results, outflow_error, stat_results, z):
    """
    Plots the outflow velocity dispersion against the star formation rate or
    against SFR surface density, depending on the input

    Parameters
    ----------
    sfr : :obj:'~numpy.ndarray'
        star formation rate OR star formation rate surface density value for each
        spaxel in the galaxy

    outflow_results : :obj:'~numpy.ndarray'
        array of outflow results from KOFFEE.  Used to calculate the outflow
        velocity. Should be (7, stat_results.shape)

    outflow_error : :obj:'~numpy.ndarray'
        array of the outflow result errors from KOFFEE.

    stat_results : :obj:'~numpy.ndarray'
        array of statistical results from KOFFEE

    z : float
        redshift

    Returns
    -------
    A one panel figure of the SFR against the outflow velocity dispersion
    """
    #create outflow mask
    flow_mask = (stat_results==1)

    #de-redshift the data first!!!
    systemic_mean = outflow_results[1,:][flow_mask]/(1+z)
    flow_mean = outflow_results[4,:][flow_mask]/(1+z)
    flow_sigma = outflow_results[3,:][flow_mask]/(1+z)

    #find the velocity difference
    #doing c*(lam_gal-lam_out)/lam_gal
    vel_diff = 299792.458*(systemic_mean-flow_mean)/systemic_mean

    #seperate them into out and in flows
    vel_diff_out = vel_diff[vel_diff > 20]
    vel_diff_mid = vel_diff[(vel_diff<=20)&(vel_diff>-20)]
    vel_diff_in = vel_diff[vel_diff <= -20]

    sigma_out = flow_sigma[vel_diff > 20]*299792.458/systemic_mean[vel_diff > 20]
    sigma_mid = flow_sigma[(vel_diff<=20)&(vel_diff>-20)]*299792.458/systemic_mean[(vel_diff<=20)&(vel_diff>-20)]
    sigma_in = flow_sigma[vel_diff <= -20]*299792.458/systemic_mean[vel_diff <= -20]

    sfr_out = sfr[flow_mask][vel_diff > 20]
    sfr_mid = sfr[flow_mask][(vel_diff<=20)&(vel_diff>-20)]
    sfr_in = sfr[flow_mask][vel_diff <= -20]

    #plot it
    plt.rcParams.update(pf.get_rc_params())
    plt.figure()
    #plt.errorbar(sfr_out, v_out, yerr=vel_err_out, marker='o', lw=0, label='Outflow', alpha=0.2)
    #plt.errorbar(sfr_in, abs(v_in), yerr=vel_err_in, marker='o', lw=0, label='Inflow', alpha=0.2)
    plt.plot(sfr_out, sigma_out, marker='o', lw=0, label='Blue shifted flow', alpha=0.4)
    plt.plot(sfr_in, sigma_in, marker='o', lw=0, label='Red shifted flow', alpha=0.4)
    plt.plot(sfr_mid, sigma_mid, marker='o', lw=0, label='Centred flow', alpha=0.4)
    plt.xscale('log')
    plt.legend(frameon=False, fontsize='small', loc='lower left')
    plt.ylabel('Flow Dispersion ($\sigma_{out}$) [km s$^{-1}$]')
    plt.xlabel('Log Star Formation Rate Surface Density (M$_\odot$ yr$^{-1}$ kpc$^{-2}$)')
    plt.show()


def plot_vel_diff_sfr(sfr, outflow_results, outflow_error, stat_results, z):
    """
    Plots the velocity difference between systemic and flow components against
    the star formation rate OR star formation rate surface density

    Parameters
    ----------
    sfr : :obj:'~numpy.ndarray'
        star formation rate OR star formation rate surface density value for each
        spaxel in the galaxy

    outflow_results : :obj:'~numpy.ndarray'
        array of outflow results from KOFFEE.  Used to calculate the outflow
        velocity. Should be (7, stat_results.shape)

    outflow_error : :obj:'~numpy.ndarray'
        array of the outflow result errors from KOFFEE.

    stat_results : :obj:'~numpy.ndarray'
        array of statistical results from KOFFEE

    z : float
        redshift

    Returns
    -------
    A one panel figure of the SFR against the outflow velocity dispersion
    """
    #create outflow mask
    flow_mask = (stat_results==1)

    #de-redshift the data first!!!
    systemic_mean = outflow_results[1,:][flow_mask]/(1+z)
    flow_mean = outflow_results[4,:][flow_mask]/(1+z)

    #find the velocity difference
    #doing c*(lam_gal-lam_out)/lam_gal
    vel_diff = 299792.458*(systemic_mean-flow_mean)/systemic_mean

    #seperate them into out and in flows
    vel_diff_out = vel_diff[vel_diff > 20]
    vel_diff_mid = vel_diff[(vel_diff<=20)&(vel_diff>-20)]
    vel_diff_in = vel_diff[vel_diff <= -20]

    sfr_out = sfr[flow_mask][vel_diff > 20]
    sfr_mid = sfr[flow_mask][(vel_diff<=20)&(vel_diff>-20)]
    sfr_in = sfr[flow_mask][vel_diff <= -20]

    #plot it
    plt.rcParams.update(pf.get_rc_params())
    plt.figure()
    #plt.errorbar(sfr_out, v_out, yerr=vel_err_out, marker='o', lw=0, label='Outflow', alpha=0.2)
    #plt.errorbar(sfr_in, abs(v_in), yerr=vel_err_in, marker='o', lw=0, label='Inflow', alpha=0.2)
    plt.plot(sfr_out, vel_diff_out, marker='o', lw=0, label='Blue shifted flow', alpha=0.4)
    plt.plot(sfr_in, vel_diff_in, marker='o', lw=0, label='Red shifted flow', alpha=0.4)
    plt.plot(sfr_mid, vel_diff_mid, marker='o', lw=0, label='Centred flow', alpha=0.4)
    plt.xscale('log')
    plt.legend(frameon=False, fontsize='small', loc='lower left')
    plt.ylabel('Velocity Difference ($v_{sys}-v_{broad}$) [km s$^{-1}$]')
    plt.xlabel('Log Star Formation Rate Surface Density (M$_\odot$ yr$^{-1}$ kpc$^{-2}$)')
    plt.show()


def plot_vdiff_amp_ratio(outflow_results, outflow_error, stat_results, z):
    """
    Plots the velocity difference between systemic and flow components against
    the amplitude ratio between broad and narrow gaussians

    Parameters
    ----------
    outflow_results : :obj:'~numpy.ndarray'
        array of outflow results from KOFFEE.  Used to calculate the outflow
        velocity. Should be (7, stat_results.shape)

    outflow_error : :obj:'~numpy.ndarray'
        array of the outflow result errors from KOFFEE.

    stat_results : :obj:'~numpy.ndarray'
        array of statistical results from KOFFEE

    z : float
        redshift

    Returns
    -------
    A one panel figure of the outflow velocity offset against the broad-to-narrow
    amplitude ratio
    """
    #create outflow mask
    flow_mask = (stat_results==1)

    #de-redshift the data first!!!
    systemic_amp = outflow_results[2,:][flow_mask]/(1+z)
    flow_amp = outflow_results[5,:][flow_mask]/(1+z)
    systemic_mean = outflow_results[1,:][flow_mask]/(1+z)
    flow_mean = outflow_results[4,:][flow_mask]/(1+z)

    #find the velocity difference
    #doing c*(lam_gal-lam_out)/lam_gal
    vel_diff = 299792.458*(systemic_mean-flow_mean)/systemic_mean

    #find the amplitude ratio
    amp_ratio = flow_amp/systemic_amp

    #seperate them into out and in flows
    amp_ratio_out = amp_ratio[vel_diff > 20]
    amp_ratio_mid = amp_ratio[(vel_diff<=20)&(vel_diff>-20)]
    amp_ratio_in = amp_ratio[vel_diff <= -20]

    vel_diff_out = vel_diff[vel_diff > 20]
    vel_diff_mid = vel_diff[(vel_diff<=20)&(vel_diff>-20)]
    vel_diff_in = vel_diff[vel_diff <= -20]

    #plot it
    plt.rcParams.update(pf.get_rc_params())
    plt.figure()
    #plt.errorbar(sfr_out, v_out, yerr=vel_err_out, marker='o', lw=0, label='Outflow', alpha=0.2)
    #plt.errorbar(sfr_in, abs(v_in), yerr=vel_err_in, marker='o', lw=0, label='Inflow', alpha=0.2)
    plt.plot(vel_diff_out, amp_ratio_out, marker='o', lw=0, label='Blue shifted flow', alpha=0.4)
    plt.plot(vel_diff_in, amp_ratio_in, marker='o', lw=0, label='Red shifted flow', alpha=0.4)
    plt.plot(vel_diff_mid, amp_ratio_mid, marker='o', lw=0, label='Centred flow', alpha=0.4)
    #plt.xscale('log')
    plt.legend(frameon=False, fontsize='small', loc='lower left')
    plt.ylabel('Amplitude Ratio (broad/systemic)')
    plt.xlabel('Velocity Difference (km/s)')
    plt.show()
