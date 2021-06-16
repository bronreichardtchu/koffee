"""
NAME:
	paper_I_plots.py

AUTHOR:
	Bronwyn Reichardt Chu
	Swinburne
	2020

EMAIL:
	<breichardtchu@swin.edu.au>

PURPOSE:
	To make plots of results from Paper I
	Written on MacOS Mojave 10.14.5, with Python 3.7

FUNCTIONS INCLUDED:
    plot_compare_fits                                   (Figure 1)
    plot_hist_out_vel_flux                              (Figure 2)
    plot_sfr_vout                                       (Figure 3 + 4)
    plot_sfr_vout_correlation_with_binning_from_file    (Figure 5)
    plot_sfr_mlf_flux                                   (Figure 6)
    maps_of_IRAS08                                      (Figure 7)
    plot_mlf_model_rad_singlepanel                      (Figure 8)

MODIFICATION HISTORY:
		v.1.0 - first created October 2020

"""
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import cmasher as cmr

from scipy.optimize import curve_fit
from scipy.signal import argrelextrema

from astropy.wcs import WCS
from astropy import units as u
from astropy.cosmology import WMAP9 as cosmo
from astropy.io import fits

from . import prepare_cubes as pc
from . import plotting_functions as pf
from . import koffee_fitting_functions as kff
from . import calculate_outflow_velocity as calc_outvel
from . import calculate_star_formation_rate as calc_sfr
from . import calculate_mass_loading_factor as calc_mlf
from . import calculate_energy_loading_factor as calc_elf
from . import calculate_equivalent_width as calc_ew
from . import brons_display_pixels_kcwi as bdpk
from . import koffee

import importlib
importlib.reload(pf)
importlib.reload(calc_mlf)
#importlib.reload(bdpk)



#===============================================================================
# PLOTTING FUNCTIONS - for paper
#===============================================================================
#Figure 1
def plot_compare_fits(lamdas, data, spaxels, z):
    """
    Plots the normalised single and double gaussian fits for the OIII 5007 line
    using a list of spaxels.  (Used spaxels [[19, 7], [26, 8], [35, 10]] for the
    paper.)

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
    fig, ax = plt.subplots(nrows=2, ncols=spaxel_num, sharex=True, sharey=True, figsize=(spaxel_num*3, 4), constrained_layout=True)
    #plt.subplots_adjust(wspace=0, hspace=0, left=0.08, right=0.99, top=0.95)

    #get colours from cmasher
    colours = cmr.take_cmap_colors('cmr.gem', 3, cmap_range=(0.25, 0.85), return_fmt='hex')

    #iterate through the spaxels
    for i in range(spaxel_num):
        #mask the data to get the flux
        flux = data[OIII_mask, spaxels[i][0], spaxels[i][1]]

        #fit data with single gaussian:obj:'~numpy.ndarray'
        gmodel1, pars1 = kff.gaussian1_const(lam_OIII, flux)
        bestfit1 = kff.fitter(gmodel1, pars1, lam_OIII, flux, verbose=False)

        #fit the data with double gaussian
        gmodel2, pars2 = kff.gaussian2_const(lam_OIII, flux)
        bestfit2 = kff.fitter(gmodel2, pars2, lam_OIII, flux, verbose=False)

        #find the significance level using the BIC difference
        BIC_diff = bestfit1.bic - bestfit2.bic
        print(BIC_diff)
        if 10 < BIC_diff <= 30:
            significance_level = 'weakly likely\n10 < $\delta_{BIC}$ < 30'
        elif 30 < BIC_diff <= 50:
            significance_level = 'moderately likely\n30 < $\delta_{BIC}$ < 50'
        elif 50 < BIC_diff:
            significance_level = 'strongly likely\n$\delta_{BIC}$ < 50'
        else:
            significance_level = str(BIC_diff)

        #get the value to normalise by
        max_value = np.nanmax(flux)

        #create a plotting mask
        plotting_mask = (lam_OIII>lam_OIII[25]) & (lam_OIII<lam_OIII[-25])
        plotting_mask2 = (fine_sampling>lam_OIII[25]) & (fine_sampling<lam_OIII[-25])

        #plot the fits on the figure
        ax[0,i].step(lam_OIII[plotting_mask], flux[plotting_mask]/max_value, where='mid', c='k')
        ax[0,i].plot(fine_sampling[plotting_mask2], bestfit1.eval(x=fine_sampling[plotting_mask2])/max_value, c=colours[0], ls='--')#, lw=1)

        ax[1,i].step(lam_OIII[plotting_mask], flux[plotting_mask]/max_value, where='mid', c='k', label='Data')
        ax[1,i].plot(fine_sampling[plotting_mask2], bestfit2.components[0].eval(bestfit2.params, x=fine_sampling[plotting_mask2])/max_value, c=colours[1], label='Narrow component')
        ax[1,i].plot(fine_sampling[plotting_mask2], bestfit2.components[1].eval(bestfit2.params, x=fine_sampling[plotting_mask2])/max_value, c=colours[2], label='Broad component')
        ax[1,i].plot(fine_sampling[plotting_mask2], bestfit2.eval(x=fine_sampling[plotting_mask2])/max_value, c=colours[0], ls='--', label='Bestfit')

        ax[1,i].set_xlabel('Wavelength ($\AA$)')
        ax[0,i].set_title(significance_level, fontsize='medium')

        if i == 0:
            ax[1,i].legend(fontsize='x-small', frameon=False, loc='upper left')
            ax[0,i].set_ylabel('Normalised Flux')
            ax[1,i].set_ylabel('Normalised Flux')
            ax[0,i].set_ylim(-0.05, 0.75)
            ax[1,i].set_ylim(-0.05, 0.75)

    plt.show()


#Figure 2
def plot_hist_out_vel_flux(outflow_results, outflow_error, outflow_results_unfixed, outflow_error_unfixed, statistical_results, lamdas, data, spaxel, z, plot_fit_parameters=False):
    """
    Plots a three panel graph of two histograms of the outflow velocity and the flux
    ratio for [OIII] for before and after koffee's selection criteria for outflows
    are applied; and an example of a fitted spaxel

    Parameters
    ----------
    outflow_results : :obj:'~numpy.ndarray'
        array of outflow results from KOFFEE for OIII line.  Used to calculate
        the outflow velocity.  Should be (7, statistical_results.shape)

    outflow_err : :obj:'~numpy.ndarray'
        array of the outflow result errors from KOFFEE

    outflow_results_unfixed : :obj:'~numpy.ndarray'
        array of outflow results from KOFFEE for Hbeta line.  Used to calculate
        the Sigma SFR.  Should be (7, statistical_results.shape)

    outflow_error_unfixed : :obj:'~numpy.ndarray'
        array of the outflow result errors from KOFFEE for Hbeta line

    statistical_results : :obj:'~numpy.ndarray'
        array of statistical results from KOFFEE.

    lamdas : :obj:'~numpy.ndarray'
        wavelength vector

    data : :obj:'~numpy.ndarray'
        3D data cube, with shape [len(lamdas), :, :]

    spaxel : list
        the spaxel of the emission line which will be fit for the third panel,
        this is usually chosen to be one of the spaxels for which not doing
        KOFFEE's checks means the fit is an unlikely one. e.g. [33, 5]

    z : float
        redshift

    plot_fit_parameters : boolean
        When True, plots histograms of the velocity difference and [OIII] amplitude
        ratio.  When False, plots the results for the outflow velocity and [OIII]
        broad-to-narrow flux ratio.  Default is False.

    Returns
    -------
    A three panel graph, two histograms and an example emission line fit
    """
    #make a stat_res array for the unfixed spaxels - this is just all the spaxels
    #that don't have NaN values in the results (they have S/N>20)
    statistical_results_unfixed = np.zeros_like(statistical_results)
    statistical_results_unfixed[~np.isnan(outflow_results_unfixed[0,:,:])] = 1

    #calculate the outflow velocity
    vel_disp, vel_disp_err, vel_diff, vel_diff_err, vel_out, vel_out_err = calc_outvel.calc_outflow_vel(outflow_results, outflow_error, statistical_results, z)

    vel_disp_unfixed, vel_disp_err_unfixed, vel_diff_unfixed, vel_diff_err_unfixed, vel_out_unfixed, vel_out_err_unfixed = calc_outvel.calc_outflow_vel(outflow_results_unfixed, outflow_error_unfixed, statistical_results_unfixed, z)

    #calculate the fluxes for systematic and flow gaussians
    systemic_flux, systemic_flux_err, outflow_flux, outflow_flux_err = calc_sfr.calc_flux_from_koffee(outflow_results, outflow_error, statistical_results, z, outflow=True)

    systemic_flux_unfixed, systemic_flux_err_unfixed, outflow_flux_unfixed, outflow_flux_err_unfixed = calc_sfr.calc_flux_from_koffee(outflow_results_unfixed, outflow_error_unfixed, statistical_results_unfixed, z, outflow=True)

    if plot_fit_parameters == True:
        #calculate the amplitude ratio
        amp_ratio = np.log10(outflow_results[2,:,:]/outflow_results[5,:,:])
        amp_ratio_unfixed = np.log10(outflow_results_unfixed[2,:,:]/outflow_results_unfixed[5,:,:])

        #calculate the medians
        amp_ratio_median = np.nanmedian(amp_ratio)
        amp_ratio_unfixed_median = np.nanmedian(amp_ratio_unfixed)

        vel_diff_median = np.nanmedian(vel_diff)
        vel_diff_unfixed_median = np.nanmedian(vel_diff_unfixed)

    elif plot_fit_parameters == False:
        #calculate flux ratios
        flux_ratio = np.log10(outflow_flux/systemic_flux)
        flux_ratio_unfixed = np.log10(outflow_flux_unfixed/systemic_flux_unfixed)

        #calculate the medians
        flux_ratio_median = np.nanmedian(flux_ratio)
        flux_ratio_unfixed_median = np.nanmedian(flux_ratio_unfixed)

        vel_out_median = np.nanmedian(vel_out)
        vel_out_unfixed_median = np.nanmedian(vel_out_unfixed)

    if plot_fit_parameters == True:
        print('All fits vel_diff median:', vel_diff_median)
        print('KOFFEE fits vel_diff median:', vel_diff_unfixed_median)
        print('All fits amp_ratio median:', amp_ratio_median)
        print('KOFFEE fits amp_ratio median:', amp_ratio_unfixed_median)

    elif plot_fit_parameters == False:
        print('All fits vel_out median:', vel_out_median)
        print('KOFFEE fits vel_out median:', vel_out_unfixed_median)
        print('All fits flux_ratio median:', flux_ratio_median)
        print('KOFFEE fits flux_ratio median:', flux_ratio_unfixed_median)

    #make a mask for the emission line
    OIII_mask = (lamdas>5008.24*(1+z)-20.0) & (lamdas<5008.24*(1+z)+20.0)

    #mask the wavelength
    lam_OIII = lamdas[OIII_mask]

    #create the fine sampling array
    fine_sampling = np.linspace(min(lam_OIII), max(lam_OIII), 1000)

    #mask the data to get the flux
    flux = data[OIII_mask, spaxel[0], spaxel[1]]

    #fit the data with double gaussian
    gmodel2, pars2 = kff.gaussian2_const(lam_OIII, flux)
    bestfit2 = kff.fitter(gmodel2, pars2, lam_OIII, flux, verbose=False)

    #get the value to normalise by
    max_value = np.nanmax(flux)

    #create a plotting mask
    plotting_mask = (lam_OIII>lam_OIII[30]) & (lam_OIII<lam_OIII[-25])
    plotting_mask2 = (fine_sampling>lam_OIII[30]) & (fine_sampling<lam_OIII[-25])

    #create the bins for the histograms
    if plot_fit_parameters == True:
        bins_panel1 = np.linspace(np.nanmin(np.concatenate((vel_diff_unfixed[statistical_results_unfixed>0], vel_diff[statistical_results>0]))), np.nanmax(np.concatenate((vel_diff_unfixed[statistical_results_unfixed>0], vel_diff[statistical_results>0]))), 20)
        bins_panel2 = np.linspace(np.nanmin(np.concatenate((amp_ratio_unfixed[statistical_results_unfixed>0], amp_ratio[statistical_results>0]))), np.nanmax(np.concatenate((amp_ratio_unfixed[statistical_results_unfixed>0], amp_ratio[statistical_results>0]))), 20)

    elif plot_fit_parameters == False:
        bins_panel1 = np.linspace(np.nanmin(np.concatenate((vel_out_unfixed[statistical_results_unfixed>0], vel_out[statistical_results>0]))), np.nanmax(np.concatenate((vel_out_unfixed[statistical_results_unfixed>0], vel_out[statistical_results>0]))), 20)
        bins_panel2 = np.linspace(np.nanmin(np.concatenate((flux_ratio_unfixed[statistical_results_unfixed>0], flux_ratio[statistical_results>0]))), np.nanmax(np.concatenate((flux_ratio_unfixed[statistical_results_unfixed>0], flux_ratio[statistical_results>0]))), 20)

    #plot the histograms
    plt.rcParams.update(pf.get_rc_params())
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(10,4), constrained_layout=True)

    #get colours from cmasher
    colours = cmr.take_cmap_colors('cmr.gem', 3, cmap_range=(0.25, 0.85), return_fmt='hex')

    if plot_fit_parameters == True:
        ax[0].hist(vel_diff_unfixed[statistical_results_unfixed>0], bins=bins_panel1, alpha=0.5, color='tab:blue', edgecolor='tab:blue', label='All spaxels S/N > 20\nNo quality tests')

        ax[0].hist(vel_diff[statistical_results>0], bins=bins_panel1, alpha=0.5, color='tab:red', edgecolor='tab:red', label='KOFFEE fits')

        ax[0].set_ylim(0,50)
        ax[0].set_xlabel('Velocity Difference $v_{narrow}-v_{broad}$ [km s$^{-1}$]')

    elif plot_fit_parameters == False:
        ax[0].hist(vel_out_unfixed[statistical_results_unfixed>0], bins=bins_panel1, alpha=0.5, color='tab:blue', edgecolor='tab:blue', label='All spaxels S/N > 20\nNo quality tests')

        ax[0].hist(vel_out[statistical_results>0], bins=bins_panel1, alpha=0.5, color='tab:red', edgecolor='tab:red', label='KOFFEE fits')

        ax[0].set_ylim(0,40)
        ax[0].set_xlabel('Maximum Outflow Velocity [km s$^{-1}$]')

    ax[0].legend(fontsize='x-small', frameon=False)
    ax[0].set_ylabel('$N_{spaxels}$')


    if plot_fit_parameters == True:
        ax[1].hist(amp_ratio_unfixed[statistical_results_unfixed>0], bins=bins_panel2, alpha=0.5, label='All spaxels S/N > 20', color='tab:blue', edgecolor='tab:blue')

        ax[1].hist(amp_ratio[statistical_results>0], bins=bins_panel2, alpha=0.5, label='KOFFEE fits', color='tab:red', edgecolor='tab:red')

        ax[1].set_ylim(0,50)
        ax[1].set_xlabel('[OIII] Log($A_{broad}$/$A_{narrow}$)')

    elif plot_fit_parameters == False:
        ax[1].hist(flux_ratio_unfixed[statistical_results_unfixed>0], bins=bins_panel2, alpha=0.5, label='All spaxels S/N > 20', color='tab:blue', edgecolor='tab:blue')

        ax[1].hist(flux_ratio[statistical_results>0], bins=bins_panel2, alpha=0.5, label='KOFFEE fits', color='tab:red', edgecolor='tab:red')

        ax[1].set_ylim(0,50)
        ax[1].set_xlabel('[OIII] Log(F$_{broad}$/F$_{narrow}$)')


    #plot the example of a bad fit
    ax[2].step(lam_OIII[plotting_mask], flux[plotting_mask]/max_value, where='mid', c='k', label='Data')
    ax[2].plot(fine_sampling[plotting_mask2], bestfit2.components[0].eval(bestfit2.params, x=fine_sampling[plotting_mask2])/max_value, c=colours[1], label='Narrow component')
    ax[2].plot(fine_sampling[plotting_mask2], bestfit2.components[1].eval(bestfit2.params, x=fine_sampling[plotting_mask2])/max_value, c=colours[2], label='Broad component')
    ax[2].plot(fine_sampling[plotting_mask2], bestfit2.eval(x=fine_sampling[plotting_mask2])/max_value, c=colours[0], ls='--', label='Bestfit')

    ax[2].xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))

    ax[2].set_xlabel('Wavelength ($\AA$)')
    ax[2].legend(fontsize='x-small', frameon=False, loc='upper left')
    ax[2].set_ylabel('Normalised Flux')
    #ax[0,i].set_ylim(-0.05, 0.75)

    #plt.tight_layout()
    plt.show()




#Figure 3 and 4
def plot_sfr_vout(OIII_outflow_results, OIII_outflow_error, hbeta_outflow_results, hbeta_outflow_error, hbeta_no_outflow_results, hbeta_no_outflow_error, BIC_outflow, BIC_no_outflow, statistical_results, z, radius, header, weighted_average=False, plot_medians=True, plot_data_fits=False, xlim_vals=None):
    """
    Plots the SFR surface density against the outflow velocity, with Sigma_SFR
    calculated using only the narrow component.

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

    weighted_average : boolean
        whether or not to take a weighted average using the errors (Default=True)

    plot_data_fits : boolean
        whether to plot the fit to the data points, and the fit to the data
        medians in red on top of the plot (default is False)

    xlim_vals : list of floats
        the xlimits for the plotted data e.g. [0.005, 13]

    Returns
    -------
    A graph of outflow velocity against the SFR surface density in three panels
    with different data selections

    """
    #calculate the outflow velocity
    vel_disp, vel_disp_err, vel_diff, vel_diff_err, vel_out, vel_out_err = calc_outvel.calc_outflow_vel(OIII_outflow_results, OIII_outflow_error, statistical_results, z)

    #calculate the sfr surface density - using just the systemic line, and including the flux line
    #don't include extinction since this was included in the continuum subtraction using ppxf
    sfr, sfr_err, total_sfr, sfr_surface_density, sfr_surface_density_err = calc_sfr.calc_sfr_koffee(hbeta_outflow_results, hbeta_outflow_error, hbeta_no_outflow_results, hbeta_no_outflow_error, statistical_results, z, header, include_extinction=False, include_outflow=False)

    #get the sfr for the outflow spaxels
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

    print(sig_sfr[physical_mask])
    print(sig_sfr[physical_mask].shape)

    #strong BIC and physical limits mask
    #clean_mask = (radius < 6.1) & (vel_disp > 51) & (BIC_diff < -50)

    #make sure none of the errors are nan values
    vel_out_err[np.where(np.isnan(vel_out_err)==True)] = np.nanmedian(vel_out_err)

    #do the calculations for all the bins
    num_bins = 5
    min_bin = None #-0.05
    max_bin = None #0.6

    if weighted_average == False:
        logspace_all, bin_center_all, v_out_bin_medians_all, v_out_bin_lower_q_all, v_out_bin_upper_q_all, v_out_bin_stdev_all = pf.binned_median_quantile_log(sig_sfr, vel_out, num_bins=num_bins, weights=None, min_bin=min_bin, max_bin=max_bin)
        logspace_physical, bin_center_physical, v_out_bin_medians_physical, v_out_bin_lower_q_physical, v_out_bin_upper_q_physical, v_out_bin_stdev_physical = pf.binned_median_quantile_log(sig_sfr[physical_mask], vel_out[physical_mask], num_bins=num_bins, weights=None, min_bin=min_bin, max_bin=max_bin)
        logspace_strong, bin_center_strong, v_out_bin_medians_strong, v_out_bin_lower_q_strong, v_out_bin_upper_q_strong, v_out_bin_stdev_strong = pf.binned_median_quantile_log(sig_sfr[BIC_diff_strong], vel_out[BIC_diff_strong], num_bins=num_bins, weights=None, min_bin=min_bin, max_bin=max_bin)

    elif weighted_average == True:
        logspace_all, bin_center_all, v_out_bin_medians_all, v_out_bin_lower_q_all, v_out_bin_upper_q_all, v_out_bin_stdev_all = pf.binned_median_quantile_log(sig_sfr, vel_out, num_bins=num_bins, weights=[vel_out_err], min_bin=min_bin, max_bin=max_bin)
        logspace_physical, bin_center_physical, v_out_bin_medians_physical, v_out_bin_lower_q_physical, v_out_bin_upper_q_physical, v_out_bin_stdev_physical = pf.binned_median_quantile_log(sig_sfr[physical_mask], vel_out[physical_mask], num_bins=num_bins, weights=[vel_out_err], min_bin=min_bin, max_bin=max_bin)
        logspace_strong, bin_center_strong, v_out_bin_medians_strong, v_out_bin_lower_q_strong, v_out_bin_upper_q_strong, v_out_bin_stdev_strong = pf.binned_median_quantile_log(sig_sfr[BIC_diff_strong], vel_out[BIC_diff_strong], num_bins=num_bins, weights=[vel_out_err], min_bin=min_bin, max_bin=max_bin)

    #calculate the r value for the median values
    r_vel_out_med_all, p_value_v_out_med_all = pf.pearson_correlation(np.log10(bin_center_all), np.log10(v_out_bin_medians_all))
    r_vel_out_med_physical, p_value_v_out_med_physical = pf.pearson_correlation(np.log10(bin_center_physical), np.log10(v_out_bin_medians_physical))
    r_vel_out_med_strong, p_value_v_out_med_strong = pf.pearson_correlation(np.log10(bin_center_strong), np.log10(v_out_bin_medians_strong))

    #calculate the r value for all the values
    r_vel_out_all, p_value_v_out_all = pf.pearson_correlation(np.log10(sig_sfr), np.log10(vel_out))
    r_vel_out_physical, p_value_v_out_physical = pf.pearson_correlation(np.log10(sig_sfr[physical_mask]), np.log10(vel_out[physical_mask]))
    r_vel_out_strong, p_value_v_out_strong = pf.pearson_correlation(np.log10(sig_sfr[BIC_diff_strong]), np.log10(vel_out[BIC_diff_strong]))

    #create vectors to plot the literature trends
    if xlim_vals:
        sfr_surface_density_chen, v_out_chen = pf.chen_et_al_2010(xlim_vals[0], xlim_vals[1], scale_factor=np.nanmedian(vel_out[BIC_diff_strong])/(np.nanmedian(sig_sfr[BIC_diff_strong])**0.1))
        sfr_surface_density_murray, v_out_murray = pf.murray_et_al_2011(xlim_vals[0], xlim_vals[1], scale_factor=np.nanmedian(vel_out[BIC_diff_strong])/(np.nanmedian(sig_sfr[BIC_diff_strong])**2))
    else:
        sfr_surface_density_chen, v_out_chen = pf.chen_et_al_2010(sig_sfr.min(), sig_sfr.max(), scale_factor=np.nanmedian(vel_out[BIC_diff_strong])/(np.nanmedian(sig_sfr[BIC_diff_strong])**0.1))
        sfr_surface_density_murray, v_out_murray = pf.murray_et_al_2011(sig_sfr.min(), sig_sfr.max(), scale_factor=np.nanmedian(vel_out[BIC_diff_strong])/(np.nanmedian(sig_sfr[BIC_diff_strong])**2))

    #fit our own trends
    popt_vout_all_medians, pcov_vout_all_medians = curve_fit(pf.fitting_function, bin_center_all, v_out_bin_medians_all)
    popt_vout_physical_medians, pcov_vout_physical_medians = curve_fit(pf.fitting_function, bin_center_physical, v_out_bin_medians_physical)
    popt_vout_strong_medians, pcov_vout_strong_medians = curve_fit(pf.fitting_function, bin_center_strong, v_out_bin_medians_strong)

    popt_vout_all, pcov_vout_all = curve_fit(pf.fitting_function, sig_sfr, vel_out)
    popt_vout_physical, pcov_vout_physical = curve_fit(pf.fitting_function, sig_sfr[physical_mask], vel_out[physical_mask])
    popt_vout_strong, pcov_vout_strong = curve_fit(pf.fitting_function, sig_sfr[BIC_diff_strong], vel_out[BIC_diff_strong])

    print(popt_vout_all, pcov_vout_all)
    print([popt_vout_all_medians[0], np.sqrt(np.diag(pcov_vout_all_medians))[0], popt_vout_all_medians[1], np.sqrt(np.diag(pcov_vout_all_medians))[1]])

    sfr_linspace = np.linspace(sig_sfr.min(), sig_sfr.max()+4, num=1000)

    #ax[0].plot(sfr_linspace, fitting_function(sfr_linspace, *popt_vout), 'r-', label='Fit: $v_{out}=%5.0f$ $\Sigma_{SFR}^{%5.2f}$' % tuple(popt_vout))

    #calculate the RMS
    rms_all = np.sqrt(np.mean(vel_out))
    rms_physical = np.sqrt(np.mean(vel_out[physical_mask]))
    rms_strong = np.sqrt(np.mean(vel_out[BIC_diff_strong]))

    #print average numbers for the different panels
    print('Number of spaxels in the first panel', vel_out.shape)
    print('All spaxels median v_out:', np.nanmedian(vel_out))
    print('All spaxels standard deviation v_out:', np.nanstd(vel_out))
    print('All spaxels median sigma_sfr:', np.nanmedian(sig_sfr))
    print('All spaxels standard deviation sigma_sfr:', np.nanstd(sig_sfr))
    print('All spaxels RMS:', rms_all)
    print('All spaxels bins lower quartiles:', v_out_bin_lower_q_all)
    print('All spaxels bins upper quartiles:', v_out_bin_upper_q_all)
    print('All spaxels p-value:', p_value_v_out_all)
    print('All spaxels medians p-value:', p_value_v_out_med_all)
    print('All spaxels logspace:', logspace_all)
    print('')

    print('All spaxels best fit coefficients:', popt_vout_all)
    print('All spaxels best fit errors', np.sqrt(np.diag(pcov_vout_all)))
    print('')

    print('Number of spaxels with broad sigmas at the instrument dispersion:', vel_out[vel_disp<=51].shape)
    print('')
    print('Number of spaxels beyond R_90:', vel_out[radius>6.1].shape)
    print('')
    print('Number of spaxels in the middle panel:', vel_out[physical_mask].shape)
    print('')

    print('Physical spaxels median v_out:', np.nanmedian(vel_out[physical_mask]))
    print('Physical spaxels standard deviation v_out:', np.nanstd(vel_out[physical_mask]))
    print('Physical spaxels median sigma_sfr:', np.nanmedian(sig_sfr[physical_mask]))
    print('Physical spaxels standard deviation sigma_sfr:', np.nanstd(sig_sfr[physical_mask]))
    print('Physical spaxels RMS:', rms_physical)
    print('Physical spaxels p-value:', p_value_v_out_physical)
    print('Physical spaxels medians p-value:', p_value_v_out_med_physical)
    print('')
    print('Physical spaxels best fit coefficients:', popt_vout_physical)
    print('Physical spaxels best fit errors', np.sqrt(np.diag(pcov_vout_physical)))
    print('')

    print('Number of spaxels with strong BIC differences:', vel_out[BIC_diff_strong].shape)
    print('')

    print('Clean spaxels median v_out:', np.nanmedian(vel_out[BIC_diff_strong]))
    print('Clean spaxels standard deviation v_out:', np.nanstd(vel_out[BIC_diff_strong]))
    print('Clean spaxels median sigma_sfr:', np.nanmedian(sig_sfr[BIC_diff_strong]))
    print('Clean spaxels standard deviation sigma_sfr:', np.nanstd(sig_sfr[BIC_diff_strong]))
    print('Clean spaxels RMS:', rms_strong)
    print('Clean spaxels p-value:', p_value_v_out_strong)
    print('Clean spaxels medians p-value:', p_value_v_out_med_strong)
    print('')
    print('Clean spaxels best fit coefficients:', popt_vout_strong)
    print('Clean spaxels best fit errors', np.sqrt(np.diag(pcov_vout_strong)))
    print('')

    #-------
    #plot it
    #-------
    plt.rcParams.update(pf.get_rc_params())
    fig, ax = plt.subplots(nrows=1, ncols=3, sharey=True, sharex=True, figsize=(10,4), constrained_layout=True)

    #get colours from cmasher
    colours = cmr.take_cmap_colors('cmr.gem', 3, cmap_range=(0.25, 0.85), return_fmt='hex')

    #plot all points
    #if plot_medians == True:
        #ax[0].fill_between(bin_center_all, v_out_bin_lower_q_all, v_out_bin_upper_q_all, color=colours[0], alpha=0.3)
        #ax[0].fill_between(bin_center_all, v_out_bin_medians_all-v_out_bin_stdev_all, v_out_bin_medians_all+v_out_bin_stdev_all, color=colours[0], alpha=0.5)

    ax[0].scatter(sig_sfr[vel_disp>51], vel_out[vel_disp>51], marker='o', s=30, label='All KOFFEE fits; R={:.2f}'.format(r_vel_out_all), color=colours[0], alpha=0.7)
    ax[0].scatter(sig_sfr[vel_disp<=51], vel_out[vel_disp<=51], marker='v', s=30, c=colours[0], alpha=0.7)

    if plot_medians == True:
        #ax[0].plot(bin_center_all, v_out_bin_medians_all, marker='', lw=3, label='Median all KOFFEE fits; R={:.2f}'.format(r_vel_out_med_all), color=colours[0])
        #indicate the error bars
        x_err = np.array([bin_center_all-logspace_all[:-1], logspace_all[1:]-bin_center_all])
        #plot the medians as error bars
        ax[0].errorbar(bin_center_all, v_out_bin_medians_all, yerr=v_out_bin_stdev_all, color=colours[0], capsize=3.0, lw=3, label='Median all KOFFEE fits; R={:.2f}'.format(r_vel_out_med_all))


    if plot_data_fits == True:
        ax[0].plot(sfr_linspace, pf.fitting_function(sfr_linspace, *popt_vout_all), 'r-', label='Fit: $v_{out}=%5.0f\pm$%2.0f $\Sigma_{SFR}^{%5.2f \pm %5.2f}$' % (popt_vout_all[0], np.sqrt(np.diag(pcov_vout_all))[0], popt_vout_all[1], np.sqrt(np.diag(pcov_vout_all))[1]))
        ax[0].plot(sfr_linspace, pf.fitting_function(sfr_linspace, *popt_vout_all_medians), 'r--', label='Median Fit: $v_{out}=%5.0f\pm$%2.0f $\Sigma_{SFR}^{%5.2f \pm %5.2f}$' %(popt_vout_all_medians[0], np.sqrt(np.diag(pcov_vout_all_medians))[0], popt_vout_all_medians[1], np.sqrt(np.diag(pcov_vout_all_medians))[1]))

    ax[0].plot(sfr_surface_density_chen, v_out_chen, ':k', label='Energy driven, $v_{out} \propto \Sigma_{SFR}^{0.1}$')
    ax[0].plot(sfr_surface_density_murray, v_out_murray, '--k', label='Momentum driven, $v_{out} \propto \Sigma_{SFR}^{2}$')

    ax[0].errorbar(0.05, 150, xerr=np.nanmedian(sig_sfr_err), yerr=np.nanmedian(vel_out_err), c='k')

    ax[0].set_ylim(100, 700)
    ax[0].set_xscale('log')
    if xlim_vals:
        ax[0].set_xlim(xlim_vals[0], xlim_vals[1])
    else:
        ax[0].set_xlim(np.nanmin(sig_sfr)-0.002, np.nanmax(sig_sfr)+2.0)
        #save the xlim values for the comparison figure
        xlim_vals = [np.nanmin(sig_sfr)-0.002, np.nanmax(sig_sfr)+2.0]

    lgnd = ax[0].legend(frameon=True, fontsize='small', loc='upper left', framealpha=0.5)
    #lgnd.legendHandles[0]._legmarker.set_markersize(3)
    ax[0].set_ylabel('Maximum Outflow Velocity [km s$^{-1}$]')
    ax[0].set_xlabel('$\Sigma_{SFR}$ [M$_\odot$ yr$^{-1}$ kpc$^{-2}$]')
    ax[0].set_title('S/N > 20 and $\delta_{BIC}$>10')

    #plot points within 90% radius
    #if plot_medians == True:
        #ax[1].fill_between(bin_center_physical, v_out_bin_lower_q_physical, v_out_bin_upper_q_physical, color=colours[1], alpha=0.3)
        #ax[1].fill_between(bin_center_physical, v_out_bin_medians_physical-v_out_bin_stdev_physical, v_out_bin_medians_physical+v_out_bin_stdev_physical, color=colours[0], alpha=0.5)

    #ax[1].scatter(sig_sfr[radius>6.1], vel_out[radius>6.1], marker='o', s=20, label='All KOFFEE fits', edgecolors=colours[0], alpha=0.3, facecolors='none')
    #ax[1].scatter(sig_sfr[vel_disp<=51], vel_out[vel_disp<=51], marker='v', s=20, edgecolors=colours[0], alpha=0.3, facecolors='none')
    ax[1].scatter(sig_sfr[physical_mask], vel_out[physical_mask], marker='o', s=30, label='Selected KOFFEE fits; R={:.2f}'.format(r_vel_out_physical), color=colours[1], alpha=0.7)

    if plot_medians == True:
        #ax[1].plot(bin_center_physical, v_out_bin_medians_physical, marker='', lw=3, label='Median of selected KOFFEE fits; R={:.2f}'.format(r_vel_out_med_physical), color=colours[1])
        x_err = np.array([bin_center_physical-logspace_physical[:-1], logspace_physical[1:]-bin_center_physical])
        #plot the medians as error bars
        ax[1].errorbar(bin_center_physical, v_out_bin_medians_physical, yerr=v_out_bin_stdev_physical, color=colours[1], capsize=3.0, lw=3, ms=5, label='Median of selected KOFFEE fits; R={:.2f}'.format(r_vel_out_med_physical))

    if plot_data_fits == True:
        ax[1].plot(sfr_linspace, pf.fitting_function(sfr_linspace, *popt_vout_physical), 'r-', label='Fit: $v_{out}=%5.0f\pm$%2.0f $\Sigma_{SFR}^{%5.2f \pm %5.2f}$' % (popt_vout_physical[0], np.sqrt(np.diag(pcov_vout_physical))[0], popt_vout_physical[1], np.sqrt(np.diag(pcov_vout_physical))[1]))
        ax[1].plot(sfr_linspace, pf.fitting_function(sfr_linspace, *popt_vout_physical_medians), 'r--', label='Median Fit: $v_{out}=%5.0f\pm$%2.0f $\Sigma_{SFR}^{%5.2f \pm %5.2f}$' %(popt_vout_physical_medians[0], np.sqrt(np.diag(pcov_vout_physical_medians))[0], popt_vout_physical_medians[1], np.sqrt(np.diag(pcov_vout_physical_medians))[1]))

    ax[1].plot(sfr_surface_density_chen, v_out_chen, ':k')#, label='Energy driven, $v_{out} \propto \Sigma_{SFR}^{0.1}$')
    #ax[1].plot(sfr_surface_density_murray, v_out_murray, '--k')#, label='Momentum driven, $v_{out} \propto \Sigma_{SFR}^{2}$')

    ax[1].errorbar(0.05, 150, xerr=np.nanmedian(sig_sfr_err[physical_mask]), yerr=np.nanmedian(vel_out_err[physical_mask]), c='k')

    #ax[1].set_xscale('log')
    lgnd = ax[1].legend(frameon=True, fontsize='small', loc='upper left', framealpha=0.5)
    #lgnd.legendHandles[0]._legmarker.set_markersize(3)
    ax[1].set_xlabel('$\Sigma_{SFR}$ [M$_\odot$ yr$^{-1}$ kpc$^{-2}$]')
    ax[1].set_title(r'$\delta_{BIC}$>10, $r$<$r_{90}$ and $\sigma_{broad}$>$\sigma_{inst}$')

    #plot points with strong BIC values
    #if plot_medians == True:
        #ax[2].fill_between(bin_center_strong, v_out_bin_lower_q_strong, v_out_bin_upper_q_strong, color=colours[2], alpha=0.3)
        #ax[2].fill_between(bin_center_strong, v_out_bin_medians_strong-v_out_bin_stdev_strong, v_out_bin_medians_strong+v_out_bin_stdev_strong, color=colours[0], alpha=0.5)

    #ax[2].scatter(sig_sfr[~BIC_diff_strong][vel_disp[~BIC_diff_strong]>51], vel_out[~BIC_diff_strong][vel_disp[~BIC_diff_strong]>51], marker='o', s=20, label='All KOFFEE fits', color=colours[0], alpha=0.3, facecolors='none')
    #ax[2].scatter(sig_sfr[~BIC_diff_strong][vel_disp[~BIC_diff_strong]<=51], vel_out[~BIC_diff_strong][vel_disp[~BIC_diff_strong]<=51], marker='v', s=20, edgecolors=colours[0], alpha=0.3, facecolors='none')
    ax[2].scatter(sig_sfr[BIC_diff_strong][vel_disp[BIC_diff_strong]>51], vel_out[BIC_diff_strong][vel_disp[BIC_diff_strong]>51], marker='o', s=30, label='Selected KOFFEE fits; R={:.2f}'.format(r_vel_out_strong), color=colours[2], alpha=0.8)
    ax[2].scatter(sig_sfr[BIC_diff_strong][vel_disp[BIC_diff_strong]<=51], vel_out[BIC_diff_strong][vel_disp[BIC_diff_strong]<=51], marker='v', s=30, color=colours[2], alpha=0.8)

    if plot_medians == True:
        #ax[2].plot(bin_center_strong, v_out_bin_medians_strong, marker='', lw=3, label='Median of selected KOFFEE fits; R={:.2f}'.format(r_vel_out_med_strong), color=colours[2])
        x_err = np.array([bin_center_strong-logspace_strong[:-1], logspace_strong[1:]-bin_center_physical])
        #plot the medians as error bars
        ax[2].errorbar(bin_center_strong, v_out_bin_medians_strong, yerr=v_out_bin_stdev_strong, color=colours[2], capsize=3.0, lw=3, ms=5, label='Median of selected KOFFEE fits; R={:.2f}'.format(r_vel_out_med_strong))

    if plot_data_fits == True:
        ax[2].plot(sfr_linspace, pf.fitting_function(sfr_linspace, *popt_vout_strong), 'r-', label='Fit: $v_{out}=%5.0f\pm$%2.0f $\Sigma_{SFR}^{%5.2f \pm %5.2f}$' % (popt_vout_strong[0], np.sqrt(np.diag(pcov_vout_strong))[0], popt_vout_strong[1], np.sqrt(np.diag(pcov_vout_strong))[1]))
        ax[2].plot(sfr_linspace, pf.fitting_function(sfr_linspace, *popt_vout_strong_medians), 'r--', label='Median Fit: $v_{out}=%5.0f\pm$%2.0f $\Sigma_{SFR}^{%5.2f \pm %5.2f}$' %(popt_vout_strong_medians[0], np.sqrt(np.diag(pcov_vout_strong_medians))[0], popt_vout_strong_medians[1], np.sqrt(np.diag(pcov_vout_strong_medians))[1]))

    ax[2].plot(sfr_surface_density_chen, v_out_chen, ':k')#, label='Energy driven, $v_{out} \propto \Sigma_{SFR}^{0.1}$')
    #ax[2].plot(sfr_surface_density_murray, v_out_murray, '--k')#, label='Momentum driven, $v_{out} \propto \Sigma_{SFR}^{2}$')

    ax[2].errorbar(0.05, 150, xerr=np.nanmedian(sig_sfr_err[BIC_diff_strong]), yerr=np.nanmedian(vel_out_err[BIC_diff_strong]), c='k')

    #ax[1].set_xscale('log')
    lgnd = ax[2].legend(frameon=True, fontsize='small', loc='upper left', framealpha=0.5)
    #lgnd.legendHandles[0]._legmarker.set_markersize(3)
    ax[2].set_xlabel('$\Sigma_{SFR}$ [M$_\odot$ yr$^{-1}$ kpc$^{-2}$]')
    ax[2].set_title('strongly likely BIC $\delta_{BIC}$>50')

    plt.show()

    return xlim_vals


#Figure 5
outflow_velocity_fits_files = [
    '../../code_outputs/koffee_results_IRAS08/IRAS08koffee_results_OIII_4_2021-03-24_resolved/IRAS08_resolved_outflow_velocity.fits',
    '../../code_outputs/koffee_results_IRAS08/IRAS08koffee_results_OIII_4_2021-03-24_resolved/IRAS08_binned_2_by_1_outflow_velocity.fits',
    '../../code_outputs/koffee_results_IRAS08/IRAS08koffee_results_OIII_4_2021-03-24_resolved/IRAS08_binned_3_by_1_outflow_velocity.fits',
    '../../code_outputs/koffee_results_IRAS08/IRAS08koffee_results_OIII_4_2021-03-24_resolved/IRAS08_binned_4_by_1_outflow_velocity.fits',
    '../../code_outputs/koffee_results_IRAS08/IRAS08koffee_results_OIII_4_2021-03-24_resolved/IRAS08_binned_5_by_1_outflow_velocity.fits',
    '../../code_outputs/koffee_results_IRAS08/IRAS08koffee_results_OIII_4_2021-03-24_resolved/IRAS08_binned_3_by_2_outflow_velocity.fits',
    '../../code_outputs/koffee_results_IRAS08/IRAS08koffee_results_OIII_4_2021-03-24_resolved/IRAS08_binned_4_by_2_outflow_velocity.fits',
    '../../code_outputs/koffee_results_IRAS08/IRAS08koffee_results_OIII_4_2021-03-24_resolved/IRAS08_binned_3_by_3_outflow_velocity.fits',
    '../../code_outputs/koffee_results_IRAS08/IRAS08koffee_results_OIII_4_2021-03-24_resolved/IRAS08_binned_5_by_2_outflow_velocity.fits',
    '../../code_outputs/koffee_results_IRAS08/IRAS08koffee_results_OIII_4_2021-03-24_resolved/IRAS08_binned_6_by_2_outflow_velocity.fits',
    '../../code_outputs/koffee_results_IRAS08/IRAS08koffee_results_OIII_4_2021-03-24_resolved/IRAS08_binned_7_by_2_outflow_velocity.fits',
    '../../code_outputs/koffee_results_IRAS08/IRAS08koffee_results_OIII_4_2021-03-24_resolved/IRAS08_binned_8_by_2_outflow_velocity.fits',
    '../../code_outputs/koffee_results_IRAS08/IRAS08koffee_results_OIII_4_2021-03-24_resolved/IRAS08_binned_9_by_2_outflow_velocity.fits']

outflow_dispersion_fits_files = [
    '../../code_outputs/koffee_results_IRAS08/IRAS08koffee_results_OIII_4_2021-03-24_resolved/IRAS08_resolved_outflow_velocity_dispersion.fits',
    '../../code_outputs/koffee_results_IRAS08/IRAS08koffee_results_OIII_4_2021-03-24_resolved/IRAS08_binned_2_by_1_outflow_velocity_dispersion.fits',
    '../../code_outputs/koffee_results_IRAS08/IRAS08koffee_results_OIII_4_2021-03-24_resolved/IRAS08_binned_3_by_1_outflow_velocity_dispersion.fits',
    '../../code_outputs/koffee_results_IRAS08/IRAS08koffee_results_OIII_4_2021-03-24_resolved/IRAS08_binned_4_by_1_outflow_velocity_dispersion.fits',
    '../../code_outputs/koffee_results_IRAS08/IRAS08koffee_results_OIII_4_2021-03-24_resolved/IRAS08_binned_5_by_1_outflow_velocity_dispersion.fits',
    '../../code_outputs/koffee_results_IRAS08/IRAS08koffee_results_OIII_4_2021-03-24_resolved/IRAS08_binned_3_by_2_outflow_velocity_dispersion.fits',
    '../../code_outputs/koffee_results_IRAS08/IRAS08koffee_results_OIII_4_2021-03-24_resolved/IRAS08_binned_4_by_2_outflow_velocity_dispersion.fits',
    '../../code_outputs/koffee_results_IRAS08/IRAS08koffee_results_OIII_4_2021-03-24_resolved/IRAS08_binned_3_by_3_outflow_velocity_dispersion.fits',
    '../../code_outputs/koffee_results_IRAS08/IRAS08koffee_results_OIII_4_2021-03-24_resolved/IRAS08_binned_5_by_2_outflow_velocity_dispersion.fits',
    '../../code_outputs/koffee_results_IRAS08/IRAS08koffee_results_OIII_4_2021-03-24_resolved/IRAS08_binned_6_by_2_outflow_velocity_dispersion.fits',
    '../../code_outputs/koffee_results_IRAS08/IRAS08koffee_results_OIII_4_2021-03-24_resolved/IRAS08_binned_7_by_2_outflow_velocity_dispersion.fits',
    '../../code_outputs/koffee_results_IRAS08/IRAS08koffee_results_OIII_4_2021-03-24_resolved/IRAS08_binned_8_by_2_outflow_velocity_dispersion.fits',
    '../../code_outputs/koffee_results_IRAS08/IRAS08koffee_results_OIII_4_2021-03-24_resolved/IRAS08_binned_9_by_2_outflow_velocity_dispersion.fits']

sig_sfr_fits_files = [
    '../../code_outputs/koffee_results_IRAS08/IRAS08koffee_results_OIII_4_2021-03-24_resolved/IRAS08_resolved_star_formation_rate_surface_density.fits',
    '../../code_outputs/koffee_results_IRAS08/IRAS08koffee_results_OIII_4_2021-03-24_resolved/IRAS08_binned_2_by_1_star_formation_rate_surface_density.fits',
    '../../code_outputs/koffee_results_IRAS08/IRAS08koffee_results_OIII_4_2021-03-24_resolved/IRAS08_binned_3_by_1_star_formation_rate_surface_density.fits',
    '../../code_outputs/koffee_results_IRAS08/IRAS08koffee_results_OIII_4_2021-03-24_resolved/IRAS08_binned_4_by_1_star_formation_rate_surface_density.fits',
    '../../code_outputs/koffee_results_IRAS08/IRAS08koffee_results_OIII_4_2021-03-24_resolved/IRAS08_binned_5_by_1_star_formation_rate_surface_density.fits',
    '../../code_outputs/koffee_results_IRAS08/IRAS08koffee_results_OIII_4_2021-03-24_resolved/IRAS08_binned_3_by_2_star_formation_rate_surface_density.fits',
    '../../code_outputs/koffee_results_IRAS08/IRAS08koffee_results_OIII_4_2021-03-24_resolved/IRAS08_binned_4_by_2_star_formation_rate_surface_density.fits',
    '../../code_outputs/koffee_results_IRAS08/IRAS08koffee_results_OIII_4_2021-03-24_resolved/IRAS08_binned_3_by_3_star_formation_rate_surface_density.fits',
    '../../code_outputs/koffee_results_IRAS08/IRAS08koffee_results_OIII_4_2021-03-24_resolved/IRAS08_binned_5_by_2_star_formation_rate_surface_density.fits',
    '../../code_outputs/koffee_results_IRAS08/IRAS08koffee_results_OIII_4_2021-03-24_resolved/IRAS08_binned_6_by_2_star_formation_rate_surface_density.fits',
    '../../code_outputs/koffee_results_IRAS08/IRAS08koffee_results_OIII_4_2021-03-24_resolved/IRAS08_binned_7_by_2_star_formation_rate_surface_density.fits',
    '../../code_outputs/koffee_results_IRAS08/IRAS08koffee_results_OIII_4_2021-03-24_resolved/IRAS08_binned_8_by_2_star_formation_rate_surface_density.fits',
    '../../code_outputs/koffee_results_IRAS08/IRAS08koffee_results_OIII_4_2021-03-24_resolved/IRAS08_binned_9_by_2_star_formation_rate_surface_density.fits']

chi_square_text_files = [
    '../../code_outputs/koffee_results_IRAS08/IRAS08koffee_results_OIII_4_2021-03-24_resolved/IRAS08_chi_squared_OIII_4.txt',
    '../../code_outputs/koffee_results_IRAS08/IRAS08koffee_results_OIII_4_2021-03-25_binned_2_by_1/IRAS08_chi_squared_OIII_4.txt',
    '../../code_outputs/koffee_results_IRAS08/IRAS08koffee_results_OIII_4_2021-03-31_binned_3_by_1/IRAS08_chi_squared_OIII_4.txt',
    '../../code_outputs/koffee_results_IRAS08/IRAS08koffee_results_OIII_4_2021-03-30_binned_4_by_1/IRAS08_chi_squared_OIII_4.txt',
    '../../code_outputs/koffee_results_IRAS08/IRAS08koffee_results_OIII_4_2021-03-31_binned_5_by_1/IRAS08_chi_squared_OIII_4.txt',
    '../../code_outputs/koffee_results_IRAS08/IRAS08koffee_results_OIII_4_2021-03-25_binned_3_by_2/IRAS08_chi_squared_OIII_4.txt',
    '../../code_outputs/koffee_results_IRAS08/IRAS08koffee_results_OIII_4_2021-03-30_binned_4_by_2/IRAS08_chi_squared_OIII_4.txt',
    '../../code_outputs/koffee_results_IRAS08/IRAS08koffee_results_OIII_4_2021-03-24_binned_3_by_3/IRAS08_chi_squared_OIII_4.txt',
    '../../code_outputs/koffee_results_IRAS08/IRAS08koffee_results_OIII_4_2021-03-30_binned_5_by_2/IRAS08_chi_squared_OIII_4.txt',
    '../../code_outputs/koffee_results_IRAS08/IRAS08koffee_results_OIII_4_2021-03-30_binned_6_by_2/IRAS08_chi_squared_OIII_4.txt',
    '../../code_outputs/koffee_results_IRAS08/IRAS08koffee_results_OIII_4_2021-03-30_binned_7_by_2/IRAS08_chi_squared_OIII_4.txt',
    '../../code_outputs/koffee_results_IRAS08/IRAS08koffee_results_OIII_4_2021-03-30_binned_8_by_2/IRAS08_chi_squared_OIII_4.txt',
    '../../code_outputs/koffee_results_IRAS08/IRAS08koffee_results_OIII_4_2021-03-30_binned_9_by_2/IRAS08_chi_squared_OIII_4.txt']

statistical_results_text_files = [
    '../../code_outputs/koffee_results_IRAS08/IRAS08koffee_results_OIII_4_2021-03-24_resolved/IRAS08_stat_results_OIII_4.txt',
    '../../code_outputs/koffee_results_IRAS08/IRAS08koffee_results_OIII_4_2021-03-25_binned_2_by_1/IRAS08_stat_results_OIII_4.txt',
    '../../code_outputs/koffee_results_IRAS08/IRAS08koffee_results_OIII_4_2021-03-31_binned_3_by_1/IRAS08_stat_results_OIII_4.txt',
    '../../code_outputs/koffee_results_IRAS08/IRAS08koffee_results_OIII_4_2021-03-30_binned_4_by_1/IRAS08_stat_results_OIII_4.txt',
    '../../code_outputs/koffee_results_IRAS08/IRAS08koffee_results_OIII_4_2021-03-31_binned_5_by_1/IRAS08_stat_results_OIII_4.txt',
    '../../code_outputs/koffee_results_IRAS08/IRAS08koffee_results_OIII_4_2021-03-25_binned_3_by_2/IRAS08_stat_results_OIII_4.txt',
    '../../code_outputs/koffee_results_IRAS08/IRAS08koffee_results_OIII_4_2021-03-30_binned_4_by_2/IRAS08_stat_results_OIII_4.txt',
    '../../code_outputs/koffee_results_IRAS08/IRAS08koffee_results_OIII_4_2021-03-24_binned_3_by_3/IRAS08_stat_results_OIII_4.txt',
    '../../code_outputs/koffee_results_IRAS08/IRAS08koffee_results_OIII_4_2021-03-30_binned_5_by_2/IRAS08_stat_results_OIII_4.txt',
    '../../code_outputs/koffee_results_IRAS08/IRAS08koffee_results_OIII_4_2021-03-30_binned_6_by_2/IRAS08_stat_results_OIII_4.txt',
    '../../code_outputs/koffee_results_IRAS08/IRAS08koffee_results_OIII_4_2021-03-30_binned_7_by_2/IRAS08_stat_results_OIII_4.txt',
    '../../code_outputs/koffee_results_IRAS08/IRAS08koffee_results_OIII_4_2021-03-30_binned_8_by_2/IRAS08_stat_results_OIII_4.txt',
    '../../code_outputs/koffee_results_IRAS08/IRAS08koffee_results_OIII_4_2021-03-30_binned_9_by_2/IRAS08_stat_results_OIII_4.txt']

data_fits_files = [
    '../../data/IRAS08_red_cubes/IRAS08339_metacube.fits',
    '../../data/IRAS08_metacube_binned_2_by_1.fits',
    '../../data/IRAS08_metacube_binned_3_by_1.fits',
    '../../data/IRAS08_metacube_binned_4_by_1.fits',
    '../../data/IRAS08_metacube_binned_5_by_1.fits',
    '../../data/IRAS08_metacube_binned_3_by_2.fits',
    '../../data/IRAS08_metacube_binned_4_by_2.fits',
    '../../data/IRAS08_metacube_binned_3_by_3.fits',
    '../../data/IRAS08_metacube_binned_5_by_2.fits',
    '../../data/IRAS08_metacube_binned_6_by_2.fits',
    '../../data/IRAS08_metacube_binned_7_by_2.fits',
    '../../data/IRAS08_metacube_binned_8_by_2.fits',
    '../../data/IRAS08_metacube_binned_9_by_2.fits']

data_descriptor = ['1x1', '2x1', '3x1', '4x1', '5x1', '3x2', '4x2', '3x3', '5x2', '6x2', '7x2', '8x2', '9x2']

def plot_sfr_vout_correlation_with_binning_from_file(outflow_velocity_fits_files, outflow_dispersion_fits_files, sig_sfr_fits_files, chi_square_text_files, statistical_results_text_files, z, data_fits_files, data_descriptor):
    """
    Plots the Pearson Correlation Coefficient for Sigma_SFR-v_out for each bin
    size against circularised bin diameter.

    Parameters
    ----------
    outflow_velocity_fits_file : list of strings
        list of fits files containing the outflow velocity results

    outflow_dispersion_fits_files : list of strings
        list of fits files containing the outflow dispersion results

    sig_sfr_fits_files : list of strings
        list of fits files containing the SFR surface density results

    chi_square_text_files : list of strings
        list of text files containing the chi square values for each fit

    statistical_results_text_files : list of strings
        list of text files containing the statistical results

    z : float
        redshift

    data_fits_files : list of strings
        list of fits files containing the data

    data_descriptor : list of strings
        list of descriptors for each data set e.g. '1x1'

    Returns
    -------
    A single panel of the correlation coefficient for Sigma_SFR against
    outflow velocity, plotted against the circularised bin diameter.

    """
    #create figure
    plt.rcParams.update(pf.get_rc_params())
    #fig, ax = plt.subplots(nrows=1, ncols=2, sharey=True, sharex=True, figsize=(8,4), constrained_layout=True)
    fig, ax = plt.subplots(nrows=1, ncols=1, sharey=True, sharex=True, figsize=(5,4), constrained_layout=True)

    #get colours from cmasher
    colours = cmr.take_cmap_colors('cmr.gem', 3, cmap_range=(0.25, 0.85), return_fmt='hex')

    #create arrays to save results to
    spaxel_area = np.full(len(outflow_velocity_fits_files), np.nan, dtype=np.double)

    corr_coeff_all = np.full(len(outflow_velocity_fits_files), np.nan, dtype=np.double)
    corr_coeff_physical = np.full(len(outflow_velocity_fits_files), np.nan, dtype=np.double)
    corr_coeff_strong = np.full(len(outflow_velocity_fits_files), np.nan, dtype=np.double)

    #calculate the proper distance
    proper_dist = cosmo.kpc_proper_per_arcmin(z).to(u.kpc/u.arcsec)


    #iterate through all of the data sets
    for i in np.arange(len(outflow_velocity_fits_files)):
        #load the outflow velocity
        with fits.open(outflow_velocity_fits_files[i]) as hdu:
            vel_out = hdu[0].data
            vel_out_err = hdu[1].data
        hdu.close()

        print(data_descriptor[i], 'vel_out shape:', vel_out.shape)

        #load the velocity dispersion
        with fits.open(outflow_dispersion_fits_files[i]) as hdu:
            vel_disp = hdu[0].data
        hdu.close()

        #load the sfr surface density
        with fits.open(sig_sfr_fits_files[i]) as hdu:
            sfr_surface_density = hdu[0].data
            sfr_surface_density_err = hdu[1].data
        hdu.close()

        print(data_descriptor[i], 'sfr_surface_density shape:', sfr_surface_density.shape)

        #load the chi squared results
        chi_square = np.loadtxt(chi_square_text_files[i])
        print(data_descriptor[i], 'chi_square shape:', chi_square.shape)
        BIC_outflow = chi_square[1,:]
        BIC_no_outflow = chi_square[0,:]
        del chi_square

        #load the statistical results
        statistical_results = np.loadtxt(statistical_results_text_files[i])
        print('stat_res shape:', statistical_results.shape)

        #reshape the things to not be flat
        statistical_results = statistical_results.reshape(vel_out.shape[0], vel_out.shape[1])
        BIC_outflow = BIC_outflow.reshape(vel_out.shape[0], vel_out.shape[1])
        BIC_no_outflow = BIC_no_outflow.reshape(vel_out.shape[0], vel_out.shape[1])

        print('stat_res shape:', statistical_results.shape)

        #load the radius
        lam, xx, yy, rad, data, xx_flat, yy_flat, rad_flat, data_flat, header = pc.prepare_single_cube(data_fits_files[i], 'IRAS08_'+data_descriptor[i], z, 'red', '../../code_outputs/koffee_results_IRAS08/')
        del lam, xx, yy, data, xx_flat, yy_flat, rad_flat, data_flat

        #get the sfr for the outflow spaxels
        flow_mask = (statistical_results>0) #& (sfr_surface_density>0.1)

        print('flow mask data_shape:', flow_mask.shape)

        #flatten all the arrays and get rid of extra spaxels
        sig_sfr = sfr_surface_density[flow_mask]
        sig_sfr_err = sfr_surface_density_err[flow_mask]
        vel_out = vel_out[flow_mask]
        vel_out_err = vel_out_err[flow_mask]
        BIC_outflow_masked = BIC_outflow[flow_mask]
        BIC_no_outflow_masked = BIC_no_outflow[flow_mask]
        vel_disp = vel_disp[flow_mask]
        radius_masked = rad[flow_mask]

        #create BIC diff
        BIC_diff = BIC_outflow_masked - BIC_no_outflow_masked
        BIC_diff_strong = (BIC_diff < -50)

        #physical limits mask -
        #for the radius mask 6.1" is the 90% radius
        #also mask out the fits which lie on the lower limit of dispersion < 51km/s
        physical_mask = (radius_masked < 6.1) & (vel_disp>51)

        print(sig_sfr[physical_mask])
        print(sig_sfr[physical_mask].shape)

        #make sure none of the errors are nan values
        vel_out_err[np.where(np.isnan(vel_out_err)==True)] = np.nanmedian(vel_out_err)

        #calculate the r value for all the values
        r_vel_out_all, p_value_v_out_all = pf.pearson_correlation(np.log10(sig_sfr), np.log10(vel_out))
        r_vel_out_physical, p_value_v_out_physical = pf.pearson_correlation(np.log10(sig_sfr[physical_mask]), np.log10(vel_out[physical_mask]))
        r_vel_out_clean, p_value_v_out_clean = pf.pearson_correlation(np.log10(sig_sfr[BIC_diff_strong]), np.log10(vel_out[BIC_diff_strong]))

        #save results to arrays
        spaxel_area[i] = ((header['CD1_2']*60*60*header['CD2_1']*60*60)*(proper_dist**2)).value

        corr_coeff_all[i] = r_vel_out_all
        corr_coeff_physical[i] = r_vel_out_physical
        corr_coeff_strong[i] = r_vel_out_clean

        #print average numbers for the different panels
        print(data_descriptor[i])
        print('Number of spaxels with outflows', vel_out.shape)
        print('All spaxels median v_out:', np.nanmedian(vel_out))
        print('All spaxels standard deviation v_out:', np.nanstd(vel_out))
        print('All spaxels median sigma_sfr:', np.nanmedian(sig_sfr))
        print('All spaxels standard deviation sigma_sfr:', np.nanstd(sig_sfr))
        print('')


        print('Number of spaxels with broad sigmas at the instrument dispersion:', vel_out[vel_disp<=51].shape)
        print('')
        print('Number of spaxels beyond R_90:', vel_out[radius_masked>6.1].shape)
        print('')
        print('Number of spaxels in the middle panel:', vel_out[physical_mask].shape)
        print('')

        print('Physical spaxels median v_out:', np.nanmedian(vel_out[physical_mask]))
        print('Physical spaxels standard deviation v_out:', np.nanstd(vel_out[physical_mask]))
        print('Physical spaxels median sigma_sfr:', np.nanmedian(sig_sfr[physical_mask]))
        print('Physical spaxels standard deviation sigma_sfr:', np.nanstd(sig_sfr[physical_mask]))
        print('')


        print('Number of spaxels with strong BIC differences:', vel_out[BIC_diff_strong].shape)
        print('')

        print('Clean spaxels median v_out:', np.nanmedian(vel_out[BIC_diff_strong]))
        print('Clean spaxels standard deviation v_out:', np.nanstd(vel_out[BIC_diff_strong]))
        print('Clean spaxels median sigma_sfr:', np.nanmedian(sig_sfr[BIC_diff_strong]))
        print('Clean spaxels standard deviation sigma_sfr:', np.nanstd(sig_sfr[BIC_diff_strong]))
        print('')


    #convert spaxel area to circularised radius Area = pi*r^2
    #so r = sqrt(Area/pi)
    circularised_radius = np.sqrt(spaxel_area/np.pi)

    #-------
    #plot it
    #-------
    ax.plot(circularised_radius*2, corr_coeff_all**2, marker='o', label='S/N>20 and $\delta_{BIC}$>10', color=colours[0])
    ax.plot(circularised_radius*2, corr_coeff_physical**2, marker='o', label=r'$\delta_{BIC}$>10, $r$<$r_{90}$ and $\sigma_{broad}$>$\sigma_{inst}$', color=colours[1])
    ax.plot(circularised_radius*2, corr_coeff_strong**2, marker='o', label='strongly likely BIC $\delta_{BIC}$>50', color=colours[2])


    lgnd = ax.legend(frameon=True, fontsize='small', loc='upper left', framealpha=0.5)
    ax.set_ylabel('Correlation Coefficient $R^2$')
    #ax.set_ylabel('Correlation Coefficient $R^2$')
    ax.set_xlabel('Circularised Bin Diameter [kpc]')

    ax.set_ylim(np.nanmin(np.array([corr_coeff_all, corr_coeff_physical, corr_coeff_strong])**2)-0.04, np.nanmax(np.array([corr_coeff_all, corr_coeff_physical, corr_coeff_strong])**2)+0.05)

    plt.show()



#Figure 6
def plot_sfr_mlf_flux(OIII_outflow_results, OIII_outflow_error, hbeta_outflow_results, hbeta_outflow_error, hbeta_no_outflow_results, hbeta_no_outflow_error, BIC_outflow, BIC_no_outflow, statistical_results, z, radius, header, weighted_average=True, plot_medians=True, plot_data_fits=False, xlim_vals=None):
    """
    Plots the SFR surface density against the mass loading factor and the Hbeta
    flux ratio, with Sigma_SFR calculated using only the narrow component.

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

    radius : :obj:'~numpy.ndarray'
        array of galaxy radius values

    header : FITS header object
        the header from the fits file

    weighted_average : boolean
        whether or not to take a weighted average using the errors (Default=True)

    Returns
    -------
    A six panel graph of the mass loading factor and Hbeta flux ratio against
    the SFR surface density
    """
    #calculate the sfr surface density - using just the systemic line, and including the flux line
    #don't include extinction since this was included in the continuum subtraction using ppxf
    sfr, sfr_err, total_sfr, sfr_surface_density, sfr_surface_density_err = calc_sfr.calc_sfr_koffee(hbeta_outflow_results, hbeta_outflow_error, hbeta_no_outflow_results, hbeta_no_outflow_error, statistical_results, z, header, include_extinction=False, include_outflow=False)

    #calculate the mass loading factor
    mlf, mlf_max, mlf_min = calc_mlf.calc_mass_loading_factor(OIII_outflow_results, OIII_outflow_error, hbeta_outflow_results, hbeta_outflow_error, hbeta_no_outflow_results, hbeta_no_outflow_error, statistical_results, z, header)

    #calculate the flux for systematic and flow gaussians
    systemic_flux, systemic_flux_err, outflow_flux, outflow_flux_err = calc_sfr.calc_flux_from_koffee(hbeta_outflow_results, hbeta_outflow_error, statistical_results, z, outflow=True)

    #calculate the velocity dispersion for the masking
    vel_disp, vel_disp_err, vel_diff, vel_diff_err, vel_out, vel_out_err = calc_outvel.calc_outflow_vel(OIII_outflow_results, OIII_outflow_error, statistical_results, z)

    #create the flow mask
    flow_mask = (statistical_results>0) #& (np.isnan(hbeta_outflow_results[3,:,:])==False)


    #flatten all the arrays and get rid of extra spaxels
    sig_sfr = sfr_surface_density[flow_mask]
    sig_sfr_err = sfr_surface_density_err[flow_mask]

    mlf = mlf[flow_mask]
    mlf_max = mlf_max[flow_mask]
    mlf_min = mlf_min[flow_mask]

    systemic_flux = systemic_flux[flow_mask]
    systemic_flux_err = systemic_flux_err[flow_mask]
    outflow_flux = outflow_flux[flow_mask]
    outflow_flux_err = outflow_flux_err[flow_mask]

    BIC_outflow = BIC_outflow[flow_mask]
    BIC_no_outflow = BIC_no_outflow[flow_mask]

    vel_disp = vel_disp[flow_mask]
    radius = radius[flow_mask]


    #take the log of the mlf
    mlf = np.log10(mlf)
    mlf_max = np.log10(mlf_max)
    mlf_min = np.log10(mlf_min)

    #calculate the errors
    mlf_err_max = mlf_max - mlf
    mlf_err_min = mlf - mlf_min

    #make sure none of the flux errors are nan values
    systemic_flux_err[np.isnan(systemic_flux_err)] = np.nanmedian(systemic_flux_err)
    outflow_flux_err[np.isnan(outflow_flux_err)] = np.nanmedian(outflow_flux_err)

    #take the log and do the flux ratio
    flux_ratio = np.log10(outflow_flux/systemic_flux)
    #flux_ratio = (outflow_flux/systemic_flux)

    #calculate the error
    flux_error = flux_ratio * np.log10(np.sqrt((outflow_flux_err/outflow_flux)**2 + (systemic_flux_err/systemic_flux)**2))

    #create BIC diff
    BIC_diff = BIC_outflow - BIC_no_outflow
    BIC_diff_strong = (BIC_diff < -50)

    #physical limits mask -
    #for the radius mask 6.1" is the 90% radius
    #also mask out the fits which lie on the lower limit of dispersion < 51km/s
    physical_mask = (radius < 6.1) & (vel_disp>51)

    #make sure none of the errors are nan values
    #vel_out_err[np.where(np.isnan(vel_out_err)==True)] = np.nanmedian(vel_out_err)

    #do the calculations for all the bins
    num_bins = 5
    min_bin = None #-0.05
    max_bin = None #0.6

    if weighted_average == False:
        logspace_all, bin_center_all, mlf_bin_medians_all, mlf_bin_lower_q_all, mlf_bin_upper_q_all, mlf_bin_stdev_all = pf.binned_median_quantile_log(sig_sfr, mlf, num_bins=num_bins, weights=None, min_bin=min_bin, max_bin=max_bin)
        logspace_physical, bin_center_physical, mlf_bin_medians_physical, mlf_bin_lower_q_physical, mlf_bin_upper_q_physical, mlf_bin_stdev_physical = pf.binned_median_quantile_log(sig_sfr[physical_mask], mlf[physical_mask], num_bins=num_bins, weights=None, min_bin=min_bin, max_bin=max_bin)
        logspace_strong, bin_center_strong, mlf_bin_medians_strong, mlf_bin_lower_q_strong, mlf_bin_upper_q_strong, mlf_bin_stdev_strong = pf.binned_median_quantile_log(sig_sfr[BIC_diff_strong], mlf[BIC_diff_strong], num_bins=num_bins, weights=None, min_bin=min_bin, max_bin=max_bin)

        logspace_all, bin_center_all, flux_bin_medians_all, flux_bin_lower_q_all, flux_bin_upper_q_all, flux_bin_stdev_all = pf.binned_median_quantile_log(sig_sfr, flux_ratio, num_bins=num_bins, weights=None, min_bin=min_bin, max_bin=max_bin)
        logspace_physical, bin_center_physical, flux_bin_medians_physical, flux_bin_lower_q_physical, flux_bin_upper_q_physical, flux_bin_stdev_physical = pf.binned_median_quantile_log(sig_sfr[physical_mask], flux_ratio[physical_mask], num_bins=num_bins, weights=None, min_bin=min_bin, max_bin=max_bin)
        logspace_strong, bin_center_strong, flux_bin_medians_strong, flux_bin_lower_q_strong, flux_bin_upper_q_strong, flux_bin_stdev_strong = pf.binned_median_quantile_log(sig_sfr[BIC_diff_strong], flux_ratio[BIC_diff_strong], num_bins=num_bins, weights=None, min_bin=min_bin, max_bin=max_bin)


    elif weighted_average == True:
        logspace_all, bin_center_all, mlf_bin_medians_all, mlf_bin_lower_q_all, mlf_bin_upper_q_all, mlf_bin_stdev_all = pf.binned_median_quantile_log(sig_sfr, mlf, num_bins=num_bins, weights=[vel_out_err], min_bin=min_bin, max_bin=max_bin)
        logspace_physical, bin_center_physical, mlf_bin_medians_physical, mlf_bin_lower_q_physical, mlf_bin_upper_q_physical, mlf_bin_stdev_physical = pf.binned_median_quantile_log(sig_sfr[physical_mask], mlf[physical_mask], num_bins=num_bins, weights=[vel_out_err], min_bin=min_bin, max_bin=max_bin)
        logspace_strong, bin_center_strong, mlf_bin_medians_strong, mlf_bin_lower_q_strong, mlf_bin_upper_q_strong, mlf_bin_stdev_strong = pf.binned_median_quantile_log(sig_sfr[BIC_diff_strong], mlf[BIC_diff_strong], num_bins=num_bins, weights=[vel_out_err], min_bin=min_bin, max_bin=max_bin)

        logspace_all, bin_center_all, flux_bin_medians_all, flux_bin_lower_q_all, flux_bin_upper_q_all, flux_bin_stdev_all = pf.binned_median_quantile_log(sig_sfr, flux_ratio, num_bins=num_bins, weights=[flux_error], min_bin=min_bin, max_bin=max_bin)
        logspace_physical, bin_center_physical, flux_bin_medians_physical, flux_bin_lower_q_physical, flux_bin_upper_q_physical, flux_bin_stdev_physical = pf.binned_median_quantile_log(sig_sfr[physical_mask], flux_ratio[physical_mask], num_bins=num_bins, weights=[flux_error], min_bin=min_bin, max_bin=max_bin)
        logspace_strong, bin_center_strong, flux_bin_medians_strong, flux_bin_lower_q_strong, flux_bin_upper_q_strong, flux_bin_stdev_strong = pf.binned_median_quantile_log(sig_sfr[BIC_diff_strong], flux_ratio[BIC_diff_strong], num_bins=num_bins, weights=[flux_error], min_bin=min_bin, max_bin=max_bin)


    #calculate the r value for the median values
    #mlf is already logged
    r_mlf_med_all, p_value_mlf_all = pf.pearson_correlation(np.log10(bin_center_all), mlf_bin_medians_all)
    r_mlf_med_physical, p_value_mlf_physical = pf.pearson_correlation(np.log10(bin_center_physical), mlf_bin_medians_physical)
    r_mlf_med_strong, p_value_mlf_strong = pf.pearson_correlation(np.log10(bin_center_strong), mlf_bin_medians_strong)

    #calculate the r value for all the values
    r_mlf_all, p_value_mlf_all = pf.pearson_correlation(np.log10(sig_sfr[~np.isnan(mlf)]), mlf[~np.isnan(mlf)])
    r_mlf_physical, p_value_mlf_physical = pf.pearson_correlation(np.log10(sig_sfr[~np.isnan(mlf)&physical_mask]), mlf[~np.isnan(mlf)&physical_mask])
    r_mlf_strong, p_value_mlf_strong = pf.pearson_correlation(np.log10(sig_sfr[~np.isnan(mlf)&BIC_diff_strong]), mlf[~np.isnan(mlf)&BIC_diff_strong])

    #calculate the r value for the median values
    #flux is already logged
    r_flux_med_all, p_value_flux_all = pf.pearson_correlation(np.log10(bin_center_all), flux_bin_medians_all)
    r_flux_med_physical, p_value_flux_physical = pf.pearson_correlation(np.log10(bin_center_physical), flux_bin_medians_physical)
    r_flux_med_strong, p_value_flux_strong = pf.pearson_correlation(np.log10(bin_center_strong), flux_bin_medians_strong)

    #calculate the r value for all the values
    r_flux_all, p_value_flux_all = pf.pearson_correlation(np.log10(sig_sfr[~np.isnan(flux_ratio)]), flux_ratio[~np.isnan(flux_ratio)])
    r_flux_physical, p_value_flux_physical = pf.pearson_correlation(np.log10(sig_sfr[~np.isnan(flux_ratio) & physical_mask]), flux_ratio[~np.isnan(flux_ratio) & physical_mask])
    r_flux_strong, p_value_flux_strong = pf.pearson_correlation(np.log10(sig_sfr[~np.isnan(flux_ratio) & BIC_diff_strong]), flux_ratio[~np.isnan(flux_ratio) & BIC_diff_strong])


    #calculate Kim et al. trend
    if xlim_vals:
        sfr_surface_density_kim, mlf_Kim = pf.kim_et_al_2020(xlim_vals[0], xlim_vals[1], scale_factor=(10**mlf_bin_medians_all[0])/(bin_center_all[0]**-0.44))#0.06)
    else:
        sfr_surface_density_kim, mlf_Kim = pf.kim_et_al_2020(sig_sfr.min(), sig_sfr.max(), scale_factor=(10**mlf_bin_medians_all[0])/(bin_center_all[0]**-0.44))#0.06)


    #fit our own trends
    if plot_data_fits == True:
        popt_mlf_all_medians, pcov_mlf_all_medians = curve_fit(pf.fitting_function, bin_center_all, 10**mlf_bin_medians_all)
        popt_mlf_physical_medians, pcov_mlf_physical_medians = curve_fit(pf.fitting_function, bin_center_physical, 10**mlf_bin_medians_physical)
        popt_mlf_strong_medians, pcov_mlf_strong_medians = curve_fit(pf.fitting_function, bin_center_strong, 10**mlf_bin_medians_strong)

        popt_mlf_all, pcov_mlf_all = curve_fit(pf.fitting_function, sig_sfr[~np.isnan(mlf)], 10**mlf[~np.isnan(mlf)])
        popt_mlf_physical, pcov_mlf_physical = curve_fit(pf.fitting_function, sig_sfr[~np.isnan(mlf) & physical_mask], 10**mlf[~np.isnan(mlf) & physical_mask])
        popt_mlf_strong, pcov_mlf_strong = curve_fit(pf.fitting_function, sig_sfr[~np.isnan(mlf) & BIC_diff_strong], 10**mlf[~np.isnan(mlf) & BIC_diff_strong])

        sfr_linspace = np.linspace(sig_sfr.min(), sig_sfr.max()+4, num=1000)




    #print average numbers for the different panels
    print('Number of spaxels in the first panel', mlf.shape)
    print('All spaxels median mlf:', np.nanmedian(10**mlf))
    print('All spaxels standard deviation mlf:', np.nanstd(10**mlf))
    print('All spaxels rms mlf:', np.sqrt(np.nanmean(10**mlf)))
    print('')

    print('All spaxels median flux_ratio:', np.nanmedian(flux_ratio))
    print('All spaxels standard deviation flux_ratio:', np.nanstd(flux_ratio))
    print('')

    print('Number of spaxels with broad sigmas at the instrument dispersion:', mlf[vel_disp<=51].shape)
    print('')
    print('Number of spaxels beyond R_90:', mlf[radius>6.1].shape)
    print('')
    print('Number of spaxels in the middle panel:', mlf[physical_mask].shape)
    print('')

    print('Physical spaxels median mlf:', np.nanmedian(10**mlf[physical_mask]))
    print('Physical spaxels standard deviation mlf:', np.nanstd(10**mlf[physical_mask]))
    print('Physical spaxels rms mlf:', np.sqrt(np.nanmean(10**mlf[physical_mask])))
    print('')

    print('Physical spaxels median flux_ratio:', np.nanmedian(flux_ratio[physical_mask]))
    print('Physical spaxels standard deviation flux_ratio:', np.nanstd(flux_ratio[physical_mask]))
    print('')

    print('Number of spaxels with strong BIC differences:', mlf[BIC_diff_strong].shape)
    print('')

    print('Clean spaxels median mlf:', np.nanmedian(10**mlf[BIC_diff_strong]))
    print('Clean spaxels standard deviation mlf:', np.nanstd(10**mlf[BIC_diff_strong]))
    print('Clean spaxels rms mlf:', np.sqrt(np.nanmean(10**mlf[BIC_diff_strong])))
    print('')

    print('Clean spaxels median flux_ratio:', np.nanmedian(flux_ratio[BIC_diff_strong]))
    print('Clean spaxels standard deviation flux_ratio:', np.nanstd(flux_ratio[BIC_diff_strong]))
    print('')

    #-------
    #plot it
    #-------
    plt.rcParams.update(pf.get_rc_params())
    fig, ax = plt.subplots(nrows=2, ncols=3, sharex=True, sharey='row', figsize=(10,7), constrained_layout=True)

    #get colours from cmasher
    colours = cmr.take_cmap_colors('cmr.gem', 3, cmap_range=(0.25, 0.85), return_fmt='hex')

    #plot all points
    #if plot_medians == True:
        #ax[1,0].fill_between(bin_center_all, mlf_bin_lower_q_all, mlf_bin_upper_q_all, color=colours[0], alpha=0.3)

    ax[1,0].scatter(sig_sfr[vel_disp>51], mlf[vel_disp>51], marker='o', s=30, label='All KOFFEE fits; R={:.2f}'.format(r_mlf_all), color=colours[0], alpha=0.7)
    ax[1,0].scatter(sig_sfr[vel_disp<=51], mlf[vel_disp<=51], marker='v', s=30, color=colours[0], alpha=0.7)

    if plot_medians == True:
        #ax[1,0].plot(bin_center_all, mlf_bin_medians_all, marker='', lw=3, label='Median all KOFFEE fits; R={:.2f}'.format(r_mlf_med_all), color=colours[0])
        #plot the medians as error bars
        ax[1,0].errorbar(bin_center_all, mlf_bin_medians_all, yerr=mlf_bin_stdev_all, capsize=3.0, lw=3, ms=5, label='Median all KOFFEE fits; R={:.2f}'.format(r_mlf_med_all), color=colours[0])


    ax[1,0].plot(sfr_surface_density_kim, np.log10(mlf_Kim), ':k', label='Kim+20, $\eta \propto \Sigma_{SFR}^{-0.44}$')

    if plot_data_fits == True:
        #fit for all
        ax[1,0].plot(sfr_linspace, np.log10(pf.fitting_function(sfr_linspace, *popt_mlf_all)), 'r-', label='Fit: $\eta=%5.0f\pm$%2.0f $\Sigma_{SFR}^{%5.2f \pm %5.2f}$' % (popt_mlf_all[0], np.sqrt(np.diag(pcov_mlf_all))[0], popt_mlf_all[1], np.sqrt(np.diag(pcov_mlf_all))[1]))
        #fit for medians
        ax[1,0].plot(sfr_linspace, np.log10(pf.fitting_function(sfr_linspace, *popt_mlf_all_medians)), 'r--', label='Median Fit: $\eta=%5.0f\pm$%2.0f $\Sigma_{SFR}^{%5.2f \pm %5.2f}$' %(popt_mlf_all_medians[0], np.sqrt(np.diag(pcov_mlf_all_medians))[0], popt_mlf_all_medians[1], np.sqrt(np.diag(pcov_mlf_all_medians))[1]))

    ax[1,0].errorbar(0.05, np.nanmin(mlf), xerr=np.nanmedian(sig_sfr_err), yerr=[[np.nanmedian(mlf_err_min)], [np.nanmedian(mlf_err_max)]], c='k')


    #ax[1,0].set_ylim(np.nanmin(mlf)-0.1, np.nanmax(mlf)+0.5)
    ax[1,0].set_xscale('log')
    if xlim_vals:
        ax[1,0].set_xlim(xlim_vals[0], xlim_vals[1])
    else:
        ax[1,0].set_xlim(np.nanmin(sig_sfr)-0.002, np.nanmax(sig_sfr+2.0))
        #save the xlim values for the comparison figure
        xlim_vals = [np.nanmin(sig_sfr)-0.002, np.nanmax(sig_sfr)+2.0]

    ax[1,0].set_ylim((np.nanmin(mlf)+np.nanmedian(mlf_err_max)-0.1), np.nanmax(mlf)+0.8)
    lgnd = ax[1,0].legend(frameon=True, fontsize='small', loc='upper right', framealpha=0.5)
    #lgnd.legendHandles[0]._legmarker.set_markersize(3)
    ax[1,0].set_ylabel(r'Log($\eta (\frac{500~\rm pc}{R_{\rm out}})$)')
    ax[1,0].set_xlabel('$\Sigma_{SFR}$ [M$_\odot$ yr$^{-1}$ kpc$^{-2}$]')



    #plot points within 90% radius
    #if plot_medians == True:
        #ax[1,1].fill_between(bin_center_physical, mlf_bin_lower_q_physical, mlf_bin_upper_q_physical, color=colours[1], alpha=0.3)

    #ax[1,1].scatter(sig_sfr[radius>6.1], mlf[radius>6.1], marker='o', s=10, label='All KOFFEE fits', edgecolors=colours[0], alpha=0.3, facecolors='none')
    #ax[1,1].scatter(sig_sfr[vel_disp<=51], mlf[vel_disp<=51], marker='v', s=10, edgecolors=colours[0], alpha=0.3, facecolors='none')
    ax[1,1].scatter(sig_sfr[physical_mask], mlf[physical_mask], marker='o', s=30, label='Selected KOFFEE fits; R={:.2f}'.format(r_mlf_physical), color=colours[1], alpha=0.7)

    if plot_medians == True:
        #ax[1,1].plot(bin_center_physical, mlf_bin_medians_physical, marker='', lw=3, label='Median of selected KOFFEE fits; R={:.2f}'.format(r_mlf_med_physical), color=colours[1])
        ax[1,1].errorbar(bin_center_physical, mlf_bin_medians_physical, yerr=mlf_bin_stdev_physical, lw=3, capsize=3.0, ms=5, label='Median of selected KOFFEE fits; R={:.2f}'.format(r_mlf_med_physical), color=colours[1])

    ax[1,1].plot(sfr_surface_density_kim, np.log10(mlf_Kim), ':k')

    if plot_data_fits == True:
        #fit for all
        ax[1,1].plot(sfr_linspace, np.log10(pf.fitting_function(sfr_linspace, *popt_mlf_physical)), 'r-', label='Fit: $\eta=%5.0f\pm$%2.0f $\Sigma_{SFR}^{%5.2f \pm %5.2f}$' % (popt_mlf_physical[0], np.sqrt(np.diag(pcov_mlf_physical))[0], popt_mlf_physical[1], np.sqrt(np.diag(pcov_mlf_physical))[1]))
        #fit for medians
        ax[1,1].plot(sfr_linspace, np.log10(pf.fitting_function(sfr_linspace, *popt_mlf_physical_medians)), 'r--', label='Median Fit: $\eta=%5.0f\pm$%2.0f $\Sigma_{SFR}^{%5.2f \pm %5.2f}$' %(popt_mlf_physical_medians[0], np.sqrt(np.diag(pcov_mlf_physical_medians))[0], popt_mlf_physical_medians[1], np.sqrt(np.diag(pcov_mlf_physical_medians))[1]))

    ax[1,1].errorbar(0.05, np.nanmin(mlf), xerr=np.nanmedian(sig_sfr_err[physical_mask]), yerr=[[np.nanmedian(mlf_err_min[physical_mask])], [np.nanmedian(mlf_err_max[physical_mask])]], c='k')

    #ax[0,1].set_xscale('log')
    lgnd = ax[1,1].legend(frameon=True, fontsize='small', loc='upper right', framealpha=0.5)
    #lgnd.legendHandles[0]._legmarker.set_markersize(3)
    ax[1,1].set_xlabel('$\Sigma_{SFR}$ [M$_\odot$ yr$^{-1}$ kpc$^{-2}$]')



    #plot points with strong BIC values
    #if plot_medians == True:
        #ax[1,2].fill_between(bin_center_strong, mlf_bin_lower_q_strong, mlf_bin_upper_q_strong, color=colours[2], alpha=0.3)

    #ax[1,2].scatter(sig_sfr[~BIC_diff_strong], mlf[~BIC_diff_strong], marker='o', s=10, label='All KOFFEE fits', color=colours[0], alpha=0.3, facecolors='none')
    ax[1,2].scatter(sig_sfr[BIC_diff_strong][vel_disp[BIC_diff_strong]>51], mlf[BIC_diff_strong][vel_disp[BIC_diff_strong]>51], marker='o', s=30, label='Selected KOFFEE fits; R={:.2f}'.format(r_mlf_strong), color=colours[2], alpha=0.8)
    ax[1,2].scatter(sig_sfr[BIC_diff_strong][vel_disp[BIC_diff_strong]<=51], mlf[BIC_diff_strong][vel_disp[BIC_diff_strong]<=51], marker='v', s=30, color=colours[2], alpha=0.8)

    if plot_medians == True:
        #ax[1,2].plot(bin_center_strong, mlf_bin_medians_strong, marker='', lw=3, label='Median of selected KOFFEE fits; R={:.2f}'.format(r_mlf_med_strong), color=colours[2])
        ax[1,2].errorbar(bin_center_strong, mlf_bin_medians_strong, yerr=mlf_bin_stdev_strong, ms=5, lw=3, capsize=3.0, label='Median of selected KOFFEE fits; R={:.2f}'.format(r_mlf_med_strong), color=colours[2])

    ax[1,2].plot(sfr_surface_density_kim, np.log10(mlf_Kim), ':k')

    if plot_data_fits == True:
        #fit for all
        ax[1,2].plot(sfr_linspace, np.log10(pf.fitting_function(sfr_linspace, *popt_mlf_strong)), 'r-', label='Fit: $\eta=%5.0f\pm$%2.0f $\Sigma_{SFR}^{%5.2f \pm %5.2f}$' % (popt_mlf_strong[0], np.sqrt(np.diag(pcov_mlf_strong))[0], popt_mlf_strong[1], np.sqrt(np.diag(pcov_mlf_strong))[1]))
        #fit for medians
        ax[1,2].plot(sfr_linspace, np.log10(pf.fitting_function(sfr_linspace, *popt_mlf_strong_medians)), 'r--', label='Median Fit: $\eta=%5.0f\pm$%2.0f $\Sigma_{SFR}^{%5.2f \pm %5.2f}$' %(popt_mlf_strong_medians[0], np.sqrt(np.diag(pcov_mlf_strong_medians))[0], popt_mlf_strong_medians[1], np.sqrt(np.diag(pcov_mlf_strong_medians))[1]))

    ax[1,2].errorbar(0.05, np.nanmin(mlf), xerr=np.nanmedian(sig_sfr_err[BIC_diff_strong]), yerr=[[np.nanmedian(mlf_err_min[BIC_diff_strong])], [np.nanmedian(mlf_err_max[BIC_diff_strong])]], c='k')

    #ax[0,1].set_xscale('log')
    lgnd = ax[1,2].legend(frameon=True, fontsize='small', loc='upper right', framealpha=0.5)
    #lgnd.legendHandles[0]._legmarker.set_markersize(3)
    ax[1,2].set_xlabel('$\Sigma_{SFR}$ [M$_\odot$ yr$^{-1}$ kpc$^{-2}$]')




    #plot all points
    #if plot_medians == True:
        #ax[0,0].fill_between(bin_center_all, flux_bin_lower_q_all, flux_bin_upper_q_all, color=colours[0], alpha=0.3)

    ax[0,0].scatter(sig_sfr[vel_disp>51], flux_ratio[vel_disp>51], marker='o', s=30, label='All KOFFEE fits; R={:.2f}'.format(r_flux_all), color=colours[0], alpha=0.7)
    ax[0,0].scatter(sig_sfr[vel_disp<=51], flux_ratio[vel_disp<=51], marker='v', s=30, color=colours[0], alpha=0.7)

    if plot_medians == True:
        #ax[0,0].plot(bin_center_all, flux_bin_medians_all, marker='', lw=3, label='Median all KOFFEE fits; R={:.2f}'.format(r_flux_med_all), color=colours[0])
        ax[0,0].errorbar(bin_center_all, flux_bin_medians_all, yerr=flux_bin_stdev_all, ms=5, lw=3, capsize=3.0, label='Median all KOFFEE fits; R={:.2f}'.format(r_flux_med_all), color=colours[0])

    ax[0,0].errorbar(0.05, np.nanmin(flux_ratio)+0.1, xerr=np.nanmedian(sig_sfr_err), yerr=np.nanmedian(flux_error), c='k')

    ax[0,0].set_ylim((np.nanmin(flux_ratio)+np.nanmedian(flux_error)-0.1), np.nanmax(flux_ratio)+0.6)
    lgnd = ax[0,0].legend(frameon=True, fontsize='small', loc='upper right', framealpha=0.5, edgecolor=None)
    #lgnd.legendHandles[0]._legmarker.set_markersize(3)
    ax[0,0].set_ylabel(r'H$\beta$ Log(F$_{\rm broad}$/F$_{\rm narrow}$)')
    ax[0,0].set_title('S/N > 20 and $\delta_{BIC}$>10')



    #plot points within 90% radius
    #if plot_medians == True:
        #ax[0,1].fill_between(bin_center_physical, flux_bin_lower_q_physical, flux_bin_upper_q_physical, color=colours[1], alpha=0.3)

    #ax[0,1].scatter(sig_sfr[radius>6.1], flux_ratio[radius>6.1], marker='o', s=10, label='All KOFFEE fits', edgecolors=colours[0], alpha=0.3, facecolors='none')
    #ax[0,1].scatter(sig_sfr[vel_disp<=51], flux_ratio[vel_disp<=51], marker='v', s=10, edgecolors=colours[0], alpha=0.3, facecolors='none')
    ax[0,1].scatter(sig_sfr[physical_mask], flux_ratio[physical_mask], marker='o', s=30, label='Selected KOFFEE fits; R={:.2f}'.format(r_flux_physical), color=colours[1], alpha=0.7)

    if plot_medians == True:
        #ax[0,1].plot(bin_center_physical, flux_bin_medians_physical, marker='', lw=3, label='Median of selected KOFFEE fits; R={:.2f}'.format(r_flux_med_physical), color=colours[1])
        ax[0,1].errorbar(bin_center_physical, flux_bin_medians_physical, yerr=flux_bin_stdev_physical, ms=5, lw=3, capsize=3.0, label='Median of selected KOFFEE fits; R={:.2f}'.format(r_flux_med_physical), color=colours[1])

    ax[0,1].errorbar(0.05, np.nanmin(flux_ratio)+0.1, xerr=np.nanmedian(sig_sfr_err[physical_mask]), yerr=np.nanmedian(flux_error[physical_mask]), c='k')

    lgnd = ax[0,1].legend(frameon=True, fontsize='small', loc='upper right', framealpha=0.5, edgecolor=None)
    #lgnd.legendHandles[0]._legmarker.set_markersize(3)
    ax[0,1].set_title(r'$\delta_{BIC}$>10, $r$<$r_{90}$ and $\sigma_{broad}$>$\sigma_{inst}$')



    #plot points with strong BIC values
    #if plot_medians == True:
        #ax[0,2].fill_between(bin_center_strong, flux_bin_lower_q_strong, flux_bin_upper_q_strong, color=colours[2], alpha=0.3)

    #ax[0,2].scatter(sig_sfr[~BIC_diff_strong], flux_ratio[~BIC_diff_strong], marker='o', s=10, label='All KOFFEE fits', edgecolors=colours[0], alpha=0.3, facecolors='none')
    ax[0,2].scatter(sig_sfr[BIC_diff_strong][vel_disp[BIC_diff_strong]>51], flux_ratio[BIC_diff_strong][vel_disp[BIC_diff_strong]>51], marker='o', s=30, label='Selected KOFFEE fits; R={:.2f}'.format(r_flux_strong), color=colours[2], alpha=0.7)
    ax[0,2].scatter(sig_sfr[BIC_diff_strong][vel_disp[BIC_diff_strong]<=51], flux_ratio[BIC_diff_strong][vel_disp[BIC_diff_strong]<=51], marker='v', s=30, color=colours[2], alpha=0.7)

    if plot_medians == True:
        #ax[0,2].plot(bin_center_strong, flux_bin_medians_strong, marker='', lw=3, label='Median of selected KOFFEE fits; R={:.2f}'.format(r_flux_med_strong), color=colours[2])
        ax[0,2].errorbar(bin_center_strong, flux_bin_medians_strong, yerr=flux_bin_stdev_strong, ms=5, lw=3, capsize=5.0, label='Median of selected KOFFEE fits; R={:.2f}'.format(r_flux_med_strong), color=colours[2])

    ax[0,2].errorbar(0.05, np.nanmin(flux_ratio)+0.1, xerr=np.nanmedian(sig_sfr_err[BIC_diff_strong]), yerr=np.nanmedian(flux_error[BIC_diff_strong]), c='k')

    lgnd = ax[0,2].legend(frameon=True, fontsize='small', loc='upper right', framealpha=0.5, edgecolor=None)
    #lgnd.legendHandles[0]._legmarker.set_markersize(3)
    ax[0,2].set_title('strongly likely BIC $\delta_{BIC}$>50')

    plt.subplots_adjust(left=0.07, right=0.99, top=0.96, bottom=0.07, wspace=0.04, hspace=0.04)

    plt.show()

    return xlim_vals



#Figure 7
def maps_of_IRAS08(halpha_fits_file, fuv_fits_file, f550m_fits_file, outflow_velocity_fits_file, flux_ratio_fits_file, flux_broad_fits_file, flux_narrow_fits_file, m_out_fits_file, mlf_fits_file, radius):
    """"
    Maps the results for IRAS08 and some flux distributions from fits files

    Parameters
    ----------
    halpha_fits_file : string
        location of the fits file with the Halpha flux

    fuv_fits_file : string
        location of the fits file with the FUV flux

    f550m_fits_file : string
        location of the fits file with the f550m filter flux

    outflow_velocity_fits_file : string
        location of the fits file with the outflow velocity measured with KOFFEE
        (must be same shape as statistical_results)

    flux_ratio_fits_file : string
        location of the fits file with the broad-to-narrow flux ratio measured
        with KOFFEE (must be same shape as statistical_results)

    flux_broad_fits_file : string
        location of the fits file with the broad flux measured with KOFFEE
        (must be same shape as statistical_results)

    flux_narrow_fits_file : string
        location of the fits file with the narrow flux measured with KOFFEE
        (must be same shape as statistical_results)

    m_out_fits_file : string
        location of the fits file with the mass outflow rate measured with KOFFEE
        (must be same shape as statistical_results)

    mlf_fits_file : string
        location of the fits file with the mass loading factor measured with KOFFEE
        (must be same shape as statistical_results)

    radius : :obj:'~numpy.ndarray'
        array of galaxy radius values

    Returns
    -------
    A nine-panel figure with the maps of flux from different emission lines and
    continuum areas, and maps of the results from KOFFEE
    """
    #read in fits files
    #need to shift the IRAS08 KCWI data to match the WCS with the HST data
    halpha_data, halpha_header, halpha_wcs = pf.read_in_create_wcs(halpha_fits_file)
    fuv_data, fuv_header, fuv_wcs = pf.read_in_create_wcs(fuv_fits_file)
    f550_data, f550_header, f550_wcs = pf.read_in_create_wcs(f550m_fits_file, index=1)
    vel_out, vel_out_header, vel_out_wcs = pf.read_in_create_wcs(outflow_velocity_fits_file, shift=['CRPIX2', 32.0])
    flux_ratio, flux_ratio_header, flux_ratio_wcs = pf.read_in_create_wcs(flux_ratio_fits_file, shift=['CRPIX2', 32.0])
    flux_broad, flux_broad_header, flux_broad_wcs = pf.read_in_create_wcs(flux_broad_fits_file, shift=['CRPIX2', 32.0])
    flux_narrow, flux_narrow_header, flux_narrow_wcs = pf.read_in_create_wcs(flux_narrow_fits_file, shift=['CRPIX2', 32.0])
    mlf, mlf_header, mlf_wcs = pf.read_in_create_wcs(mlf_fits_file, shift=['CRPIX2', 32.0])
    m_out, m_out_header, m_out_wcs = pf.read_in_create_wcs(m_out_fits_file, shift=['CRPIX2', 32.0])

    #take the log of the velocity and the flux ratio
    vel_out = np.log10(vel_out)
    flux_ratio = np.log10(flux_ratio)
    flux_broad = np.log10(flux_broad)
    flux_narrow = np.log10(flux_narrow)
    mlf = np.log10(mlf)
    m_out = np.log10(m_out)

    #creating the x and y limits
    xlim = [4, 16]
    ylim = [2, 58]

    kcwi_low_lim = vel_out_wcs.all_pix2world(xlim[0], ylim[0], 0)
    kcwi_high_lim = vel_out_wcs.all_pix2world(xlim[1], ylim[1], 0)
    print('Limits')
    print(kcwi_low_lim)
    print(kcwi_high_lim)

    kcwi_low_lim_halpha = halpha_wcs.all_world2pix(kcwi_low_lim[0], kcwi_low_lim[1], 0)
    kcwi_high_lim_halpha = halpha_wcs.all_world2pix(kcwi_high_lim[0], kcwi_high_lim[1], 0)
    print(kcwi_low_lim_halpha)
    print(kcwi_high_lim_halpha)

    kcwi_low_lim_fuv = fuv_wcs.all_world2pix(kcwi_low_lim[0], kcwi_low_lim[1], 0)
    kcwi_high_lim_fuv = fuv_wcs.all_world2pix(kcwi_high_lim[0], kcwi_high_lim[1], 0)
    print(kcwi_low_lim_fuv)
    print(kcwi_high_lim_fuv)

    kcwi_low_lim_f550 = f550_wcs.all_world2pix(kcwi_low_lim[0], kcwi_low_lim[1], 0)
    kcwi_high_lim_f550 = f550_wcs.all_world2pix(kcwi_high_lim[0], kcwi_high_lim[1], 0)
    print(kcwi_low_lim_f550)
    print(kcwi_high_lim_f550)


    #find the peak of the OIII flux and convert to wcs
    print('Max flux ratio')
    OIII_peak_pixel = np.argwhere(flux_ratio==np.nanmax(flux_ratio[radius<6.1]))
    print(OIII_peak_pixel)

    OIII_local_maxima_pixel = argrelextrema(flux_ratio[radius<6.1], np.greater)
    #OIII_max_local_maxima_pixel = np.argwhere(flux_ratio==np.nanmax(flux_ratio[radius<6.1][OIII_local_maxima_pixel]))
    OIII_max_local_maxima_pixel = np.argwhere(flux_ratio==np.sort(flux_ratio[radius<6.1][OIII_local_maxima_pixel])[-2])

    OIII_peak_world = flux_ratio_wcs.all_pix2world(OIII_peak_pixel[0,1], OIII_peak_pixel[0,0], 0)
    print(OIII_peak_world)
    OIII_peak_halpha_pixel = halpha_wcs.all_world2pix(OIII_peak_world[0], OIII_peak_world[1], 0)
    print(OIII_peak_halpha_pixel)
    OIII_peak_fuv_pixel = fuv_wcs.all_world2pix(OIII_peak_world[0], OIII_peak_world[1], 0)
    print(OIII_peak_fuv_pixel)

    OIII_local_max_world = vel_out_wcs.all_pix2world(OIII_max_local_maxima_pixel[0,1], OIII_max_local_maxima_pixel[0,0], 0)#, 0)
    print(OIII_local_max_world)
    OIII_local_max_halpha_pixel = halpha_wcs.all_world2pix(OIII_local_max_world[0], OIII_local_max_world[1], 0)
    print(OIII_local_max_halpha_pixel)
    OIII_local_max_fuv_pixel = fuv_wcs.all_world2pix(OIII_local_max_world[0], OIII_local_max_world[1], 0)
    print(OIII_local_max_fuv_pixel)

    #find the peak of the outflow velocity and convert to wcs
    print('Max outflow velocity')
    outvel_peak_pixel = np.argwhere(vel_out==np.nanmax(vel_out[radius<6.1]))
    print(outvel_peak_pixel)

    outvel_local_maxima_pixel = argrelextrema(vel_out[radius<6.1], np.greater)
    #outvel_max_local_maxima_pixel = np.argwhere(vel_out==np.nanmax(vel_out[outvel_local_maxima_pixel]))
    outvel_max_local_maxima_pixel = np.argwhere(vel_out==np.sort(vel_out[radius<6.1][outvel_local_maxima_pixel])[-2])

    out_vel_peak_world = vel_out_wcs.all_pix2world(outvel_peak_pixel[0,1], outvel_peak_pixel[0,0], 0)#, 0)
    print(out_vel_peak_world)
    out_vel_peak_halpha_pixel = halpha_wcs.all_world2pix(out_vel_peak_world[0], out_vel_peak_world[1], 0)
    print(out_vel_peak_halpha_pixel)
    out_vel_peak_fuv_pixel = fuv_wcs.all_world2pix(out_vel_peak_world[0], out_vel_peak_world[1], 0)
    print(out_vel_peak_fuv_pixel)

    out_vel_local_max_world = vel_out_wcs.all_pix2world(outvel_max_local_maxima_pixel[0,1], outvel_max_local_maxima_pixel[0,0], 0)#, 0)
    print(out_vel_local_max_world)
    out_vel_local_max_halpha_pixel = halpha_wcs.all_world2pix(out_vel_local_max_world[0], out_vel_local_max_world[1], 0)
    print(out_vel_local_max_halpha_pixel)
    out_vel_local_max_fuv_pixel = fuv_wcs.all_world2pix(out_vel_local_max_world[0], out_vel_local_max_world[1], 0)
    print(out_vel_local_max_fuv_pixel)

    print('Max m_out')
    m_out_peak_pixel = np.argwhere(m_out==np.nanmax(m_out[radius<6.1]))
    print(m_out_peak_pixel)

    m_out_local_maxima_pixel = argrelextrema(m_out[radius<6.1], np.greater)
    #OIII_max_local_maxima_pixel = np.argwhere(flux_ratio==np.nanmax(flux_ratio[radius<6.1][OIII_local_maxima_pixel]))
    m_out_max_local_maxima_pixel = np.argwhere(m_out==np.sort(m_out[radius<6.1][m_out_local_maxima_pixel])[-2])

    m_out_peak_world = m_out_wcs.all_pix2world(m_out_peak_pixel[0,1], m_out_peak_pixel[0,0], 0)
    print(m_out_peak_world)
    m_out_peak_halpha_pixel = halpha_wcs.all_world2pix(m_out_peak_world[0], m_out_peak_world[1], 0)
    print(m_out_peak_halpha_pixel)
    m_out_peak_fuv_pixel = fuv_wcs.all_world2pix(m_out_peak_world[0], m_out_peak_world[1], 0)
    print(m_out_peak_fuv_pixel)

    m_out_local_max_world = m_out_wcs.all_pix2world(m_out_max_local_maxima_pixel[0,1], m_out_max_local_maxima_pixel[0,0], 0)#, 0)
    print(m_out_local_max_world)
    m_out_local_max_halpha_pixel = halpha_wcs.all_world2pix(m_out_local_max_world[0], m_out_local_max_world[1], 0)
    print(m_out_local_max_halpha_pixel)
    m_out_local_max_fuv_pixel = fuv_wcs.all_world2pix(m_out_local_max_world[0], m_out_local_max_world[1], 0)
    print(m_out_local_max_fuv_pixel)

    print('Max f550')
    f550_peak_pixel = np.argwhere(f550_data==np.nanmax(f550_data))
    print(f550_peak_pixel)

    f550_peak_world = f550_wcs.all_pix2world(f550_peak_pixel[0,1], f550_peak_pixel[0,0], 0)
    print(f550_peak_world)



    #calculate the beginning and end of 5 arcsec
    halpha_10arcsec_pixel_length = abs(5/(halpha_header['CD1_1']*60*60))

    halpha_start_10_arcsec_xpixel = kcwi_high_lim_halpha[0]+20
    halpha_start_10_arcsec_ypixel = kcwi_low_lim_halpha[1]+20
    halpha_end_10_arcsec_xpixel = kcwi_high_lim_halpha[0]+20+halpha_10arcsec_pixel_length

    halpha_start_10_arcsec_world = halpha_wcs.all_pix2world(halpha_start_10_arcsec_xpixel, halpha_start_10_arcsec_ypixel, 0)
    halpha_end_10_arcsec_world = halpha_wcs.all_pix2world(halpha_end_10_arcsec_xpixel, halpha_start_10_arcsec_ypixel, 0)

    fuv_start_10_arcsec_pixel = fuv_wcs.all_world2pix(halpha_start_10_arcsec_world[0], halpha_start_10_arcsec_world[1], 0)
    fuv_end_10_arcsec_pixel = fuv_wcs.all_world2pix(halpha_end_10_arcsec_world[0], halpha_end_10_arcsec_world[1], 0)

    f550_start_10_arcsec_pixel = f550_wcs.all_world2pix(halpha_start_10_arcsec_world[0], halpha_start_10_arcsec_world[1], 0)
    f550_end_10_arcsec_pixel = f550_wcs.all_world2pix(halpha_end_10_arcsec_world[0], halpha_end_10_arcsec_world[1], 0)



    #create the figure
    plt.rcParams.update(pf.get_rc_params())

    plt.figure(figsize=(10,10))#constrained_layout=True)

    ax1 = plt.subplot(331, projection=halpha_wcs)
    ax1.set_facecolor('black')
    #do the plotting
    halpha_map = ax1.imshow(np.log10(halpha_data), origin='lower', cmap=cmr.ember, vmin=-1.75, vmax=-0.25)
    #ax1.imshow(np.log10(halpha_data), origin='lower', cmap=cmr.ember, vmin=-20, vmax=-17.5)
    #ax1.hlines(halpha_start_10_arcsec_ypixel, halpha_start_10_arcsec_xpixel, halpha_end_10_arcsec_xpixel, colors='white')
    #ax1.scatter(OIII_peak_halpha_pixel[0], OIII_peak_halpha_pixel[1], c='white', marker='x', s=20)
    #ax1.scatter(OIII_local_max_halpha_pixel[0], OIII_local_max_halpha_pixel[1], c='white', marker='o', s=20)
    #ax1.scatter(out_vel_peak_halpha_pixel[0], out_vel_peak_halpha_pixel[1], c='grey', marker='x', s=20)
    #ax1.scatter(out_vel_local_max_halpha_pixel[0], out_vel_local_max_halpha_pixel[1], c='grey', marker='o', s=20)
    #ax1.scatter(mlf_peak_halpha_pixel[0], mlf_peak_halpha_pixel[1], c='white', marker='x', s=20)

    #this arrow works!
    #ax1.arrow(m_out_peak_halpha_pixel[0]-55, m_out_peak_halpha_pixel[1]+55, 50, -50, width=5, length_includes_head=True, color='white')

    lon1 = ax1.coords[0]
    lat1 = ax1.coords[1]
    lon1.set_ticks_visible(False)
    lon1.set_ticklabel_visible(False)
    lat1.set_ticks_visible(False)
    lat1.set_ticklabel_visible(False)
    ax1.set_title(r'H$\alpha$ Flux')
    ax1.set_xlim(kcwi_high_lim_halpha[0], kcwi_low_lim_halpha[0])
    ax1.set_ylim(kcwi_low_lim_halpha[1], kcwi_high_lim_halpha[1])
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("left", size="20%", pad=0.1)
    cax.axis('off')

    #cbar = plt.colorbar(halpha_map, ax=ax1, shrink=0.8)

    ax3 = plt.subplot(336, projection=flux_ratio_wcs, slices=('y', 'x'))
    flux_ratio_spax = ax3.imshow(flux_ratio.T, origin='lower', aspect=flux_ratio_header['CD2_1']/flux_ratio_header['CD1_2'], cmap=cmr.gem, vmin=-1.5, vmax=0.1)
    #ax3.hlines(ymin+0.75, xmin+4, xmin+4+koffee_10arcsec_pixel_length, colors='black')
    #ax3.hlines(koffee_start_10_arcsec_pixel[1], koffee_start_10_arcsec_pixel[0], koffee_end_10_arcsec_pixel[0], colors='black')
    #ax3.text(low_lim_rad[0]-5, high_lim_rad[1]-5, '[OIII]', c='black')
    ax3.grid(False)
    ax3.coords.grid(False)
    lon3 = ax3.coords[0]
    lat3 = ax3.coords[1]
    lon3.set_ticks_visible(False)
    lon3.set_ticklabel_visible(False)
    lat3.set_ticks_visible(False)
    lat3.set_ticklabel_visible(False)

    ax3.set_xlim(ylim[0], ylim[1])
    ax3.set_ylim(xlim[0], xlim[1])
    ax3.invert_xaxis()

    ax3.set_title(r'Log(H$\beta$ F$_{broad}$/F$_{narrow}$)')
    cbar = plt.colorbar(flux_ratio_spax, ax=ax3, shrink=0.8)


    ax2 = plt.subplot(334, projection=fuv_wcs)
    ax2.set_facecolor('black')
    #do the plotting
    fuv_map = ax2.imshow(np.log10(fuv_data), origin='lower', cmap=cmr.ember, vmin=-3, vmax=-0.75)
    #ax2.hlines(fuv_start_10_arcsec_pixel[1], fuv_start_10_arcsec_pixel[0], fuv_end_10_arcsec_pixel[0], colors='white')
    #ax2.text(fuv_start_10_arcsec_pixel[0]+5, fuv_start_10_arcsec_pixel[1]+10, '5" ', c='white')
    #ax2.scatter(OIII_peak_fuv_pixel[0], OIII_peak_fuv_pixel[1], c='white', marker='x', s=20)
    #ax2.scatter(OIII_local_max_fuv_pixel[0], OIII_local_max_fuv_pixel[1], c='white', marker='o', s=20)
    #ax2.scatter(out_vel_peak_fuv_pixel[0], out_vel_peak_fuv_pixel[1], c='grey', marker='x', s=20)
    #ax2.scatter(out_vel_local_max_fuv_pixel[0], out_vel_local_max_fuv_pixel[1], c='grey', marker='o', s=20)
    #ax2.scatter(mlf_peak_fuv_pixel[0], mlf_peak_fuv_pixel[1], c='white', marker='x', s=20)
    #ax2.arrow(m_out_peak_fuv_pixel[0]-55, m_out_peak_fuv_pixel[1]+55, 50, -50, width=5, length_includes_head=True, color='white')
    #ax2.arrow(m_out_peak_fuv_pixel[0]-50, m_out_peak_fuv_pixel[1]-50, 50, 50, width=5, length_includes_head=True, color='white')
    lon2 = ax2.coords[0]
    lat2 = ax2.coords[1]
    lon2.set_ticks_visible(False)
    lon2.set_ticklabel_visible(False)
    lat2.set_ticks_visible(False)
    lat2.set_ticklabel_visible(False)
    ax2.set_title('FUV flux')
    ax2.set_xlim(kcwi_high_lim_fuv[0], kcwi_low_lim_fuv[0])
    ax2.set_ylim(kcwi_low_lim_fuv[1], kcwi_high_lim_fuv[1])
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes("left", size="20%", pad=0.1)
    cax.axis('off')


    ax4 = plt.subplot(335, projection=vel_out_wcs, slices=('y', 'x'))
    #outvel_spax = bdpk.display_pixels(xx_flat_out, yy_flat_out, vel_out, angle=360, axes=ax4, cmap=cmr.gem, vmin=2.2, vmax=2.65)
    outvel_spax = ax4.imshow(vel_out.T, origin='lower', aspect=vel_out_header['CD2_1']/vel_out_header['CD1_2'], cmap=cmr.gem, vmin=2.2, vmax=2.65)
    #ax4.hlines(ymin+0.75, xmin+4, xmin+4+10, colors='black')
    #ax4.arrow(out_vel_peak_[0], out_vel_peak_world[1], 5, 5, color='k')
    ax4.grid(False)
    ax4.coords.grid(False)
    #ax4.get_xaxis().set_visible(False)
    #ax4.get_yaxis().set_visible(False)
    lon4 = ax4.coords[0]
    lat4 = ax4.coords[1]
    lon4.set_ticks_visible(False)
    lon4.set_ticklabel_visible(False)
    lat4.set_ticks_visible(False)
    lat4.set_ticklabel_visible(False)

    ax4.set_xlim(ylim[0], ylim[1])
    ax4.set_ylim(xlim[0], xlim[1])
    ax4.invert_xaxis()
    ax4.set_title(r'Log($v_{out}$)')
    cbar = plt.colorbar(outvel_spax, ax=ax4, shrink=0.8)
    cbar.set_label('Log(km s$^{-1}$)')

    ax5 = plt.subplot(333, projection=flux_broad_wcs, slices=('y', 'x'))
    #flux_broad_spax = bdpk.display_pixels(xx_flat_out, yy_flat_out, flux_broad, angle=360, axes=ax5, cmap=cmr.gem, vmin=-1.5, vmax=1.7)
    flux_broad_spax = ax5.imshow(flux_broad.T, origin='lower', aspect=flux_broad_header['CD2_1']/flux_broad_header['CD1_2'], cmap=cmr.gem, vmin=-0.5, vmax=1.7)
    #ax5.hlines(ymin+0.75, xmin+4, xmin+4+10, colors='black')
    ax5.grid(False)
    ax5.coords.grid(False)
    #ax5.get_xaxis().set_visible(False)
    #ax5.get_yaxis().set_visible(False)
    lon5 = ax5.coords[0]
    lat5 = ax5.coords[1]
    lon5.set_ticks_visible(False)
    lon5.set_ticklabel_visible(False)
    lat5.set_ticks_visible(False)
    lat5.set_ticklabel_visible(False)

    ax5.set_xlim(ylim[0], ylim[1])
    ax5.set_ylim(xlim[0], xlim[1])
    ax5.invert_xaxis()
    ax5.set_title(r'Log(H$\beta$ F$_{broad}$)')
    cbar = plt.colorbar(flux_broad_spax, ax=ax5, shrink=0.8)
    #10^-16 erg/s/cm^2
    cbar.set_label(r'Log($10^{-16}$ erg s$^{-1}$ cm$^{-2}$)')

    ax6 = plt.subplot(332, projection=flux_narrow_wcs, slices=('y', 'x'))
    #flux_narrow_spax = bdpk.display_pixels(xx_flat_out, yy_flat_out, flux_narrow, angle=360, axes=ax6, cmap=cmr.gem, vmin=-0.0, vmax=2.3)
    flux_narrow_spax = ax6.imshow(flux_narrow.T, origin='lower', aspect=flux_narrow_header['CD2_1']/flux_narrow_header['CD1_2'], cmap=cmr.gem, vmin=0.0, vmax=2.3)
    ax6.grid(False)
    ax6.coords.grid(False)
    lon6 = ax6.coords[0]
    lat6 = ax6.coords[1]
    lon6.set_ticks_visible(False)
    lon6.set_ticklabel_visible(False)
    lat6.set_ticks_visible(False)
    lat6.set_ticklabel_visible(False)

    ax6.set_xlim(ylim[0], ylim[1])
    ax6.set_ylim(xlim[0], xlim[1])
    ax6.invert_xaxis()
    ax6.set_title(r'Log(H$\beta$ F$_{narrow}$)')
    cbar = plt.colorbar(flux_narrow_spax, ax=ax6, shrink=0.8)
    cbar.set_label(r'Log($10^{-16}$ erg s$^{-1}$ cm$^{-2}$)')

    ax7 = plt.subplot(337, projection=f550_wcs)
    ax7.set_facecolor('black')
    #do the plotting
    f550_map = ax7.imshow(np.log10(f550_data), origin='lower', cmap=cmr.ember, vmin=-1.5, vmax=0.1)
    #ax2.arrow(mlf_peak_fuv_pixel[0]-55, mlf_peak_fuv_pixel[1]+55, 50, -50, width=5, length_includes_head=True, color='white')
    ax7.hlines(f550_start_10_arcsec_pixel[1], f550_start_10_arcsec_pixel[0], f550_end_10_arcsec_pixel[0], colors='white')
    ax7.text(f550_start_10_arcsec_pixel[0]+5, f550_start_10_arcsec_pixel[1]+10, '2 kpc ', c='white')
    lon7 = ax7.coords[0]
    lat7 = ax7.coords[1]
    lon7.set_ticks_visible(False)
    lon7.set_ticklabel_visible(False)
    lat7.set_ticks_visible(False)
    lat7.set_ticklabel_visible(False)
    ax7.set_title('F550m flux')
    ax7.set_xlim(kcwi_high_lim_f550[0], kcwi_low_lim_f550[0])
    ax7.set_ylim(kcwi_low_lim_f550[1], kcwi_high_lim_f550[1])
    divider = make_axes_locatable(ax7)
    cax = divider.append_axes("left", size="20%", pad=0.1)
    cax.axis('off')

    ax8 = plt.subplot(338, projection=m_out_wcs, slices=('y', 'x'))
    #m_out_spax = bdpk.display_pixels(xx_flat_out, yy_flat_out, m_out, angle=360, axes=ax8, cmap=cmr.gem, vmin=-3.2, vmax=-0.9)
    m_out_spax = ax8.imshow(m_out.T, origin='lower', aspect=m_out_header['CD2_1']/m_out_header['CD1_2'], cmap=cmr.gem, vmin=-2.8, vmax=-0.5)
    #ax8.hlines(ymin+0.75, xmin+4, xmin+4+10, colors='black')
    #ax8.arrow(m_out_peak_pixel[0][0]+5, m_out_peak_pixel[0][1]-2, -5, 2, width=0.2, length_includes_head=True, color='k')
    ax8.grid(False)
    ax8.coords.grid(False)
    lon8 = ax8.coords[0]
    lat8 = ax8.coords[1]
    lon8.set_ticks_visible(False)
    lon8.set_ticklabel_visible(False)
    lat8.set_ticks_visible(False)
    lat8.set_ticklabel_visible(False)

    ax8.set_xlim(ylim[0], ylim[1])
    ax8.set_ylim(xlim[0], xlim[1])
    ax8.invert_xaxis()
    ax8.set_title(r'Log($\dot{M}_{out}$) ')
    cbar = plt.colorbar(m_out_spax, ax=ax8, shrink=0.8)
    cbar.set_label(r'Log(M$_\odot$ yr$^{-1}$)')

    ax9 = plt.subplot(339, projection=mlf_wcs, slices=('y', 'x'))
    #mlf_spax = bdpk.display_pixels(xx_flat_out, yy_flat_out, mlf, angle=360, axes=ax9, cmap=cmr.gem, vmin=-1.5, vmax=0.3)
    mlf_spax = ax9.imshow(mlf.T, origin='lower', aspect=mlf_header['CD2_1']/mlf_header['CD1_2'], cmap=cmr.gem, vmin=-1.1, vmax=0.7)
    #ax9.hlines(ymin+0.75, xmin+4, xmin+4+10, colors='black')
    ax9.grid(False)
    ax9.coords.grid(False)
    lon9 = ax9.coords[0]
    lat9 = ax9.coords[1]
    lon9.set_ticks_visible(False)
    lon9.set_ticklabel_visible(False)
    lat9.set_ticks_visible(False)
    lat9.set_ticklabel_visible(False)

    ax9.set_xlim(ylim[0], ylim[1])
    ax9.set_ylim(xlim[0], xlim[1])
    ax9.invert_xaxis()
    ax9.set_title(r'Log($\eta$) ')
    cbar = plt.colorbar(mlf_spax, ax=ax9, shrink=0.8)
    #cbar.set_label(r'Log(M$_\odot$ yr$^{-1}$)')

    plt.subplots_adjust(left=0.0, right=0.96, top=0.99, bottom=0.0, wspace=0.1, hspace=0.0)

    plt.show()



#Figure 8
def plot_mlf_model_rad_singlepanel(OIII_outflow_results, OIII_outflow_error, hbeta_outflow_results, hbeta_outflow_error, hbeta_no_outflow_results, hbeta_no_outflow_error, BIC_outflow, BIC_no_outflow, statistical_results, z, radius, header, compare='divide', plot_medians=True):
    """
    Plots the mass loading factor values found with KOFFEE either divided by or
    subtracted from the expected values calculated with the model from
    Kim et al. 2020 against the galaxy radius

    Parameters
    ----------
    OIII_outflow_results : :obj:'~numpy.ndarray'
        array of outflow results from KOFFEE for OIII line.  Used to calculate
        the outflow velocity.  Should be (7, statistical_results.shape)

    OIII_outflow_err : :obj:'~numpy.ndarray'
        array of the outflow result errors from KOFFEE for OIII line

    hbeta_outflow_results : :obj:'~numpy.ndarray'
        array of outflow results from KOFFEE for Hbeta line.  Used to calculate
        the Sigma SFR.  Should be (7, statistical_results.shape)

    hbeta_outflow_err : :obj:'~numpy.ndarray'
        array of the outflow result errors from KOFFEE for Hbeta line

    hbeta_no_outflow_results : :obj:'~numpy.ndarray'
        array of single gaussian results from KOFFEE for Hbeta line.  Used to
        calculate the Sigma SFR.  Should be (4, statistical_results.shape)

    hbeta_no_outflow_err : :obj:'~numpy.ndarray'
        array of the single gaussian result errors from KOFFEE for Hbeta line

    BIC_outflow : :obj:'~numpy.ndarray'
        array of BIC values from the double gaussian fits

    BIC_no_outflow : :obj:'~numpy.ndarray'
        array of BIC values from the single gaussian fits

    statistical_results : :obj:'~numpy.ndarray'
        array of statistical results from KOFFEE.

    z : float
        redshift

    radius : :obj:'~numpy.ndarray'
        array of galaxy radius values (must be same shape as statistical_results)

    header : FITS header object
        the header from the fits file

    compare : string
        string defining how to compare the results to the model.  Can be 'divide'
        or 'subtract' (Default='divide')

    Returns
    -------
    A single panel figure of the mass loading factor compared to the expected
    model against the galaxy radius
    """
    #calculate the sfr surface density - using just the systemic line, and including the flux line
    #don't include extinction since this was included in the continuum subtraction using ppxf
    sfr, sfr_err, total_sfr, sfr_surface_density, sfr_surface_density_err = calc_sfr.calc_sfr_koffee(hbeta_outflow_results, hbeta_outflow_error, hbeta_no_outflow_results, hbeta_no_outflow_error, statistical_results, z, header, include_extinction=False, include_outflow=False)

    #calculate the mass loading factor
    mlf, mlf_max, mlf_min = calc_mlf.calc_mass_loading_factor(OIII_outflow_results, OIII_outflow_error, hbeta_outflow_results, hbeta_outflow_error, hbeta_no_outflow_results, hbeta_no_outflow_error, statistical_results, z, header)

    #calculate the velocity dispersion for the masking
    vel_disp, vel_disp_err, vel_diff, vel_diff_err, vel_out, vel_out_err = calc_outvel.calc_outflow_vel(OIII_outflow_results, OIII_outflow_error, statistical_results, z)

    #create the flow mask
    flow_mask = (statistical_results>0) #& (np.isnan(hbeta_outflow_results[3,:,:])==False)


    #flatten all the arrays and get rid of extra spaxels
    sig_sfr = sfr_surface_density[flow_mask]
    sig_sfr_err = sfr_surface_density_err[flow_mask]

    mlf = mlf[flow_mask]
    mlf_max = mlf_max[flow_mask]
    mlf_min = mlf_min[flow_mask]

    radius = radius[flow_mask]

    BIC_outflow = BIC_outflow[flow_mask]
    BIC_no_outflow = BIC_no_outflow[flow_mask]
    vel_disp = vel_disp[flow_mask]

    #calculate the errors
    mlf_err_max = mlf_max - mlf
    mlf_err_min = mlf - mlf_min

    #create BIC diff
    BIC_diff = BIC_outflow - BIC_no_outflow
    BIC_diff_strong = (BIC_diff < -50)

    #physical limits mask -
    #for the radius mask 6.1" is the 90% radius
    #also mask out the fits which lie on the lower limit of dispersion < 51km/s
    physical_mask = (radius < 6.1) & (vel_disp>51)

    logspace_all, bin_center_all, mlf_bin_medians_all, mlf_bin_lower_q_all, mlf_bin_upper_q_all, mlf_bin_stdev_all = pf.binned_median_quantile_log(sig_sfr, mlf, num_bins=5, weights=None, min_bin=None, max_bin=None)

    #divide the mass loading factor by the model
    mlf_model = np.full_like(mlf, np.nan, dtype=np.double)

    for i in np.arange(mlf.shape[0]):
        #calculate the expected mlf at each sigma_sfr
        sigma_sfr_model, mlf_expected = pf.kim_et_al_2020(sig_sfr[i], sig_sfr[i], scale_factor = (10**mlf_bin_medians_all[0])/(bin_center_all[0]**-0.44)) #0.06
        sigma_sfr_model = sigma_sfr_model[0]
        mlf_expected = mlf_expected[0]
        #divide the mlf by the expected mlf
        if compare == 'divide':
            mlf_model[i] = np.log10(mlf[i]/mlf_expected)
        elif compare == 'subtract':
            mlf_model[i] = mlf[i] - mlf_expected

    print(mlf_model)



    #do the calculations for all the bins
    num_bins = 5
    min_bin = None #-0.05
    max_bin = None #0.6

    linspace_all, bin_center_all, mlf_bin_medians_all, mlf_bin_lower_q_all, mlf_bin_upper_q_all, mlf_bin_stdev_all = pf.binned_median_quantile_lin(radius, mlf_model, num_bins=num_bins, weights=None, min_bin=min_bin, max_bin=max_bin)

    print(bin_center_all, mlf_bin_medians_all)

    #use constant to bring the first median point to zero
    zero_constant = mlf_bin_medians_all[0]
    mlf_bin_medians_all = mlf_bin_medians_all - zero_constant
    mlf_bin_lower_q_all = mlf_bin_lower_q_all - zero_constant
    mlf_bin_upper_q_all = mlf_bin_upper_q_all - zero_constant
    mlf_model = mlf_model - zero_constant


    #calculate the r value for the median values
    #model is already logged
    r_mlf_med_all, p_value_mlf_all = pf.pearson_correlation(bin_center_all, mlf_bin_medians_all)
    #r_mlf_med_all, p_value_mlf_all = pearson_correlation(bin_center_all, mlf_model_medians)

    #calculate the r value for all the values
    r_mlf_all, p_value_mlf_all = pf.pearson_correlation(radius[~np.isnan(mlf_model)], mlf_model[~np.isnan(mlf_model)])



    #calculate the proper distance
    proper_dist = cosmo.kpc_proper_per_arcmin(z).to(u.kpc/u.arcsec)

    #convert radius to kpc
    radius = radius*proper_dist
    bin_center_all = bin_center_all*proper_dist



    #print average numbers for the different panels
    print('Number of spaxels in the first panel', mlf.shape)
    print('All spaxels median mlf:', np.nanmedian(mlf))
    print('All spaxels standard deviation mlf:', np.nanstd(mlf))
    print('')

    print('All spaxels median mlf/model:', np.nanmedian(mlf_model))
    print('All spaxels standard deviation mlf/model:', np.nanstd(mlf_model))
    print('')



    #-------
    #plot it
    #-------
    plt.rcParams.update(pf.get_rc_params())
    fig, ax = plt.subplots(nrows=1, ncols=1, sharex=True, sharey='row', figsize=(4,4), constrained_layout=True)

    #get colours from cmasher
    colours = cmr.take_cmap_colors('cmr.gem', 3, cmap_range=(0.25, 0.85), return_fmt='hex')

    #plot all points
    #if plot_medians == True:
        #ax.fill_between(bin_center_all, mlf_bin_lower_q_all, mlf_bin_upper_q_all, color=colours[0], alpha=0.3

    ax.axhline(0, ls='--', color='k')

    ax.scatter(radius[vel_disp>51], mlf_model[vel_disp>51], marker='o', s=30, label='All KOFFEE fits; R={:.2f}'.format(r_mlf_all), c=colours[0], alpha=0.7)
    ax.scatter(radius[vel_disp<=51], mlf_model[vel_disp<=51], marker='v', s=30, c=colours[0], alpha=0.7)

    #ax.errorbar(radius[vel_disp>51].value, mlf_model[vel_disp>51], yerr=np.array([mlf_max[vel_disp>51], mlf_min[vel_disp>51]]), marker='o', ms=3, ls='none', label='All KOFFEE fits; R={:.2f}'.format(r_mlf_all), c=colours[0], alpha=0.7)
    #ax.errorbar(radius[vel_disp<=51].value, mlf_model[vel_disp<=51], yerr=np.array([mlf_max[vel_disp<=51], mlf_min[vel_disp<=51]]), marker='v', ms=3, ls='none', c=colours[0], alpha=0.7)

    if plot_medians == True:
        #ax.plot(bin_center_all, mlf_bin_medians_all, marker='', lw=3, label='Median all KOFFEE fits; R={:.2f}'.format(r_mlf_med_all), color=colours[0])
        ax.errorbar(bin_center_all.value, mlf_bin_medians_all, yerr=mlf_bin_stdev_all, capsize=3.0, ms=5, lw=3, label='Median all KOFFEE fits; R={:.2f}'.format(r_mlf_med_all), color=colours[0])

    lgnd = ax.legend(frameon=True, fontsize='small', loc='lower left', framealpha=0.5)
    #lgnd.legendHandles[0]._legmarker.set_markersize(3)

    if compare == 'divide':
        ax.set_ylabel(r'Log($\eta$/model)+$const$')
    elif compare == 'subtract':
        ax.set_ylabel(r'$\eta$-model+$const$')

    ax.set_xlabel('Galaxy Radius [kpc]')
    ax.set_title('S/N > 20 and $\delta_{BIC}$ > 10')

    ax.set_ylim(np.nanmin(mlf_model)-0.4, np.nanmax(mlf_model)+0.1)

    plt.show()


#possible other plot
def plot_sfr_elf(OIII_outflow_results, OIII_outflow_error, hbeta_outflow_results, hbeta_outflow_error, hbeta_no_outflow_results, hbeta_no_outflow_error, BIC_outflow, BIC_no_outflow, statistical_results, z, radius, header, weighted_average=True):
    """
    Plots the SFR surface density against the energy loading factor with
    Sigma_SFR calculated using only the narrow component.

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

    radius : :obj:'~numpy.ndarray'
        array of galaxy radius values

    header : FITS header object
        the header from the fits file

    weighted_average : boolean
        whether or not to take a weighted average using the errors (Default=True)

    Returns
    -------
    A six panel graph of the mass loading factor and Hbeta flux ratio against
    the SFR surface density

    """
    #calculate the sfr surface density - using just the systemic line, and including the flux line
    #don't include extinction since this was included in the continuum subtraction using ppxf
    sfr, sfr_err, total_sfr, sfr_surface_density, sfr_surface_density_err = calc_sfr.calc_sfr_koffee(hbeta_outflow_results, hbeta_outflow_error, hbeta_no_outflow_results, hbeta_no_outflow_error, statistical_results, z, header, include_extinction=False, include_outflow=False)

    #calculate the mass loading factor
    elf, elf_max, elf_min = calc_elf.calc_energy_loading_factor(OIII_outflow_results, OIII_outflow_error, hbeta_outflow_results, hbeta_outflow_error, hbeta_no_outflow_results, hbeta_no_outflow_error, statistical_results, z)

    #calculate the velocity dispersion for the masking
    vel_disp, vel_disp_err, vel_diff, vel_diff_err, vel_out, vel_out_err = calc_outvel.calc_outflow_vel(OIII_outflow_results, OIII_outflow_error, statistical_results, z)

    #create the flow mask
    flow_mask = (statistical_results>0) & (np.isnan(hbeta_outflow_results[3,:,:])==False)


    #flatten all the arrays and get rid of extra spaxels
    sig_sfr = sfr_surface_density[flow_mask]
    sig_sfr_err = sfr_surface_density_err[flow_mask]

    elf = elf[flow_mask]
    elf_max = elf_max[flow_mask]
    elf_min = elf_min[flow_mask]

    BIC_outflow = BIC_outflow[flow_mask]
    BIC_no_outflow = BIC_no_outflow[flow_mask]

    vel_disp = vel_disp[flow_mask]
    radius = radius[flow_mask]


    #take the log of the elf
    elf = np.log10(elf)
    elf_max = np.log10(elf_max)
    elf_min = np.log10(elf_min)

    #calculate the errors
    elf_err_max = elf_max - elf
    elf_err_min = elf - elf_min

    #create BIC diff
    BIC_diff = BIC_outflow - BIC_no_outflow
    BIC_diff_strong = (BIC_diff < -50)

    #physical limits mask -
    #for the radius mask 6.1" is the 90% radius
    #also mask out the fits which lie on the lower limit of dispersion < 51km/s
    physical_mask = (radius < 6.1) & (vel_disp>51)

    #make sure none of the errors are nan values
    #vel_out_err[np.where(np.isnan(vel_out_err)==True)] = np.nanmedian(vel_out_err)

    #do the calculations for all the bins
    num_bins = 5
    min_bin = None #-0.05
    max_bin = None #0.6

    if weighted_average == False:
        bin_center_all, elf_bin_medians_all, elf_bin_lower_q_all, elf_bin_upper_q_all = pf.binned_median_quantile_log(sig_sfr, elf, num_bins=num_bins, weights=None, min_bin=min_bin, max_bin=max_bin)
        bin_center_physical, elf_bin_medians_physical, elf_bin_lower_q_physical, elf_bin_upper_q_physical = pf.binned_median_quantile_log(sig_sfr[physical_mask], elf[physical_mask], num_bins=num_bins, weights=None, min_bin=min_bin, max_bin=max_bin)
        bin_center_strong, elf_bin_medians_strong, elf_bin_lower_q_strong, elf_bin_upper_q_strong = pf.binned_median_quantile_log(sig_sfr[BIC_diff_strong], elf[BIC_diff_strong], num_bins=num_bins, weights=None, min_bin=min_bin, max_bin=max_bin)


    elif weighted_average == True:
        bin_center_all, elf_bin_medians_all, elf_bin_lower_q_all, elf_bin_upper_q_all = pf.binned_median_quantile_log(sig_sfr, elf, num_bins=num_bins, weights=[vel_out_err], min_bin=min_bin, max_bin=max_bin)
        bin_center_physical, elf_bin_medians_physical, elf_bin_lower_q_physical, elf_bin_upper_q_physical = pf.binned_median_quantile_log(sig_sfr[physical_mask], elf[physical_mask], num_bins=num_bins, weights=[vel_out_err], min_bin=min_bin, max_bin=max_bin)
        bin_center_strong, elf_bin_medians_strong, elf_bin_lower_q_strong, elf_bin_upper_q_strong = pf.binned_median_quantile_log(sig_sfr[BIC_diff_strong], elf[BIC_diff_strong], num_bins=num_bins, weights=[vel_out_err], min_bin=min_bin, max_bin=max_bin)


    #calculate the r value for the median values
    r_elf_med_all, p_value_elf_all = pf.pearson_correlation(bin_center_all, elf_bin_medians_all)
    r_elf_med_physical, p_value_elf_physical = pf.pearson_correlation(bin_center_physical, elf_bin_medians_physical)
    r_elf_med_strong, p_value_elf_strong = pf.pearson_correlation(bin_center_strong, elf_bin_medians_strong)

    #calculate the r value for all the values
    r_elf_all, p_value_elf_all = pf.pearson_correlation(sig_sfr[~np.isnan(elf)], elf[~np.isnan(elf)])
    r_elf_physical, p_value_elf_physical = pf.pearson_correlation(sig_sfr[~np.isnan(elf)&physical_mask], elf[~np.isnan(elf)&physical_mask])
    r_elf_strong, p_value_elf_strong = pf.pearson_correlation(sig_sfr[~np.isnan(elf)&BIC_diff_strong], elf[~np.isnan(elf)&BIC_diff_strong])


    #calculate Kim et al. trend
    #sfr_surface_density_kim, mlf_Kim = pf.kim_et_al_2020(sig_sfr.min(), sig_sfr.max(), scale_factor=0.06)#(10**mlf_bin_medians_all[0])/(bin_center_all[0]**-0.44))#0.06)


    #print average numbers for the different panels
    print('Number of spaxels in the first panel', elf.shape)
    print('All spaxels median elf:', np.nanmedian(elf))
    print('All spaxels standard deviation elf:', np.nanstd(elf))
    print('')

    print('Number of spaxels with broad sigmas at the instrument dispersion:', elf[vel_disp<=51].shape)
    print('')
    print('Number of spaxels beyond R_90:', elf[radius>6.1].shape)
    print('')
    print('Number of spaxels in the middle panel:', elf[physical_mask].shape)
    print('')

    print('Physical spaxels median elf:', np.nanmedian(elf[physical_mask]))
    print('Physical spaxels standard deviation elf:', np.nanstd(elf[physical_mask]))
    print('')

    print('Number of spaxels with strong BIC differences:', elf[BIC_diff_strong].shape)
    print('')

    print('Clean spaxels median elf:', np.nanmedian(elf[BIC_diff_strong]))
    print('Clean spaxels standard deviation elf:', np.nanstd(elf[BIC_diff_strong]))
    print('')

    #-------
    #plot it
    #-------
    plt.rcParams.update(pf.get_rc_params())
    fig, ax = plt.subplots(nrows=1, ncols=3, sharex=True, sharey='row', figsize=(10,4), constrained_layout=True)

    #get colours from cmasher
    colours = cmr.take_cmap_colors('cmr.gem', 3, cmap_range=(0.25, 0.85), return_fmt='hex')

    #plot all points
    #ax[0].errorbar(sig_sfr, elf, xerr=sig_sfr_err, yerr=[elf_err_min, elf_err_max], fmt='o', ms=3, color=colours[0], alpha=0.6, label='All KOFFEE fits; R={:.2f}'.format(r_elf_all))
    ax[0].fill_between(bin_center_all, elf_bin_lower_q_all, elf_bin_upper_q_all, color=colours[0], alpha=0.3)
    ax[0].scatter(sig_sfr, elf, marker='o', s=10, label='All KOFFEE fits; R={:.2f}'.format(r_elf_all), color=colours[0], alpha=0.8)
    ax[0].plot(bin_center_all, elf_bin_medians_all, marker='', lw=3, label='Median all KOFFEE fits; R={:.2f}'.format(r_elf_med_all), color=colours[0])

    #ax[0].plot(sfr_surface_density_kim, np.log10(elf_Kim), ':k', label='Kim+20, $\eta \propto \Sigma_{SFR}^{-0.44}$')

    ax[0].errorbar(0.03, np.nanmin(elf)+1.0, xerr=np.nanmedian(sig_sfr_err), yerr=[[np.nanmedian(elf_err_min)], [np.nanmedian(elf_err_max)]], c='k')

    #ax[0].set_ylim(-2.4, 0.7)
    ax[0].set_xscale('log')
    ax[0].set_xlim(0.003, 2)
    lgnd = ax[0].legend(frameon=True, fontsize='small', loc='lower left', framealpha=0.5)
    lgnd.legendHandles[0]._legmarker.set_markersize(3)
    ax[0].set_ylabel(r'Log($\eta_{E}$)')
    ax[0].set_xlabel('$\Sigma_{SFR}$ [M$_\odot$ yr$^{-1}$ kpc$^{-2}$]')
    ax[0].set_title('S/N > 20 and $\delta_{BIC}$<-10')


    #plot points within 90% radius
    ax[1].fill_between(bin_center_physical, elf_bin_lower_q_physical, elf_bin_upper_q_physical, color=colours[1], alpha=0.3)
    ax[1].scatter(sig_sfr[radius>6.1], elf[radius>6.1], marker='o', s=10, label='All KOFFEE fits', edgecolors=colours[0], alpha=0.3, facecolors='none')
    ax[1].scatter(sig_sfr[vel_disp<=51], elf[vel_disp<=51], marker='v', s=10, edgecolors=colours[0], alpha=0.3, facecolors='none')
    ax[1].scatter(sig_sfr[physical_mask], elf[physical_mask], marker='o', s=10, label='Selected KOFFEE fits; R={:.2f}'.format(r_elf_physical), color=colours[1], alpha=0.8)
    ax[1].plot(bin_center_physical, elf_bin_medians_physical, marker='', lw=3, label='Median of selected KOFFEE fits; R={:.2f}'.format(r_elf_med_physical), color=colours[1])

    #ax[1].plot(sfr_surface_density_kim, np.log10(elf_Kim), ':k')

    ax[1].errorbar(0.03, np.nanmin(elf)+1.0, xerr=np.nanmedian(sig_sfr_err[physical_mask]), yerr=[[np.nanmedian(elf_err_min[physical_mask])], [np.nanmedian(elf_err_max[physical_mask])]], c='k')

    #ax[1].set_xscale('log')
    lgnd = ax[1].legend(frameon=True, fontsize='small', loc='lower left', framealpha=0.5)
    lgnd.legendHandles[0]._legmarker.set_markersize(3)
    ax[1].set_xlabel('$\Sigma_{SFR}$ [M$_\odot$ yr$^{-1}$ kpc$^{-2}$]')
    ax[1].set_title(r'$\delta_{BIC}$<-10, $r$<$r_{90}$ and $\sigma_{broad}$>$\sigma_{inst}$')


    #plot points with strong BIC values
    ax[2].fill_between(bin_center_strong, elf_bin_lower_q_strong, elf_bin_upper_q_strong, color=colours[2], alpha=0.3)
    ax[2].scatter(sig_sfr[~BIC_diff_strong], elf[~BIC_diff_strong], marker='o', s=10, label='All KOFFEE fits', color=colours[0], alpha=0.3, facecolors='none')
    ax[2].scatter(sig_sfr[BIC_diff_strong], elf[BIC_diff_strong], marker='o', s=10, label='Selected KOFFEE fits; R={:.2f}'.format(r_elf_strong), color=colours[2], alpha=1.0)
    ax[2].plot(bin_center_strong, elf_bin_medians_strong, marker='', lw=3, label='Median of selected KOFFEE fits; R={:.2f}'.format(r_elf_med_strong), color=colours[2])

    #ax[2].plot(sfr_surface_density_kim, np.log10(elf_Kim), ':k')

    ax[2].errorbar(0.03, np.nanmin(elf)+1.0, xerr=np.nanmedian(sig_sfr_err[BIC_diff_strong]), yerr=[[np.nanmedian(elf_err_min[BIC_diff_strong])], [np.nanmedian(elf_err_max[BIC_diff_strong])]], c='k')

    #ax[1].set_xscale('log')
    lgnd = ax[2].legend(frameon=True, fontsize='small', loc='lower left', framealpha=0.5)
    lgnd.legendHandles[0]._legmarker.set_markersize(3)
    ax[2].set_xlabel('$\Sigma_{SFR}$ [M$_\odot$ yr$^{-1}$ kpc$^{-2}$]')
    ax[2].set_title('strongly likely BIC $\delta_{BIC}$<-50')

    #plt.subplots_adjust(left=0.06, right=0.99, top=0.96, bottom=0.07, wspace=0.04, hspace=0.04)

    plt.show()
