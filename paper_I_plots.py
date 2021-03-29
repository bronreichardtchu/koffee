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
    plot_compare_fits               (Figure 1)
    plot_hist_out_vel_flux          (Figure 2)
    plot_sfr_vout                   (Figure 3)
    plot_sfr_mlf_flux               (Figure 4)
    maps_of_IRAS08                  (Figure 5)
    plot_mlf_model_rad_singlepanel  (Figure 6)

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
importlib.reload(calc_sfr)
#importlib.reload(calc_mlf)
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
        BIC_diff = bestfit2.bic - bestfit1.bic
        print(BIC_diff)
        if -10 > BIC_diff >= -30:
            significance_level = 'weakly likely\n -10 > $\delta_{BIC}$ > -30'
        elif -30 > BIC_diff >= -50:
            significance_level = 'moderately likely\n -30 > $\delta_{BIC}$ > -50'
        elif -50 > BIC_diff:
            significance_level = 'strongly likely\n -50 > $\delta_{BIC}$'
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
def plot_hist_out_vel_flux(outflow_results, outflow_error, outflow_results_unfixed, outflow_error_unfixed, statistical_results, lamdas, data, spaxel, z):
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
        KOFFEE's checks means the fit is an unlikely one.

    z : float
        redshift

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

    #calculate flux ratios
    flux_ratio = np.log10(outflow_flux/systemic_flux)

    flux_ratio_unfixed = np.log10(outflow_flux_unfixed/systemic_flux_unfixed)

    #calculate the medians
    vel_out_median = np.nanmedian(vel_out)
    vel_out_unfixed_median = np.nanmedian(vel_out_unfixed)

    flux_ratio_median = np.nanmedian(flux_ratio)
    flux_ratio_unfixed_median = np.nanmedian(flux_ratio_unfixed)

    print('All fits vel_out median:', vel_out_median)
    print('KOFFEE fits vel_out median:', vel_out_unfixed_median)
    print('All fits flux_ratio median:', flux_ratio_median)
    print('KOFFEE fits flux_ratio median:', flux_ratio_unfixed_median)

    #calculate the means
    #vel_out_mean = np.nanmean(vel_out)
    #vel_out_unfixed_mean = np.nanmean(vel_out_unfixed)

    #flux_ratio_mean = np.nanmean(flux_ratio)
    #flux_ratio_unfixed_mean = np.nanmean(flux_ratio_unfixed)

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

    #plot the histograms
    plt.rcParams.update(pf.get_rc_params())
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(10,4), constrained_layout=True)

    #get colours from cmasher
    colours = cmr.take_cmap_colors('cmr.gem', 3, cmap_range=(0.25, 0.85), return_fmt='hex')

    ax[0].hist(vel_out_unfixed[statistical_results_unfixed>0], alpha=0.5, color='tab:blue', label='All fits')
    #ax[0].axvline(vel_out_unfixed_median, color='tab:blue', label='All fits median: {:.2f}'.format(vel_out_unfixed_median))
    #ax[0].axvline(vel_out_unfixed_mean, ls='--', color='tab:blue', label='All fits mean: {:.2f}'.format(vel_out_unfixed_mean))

    ax[0].hist(vel_out[statistical_results>0], alpha=0.5, color='tab:red', label='KOFFEE fits')
    #ax[0].axvline(vel_out_median, color='tab:red', label='KOFFEE fits median: {:.2f}'.format(vel_out_median))
    #ax[0].axvline(vel_out_mean, color='tab:red', ls='--', label='KOFFEE fits mean: {:.2f}'.format(vel_out_mean))

    ax[0].legend(fontsize='x-small', frameon=False)
    ax[0].set_ylim(0,90)
    ax[0].set_xlabel('Outflow Velocity (km/s)')
    ax[0].set_ylabel('$N_{spaxels}$')

    ax[1].hist(flux_ratio_unfixed[statistical_results_unfixed>0], alpha=0.5, label='All fits', color='tab:blue')
    #ax[1].axvline(flux_ratio_unfixed_median, color='tab:blue', label='All fits median: {:.2f}'.format(flux_ratio_unfixed_median))
    #ax[1].axvline(flux_ratio_unfixed_mean, color='tab:blue', ls='--', label='All fits mean: {:.2f}'.format(flux_ratio_unfixed_mean))

    ax[1].hist(flux_ratio[statistical_results>0], alpha=0.5, label='KOFFEE fits', color='tab:red')
    #ax[1].axvline(flux_ratio_median, color='tab:red', label='KOFFEE fits median: {:.2f}'.format(flux_ratio_median))
    #ax[1].axvline(flux_ratio_mean, color='tab:red', ls='--', label='KOFFEE fits mean: {:.2f}'.format(flux_ratio_mean))

    #ax[1].legend(fontsize='x-small', frameon=False)
    ax[1].set_ylim(0,90)
    ax[1].set_xlabel('[OIII] Log($f_{broad}/f_{narrow}$)')

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




#Figure 3
def plot_sfr_vout(OIII_outflow_results, OIII_outflow_error, hbeta_outflow_results, hbeta_outflow_error, hbeta_no_outflow_results, hbeta_no_outflow_error, BIC_outflow, BIC_no_outflow, statistical_results, z, radius, header, weighted_average=True, plot_data_fits=False):
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
        bin_center_all, v_out_bin_medians_all, v_out_bin_lower_q_all, v_out_bin_upper_q_all = pf.binned_median_quantile_log(sig_sfr, vel_out, num_bins=num_bins, weights=None, min_bin=min_bin, max_bin=max_bin)
        bin_center_physical, v_out_bin_medians_physical, v_out_bin_lower_q_physical, v_out_bin_upper_q_physical = pf.binned_median_quantile_log(sig_sfr[physical_mask], vel_out[physical_mask], num_bins=num_bins, weights=None, min_bin=min_bin, max_bin=max_bin)
        bin_center_clean, v_out_bin_medians_clean, v_out_bin_lower_q_clean, v_out_bin_upper_q_clean = pf.binned_median_quantile_log(sig_sfr[BIC_diff_strong], vel_out[BIC_diff_strong], num_bins=num_bins, weights=None, min_bin=min_bin, max_bin=max_bin)

        bin_center_moderate, v_out_bin_medians_moderate, v_out_bin_lower_q_moderate, v_out_bin_upper_q_moderate = pf.binned_median_quantile_log(sig_sfr[BIC_diff<-30], vel_out[BIC_diff<-30], num_bins=num_bins, weights=None, min_bin=min_bin, max_bin=max_bin)
        bin_center_strong, v_out_bin_medians_strong, v_out_bin_lower_q_strong, v_out_bin_upper_q_strong = pf.binned_median_quantile_log(sig_sfr[BIC_diff<-50], vel_out[BIC_diff<-50], num_bins=num_bins, weights=None, min_bin=min_bin, max_bin=max_bin)

    elif weighted_average == True:
        bin_center_all, v_out_bin_medians_all, v_out_bin_lower_q_all, v_out_bin_upper_q_all = pf.binned_median_quantile_log(sig_sfr, vel_out, num_bins=num_bins, weights=[vel_out_err], min_bin=min_bin, max_bin=max_bin)
        bin_center_physical, v_out_bin_medians_physical, v_out_bin_lower_q_physical, v_out_bin_upper_q_physical = pf.binned_median_quantile_log(sig_sfr[physical_mask], vel_out[physical_mask], num_bins=num_bins, weights=[vel_out_err], min_bin=min_bin, max_bin=max_bin)
        bin_center_clean, v_out_bin_medians_clean, v_out_bin_lower_q_clean, v_out_bin_upper_q_clean = pf.binned_median_quantile_log(sig_sfr[BIC_diff_strong], vel_out[BIC_diff_strong], num_bins=num_bins, weights=[vel_out_err], min_bin=min_bin, max_bin=max_bin)

        bin_center_moderate, v_out_bin_medians_moderate, v_out_bin_lower_q_moderate, v_out_bin_upper_q_moderate = pf.binned_median_quantile_log(sig_sfr[BIC_diff<-30], vel_out[BIC_diff<-30], num_bins=num_bins, weights=[vel_out_err], min_bin=min_bin, max_bin=max_bin)
        bin_center_strong, v_out_bin_medians_strong, v_out_bin_lower_q_strong, v_out_bin_upper_q_strong = pf.binned_median_quantile_log(sig_sfr[BIC_diff<-50], vel_out[BIC_diff<-50], num_bins=num_bins, weights=[vel_out_err], min_bin=min_bin, max_bin=max_bin)

    #calculate the r value for the median values
    r_vel_out_med_all, p_value_v_out_all = pf.pearson_correlation(bin_center_all, v_out_bin_medians_all)
    r_vel_out_med_physical, p_value_v_out_physical = pf.pearson_correlation(bin_center_physical, v_out_bin_medians_physical)
    r_vel_out_med_clean, p_value_v_out_clean = pf.pearson_correlation(bin_center_clean, v_out_bin_medians_clean)

    r_vel_out_med_moderate, p_value_v_out_moderate = pf.pearson_correlation(bin_center_moderate, v_out_bin_medians_moderate)
    r_vel_out_med_strong, p_value_v_out_strong = pf.pearson_correlation(bin_center_strong, v_out_bin_medians_strong)

    #calculate the r value for all the values
    r_vel_out_all, p_value_v_out_all = pf.pearson_correlation(sig_sfr, vel_out)
    r_vel_out_physical, p_value_v_out_physical = pf.pearson_correlation(sig_sfr[physical_mask], vel_out[physical_mask])
    r_vel_out_clean, p_value_v_out_clean = pf.pearson_correlation(sig_sfr[BIC_diff_strong], vel_out[BIC_diff_strong])

    r_vel_out_moderate, p_value_v_out_moderate = pf.pearson_correlation(sig_sfr[BIC_diff<-30], vel_out[BIC_diff<-30])
    r_vel_out_strong, p_value_v_out_strong = pf.pearson_correlation(sig_sfr[BIC_diff<-50], vel_out[BIC_diff<-50])

    #create vectors to plot the literature trends
    sfr_surface_density_chen, v_out_chen = pf.chen_et_al_2010(sig_sfr.min(), sig_sfr.max(), scale_factor=np.nanmedian(vel_out[BIC_diff_strong])/(np.nanmedian(sig_sfr[BIC_diff_strong])**0.1))
    sfr_surface_density_murray, v_out_murray = pf.murray_et_al_2011(sig_sfr.min(), sig_sfr.max(), scale_factor=np.nanmedian(vel_out[BIC_diff_strong])/(np.nanmedian(sig_sfr[BIC_diff_strong])**2))

    #fit our own trends
    popt_vout_all_medians, pcov_vout_all_medians = curve_fit(pf.fitting_function, bin_center_all, v_out_bin_medians_all)
    popt_vout_physical_medians, pcov_vout_physical_medians = curve_fit(pf.fitting_function, bin_center_physical, v_out_bin_medians_physical)
    popt_vout_clean_medians, pcov_vout_clean_medians = curve_fit(pf.fitting_function, bin_center_clean, v_out_bin_medians_clean)

    popt_vout_all, pcov_vout_all = curve_fit(pf.fitting_function, sig_sfr, vel_out)
    popt_vout_physical, pcov_vout_physical = curve_fit(pf.fitting_function, sig_sfr[physical_mask], vel_out[physical_mask])
    popt_vout_clean, pcov_vout_clean = curve_fit(pf.fitting_function, sig_sfr[BIC_diff_strong], vel_out[BIC_diff_strong])

    print(popt_vout_all, pcov_vout_all)
    print([popt_vout_all_medians[0], np.sqrt(np.diag(pcov_vout_all_medians))[0], popt_vout_all_medians[1], np.sqrt(np.diag(pcov_vout_all_medians))[1]])

    sfr_linspace = np.linspace(sig_sfr.min(), sig_sfr.max()+4, num=1000)

    #ax[0].plot(sfr_linspace, fitting_function(sfr_linspace, *popt_vout), 'r-', label='Fit: $v_{out}=%5.0f$ $\Sigma_{SFR}^{%5.2f}$' % tuple(popt_vout))

    #print average numbers for the different panels
    print('Number of spaxels in the first panel', vel_out.shape)
    print('All spaxels median v_out:', np.nanmedian(vel_out))
    print('All spaxels standard deviation v_out:', np.nanstd(vel_out))
    print('All spaxels median sigma_sfr:', np.nanmedian(sig_sfr))
    print('All spaxels standard deviation sigma_sfr:', np.nanstd(sig_sfr))
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
    print('')
    print('Clean spaxels best fit coefficients:', popt_vout_clean)
    print('Clean spaxels best fit errors', np.sqrt(np.diag(pcov_vout_clean)))
    print('')

    #-------
    #plot it
    #-------
    plt.rcParams.update(pf.get_rc_params())
    fig, ax = plt.subplots(nrows=1, ncols=3, sharey=True, sharex=True, figsize=(10,4), constrained_layout=True)

    #get colours from cmasher
    colours = cmr.take_cmap_colors('cmr.gem', 3, cmap_range=(0.25, 0.85), return_fmt='hex')

    #plot all points
    ax[0].fill_between(bin_center_all, v_out_bin_lower_q_all, v_out_bin_upper_q_all, color=colours[0], alpha=0.3)
    ax[0].scatter(sig_sfr[vel_disp>51], vel_out[vel_disp>51], marker='o', s=10, label='All KOFFEE fits; R={:.2f}'.format(r_vel_out_all), color=colours[0], alpha=0.8)
    ax[0].scatter(sig_sfr[vel_disp<=51], vel_out[vel_disp<=51], marker='v', s=10, c=colours[0])
    ax[0].plot(bin_center_all, v_out_bin_medians_all, marker='', lw=3, label='Median all KOFFEE fits; R={:.2f}'.format(r_vel_out_med_all), color=colours[0])


    if plot_data_fits == True:
        ax[0].plot(sfr_linspace, pf.fitting_function(sfr_linspace, *popt_vout_all), 'r-', label='Fit: $v_{out}=%5.0f\pm$%2.0f $\Sigma_{SFR}^{%5.2f \pm %5.2f}$' % (popt_vout_all[0], np.sqrt(np.diag(pcov_vout_all))[0], popt_vout_all[1], np.sqrt(np.diag(pcov_vout_all))[1]))
        ax[0].plot(sfr_linspace, pf.fitting_function(sfr_linspace, *popt_vout_all_medians), 'r--', label='Median Fit: $v_{out}=%5.0f\pm$%2.0f $\Sigma_{SFR}^{%5.2f \pm %5.2f}$' %(popt_vout_all_medians[0], np.sqrt(np.diag(pcov_vout_all_medians))[0], popt_vout_all_medians[1], np.sqrt(np.diag(pcov_vout_all_medians))[1]))

    ax[0].plot(sfr_surface_density_chen, v_out_chen, ':k', label='Energy driven, $v_{out} \propto \Sigma_{SFR}^{0.1}$')
    ax[0].plot(sfr_surface_density_murray, v_out_murray, '--k', label='Momentum driven, $v_{out} \propto \Sigma_{SFR}^{2}$')

    ax[0].errorbar(0.05, 150, xerr=np.nanmedian(sig_sfr_err), yerr=np.nanmedian(vel_out_err), c='k')

    ax[0].set_ylim(100, 700)
    ax[0].set_xscale('log')
    ax[0].set_xlim(np.nanmin(sig_sfr)-0.001, np.nanmax(sig_sfr)+1.0)
    lgnd = ax[0].legend(frameon=True, fontsize='small', loc='upper left', framealpha=0.5)
    lgnd.legendHandles[0]._legmarker.set_markersize(3)
    ax[0].set_ylabel('Maximum Outflow Velocity [km s$^{-1}$]')
    ax[0].set_xlabel('$\Sigma_{SFR}$ [M$_\odot$ yr$^{-1}$ kpc$^{-2}$]')
    ax[0].set_title('S/N > 20 and $\delta_{BIC}$<-10')

    #plot points within 90% radius
    ax[1].fill_between(bin_center_physical, v_out_bin_lower_q_physical, v_out_bin_upper_q_physical, color=colours[1], alpha=0.3)
    ax[1].scatter(sig_sfr[radius>6.1], vel_out[radius>6.1], marker='o', s=10, label='All KOFFEE fits', edgecolors=colours[0], alpha=0.3, facecolors='none')
    ax[1].scatter(sig_sfr[vel_disp<=51], vel_out[vel_disp<=51], marker='v', s=10, edgecolors=colours[0], alpha=0.3, facecolors='none')
    ax[1].scatter(sig_sfr[physical_mask], vel_out[physical_mask], marker='o', s=10, label='Selected KOFFEE fits; R={:.2f}'.format(r_vel_out_physical), color=colours[1], alpha=0.8)
    ax[1].plot(bin_center_physical, v_out_bin_medians_physical, marker='', lw=3, label='Median of selected KOFFEE fits; R={:.2f}'.format(r_vel_out_med_physical), color=colours[1])

    if plot_data_fits == True:
        ax[1].plot(sfr_linspace, pf.fitting_function(sfr_linspace, *popt_vout_physical), 'r-', label='Fit: $v_{out}=%5.0f\pm$%2.0f $\Sigma_{SFR}^{%5.2f \pm %5.2f}$' % (popt_vout_physical[0], np.sqrt(np.diag(pcov_vout_physical))[0], popt_vout_physical[1], np.sqrt(np.diag(pcov_vout_physical))[1]))
        ax[1].plot(sfr_linspace, pf.fitting_function(sfr_linspace, *popt_vout_physical_medians), 'r--', label='Median Fit: $v_{out}=%5.0f\pm$%2.0f $\Sigma_{SFR}^{%5.2f \pm %5.2f}$' %(popt_vout_physical_medians[0], np.sqrt(np.diag(pcov_vout_physical_medians))[0], popt_vout_physical_medians[1], np.sqrt(np.diag(pcov_vout_physical_medians))[1]))

    ax[1].plot(sfr_surface_density_chen, v_out_chen, ':k')#, label='Energy driven, $v_{out} \propto \Sigma_{SFR}^{0.1}$')
    ax[1].plot(sfr_surface_density_murray, v_out_murray, '--k')#, label='Momentum driven, $v_{out} \propto \Sigma_{SFR}^{2}$')

    ax[1].errorbar(0.05, 150, xerr=np.nanmedian(sig_sfr_err[physical_mask]), yerr=np.nanmedian(vel_out_err[physical_mask]), c='k')

    #ax[1].set_xscale('log')
    lgnd = ax[1].legend(frameon=True, fontsize='small', loc='upper left', framealpha=0.5)
    lgnd.legendHandles[0]._legmarker.set_markersize(3)
    ax[1].set_xlabel('$\Sigma_{SFR}$ [M$_\odot$ yr$^{-1}$ kpc$^{-2}$]')
    ax[1].set_title(r'$\delta_{BIC}$<-10, $r$<$r_{90}$ and $\sigma_{broad}$>$\sigma_{inst}$')

    #plot points with strong BIC values
    ax[2].fill_between(bin_center_clean, v_out_bin_lower_q_clean, v_out_bin_upper_q_clean, color=colours[2], alpha=0.3)
    #ax[2].scatter(sig_sfr[radius>6.1], vel_out[radius>6.1], marker='o', s=10, label='All KOFFEE fits', edgecolors=colours[0], alpha=0.3, facecolors='none')
    #ax[2].scatter(sig_sfr[vel_disp<=51], vel_out[vel_disp<=51], marker='v', s=10, edgecolors=colours[0], alpha=0.3, facecolors='none')
    #ax[2].scatter(sig_sfr[physical_mask][BIC_diff[physical_mask]>=-51], vel_out[physical_mask][BIC_diff[physical_mask]>=-51], marker='o', s=10, edgecolors=colours[1], alpha=0.3, facecolors='none')
    ax[2].scatter(sig_sfr[~BIC_diff_strong][vel_disp[~BIC_diff_strong]>51], vel_out[~BIC_diff_strong][vel_disp[~BIC_diff_strong]>51], marker='o', s=10, label='All KOFFEE fits', color=colours[0], alpha=0.3, facecolors='none')
    ax[2].scatter(sig_sfr[~BIC_diff_strong][vel_disp[~BIC_diff_strong]<=51], vel_out[~BIC_diff_strong][vel_disp[~BIC_diff_strong]<=51], marker='v', s=10, edgecolors=colours[0], alpha=0.3, facecolors='none')
    ax[2].scatter(sig_sfr[BIC_diff_strong][vel_disp[BIC_diff_strong]>51], vel_out[BIC_diff_strong][vel_disp[BIC_diff_strong]>51], marker='o', s=10, label='Selected KOFFEE fits; R={:.2f}'.format(r_vel_out_clean), color=colours[2], alpha=1.0)
    ax[2].scatter(sig_sfr[BIC_diff_strong][vel_disp[BIC_diff_strong]<=51], vel_out[BIC_diff_strong][vel_disp[BIC_diff_strong]<=51], marker='v', s=10, color=colours[2], alpha=1.0)
    ax[2].plot(bin_center_clean, v_out_bin_medians_clean, marker='', lw=3, label='Median of selected KOFFEE fits; R={:.2f}'.format(r_vel_out_med_clean), color=colours[2])

    if plot_data_fits == True:
        ax[2].plot(sfr_linspace, pf.fitting_function(sfr_linspace, *popt_vout_clean), 'r-', label='Fit: $v_{out}=%5.0f\pm$%2.0f $\Sigma_{SFR}^{%5.2f \pm %5.2f}$' % (popt_vout_clean[0], np.sqrt(np.diag(pcov_vout_clean))[0], popt_vout_clean[1], np.sqrt(np.diag(pcov_vout_clean))[1]))
        ax[2].plot(sfr_linspace, pf.fitting_function(sfr_linspace, *popt_vout_clean_medians), 'r--', label='Median Fit: $v_{out}=%5.0f\pm$%2.0f $\Sigma_{SFR}^{%5.2f \pm %5.2f}$' %(popt_vout_clean_medians[0], np.sqrt(np.diag(pcov_vout_clean_medians))[0], popt_vout_clean_medians[1], np.sqrt(np.diag(pcov_vout_clean_medians))[1]))

    ax[2].plot(sfr_surface_density_chen, v_out_chen, ':k')#, label='Energy driven, $v_{out} \propto \Sigma_{SFR}^{0.1}$')
    ax[2].plot(sfr_surface_density_murray, v_out_murray, '--k')#, label='Momentum driven, $v_{out} \propto \Sigma_{SFR}^{2}$')

    ax[2].errorbar(0.05, 150, xerr=np.nanmedian(sig_sfr_err[BIC_diff_strong]), yerr=np.nanmedian(vel_out_err[BIC_diff_strong]), c='k')

    #ax[1].set_xscale('log')
    lgnd = ax[2].legend(frameon=True, fontsize='small', loc='upper left', framealpha=0.5)
    lgnd.legendHandles[0]._legmarker.set_markersize(3)
    ax[2].set_xlabel('$\Sigma_{SFR}$ [M$_\odot$ yr$^{-1}$ kpc$^{-2}$]')
    ax[2].set_title('strongly likely BIC $\delta_{BIC}$<-50')

    plt.show()


#Figure 4
def plot_sfr_mlf_flux(OIII_outflow_results, OIII_outflow_error, hbeta_outflow_results, hbeta_outflow_error, hbeta_no_outflow_results, hbeta_no_outflow_error, BIC_outflow, BIC_no_outflow, statistical_results, z, radius, header, weighted_average=True):
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
    mlf, mlf_max, mlf_min = calc_mlf.calc_mass_loading_factor(OIII_outflow_results, OIII_outflow_error, hbeta_outflow_results, hbeta_outflow_error, hbeta_no_outflow_results, hbeta_no_outflow_error, statistical_results, z)

    #calculate the flux for systematic and flow gaussians
    systemic_flux, systemic_flux_err, outflow_flux, outflow_flux_err = calc_sfr.calc_flux_from_koffee(hbeta_outflow_results, hbeta_outflow_error, statistical_results, z, outflow=True)

    #calculate the velocity dispersion for the masking
    vel_disp, vel_disp_err, vel_diff, vel_diff_err, vel_out, vel_out_err = calc_outvel.calc_outflow_vel(OIII_outflow_results, OIII_outflow_error, statistical_results, z)

    #create the flow mask
    flow_mask = (statistical_results>0) & (np.isnan(hbeta_outflow_results[3,:,:])==False)


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
        bin_center_all, mlf_bin_medians_all, mlf_bin_lower_q_all, mlf_bin_upper_q_all = pf.binned_median_quantile_log(sig_sfr, mlf, num_bins=num_bins, weights=None, min_bin=min_bin, max_bin=max_bin)
        bin_center_physical, mlf_bin_medians_physical, mlf_bin_lower_q_physical, mlf_bin_upper_q_physical = pf.binned_median_quantile_log(sig_sfr[physical_mask], mlf[physical_mask], num_bins=num_bins, weights=None, min_bin=min_bin, max_bin=max_bin)
        bin_center_strong, mlf_bin_medians_strong, mlf_bin_lower_q_strong, mlf_bin_upper_q_strong = pf.binned_median_quantile_log(sig_sfr[BIC_diff_strong], mlf[BIC_diff_strong], num_bins=num_bins, weights=None, min_bin=min_bin, max_bin=max_bin)

        bin_center_all, flux_bin_medians_all, flux_bin_lower_q_all, flux_bin_upper_q_all = pf.binned_median_quantile_log(sig_sfr, flux_ratio, num_bins=num_bins, weights=None, min_bin=min_bin, max_bin=max_bin)
        bin_center_physical, flux_bin_medians_physical, flux_bin_lower_q_physical, flux_bin_upper_q_physical = pf.binned_median_quantile_log(sig_sfr[physical_mask], flux_ratio[physical_mask], num_bins=num_bins, weights=None, min_bin=min_bin, max_bin=max_bin)
        bin_center_strong, flux_bin_medians_strong, flux_bin_lower_q_strong, flux_bin_upper_q_strong = pf.binned_median_quantile_log(sig_sfr[BIC_diff_strong], flux_ratio[BIC_diff_strong], num_bins=num_bins, weights=None, min_bin=min_bin, max_bin=max_bin)


    elif weighted_average == True:
        bin_center_all, mlf_bin_medians_all, mlf_bin_lower_q_all, mlf_bin_upper_q_all = pf.binned_median_quantile_log(sig_sfr, mlf, num_bins=num_bins, weights=[vel_out_err], min_bin=min_bin, max_bin=max_bin)
        bin_center_physical, mlf_bin_medians_physical, mlf_bin_lower_q_physical, mlf_bin_upper_q_physical = pf.binned_median_quantile_log(sig_sfr[physical_mask], mlf[physical_mask], num_bins=num_bins, weights=[vel_out_err], min_bin=min_bin, max_bin=max_bin)
        bin_center_strong, mlf_bin_medians_strong, mlf_bin_lower_q_strong, mlf_bin_upper_q_strong = pf.binned_median_quantile_log(sig_sfr[BIC_diff_strong], mlf[BIC_diff_strong], num_bins=num_bins, weights=[vel_out_err], min_bin=min_bin, max_bin=max_bin)

        bin_center_all, flux_bin_medians_all, flux_bin_lower_q_all, flux_bin_upper_q_all = pf.binned_median_quantile_log(sig_sfr, flux_ratio, num_bins=num_bins, weights=[flux_error], min_bin=min_bin, max_bin=max_bin)
        bin_center_physical, flux_bin_medians_physical, flux_bin_lower_q_physical, flux_bin_upper_q_physical = pf.binned_median_quantile_log(sig_sfr[physical_mask], flux_ratio[physical_mask], num_bins=num_bins, weights=[flux_error], min_bin=min_bin, max_bin=max_bin)
        bin_center_strong, flux_bin_medians_strong, flux_bin_lower_q_strong, flux_bin_upper_q_strong = pf.binned_median_quantile_log(sig_sfr[BIC_diff_strong], flux_ratio[BIC_diff_strong], num_bins=num_bins, weights=[flux_error], min_bin=min_bin, max_bin=max_bin)


    #calculate the r value for the median values
    r_mlf_med_all, p_value_mlf_all = pf.pearson_correlation(bin_center_all, mlf_bin_medians_all)
    r_mlf_med_physical, p_value_mlf_physical = pf.pearson_correlation(bin_center_physical, mlf_bin_medians_physical)
    r_mlf_med_strong, p_value_mlf_strong = pf.pearson_correlation(bin_center_strong, mlf_bin_medians_strong)

    #calculate the r value for all the values
    r_mlf_all, p_value_mlf_all = pf.pearson_correlation(sig_sfr[~np.isnan(mlf)], mlf[~np.isnan(mlf)])
    r_mlf_physical, p_value_mlf_physical = pf.pearson_correlation(sig_sfr[~np.isnan(mlf)&physical_mask], mlf[~np.isnan(mlf)&physical_mask])
    r_mlf_strong, p_value_mlf_strong = pf.pearson_correlation(sig_sfr[~np.isnan(mlf)&BIC_diff_strong], mlf[~np.isnan(mlf)&BIC_diff_strong])

    #calculate the r value for the median values
    r_flux_med_all, p_value_flux_all = pf.pearson_correlation(bin_center_all, flux_bin_medians_all)
    r_flux_med_physical, p_value_flux_physical = pf.pearson_correlation(bin_center_physical, flux_bin_medians_physical)
    r_flux_med_strong, p_value_flux_strong = pf.pearson_correlation(bin_center_strong, flux_bin_medians_strong)

    #calculate the r value for all the values
    r_flux_all, p_value_flux_all = pf.pearson_correlation(sig_sfr, flux_ratio)
    r_flux_physical, p_value_flux_physical = pf.pearson_correlation(sig_sfr[physical_mask], flux_ratio[physical_mask])
    r_flux_strong, p_value_flux_strong = pf.pearson_correlation(sig_sfr[BIC_diff_strong], flux_ratio[BIC_diff_strong])


    #calculate Kim et al. trend
    sfr_surface_density_kim, mlf_Kim = pf.kim_et_al_2020(sig_sfr.min(), sig_sfr.max(), scale_factor=0.06)#(10**mlf_bin_medians_all[0])/(bin_center_all[0]**-0.44))#0.06)


    #print average numbers for the different panels
    print('Number of spaxels in the first panel', mlf.shape)
    print('All spaxels median mlf:', np.nanmedian(mlf))
    print('All spaxels standard deviation mlf:', np.nanstd(mlf))
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

    print('Physical spaxels median mlf:', np.nanmedian(mlf[physical_mask]))
    print('Physical spaxels standard deviation mlf:', np.nanstd(mlf[physical_mask]))
    print('')

    print('Physical spaxels median flux_ratio:', np.nanmedian(flux_ratio[physical_mask]))
    print('Physical spaxels standard deviation flux_ratio:', np.nanstd(flux_ratio[physical_mask]))
    print('')

    print('Number of spaxels with strong BIC differences:', mlf[BIC_diff_strong].shape)
    print('')

    print('Clean spaxels median mlf:', np.nanmedian(mlf[BIC_diff_strong]))
    print('Clean spaxels standard deviation mlf:', np.nanstd(mlf[BIC_diff_strong]))
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
    #ax[0].errorbar(sig_sfr, mlf, xerr=sig_sfr_err, yerr=[mlf_err_min, mlf_err_max], fmt='o', ms=3, color=colours[0], alpha=0.6, label='All KOFFEE fits; R={:.2f}'.format(r_mlf_all))
    ax[1,0].fill_between(bin_center_all, mlf_bin_lower_q_all, mlf_bin_upper_q_all, color=colours[0], alpha=0.3)
    ax[1,0].scatter(sig_sfr, mlf, marker='o', s=10, label='All KOFFEE fits; R={:.2f}'.format(r_mlf_all), color=colours[0], alpha=0.8)
    ax[1,0].plot(bin_center_all, mlf_bin_medians_all, marker='', lw=3, label='Median all KOFFEE fits; R={:.2f}'.format(r_mlf_med_all), color=colours[0])

    ax[1,0].plot(sfr_surface_density_kim, np.log10(mlf_Kim), ':k', label='Kim+20, $\eta \propto \Sigma_{SFR}^{-0.44}$')

    ax[1,0].errorbar(0.03, np.nanmin(mlf)+1.5, xerr=np.nanmedian(sig_sfr_err), yerr=[[np.nanmedian(mlf_err_min)], [np.nanmedian(mlf_err_max)]], c='k')

    ax[1,0].set_ylim(np.nanmin(mlf)-0.1, np.nanmax(mlf)+0.5)
    ax[1,0].set_xscale('log')
    ax[1,0].set_xlim(np.nanmin(sig_sfr)-0.001, np.nanmax(sig_sfr+1))
    lgnd = ax[1,0].legend(frameon=True, fontsize='small', loc='lower left', framealpha=0.5)
    lgnd.legendHandles[0]._legmarker.set_markersize(3)
    ax[1,0].set_ylabel(r'Log($\eta$)')
    ax[1,0].set_xlabel('$\Sigma_{SFR}$ [M$_\odot$ yr$^{-1}$ kpc$^{-2}$]')


    #plot points within 90% radius
    ax[1,1].fill_between(bin_center_physical, mlf_bin_lower_q_physical, mlf_bin_upper_q_physical, color=colours[1], alpha=0.3)
    ax[1,1].scatter(sig_sfr[radius>6.1], mlf[radius>6.1], marker='o', s=10, label='All KOFFEE fits', edgecolors=colours[0], alpha=0.3, facecolors='none')
    ax[1,1].scatter(sig_sfr[vel_disp<=51], mlf[vel_disp<=51], marker='v', s=10, edgecolors=colours[0], alpha=0.3, facecolors='none')
    ax[1,1].scatter(sig_sfr[physical_mask], mlf[physical_mask], marker='o', s=10, label='Selected KOFFEE fits; R={:.2f}'.format(r_mlf_physical), color=colours[1], alpha=0.8)
    ax[1,1].plot(bin_center_physical, mlf_bin_medians_physical, marker='', lw=3, label='Median of selected KOFFEE fits; R={:.2f}'.format(r_mlf_med_physical), color=colours[1])

    ax[1,1].plot(sfr_surface_density_kim, np.log10(mlf_Kim), ':k')

    ax[1,1].errorbar(0.03, np.nanmin(mlf)+1.5, xerr=np.nanmedian(sig_sfr_err[physical_mask]), yerr=[[np.nanmedian(mlf_err_min[physical_mask])], [np.nanmedian(mlf_err_max[physical_mask])]], c='k')

    #ax[0,1].set_xscale('log')
    lgnd = ax[1,1].legend(frameon=True, fontsize='small', loc='lower left', framealpha=0.5)
    lgnd.legendHandles[0]._legmarker.set_markersize(3)
    ax[1,1].set_xlabel('$\Sigma_{SFR}$ [M$_\odot$ yr$^{-1}$ kpc$^{-2}$]')


    #plot points with strong BIC values
    ax[1,2].fill_between(bin_center_strong, mlf_bin_lower_q_strong, mlf_bin_upper_q_strong, color=colours[2], alpha=0.3)
    ax[1,2].scatter(sig_sfr[~BIC_diff_strong], mlf[~BIC_diff_strong], marker='o', s=10, label='All KOFFEE fits', color=colours[0], alpha=0.3, facecolors='none')
    ax[1,2].scatter(sig_sfr[BIC_diff_strong], mlf[BIC_diff_strong], marker='o', s=10, label='Selected KOFFEE fits; R={:.2f}'.format(r_mlf_strong), color=colours[2], alpha=1.0)
    ax[1,2].plot(bin_center_strong, mlf_bin_medians_strong, marker='', lw=3, label='Median of selected KOFFEE fits; R={:.2f}'.format(r_mlf_med_strong), color=colours[2])

    ax[1,2].plot(sfr_surface_density_kim, np.log10(mlf_Kim), ':k')

    ax[1,2].errorbar(0.03, np.nanmin(mlf)+1.5, xerr=np.nanmedian(sig_sfr_err[BIC_diff_strong]), yerr=[[np.nanmedian(mlf_err_min[BIC_diff_strong])], [np.nanmedian(mlf_err_max[BIC_diff_strong])]], c='k')

    #ax[0,1].set_xscale('log')
    lgnd = ax[1,2].legend(frameon=True, fontsize='small', loc='lower left', framealpha=0.5)
    lgnd.legendHandles[0]._legmarker.set_markersize(3)
    ax[1,2].set_xlabel('$\Sigma_{SFR}$ [M$_\odot$ yr$^{-1}$ kpc$^{-2}$]')




    #plot all points
    ax[0,0].fill_between(bin_center_all, flux_bin_lower_q_all, flux_bin_upper_q_all, color=colours[0], alpha=0.3)
    ax[0,0].scatter(sig_sfr, flux_ratio, marker='o', s=10, label='All KOFFEE fits; R={:.2f}'.format(r_flux_all), color=colours[0], alpha=0.8)
    ax[0,0].plot(bin_center_all, flux_bin_medians_all, marker='', lw=3, label='Median all KOFFEE fits; R={:.2f}'.format(r_flux_med_all), color=colours[0])

    ax[0,0].errorbar(0.03, np.nanmin(flux_ratio)+1.0, xerr=np.nanmedian(sig_sfr_err), yerr=np.nanmedian(flux_error), c='k')

    ax[0,0].set_ylim((np.nanmin(flux_ratio)+np.nanmedian(flux_error)-0.1), np.nanmax(flux_ratio)+0.6)
    lgnd = ax[0,0].legend(frameon=True, fontsize='small', loc='lower left', framealpha=0.5, edgecolor=None)
    lgnd.legendHandles[0]._legmarker.set_markersize(3)
    ax[0,0].set_ylabel(r'H$\beta$ Log(F$_{broad}$/F$_{narrow}$)')
    ax[0,0].set_title('S/N > 20 and $\delta_{BIC}$<-10')



    #plot points within 90% radius
    ax[0,1].fill_between(bin_center_physical, flux_bin_lower_q_physical, flux_bin_upper_q_physical, color=colours[1], alpha=0.3)
    ax[0,1].scatter(sig_sfr[radius>6.1], flux_ratio[radius>6.1], marker='o', s=10, label='All KOFFEE fits', edgecolors=colours[0], alpha=0.3, facecolors='none')
    ax[0,1].scatter(sig_sfr[vel_disp<=51], flux_ratio[vel_disp<=51], marker='v', s=10, edgecolors=colours[0], alpha=0.3, facecolors='none')
    ax[0,1].scatter(sig_sfr[physical_mask], flux_ratio[physical_mask], marker='o', s=10, label='Selected KOFFEE fits; R={:.2f}'.format(r_flux_physical), color=colours[1], alpha=0.8)
    ax[0,1].plot(bin_center_physical, flux_bin_medians_physical, marker='', lw=3, label='Median of selected KOFFEE fits; R={:.2f}'.format(r_flux_med_physical), color=colours[1])

    ax[0,1].errorbar(0.03, np.nanmin(flux_ratio)+1.0, xerr=np.nanmedian(sig_sfr_err[physical_mask]), yerr=np.nanmedian(flux_error[physical_mask]), c='k')

    lgnd = ax[0,1].legend(frameon=True, fontsize='small', loc='lower left', framealpha=0.5, edgecolor=None)
    lgnd.legendHandles[0]._legmarker.set_markersize(3)
    ax[0,1].set_title(r'$\delta_{BIC}$<-10, $r$<$r_{90}$ and $\sigma_{broad}$>$\sigma_{inst}$')



    #plot points with strong BIC values
    ax[0,2].fill_between(bin_center_strong, flux_bin_lower_q_strong, flux_bin_upper_q_strong, color=colours[2], alpha=0.3)
    ax[0,2].scatter(sig_sfr[~BIC_diff_strong], flux_ratio[~BIC_diff_strong], marker='o', s=10, label='All KOFFEE fits', edgecolors=colours[0], alpha=0.3, facecolors='none')
    ax[0,2].scatter(sig_sfr[BIC_diff_strong], flux_ratio[BIC_diff_strong], marker='o', s=10, label='Selected KOFFEE fits; R={:.2f}'.format(r_flux_strong), color=colours[2], alpha=1.0)
    ax[0,2].plot(bin_center_strong, flux_bin_medians_strong, marker='', lw=3, label='Median of selected KOFFEE fits; R={:.2f}'.format(r_flux_med_strong), color=colours[2])

    ax[0,2].errorbar(0.03, np.nanmin(flux_ratio)+1.0, xerr=np.nanmedian(sig_sfr_err[BIC_diff_strong]), yerr=np.nanmedian(flux_error[BIC_diff_strong]), c='k')

    lgnd = ax[0,2].legend(frameon=True, fontsize='small', loc='lower left', framealpha=0.5, edgecolor=None)
    lgnd.legendHandles[0]._legmarker.set_markersize(3)
    ax[0,2].set_title('strongly likely BIC $\delta_{BIC}$<-50')

    plt.subplots_adjust(left=0.06, right=0.99, top=0.96, bottom=0.07, wspace=0.04, hspace=0.04)

    plt.show()


#Figure 5
def maps_of_IRAS08(halpha_fits_file, fuv_fits_file, f550m_fits_file, outflow_velocity_fits_file, flux_ratio_fits_file, flux_broad_fits_file, flux_narrow_fits_file, m_out_fits_file, mlf_fits_file, xx_flat, yy_flat, radius, statistical_results):#, header, z, OIII_outflow_results, OIII_outflow_error, statistical_results):
    """
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

    xx_flat : :obj:'~numpy.ndarray'
        x-coordinates of the data cube used in KOFFEE

    yy_flat : :obj:'~numpy.ndarray'
        y-coordinates of the data cube used in KOFFEE (must be same shape as xx_flat)

    radius : :obj:'~numpy.ndarray'
        array of galaxy radius values (must be same shape as statistical_results)

    statistical_results : :obj:'~numpy.ndarray'
        array of statistical results from KOFFEE


    Returns
    -------
    A nine-panel figure with the maps of flux from different emission lines and
    continuum areas, and maps of the results from KOFFEE
    """
    #read in fits files
    halpha_data, halpha_header, halpha_wcs = pf.read_in_create_wcs(halpha_fits_file)
    fuv_data, fuv_header, fuv_wcs = pf.read_in_create_wcs(fuv_fits_file)
    f550_data, f550_header, f550_wcs = pf.read_in_create_wcs(f550m_fits_file, index=1)
    vel_out, vel_out_header, vel_out_wcs = pf.read_in_create_wcs(outflow_velocity_fits_file)
    flux_ratio, flux_ratio_header, flux_ratio_wcs = pf.read_in_create_wcs(flux_ratio_fits_file)
    flux_broad, flux_broad_header, flux_broad_wcs = pf.read_in_create_wcs(flux_broad_fits_file)
    flux_narrow, flux_narrow_header, flux_narrow_wcs = pf.read_in_create_wcs(flux_narrow_fits_file)
    mlf, mlf_header, mlf_wcs = pf.read_in_create_wcs(mlf_fits_file)
    m_out, m_out_header, m_out_wcs = pf.read_in_create_wcs(m_out_fits_file)

    #take the log of the velocity and the flux ratio
    vel_out = np.log10(vel_out)
    flux_ratio = np.log10(flux_ratio)
    flux_broad = np.log10(flux_broad)
    flux_narrow = np.log10(flux_narrow)
    mlf = np.log10(mlf)
    m_out = np.log10(m_out)

    #creating the x and y limits
    #xlim = [0, 23]
    #ylim = [0, 66]
    xlim = [4, 16]
    ylim = [2, 58]
    low_lim_rad = [xx_flat.reshape(67,24).transpose()[xlim[0], ylim[0]], yy_flat.reshape(67, 24).transpose()[xlim[0], ylim[0]]]
    high_lim_rad = [xx_flat.reshape(67,24).transpose()[xlim[1], ylim[1]], yy_flat.reshape(67, 24).transpose()[xlim[1], ylim[1]]]
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





    #create outflow mask
    flow_mask = (statistical_results>0)

    xx_flat_out = xx_flat[flow_mask.reshape(-1)]
    yy_flat_out = yy_flat[flow_mask.reshape(-1)]
    flux_ratio = flux_ratio[flow_mask]
    vel_out = vel_out[flow_mask]
    flux_broad = flux_broad[flow_mask]
    flux_narrow = flux_narrow[flow_mask]
    mlf = mlf[flow_mask]
    m_out = m_out[flow_mask]

    #calculate the beginning and end of 5 arcsec
    halpha_10arcsec_pixel_length = abs(5/(halpha_header['CD1_1']*60*60))
    #koffee_10arcsec_pixel_length = abs(10/(xx_flat.reshape(67,24)[1,0]-xx_flat.reshape(67,24)[0,0]))

    halpha_start_10_arcsec_xpixel = kcwi_high_lim_halpha[0]+20
    halpha_start_10_arcsec_ypixel = kcwi_low_lim_halpha[1]+20
    halpha_end_10_arcsec_xpixel = kcwi_high_lim_halpha[0]+20+halpha_10arcsec_pixel_length

    halpha_start_10_arcsec_world = halpha_wcs.all_pix2world(halpha_start_10_arcsec_xpixel, halpha_start_10_arcsec_ypixel, 0)
    halpha_end_10_arcsec_world = halpha_wcs.all_pix2world(halpha_end_10_arcsec_xpixel, halpha_start_10_arcsec_ypixel, 0)

    fuv_start_10_arcsec_pixel = fuv_wcs.all_world2pix(halpha_start_10_arcsec_world[0], halpha_start_10_arcsec_world[1], 0)
    fuv_end_10_arcsec_pixel = fuv_wcs.all_world2pix(halpha_end_10_arcsec_world[0], halpha_end_10_arcsec_world[1], 0)

    f550_start_10_arcsec_pixel = f550_wcs.all_world2pix(halpha_start_10_arcsec_world[0], halpha_start_10_arcsec_world[1], 0)
    f550_end_10_arcsec_pixel = f550_wcs.all_world2pix(halpha_end_10_arcsec_world[0], halpha_end_10_arcsec_world[1], 0)


    koffee_start_10_arcsec_pixel = vel_out_wcs.all_world2pix(halpha_start_10_arcsec_world[0], halpha_start_10_arcsec_world[1], 0)
    koffee_end_10_arcsec_pixel = vel_out_wcs.all_world2pix(halpha_end_10_arcsec_world[0], halpha_end_10_arcsec_world[1], 0)
    koffee_start_10_arcsec_rad = [xx_flat.reshape(67, 24)[int(koffee_start_10_arcsec_pixel[1]), int(koffee_start_10_arcsec_pixel[0])], yy_flat.reshape(67, 24)[int(koffee_start_10_arcsec_pixel[1]), int(koffee_start_10_arcsec_pixel[0])]]
    koffee_end_10_arcsec_rad = [xx_flat.reshape(67, 24)[int(koffee_end_10_arcsec_pixel[1]), int(koffee_end_10_arcsec_pixel[0])], yy_flat.reshape(67, 24)[int(koffee_end_10_arcsec_pixel[1]), int(koffee_end_10_arcsec_pixel[0])]]




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
    #flux_ratio_spax = bdpk.display_pixels(xx_flat.reshape(67, 24).transpose(), yy_flat.reshape(67, 24).transpose(), flux_ratio.transpose(), axes=ax3, cmap=cmr.gem, vmin=-1.5, vmax=0.0, angle=360, snap=True)
    #OIII limits
    #flux_ratio_spax = bdpk.display_pixels(xx_flat_out, yy_flat_out, flux_ratio, axes=ax3, cmap=cmr.gem, vmin=-1.5, vmax=0.0, angle=360, snap=True)
    #Hbeta limits
    flux_ratio_spax = bdpk.display_pixels(xx_flat_out, yy_flat_out, flux_ratio, axes=ax3, cmap=cmr.gem, vmin=-1.5, vmax=0.1, angle=360, snap=True)
    #ax3.hlines(ymin+0.75, xmin+4, xmin+4+koffee_10arcsec_pixel_length, colors='black')
    #ax3.hlines(koffee_start_10_arcsec_pixel[1], koffee_start_10_arcsec_pixel[0], koffee_end_10_arcsec_pixel[0], colors='black')
    #ax3.text(low_lim_rad[0]-5, high_lim_rad[1]-5, '[OIII]', c='black')
    ax3.grid(False)
    #ax3.get_xaxis().set_visible(False)
    #ax3.get_yaxis().set_visible(False)
    lon3 = ax3.coords[0]
    lat3 = ax3.coords[1]
    lon3.set_ticks_visible(False)
    lon3.set_ticklabel_visible(False)
    lat3.set_ticks_visible(False)
    lat3.set_ticklabel_visible(False)
    #ax3.set_xlim(np.nanmin(xx_flat), np.nanmax(xx_flat))
    #ax3.set_ylim(np.nanmin(yy_flat), np.nanmax(yy_flat))
    ax3.set_xlim(low_lim_rad[0], high_lim_rad[0])
    ax3.set_ylim(low_lim_rad[1], high_lim_rad[1])
    ax3.invert_xaxis()
    #ax3.set_ylabel('Arcseconds')
    #ax3.set_xlabel('Arcseconds')
    #ax3.set_title(r'Log([OIII] F$_{broad}$/F$_{narrow}$)')
    ax3.set_title(r'Log(H$\beta$ F$_{broad}$/F$_{narrow}$)')
    cbar = plt.colorbar(flux_ratio_spax, ax=ax3, shrink=0.8)
    #cbar.set_label(r'Log([OIII] F$_{broad}$/F$_{narrow}$)')

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
    #outvel_spax = bdpk.display_pixels(xx_flat.reshape(67, 24).transpose(), yy_flat.reshape(67, 24).transpose(), vel_out.transpose(), angle=360, axes=ax4, cmap=cmr.gem, vmin=2.2, vmax=2.6)
    outvel_spax = bdpk.display_pixels(xx_flat_out, yy_flat_out, vel_out, angle=360, axes=ax4, cmap=cmr.gem, vmin=2.2, vmax=2.6)
    #ax4.hlines(ymin+0.75, xmin+4, xmin+4+10, colors='black')
    ax4.grid(False)
    #ax4.get_xaxis().set_visible(False)
    #ax4.get_yaxis().set_visible(False)
    lon4 = ax4.coords[0]
    lat4 = ax4.coords[1]
    lon4.set_ticks_visible(False)
    lon4.set_ticklabel_visible(False)
    lat4.set_ticks_visible(False)
    lat4.set_ticklabel_visible(False)
    #ax4.set_ylabel('Arcseconds')
    #ax4.set_xlabel('Arcseconds')
    #ax4.set_xlim(np.nanmin(xx_flat), np.nanmax(xx_flat))
    #ax4.set_ylim(np.nanmin(yy_flat), np.nanmax(yy_flat))
    ax4.set_xlim(low_lim_rad[0], high_lim_rad[0])
    ax4.set_ylim(low_lim_rad[1], high_lim_rad[1])
    ax4.invert_xaxis()
    ax4.set_title(r'Log($v_{out}$)')
    cbar = plt.colorbar(outvel_spax, ax=ax4, shrink=0.8)
    cbar.set_label('Log(km s$^{-1}$)')

    ax5 = plt.subplot(333, projection=flux_broad_wcs, slices=('y', 'x'))
    #outvel_spax = bdpk.display_pixels(xx_flat.reshape(67, 24).transpose(), yy_flat.reshape(67, 24).transpose(), flux_broad.transpose(), angle=360, axes=ax5, cmap=cmr.gem, vmin=2.2, vmax=2.6)
    #OIII limits
    #flux_broad_spax = bdpk.display_pixels(xx_flat_out, yy_flat_out, flux_broad, angle=360, axes=ax5, cmap=cmr.gem, vmin=-0.5, vmax=2.0)
    #Hbeta limits
    flux_broad_spax = bdpk.display_pixels(xx_flat_out, yy_flat_out, flux_broad, angle=360, axes=ax5, cmap=cmr.gem, vmin=-1.5, vmax=1.7)
    #ax5.hlines(ymin+0.75, xmin+4, xmin+4+10, colors='black')
    ax5.grid(False)
    #ax5.get_xaxis().set_visible(False)
    #ax5.get_yaxis().set_visible(False)
    lon5 = ax5.coords[0]
    lat5 = ax5.coords[1]
    lon5.set_ticks_visible(False)
    lon5.set_ticklabel_visible(False)
    lat5.set_ticks_visible(False)
    lat5.set_ticklabel_visible(False)
    #ax5.set_ylabel('Arcseconds')
    #ax5.set_xlabel('Arcseconds')
    #ax5.set_xlim(np.nanmin(xx_flat), np.nanmax(xx_flat))
    #ax5.set_ylim(np.nanmin(yy_flat), np.nanmax(yy_flat))
    ax5.set_xlim(low_lim_rad[0], high_lim_rad[0])
    ax5.set_ylim(low_lim_rad[1], high_lim_rad[1])
    ax5.invert_xaxis()
    #ax5.set_title(r'Log([OIII] F$_{broad}$)')
    ax5.set_title(r'Log(H$\beta$ F$_{broad}$)')
    cbar = plt.colorbar(flux_broad_spax, ax=ax5, shrink=0.8)
    #10^-16 erg/s/cm^2
    cbar.set_label(r'Log($10^{-16}$ erg s$^{-1}$ cm$^{-2}$)')

    ax6 = plt.subplot(332, projection=flux_narrow_wcs, slices=('y', 'x'))
    #flux_narrow_spax = bdpk.display_pixels(xx_flat.reshape(67, 24).transpose(), yy_flat.reshape(67, 24).transpose(), flux_narrow.transpose(), angle=360, axes=ax6, cmap=cmr.gem, vmin=2.2, vmax=2.6)
    #OIII limits
    #flux_narrow_spax = bdpk.display_pixels(xx_flat_out, yy_flat_out, flux_narrow, angle=360, axes=ax6, cmap=cmr.gem, vmin=0, vmax=2.0)
    #Hbeta limits
    flux_narrow_spax = bdpk.display_pixels(xx_flat_out, yy_flat_out, flux_narrow, angle=360, axes=ax6, cmap=cmr.gem, vmin=-0.0, vmax=2.3)
    ax6.grid(False)
    lon6 = ax6.coords[0]
    lat6 = ax6.coords[1]
    lon6.set_ticks_visible(False)
    lon6.set_ticklabel_visible(False)
    lat6.set_ticks_visible(False)
    lat6.set_ticklabel_visible(False)
    ax6.set_xlim(low_lim_rad[0], high_lim_rad[0])
    ax6.set_ylim(low_lim_rad[1], high_lim_rad[1])
    ax6.invert_xaxis()
    #ax6.set_title(r'Log([OIII] F$_{narrow}$)')
    ax6.set_title(r'Log(H$\beta$ F$_{narrow}$)')
    cbar = plt.colorbar(flux_narrow_spax, ax=ax6, shrink=0.8)
    cbar.set_label(r'Log($10^{-16}$ erg s$^{-1}$ cm$^{-2}$)')

    ax7 = plt.subplot(337, projection=f550_wcs)
    ax7.set_facecolor('black')
    #do the plotting
    f550_map = ax7.imshow(np.log10(f550_data), origin='lower', cmap=cmr.ember, vmin=-1.5, vmax=0.1)
    #ax2.arrow(mlf_peak_fuv_pixel[0]-55, mlf_peak_fuv_pixel[1]+55, 50, -50, width=5, length_includes_head=True, color='white')
    ax7.hlines(f550_start_10_arcsec_pixel[1], f550_start_10_arcsec_pixel[0], f550_end_10_arcsec_pixel[0], colors='white')
    ax7.text(f550_start_10_arcsec_pixel[0]+5, f550_start_10_arcsec_pixel[1]+10, '5" ', c='white')
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
    #m_out_spax = bdpk.display_pixels(xx_flat.reshape(67, 24).transpose(), yy_flat.reshape(67, 24).transpose(), m_out.transpose(), angle=360, axes=ax8, cmap=cmr.gem, vmin=2.2, vmax=2.6)
    m_out_spax = bdpk.display_pixels(xx_flat_out, yy_flat_out, m_out, angle=360, axes=ax8, cmap=cmr.gem)#, vmin=19.5, vmax=24)
    #ax8.hlines(ymin+0.75, xmin+4, xmin+4+10, colors='black')
    ax8.grid(False)
    lon8 = ax8.coords[0]
    lat8 = ax8.coords[1]
    lon8.set_ticks_visible(False)
    lon8.set_ticklabel_visible(False)
    lat8.set_ticks_visible(False)
    lat8.set_ticklabel_visible(False)
    ax8.set_xlim(low_lim_rad[0], high_lim_rad[0])
    ax8.set_ylim(low_lim_rad[1], high_lim_rad[1])
    ax8.invert_xaxis()
    ax8.set_title(r'Log($\dot{M}_{out}$) ')
    cbar = plt.colorbar(m_out_spax, ax=ax8, shrink=0.8)
    cbar.set_label(r'Log(M$_\odot$ yr$^{-1}$)')

    ax9 = plt.subplot(339, projection=mlf_wcs, slices=('y', 'x'))
    #mlf_spax = bdpk.display_pixels(xx_flat.reshape(67, 24).transpose(), yy_flat.reshape(67, 24).transpose(), mlf.transpose(), angle=360, axes=ax9, cmap=cmr.gem, vmin=2.2, vmax=2.6)
    mlf_spax = bdpk.display_pixels(xx_flat_out, yy_flat_out, mlf, angle=360, axes=ax9, cmap=cmr.gem, vmin=-2.0, vmax=-0.5)
    #ax9.hlines(ymin+0.75, xmin+4, xmin+4+10, colors='black')
    ax9.grid(False)
    lon9 = ax9.coords[0]
    lat9 = ax9.coords[1]
    lon9.set_ticks_visible(False)
    lon9.set_ticklabel_visible(False)
    lat9.set_ticks_visible(False)
    lat9.set_ticklabel_visible(False)
    ax9.set_xlim(low_lim_rad[0], high_lim_rad[0])
    ax9.set_ylim(low_lim_rad[1], high_lim_rad[1])
    ax9.invert_xaxis()
    ax9.set_title(r'Log($\eta$) ')
    cbar = plt.colorbar(mlf_spax, ax=ax9, shrink=0.8)
    #cbar.set_label(r'Log(M$_\odot$ yr$^{-1}$)')

    plt.subplots_adjust(left=0.0, right=0.96, top=0.99, bottom=0.0, wspace=0.1, hspace=0.0)

    plt.show()


#Figure 6
def plot_mlf_model_rad_singlepanel(OIII_outflow_results, OIII_outflow_error, hbeta_outflow_results, hbeta_outflow_error, hbeta_no_outflow_results, hbeta_no_outflow_error, BIC_outflow, BIC_no_outflow, statistical_results, z, radius, header, compare='divide'):
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
    mlf, mlf_max, mlf_min = calc_mlf.calc_mass_loading_factor(OIII_outflow_results, OIII_outflow_error, hbeta_outflow_results, hbeta_outflow_error, hbeta_no_outflow_results, hbeta_no_outflow_error, statistical_results, z)

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

    #divide the mass loading factor by the model
    mlf_model = np.full_like(mlf, np.nan, dtype=np.double)

    for i in np.arange(mlf.shape[0]):
        #calculate the expected mlf at each sigma_sfr
        sigma_sfr_model, mlf_expected = pf.kim_et_al_2020(sig_sfr[i], sig_sfr[i], scale_factor=0.06) #scale_factor=(10**mlf_bin_medians_all[0])/(bin_center_all[0]**-0.44))
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

    bin_center_all, mlf_bin_medians_all, mlf_bin_lower_q_all, mlf_bin_upper_q_all = pf.binned_median_quantile_lin(radius, mlf_model, num_bins=num_bins, weights=None, min_bin=min_bin, max_bin=max_bin)

    print(bin_center_all, mlf_bin_medians_all)

    #use constant to bring the first median point to zero
    zero_constant = mlf_bin_medians_all[0]
    mlf_bin_medians_all = mlf_bin_medians_all - zero_constant
    mlf_bin_lower_q_all = mlf_bin_lower_q_all - zero_constant
    mlf_bin_upper_q_all = mlf_bin_upper_q_all - zero_constant
    mlf_model = mlf_model - zero_constant


    #calculate the r value for the median values
    r_mlf_med_all, p_value_mlf_all = pf.pearson_correlation(bin_center_all, mlf_bin_medians_all)
    #r_mlf_med_all, p_value_mlf_all = pearson_correlation(bin_center_all, mlf_model_medians)

    #calculate the r value for all the values
    r_mlf_all, p_value_mlf_all = pf.pearson_correlation(radius[~np.isnan(mlf_model)], mlf_model[~np.isnan(mlf_model)])



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
    ax.fill_between(bin_center_all, mlf_bin_lower_q_all, mlf_bin_upper_q_all, color=colours[0], alpha=0.3)
    ax.scatter(radius, mlf_model, marker='o', s=10, label='All KOFFEE fits; R={:.2f}'.format(r_mlf_all), c=colours[0], alpha=0.8)
    ax.plot(bin_center_all, mlf_bin_medians_all, marker='', lw=3, label='Median all KOFFEE fits; R={:.2f}'.format(r_mlf_med_all), color=colours[0])

    lgnd = ax.legend(frameon=True, fontsize='small', loc='lower left', framealpha=0.5)
    lgnd.legendHandles[0]._legmarker.set_markersize(3)
    if compare == 'divide':
        ax.set_ylabel(r'Log($\eta$/model)+$const$')
    elif compare == 'subtract':
        ax.set_ylabel(r'$\eta$-model+$const$')
    ax.set_xlabel('Radius (Arcseconds)')
    ax.set_title('all spaxels')

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
