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

MODIFICATION HISTORY:
		v.1.0 - first created October 2020

"""
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
import cmasher as cmr

from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
import scipy.stats as stats

from . import calculate_outflow_velocity as calc_outvel
from . import calculate_star_formation_rate as calc_sfr
from . import koffee

import importlib
importlib.reload(calc_outvel)

#===============================================================================
# DEFINE PLOTTING PARAMETERS
#===============================================================================


def get_rc_params():
    """
    Get the rcParams that will be used in all the plots.
    """

    rc_params = {
        "text.usetex": False,
        "axes.facecolor": 'white',

        #"figure.dpi": 125,
        #"legend.fontsize": 12,
        "legend.frameon": False,
        #"legend.markerscale": 1.0,

        #"axes.labelsize": 18,

        "xtick.direction": 'in',
        #"xtick.labelsize": 14,
        "xtick.minor.visible": True,
        "xtick.top": True,
        "xtick.major.width": 1,

        "ytick.direction": 'in',
        #"ytick.labelsize": 14,
        "ytick.minor.visible": True,
        "ytick.right": True,
        "ytick.major.width": 1,
    }

    return rc_params


#===============================================================================
# RELATIONS FROM OTHER PAPERS
#===============================================================================


def chen_et_al_2010(sfr_surface_density_min, sfr_surface_density_max, scale_factor=1):
    """
    The trendline from Chen et al. (2010) where v_out is proportional to (SFR surface density)^0.1
    (Energy driven winds - SNe feedback)
    """
    #create a vector for sfr surface density
    sfr_surface_density = np.linspace(sfr_surface_density_min, sfr_surface_density_max+4, num=1000)

    #use the relationship to predict the v_out
    v_out = scale_factor*sfr_surface_density**0.1

    return sfr_surface_density, v_out


def murray_et_al_2011(sfr_surface_density_min, sfr_surface_density_max, scale_factor=1):
    """
    The trendline from Murray et al. (2011) where v_out is proportional to (SFR surface density)^2
    (Momentum driven winds - radiative feedback from young stars)
    """
    #create a vector for sfr surface density
    sfr_surface_density = np.linspace(sfr_surface_density_min, sfr_surface_density_max+4, num=1000)

    #use the relationship to predict the v_out
    v_out = scale_factor*sfr_surface_density**2

    return sfr_surface_density, v_out


#===============================================================================
# USEFUL LITTLE FUNCTIONS
#===============================================================================

def fitting_function(x, a, b):
    """
    My fitting function to be fit to the v_out to sfr surface density data

    Inputs:
        x: (vector) the star formation surface density
        a, b: (int) constants to be fit

    Returns:
        y: (vector) the v_out
    """
    return a*(x**b)


def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)

def lower_quantile(x):
    """
    Calculate the lower quantile
    """
    return np.nanquantile(x, 0.33)

def upper_quantile(x):
    """
    Calculate the upper quantile
    """
    return np.nanquantile(x, 0.66)


def binned_median_quantile_log(x, y, num_bins, weights=None, min_bin=None, max_bin=None):
    """
    Calculate the mean, upper and lower quantile for an array of data
    """
    if min_bin == None:
        min_bin = np.nanmin(x)
    if max_bin == None:
        max_bin = np.nanmax(x)

    #create the logspace - these are the bin edges
    logspace = np.logspace(np.log10(min_bin), np.log10(max_bin), num=num_bins+1)

    def lower_quantile(x):
        return np.nanquantile(x, 0.33)

    def upper_quantile(x):
        return np.nanquantile(x, 0.66)

    #calculate the upper and lower quartiles
    upper_quantile, bin_edges, binnumber = stats.binned_statistic(x, y, statistic=upper_quantile, bins=logspace)

    lower_quantile, bin_edges, binnumber = stats.binned_statistic(x, y, statistic=lower_quantile, bins=logspace)

    #calculate the average
    bin_avg = np.zeros(len(logspace)-1)

    for i in range(0, len(logspace)-1):
        left_bound = logspace[i]
        right_bound = logspace[i+1]
        items_in_bin = y[(x>left_bound)&(x<=right_bound)]
        if weights == None:
            bin_avg[i] = np.nanmedian(items_in_bin)
        else:
            weights_in_bin = weights[0][(x>left_bound)&(x<=right_bound)]
            weights_in_bin = 1.0 - weights_in_bin/items_in_bin
            bin_avg[i] = np.average(items_in_bin, weights=weights_in_bin)

    #calculate the bin center for plotting
    bin_center = np.zeros(len(logspace)-1)
    for i in range(0, len(logspace)-1):
        bin_center[i] = np.nanmean([logspace[i],logspace[i+1]])

    return bin_center, bin_avg, lower_quantile, upper_quantile


def binned_median_quantile_lin(x, y, num_bins, weights=None, min_bin=None, max_bin=None):
    """
    Calculate the mean, upper and lower quantile for an array of data
    """
    if min_bin == None:
        min_bin = np.nanmin(x)
    if max_bin == None:
        max_bin = np.nanmax(x)

    #create the logspace - these are the bin edges
    logspace = np.linspace(min_bin, max_bin, num=num_bins+1)

    def lower_quantile(x):
        return np.nanquantile(x, 0.33)

    def upper_quantile(x):
        return np.nanquantile(x, 0.66)

    #calculate the upper and lower quartiles
    upper_quantile, bin_edges, binnumber = stats.binned_statistic(x, y, statistic=upper_quantile, bins=logspace)

    lower_quantile, bin_edges, binnumber = stats.binned_statistic(x, y, statistic=lower_quantile, bins=logspace)

    #calculate the average
    bin_avg = np.zeros(len(logspace)-1)

    for i in range(0, len(logspace)-1):
        left_bound = logspace[i]
        right_bound = logspace[i+1]
        items_in_bin = y[(x>left_bound)&(x<=right_bound)]
        if weights == None:
            bin_avg[i] = np.nanmedian(items_in_bin)
        else:
            weights_in_bin = weights[0][(x>left_bound)&(x<=right_bound)]
            weights_in_bin = 1.0 - weights_in_bin/items_in_bin
            bin_avg[i] = np.average(items_in_bin, weights=weights_in_bin)

    #calculate the bin center for plotting
    bin_center = np.zeros(len(logspace)-1)
    for i in range(0, len(logspace)-1):
        bin_center[i] = np.nanmean([logspace[i],logspace[i+1]])

    return bin_center, bin_avg, lower_quantile, upper_quantile


def pearson_correlation(x, y):
    """
    Calculate the Pearson correlation coefficient and p-value

    Parameters
    ----------
    x : :obj:'~numpy.ndarray' object
        Input array - x values

    y : :obj:'~numpy.ndarray' object
        Input array - y values

    Returns
    -------
    r : float
        Pearson's correlation coefficient

    p_value : float
        Two-tailed p-value
    """
    r, p_value = stats.pearsonr(x, y)

    return r, p_value


#===============================================================================
# PLOTTING FUNCTIONS - for paper
#===============================================================================
#Figure 1
def plot_compare_fits(lamdas, data, spaxels, z):
    """
    Plots the normalised single and double gaussian fits for the OIII 5007 line
    using a list of spaxels.
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
    plt.rcParams.update(get_rc_params())
    fig, ax = plt.subplots(nrows=2, ncols=spaxel_num, sharex=True, sharey=True, figsize=(spaxel_num*3, 4), constrained_layout=True)
    #plt.subplots_adjust(wspace=0, hspace=0, left=0.08, right=0.99, top=0.95)

    #get colours from cmasher
    colours = cmr.take_cmap_colors('cmr.gem', 3, cmap_range=(0.25, 0.85), return_fmt='hex')

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
    Plots a two panel graph of histograms of the outflow velocity and the flux
    ratio for [OIII] for before and after koffee's selection criteria for outflows
    are applied.
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
    gmodel2, pars2 = koffee.gaussian2_const(lam_OIII, flux)
    bestfit2 = koffee.fitter(gmodel2, pars2, lam_OIII, flux, verbose=False)

    #get the value to normalise by
    max_value = np.nanmax(flux)

    #create a plotting mask
    plotting_mask = (lam_OIII>lam_OIII[30]) & (lam_OIII<lam_OIII[-25])
    plotting_mask2 = (fine_sampling>lam_OIII[30]) & (fine_sampling<lam_OIII[-25])

    #plot the histograms
    plt.rcParams.update(get_rc_params())
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
    ax[0].set_ylim(0,60)
    ax[0].set_xlabel('Outflow Velocity (km/s)')
    ax[0].set_ylabel('$N_{spaxels}$')

    ax[1].hist(flux_ratio_unfixed[statistical_results_unfixed>0], alpha=0.5, label='All fits', color='tab:blue')
    #ax[1].axvline(flux_ratio_unfixed_median, color='tab:blue', label='All fits median: {:.2f}'.format(flux_ratio_unfixed_median))
    #ax[1].axvline(flux_ratio_unfixed_mean, color='tab:blue', ls='--', label='All fits mean: {:.2f}'.format(flux_ratio_unfixed_mean))

    ax[1].hist(flux_ratio[statistical_results>0], alpha=0.5, label='KOFFEE fits', color='tab:red')
    #ax[1].axvline(flux_ratio_median, color='tab:red', label='KOFFEE fits median: {:.2f}'.format(flux_ratio_median))
    #ax[1].axvline(flux_ratio_mean, color='tab:red', ls='--', label='KOFFEE fits mean: {:.2f}'.format(flux_ratio_mean))

    #ax[1].legend(fontsize='x-small', frameon=False)
    ax[1].set_ylim(0,60)
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
def plot_sfr_vout(OIII_outflow_results, OIII_outflow_error, hbeta_outflow_results, hbeta_outflow_error, hbeta_no_outflow_results, hbeta_no_outflow_error, BIC_outflow, BIC_no_outflow, statistical_results, z, radius, weighted_average=True):
    """
    Plots the SFR surface density against the outflow velocity, with Sigma_SFR calculated
    using only the narrow component.

    Parameters
    ----------
    OIII_outflow_results : (array)
        array of outflow results from KOFFEE for OIII line.  Used to calculate the outflow velocity.  Should be (7, statistical_results.shape)

    OIII_outflow_err : (array)
        array of the outflow result errors from KOFFEE for OIII line

    hbeta_outflow_results : (array)
        array of outflow results from KOFFEE for Hbeta line.  Used to calculate the Sigma SFR.  Should be (7, statistical_results.shape)

    hbeta_outflow_err : (array)
        array of the outflow result errors from KOFFEE for Hbeta line

    hbeta_no_outflow_results : (array)
        array of single gaussian results from KOFFEE for Hbeta line.  Used to calculate the Sigma SFR.  Should be (4, statistical_results.shape)

    hbeta_no_outflow_err : (array)
        array of the single gaussian result errors from KOFFEE for Hbeta line

    BIC_outflow : (array)
        array of BIC values from the double gaussian fits

    BIC_no_outflow : (array)
        array of BIC values from the single gaussian fits

    statistical_results : (array)
        array of statistical results from KOFFEE.

    z : float
        redshift

    Returns
    -------
    A graph of outflow velocity against
    the SFR surface density

    """
    #calculate the outflow velocity
    vel_disp, vel_disp_err, vel_diff, vel_diff_err, vel_out, vel_out_err = calc_outvel.calc_outflow_vel(OIII_outflow_results, OIII_outflow_error, statistical_results, z)

    #calculate the sfr surface density - using just the systemic line, and including the flux line
    #don't include extinction since this was included in the continuum subtraction using ppxf
    sfr, total_sfr, sfr_surface_density, h_beta_integral_err = calc_sfr.calc_sfr_koffee(hbeta_outflow_results, hbeta_outflow_error, hbeta_no_outflow_results, hbeta_no_outflow_error, statistical_results, z, include_extinction=False, include_outflow=False)

    #get the sfr for the outflow spaxels
    flow_mask = (statistical_results>0)# & (sfr_surface_density>0.1)

    #flatten all the arrays and get rid of extra spaxels
    sig_sfr = sfr_surface_density[flow_mask]
    sig_sfr_err = h_beta_integral_err[flow_mask]
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
    #for the radius mask 5" is the 90% radius
    #also mask out the fits which lie on the lower limit of dispersion < 51km/s
    physical_mask = (radius < 5) & (vel_disp>51)

    #strong BIC and physical limits mask
    #clean_mask = (radius < 5) & (vel_disp > 51) & (BIC_diff < -50)

    #make sure none of the errors are nan values
    vel_out_err[np.where(np.isnan(vel_out_err)==True)] = np.nanmedian(vel_out_err)

    #do the calculations for all the bins
    num_bins = 5
    min_bin = None #-0.05
    max_bin = None #0.6

    if weighted_average == False:
        bin_center_all, v_out_bin_medians_all, v_out_bin_lower_q_all, v_out_bin_upper_q_all = binned_median_quantile_log(sig_sfr, vel_out, num_bins=num_bins, weights=None, min_bin=min_bin, max_bin=max_bin)
        bin_center_physical, v_out_bin_medians_physical, v_out_bin_lower_q_physical, v_out_bin_upper_q_physical = binned_median_quantile_log(sig_sfr[physical_mask], vel_out[physical_mask], num_bins=num_bins, weights=None, min_bin=min_bin, max_bin=max_bin)
        bin_center_clean, v_out_bin_medians_clean, v_out_bin_lower_q_clean, v_out_bin_upper_q_clean = binned_median_quantile_log(sig_sfr[BIC_diff_strong], vel_out[BIC_diff_strong], num_bins=num_bins, weights=None, min_bin=min_bin, max_bin=max_bin)

        bin_center_moderate, v_out_bin_medians_moderate, v_out_bin_lower_q_moderate, v_out_bin_upper_q_moderate = binned_median_quantile_log(sig_sfr[BIC_diff<-30], vel_out[BIC_diff<-30], num_bins=num_bins, weights=None, min_bin=min_bin, max_bin=max_bin)
        bin_center_strong, v_out_bin_medians_strong, v_out_bin_lower_q_strong, v_out_bin_upper_q_strong = binned_median_quantile_log(sig_sfr[BIC_diff<-50], vel_out[BIC_diff<-50], num_bins=num_bins, weights=None, min_bin=min_bin, max_bin=max_bin)

    elif weighted_average == True:
        bin_center_all, v_out_bin_medians_all, v_out_bin_lower_q_all, v_out_bin_upper_q_all = binned_median_quantile_log(sig_sfr, vel_out, num_bins=num_bins, weights=[vel_out_err], min_bin=min_bin, max_bin=max_bin)
        bin_center_physical, v_out_bin_medians_physical, v_out_bin_lower_q_physical, v_out_bin_upper_q_physical = binned_median_quantile_log(sig_sfr[physical_mask], vel_out[physical_mask], num_bins=num_bins, weights=[vel_out_err], min_bin=min_bin, max_bin=max_bin)
        bin_center_clean, v_out_bin_medians_clean, v_out_bin_lower_q_clean, v_out_bin_upper_q_clean = binned_median_quantile_log(sig_sfr[BIC_diff_strong], vel_out[BIC_diff_strong], num_bins=num_bins, weights=[vel_out_err], min_bin=min_bin, max_bin=max_bin)

        bin_center_moderate, v_out_bin_medians_moderate, v_out_bin_lower_q_moderate, v_out_bin_upper_q_moderate = binned_median_quantile_log(sig_sfr[BIC_diff<-30], vel_out[BIC_diff<-30], num_bins=num_bins, weights=[vel_out_err], min_bin=min_bin, max_bin=max_bin)
        bin_center_strong, v_out_bin_medians_strong, v_out_bin_lower_q_strong, v_out_bin_upper_q_strong = binned_median_quantile_log(sig_sfr[BIC_diff<-50], vel_out[BIC_diff<-50], num_bins=num_bins, weights=[vel_out_err], min_bin=min_bin, max_bin=max_bin)

    #calculate the r value for the median values
    r_vel_out_med_all, p_value_v_out_all = pearson_correlation(bin_center_all, v_out_bin_medians_all)
    r_vel_out_med_physical, p_value_v_out_physical = pearson_correlation(bin_center_physical, v_out_bin_medians_physical)
    r_vel_out_med_clean, p_value_v_out_clean = pearson_correlation(bin_center_clean, v_out_bin_medians_clean)

    r_vel_out_med_moderate, p_value_v_out_moderate = pearson_correlation(bin_center_moderate, v_out_bin_medians_moderate)
    r_vel_out_med_strong, p_value_v_out_strong = pearson_correlation(bin_center_strong, v_out_bin_medians_strong)

    #calculate the r value for all the values
    r_vel_out_all, p_value_v_out_all = pearson_correlation(sig_sfr, vel_out)
    r_vel_out_physical, p_value_v_out_physical = pearson_correlation(sig_sfr[physical_mask], vel_out[physical_mask])
    r_vel_out_clean, p_value_v_out_clean = pearson_correlation(sig_sfr[BIC_diff_strong], vel_out[BIC_diff_strong])

    r_vel_out_moderate, p_value_v_out_moderate = pearson_correlation(sig_sfr[BIC_diff<-30], vel_out[BIC_diff<-30])
    r_vel_out_strong, p_value_v_out_strong = pearson_correlation(sig_sfr[BIC_diff<-50], vel_out[BIC_diff<-50])

    #create vectors to plot the literature trends
    sfr_surface_density_chen, v_out_chen = chen_et_al_2010(sig_sfr.min(), sig_sfr.max(), scale_factor=np.nanmedian(vel_out[BIC_diff_strong])/(np.nanmedian(sig_sfr[BIC_diff_strong])**0.1))
    sfr_surface_density_murray, v_out_murray = murray_et_al_2011(sig_sfr.min(), sig_sfr.max(), scale_factor=np.nanmedian(vel_out[BIC_diff_strong])/(np.nanmedian(sig_sfr[BIC_diff_strong])**2))

    #fit our own trends
    #popt_vout_all, pcov_vout_all = curve_fit(fitting_function, bin_center_all, v_out_bin_medians_all)
    #popt_vout_physical, pcov_vout_physical = curve_fit(fitting_function, bin_center_physical, v_out_bin_medians_physical)
    #popt_vout_clean, pcov_vout_clean = curve_fit(fitting_function, bin_center_clean, v_out_bin_medians_clean)

    popt_vout_all, pcov_vout_all = curve_fit(fitting_function, sig_sfr, vel_out)
    popt_vout_physical, pcov_vout_physical = curve_fit(fitting_function, sig_sfr[physical_mask], vel_out[physical_mask])
    popt_vout_clean, pcov_vout_clean = curve_fit(fitting_function, sig_sfr[BIC_diff_strong], vel_out[BIC_diff_strong])

    #ax[0].plot(sfr_linspace, fitting_function(sfr_linspace, *popt_vout), 'r-', label='Fit: $v_{out}=%5.0f$ $\Sigma_{SFR}^{%5.2f}$' % tuple(popt_vout))

    #print average numbers for the different panels
    print('Number of spaxels in the first panel', vel_out.shape)
    print('All spaxels median v_out:', np.nanmedian(vel_out))
    print('All spaxels standard deviation v_out:', np.nanstd(vel_out))
    print('')

    print('All spaxels best fit coefficients:', popt_vout_all)
    print('All spaxels best fit errors', np.sqrt(np.diag(pcov_vout_all)))
    print('')

    print('Number of spaxels with broad sigmas at the instrument dispersion:', vel_out[vel_disp<=51].shape)
    print('')
    print('Number of spaxels beyond R_90:', vel_out[radius>5].shape)
    print('')
    print('Number of spaxels in the middle panel:', vel_out[physical_mask].shape)
    print('')

    print('Physical spaxels median v_out:', np.nanmedian(vel_out[physical_mask]))
    print('Physical spaxels standard deviation v_out:', np.nanstd(vel_out[physical_mask]))
    print('')
    print('Physical spaxels best fit coefficients:', popt_vout_physical)
    print('Physical spaxels best fit errors', np.sqrt(np.diag(pcov_vout_physical)))
    print('')

    print('Number of spaxels with strong BIC differences:', vel_out[BIC_diff_strong].shape)
    print('')

    print('Clean spaxels median v_out:', np.nanmedian(vel_out[BIC_diff_strong]))
    print('Clean spaxels standard deviation v_out:', np.nanstd(vel_out[BIC_diff_strong]))
    print('')
    print('Clean spaxels best fit coefficients:', popt_vout_clean)
    print('Clean spaxels best fit errors', np.sqrt(np.diag(pcov_vout_clean)))
    print('')

    #-------
    #plot it
    #-------
    plt.rcParams.update(get_rc_params())
    fig, ax = plt.subplots(nrows=1, ncols=3, sharey=True, sharex=True, figsize=(10,4), constrained_layout=True)

    #get colours from cmasher
    colours = cmr.take_cmap_colors('cmr.gem', 3, cmap_range=(0.25, 0.85), return_fmt='hex')

    #plot all points
    ax[0].fill_between(bin_center_all, v_out_bin_lower_q_all, v_out_bin_upper_q_all, color=colours[0], alpha=0.3)
    ax[0].scatter(sig_sfr, vel_out, marker='o', s=10, label='All KOFFEE fits; R={:.2f}'.format(r_vel_out_all), color=colours[0], alpha=0.8)
    ax[0].plot(bin_center_all, v_out_bin_medians_all, marker='', lw=3, label='Median all KOFFEE fits; R={:.2f}'.format(r_vel_out_med_all), color=colours[0])

    ax[0].plot(sfr_surface_density_chen, v_out_chen, ':k', label='Energy driven, $v_{out} \propto \Sigma_{SFR}^{0.1}$')
    ax[0].plot(sfr_surface_density_murray, v_out_murray, '--k', label='Momentum driven, $v_{out} \propto \Sigma_{SFR}^{2}$')

    ax[0].errorbar(5, 150, xerr=np.nanmedian(sig_sfr_err), yerr=np.nanmedian(vel_out_err), c='k')

    ax[0].set_ylim(100, 500)
    ax[0].set_xscale('log')
    ax[0].set_xlim(0.003, 10)
    lgnd = ax[0].legend(frameon=True, fontsize='small', loc='lower left', framealpha=0.5, edgecolor='white')
    lgnd.legendHandles[0]._legmarker.set_markersize(3)
    ax[0].set_ylabel('Maximum Outflow Velocity [km s$^{-1}$]')
    ax[0].set_xlabel('$\Sigma_{SFR}$ [M$_\odot$ yr$^{-1}$ kpc$^{-2}$]')
    ax[0].set_title('all spaxels')

    #plot points within 90% radius
    ax[1].fill_between(bin_center_physical, v_out_bin_lower_q_physical, v_out_bin_upper_q_physical, color=colours[1], alpha=0.3)
    ax[1].scatter(sig_sfr[radius>5], vel_out[radius>5], marker='o', s=10, label='All KOFFEE fits', edgecolors=colours[0], alpha=0.3, facecolors='none')
    ax[1].scatter(sig_sfr[vel_disp<=51], vel_out[vel_disp<=51], marker='v', s=10, edgecolors=colours[0], alpha=0.3, facecolors='none')
    ax[1].scatter(sig_sfr[physical_mask], vel_out[physical_mask], marker='o', s=10, label='Physical KOFFEE fits; R={:.2f}'.format(r_vel_out_physical), color=colours[1], alpha=0.8)
    ax[1].plot(bin_center_physical, v_out_bin_medians_physical, marker='', lw=3, label='Median of physical KOFFEE fits;\n R={:.2f}'.format(r_vel_out_med_physical), color=colours[1])

    ax[1].plot(sfr_surface_density_chen, v_out_chen, ':k')#, label='Energy driven, $v_{out} \propto \Sigma_{SFR}^{0.1}$')
    ax[1].plot(sfr_surface_density_murray, v_out_murray, '--k')#, label='Momentum driven, $v_{out} \propto \Sigma_{SFR}^{2}$')

    ax[1].errorbar(5, 150, xerr=np.nanmedian(sig_sfr_err), yerr=np.nanmedian(vel_out_err), c='k')

    #ax[1].set_xscale('log')
    lgnd = ax[1].legend(frameon=True, fontsize='small', loc='lower left', framealpha=0.5, edgecolor='white')
    lgnd.legendHandles[0]._legmarker.set_markersize(3)
    ax[1].set_xlabel('$\Sigma_{SFR}$ [M$_\odot$ yr$^{-1}$ kpc$^{-2}$]')
    ax[1].set_title(r'$r$<$r_{90}$ and $\sigma_{broad}$>$\sigma_{inst}$')

    #plot points with strong BIC values
    ax[2].fill_between(bin_center_clean, v_out_bin_lower_q_clean, v_out_bin_upper_q_clean, color=colours[2], alpha=0.3)
    #ax[2].scatter(sig_sfr[radius>5], vel_out[radius>5], marker='o', s=10, label='All KOFFEE fits', edgecolors=colours[0], alpha=0.3, facecolors='none')
    #ax[2].scatter(sig_sfr[vel_disp<=51], vel_out[vel_disp<=51], marker='v', s=10, edgecolors=colours[0], alpha=0.3, facecolors='none')
    #ax[2].scatter(sig_sfr[physical_mask][BIC_diff[physical_mask]>=-51], vel_out[physical_mask][BIC_diff[physical_mask]>=-51], marker='o', s=10, edgecolors=colours[1], alpha=0.3, facecolors='none')
    ax[2].scatter(sig_sfr[~BIC_diff_strong], vel_out[~BIC_diff_strong], marker='o', s=10, label='All KOFFEE fits', color=colours[0], alpha=0.3, facecolors='none')
    ax[2].scatter(sig_sfr[BIC_diff_strong], vel_out[BIC_diff_strong], marker='o', s=10, label='Strong BIC KOFFEE fits; R={:.2f}'.format(r_vel_out_clean), color=colours[2], alpha=1.0)
    ax[2].plot(bin_center_clean, v_out_bin_medians_clean, marker='', lw=3, label='Median of strong BIC KOFFEE fits;\n R={:.2f}'.format(r_vel_out_med_clean), color=colours[2])

    ax[2].plot(sfr_surface_density_chen, v_out_chen, ':k')#, label='Energy driven, $v_{out} \propto \Sigma_{SFR}^{0.1}$')
    ax[2].plot(sfr_surface_density_murray, v_out_murray, '--k')#, label='Momentum driven, $v_{out} \propto \Sigma_{SFR}^{2}$')

    ax[2].errorbar(5, 150, xerr=np.nanmedian(sig_sfr_err), yerr=np.nanmedian(vel_out_err), c='k')

    #ax[1].set_xscale('log')
    lgnd = ax[2].legend(frameon=True, fontsize='small', loc='lower left', framealpha=0.5, edgecolor='white')
    lgnd.legendHandles[0]._legmarker.set_markersize(3)
    ax[2].set_xlabel('$\Sigma_{SFR}$ [M$_\odot$ yr$^{-1}$ kpc$^{-2}$]')
    ax[2].set_title('strongly likely BIC')

    plt.show()



#Figure 4
def plot_sfr_flux(flux_outflow_results, flux_outflow_error, hbeta_outflow_results, hbeta_outflow_error, hbeta_no_outflow_results, hbeta_no_outflow_error, BIC_outflow, BIC_no_outflow, statistical_results, z, radius, flux_ratio_line='OIII', weighted_average=True):
    """
    Plots the SFR surface density against the flux ratio, with Sigma_SFR calculated
    using only the narrow component.

    Parameters
    ----------
    flux_outflow_results : (array)
        array of outflow results from KOFFEE for the line we want to calculate
        the flux ratio for.  Should be (7, statistical_results.shape)

    flux_outflow_err : (array)
        array of the outflow result errors from KOFFEE for flux ratio line

    hbeta_outflow_results : (array)
        array of outflow results from KOFFEE for Hbeta line.  Used to calculate the Sigma SFR.  Should be (7, statistical_results.shape)

    hbeta_outflow_err : (array)
        array of the outflow result errors from KOFFEE for Hbeta line

    hbeta_no_outflow_results : (array)
        array of single gaussian results from KOFFEE for Hbeta line.  Used to calculate the Sigma SFR.  Should be (4, statistical_results.shape)

    hbeta_no_outflow_err : (array)
        array of the single gaussian result errors from KOFFEE for Hbeta line

    BIC_outflow : (array)
        array of BIC values from the double gaussian fits

    BIC_no_outflow : (array)
        array of BIC values from the single gaussian fits

    statistical_results : (array)
        array of statistical results from KOFFEE.

    z : float
        redshift

    Returns
    -------
    A graph of flux ratio against
    the SFR surface density

    """
    #calculate the sfr surface density - using just the systemic line, and including the flux line
    #don't include extinction since this was included in the continuum subtraction using ppxf
    sfr, total_sfr, sfr_surface_density, h_beta_integral_err = calc_sfr.calc_sfr_koffee(hbeta_outflow_results, hbeta_outflow_error, hbeta_no_outflow_results, hbeta_no_outflow_error, statistical_results, z, include_extinction=False, include_outflow=False)

    #calculate the flux for systematic and flow gaussians
    systemic_flux, systemic_flux_err, outflow_flux, outflow_flux_err = calc_sfr.calc_flux_from_koffee(flux_outflow_results, flux_outflow_error, statistical_results, z, outflow=True)

    #create the flow mask
    if flux_ratio_line == 'Hbeta':
        flow_mask = (statistical_results>0) & (np.isnan(flux_outflow_results[3,:,:])==False)
        print('For H$\beta$ the number of outflow fitted spaxels is:', flow_mask[flow_mask].shape)
    else:
        flow_mask = (statistical_results>0)


    #calculate the velocity dispersion for the flux lines
    systemic_mean = flux_outflow_results[1,:,:][flow_mask]/(1+z)
    flow_sigma = flux_outflow_results[3,:,:][flow_mask]/(1+z)
    vel_disp = flow_sigma*299792.458/systemic_mean

    #flatten all the arrays and get rid of extra spaxels
    sig_sfr = sfr_surface_density[flow_mask]
    sig_sfr_err = h_beta_integral_err[flow_mask]

    BIC_outflow = BIC_outflow[flow_mask]
    BIC_no_outflow = BIC_no_outflow[flow_mask]

    systemic_flux = systemic_flux[flow_mask]
    systemic_flux_err = systemic_flux_err[flow_mask]
    outflow_flux = outflow_flux[flow_mask]
    outflow_flux_err = outflow_flux_err[flow_mask]

    radius = radius[flow_mask]

    #make sure none of the errors are nan values
    systemic_flux_err[np.where(np.isnan(systemic_flux_err)==True)] = np.nanmedian(systemic_flux_err)
    outflow_flux_err[np.where(np.isnan(outflow_flux_err)==True)] = np.nanmedian(outflow_flux_err)

    #take the log and do the flux ratio
    #flux_ratio = np.log10(outflow_flux/systemic_flux)
    flux_ratio = (outflow_flux/systemic_flux)

    #calculate the error
    flux_error = flux_ratio * np.sqrt((outflow_flux_err/outflow_flux)**2 + (systemic_flux_err/systemic_flux)**2)

    #create BIC diff
    BIC_diff = BIC_outflow - BIC_no_outflow
    BIC_diff_weak = (BIC_diff < -10) & (BIC_diff >= -30)
    BIC_diff_moderate = (BIC_diff < -30) & (BIC_diff >= -50)
    BIC_diff_strong = (BIC_diff < -50)

    #physical limits mask -
    #for the radius mask 5" is the 90% radius
    #also mask out the fits which lie on the lower limit of dispersion < 51km/s
    physical_mask = (radius < 5) & (vel_disp>51)

    #strong BIC and physical limits mask
    clean_mask = (radius < 5) & (vel_disp > 51) & (BIC_diff < -50)

    #do the calculations for all the bins
    num_bins = 5
    min_bin = None #-0.05
    max_bin = None #0.6

    if weighted_average == False:
        bin_center_all, flux_bin_medians_all, flux_bin_lower_q_all, flux_bin_upper_q_all = binned_median_quantile_log(sig_sfr, flux_ratio, num_bins=num_bins, weights=None, min_bin=min_bin, max_bin=max_bin)
        bin_center_physical, flux_bin_medians_physical, flux_bin_lower_q_physical, flux_bin_upper_q_physical = binned_median_quantile_log(sig_sfr[physical_mask], flux_ratio[physical_mask], num_bins=num_bins, weights=None, min_bin=min_bin, max_bin=max_bin)
        bin_center_clean, flux_bin_medians_clean, flux_bin_lower_q_clean, flux_bin_upper_q_clean = binned_median_quantile_log(sig_sfr[clean_mask], flux_ratio[clean_mask], num_bins=num_bins, weights=None, min_bin=min_bin, max_bin=max_bin)

        bin_center_moderate, flux_bin_medians_moderate, flux_bin_lower_q_moderate, flux_bin_upper_q_moderate = binned_median_quantile_log(sig_sfr[BIC_diff<-30], flux_ratio[BIC_diff<-30], num_bins=num_bins, weights=None, min_bin=min_bin, max_bin=max_bin)
        bin_center_strong, flux_bin_medians_strong, flux_bin_lower_q_strong, flux_bin_upper_q_strong = binned_median_quantile_log(sig_sfr[BIC_diff<-50], flux_ratio[BIC_diff<-50], num_bins=num_bins, weights=None, min_bin=min_bin, max_bin=max_bin)

    elif weighted_average == True:
        bin_center_all, flux_bin_medians_all, flux_bin_lower_q_all, flux_bin_upper_q_all = binned_median_quantile_log(sig_sfr, flux_ratio, num_bins=num_bins, weights=[flux_error], min_bin=min_bin, max_bin=max_bin)
        bin_center_physical, flux_bin_medians_physical, flux_bin_lower_q_physical, flux_bin_upper_q_physical = binned_median_quantile_log(sig_sfr[physical_mask], flux_ratio[physical_mask], num_bins=num_bins, weights=[flux_error], min_bin=min_bin, max_bin=max_bin)
        bin_center_clean, flux_bin_medians_clean, flux_bin_lower_q_clean, flux_bin_upper_q_clean = binned_median_quantile_log(sig_sfr[clean_mask], flux_ratio[clean_mask], num_bins=num_bins, weights=[flux_error], min_bin=min_bin, max_bin=max_bin)

        bin_center_moderate, flux_bin_medians_moderate, flux_bin_lower_q_moderate, flux_bin_upper_q_moderate = binned_median_quantile_log(sig_sfr[BIC_diff<-30], flux_ratio[BIC_diff<-30], num_bins=num_bins, weights=[flux_error], min_bin=min_bin, max_bin=max_bin)
        bin_center_strong, flux_bin_medians_strong, flux_bin_lower_q_strong, flux_bin_upper_q_strong = binned_median_quantile_log(sig_sfr[BIC_diff<-50], flux_ratio[BIC_diff<-50], num_bins=num_bins, weights=[flux_error], min_bin=min_bin, max_bin=max_bin)

    #calculate the r value for the median values
    r_flux_med_all, p_value_flux_all = pearson_correlation(bin_center_all, flux_bin_medians_all)
    r_flux_med_physical, p_value_flux_physical = pearson_correlation(bin_center_physical, flux_bin_medians_physical)
    r_flux_med_clean, p_value_flux_clean = pearson_correlation(bin_center_clean, flux_bin_medians_clean)

    r_flux_med_moderate, p_value_flux_moderate = pearson_correlation(bin_center_moderate, flux_bin_medians_moderate)
    r_flux_med_strong, p_value_flux_strong = pearson_correlation(bin_center_strong, flux_bin_medians_strong)

    #calculate the r value for all the values
    r_flux_all, p_value_flux_all = pearson_correlation(sig_sfr, flux_ratio)
    r_flux_physical, p_value_flux_physical = pearson_correlation(sig_sfr[physical_mask], flux_ratio[physical_mask])
    r_flux_clean, p_value_flux_clean = pearson_correlation(sig_sfr[clean_mask], flux_ratio[clean_mask])

    r_flux_moderate, p_value_flux_moderate = pearson_correlation(sig_sfr[BIC_diff<-30], flux_ratio[BIC_diff<-30])
    r_flux_strong, p_value_flux_strong = pearson_correlation(sig_sfr[BIC_diff<-50], flux_ratio[BIC_diff<-50])

    #print average numbers for the different panels
    print('All spaxels median flux_ratio:', np.nanmedian(flux_ratio))
    print('All spaxels standard deviation flux_ratio:', np.nanstd(flux_ratio))
    print('')

    print('Number of spaxels with broad sigmas at the instrument dispersion:', flux_ratio[vel_disp<=51].shape)
    print('')
    print('Number of spaxels beyond R_90:', flux_ratio[radius>5].shape)
    print('')

    print('Physical spaxels median flux_ratio:', np.nanmedian(flux_ratio[physical_mask]))
    print('Physical spaxels standard deviation flux_ratio:', np.nanstd(flux_ratio[physical_mask]))
    print('')

    print('Clean spaxels median flux_ratio:', np.nanmedian(flux_ratio[clean_mask]))
    print('Clean spaxels standard deviation flux_ratio:', np.nanstd(flux_ratio[clean_mask]))
    print('')

    #-------
    #plot it
    #-------
    plt.rcParams.update(get_rc_params())
    fig, ax = plt.subplots(nrows=1, ncols=3, sharey=True, sharex=True, figsize=(10,4), constrained_layout=True)

    #get colours from cmasher
    colours = cmr.take_cmap_colors('cmr.gem', 3, cmap_range=(0.25, 0.85), return_fmt='hex')

    #plot all points
    ax[0].fill_between(bin_center_all, flux_bin_lower_q_all, flux_bin_upper_q_all, color=colours[0], alpha=0.3)
    ax[0].scatter(sig_sfr, flux_ratio, marker='o', s=10, label='All KOFFEE fits; R={:.2f}'.format(r_flux_all), color=colours[0], alpha=0.8)
    ax[0].plot(bin_center_all, flux_bin_medians_all, marker='', lw=3, label='Median all KOFFEE fits; R={:.2f}'.format(r_flux_med_all), color=colours[0])

    ax[0].errorbar(7, np.nanmedian(flux_ratio), xerr=np.nanmedian(sig_sfr_err), yerr=np.nanmedian(flux_error), c='k')

    ax[0].set_ylim(np.nanmedian(flux_ratio)-np.nanmedian(flux_error)-0.05, 2.1)
    ax[0].set_xscale('log')
    ax[0].set_xlim(0.003, 10)
    lgnd = ax[0].legend(frameon=True, fontsize='small', loc='upper right', framealpha=0.5, edgecolor=None)
    lgnd.legendHandles[0]._legmarker.set_markersize(3)
    ax[0].set_ylabel(flux_ratio_line+' $F_{broad}/F_{narrow}$')
    ax[0].set_xlabel('$\Sigma_{SFR}$ [M$_\odot$ yr$^{-1}$ kpc$^{-2}$]')

    #plot points within 90% radius
    ax[1].fill_between(bin_center_physical, flux_bin_lower_q_physical, flux_bin_upper_q_physical, color=colours[1], alpha=0.3)
    ax[1].scatter(sig_sfr[radius>5], flux_ratio[radius>5], marker='o', s=10, label='All KOFFEE fits', edgecolors=colours[0], alpha=0.3, facecolors='none')
    ax[1].scatter(sig_sfr[vel_disp<=51], flux_ratio[vel_disp<=51], marker='v', s=10, edgecolors=colours[0], alpha=0.3, facecolors='none')
    ax[1].scatter(sig_sfr[physical_mask], flux_ratio[physical_mask], marker='o', s=10, label='Physical KOFFEE fits; R={:.2f}'.format(r_flux_physical), color=colours[1], alpha=0.8)
    ax[1].plot(bin_center_physical, flux_bin_medians_physical, marker='', lw=3, label='Median of physical KOFFEE fits;\n R={:.2f}'.format(r_flux_med_physical), color=colours[1])

    ax[1].errorbar(7, np.nanmedian(flux_ratio), xerr=np.nanmedian(sig_sfr_err), yerr=np.nanmedian(flux_error), c='k')

    lgnd = ax[1].legend(frameon=True, fontsize='small', loc='upper right', framealpha=0.5, edgecolor=None)
    lgnd.legendHandles[0]._legmarker.set_markersize(3)
    ax[1].set_xlabel('$\Sigma_{SFR}$ [M$_\odot$ yr$^{-1}$ kpc$^{-2}$]')

    #plot points with strong BIC values
    ax[2].fill_between(bin_center_clean, flux_bin_lower_q_clean, flux_bin_upper_q_clean, color=colours[2], alpha=0.3)
    ax[2].scatter(sig_sfr[radius>5], flux_ratio[radius>5], marker='o', s=10, label='All KOFFEE fits', edgecolors=colours[0], alpha=0.3, facecolors='none')
    ax[2].scatter(sig_sfr[vel_disp<=51], flux_ratio[vel_disp<=51], marker='v', s=10, edgecolors=colours[0], alpha=0.3, facecolors='none')
    ax[2].scatter(sig_sfr[physical_mask][BIC_diff[physical_mask]>=-51], flux_ratio[physical_mask][BIC_diff[physical_mask]>=-51], marker='o', s=10, edgecolors=colours[1], alpha=0.3, facecolors='none')
    ax[2].scatter(sig_sfr[clean_mask], flux_ratio[clean_mask], marker='o', s=10, label='Strong BIC KOFFEE fits; R={:.2f}'.format(r_flux_clean), color=colours[2], alpha=1.0)
    ax[2].plot(bin_center_clean, flux_bin_medians_clean, marker='', lw=3, label='Median of strong BIC KOFFEE fits;\n R={:.2f}'.format(r_flux_med_clean), color=colours[2])

    ax[2].errorbar(7, np.nanmedian(flux_ratio), xerr=np.nanmedian(sig_sfr_err), yerr=np.nanmedian(flux_error), c='k')

    lgnd = ax[2].legend(frameon=True, fontsize='small', loc='upper right', framealpha=0.5, edgecolor=None)
    lgnd.legendHandles[0]._legmarker.set_markersize(3)
    ax[2].set_xlabel('$\Sigma_{SFR}$ [M$_\odot$ yr$^{-1}$ kpc$^{-2}$]')

    plt.show()
