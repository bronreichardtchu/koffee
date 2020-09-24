"""
NAME:
	sfr_plots.py

AUTHOR:
	Bronwyn Reichardt Chu
	Swinburne
	2020

EMAIL:
	<breichardtchu@swin.edu.au>

PURPOSE:
	To make plots of results from koffee against the SFR or Sigma SFR
	Written on MacOS Mojave 10.14.5, with Python 3.7

MODIFICATION HISTORY:
		v.1.0 - first created September 2020

"""
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
import scipy.stats as stats

from . import calculate_outflow_velocity as calc_outvel
from . import calculate_star_formation_rate as calc_sfr


import importlib
importlib.reload(calc_outvel)


#===============================================================================
# RELATIONS FROM OTHER PAPERS
#===============================================================================


def chen_et_al_2010(sfr_surface_density_min, sfr_surface_density_max, scale_factor=1):
    """
    The trendline from Chen et al. (2010) where v_out is proportional to (SFR surface density)^0.1
    """
    #create a vector for sfr surface density
    sfr_surface_density = np.linspace(sfr_surface_density_min, sfr_surface_density_max+4, num=1000)

    #use the relationship to predict the v_out
    v_out = scale_factor*sfr_surface_density**0.1

    return sfr_surface_density, v_out


def murray_et_al_2011(sfr_surface_density_min, sfr_surface_density_max, scale_factor=1):
    """
    The trendline from Murray et al. (2011) where v_out is proportional to (SFR surface density)^2
    """
    #create a vector for sfr surface density
    sfr_surface_density = np.linspace(sfr_surface_density_min, sfr_surface_density_max+4, num=1000)

    #use the relationship to predict the v_out
    v_out = scale_factor*sfr_surface_density**2

    return sfr_surface_density, v_out

def davies_et_al_2019(sfr_surface_density_min, sfr_surface_density_max):
    """
    The trendline from Davies et al. (2019) where the flow velocity dispersion
    is proportional to SFR surface density.
    """
    #create a vector for sfr surface density
    sfr_surface_density = np.linspace(sfr_surface_density_min, sfr_surface_density_max+4, num=1000)

    #use the relationship to predict the v_out
    vel_disp = 241*sfr_surface_density**0.3

    return sfr_surface_density, vel_disp

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
def plot_sfr_vout(OIII_outflow_results, OIII_outflow_error, hbeta_outflow_results, hbeta_outflow_error, hbeta_no_outflow_results, hbeta_no_outflow_error, statistical_results, z, radius, weighted_average=True, colour_by_radius=False):
    """
    Plots the SFR surface density against the outflow velocity, with Sigma_SFR calculated
    using only the narrow component.

    Parameters
    ----------
    outflow_results : (array)
        array of outflow results from KOFFEE.  Should be (6, sfr.shape)

    outflow_err : (array)
        array of the outflow result errors from KOFFEE

    stat_results : (array)
        array of statistical results from KOFFEE.  Should be same shape as sfr.

    z : float
        redshift

    Returns
    -------
    A six panel graph of velocity offset, velocity dispersion and outflow velocity against
    the SFR surface density

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
    radius = radius[flow_mask]

    #make sure none of the errors are nan values
    vel_out_err[np.where(np.isnan(vel_out_err)==True)] = np.nanmedian(vel_out_err)

    #do the calculations for all the bins
    num_bins = 5
    min_bin = None #-0.05
    max_bin = None #0.6

    if weighted_average == False:
        bin_center, v_out_bin_medians, v_out_bin_lower_q, v_out_bin_upper_q = binned_median_quantile_log(sig_sfr, vel_out, num_bins=num_bins, weights=None, min_bin=min_bin, max_bin=max_bin)

    elif weighted_average == True:
        bin_center, v_out_bin_medians, v_out_bin_lower_q, v_out_bin_upper_q = binned_median_quantile_log(sig_sfr, vel_out, num_bins=num_bins, weights=[vel_out_err], min_bin=min_bin, max_bin=max_bin)


    print(bin_center)
    print(v_out_bin_medians)

    #calculate the r value for the median values
    r_vel_out_med, p_value_v_out = pearson_correlation(bin_center, v_out_bin_medians)

    #calculate the r value for all the values
    r_vel_out, p_value_v_out = pearson_correlation(sig_sfr, vel_out)

    #create vectors to plot the literature trends
    sfr_surface_density_chen, v_out_chen = chen_et_al_2010(sig_sfr.min(), sig_sfr.max(), scale_factor=np.nanmedian(vel_out)/(np.nanmedian(sig_sfr)**0.1))
    sfr_surface_density_murray, v_out_murray = murray_et_al_2011(sig_sfr.min(), sig_sfr.max(), scale_factor=np.nanmedian(vel_out)/(np.nanmedian(sig_sfr)**2))

    #plot it
    plt.figure(figsize=(5,4))
    plt.rcParams['axes.facecolor']='white'

    #----------------
    #Including Outflow Line Plots
    #-----------------
    if colour_by_radius == True:
        plt.scatter(sig_sfr, vel_out, marker='o', lw=0, label='Flow spaxels; R={:.2f}'.format(r_vel_out), alpha=0.6, c=radius)
        cbar = plt.colorbar()
        cbar.ax.set_ylabel('Radius (Arcsec)')
        plt.errorbar(5, 150, xerr=np.nanmedian(sig_sfr_err), yerr=np.nanmedian(vel_out_err), c='k')

    elif colour_by_radius == False:
        plt.errorbar(sig_sfr, vel_out, xerr=sig_sfr_err, yerr=vel_out_err, marker='o', lw=0, label='Flow spaxels; R={:.2f}'.format(r_vel_out), alpha=0.4, color='tab:blue', ecolor='tab:blue', elinewidth=1)


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


def plot_sfr_vseparate(OIII_outflow_results, OIII_outflow_error, hbeta_outflow_results, hbeta_outflow_error, hbeta_no_outflow_results, hbeta_no_outflow_error, statistical_results, z, radius, weighted_average=True, colour_by_radius=False):
    """
    Plots the SFR surface density against the outflow velocity, comparing Sigma_SFR calculated
    from the full line to Sigma_SFR calculated using only the narrow component.

    Parameters
    ----------
    outflow_results : (array)
        array of outflow results from KOFFEE.  Should be (6, sfr.shape)

    outflow_err : (array)
        array of the outflow result errors from KOFFEE

    stat_results : (array)
        array of statistical results from KOFFEE.  Should be same shape as sfr.

    z : float
        redshift

    Returns
    -------
    A two panel graph of velocity offset and velocity dispersion against
    the SFR surface density

    """
    #calculate the outflow velocity
    vel_diff, vel_diff_err, vel_out, vel_out_err = calc_outvel.calc_outflow_vel(OIII_outflow_results, OIII_outflow_error, statistical_results, z)

    #calculate the sfr surface density - using just the systemic line, and including the flux line
    #don't include extinction since this was included in the continuum subtraction using ppxf
    sfr, total_sfr, sfr_surface_density, h_beta_integral_err = calc_sfr.calc_sfr_koffee(hbeta_outflow_results, hbeta_outflow_error, hbeta_no_outflow_results, hbeta_no_outflow_error, statistical_results, z, include_extinction=False, include_outflow=False)

    #get the sfr for the outflow spaxels
    flow_mask = (statistical_results>0)

    #create id array
    id_array = np.arange(radius.shape[0]*radius.shape[1])
    id_array = id_array.reshape(radius.shape)
    x_id = np.tile(np.arange(radius.shape[0]), radius.shape[1]).reshape(radius.shape[1], radius.shape[0]).T
    y_id = np.tile(np.arange(radius.shape[1]), radius.shape[0]).reshape(radius.shape)
    print(id_array)
    print(x_id)
    print(y_id)

    #convert the sigma to km/s instead of Angstroms
    flow_sigma = OIII_outflow_results[3,:,:][flow_mask]/(1+z)
    systemic_mean = OIII_outflow_results[1,:,:][flow_mask]/(1+z)
    vel_disp = flow_sigma*299792.458/systemic_mean

    vel_disp_err = (flow_sigma/systemic_mean)*np.sqrt((OIII_outflow_error[3,:,:][flow_mask]/flow_sigma)**2 + (OIII_outflow_error[1,:,:][flow_mask]/systemic_mean)**2)

    #flatten all the arrays and get rid of extra spaxels
    sfr = sfr_surface_density[flow_mask]
    sfr_err = h_beta_integral_err[flow_mask]
    vel_diff = vel_diff[flow_mask]
    vel_diff_err = vel_diff_err[flow_mask]
    radius = radius[flow_mask]

    id_array = id_array[flow_mask]
    x_id = x_id[flow_mask]
    y_id = y_id[flow_mask]

    #print the id values where the sigma_sfr is below 0.05 and the vel_disp is above 160
    print('ID values:', id_array[(sfr<0.05)&(vel_disp>160)])
    print('x ID:', x_id[(sfr<0.05)&(vel_disp>160)])
    print('y ID:', y_id[(sfr<0.05)&(vel_disp>160)])

    #make sure none of the errors are nan values
    vel_diff_err[np.where(np.isnan(vel_diff_err)==True)] = np.nanmedian(vel_diff_err)
    vel_disp_err[np.where(np.isnan(vel_disp_err)==True)] = np.nanmedian(vel_disp_err)

    #do the calculations for all the bins
    num_bins = 5
    min_bin = None #-0.05
    max_bin = None #0.6

    if weighted_average == False:
        bin_center, vel_diff_bin_medians, vel_diff_bin_lower_q, vel_diff_bin_upper_q = binned_median_quantile_log(sfr, vel_diff, num_bins=num_bins, weights=None, min_bin=min_bin, max_bin=max_bin)
        bin_center, disp_bin_medians, disp_bin_lower_q, disp_bin_upper_q = binned_median_quantile_log(sfr, vel_disp, num_bins=num_bins, weights=None, min_bin=min_bin, max_bin=max_bin)

    elif weighted_average == True:
        bin_center, vel_diff_bin_medians, vel_diff_bin_lower_q, vel_diff_bin_upper_q = binned_median_quantile_log(sfr, vel_diff, num_bins=num_bins, weights=[vel_diff_err], min_bin=min_bin, max_bin=max_bin)
        bin_center, disp_bin_medians, disp_bin_lower_q, disp_bin_upper_q = binned_median_quantile_log(sfr, vel_disp, num_bins=num_bins, weights=[vel_disp_err], min_bin=min_bin, max_bin=max_bin)


    #calculate the r value for the median values
    r_vel_diff_med, p_value_v_diff = pearson_correlation(bin_center, vel_diff_bin_medians)
    r_disp_med, p_value_disp = pearson_correlation(bin_center, disp_bin_medians)

    #calculate the r value for all the values
    r_vel_diff, p_value_v_diff = pearson_correlation(sfr, vel_diff)
    r_disp, p_value_disp = pearson_correlation(sfr, vel_disp)


    #create vectors to plot the literature trends
    sfr_surface_density_chen, vel_diff_chen = chen_et_al_2010(sfr.min(), sfr.max(), scale_factor=np.nanmedian(vel_diff)/(np.nanmedian(sfr)**0.1))
    sfr_surface_density_murray, vel_diff_murray = murray_et_al_2011(sfr.min(), sfr.max(), scale_factor=np.nanmedian(vel_diff)/(np.nanmedian(sfr)**2))

    #plot it
    fig, ax = plt.subplots(nrows=1, ncols=2, sharex=True, figsize=(8,5))
    plt.rcParams['axes.facecolor']='white'

    #----------------
    #Plots
    #-----------------
    if colour_by_radius == True:
        ax[0].scatter(sfr[vel_disp>=51], vel_diff[vel_disp>=51], marker='o', lw=0, alpha=0.6, label='Flow spaxels; R={:.2f}'.format(r_vel_diff), c=radius[vel_disp>=51])
        ax[0].scatter(sfr[vel_disp<51], vel_diff[vel_disp<51], marker='v', lw=0, alpha=0.6, c=radius[vel_disp<51])

        im = ax[1].scatter(sfr[vel_disp>=51], vel_disp[vel_disp>=51], marker='o', lw=0, alpha=0.6, label='Flow spaxels; R={:.2f}'.format(r_disp), c=radius[vel_disp>=51])
        ax[1].scatter(sfr[vel_disp<51], vel_disp[vel_disp<51], marker='v', lw=0, alpha=0.6, c=radius[vel_disp<51])
        cbar = plt.colorbar(im, ax=ax[1])
        cbar.ax.set_ylabel('Radius (Arcsec)')

    elif colour_by_radius == False:
        #ax[0].plot(sfr[vel_disp>=51], vel_diff[vel_disp>=51], marker='o', lw=0, alpha=0.4, label='Flow spaxels; R={:.2f}'.format(r_vel_diff), color='tab:blue')
        #ax[0].plot(sfr[vel_disp<51], vel_diff[vel_disp<51], marker='v', lw=0, alpha=0.4, color='tab:blue')

        #ax[1].plot(sfr[vel_disp>=51], vel_disp[vel_disp>=51], marker='o', lw=0, alpha=0.4, label='Flow spaxels; R={:.2f}'.format(r_disp), color='tab:blue')
        #ax[1].plot(sfr[vel_disp<51], vel_disp[vel_disp<51], marker='v', lw=0, alpha=0.4, color='tab:blue')

        ax[0].errorbar(sfr[vel_disp>=51], vel_diff[vel_disp>=51], xerr=sfr_err[vel_disp>=51], yerr=vel_diff_err[vel_disp>=51], marker='o', lw=0, alpha=0.4, elinewidth=1, label='Flow spaxels; R={:.2f}'.format(r_vel_diff), color='tab:blue')
        ax[0].errorbar(sfr[vel_disp<51], vel_diff[vel_disp<51], xerr=sfr_err[vel_disp<51], yerr=vel_diff_err[vel_disp<51], marker='v', lw=0, alpha=0.4, elinewidth=1, color='tab:blue')

        ax[1].errorbar(sfr[vel_disp>=51], vel_disp[vel_disp>=51], xerr=sfr_err[vel_disp>=51], yerr=vel_disp_err[vel_disp>=51], marker='o', lw=0, alpha=0.4, elinewidth=1, label='Flow spaxels; R={:.2f}'.format(r_disp), color='tab:blue')
        ax[1].errorbar(sfr[vel_disp<51], vel_disp[vel_disp<51], xerr=sfr_err[vel_disp<51], yerr=vel_disp_err[vel_disp<51], marker='v', lw=0, alpha=0.4, elinewidth=1, color='tab:blue')


    ax[0].fill_between(bin_center, vel_diff_bin_lower_q, vel_diff_bin_upper_q, color='tab:blue', alpha=0.3)
    ax[0].plot(bin_center, vel_diff_bin_medians, marker='', color='tab:blue', lw=3.0, label='Median; R={:.2f}'.format(r_vel_diff_med))
    ax[0].errorbar(5, -100, xerr=np.nanmedian(sfr_err), yerr=np.nanmedian(vel_diff_err), c='k')
    ax[0].plot(sfr_surface_density_chen, vel_diff_chen, ':k', label='Energy driven, $v_{out} \propto \Sigma_{SFR}^{0.1}$')
    ax[0].plot(sfr_surface_density_murray, vel_diff_murray, '--k', label='Momentum driven, $v_{out} \propto \Sigma_{SFR}^{2}$')
    ax[0].set_ylim(-150,250)
    ax[0].set_xscale('log')
    lgnd = ax[0].legend(frameon=False, fontsize='x-small', loc='lower left')
    lgnd.legendHandles[0]._legmarker.set_markersize(4)
    ax[0].set_ylabel('Velocity Offset [km s$^{-1}$]')
    ax[0].set_xlabel('$\Sigma_{SFR}$ [M$_\odot$ yr$^{-1}$ kpc$^{-2}$]')


    ax[1].fill_between(bin_center, disp_bin_lower_q, disp_bin_upper_q, color='tab:blue', alpha=0.3)
    ax[1].plot(bin_center, disp_bin_medians, marker='', color='tab:blue', lw=3.0, label='Median; R={:.2f}'.format(r_disp_med))
    ax[1].errorbar(5, -50, xerr=np.nanmedian(sfr_err), yerr=np.nanmedian(vel_disp_err), c='k')
    ax[1].set_xscale('log')
    ax[1].set_ylim(-100,300)
    lgnd = ax[1].legend(frameon=False, fontsize='x-small', loc='lower left')
    lgnd.legendHandles[0]._legmarker.set_markersize(4)
    ax[1].set_ylabel('Velocity Dispersion [km s$^{-1}$]')
    ax[1].set_xlabel('$\Sigma_{SFR}$ [M$_\odot$ yr$^{-1}$ kpc$^{-2}$]')

    plt.tight_layout()
    plt.show()



def plot_sfr_flux(flux_line_outflow_results, flux_line_outflow_error, hbeta_outflow_results, hbeta_outflow_error, hbeta_no_outflow_results, hbeta_no_outflow_error, statistical_results, z, radius, weighted_average=True, colour_by_radius=False):
    """
    Plots the SFR surface density against the outflow velocity, with Sigma_SFR calculated
    using only the narrow component.

    Parameters
    ----------
    outflow_results : (array)
        array of outflow results from KOFFEE.  Should be (6, sfr.shape)

    outflow_err : (array)
        array of the outflow result errors from KOFFEE

    stat_results : (array)
        array of statistical results from KOFFEE.  Should be same shape as sfr.

    z : float
        redshift

    Returns
    -------
    A graph of velocity offset, velocity dispersion and outflow velocity against
    the SFR surface density

    """

    #calculate the sfr surface density - using just the systemic line, and including the flux line
    #don't include extinction since this was included in the continuum subtraction using ppxf
    sfr, total_sfr, sfr_surface_density, h_beta_integral_err = calc_sfr.calc_sfr_koffee(hbeta_outflow_results, hbeta_outflow_error, hbeta_no_outflow_results, hbeta_no_outflow_error, statistical_results, z, include_extinction=False, include_outflow=False)

    #calculate the flux for systematic and flow gaussians
    systemic_flux, systemic_flux_err, outflow_flux, outflow_flux_err = calc_sfr.calc_flux_from_koffee(flux_line_outflow_results, flux_line_outflow_error, statistical_results, z, outflow=True)

    #get the sfr for the outflow spaxels
    flow_mask = (statistical_results>0)

    #flatten all the arrays and get rid of extra spaxels
    sig_sfr = sfr_surface_density[flow_mask]
    sig_sfr_err = h_beta_integral_err[flow_mask]
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
    #print(outflow_flux)
    #print(systemic_flux)
    #print(flux_ratio)

    #calculate the error
    flux_error = (outflow_flux/systemic_flux) * np.sqrt((outflow_flux_err/outflow_flux)**2 + (systemic_flux_err/systemic_flux)**2)

    #do the calculations for all the bins
    num_bins = 5
    min_bin = None #-0.05
    max_bin = None #0.6

    if weighted_average == False:
        bin_center, flux_bin_medians, flux_bin_lower_q, flux_bin_upper_q = binned_median_quantile_log(sig_sfr, flux_ratio, num_bins=num_bins, weights=None, min_bin=min_bin, max_bin=max_bin)

    elif weighted_average == True:
        bin_center, flux_bin_medians, flux_bin_lower_q, flux_bin_upper_q = binned_median_quantile_log(sig_sfr, flux_ratio, num_bins=num_bins, weights=[flux_error], min_bin=min_bin, max_bin=max_bin)


    print(bin_center)
    print(flux_bin_medians)

    #calculate the r value for the median values
    r_flux_med, p_value_flux = pearson_correlation(bin_center, flux_bin_medians)

    #calculate the r value for all the values
    #r_flux, p_value_flux = pearson_correlation(sig_sfr, flux_ratio)

    #plot it
    plt.figure(figsize=(5,4))
    plt.rcParams['axes.facecolor']='white'

    #----------------
    #Including Outflow Line Plots
    #-----------------
    if colour_by_radius == True:
        #plt.scatter(sig_sfr, flux_ratio, marker='o', lw=0, label='Flow spaxels; R={:.2f}'.format(r_flux), alpha=0.6, c=radius)
        plt.scatter(sig_sfr, flux_ratio, marker='o', lw=0, label='Flow spaxels', alpha=0.6, c=radius)
        cbar = plt.colorbar()
        cbar.ax.set_ylabel('Radius (Arcsec)')
        plt.errorbar(5, -1, xerr=np.nanmedian(sig_sfr_err), yerr=np.nanmedian(flux_error), c='k')

    elif colour_by_radius == False:
        #plt.errorbar(sig_sfr, flux_ratio, xerr=sig_sfr_err, yerr=flux_error, marker='o', lw=0, label='Flow spaxels; R={:.2f}'.format(r_flux), alpha=0.4, color='tab:blue', ecolor='tab:blue', elinewidth=1)
        plt.errorbar(sig_sfr, flux_ratio, xerr=sig_sfr_err, yerr=flux_error, marker='o', lw=0, label='Flow spaxels', alpha=0.4, color='tab:blue', ecolor='tab:blue', elinewidth=1)


    plt.fill_between(bin_center, flux_bin_lower_q, flux_bin_upper_q, color='tab:blue', alpha=0.3)
    plt.plot(bin_center, flux_bin_medians, marker='', color='tab:blue', lw=3.0, label='Median; R={:.2f}'.format(r_flux_med))
    #plt.ylim(-2, 1)
    plt.ylim(-0.5, 2.0)
    plt.xscale('log')
    lgnd = plt.legend(frameon=False, fontsize='x-small', loc='lower left')
    lgnd.legendHandles[0]._legmarker.set_markersize(4)
    #plt.ylabel('Log Broad/Narrow Flux')
    plt.ylabel('Broad/Narrow Flux')
    plt.xlabel('$\Sigma_{SFR}$ [M$_\odot$ yr$^{-1}$ kpc$^{-2}$]')

    plt.tight_layout()
    plt.show()




def plot_sfr_vout_compare_sfr_calcs(OIII_outflow_results, OIII_outflow_error, hbeta_outflow_results, hbeta_outflow_error, hbeta_no_outflow_results, hbeta_no_outflow_error, statistical_results, z, radius, weighted_average=True, colour_by_radius=False):
    """
    Plots the SFR surface density against the outflow velocity, comparing Sigma_SFR calculated
    from the full line to Sigma_SFR calculated using only the narrow component.

    Parameters
    ----------
    outflow_results : (array)
        array of outflow results from KOFFEE.  Should be (6, sfr.shape)

    outflow_err : (array)
        array of the outflow result errors from KOFFEE

    stat_results : (array)
        array of statistical results from KOFFEE.  Should be same shape as sfr.

    z : float
        redshift

    Returns
    -------
    A six panel graph of velocity offset, velocity dispersion and outflow velocity against
    the SFR surface density

    """
    #calculate the outflow velocity
    vel_diff, vel_diff_err, vel_out, vel_out_err = calc_outvel.calc_outflow_vel(OIII_outflow_results, OIII_outflow_error, statistical_results, z)

    #calculate the sfr surface density - using just the systemic line, and including the flux line
    #don't include extinction since this was included in the continuum subtraction using ppxf
    sfr_sys, total_sfr_sys, sfr_surface_density_sys, h_beta_integral_err_sys = calc_sfr.calc_sfr_koffee(hbeta_outflow_results, hbeta_outflow_error, hbeta_no_outflow_results, hbeta_no_outflow_error, statistical_results, z, include_extinction=False, include_outflow=False)
    sfr_flow, total_sfr_flow, sfr_surface_density_flow, h_beta_integral_err_flow = calc_sfr.calc_sfr_koffee(hbeta_outflow_results, hbeta_outflow_error, hbeta_no_outflow_results, hbeta_no_outflow_error, statistical_results, z, include_extinction=False, include_outflow=True)

    print('$\Sigma_{SFR}$ difference between including and not including flow for whole cube:', np.nanmedian(sfr_surface_density_sys)-np.nanmedian(sfr_surface_density_flow))
    print('')

    print('$\Sigma_{SFR}$ difference between including and not including flow for most SFing spaxels:', np.nanmedian(sfr_surface_density_sys.reshape(-1)[sfr_surface_density_sys.reshape(-1)>np.nanquantile(sfr_surface_density_sys, 0.5)])-np.nanmedian(sfr_surface_density_flow.reshape(-1)[sfr_surface_density_flow.reshape(-1)>np.nanquantile(sfr_surface_density_flow, 0.5)]))
    print('')

    print('$\Sigma_{SFR}$ difference between including and not including flow for least SFing spaxels:', np.nanmedian(sfr_surface_density_sys.reshape(-1)[sfr_surface_density_sys.reshape(-1)<=np.nanquantile(sfr_surface_density_sys, 0.5)])-np.nanmedian(sfr_surface_density_flow.reshape(-1)[sfr_surface_density_flow.reshape(-1)<=np.nanquantile(sfr_surface_density_flow, 0.5)]))
    print('')

    #get the sfr for the outflow spaxels
    flow_mask = (statistical_results>0)

    #convert the sigma to km/s instead of Angstroms
    flow_sigma = OIII_outflow_results[3,:,:][flow_mask]/(1+z)
    systemic_mean = OIII_outflow_results[1,:,:][flow_mask]/(1+z)
    vel_disp = flow_sigma*299792.458/systemic_mean

    vel_disp_err = (flow_sigma/systemic_mean)*np.sqrt((OIII_outflow_error[3,:,:][flow_mask]/flow_sigma)**2 + (OIII_outflow_error[1,:,:][flow_mask]/systemic_mean)**2)

    #flatten all the arrays and get rid of extra spaxels
    sfr_sys = sfr_surface_density_sys[flow_mask]
    sfr_flow = sfr_surface_density_flow[flow_mask]
    sfr_sys_err = h_beta_integral_err_sys[flow_mask]
    sfr_flow_err = h_beta_integral_err_flow[flow_mask]
    vel_diff = vel_diff[flow_mask]
    vel_diff_err = vel_diff_err[flow_mask]
    vel_out = vel_out[flow_mask]
    vel_out_err = vel_out_err[flow_mask]
    radius = radius[flow_mask]

    #make sure none of the errors are nan values
    vel_diff_err[np.where(np.isnan(vel_diff_err)==True)] = np.nanmedian(vel_diff_err)
    vel_out_err[np.where(np.isnan(vel_out_err)==True)] = np.nanmedian(vel_out_err)
    vel_disp_err[np.where(np.isnan(vel_disp_err)==True)] = np.nanmedian(vel_disp_err)

    #do the calculations for all the bins
    num_bins = 5
    min_bin = None #-0.05
    max_bin = None #0.6

    if weighted_average == False:
        bin_center, v_out_bin_medians_sys, v_out_bin_lower_q_sys, v_out_bin_upper_q_sys = binned_median_quantile_log(sfr_sys, vel_out, num_bins=num_bins, weights=None, min_bin=None, max_bin=None)
        bin_center, vel_diff_bin_medians_sys, vel_diff_bin_lower_q_sys, vel_diff_bin_upper_q_sys = binned_median_quantile_log(sfr_sys, vel_diff, num_bins=num_bins, weights=None, min_bin=min_bin, max_bin=max_bin)
        bin_center, disp_bin_medians_sys, disp_bin_lower_q_sys, disp_bin_upper_q_sys = binned_median_quantile_log(sfr_sys, vel_disp, num_bins=num_bins, weights=None, min_bin=min_bin, max_bin=max_bin)

        bin_center, v_out_bin_medians_flow, v_out_bin_lower_q_flow, v_out_bin_upper_q_flow = binned_median_quantile_log(sfr_flow, vel_out, num_bins=num_bins, weights=None, min_bin=min_bin, max_bin=max_bin)
        bin_center, vel_diff_bin_medians_flow, vel_diff_bin_lower_q_flow, vel_diff_bin_upper_q_flow = binned_median_quantile_log(sfr_flow, vel_diff, num_bins=num_bins, weights=None, min_bin=min_bin, max_bin=max_bin)
        bin_center, disp_bin_medians_flow, disp_bin_lower_q_flow, disp_bin_upper_q_flow = binned_median_quantile_log(sfr_flow, vel_disp, num_bins=num_bins, weights=None, min_bin=min_bin, max_bin=max_bin)

    elif weighted_average == True:
        bin_center, v_out_bin_medians_sys, v_out_bin_lower_q_sys, v_out_bin_upper_q_sys = binned_median_quantile_log(sfr_sys, vel_out, num_bins=num_bins, weights=[vel_out_err], min_bin=None, max_bin=None)
        bin_center, vel_diff_bin_medians_sys, vel_diff_bin_lower_q_sys, vel_diff_bin_upper_q_sys = binned_median_quantile_log(sfr_sys, vel_diff, num_bins=num_bins, weights=[vel_diff_err], min_bin=min_bin, max_bin=max_bin)
        bin_center, disp_bin_medians_sys, disp_bin_lower_q_sys, disp_bin_upper_q_sys = binned_median_quantile_log(sfr_sys, vel_disp, num_bins=num_bins, weights=[vel_disp_err], min_bin=min_bin, max_bin=max_bin)

        bin_center, v_out_bin_medians_flow, v_out_bin_lower_q_flow, v_out_bin_upper_q_flow = binned_median_quantile_log(sfr_flow, vel_out, num_bins=num_bins, weights=[vel_out_err], min_bin=min_bin, max_bin=max_bin)
        bin_center, vel_diff_bin_medians_flow, vel_diff_bin_lower_q_flow, vel_diff_bin_upper_q_flow = binned_median_quantile_log(sfr_flow, vel_diff, num_bins=num_bins, weights=[vel_diff_err], min_bin=min_bin, max_bin=max_bin)
        bin_center, disp_bin_medians_flow, disp_bin_lower_q_flow, disp_bin_upper_q_flow = binned_median_quantile_log(sfr_flow, vel_disp, num_bins=num_bins, weights=[vel_disp_err], min_bin=min_bin, max_bin=max_bin)


    print(bin_center)
    print(v_out_bin_medians_sys)

    #calculate the r value for the median values
    r_vel_out_med_sys, p_value_v_out = pearson_correlation(bin_center, v_out_bin_medians_sys)
    r_vel_diff_med_sys, p_value_v_diff = pearson_correlation(bin_center, vel_diff_bin_medians_sys)
    r_disp_med_sys, p_value_disp = pearson_correlation(bin_center, disp_bin_medians_sys)

    r_vel_out_med_flow, p_value_v_out = pearson_correlation(bin_center, v_out_bin_medians_flow)
    r_vel_diff_med_flow, p_value_v_diff = pearson_correlation(bin_center, vel_diff_bin_medians_flow)
    r_disp_med_flow, p_value_disp = pearson_correlation(bin_center, disp_bin_medians_flow)

    #calculate the r value for all the values
    r_vel_out_sys, p_value_v_out = pearson_correlation(sfr_sys, vel_out)
    r_vel_diff_sys, p_value_v_diff = pearson_correlation(sfr_sys, vel_diff)
    r_disp_sys, p_value_disp = pearson_correlation(sfr_sys, vel_disp)

    r_vel_out_flow, p_value_v_out = pearson_correlation(sfr_flow, vel_out)
    r_vel_diff_flow, p_value_v_diff = pearson_correlation(sfr_flow, vel_diff)
    r_disp_flow, p_value_disp = pearson_correlation(sfr_flow, vel_disp)


    #create vectors to plot the literature trends
    sfr_surface_density_chen, v_out_chen = chen_et_al_2010(sfr_sys.min(), sfr_sys.max(), scale_factor=np.nanmedian(vel_out)/(np.nanmedian(sfr_flow)**0.1))
    sfr_surface_density_murray, v_out_murray = murray_et_al_2011(sfr_sys.min(), sfr_sys.max(), scale_factor=np.nanmedian(vel_out)/(np.nanmedian(sfr_flow)**2))

    sfr_surface_density_chen, vel_diff_chen = chen_et_al_2010(sfr_sys.min(), sfr_sys.max(), scale_factor=np.nanmedian(vel_diff)/(np.nanmedian(sfr_flow)**0.1))
    sfr_surface_density_murray, vel_diff_murray = murray_et_al_2011(sfr_sys.min(), sfr_sys.max(), scale_factor=np.nanmedian(vel_diff)/(np.nanmedian(sfr_flow)**2))

    #plot it
    fig, ax = plt.subplots(nrows=2, ncols=3, sharex=True, figsize=(12,7))
    plt.rcParams['axes.facecolor']='white'

    #----------------
    #Including Outflow Line Plots
    #-----------------
    if colour_by_radius == True:
        ax[0,0].scatter(sfr_flow[vel_disp>=51], vel_out[vel_disp>=51], marker='o', lw=0, label='Flow spaxels; R={:.2f}'.format(r_vel_out_flow), alpha=0.6, c=radius[vel_disp>=51])
        ax[0,0].scatter(sfr_flow[vel_disp<51], vel_out[vel_disp<51], marker='v', lw=0, alpha=0.6, c=radius[vel_disp<51])

        ax[0,1].scatter(sfr_flow[vel_disp>=51], vel_diff[vel_disp>=51], marker='o', lw=0, alpha=0.6, label='Flow spaxels; R={:.2f}'.format(r_vel_diff_flow), c=radius[vel_disp>=51])
        ax[0,1].scatter(sfr_flow[vel_disp<51], vel_diff[vel_disp<51], marker='v', lw=0, alpha=0.6, c=radius[vel_disp<51])

        im = ax[0,2].scatter(sfr_flow[vel_disp>=51], vel_disp[vel_disp>=51], marker='o', lw=0, alpha=0.6, label='Flow spaxels; R={:.2f}'.format(r_disp_flow), c=radius[vel_disp>=51])
        ax[0,2].scatter(sfr_flow[vel_disp<51], vel_disp[vel_disp<51], marker='v', lw=0, alpha=0.6, c=radius[vel_disp<51])
        cbar = plt.colorbar(im, ax=ax[0,2])
        cbar.ax.set_ylabel('Radius (Arcsec)')

    elif colour_by_radius == False:
        ax[0,0].plot(sfr_flow[vel_disp>=51], vel_out[vel_disp>=51], marker='o', lw=0, label='Flow spaxels; R={:.2f}'.format(r_vel_out_flow), alpha=0.4, color='tab:blue')
        ax[0,0].plot(sfr_flow[vel_disp<51], vel_out[vel_disp<51], marker='v', lw=0, alpha=0.4, color='tab:blue')

        ax[0,1].plot(sfr_flow[vel_disp>=51], vel_diff[vel_disp>=51], marker='o', lw=0, alpha=0.4, label='Flow spaxels; R={:.2f}'.format(r_vel_diff_flow), color='tab:blue')
        ax[0,1].plot(sfr_flow[vel_disp<51], vel_diff[vel_disp<51], marker='v', lw=0, alpha=0.4, color='tab:blue')

        ax[0,2].plot(sfr_flow[vel_disp>=51], vel_disp[vel_disp>=51], marker='o', lw=0, alpha=0.4, label='Flow spaxels; R={:.2f}'.format(r_disp_flow), color='tab:blue')
        ax[0,2].plot(sfr_flow[vel_disp<51], vel_disp[vel_disp<51], marker='v', lw=0, alpha=0.4, color='tab:blue')

    ax[0,0].fill_between(bin_center, v_out_bin_lower_q_flow, v_out_bin_upper_q_flow, color='tab:blue', alpha=0.3)
    ax[0,0].plot(bin_center, v_out_bin_medians_flow, marker='', color='tab:blue', lw=3.0, label='Median; R={:.2f}'.format(r_vel_out_med_flow))
    ax[0,0].errorbar(5, 150, xerr=np.nanmedian(sfr_flow_err), yerr=np.nanmedian(vel_out_err), c='k')
    ax[0,0].plot(sfr_surface_density_chen, v_out_chen, ':k', label='Energy driven, $v_{out} \propto \Sigma_{SFR}^{0.1}$')
    ax[0,0].plot(sfr_surface_density_murray, v_out_murray, '--k', label='Momentum driven, $v_{out} \propto \Sigma_{SFR}^{2}$')
    ax[0,0].set_ylim(100, 500)
    ax[0,0].set_xscale('log')
    lgnd = ax[0,0].legend(frameon=False, fontsize='x-small', loc='lower left')
    lgnd.legendHandles[0]._legmarker.set_markersize(4)
    ax[0,0].set_ylabel('Maximum Outflow Velocity [km s$^{-1}$]')
    #ax[0,0].set_xlabel('$\Sigma_{SFR}$ [M$_\odot$ yr$^{-1}$ kpc$^{-2}$]')


    ax[0,1].fill_between(bin_center, vel_diff_bin_lower_q_flow, vel_diff_bin_upper_q_flow, color='tab:blue', alpha=0.3)
    ax[0,1].plot(bin_center, vel_diff_bin_medians_flow, marker='', color='tab:blue', lw=3.0, label='Median; R={:.2f}'.format(r_vel_diff_med_flow))
    ax[0,1].errorbar(5, -100, xerr=np.nanmedian(sfr_flow_err), yerr=np.nanmedian(vel_diff_err), c='k')
    ax[0,1].plot(sfr_surface_density_chen, vel_diff_chen, ':k', label='Energy driven, $v_{out} \propto \Sigma_{SFR}^{0.1}$')
    ax[0,1].plot(sfr_surface_density_murray, vel_diff_murray, '--k', label='Momentum driven, $v_{out} \propto \Sigma_{SFR}^{2}$')
    ax[0,1].set_ylim(-150,250)
    ax[0,1].set_xscale('log')
    lgnd = ax[0,1].legend(frameon=False, fontsize='x-small', loc='lower left')
    lgnd.legendHandles[0]._legmarker.set_markersize(4)
    ax[0,1].set_title('$\Sigma_{SFR}$ calculated using full line')
    ax[0,1].set_ylabel('Velocity Offset [km s$^{-1}$]')
    #ax[0,1].set_xlabel('$\Sigma_{SFR}$ [M$_\odot$ yr$^{-1}$ kpc$^{-2}$]')


    ax[0,2].fill_between(bin_center, disp_bin_lower_q_flow, disp_bin_upper_q_flow, color='tab:blue', alpha=0.3)
    ax[0,2].plot(bin_center, disp_bin_medians_flow, marker='', color='tab:blue', lw=3.0, label='Median; R={:.2f}'.format(r_disp_med_flow))
    ax[0,2].errorbar(5, -50, xerr=np.nanmedian(sfr_flow_err), yerr=np.nanmedian(vel_disp_err), c='k')
    ax[0,2].set_xscale('log')
    ax[0,2].set_ylim(-100,300)
    lgnd = ax[0,2].legend(frameon=False, fontsize='x-small', loc='lower left')
    lgnd.legendHandles[0]._legmarker.set_markersize(4)
    ax[0,2].set_ylabel('Velocity Dispersion [km s$^{-1}$]')
    #ax[0,2].set_xlabel('$\Sigma_{SFR}$ [M$_\odot$ yr$^{-1}$ kpc$^{-2}$]')

    #----------------
    #Systemic Line Plots
    #-----------------
    if colour_by_radius == True:
        ax[1,0].scatter(sfr_sys[vel_disp>=51], vel_out[vel_disp>=51], marker='o', lw=0, label='Flow spaxels; R={:.2f}'.format(r_vel_out_sys), alpha=0.6, c=radius[vel_disp>=51])
        ax[1,0].scatter(sfr_sys[vel_disp<51], vel_out[vel_disp<51], marker='v', lw=0, alpha=0.6, c=radius[vel_disp<51])

        ax[1,1].scatter(sfr_sys[vel_disp>=51], vel_diff[vel_disp>=51], marker='o', lw=0, alpha=0.6, label='Flow spaxels; R={:.2f}'.format(r_vel_diff_sys), c=radius[vel_disp>=51])
        ax[1,1].scatter(sfr_sys[vel_disp<51], vel_diff[vel_disp<51], marker='v', lw=0, alpha=0.6, c=radius[vel_disp<51])

        im = ax[1,2].scatter(sfr_sys[vel_disp>=51], vel_disp[vel_disp>=51], marker='o', lw=0, alpha=0.6, label='Flow spaxels; R={:.2f}'.format(r_disp_sys), c=radius[vel_disp>=51])
        ax[1,2].scatter(sfr_sys[vel_disp<51], vel_disp[vel_disp<51], marker='v', lw=0, alpha=0.6, c=radius[vel_disp<51])
        cbar = plt.colorbar(im, ax=ax[1,2])
        cbar.ax.set_ylabel('Radius (Arcsec)')

    elif colour_by_radius == False:
        ax[1,0].plot(sfr_sys[vel_disp>=51], vel_out[vel_disp>=51], marker='o', lw=0, label='Flow spaxels; R={:.2f}'.format(r_vel_out_sys), alpha=0.4, color='tab:blue')
        ax[1,0].plot(sfr_sys[vel_disp<51], vel_out[vel_disp<51], marker='v', lw=0, alpha=0.4, color='tab:blue')

        ax[1,1].plot(sfr_sys[vel_disp>=51], vel_diff[vel_disp>=51], marker='o', lw=0, alpha=0.4, label='Flow spaxels; R={:.2f}'.format(r_vel_diff_sys), color='tab:blue')
        ax[1,1].plot(sfr_sys[vel_disp<51], vel_diff[vel_disp<51], marker='v', lw=0, alpha=0.4, color='tab:blue')

        ax[1,2].plot(sfr_sys[vel_disp>=51], vel_disp[vel_disp>=51], marker='o', lw=0, alpha=0.4, label='Flow spaxels; R={:.2f}'.format(r_disp_sys), color='tab:blue')
        ax[1,2].plot(sfr_sys[vel_disp<51], vel_disp[vel_disp<51], marker='v', lw=0, alpha=0.4, color='tab:blue')

    ax[1,0].fill_between(bin_center, v_out_bin_lower_q_sys, v_out_bin_upper_q_sys, color='tab:blue', alpha=0.3)
    ax[1,0].plot(bin_center, v_out_bin_medians_sys, marker='', color='tab:blue', lw=3.0, label='Median; R={:.2f}'.format(r_vel_out_med_sys))
    ax[1,0].errorbar(5, 150, xerr=np.nanmedian(sfr_sys_err), yerr=np.nanmedian(vel_out_err), c='k')
    ax[1,0].plot(sfr_surface_density_chen, v_out_chen, ':k', label='Energy driven, $v_{out} \propto \Sigma_{SFR}^{0.1}$')
    ax[1,0].plot(sfr_surface_density_murray, v_out_murray, '--k', label='Momentum driven, $v_{out} \propto \Sigma_{SFR}^{2}$')
    ax[1,0].set_ylim(100, 500)
    ax[1,0].set_xscale('log')
    lgnd = ax[1,0].legend(frameon=False, fontsize='x-small', loc='lower left')
    lgnd.legendHandles[0]._legmarker.set_markersize(4)
    ax[1,0].set_ylabel('Maximum Outflow Velocity [km s$^{-1}$]')
    ax[1,0].set_xlabel('$\Sigma_{SFR}$ [M$_\odot$ yr$^{-1}$ kpc$^{-2}$]')


    ax[1,1].fill_between(bin_center, vel_diff_bin_lower_q_sys, vel_diff_bin_upper_q_sys, color='tab:blue', alpha=0.3)
    ax[1,1].plot(bin_center, vel_diff_bin_medians_sys, marker='', color='tab:blue', lw=3.0, label='Median; R={:.2f}'.format(r_vel_diff_med_sys))
    ax[1,1].errorbar(5, -100, xerr=np.nanmedian(sfr_sys_err), yerr=np.nanmedian(vel_diff_err), c='k')
    ax[1,1].plot(sfr_surface_density_chen, vel_diff_chen, ':k', label='Energy driven, $v_{out} \propto \Sigma_{SFR}^{0.1}$')
    ax[1,1].plot(sfr_surface_density_murray, vel_diff_murray, '--k', label='Momentum driven, $v_{out} \propto \Sigma_{SFR}^{2}$')
    ax[1,1].set_ylim(-150,250)
    ax[1,1].set_xscale('log')
    lgnd = ax[1,1].legend(frameon=False, fontsize='x-small', loc='lower left')
    lgnd.legendHandles[0]._legmarker.set_markersize(4)
    ax[1,1].set_title('$\Sigma_{SFR}$ calculated using narrow gaussian')
    ax[1,1].set_ylabel('Velocity Offset [km s$^{-1}$]')
    ax[1,1].set_xlabel('$\Sigma_{SFR}$ [M$_\odot$ yr$^{-1}$ kpc$^{-2}$]')


    ax[1,2].fill_between(bin_center, disp_bin_lower_q_sys, disp_bin_upper_q_sys, color='tab:blue', alpha=0.3)
    ax[1,2].plot(bin_center, disp_bin_medians_sys, marker='', color='tab:blue', lw=3.0, label='Median; R={:.2f}'.format(r_disp_med_sys))
    ax[1,2].errorbar(5, -50, xerr=np.nanmedian(sfr_sys_err), yerr=np.nanmedian(vel_disp_err), c='k')
    ax[1,2].set_xscale('log')
    ax[1,2].set_xlim(0.007, 10)
    ax[1,2].set_ylim(-100,300)
    lgnd = ax[1,2].legend(frameon=False, fontsize='x-small', loc='lower left')
    lgnd.legendHandles[0]._legmarker.set_markersize(4)
    ax[1,2].set_ylabel('Velocity Dispersion [km s$^{-1}$]')
    ax[1,2].set_xlabel('$\Sigma_{SFR}$ [M$_\odot$ yr$^{-1}$ kpc$^{-2}$]')

    plt.tight_layout()
    plt.show()


def plot_flux_vout(OIII_outflow_results, OIII_outflow_error, flux_line_outflow_results, flux_line_outflow_error, statistical_results, radius, z):
    """
    Plots the line flux ratio against the outflow velocity.

    Parameters
    ----------
    outflow_results : (array)
        array of outflow results from KOFFEE.  Should be (6, sfr.shape)

    outflow_err : (array)
        array of the outflow result errors from KOFFEE

    stat_results : (array)
        array of statistical results from KOFFEE.  Should be same shape as sfr.

    z : float
        redshift

    Returns
    -------
    A three panel graph of velocity offset, velocity dispersion and outflow velocity against
    the flux ratio

    """
    #calculate the outflow velocity
    vel_diff, vel_diff_err, vel_out, vel_out_err = calc_outvel.calc_outflow_vel(OIII_outflow_results, OIII_outflow_error, statistical_results, z)

    #calculate the flux for systematic and flow gaussians
    systemic_flux, systemic_flux_err, outflow_flux, outflow_flux_err = calc_sfr.calc_flux_from_koffee(flux_line_outflow_results, flux_line_outflow_error, statistical_results, z, outflow=True)

    #make the flow mask
    flow_mask = (statistical_results>0)

    #convert the sigma to km/s instead of Angstroms
    flow_sigma = OIII_outflow_results[3,:,:][flow_mask]/(1+z)
    systemic_mean = OIII_outflow_results[1,:,:][flow_mask]/(1+z)
    vel_disp = flow_sigma*299792.458/systemic_mean

    vel_disp_err = (flow_sigma/systemic_mean)*np.sqrt((OIII_outflow_error[3,:,:][flow_mask]/flow_sigma)**2 + (OIII_outflow_error[1,:,:][flow_mask]/systemic_mean)**2)

    #flatten all the arrays and get rid of extra spaxels
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
    bin_center, v_out_bin_medians, v_out_bin_lower_q, v_out_bin_upper_q = binned_median_quantile_lin(flux_ratio, vel_out, num_bins=num_bins, weights=None, min_bin=None, max_bin=None)
    bin_center, vel_diff_bin_medians, vel_diff_bin_lower_q, vel_diff_bin_upper_q = binned_median_quantile_lin(flux_ratio, vel_diff, num_bins=num_bins, weights=None, min_bin=min_bin, max_bin=max_bin)
    bin_center, disp_bin_medians, disp_bin_lower_q, disp_bin_upper_q = binned_median_quantile_lin(flux_ratio, vel_disp, num_bins=num_bins, weights=None, min_bin=min_bin, max_bin=max_bin)



    #fit our own trends
    #popt_vout, pcov_vout = curve_fit(fitting_function, flux_ratio_bin_medians, v_out_bin_medians)
    #popt_vel_diff, pcov_vel_diff = curve_fit(fitting_function, flux_ratio_bin_medians, vel_diff_bin_medians)
    #popt_disp, pcov_disp = curve_fit(fitting_function, flux_ratio_bin_medians, disp_bin_medians)

    #calculate the r value for the median values
    r_vel_out, p_value_v_out = pearson_correlation(bin_center, v_out_bin_medians)
    r_vel_diff, p_value_v_diff = pearson_correlation(bin_center, vel_diff_bin_medians)
    r_disp, p_value_disp = pearson_correlation(bin_center, disp_bin_medians)



    #plot it
    fig, ax = plt.subplots(nrows=1, ncols=3, sharex=True, figsize=(12,5))
    plt.rcParams['axes.facecolor']='white'

    ax[0].scatter(flux_ratio, vel_out, marker='o', lw=0, label='Flow spaxels', alpha=0.6, c=radius)
    ax[0].fill_between(bin_center, v_out_bin_lower_q, v_out_bin_upper_q, color='tab:blue', alpha=0.3)
    ax[0].plot(bin_center, v_out_bin_medians, marker='', color='tab:blue', lw=3.0, label='Median; R={:.2f}'.format(r_vel_out))
    #ax[0].plot(sfr_linspace, fitting_function(sfr_linspace, *popt_vout), 'r-', label='Fit: $v_{out}=%5.0f$ $\Sigma_{SFR}^{%5.2f}$' % tuple(popt_vout))
    ax[0].legend(frameon=False, fontsize='x-small', loc='lower left')
    ax[0].set_ylabel('Maximum Outflow Velocity [km s$^{-1}$]')
    ax[0].set_xlabel('Log Broad/Narrow Flux')

    ax[1].scatter(flux_ratio, vel_diff, marker='o', lw=0, alpha=0.6, c=radius)
    ax[1].fill_between(bin_center, vel_diff_bin_lower_q, vel_diff_bin_upper_q, color='tab:blue', alpha=0.3)
    ax[1].plot(bin_center, vel_diff_bin_medians, marker='', color='tab:blue', lw=3.0, label='Median; R={:.2f}'.format(r_vel_diff))
    #ax[1].plot(sfr_linspace, fitting_function(sfr_linspace, *popt_vel_diff), 'r-', label='Fit: $\mu_{sys}-\mu_{flow}=%5.0f$ $\Sigma_{SFR}^{%5.2f}$' % tuple(popt_vel_diff))
    #ax[1].set_xscale('log')
    ax[1].legend(frameon=False, fontsize='x-small', loc='lower left')
    ax[1].set_ylabel('Velocity Offset [km s$^{-1}$]')
    ax[1].set_xlabel('Log Broad/Narrow Flux')


    im = ax[2].scatter(flux_ratio, vel_disp, marker='o', lw=0, alpha=0.6, c=radius)
    ax[2].fill_between(bin_center, disp_bin_lower_q, disp_bin_upper_q, color='tab:blue', alpha=0.3)
    ax[2].plot(bin_center, disp_bin_medians, marker='', color='tab:blue', lw=3.0, label='Median; R={:.2f}'.format(r_disp))
    #ax[2].plot(sfr_linspace, fitting_function(sfr_linspace, *popt_disp), 'r-', label='Fit: $\sigma_{flow}=%5.0f$ $\Sigma_{SFR}^{%5.2f}$' % tuple(popt_disp))
    #ax[2].set_xscale('log')
    #ax[2].set_ylim(30,230)
    plt.colorbar(im)
    ax[2].legend(frameon=False, fontsize='x-small', loc='lower left')
    ax[2].set_ylabel('Velocity Dispersion [km s$^{-1}$]')
    ax[2].set_xlabel('Log Broad/Narrow Flux')

    plt.tight_layout()
    plt.show()

#===============================================================================
# PLOTTING FUNCTIONS - other
#===============================================================================
def plot_sfr_surface_density_radius(sig_sfr, rad_flat, stat_results):
    """
    Plots the SFR surface density against galaxy radius

    Inputs:
        sig_sfr: the SFR surface density
        rad: the flattend array of radius
        stat_results: the statistical results from KOFFEE

    """
    #create the flow mask
    flow_mask = (stat_results > 0)

    #create the no-flow mask
    no_flow_mask = (stat_results == 0)

    #apply to the rad array
    rad_flow = rad_flat[flow_mask]
    rad_no_flow = rad_flat[no_flow_mask]

    #apply to the SFR surface density array
    sig_sfr_flow = sig_sfr[flow_mask]
    sig_sfr_no_flow = sig_sfr[no_flow_mask]

    #plot
    plt.figure()
    plt.scatter(rad_flow, sig_sfr_flow, label='Outflow spaxels')
    plt.scatter(rad_no_flow, sig_sfr_no_flow, label='No outflow spaxels')
    plt.yscale('log')
    plt.xlabel('Radius [Arcseconds]')
    plt.ylabel('Log $\Sigma_{SFR}$ [M$_\odot$ yr$^{-1}$]')
    plt.legend()
    plt.show()

def plot_vel_out_radius(rad, outflow_results, stat_results, z):
    """
    Plots the SFR surface density against galaxy radius

    Inputs:
        sig_sfr: the SFR surface density
        rad: the array of radius
        stat_results: the statistical results from KOFFEE

    """
    #create the flow mask
    flow_mask = (stat_results > 0)

    #de-redshift the data first!!!
    systemic_mean = outflow_results[1,:]/(1+z)
    flow_mean = outflow_results[4,:]/(1+z)
    flow_sigma = outflow_results[3,:]/(1+z)

    #find the velocity difference
    #doing c*(lam_gal-lam_out)/lam_gal
    vel_diff = 299792.458*(systemic_mean-flow_mean)/systemic_mean

    v_out = 2*flow_sigma*299792.458/systemic_mean + vel_diff

    vel_out = v_out[flow_mask]
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

    return v_out

def plot_outflow_frequency_radius(rad_flat, stat_results):
    """
    Plots the frequency of outflow spaxels, and non-outflow spaxels against radius, using the flattened arrays.
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


def plot_outflow_frequency_sfr_surface_density(sig_sfr, stat_results):
    """
    Plots the frequency of outflow spaxels, and non-outflow spaxels against radius, using the flattened arrays.
    """
    #create flow mask
    flow_mask = (stat_results > 0)

    #create no_flow mask
    no_flow_mask = (stat_results == 0)

    #create low S/N mask
    low_sn_mask = np.isnan(stat_results)

    #get the radius array for each type of spaxel
    sig_sfr_flow = sig_sfr[flow_mask]
    sig_sfr_no_flow = sig_sfr[no_flow_mask]
    sig_sfr_low_sn = sig_sfr[low_sn_mask]

    #get total number of spaxels
    total_spaxels = stat_results.shape[0]
    print(total_spaxels)
    total_outflows = sig_sfr_flow.shape[0]
    total_no_flows = sig_sfr_no_flow.shape[0]

    #iterate through the sfr surface densities and get the number of spaxels in that range
    num_flow = []
    num_no_flow = []
    num_low_sn = []
    num_spax_sig_sfr = []
    num_spax_flow_sig_sfr = []

    cum_spax = []
    cumulative_spaxels = 0

    step_range = 0.1

    for i in np.arange(sig_sfr.min(),sig_sfr.max(),step_range):
        flow_bin = sig_sfr_flow[(sig_sfr_flow>=i)&(sig_sfr_flow<(i+step_range))].shape[0]
        no_flow_bin = sig_sfr_no_flow[(sig_sfr_no_flow>=i)&(sig_sfr_no_flow<(i+step_range))].shape[0]
        low_sn_bin = sig_sfr_low_sn[(sig_sfr_low_sn>=i)&(sig_sfr_low_sn<(i+step_range))].shape[0]

        sig_sfr_bin = flow_bin+no_flow_bin+low_sn_bin
        sig_sfr_flow_bin = flow_bin + no_flow_bin

        cumulative_spaxels = cumulative_spaxels + sig_sfr_flow_bin
        cum_spax.append(cumulative_spaxels)

        num_flow.append(flow_bin)
        num_no_flow.append(no_flow_bin)
        num_low_sn.append(low_sn_bin)
        num_spax_sig_sfr.append(sig_sfr_bin)
        num_spax_flow_sig_sfr.append(sig_sfr_flow_bin)

    num_flow = np.array(num_flow)
    num_no_flow = np.array(num_no_flow)
    num_low_sn = np.array(num_low_sn)
    num_spax_sig_sfr = np.array(num_spax_sig_sfr)
    num_spax_flow_sig_sfr = np.array(num_spax_flow_sig_sfr)
    cum_spax = np.array(cum_spax)
    cum_spax = cum_spax/cum_spax[-1]

    #create the fractions
    flow_fraction = num_flow/num_spax_flow_sig_sfr
    no_flow_fraction = num_no_flow/num_spax_flow_sig_sfr

    #replace nan values with the mean of the two values in front and behind.
    for index, val in np.ndenumerate(flow_fraction):
        if np.isnan(val):
            flow_fraction[index] = np.nanmean([flow_fraction[index[0]-1],flow_fraction[index[0]+1]])

    for index, val in np.ndenumerate(no_flow_fraction):
        if np.isnan(val):
            no_flow_fraction[index] = np.nanmean([no_flow_fraction[index[0]-1],no_flow_fraction[index[0]+1]])

    #fit the frequencies using a spline fit
    sig_sfr_range = np.log10(np.arange(sig_sfr.min(),sig_sfr.max(),step_range))
    fit_flow = interp1d(sig_sfr_range, flow_fraction)
    fit_no_flow = interp1d(sig_sfr_range, no_flow_fraction)
    #new_sig_sfr_range = np.log10(np.arange(sig_sfr.min(),sig_sfr.max(),step_range/2))

    #create the plot
    fig, ax = plt.subplots()
    ax.fill_between(np.arange(sig_sfr.min(),sig_sfr.max(),step_range),0.0, cum_spax, color='grey', alpha=0.4)
    ax.plot(np.arange(sig_sfr.min(),sig_sfr.max(),step_range), flow_fraction, label="Outflow Spaxels", ls='-', lw=3, color='tab:blue')
    #ax.plot(sig_sfr_range, num_flow/num_spax_flow_sig_sfr, label="Outflow Spaxels", ls='-', lw=3)
    #ax.plot(sig_sfr_range, fit_flow(sig_sfr_range), label='Fit to Outflow Spaxels')
    ax.text(1.2, 0.91, 'Outflow Spaxels', fontsize=14, color='tab:blue')
    ax.plot(np.arange(sig_sfr.min(),sig_sfr.max(),step_range), no_flow_fraction, label="No Flow Spaxels", ls='-', lw=3, color='tab:orange')
    #ax.plot(sig_sfr_range, fit_no_flow(sig_sfr_range), label='Fit to No flow Spaxels')
    ax.text(1.1, 0.05, 'No Flow Spaxels', fontsize=14, color='tab:orange')
    ax.axvline(1.0, ls='--', c='k', lw=2, label='Genzel+2011, Newman+2012')
    ax.text(0.83, 0.2, 'Genzel+2011, Newman+2012', rotation='vertical')
    ax.axvline(0.1, ls=':', c='k', lw=2, label='Heckman 2000')
    ax.text(0.083, 0.65, 'Heckman 2000', rotation='vertical')
    ax.set_xscale('log')
    #ax.set_xticks([0,0.5,1,5,10])
    #ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    ax.set_xlabel('Log $\Sigma_{SFR}$ [M$_\odot$ yr$^{-1}$ kpc$^{-2}$]')
    ax.set_ylabel('Spaxel PDF')
    ax.set_xlim(0.02,11)
    #plt.legend(frameon=False, fontsize='x-small', loc=(0.12,0.87))
    plt.show()




def plot_sfr_surface_density_vout_loops(sfr, outflow_results, outflow_error, stat_results, z, h_beta_spec):
    """
    Try the same thing as above but with for loops
    """
    #create arrays to put the flow velocity results in
    vel_res = np.empty_like(sfr)
    vel_diffs = np.empty_like(sfr)
    vel_res_type = np.empty_like(sfr)

    for i in np.arange(sfr.shape[0]):
        #if the maximum flux in the hbeta area is greater than 1.0, we do our maths on it
        if h_beta_spec[:,i].max() > 1.0:
            #find whether stat_results decided if there was an outflow or not
            if stat_results[i] == 1:
                #de-redshift the data
                systemic_mean = outflow_results[1,i]/(1+z)
                flow_mean = outflow_results[4,i]/(1+z)
                flow_sigma = outflow_results[3,i]/(1+z)

                #calculate the velocity difference
                vel_diff = 299792.458*(systemic_mean-flow_mean)/systemic_mean

                #use the velocity difference to decide if it is an in or an outflow
                # if it is positive, it's an outflow
                if vel_diff > 0:
                    vel_res_type[i] = 1.0
                    #calculate the v_out
                    vel_res[i] = 2*flow_sigma*299792.458/systemic_mean + vel_diff
                    vel_diffs[i] = vel_diff

                elif vel_diff <= 0:
                    vel_res_type[i] = -1.0
                    #calculate the v_out
                    vel_res[i] = 2*flow_sigma*299792.458/systemic_mean + vel_diff
                    vel_diffs[i] = vel_diff


            #if stat_results is zero, KOFFEE didn't find an outflow
            elif stat_results[i] == 0:
                vel_res[i] = 0.0
                vel_res_type[i] = 0.0
                vel_diffs[i] = 0.0

        #otherwise, we cannot calculate the sfr reliably
        else:
            vel_res[i] = 0.0
            vel_res_type[i] = 0.0
            vel_diffs[i] = 0.0

    #now do the plotting
    plt.figure()
    plt.plot(sfr[vel_res_type==1], vel_res[vel_res_type==1], marker='o', lw=0, label='Outflow', alpha=0.2)
    plt.plot(sfr[vel_res_type==-1], vel_res[vel_res_type==-1], marker='o', lw=0, label='Inflow', alpha=0.2)
    plt.xscale('log')
    plt.legend(frameon=False, fontsize='small', loc='lower left')
    plt.ylabel('Flow velocity (km s$^{-1}$)')
    plt.xlabel('Log Star Formation Rate Surface Density (M$_\odot$ yr$^{-1}$ kpc$^{-2}$)')
    plt.show()

    return vel_res, vel_res_type, vel_diffs


def plot_sigma_vel_diff(outflow_results, outflow_error, stat_results, z):
    """
    Plots the velocity difference between systemic and flow components against the flow sigma
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
    Plots the velocity difference between systemic and flow components against the flow sigma
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
    Plots the velocity difference between systemic and flow components against the flow sigma
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
    Plots the velocity difference between systemic and flow components against the flow sigma
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
