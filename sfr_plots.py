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
from scipy.stats import pearsonr

from . import calculate_outflow_velocity as calc_outvel




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
    r, p_value = pearsonr(x, y)

    return r, p_value


#===============================================================================
# PLOTTING FUNCTIONS - for paper
#===============================================================================

def plot_sfr_vout(sfr, outflow_results, outflow_error, statistical_results, z):
    """
    Plots the SFR against the outflow velocity.

    Parameters
    ----------
    sfr : (array)
        the star formation rate (shape (67, 24))

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
    the SFR surface density

    """
    #calculate the outflow velocity
    vel_diff, vel_diff_err, vel_out, vel_out_err = calc_outvel.calc_outflow_vel(outflow_results, outflow_error, statistical_results, z)

    #get the sfr for the outflow spaxels
    flow_mask = (statistical_results>0)

    #convert the sigma to km/s instead of Angstroms
    flow_sigma = outflow_results[3,:,:][flow_mask]/(1+z)
    systemic_mean = outflow_results[1,:,:][flow_mask]/(1+z)
    vel_disp = flow_sigma*299792.458/systemic_mean

    #flatten all the arrays and get rid of extra spaxels
    sfr = sfr[flow_mask]
    vel_diff = vel_diff[flow_mask]
    vel_diff_err = vel_diff_err[flow_mask]
    vel_out = vel_out[flow_mask]
    vel_out_err = vel_out_err[flow_mask]

    #take the log of sfr_out
    log_sfr = np.log10(sfr)

    #create the median points for the sfr
    first_bin, last_bin = -1., np.nanmax(log_sfr)
    bin_width = (last_bin-first_bin)/8

    #loop through all the bins
    bin_edges = [first_bin, first_bin+bin_width]

    sfr_bin_medians = []
    v_out_bin_medians = []
    vel_diff_bin_medians = []
    disp_bin_medians = []

    sfr_bin_stdev = []
    v_out_bin_stdev = []
    vel_diff_bin_stdev = []
    disp_bin_stdev = []

    while bin_edges[1] <= last_bin+bin_width-bin_width/6:
        #create the bin
        sfr_bin = log_sfr[(log_sfr>=bin_edges[0]) & (log_sfr<bin_edges[1])]
        v_out_bin = vel_out[(log_sfr>=bin_edges[0]) & (log_sfr<bin_edges[1])]
        vel_diff_bin = vel_diff[(log_sfr>=bin_edges[0]) & (log_sfr<bin_edges[1])]
        disp_bin = vel_disp[(log_sfr>=bin_edges[0]) & (log_sfr<bin_edges[1])]

        #find the median in the bin
        sfr_median = np.nanmedian(sfr_bin)
        v_out_median = np.nanmedian(v_out_bin)
        vel_diff_median = np.nanmedian(vel_diff_bin)
        disp_median = np.nanmedian(disp_bin)

        #find the standard deviation in the bin
        sfr_stdev = np.nanstd(sfr_bin)
        v_out_stdev = np.nanstd(v_out_bin)
        vel_diff_stdev = np.nanstd(vel_diff_bin)
        disp_stdev = np.nanstd(disp_bin)

        #use the stdev to cut out any points greater than 2 sigma away from the median
        if np.any(v_out_bin >= v_out_median+2*v_out_stdev) or np.any(v_out_bin <= v_out_median-2*v_out_stdev):
            v_out_median = np.nanmedian(v_out_bin[(v_out_bin>v_out_median-2*v_out_stdev) & (v_out_bin<v_out_median+2*v_out_stdev)])
            v_out_stdev = np.nanstd(v_out_bin[(v_out_bin>v_out_median-2*v_out_stdev) & (v_out_bin<v_out_median+2*v_out_stdev)])

        if np.any(vel_diff_bin >= vel_diff_median+2*vel_diff_stdev) or np.any(vel_diff_bin <= vel_diff_median-2*vel_diff_stdev):
            vel_diff_median = np.nanmedian(vel_diff_bin[(vel_diff_bin>vel_diff_median-2*vel_diff_stdev) & (vel_diff_bin<vel_diff_median+2*vel_diff_stdev)])
            v_out_stdev = np.nanstd(vel_diff_bin[(vel_diff_bin>vel_diff_median-2*vel_diff_stdev) & (vel_diff_bin<vel_diff_median+2*vel_diff_stdev)])

        if np.any(disp_bin >= disp_median+2*disp_stdev) or np.any(disp_bin <= disp_median-2*disp_stdev):
            disp_median = np.nanmedian(disp_bin[(disp_bin>disp_median-2*disp_stdev) & (disp_bin<disp_median+2*disp_stdev)])
            disp_stdev = np.nanstd(disp_bin[(disp_bin>disp_median-2*disp_stdev) & (disp_bin<disp_median+2*disp_stdev)])

        sfr_bin_medians.append(sfr_median)
        v_out_bin_medians.append(v_out_median)
        vel_diff_bin_medians.append(vel_diff_median)
        disp_bin_medians.append(disp_median)

        sfr_bin_stdev.append(sfr_stdev)
        v_out_bin_stdev.append(v_out_stdev)
        vel_diff_bin_stdev.append(vel_diff_stdev)
        disp_bin_stdev.append(disp_stdev)

        #change bin_edges
        bin_edges = [bin_edges[0]+0.15, bin_edges[1]+0.15]

    sfr_bin_medians = np.array(sfr_bin_medians)
    v_out_bin_medians = np.array(v_out_bin_medians)
    vel_diff_bin_medians = np.array(vel_diff_bin_medians)
    disp_bin_medians = np.array(disp_bin_medians)

    sfr_bin_stdev = np.array(sfr_bin_stdev)
    v_out_bin_stdev = np.array(v_out_bin_stdev)
    vel_diff_bin_stdev = np.array(vel_diff_bin_stdev)
    disp_bin_stdev = np.array(disp_bin_stdev)


    #fit our own trends
    popt_vout, pcov_vout = curve_fit(fitting_function, 10**(sfr_bin_medians), v_out_bin_medians)
    popt_vel_diff, pcov_vel_diff = curve_fit(fitting_function, 10**(sfr_bin_medians), vel_diff_bin_medians)
    popt_disp, pcov_disp = curve_fit(fitting_function, 10**(sfr_bin_medians), disp_bin_medians)

    #calculate the r value for the median values
    r_vel_out, p_value_v_out = pearson_correlation(10**sfr_bin_medians, v_out_bin_medians)
    r_vel_diff, p_value_v_diff = pearson_correlation(10**sfr_bin_medians, vel_diff_bin_medians)
    r_disp, p_value_disp = pearson_correlation(10**sfr_bin_medians, disp_bin_medians)

    #create a vector for sfr_out
    sfr_linspace = np.linspace(sfr.min(), sfr.max()+4.0, num=1000)

    #create vectors to plot the literature trends
    sfr_surface_density_chen, v_out_chen = chen_et_al_2010(sfr.min(), sfr.max(), scale_factor=popt_vout[0])
    sfr_surface_density_murray, v_out_murray = murray_et_al_2011(sfr.min(), sfr.max(), scale_factor=50)
    sfr_surface_density_davies, v_disp_davies = davies_et_al_2019(sfr.min(), sfr.max())

    #plot it
    fig, ax = plt.subplots(nrows=1, ncols=3, sharex=True, figsize=(12,5))
    plt.rcParams['axes.facecolor']='white'

    ax[0].plot(10**log_sfr, vel_out, marker='o', lw=0, label='Flow spaxels', alpha=0.4)
    ax[0].fill_between(10**sfr_bin_medians, v_out_bin_medians+v_out_bin_stdev, v_out_bin_medians-v_out_bin_stdev, color='tab:blue', alpha=0.3)
    ax[0].plot(10**sfr_bin_medians, v_out_bin_medians, marker='', color='tab:blue', lw=3.0, label='Median; R={:.2f}'.format(r_vel_out))
    ax[0].plot(sfr_linspace, fitting_function(sfr_linspace, *popt_vout), 'r-', label='Fit: $v_{out}=%5.0f$ $\Sigma_{SFR}^{%5.2f}$' % tuple(popt_vout))
    ax[0].plot(sfr_surface_density_chen, v_out_chen, ':k', label='Energy driven, $v_{out} \propto \Sigma_{SFR}^{0.1}$')
    ax[0].plot(sfr_surface_density_murray, v_out_murray, '--k', label='Momentum driven, $v_{out} \propto \Sigma_{SFR}^{2}$')
    ax[0].set_ylim(100, 500)
    ax[0].set_xscale('log')
    ax[0].legend(frameon=False, fontsize='x-small', loc='lower left')
    ax[0].set_ylabel('Maximum Outflow Velocity [km s$^{-1}$]')
    ax[0].set_xlabel('Log $\Sigma_{SFR}$ [M$_\odot$ yr$^{-1}$ kpc$^{-2}$]')

    ax[1].plot(10**log_sfr, vel_diff, marker='o', lw=0, alpha=0.4)
    ax[1].fill_between(10**sfr_bin_medians, vel_diff_bin_medians+vel_diff_bin_stdev, vel_diff_bin_medians-vel_diff_bin_stdev, color='tab:blue', alpha=0.3)
    ax[1].plot(10**sfr_bin_medians, vel_diff_bin_medians, marker='', color='tab:blue', lw=3.0, label='Median; R={:.2f}'.format(r_vel_diff))
    ax[1].plot(sfr_linspace, fitting_function(sfr_linspace, *popt_vel_diff), 'r-', label='Fit: $\mu_{sys}-\mu_{flow}=%5.0f$ $\Sigma_{SFR}^{%5.2f}$' % tuple(popt_vel_diff))
    ax[1].set_xscale('log')
    ax[1].legend(frameon=False, fontsize='x-small', loc='lower left')
    ax[1].set_ylabel('Velocity Offset [km s$^{-1}$]')
    ax[1].set_xlabel('Log $\Sigma_{SFR}$ [M$_\odot$ yr$^{-1}$ kpc$^{-2}$]')

    ax[2].plot(10**log_sfr, vel_disp, marker='o', lw=0, alpha=0.4)
    ax[2].fill_between(10**sfr_bin_medians, disp_bin_medians+disp_bin_stdev, disp_bin_medians-disp_bin_stdev, color='tab:blue', alpha=0.3)
    ax[2].plot(10**sfr_bin_medians, disp_bin_medians, marker='', color='tab:blue', lw=3.0, label='Median; R={:.2f}'.format(r_disp))
    ax[2].plot(sfr_linspace, fitting_function(sfr_linspace, *popt_disp), 'r-', label='Fit: $\sigma_{flow}=%5.0f$ $\Sigma_{SFR}^{%5.2f}$' % tuple(popt_disp))
    ax[2].plot(sfr_surface_density_davies, v_disp_davies, '--k', label='Davies+19, $\sigma_{out}=241\Sigma_{SFR}^{0.3}$')
    ax[2].set_xscale('log')
    ax[2].set_ylim(30,230)
    ax[2].legend(frameon=False, fontsize='x-small', loc='lower left')
    ax[2].set_ylabel('Velocity Dispersion [km s$^{-1}$]')
    ax[2].set_xlabel('Log $\Sigma_{SFR}$ [M$_\odot$ yr$^{-1}$ kpc$^{-2}$]')

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
