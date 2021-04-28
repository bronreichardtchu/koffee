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

FUNCTIONS INCLUDED:
    plot_sfr_vout
    plot_sfr_mlf_flux
    plot_sfr_flux
    plot_radius_flux
    maps_of_halpha_hbeta
    plot_sfr_vout_multiple
    plot_sfr_vout_correlation_with_binning
    plot_sfr_vout_compare_sfr_calcs
    plot_sfr_surface_density_radius
    plot_outflow_frequency_sfr_surface_density
    plot_sfr_surface_density_vout_loops

MODIFICATION HISTORY:
		v.1.0 - first created September 2020

"""

import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
import cmasher as cmr

from scipy.optimize import curve_fit
from scipy.interpolate import interp1d

from astropy.wcs import WCS
from astropy.cosmology import WMAP9 as cosmo
from astropy.constants import c
from astropy import units as u
from astropy.io import fits

from . import prepare_cubes as pc
from . import plotting_functions as pf
from . import calculate_outflow_velocity as calc_outvel
from . import calculate_star_formation_rate as calc_sfr
from . import calculate_mass_loading_factor as calc_mlf
from . import brons_display_pixels_kcwi as bdpk
from . import koffee




#===============================================================================
# PLOTTING FUNCTION FOR PAPER I
#===============================================================================
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

    ax[1,0].errorbar(0.03, np.nanmin(mlf)-0.1, xerr=np.nanmedian(sig_sfr_err), yerr=[[np.nanmedian(mlf_err_min)], [np.nanmedian(mlf_err_max)]], c='k')

    ax[1,0].set_ylim(-2.4, 0.7)
    ax[1,0].set_xscale('log')
    ax[1,0].set_xlim(0.003, 2)
    lgnd = ax[1,0].legend(frameon=True, fontsize='small', loc='upper left', framealpha=0.5)
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

    ax[1,1].errorbar(0.03, np.nanmin(mlf)-0.1, xerr=np.nanmedian(sig_sfr_err[physical_mask]), yerr=[[np.nanmedian(mlf_err_min[physical_mask])], [np.nanmedian(mlf_err_max[physical_mask])]], c='k')

    #ax[0,1].set_xscale('log')
    lgnd = ax[1,1].legend(frameon=True, fontsize='small', loc='upper left', framealpha=0.5)
    lgnd.legendHandles[0]._legmarker.set_markersize(3)
    ax[1,1].set_xlabel('$\Sigma_{SFR}$ [M$_\odot$ yr$^{-1}$ kpc$^{-2}$]')


    #plot points with strong BIC values
    ax[1,2].fill_between(bin_center_strong, mlf_bin_lower_q_strong, mlf_bin_upper_q_strong, color=colours[2], alpha=0.3)
    ax[1,2].scatter(sig_sfr[~BIC_diff_strong], mlf[~BIC_diff_strong], marker='o', s=10, label='All KOFFEE fits', color=colours[0], alpha=0.3, facecolors='none')
    ax[1,2].scatter(sig_sfr[BIC_diff_strong], mlf[BIC_diff_strong], marker='o', s=10, label='Selected KOFFEE fits; R={:.2f}'.format(r_mlf_strong), color=colours[2], alpha=1.0)
    ax[1,2].plot(bin_center_strong, mlf_bin_medians_strong, marker='', lw=3, label='Median of selected KOFFEE fits; R={:.2f}'.format(r_mlf_med_strong), color=colours[2])

    ax[1,2].plot(sfr_surface_density_kim, np.log10(mlf_Kim), ':k')

    ax[1,2].errorbar(0.03, np.nanmin(mlf)-0.1, xerr=np.nanmedian(sig_sfr_err[BIC_diff_strong]), yerr=[[np.nanmedian(mlf_err_min[BIC_diff_strong])], [np.nanmedian(mlf_err_max[BIC_diff_strong])]], c='k')

    #ax[0,1].set_xscale('log')
    lgnd = ax[1,2].legend(frameon=True, fontsize='small', loc='upper left', framealpha=0.5)
    lgnd.legendHandles[0]._legmarker.set_markersize(3)
    ax[1,2].set_xlabel('$\Sigma_{SFR}$ [M$_\odot$ yr$^{-1}$ kpc$^{-2}$]')




    #plot all points
    ax[0,0].fill_between(bin_center_all, flux_bin_lower_q_all, flux_bin_upper_q_all, color=colours[0], alpha=0.3)
    ax[0,0].scatter(sig_sfr, flux_ratio, marker='o', s=10, label='All KOFFEE fits; R={:.2f}'.format(r_flux_all), color=colours[0], alpha=0.8)
    ax[0,0].plot(bin_center_all, flux_bin_medians_all, marker='', lw=3, label='Median all KOFFEE fits; R={:.2f}'.format(r_flux_med_all), color=colours[0])

    ax[0,0].errorbar(0.03, np.nanmin(flux_ratio), xerr=np.nanmedian(sig_sfr_err), yerr=np.nanmedian(flux_error), c='k')

    ax[0,0].set_ylim((np.nanmin(flux_ratio)+np.nanmedian(flux_error)-0.1), np.nanmax(flux_ratio)+0.6)
    lgnd = ax[0,0].legend(frameon=True, fontsize='small', loc='upper left', framealpha=0.5, edgecolor=None)
    lgnd.legendHandles[0]._legmarker.set_markersize(3)
    ax[0,0].set_ylabel(r'H$\beta$ Log(F$_{broad}$/F$_{narrow}$)')
    ax[0,0].set_title('all spaxels')



    #plot points within 90% radius
    ax[0,1].fill_between(bin_center_physical, flux_bin_lower_q_physical, flux_bin_upper_q_physical, color=colours[1], alpha=0.3)
    ax[0,1].scatter(sig_sfr[radius>6.1], flux_ratio[radius>6.1], marker='o', s=10, label='All KOFFEE fits', edgecolors=colours[0], alpha=0.3, facecolors='none')
    ax[0,1].scatter(sig_sfr[vel_disp<=51], flux_ratio[vel_disp<=51], marker='v', s=10, edgecolors=colours[0], alpha=0.3, facecolors='none')
    ax[0,1].scatter(sig_sfr[physical_mask], flux_ratio[physical_mask], marker='o', s=10, label='Selected KOFFEE fits; R={:.2f}'.format(r_flux_physical), color=colours[1], alpha=0.8)
    ax[0,1].plot(bin_center_physical, flux_bin_medians_physical, marker='', lw=3, label='Median of selected KOFFEE fits; R={:.2f}'.format(r_flux_med_physical), color=colours[1])

    ax[0,1].errorbar(0.03, np.nanmin(flux_ratio), xerr=np.nanmedian(sig_sfr_err[physical_mask]), yerr=np.nanmedian(flux_error[physical_mask]), c='k')

    lgnd = ax[0,1].legend(frameon=True, fontsize='small', loc='upper left', framealpha=0.5, edgecolor=None)
    lgnd.legendHandles[0]._legmarker.set_markersize(3)
    ax[0,1].set_title(r'$r$<$r_{90}$ and $\sigma_{broad}$>$\sigma_{inst}$')



    #plot points with strong BIC values
    ax[0,2].fill_between(bin_center_strong, flux_bin_lower_q_strong, flux_bin_upper_q_strong, color=colours[2], alpha=0.3)
    ax[0,2].scatter(sig_sfr[~BIC_diff_strong], flux_ratio[~BIC_diff_strong], marker='o', s=10, label='All KOFFEE fits', edgecolors=colours[0], alpha=0.3, facecolors='none')
    ax[0,2].scatter(sig_sfr[BIC_diff_strong], flux_ratio[BIC_diff_strong], marker='o', s=10, label='Selected KOFFEE fits; R={:.2f}'.format(r_flux_strong), color=colours[2], alpha=1.0)
    ax[0,2].plot(bin_center_strong, flux_bin_medians_strong, marker='', lw=3, label='Median of selected KOFFEE fits; R={:.2f}'.format(r_flux_med_strong), color=colours[2])

    ax[0,2].errorbar(0.03, np.nanmin(flux_ratio), xerr=np.nanmedian(sig_sfr_err[BIC_diff_strong]), yerr=np.nanmedian(flux_error[BIC_diff_strong]), c='k')

    lgnd = ax[0,2].legend(frameon=True, fontsize='small', loc='upper left', framealpha=0.5, edgecolor=None)
    lgnd.legendHandles[0]._legmarker.set_markersize(3)
    ax[0,2].set_title('strongly likely BIC')

    plt.subplots_adjust(left=0.06, right=0.99, top=0.96, bottom=0.07, wspace=0.04, hspace=0.04)

    plt.show()



#===============================================================================
# OTHER PLOTTING FUNCTIONS
#===============================================================================

def plot_sfr_flux(flux_outflow_results, flux_outflow_error, hbeta_outflow_results, hbeta_outflow_error, hbeta_no_outflow_results, hbeta_no_outflow_error, BIC_outflow, BIC_no_outflow, statistical_results, z, radius, header, flux_ratio_line='OIII', weighted_average=True):
    """
    Plots the SFR surface density against the broad-to-narrow flux ratio, with
    Sigma_SFR calculated using only the narrow component.

    Parameters
    ----------
    flux_outflow_results : :obj:'~numpy.ndarray'
        array of outflow results from KOFFEE for the line we want to calculate
        the flux ratio for.  Should be (7, statistical_results.shape)

    flux_outflow_err : :obj:'~numpy.ndarray'
        array of the outflow result errors from KOFFEE for flux ratio line

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
        array of galaxy radius values

    header : FITS header object
        the header from the fits file

    flux_ratio_line : string
        the emission line being used for the broad-to-narrow flux ratio, used for
        plotting labels (Default='OIII')

    weighted_average : boolean
        whether or not to take a weighted average using the errors (Default=True)

    Returns
    -------
    A graph of broad-to-narrow flux ratio against the SFR surface density

    """
    #calculate the sfr surface density - using just the systemic line, and including the flux line
    #don't include extinction since this was included in the continuum subtraction using ppxf
    sfr, sfr_err, total_sfr, sfr_surface_density, sfr_surface_density_err = calc_sfr.calc_sfr_koffee(hbeta_outflow_results, hbeta_outflow_error, hbeta_no_outflow_results, hbeta_no_outflow_error, statistical_results, z, header, include_extinction=False, include_outflow=False)

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
    sig_sfr_err = sfr_surface_density_err[flow_mask]

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
    flux_ratio = np.log10(outflow_flux/systemic_flux)
    #flux_ratio = (outflow_flux/systemic_flux)

    #calculate the error
    flux_error = flux_ratio * np.log10(np.sqrt((outflow_flux_err/outflow_flux)**2 + (systemic_flux_err/systemic_flux)**2))

    #create BIC diff
    BIC_diff = BIC_outflow - BIC_no_outflow
    BIC_diff_weak = (BIC_diff < -10) & (BIC_diff >= -30)
    BIC_diff_moderate = (BIC_diff < -30) & (BIC_diff >= -50)
    BIC_diff_strong = (BIC_diff < -50)

    #physical limits mask -
    #for the radius mask 6.1" is the 90% radius
    #also mask out the fits which lie on the lower limit of dispersion < 51km/s
    physical_mask = (radius < 6.1) & (vel_disp>51)

    #strong BIC and physical limits mask
    #clean_mask = (radius < 6.1) & (vel_disp > 51) & (BIC_diff < -50)

    #do the calculations for all the bins
    num_bins = 3
    min_bin = None #-0.05
    max_bin = None #0.6

    if weighted_average == False:
        bin_center_all, flux_bin_medians_all, flux_bin_lower_q_all, flux_bin_upper_q_all = pf.binned_median_quantile_log(sig_sfr, flux_ratio, num_bins=num_bins, weights=None, min_bin=min_bin, max_bin=max_bin)
        bin_center_physical, flux_bin_medians_physical, flux_bin_lower_q_physical, flux_bin_upper_q_physical = pf.binned_median_quantile_log(sig_sfr[physical_mask], flux_ratio[physical_mask], num_bins=num_bins, weights=None, min_bin=min_bin, max_bin=max_bin)
        bin_center_strong, flux_bin_medians_strong, flux_bin_lower_q_strong, flux_bin_upper_q_strong = pf.binned_median_quantile_log(sig_sfr[BIC_diff_strong], flux_ratio[BIC_diff_strong], num_bins=num_bins, weights=None, min_bin=min_bin, max_bin=max_bin)

    elif weighted_average == True:
        bin_center_all, flux_bin_medians_all, flux_bin_lower_q_all, flux_bin_upper_q_all = pf.binned_median_quantile_log(sig_sfr, flux_ratio, num_bins=num_bins, weights=[flux_error], min_bin=min_bin, max_bin=max_bin)
        bin_center_physical, flux_bin_medians_physical, flux_bin_lower_q_physical, flux_bin_upper_q_physical = pf.binned_median_quantile_log(sig_sfr[physical_mask], flux_ratio[physical_mask], num_bins=num_bins, weights=[flux_error], min_bin=min_bin, max_bin=max_bin)
        bin_center_strong, flux_bin_medians_strong, flux_bin_lower_q_strong, flux_bin_upper_q_strong = pf.binned_median_quantile_log(sig_sfr[BIC_diff_strong], flux_ratio[BIC_diff_strong], num_bins=num_bins, weights=[flux_error], min_bin=min_bin, max_bin=max_bin)

    #calculate the r value for the median values
    r_flux_med_all, p_value_flux_all = pf.pearson_correlation(bin_center_all, flux_bin_medians_all)
    r_flux_med_physical, p_value_flux_physical = pf.pearson_correlation(bin_center_physical, flux_bin_medians_physical)
    r_flux_med_strong, p_value_flux_strong = pf.pearson_correlation(bin_center_strong, flux_bin_medians_strong)

    #calculate the r value for all the values
    r_flux_all, p_value_flux_all = pf.pearson_correlation(sig_sfr, flux_ratio)
    r_flux_physical, p_value_flux_physical = pf.pearson_correlation(sig_sfr[physical_mask], flux_ratio[physical_mask])
    r_flux_strong, p_value_flux_strong = pf.pearson_correlation(sig_sfr[BIC_diff_strong], flux_ratio[BIC_diff_strong])

    #print average numbers for the different panels
    print('All spaxels median flux_ratio:', np.nanmedian(flux_ratio))
    print('All spaxels standard deviation flux_ratio:', np.nanstd(flux_ratio))
    print('')

    print('Number of spaxels with broad sigmas at the instrument dispersion:', flux_ratio[vel_disp<=51].shape)
    print('')
    print('Number of spaxels beyond R_90:', flux_ratio[radius>6.1].shape)
    print('')

    print('Physical spaxels median flux_ratio:', np.nanmedian(flux_ratio[physical_mask]))
    print('Physical spaxels standard deviation flux_ratio:', np.nanstd(flux_ratio[physical_mask]))
    print('')

    print('Clean spaxels median flux_ratio:', np.nanmedian(flux_ratio[BIC_diff_strong]))
    print('Clean spaxels standard deviation flux_ratio:', np.nanstd(flux_ratio[BIC_diff_strong]))
    print('')

    #-------
    #plot it
    #-------
    plt.rcParams.update(pf.get_rc_params())
    fig, ax = plt.subplots(nrows=1, ncols=3, sharey=True, sharex=True, figsize=(10,4), constrained_layout=True)

    #get colours from cmasher
    colours = cmr.take_cmap_colors('cmr.gem', 3, cmap_range=(0.25, 0.85), return_fmt='hex')

    #plot all points
    ax[0].fill_between(bin_center_all, flux_bin_lower_q_all, flux_bin_upper_q_all, color=colours[0], alpha=0.3)
    ax[0].scatter(sig_sfr, flux_ratio, marker='o', s=10, label='All KOFFEE fits; R={:.2f}'.format(r_flux_all), color=colours[0], alpha=0.8)
    ax[0].plot(bin_center_all, flux_bin_medians_all, marker='', lw=3, label='Median all KOFFEE fits; R={:.2f}'.format(r_flux_med_all), color=colours[0])

    ax[0].errorbar(0.03, np.nanmin(flux_ratio), xerr=np.nanmedian(sig_sfr_err), yerr=np.nanmedian(flux_error), c='k')

    ax[0].set_ylim((np.nanmin(flux_ratio)+np.nanmedian(flux_error)-0.1), np.nanmax(flux_ratio)+0.5)
    ax[0].set_xscale('log')
    ax[0].set_xlim(0.002, 3)
    lgnd = ax[0].legend(frameon=True, fontsize='small', loc='upper left', framealpha=0.5, edgecolor=None)
    lgnd.legendHandles[0]._legmarker.set_markersize(3)
    ax[0].set_ylabel(flux_ratio_line+' Log(F$_{broad}$/F$_{narrow}$)')
    ax[0].set_xlabel('$\Sigma_{SFR}$ [M$_\odot$ yr$^{-1}$ kpc$^{-2}$]')
    ax[0].set_title('all spaxels')


    #plot points within 90% radius
    ax[1].fill_between(bin_center_physical, flux_bin_lower_q_physical, flux_bin_upper_q_physical, color=colours[1], alpha=0.3)
    ax[1].scatter(sig_sfr[radius>6.1], flux_ratio[radius>6.1], marker='o', s=10, label='All KOFFEE fits', edgecolors=colours[0], alpha=0.3, facecolors='none')
    ax[1].scatter(sig_sfr[vel_disp<=51], flux_ratio[vel_disp<=51], marker='v', s=10, edgecolors=colours[0], alpha=0.3, facecolors='none')
    ax[1].scatter(sig_sfr[physical_mask], flux_ratio[physical_mask], marker='o', s=10, label='Selected KOFFEE fits; R={:.2f}'.format(r_flux_physical), color=colours[1], alpha=0.8)
    ax[1].plot(bin_center_physical, flux_bin_medians_physical, marker='', lw=3, label='Median of selected KOFFEE fits; R={:.2f}'.format(r_flux_med_physical), color=colours[1])

    ax[1].errorbar(0.03, np.nanmin(flux_ratio), xerr=np.nanmedian(sig_sfr_err[physical_mask]), yerr=np.nanmedian(flux_error[physical_mask]), c='k')

    lgnd = ax[1].legend(frameon=True, fontsize='small', loc='upper left', framealpha=0.5, edgecolor=None)
    lgnd.legendHandles[0]._legmarker.set_markersize(3)
    ax[1].set_xlabel('$\Sigma_{SFR}$ [M$_\odot$ yr$^{-1}$ kpc$^{-2}$]')
    ax[1].set_title(r'$r$<$r_{90}$ and $\sigma_{broad}$>$\sigma_{inst}$')


    #plot points with strong BIC values
    ax[2].fill_between(bin_center_strong, flux_bin_lower_q_strong, flux_bin_upper_q_strong, color=colours[2], alpha=0.3)
    ax[2].scatter(sig_sfr[~BIC_diff_strong], flux_ratio[~BIC_diff_strong], marker='o', s=10, label='All KOFFEE fits', edgecolors=colours[0], alpha=0.3, facecolors='none')
    ax[2].scatter(sig_sfr[BIC_diff_strong], flux_ratio[BIC_diff_strong], marker='o', s=10, label='Selected KOFFEE fits; R={:.2f}'.format(r_flux_strong), color=colours[2], alpha=1.0)
    ax[2].plot(bin_center_strong, flux_bin_medians_strong, marker='', lw=3, label='Median of selected KOFFEE fits; R={:.2f}'.format(r_flux_med_strong), color=colours[2])

    ax[2].errorbar(0.03, np.nanmin(flux_ratio), xerr=np.nanmedian(sig_sfr_err[BIC_diff_strong]), yerr=np.nanmedian(flux_error[BIC_diff_strong]), c='k')

    lgnd = ax[2].legend(frameon=True, fontsize='small', loc='upper left', framealpha=0.5, edgecolor=None)
    lgnd.legendHandles[0]._legmarker.set_markersize(3)
    ax[2].set_xlabel('$\Sigma_{SFR}$ [M$_\odot$ yr$^{-1}$ kpc$^{-2}$]')
    ax[2].set_title('strongly likely BIC')

    plt.show()




def plot_radius_flux(flux_outflow_results, flux_outflow_error, OIII_outflow_results, OIII_outflow_error, BIC_outflow, BIC_no_outflow, statistical_results, z, radius, flux_ratio_line='OIII', weighted_average=True):
    """
    Plots the galaxy radius against the broad-to-narrow flux ratio.

    Parameters
    ----------
    flux_outflow_results : :obj:'~numpy.ndarray'
        array of outflow results from KOFFEE for the line we want to calculate
        the flux ratio for.  Should be (7, statistical_results.shape)

    flux_outflow_err : :obj:'~numpy.ndarray'
        array of the outflow result errors from KOFFEE for flux ratio line

    OIII_outflow_results : :obj:'~numpy.ndarray'
        array of outflow results from KOFFEE for OIII line.  Used to calculate
        the Sigma SFR.  Should be (7, statistical_results.shape)

    OIII_outflow_err : :obj:'~numpy.ndarray'
        array of the outflow result errors from KOFFEE for OIII line

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

    flux_ratio_line : string
        the emission line being used for the broad-to-narrow flux ratio, used for
        plotting labels (Default='OIII')

    weighted_average : boolean
        whether or not to take a weighted average using the errors (Default=True)

    Returns
    -------
    A graph of broad-to-narrow flux ratio against galaxy radius.

    """
    #calculate the flux for systematic and flow gaussians
    systemic_flux, systemic_flux_err, outflow_flux, outflow_flux_err = calc_sfr.calc_flux_from_koffee(flux_outflow_results, flux_outflow_error, statistical_results, z, outflow=True)

    #calculate the velocity dispersion for the masking
    vel_disp, vel_disp_err, vel_diff, vel_diff_err, vel_out, vel_out_err = calc_outvel.calc_outflow_vel(OIII_outflow_results, OIII_outflow_error, statistical_results, z)

    #create the flow mask
    if flux_ratio_line == 'Hbeta':
        flow_mask = (statistical_results>0) & (np.isnan(flux_outflow_results[3,:,:])==False)
        print('For H$\beta$ the number of outflow fitted spaxels is:', flow_mask[flow_mask].shape)
    else:
        flow_mask = (statistical_results>0)

    #flatten all the arrays and get rid of extra spaxels
    vel_disp = vel_disp[flow_mask]

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
    flux_ratio = np.log10(outflow_flux/systemic_flux)
    #flux_ratio = (outflow_flux/systemic_flux)

    #calculate the error
    flux_error = flux_ratio * np.log10(np.sqrt((outflow_flux_err/outflow_flux)**2 + (systemic_flux_err/systemic_flux)**2))

    #create BIC diff
    BIC_diff = BIC_outflow - BIC_no_outflow
    BIC_diff_weak = (BIC_diff < -10) & (BIC_diff >= -30)
    BIC_diff_moderate = (BIC_diff < -30) & (BIC_diff >= -50)
    BIC_diff_strong = (BIC_diff < -50)

    #physical limits mask -
    #for the radius mask 6.1" is the 90% radius
    #also mask out the fits which lie on the lower limit of dispersion < 51km/s
    physical_mask = (radius < 6.1) & (vel_disp>51)

    #strong BIC and physical limits mask
    #clean_mask = (radius < 6.1) & (vel_disp > 51) & (BIC_diff < -50)

    #do the calculations for all the bins
    num_bins = 4
    min_bin = None #-0.05
    max_bin = None #0.6

    if weighted_average == False:
        bin_center_all, flux_bin_medians_all, flux_bin_lower_q_all, flux_bin_upper_q_all = pf.binned_median_quantile_lin(radius, flux_ratio, num_bins=num_bins, weights=None, min_bin=min_bin, max_bin=max_bin)
        bin_center_physical, flux_bin_medians_physical, flux_bin_lower_q_physical, flux_bin_upper_q_physical = pf.binned_median_quantile_lin(radius[physical_mask], flux_ratio[physical_mask], num_bins=num_bins, weights=None, min_bin=min_bin, max_bin=max_bin)
        bin_center_strong, flux_bin_medians_strong, flux_bin_lower_q_strong, flux_bin_upper_q_strong = pf.binned_median_quantile_lin(radius[BIC_diff_strong], flux_ratio[BIC_diff_strong], num_bins=num_bins, weights=None, min_bin=min_bin, max_bin=max_bin)

    elif weighted_average == True:
        bin_center_all, flux_bin_medians_all, flux_bin_lower_q_all, flux_bin_upper_q_all = pf.binned_median_quantile_lin(radius, flux_ratio, num_bins=num_bins, weights=[flux_error], min_bin=min_bin, max_bin=max_bin)
        bin_center_physical, flux_bin_medians_physical, flux_bin_lower_q_physical, flux_bin_upper_q_physical = pf.binned_median_quantile_lin(radius[physical_mask], flux_ratio[physical_mask], num_bins=num_bins, weights=[flux_error], min_bin=min_bin, max_bin=max_bin)
        bin_center_strong, flux_bin_medians_strong, flux_bin_lower_q_strong, flux_bin_upper_q_strong = pf.binned_median_quantile_lin(radius[BIC_diff_strong], flux_ratio[BIC_diff_strong], num_bins=num_bins, weights=[flux_error], min_bin=min_bin, max_bin=max_bin)

    #calculate the r value for the median values
    r_flux_med_all, p_value_flux_all = pf.pearson_correlation(bin_center_all, flux_bin_medians_all)
    r_flux_med_physical, p_value_flux_physical = pf.pearson_correlation(bin_center_physical, flux_bin_medians_physical)
    r_flux_med_strong, p_value_flux_strong = pf.pearson_correlation(bin_center_strong, flux_bin_medians_strong)

    #calculate the r value for all the values
    r_flux_all, p_value_flux_all = pf.pearson_correlation(radius, flux_ratio)
    r_flux_physical, p_value_flux_physical = pf.pearson_correlation(radius[physical_mask], flux_ratio[physical_mask])
    r_flux_strong, p_value_flux_strong = pf.pearson_correlation(radius[BIC_diff_strong], flux_ratio[BIC_diff_strong])

    #print average numbers for the different panels
    print('All spaxels median flux_ratio:', np.nanmedian(flux_ratio))
    print('All spaxels standard deviation flux_ratio:', np.nanstd(flux_ratio))
    print('')

    print('Number of spaxels with broad sigmas at the instrument dispersion:', flux_ratio[vel_disp<=51].shape)
    print('')
    print('Number of spaxels beyond R_90:', flux_ratio[radius>6.1].shape)
    print('')

    print('Physical spaxels median flux_ratio:', np.nanmedian(flux_ratio[physical_mask]))
    print('Physical spaxels standard deviation flux_ratio:', np.nanstd(flux_ratio[physical_mask]))
    print('')

    print('Strong spaxels median flux_ratio:', np.nanmedian(flux_ratio[BIC_diff_strong]))
    print('Strong spaxels standard deviation flux_ratio:', np.nanstd(flux_ratio[BIC_diff_strong]))
    print('')

    #-------
    #plot it
    #-------
    plt.rcParams.update(pf.get_rc_params())
    fig, ax = plt.subplots(nrows=1, ncols=3, sharey=True, sharex=True, figsize=(10,4), constrained_layout=True)

    #get colours from cmasher
    colours = cmr.take_cmap_colors('cmr.gem', 3, cmap_range=(0.25, 0.85), return_fmt='hex')

    #plot all points
    ax[0].fill_between(bin_center_all, flux_bin_lower_q_all, flux_bin_upper_q_all, color=colours[0], alpha=0.3)
    ax[0].scatter(radius, flux_ratio, marker='o', s=10, label='All KOFFEE fits; R={:.2f}'.format(r_flux_all), color=colours[0], alpha=0.8)
    ax[0].plot(bin_center_all, flux_bin_medians_all, marker='', lw=3, label='Median all KOFFEE fits; R={:.2f}'.format(r_flux_med_all), color=colours[0])

    ax[0].errorbar(7, np.nanmin(flux_ratio)+0.25, xerr=0.7, yerr=np.nanmedian(flux_error), c='k')

    ax[0].set_ylim(np.nanmin(flux_ratio)-0.5, np.nanmax(flux_ratio)+0.1)
    #ax[0].set_xscale('log')
    #ax[0].set_xlim(0.003, 10)
    lgnd = ax[0].legend(frameon=True, fontsize='small', loc='lower left', framealpha=0.5, edgecolor=None)
    lgnd.legendHandles[0]._legmarker.set_markersize(3)
    ax[0].set_ylabel(flux_ratio_line+' Log(F$_{broad}$/F$_{narrow}$)')
    ax[0].set_xlabel('Radius [Arcsec]')
    ax[0].set_title('all spaxels')

    #plot points within 90% radius
    ax[1].fill_between(bin_center_physical, flux_bin_lower_q_physical, flux_bin_upper_q_physical, color=colours[1], alpha=0.3)
    ax[1].scatter(radius[radius>6.1], flux_ratio[radius>6.1], marker='o', s=10, label='All KOFFEE fits', edgecolors=colours[0], alpha=0.3, facecolors='none')
    ax[1].scatter(radius[vel_disp<=51], flux_ratio[vel_disp<=51], marker='v', s=10, edgecolors=colours[0], alpha=0.3, facecolors='none')
    ax[1].scatter(radius[physical_mask], flux_ratio[physical_mask], marker='o', s=10, label='Selected KOFFEE fits; R={:.2f}'.format(r_flux_physical), color=colours[1], alpha=0.8)
    ax[1].plot(bin_center_physical, flux_bin_medians_physical, marker='', lw=3, label='Median of selected KOFFEE fits; R={:.2f}'.format(r_flux_med_physical), color=colours[1])

    ax[1].errorbar(7, np.nanmin(flux_ratio)+0.25, xerr=0.7, yerr=np.nanmedian(flux_error[physical_mask]), c='k')

    lgnd = ax[1].legend(frameon=True, fontsize='small', loc='lower left', framealpha=0.5, edgecolor=None)
    lgnd.legendHandles[0]._legmarker.set_markersize(3)
    ax[1].set_xlabel('Radius [Arcsec]')
    ax[1].set_title(r'$r$<$r_{90}$ and $\sigma_{broad}$>$\sigma_{inst}$')

    #plot points with strong BIC values
    ax[2].fill_between(bin_center_strong, flux_bin_lower_q_strong, flux_bin_upper_q_strong, color=colours[2], alpha=0.3)
    ax[2].scatter(radius[~BIC_diff_strong], flux_ratio[~BIC_diff_strong], marker='o', s=10, label='All KOFFEE fits', edgecolors=colours[0], alpha=0.3, facecolors='none')
    ax[2].scatter(radius[BIC_diff_strong], flux_ratio[BIC_diff_strong], marker='o', s=10, label='Selected KOFFEE fits; R={:.2f}'.format(r_flux_strong), color=colours[2], alpha=1.0)
    ax[2].plot(bin_center_strong, flux_bin_medians_strong, marker='', lw=3, label='Median of selected KOFFEE fits; R={:.2f}'.format(r_flux_med_strong), color=colours[2])

    ax[2].errorbar(7, np.nanmin(flux_ratio)+0.25, xerr=0.7, yerr=np.nanmedian(flux_error[BIC_diff_strong]), c='k')

    lgnd = ax[2].legend(frameon=True, fontsize='small', loc='lower left', framealpha=0.5, edgecolor=None)
    lgnd.legendHandles[0]._legmarker.set_markersize(3)
    ax[2].set_xlabel('Radius [Arcsec]')
    ax[2].set_title('strongly likely BIC')

    plt.show()



def maps_of_halpha_hbeta(halpha_fits_file, hbeta_fits_file, xx_flat, yy_flat):#, header, z, OIII_outflow_results, OIII_outflow_error, statistical_results):
    """
    Maps the Halpha flux from the fits file

    Parameters
    ----------
    halpha_fits_file : string
        location of the halpha fits file

    hbeta_fits_file : string
        location of the hbeta fits file

    xx_flat : :obj:'~numpy.ndarray'
        flat (1D) array of the x-coordinates for the fits files

    yy_flat : :obj:'~numpy.ndarray'
        flat (1D) array of the y-coordinates for the fits files

    Returns
    -------
    A two-panel graph of maps of the Halpha and Hbeta flux
    """
    #read in fits files
    halpha_data, halpha_header, halpha_wcs = pf.read_in_create_wcs(halpha_fits_file, index=0)
    hbeta_data, hbeta_header, hbeta_wcs = pf.read_in_create_wcs(hbeta_fits_file, index=0)

    #find the peak of the hbeta flux and convert to wcs
    hbeta_peak_pixel = np.argwhere(hbeta_data==np.nanmax(hbeta_data))
    hbeta_peak_world = hbeta_wcs.all_pix2world(hbeta_peak_pixel[0,1], hbeta_peak_pixel[0,0], 0)
    hbeta_peak_halpha_pixel = halpha_wcs.all_world2pix(hbeta_peak_world[0], hbeta_peak_world[1], 0)

    #calculate the beginning and end of 10 arcsec
    halpha_10arcsec_pixel_length = abs(10/(halpha_header['CD1_1']*60*60))
    hbeta_10arcsec_pixel_length = abs(10/(hbeta_header['CD1_2']*60*60))

    #create the figure
    plt.rcParams.update(pf.get_rc_params())

    plt.figure()#constrained_layout=True)

    ax1 = plt.subplot(121, projection=halpha_wcs)
    ax1.set_facecolor('black')
    #set the x and y limits of the plot
    halpha_xlim = 210
    halpha_ylim = 200
    #beginning and ending of scale bar
    scale_ypos = halpha_ylim+20
    scale_begin = halpha_xlim+20
    scale_end = halpha_xlim+20+halpha_10arcsec_pixel_length
    #convert to world coords
    scale_begin_world = halpha_wcs.all_pix2world(scale_begin, scale_ypos, 0)
    scale_end_world = halpha_wcs.all_pix2world(scale_end, scale_ypos, 0)
    print(scale_begin_world)
    print(scale_end_world)
    #convert to hbeta pixel coords
    scale_begin_hbeta = hbeta_wcs.wcs_world2pix(scale_begin_world[0], scale_begin_world[1], 0)
    scale_end_hbeta = hbeta_wcs.wcs_world2pix(scale_end_world[0], scale_end_world[1], 0)
    print(scale_begin_hbeta)
    print(scale_end_hbeta)

    #do the plotting
    ax1.imshow(np.log10(halpha_data), origin='lower', cmap=cmr.ember, vmin=-1.75, vmax=-0.25)
    #ax1.imshow(np.log10(halpha_data), origin='lower', cmap=cmr.ember, vmin=-20, vmax=-17.5)
    ax1.hlines(halpha_ylim+20, halpha_xlim+20, halpha_xlim+20+halpha_10arcsec_pixel_length, colors='white')
    ax1.scatter(hbeta_peak_halpha_pixel[0], hbeta_peak_halpha_pixel[1], c='white', marker='x', s=20)
    lon1 = ax1.coords[0]
    lat1 = ax1.coords[1]
    lon1.set_ticks_visible(False)
    lon1.set_ticklabel_visible(False)
    lat1.set_ticks_visible(False)
    lat1.set_ticklabel_visible(False)
    ax1.set_title(r'H$\alpha$ Flux')
    #ax1.set_xlim(halpha_xlim, halpha_data.shape[1]-halpha_xlim)
    #ax1.set_ylim(halpha_ylim, halpha_data.shape[0]-halpha_ylim)

    ax2 = plt.subplot(122, projection=hbeta_wcs, slices=('y', 'x'))
    ax2.set_facecolor('black')
    #set the x and y limits of the plot

    #do the plotting
    hbeta_spax = bdpk.display_pixels(xx_flat.reshape(67,24).transpose(), yy_flat.reshape(67,24).transpose(), np.log10(hbeta_data).transpose(), angle=360, axes=ax2, cmap=cmr.ember, vmin=-0.5, vmax=2.0)
    ax2.hlines(1, 1, 10, colors='white')
    ax2.hlines(scale_begin_world[1], scale_begin_world[0], scale_end_world[0], colors='white', transform=ax2.get_transform('world'))
    #ax2.text(fuv_xlim+105, fuv_ylim+105, '10" ', c='white')
    ax2.grid(False)
    ax2.invert_xaxis()
    lon2 = ax2.coords[0]
    lat2 = ax2.coords[1]
    #lon2.set_ticks_visible(False)
    #lon2.set_ticklabel_visible(False)
    #lat2.set_ticks_visible(False)
    #lat2.set_ticklabel_visible(False)
    ax2.set_title(r'H$\beta$ flux')
    #ax2.set_xlim(hbeta_lim2_pixel[0], hbeta_lim1_pixel[0])
    #ax2.set_ylim(hbeta_lim2_pixel[1], hbeta_lim1_pixel[1])

    plt.show()



def plot_sfr_vout_multiple(OIII_outflow_results, OIII_outflow_error, hbeta_outflow_results, hbeta_outflow_error, hbeta_no_outflow_results, hbeta_no_outflow_error, statistical_results, z, header, data_descriptor, plot_data_fits=False):
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
    #create the figure
    plt.rcParams.update(pf.get_rc_params())
    fig, ax = plt.subplots(nrows=1, ncols=2, sharey=True, sharex=True, figsize=(8,4), constrained_layout=True)

    #get colours from cmasher, using as many colours as there are data sets
    colours = cmr.take_cmap_colors('cmr.gem', len(OIII_outflow_results), cmap_range=(0.25, 0.85), return_fmt='hex')

    for i in np.arange(len(OIII_outflow_results)):

        #calculate the outflow velocity
        vel_disp, vel_disp_err, vel_diff, vel_diff_err, vel_out, vel_out_err = calc_outvel.calc_outflow_vel(OIII_outflow_results[i], OIII_outflow_error[i], statistical_results[i], z)

        #calculate the sfr surface density - using just the systemic line, and including the flux line
        #don't include extinction since this was included in the continuum subtraction using ppxf
        sfr, sfr_err, total_sfr, sfr_surface_density, sfr_surface_density_err = calc_sfr.calc_sfr_koffee(hbeta_outflow_results[i], hbeta_outflow_error[i], hbeta_no_outflow_results[i], hbeta_no_outflow_error[i], statistical_results[i], z, header[i], include_extinction=False, include_outflow=False)

        #get the sfr for the outflow spaxels
        flow_mask = (statistical_results[i]>0) #& (sfr_surface_density>0.1)

        #flatten all the arrays and get rid of extra spaxels
        sig_sfr = sfr_surface_density[flow_mask]
        sig_sfr_err = sfr_surface_density_err[flow_mask]
        vel_out = vel_out[flow_mask]
        vel_out_err = vel_out_err[flow_mask]
        vel_disp = vel_disp[flow_mask]

        #make sure none of the errors are nan values
        vel_out_err[np.where(np.isnan(vel_out_err)==True)] = np.nanmedian(vel_out_err)

        #do the calculations for all the bins
        num_bins = 5
        min_bin = None #-0.05
        max_bin = None #0.6


        bin_center_all, v_out_bin_medians_all, v_out_bin_lower_q_all, v_out_bin_upper_q_all = pf.binned_median_quantile_log(sig_sfr, vel_out, num_bins=num_bins, weights=None, min_bin=min_bin, max_bin=max_bin)


        #calculate the r value for the median values
        r_vel_out_med_all, p_value_v_out_all = pf.pearson_correlation(bin_center_all, v_out_bin_medians_all)

        #calculate the r value for all the values
        r_vel_out_all, p_value_v_out_all = pf.pearson_correlation(sig_sfr, vel_out)


        #fit our own trends
        popt_vout_all_medians, pcov_vout_all_medians = curve_fit(pf.fitting_function, bin_center_all, v_out_bin_medians_all)

        popt_vout_all, pcov_vout_all = curve_fit(pf.fitting_function, sig_sfr, vel_out)

        print(popt_vout_all, pcov_vout_all)
        print([popt_vout_all_medians[0], np.sqrt(np.diag(pcov_vout_all_medians))[0], popt_vout_all_medians[1], np.sqrt(np.diag(pcov_vout_all_medians))[1]])

        sfr_linspace = np.linspace(sig_sfr.min(), sig_sfr.max()+4, num=1000)



        #print average numbers for the different panels
        print('Number of spaxels in data set', data_descriptor, vel_out.shape)
        print('All spaxels median v_out:', np.nanmedian(vel_out))
        print('All spaxels standard deviation v_out:', np.nanstd(vel_out))
        print('All spaxels median sigma_sfr:', np.nanmedian(sig_sfr))
        print('All spaxels standard deviation sigma_sfr:', np.nanstd(sig_sfr))
        print('')

        print('All spaxels best fit coefficients:', popt_vout_all)
        print('All spaxels best fit errors', np.sqrt(np.diag(pcov_vout_all)))
        print('')

        #-------
        #plot it
        #-------

        #plot all points
        ax[0].scatter(sig_sfr[vel_disp>51], vel_out[vel_disp>51], marker='o', s=10, label=data_descriptor[i]+'; R={:.2f}'.format(r_vel_out_all), color=colours[i], alpha=0.8)
        ax[0].scatter(sig_sfr[vel_disp<=51], vel_out[vel_disp<=51], marker='v', s=10, c=colours[i])

        #plot medians
        ax[1].fill_between(bin_center_all, v_out_bin_lower_q_all, v_out_bin_upper_q_all, color=colours[i], alpha=0.3)
        ax[1].plot(bin_center_all, v_out_bin_medians_all, marker='', lw=3, label=data_descriptor[i]+' Median; R={:.2f}'.format(r_vel_out_med_all), color=colours[i])


        if plot_data_fits == True:
            ax[0].plot(sfr_linspace, pf.fitting_function(sfr_linspace, *popt_vout_all), ls='--', color=colours[i], label=data_descriptor[i]+' Fit: $v_{out}=%5.0f\pm$%2.0f $\Sigma_{SFR}^{%5.2f \pm %5.2f}$' % (popt_vout_all[0], np.sqrt(np.diag(pcov_vout_all))[0], popt_vout_all[1], np.sqrt(np.diag(pcov_vout_all))[1]))
            ax[1].plot(sfr_linspace, pf.fitting_function(sfr_linspace, *popt_vout_all_medians), ls='--', color=colours[i], label=data_descriptor[i]+' Fit: $v_{out}=%5.0f\pm$%2.0f $\Sigma_{SFR}^{%5.2f \pm %5.2f}$' %(popt_vout_all_medians[0], np.sqrt(np.diag(pcov_vout_all_medians))[0], popt_vout_all_medians[1], np.sqrt(np.diag(pcov_vout_all_medians))[1]))

        if i == 0:
            ax[0].set_xlim(np.nanmin(sig_sfr)-0.001, np.nanmax(sig_sfr)+1.0)



    #create vectors to plot the literature trends
    sfr_surface_density_chen, v_out_chen = pf.chen_et_al_2010(sig_sfr.min(), sig_sfr.max(), scale_factor=np.nanmedian(vel_out)/(np.nanmedian(sig_sfr)**0.1))
    sfr_surface_density_murray, v_out_murray = pf.murray_et_al_2011(sig_sfr.min(), sig_sfr.max(), scale_factor=np.nanmedian(vel_out)/(np.nanmedian(sig_sfr)**2))

    ax[0].plot(sfr_surface_density_chen, v_out_chen, ':k', label='Energy driven, $v_{out} \propto \Sigma_{SFR}^{0.1}$')
    ax[0].plot(sfr_surface_density_murray, v_out_murray, '--k', label='Momentum driven, $v_{out} \propto \Sigma_{SFR}^{2}$')

    ax[1].plot(sfr_surface_density_chen, v_out_chen, ':k', label='Energy driven, $v_{out} \propto \Sigma_{SFR}^{0.1}$')
    ax[1].plot(sfr_surface_density_murray, v_out_murray, '--k', label='Momentum driven, $v_{out} \propto \Sigma_{SFR}^{2}$')

    ax[0].errorbar(0.05, 150, xerr=np.nanmedian(sig_sfr_err), yerr=np.nanmedian(vel_out_err), c='k')

    ax[0].set_ylim(100, 700)
    ax[0].set_xscale('log')

    lgnd = ax[0].legend(frameon=True, fontsize='small', loc='upper left', framealpha=0.5)
    lgnd.legendHandles[0]._legmarker.set_markersize(3)
    ax[0].set_ylabel('Maximum Outflow Velocity [km s$^{-1}$]')
    ax[0].set_xlabel('$\Sigma_{SFR}$ [M$_\odot$ yr$^{-1}$ kpc$^{-2}$]')
    ax[0].set_title('S/N > 20 and $\delta_{BIC}$<-10')

    lgnd = ax[1].legend(frameon=True, fontsize='small', loc='upper left', framealpha=0.5)
    lgnd.legendHandles[0]._legmarker.set_markersize(3)
    ax[1].set_xlabel('$\Sigma_{SFR}$ [M$_\odot$ yr$^{-1}$ kpc$^{-2}$]')
    ax[1].set_title('Median values of all points')

    plt.show()


def plot_sfr_vout_correlation_with_binning(OIII_outflow_results, OIII_outflow_error, hbeta_outflow_results, hbeta_outflow_error, hbeta_no_outflow_results, hbeta_no_outflow_error, BIC_outflow, BIC_no_outflow, statistical_results, z, radius, header, data_descriptor, plot_data_fits=False):
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
    #create figure
    plt.rcParams.update(pf.get_rc_params())
    #fig, ax = plt.subplots(nrows=1, ncols=2, sharey=True, sharex=True, figsize=(8,4), constrained_layout=True)
    fig, ax = plt.subplots(nrows=1, ncols=1, sharey=True, sharex=True, figsize=(5,4), constrained_layout=True)

    #get colours from cmasher
    colours = cmr.take_cmap_colors('cmr.gem', 3, cmap_range=(0.25, 0.85), return_fmt='hex')

    #create arrays to save results to
    spaxel_area = np.full(len(OIII_outflow_results), np.nan, dtype=np.double)

    corr_coeff_all = np.full(len(OIII_outflow_results), np.nan, dtype=np.double)
    corr_coeff_physical = np.full(len(OIII_outflow_results), np.nan, dtype=np.double)
    corr_coeff_strong = np.full(len(OIII_outflow_results), np.nan, dtype=np.double)

    corr_coeff_all_median = np.full(len(OIII_outflow_results), np.nan, dtype=np.double)
    corr_coeff_physical_median = np.full(len(OIII_outflow_results), np.nan, dtype=np.double)
    corr_coeff_strong_median = np.full(len(OIII_outflow_results), np.nan, dtype=np.double)

    #calculate the proper distance
    proper_dist = cosmo.kpc_proper_per_arcmin(z).to(u.kpc/u.arcsec)


    #iterate through all of the data sets
    for i in np.arange(len(OIII_outflow_results)):
        #calculate the outflow velocity
        vel_disp, vel_disp_err, vel_diff, vel_diff_err, vel_out, vel_out_err = calc_outvel.calc_outflow_vel(OIII_outflow_results[i], OIII_outflow_error[i], statistical_results[i], z)

        #calculate the sfr surface density - using just the systemic line, and including the flux line
        #don't include extinction since this was included in the continuum subtraction using ppxf
        sfr, sfr_err, total_sfr, sfr_surface_density, sfr_surface_density_err = calc_sfr.calc_sfr_koffee(hbeta_outflow_results[i], hbeta_outflow_error[i], hbeta_no_outflow_results[i], hbeta_no_outflow_error[i], statistical_results[i], z, header[i], include_extinction=False, include_outflow=False)

        #get the sfr for the outflow spaxels
        flow_mask = (statistical_results[i]>0) #& (sfr_surface_density>0.1)

        #flatten all the arrays and get rid of extra spaxels
        sig_sfr = sfr_surface_density[flow_mask]
        sig_sfr_err = sfr_surface_density_err[flow_mask]
        vel_out = vel_out[flow_mask]
        vel_out_err = vel_out_err[flow_mask]
        BIC_outflow_masked = BIC_outflow[i][flow_mask]
        BIC_no_outflow_masked = BIC_no_outflow[i][flow_mask]
        vel_disp = vel_disp[flow_mask]
        radius_masked = radius[i][flow_mask]

        #create BIC diff
        BIC_diff = BIC_outflow_masked - BIC_no_outflow_masked
        #BIC_diff_weak = (BIC_diff < -10) & (BIC_diff >= -30)
        #BIC_diff_moderate = (BIC_diff < -30) & (BIC_diff >= -50)
        BIC_diff_strong = (BIC_diff < -50)

        #physical limits mask -
        #for the radius mask 6.1" is the 90% radius
        #also mask out the fits which lie on the lower limit of dispersion < 51km/s
        physical_mask = (radius_masked < 6.1) & (vel_disp>51)

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


        bin_center_all, v_out_bin_medians_all, v_out_bin_lower_q_all, v_out_bin_upper_q_all = pf.binned_median_quantile_log(sig_sfr, vel_out, num_bins=num_bins, weights=None, min_bin=min_bin, max_bin=max_bin)
        bin_center_physical, v_out_bin_medians_physical, v_out_bin_lower_q_physical, v_out_bin_upper_q_physical = pf.binned_median_quantile_log(sig_sfr[physical_mask], vel_out[physical_mask], num_bins=num_bins, weights=None, min_bin=min_bin, max_bin=max_bin)
        bin_center_clean, v_out_bin_medians_clean, v_out_bin_lower_q_clean, v_out_bin_upper_q_clean = pf.binned_median_quantile_log(sig_sfr[BIC_diff_strong], vel_out[BIC_diff_strong], num_bins=num_bins, weights=None, min_bin=min_bin, max_bin=max_bin)

        #calculate the r value for the median values
        r_vel_out_med_all, p_value_v_out_all = pf.pearson_correlation(bin_center_all, v_out_bin_medians_all)
        r_vel_out_med_physical, p_value_v_out_physical = pf.pearson_correlation(bin_center_physical, v_out_bin_medians_physical)
        r_vel_out_med_clean, p_value_v_out_clean = pf.pearson_correlation(bin_center_clean, v_out_bin_medians_clean)

        #calculate the r value for all the values
        r_vel_out_all, p_value_v_out_all = pf.pearson_correlation(sig_sfr, vel_out)
        r_vel_out_physical, p_value_v_out_physical = pf.pearson_correlation(sig_sfr[physical_mask], vel_out[physical_mask])
        r_vel_out_clean, p_value_v_out_clean = pf.pearson_correlation(sig_sfr[BIC_diff_strong], vel_out[BIC_diff_strong])

        #save results to arrays
        spaxel_area[i] = ((header[i]['CD1_2']*60*60*header[i]['CD2_1']*60*60)*(proper_dist**2)).value

        corr_coeff_all[i] = r_vel_out_all
        corr_coeff_physical[i] = r_vel_out_physical
        corr_coeff_strong[i] = r_vel_out_clean

        corr_coeff_all_median[i] = r_vel_out_med_all
        corr_coeff_physical_median[i] = r_vel_out_med_physical
        corr_coeff_strong_median[i] = r_vel_out_med_clean

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
    ax.plot(circularised_radius, corr_coeff_all, marker='o', label='S/N>20 and $\delta_{BIC}$<-10', color=colours[0])
    ax.plot(circularised_radius, corr_coeff_physical, marker='o', label=r'$\delta_{BIC}$<-10, $r$<$r_{90}$ and $\sigma_{broad}$>$\sigma_{inst}$', color=colours[1])
    ax.plot(circularised_radius, corr_coeff_strong, marker='o', label='strongly likely BIC $\delta_{BIC}$<-50', color=colours[2])

    #ax[1].scatter(spaxel_area, corr_coeff_all_median, marker='s', s=20, color=colours[0])
    #ax[1].scatter(spaxel_area, corr_coeff_physical_median, marker='s', s=20, color=colours[1])
    #ax[1].scatter(spaxel_area, corr_coeff_strong_median, marker='s', s=20, color=colours[2])


    lgnd = ax.legend(frameon=True, fontsize='small', loc='upper left', framealpha=0.5)
    ax.set_ylabel('Pearson Correlation Coefficient')
    ax.set_xlabel('Circularised Bin Radius [kpc]')
    #ax[0].set_xlabel('Spaxel Area [kpc$^{2}$]')
    #ax[1].set_xlabel('Spaxel Area [kpc$^{2}$]')

    #ax[0].set_title('All points')
    #ax[1].set_title('Median values')

    plt.show()


def plot_sfr_vout_correlation_with_binning_from_file(outflow_velocity_fits_files, outflow_dispersion_fits_files, sig_sfr_fits_files, chi_square_text_files, statistical_results_text_files, z, data_fits_files, data_descriptor):
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
        r_vel_out_all, p_value_v_out_all = pf.pearson_correlation(sig_sfr, vel_out)
        r_vel_out_physical, p_value_v_out_physical = pf.pearson_correlation(sig_sfr[physical_mask], vel_out[physical_mask])
        r_vel_out_clean, p_value_v_out_clean = pf.pearson_correlation(sig_sfr[BIC_diff_strong], vel_out[BIC_diff_strong])

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
    ax.plot(circularised_radius*2, corr_coeff_all, marker='o', label='S/N>20 and $\delta_{BIC}$<-10', color=colours[0])
    ax.plot(circularised_radius*2, corr_coeff_physical, marker='o', label=r'$\delta_{BIC}$<-10, $r$<$r_{90}$ and $\sigma_{broad}$>$\sigma_{inst}$', color=colours[1])
    ax.plot(circularised_radius*2, corr_coeff_strong, marker='o', label='strongly likely BIC $\delta_{BIC}$<-50', color=colours[2])


    lgnd = ax.legend(frameon=True, fontsize='small', loc='upper left', framealpha=0.5)
    ax.set_ylabel('Pearson Correlation Coefficient')
    ax.set_xlabel('Circularised Bin Diameter [kpc]')

    plt.show()



def plot_sfr_vout_data_model_comparison(OIII_outflow_results, OIII_outflow_error, hbeta_outflow_results, hbeta_outflow_error, hbeta_no_outflow_results, hbeta_no_outflow_error, BIC_outflow, BIC_no_outflow, statistical_results, z, radius, header, weighted_average=False, plot_medians=True, compare='subtract'):
    """
    Plots the model-data for the SFR surface density against the outflow velocity,
    with Sigma_SFR calculated using only the narrow component.

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

    #strong BIC and physical limits mask
    #clean_mask = (radius < 6.1) & (vel_disp > 51) & (BIC_diff < -50)

    #make sure none of the errors are nan values
    vel_out_err[np.where(np.isnan(vel_out_err)==True)] = np.nanmedian(vel_out_err)

    #do the calculations for all the bins
    num_bins = 5
    min_bin = None #-0.05
    max_bin = None #0.6

    logspace_all, bin_center_all, v_out_bin_medians_all, v_out_bin_lower_q_all, v_out_bin_upper_q_all, v_out_bin_stdev_all = pf.binned_median_quantile_log(sig_sfr, vel_out, num_bins=num_bins, weights=None, min_bin=min_bin, max_bin=max_bin)
    logspace_physical, bin_center_physical, v_out_bin_medians_physical, v_out_bin_lower_q_physical, v_out_bin_upper_q_physical, v_out_bin_stdev_physical = pf.binned_median_quantile_log(sig_sfr[physical_mask], vel_out[physical_mask], num_bins=num_bins, weights=None, min_bin=min_bin, max_bin=max_bin)
    logspace_strong, bin_center_strong, v_out_bin_medians_strong, v_out_bin_lower_q_strong, v_out_bin_upper_q_strong, v_out_bin_stdev_strong = pf.binned_median_quantile_log(sig_sfr[BIC_diff<-50], vel_out[BIC_diff<-50], num_bins=num_bins, weights=None, min_bin=min_bin, max_bin=max_bin)



    #minus the data from the model
    chen_model = np.full_like(vel_out, np.nan, dtype=np.double)
    murray_model = np.full_like(vel_out, np.nan, dtype=np.double)

    for i in np.arange(vel_out.shape[0]):
        #calculate the expected velocity at each sigma_sfr
        sigma_sfr_chen, vel_out_expected_chen = pf.chen_et_al_2010(sig_sfr[i], sig_sfr[i], scale_factor=np.nanmedian(vel_out[BIC_diff_strong])/(np.nanmedian(sig_sfr[BIC_diff_strong])**0.1))
        sigma_sfr_murray, vel_out_expected_murray = pf.chen_et_al_2010(sig_sfr[i], sig_sfr[i], scale_factor=np.nanmedian(vel_out[BIC_diff_strong])/(np.nanmedian(sig_sfr[BIC_diff_strong])**2))

        vel_out_expected_chen = vel_out_expected_chen[0]
        vel_out_expected_murray = vel_out_expected_murray[0]

        #expected velocity - measured velocity
        if compare == 'subtract':
            chen_model[i] = vel_out_expected_chen-vel_out[i]
            murray_model[i] = vel_out_expected_murray-vel_out[i]
        elif compare == 'divide':
            chen_model[i] = np.log10(vel_out[i]/vel_out_expected_chen)
            murray_model[i] = np.log10(vel_out[i]/vel_out_expected_murray)


    chen_model_medians_all = np.full_like(v_out_bin_medians_all, np.nan, dtype=np.double)
    murray_model_medians_all = np.full_like(v_out_bin_medians_all, np.nan, dtype=np.double)

    chen_model_medians_physical = np.full_like(v_out_bin_medians_all, np.nan, dtype=np.double)
    murray_model_medians_physical = np.full_like(v_out_bin_medians_all, np.nan, dtype=np.double)

    chen_model_medians_strong = np.full_like(v_out_bin_medians_all, np.nan, dtype=np.double)
    murray_model_medians_strong = np.full_like(v_out_bin_medians_all, np.nan, dtype=np.double)


    for i in np.arange(v_out_bin_medians_all.shape[0]):
        #calculate the expected velocity at each sigma_sfr
        sigma_sfr_chen, vel_out_expected_chen_all = pf.chen_et_al_2010(bin_center_all[i], bin_center_all[i], scale_factor=np.nanmedian(vel_out[BIC_diff_strong])/(np.nanmedian(sig_sfr[BIC_diff_strong])**0.1))
        sigma_sfr_murray, vel_out_expected_murray_all = pf.chen_et_al_2010(bin_center_all[i], bin_center_all[i], scale_factor=np.nanmedian(vel_out[BIC_diff_strong])/(np.nanmedian(sig_sfr[BIC_diff_strong])**2))

        vel_out_expected_chen_all = vel_out_expected_chen_all[0]
        vel_out_expected_murray_all = vel_out_expected_murray_all[0]

        #calculate the expected velocity at each sigma_sfr
        sigma_sfr_chen, vel_out_expected_chen_physical = pf.chen_et_al_2010(bin_center_physical[i], bin_center_physical[i], scale_factor=np.nanmedian(vel_out[BIC_diff_strong])/(np.nanmedian(sig_sfr[BIC_diff_strong])**0.1))
        sigma_sfr_murray, vel_out_expected_murray_physical = pf.chen_et_al_2010(bin_center_physical[i], bin_center_physical[i], scale_factor=np.nanmedian(vel_out[BIC_diff_strong])/(np.nanmedian(sig_sfr[BIC_diff_strong])**2))

        vel_out_expected_chen_physical = vel_out_expected_chen_physical[0]
        vel_out_expected_murray_physical = vel_out_expected_murray_physical[0]

        #calculate the expected velocity at each sigma_sfr
        sigma_sfr_chen, vel_out_expected_chen_strong = pf.chen_et_al_2010(bin_center_strong[i], bin_center_strong[i], scale_factor=np.nanmedian(vel_out[BIC_diff_strong])/(np.nanmedian(sig_sfr[BIC_diff_strong])**0.1))
        sigma_sfr_murray, vel_out_expected_murray_strong = pf.chen_et_al_2010(bin_center_strong[i], bin_center_strong[i], scale_factor=np.nanmedian(vel_out[BIC_diff_strong])/(np.nanmedian(sig_sfr[BIC_diff_strong])**2))

        vel_out_expected_chen_strong = vel_out_expected_chen_strong[0]
        vel_out_expected_murray_strong = vel_out_expected_murray_strong[0]

        #expected velocity - measured velocity
        if compare == 'subtract':
            chen_model_medians_all[i] = vel_out_expected_chen_all-v_out_bin_medians_all[i]
            murray_model_medians_all[i] = vel_out_expected_murray_all-v_out_bin_medians_all[i]

            chen_model_medians_physical[i] = vel_out_expected_chen_physical-v_out_bin_medians_physical[i]
            murray_model_medians_physical[i] = vel_out_expected_murray_physical-v_out_bin_medians_physical[i]

            chen_model_medians_strong[i] = vel_out_expected_chen_strong-v_out_bin_medians_strong[i]
            murray_model_medians_strong[i] = vel_out_expected_murray_strong-v_out_bin_medians_strong[i]
        elif compare =='divide':
            chen_model_medians_all[i] = np.log10(v_out_bin_medians_all[i]/vel_out_expected_chen_all)
            murray_model_medians_all[i] = np.log10(v_out_bin_medians_all[i]/vel_out_expected_murray_all)

            chen_model_medians_physical[i] = np.log10(v_out_bin_medians_physical[i]/vel_out_expected_chen_physical)
            murray_model_medians_physical[i] = np.log10(v_out_bin_medians_physical[i]/vel_out_expected_murray_physical)

            chen_model_medians_strong[i] = np.log10(v_out_bin_medians_strong[i]/vel_out_expected_chen_strong)
            murray_model_medians_strong[i] = np.log10(v_out_bin_medians_strong[i]/vel_out_expected_murray_strong)


    #calculate the r value for the median values
    r_vel_out_med_chen_all, p_value_v_out_chen_all = pf.pearson_correlation(bin_center_all, chen_model_medians_all)
    r_vel_out_med_chen_physical, p_value_v_out_chen_physical = pf.pearson_correlation(bin_center_physical, chen_model_medians_physical)
    r_vel_out_med_chen_strong, p_value_v_out_chen_strong = pf.pearson_correlation(bin_center_strong, chen_model_medians_strong)

    r_vel_out_med_murray_all, p_value_v_out_murray_all = pf.pearson_correlation(bin_center_all, murray_model_medians_all)
    r_vel_out_med_murray_physical, p_value_v_out_murray_physical = pf.pearson_correlation(bin_center_physical, murray_model_medians_physical)
    r_vel_out_med_murray_strong, p_value_v_out_murray_strong = pf.pearson_correlation(bin_center_strong, murray_model_medians_strong)

    #calculate the r value for all the values
    r_vel_out_chen_all, p_value_v_out_chen_all = pf.pearson_correlation(sig_sfr, chen_model)
    r_vel_out_chen_physical, p_value_v_out_chen_physical = pf.pearson_correlation(sig_sfr[physical_mask], chen_model[physical_mask])
    r_vel_out_chen_strong, p_value_v_out_chen_strong = pf.pearson_correlation(sig_sfr[BIC_diff<-50], chen_model[BIC_diff<-50])

    r_vel_out_murray_all, p_value_v_out_murray_all = pf.pearson_correlation(sig_sfr, murray_model)
    r_vel_out_murray_physical, p_value_v_out_murray_physical = pf.pearson_correlation(sig_sfr[physical_mask], murray_model[physical_mask])
    r_vel_out_murray_strong, p_value_v_out_murray_strong = pf.pearson_correlation(sig_sfr[BIC_diff<-50], murray_model[BIC_diff<-50])

    #print average numbers for the different panels
    print('Number of spaxels in the first panel', vel_out.shape)
    print('All spaxels median v_out:', np.nanmedian(vel_out))
    print('All spaxels standard deviation v_out:', np.nanstd(vel_out))
    print('All spaxels median sigma_sfr:', np.nanmedian(sig_sfr))
    print('All spaxels standard deviation sigma_sfr:', np.nanstd(sig_sfr))
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


    print('Number of spaxels with strong BIC differences:', vel_out[BIC_diff_strong].shape)
    print('')

    print('Clean spaxels median v_out:', np.nanmedian(vel_out[BIC_diff_strong]))
    print('Clean spaxels standard deviation v_out:', np.nanstd(vel_out[BIC_diff_strong]))
    print('Clean spaxels median sigma_sfr:', np.nanmedian(sig_sfr[BIC_diff_strong]))
    print('Clean spaxels standard deviation sigma_sfr:', np.nanstd(sig_sfr[BIC_diff_strong]))
    print('')


    #-------
    #plot it
    #-------
    plt.rcParams.update(pf.get_rc_params())
    fig, ax = plt.subplots(nrows=2, ncols=3, sharey='row', sharex=True, figsize=(10,6), constrained_layout=True)

    #get colours from cmasher
    colours = cmr.take_cmap_colors('cmr.gem', 3, cmap_range=(0.25, 0.85), return_fmt='hex')

    #plot all points
    ax[0,0].scatter(sig_sfr[vel_disp>51], chen_model[vel_disp>51], marker='o', s=20, label='All KOFFEE fits; R={:.2f}'.format(r_vel_out_chen_all), color=colours[0], alpha=0.8)
    ax[0,0].scatter(sig_sfr[vel_disp<=51], chen_model[vel_disp<=51], marker='v', s=20, c=colours[0], alpha=0.8)

    if plot_medians == True:
        #indicate the error bars
        x_err = np.array([bin_center_all-logspace_all[:-1], logspace_all[1:]-bin_center_all])
        #plot the medians as error bars
        ax[0,0].errorbar(bin_center_all, chen_model_medians_all, xerr=x_err, marker='s', color='k', capsize=3.0, ls='none', ms=5, label='Median all KOFFEE fits; R={:.2f}'.format(r_vel_out_med_chen_all))

    #ax[0,0].set_ylim(-10, 10)
    ax[0,0].set_xscale('log')
    ax[0,0].set_xlim(np.nanmin(sig_sfr)-0.002, np.nanmax(sig_sfr)+2.0)

    lgnd = ax[0,0].legend(frameon=True, fontsize='small', loc='lower left', framealpha=0.5)
    #lgnd.legendHandles[0]._legmarker.set_markersize(3)
    if compare == 'subtract':
        ax[0,0].set_ylabel('Energy-Driven Model - Data [km s$^{-1}$]')
    elif compare == 'divide':
        ax[0,0].set_ylabel('log(Data/Energy-Driven Model)')
    #ax[0,0].set_xlabel('$\Sigma_{SFR}$ [M$_\odot$ yr$^{-1}$ kpc$^{-2}$]')
    ax[0,0].set_title('S/N > 20 and $\delta_{BIC}$<-10')

    #plot points within 90% radius
    ax[0,1].scatter(sig_sfr[radius>6.1], chen_model[radius>6.1], marker='o', s=20, label='All KOFFEE fits', edgecolors=colours[0], alpha=0.3, facecolors='none')
    ax[0,1].scatter(sig_sfr[vel_disp<=51], chen_model[vel_disp<=51], marker='v', s=20, edgecolors=colours[0], alpha=0.3, facecolors='none')
    ax[0,1].scatter(sig_sfr[physical_mask], chen_model[physical_mask], marker='o', s=20, label='Selected KOFFEE fits; R={:.2f}'.format(r_vel_out_chen_physical), color=colours[1], alpha=0.8)

    if plot_medians == True:
        x_err = np.array([bin_center_physical-logspace_physical[:-1], logspace_physical[1:]-bin_center_physical])
        #plot the medians as error bars
        ax[0,1].errorbar(bin_center_physical, chen_model_medians_physical, xerr=x_err, marker='s', color='k', capsize=3.0, ls='none', ms=5, label='Median of selected KOFFEE fits; R={:.2f}'.format(r_vel_out_med_chen_physical))

    #ax[1].set_xscale('log')
    lgnd = ax[0,1].legend(frameon=True, fontsize='small', loc='lower left', framealpha=0.5)
    #lgnd.legendHandles[0]._legmarker.set_markersize(3)
    #ax[0,1].set_xlabel('$\Sigma_{SFR}$ [M$_\odot$ yr$^{-1}$ kpc$^{-2}$]')
    ax[0,1].set_title(r'$\delta_{BIC}$<-10, $r$<$r_{90}$ and $\sigma_{broad}$>$\sigma_{inst}$')

    #plot points with strong BIC values
    ax[0,2].scatter(sig_sfr[~BIC_diff_strong][vel_disp[~BIC_diff_strong]>51], chen_model[~BIC_diff_strong][vel_disp[~BIC_diff_strong]>51], marker='o', s=20, label='All KOFFEE fits', color=colours[0], alpha=0.3, facecolors='none')
    ax[0,2].scatter(sig_sfr[~BIC_diff_strong][vel_disp[~BIC_diff_strong]<=51], chen_model[~BIC_diff_strong][vel_disp[~BIC_diff_strong]<=51], marker='v', s=20, edgecolors=colours[0], alpha=0.3, facecolors='none')
    ax[0,2].scatter(sig_sfr[BIC_diff_strong][vel_disp[BIC_diff_strong]>51], chen_model[BIC_diff_strong][vel_disp[BIC_diff_strong]>51], marker='o', s=20, label='Selected KOFFEE fits; R={:.2f}'.format(r_vel_out_chen_strong), color=colours[2], alpha=1.0)
    ax[0,2].scatter(sig_sfr[BIC_diff_strong][vel_disp[BIC_diff_strong]<=51], chen_model[BIC_diff_strong][vel_disp[BIC_diff_strong]<=51], marker='v', s=20, color=colours[2], alpha=1.0)

    if plot_medians == True:
        x_err = np.array([bin_center_strong-logspace_strong[:-1], logspace_strong[1:]-bin_center_physical])
        #plot the medians as error bars
        ax[0,2].errorbar(bin_center_strong, chen_model_medians_strong, xerr=x_err, marker='s', color='k', capsize=3.0, ls='none', ms=5, label='Median of selected KOFFEE fits; R={:.2f}'.format(r_vel_out_med_chen_strong))

    #ax[1].set_xscale('log')
    lgnd = ax[0,2].legend(frameon=True, fontsize='small', loc='lower left', framealpha=0.5)
    #lgnd.legendHandles[0]._legmarker.set_markersize(3)
    #ax[0,2].set_xlabel('$\Sigma_{SFR}$ [M$_\odot$ yr$^{-1}$ kpc$^{-2}$]')
    ax[0,2].set_title('strongly likely BIC $\delta_{BIC}$<-50')



    #plot all points
    ax[1,0].scatter(sig_sfr[vel_disp>51], murray_model[vel_disp>51], marker='o', s=20, label='All KOFFEE fits; R={:.2f}'.format(r_vel_out_murray_all), color=colours[0], alpha=0.8)
    ax[1,0].scatter(sig_sfr[vel_disp<=51], murray_model[vel_disp<=51], marker='v', s=20, c=colours[0], alpha=0.8)

    if plot_medians == True:
        #indicate the error bars
        x_err = np.array([bin_center_all-logspace_all[:-1], logspace_all[1:]-bin_center_all])
        #plot the medians as error bars
        ax[1,0].errorbar(bin_center_all, murray_model_medians_all, xerr=x_err, marker='s', color='k', capsize=3.0, ls='none', ms=5, label='Median all KOFFEE fits; R={:.2f}'.format(r_vel_out_med_murray_all))

    #ax[0].set_ylim(100, 700)
    ax[1,0].set_xscale('log')
    ax[1,0].set_xlim(np.nanmin(sig_sfr)-0.002, np.nanmax(sig_sfr)+2.0)

    lgnd = ax[1,0].legend(frameon=True, fontsize='small', loc='upper left', framealpha=0.5)
    #lgnd.legendHandles[0]._legmarker.set_markersize(3)
    if compare == 'subtract':
        ax[1,0].set_ylabel('Momentum-Driven Model - Data [km s$^{-1}$]')
    elif compare == 'divide':
        ax[1,0].set_ylabel('log(Data/Momentum-Driven Model)')

    ax[1,0].set_xlabel('$\Sigma_{SFR}$ [M$_\odot$ yr$^{-1}$ kpc$^{-2}$]')
    #ax[1,0].set_title('S/N > 20 and $\delta_{BIC}$<-10')

    #plot points within 90% radius
    ax[1,1].scatter(sig_sfr[radius>6.1], murray_model[radius>6.1], marker='o', s=20, label='All KOFFEE fits', edgecolors=colours[0], alpha=0.3, facecolors='none')
    ax[1,1].scatter(sig_sfr[vel_disp<=51], murray_model[vel_disp<=51], marker='v', s=20, edgecolors=colours[0], alpha=0.3, facecolors='none')
    ax[1,1].scatter(sig_sfr[physical_mask], murray_model[physical_mask], marker='o', s=20, label='Selected KOFFEE fits; R={:.2f}'.format(r_vel_out_murray_physical), color=colours[1], alpha=0.8)

    if plot_medians == True:
        x_err = np.array([bin_center_physical-logspace_physical[:-1], logspace_physical[1:]-bin_center_physical])
        #plot the medians as error bars
        ax[1,1].errorbar(bin_center_physical, murray_model_medians_physical, xerr=x_err, marker='s', color='k', capsize=3.0, ls='none', ms=5, label='Median of selected KOFFEE fits; R={:.2f}'.format(r_vel_out_med_murray_physical))

    #ax[1].set_xscale('log')
    lgnd = ax[1,1].legend(frameon=True, fontsize='small', loc='upper left', framealpha=0.5)
    #lgnd.legendHandles[0]._legmarker.set_markersize(3)
    ax[1,1].set_xlabel('$\Sigma_{SFR}$ [M$_\odot$ yr$^{-1}$ kpc$^{-2}$]')
    #ax[1,1].set_title(r'$\delta_{BIC}$<-10, $r$<$r_{90}$ and $\sigma_{broad}$>$\sigma_{inst}$')

    #plot points with strong BIC values
    ax[1,2].scatter(sig_sfr[~BIC_diff_strong][vel_disp[~BIC_diff_strong]>51], murray_model[~BIC_diff_strong][vel_disp[~BIC_diff_strong]>51], marker='o', s=20, label='All KOFFEE fits', color=colours[0], alpha=0.3, facecolors='none')
    ax[1,2].scatter(sig_sfr[~BIC_diff_strong][vel_disp[~BIC_diff_strong]<=51], murray_model[~BIC_diff_strong][vel_disp[~BIC_diff_strong]<=51], marker='v', s=20, edgecolors=colours[0], alpha=0.3, facecolors='none')
    ax[1,2].scatter(sig_sfr[BIC_diff_strong][vel_disp[BIC_diff_strong]>51], murray_model[BIC_diff_strong][vel_disp[BIC_diff_strong]>51], marker='o', s=20, label='Selected KOFFEE fits; R={:.2f}'.format(r_vel_out_murray_strong), color=colours[2], alpha=1.0)
    ax[1,2].scatter(sig_sfr[BIC_diff_strong][vel_disp[BIC_diff_strong]<=51], murray_model[BIC_diff_strong][vel_disp[BIC_diff_strong]<=51], marker='v', s=20, color=colours[2], alpha=1.0)

    if plot_medians == True:
        x_err = np.array([bin_center_strong-logspace_strong[:-1], logspace_strong[1:]-bin_center_physical])
        #plot the medians as error bars
        ax[1,2].errorbar(bin_center_strong, murray_model_medians_strong, xerr=x_err, marker='s', color='k', capsize=3.0, ls='none', ms=5, label='Median of selected KOFFEE fits; R={:.2f}'.format(r_vel_out_med_murray_strong))

    #ax[1].set_xscale('log')
    lgnd = ax[1,2].legend(frameon=True, fontsize='small', loc='upper left', framealpha=0.5)
    #lgnd.legendHandles[0]._legmarker.set_markersize(3)
    ax[1,2].set_xlabel('$\Sigma_{SFR}$ [M$_\odot$ yr$^{-1}$ kpc$^{-2}$]')
    #ax[1,2].set_title('strongly likely BIC $\delta_{BIC}$<-50')

    plt.show()



#===============================================================================
# OLDER PLOTTING FUNCTIONS
#===============================================================================

def plot_sfr_vout_compare_sfr_calcs(OIII_outflow_results, OIII_outflow_error, hbeta_outflow_results, hbeta_outflow_error, hbeta_no_outflow_results, hbeta_no_outflow_error, statistical_results, z, radius, header, weighted_average=True, colour_by_radius=False):
    """
    Plots the SFR surface density against the outflow velocity, comparing Sigma_SFR
    calculated from the full line to Sigma_SFR calculated using only the narrow
    component.

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
        array of galaxy radius values

    colour_by_radius : boolean
        Whether or not to colour the points by galaxy radius. (Default=False)

    Returns
    -------
    A six panel graph of velocity offset, velocity dispersion and outflow velocity
    against the SFR surface density calculated two different ways.

    """
    #calculate the outflow velocity
    vel_disp, vel_disp_err, vel_diff, vel_diff_err, vel_out, vel_out_err = calc_outvel.calc_outflow_vel(OIII_outflow_results, OIII_outflow_error, statistical_results, z)

    #calculate the sfr surface density - using just the systemic line, and including the flux line
    #don't include extinction since this was included in the continuum subtraction using ppxf
    sfr_sys, total_sfr_sys, sfr_surface_density_sys, h_beta_integral_err_sys = calc_sfr.calc_sfr_koffee(hbeta_outflow_results, hbeta_outflow_error, hbeta_no_outflow_results, hbeta_no_outflow_error, statistical_results, z, header, include_extinction=False, include_outflow=False)
    sfr_flow, total_sfr_flow, sfr_surface_density_flow, h_beta_integral_err_flow = calc_sfr.calc_sfr_koffee(hbeta_outflow_results, hbeta_outflow_error, hbeta_no_outflow_results, hbeta_no_outflow_error, statistical_results, z, header, include_extinction=False, include_outflow=True)

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
        bin_center, v_out_bin_medians_sys, v_out_bin_lower_q_sys, v_out_bin_upper_q_sys = pf.binned_median_quantile_log(sfr_sys, vel_out, num_bins=num_bins, weights=None, min_bin=None, max_bin=None)
        bin_center, vel_diff_bin_medians_sys, vel_diff_bin_lower_q_sys, vel_diff_bin_upper_q_sys = pf.binned_median_quantile_log(sfr_sys, vel_diff, num_bins=num_bins, weights=None, min_bin=min_bin, max_bin=max_bin)
        bin_center, disp_bin_medians_sys, disp_bin_lower_q_sys, disp_bin_upper_q_sys = pf.binned_median_quantile_log(sfr_sys, vel_disp, num_bins=num_bins, weights=None, min_bin=min_bin, max_bin=max_bin)

        bin_center, v_out_bin_medians_flow, v_out_bin_lower_q_flow, v_out_bin_upper_q_flow = pf.binned_median_quantile_log(sfr_flow, vel_out, num_bins=num_bins, weights=None, min_bin=min_bin, max_bin=max_bin)
        bin_center, vel_diff_bin_medians_flow, vel_diff_bin_lower_q_flow, vel_diff_bin_upper_q_flow = pf.binned_median_quantile_log(sfr_flow, vel_diff, num_bins=num_bins, weights=None, min_bin=min_bin, max_bin=max_bin)
        bin_center, disp_bin_medians_flow, disp_bin_lower_q_flow, disp_bin_upper_q_flow = pf.binned_median_quantile_log(sfr_flow, vel_disp, num_bins=num_bins, weights=None, min_bin=min_bin, max_bin=max_bin)

    elif weighted_average == True:
        bin_center, v_out_bin_medians_sys, v_out_bin_lower_q_sys, v_out_bin_upper_q_sys = pf.binned_median_quantile_log(sfr_sys, vel_out, num_bins=num_bins, weights=[vel_out_err], min_bin=None, max_bin=None)
        bin_center, vel_diff_bin_medians_sys, vel_diff_bin_lower_q_sys, vel_diff_bin_upper_q_sys = pf.binned_median_quantile_log(sfr_sys, vel_diff, num_bins=num_bins, weights=[vel_diff_err], min_bin=min_bin, max_bin=max_bin)
        bin_center, disp_bin_medians_sys, disp_bin_lower_q_sys, disp_bin_upper_q_sys = pf.binned_median_quantile_log(sfr_sys, vel_disp, num_bins=num_bins, weights=[vel_disp_err], min_bin=min_bin, max_bin=max_bin)

        bin_center, v_out_bin_medians_flow, v_out_bin_lower_q_flow, v_out_bin_upper_q_flow = pf.binned_median_quantile_log(sfr_flow, vel_out, num_bins=num_bins, weights=[vel_out_err], min_bin=min_bin, max_bin=max_bin)
        bin_center, vel_diff_bin_medians_flow, vel_diff_bin_lower_q_flow, vel_diff_bin_upper_q_flow = pf.binned_median_quantile_log(sfr_flow, vel_diff, num_bins=num_bins, weights=[vel_diff_err], min_bin=min_bin, max_bin=max_bin)
        bin_center, disp_bin_medians_flow, disp_bin_lower_q_flow, disp_bin_upper_q_flow = pf.binned_median_quantile_log(sfr_flow, vel_disp, num_bins=num_bins, weights=[vel_disp_err], min_bin=min_bin, max_bin=max_bin)


    print(bin_center)
    print(v_out_bin_medians_sys)

    #calculate the r value for the median values
    r_vel_out_med_sys, p_value_v_out = pf.pearson_correlation(bin_center, v_out_bin_medians_sys)
    r_vel_diff_med_sys, p_value_v_diff = pf.pearson_correlation(bin_center, vel_diff_bin_medians_sys)
    r_disp_med_sys, p_value_disp = pf.pearson_correlation(bin_center, disp_bin_medians_sys)

    r_vel_out_med_flow, p_value_v_out = pf.pearson_correlation(bin_center, v_out_bin_medians_flow)
    r_vel_diff_med_flow, p_value_v_diff = pf.pearson_correlation(bin_center, vel_diff_bin_medians_flow)
    r_disp_med_flow, p_value_disp = pf.pearson_correlation(bin_center, disp_bin_medians_flow)

    #calculate the r value for all the values
    r_vel_out_sys, p_value_v_out = pf.pearson_correlation(sfr_sys, vel_out)
    r_vel_diff_sys, p_value_v_diff = pf.pearson_correlation(sfr_sys, vel_diff)
    r_disp_sys, p_value_disp = pf.pearson_correlation(sfr_sys, vel_disp)

    r_vel_out_flow, p_value_v_out = pf.pearson_correlation(sfr_flow, vel_out)
    r_vel_diff_flow, p_value_v_diff = pf.pearson_correlation(sfr_flow, vel_diff)
    r_disp_flow, p_value_disp = pf.pearson_correlation(sfr_flow, vel_disp)


    #create vectors to plot the literature trends
    sfr_surface_density_chen, v_out_chen = pf.chen_et_al_2010(sfr_sys.min(), sfr_sys.max(), scale_factor=np.nanmedian(vel_out)/(np.nanmedian(sfr_flow)**0.1))
    sfr_surface_density_murray, v_out_murray = pf.murray_et_al_2011(sfr_sys.min(), sfr_sys.max(), scale_factor=np.nanmedian(vel_out)/(np.nanmedian(sfr_flow)**2))

    sfr_surface_density_chen, vel_diff_chen = pf.chen_et_al_2010(sfr_sys.min(), sfr_sys.max(), scale_factor=np.nanmedian(vel_diff)/(np.nanmedian(sfr_flow)**0.1))
    sfr_surface_density_murray, vel_diff_murray = pf.murray_et_al_2011(sfr_sys.min(), sfr_sys.max(), scale_factor=np.nanmedian(vel_diff)/(np.nanmedian(sfr_flow)**2))

    #plot it
    plt.rcParams.update(pf.get_rc_params())
    fig, ax = plt.subplots(nrows=2, ncols=3, sharex=True, figsize=(12,7))

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



#===============================================================================
# PLOTTING FUNCTIONS - other
#===============================================================================
def plot_sfr_surface_density_radius(sig_sfr, rad_flat, stat_results):
    """
    Plots the SFR surface density against galaxy radius

    Parameters
    ----------
    sig_sfr : :obj:'~numpy.ndarray'
        the SFR surface density

    rad : :obj:'~numpy.ndarray'
        the array of galaxy radius

    stat_results : :obj:'~numpy.ndarray'
        the statistical results from KOFFEE, same shape as rad

    Returns
    -------
    A graph of the SFR surface density against galaxy radius
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
    plt.rcParams.update(pf.get_rc_params())
    plt.figure()
    plt.scatter(rad_flow, sig_sfr_flow, label='Outflow spaxels')
    plt.scatter(rad_no_flow, sig_sfr_no_flow, label='No outflow spaxels')
    plt.yscale('log')
    plt.xlabel('Radius [Arcseconds]')
    plt.ylabel('Log $\Sigma_{SFR}$ [M$_\odot$ yr$^{-1}$]')
    plt.legend()
    plt.show()



def plot_outflow_frequency_sfr_surface_density(sig_sfr, stat_results):
    """
    Plots the frequency of outflow spaxels, and non-outflow spaxels against
    radius, using the flattened arrays.

    Parameters
    ----------
    sig_sfr : :obj:'~numpy.ndarray'
        the SFR surface density

    stat_results : :obj:'~numpy.ndarray'
        the statistical results from KOFFEE, same shape as sig_sfr

    Returns
    -------
    A histogram of spaxels at each SFR surface density
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
    plt.rcParams.update(pf.get_rc_params())
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
    Plots the frequency of outflow spaxels, and non-outflow spaxels against
    radius, using the flattened arrays.
    (Does the same thing as plot_outflow_frequency_sfr_surface_density but with
    for loops)

    Parameters
    ----------
    sig_sfr : :obj:'~numpy.ndarray'
        the SFR surface density

    stat_results : :obj:'~numpy.ndarray'
        the statistical results from KOFFEE, same shape as sig_sfr

    Returns
    -------
    A histogram of spaxels at each SFR surface density
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
    plt.rcParams.update(pf.get_rc_params())
    plt.figure()
    plt.plot(sfr[vel_res_type==1], vel_res[vel_res_type==1], marker='o', lw=0, label='Outflow', alpha=0.2)
    plt.plot(sfr[vel_res_type==-1], vel_res[vel_res_type==-1], marker='o', lw=0, label='Inflow', alpha=0.2)
    plt.xscale('log')
    plt.legend(frameon=False, fontsize='small', loc='lower left')
    plt.ylabel('Flow velocity (km s$^{-1}$)')
    plt.xlabel('Log Star Formation Rate Surface Density (M$_\odot$ yr$^{-1}$ kpc$^{-2}$)')
    plt.show()

    return vel_res, vel_res_type, vel_diffs
