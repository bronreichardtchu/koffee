"""
NAME:
	outflow_velocity_plots.py

AUTHOR:
	Bronwyn Reichardt Chu
	Swinburne
	2021

EMAIL:
	<breichardtchu@swin.edu.au>

PURPOSE:
	To make plots of results from koffee against the outflow velocity
	Written on MacOS Mojave 10.14.5, with Python 3.7

FUNCTIONS INCLUDED:
    plot_sfr_vout
    plot_sfr_vout_no_koffee_checks
    plot_vel_out_radius
    plot_out_vel_model_rad
    plot_out_vel_model_sigsfr
    plot_out_vel_mlf
    plot_out_vel_disp_mlf
    plot_sfr_vseparate
    plot_vel_diff_mlf
    plot_out_vel_flux
    map_of_outflows

MODIFICATION HISTORY:
		v.1.0 - first created January 2021

"""
import numpy as np

import matplotlib.pyplot as plt
import cmasher as cmr

from scipy.optimize import curve_fit

from astropy import units as u

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
def plot_sfr_vout(OIII_outflow_results, OIII_outflow_error, hbeta_outflow_results, hbeta_outflow_error, hbeta_no_outflow_results, hbeta_no_outflow_error, BIC_outflow, BIC_no_outflow, statistical_results, z, radius, weighted_average=True):
    """
    Plots the SFR surface density against the outflow velocity, with Sigma_SFR
    calculated using only the narrow component.

    Parameters
    ----------
    OIII_outflow_results : :obj:'~numpy.ndarray'
        array of outflow results from KOFFEE for OIII line.  Used to calculate the outflow velocity.  Should be (7, statistical_results.shape)

    OIII_outflow_err : :obj:'~numpy.ndarray'
        array of the outflow result errors from KOFFEE for OIII line

    hbeta_outflow_results : :obj:'~numpy.ndarray'
        array of outflow results from KOFFEE for Hbeta line.  Used to calculate the Sigma SFR.  Should be (7, statistical_results.shape)

    hbeta_outflow_err : :obj:'~numpy.ndarray'
        array of the outflow result errors from KOFFEE for Hbeta line

    hbeta_no_outflow_results : :obj:'~numpy.ndarray'
        array of single gaussian results from KOFFEE for Hbeta line.  Used to calculate the Sigma SFR.  Should be (4, statistical_results.shape)

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

    weighted_average : boolean
        whether or not to take a weighted average using the errors (Default=True)

    Returns
    -------
    A graph of outflow velocity against the SFR surface density in three panels
    with different data selections

    """
    #calculate the outflow velocity
    vel_disp, vel_disp_err, vel_diff, vel_diff_err, vel_out, vel_out_err = calc_outvel.calc_outflow_vel(OIII_outflow_results, OIII_outflow_error, statistical_results, z)

    #calculate the sfr surface density - using just the systemic line, and including the flux line
    #don't include extinction since this was included in the continuum subtraction using ppxf
    sfr, sfr_err, total_sfr, sfr_surface_density, sfr_surface_density_err = calc_sfr.calc_sfr_koffee(hbeta_outflow_results, hbeta_outflow_error, hbeta_no_outflow_results, hbeta_no_outflow_error, statistical_results, z, include_extinction=False, include_outflow=False)

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
    plt.rcParams.update(pf.get_rc_params())
    fig, ax = plt.subplots(nrows=1, ncols=3, sharey=True, sharex=True, figsize=(10,4), constrained_layout=True)

    #get colours from cmasher
    colours = cmr.take_cmap_colors('cmr.gem', 3, cmap_range=(0.25, 0.85), return_fmt='hex')

    #plot all points
    ax[0].fill_between(bin_center_all, v_out_bin_lower_q_all, v_out_bin_upper_q_all, color=colours[0], alpha=0.3)
    ax[0].scatter(sig_sfr, vel_out, marker='o', s=10, label='All KOFFEE fits; R={:.2f}'.format(r_vel_out_all), color=colours[0], alpha=0.8)
    ax[0].plot(bin_center_all, v_out_bin_medians_all, marker='', lw=3, label='Median all KOFFEE fits; R={:.2f}'.format(r_vel_out_med_all), color=colours[0])


    ax[0].plot(sfr_linspace, pf.fitting_function(sfr_linspace, *popt_vout_all), 'r-', label='Fit: $v_{out}=%5.0f\pm$%2.0f $\Sigma_{SFR}^{%5.2f \pm %5.2f}$' % (popt_vout_all[0], np.sqrt(np.diag(pcov_vout_all))[0], popt_vout_all[1], np.sqrt(np.diag(pcov_vout_all))[1]))
    ax[0].plot(sfr_linspace, pf.fitting_function(sfr_linspace, *popt_vout_all_medians), 'r--', label='Median Fit: $v_{out}=%5.0f\pm$%2.0f $\Sigma_{SFR}^{%5.2f \pm %5.2f}$' %(popt_vout_all_medians[0], np.sqrt(np.diag(pcov_vout_all_medians))[0], popt_vout_all_medians[1], np.sqrt(np.diag(pcov_vout_all_medians))[1]))

    ax[0].plot(sfr_surface_density_chen, v_out_chen, ':k', label='Energy driven, $v_{out} \propto \Sigma_{SFR}^{0.1}$')
    ax[0].plot(sfr_surface_density_murray, v_out_murray, '--k', label='Momentum driven, $v_{out} \propto \Sigma_{SFR}^{2}$')

    ax[0].errorbar(0.03, 150, xerr=np.nanmedian(sig_sfr_err), yerr=np.nanmedian(vel_out_err), c='k')

    ax[0].set_ylim(100, 610)
    ax[0].set_xscale('log')
    ax[0].set_xlim(0.002, 3)
    lgnd = ax[0].legend(frameon=True, fontsize='small', loc='upper left', framealpha=0.5)
    lgnd.legendHandles[0]._legmarker.set_markersize(3)
    ax[0].set_ylabel('Maximum Outflow Velocity [km s$^{-1}$]')
    ax[0].set_xlabel('$\Sigma_{SFR}$ [M$_\odot$ yr$^{-1}$ kpc$^{-2}$]')
    ax[0].set_title('all spaxels')

    #plot points within 90% radius
    ax[1].fill_between(bin_center_physical, v_out_bin_lower_q_physical, v_out_bin_upper_q_physical, color=colours[1], alpha=0.3)
    ax[1].scatter(sig_sfr[radius>6.1], vel_out[radius>6.1], marker='o', s=10, label='All KOFFEE fits', edgecolors=colours[0], alpha=0.3, facecolors='none')
    ax[1].scatter(sig_sfr[vel_disp<=51], vel_out[vel_disp<=51], marker='v', s=10, edgecolors=colours[0], alpha=0.3, facecolors='none')
    ax[1].scatter(sig_sfr[physical_mask], vel_out[physical_mask], marker='o', s=10, label='Selected KOFFEE fits; R={:.2f}'.format(r_vel_out_physical), color=colours[1], alpha=0.8)
    ax[1].plot(bin_center_physical, v_out_bin_medians_physical, marker='', lw=3, label='Median of selected KOFFEE fits; R={:.2f}'.format(r_vel_out_med_physical), color=colours[1])

    ax[1].plot(sfr_linspace, pf.fitting_function(sfr_linspace, *popt_vout_physical), 'r-', label='Fit: $v_{out}=%5.0f\pm$%2.0f $\Sigma_{SFR}^{%5.2f \pm %5.2f}$' % (popt_vout_physical[0], np.sqrt(np.diag(pcov_vout_physical))[0], popt_vout_physical[1], np.sqrt(np.diag(pcov_vout_physical))[1]))
    ax[1].plot(sfr_linspace, pf.fitting_function(sfr_linspace, *popt_vout_physical_medians), 'r--', label='Median Fit: $v_{out}=%5.0f\pm$%2.0f $\Sigma_{SFR}^{%5.2f \pm %5.2f}$' %(popt_vout_physical_medians[0], np.sqrt(np.diag(pcov_vout_physical_medians))[0], popt_vout_physical_medians[1], np.sqrt(np.diag(pcov_vout_physical_medians))[1]))

    ax[1].plot(sfr_surface_density_chen, v_out_chen, ':k')#, label='Energy driven, $v_{out} \propto \Sigma_{SFR}^{0.1}$')
    ax[1].plot(sfr_surface_density_murray, v_out_murray, '--k')#, label='Momentum driven, $v_{out} \propto \Sigma_{SFR}^{2}$')

    ax[1].errorbar(0.03, 150, xerr=np.nanmedian(sig_sfr_err[physical_mask]), yerr=np.nanmedian(vel_out_err[physical_mask]), c='k')

    #ax[1].set_xscale('log')
    lgnd = ax[1].legend(frameon=True, fontsize='small', loc='upper left', framealpha=0.5)
    lgnd.legendHandles[0]._legmarker.set_markersize(3)
    ax[1].set_xlabel('$\Sigma_{SFR}$ [M$_\odot$ yr$^{-1}$ kpc$^{-2}$]')
    ax[1].set_title(r'$r$<$r_{90}$ and $\sigma_{broad}$>$\sigma_{inst}$')

    #plot points with strong BIC values
    ax[2].fill_between(bin_center_clean, v_out_bin_lower_q_clean, v_out_bin_upper_q_clean, color=colours[2], alpha=0.3)
    #ax[2].scatter(sig_sfr[radius>6.1], vel_out[radius>6.1], marker='o', s=10, label='All KOFFEE fits', edgecolors=colours[0], alpha=0.3, facecolors='none')
    #ax[2].scatter(sig_sfr[vel_disp<=51], vel_out[vel_disp<=51], marker='v', s=10, edgecolors=colours[0], alpha=0.3, facecolors='none')
    #ax[2].scatter(sig_sfr[physical_mask][BIC_diff[physical_mask]>=-51], vel_out[physical_mask][BIC_diff[physical_mask]>=-51], marker='o', s=10, edgecolors=colours[1], alpha=0.3, facecolors='none')
    ax[2].scatter(sig_sfr[~BIC_diff_strong], vel_out[~BIC_diff_strong], marker='o', s=10, label='All KOFFEE fits', color=colours[0], alpha=0.3, facecolors='none')
    ax[2].scatter(sig_sfr[BIC_diff_strong], vel_out[BIC_diff_strong], marker='o', s=10, label='Selected KOFFEE fits; R={:.2f}'.format(r_vel_out_clean), color=colours[2], alpha=1.0)
    ax[2].plot(bin_center_clean, v_out_bin_medians_clean, marker='', lw=3, label='Median of selected KOFFEE fits; R={:.2f}'.format(r_vel_out_med_clean), color=colours[2])

    ax[2].plot(sfr_linspace, pf.fitting_function(sfr_linspace, *popt_vout_clean), 'r-', label='Fit: $v_{out}=%5.0f\pm$%2.0f $\Sigma_{SFR}^{%5.2f \pm %5.2f}$' % (popt_vout_clean[0], np.sqrt(np.diag(pcov_vout_clean))[0], popt_vout_clean[1], np.sqrt(np.diag(pcov_vout_clean))[1]))
    ax[2].plot(sfr_linspace, pf.fitting_function(sfr_linspace, *popt_vout_clean_medians), 'r--', label='Median Fit: $v_{out}=%5.0f\pm$%2.0f $\Sigma_{SFR}^{%5.2f \pm %5.2f}$' %(popt_vout_clean_medians[0], np.sqrt(np.diag(pcov_vout_clean_medians))[0], popt_vout_clean_medians[1], np.sqrt(np.diag(pcov_vout_clean_medians))[1]))

    ax[2].plot(sfr_surface_density_chen, v_out_chen, ':k')#, label='Energy driven, $v_{out} \propto \Sigma_{SFR}^{0.1}$')
    ax[2].plot(sfr_surface_density_murray, v_out_murray, '--k')#, label='Momentum driven, $v_{out} \propto \Sigma_{SFR}^{2}$')

    ax[2].errorbar(0.03, 150, xerr=np.nanmedian(sig_sfr_err[BIC_diff_strong]), yerr=np.nanmedian(vel_out_err[BIC_diff_strong]), c='k')

    #ax[1].set_xscale('log')
    lgnd = ax[2].legend(frameon=True, fontsize='small', loc='upper left', framealpha=0.5)
    lgnd.legendHandles[0]._legmarker.set_markersize(3)
    ax[2].set_xlabel('$\Sigma_{SFR}$ [M$_\odot$ yr$^{-1}$ kpc$^{-2}$]')
    ax[2].set_title('strongly likely BIC')

    plt.show()



#===============================================================================
# OTHER PLOTTING FUNCTIONS
#===============================================================================

def plot_sfr_vout_no_koffee_checks(lamdas, data, OIII_outflow_results, OIII_outflow_error, BIC_outflow, BIC_no_outflow, statistical_results, z, radius, weighted_average=True):
    """
    Plots the SFR surface density against the outflow velocity calculated from
    KOFFEE without the usual checks, with Sigma_SFR calculated using only the
    narrow component.

    Parameters
    ----------
    lamdas : :obj:'~numpy.ndarray'
        wavelength vector for the data

    data : :obj:'~numpy.ndarray'
        3D data cube for the galaxy with shape [len(lamdas),:,:]

    OIII_outflow_results : :obj:'~numpy.ndarray'
        array of outflow results from KOFFEE for OIII line.  Used to calculate
        the outflow velocity.  Should be (7, statistical_results.shape)

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

    weighted_average : boolean
        whether or not to take a weighted average using the errors (Default=True)

    Returns
    -------
    A graph of outflow velocity against the SFR surface density

    """
    #calculate the outflow velocity
    vel_disp, vel_disp_err, vel_diff, vel_diff_err, vel_out, vel_out_err = calc_outvel.calc_outflow_vel(OIII_outflow_results, OIII_outflow_error, statistical_results, z)

    #calculate the sfr surface density - using just the systemic line, and including the flux line
    #don't include extinction since this was included in the continuum subtraction using ppxf
    sfr, total_sfr, sfr_surface_density, h_beta_spec = calc_sfr.calc_sfr_integrate(lamdas, data, z, cont_subtract=False, include_extinction=False)

    #get the sfr for the outflow spaxels
    flow_mask = (statistical_results>0) #& (sfr_surface_density>0.1)

    #flatten all the arrays and get rid of extra spaxels
    sig_sfr = sfr_surface_density[flow_mask]
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
    plt.rcParams.update(pf.get_rc_params())
    fig, ax = plt.subplots(nrows=1, ncols=3, sharey=True, sharex=True, figsize=(10,4), constrained_layout=True)

    #get colours from cmasher
    colours = cmr.take_cmap_colors('cmr.gem', 3, cmap_range=(0.25, 0.85), return_fmt='hex')

    #plot all points
    ax[0].fill_between(bin_center_all, v_out_bin_lower_q_all, v_out_bin_upper_q_all, color=colours[0], alpha=0.3)
    ax[0].scatter(sig_sfr, vel_out, marker='o', s=10, label='All KOFFEE fits; R={:.2f}'.format(r_vel_out_all), color=colours[0], alpha=0.8)
    ax[0].plot(bin_center_all, v_out_bin_medians_all, marker='', lw=3, label='Median all KOFFEE fits; R={:.2f}'.format(r_vel_out_med_all), color=colours[0])


    ax[0].plot(sfr_linspace, pf.fitting_function(sfr_linspace, *popt_vout_all), 'r-', label='Fit: $v_{out}=%5.0f\pm$%2.0f $\Sigma_{SFR}^{%5.2f \pm %5.2f}$' % (popt_vout_all[0], np.sqrt(np.diag(pcov_vout_all))[0], popt_vout_all[1], np.sqrt(np.diag(pcov_vout_all))[1]))
    ax[0].plot(sfr_linspace, pf.fitting_function(sfr_linspace, *popt_vout_all_medians), 'r--', label='Median Fit: $v_{out}=%5.0f\pm$%2.0f $\Sigma_{SFR}^{%5.2f \pm %5.2f}$' %(popt_vout_all_medians[0], np.sqrt(np.diag(pcov_vout_all_medians))[0], popt_vout_all_medians[1], np.sqrt(np.diag(pcov_vout_all_medians))[1]))

    ax[0].plot(sfr_surface_density_chen, v_out_chen, ':k', label='Energy driven, $v_{out} \propto \Sigma_{SFR}^{0.1}$')
    ax[0].plot(sfr_surface_density_murray, v_out_murray, '--k', label='Momentum driven, $v_{out} \propto \Sigma_{SFR}^{2}$')

    #ax[0].errorbar(0.03, 150, xerr=np.nanmedian(sig_sfr_err), yerr=np.nanmedian(vel_out_err), c='k')

    ax[0].set_ylim(100, 610)
    ax[0].set_xscale('log')
    ax[0].set_xlim(0.002, 3)
    lgnd = ax[0].legend(frameon=True, fontsize='small', loc='upper left', framealpha=0.5)
    lgnd.legendHandles[0]._legmarker.set_markersize(3)
    ax[0].set_ylabel('Maximum Outflow Velocity [km s$^{-1}$]')
    ax[0].set_xlabel('$\Sigma_{SFR}$ [M$_\odot$ yr$^{-1}$ kpc$^{-2}$]')
    ax[0].set_title('all spaxels')

    #plot points within 90% radius
    ax[1].fill_between(bin_center_physical, v_out_bin_lower_q_physical, v_out_bin_upper_q_physical, color=colours[1], alpha=0.3)
    ax[1].scatter(sig_sfr[radius>6.1], vel_out[radius>6.1], marker='o', s=10, label='All KOFFEE fits', edgecolors=colours[0], alpha=0.3, facecolors='none')
    ax[1].scatter(sig_sfr[vel_disp<=51], vel_out[vel_disp<=51], marker='v', s=10, edgecolors=colours[0], alpha=0.3, facecolors='none')
    ax[1].scatter(sig_sfr[physical_mask], vel_out[physical_mask], marker='o', s=10, label='Selected KOFFEE fits; R={:.2f}'.format(r_vel_out_physical), color=colours[1], alpha=0.8)
    ax[1].plot(bin_center_physical, v_out_bin_medians_physical, marker='', lw=3, label='Median of selected KOFFEE fits; R={:.2f}'.format(r_vel_out_med_physical), color=colours[1])

    ax[1].plot(sfr_linspace, pf.fitting_function(sfr_linspace, *popt_vout_physical), 'r-', label='Fit: $v_{out}=%5.0f\pm$%2.0f $\Sigma_{SFR}^{%5.2f \pm %5.2f}$' % (popt_vout_physical[0], np.sqrt(np.diag(pcov_vout_physical))[0], popt_vout_physical[1], np.sqrt(np.diag(pcov_vout_physical))[1]))
    ax[1].plot(sfr_linspace, pf.fitting_function(sfr_linspace, *popt_vout_physical_medians), 'r--', label='Median Fit: $v_{out}=%5.0f\pm$%2.0f $\Sigma_{SFR}^{%5.2f \pm %5.2f}$' %(popt_vout_physical_medians[0], np.sqrt(np.diag(pcov_vout_physical_medians))[0], popt_vout_physical_medians[1], np.sqrt(np.diag(pcov_vout_physical_medians))[1]))

    ax[1].plot(sfr_surface_density_chen, v_out_chen, ':k')#, label='Energy driven, $v_{out} \propto \Sigma_{SFR}^{0.1}$')
    ax[1].plot(sfr_surface_density_murray, v_out_murray, '--k')#, label='Momentum driven, $v_{out} \propto \Sigma_{SFR}^{2}$')

    #ax[1].errorbar(0.03, 150, xerr=np.nanmedian(sig_sfr_err[physical_mask]), yerr=np.nanmedian(vel_out_err[physical_mask]), c='k')

    #ax[1].set_xscale('log')
    lgnd = ax[1].legend(frameon=True, fontsize='small', loc='upper left', framealpha=0.5)
    lgnd.legendHandles[0]._legmarker.set_markersize(3)
    ax[1].set_xlabel('$\Sigma_{SFR}$ [M$_\odot$ yr$^{-1}$ kpc$^{-2}$]')
    ax[1].set_title(r'$r$<$r_{90}$ and $\sigma_{broad}$>$\sigma_{inst}$')

    #plot points with strong BIC values
    ax[2].fill_between(bin_center_clean, v_out_bin_lower_q_clean, v_out_bin_upper_q_clean, color=colours[2], alpha=0.3)
    #ax[2].scatter(sig_sfr[radius>6.1], vel_out[radius>6.1], marker='o', s=10, label='All KOFFEE fits', edgecolors=colours[0], alpha=0.3, facecolors='none')
    #ax[2].scatter(sig_sfr[vel_disp<=51], vel_out[vel_disp<=51], marker='v', s=10, edgecolors=colours[0], alpha=0.3, facecolors='none')
    #ax[2].scatter(sig_sfr[physical_mask][BIC_diff[physical_mask]>=-51], vel_out[physical_mask][BIC_diff[physical_mask]>=-51], marker='o', s=10, edgecolors=colours[1], alpha=0.3, facecolors='none')
    ax[2].scatter(sig_sfr[~BIC_diff_strong], vel_out[~BIC_diff_strong], marker='o', s=10, label='All KOFFEE fits', color=colours[0], alpha=0.3, facecolors='none')
    ax[2].scatter(sig_sfr[BIC_diff_strong], vel_out[BIC_diff_strong], marker='o', s=10, label='Selected KOFFEE fits; R={:.2f}'.format(r_vel_out_clean), color=colours[2], alpha=1.0)
    ax[2].plot(bin_center_clean, v_out_bin_medians_clean, marker='', lw=3, label='Median of selected KOFFEE fits; R={:.2f}'.format(r_vel_out_med_clean), color=colours[2])

    ax[2].plot(sfr_linspace, pf.fitting_function(sfr_linspace, *popt_vout_clean), 'r-', label='Fit: $v_{out}=%5.0f\pm$%2.0f $\Sigma_{SFR}^{%5.2f \pm %5.2f}$' % (popt_vout_clean[0], np.sqrt(np.diag(pcov_vout_clean))[0], popt_vout_clean[1], np.sqrt(np.diag(pcov_vout_clean))[1]))
    ax[2].plot(sfr_linspace, pf.fitting_function(sfr_linspace, *popt_vout_clean_medians), 'r--', label='Median Fit: $v_{out}=%5.0f\pm$%2.0f $\Sigma_{SFR}^{%5.2f \pm %5.2f}$' %(popt_vout_clean_medians[0], np.sqrt(np.diag(pcov_vout_clean_medians))[0], popt_vout_clean_medians[1], np.sqrt(np.diag(pcov_vout_clean_medians))[1]))

    ax[2].plot(sfr_surface_density_chen, v_out_chen, ':k')#, label='Energy driven, $v_{out} \propto \Sigma_{SFR}^{0.1}$')
    ax[2].plot(sfr_surface_density_murray, v_out_murray, '--k')#, label='Momentum driven, $v_{out} \propto \Sigma_{SFR}^{2}$')

    #ax[2].errorbar(0.03, 150, xerr=np.nanmedian(sig_sfr_err[BIC_diff_strong]), yerr=np.nanmedian(vel_out_err[BIC_diff_strong]), c='k')

    #ax[1].set_xscale('log')
    lgnd = ax[2].legend(frameon=True, fontsize='small', loc='upper left', framealpha=0.5)
    lgnd.legendHandles[0]._legmarker.set_markersize(3)
    ax[2].set_xlabel('$\Sigma_{SFR}$ [M$_\odot$ yr$^{-1}$ kpc$^{-2}$]')
    ax[2].set_title('strongly likely BIC')

    plt.show()



def plot_vel_out_radius(rad, OIII_outflow_results, OIII_outflow_error, statistical_results, z):
    """
    Plots the SFR surface density against galaxy radius

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
    A plot of outflow velocity against galaxy radius
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




def plot_out_vel_model_rad(OIII_outflow_results, OIII_outflow_error, hbeta_outflow_results, hbeta_outflow_error, hbeta_no_outflow_results, hbeta_no_outflow_error, BIC_outflow, BIC_no_outflow, statistical_results, z, radius, compare='divide'):
    """
    Plots the outflow velocity values found with KOFFEE either divided by or
    subtracted from the expected values calculated with the model from
    Chen et al. 2010 against the galaxy radius

    Parameters
    ----------
    OIII_outflow_results : :obj:'~numpy.ndarray'
        array of outflow results from KOFFEE for OIII line.  Used to calculate the outflow velocity.  Should be (7, statistical_results.shape)

    OIII_outflow_err : :obj:'~numpy.ndarray'
        array of the outflow result errors from KOFFEE for OIII line

    hbeta_outflow_results : :obj:'~numpy.ndarray'
        array of outflow results from KOFFEE for Hbeta line.  Used to calculate the Sigma SFR.  Should be (7, statistical_results.shape)

    hbeta_outflow_err : :obj:'~numpy.ndarray'
        array of the outflow result errors from KOFFEE for Hbeta line

    hbeta_no_outflow_results : :obj:'~numpy.ndarray'
        array of single gaussian results from KOFFEE for Hbeta line.  Used to calculate the Sigma SFR.  Should be (4, statistical_results.shape)

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

    compare : string
        string defining how to compare the results to the model.  Can be 'divide'
        or 'subtract' (Default='divide')

    Returns
    -------
    A graph of the outflow velocity compared to the expected model against galaxy
    radius

    """
    #calculate the sfr surface density - using just the systemic line, and including the flux line
    #don't include extinction since this was included in the continuum subtraction using ppxf
    sfr, sfr_err, total_sfr, sfr_surface_density, sfr_surface_density_err = calc_sfr.calc_sfr_koffee(hbeta_outflow_results, hbeta_outflow_error, hbeta_no_outflow_results, hbeta_no_outflow_error, statistical_results, z, include_extinction=False, include_outflow=False)

    #calculate the velocity dispersion for the masking
    vel_disp, vel_disp_err, vel_diff, vel_diff_err, vel_out, vel_out_err = calc_outvel.calc_outflow_vel(OIII_outflow_results, OIII_outflow_error, statistical_results, z)

    #create the flow mask
    flow_mask = (statistical_results>0) #& (np.isnan(hbeta_outflow_results[3,:,:])==False)


    #flatten all the arrays and get rid of extra spaxels
    sig_sfr = sfr_surface_density[flow_mask]
    sig_sfr_err = sfr_surface_density_err[flow_mask]

    vel_out = vel_out[flow_mask]

    radius = radius[flow_mask]

    BIC_outflow = BIC_outflow[flow_mask]
    BIC_no_outflow = BIC_no_outflow[flow_mask]
    vel_disp = vel_disp[flow_mask]

    #create BIC diff
    BIC_diff = BIC_outflow - BIC_no_outflow
    BIC_diff_strong = (BIC_diff < -50)

    #physical limits mask -
    #for the radius mask 6.1" is the 90% radius
    #also mask out the fits which lie on the lower limit of dispersion < 51km/s
    physical_mask = (radius < 6.1) & (vel_disp>51)

    #calculate Chen et al. trend
    sfr_surface_density_chen, v_out_chen = pf.chen_et_al_2010(sig_sfr.min(), sig_sfr.max(), scale_factor=np.nanmedian(vel_out[BIC_diff_strong])/(np.nanmedian(sig_sfr[BIC_diff_strong])**0.1))

    #divide the mass loading factor by the model
    vout_model = np.full_like(vel_out, np.nan, dtype=np.double)

    for i in np.arange(vel_out.shape[0]):
        #calculate the expected outflow velocity at each sigma_sfr
        sigma_sfr_model, vout_expected = pf.chen_et_al_2010(sig_sfr[i], sig_sfr[i], scale_factor=np.nanmedian(vel_out[BIC_diff_strong])/(np.nanmedian(sig_sfr[BIC_diff_strong])**0.1))
        sigma_sfr_model = sigma_sfr_model[0]
        vout_expected = vout_expected[0]
        #divide the mlf by the expected mlf
        if compare == 'divide':
            vout_model[i] = np.log10(vel_out[i]/vout_expected)
        elif compare == 'subtract':
            vout_model[i] = vel_out[i] - vout_expected


    #do the calculations for all the bins
    num_bins = 5
    min_bin = None #-0.05
    max_bin = None #0.6

    bin_center_all, vout_bin_medians_all, vout_bin_lower_q_all, vout_bin_upper_q_all = pf.binned_median_quantile_lin(radius, vout_model, num_bins=num_bins, weights=None, min_bin=min_bin, max_bin=max_bin)

    bin_center_physical, vout_bin_medians_physical, vout_bin_lower_q_physical, vout_bin_upper_q_physical = pf.binned_median_quantile_lin(radius[physical_mask], vout_model[physical_mask], num_bins=num_bins, weights=None, min_bin=min_bin, max_bin=max_bin)

    bin_center_strong, vout_bin_medians_strong, vout_bin_lower_q_strong, vout_bin_upper_q_strong = pf.binned_median_quantile_lin(radius[BIC_diff_strong], vout_model[BIC_diff_strong], num_bins=num_bins, weights=None, min_bin=min_bin, max_bin=max_bin)


    #calculate the r value for the median values
    r_vout_med_all, p_value_vout_all = pf.pearson_correlation(bin_center_all, vout_bin_medians_all)
    r_vout_med_physical, p_value_vout_physical = pf.pearson_correlation(bin_center_physical, vout_bin_medians_physical)
    r_vout_med_strong, p_value_vout_strong = pf.pearson_correlation(bin_center_strong, vout_bin_medians_strong)

    #calculate the r value for all the values
    r_vout_all, p_value_vout_all = pf.pearson_correlation(radius[~np.isnan(vout_model)], vout_model[~np.isnan(vout_model)])
    r_vout_physical, p_value_vout_physical = pf.pearson_correlation(radius[~np.isnan(vout_model)&physical_mask], vout_model[~np.isnan(vout_model)&physical_mask])
    r_vout_strong, p_value_vout_strong = pf.pearson_correlation(radius[~np.isnan(vout_model)&BIC_diff_strong], vout_model[~np.isnan(vout_model)&BIC_diff_strong])



    #print average numbers for the different panels
    print('Number of spaxels in the first panel', vel_out.shape)
    print('All spaxels median out vel:', np.nanmedian(vel_out))
    print('All spaxels standard deviation out vel:', np.nanstd(vel_out))
    print('')

    print('All spaxels median vout/model:', np.nanmedian(vout_model))
    print('All spaxels standard deviation vout/model:', np.nanstd(vout_model))
    print('')

    print('Number of spaxels with broad sigmas at the instrument dispersion:', vel_out[vel_disp<=51].shape)
    print('')
    print('Number of spaxels beyond R_90:', vel_out[radius>6.1].shape)
    print('')
    print('Number of spaxels in the middle panel:', vel_out[physical_mask].shape)
    print('')

    print('Physical spaxels median vout:', np.nanmedian(vel_out[physical_mask]))
    print('Physical spaxels standard deviation vout:', np.nanstd(vel_out[physical_mask]))
    print('')

    print('Physical spaxels median vout/model:', np.nanmedian(vout_model[physical_mask]))
    print('Physical spaxels standard deviation vout/model:', np.nanstd(vout_model[physical_mask]))
    print('')

    print('Number of spaxels with strong BIC differences:', vel_out[BIC_diff_strong].shape)
    print('')

    print('Clean spaxels median vout:', np.nanmedian(vel_out[BIC_diff_strong]))
    print('Clean spaxels standard deviation vout:', np.nanstd(vel_out[BIC_diff_strong]))
    print('')

    print('Clean spaxels median vout/model:', np.nanmedian(vout_model[BIC_diff_strong]))
    print('Clean spaxels standard deviation vout/model:', np.nanstd(vout_model[BIC_diff_strong]))
    print('')




    #-------
    #plot it
    #-------
    plt.rcParams.update(pf.get_rc_params())
    fig, ax = plt.subplots(nrows=1, ncols=3, sharex=True, sharey='row', figsize=(10,4), constrained_layout=True)

    #get colours from cmasher
    colours = cmr.take_cmap_colors('cmr.gem', 3, cmap_range=(0.25, 0.85), return_fmt='hex')

    #plot all points
    ax[0].fill_between(bin_center_all, vout_bin_lower_q_all, vout_bin_upper_q_all, color=colours[0], alpha=0.3)
    ax[0].scatter(radius, vout_model, marker='o', s=10, label='All KOFFEE fits; R={:.2f}'.format(r_vout_all), c=colours[0], alpha=0.8)
    ax[0].plot(bin_center_all, vout_bin_medians_all, marker='', lw=3, label='Median all KOFFEE fits; R={:.2f}'.format(r_vout_med_all), color=colours[0])

    lgnd = ax[0].legend(frameon=True, fontsize='small', loc='upper right', framealpha=0.5)
    lgnd.legendHandles[0]._legmarker.set_markersize(3)
    if compare == 'divide':
        ax[0].set_ylabel(r'log($v_{out}$/model)')
    elif compare == 'subtract':
        ax[0].set_ylabel(r'$v_{out}$-model')
    ax[0].set_xlabel('Radius (Arcseconds)')
    ax[0].set_title('all spaxels')

    #plot points within 90% radius
    ax[1].fill_between(bin_center_physical, vout_bin_lower_q_physical, vout_bin_upper_q_physical, color=colours[1], alpha=0.3)
    ax[1].scatter(radius[radius>6.1], vout_model[radius>6.1], marker='o', s=10, label='All KOFFEE fits', edgecolors=colours[0], alpha=0.3, facecolors='none')
    ax[1].scatter(radius[vel_disp<=51], vout_model[vel_disp<=51], marker='v', s=10, edgecolors=colours[0], alpha=0.3, facecolors='none')
    ax[1].scatter(radius[physical_mask], vout_model[physical_mask], marker='o', s=10, label='Selected KOFFEE fits; R={:.2f}'.format(r_vout_physical), color=colours[1], alpha=0.8)
    ax[1].plot(bin_center_physical, vout_bin_medians_physical, marker='', lw=3, label='Median of selected KOFFEE fits; R={:.2f}'.format(r_vout_med_physical), color=colours[1])

    lgnd = ax[1].legend(frameon=True, fontsize='small', loc='upper right', framealpha=0.5)
    lgnd.legendHandles[0]._legmarker.set_markersize(3)
    ax[1].set_xlabel('Radius (Arcseconds)')
    ax[1].set_title(r'$r$<$r_{90}$ and $\sigma_{broad}$>$\sigma_{inst}$')

    #plot points with strong BIC values
    ax[2].fill_between(bin_center_strong, vout_bin_lower_q_strong, vout_bin_upper_q_strong, color=colours[2], alpha=0.3)
    ax[2].scatter(radius[~BIC_diff_strong], vout_model[~BIC_diff_strong], marker='o', s=10, label='All KOFFEE fits', color=colours[0], alpha=0.3, facecolors='none')
    ax[2].scatter(radius[BIC_diff_strong], vout_model[BIC_diff_strong], marker='o', s=10, label='Selected KOFFEE fits; R={:.2f}'.format(r_vout_strong), color=colours[2], alpha=1.0)
    ax[2].plot(bin_center_strong, vout_bin_medians_strong, marker='', lw=3, label='Median of selected KOFFEE fits; R={:.2f}'.format(r_vout_med_strong), color=colours[2])

    lgnd = ax[2].legend(frameon=True, fontsize='small', loc='upper right', framealpha=0.5)
    lgnd.legendHandles[0]._legmarker.set_markersize(3)
    ax[2].set_xlabel('Radius (Arcseconds)')
    ax[2].set_title('strongly likely BIC')

    plt.show()


def plot_out_vel_model_sigsfr(OIII_outflow_results, OIII_outflow_error, hbeta_outflow_results, hbeta_outflow_error, hbeta_no_outflow_results, hbeta_no_outflow_error, BIC_outflow, BIC_no_outflow, statistical_results, z, radius, compare='divide'):
    """
    Plots the outflow velocity values found with KOFFEE either divided by or
    subtracted from the expected values calculated with the model from
    Chen et al. 2010 against the SFR surface density calculated using the narrow
    line flux

    Parameters
    ----------
    OIII_outflow_results : :obj:'~numpy.ndarray'
        array of outflow results from KOFFEE for OIII line.  Used to calculate the outflow velocity.  Should be (7, statistical_results.shape)

    OIII_outflow_err : :obj:'~numpy.ndarray'
        array of the outflow result errors from KOFFEE for OIII line

    hbeta_outflow_results : :obj:'~numpy.ndarray'
        array of outflow results from KOFFEE for Hbeta line.  Used to calculate the Sigma SFR.  Should be (7, statistical_results.shape)

    hbeta_outflow_err : :obj:'~numpy.ndarray'
        array of the outflow result errors from KOFFEE for Hbeta line

    hbeta_no_outflow_results : :obj:'~numpy.ndarray'
        array of single gaussian results from KOFFEE for Hbeta line.  Used to calculate the Sigma SFR.  Should be (4, statistical_results.shape)

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

    compare : string
        string defining how to compare the results to the model.  Can be 'divide'
        or 'subtract' (Default='divide')

    Returns
    -------
    A graph of the outflow velocity compared to the expected model against SFR
    surface density
    """
    #calculate the sfr surface density - using just the systemic line, and including the flux line
    #don't include extinction since this was included in the continuum subtraction using ppxf
    sfr, sfr_err, total_sfr, sfr_surface_density, sfr_surface_density_err = calc_sfr.calc_sfr_koffee(hbeta_outflow_results, hbeta_outflow_error, hbeta_no_outflow_results, hbeta_no_outflow_error, statistical_results, z, include_extinction=False, include_outflow=False)

    #calculate the velocity dispersion for the masking
    vel_disp, vel_disp_err, vel_diff, vel_diff_err, vel_out, vel_out_err = calc_outvel.calc_outflow_vel(OIII_outflow_results, OIII_outflow_error, statistical_results, z)

    #create the flow mask
    flow_mask = (statistical_results>0) #& (np.isnan(hbeta_outflow_results[3,:,:])==False)


    #flatten all the arrays and get rid of extra spaxels
    sig_sfr = sfr_surface_density[flow_mask]
    sig_sfr_err = sfr_surface_density_err[flow_mask]

    vel_out = vel_out[flow_mask]

    radius = radius[flow_mask]

    BIC_outflow = BIC_outflow[flow_mask]
    BIC_no_outflow = BIC_no_outflow[flow_mask]
    vel_disp = vel_disp[flow_mask]

    #create BIC diff
    BIC_diff = BIC_outflow - BIC_no_outflow
    BIC_diff_strong = (BIC_diff < -50)

    #physical limits mask -
    #for the radius mask 6.1" is the 90% radius
    #also mask out the fits which lie on the lower limit of dispersion < 51km/s
    physical_mask = (radius < 6.1) & (vel_disp>51)

    #calculate Chen et al. trend
    sfr_surface_density_chen, v_out_chen = pf.chen_et_al_2010(sig_sfr.min(), sig_sfr.max(), scale_factor=np.nanmedian(vel_out[BIC_diff_strong])/(np.nanmedian(sig_sfr[BIC_diff_strong])**0.1))

    #divide the mass loading factor by the model
    vout_model = np.full_like(vel_out, np.nan, dtype=np.double)

    for i in np.arange(vel_out.shape[0]):
        #calculate the expected outflow velocity at each sigma_sfr
        sigma_sfr_model, vout_expected = pf.chen_et_al_2010(sig_sfr[i], sig_sfr[i], scale_factor=np.nanmedian(vel_out[BIC_diff_strong])/(np.nanmedian(sig_sfr[BIC_diff_strong])**0.1))
        sigma_sfr_model = sigma_sfr_model[0]
        vout_expected = vout_expected[0]
        #divide the mlf by the expected mlf
        if compare == 'divide':
            vout_model[i] = np.log10(vel_out[i]/vout_expected)
        elif compare == 'subtract':
            vout_model[i] = vel_out[i] - vout_expected


    #do the calculations for all the bins
    num_bins = 5
    min_bin = None #-0.05
    max_bin = None #0.6

    bin_center_all, vout_bin_medians_all, vout_bin_lower_q_all, vout_bin_upper_q_all = pf.binned_median_quantile_lin(sig_sfr, vout_model, num_bins=num_bins, weights=None, min_bin=min_bin, max_bin=max_bin)

    bin_center_physical, vout_bin_medians_physical, vout_bin_lower_q_physical, vout_bin_upper_q_physical = pf.binned_median_quantile_lin(sig_sfr[physical_mask], vout_model[physical_mask], num_bins=num_bins, weights=None, min_bin=min_bin, max_bin=max_bin)

    bin_center_strong, vout_bin_medians_strong, vout_bin_lower_q_strong, vout_bin_upper_q_strong = pf.binned_median_quantile_lin(sig_sfr[BIC_diff_strong], vout_model[BIC_diff_strong], num_bins=num_bins, weights=None, min_bin=min_bin, max_bin=max_bin)


    #calculate the r value for the median values
    r_vout_med_all, p_value_vout_all = pf.pearson_correlation(bin_center_all, vout_bin_medians_all)
    r_vout_med_physical, p_value_vout_physical = pf.pearson_correlation(bin_center_physical, vout_bin_medians_physical)
    r_vout_med_strong, p_value_vout_strong = pf.pearson_correlation(bin_center_strong, vout_bin_medians_strong)

    #calculate the r value for all the values
    r_vout_all, p_value_vout_all = pf.pearson_correlation(sig_sfr[~np.isnan(vout_model)], vout_model[~np.isnan(vout_model)])
    r_vout_physical, p_value_vout_physical = pf.pearson_correlation(sig_sfr[~np.isnan(vout_model)&physical_mask], vout_model[~np.isnan(vout_model)&physical_mask])
    r_vout_strong, p_value_vout_strong = pf.pearson_correlation(sig_sfr[~np.isnan(vout_model)&BIC_diff_strong], vout_model[~np.isnan(vout_model)&BIC_diff_strong])



    #print average numbers for the different panels
    print('Number of spaxels in the first panel', vel_out.shape)
    print('All spaxels median out vel:', np.nanmedian(vel_out))
    print('All spaxels standard deviation out vel:', np.nanstd(vel_out))
    print('')

    print('All spaxels median vout/model:', np.nanmedian(vout_model))
    print('All spaxels standard deviation vout/model:', np.nanstd(vout_model))
    print('')

    print('Number of spaxels with broad sigmas at the instrument dispersion:', vel_out[vel_disp<=51].shape)
    print('')
    print('Number of spaxels beyond R_90:', vel_out[radius>6.1].shape)
    print('')
    print('Number of spaxels in the middle panel:', vel_out[physical_mask].shape)
    print('')

    print('Physical spaxels median vout:', np.nanmedian(vel_out[physical_mask]))
    print('Physical spaxels standard deviation vout:', np.nanstd(vel_out[physical_mask]))
    print('')

    print('Physical spaxels median vout/model:', np.nanmedian(vout_model[physical_mask]))
    print('Physical spaxels standard deviation vout/model:', np.nanstd(vout_model[physical_mask]))
    print('')

    print('Number of spaxels with strong BIC differences:', vel_out[BIC_diff_strong].shape)
    print('')

    print('Clean spaxels median vout:', np.nanmedian(vel_out[BIC_diff_strong]))
    print('Clean spaxels standard deviation vout:', np.nanstd(vel_out[BIC_diff_strong]))
    print('')

    print('Clean spaxels median vout/model:', np.nanmedian(vout_model[BIC_diff_strong]))
    print('Clean spaxels standard deviation vout/model:', np.nanstd(vout_model[BIC_diff_strong]))
    print('')




    #-------
    #plot it
    #-------
    plt.rcParams.update(pf.get_rc_params())
    fig, ax = plt.subplots(nrows=1, ncols=3, sharex=True, sharey='row', figsize=(10,4), constrained_layout=True)

    #get colours from cmasher
    colours = cmr.take_cmap_colors('cmr.gem', 3, cmap_range=(0.25, 0.85), return_fmt='hex')

    #plot all points
    ax[0].fill_between(bin_center_all, vout_bin_lower_q_all, vout_bin_upper_q_all, color=colours[0], alpha=0.3)
    ax[0].scatter(sig_sfr, vout_model, marker='o', s=10, label='All KOFFEE fits; R={:.2f}'.format(r_vout_all), c=colours[0], alpha=0.8)
    ax[0].plot(bin_center_all, vout_bin_medians_all, marker='', lw=3, label='Median all KOFFEE fits; R={:.2f}'.format(r_vout_med_all), color=colours[0])

    lgnd = ax[0].legend(frameon=True, fontsize='small', loc='upper right', framealpha=0.5)
    lgnd.legendHandles[0]._legmarker.set_markersize(3)
    if compare == 'divide':
        ax[0].set_ylabel(r'log($v_{out}$/model)')
    elif compare == 'subtract':
        ax[0].set_ylabel(r'$v_{out}$-model')
    ax[0].set_xlabel('$\Sigma_{SFR}$ [M$_\odot$ yr$^{-1}$ kpc$^{-2}$]')
    ax[0].set_title('all spaxels')

    #plot points within 90% radius
    ax[1].fill_between(bin_center_physical, vout_bin_lower_q_physical, vout_bin_upper_q_physical, color=colours[1], alpha=0.3)
    ax[1].scatter(sig_sfr[radius>6.1], vout_model[radius>6.1], marker='o', s=10, label='All KOFFEE fits', edgecolors=colours[0], alpha=0.3, facecolors='none')
    ax[1].scatter(sig_sfr[vel_disp<=51], vout_model[vel_disp<=51], marker='v', s=10, edgecolors=colours[0], alpha=0.3, facecolors='none')
    ax[1].scatter(sig_sfr[physical_mask], vout_model[physical_mask], marker='o', s=10, label='Selected KOFFEE fits; R={:.2f}'.format(r_vout_physical), color=colours[1], alpha=0.8)
    ax[1].plot(bin_center_physical, vout_bin_medians_physical, marker='', lw=3, label='Median of selected KOFFEE fits; R={:.2f}'.format(r_vout_med_physical), color=colours[1])

    lgnd = ax[1].legend(frameon=True, fontsize='small', loc='upper right', framealpha=0.5)
    lgnd.legendHandles[0]._legmarker.set_markersize(3)
    ax[1].set_xlabel('$\Sigma_{SFR}$ [M$_\odot$ yr$^{-1}$ kpc$^{-2}$]')
    ax[1].set_title(r'$r$<$r_{90}$ and $\sigma_{broad}$>$\sigma_{inst}$')

    #plot points with strong BIC values
    ax[2].fill_between(bin_center_strong, vout_bin_lower_q_strong, vout_bin_upper_q_strong, color=colours[2], alpha=0.3)
    ax[2].scatter(sig_sfr[~BIC_diff_strong], vout_model[~BIC_diff_strong], marker='o', s=10, label='All KOFFEE fits', color=colours[0], alpha=0.3, facecolors='none')
    ax[2].scatter(sig_sfr[BIC_diff_strong], vout_model[BIC_diff_strong], marker='o', s=10, label='Selected KOFFEE fits; R={:.2f}'.format(r_vout_strong), color=colours[2], alpha=1.0)
    ax[2].plot(bin_center_strong, vout_bin_medians_strong, marker='', lw=3, label='Median of selected KOFFEE fits; R={:.2f}'.format(r_vout_med_strong), color=colours[2])

    lgnd = ax[2].legend(frameon=True, fontsize='small', loc='upper right', framealpha=0.5)
    lgnd.legendHandles[0]._legmarker.set_markersize(3)
    ax[2].set_xlabel('$\Sigma_{SFR}$ [M$_\odot$ yr$^{-1}$ kpc$^{-2}$]')
    ax[2].set_title('strongly likely BIC')

    plt.show()



def plot_out_vel_mlf(OIII_outflow_results, OIII_outflow_error, hbeta_outflow_results, hbeta_outflow_error, hbeta_no_outflow_results, hbeta_no_outflow_error, BIC_outflow, BIC_no_outflow, statistical_results, z, radius, weighted_average=True):
    """
    Plots the mass loading factor against the outflow velocity.

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

    weighted_average : boolean
        whether or not to take a weighted average using the errors (Default=True)

    Returns
    -------
    A graph of the mass loading factor against the outflow velocity

    """
    #calculate the mass loading factor
    mlf, mlf_max, mlf_min = calc_mlf.calc_mass_loading_factor(OIII_outflow_results, OIII_outflow_error, hbeta_outflow_results, hbeta_outflow_error, hbeta_no_outflow_results, hbeta_no_outflow_error, statistical_results, z)

    #calculate the velocity dispersion for the masking
    vel_disp, vel_disp_err, vel_diff, vel_diff_err, vel_out, vel_out_err = calc_outvel.calc_outflow_vel(OIII_outflow_results, OIII_outflow_error, statistical_results, z)

    #make the mask for the outflow spaxels
    flow_mask = (statistical_results>0)

    #flatten all the arrays and get rid of extra spaxels
    vel_out = vel_out[flow_mask]
    vel_out_err = vel_out_err[flow_mask]
    mlf = mlf[flow_mask]
    mlf_max = mlf_max[flow_mask]
    mlf_min = mlf_min[flow_mask]
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

    #create BIC diff
    BIC_diff = BIC_outflow - BIC_no_outflow
    BIC_diff_strong = (BIC_diff < -50)

    #physical limits mask -
    #for the radius mask 6.1" is the 90% radius
    #also mask out the fits which lie on the lower limit of dispersion < 51km/s
    physical_mask = (radius < 6.1) & (vel_disp>51)

    #make sure none of the errors are nan values
    vel_out_err[np.where(np.isnan(vel_out_err)==True)] = np.nanmedian(vel_out_err)

    #do the calculations for all the bins
    num_bins = 3
    min_bin = None #-0.05
    max_bin = None #0.6

    if weighted_average == False:
        bin_center_all, mlf_bin_medians_all, mlf_bin_lower_q_all, mlf_bin_upper_q_all = pf.binned_median_quantile_lin(vel_out, mlf, num_bins=num_bins, weights=None, min_bin=min_bin, max_bin=max_bin)
        bin_center_physical, mlf_bin_medians_physical, mlf_bin_lower_q_physical, mlf_bin_upper_q_physical = pf.binned_median_quantile_lin(vel_out[physical_mask], mlf[physical_mask], num_bins=num_bins, weights=None, min_bin=min_bin, max_bin=max_bin)
        bin_center_strong, mlf_bin_medians_strong, mlf_bin_lower_q_strong, mlf_bin_upper_q_strong = pf.binned_median_quantile_lin(vel_out[BIC_diff_strong], mlf[BIC_diff_strong], num_bins=num_bins, weights=None, min_bin=min_bin, max_bin=max_bin)


    elif weighted_average == True:
        bin_center_all, mlf_bin_medians_all, mlf_bin_lower_q_all, mlf_bin_upper_q_all = pf.binned_median_quantile_lin(vel_out, mlf, num_bins=num_bins, weights=[vel_out_err], min_bin=min_bin, max_bin=max_bin)
        bin_center_physical, mlf_bin_medians_physical, mlf_bin_lower_q_physical, mlf_bin_upper_q_physical = pf.binned_median_quantile_lin(vel_out[physical_mask], mlf[physical_mask], num_bins=num_bins, weights=[vel_out_err], min_bin=min_bin, max_bin=max_bin)
        bin_center_strong, mlf_bin_medians_strong, mlf_bin_lower_q_strong, mlf_bin_upper_q_strong = pf.binned_median_quantile_lin(vel_out[BIC_diff_strong], mlf[BIC_diff_strong], num_bins=num_bins, weights=[vel_out_err], min_bin=min_bin, max_bin=max_bin)


    #calculate the r value for the median values
    r_mlf_med_all, p_value_mlf_all = pf.pearson_correlation(bin_center_all, mlf_bin_medians_all)
    r_mlf_med_physical, p_value_mlf_physical = pf.pearson_correlation(bin_center_physical, mlf_bin_medians_physical)
    r_mlf_med_strong, p_value_mlf_strong = pf.pearson_correlation(bin_center_strong, mlf_bin_medians_strong)

    #calculate the r value for all the values
    r_mlf_all, p_value_mlf_all = pf.pearson_correlation(vel_out[~np.isnan(mlf)], mlf[~np.isnan(mlf)])
    r_mlf_physical, p_value_mlf_physical = pf.pearson_correlation(vel_out[~np.isnan(mlf)&physical_mask], mlf[~np.isnan(mlf)&physical_mask])
    r_mlf_strong, p_value_mlf_strong = pf.pearson_correlation(vel_out[~np.isnan(mlf)&BIC_diff_strong], mlf[~np.isnan(mlf)&BIC_diff_strong])


    #print average numbers for the different panels
    print('Number of spaxels in the first panel', mlf.shape)
    print('All spaxels median mlf:', np.nanmedian(mlf))
    print('All spaxels standard deviation mlf:', np.nanstd(mlf))
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

    print('Number of spaxels with strong BIC differences:', mlf[BIC_diff_strong].shape)
    print('')

    print('Clean spaxels median mlf:', np.nanmedian(mlf[BIC_diff_strong]))
    print('Clean spaxels standard deviation mlf:', np.nanstd(mlf[BIC_diff_strong]))
    print('')

    #-------
    #plot it
    #-------
    plt.rcParams.update(pf.get_rc_params())
    fig, ax = plt.subplots(nrows=1, ncols=3, sharey=True, sharex=True, figsize=(10,4), constrained_layout=True)

    #get colours from cmasher
    colours = cmr.take_cmap_colors('cmr.gem', 3, cmap_range=(0.25, 0.85), return_fmt='hex')

    #plot all points
    #ax[0].errorbar(vel_out, mlf, xerr=vel_out_err, yerr=[mlf-mlf_min, mlf_max-mlf], fmt='o', ms=3, color=colours[0], alpha=0.6, label='All KOFFEE fits; R={:.2f}'.format(r_mlf_all))
    ax[0].fill_between(bin_center_all, mlf_bin_lower_q_all, mlf_bin_upper_q_all, color=colours[0], alpha=0.3)
    ax[0].scatter(vel_out, mlf, marker='o', s=10, label='All KOFFEE fits; R={:.2f}'.format(r_mlf_all), color=colours[0], alpha=0.8)
    ax[0].plot(bin_center_all, mlf_bin_medians_all, marker='', lw=3, label='Median all KOFFEE fits; R={:.2f}'.format(r_mlf_med_all), color=colours[0])

    ax[0].errorbar(480, np.nanmin(mlf), xerr=np.nanmedian(vel_out_err), yerr=[[np.nanmedian(mlf_err_min)], [np.nanmedian(mlf_err_max)]], c='k')

    #ax[0].set_ylim(100, 500)
    #ax[0].set_xscale('log')
    #ax[0].set_xlim(0.003, 10)
    lgnd = ax[0].legend(frameon=True, fontsize='small', loc='lower left', framealpha=0.5, edgecolor='white')
    lgnd.legendHandles[0]._legmarker.set_markersize(3)
    ax[0].set_ylabel('Log(Mass Loading Factor)')
    ax[0].set_xlabel('Maximum Outflow Velocity [km s$^{-1}$]')
    ax[0].set_title('all spaxels')

    #plot points within 90% radius
    ax[1].fill_between(bin_center_physical, mlf_bin_lower_q_physical, mlf_bin_upper_q_physical, color=colours[1], alpha=0.3)
    ax[1].scatter(vel_out[radius>6.1], mlf[radius>6.1], marker='o', s=10, label='All KOFFEE fits', edgecolors=colours[0], alpha=0.3, facecolors='none')
    ax[1].scatter(vel_out[vel_disp<=51], mlf[vel_disp<=51], marker='v', s=10, edgecolors=colours[0], alpha=0.3, facecolors='none')
    ax[1].scatter(vel_out[physical_mask], mlf[physical_mask], marker='o', s=10, label='Selected KOFFEE fits; R={:.2f}'.format(r_mlf_physical), color=colours[1], alpha=0.8)
    ax[1].plot(bin_center_physical, mlf_bin_medians_physical, marker='', lw=3, label='Median of selected KOFFEE fits;\n R={:.2f}'.format(r_mlf_med_physical), color=colours[1])

    ax[1].errorbar(480, np.nanmin(mlf), xerr=np.nanmedian(vel_out_err[physical_mask]), yerr=[[np.nanmedian(mlf_err_min[physical_mask])], [np.nanmedian(mlf_err_max[physical_mask])]], c='k')

    #ax[1].set_xscale('log')
    lgnd = ax[1].legend(frameon=True, fontsize='small', loc='lower left', framealpha=0.5, edgecolor='white')
    lgnd.legendHandles[0]._legmarker.set_markersize(3)
    ax[1].set_xlabel('Maximum Outflow Velocity [km s$^{-1}$]')
    ax[1].set_title(r'$r$<$r_{90}$ and $\sigma_{broad}$>$\sigma_{inst}$')

    #plot points with strong BIC values
    ax[2].fill_between(bin_center_strong, mlf_bin_lower_q_strong, mlf_bin_upper_q_strong, color=colours[2], alpha=0.3)
    ax[2].scatter(vel_out[~BIC_diff_strong], mlf[~BIC_diff_strong], marker='o', s=10, label='All KOFFEE fits', color=colours[0], alpha=0.3, facecolors='none')
    ax[2].scatter(vel_out[BIC_diff_strong], mlf[BIC_diff_strong], marker='o', s=10, label='Selected KOFFEE fits; R={:.2f}'.format(r_mlf_strong), color=colours[2], alpha=1.0)
    ax[2].plot(bin_center_strong, mlf_bin_medians_strong, marker='', lw=3, label='Median of selected KOFFEE fits;\n R={:.2f}'.format(r_mlf_med_strong), color=colours[2])

    ax[2].errorbar(480, np.nanmin(mlf), xerr=np.nanmedian(vel_out_err[BIC_diff_strong]), yerr=[[np.nanmedian(mlf_err_min[BIC_diff_strong])], [np.nanmedian(mlf_err_max[BIC_diff_strong])]], c='k')

    #ax[1].set_xscale('log')
    lgnd = ax[2].legend(frameon=True, fontsize='small', loc='lower left', framealpha=0.5, edgecolor='white')
    lgnd.legendHandles[0]._legmarker.set_markersize(3)
    ax[2].set_xlabel('Maximum Outflow Velocity [km s$^{-1}$]')
    ax[2].set_title('strongly likely BIC')

    plt.show()



def plot_out_vel_disp_mlf(OIII_outflow_results, OIII_outflow_error, hbeta_outflow_results, hbeta_outflow_error, hbeta_no_outflow_results, hbeta_no_outflow_error, BIC_outflow, BIC_no_outflow, statistical_results, z, radius, weighted_average=True):
    """
    Plots the mass loading factor against the outflow velocity dispersion.

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

    weighted_average : boolean
        whether or not to take a weighted average using the errors (Default=True)

    Returns
    -------
    A graph of the mass loading factor against the outflow velocity dispersion

    """
    #calculate the mass loading factor
    mlf, mlf_max, mlf_min = calc_mlf.calc_mass_loading_factor(OIII_outflow_results, OIII_outflow_error, hbeta_outflow_results, hbeta_outflow_error, hbeta_no_outflow_results, hbeta_no_outflow_error, statistical_results, z)

    #calculate the velocity dispersion for the masking
    vel_disp, vel_disp_err, vel_diff, vel_diff_err, vel_out, vel_out_err = calc_outvel.calc_outflow_vel(OIII_outflow_results, OIII_outflow_error, statistical_results, z)

    #make the mask for the outflow spaxels
    flow_mask = (statistical_results>0)

    #flatten all the arrays and get rid of extra spaxels
    vel_disp = vel_disp[flow_mask]
    vel_disp_err = vel_disp_err[flow_mask]
    mlf = mlf[flow_mask]
    mlf_max = mlf_max[flow_mask]
    mlf_min = mlf_min[flow_mask]
    BIC_outflow = BIC_outflow[flow_mask]
    BIC_no_outflow = BIC_no_outflow[flow_mask]
    radius = radius[flow_mask]

    #take the log of the mlf
    mlf = np.log10(mlf)
    mlf_max = np.log10(mlf_max)
    mlf_min = np.log10(mlf_min)

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

    #make sure none of the errors are nan values
    vel_disp_err[np.where(np.isnan(vel_disp_err)==True)] = np.nanmedian(vel_disp_err)

    #do the calculations for all the bins
    num_bins = 4
    min_bin = None #-0.05
    max_bin = None #0.6

    if weighted_average == False:
        bin_center_all, mlf_bin_medians_all, mlf_bin_lower_q_all, mlf_bin_upper_q_all = pf.binned_median_quantile_lin(vel_disp, mlf, num_bins=num_bins, weights=None, min_bin=min_bin, max_bin=max_bin)
        bin_center_physical, mlf_bin_medians_physical, mlf_bin_lower_q_physical, mlf_bin_upper_q_physical = pf.binned_median_quantile_lin(vel_disp[physical_mask], mlf[physical_mask], num_bins=num_bins, weights=None, min_bin=min_bin, max_bin=max_bin)
        bin_center_strong, mlf_bin_medians_strong, mlf_bin_lower_q_strong, mlf_bin_upper_q_strong = pf.binned_median_quantile_lin(vel_disp[BIC_diff_strong], mlf[BIC_diff_strong], num_bins=num_bins, weights=None, min_bin=min_bin, max_bin=max_bin)


    elif weighted_average == True:
        bin_center_all, mlf_bin_medians_all, mlf_bin_lower_q_all, mlf_bin_upper_q_all = pf.binned_median_quantile_lin(vel_disp, mlf, num_bins=num_bins, weights=[vel_out_err], min_bin=min_bin, max_bin=max_bin)
        bin_center_physical, mlf_bin_medians_physical, mlf_bin_lower_q_physical, mlf_bin_upper_q_physical = pf.binned_median_quantile_lin(vel_disp[physical_mask], mlf[physical_mask], num_bins=num_bins, weights=[vel_out_err], min_bin=min_bin, max_bin=max_bin)
        bin_center_strong, mlf_bin_medians_strong, mlf_bin_lower_q_strong, mlf_bin_upper_q_strong = pf.binned_median_quantile_lin(vel_disp[BIC_diff_strong], mlf[BIC_diff_strong], num_bins=num_bins, weights=[vel_out_err], min_bin=min_bin, max_bin=max_bin)


    #calculate the r value for the median values
    r_mlf_med_all, p_value_mlf_all = pf.pearson_correlation(bin_center_all, mlf_bin_medians_all)
    r_mlf_med_physical, p_value_mlf_physical = pf.pearson_correlation(bin_center_physical, mlf_bin_medians_physical)
    r_mlf_med_strong, p_value_mlf_strong = pf.pearson_correlation(bin_center_strong, mlf_bin_medians_strong)

    #calculate the r value for all the values
    r_mlf_all, p_value_mlf_all = pf.pearson_correlation(vel_disp[~np.isnan(mlf)], mlf[~np.isnan(mlf)])
    r_mlf_physical, p_value_mlf_physical = pf.pearson_correlation(vel_disp[~np.isnan(mlf)&physical_mask], mlf[~np.isnan(mlf)&physical_mask])
    r_mlf_strong, p_value_mlf_strong = pf.pearson_correlation(vel_disp[~np.isnan(mlf)&BIC_diff_strong], mlf[~np.isnan(mlf)&BIC_diff_strong])


    #print average numbers for the different panels
    print('Number of spaxels in the first panel', mlf.shape)
    print('All spaxels median mlf:', np.nanmedian(mlf))
    print('All spaxels standard deviation mlf:', np.nanstd(mlf))
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

    print('Number of spaxels with strong BIC differences:', mlf[BIC_diff_strong].shape)
    print('')

    print('Clean spaxels median mlf:', np.nanmedian(mlf[BIC_diff_strong]))
    print('Clean spaxels standard deviation mlf:', np.nanstd(mlf[BIC_diff_strong]))
    print('')

    #-------
    #plot it
    #-------
    plt.rcParams.update(pf.get_rc_params())
    fig, ax = plt.subplots(nrows=1, ncols=3, sharey=True, sharex=True, figsize=(10,4), constrained_layout=True)

    #get colours from cmasher
    colours = cmr.take_cmap_colors('cmr.gem', 3, cmap_range=(0.25, 0.85), return_fmt='hex')

    #plot all points
    #ax[0].errorbar(vel_out, mlf, xerr=vel_out_err, yerr=[mlf-mlf_min, mlf_max-mlf], fmt='o', ms=3, color=colours[0], alpha=0.6, label='All KOFFEE fits; R={:.2f}'.format(r_mlf_all))
    ax[0].fill_between(bin_center_all, mlf_bin_lower_q_all, mlf_bin_upper_q_all, color=colours[0], alpha=0.3)
    ax[0].scatter(vel_disp, mlf, marker='o', s=10, label='All KOFFEE fits; R={:.2f}'.format(r_mlf_all), color=colours[0], alpha=0.8)
    ax[0].plot(bin_center_all, mlf_bin_medians_all, marker='', lw=3, label='Median all KOFFEE fits; R={:.2f}'.format(r_mlf_med_all), color=colours[0])

    ax[0].errorbar(230, np.nanmin(mlf), xerr=np.nanmedian(vel_disp_err), yerr=[[np.nanmedian(mlf_err_min)], [np.nanmedian(mlf_err_max)]], c='k')

    #ax[0].set_ylim(100, 500)
    #ax[0].set_xscale('log')
    #ax[0].set_xlim(0.003, 10)
    lgnd = ax[0].legend(frameon=True, fontsize='small', loc='lower left', framealpha=0.5, edgecolor='white')
    lgnd.legendHandles[0]._legmarker.set_markersize(3)
    ax[0].set_ylabel('Log(Mass Loading Factor)')
    ax[0].set_xlabel('Outflow Velocity Dispersion [km s$^{-1}$]')
    ax[0].set_title('all spaxels')

    #plot points within 90% radius
    ax[1].fill_between(bin_center_physical, mlf_bin_lower_q_physical, mlf_bin_upper_q_physical, color=colours[1], alpha=0.3)
    ax[1].scatter(vel_disp[radius>6.1], mlf[radius>6.1], marker='o', s=10, label='All KOFFEE fits', edgecolors=colours[0], alpha=0.3, facecolors='none')
    ax[1].scatter(vel_disp[vel_disp<=51], mlf[vel_disp<=51], marker='v', s=10, edgecolors=colours[0], alpha=0.3, facecolors='none')
    ax[1].scatter(vel_disp[physical_mask], mlf[physical_mask], marker='o', s=10, label='Selected KOFFEE fits; R={:.2f}'.format(r_mlf_physical), color=colours[1], alpha=0.8)
    ax[1].plot(bin_center_physical, mlf_bin_medians_physical, marker='', lw=3, label='Median of selected KOFFEE fits;\n R={:.2f}'.format(r_mlf_med_physical), color=colours[1])

    ax[1].errorbar(230, np.nanmin(mlf), xerr=np.nanmedian(vel_disp_err[physical_mask]), yerr=[[np.nanmedian(mlf_err_min[physical_mask])], [np.nanmedian(mlf_err_max[physical_mask])]], c='k')

    #ax[1].set_xscale('log')
    lgnd = ax[1].legend(frameon=True, fontsize='small', loc='lower left', framealpha=0.5, edgecolor='white')
    lgnd.legendHandles[0]._legmarker.set_markersize(3)
    ax[1].set_xlabel('Outflow Velocity Dispersion [km s$^{-1}$]')
    ax[1].set_title(r'$r$<$r_{90}$ and $\sigma_{broad}$>$\sigma_{inst}$')

    #plot points with strong BIC values
    ax[2].fill_between(bin_center_strong, mlf_bin_lower_q_strong, mlf_bin_upper_q_strong, color=colours[2], alpha=0.3)
    ax[2].scatter(vel_disp[~BIC_diff_strong], mlf[~BIC_diff_strong], marker='o', s=10, label='All KOFFEE fits', color=colours[0], alpha=0.3, facecolors='none')
    ax[2].scatter(vel_disp[BIC_diff_strong], mlf[BIC_diff_strong], marker='o', s=10, label='Selected KOFFEE fits; R={:.2f}'.format(r_mlf_strong), color=colours[2], alpha=1.0)
    ax[2].plot(bin_center_strong, mlf_bin_medians_strong, marker='', lw=3, label='Median of selected KOFFEE fits;\n R={:.2f}'.format(r_mlf_med_strong), color=colours[2])

    ax[2].errorbar(230, np.nanmin(mlf), xerr=np.nanmedian(vel_disp_err[BIC_diff_strong]), yerr=[[np.nanmedian(mlf_err_min[BIC_diff_strong])], [np.nanmedian(mlf_err_max[BIC_diff_strong])]], c='k')

    #ax[1].set_xscale('log')
    lgnd = ax[2].legend(frameon=True, fontsize='small', loc='lower left', framealpha=0.5, edgecolor='white')
    lgnd.legendHandles[0]._legmarker.set_markersize(3)
    ax[2].set_xlabel('Outflow Velocity Dispersion [km s$^{-1}$]')
    ax[2].set_title('strongly likely BIC')

    plt.show()



def plot_sfr_vseparate(OIII_outflow_results, OIII_outflow_error, hbeta_outflow_results, hbeta_outflow_error, hbeta_no_outflow_results, hbeta_no_outflow_error, statistical_results, z, colour_by=None, colour_by_array=None, weighted_average=True):
    """
    Plots the SFR surface density against the outflow velocity offset and dispersion,
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

    colour_by : string
        the variable used for colouring the points on the graph, used in the
        plotting labels for the colourbar (Default=None)

    colour_by_array : :obj:'~numpy.ndarray'
        the array used for colouring the points on the graph (Default=None)

    weighted_average : boolean
        whether or not to take a weighted average using the errors (Default=True)

    Returns
    -------
    A two panel graph of velocity offset and velocity dispersion against the
    SFR surface density

    """
    #calculate the outflow velocity
    vel_diff, vel_diff_err, vel_out, vel_out_err = calc_outvel.calc_outflow_vel(OIII_outflow_results, OIII_outflow_error, statistical_results, z)

    #calculate the sfr surface density - using just the systemic line, and including the flux line
    #don't include extinction since this was included in the continuum subtraction using ppxf
    sfr, total_sfr, sfr_surface_density, h_beta_integral_err = calc_sfr.calc_sfr_koffee(hbeta_outflow_results, hbeta_outflow_error, hbeta_no_outflow_results, hbeta_no_outflow_error, statistical_results, z, include_extinction=False, include_outflow=False)

    #get the sfr for the outflow spaxels
    flow_mask = (statistical_results>0)

    #create id array
    """
    id_array = np.arange(radius.shape[0]*radius.shape[1])
    id_array = id_array.reshape(radius.shape)
    x_id = np.tile(np.arange(radius.shape[0]), radius.shape[1]).reshape(radius.shape[1], radius.shape[0]).T
    y_id = np.tile(np.arange(radius.shape[1]), radius.shape[0]).reshape(radius.shape)
    print(id_array)
    print(x_id)
    print(y_id)
    """

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
    if colour_by is not None:
        colour_by_array = colour_by_array[flow_mask]

    #
    colour_by_array = colour_by_array[flow_mask]
    BIC_mask = (colour_by_array<-10)

    """id_array = id_array[flow_mask]
    x_id = x_id[flow_mask]
    y_id = y_id[flow_mask]

    #print the id values where the sigma_sfr is below 0.05 and the vel_disp is above 160
    print('ID values:', id_array[(sfr<0.05)&(vel_disp>160)])
    print('x ID:', x_id[(sfr<0.05)&(vel_disp>160)])
    print('y ID:', y_id[(sfr<0.05)&(vel_disp>160)])
    """

    #make sure none of the errors are nan values
    vel_diff_err[np.where(np.isnan(vel_diff_err)==True)] = np.nanmedian(vel_diff_err)
    vel_disp_err[np.where(np.isnan(vel_disp_err)==True)] = np.nanmedian(vel_disp_err)

    #do the calculations for all the bins
    num_bins = 5
    min_bin = None #-0.05
    max_bin = None #0.6

    if weighted_average == False:
        bin_center, vel_diff_bin_medians, vel_diff_bin_lower_q, vel_diff_bin_upper_q = pf.binned_median_quantile_log(sfr[BIC_mask], vel_diff[BIC_mask], num_bins=num_bins, weights=None, min_bin=min_bin, max_bin=max_bin)
        bin_center, disp_bin_medians, disp_bin_lower_q, disp_bin_upper_q = pf.binned_median_quantile_log(sfr[BIC_mask], vel_disp[BIC_mask], num_bins=num_bins, weights=None, min_bin=min_bin, max_bin=max_bin)

    elif weighted_average == True:
        bin_center, vel_diff_bin_medians, vel_diff_bin_lower_q, vel_diff_bin_upper_q = pf.binned_median_quantile_log(sfr, vel_diff, num_bins=num_bins, weights=[vel_diff_err], min_bin=min_bin, max_bin=max_bin)
        bin_center, disp_bin_medians, disp_bin_lower_q, disp_bin_upper_q = pf.binned_median_quantile_log(sfr, vel_disp, num_bins=num_bins, weights=[vel_disp_err], min_bin=min_bin, max_bin=max_bin)


    #calculate the r value for the median values
    r_vel_diff_med, p_value_v_diff = pf.pearson_correlation(bin_center, vel_diff_bin_medians)
    r_disp_med, p_value_disp = pf.pearson_correlation(bin_center, disp_bin_medians)

    #calculate the r value for all the values
    r_vel_diff, p_value_v_diff = pf.pearson_correlation(sfr, vel_diff)
    r_disp, p_value_disp = pf.pearson_correlation(sfr, vel_disp)

    r_vel_diff_BIC, p_value_v_diff_BIC = pf.pearson_correlation(sfr[BIC_mask], vel_diff[BIC_mask])
    r_disp_BIC, p_value_disp_BIC = pf.pearson_correlation(sfr[BIC_mask], vel_disp[BIC_mask])


    #create vectors to plot the literature trends
    sfr_surface_density_chen, vel_diff_chen = pf.chen_et_al_2010(sfr.min(), sfr.max(), scale_factor=np.nanmedian(vel_diff[BIC_mask])/(np.nanmedian(sfr[BIC_mask])**0.1))
    sfr_surface_density_murray, vel_diff_murray = pf.murray_et_al_2011(sfr.min(), sfr.max(), scale_factor=np.nanmedian(vel_diff[BIC_mask])/(np.nanmedian(sfr[BIC_mask])**2))

    #plot it
    plt.rcParams.update(pf.get_rc_params())
    fig, ax = plt.subplots(nrows=1, ncols=2, sharex=True, figsize=(8,5))

    #----------------
    #Plots
    #-----------------
    if colour_by is not None:
        ax[0].scatter(sfr[vel_disp>=51], vel_diff[vel_disp>=51], marker='o', lw=0, alpha=0.6, label='Flow spaxels; R={:.2f}'.format(r_vel_diff), c=colour_by_array[vel_disp>=51])
        ax[0].scatter(sfr[vel_disp<51], vel_diff[vel_disp<51], marker='v', lw=0, alpha=0.6, c=colour_by_array[vel_disp<51])

        im = ax[1].scatter(sfr[vel_disp>=51], vel_disp[vel_disp>=51], marker='o', lw=0, alpha=0.6, label='Flow spaxels; R={:.2f}'.format(r_disp), c=colour_by_array[vel_disp>=51])
        ax[1].scatter(sfr[vel_disp<51], vel_disp[vel_disp<51], marker='v', lw=0, alpha=0.6, c=colour_by_array[vel_disp<51])
        cbar = plt.colorbar(im, ax=ax[1])
        cbar.ax.set_ylabel(colour_by)

    elif colour_by is None:
        ax[0].plot(sfr[BIC_mask][vel_disp[BIC_mask]>=51], vel_diff[BIC_mask][vel_disp[BIC_mask]>=51], marker='o', lw=0, alpha=0.4, label='Definite Flow spaxels; R={:.2f}'.format(r_vel_diff_BIC), color='tab:blue')
        ax[0].plot(sfr[BIC_mask][vel_disp[BIC_mask]<51], vel_diff[BIC_mask][vel_disp[BIC_mask]<51], marker='v', lw=0, alpha=0.4, color='tab:blue')

        ax[1].plot(sfr[BIC_mask][vel_disp[BIC_mask]>=51], vel_disp[BIC_mask][vel_disp[BIC_mask]>=51], marker='o', lw=0, alpha=0.4, label='Definite Flow spaxels; R={:.2f}'.format(r_disp_BIC), color='tab:blue')
        ax[1].plot(sfr[BIC_mask][vel_disp[BIC_mask]<51], vel_disp[BIC_mask][vel_disp[BIC_mask]<51], marker='v', lw=0, alpha=0.4, color='tab:blue')

        ax[0].plot(sfr[~BIC_mask][vel_disp[~BIC_mask]>=51], vel_diff[~BIC_mask][vel_disp[~BIC_mask]>=51], marker='o', lw=0, alpha=0.4, label='Likely Flow spaxels; R={:.2f}'.format(r_vel_diff), color='tab:pink')
        ax[0].plot(sfr[~BIC_mask][vel_disp[~BIC_mask]<51], vel_diff[~BIC_mask][vel_disp[~BIC_mask]<51], marker='v', lw=0, alpha=0.4, color='tab:pink')

        ax[1].plot(sfr[~BIC_mask][vel_disp[~BIC_mask]>=51], vel_disp[~BIC_mask][vel_disp[~BIC_mask]>=51], marker='o', lw=0, alpha=0.4, label='Likely Flow spaxels; R={:.2f}'.format(r_disp), color='tab:pink')
        ax[1].plot(sfr[~BIC_mask][vel_disp[~BIC_mask]<51], vel_disp[~BIC_mask][vel_disp[~BIC_mask]<51], marker='v', lw=0, alpha=0.4, color='tab:pink')

        #ax[0].errorbar(sfr[vel_disp>=51], vel_diff[vel_disp>=51], xerr=sfr_err[vel_disp>=51], yerr=vel_diff_err[vel_disp>=51], marker='o', lw=0, alpha=0.4, elinewidth=1, label='Flow spaxels; R={:.2f}'.format(r_vel_diff), color='tab:blue')
        #ax[0].errorbar(sfr[vel_disp<51], vel_diff[vel_disp<51], xerr=sfr_err[vel_disp<51], yerr=vel_diff_err[vel_disp<51], marker='v', lw=0, alpha=0.4, elinewidth=1, color='tab:blue')

        #ax[1].errorbar(sfr[vel_disp>=51], vel_disp[vel_disp>=51], xerr=sfr_err[vel_disp>=51], yerr=vel_disp_err[vel_disp>=51], marker='o', lw=0, alpha=0.4, elinewidth=1, label='Flow spaxels; R={:.2f}'.format(r_disp), color='tab:blue')
        #ax[1].errorbar(sfr[vel_disp<51], vel_disp[vel_disp<51], xerr=sfr_err[vel_disp<51], yerr=vel_disp_err[vel_disp<51], marker='v', lw=0, alpha=0.4, elinewidth=1, color='tab:blue')


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



def plot_vel_diff_mlf(OIII_outflow_results, OIII_outflow_error, hbeta_outflow_results, hbeta_outflow_error, hbeta_no_outflow_results, hbeta_no_outflow_error, BIC_outflow, BIC_no_outflow, statistical_results, z, radius, weighted_average=True):
    """
    Plots the mass loading factor against the velocity difference.

    Parameters
    ----------
    OIII_outflow_results : :obj:'~numpy.ndarray'
        array of outflow results from KOFFEE for OIII line.  Used to calculate the outflow velocity.  Should be (7, statistical_results.shape)

    OIII_outflow_err : :obj:'~numpy.ndarray'
        array of the outflow result errors from KOFFEE for OIII line

    hbeta_outflow_results : :obj:'~numpy.ndarray'
        array of outflow results from KOFFEE for Hbeta line.  Used to calculate the Sigma SFR.  Should be (7, statistical_results.shape)

    hbeta_outflow_err : :obj:'~numpy.ndarray'
        array of the outflow result errors from KOFFEE for Hbeta line

    hbeta_no_outflow_results : :obj:'~numpy.ndarray'
        array of single gaussian results from KOFFEE for Hbeta line.  Used to calculate the Sigma SFR.  Should be (4, statistical_results.shape)

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

    weighted_average : boolean
        whether or not to take a weighted average using the errors (Default=True)

    Returns
    -------
    A graph of the mass loading factor against the outflow velocity offset

    """
    #calculate the mass loading factor
    mlf, mlf_max, mlf_min = calc_mlf.calc_mass_loading_factor(OIII_outflow_results, OIII_outflow_error, hbeta_outflow_results, hbeta_outflow_error, hbeta_no_outflow_results, hbeta_no_outflow_error, statistical_results, z)

    #calculate the velocity dispersion for the masking
    vel_disp, vel_disp_err, vel_diff, vel_diff_err, vel_out, vel_out_err = calc_outvel.calc_outflow_vel(OIII_outflow_results, OIII_outflow_error, statistical_results, z)

    #make the mask for the outflow spaxels
    flow_mask = (statistical_results>0)

    #flatten all the arrays and get rid of extra spaxels
    vel_diff = vel_diff[flow_mask]
    vel_diff_err = vel_diff_err[flow_mask]
    mlf = mlf[flow_mask]
    mlf_max = mlf_max[flow_mask]
    mlf_min = mlf_min[flow_mask]
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

    #create BIC diff
    BIC_diff = BIC_outflow - BIC_no_outflow
    BIC_diff_strong = (BIC_diff < -50)

    #physical limits mask -
    #for the radius mask 6.1" is the 90% radius
    #also mask out the fits which lie on the lower limit of dispersion < 51km/s
    physical_mask = (radius < 6.1) & (vel_disp>51)

    #make sure none of the errors are nan values
    vel_diff_err[np.where(np.isnan(vel_diff_err)==True)] = np.nanmedian(vel_diff_err)

    #do the calculations for all the bins
    num_bins = 4
    min_bin = None #-0.05
    max_bin = None #0.6

    if weighted_average == False:
        bin_center_all, mlf_bin_medians_all, mlf_bin_lower_q_all, mlf_bin_upper_q_all = pf.binned_median_quantile_lin(vel_diff, mlf, num_bins=num_bins, weights=None, min_bin=min_bin, max_bin=max_bin)
        bin_center_physical, mlf_bin_medians_physical, mlf_bin_lower_q_physical, mlf_bin_upper_q_physical = pf.binned_median_quantile_lin(vel_diff[physical_mask], mlf[physical_mask], num_bins=num_bins, weights=None, min_bin=min_bin, max_bin=max_bin)
        bin_center_strong, mlf_bin_medians_strong, mlf_bin_lower_q_strong, mlf_bin_upper_q_strong = pf.binned_median_quantile_lin(vel_diff[BIC_diff_strong], mlf[BIC_diff_strong], num_bins=num_bins, weights=None, min_bin=min_bin, max_bin=max_bin)


    elif weighted_average == True:
        bin_center_all, mlf_bin_medians_all, mlf_bin_lower_q_all, mlf_bin_upper_q_all = pf.binned_median_quantile_lin(vel_diff, mlf, num_bins=num_bins, weights=[vel_out_err], min_bin=min_bin, max_bin=max_bin)
        bin_center_physical, mlf_bin_medians_physical, mlf_bin_lower_q_physical, mlf_bin_upper_q_physical = pf.binned_median_quantile_lin(vel_diff[physical_mask], mlf[physical_mask], num_bins=num_bins, weights=[vel_out_err], min_bin=min_bin, max_bin=max_bin)
        bin_center_strong, mlf_bin_medians_strong, mlf_bin_lower_q_strong, mlf_bin_upper_q_strong = pf.binned_median_quantile_lin(vel_diff[BIC_diff_strong], mlf[BIC_diff_strong], num_bins=num_bins, weights=[vel_out_err], min_bin=min_bin, max_bin=max_bin)


    #calculate the r value for the median values
    r_mlf_med_all, p_value_mlf_all = pf.pearson_correlation(bin_center_all, mlf_bin_medians_all)
    r_mlf_med_physical, p_value_mlf_physical = pf.pearson_correlation(bin_center_physical, mlf_bin_medians_physical)
    r_mlf_med_strong, p_value_mlf_strong = pf.pearson_correlation(bin_center_strong, mlf_bin_medians_strong)

    #calculate the r value for all the values
    r_mlf_all, p_value_mlf_all = pf.pearson_correlation(vel_diff[~np.isnan(mlf)], mlf[~np.isnan(mlf)])
    r_mlf_physical, p_value_mlf_physical = pf.pearson_correlation(vel_diff[~np.isnan(mlf)&physical_mask], mlf[~np.isnan(mlf)&physical_mask])
    r_mlf_strong, p_value_mlf_strong = pf.pearson_correlation(vel_diff[~np.isnan(mlf)&BIC_diff_strong], mlf[~np.isnan(mlf)&BIC_diff_strong])


    #print average numbers for the different panels
    print('Number of spaxels in the first panel', mlf.shape)
    print('All spaxels median mlf:', np.nanmedian(mlf))
    print('All spaxels standard deviation mlf:', np.nanstd(mlf))
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

    print('Number of spaxels with strong BIC differences:', mlf[BIC_diff_strong].shape)
    print('')

    print('Clean spaxels median mlf:', np.nanmedian(mlf[BIC_diff_strong]))
    print('Clean spaxels standard deviation mlf:', np.nanstd(mlf[BIC_diff_strong]))
    print('')

    #-------
    #plot it
    #-------
    plt.rcParams.update(pf.get_rc_params())
    fig, ax = plt.subplots(nrows=1, ncols=3, sharey=True, sharex=True, figsize=(10,4), constrained_layout=True)

    #get colours from cmasher
    colours = cmr.take_cmap_colors('cmr.gem', 3, cmap_range=(0.25, 0.85), return_fmt='hex')

    #plot all points
    #ax[0].errorbar(vel_diff, mlf, xerr=vel_diff_err, yerr=[mlf-mlf_min, mlf_max-mlf], fmt='o', ms=3, color=colours[0], alpha=0.6, label='All KOFFEE fits; R={:.2f}'.format(r_mlf_all))
    ax[0].fill_between(bin_center_all, mlf_bin_lower_q_all, mlf_bin_upper_q_all, color=colours[0], alpha=0.3)
    ax[0].scatter(vel_diff, mlf, marker='o', s=10, label='All KOFFEE fits; R={:.2f}'.format(r_mlf_all), color=colours[0], alpha=0.8)
    ax[0].plot(bin_center_all, mlf_bin_medians_all, marker='', lw=3, label='Median all KOFFEE fits; R={:.2f}'.format(r_mlf_med_all), color=colours[0])

    ax[0].errorbar(230, np.nanmin(mlf), xerr=np.nanmedian(vel_diff_err), yerr=[[np.nanmedian(mlf_err_min)], [np.nanmedian(mlf_err_max)]], c='k')

    #ax[0].set_ylim(100, 500)
    #ax[0].set_xscale('log')
    #ax[0].set_xlim(0.003, 10)
    lgnd = ax[0].legend(frameon=True, fontsize='small', loc='lower left', framealpha=0.5, edgecolor='white')
    lgnd.legendHandles[0]._legmarker.set_markersize(3)
    ax[0].set_ylabel('Log(Mass Loading Factor)')
    ax[0].set_xlabel('Velocity Difference [km s$^{-1}$]')
    ax[0].set_title('all spaxels')

    #plot points within 90% radius
    ax[1].fill_between(bin_center_physical, mlf_bin_lower_q_physical, mlf_bin_upper_q_physical, color=colours[1], alpha=0.3)
    ax[1].scatter(vel_diff[radius>6.1], mlf[radius>6.1], marker='o', s=10, label='All KOFFEE fits', edgecolors=colours[0], alpha=0.3, facecolors='none')
    ax[1].scatter(vel_diff[vel_disp<=51], mlf[vel_disp<=51], marker='v', s=10, edgecolors=colours[0], alpha=0.3, facecolors='none')
    ax[1].scatter(vel_diff[physical_mask], mlf[physical_mask], marker='o', s=10, label='Selected KOFFEE fits; R={:.2f}'.format(r_mlf_physical), color=colours[1], alpha=0.8)
    ax[1].plot(bin_center_physical, mlf_bin_medians_physical, marker='', lw=3, label='Median of selected KOFFEE fits;\n R={:.2f}'.format(r_mlf_med_physical), color=colours[1])

    ax[1].errorbar(230, np.nanmin(mlf), xerr=np.nanmedian(vel_diff_err[physical_mask]), yerr=[[np.nanmedian(mlf_err_min[physical_mask])], [np.nanmedian(mlf_err_max[physical_mask])]], c='k')

    #ax[1].set_xscale('log')
    lgnd = ax[1].legend(frameon=True, fontsize='small', loc='lower left', framealpha=0.5, edgecolor='white')
    lgnd.legendHandles[0]._legmarker.set_markersize(3)
    ax[1].set_xlabel('Velocity Difference [km s$^{-1}$]')
    ax[1].set_title(r'$r$<$r_{90}$ and $\sigma_{broad}$>$\sigma_{inst}$')

    #plot points with strong BIC values
    ax[2].fill_between(bin_center_strong, mlf_bin_lower_q_strong, mlf_bin_upper_q_strong, color=colours[2], alpha=0.3)
    ax[2].scatter(vel_diff[~BIC_diff_strong], mlf[~BIC_diff_strong], marker='o', s=10, label='All KOFFEE fits', color=colours[0], alpha=0.3, facecolors='none')
    ax[2].scatter(vel_diff[BIC_diff_strong], mlf[BIC_diff_strong], marker='o', s=10, label='Selected KOFFEE fits; R={:.2f}'.format(r_mlf_strong), color=colours[2], alpha=1.0)
    ax[2].plot(bin_center_strong, mlf_bin_medians_strong, marker='', lw=3, label='Median of selected KOFFEE fits;\n R={:.2f}'.format(r_mlf_med_strong), color=colours[2])

    ax[2].errorbar(230, np.nanmin(mlf), xerr=np.nanmedian(vel_diff_err[BIC_diff_strong]), yerr=[[np.nanmedian(mlf_err_min[BIC_diff_strong])], [np.nanmedian(mlf_err_max[BIC_diff_strong])]], c='k')

    #ax[1].set_xscale('log')
    lgnd = ax[2].legend(frameon=True, fontsize='small', loc='lower left', framealpha=0.5, edgecolor='white')
    lgnd.legendHandles[0]._legmarker.set_markersize(3)
    ax[2].set_xlabel('Velocity Difference [km s$^{-1}$]')
    ax[2].set_title('strongly likely BIC')

    plt.show()




def plot_out_vel_flux(flux_outflow_results, flux_outflow_error, OIII_outflow_results, OIII_outflow_error, BIC_outflow, BIC_no_outflow, statistical_results, z, radius, flux_ratio_line='OIII', weighted_average=True):
    """
    Plots the outflow velocity against the broad-to-narrow flux ratio

    Parameters
    ----------
    flux_outflow_results : :obj:'~numpy.ndarray'
        array of outflow results from KOFFEE for the line we want to calculate
        the flux ratio for.  Should be (7, statistical_results.shape)

    flux_outflow_err : :obj:'~numpy.ndarray'
        array of the outflow result errors from KOFFEE for flux ratio line

    OIII_outflow_results : :obj:'~numpy.ndarray'
        array of outflow results from KOFFEE for OIII 5007 line.  Should be
        (7, statistical_results.shape)

    OIII_outflow_err : :obj:'~numpy.ndarray'
        array of the outflow result errors from KOFFEE for OIII 5007 line

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
        the line we calculate the broad-to-narrow flux ratio for.  Used in the
        plotting labels (Default='OIII')

    weighted_average : boolean
        whether or not to take a weighted average using the errors (Default=True)

    Returns
    -------
    A graph of flux ratio against the outflow velocity

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
    vel_out = vel_out[flow_mask]
    vel_out_err = vel_out_err[flow_mask]
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
    vel_out_err[np.where(np.isnan(vel_out_err)==True)] = np.nanmedian(vel_out_err)

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
        bin_center_all, flux_bin_medians_all, flux_bin_lower_q_all, flux_bin_upper_q_all = pf.binned_median_quantile_lin(vel_out, flux_ratio, num_bins=num_bins, weights=None, min_bin=min_bin, max_bin=max_bin)
        bin_center_physical, flux_bin_medians_physical, flux_bin_lower_q_physical, flux_bin_upper_q_physical = pf.binned_median_quantile_lin(vel_out[physical_mask], flux_ratio[physical_mask], num_bins=num_bins, weights=None, min_bin=min_bin, max_bin=max_bin)
        bin_center_strong, flux_bin_medians_strong, flux_bin_lower_q_strong, flux_bin_upper_q_strong = pf.binned_median_quantile_lin(vel_out[BIC_diff_strong], flux_ratio[BIC_diff_strong], num_bins=num_bins, weights=None, min_bin=min_bin, max_bin=max_bin)

    elif weighted_average == True:
        bin_center_all, flux_bin_medians_all, flux_bin_lower_q_all, flux_bin_upper_q_all = pf.binned_median_quantile_lin(vel_out, flux_ratio, num_bins=num_bins, weights=[flux_error], min_bin=min_bin, max_bin=max_bin)
        bin_center_physical, flux_bin_medians_physical, flux_bin_lower_q_physical, flux_bin_upper_q_physical = pf.binned_median_quantile_lin(vel_out[physical_mask], flux_ratio[physical_mask], num_bins=num_bins, weights=[flux_error], min_bin=min_bin, max_bin=max_bin)
        bin_center_strong, flux_bin_medians_strong, flux_bin_lower_q_strong, flux_bin_upper_q_strong = pf.binned_median_quantile_lin(vel_out[BIC_diff_strong], flux_ratio[BIC_diff_strong], num_bins=num_bins, weights=[flux_error], min_bin=min_bin, max_bin=max_bin)

    #calculate the r value for the median values
    r_flux_med_all, p_value_flux_all = pf.pearson_correlation(bin_center_all, flux_bin_medians_all)
    r_flux_med_physical, p_value_flux_physical = pf.pearson_correlation(bin_center_physical, flux_bin_medians_physical)
    r_flux_med_strong, p_value_flux_strong = pf.pearson_correlation(bin_center_strong, flux_bin_medians_strong)

    #calculate the r value for all the values
    r_flux_all, p_value_flux_all = pf.pearson_correlation(vel_out, flux_ratio)
    r_flux_physical, p_value_flux_physical = pf.pearson_correlation(vel_out[physical_mask], flux_ratio[physical_mask])
    r_flux_strong, p_value_flux_strong = pf.pearson_correlation(vel_out[BIC_diff_strong], flux_ratio[BIC_diff_strong])

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
    ax[0].scatter(vel_out, flux_ratio, marker='o', s=10, label='All KOFFEE fits; R={:.2f}'.format(r_flux_all), color=colours[0], alpha=0.8)
    ax[0].plot(bin_center_all, flux_bin_medians_all, marker='', lw=3, label='Median all KOFFEE fits; R={:.2f}'.format(r_flux_med_all), color=colours[0])

    ax[0].errorbar(500, np.nanmin(flux_ratio)+0.3, xerr=np.nanmedian(vel_out_err), yerr=np.nanmedian(flux_error), c='k')

    ax[0].set_ylim(np.nanmin(flux_ratio)-0.1, np.nanmax(flux_ratio)+0.5)
    lgnd = ax[0].legend(frameon=True, fontsize='small', loc='upper left', framealpha=0.5, edgecolor=None)
    lgnd.legendHandles[0]._legmarker.set_markersize(3)
    ax[0].set_ylabel(flux_ratio_line+' Log(F$_{broad}$/F$_{narrow}$)')
    ax[0].set_xlabel('Maximum Outflow Velocity [km s$^{-1}$]')
    ax[0].set_title('all spaxels')

    #plot points within 90% radius
    ax[1].fill_between(bin_center_physical, flux_bin_lower_q_physical, flux_bin_upper_q_physical, color=colours[1], alpha=0.3)
    ax[1].scatter(vel_out[radius>6.1], flux_ratio[radius>6.1], marker='o', s=10, label='All KOFFEE fits', edgecolors=colours[0], alpha=0.3, facecolors='none')
    ax[1].scatter(vel_out[vel_disp<=51], flux_ratio[vel_disp<=51], marker='v', s=10, edgecolors=colours[0], alpha=0.3, facecolors='none')
    ax[1].scatter(vel_out[physical_mask], flux_ratio[physical_mask], marker='o', s=10, label='Selected KOFFEE fits; R={:.2f}'.format(r_flux_physical), color=colours[1], alpha=0.8)
    ax[1].plot(bin_center_physical, flux_bin_medians_physical, marker='', lw=3, label='Median of selected KOFFEE fits; R={:.2f}'.format(r_flux_med_physical), color=colours[1])

    ax[1].errorbar(500, np.nanmin(flux_ratio)+0.3, xerr=np.nanmedian(vel_out_err[physical_mask]), yerr=np.nanmedian(flux_error[physical_mask]), c='k')

    lgnd = ax[1].legend(frameon=True, fontsize='small', loc='upper left', framealpha=0.5, edgecolor=None)
    lgnd.legendHandles[0]._legmarker.set_markersize(3)
    ax[1].set_xlabel('Maximum Outflow Velocity [km s$^{-1}$]')
    ax[1].set_title(r'$r$<$r_{90}$ and $\sigma_{broad}$>$\sigma_{inst}$')

    #plot points with strong BIC values
    ax[2].fill_between(bin_center_strong, flux_bin_lower_q_strong, flux_bin_upper_q_strong, color=colours[2], alpha=0.3)
    ax[2].scatter(vel_out[~BIC_diff_strong], flux_ratio[~BIC_diff_strong], marker='o', s=10, label='All KOFFEE fits', edgecolors=colours[0], alpha=0.3, facecolors='none')
    ax[2].scatter(vel_out[BIC_diff_strong], flux_ratio[BIC_diff_strong], marker='o', s=10, label='Selected KOFFEE fits; R={:.2f}'.format(r_flux_strong), color=colours[2], alpha=1.0)
    ax[2].plot(bin_center_strong, flux_bin_medians_strong, marker='', lw=3, label='Median of selected KOFFEE fits; R={:.2f}'.format(r_flux_med_strong), color=colours[2])

    ax[2].errorbar(500, np.nanmin(flux_ratio)+0.3, xerr=np.nanmedian(vel_out_err[BIC_diff_strong]), yerr=np.nanmedian(flux_error[BIC_diff_strong]), c='k')

    lgnd = ax[2].legend(frameon=True, fontsize='small', loc='upper left', framealpha=0.5, edgecolor=None)
    lgnd.legendHandles[0]._legmarker.set_markersize(3)
    ax[2].set_xlabel('Maximum Outflow Velocity [km s$^{-1}$]')
    ax[2].set_title('strongly likely BIC')

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
