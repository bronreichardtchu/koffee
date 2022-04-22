"""
NAME:
	mlf_plots.py

AUTHOR:
	Bronwyn Reichardt Chu
	Swinburne
	2021

EMAIL:
	<breichardtchu@swin.edu.au>

PURPOSE:
	To make plots of results from koffee against the mass loading factor
	Written on MacOS Mojave 10.14.5, with Python 3.7

FUNCTIONS INCLUDED:
    plot_sfr_mlf_flux
    plot_sfr_mlf_rad
    plot_mlf_model_rad
    plot_mlf_model_sigsfr
    plot_sfr_mlf
    plot_sfr_mout
    plot_sfr_momentum_out
    plot_sfr_momentum_out2
    plot_radius_mlf
    plot_flux_mlf
    plot_ew_mlf
    plot_ew_mout
    map_of_mlf

MODIFICATION HISTORY:
		v.1.0 - first created January 2021

"""
import numpy as np

import matplotlib.pyplot as plt
import cmasher as cmr

from scipy.optimize import curve_fit

from astropy import units as u

import plotting_functions as pf
import calculate_outflow_velocity as calc_outvel
import calculate_star_formation_rate as calc_sfr
import calculate_mass_loading_factor as calc_mlf
import calculate_equivalent_width as calc_ew
import brons_display_pixels_kcwi as bdpk
import koffee



#===============================================================================
# PLOTTING FUNCTION FOR PAPER I
#===============================================================================

#Figure 4
def plot_sfr_mlf_flux(OIII_outflow_results, OIII_outflow_error, hbeta_outflow_results, hbeta_outflow_error, hbeta_no_outflow_results, hbeta_no_outflow_error, BIC_outflow, BIC_no_outflow, statistical_results, z, radius, weighted_average=True):
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

    weighted_average : boolean
        whether or not to take a weighted average using the errors (Default=True)

    Returns
    -------
    A six panel graph of the mass loading factor and Hbeta flux ratio against
    the SFR surface density

    """
    #calculate the sfr surface density - using just the systemic line, and including the flux line
    #don't include extinction since this was included in the continuum subtraction using ppxf
    sfr, sfr_err, total_sfr, sfr_surface_density, sfr_surface_density_err = calc_sfr.calc_sfr_koffee(hbeta_outflow_results, hbeta_outflow_error, hbeta_no_outflow_results, hbeta_no_outflow_error, statistical_results, z, include_extinction=False, include_outflow=False)

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

def plot_sfr_mlf_rad(OIII_outflow_results, OIII_outflow_error, hbeta_outflow_results, hbeta_outflow_error, hbeta_no_outflow_results, hbeta_no_outflow_error, statistical_results, z, radius):
    """
    Plots the SFR surface density against the mass loading factor coloured by
    radius, with Sigma_SFR calculated using only the narrow component.

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

    Returns
    -------
    A graph of the mass loading factor against the SFR surface density, coloured
    by radius

    """
    #calculate the sfr surface density - using just the systemic line, and including the flux line
    #don't include extinction since this was included in the continuum subtraction using ppxf
    sfr, sfr_err, total_sfr, sfr_surface_density, sfr_surface_density_err = calc_sfr.calc_sfr_koffee(hbeta_outflow_results, hbeta_outflow_error, hbeta_no_outflow_results, hbeta_no_outflow_error, statistical_results, z, include_extinction=False, include_outflow=False)

    #calculate the mass loading factor
    mlf, mlf_max, mlf_min = calc_mlf.calc_mass_loading_factor(OIII_outflow_results, OIII_outflow_error, hbeta_outflow_results, hbeta_outflow_error, hbeta_no_outflow_results, hbeta_no_outflow_error, statistical_results, z)

    #create the flow mask
    flow_mask = (statistical_results>0) & (np.isnan(hbeta_outflow_results[3,:,:])==False)


    #flatten all the arrays and get rid of extra spaxels
    sig_sfr = sfr_surface_density[flow_mask]
    sig_sfr_err = sfr_surface_density_err[flow_mask]

    mlf = mlf[flow_mask]
    mlf_max = mlf_max[flow_mask]
    mlf_min = mlf_min[flow_mask]

    radius = radius[flow_mask]


    #take the log of the mlf
    mlf = np.log10(mlf)
    mlf_max = np.log10(mlf_max)
    mlf_min = np.log10(mlf_min)

    #calculate the errors
    mlf_err_max = mlf_max - mlf
    mlf_err_min = mlf - mlf_min

    #do the calculations for all the bins
    num_bins = 5
    min_bin = None #-0.05
    max_bin = None #0.6

    bin_center_all, mlf_bin_medians_all, mlf_bin_lower_q_all, mlf_bin_upper_q_all = pf.binned_median_quantile_log(sig_sfr, mlf, num_bins=num_bins, weights=None, min_bin=min_bin, max_bin=max_bin)


    #calculate the r value for the median values
    r_mlf_med_all, p_value_mlf_all = pf.pearson_correlation(bin_center_all, mlf_bin_medians_all)

    #calculate the r value for all the values
    r_mlf_all, p_value_mlf_all = pf.pearson_correlation(sig_sfr[~np.isnan(mlf)], mlf[~np.isnan(mlf)])

    #calculate Kim et al. trend
    sfr_surface_density_kim, mlf_Kim = pf.kim_et_al_2020(sig_sfr.min(), sig_sfr.max(), scale_factor=0.06) #scale_factor=abs(mlf_bin_medians_all[-1]/bin_center_all[-1]**-0.44)))


    #print average numbers for the different panels
    print('Number of spaxels in the first panel', mlf.shape)
    print('All spaxels median mlf:', np.nanmedian(mlf))
    print('All spaxels standard deviation mlf:', np.nanstd(mlf))
    print('')



    #-------
    #plot it
    #-------
    plt.rcParams.update(pf.get_rc_params())
    fig, ax = plt.subplots(nrows=1, ncols=1, sharex=True, sharey='row', figsize=(5,4), constrained_layout=True)

    #get colours from cmasher
    colours = cmr.take_cmap_colors('cmr.gem', 3, cmap_range=(0.25, 0.85), return_fmt='hex')

    #plot all points
    #ax[0].errorbar(sig_sfr, mlf, xerr=sig_sfr_err, yerr=[mlf_err_min, mlf_err_max], fmt='o', ms=3, color=colours[0], alpha=0.6, label='All KOFFEE fits; R={:.2f}'.format(r_mlf_all))
    ax.fill_between(bin_center_all, mlf_bin_lower_q_all, mlf_bin_upper_q_all, color=colours[0], alpha=0.3)
    scatter_points = ax.scatter(sig_sfr, mlf, marker='o', s=10, label='All KOFFEE fits; R={:.2f}'.format(r_mlf_all), c=radius, cmap=cmr.gem, alpha=0.8)
    ax.plot(bin_center_all, mlf_bin_medians_all, marker='', lw=3, label='Median all KOFFEE fits; R={:.2f}'.format(r_mlf_med_all), color=colours[0])

    ax.plot(sfr_surface_density_kim, np.log10(mlf_Kim), ':k', label='Kim+20, $\eta \propto \Sigma_{SFR}^{-0.44}$')

    ax.errorbar(0.03, np.nanmin(mlf)-0.1, xerr=np.nanmedian(sig_sfr_err), yerr=[[np.nanmedian(mlf_err_min)], [np.nanmedian(mlf_err_max)]], c='k')

    ax.set_ylim(-2.4, 0.7)
    ax.set_xscale('log')
    ax.set_xlim(0.003, 2)
    lgnd = ax.legend(frameon=True, fontsize='small', loc='upper left', framealpha=0.5)
    lgnd.legendHandles[0]._legmarker.set_markersize(3)
    plt.colorbar(scatter_points, label='Radius (Arcseconds)')
    ax.set_ylabel(r'Log($\eta$)')
    ax.set_xlabel('$\Sigma_{SFR}$ [M$_\odot$ yr$^{-1}$ kpc$^{-2}$]')

    plt.show()



def plot_mlf_model_rad(OIII_outflow_results, OIII_outflow_error, hbeta_outflow_results, hbeta_outflow_error, hbeta_no_outflow_results, hbeta_no_outflow_error, BIC_outflow, BIC_no_outflow, statistical_results, z, radius, compare='divide'):
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

    compare : string
        string defining how to compare the results to the model.  Can be 'divide'
        or 'subtract' (Default='divide')

    Returns
    -------
    A three panel figure of the mass loading factor compared to the expected
    model against the galaxy radius, with the usual three data selection panels

    """
    #calculate the sfr surface density - using just the systemic line, and including the flux line
    #don't include extinction since this was included in the continuum subtraction using ppxf
    sfr, sfr_err, total_sfr, sfr_surface_density, sfr_surface_density_err = calc_sfr.calc_sfr_koffee(hbeta_outflow_results, hbeta_outflow_error, hbeta_no_outflow_results, hbeta_no_outflow_error, statistical_results, z, include_extinction=False, include_outflow=False)

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

    #find the median bins of all the points when mlf is plotted against sigma sfr
    num_bins = 5
    min_bin = None #-0.05
    max_bin = None #0.6
    bin_center_all, mlf_bin_medians_all, mlf_bin_lower_q_all, mlf_bin_upper_q_all = pf.binned_median_quantile_lin(sig_sfr, np.log10(mlf), num_bins=num_bins, weights=None, min_bin=min_bin, max_bin=max_bin)

    #print((10**mlf_bin_medians_all[0])/(bin_center_all[0]**-0.44))

    #divide the mass loading factor by the model
    mlf_model = np.full_like(mlf, np.nan, dtype=np.double)

    for i in np.arange(mlf.shape[0]):
        #calculate the expected mlf at each sigma_sfr
        sigma_sfr_model, mlf_expected = pf.kim_et_al_2020(sig_sfr[i], sig_sfr[i], scale_factor=(10**mlf_bin_medians_all[0])/(bin_center_all[0]**-0.44))#0.06)
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

    bin_center_physical, mlf_bin_medians_physical, mlf_bin_lower_q_physical, mlf_bin_upper_q_physical = pf.binned_median_quantile_lin(radius[physical_mask], mlf_model[physical_mask], num_bins=num_bins, weights=None, min_bin=min_bin, max_bin=max_bin)

    bin_center_strong, mlf_bin_medians_strong, mlf_bin_lower_q_strong, mlf_bin_upper_q_strong = pf.binned_median_quantile_lin(radius[BIC_diff_strong], mlf_model[BIC_diff_strong], num_bins=num_bins, weights=None, min_bin=min_bin, max_bin=max_bin)


    #calculate the r value for the median values
    r_mlf_med_all, p_value_mlf_all = pf.pearson_correlation(bin_center_all, mlf_bin_medians_all)
    r_mlf_med_physical, p_value_mlf_physical = pf.pearson_correlation(bin_center_physical, mlf_bin_medians_physical)
    r_mlf_med_strong, p_value_mlf_strong = pf.pearson_correlation(bin_center_strong, mlf_bin_medians_strong)

    #calculate the r value for all the values
    r_mlf_all, p_value_mlf_all = pf.pearson_correlation(radius[~np.isnan(mlf_model)], mlf_model[~np.isnan(mlf_model)])
    r_mlf_physical, p_value_mlf_physical = pf.pearson_correlation(radius[~np.isnan(mlf_model)&physical_mask], mlf_model[~np.isnan(mlf_model)&physical_mask])
    r_mlf_strong, p_value_mlf_strong = pf.pearson_correlation(radius[~np.isnan(mlf_model)&BIC_diff_strong], mlf_model[~np.isnan(mlf_model)&BIC_diff_strong])



    #print average numbers for the different panels
    print('Number of spaxels in the first panel', mlf.shape)
    print('All spaxels median mlf:', np.nanmedian(mlf))
    print('All spaxels standard deviation mlf:', np.nanstd(mlf))
    print('')

    print('All spaxels median mlf/model:', np.nanmedian(mlf_model))
    print('All spaxels standard deviation mlf/model:', np.nanstd(mlf_model))
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

    print('Physical spaxels median mlf/model:', np.nanmedian(mlf_model[physical_mask]))
    print('Physical spaxels standard deviation mlf/model:', np.nanstd(mlf_model[physical_mask]))
    print('')

    print('Number of spaxels with strong BIC differences:', mlf[BIC_diff_strong].shape)
    print('')

    print('Clean spaxels median mlf:', np.nanmedian(mlf[BIC_diff_strong]))
    print('Clean spaxels standard deviation mlf:', np.nanstd(mlf[BIC_diff_strong]))
    print('')

    print('Clean spaxels median mlf/model:', np.nanmedian(mlf_model[BIC_diff_strong]))
    print('Clean spaxels standard deviation mlf/model:', np.nanstd(mlf_model[BIC_diff_strong]))
    print('')




    #-------
    #plot it
    #-------
    plt.rcParams.update(pf.get_rc_params())
    fig, ax = plt.subplots(nrows=1, ncols=3, sharex=True, sharey='row', figsize=(10,4), constrained_layout=True)

    #get colours from cmasher
    colours = cmr.take_cmap_colors('cmr.gem', 3, cmap_range=(0.25, 0.85), return_fmt='hex')

    #plot all points
    ax[0].fill_between(bin_center_all, mlf_bin_lower_q_all, mlf_bin_upper_q_all, color=colours[0], alpha=0.3)
    ax[0].scatter(radius, mlf_model, marker='o', s=10, label='All KOFFEE fits; R={:.2f}'.format(r_mlf_all), c=colours[0], alpha=0.8)
    ax[0].plot(bin_center_all, mlf_bin_medians_all, marker='', lw=3, label='Median all KOFFEE fits; R={:.2f}'.format(r_mlf_med_all), color=colours[0])

    lgnd = ax[0].legend(frameon=True, fontsize='small', loc='upper right', framealpha=0.5)
    lgnd.legendHandles[0]._legmarker.set_markersize(3)
    if compare == 'divide':
        ax[0].set_ylabel(r'Log($\eta$/model)')
    elif compare == 'subtract':
        ax[0].set_ylabel(r'$\eta$-model')
    ax[0].set_xlabel('Radius (Arcseconds)')
    ax[0].set_title('all spaxels')

    #plot points within 90% radius
    ax[1].fill_between(bin_center_physical, mlf_bin_lower_q_physical, mlf_bin_upper_q_physical, color=colours[1], alpha=0.3)
    ax[1].scatter(radius[radius>6.1], mlf_model[radius>6.1], marker='o', s=10, label='All KOFFEE fits', edgecolors=colours[0], alpha=0.3, facecolors='none')
    ax[1].scatter(radius[vel_disp<=51], mlf_model[vel_disp<=51], marker='v', s=10, edgecolors=colours[0], alpha=0.3, facecolors='none')
    ax[1].scatter(radius[physical_mask], mlf_model[physical_mask], marker='o', s=10, label='Selected KOFFEE fits; R={:.2f}'.format(r_mlf_physical), color=colours[1], alpha=0.8)
    ax[1].plot(bin_center_physical, mlf_bin_medians_physical, marker='', lw=3, label='Median of selected KOFFEE fits; R={:.2f}'.format(r_mlf_med_physical), color=colours[1])

    lgnd = ax[1].legend(frameon=True, fontsize='small', loc='upper right', framealpha=0.5)
    lgnd.legendHandles[0]._legmarker.set_markersize(3)
    ax[1].set_xlabel('Radius (Arcseconds)')
    ax[1].set_title(r'$r$<$r_{90}$ and $\sigma_{broad}$>$\sigma_{inst}$')

    #plot points with strong BIC values
    ax[2].fill_between(bin_center_strong, mlf_bin_lower_q_strong, mlf_bin_upper_q_strong, color=colours[2], alpha=0.3)
    ax[2].scatter(radius[~BIC_diff_strong], mlf_model[~BIC_diff_strong], marker='o', s=10, label='All KOFFEE fits', color=colours[0], alpha=0.3, facecolors='none')
    ax[2].scatter(radius[BIC_diff_strong], mlf_model[BIC_diff_strong], marker='o', s=10, label='Selected KOFFEE fits; R={:.2f}'.format(r_mlf_strong), color=colours[2], alpha=1.0)
    ax[2].plot(bin_center_strong, mlf_bin_medians_strong, marker='', lw=3, label='Median of selected KOFFEE fits; R={:.2f}'.format(r_mlf_med_strong), color=colours[2])

    lgnd = ax[2].legend(frameon=True, fontsize='small', loc='upper right', framealpha=0.5)
    lgnd.legendHandles[0]._legmarker.set_markersize(3)
    ax[2].set_xlabel('Radius (Arcseconds)')
    ax[2].set_title('strongly likely BIC')

    plt.show()



def plot_mlf_model_sigsfr(OIII_outflow_results, OIII_outflow_error, hbeta_outflow_results, hbeta_outflow_error, hbeta_no_outflow_results, hbeta_no_outflow_error, BIC_outflow, BIC_no_outflow, statistical_results, z, radius, compare='divide'):
    """
    Plots the mass loading factor values found with KOFFEE either divided by or
    subtracted from the expected values calculated with the model from
    Kim et al. 2020 against the SFR surface density, with Sigma_SFR calculated
    using only the narrow line

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

    compare : string
        string defining how to compare the results to the model.  Can be 'divide'
        or 'subtract' (Default='divide')

    Returns
    -------
    A three panel figure of the mass loading factor compared to the expected
    model against the SFR surface density

    """
    #calculate the sfr surface density - using just the systemic line, and including the flux line
    #don't include extinction since this was included in the continuum subtraction using ppxf
    sfr, sfr_err, total_sfr, sfr_surface_density, sfr_surface_density_err = calc_sfr.calc_sfr_koffee(hbeta_outflow_results, hbeta_outflow_error, hbeta_no_outflow_results, hbeta_no_outflow_error, statistical_results, z, include_extinction=False, include_outflow=False)

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

    #calculate Kim et al. trend
    sfr_surface_density_kim, mlf_Kim = pf.kim_et_al_2020(sig_sfr.min(), sig_sfr.max(), scale_factor=0.06) #scale_factor=abs(mlf_bin_medians_all[-1]/bin_center_all[-1]**-0.44)))

    #divide the mass loading factor by the model
    mlf_model = np.full_like(mlf, np.nan, dtype=np.double)

    for i in np.arange(mlf.shape[0]):
        #calculate the expected mlf at each sigma_sfr
        sigma_sfr_model, mlf_expected = pf.kim_et_al_2020(sig_sfr[i], sig_sfr[i], scale_factor=0.06)
        sigma_sfr_model = sigma_sfr_model[0]
        mlf_expected = mlf_expected[0]
        #divide the mlf by the expected mlf
        if compare == 'divide':
            mlf_model[i] = np.log10(mlf[i]/mlf_expected)
        elif compare == 'subtract':
            mlf_model[i] = mlf[i] - mlf_expected


    #do the calculations for all the bins
    num_bins = 5
    min_bin = None #-0.05
    max_bin = None #0.6

    bin_center_all, mlf_bin_medians_all, mlf_bin_lower_q_all, mlf_bin_upper_q_all = pf.binned_median_quantile_lin(sig_sfr, mlf_model, num_bins=num_bins, weights=None, min_bin=min_bin, max_bin=max_bin)

    bin_center_physical, mlf_bin_medians_physical, mlf_bin_lower_q_physical, mlf_bin_upper_q_physical = pf.binned_median_quantile_lin(sig_sfr[physical_mask], mlf_model[physical_mask], num_bins=num_bins, weights=None, min_bin=min_bin, max_bin=max_bin)

    bin_center_strong, mlf_bin_medians_strong, mlf_bin_lower_q_strong, mlf_bin_upper_q_strong = pf.binned_median_quantile_lin(sig_sfr[BIC_diff_strong], mlf_model[BIC_diff_strong], num_bins=num_bins, weights=None, min_bin=min_bin, max_bin=max_bin)


    #calculate the r value for the median values
    r_mlf_med_all, p_value_mlf_all = pf.pearson_correlation(bin_center_all, mlf_bin_medians_all)
    r_mlf_med_physical, p_value_mlf_physical = pf.pearson_correlation(bin_center_physical, mlf_bin_medians_physical)
    r_mlf_med_strong, p_value_mlf_strong = pf.pearson_correlation(bin_center_strong, mlf_bin_medians_strong)

    #calculate the r value for all the values
    r_mlf_all, p_value_mlf_all = pf.pearson_correlation(sig_sfr[~np.isnan(mlf_model)], mlf_model[~np.isnan(mlf_model)])
    r_mlf_physical, p_value_mlf_physical = pf.pearson_correlation(sig_sfr[~np.isnan(mlf_model)&physical_mask], mlf_model[~np.isnan(mlf_model)&physical_mask])
    r_mlf_strong, p_value_mlf_strong = pf.pearson_correlation(sig_sfr[~np.isnan(mlf_model)&BIC_diff_strong], mlf_model[~np.isnan(mlf_model)&BIC_diff_strong])



    #print average numbers for the different panels
    print('Number of spaxels in the first panel', mlf.shape)
    print('All spaxels median mlf:', np.nanmedian(mlf))
    print('All spaxels standard deviation mlf:', np.nanstd(mlf))
    print('')

    print('All spaxels median mlf/model:', np.nanmedian(mlf_model))
    print('All spaxels standard deviation mlf/model:', np.nanstd(mlf_model))
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

    print('Physical spaxels median mlf/model:', np.nanmedian(mlf_model[physical_mask]))
    print('Physical spaxels standard deviation mlf/model:', np.nanstd(mlf_model[physical_mask]))
    print('')

    print('Number of spaxels with strong BIC differences:', mlf[BIC_diff_strong].shape)
    print('')

    print('Clean spaxels median mlf:', np.nanmedian(mlf[BIC_diff_strong]))
    print('Clean spaxels standard deviation mlf:', np.nanstd(mlf[BIC_diff_strong]))
    print('')

    print('Clean spaxels median mlf/model:', np.nanmedian(mlf_model[BIC_diff_strong]))
    print('Clean spaxels standard deviation mlf/model:', np.nanstd(mlf_model[BIC_diff_strong]))
    print('')




    #-------
    #plot it
    #-------
    plt.rcParams.update(pf.get_rc_params())
    fig, ax = plt.subplots(nrows=1, ncols=3, sharex=True, sharey='row', figsize=(10,4), constrained_layout=True)

    #get colours from cmasher
    colours = cmr.take_cmap_colors('cmr.gem', 3, cmap_range=(0.25, 0.85), return_fmt='hex')

    #plot all points
    ax[0].fill_between(bin_center_all, mlf_bin_lower_q_all, mlf_bin_upper_q_all, color=colours[0], alpha=0.3)
    ax[0].scatter(sig_sfr, mlf_model, marker='o', s=10, label='All KOFFEE fits; R={:.2f}'.format(r_mlf_all), c=colours[0], alpha=0.8)
    ax[0].plot(bin_center_all, mlf_bin_medians_all, marker='', lw=3, label='Median all KOFFEE fits; R={:.2f}'.format(r_mlf_med_all), color=colours[0])

    lgnd = ax[0].legend(frameon=True, fontsize='small', loc='upper right', framealpha=0.5)
    lgnd.legendHandles[0]._legmarker.set_markersize(3)
    if compare == 'divide':
        ax[0].set_ylabel(r'log($\eta$/model)')
    elif compare == 'subtract':
        ax[0].set_ylabel(r'$\eta$-model')
    ax[0].set_xlabel('$\Sigma_{SFR}$ [M$_\odot$ yr$^{-1}$ kpc$^{-2}$]')
    ax[0].set_title('all spaxels')

    #plot points within 90% radius
    ax[1].fill_between(bin_center_physical, mlf_bin_lower_q_physical, mlf_bin_upper_q_physical, color=colours[1], alpha=0.3)
    ax[1].scatter(sig_sfr[radius>6.1], mlf_model[radius>6.1], marker='o', s=10, label='All KOFFEE fits', edgecolors=colours[0], alpha=0.3, facecolors='none')
    ax[1].scatter(sig_sfr[vel_disp<=51], mlf_model[vel_disp<=51], marker='v', s=10, edgecolors=colours[0], alpha=0.3, facecolors='none')
    ax[1].scatter(sig_sfr[physical_mask], mlf_model[physical_mask], marker='o', s=10, label='Selected KOFFEE fits; R={:.2f}'.format(r_mlf_physical), color=colours[1], alpha=0.8)
    ax[1].plot(bin_center_physical, mlf_bin_medians_physical, marker='', lw=3, label='Median of selected KOFFEE fits; R={:.2f}'.format(r_mlf_med_physical), color=colours[1])

    lgnd = ax[1].legend(frameon=True, fontsize='small', loc='upper right', framealpha=0.5)
    lgnd.legendHandles[0]._legmarker.set_markersize(3)
    ax[1].set_xlabel('$\Sigma_{SFR}$ [M$_\odot$ yr$^{-1}$ kpc$^{-2}$]')
    ax[1].set_title(r'$r$<$r_{90}$ and $\sigma_{broad}$>$\sigma_{inst}$')

    #plot points with strong BIC values
    ax[2].fill_between(bin_center_strong, mlf_bin_lower_q_strong, mlf_bin_upper_q_strong, color=colours[2], alpha=0.3)
    ax[2].scatter(sig_sfr[~BIC_diff_strong], mlf_model[~BIC_diff_strong], marker='o', s=10, label='All KOFFEE fits', color=colours[0], alpha=0.3, facecolors='none')
    ax[2].scatter(sig_sfr[BIC_diff_strong], mlf_model[BIC_diff_strong], marker='o', s=10, label='Selected KOFFEE fits; R={:.2f}'.format(r_mlf_strong), color=colours[2], alpha=1.0)
    ax[2].plot(bin_center_strong, mlf_bin_medians_strong, marker='', lw=3, label='Median of selected KOFFEE fits; R={:.2f}'.format(r_mlf_med_strong), color=colours[2])

    lgnd = ax[2].legend(frameon=True, fontsize='small', loc='upper right', framealpha=0.5)
    lgnd.legendHandles[0]._legmarker.set_markersize(3)
    ax[2].set_xlabel('$\Sigma_{SFR}$ [M$_\odot$ yr$^{-1}$ kpc$^{-2}$]')
    ax[2].set_title('strongly likely BIC')

    plt.show()



def plot_sfr_mlf(OIII_outflow_results, OIII_outflow_error, hbeta_outflow_results, hbeta_outflow_error, hbeta_no_outflow_results, hbeta_no_outflow_error, BIC_outflow, BIC_no_outflow, statistical_results, z, radius, weighted_average=True):
    """
    Plots the SFR surface density against the mass loading factor

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
    A three panel figure of the mass loading factor against the SFR surface density

    """
    #calculate the sfr surface density - using just the systemic line, and including the flux line
    #don't include extinction since this was included in the continuum subtraction using ppxf
    sfr, sfr_err, total_sfr, sfr_surface_density, sfr_surface_density_err = calc_sfr.calc_sfr_koffee(hbeta_outflow_results, hbeta_outflow_error, hbeta_no_outflow_results, hbeta_no_outflow_error, statistical_results, z, include_extinction=False, include_outflow=False)

    #calculate the mass loading factor
    mlf, mlf_max, mlf_min = calc_mlf.calc_mass_loading_factor(OIII_outflow_results, OIII_outflow_error, hbeta_outflow_results, hbeta_outflow_error, hbeta_no_outflow_results, hbeta_no_outflow_error, statistical_results, z)

    #calculate the velocity dispersion for the masking
    vel_disp, vel_disp_err, vel_diff, vel_diff_err, vel_out, vel_out_err = calc_outvel.calc_outflow_vel(OIII_outflow_results, OIII_outflow_error, statistical_results, z)

    #make the mask for the outflow spaxels
    flow_mask = (statistical_results>0)# & (sfr_surface_density>0.1)

    #flatten all the arrays and get rid of extra spaxels
    sig_sfr = sfr_surface_density[flow_mask]
    sig_sfr_err = sfr_surface_density_err[flow_mask]
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
    #vel_out_err[np.where(np.isnan(vel_out_err)==True)] = np.nanmedian(vel_out_err)

    #do the calculations for all the bins
    num_bins = 5
    min_bin = None #-0.05
    max_bin = None #0.6

    if weighted_average == False:
        bin_center_all, mlf_bin_medians_all, mlf_bin_lower_q_all, mlf_bin_upper_q_all = pf.binned_median_quantile_log(sig_sfr, mlf, num_bins=num_bins, weights=None, min_bin=min_bin, max_bin=max_bin)
        bin_center_physical, mlf_bin_medians_physical, mlf_bin_lower_q_physical, mlf_bin_upper_q_physical = pf.binned_median_quantile_log(sig_sfr[physical_mask], mlf[physical_mask], num_bins=num_bins, weights=None, min_bin=min_bin, max_bin=max_bin)
        bin_center_strong, mlf_bin_medians_strong, mlf_bin_lower_q_strong, mlf_bin_upper_q_strong = pf.binned_median_quantile_log(sig_sfr[BIC_diff_strong], mlf[BIC_diff_strong], num_bins=num_bins, weights=None, min_bin=min_bin, max_bin=max_bin)


    elif weighted_average == True:
        bin_center_all, mlf_bin_medians_all, mlf_bin_lower_q_all, mlf_bin_upper_q_all = pf.binned_median_quantile_log(sig_sfr, mlf, num_bins=num_bins, weights=[vel_out_err], min_bin=min_bin, max_bin=max_bin)
        bin_center_physical, mlf_bin_medians_physical, mlf_bin_lower_q_physical, mlf_bin_upper_q_physical = pf.binned_median_quantile_log(sig_sfr[physical_mask], mlf[physical_mask], num_bins=num_bins, weights=[vel_out_err], min_bin=min_bin, max_bin=max_bin)
        bin_center_strong, mlf_bin_medians_strong, mlf_bin_lower_q_strong, mlf_bin_upper_q_strong = pf.binned_median_quantile_log(sig_sfr[BIC_diff_strong], mlf[BIC_diff_strong], num_bins=num_bins, weights=[vel_out_err], min_bin=min_bin, max_bin=max_bin)


    #calculate the r value for the median values
    r_mlf_med_all, p_value_mlf_all = pf.pearson_correlation(bin_center_all, mlf_bin_medians_all)
    r_mlf_med_physical, p_value_mlf_physical = pf.pearson_correlation(bin_center_physical, mlf_bin_medians_physical)
    r_mlf_med_strong, p_value_mlf_strong = pf.pearson_correlation(bin_center_strong, mlf_bin_medians_strong)

    #calculate the r value for all the values
    r_mlf_all, p_value_mlf_all = pf.pearson_correlation(sig_sfr[~np.isnan(mlf)], mlf[~np.isnan(mlf)])
    r_mlf_physical, p_value_mlf_physical = pf.pearson_correlation(sig_sfr[~np.isnan(mlf)&physical_mask], mlf[~np.isnan(mlf)&physical_mask])
    r_mlf_strong, p_value_mlf_strong = pf.pearson_correlation(sig_sfr[~np.isnan(mlf)&BIC_diff_strong], mlf[~np.isnan(mlf)&BIC_diff_strong])

    #calculate Kim et al. trend
    sfr_surface_density_kim, mlf_Kim = pf.kim_et_al_2020(sig_sfr.min(), sig_sfr.max(), scale_factor=0.06) #scale_factor=abs(np.nanmedian(mlf)/(np.nanmedian(sig_sfr)**-0.44)))

    #fit our own trends
    popt_mlf_all_medians, pcov_mlf_all_medians = curve_fit(pf.fitting_function, bin_center_all, 10**mlf_bin_medians_all)
    popt_mlf_physical_medians, pcov_mlf_physical_medians = curve_fit(pf.fitting_function, bin_center_physical, 10**mlf_bin_medians_physical)
    popt_mlf_clean_medians, pcov_mlf_clean_medians = curve_fit(pf.fitting_function, bin_center_strong, 10**mlf_bin_medians_strong)

    popt_mlf_all, pcov_mlf_all = curve_fit(pf.fitting_function, sig_sfr[~np.isnan(mlf)], 10**mlf[~np.isnan(mlf)])
    popt_mlf_physical, pcov_mlf_physical = curve_fit(pf.fitting_function, sig_sfr[~np.isnan(mlf)&physical_mask], 10**mlf[~np.isnan(mlf)&physical_mask])
    popt_mlf_clean, pcov_mlf_clean = curve_fit(pf.fitting_function, sig_sfr[~np.isnan(mlf)&BIC_diff_strong], 10**mlf[~np.isnan(mlf)&BIC_diff_strong])

    print(popt_mlf_all, pcov_mlf_all)
    print([popt_mlf_all_medians[0], np.sqrt(np.diag(pcov_mlf_all_medians))[0], popt_mlf_all_medians[1], np.sqrt(np.diag(pcov_mlf_all_medians))[1]])

    sfr_linspace = np.linspace(sig_sfr.min(), sig_sfr.max()+4, num=1000)


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
    #ax[0].errorbar(sig_sfr, mlf, xerr=sig_sfr_err, yerr=[mlf_err_min, mlf_err_max], fmt='o', ms=3, color=colours[0], alpha=0.6, label='All KOFFEE fits; R={:.2f}'.format(r_mlf_all))
    ax[0].fill_between(bin_center_all, mlf_bin_lower_q_all, mlf_bin_upper_q_all, color=colours[0], alpha=0.3)
    ax[0].scatter(sig_sfr, mlf, marker='o', s=10, label='All KOFFEE fits; R={:.2f}'.format(r_mlf_all), color=colours[0], alpha=0.8)
    ax[0].plot(bin_center_all, mlf_bin_medians_all, marker='', lw=3, label='Median all KOFFEE fits; R={:.2f}'.format(r_mlf_med_all), color=colours[0])

    ax[0].plot(sfr_surface_density_kim, np.log10(mlf_Kim), ':k', label='Kim+20, $\eta \propto \Sigma_{SFR}^{-0.44}$')

    ax[0].plot(sfr_linspace, np.log10(pf.fitting_function(sfr_linspace, *popt_mlf_all)), 'r-', label='Fit: $\eta=%5.0f\pm$%2.0f $\Sigma_{SFR}^{%5.2f \pm %5.2f}$' % (popt_mlf_all[0], np.sqrt(np.diag(pcov_mlf_all))[0], popt_mlf_all[1], np.sqrt(np.diag(pcov_mlf_all))[1]))
    ax[0].plot(sfr_linspace, np.log10(pf.fitting_function(sfr_linspace, *popt_mlf_all_medians)), 'r--', label='Median Fit: $\eta=%5.0f\pm$%2.0f $\Sigma_{SFR}^{%5.2f \pm %5.2f}$' %(popt_mlf_all_medians[0], np.sqrt(np.diag(pcov_mlf_all_medians))[0], popt_mlf_all_medians[1], np.sqrt(np.diag(pcov_mlf_all_medians))[1]))

    ax[0].errorbar(0.03, np.nanmin(mlf)-0.1, xerr=np.nanmedian(sig_sfr_err), yerr=[[np.nanmedian(mlf_err_min)], [np.nanmedian(mlf_err_max)]], c='k')

    ax[0].set_ylim(-2.4, 0.7)
    ax[0].set_xscale('log')
    ax[0].set_xlim(0.003, 2)
    lgnd = ax[0].legend(frameon=True, fontsize='small', loc='upper left', framealpha=0.5)
    lgnd.legendHandles[0]._legmarker.set_markersize(3)
    ax[0].set_ylabel('Log(Mass Loading Factor)')
    ax[0].set_xlabel('$\Sigma_{SFR}$ [M$_\odot$ yr$^{-1}$ kpc$^{-2}$]')
    ax[0].set_title('all spaxels')

    #plot points within 90% radius
    ax[1].fill_between(bin_center_physical, mlf_bin_lower_q_physical, mlf_bin_upper_q_physical, color=colours[1], alpha=0.3)
    ax[1].scatter(sig_sfr[radius>6.1], mlf[radius>6.1], marker='o', s=10, label='All KOFFEE fits', edgecolors=colours[0], alpha=0.3, facecolors='none')
    ax[1].scatter(sig_sfr[vel_disp<=51], mlf[vel_disp<=51], marker='v', s=10, edgecolors=colours[0], alpha=0.3, facecolors='none')
    ax[1].scatter(sig_sfr[physical_mask], mlf[physical_mask], marker='o', s=10, label='Selected KOFFEE fits; R={:.2f}'.format(r_mlf_physical), color=colours[1], alpha=0.8)
    ax[1].plot(bin_center_physical, mlf_bin_medians_physical, marker='', lw=3, label='Median of selected KOFFEE fits; R={:.2f}'.format(r_mlf_med_physical), color=colours[1])

    ax[1].plot(sfr_surface_density_kim, np.log10(mlf_Kim), ':k')

    ax[1].plot(sfr_linspace, np.log10(pf.fitting_function(sfr_linspace, *popt_mlf_physical)), 'r-', label='Fit: $\eta=%5.0f\pm$%2.0f $\Sigma_{SFR}^{%5.2f \pm %5.2f}$' % (popt_mlf_physical[0], np.sqrt(np.diag(pcov_mlf_physical))[0], popt_mlf_physical[1], np.sqrt(np.diag(pcov_mlf_physical))[1]))
    ax[1].plot(sfr_linspace, np.log10(pf.fitting_function(sfr_linspace, *popt_mlf_physical_medians)), 'r--', label='Median Fit: $\eta=%5.0f\pm$%2.0f $\Sigma_{SFR}^{%5.2f \pm %5.2f}$' %(popt_mlf_physical_medians[0], np.sqrt(np.diag(pcov_mlf_physical_medians))[0], popt_mlf_physical_medians[1], np.sqrt(np.diag(pcov_mlf_physical_medians))[1]))

    ax[1].errorbar(0.03, np.nanmin(mlf)-0.1, xerr=np.nanmedian(sig_sfr_err[physical_mask]), yerr=[[np.nanmedian(mlf_err_min[physical_mask])], [np.nanmedian(mlf_err_max[physical_mask])]], c='k')

    #ax[1].set_xscale('log')
    lgnd = ax[1].legend(frameon=True, fontsize='small', loc='upper left', framealpha=0.5)
    lgnd.legendHandles[0]._legmarker.set_markersize(3)
    ax[1].set_xlabel('$\Sigma_{SFR}$ [M$_\odot$ yr$^{-1}$ kpc$^{-2}$]')
    ax[1].set_title(r'$r$<$r_{90}$ and $\sigma_{broad}$>$\sigma_{inst}$')

    #plot points with strong BIC values
    ax[2].fill_between(bin_center_strong, mlf_bin_lower_q_strong, mlf_bin_upper_q_strong, color=colours[2], alpha=0.3)
    ax[2].scatter(sig_sfr[~BIC_diff_strong], mlf[~BIC_diff_strong], marker='o', s=10, label='All KOFFEE fits', color=colours[0], alpha=0.3, facecolors='none')
    ax[2].scatter(sig_sfr[BIC_diff_strong], mlf[BIC_diff_strong], marker='o', s=10, label='Selected KOFFEE fits; R={:.2f}'.format(r_mlf_strong), color=colours[2], alpha=1.0)
    ax[2].plot(bin_center_strong, mlf_bin_medians_strong, marker='', lw=3, label='Median of selected KOFFEE fits; R={:.2f}'.format(r_mlf_med_strong), color=colours[2])

    ax[2].plot(sfr_surface_density_kim, np.log10(mlf_Kim), ':k')

    ax[2].plot(sfr_linspace, np.log10(pf.fitting_function(sfr_linspace, *popt_mlf_clean)), 'r-', label='Fit: $\eta=%5.0f\pm$%2.0f $\Sigma_{SFR}^{%5.2f \pm %5.2f}$' % (popt_mlf_clean[0], np.sqrt(np.diag(pcov_mlf_clean))[0], popt_mlf_clean[1], np.sqrt(np.diag(pcov_mlf_clean))[1]))
    ax[2].plot(sfr_linspace, np.log10(pf.fitting_function(sfr_linspace, *popt_mlf_clean_medians)), 'r--', label='Median Fit: $\eta=%5.0f\pm$%2.0f $\Sigma_{SFR}^{%5.2f \pm %5.2f}$' %(popt_mlf_clean_medians[0], np.sqrt(np.diag(pcov_mlf_clean_medians))[0], popt_mlf_clean_medians[1], np.sqrt(np.diag(pcov_mlf_clean_medians))[1]))

    ax[2].errorbar(0.03, np.nanmin(mlf)-0.1, xerr=np.nanmedian(sig_sfr_err[BIC_diff_strong]), yerr=[[np.nanmedian(mlf_err_min[BIC_diff_strong])], [np.nanmedian(mlf_err_max[BIC_diff_strong])]], c='k')

    #ax[1].set_xscale('log')
    lgnd = ax[2].legend(frameon=True, fontsize='small', loc='upper left', framealpha=0.5)
    lgnd.legendHandles[0]._legmarker.set_markersize(3)
    ax[2].set_xlabel('$\Sigma_{SFR}$ [M$_\odot$ yr$^{-1}$ kpc$^{-2}$]')
    ax[2].set_title('strongly likely BIC')

    plt.show()




def plot_sfr_mout(OIII_outflow_results, OIII_outflow_error, hbeta_outflow_results, hbeta_outflow_error, hbeta_no_outflow_results, hbeta_no_outflow_error, BIC_outflow, BIC_no_outflow, statistical_results, z, radius, weighted_average=True):
    """
    Plots the SFR surface density against the mass outflow rate, with Sigma_SFR
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
    A three panel figure of the mass outflow rate against the SFR surface density

    """
    #calculate the sfr surface density - using just the systemic line, and including the flux line
    #don't include extinction since this was included in the continuum subtraction using ppxf
    sfr, sfr_err, total_sfr, sfr_surface_density, sfr_surface_density_err = calc_sfr.calc_sfr_koffee(hbeta_outflow_results, hbeta_outflow_error, hbeta_no_outflow_results, hbeta_no_outflow_error, statistical_results, z, include_extinction=False, include_outflow=False)

    #calculate the mass outflow rate (in g/s)
    m_out, m_out_max, m_out_min = calc_mlf.calc_mass_outflow_rate(OIII_outflow_results, OIII_outflow_error, hbeta_outflow_results, hbeta_outflow_error, statistical_results, z)

    #convert to solar masses per year
    m_out = m_out.to(u.solMass/u.yr)
    m_out_max = m_out_max.to(u.solMass/u.yr)
    m_out_min = m_out_min.to(u.solMass/u.yr)

    #calculate the velocity dispersion for the masking
    vel_disp, vel_disp_err, vel_diff, vel_diff_err, vel_out, vel_out_err = calc_outvel.calc_outflow_vel(OIII_outflow_results, OIII_outflow_error, statistical_results, z)

    #make the mask for the outflow spaxels
    flow_mask = (statistical_results>0)# & (sfr_surface_density>0.1)

    #flatten all the arrays and get rid of extra spaxels
    sig_sfr = sfr_surface_density[flow_mask]
    sig_sfr_err = sfr_surface_density_err[flow_mask]
    m_out = m_out[flow_mask]
    m_out_max = m_out_max[flow_mask]
    m_out_min = m_out_min[flow_mask]
    BIC_outflow = BIC_outflow[flow_mask]
    BIC_no_outflow = BIC_no_outflow[flow_mask]
    vel_disp = vel_disp[flow_mask]
    radius = radius[flow_mask]

    #take the log of the mlf
    m_out = np.log10(m_out.value)
    m_out_max = np.log10(m_out_max.value)
    m_out_min = np.log10(m_out_min.value)

    #calculate the errors
    m_out_err_max = m_out_max - m_out
    m_out_err_min = m_out - m_out_min

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
        bin_center_all, m_out_bin_medians_all, m_out_bin_lower_q_all, m_out_bin_upper_q_all = pf.binned_median_quantile_log(sig_sfr, m_out, num_bins=num_bins, weights=None, min_bin=min_bin, max_bin=max_bin)
        bin_center_physical, m_out_bin_medians_physical, m_out_bin_lower_q_physical, m_out_bin_upper_q_physical = pf.binned_median_quantile_log(sig_sfr[physical_mask], m_out[physical_mask], num_bins=num_bins, weights=None, min_bin=min_bin, max_bin=max_bin)
        bin_center_strong, m_out_bin_medians_strong, m_out_bin_lower_q_strong, m_out_bin_upper_q_strong = pf.binned_median_quantile_log(sig_sfr[BIC_diff_strong], m_out[BIC_diff_strong], num_bins=num_bins, weights=None, min_bin=min_bin, max_bin=max_bin)


    elif weighted_average == True:
        bin_center_all, m_out_bin_medians_all, m_out_bin_lower_q_all, m_out_bin_upper_q_all = pf.binned_median_quantile_log(sig_sfr, m_out, num_bins=num_bins, weights=[vel_out_err], min_bin=min_bin, max_bin=max_bin)
        bin_center_physical, m_out_bin_medians_physical, m_out_bin_lower_q_physical, m_out_bin_upper_q_physical = pf.binned_median_quantile_log(sig_sfr[physical_mask], m_out[physical_mask], num_bins=num_bins, weights=[vel_out_err], min_bin=min_bin, max_bin=max_bin)
        bin_center_strong, m_out_bin_medians_strong, m_out_bin_lower_q_strong, m_out_bin_upper_q_strong = pf.binned_median_quantile_log(sig_sfr[BIC_diff_strong], m_out[BIC_diff_strong], num_bins=num_bins, weights=[vel_out_err], min_bin=min_bin, max_bin=max_bin)


    #calculate the r value for the median values
    r_m_out_med_all, p_value_m_out_all = pf.pearson_correlation(bin_center_all, m_out_bin_medians_all)
    r_m_out_med_physical, p_value_m_out_physical = pf.pearson_correlation(bin_center_physical, m_out_bin_medians_physical)
    r_m_out_med_strong, p_value_m_out_strong = pf.pearson_correlation(bin_center_strong, m_out_bin_medians_strong)

    #calculate the r value for all the values
    r_m_out_all, p_value_m_out_all = pf.pearson_correlation(sig_sfr[~np.isnan(m_out)], m_out[~np.isnan(m_out)])
    r_m_out_physical, p_value_m_out_physical = pf.pearson_correlation(sig_sfr[~np.isnan(m_out)&physical_mask], m_out[~np.isnan(m_out)&physical_mask])
    r_m_out_strong, p_value_m_out_strong = pf.pearson_correlation(sig_sfr[~np.isnan(m_out)&BIC_diff_strong], m_out[~np.isnan(m_out)&BIC_diff_strong])

    #calculate Kim et al. trend
    #sfr_surface_density_kim, mlf_Kim = kim_et_al_2020(sig_sfr.min(), sig_sfr.max(), scale_factor=0.06) #scale_factor=abs(np.nanmedian(mlf)/(np.nanmedian(sig_sfr)**-0.44)))


    #print average numbers for the different panels
    print('Number of spaxels in the first panel', m_out.shape)
    print('All spaxels median m_out:', np.nanmedian(m_out))
    print('All spaxels standard deviation m_out:', np.nanstd(m_out))
    print('')

    print('Number of spaxels with broad sigmas at the instrument dispersion:', m_out[vel_disp<=51].shape)
    print('')
    print('Number of spaxels beyond R_90:', m_out[radius>6.1].shape)
    print('')
    print('Number of spaxels in the middle panel:', m_out[physical_mask].shape)
    print('')

    print('Physical spaxels median m_out:', np.nanmedian(m_out[physical_mask]))
    print('Physical spaxels standard deviation m_out:', np.nanstd(m_out[physical_mask]))
    print('')

    print('Number of spaxels with strong BIC differences:', m_out[BIC_diff_strong].shape)
    print('')

    print('Clean spaxels median m_out:', np.nanmedian(m_out[BIC_diff_strong]))
    print('Clean spaxels standard deviation m_out:', np.nanstd(m_out[BIC_diff_strong]))
    print('')

    #-------
    #plot it
    #-------
    plt.rcParams.update(pf.get_rc_params())
    fig, ax = plt.subplots(nrows=1, ncols=3, sharey=True, sharex=True, figsize=(10,4), constrained_layout=True)

    #get colours from cmasher
    colours = cmr.take_cmap_colors('cmr.gem', 3, cmap_range=(0.25, 0.85), return_fmt='hex')

    #plot all points
    ax[0].fill_between(bin_center_all, m_out_bin_lower_q_all, m_out_bin_upper_q_all, color=colours[0], alpha=0.3)
    ax[0].scatter(sig_sfr, m_out, marker='o', s=10, label='All KOFFEE fits; R={:.2f}'.format(r_m_out_all), color=colours[0], alpha=0.8)
    ax[0].plot(bin_center_all, m_out_bin_medians_all, marker='', lw=3, label='Median all KOFFEE fits; R={:.2f}'.format(r_m_out_med_all), color=colours[0])

    #ax[0].plot(sfr_surface_density_kim, np.log10(mlf_Kim), ':k', label='Kim+20, $\eta \propto \Sigma_{SFR}^{-0.44}$')

    ax[0].errorbar(0.03, np.nanmin(m_out)-0.1, xerr=np.nanmedian(sig_sfr_err), yerr=[[np.nanmedian(m_out_err_min)], [np.nanmedian(m_out_err_max)]], c='k')

    #ax[0].set_ylim(-2.4, 0.7)
    ax[0].set_xscale('log')
    ax[0].set_xlim(0.003, 2)
    lgnd = ax[0].legend(frameon=True, fontsize='small', loc='upper left', framealpha=0.5)
    lgnd.legendHandles[0]._legmarker.set_markersize(3)
    ax[0].set_ylabel('Log(Mass Outflow Rate [M$_\odot$ yr$^{-1}$])')
    ax[0].set_xlabel('$\Sigma_{SFR}$ [M$_\odot$ yr$^{-1}$ kpc$^{-2}$]')
    ax[0].set_title('all spaxels')

    #plot points within 90% radius
    ax[1].fill_between(bin_center_physical, m_out_bin_lower_q_physical, m_out_bin_upper_q_physical, color=colours[1], alpha=0.3)
    ax[1].scatter(sig_sfr[radius>6.1], m_out[radius>6.1], marker='o', s=10, label='All KOFFEE fits', edgecolors=colours[0], alpha=0.3, facecolors='none')
    ax[1].scatter(sig_sfr[vel_disp<=51], m_out[vel_disp<=51], marker='v', s=10, edgecolors=colours[0], alpha=0.3, facecolors='none')
    ax[1].scatter(sig_sfr[physical_mask], m_out[physical_mask], marker='o', s=10, label='Selected KOFFEE fits; R={:.2f}'.format(r_m_out_physical), color=colours[1], alpha=0.8)
    ax[1].plot(bin_center_physical, m_out_bin_medians_physical, marker='', lw=3, label='Median of selected KOFFEE fits; R={:.2f}'.format(r_m_out_med_physical), color=colours[1])

    #ax[1].plot(sfr_surface_density_kim, np.log10(m_out_Kim), ':k')

    ax[1].errorbar(0.03, np.nanmin(m_out)-0.1, xerr=np.nanmedian(sig_sfr_err[physical_mask]), yerr=[[np.nanmedian(m_out_err_min[physical_mask])], [np.nanmedian(m_out_err_max[physical_mask])]], c='k')

    #ax[1].set_xscale('log')
    lgnd = ax[1].legend(frameon=True, fontsize='small', loc='upper left', framealpha=0.5)
    lgnd.legendHandles[0]._legmarker.set_markersize(3)
    ax[1].set_xlabel('$\Sigma_{SFR}$ [M$_\odot$ yr$^{-1}$ kpc$^{-2}$]')
    ax[1].set_title(r'$r$<$r_{90}$ and $\sigma_{broad}$>$\sigma_{inst}$')

    #plot points with strong BIC values
    ax[2].fill_between(bin_center_strong, m_out_bin_lower_q_strong, m_out_bin_upper_q_strong, color=colours[2], alpha=0.3)
    ax[2].scatter(sig_sfr[~BIC_diff_strong], m_out[~BIC_diff_strong], marker='o', s=10, label='All KOFFEE fits', color=colours[0], alpha=0.3, facecolors='none')
    ax[2].scatter(sig_sfr[BIC_diff_strong], m_out[BIC_diff_strong], marker='o', s=10, label='Selected KOFFEE fits; R={:.2f}'.format(r_m_out_strong), color=colours[2], alpha=1.0)
    ax[2].plot(bin_center_strong, m_out_bin_medians_strong, marker='', lw=3, label='Median of selected KOFFEE fits; R={:.2f}'.format(r_m_out_med_strong), color=colours[2])

    #ax[2].plot(sfr_surface_density_kim, np.log10(mlf_Kim), ':k')

    ax[2].errorbar(0.03, np.nanmin(m_out)-0.1, xerr=np.nanmedian(sig_sfr_err[BIC_diff_strong]), yerr=[[np.nanmedian(m_out_err_min[BIC_diff_strong])], [np.nanmedian(m_out_err_max[BIC_diff_strong])]], c='k')

    #ax[1].set_xscale('log')
    lgnd = ax[2].legend(frameon=True, fontsize='small', loc='upper left', framealpha=0.5)
    lgnd.legendHandles[0]._legmarker.set_markersize(3)
    ax[2].set_xlabel('$\Sigma_{SFR}$ [M$_\odot$ yr$^{-1}$ kpc$^{-2}$]')
    ax[2].set_title('strongly likely BIC')

    plt.show()



def plot_sfr_momentum_out(OIII_outflow_results, OIII_outflow_error, hbeta_outflow_results, hbeta_outflow_error, hbeta_no_outflow_results, hbeta_no_outflow_error, BIC_outflow, BIC_no_outflow, statistical_results, z, radius, weighted_average=True):
    """
    Plots the SFR surface density against the outflow momentum rate, with Sigma_SFR
    calculated using only the narrow component.
    Momentum is defined here as m_out*v_out [M_sun km yr^-2]

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
    A three panel figure of the momentum outflow rate against the SFR surface density

    """
    #calculate the sfr surface density - using just the systemic line, and including the flux line
    #don't include extinction since this was included in the continuum subtraction using ppxf
    sfr, sfr_err, total_sfr, sfr_surface_density, sfr_surface_density_err = calc_sfr.calc_sfr_koffee(hbeta_outflow_results, hbeta_outflow_error, hbeta_no_outflow_results, hbeta_no_outflow_error, statistical_results, z, include_extinction=False, include_outflow=False)

    #calculate the mass outflow rate (in g/s)
    m_out, m_out_max, m_out_min = calc_mlf.calc_mass_outflow_rate(OIII_outflow_results, OIII_outflow_error, hbeta_outflow_results, hbeta_outflow_error, statistical_results, z)

    #convert to solar masses per year
    m_out = m_out.to(u.solMass/u.yr)
    m_out_max = m_out_max.to(u.solMass/u.yr)
    m_out_min = m_out_min.to(u.solMass/u.yr)

    #calculate the velocity dispersion for the masking, vel_out for the momentum
    vel_disp, vel_disp_err, vel_diff, vel_diff_err, vel_out, vel_out_err = calc_outvel.calc_outflow_vel(OIII_outflow_results, OIII_outflow_error, statistical_results, z)

    #multiply mass outflow rate by velocity
    momentum = m_out * (vel_out*(u.km/u.s))

    momentum = momentum.to(u.solMass*u.km/(u.yr**2))

    #make the mask for the outflow spaxels
    flow_mask = (statistical_results>0)# & (sfr_surface_density>0.1)

    #flatten all the arrays and get rid of extra spaxels
    sig_sfr = sfr_surface_density[flow_mask]
    sig_sfr_err = sfr_surface_density_err[flow_mask]
    momentum = momentum[flow_mask]
    #m_out_max = m_out_max[flow_mask]
    #m_out_min = m_out_min[flow_mask]
    BIC_outflow = BIC_outflow[flow_mask]
    BIC_no_outflow = BIC_no_outflow[flow_mask]
    vel_disp = vel_disp[flow_mask]
    radius = radius[flow_mask]

    #take the log of the mlf
    momentum = np.log10(momentum.value)

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
        bin_center_all, momentum_bin_medians_all, momentum_bin_lower_q_all, momentum_bin_upper_q_all = pf.binned_median_quantile_log(sig_sfr, momentum, num_bins=num_bins, weights=None, min_bin=min_bin, max_bin=max_bin)
        bin_center_physical, momentum_bin_medians_physical, momentum_bin_lower_q_physical, momentum_bin_upper_q_physical = pf.binned_median_quantile_log(sig_sfr[physical_mask], momentum[physical_mask], num_bins=num_bins, weights=None, min_bin=min_bin, max_bin=max_bin)
        bin_center_strong, momentum_bin_medians_strong, momentum_bin_lower_q_strong, momentum_bin_upper_q_strong = pf.binned_median_quantile_log(sig_sfr[BIC_diff_strong], momentum[BIC_diff_strong], num_bins=num_bins, weights=None, min_bin=min_bin, max_bin=max_bin)


    elif weighted_average == True:
        bin_center_all, momentum_bin_medians_all, momentum_bin_lower_q_all, momentum_bin_upper_q_all = pf.binned_median_quantile_log(sig_sfr, momentum, num_bins=num_bins, weights=[vel_out_err], min_bin=min_bin, max_bin=max_bin)
        bin_center_physical, momentum_bin_medians_physical, momentum_bin_lower_q_physical, momentum_bin_upper_q_physical = pf.binned_median_quantile_log(sig_sfr[physical_mask], momentum[physical_mask], num_bins=num_bins, weights=[vel_out_err], min_bin=min_bin, max_bin=max_bin)
        bin_center_strong, momentum_bin_medians_strong, momentum_bin_lower_q_strong, momentum_bin_upper_q_strong = pf.binned_median_quantile_log(sig_sfr[BIC_diff_strong], momentum[BIC_diff_strong], num_bins=num_bins, weights=[vel_out_err], min_bin=min_bin, max_bin=max_bin)


    #calculate the r value for the median values
    r_momentum_med_all, p_value_momentum_all = pf.pearson_correlation(bin_center_all, momentum_bin_medians_all)
    r_momentum_med_physical, p_value_momentum_physical = pf.pearson_correlation(bin_center_physical, momentum_bin_medians_physical)
    r_momentum_med_strong, p_value_momentum_strong = pf.pearson_correlation(bin_center_strong, momentum_bin_medians_strong)

    #calculate the r value for all the values
    r_momentum_all, p_value_momentum_all = pf.pearson_correlation(sig_sfr[~np.isnan(momentum)], momentum[~np.isnan(momentum)])
    r_momentum_physical, p_value_momentum_physical = pf.pearson_correlation(sig_sfr[~np.isnan(momentum)&physical_mask], momentum[~np.isnan(momentum)&physical_mask])
    r_momentum_strong, p_value_momentum_strong = pf.pearson_correlation(sig_sfr[~np.isnan(momentum)&BIC_diff_strong], momentum[~np.isnan(momentum)&BIC_diff_strong])


    #print average numbers for the different panels
    print('Number of spaxels in the first panel', momentum.shape)
    print('All spaxels median momentum:', np.nanmedian(momentum))
    print('All spaxels standard deviation momentum:', np.nanstd(momentum))
    print('')

    print('Number of spaxels with broad sigmas at the instrument dispersion:', momentum[vel_disp<=51].shape)
    print('')
    print('Number of spaxels beyond R_90:', momentum[radius>6.1].shape)
    print('')
    print('Number of spaxels in the middle panel:', momentum[physical_mask].shape)
    print('')

    print('Physical spaxels median momentum:', np.nanmedian(momentum[physical_mask]))
    print('Physical spaxels standard deviation momentum:', np.nanstd(momentum[physical_mask]))
    print('')

    print('Number of spaxels with strong BIC differences:', momentum[BIC_diff_strong].shape)
    print('')

    print('Clean spaxels median momentum:', np.nanmedian(momentum[BIC_diff_strong]))
    print('Clean spaxels standard deviation momentum:', np.nanstd(momentum[BIC_diff_strong]))
    print('')

    #-------
    #plot it
    #-------
    plt.rcParams.update(pf.get_rc_params())
    fig, ax = plt.subplots(nrows=1, ncols=3, sharey=True, sharex=True, figsize=(10,4), constrained_layout=True)

    #get colours from cmasher
    colours = cmr.take_cmap_colors('cmr.gem', 3, cmap_range=(0.25, 0.85), return_fmt='hex')

    #plot all points
    ax[0].fill_between(bin_center_all, momentum_bin_lower_q_all, momentum_bin_upper_q_all, color=colours[0], alpha=0.3)
    ax[0].scatter(sig_sfr, momentum, marker='o', s=10, label='All KOFFEE fits; R={:.2f}'.format(r_momentum_all), color=colours[0], alpha=0.8)
    ax[0].plot(bin_center_all, momentum_bin_medians_all, marker='', lw=3, label='Median all KOFFEE fits; R={:.2f}'.format(r_momentum_med_all), color=colours[0])

    #ax[0].set_ylim(-2.4, 0.7)
    ax[0].set_xscale('log')
    ax[0].set_xlim(0.003, 2)
    lgnd = ax[0].legend(frameon=True, fontsize='small', loc='upper left', framealpha=0.5)
    lgnd.legendHandles[0]._legmarker.set_markersize(3)
    ax[0].set_ylabel('Log(Momentum Outflow Rate [M$_\odot$ km yr$^{-2}$])')
    ax[0].set_xlabel('$\Sigma_{SFR}$ [M$_\odot$ yr$^{-1}$ kpc$^{-2}$]')
    ax[0].set_title('all spaxels')

    #plot points within 90% radius
    ax[1].fill_between(bin_center_physical, momentum_bin_lower_q_physical, momentum_bin_upper_q_physical, color=colours[1], alpha=0.3)
    ax[1].scatter(sig_sfr[radius>6.1], momentum[radius>6.1], marker='o', s=10, label='All KOFFEE fits', edgecolors=colours[0], alpha=0.3, facecolors='none')
    ax[1].scatter(sig_sfr[vel_disp<=51], momentum[vel_disp<=51], marker='v', s=10, edgecolors=colours[0], alpha=0.3, facecolors='none')
    ax[1].scatter(sig_sfr[physical_mask], momentum[physical_mask], marker='o', s=10, label='Selected KOFFEE fits; R={:.2f}'.format(r_momentum_physical), color=colours[1], alpha=0.8)
    ax[1].plot(bin_center_physical, momentum_bin_medians_physical, marker='', lw=3, label='Median of selected KOFFEE fits; R={:.2f}'.format(r_momentum_med_physical), color=colours[1])

    #ax[1].set_xscale('log')
    lgnd = ax[1].legend(frameon=True, fontsize='small', loc='upper left', framealpha=0.5)
    lgnd.legendHandles[0]._legmarker.set_markersize(3)
    ax[1].set_xlabel('$\Sigma_{SFR}$ [M$_\odot$ yr$^{-1}$ kpc$^{-2}$]')
    ax[1].set_title(r'$r$<$r_{90}$ and $\sigma_{broad}$>$\sigma_{inst}$')

    #plot points with strong BIC values
    ax[2].fill_between(bin_center_strong, momentum_bin_lower_q_strong, momentum_bin_upper_q_strong, color=colours[2], alpha=0.3)
    ax[2].scatter(sig_sfr[~BIC_diff_strong], momentum[~BIC_diff_strong], marker='o', s=10, label='All KOFFEE fits', color=colours[0], alpha=0.3, facecolors='none')
    ax[2].scatter(sig_sfr[BIC_diff_strong], momentum[BIC_diff_strong], marker='o', s=10, label='Selected KOFFEE fits; R={:.2f}'.format(r_momentum_strong), color=colours[2], alpha=1.0)
    ax[2].plot(bin_center_strong, momentum_bin_medians_strong, marker='', lw=3, label='Median of selected KOFFEE fits; R={:.2f}'.format(r_momentum_med_strong), color=colours[2])

    #ax[1].set_xscale('log')
    lgnd = ax[2].legend(frameon=True, fontsize='small', loc='upper left', framealpha=0.5)
    lgnd.legendHandles[0]._legmarker.set_markersize(3)
    ax[2].set_xlabel('$\Sigma_{SFR}$ [M$_\odot$ yr$^{-1}$ kpc$^{-2}$]')
    ax[2].set_title('strongly likely BIC')

    plt.show()



def plot_sfr_momentum_out2(OIII_outflow_results, OIII_outflow_error, hbeta_outflow_results, hbeta_outflow_error, hbeta_no_outflow_results, hbeta_no_outflow_error, BIC_outflow, BIC_no_outflow, statistical_results, z, radius, weighted_average=True):
    """
    Plots the SFR surface density against the outflow momentum, with Sigma_SFR calculated
    using only the narrow component.
    Momentum is defined here as m_out*v_out/sfr [km/s]

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
    A three panel figure of the momentum outflow rate against the SFR surface density

    """
    #calculate the sfr surface density - using just the systemic line, and including the flux line
    #don't include extinction since this was included in the continuum subtraction using ppxf
    sfr, sfr_err, total_sfr, sfr_surface_density, sfr_surface_density_err = calc_sfr.calc_sfr_koffee(hbeta_outflow_results, hbeta_outflow_error, hbeta_no_outflow_results, hbeta_no_outflow_error, statistical_results, z, include_extinction=False, include_outflow=False)

    #calculate the mass outflow rate (in g/s)
    m_out, m_out_max, m_out_min = calc_mlf.calc_mass_outflow_rate(OIII_outflow_results, OIII_outflow_error, hbeta_outflow_results, hbeta_outflow_error, statistical_results, z)

    #convert to solar masses per year
    m_out = m_out.to(u.solMass/u.yr)
    m_out_max = m_out_max.to(u.solMass/u.yr)
    m_out_min = m_out_min.to(u.solMass/u.yr)

    #calculate the velocity dispersion for the masking, vel_out for the momentum
    vel_disp, vel_disp_err, vel_diff, vel_diff_err, vel_out, vel_out_err = calc_outvel.calc_outflow_vel(OIII_outflow_results, OIII_outflow_error, statistical_results, z)

    #multiply mass outflow rate by velocity
    momentum = (m_out * (vel_out*(u.km/u.s))) / (sfr*(u.solMass/u.yr))

    momentum = momentum.to(u.km/(u.s))

    #make the mask for the outflow spaxels
    flow_mask = (statistical_results>0)# & (sfr_surface_density>0.1)

    #flatten all the arrays and get rid of extra spaxels
    sig_sfr = sfr_surface_density[flow_mask]
    sig_sfr_err = sfr_surface_density_err[flow_mask]
    momentum = momentum[flow_mask]
    #m_out_max = m_out_max[flow_mask]
    #m_out_min = m_out_min[flow_mask]
    BIC_outflow = BIC_outflow[flow_mask]
    BIC_no_outflow = BIC_no_outflow[flow_mask]
    vel_disp = vel_disp[flow_mask]
    radius = radius[flow_mask]

    #take the log of the mlf
    momentum = np.log10(momentum.value)

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
        bin_center_all, momentum_bin_medians_all, momentum_bin_lower_q_all, momentum_bin_upper_q_all = pf.binned_median_quantile_log(sig_sfr, momentum, num_bins=num_bins, weights=None, min_bin=min_bin, max_bin=max_bin)
        bin_center_physical, momentum_bin_medians_physical, momentum_bin_lower_q_physical, momentum_bin_upper_q_physical = pf.binned_median_quantile_log(sig_sfr[physical_mask], momentum[physical_mask], num_bins=num_bins, weights=None, min_bin=min_bin, max_bin=max_bin)
        bin_center_strong, momentum_bin_medians_strong, momentum_bin_lower_q_strong, momentum_bin_upper_q_strong = pf.binned_median_quantile_log(sig_sfr[BIC_diff_strong], momentum[BIC_diff_strong], num_bins=num_bins, weights=None, min_bin=min_bin, max_bin=max_bin)


    elif weighted_average == True:
        bin_center_all, momentum_bin_medians_all, momentum_bin_lower_q_all, momentum_bin_upper_q_all = pf.binned_median_quantile_log(sig_sfr, momentum, num_bins=num_bins, weights=[vel_out_err], min_bin=min_bin, max_bin=max_bin)
        bin_center_physical, momentum_bin_medians_physical, momentum_bin_lower_q_physical, momentum_bin_upper_q_physical = pf.binned_median_quantile_log(sig_sfr[physical_mask], momentum[physical_mask], num_bins=num_bins, weights=[vel_out_err], min_bin=min_bin, max_bin=max_bin)
        bin_center_strong, momentum_bin_medians_strong, momentum_bin_lower_q_strong, momentum_bin_upper_q_strong = pf.binned_median_quantile_log(sig_sfr[BIC_diff_strong], momentum[BIC_diff_strong], num_bins=num_bins, weights=[vel_out_err], min_bin=min_bin, max_bin=max_bin)


    #calculate the r value for the median values
    r_momentum_med_all, p_value_momentum_all = pf.pearson_correlation(bin_center_all, momentum_bin_medians_all)
    r_momentum_med_physical, p_value_momentum_physical = pf.pearson_correlation(bin_center_physical, momentum_bin_medians_physical)
    r_momentum_med_strong, p_value_momentum_strong = pf.pearson_correlation(bin_center_strong, momentum_bin_medians_strong)

    #calculate the r value for all the values
    r_momentum_all, p_value_momentum_all = pf.pearson_correlation(sig_sfr[~np.isnan(momentum)], momentum[~np.isnan(momentum)])
    r_momentum_physical, p_value_momentum_physical = pf.pearson_correlation(sig_sfr[~np.isnan(momentum)&physical_mask], momentum[~np.isnan(momentum)&physical_mask])
    r_momentum_strong, p_value_momentum_strong = pf.pearson_correlation(sig_sfr[~np.isnan(momentum)&BIC_diff_strong], momentum[~np.isnan(momentum)&BIC_diff_strong])


    #print average numbers for the different panels
    print('Number of spaxels in the first panel', momentum.shape)
    print('All spaxels median momentum:', np.nanmedian(momentum))
    print('All spaxels standard deviation momentum:', np.nanstd(momentum))
    print('')

    print('Number of spaxels with broad sigmas at the instrument dispersion:', momentum[vel_disp<=51].shape)
    print('')
    print('Number of spaxels beyond R_90:', momentum[radius>6.1].shape)
    print('')
    print('Number of spaxels in the middle panel:', momentum[physical_mask].shape)
    print('')

    print('Physical spaxels median momentum:', np.nanmedian(momentum[physical_mask]))
    print('Physical spaxels standard deviation momentum:', np.nanstd(momentum[physical_mask]))
    print('')

    print('Number of spaxels with strong BIC differences:', momentum[BIC_diff_strong].shape)
    print('')

    print('Clean spaxels median momentum:', np.nanmedian(momentum[BIC_diff_strong]))
    print('Clean spaxels standard deviation momentum:', np.nanstd(momentum[BIC_diff_strong]))
    print('')

    #-------
    #plot it
    #-------
    plt.rcParams.update(pf.get_rc_params())
    fig, ax = plt.subplots(nrows=1, ncols=3, sharey=True, sharex=True, figsize=(10,4), constrained_layout=True)

    #get colours from cmasher
    colours = cmr.take_cmap_colors('cmr.gem', 3, cmap_range=(0.25, 0.85), return_fmt='hex')

    #plot all points
    ax[0].fill_between(bin_center_all, momentum_bin_lower_q_all, momentum_bin_upper_q_all, color=colours[0], alpha=0.3)
    ax[0].scatter(sig_sfr, momentum, marker='o', s=10, label='All KOFFEE fits; R={:.2f}'.format(r_momentum_all), color=colours[0], alpha=0.8)
    ax[0].plot(bin_center_all, momentum_bin_medians_all, marker='', lw=3, label='Median all KOFFEE fits; R={:.2f}'.format(r_momentum_med_all), color=colours[0])

    #ax[0].set_ylim(-2.4, 0.7)
    ax[0].set_xscale('log')
    ax[0].set_xlim(0.003, 2)
    lgnd = ax[0].legend(frameon=True, fontsize='small', loc='upper left', framealpha=0.5)
    lgnd.legendHandles[0]._legmarker.set_markersize(3)
    ax[0].set_ylabel('Log(Momentum Outflow Rate/SFR [km s$^{-1}$])')
    ax[0].set_xlabel('$\Sigma_{SFR}$ [M$_\odot$ yr$^{-1}$ kpc$^{-2}$]')
    ax[0].set_title('all spaxels')

    #plot points within 90% radius
    ax[1].fill_between(bin_center_physical, momentum_bin_lower_q_physical, momentum_bin_upper_q_physical, color=colours[1], alpha=0.3)
    ax[1].scatter(sig_sfr[radius>6.1], momentum[radius>6.1], marker='o', s=10, label='All KOFFEE fits', edgecolors=colours[0], alpha=0.3, facecolors='none')
    ax[1].scatter(sig_sfr[vel_disp<=51], momentum[vel_disp<=51], marker='v', s=10, edgecolors=colours[0], alpha=0.3, facecolors='none')
    ax[1].scatter(sig_sfr[physical_mask], momentum[physical_mask], marker='o', s=10, label='Selected KOFFEE fits; R={:.2f}'.format(r_momentum_physical), color=colours[1], alpha=0.8)
    ax[1].plot(bin_center_physical, momentum_bin_medians_physical, marker='', lw=3, label='Median of selected KOFFEE fits; R={:.2f}'.format(r_momentum_med_physical), color=colours[1])

    #ax[1].set_xscale('log')
    lgnd = ax[1].legend(frameon=True, fontsize='small', loc='upper left', framealpha=0.5)
    lgnd.legendHandles[0]._legmarker.set_markersize(3)
    ax[1].set_xlabel('$\Sigma_{SFR}$ [M$_\odot$ yr$^{-1}$ kpc$^{-2}$]')
    ax[1].set_title(r'$r$<$r_{90}$ and $\sigma_{broad}$>$\sigma_{inst}$')

    #plot points with strong BIC values
    ax[2].fill_between(bin_center_strong, momentum_bin_lower_q_strong, momentum_bin_upper_q_strong, color=colours[2], alpha=0.3)
    ax[2].scatter(sig_sfr[~BIC_diff_strong], momentum[~BIC_diff_strong], marker='o', s=10, label='All KOFFEE fits', color=colours[0], alpha=0.3, facecolors='none')
    ax[2].scatter(sig_sfr[BIC_diff_strong], momentum[BIC_diff_strong], marker='o', s=10, label='Selected KOFFEE fits; R={:.2f}'.format(r_momentum_strong), color=colours[2], alpha=1.0)
    ax[2].plot(bin_center_strong, momentum_bin_medians_strong, marker='', lw=3, label='Median of selected KOFFEE fits; R={:.2f}'.format(r_momentum_med_strong), color=colours[2])

    #ax[1].set_xscale('log')
    lgnd = ax[2].legend(frameon=True, fontsize='small', loc='upper left', framealpha=0.5)
    lgnd.legendHandles[0]._legmarker.set_markersize(3)
    ax[2].set_xlabel('$\Sigma_{SFR}$ [M$_\odot$ yr$^{-1}$ kpc$^{-2}$]')
    ax[2].set_title('strongly likely BIC')

    plt.show()



def plot_radius_mlf(OIII_outflow_results, OIII_outflow_error, hbeta_outflow_results, hbeta_outflow_error, hbeta_no_outflow_results, hbeta_no_outflow_error, BIC_outflow, BIC_no_outflow, statistical_results, z, radius, weighted_average=True):
    """
    Plots the mass loading factor against galaxy radius.

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

    weighted_average : boolean
        whether or not to take a weighted average using the errors (Default=True)

    Returns
    -------
    A three panel figure of the mass loading factor against the galaxy radius

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
    num_bins = 4
    min_bin = None #-0.05
    max_bin = None #0.6

    if weighted_average == False:
        bin_center_all, mlf_bin_medians_all, mlf_bin_lower_q_all, mlf_bin_upper_q_all = pf.binned_median_quantile_lin(radius, mlf, num_bins=num_bins, weights=None, min_bin=min_bin, max_bin=max_bin)
        bin_center_physical, mlf_bin_medians_physical, mlf_bin_lower_q_physical, mlf_bin_upper_q_physical = pf.binned_median_quantile_lin(radius[physical_mask], mlf[physical_mask], num_bins=num_bins, weights=None, min_bin=min_bin, max_bin=max_bin)
        bin_center_strong, mlf_bin_medians_strong, mlf_bin_lower_q_strong, mlf_bin_upper_q_strong = pf.binned_median_quantile_lin(radius[BIC_diff_strong], mlf[BIC_diff_strong], num_bins=num_bins, weights=None, min_bin=min_bin, max_bin=max_bin)


    elif weighted_average == True:
        bin_center_all, mlf_bin_medians_all, mlf_bin_lower_q_all, mlf_bin_upper_q_all = pf.binned_median_quantile_lin(radius, mlf, num_bins=num_bins, weights=[vel_out_err], min_bin=min_bin, max_bin=max_bin)
        bin_center_physical, mlf_bin_medians_physical, mlf_bin_lower_q_physical, mlf_bin_upper_q_physical = pf.binned_median_quantile_lin(radius[physical_mask], mlf[physical_mask], num_bins=num_bins, weights=[vel_out_err], min_bin=min_bin, max_bin=max_bin)
        bin_center_strong, mlf_bin_medians_strong, mlf_bin_lower_q_strong, mlf_bin_upper_q_strong = pf.binned_median_quantile_lin(radius[BIC_diff_strong], mlf[BIC_diff_strong], num_bins=num_bins, weights=[vel_out_err], min_bin=min_bin, max_bin=max_bin)


    #calculate the r value for the median values
    r_mlf_med_all, p_value_mlf_all = pf.pearson_correlation(bin_center_all, mlf_bin_medians_all)
    r_mlf_med_physical, p_value_mlf_physical = pf.pearson_correlation(bin_center_physical, mlf_bin_medians_physical)
    r_mlf_med_strong, p_value_mlf_strong = pf.pearson_correlation(bin_center_strong, mlf_bin_medians_strong)

    #calculate the r value for all the values
    r_mlf_all, p_value_mlf_all = pf.pearson_correlation(radius[~np.isnan(mlf)], mlf[~np.isnan(mlf)])
    r_mlf_physical, p_value_mlf_physical = pf.pearson_correlation(radius[~np.isnan(mlf)&physical_mask], mlf[~np.isnan(mlf)&physical_mask])
    r_mlf_strong, p_value_mlf_strong = pf.pearson_correlation(radius[~np.isnan(mlf)&BIC_diff_strong], mlf[~np.isnan(mlf)&BIC_diff_strong])


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
    #ax[0].errorbar(radius, mlf, xerr=vel_out_err, yerr=[mlf-mlf_min, mlf_max-mlf], fmt='o', ms=3, color=colours[0], alpha=0.6, label='All KOFFEE fits; R={:.2f}'.format(r_mlf_all))
    ax[0].fill_between(bin_center_all, mlf_bin_lower_q_all, mlf_bin_upper_q_all, color=colours[0], alpha=0.3)
    ax[0].scatter(radius, mlf, marker='o', s=10, label='All KOFFEE fits; R={:.2f}'.format(r_mlf_all), color=colours[0], alpha=0.8)
    ax[0].plot(bin_center_all, mlf_bin_medians_all, marker='', lw=3, label='Median all KOFFEE fits; R={:.2f}'.format(r_mlf_med_all), color=colours[0])

    ax[0].errorbar(7.5, np.nanmin(mlf), xerr=0.7, yerr=[[np.nanmedian(mlf_err_min)], [np.nanmedian(mlf_err_max)]], c='k')

    #ax[0].set_ylim(100, 500)
    #ax[0].set_xscale('log')
    #ax[0].set_xlim(0.003, 10)
    lgnd = ax[0].legend(frameon=True, fontsize='small', loc='lower left', framealpha=0.5, edgecolor='white')
    lgnd.legendHandles[0]._legmarker.set_markersize(3)
    ax[0].set_ylabel('Log(Mass Loading Factor)')
    ax[0].set_xlabel('Radius [Arcsec]')
    ax[0].set_title('all spaxels')

    #plot points within 90% radius
    ax[1].fill_between(bin_center_physical, mlf_bin_lower_q_physical, mlf_bin_upper_q_physical, color=colours[1], alpha=0.3)
    ax[1].scatter(radius[radius>6.1], mlf[radius>6.1], marker='o', s=10, label='All KOFFEE fits', edgecolors=colours[0], alpha=0.3, facecolors='none')
    ax[1].scatter(radius[vel_disp<=51], mlf[vel_disp<=51], marker='v', s=10, edgecolors=colours[0], alpha=0.3, facecolors='none')
    ax[1].scatter(radius[physical_mask], mlf[physical_mask], marker='o', s=10, label='Selected KOFFEE fits; R={:.2f}'.format(r_mlf_physical), color=colours[1], alpha=0.8)
    ax[1].plot(bin_center_physical, mlf_bin_medians_physical, marker='', lw=3, label='Median of selected KOFFEE fits;\n R={:.2f}'.format(r_mlf_med_physical), color=colours[1])

    ax[1].errorbar(7.5, np.nanmin(mlf), xerr=0.7, yerr=[[np.nanmedian(mlf_err_min[physical_mask])], [np.nanmedian(mlf_err_max[physical_mask])]], c='k')

    #ax[1].set_xscale('log')
    lgnd = ax[1].legend(frameon=True, fontsize='small', loc='lower left', framealpha=0.5, edgecolor='white')
    lgnd.legendHandles[0]._legmarker.set_markersize(3)
    ax[1].set_xlabel('Radius [Arcsec]')
    ax[1].set_title(r'$r$<$r_{90}$ and $\sigma_{broad}$>$\sigma_{inst}$')

    #plot points with strong BIC values
    ax[2].fill_between(bin_center_strong, mlf_bin_lower_q_strong, mlf_bin_upper_q_strong, color=colours[2], alpha=0.3)
    ax[2].scatter(radius[~BIC_diff_strong], mlf[~BIC_diff_strong], marker='o', s=10, label='All KOFFEE fits', color=colours[0], alpha=0.3, facecolors='none')
    ax[2].scatter(radius[BIC_diff_strong], mlf[BIC_diff_strong], marker='o', s=10, label='Selected KOFFEE fits; R={:.2f}'.format(r_mlf_strong), color=colours[2], alpha=1.0)
    ax[2].plot(bin_center_strong, mlf_bin_medians_strong, marker='', lw=3, label='Median of selected KOFFEE fits;\n R={:.2f}'.format(r_mlf_med_strong), color=colours[2])

    ax[2].errorbar(7.5, np.nanmin(mlf), xerr=0.7, yerr=[[np.nanmedian(mlf_err_min[BIC_diff_strong])], [np.nanmedian(mlf_err_max[BIC_diff_strong])]], c='k')

    #ax[1].set_xscale('log')
    lgnd = ax[2].legend(frameon=True, fontsize='small', loc='lower left', framealpha=0.5, edgecolor='white')
    lgnd.legendHandles[0]._legmarker.set_markersize(3)
    ax[2].set_xlabel('Radius [Arcsec]')
    ax[2].set_title('strongly likely BIC')

    plt.show()




def plot_flux_mlf(flux_outflow_results, flux_outflow_error, OIII_outflow_results, OIII_outflow_error, hbeta_outflow_results, hbeta_outflow_error, hbeta_no_outflow_results, hbeta_no_outflow_error, BIC_outflow, BIC_no_outflow, statistical_results, z, radius, header, flux_ratio_line='Hbeta', weighted_average=True):
    """
    Plots the broad-to-narrow flux ratio against the mass loading factor.

    Parameters
    ----------
    flux_outflow_results : :obj:'~numpy.ndarray'
        array of outflow results from KOFFEE for the emission line for which we
        want to do the ratio.  Should be (7, statistical_results.shape)

    flux_outflow_error : :obj:'~numpy.ndarray'
        array of the outflow result errors from KOFFEE for the emission line for
        which we want to do the ratio.

    OIII_outflow_results : :obj:'~numpy.ndarray'
        array of outflow results from KOFFEE for OIII line.  Used to calculate
        the outflow velocity.  Should be (7, statistical_results.shape)

    OIII_outflow_err : :obj:'~numpy.ndarray'
        array of the outflow result errors from KOFFEE for OIII line

    hbeta_outflow_results : :obj:'~numpy.ndarray'
        array of outflow results from KOFFEE for Hbeta line.  Used to calculate
        the Sigma SFR.  Should be (7, statistical_results.shape)

    hbeta_outflow_error : :obj:'~numpy.ndarray'
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
    A three panel figure of the broad-to-narrow flux ratio against the mass
    loading factor

    """
    #calculate the mass loading factor
    mlf, mlf_max, mlf_min = calc_mlf.calc_mass_loading_factor(OIII_outflow_results, OIII_outflow_error, hbeta_outflow_results, hbeta_outflow_error, hbeta_no_outflow_results, hbeta_no_outflow_error, statistical_results, z, header)

    #calculate the flux for systematic and flow gaussians
    systemic_flux, systemic_flux_err, outflow_flux, outflow_flux_err = calc_sfr.calc_flux_from_koffee(flux_outflow_results, flux_outflow_error, statistical_results, z, outflow=True)

    #calculate the velocity dispersion for the masking
    vel_disp, vel_disp_err, vel_diff, vel_diff_err, vel_out, vel_out_err = calc_outvel.calc_outflow_vel(OIII_outflow_results, OIII_outflow_error, statistical_results, z)

    #make the mask for the outflow spaxels
    #create the flow mask
    if flux_ratio_line == 'Hbeta':
        flow_mask = (statistical_results>0) & (np.isnan(flux_outflow_results[3,:,:])==False)
        print('For H$\beta$ the number of outflow fitted spaxels is:', flow_mask[flow_mask].shape)
    else:
        flow_mask = (statistical_results>0)

    #flatten all the arrays and get rid of extra spaxels
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

    #make sure none of the errors are nan values
    systemic_flux_err[np.where(np.isnan(systemic_flux_err)==True)] = np.nanmedian(systemic_flux_err)
    outflow_flux_err[np.where(np.isnan(outflow_flux_err)==True)] = np.nanmedian(outflow_flux_err)

    #take the log and do the flux ratio
    flux_ratio = np.log10(outflow_flux/systemic_flux)

    #calculate the error
    flux_error = flux_ratio * np.log10(np.sqrt((outflow_flux_err/outflow_flux)**2 + (systemic_flux_err/systemic_flux)**2))

    #take the log of the mlf
    mlf = np.log10(mlf)
    mlf_max = np.log10(mlf_max)
    mlf_min = np.log10(mlf_min)

    #calculate the errors
    mlf_err_max = mlf_max - mlf
    mlf_err_min = mlf - mlf_min

    #create BIC diff
    BIC_diff = BIC_outflow - BIC_no_outflow
    BIC_diff_strong = (BIC_diff < -2000)

    #physical limits mask -
    #for the radius mask 6.1" is the 90% radius
    #also mask out the fits which lie on the lower limit of dispersion < 51km/s
    physical_mask = (radius < 6.1) & (vel_disp>51)

    #do the calculations for all the bins
    num_bins = 3
    min_bin = None #-0.05
    max_bin = None #0.6

    if weighted_average == False:
        linspace_all, bin_center_all, mlf_bin_medians_all, mlf_bin_lower_q_all, mlf_bin_upper_q_all, mlf_bin_stdev_all = pf.binned_median_quantile_lin(flux_ratio, mlf, num_bins=num_bins, weights=None, min_bin=min_bin, max_bin=max_bin)
        linspace_physical, bin_center_physical, mlf_bin_medians_physical, mlf_bin_lower_q_physical, mlf_bin_upper_q_physical, mlf_bin_stdev_physical = pf.binned_median_quantile_lin(flux_ratio[physical_mask], mlf[physical_mask], num_bins=num_bins, weights=None, min_bin=min_bin, max_bin=max_bin)
        linspace_strong, bin_center_strong, mlf_bin_medians_strong, mlf_bin_lower_q_strong, mlf_bin_upper_q_strong, mlf_bin_stdev_strong = pf.binned_median_quantile_lin(flux_ratio[BIC_diff_strong], mlf[BIC_diff_strong], num_bins=num_bins, weights=None, min_bin=min_bin, max_bin=max_bin)


    elif weighted_average == True:
        linspace_all, bin_center_all, mlf_bin_medians_all, mlf_bin_lower_q_all, mlf_bin_upper_q_all, mlf_bin_stdev_all = pf.binned_median_quantile_lin(flux_ratio, mlf, num_bins=num_bins, weights=[vel_out_err], min_bin=min_bin, max_bin=max_bin)
        linspace_physical, bin_center_physical, mlf_bin_medians_physical, mlf_bin_lower_q_physical, mlf_bin_upper_q_physical, mlf_bin_stdev_physical = pf.binned_median_quantile_lin(flux_ratio[physical_mask], mlf[physical_mask], num_bins=num_bins, weights=[vel_out_err], min_bin=min_bin, max_bin=max_bin)
        linspace_strong, bin_center_strong, mlf_bin_medians_strong, mlf_bin_lower_q_strong, mlf_bin_upper_q_strong, mlf_bin_stdev_strong = pf.binned_median_quantile_lin(flux_ratio[BIC_diff_strong], mlf[BIC_diff_strong], num_bins=num_bins, weights=[vel_out_err], min_bin=min_bin, max_bin=max_bin)


    #calculate the r value for the median values
    r_mlf_med_all, p_value_mlf_all = pf.pearson_correlation(bin_center_all, mlf_bin_medians_all)
    r_mlf_med_physical, p_value_mlf_physical = pf.pearson_correlation(bin_center_physical, mlf_bin_medians_physical)
    r_mlf_med_strong, p_value_mlf_strong = pf.pearson_correlation(bin_center_strong, mlf_bin_medians_strong)

    #calculate the r value for all the values
    r_mlf_all, p_value_mlf_all = pf.pearson_correlation(flux_ratio[~np.isnan(mlf)], mlf[~np.isnan(mlf)])
    r_mlf_physical, p_value_mlf_physical = pf.pearson_correlation(flux_ratio[~np.isnan(mlf)&physical_mask], mlf[~np.isnan(mlf)&physical_mask])
    r_mlf_strong, p_value_mlf_strong = pf.pearson_correlation(flux_ratio[~np.isnan(mlf)&BIC_diff_strong], mlf[~np.isnan(mlf)&BIC_diff_strong])


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
    #ax[0].errorbar(flux_ratio, mlf, xerr=vel_out_err, yerr=[mlf-mlf_min, mlf_max-mlf], fmt='o', ms=3, color=colours[0], alpha=0.6, label='All KOFFEE fits; R={:.2f}'.format(r_mlf_all))
    ax[0].fill_between(bin_center_all, mlf_bin_lower_q_all, mlf_bin_upper_q_all, color=colours[0], alpha=0.3)
    ax[0].scatter(flux_ratio, mlf, marker='o', s=10, label='All KOFFEE fits; R={:.2f}'.format(r_mlf_all), color=colours[0], alpha=0.8)
    ax[0].plot(bin_center_all, mlf_bin_medians_all, marker='', lw=3, label='Median all KOFFEE fits; R={:.2f}'.format(r_mlf_med_all), color=colours[0])

    ax[0].errorbar(0.25, np.nanmin(mlf), xerr=np.nanmedian(flux_error), yerr=[[np.nanmedian(mlf_err_min)], [np.nanmedian(mlf_err_max)]], c='k')

    ax[0].set_ylim((np.nanmin(mlf)+np.nanmedian(mlf_err_max)-0.1), np.nanmax(mlf)+0.5)
    #ax[0].set_xlim(0.003, 10)
    lgnd = ax[0].legend(frameon=True, fontsize='small', loc='upper left', framealpha=0.5, edgecolor='white')
    lgnd.legendHandles[0]._legmarker.set_markersize(3)
    ax[0].set_ylabel('Log(Mass Loading Factor)')
    ax[0].set_xlabel(flux_ratio_line+' Log(F$_{broad}$/F$_{narrow}$)')
    ax[0].set_title('all spaxels')

    #plot points within 90% radius
    ax[1].fill_between(bin_center_physical, mlf_bin_lower_q_physical, mlf_bin_upper_q_physical, color=colours[1], alpha=0.3)
    ax[1].scatter(flux_ratio[radius>6.1], mlf[radius>6.1], marker='o', s=10, label='All KOFFEE fits', edgecolors=colours[0], alpha=0.3, facecolors='none')
    ax[1].scatter(flux_ratio[vel_disp<=51], mlf[vel_disp<=51], marker='v', s=10, edgecolors=colours[0], alpha=0.3, facecolors='none')
    ax[1].scatter(flux_ratio[physical_mask], mlf[physical_mask], marker='o', s=10, label='Selected KOFFEE fits; R={:.2f}'.format(r_mlf_physical), color=colours[1], alpha=0.8)
    ax[1].plot(bin_center_physical, mlf_bin_medians_physical, marker='', lw=3, label='Median of selected KOFFEE fits; R={:.2f}'.format(r_mlf_med_physical), color=colours[1])

    ax[1].errorbar(0.25, np.nanmin(mlf), xerr=np.nanmedian(flux_error), yerr=[[np.nanmedian(mlf_err_min[physical_mask])], [np.nanmedian(mlf_err_max[physical_mask])]], c='k')

    #ax[1].set_xscale('log')
    lgnd = ax[1].legend(frameon=True, fontsize='small', loc='upper left', framealpha=0.5, edgecolor='white')
    lgnd.legendHandles[0]._legmarker.set_markersize(3)
    ax[1].set_xlabel(flux_ratio_line+' Log(F$_{broad}$/F$_{narrow}$)')
    ax[1].set_title(r'$r$<$r_{90}$ and $\sigma_{broad}$>$\sigma_{inst}$')

    #plot points with strong BIC values
    ax[2].fill_between(bin_center_strong, mlf_bin_lower_q_strong, mlf_bin_upper_q_strong, color=colours[2], alpha=0.3)
    ax[2].scatter(flux_ratio[~BIC_diff_strong], mlf[~BIC_diff_strong], marker='o', s=10, label='All KOFFEE fits', color=colours[0], alpha=0.3, facecolors='none')
    ax[2].scatter(flux_ratio[BIC_diff_strong], mlf[BIC_diff_strong], marker='o', s=10, label='Selected KOFFEE fits; R={:.2f}'.format(r_mlf_strong), color=colours[2], alpha=1.0)
    ax[2].plot(bin_center_strong, mlf_bin_medians_strong, marker='', lw=3, label='Median of selected KOFFEE fits; R={:.2f}'.format(r_mlf_med_strong), color=colours[2])

    ax[2].errorbar(0.25, np.nanmin(mlf), xerr=np.nanmedian(flux_error), yerr=[[np.nanmedian(mlf_err_min[BIC_diff_strong])], [np.nanmedian(mlf_err_max[BIC_diff_strong])]], c='k')

    #ax[1].set_xscale('log')
    lgnd = ax[2].legend(frameon=True, fontsize='small', loc='upper left', framealpha=0.5, edgecolor='white')
    lgnd.legendHandles[0]._legmarker.set_markersize(3)
    ax[2].set_xlabel(flux_ratio_line+' Log(F$_{broad}$/F$_{narrow}$)')
    ax[2].set_title('strongly likely BIC')

    plt.show()




def plot_ew_mlf(lamda, data, OIII_outflow_results, OIII_outflow_error, hbeta_outflow_results, hbeta_outflow_error, hbeta_no_outflow_results, hbeta_no_outflow_error, BIC_outflow, BIC_no_outflow, statistical_results, z, radius, weighted_average=True):
    """
    Plots the equivalent width of Hbeta against the mass loading factor

    Parameters
    ----------
    lamda : :obj:'~numpy.ndarray'
        1D array of wavelengths

    data : :obj:'~numpy.ndarray' of shape (len(lamda), i, j)
        array of flux values for the data (KCWI data cube)

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
    A three panel figure of the mass loading factor against the Hbeta equivalent
    width

    """
    #calculate the mass loading factor
    mlf, mlf_max, mlf_min = calc_mlf.calc_mass_loading_factor(OIII_outflow_results, OIII_outflow_error, hbeta_outflow_results, hbeta_outflow_error, hbeta_no_outflow_results, hbeta_no_outflow_error, statistical_results, z)

    #calculate the equivalent width of Hbeta
    ew = calc_ew.calc_ew(lamda, data, z)

    #calculate the velocity dispersion for the masking
    vel_disp, vel_disp_err, vel_diff, vel_diff_err, vel_out, vel_out_err = calc_outvel.calc_outflow_vel(OIII_outflow_results, OIII_outflow_error, statistical_results, z)

    #create the flow mask
    flow_mask = (statistical_results>0)

    #flatten all the arrays and get rid of extra spaxels
    mlf = mlf[flow_mask]
    mlf_max = mlf_max[flow_mask]
    mlf_min = mlf_min[flow_mask]

    ew = ew[flow_mask]

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

    #do the calculations for all the bins
    num_bins = 4
    min_bin = None #-0.05
    max_bin = None #0.6

    if weighted_average == False:
        bin_center_all, mlf_bin_medians_all, mlf_bin_lower_q_all, mlf_bin_upper_q_all = pf.binned_median_quantile_lin(ew, mlf, num_bins=num_bins, weights=None, min_bin=min_bin, max_bin=max_bin)
        bin_center_physical, mlf_bin_medians_physical, mlf_bin_lower_q_physical, mlf_bin_upper_q_physical = pf.binned_median_quantile_lin(ew[physical_mask], mlf[physical_mask], num_bins=num_bins, weights=None, min_bin=min_bin, max_bin=max_bin)
        bin_center_strong, mlf_bin_medians_strong, mlf_bin_lower_q_strong, mlf_bin_upper_q_strong = pf.binned_median_quantile_lin(ew[BIC_diff_strong], mlf[BIC_diff_strong], num_bins=num_bins, weights=None, min_bin=min_bin, max_bin=max_bin)


    elif weighted_average == True:
        bin_center_all, mlf_bin_medians_all, mlf_bin_lower_q_all, mlf_bin_upper_q_all = pf.binned_median_quantile_lin(ew, mlf, num_bins=num_bins, weights=[vel_out_err], min_bin=min_bin, max_bin=max_bin)
        bin_center_physical, mlf_bin_medians_physical, mlf_bin_lower_q_physical, mlf_bin_upper_q_physical = pf.binned_median_quantile_lin(ew[physical_mask], mlf[physical_mask], num_bins=num_bins, weights=[vel_out_err], min_bin=min_bin, max_bin=max_bin)
        bin_center_strong, mlf_bin_medians_strong, mlf_bin_lower_q_strong, mlf_bin_upper_q_strong = pf.binned_median_quantile_lin(ew[BIC_diff_strong], mlf[BIC_diff_strong], num_bins=num_bins, weights=[vel_out_err], min_bin=min_bin, max_bin=max_bin)


    #calculate the r value for the median values
    r_mlf_med_all, p_value_mlf_all = pf.pearson_correlation(bin_center_all, mlf_bin_medians_all)
    r_mlf_med_physical, p_value_mlf_physical = pf.pearson_correlation(bin_center_physical, mlf_bin_medians_physical)
    r_mlf_med_strong, p_value_mlf_strong = pf.pearson_correlation(bin_center_strong, mlf_bin_medians_strong)

    #calculate the r value for all the values
    r_mlf_all, p_value_mlf_all = pf.pearson_correlation(ew[~np.isnan(mlf)], mlf[~np.isnan(mlf)])
    r_mlf_physical, p_value_mlf_physical = pf.pearson_correlation(ew[~np.isnan(mlf)&physical_mask], mlf[~np.isnan(mlf)&physical_mask])
    r_mlf_strong, p_value_mlf_strong = pf.pearson_correlation(ew[~np.isnan(mlf)&BIC_diff_strong], mlf[~np.isnan(mlf)&BIC_diff_strong])


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
    #ax[0].errorbar(ew, mlf, xerr=vel_out_err, yerr=[mlf-mlf_min, mlf_max-mlf], fmt='o', ms=3, color=colours[0], alpha=0.6, label='All KOFFEE fits; R={:.2f}'.format(r_mlf_all))
    ax[0].fill_between(bin_center_all, mlf_bin_lower_q_all, mlf_bin_upper_q_all, color=colours[0], alpha=0.3)
    ax[0].scatter(ew, mlf, marker='o', s=10, label='All KOFFEE fits; R={:.2f}'.format(r_mlf_all), color=colours[0], alpha=0.8)
    ax[0].plot(bin_center_all, mlf_bin_medians_all, marker='', lw=3, label='Median all KOFFEE fits; R={:.2f}'.format(r_mlf_med_all), color=colours[0])

    #ax[0].errorbar(0.25, np.nanmin(mlf), xerr=np.nanmedian(flux_error), yerr=[[np.nanmedian(mlf_err_min)], [np.nanmedian(mlf_err_max)]], c='k')

    ax[0].set_ylim((np.nanmin(mlf)+np.nanmedian(mlf_err_max)-0.1), np.nanmax(mlf)+0.5)
    #ax[0].set_xlim(0.003, 10)
    lgnd = ax[0].legend(frameon=True, fontsize='small', loc='upper left', framealpha=0.5, edgecolor='white')
    lgnd.legendHandles[0]._legmarker.set_markersize(3)
    ax[0].set_ylabel(r'Log($\eta$)')
    ax[0].set_xlabel(r'EW(H$\beta$)')
    ax[0].set_title('all spaxels')

    #plot points within 90% radius
    ax[1].fill_between(bin_center_physical, mlf_bin_lower_q_physical, mlf_bin_upper_q_physical, color=colours[1], alpha=0.3)
    ax[1].scatter(ew[radius>6.1], mlf[radius>6.1], marker='o', s=10, label='All KOFFEE fits', edgecolors=colours[0], alpha=0.3, facecolors='none')
    ax[1].scatter(ew[vel_disp<=51], mlf[vel_disp<=51], marker='v', s=10, edgecolors=colours[0], alpha=0.3, facecolors='none')
    ax[1].scatter(ew[physical_mask], mlf[physical_mask], marker='o', s=10, label='Selected KOFFEE fits; R={:.2f}'.format(r_mlf_physical), color=colours[1], alpha=0.8)
    ax[1].plot(bin_center_physical, mlf_bin_medians_physical, marker='', lw=3, label='Median of selected KOFFEE fits; R={:.2f}'.format(r_mlf_med_physical), color=colours[1])

    #ax[1].errorbar(0.25, np.nanmin(mlf), xerr=np.nanmedian(flux_error), yerr=[[np.nanmedian(mlf_err_min[physical_mask])], [np.nanmedian(mlf_err_max[physical_mask])]], c='k')

    #ax[1].set_xscale('log')
    lgnd = ax[1].legend(frameon=True, fontsize='small', loc='upper left', framealpha=0.5, edgecolor='white')
    lgnd.legendHandles[0]._legmarker.set_markersize(3)
    ax[1].set_xlabel(r'EW(H$\beta$)')
    ax[1].set_title(r'$r$<$r_{90}$ and $\sigma_{broad}$>$\sigma_{inst}$')

    #plot points with strong BIC values
    ax[2].fill_between(bin_center_strong, mlf_bin_lower_q_strong, mlf_bin_upper_q_strong, color=colours[2], alpha=0.3)
    ax[2].scatter(ew[~BIC_diff_strong], mlf[~BIC_diff_strong], marker='o', s=10, label='All KOFFEE fits', color=colours[0], alpha=0.3, facecolors='none')
    ax[2].scatter(ew[BIC_diff_strong], mlf[BIC_diff_strong], marker='o', s=10, label='Selected KOFFEE fits; R={:.2f}'.format(r_mlf_strong), color=colours[2], alpha=1.0)
    ax[2].plot(bin_center_strong, mlf_bin_medians_strong, marker='', lw=3, label='Median of selected KOFFEE fits; R={:.2f}'.format(r_mlf_med_strong), color=colours[2])

    #ax[2].errorbar(0.25, np.nanmin(mlf), xerr=np.nanmedian(flux_error), yerr=[[np.nanmedian(mlf_err_min[BIC_diff_strong])], [np.nanmedian(mlf_err_max[BIC_diff_strong])]], c='k')

    #ax[1].set_xscale('log')
    lgnd = ax[2].legend(frameon=True, fontsize='small', loc='upper left', framealpha=0.5, edgecolor='white')
    lgnd.legendHandles[0]._legmarker.set_markersize(3)
    ax[2].set_xlabel(r'EW(H$\beta$)')
    ax[2].set_title('strongly likely BIC')

    plt.show()




def plot_ew_mout(lamda, data, OIII_outflow_results, OIII_outflow_error, hbeta_outflow_results, hbeta_outflow_error, BIC_outflow, BIC_no_outflow, statistical_results, z, radius, weighted_average=True):
    """
    Plots the equivalent width of Hbeta against the mass outflow rate

    Parameters
    ----------
    lamda : :obj:'~numpy.ndarray'
        1D array of wavelengths

    data : :obj:'~numpy.ndarray' of shape (len(lamda), i, j)
        array of flux values for the data (KCWI data cube)

    OIII_outflow_results : :obj:'~numpy.ndarray'
        array of outflow results from KOFFEE for OIII line.  Used to calculate
        the outflow velocity.  Should be (7, statistical_results.shape)

    OIII_outflow_error : :obj:'~numpy.ndarray'
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
    A three panel figure of the mass outflow rate against the Hbeta equivalent
    width

    """
    #calculate the mass outflow rate (in g/s)
    m_out, m_out_max, m_out_min = calc_mlf.calc_mass_outflow_rate(OIII_outflow_results, OIII_outflow_error, hbeta_outflow_results, hbeta_outflow_error, statistical_results, z)

    #convert to solar mass/yr
    m_out = m_out.to(u.solMass/u.yr)
    m_out_max = m_out_max.to(u.solMass/u.yr)
    m_out_min = m_out_min.to(u.solMass/u.yr)

    #calculate the equivalent width of Hbeta
    ew = calc_ew.calc_ew(lamda, data, z)

    #calculate the velocity dispersion for the masking
    vel_disp, vel_disp_err, vel_diff, vel_diff_err, vel_out, vel_out_err = calc_outvel.calc_outflow_vel(OIII_outflow_results, OIII_outflow_error, statistical_results, z)

    #create the flow mask
    flow_mask = (statistical_results>0)

    #flatten all the arrays and get rid of extra spaxels
    m_out = m_out[flow_mask]
    m_out_max = m_out_max[flow_mask]
    m_out_min = m_out_min[flow_mask]

    ew = ew[flow_mask]

    BIC_outflow = BIC_outflow[flow_mask]
    BIC_no_outflow = BIC_no_outflow[flow_mask]
    vel_disp = vel_disp[flow_mask]
    radius = radius[flow_mask]

    #take the log of the mass outflow rate
    m_out = np.log10(m_out.value)
    m_out_max = np.log10(m_out_max.value)
    m_out_min = np.log10(m_out_min.value)

    #calculate the errors
    m_out_err_max = m_out_max - m_out
    m_out_err_min = m_out - m_out_min

    #create BIC diff
    BIC_diff = BIC_outflow - BIC_no_outflow
    BIC_diff_strong = (BIC_diff < -50)

    #physical limits mask -
    #for the radius mask "6.1 is the 90% radius
    #also mask out the fits which lie on the lower limit of dispersion < 51km/s
    physical_mask = (radius < 6.1) & (vel_disp>51)

    #do the calculations for all the bins
    num_bins = 4
    min_bin = None #-0.05
    max_bin = None #0.6

    if weighted_average == False:
        bin_center_all, m_out_bin_medians_all, m_out_bin_lower_q_all, m_out_bin_upper_q_all = pf.binned_median_quantile_lin(ew, m_out, num_bins=num_bins, weights=None, min_bin=min_bin, max_bin=max_bin)
        bin_center_physical, m_out_bin_medians_physical, m_out_bin_lower_q_physical, m_out_bin_upper_q_physical = pf.binned_median_quantile_lin(ew[physical_mask], m_out[physical_mask], num_bins=num_bins, weights=None, min_bin=min_bin, max_bin=max_bin)
        bin_center_strong, m_out_bin_medians_strong, m_out_bin_lower_q_strong, m_out_bin_upper_q_strong = pf.binned_median_quantile_lin(ew[BIC_diff_strong], m_out[BIC_diff_strong], num_bins=num_bins, weights=None, min_bin=min_bin, max_bin=max_bin)


    elif weighted_average == True:
        bin_center_all, m_out_bin_medians_all, m_out_bin_lower_q_all, m_out_bin_upper_q_all = pf.binned_median_quantile_lin(ew, m_out, num_bins=num_bins, weights=[vel_out_err], min_bin=min_bin, max_bin=max_bin)
        bin_center_physical, m_out_bin_medians_physical, m_out_bin_lower_q_physical, m_out_bin_upper_q_physical = pf.binned_median_quantile_lin(ew[physical_mask], m_out[physical_mask], num_bins=num_bins, weights=[vel_out_err], min_bin=min_bin, max_bin=max_bin)
        bin_center_strong, m_out_bin_medians_strong, m_out_bin_lower_q_strong, m_out_bin_upper_q_strong = pf.binned_median_quantile_lin(ew[BIC_diff_strong], m_out[BIC_diff_strong], num_bins=num_bins, weights=[vel_out_err], min_bin=min_bin, max_bin=max_bin)


    #calculate the r value for the median values
    r_m_out_med_all, p_value_m_out_all = pf.pearson_correlation(bin_center_all, m_out_bin_medians_all)
    r_m_out_med_physical, p_value_m_out_physical = pf.pearson_correlation(bin_center_physical, m_out_bin_medians_physical)
    r_m_out_med_strong, p_value_m_out_strong = pf.pearson_correlation(bin_center_strong, m_out_bin_medians_strong)

    #calculate the r value for all the values
    r_m_out_all, p_value_m_out_all = pf.pearson_correlation(ew[~np.isnan(m_out)], m_out[~np.isnan(m_out)])
    r_m_out_physical, p_value_m_out_physical = pf.pearson_correlation(ew[~np.isnan(m_out)&physical_mask], m_out[~np.isnan(m_out)&physical_mask])
    r_m_out_strong, p_value_m_out_strong = pf.pearson_correlation(ew[~np.isnan(m_out)&BIC_diff_strong], m_out[~np.isnan(m_out)&BIC_diff_strong])


    #print average numbers for the different panels
    print('Number of spaxels in the first panel', m_out.shape)
    print('All spaxels median m_out:', np.nanmedian(m_out))
    print('All spaxels standard deviation m_out:', np.nanstd(m_out))
    print('')

    print('Number of spaxels with broad sigmas at the instrument dispersion:', m_out[vel_disp<=51].shape)
    print('')
    print('Number of spaxels beyond R_90:', m_out[radius>6.1].shape)
    print('')
    print('Number of spaxels in the middle panel:', m_out[physical_mask].shape)
    print('')

    print('Physical spaxels median mlf:', np.nanmedian(m_out[physical_mask]))
    print('Physical spaxels standard deviation m_out:', np.nanstd(m_out[physical_mask]))
    print('')

    print('Number of spaxels with strong BIC differences:', m_out[BIC_diff_strong].shape)
    print('')

    print('Clean spaxels median mlf:', np.nanmedian(m_out[BIC_diff_strong]))
    print('Clean spaxels standard deviation m_out:', np.nanstd(m_out[BIC_diff_strong]))
    print('')

    #-------
    #plot it
    #-------
    plt.rcParams.update(pf.get_rc_params())
    fig, ax = plt.subplots(nrows=1, ncols=3, sharey=True, sharex=True, figsize=(10,4), constrained_layout=True)

    #get colours from cmasher
    colours = cmr.take_cmap_colors('cmr.gem', 3, cmap_range=(0.25, 0.85), return_fmt='hex')

    #plot all points
    #ax[0].errorbar(ew, m_out, xerr=vel_out_err, yerr=[m_out-m_out_min, m_out_max-m_out], fmt='o', ms=3, color=colours[0], alpha=0.6, label='All KOFFEE fits; R={:.2f}'.format(r_mlf_all))
    ax[0].fill_between(bin_center_all, m_out_bin_lower_q_all, m_out_bin_upper_q_all, color=colours[0], alpha=0.3)
    ax[0].scatter(ew, m_out, marker='o', s=10, label='All KOFFEE fits; R={:.2f}'.format(r_m_out_all), color=colours[0], alpha=0.8)
    ax[0].plot(bin_center_all, m_out_bin_medians_all, marker='', lw=3, label='Median all KOFFEE fits; R={:.2f}'.format(r_m_out_med_all), color=colours[0])

    #ax[0].errorbar(0.25, np.nanmin(m_out), xerr=np.nanmedian(flux_error), yerr=[[np.nanmedian(m_out_err_min)], [np.nanmedian(m_out_err_max)]], c='k')

    ax[0].set_ylim((np.nanmin(m_out)+np.nanmedian(m_out_err_max)-0.1), np.nanmax(m_out)+0.5)
    #ax[0].set_xlim(0.003, 10)
    lgnd = ax[0].legend(frameon=True, fontsize='small', loc='upper left', framealpha=0.5, edgecolor='white')
    lgnd.legendHandles[0]._legmarker.set_markersize(3)
    ax[0].set_ylabel('Log(Mass outflow rate)')
    ax[0].set_xlabel(r'EW(H$\beta$)')
    ax[0].set_title('all spaxels')

    #plot points within 90% radius
    ax[1].fill_between(bin_center_physical, m_out_bin_lower_q_physical, m_out_bin_upper_q_physical, color=colours[1], alpha=0.3)
    ax[1].scatter(ew[radius>6.1], m_out[radius>6.1], marker='o', s=10, label='All KOFFEE fits', edgecolors=colours[0], alpha=0.3, facecolors='none')
    ax[1].scatter(ew[vel_disp<=51], m_out[vel_disp<=51], marker='v', s=10, edgecolors=colours[0], alpha=0.3, facecolors='none')
    ax[1].scatter(ew[physical_mask], m_out[physical_mask], marker='o', s=10, label='Selected KOFFEE fits; R={:.2f}'.format(r_m_out_physical), color=colours[1], alpha=0.8)
    ax[1].plot(bin_center_physical, m_out_bin_medians_physical, marker='', lw=3, label='Median of selected KOFFEE fits; R={:.2f}'.format(r_m_out_med_physical), color=colours[1])

    #ax[1].errorbar(0.25, np.nanmin(m_out), xerr=np.nanmedian(flux_error), yerr=[[np.nanmedian(m_out_err_min[physical_mask])], [np.nanmedian(m_out_err_max[physical_mask])]], c='k')

    #ax[1].set_xscale('log')
    lgnd = ax[1].legend(frameon=True, fontsize='small', loc='upper left', framealpha=0.5, edgecolor='white')
    lgnd.legendHandles[0]._legmarker.set_markersize(3)
    ax[1].set_xlabel(r'EW(H$\beta$)')
    ax[1].set_title(r'$r$<$r_{90}$ and $\sigma_{broad}$>$\sigma_{inst}$')

    #plot points with strong BIC values
    ax[2].fill_between(bin_center_strong, m_out_bin_lower_q_strong, m_out_bin_upper_q_strong, color=colours[2], alpha=0.3)
    ax[2].scatter(ew[~BIC_diff_strong], m_out[~BIC_diff_strong], marker='o', s=10, label='All KOFFEE fits', color=colours[0], alpha=0.3, facecolors='none')
    ax[2].scatter(ew[BIC_diff_strong], m_out[BIC_diff_strong], marker='o', s=10, label='Selected KOFFEE fits; R={:.2f}'.format(r_m_out_strong), color=colours[2], alpha=1.0)
    ax[2].plot(bin_center_strong, m_out_bin_medians_strong, marker='', lw=3, label='Median of selected KOFFEE fits; R={:.2f}'.format(r_m_out_med_strong), color=colours[2])

    #ax[2].errorbar(0.25, np.nanmin(m_out), xerr=np.nanmedian(flux_error), yerr=[[np.nanmedian(m_out_err_min[BIC_diff_strong])], [np.nanmedian(m_out_err_max[BIC_diff_strong])]], c='k')

    #ax[1].set_xscale('log')
    lgnd = ax[2].legend(frameon=True, fontsize='small', loc='upper left', framealpha=0.5, edgecolor='white')
    lgnd.legendHandles[0]._legmarker.set_markersize(3)
    ax[2].set_xlabel(r'EW(H$\beta$)')
    ax[2].set_title('strongly likely BIC')

    plt.show()



def map_of_mlf(lamdas, xx_flat, yy_flat, rad_flat, data_flat, z, OIII_outflow_results, OIII_outflow_error, hbeta_outflow_results, hbeta_outflow_error, hbeta_no_outflow_results, hbeta_no_outflow_error, statistical_results):
    """
    Plots the map of the mass loading factor

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

    OIII_outflow_results : :obj:'~numpy.ndarray'
        array of results from KOFFEE.  Used to calculate the outflow velocity.
        Should have shape [7, :, :]

    OIII_outflow_error : :obj:'~numpy.ndarray'
        array of the outflow result errors from KOFFEE.  Same shape as outflow_results

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

    statistical_results : :obj:'~numpy.ndarray'
        array of statistical results from KOFFEE.  Should have same shape as the
        second two dimensions of outflow_results.

    Returns
    -------
    A map of the outflow velocities
    """
    #calculate the mass loading factor
    mlf, mlf_max, mlf_min = calc_mlf.calc_mass_loading_factor(OIII_outflow_results, OIII_outflow_error, hbeta_outflow_results, hbeta_outflow_error, hbeta_no_outflow_results, hbeta_no_outflow_error, statistical_results, z)

    #create outflow mask
    flow_mask = (statistical_results>0)

    xx_flat_out = xx_flat[flow_mask.reshape(-1)]
    yy_flat_out = yy_flat[flow_mask.reshape(-1)]
    mlf = np.log10(mlf[flow_mask])

    #make limits for the plots
    xmin, xmax = xx_flat.min(), xx_flat.max()
    ymin, ymax = yy_flat.min(), yy_flat.max()
    vmin, vmax = np.nanmin(mlf), np.nanmax(mlf)

    #create figure and subplots
    plt.rcParams.update(pf.get_rc_params())
    #fig, (ax1, ax2) = plt.subplots(1,2, sharey=True)
    fig, ax1 = plt.subplots(1,1)

    #get the continuum contours
    i, j = statistical_results.shape
    #cont_contours1 = plot_continuum_contours(lamdas, np.reshape(xx_flat, (i,j)), np.reshape(yy_flat, (i, j)), np.reshape(data_flat, (data_flat.shape[0],i,j)), z, ax1)

    #create figure of just outflows
    outflow_spax = bdpk.display_pixels(xx_flat_out, yy_flat_out, mlf.reshape(1,-1), axes=ax1, vmin=vmin, vmax=vmax)
    ax1.set_xlim(xmin, xmax)
    ax1.invert_xaxis()
    ax1.set_ylabel('Arcseconds')
    ax1.set_xlabel('Arcseconds')
    cbar = plt.colorbar(outflow_spax, ax=ax1, shrink=0.8)
    cbar.set_label('Log(Mass Loading Factor)')
    #cont_contours1

    plt.show()
