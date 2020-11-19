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




def plot_continuum_contours(lamdas, xx, yy, data, z, ax):
    """
    Plots the continuum contours, using the rest wavelengths between 4600 and 4800 to define the continuum.

    Args:
        lamdas: wavelength vector (1D)
        xx: x coordinate array (2D)
        yy: y coordinate array (2D)
        data: data array (3D)
        z: redshift of the galaxy
        ax: axis for matplotlib to draw on

    Returns:
        cont_contours: matplotlib.contour.QuadContourSet instance

    """
    #create a mask for the continuum
    cont_mask = (lamdas>4600*(1+z))&(lamdas<4800*(1+z))

    #find the median of the continuum
    cont_median = np.median(data[cont_mask,:,:], axis=0)

    #create the contours
    cont_contours = ax.contour(xx, yy, cont_median, colors='white', linewidths=0.7, alpha=0.7, levels=(0.2,0.3,0.4,0.7,1.0,2.0,4.0))

    return cont_contours

def map_of_outflows(lamdas, xx_flat, yy_flat, rad_flat, data_flat, z, outflow_results, outflow_error, statistical_results):
    """
    Plots the map of outflow velocities.
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
    plt.rcParams['axes.facecolor']='white'
    #fig, (ax1, ax2) = plt.subplots(1,2, sharey=True)
    fig, ax1 = plt.subplots(1,1)

    #get the continuum contours
    i, j = statistical_results.shape
    #cont_contours1 = plot_continuum_contours(lamdas, np.reshape(xx_flat, (i,j)), np.reshape(yy_flat, (i, j)), np.reshape(data_flat, (data_flat.shape[0],i,j)), z, ax1)
    #cont_contours2 = plot_continuum_contours(lamdas, np.reshape(xx_flat, (67,24)), np.reshape(yy_flat, (67, 24)), np.reshape(data_flat, (data_flat.shape[0],67,24)), z, ax2)

    #create figure of just outflows
    outflow_spax = bdpk.display_pixels(xx_flat_out, yy_flat_out, vel_out.reshape(1,-1), axes=ax1)#, vmin=100, vmax=500)
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


def map_of_mlf(lamdas, xx_flat, yy_flat, rad_flat, data_flat, z, OIII_outflow_results, OIII_outflow_error, hbeta_outflow_results, hbeta_outflow_error, hbeta_no_outflow_results, hbeta_no_outflow_error, statistical_results):
    """
    Plots the map of the mass loading factor
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
    plt.rcParams['axes.facecolor']='white'
    #fig, (ax1, ax2) = plt.subplots(1,2, sharey=True)
    fig, ax1 = plt.subplots(1,1)

    #get the continuum contours
    i, j = statistical_results.shape
    #cont_contours1 = plot_continuum_contours(lamdas, np.reshape(xx_flat, (i,j)), np.reshape(yy_flat, (i, j)), np.reshape(data_flat, (data_flat.shape[0],i,j)), z, ax1)
    #cont_contours2 = plot_continuum_contours(lamdas, np.reshape(xx_flat, (67,24)), np.reshape(yy_flat, (67, 24)), np.reshape(data_flat, (data_flat.shape[0],67,24)), z, ax2)

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


def sn_cut_plot(lamdas, xx_flat, yy_flat, rad_flat, data_flat, z, sn):
    """
    Plots the two maps before and after the S/N cut is made, so that you can check which pixels have been removed.

    Args:

    Returns:

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
    fig, (ax1, ax2) = plt.subplots(1,2, sharey=True)

    #get the continuum contours
    cont_contours1 = plot_continuum_contours(lamdas, np.reshape(xx_flat, (67,24)), np.reshape(yy_flat, (67, 24)), np.reshape(data_flat, (data_flat.shape[0],67,24)), z, ax1)
    cont_contours2 = plot_continuum_contours(lamdas, np.reshape(xx_flat, (67,24)), np.reshape(yy_flat, (67, 24)), np.reshape(data_flat, (data_flat.shape[0],67,24)), z, ax2)

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
