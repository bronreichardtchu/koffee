"""
NAME:
	paper_II_plots.py

AUTHOR:
	Bronwyn Reichardt Chu
	Swinburne
	2022

EMAIL:
	<breichardtchu@swin.edu.au>

PURPOSE:
	To make plots of results from Paper II
	Written on MacOS Mojave 10.14.5, with Python 3.7

FUNCTIONS INCLUDED
    maps_of_IRAS08                                      (Figure 1)


"""
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.lines import Line2D
import cmasher as cmr

from astropy.wcs import WCS
from astropy import units as u
from astropy.cosmology import WMAP9 as cosmo
from astropy.io import fits
from astropy.table import Table

from scipy.signal import argrelextrema

import prepare_cubes as pc
import plotting_functions as pf
import koffee_fitting_functions as kff
import calculate_outflow_velocity as calc_outvel
import calculate_star_formation_rate as calc_sfr
import calculate_mass_loading_factor as calc_mlf
import calculate_energy_loading_factor as calc_elf
import calculate_equivalent_width as calc_ew
import brons_display_pixels_kcwi as bdpk
import koffee

import importlib
importlib.reload(pf)
importlib.reload(calc_sfr)
#importlib.reload(bdpk)



def maps_of_IRAS08(halpha_fits_file, m_out_fits_file, radius):
    """
    Maps the HST image and mass outflow rate results for IRAS08 from fits files

    Parameters
    ----------
    halpha_fits_file : string
        location of the fits file with the Halpha flux

    m_out_fits_file : string
        location of the fits file with the mass outflow rate measured with KOFFEE
        (must be same shape as statistical_results)

    radius : :obj:'~numpy.ndarray'
        array of galaxy radius values

    Returns
    -------
    A two-panel figure with a map of flux, and a map of the results from KOFFEE
    """
    #read in fits files
    #need to shift the IRAS08 KCWI data to match the WCS with the HST data
    halpha_data, halpha_header, halpha_wcs = pf.read_in_create_wcs(halpha_fits_file)
    m_out, m_out_header, m_out_wcs = pf.read_in_create_wcs(m_out_fits_file, shift=['CRPIX2', 32.0])

    #multiply the mass outflow rate by 380 because when I calculated it, I
    #assumed that n_e = 380 cm^-3; but in this paper we're assuming n_e = 100 cm^-3
    m_out = m_out*380/100

    #take the log of the velocity and the flux ratio
    m_out = np.log10(m_out)

    #creating the x and y limits
    xlim = [4, 16]
    ylim = [2, 58]

    kcwi_low_lim = m_out_wcs.all_pix2world(xlim[0], ylim[0], 0)
    kcwi_high_lim = m_out_wcs.all_pix2world(xlim[1], ylim[1], 0)
    print('Limits')
    print(kcwi_low_lim)
    print(kcwi_high_lim)

    kcwi_low_lim_halpha = halpha_wcs.all_world2pix(kcwi_low_lim[0], kcwi_low_lim[1], 0)
    kcwi_high_lim_halpha = halpha_wcs.all_world2pix(kcwi_high_lim[0], kcwi_high_lim[1], 0)
    print(kcwi_low_lim_halpha)
    print(kcwi_high_lim_halpha)


    #find the peak of the halpha
    halpha_peak_pixel = np.argwhere(halpha_data==np.nanmax(halpha_data))
    print(halpha_peak_pixel)

    halpha_peak_world = halpha_wcs.all_pix2world(halpha_peak_pixel[0,1], halpha_peak_pixel[0,0], 0)
    print(halpha_peak_world)


    #find the peak of the mass outflow rate and convert to wcs
    print('Max m_out')
    m_out_peak_pixel = np.argwhere(m_out==np.nanmax(m_out[radius<6.1]))
    print(m_out_peak_pixel)

    m_out_local_maxima_pixel = argrelextrema(m_out[radius<6.1], np.greater)

    m_out_max_local_maxima_pixel = np.argwhere(m_out==np.sort(m_out[radius<6.1][m_out_local_maxima_pixel])[-2])

    m_out_peak_world = m_out_wcs.all_pix2world(m_out_peak_pixel[0,1], m_out_peak_pixel[0,0], 0)
    print(m_out_peak_world)
    m_out_peak_halpha_pixel = halpha_wcs.all_world2pix(m_out_peak_world[0], m_out_peak_world[1], 0)
    print(m_out_peak_halpha_pixel)

    m_out_local_max_world = m_out_wcs.all_pix2world(m_out_max_local_maxima_pixel[0,1], m_out_max_local_maxima_pixel[0,0], 0)#, 0)
    print(m_out_local_max_world)
    m_out_local_max_halpha_pixel = halpha_wcs.all_world2pix(m_out_local_max_world[0], m_out_local_max_world[1], 0)
    print(m_out_local_max_halpha_pixel)
    #m_out_local_max_fuv_pixel = fuv_wcs.all_world2pix(m_out_local_max_world[0], m_out_local_max_world[1], 0)
    #print(m_out_local_max_fuv_pixel)




    #calculate the beginning and end of 6.1 arcsec (r_90)
    halpha_r90_pixel_length = abs(6.1/(halpha_header['CD1_1']*60*60))

    halpha_start_r90_xpixel = halpha_peak_pixel[0][1] - kcwi_low_lim_halpha[1] + 10
    halpha_start_r90_ypixel = kcwi_low_lim_halpha[1] + 20
    halpha_end_r90_xpixel = halpha_start_r90_xpixel + halpha_r90_pixel_length

    #calculate the beginning and end of 2.6 arcsec (r_50)
    halpha_r50_pixel_length = abs(2.6/(halpha_header['CD1_1']*60*60))

    halpha_start_r50_xpixel = halpha_peak_pixel[0][1] - kcwi_low_lim_halpha[1] + 10
    halpha_start_r50_ypixel = kcwi_low_lim_halpha[1] + 100
    halpha_end_r50_xpixel = halpha_start_r50_xpixel + halpha_r50_pixel_length

    #create the figure
    plt.rcParams.update(pf.get_rc_params())

    plt.figure(figsize=(8,4))#constrained_layout=True)

    ax1 = plt.subplot(121, projection=halpha_wcs)
    ax1.set_facecolor('black')
    #do the plotting
    halpha_map = ax1.imshow(np.log10(halpha_data), origin='lower', cmap=cmr.ember, vmin=-1.75, vmax=-0.25)

    ax1.hlines(halpha_start_r90_ypixel, halpha_start_r90_xpixel, halpha_end_r90_xpixel, colors='white')
    ax1.text(halpha_start_r90_xpixel+5, halpha_start_r90_ypixel+10, '$r_{90}$ = 2.4 kpc ', c='white')
    ax1.hlines(halpha_start_r50_ypixel, halpha_start_r50_xpixel, halpha_end_r50_xpixel, colors='white')
    ax1.text(halpha_start_r50_xpixel+5, halpha_start_r50_ypixel+10, '$r_{50}$ = 1 kpc ', c='white')


    lon1 = ax1.coords[0]
    lat1 = ax1.coords[1]
    #lon1.set_ticks_visible(False)
    lon1.tick_params(colors='white')
    lon1.set_ticklabel_visible(False)
    #lat1.set_ticks_visible(False)
    lat1.tick_params(colors='white')
    lat1.set_ticklabel_visible(False)
    ax1.set_title(r'H$\alpha$ Flux')
    ax1.set_xlim(kcwi_high_lim_halpha[0], kcwi_low_lim_halpha[0])
    ax1.set_ylim(kcwi_low_lim_halpha[1], kcwi_high_lim_halpha[1])
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("left", size="20%", pad=0.1)
    cax.axis('off')

    ax2 = plt.subplot(122, projection=m_out_wcs, slices=('y', 'x'))
    #m_out_spax = bdpk.display_pixels(xx_flat_out, yy_flat_out, m_out, angle=360, axes=ax8, cmap=cmr.gem, vmin=-3.2, vmax=-0.9)
    m_out_spax = ax2.imshow(m_out.T, origin='lower', aspect=m_out_header['CD2_1']/m_out_header['CD1_2'], cmap=cmr.gem, vmin=-3.0, vmax=0.0)
    #ax8.hlines(ymin+0.75, xmin+4, xmin+4+10, colors='black')
    #ax8.arrow(m_out_peak_pixel[0][0]+5, m_out_peak_pixel[0][1]-2, -5, 2, width=0.2, length_includes_head=True, color='k')
    ax2.grid(False)
    ax2.coords.grid(False)
    lon2 = ax2.coords[0]
    lat2 = ax2.coords[1]
    #lon2.set_ticks_visible(False)
    lon2.set_ticklabel_visible(False)
    #lat2.set_ticks_visible(False)
    lat2.set_ticklabel_visible(False)

    ax2.set_xlim(ylim[0], ylim[1])
    ax2.set_ylim(xlim[0], xlim[1])
    ax2.invert_xaxis()
    ax2.set_title(r'Log($\dot{M}_{out}$) ')
    cbar = plt.colorbar(m_out_spax, ax=ax2, shrink=0.8)
    cbar.set_label(r'Log(M$_\odot$ yr$^{-1}$)')

    #plt.subplots_adjust(left=0.0, right=0.96, top=0.99, bottom=0.0, wspace=0.1, hspace=0.0)

    plt.show()


def plot_compare_pandya21(fire_data, m_out_fits_file, sfr_fits_file, co_fits_file, z, adjust_FIRE=False, adjust_mout_vel=False, vout_fits_file=None, vel_diff_fits_file=None):
    """
    This plot compares our data to the results from the FIRE-2 simulation found
    in Pandya et al (2021), Figure 11.  The plot shows the mass loading factor
    for the warm gas phase plotted against the gas and SFR surface densitites.
    Thanks to Viraj Pandya for sharing a simpler version of the plotting script
    for their Figure 11.

    Parameters
    ----------
    fire_data : str
        the location of the data table file from FIRE-2
        'time_catalog_AllHalos.dat'
    """
    #plot x-limits
    xlim_sigma_mol = [0.5, 4]
    xlim_sigma_sfr = [-5, np.log10(5e1)]

    #read in the relevant supplementary appendix Table B4 for Pandya+21
    fire_table = Table.read(fire_data, format='ascii')

    #read in the kcwi sfr and mass outflow rate
    #use the resampled and rotated data, so the WCS of the KCWI data is already
    #shifted match the WCS with the HST data and CO data
    m_out, m_out_header, m_out_wcs = pf.read_in_create_wcs(m_out_fits_file, index=1, shift=None)

    sfr, sfr_header, sfr_wcs = pf.read_in_create_wcs(sfr_fits_file, index=1, shift=None)

    #read in the CO data
    co_data, co_header, co_wcs = pf.read_in_create_wcs(co_fits_file, index=1)

    #create a mask to get rid of any m_out values below 1e-10
    m_out_mask = m_out > 1e-10

    #multiply the mass outflow rate by 380 because when I calculated it, I
    #assumed that n_e = 380 cm^-3; but in this paper we're assuming n_e = 100 cm^-3
    m_out = m_out*380/100

    #if the m_out is still calculated with vel_out which is the maximum (90%)
    #velocity from Paper I, we want to use the central velocity instead
    if adjust_mout_vel == True:
        #need to read in the files
        vel_out, vel_out_header, vel_out_wcs = pf.read_in_create_wcs(vout_fits_file, index=1, shift=None)
        vel_diff, vel_diff_header, vel_diff_wcs = pf.read_in_create_wcs(vel_diff_fits_file, index=1, shift=None)

        #now multiply the m_out to adjust it (vout is in the numerator in mout)
        m_out = m_out * vel_diff/vel_out

    #calculate the mass outflow rate which we would get if we used R_out = 20kpc
    #instead of R_out = 500pc
    m_out_20kpc = m_out*500/20000

    #calculate the mass loading factor
    mlf = m_out/sfr
    mlf_20kpc = m_out_20kpc/sfr
    #mlf_lower_lim = mlf - mlf_20kpc

    print('Median mass loading factor:', np.nanmedian(mlf))


    #get the proper distance per arcsecond
    proper_dist_kpc = cosmo.kpc_proper_per_arcmin(z).to(u.kpc/u.arcsec)
    print(proper_dist_kpc)
    proper_dist_pc = cosmo.kpc_proper_per_arcmin(z).to(u.pc/u.arcsec)
    print(proper_dist_pc)

    #calculate the sfr surface density
    print('SFR spaxel size:', abs(sfr_header['CD1_1']*60*60)*proper_dist_kpc, 'by', abs(sfr_header['CD2_2']*60*60)*proper_dist_kpc, '=', abs(sfr_header['CD1_1']*60*60*sfr_header['CD2_2']*60*60)*(proper_dist_kpc**2))

    sfr_surface_density = sfr/(abs(sfr_header['CD1_1']*60*60*sfr_header['CD2_2']*60*60)*(proper_dist_kpc**2))
    #sfr_surface_density_err = sfr_err/((header['CD1_2']*60*60*header['CD2_1']*60*60)*(proper_dist**2))

    print('Median SFR surface density:', np.nanmedian(sfr_surface_density))

    #calculate the molecular mass surface density
    #first convert CO luminosity to gas mass
    molgas = co_data * 1.06e8/7.3


    #now divide through by the spaxel size
    print('CO spaxel size:', (co_header['CDELT1']*60*60)*proper_dist_pc, 'by', (co_header['CDELT2']*60*60)*proper_dist_pc, '=', (co_header['CDELT1']*60*60*co_header['CDELT2']*60*60)*(proper_dist_pc**2))

    #molgas_surface_density = molgas/((co_header['CDELT1']*60*60*co_header['CDELT2']*60*60)*(proper_dist**2))
    #molgas_surface_density = molgas/(abs(sfr_header['CD1_1']*60*60*sfr_header['CD2_2']*60*60)*(proper_dist_pc**2))

    molgas_surface_density = molgas/(525.8564*525.8564*(u.pc)**2)

    print('Median molecular gas surface density:', np.nanmedian(molgas_surface_density))

    #calculate the depletion times
    tdep_sf = molgas/sfr
    tdep_out = molgas/m_out

    #calculate the Kim et al. 2020 trends
    log_sigma_sfr_kim, log_mlf_kim = pf.kim_et_al_2020_sigma_sfr_log(-4, 0)
    log_sigma_sfr_kim_extrapolate, log_mlf_kim_extrapolate = pf.kim_et_al_2020_sigma_sfr_log(xlim_sigma_sfr[0], xlim_sigma_sfr[1])

    log_sigma_mol_kim, log_mlf_mol_kim = pf.kim_et_al_2020_sigma_mol_log(0, 2)
    log_sigma_mol_kim_extrapolate, log_mlf_mol_kim_extrapolate = pf.kim_et_al_2020_sigma_mol_log(xlim_sigma_mol[0], xlim_sigma_mol[1])


    #create the figure
    plt.rcParams.update(pf.get_rc_params())

    with plt.rc_context({"xtick.direction": 'out', "xtick.labelsize": 'medium', "xtick.minor.visible": False, "xtick.top": False, "ytick.direction": 'out', "ytick.labelsize": 'medium', "ytick.minor.visible": False, "ytick.right": False}):

        # initialize figure with axes objects
        fig, axes = plt.subplots(nrows=1,ncols=2,figsize=(8,4),dpi=100, sharey=True)

        #get colors
        colours = cmr.take_cmap_colors(cmr.gem, 2, cmap_range=(0.45, 0.85), return_fmt='hex')

        # loop over halo names
        halo_names = np.unique(fire_table['Halo'])
        for halo_name in halo_names:
            # filter the table to select all bursts belonging to current halo
            thalo = fire_table[fire_table['Halo']==halo_name]

            # use a different color for each halo mass bin/category (same symbols)
            if 'm10' in halo_name:
                plt_color = 'dimgrey'
            elif 'm11' in halo_name:
                plt_color = 'dimgrey'
            elif 'm12' in halo_name:
                plt_color = 'dimgrey'
            elif 'A' in halo_name:
                plt_color = 'dimgrey'

            """ NOTE: the y-axis is now showing the warm-phase mass loading factor, log_etaM_warm """
            # left panel: etaM vs Sigma_gas for each individual burst
            if adjust_FIRE == False:
                axes[0].scatter(thalo['log_sigmaGas_weighted'],thalo['log_etaM_warm'],
                                c=plt_color,marker='s',s=10,alpha=0.5,edgecolor='none')

                # right panel: etaM vs Sigma_SFR
                axes[1].scatter(thalo['log_sigmaSFR_weighted'], thalo['log_etaM_warm'],
                                c=plt_color,marker='s',s=10,alpha=0.5,edgecolor='none')

            elif adjust_FIRE == True:
                axes[0].scatter(thalo['log_sigmaGas_weighted'], thalo['log_etaM_warm']+np.log10(3), c=plt_color, marker='s', s=10, alpha=0.5, edgecolor='none')

                # right panel: etaM vs Sigma_SFR
                axes[1].scatter(thalo['log_sigmaSFR_weighted'],  thalo['log_etaM_warm']+np.log10(3),  c=plt_color, marker='s', s=10, alpha=0.5, edgecolor='none')

        #scatter the kcwi and co data on the plots
        #axes[0].scatter(np.log10(molgas_surface_density[m_out_mask].flatten().value), np.log10(mlf[m_out_mask].flatten()), cmap='cool', c=np.log10(tdep_out[m_out_mask]), s=50, alpha=1, vmin=7.5, vmax=11)#, zorder=8)

        #tdep_colour = axes[1].scatter(np.log10(sfr_surface_density[m_out_mask].flatten().value), np.log10(mlf[m_out_mask].flatten()), cmap='cool', c=np.log10(tdep_out[m_out_mask]), s=50, alpha=1, vmin=7.5, vmax=11)#, zorder=8)

        sigma_sfr_mask = (sfr_surface_density.value<1) & (sfr_surface_density.value>0.3)

        axes[0].scatter(np.log10(molgas_surface_density[m_out_mask&sigma_sfr_mask].flatten().value), np.log10(mlf[m_out_mask&sigma_sfr_mask].flatten()), c=colours[0], s=50, alpha=1)#, zorder=8)

        axes[0].scatter(np.log10(molgas_surface_density[m_out_mask&(~sigma_sfr_mask)].flatten().value), np.log10(mlf[m_out_mask&(~sigma_sfr_mask)].flatten()), c=colours[0], s=50, alpha=0.5)#, zorder=8)

        axes[1].scatter(np.log10(sfr_surface_density[m_out_mask].flatten().value), np.log10(mlf[m_out_mask].flatten()), c=colours[0], s=50, alpha=1)#, zorder=8)

        #axes[0].scatter(np.log10(molgas_surface_density[m_out_mask].flatten().value), np.log10(mlf_20kpc[m_out_mask].flatten()), cmap='cool', c=np.log10(tdep_out[m_out_mask]), s=50, facecolors='none', alpha=0.5, vmin=7.5, vmax=11)#, zorder=8)

        #axes[1].scatter(np.log10(sfr_surface_density[m_out_mask].flatten().value), np.log10(mlf_20kpc[m_out_mask].flatten()), cmap='cool', c=np.log10(tdep_out[m_out_mask]), s=50, facecolors='none', alpha=0.5, vmin=7.5, vmax=11)#, zorder=8)

        #divider = make_axes_locatable(axes[1])
        #cax = divider.append_axes('right', size='5%', pad=0.05)

        #cbar = fig.colorbar(tdep_colour, cax=cax)
        #cbar.set_label(label='log($t_{dep,out}$ [Gyr])', rotation=270, labelpad=10)

        axes[0].scatter(np.log10(molgas_surface_density[m_out_mask].flatten().value), np.log10(mlf_20kpc[m_out_mask].flatten()), c=colours[1], s=50, facecolors='none', alpha=0.4)#, zorder=8)

        axes[1].scatter(np.log10(sfr_surface_density[m_out_mask].flatten().value), np.log10(mlf_20kpc[m_out_mask].flatten()), c=colours[1], s=50, facecolors='none', alpha=0.4)#, zorder=8)

        #put the Kim et al. 2020 trends on the plot
        axes[0].plot(log_sigma_mol_kim_extrapolate, log_mlf_mol_kim_extrapolate, c='k', lw=2, ls='--')
        axes[0].plot(log_sigma_mol_kim, log_mlf_mol_kim, c='k', lw=2, ls='-')

        axes[1].plot(log_sigma_sfr_kim_extrapolate, log_mlf_kim_extrapolate, c='k', lw=2, ls='--')
        axes[1].plot(log_sigma_sfr_kim, log_mlf_kim, c='k', lw=2, ls='-')


        # use logscale and set same y-limits for both plots
        for ax in axes:
            #ax.tick_params(top=True,right=True,which='both')

            ax.set_ylim(-4, 4)

            #ax.axhline(1.0,alpha=0.3,color='k',zorder=0,label='__none__')

        # set different xlims for each plot
        axes[0].set_xlim(xlim_sigma_mol[0], xlim_sigma_mol[1])
        axes[1].set_xlim(xlim_sigma_sfr[0], xlim_sigma_sfr[1])

        # axis labels
        axes[0].set_xlabel('log($\Sigma_{mol} [\mathrm{M_\odot pc^{-2}}]$)')
        axes[1].set_xlabel('log($\Sigma_{SFR} [\mathrm{M_\odot kpc^{-2}}]$)')
        axes[0].set_ylabel(r'log($\eta_{\rm ion}$)')


        # halo mass bin color-coding label
        #axes[0].text(0.85,0.95,u'm10',color='black',transform=axes[0].transAxes,fontsize=8)
        #axes[0].text(0.85,0.9,u'm11',color='dimgrey',transform=axes[0].transAxes,fontsize=8)
        #axes[0].text(0.85,0.85,u'm12',color='grey',transform=axes[0].transAxes,fontsize=8)
        #axes[0].text(0.85,0.8,u'm13',color='darkgrey',transform=axes[0].transAxes,fontsize=8)

        # colour coding labels
        legend_elements = [Line2D([0], [0], color='k', lw=2, ls='-', label=u'Kim+2020'),
                    Line2D([0], [0], color='k', lw=2, ls='--', label=u'Kim+2020 extrapolated'),
                    Line2D([0], [0], color='w', marker='s', markersize=5, markerfacecolor='dimgrey', label=u'FIRE-2 (Pindya+2022)')]

        axes[0].legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.0, 1.0), fontsize=8, frameon=False)


        #axes[1].text(0.98, 0.95, u'solid: Kim+2020', color='k', transform=axes[1].transAxes, fontsize=8, ha='right')
        #axes[1].text(0.98, 0.9, u'dashed: Kim+2020 extrapolated', color='k', transform=axes[1].transAxes, fontsize=8, ha='right')
        #axes[1].text(0.98, 0.94, u'FIRE-2 (Pindya+2022)', color='dimgrey', transform=axes[1].transAxes, fontsize=8, ha='right')
        axes[1].text(0.98, 0.95, u'IRAS08', color=colours[0], transform=axes[1].transAxes, fontsize=8, ha='right')
        #axes[1].text(0.98, 0.9, u'Solid: $R_{out}$=0.5kpc', color='k', transform=axes[1].transAxes, fontsize=8, ha='right')
        axes[1].text(0.98, 0.9, r'IRAS08 rescaled to $0.1R_{vir}$', color=colours[1], transform=axes[1].transAxes, fontsize=8, ha='right')


        plt.subplots_adjust(left=0.1, right=0.98, top=0.96, bottom=0.17, wspace=0.1, hspace=0.0)
        #plt.subplots_adjust(left=0.08, right=0.91, top=0.96, bottom=0.17, wspace=0.1, hspace=0.0)

    plt.show()
