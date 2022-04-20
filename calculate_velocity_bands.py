"""
NAME:
	calculate_velocity_bands.py

AUTHOR:
	Bronwyn Reichardt Chu
	Swinburne
	2022

EMAIL:
	<breichardtchu@swin.edu.au>

PURPOSE:
	To calculate velocity bands of the outflowing flux.
	Written on MacOS Mojave 10.14.5, with Python 3.7

FUNCTIONS INCLUDED:

    calc_save_as_fits

MODIFICATION HISTORY:
		v.1.0 - first created January 2022

"""
import numpy as np
import matplotlib.pyplot as plt
import cmasher as cmr
import matplotlib.cm as cm
from matplotlib.colors import Normalize

from astropy.cosmology import WMAP9 as cosmo
from astropy.constants import c
from astropy import units

from astropy.io import fits
from astropy.wcs import WCS

import prepare_cubes as pc
import koffee_fitting_functions as kff
import plotting_functions as pf
import koffee

import importlib
importlib.reload(pc)

"""
Steps:
1. Read in data
2. Read in fits
3. Find location of maximum flux
4. Subtract narrow line from data
5. Convert wavelength vector into velocity vector
6. Split remaining flux into:
    a. v_cen to sigma_disk
    b. sigma_disk to v_esc
    c. v_esc to noise level?
7. Add it up and save to file.
"""

def calc_line_flux(spectrum, dx):
    """
    Calculates the line flux over the input spectrum

    Parameters
    ----------
    spectrum : :obj:'~numpy.ndarray'
        the area of the spectrum over which to add the line flux

    dx : float
        the channel width of the spectrum (usually 0.5 A)
    """
    #attach the units to the spectrum and dx
    spectrum = spectrum*units.erg/(units.s*(units.cm*units.cm)*units.AA)
    dx = dx*units.AA

    line_flux = np.nansum(spectrum*dx, axis=0)

    return line_flux


def convert_wave_to_vel(lamdas, lam_cen, z, deredshift=True):
    """
    Converts the wavelength vector into a velocity vector using:
        v = c*(lam_cen-lamdas)/lam_cen

    Parameters
    ----------
    lamdas : :obj:'~numpy.ndarray'
        the wavelength vector

    lam_cen : float or :obj:'~numpy.ndarray'
        the central wavelength value to measure the velocity from (generally the
        location of the maximum flux for the emission line)
        If lam_cen is a :obj:'~numpy.ndarray' then lamdas must be a cube with
        the same spatial dimensions.

    z : float
        redshift

    deredshift : boolean
        whether to de-redshift the observed wavelength (default is True)

    Returns
    -------
    vels : :obj:'~numpy.ndarray'
        the velocity vector in km/s
    """
    # de-redshift the vector first
    if deredshift == True:
        lamdas = lamdas/(1+z)
        lam_cen = lam_cen/(1+z)

    #calculate the velocities
    lam_diff = lamdas-lam_cen
    vels = c.to('km/s')*lam_diff/lam_cen

    return vels


def narrow_line_models(outflow_results, no_outflow_results, statistical_results, lamdas):
    """
    Turns the narrow line fits from koffee into gaussians that can be subtracted
    from the data cube

    Parameters
    ----------
    outflow_results : :obj:'~numpy.ndarray' object
        array with galaxy sigma, center, amplitude and outflow sigma, center,
        amplitude values

    no_outflow_results : :obj:'~numpy.ndarray'
        array with single gaussian sigma, center, amplitude values

    statistical results : :obj:'~numpy.ndarray'
        array with 0 where one gaussian gave a better BIC value, 1 where two
        gaussians gave a better BIC value, and -1 where the S/N wasn't high
        enough for fitting

    lamdas : :obj:'~numpy.ndarray'
        the wavelength vector

    Returns
    -------
    narrow_line_array : :obj:'~numpy.ndarray'
        the array of gaussians which were fit to the narrow line
    """
    #create an array to keep the gaussians in
    narrow_line_array = np.full((lamdas.shape[0], outflow_results.shape[1], outflow_results.shape[2]), np.nan, dtype=np.double)

    #iterate through the cube and create the gaussians
    for i in np.arange(outflow_results.shape[1]):
        for j in np.arange(outflow_results.shape[2]):
            #if there was no fitting done, make it a zero array
            if statistical_results[i,j] < 0:
                narrow_line_array[:,i,j] = np.full_like(lamdas, np.nan, dtype=np.double)

            #if one gaussian was fit, use the no_outflow_results array
            if statistical_results[i,j] == 0:
                narrow_line_array[:,i,j] = kff.gaussian_func(x=lamdas, amp=no_outflow_results[2,i,j], mean=no_outflow_results[1,i,j], sigma=no_outflow_results[0,i,j])

            #if two gaussians were fit, use the outflow_results array
            if statistical_results[i,j] > 0:
                narrow_line_array[:,i,j] = kff.gaussian_func(x=lamdas, amp=outflow_results[2,i,j], mean=outflow_results[1,i,j], sigma=outflow_results[0,i,j])

    return narrow_line_array


def map_flux(flux_array, title, header, radius, vmin=None, vmax=None):
    """
    Maps the flux

    Parameters
    ----------
    flux_array : :obj:'~numpy.ndarray' or list of :obj:'~numpy.ndarray'
        An array with the flux in it

    title : 'str' or list of :obj:'~numpy.ndarray'
        The title for the plot

    header : FITs header object
        the header from the fits file (assumes all arrays have the same header)

    radius : :obj:'~numpy.ndarray'
        An array with the radius in it (assumes all arrays are from the same
        galaxy and so have the same radius array)

    vmin : float or None
        the minimum value for the colourmap (Default is None)
        uses same vmin in all plots

    vmax : float or None
        the maximum value for the colourmap (Default is None)
        uses same vmax in all plots

    Returns
    -------
    A plot mapping the flux with a colorbar.
    """
    #if it's just one array, put it in a list
    if type(flux_array) != list:
        flux_array = [flux_array]
    if type(title) != list:
        title = [title]

    #get the number of arrays
    array_num = len(flux_array)

    #create the wcs coords
    fits_wcs = WCS(header)

    #centre pixel
    centre_pixel = [30,10]
    ylim = [4, 16]
    xlim = [2, 58]

    #calculate the beginning and end of 6.1 arcsec (r_90)
    try:
        r90_pixel_length = abs(6.1/(header['CD1_2']*60*60))
    except KeyError:
        r90_pixel_length = abs(6.1/(header['CDELT1']*60*60))


    start_r90_xpixel = centre_pixel[0]
    start_r90_ypixel = centre_pixel[1] - 5.6
    end_r90_xpixel = start_r90_xpixel - r90_pixel_length

    #calculate the beginning and end of 2.6 arcsec (r_50)
    try:
        r50_pixel_length = abs(2.6/(header['CD1_2']*60*60))
    except KeyError:
        r50_pixel_length = abs(2.6/(header['CDELT1']*60*60))

    start_r50_xpixel = centre_pixel[0]
    start_r50_ypixel = centre_pixel[1] - 4.5
    end_r50_xpixel = start_r50_xpixel - r50_pixel_length


    #create the figure
    plt.rcParams.update(pf.get_rc_params())
    fig, ax = plt.subplots(nrows=1, ncols=array_num, sharey=True, figsize=(array_num*3, 3), subplot_kw={'projection': fits_wcs[0], 'slices': ('y', 'x')}, constrained_layout=True)

    #iterate through the arrays to make maps
    if array_num <= 1:
        plot_array = flux_array[0]
        try:
            cmap = ax.imshow(np.log10(plot_array.value).T, origin='lower', aspect=header['CD2_1']/header['CD1_2'], cmap=cmr.gem, vmin=vmin, vmax=vmax)
        except KeyError:
            cmap = ax.imshow(np.log10(plot_array.value).T, origin='lower', aspect=abs(header['CDELT1']/header['CDELT2']), cmap=cmr.gem, vmin=vmin, vmax=vmax)

        ax.hlines(start_r90_ypixel, start_r90_xpixel, end_r90_xpixel, colors='k')
        ax.text(start_r90_xpixel, start_r90_ypixel+0.4, '$r_{90}$ = 2.4 kpc ', c='k')
        ax.hlines(start_r50_ypixel, start_r50_xpixel, end_r50_xpixel, colors='k')
        ax.text(start_r50_xpixel, start_r50_ypixel+0.4, '$r_{50}$ = 1 kpc ', c='k')

        ax.set_xlim(xlim[0], xlim[1])
        ax.set_ylim(ylim[0], ylim[1])
        ax.invert_xaxis()
        ax.set_title(title[0])

        #find the fraction of flux within 1 kpc
        frac_flux = np.nansum(flux_array[0][radius<1])/np.nansum(flux_array[0])

        #write this on the plot
        ax.text(55, 15, str(frac_flux)+' within 1 kpc', c='k')

        plt.colorbar(cmap, label='Log(10$^{-16}$ erg s$^{-1}$ cm$^{-2}$)')
    else:
        for i in range(array_num):
            plot_array =flux_array[i]

            #map the array
            try:
                cmap = ax[i].imshow(np.log10(plot_array.value).T, origin='lower', aspect=header['CD2_1']/header['CD1_2'], cmap=cmr.gem, vmin=vmin, vmax=vmax)
            except KeyError:
                cmap = ax[i].imshow(np.log10(plot_array.value).T, origin='lower', aspect=abs(header['CDELT2']/header['CDELT1']), cmap=cmr.gem, vmin=vmin, vmax=vmax)


            ax[i].set_xlim(xlim[0], xlim[1])
            ax[i].set_ylim(ylim[0], ylim[1])
            ax[i].invert_xaxis()
            ax[i].set_title(title[i])

            #find the fraction of flux within 1 kpc
            frac_flux = np.nansum(plot_array[radius<1])/np.nansum(plot_array)

            #write this on the plot
            ax[i].text(55, 15, '{:.2%} within 1 kpc'.format(frac_flux), c='k')

        #add bars to the last plot
        ax[i].hlines(start_r90_ypixel, start_r90_xpixel, end_r90_xpixel, colors='k')
        ax[i].text(start_r90_xpixel, start_r90_ypixel+0.4, '$r_{90}$ = 2.4 kpc ', c='k')
        ax[i].hlines(start_r50_ypixel, start_r50_xpixel, end_r50_xpixel, colors='k')
        ax[i].text(start_r50_xpixel, start_r50_ypixel+0.4, '$r_{50}$ = 1 kpc ', c='k')

        plt.colorbar(cmap, label='Log(10$^{-16}$ erg s$^{-1}$ cm$^{-2}$)')

    plt.subplots_adjust(left=0.02, right=0.85, top=0.9, bottom=0.12, wspace=0.06)
    plt.show()


def plot_example_fits(lamdas, data, narrow_gaussians, vels_array, spaxel, disk_limit):
    """
    Plots an example of how the fit works.  Two panels.
    """
    #create the figure
    plt.rcParams.update(pf.get_rc_params())
    fig, ax = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(8,4), constrained_layout=True)

    ax[0].step(lamdas, data[:, spaxel[0], spaxel[1]], where='mid')
    ax[0].plot(lamdas, narrow_gaussians[:, spaxel[0], spaxel[1]], color='grey')

    ax[0].set_ylabel('Flux')
    ax[0].set_xlabel('Wavelength ($\AA$)')

    #subtract the narrow gaussian
    data_subtracted = data[:, spaxel[0], spaxel[1]] - narrow_gaussians[:, spaxel[0], spaxel[1]]
    ax[1].step(vels_array[:, spaxel[0], spaxel[1]], data_subtracted, where='mid')

    ax[1].axvline(0, color='k')
    ax[1].axvline(-disk_limit, c='k', ls='--')
    ax[1].axvline(-300, c='k', ls='--')

    ax[1].text(5, 15, '0 km/s', color='k')
    ax[1].text(-disk_limit-70, 20, '$\sigma_{disk}$='+str(disk_limit)+' km/s', color='k', rotation='vertical', va='top')
    ax[1].text(-400, 20, '$v_{esc}\approx$ 300 km/s', color='k', rotation='vertical', va='top')

    ax[1].set_xlim(-700,700)

    ax[1].set_xlabel('Velocity (km/s)')

    plt.show()


def plot_all_spaxels_vels(vels_array, data_subtracted, radius, disk_limit):
    """
    Plots all the spaxels once their narrow line has been subtracted.
    Coloured by radius.
    """
    #create the figure
    plt.rcParams.update(pf.get_rc_params())
    plt.figure(figsize=(6,4))

    #find the number of individual radii within 5 kpc
    radii_unique = np.unique(np.around(radius[radius<=5], decimals=1))
    radii_num = radii_unique.shape[0]

    #get colours from cmasher
    colours = cmr.take_cmap_colors('cmr.gem', radii_num, cmap_range=(0.15, 0.95), return_fmt='hex')

    cmap = cmr.gem
    norm = Normalize(vmin=radii_unique.min(), vmax=radii_unique.max())

    #iterate through the radius array
    for i in np.arange(radius.shape[0]):
        for j in np.arange(radius.shape[1]):
            #plot it if the radius is less than 5 kpc
            if radius[i,j] <= 5:
                rad = np.around(radius[i,j], decimals=1)
                #plt.step(vels_array[:,i,j], data_subtracted[:,i,j]/np.nanmax(data_subtracted[:,i,j]), where='mid', c=colours[np.where(radii_unique==rad)[0][0]], alpha=0.5)
                plt.step(vels_array[:,i,j], data_subtracted[:,i,j], where='mid', c=colours[np.where(radii_unique==rad)[0][0]], alpha=0.5)

    plt.axvline(0, color='k')
    plt.axvline(-disk_limit, c='k', ls='--')
    plt.axvline(-300, c='k', ls='--')

    plt.text(5, 180, '0 km/s', color='k')
    plt.text(-disk_limit-70, 180, '$\sigma_{disk}$='+str(disk_limit)+' km/s', color='k', rotation='vertical', va='top')
    plt.text(-400, 180, '$v_{esc}\approx$ 300 km/s', color='k', rotation='vertical', va='top')

    cbar = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap))
    cbar.set_label(label='Radius (kpc)')#, rotation=270, labelpad=20)

    plt.xlim(-700,700)

    plt.xlabel('Velocity (km/s)')
    #plt.ylabel('Normalised Flux')
    plt.ylabel('Flux')

    plt.show()


def plot_sigma_sfr_flux(flux_array, sigma_sfr, labels):
    """
    Plots the flux in the different bands against the SFR surface density
    """
    #if it's just one array, put it in a list
    if type(flux_array) != list:
        flux_array = [flux_array]
    if type(labels) != list:
        labels = [labels]

    #get number of flux arrays
    flux_array_num = len(flux_array)

    #create the figure
    fig = plt.figure(figsize=(5,4))

    #get colours from cmasher
    colours = cmr.take_cmap_colors('cmr.gem', flux_array_num, cmap_range=(0.25, 0.85), return_fmt='hex')

    #plot the points
    for i in np.arange(flux_array_num):
        print(sigma_sfr.shape, flux_array[i].shape)
        plt.scatter(np.log10(sigma_sfr), np.log10(flux_array[i].value), c=colours[i], label=labels[i])

    plt.xlabel('Log($\Sigma_{SFR}$ [M$_\odot$ yr$^{-1}$ kpc$^{-2}$])')
    plt.ylabel('Log(Flux [10$^{-16}$ erg s$^{-1}$ cm$^{-2}$])')
    plt.legend()


def plot_sigma_sfr_flux_ratio(flux_array, sigma_sfr, labels):
    """
    Plots the ratio of flux to total band flux in the different bands
    against the SFR surface density
    """
    #if it's just one array, put it in a list
    if type(flux_array) != list:
        flux_array = [flux_array]
    if type(labels) != list:
        labels = [labels]

    #get number of flux arrays
    flux_array_num = len(flux_array)

    #create the figure
    fig = plt.figure(figsize=(5,4))

    #get colours from cmasher
    colours = cmr.take_cmap_colors('cmr.gem', flux_array_num, cmap_range=(0.25, 0.85), return_fmt='hex')

    #plot the points
    for i in np.arange(flux_array_num):
        print(sigma_sfr.shape, flux_array[i].shape)
        plt.scatter(np.log10(sigma_sfr), np.log10(flux_array[i]/np.nansum(flux_array[i])), c=colours[i], label=labels[i])

    plt.xlabel('Log($\Sigma_{SFR}$ [M$_\odot$ yr$^{-1}$ kpc$^{-2}$])')
    plt.ylabel('Log(Flux/Band Flux)')
    plt.legend()


def plot_sigma_sfr_total_flux_ratio(flux_array, sigma_sfr, labels):
    """
    Plots the ratio of flux to total flux in the different bands against
    the SFR surface density
    """
    #if it's just one array, put it in a list
    if type(flux_array) != list:
        flux_array = [flux_array]
    if type(labels) != list:
        labels = [labels]

    #get number of flux arrays
    flux_array_num = len(flux_array)

    #add all the flux up in each spaxel
    # NOTE: IF THERE ARE TWO NAN VALUES THIS RETURNS A ZERO
    total_flux = np.full_like(flux_array[0], np.nan, dtype=np.double)
    for i in np.arange(flux_array_num):
        total_flux = np.nansum(np.dstack((total_flux, flux_array[i])), axis=2)

    #get rid of the zero values
    total_flux[total_flux==0.0] = np.nan


    #create the figure
    fig = plt.figure(figsize=(5,4))

    #get colours from cmasher
    colours = cmr.take_cmap_colors('cmr.gem', flux_array_num, cmap_range=(0.25, 0.85), return_fmt='hex')

    #plot the points
    for i in np.arange(flux_array_num):
        print(sigma_sfr.shape, flux_array[i].shape)
        plt.scatter(np.log10(sigma_sfr), np.log10(flux_array[i]/total_flux), c=colours[i], label=labels[i])

    plt.xlabel('Log($\Sigma_{SFR}$ [M$_\odot$ yr$^{-1}$ kpc$^{-2}$])')
    plt.ylabel('Log(Band Flux/Total Spaxel Flux)')
    plt.legend()

    plt.subplots_adjust(left=0.14, right=0.95, top=0.95, bottom=0.12)



def calc_vel_bands(data_file, original_data_file, koffee_folder, galaxy_name, z, results_folder, data_shape=[67,24]):
    """
    Does all the things
    """
    #set the velocity limit for the disk gas
    #disk_limit = 30 #km/s molecular gas velocity dispersion
    #disk_limit = 40 * 2.35
    disk_limit = 40 * 2.35 / 2

    #read in the data fits file
    data_stuff = pc.load_data(data_file, mw_correction=False)

    if len(data_stuff) > 3:
        lamdas, data, var, header = data_stuff
    else:
        lamdas, data, header = data_stuff

    #read in the non-continuum subtracted cube to find the radius array
    data_stuff = pc.prepare_single_cube(original_data_file, galaxy_name, z, 'red', results_folder, data_crop=False, var_filepath=None, var_crop=False, lamda_crop=False, mw_correction=False)

    if len(data_stuff) > 10:
        lamdas_metacube, var_lamdas_metacube, xx_metacube, yy_metacube, rad_metacube, data_metacube, var_metacube, xx_flat_metacube, yy_flat_metacube, rad_flat_metacube, data_flat_metacube, var_flat_metacube, header_metacube = data_stuff
        del lamdas_metacube, xx_metacube, yy_metacube, data_metacube, xx_flat_metacube, yy_flat_metacube, rad_flat_metacube, data_flat_metacube, var_lamdas_metacube, var_metacube, var_flat_metacube
    else:
        lamdas_metacube, xx_metacube, yy_metacube, rad_metacube, data_metacube, xx_flat_metacube, yy_flat_metacube, rad_flat_metacube, data_flat_metacube, header_metacube = data_stuff
        del lamdas_metacube, xx_metacube, yy_metacube, data_metacube, xx_flat_metacube, yy_flat_metacube, rad_flat_metacube, data_flat_metacube

    #the radius array is in arcseconds - need to convert to kpc
    #get the proper distance per arcsecond
    proper_dist = cosmo.kpc_proper_per_arcmin(z).to(units.kpc/units.arcsec)
    #multiply the array
    rad_metacube = rad_metacube*proper_dist.value

    #read in the koffee fits
    out_res, out_err, no_out_res, no_out_err, stat_res, bic = koffee.read_output_files(koffee_folder, galaxy_name, include_const=True, emission_line1='OIII_4', emission_line2=None, OII_doublet=False, data_shape=data_shape)

    #read in the sigma sfr
    sigma_sfr, sigma_sfr_header, sigma_sfr_wcs = pf.read_in_create_wcs(koffee_folder+galaxy_name+'_star_formation_rate_surface_density.fits')
    print(sigma_sfr.shape)


    #create mask for OIII 5007
    OIII_mask = (lamdas > 5007*(1+z)-20.0) & (lamdas < 5007*(1+z)+20.0)

    #mask the lamdas and the data
    masked_lamdas = lamdas[OIII_mask]
    masked_data = data[OIII_mask,:,:]
    #masked_var = var[OIII_mask,:,:]

    print(masked_data.shape)

    #find the location of the maximum flux
    max_flux_loc = np.argmax(masked_data, axis=0)

    #use that to find where the zero velocity should be
    #create a new array to fill with the answers
    lam_cen = np.full_like(max_flux_loc, np.nan, dtype=np.double)
    for i in np.arange(max_flux_loc.shape[0]):
        for j in np.arange(max_flux_loc.shape[1]):
            lam_cen[i,j] = masked_lamdas[max_flux_loc[i,j]]


    #now create a new array to keep the velocity vectors in
    vels_array = np.full_like(masked_data, np.nan, dtype=np.double)

    #iterate through the array and create the velocity vectors
    for i in np.arange(lam_cen.shape[0]):
        for j in np.arange(lam_cen.shape[1]):
            vels_array[:,i,j] = convert_wave_to_vel(masked_lamdas, lam_cen[i,j], z, deredshift=True)

    #create a cube of narrow line gaussian models from the fits
    narrow_gaussians = narrow_line_models(out_res, no_out_res, stat_res, masked_lamdas)
    print(narrow_gaussians.shape)

    #create an example plot
    #plot_example_fits(masked_lamdas, masked_data, narrow_gaussians, vels_array, spaxel=[30,10], disk_limit=disk_limit)
    #plot_example_fits(masked_lamdas, masked_data, narrow_gaussians, vels_array, spaxel=[10,25], disk_limit=disk_limit)

    #subtract the fitted narrow line from the data
    data_subtracted = masked_data - narrow_gaussians

    #if it's IRAS08, get rid of spaxel (4,11)
    if galaxy_name == 'IRAS08':
        data_subtracted[:,4,11] = np.full_like(masked_lamdas, np.nan, dtype=np.double)
    #try:
    #    data_subtracted = masked_data - narrow_gaussians
    #except ValueError:
    #    data_subtracted = masked_data - narrow_gaussians[:,-masked_data.shape[1]:,:]
    #    stat_res = stat_res[-masked_data.shape[1]:,:]
    #    sigma_sfr = sigma_sfr.T[-masked_data.shape[1]:,:]
    #    rad_metacube = rad_metacube[-masked_data.shape[1]:,:]

    #plot those
    #plot_all_spaxels_vels(vels_array, data_subtracted, rad_metacube, disk_limit)

    #create masks for the different bands using the velocities in km/s
    disk_mask = (vels_array>0) & (vels_array<=disk_limit)
    fountain_mask = (vels_array>disk_limit) & (vels_array<300)
    escape_mask = (vels_array>=300) & (vels_array<600)

    #create arrays of spectra split into the different bands
    disk_array = np.full_like(masked_data, np.nan, dtype=np.double)
    disk_array[fountain_mask] = data_subtracted[fountain_mask]

    fountain_array = np.full_like(masked_data, np.nan, dtype=np.double)
    fountain_array[disk_mask] = data_subtracted[disk_mask]

    escape_array = np.full_like(masked_data, np.nan, dtype=np.double)
    escape_array[escape_mask] = data_subtracted[escape_mask]

    #mask out things that didn't get past the S/N requirements for koffee
    disk_array[:,(stat_res<0)] = np.nan
    fountain_array[:,(stat_res<0)] = np.nan
    escape_array[:,(stat_res<0)] = np.nan

    #find the flux for the different bands
    disk_flux = calc_line_flux(disk_array, dx=0.5)
    fountain_flux = calc_line_flux(fountain_array, dx=0.5)
    escape_flux = calc_line_flux(escape_array, dx=0.5)

    #mask the flux that isn't above 10^-0.1
    #disk_flux[np.log10(disk_flux.value)<-0.1] = np.nan
    #fountain_flux[np.log10(fountain_flux.value)<-0.1] = np.nan
    #escape_flux[np.log10(escape_flux.value)<-0.1] = np.nan

    #make all the plots
    plot_sigma_sfr_flux([disk_flux, fountain_flux, escape_flux], sigma_sfr, labels=['Flux from disk gas', 'Flux from fountain gas', 'Flux from escaping gas'])

    plot_sigma_sfr_flux_ratio([disk_flux, fountain_flux, escape_flux], sigma_sfr, labels=['Flux from disk gas', 'Flux from fountain gas', 'Flux from escaping gas'])

    plot_sigma_sfr_total_flux_ratio([disk_flux, fountain_flux, escape_flux], sigma_sfr, labels=['Flux from disk gas', 'Flux from fountain gas', 'Flux from escaping gas'])

    #map the flux
    vmin=-0.5 #-2.0
    vmax=1.7 #1.5
    #map_flux(disk_flux, title='Flux from disk gas', header=header, radius=rad_metacube, vmin=vmin, vmax=vmax)
    #map_flux(fountain_flux, title='Flux from fountain gas', header=header, radius=rad_metacube, vmin=vmin, vmax=vmax)
    #map_flux(escape_flux, title='Flux from escaping gas', header=header, radius=rad_metacube, vmin=vmin, vmax=vmax)

    map_flux([disk_flux, fountain_flux, escape_flux], title=['Flux from disk gas', 'Flux from fountain gas', 'Flux from escaping gas'], header=header, radius=rad_metacube, vmin=vmin, vmax=vmax)


    return disk_flux, fountain_flux, escape_flux





#===============================================================================
# MAIN
#===============================================================================
if __name__ == '__main__':
    #data file
    data_file = '../../../data/IRAS08/IRAS08_combined_metacube_all_corrections.fits'

    #folder where the fits are
    koffee_folder = '../../../code_outputs/koffee_results_IRAS08/IRAS08koffee_results_OIII_4_2022-01-28_resolved/'

    #galaxy name
    galaxy_name = 'IRAS08'

    #redshift
    z = 0.018950

    #data_shape
    data_shape_IRAS08 = [67, 24]
    data_shape_cgcg453 = [36,23]
