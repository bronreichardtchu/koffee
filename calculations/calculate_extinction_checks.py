"""
NAME:
	calculate_extinction_checks.py

AUTHOR:
	Bronwyn Reichardt Chu
	Swinburne
	2020

EMAIL:
	<breichardtchu@swin.edu.au>

PURPOSE:
	To check the reddening and extinction corrections applied to the data cubes.
	Written on MacOS Mojave 10.14.5, with Python 3.7

FUNCTIONS INCLUDED:
    integrate_spectrum
    subtract_continuum
    calc_hbeta_hgamma_integrals
    calc_hbeta_hgamma_amps
    calc_OIII_doublet_ratio
    plot_ratio_histograms

DICTIONARIES INCLUDED:
    all_the_lines
    emission_line_limits
    continuum_band_limits

MODIFICATION HISTORY:
		v.1.0 - first created February 2021

"""
import numpy as np

import cmasher as cmr
import matplotlib.pyplot as plt

from astropy.cosmology import WMAP9 as cosmo
from astropy.constants import c
from astropy.constants import m_p
from astropy import units as u

from koffee.calculations import calculate_star_formation_rate as calc_sfr
from koffee.plotting_scripts import plotting_functions as pf
#import calculate_star_formation_rate as calc_sfr
#import plotting_functions as pf

#===============================================================================
#DICTIONARIES
#===============================================================================
#wavelengths of emission lines at rest in vacuum, taken from
#http://classic.sdss.org/dr6/algorithms/linestable.html
all_the_lines = {
    "Hdelta" : 4102.89,
    "Hgamma" : 4341.68,
    "Hbeta" : 4862.68,
    "Halpha": 6564.61,
    "OII_1" : 3727.092,
    "OII_2" : 3729.875,
    "HeI" : 3889.0,
    "SII" : 4072.3,
    "OIII_1" : 4364.436,
    "OIII_2" : 4932.603,
    "OIII_3" : 4960.295,
    "OIII_4" : 5008.240
}

#giving left and right limits of where to look for emission lines
emission_line_limits = {
    "Hgamma_left" : 4335.68,
    "Hgamma_right" : 4347.68,
    "Hbeta_left" : 4855.68,
    "Hbeta_right" : 4869.68,
    "OIII_3_left" : 4953.0,
    "OIII_3_right" : 4967.0,
    "OIII_4_left" : 5003.0,
    "OIII_4_right" : 5013.0
}

#giving left and right limits of where to take the median in the continuum near
#emission lines
continuum_band_limits = {
    "Hgamma_left" : 4315.0,
    "Hgamma_right" : 4320.0,
    "Hbeta_left" : 4850.0,
    "Hbeta_right" : 4855.0,
    "OIII_3_left" : 4941.0,
    "OIII_3_right" : 4946.0,
    "OIII_4_left" : 4990.0,
    "OIII_4_right" : 4995.0
}

#===============================================================================
#FUNCTIONS
#===============================================================================

def integrate_spectrum(lamdas, spectrum, left_limit, right_limit, plot=False):
    """
    Calculate integrates along the spectrum between the given wavelength limits
    using np.trapz.  The spectrum is in 10^-16 erg/s/cm^2/Ang.  Need to change
    the returned result to erg/s to get the flux

    Luminosities should be around 10^40

    Parameters
    ----------
    lamdas : :obj:'~numpy.ndarray'
        the wavelength vector, same length as the given spectrum

    spectrum : :obj:'~numpy.ndarray'
        the spectrum or array of spectra.  If in array, needs to be in shape
        [npix, nspec]

    left_limit : :obj:'~numpy.ndarray'
        The left-hand wavelength limit of the region to integrate over

    right_limit : :obj:'~numpy.ndarray'
        The right-hand wavelength limit of the region to integrate over

    plot : boolean
        if True, plots the area of the spectrum used for the integral.  Default
        is False.

    Returns
    -------
    integral : float or :obj:'~numpy.ndarray'
        the flux of the line
    """
    # create a new array to keep the integrals in 
    integral = np.zeros((spectrum.shape[1]))

    if plot == True:
        fig, ax = plt.subplots(ncols=2, nrows=1)

    # iterate over each spectrum
    for i in np.arange(spectrum.shape[1]):
        # use the wavelengths to find the values in the spectrum to integrate over
        spec = spectrum[(lamdas>=left_limit[i])&(lamdas<=right_limit[i]),i]
        lam = lamdas[(lamdas>=left_limit[i]) & (lamdas<=right_limit[i])]

        #plot the area used
        if plot == True:
            #ax.step(lam, spec.reshape([lam.shape[0],-1]), where='mid')
            ax[0].step(lam, spec, where='mid')
            ax[1].step(np.arange(lam.shape[0])-lam.shape[0]/2, spec, where='mid')
            plt.show()

        #integrate along the spectrum
        #by integrating, the units are now 10^-16 erg/s/cm^2
        integral[i] = np.trapz(spec, lam, axis=0)
        
    integral = integral*10**(-16)*u.erg/(u.s*(u.cm*u.cm))

    return integral


def subtract_continuum(lamdas, spectrum, left_limit, right_limit):
    """
    Subtract a median value of a section of continuum to the blue of the emission
    line using the bounds given by left_limit and right_limit.

    Parameters
    ----------
    lamdas : :obj:'~numpy.ndarray'
        the wavelength vector, same length as the given spectrum

    spectrum : :obj:'~numpy.ndarray'
        the spectrum or array of spectra.  If in array, needs to be in shape
        [npix, nspec]

    left_limit : float
        The left-hand wavelength limit of the region to integrate over

    right_limit : float
        The right-hand wavelength limit of the region to integrate over

    plot : boolean
        if True, plots the area of the spectrum used for the integral.  Default
        is False.

    Returns
    -------
    spectrum : :obj:'~numpy.ndarray'
        the continuum subtracted continuum

    s_n : :obj:'~numpy.ndarray'
        an array the shape of the input array of spectra giving the signal to
        noise ratio in the continuum for each spectrum
    """
    #use the given limits to define the section of continuum
    cont = spectrum[(lamdas >= left_limit) & (lamdas <= right_limit),]

    #find the median of the continuum
    cont_median = np.nanmedian(cont, axis=0)

    #minus off the continuum value
    spectrum = spectrum - cont_median

    #find the standard deviation of the continuum section
    noise = np.std(cont, axis=0)

    #calculate the signal to noise ratio
    s_n = (cont_median/noise)

    return spectrum, s_n


def create_new_bandwidth(lamdas, spectrum, left_limit, right_limit):
    """Finds the wavelength of the peak within the given limits, and then creates
    a band +/- 5A around that wavelength.

    Parameters
    ----------
    lamdas : :obj:'~numpy.ndarray'
        the wavelength vector, same length as the given spectrum
    spectrum : :obj:'~numpy.ndarray'
        spectrum or array of spectra. If in array, needs to be in shape
        [npix, nspec]
    left_limit : float
        the left (blue) wavelength limit of the range to search for the max in 
    right_limit : float
        the right (red) wavelength limit of the range to search for the max in

    Returns
    -------
    float
        the new wavelength boundaries
    """
    # create the range 
    lam_mask = (lamdas>=left_limit) & (lamdas<=right_limit)

    # find the peak value within the range
    peak_idx = np.nanargmax(spectrum[lam_mask], axis=0)

    # find the wavelength value 
    peak_lam = lamdas[lam_mask][peak_idx]

    # create new limits
    new_left_limit = peak_lam - 6.0
    new_right_limit = peak_lam + 6.0

    return new_left_limit, new_right_limit


def calc_hbeta_hgamma_integrals(lamdas, spectrum, z, cont_subtract=False, plot=False):
    """
    Calculate the Hbeta/hgamma ratio, which should be 2.13, by integrating over the
    emission lines.

    Parameters
    ----------
    lamdas : :obj:'~numpy.ndarray'
        the wavelength vector (must be in rest wavelengths, not observed)

    spectrum : :obj:'~numpy.ndarray'
        the spectrum or array of spectra.

    z : float
        redshift of the galaxy - used to calculate the flux, not applied to the 
        wavelengths.

    cont_subtract : boolean
        if True, assumes continuum has not already been subtracted.

    plot : boolean
        if True, plots the area of the spectrum used for the integral.  Default
        is False.

    Returns
    -------
    hbeta_hgamma : float or :obj:'~numpy.ndarray'
        the hbeta to hgamma ratio for each spectrum

    s_n_mask : :obj:'~numpy.ndarray'
        Returned if cont_subtract==True.  The spectrum is used before it is
        continuum subtracted to find the signal-to-noise value near the Hbeta
        line for each spectrum
    """
    #create bounds to look for the peak value within
    hgamma_left_limit = emission_line_limits["Hgamma_left"]*(1+z)
    hgamma_right_limit = emission_line_limits["Hgamma_right"]*(1+z)

    hbeta_left_limit = emission_line_limits["Hbeta_left"]*(1+z)
    hbeta_right_limit = emission_line_limits["Hbeta_right"]*(1+z)

    #create bounds for the continuum
    hgamma_cont_left_limit = continuum_band_limits["Hgamma_left"]*(1+z)
    hgamma_cont_right_limit = continuum_band_limits["Hgamma_right"]*(1+z)

    hbeta_cont_left_limit = continuum_band_limits["Hbeta_left"]*(1+z)
    hbeta_cont_right_limit = continuum_band_limits["Hbeta_right"]*(1+z)

    # flatten data 
    flat_data = spectrum.reshape((lamdas.shape[0],-1))

    #if the continuum has not already been fit and subtracted, use an approximation
    #to subtract it off
    #also use the continuum to find the S/N and mask things
    if cont_subtract == True:
        hgamma_spec, s_n_hgamma = subtract_continuum(lamdas, flat_data, hgamma_cont_left_limit, hgamma_cont_right_limit)

        hbeta_spec, s_n_hbeta = subtract_continuum(lamdas, flat_data, hbeta_cont_left_limit, hbeta_cont_right_limit)

        #create the S/N mask
        s_n_mask = s_n_hbeta > 20

        # create the new limits from the peak wavelength
        hgamma_left_limit, hgamma_right_limit = create_new_bandwidth(lamdas, hgamma_spec, hgamma_left_limit, hgamma_right_limit)
        hbeta_left_limit, hbeta_right_limit = create_new_bandwidth(lamdas, hbeta_spec, hbeta_left_limit, hbeta_right_limit)

        #integrate over the emission lines
        hgamma_integral = integrate_spectrum(lamdas, hgamma_spec, hgamma_left_limit, hgamma_right_limit, plot=plot)
        hbeta_integral = integrate_spectrum(lamdas, hbeta_spec, hbeta_left_limit, hbeta_right_limit, plot=plot)

    elif cont_subtract == False:
        # create the new limits from the peak wavelength
        hgamma_left_limit, hgamma_right_limit = create_new_bandwidth(lamdas, flat_data, hgamma_left_limit, hgamma_right_limit)
        hbeta_left_limit, hbeta_right_limit = create_new_bandwidth(lamdas, flat_data, hbeta_left_limit, hbeta_right_limit)

        #integrate over the emission lines
        hgamma_integral = integrate_spectrum(lamdas, flat_data, hgamma_left_limit, hgamma_right_limit, plot=plot)
        hbeta_integral = integrate_spectrum(lamdas, flat_data, hbeta_left_limit, hbeta_right_limit, plot=plot)

    #now get rid of the cm^2
    #get the Hubble constant at z=0; this is in km/Mpc/s
    H_0 = cosmo.H(0)
    #use d = cz/H0 to find the distance in cm
    dist = (c*z/H_0).decompose().to('cm')
    print('distance:', dist)

    #multiply by 4*pi*d^2 to get rid of the cm
    hgamma_flux = (hgamma_integral*(4*np.pi*(dist**2))).to('erg/s')
    hbeta_flux = (hbeta_integral*(4*np.pi*(dist**2))).to('erg/s')

    #print('Hgamma flux:', hgamma_flux)
    #print('Hbeta flux:', hbeta_flux)

    #calculate the hbeta/hgamma ratio
    hbeta_hgamma = hbeta_flux/hgamma_flux

    print('Median Hbeta/Hgamma:', np.nanmedian(hbeta_hgamma))

    # reshape everything to the data shape 
    if len(spectrum.shape)>2:
        hbeta_flux = hbeta_flux.reshape((spectrum.shape[1], spectrum.shape[2]))
        hgamma_flux = hgamma_flux.reshape((spectrum.shape[1], spectrum.shape[2]))
        hbeta_hgamma = hbeta_hgamma.reshape((spectrum.shape[1], spectrum.shape[2]))

    if cont_subtract == True:
        return hbeta_flux, hgamma_flux, hbeta_hgamma, s_n_mask
    else:
        return hbeta_flux, hgamma_flux, hbeta_hgamma



def calc_hbeta_hgamma_amps(lamdas, spectrum, z=0.0, cont_subtract=False):
    """
    Calculate the Hbeta/hgamma ratio, which should be 2.13, by using the amplitudes
    of the emission lines

    Parameters
    ----------
    lamdas : :obj:'~numpy.ndarray'
        the wavelength vector

    spectrum : :obj:'~numpy.ndarray'
        the spectrum or array of spectra.  If in array, needs to be in shape
        [npix, nspec]

    z : float
        redshift of the galaxy, applied to the wavelengths.  Default is zero, 
        which assumes that the lamdas are in rest wavelength.

    cont_subtract : boolean
        if True, assumes continuum has not already been subtracted.

    Returns
    -------
    hbeta_hgamma : float or :obj:'~numpy.ndarray'
        the hbeta to hgamma ratio for each spectrum

    s_n_mask : :obj:'~numpy.ndarray'
        Returned if cont_subtract==True.  The spectrum is used before it is
        continuum subtracted to find the signal-to-noise value near the Hbeta
        line for each spectrum
    """
    #create bounds to look for highest amplitude value within
    hgamma_left_limit = emission_line_limits["Hgamma_left"]*(1+z)
    hgamma_right_limit = emission_line_limits["Hgamma_right"]*(1+z)

    hbeta_left_limit = emission_line_limits["Hbeta_left"]*(1+z)
    hbeta_right_limit = emission_line_limits["Hbeta_right"]*(1+z)

    #create bounds for the continuum
    hgamma_cont_left_limit = continuum_band_limits["Hgamma_left"]*(1+z)
    hgamma_cont_right_limit = continuum_band_limits["Hgamma_right"]*(1+z)

    hbeta_cont_left_limit = continuum_band_limits["Hbeta_left"]*(1+z)
    hbeta_cont_right_limit = continuum_band_limits["Hbeta_right"]*(1+z)

    #if the continuum has not already been fit and subtracted, use an approximation
    #to subtract it off
    #also use the continuum to find the S/N and mask things
    if cont_subtract == True:
        hgamma_spec, s_n_hgamma = subtract_continuum(lamdas, spectrum, hgamma_cont_left_limit, hgamma_cont_right_limit)

        hbeta_spec, s_n_hbeta = subtract_continuum(lamdas, spectrum, hbeta_cont_left_limit, hbeta_cont_right_limit)

        #create the S/N mask
        s_n_mask = s_n_hbeta > 20

        #find the highest amplitude
        hgamma_amp = np.nanmax(hgamma_spec[(lamdas>=hgamma_left_limit)&(lamdas<=hgamma_right_limit),], axis=0)
        hbeta_amp = np.nanmax(hbeta_spec[(lamdas>=hbeta_left_limit)&(lamdas<=hbeta_right_limit),], axis=0)

    elif cont_subtract == False:
        #find the highest amplitude
        hgamma_amp = np.nanmax(spectrum[(lamdas>=hgamma_left_limit)&(lamdas<=hgamma_right_limit),], axis=0)
        hbeta_amp = np.nanmax(spectrum[(lamdas>=hbeta_left_limit)&(lamdas<=hbeta_right_limit),], axis=0)

    #print('Hgamma amplitudes:', hgamma_amp)
    #print('Hbeta amplitudes:', hbeta_amp)

    #calculate the hbeta/hgamma ratio
    hbeta_hgamma = hbeta_amp/hgamma_amp

    print('Median Hbeta/Hgamma:', np.nanmedian(hbeta_hgamma))

    if cont_subtract == True:
        return hbeta_amp, hgamma_amp, hbeta_hgamma, s_n_mask
    else:
        return hbeta_amp, hgamma_amp, hbeta_hgamma


def calc_OIII_doublet_ratio(lamdas, spectrum, z, cont_subtract=False, plot=False):
    """
    Calculate the OIII 4959/5007 ratio, which should be 0.34

    Parameters
    ----------
    lamdas : :obj:'~numpy.ndarray'
        the wavelength vector (must be in rest wavelengths, not observed)

    spectrum : :obj:'~numpy.ndarray'
        the spectrum or array of spectra.  If in array, needs to be in shape
        [npix, nspec]

    z : float
        redshift of the galaxy - used to calculate the flux, not applied to the 
        wavelengths.

    cont_subtract : boolean
        if True, assumes continuum has not already been subtracted.

    plot : boolean
        if True, plots the area of the spectrum used for the integral.  Default
        is False.

    Returns
    -------
    hbeta_hgamma : float or :obj:'~numpy.ndarray'
        the hbeta to hgamma ratio for each spectrum

    s_n_mask : :obj:'~numpy.ndarray'
        Returned if cont_subtract==True.  The spectrum is used before it is
        continuum subtracted to find the signal-to-noise value near the Hbeta
        line for each spectrum
    """
    #create bounds to integrate over
    OIII_4959_left_limit = emission_line_limits["OIII_3_left"]*(1+z)
    OIII_4959_right_limit = emission_line_limits["OIII_3_right"]*(1+z)

    OIII_5007_left_limit = emission_line_limits["OIII_4_left"]*(1+z)
    OIII_5007_right_limit = emission_line_limits["OIII_4_right"]*(1+z)

    #create bounds for the continuum
    OIII_4959_cont_left_limit = continuum_band_limits["OIII_3_left"]*(1+z)
    OIII_4959_cont_right_limit = continuum_band_limits["OIII_3_right"]*(1+z)

    OIII_5007_cont_left_limit = continuum_band_limits["OIII_4_left"]*(1+z)
    OIII_5007_cont_right_limit = continuum_band_limits["OIII_4_right"]*(1+z)

    # flatten data 
    flat_data = spectrum.reshape((lamdas.shape[0],-1))

    #if the continuum has not already been fit and subtracted, use an approximation
    #to subtract it off
    #also use the continuum to find the S/N and mask things
    if cont_subtract == True:
        OIII_4959_spec, s_n_OIII_4959 = subtract_continuum(lamdas, flat_data, OIII_4959_cont_left_limit, OIII_4959_cont_right_limit)
        OIII_5007_spec, s_n_OIII_5007 = subtract_continuum(lamdas, flat_data, OIII_5007_cont_left_limit, OIII_5007_cont_right_limit)

        # create the new limits from the peak wavelength
        OIII_4959_left_limit, OIII_4959_right_limit = create_new_bandwidth(lamdas, OIII_4959_spec, OIII_4959_left_limit, OIII_4959_right_limit)
        OIII_5007_left_limit, OIII_5007_right_limit = create_new_bandwidth(lamdas, OIII_5007_spec, OIII_5007_left_limit, OIII_5007_right_limit)

        #integrate over the emission lines
        OIII_4959_integral = integrate_spectrum(lamdas, OIII_4959_spec, OIII_4959_left_limit, OIII_4959_right_limit, plot=plot)
        OIII_5007_integral = integrate_spectrum(lamdas, OIII_5007_spec, OIII_5007_left_limit, OIII_5007_right_limit, plot=plot)

    elif cont_subtract == False:
        # create the new limits from the peak wavelength
        OIII_4959_left_limit, OIII_4959_right_limit = create_new_bandwidth(lamdas, flat_data, OIII_4959_left_limit, OIII_4959_right_limit)
        OIII_5007_left_limit, OIII_5007_right_limit = create_new_bandwidth(lamdas, flat_data, OIII_5007_left_limit, OIII_5007_right_limit)

        #integrate over the emission lines
        OIII_4959_integral = integrate_spectrum(lamdas, flat_data, OIII_4959_left_limit, OIII_4959_right_limit, plot=plot)
        OIII_5007_integral = integrate_spectrum(lamdas, flat_data, OIII_5007_left_limit, OIII_5007_right_limit, plot=plot)

    #now get rid of the cm^2
    #get the Hubble constant at z=0; this is in km/Mpc/s
    H_0 = cosmo.H(0)
    #use d = cz/H0 to find the distance in cm
    dist = (c*z/H_0).decompose().to('cm')
    print('distance:', dist)

    #multiply by 4*pi*d^2 to get rid of the cm
    OIII_4959_flux = (OIII_4959_integral*(4*np.pi*(dist**2))).to('erg/s')
    OIII_5007_flux = (OIII_5007_integral*(4*np.pi*(dist**2))).to('erg/s')

    #print(OIII_4959_flux)
    #print(OIII_5007_flux)

    #calculate the hbeta/hgamma ratio
    OIII_4959_OIII_5007 = OIII_4959_flux/OIII_5007_flux

    print('Median [OIII]4959/[OIII]5007:', np.nanmedian(OIII_4959_OIII_5007))

    # reshape everything to the data shape 
    if len(spectrum.shape)>2:
        OIII_4959_flux = OIII_4959_flux.reshape((spectrum.shape[1], spectrum.shape[2]))
        OIII_5007_flux = OIII_5007_flux.reshape((spectrum.shape[1], spectrum.shape[2]))
        OIII_4959_OIII_5007 = OIII_4959_OIII_5007.reshape((spectrum.shape[1], spectrum.shape[2]))

    return OIII_4959_OIII_5007


def plot_ratio_histograms(ratio_arrays, array_labels, xlabel, range=None):
    """
    Plot the OIII 4959/5007 ratio, which should be 0.34 (2.94 is the inverse)

    Parameters
    ----------
    ratio_arrays : list of :obj:'~numpy.ndarray'
        a list of all the arrays to be plotted on the histogram, these should be
        flattened

    array_labels : list of strings
        a list containing descriptors/labels for the arrays, to use in the legend

    xlabel : string
        the xlabel, which will also be used within the plot title
        e.g. Hbeta/Hgamma

    range : tuple or None
        The lower and upper range of the bins. Lower and upper outliers are
        ignored. If not provided, range is (x.min(), x.max()). If range is
        specified, autoscaling is based on the specified bin range instead of
        the range of x values.

    Returns
    -------
    A one panel figure of histograms of the input ratios, with vertical lines
    showing the median values
    """
    #create the figure
    plt.rcParams.update(pf.get_rc_params())
    plt.figure()

    #get colours from cmasher
    colours = cmr.take_cmap_colors('cmr.gem', len(ratio_arrays), cmap_range=(0.25, 0.85), return_fmt='hex')

    #for each array, create the histogram
    for i in np.arange(len(ratio_arrays)):
        try:
            plt.hist(ratio_arrays[i].value, bins=50, range=range, alpha=0.5, color=colours[i])
        except:
            plt.hist(ratio_arrays[i], bins=50, range=range, alpha=0.5, color=colours[i])

    #over the top, plot the median value lines
    for i in np.arange(len(ratio_arrays)):
        plt.axvline(np.nanmedian(ratio_arrays[i]), c=colours[i], label=array_labels[i]+'\n Median: {:.2f}'.format(np.nanmedian(ratio_arrays[i])))

    #plot labels
    plt.legend()
    plt.title(xlabel+' ratios for different cubes')
    plt.xlabel(xlabel)

    plt.show()
