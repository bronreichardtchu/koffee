"""
NAME:
	calculate_star_formation_rate.py

AUTHOR:
	Bronwyn Reichardt Chu
	Swinburne
	2020

EMAIL:
	<breichardtchu@swin.edu.au>

PURPOSE:
	To calculate the SFR and SFR surface density of a data cube.
	Written on MacOS Mojave 10.14.5, with Python 3.7

FUNCTIONS INCLUDED:
    calc_hbeta_extinction
    calc_flux_from_koffee
    calc_doublet_flux_from_koffee
    calc_hbeta_luminosity
    calc_hgamma_luminosity
    calc_sfr_integrate
    calc_sfr_koffee
    calc_save_as_fits

MODIFICATION HISTORY:
		v.1.0 - first created January 2020

"""
import numpy as np
import matplotlib.pyplot as plt

from astropy.cosmology import WMAP9 as cosmo
from astropy.constants import c
from astropy import units

from astropy.io import fits


#-------------------------------------------------------------------------------
# EXTINCTION CALCULATIONS
#-------------------------------------------------------------------------------
def calc_hbeta_extinction(lamdas, z):
    """
    Calculates the H_beta extinction - corrects for the extinction caused by light
    travelling through the dust and gas of the original galaxy, using the
    Cardelli et al. 1989 curves and Av = E(B-V)*Rv.
    The value for Av ~ 2.11 x C(Hbeta) where C(Hbeta) = 0.24 from
    Lopez-Sanchez et al. 2006 A&A 449.

    Parameters
    ----------
    lamdas : :obj:'~numpy.ndarray'
        the wavelength vector
    z : float
        redshift

    Returns
    -------
    A_hbeta : float
        the extinction correction factor at the Hbeta line
    """
    #convert lamdas from Angstroms into micrometers
    lamdas = lamdas/10000

    #define the equations from the paper
    y = lamdas**(-1) - 1.82
    a_x = 1.0 + 0.17699*y - 0.50447*(y**2) - 0.02427*(y**3) + 0.72085*(y**4) + 0.01979*(y**5) - 0.77530*(y**6) + 0.32999*(y**7)
    b_x = 1.41338*y + 2.28305*(y**2) + 1.07233*(y**3) - 5.38434*(y**4) - 0.62251*(y**5) + 5.30260*(y**6) - 2.09002*(y**7)

    #define the constants
    Rv = 3.1
    Av = 2.11*0.24

    #find A(lambda)
    A_lam = (a_x + b_x/Rv)*Av

    #find A_hbeta
    #first redshift the hbeta wavelength and convert to micrometers
    hbeta = (4861.333*(1+z))/10000
    #then find in lamdas array
    index = (np.abs(lamdas - hbeta)).argmin()
    #use the index to find A_hbeta
    A_hbeta = A_lam[index]

    return A_hbeta


#-------------------------------------------------------------------------------
# FLUX CALCULATIONS
#-------------------------------------------------------------------------------

def calc_flux_from_koffee(outflow_results, outflow_error, statistical_results, z, outflow=True):
    """
    Uses koffee outputs to calculate the flux of a single emission line with or
    without an outflow. In koffee, a gaussian is defined as:
        amp*e^[-(x-mean)**2/2sigma**2]
    so the integral (which gives us the flux) is:
        sqrt(2*pi)*amp*sigma.

    Parameters
    ----------
    outflow_results : :obj:'~numpy.ndarray'
        Array containing the outflow results found in koffee fits.  This will have
        either shape [6, i, j] or [7, i, j] depending on whether a constant was
        included in the koffee fit, or [3, i, j] or [4, i, j] if an outflow was
        not included. Either way, the flow and galaxy parameters are in the same
        shape.
        [[gal_sigma, gal_mean, gal_amp, flow_sigma, flow_mean, flow_amp], i, j]
        [[gal_sigma, gal_mean, gal_amp, flow_sigma, flow_mean, flow_amp, continuum_const], i, j]
        or
        [[gal_sigma, gal_mean, gal_amp], i, j]
        [[gal_sigma, gal_mean, gal_amp, continuum_const], i, j]
        for non-outflow fits.

    outflow_error : :obj:'~numpy.ndarray'
        Array containing the outflow errors found in koffee fits.  This will have
        either shape [6, i, j] or [7, i, j] depending on whether a constant was
        included in the koffee fit, or [3, i, j] or [4, i, j] if an outflow was
        not included. Either way, the flow and galaxy parameters are in the same
        shape.
        [[gal_sigma, gal_mean, gal_amp, flow_sigma, flow_mean, flow_amp], i, j]
        [[gal_sigma, gal_mean, gal_amp, flow_sigma, flow_mean, flow_amp, continuum_const], i, j]
        or
        [[gal_sigma, gal_mean, gal_amp], i, j]
        [[gal_sigma, gal_mean, gal_amp, continuum_const], i, j]
        for non-outflow fits.

    statistical_results : :obj:'~numpy.ndarray'
        Array containing the statistical results from koffee.  This has 0 if no
        flow was found, 1 if a flow was found, 2 if an outflow was found using a
        forced second fit due to the blue chi square test.  This should have the
        shape [i, j]

    z : float
        The redshift of the galaxy

    outflow : boolean
        Whether to also calculate the outflow flux.  Default is True, set to False
        for single gaussian fits.

    Returns
    -------
    systemic_flux : :obj:'~numpy.ndarray'
        Array with the systemic line fluxes, and np.nan where no outflow was found.

    systemic_flux_err : :obj:'~numpy.ndarray'
        Array with the systemic line flux errors, and np.nan where no outflow was
        found.

    outflow_flux : :obj:'~numpy.ndarray'
        Array with the outflow line fluxes, and np.nan where no outflow was found,
        if outflow==True.

    outflow_flux_err : :obj:'~numpy.ndarray'
        Array with the outflow line flux errors, and np.nan where no outflow was
        found, if outflow==True.
    """
    ##create array to keep velocity differences in, filled with np.nan
    systemic_flux = np.full_like(statistical_results, np.nan, dtype=np.double)
    systemic_flux_err = np.full_like(statistical_results, np.nan, dtype=np.double)

    if outflow == True:
        outflow_flux = np.full_like(statistical_results, np.nan, dtype=np.double)
        outflow_flux_err = np.full_like(statistical_results, np.nan, dtype=np.double)

    #create an outflow mask - outflows found at 1 and 2
    flow_mask = (statistical_results > 0)

    #de-redshift the sigma results
    systemic_sigma = outflow_results[0,:,:][flow_mask]/(1+z)

    if outflow == True:
        flow_sigma = outflow_results[3,:,:][flow_mask]/(1+z)

    #calculate the flux, which is srt(2*pi)*sigma*amplitude
    sys_flux = np.sqrt(2*np.pi) * systemic_sigma * outflow_results[2,:,:][flow_mask]

    #and calculate the error
    sys_flux_err = sys_flux * np.sqrt(2*np.pi) * np.sqrt((outflow_error[0,:,:][flow_mask]/systemic_sigma)**2 + (outflow_error[2,:,:][flow_mask]/outflow_results[2,:,:][flow_mask])**2)

    #save the results into the array
    systemic_flux[flow_mask] = sys_flux
    systemic_flux_err[flow_mask] = sys_flux_err

    #if also finding the flux of the outflow
    if outflow == True:
        flow_flux = np.sqrt(2*np.pi) * flow_sigma * outflow_results[5,:,:][flow_mask]

        flow_flux_err = flow_flux * np.sqrt(2*np.pi) * np.sqrt((outflow_error[3,:,:][flow_mask]/systemic_sigma)**2 + (outflow_error[5,:,:][flow_mask]/outflow_results[5,:,:][flow_mask])**2)

        #save to array
        outflow_flux[flow_mask] = flow_flux
        outflow_flux_err[flow_mask] = flow_flux_err

    #and return the results
    if outflow == True:
        return systemic_flux, systemic_flux_err, outflow_flux, outflow_flux_err
    else:
        return systemic_flux, systemic_flux_err


def calc_doublet_flux_from_koffee(outflow_results, outflow_error, statistical_results, z, outflow=True):
    """
    Uses koffee outputs to calculate the flux of a doublet emission line with or
    without an outflow. In koffee, a gaussian is defined as:
        amp*e^[-(x-mean)**2/2sigma**2]
    so the integral (which gives us the flux) is:
        sqrt(2*pi)*amp*sigma.
    Since this is a doublet, we therefore have:
        sqrt(2*pi)*amp1*sigma + sqrt(2*pi)*amp2*sigma.

    Parameters
    ----------
    outflow_results : :obj:'~numpy.ndarray'
        Array containing the outflow results found in koffee fits.  This will have
        either shape [13, i, j] or [7, i, j] if an outflow was not included in
        the koffee fit. Either way, the galaxy parameters are in the same position.
        [[gal_blue_sigma, gal_blue_mean, gal_blue_amp, gal_red_sigma, gal_red_mean,
        gal_red_amp, flow_blue_sigma, flow_blue_mean, flow_blue_amp, flow_red_sigma,
        flow_red_mean, flow_red_amp, continuum_const], i, j]
        or
        [[gal_blue_sigma, gal_blue_mean, gal_blue_amp, gal_red_sigma, gal_red_mean,
        gal_red_amp, continuum_const], i, j]
        for non-outflow fits.

    outflow_error : :obj:'~numpy.ndarray'
        Array containing the outflow errors found in koffee fits.  This will have
        either shape [13, i, j] or [7, i, j] if an outflow was not included in
        the koffee fit. Either way, the galaxy parameters are in the same position.
        [[gal_blue_sigma, gal_blue_mean, gal_blue_amp, gal_red_sigma, gal_red_mean,
        gal_red_amp, flow_blue_sigma, flow_blue_mean, flow_blue_amp, flow_red_sigma,
        flow_red_mean, flow_red_amp, continuum_const], i, j]
        or
        [[gal_blue_sigma, gal_blue_mean, gal_blue_amp, gal_red_sigma, gal_red_mean,
        gal_red_amp, continuum_const], i, j]
        for non-outflow fits.

    statistical_results : :obj:'~numpy.ndarray'
        Array containing the statistical results from koffee.  This has 0 if no
        flow was found, 1 if a flow was found, 2 if an outflow was found using a
        forced second fit due to the blue chi square test.  This should have the
        shape [i, j]

    z : float
        The redshift of the galaxy

    outflow : boolean
        Whether to also calculate the outflow flux.  Default is True, set to False
        for single gaussian fits.

    Returns
    -------
    systemic_flux : :obj:'~numpy.ndarray'
        Array with the systemic line fluxes, and np.nan where no outflow was found.

    systemic_flux_err : :obj:'~numpy.ndarray'
        Array with the systemic line flux errors, and np.nan where no outflow was
        found.

    outflow_flux : :obj:'~numpy.ndarray'
        Array with the outflow line fluxes, and np.nan where no outflow was found,
        if outflow==True.

    outflow_flux_err : :obj:'~numpy.ndarray'
        Array with the outflow line flux errors, and np.nan where no outflow was
        found, if outflow==True.
    """
    ##create array to keep velocity differences in, filled with np.nan
    systemic_flux = np.full_like(statistical_results, np.nan, dtype=np.double)
    systemic_flux_err = np.full_like(statistical_results, np.nan, dtype=np.double)

    if outflow == True:
        outflow_flux = np.full_like(statistical_results, np.nan, dtype=np.double)
        outflow_flux_err = np.full_like(statistical_results, np.nan, dtype=np.double)

    #create an outflow mask - outflows found at 1 and 2
    flow_mask = (statistical_results > 0)

    #de-redshift the sigma results... the doublet is set to have the same sigma
    #for both systemic components, so only need to do this once
    systemic_sigma = outflow_results[0,:,:][flow_mask]/(1+z)

    if outflow == True:
        flow_sigma = outflow_results[6,:,:][flow_mask]/(1+z)

    #calculate the flux using sqrt(2*pi)*sigma*(amplitude1+amplitude2)
    sys_flux = np.sqrt(2*np.pi) * systemic_sigma * (outflow_results[2,:,:][flow_mask] + outflow_results[5,:,:][flow_mask])

    #and calculate the error
    sys_flux_err = sys_flux * np.sqrt((outflow_error[0,:,:][flow_mask]/systemic_sigma)**2 + ((outflow_error[2,:,:][flow_mask]/outflow_results[2,:,:][flow_mask])**2 + (outflow_error[5,:,:][flow_mask]/outflow_results[5,:,:][flow_mask])**2))

    #save the results into the array
    systemic_flux[flow_mask] = sys_flux
    systemic_flux_err[flow_mask] = sys_flux_err

    #if also finding the flux of the outflow
    if outflow == True:
        flow_flux = np.sqrt(2*np.pi) * flow_sigma * (outflow_results[8,:,:][flow_mask] + outflow_results[11,:,:][flow_mask])

        flow_flux_err = sys_flux * np.sqrt((outflow_error[6,:,:][flow_mask]/systemic_sigma)**2 + ((outflow_error[8,:,:][flow_mask]/outflow_results[8,:,:][flow_mask])**2 + (outflow_error[11,:,:][flow_mask]/outflow_results[11,:,:][flow_mask])**2))

        #save to array
        outflow_flux[flow_mask] = flow_flux
        outflow_flux_err[flow_mask] = flow_flux_err

    #and return the results
    if outflow == True:
        return systemic_flux, systemic_flux_err, outflow_flux, outflow_flux_err
    else:
        return systemic_flux, systemic_flux_err


def calc_hbeta_luminosity(lamdas, spectrum, z, cont_subtract=False, plot=False):
    """
    Calculate the luminosity of the H_beta line by integrating along the line
    using np.trapz.  The spectrum is in 10^-16 erg/s/cm^2/Ang.  Need to change
    it to erg/s

    Luminosities should be around 10^40

    Parameters
    ----------
    lamdas : :obj:'~numpy.ndarray'
        the wavelength vector

    spectrum : :obj:'~numpy.ndarray'
        the spectrum or array of spectra.  If in array, needs to be in shape
        [npix, nspec]

    z : float
        redshift of the galaxy

    cont_subtract : boolean
        if True, assumes continuum has not already been subtracted.  Uses the
        median value of the wavelength range 4850-4855A.

    plot : boolean
        if True, plots the area of the spectrum used for the integral.  Default
        is False.

    Returns
    -------
    h_beta_flux : float or :obj:'~numpy.ndarray'
        the flux of the h_beta line

    s_n_mask : :obj:'~numpy.ndarray'
        Returned if cont_subtract==True.  The spectrum is used before it is
        continuum subtracted to find the signal-to-noise value near the Hbeta
        line for each spectrum

    h_beta_spec : :obj:'~numpy.ndarray'
        the region of the spectrum used to calculate the Hbeta flux.  Can be
        plotted to check the whole line is included in the integral.
    """
    #create bounds to integrate over
    #Hbeta is at 4862.68A, allowing 5.5A on either side
    left_limit = 4857.18*(1+z)
    right_limit = 4868.18*(1+z)

    #use the wavelengths to find the values in the spectrum to integrate over
    h_beta_spec = spectrum[(lamdas>=left_limit)&(lamdas<=right_limit),]
    h_beta_lam = lamdas[(lamdas>=left_limit) & (lamdas<=right_limit)]

    #create a mask to cut out all spectra with a flux less than 1.0 at its peak
    #flux_mask = np.amax(h_beta_spec, axis=0) < 1.0

    #if the continuum has not already been fit and subtracted, use an approximation to subtract it off
    #also use the continuum to find the S/N and mask things
    #s_n = []
    if cont_subtract == True:
        cont = spectrum[(lamdas>=4850.0*(1+z))&(lamdas<=4855.0*(1+z)),]
        cont_median = np.nanmedian(cont, axis=0)
        h_beta_spec = h_beta_spec - cont_median
        #find the standard deviation of the continuum section
        noise = np.std(cont, axis=0)
        s_n = (cont_median/noise)
        #create the S/N mask
        s_n_mask = s_n > 20

    if plot == True:
        plt.figure()
        plt.step(h_beta_lam, h_beta_spec.reshape([h_beta_lam.shape[0],-1]), where='mid')
        plt.show()

    #integrate along the spectrum
    #by integrating, the units are now 10^-16 erg/s/cm^2
    h_beta_integral = np.trapz(h_beta_spec, h_beta_lam, axis=0)
    h_beta_integral = h_beta_integral*10**(-16)*units.erg/(units.s*(units.cm*units.cm))

    #now get rid of the cm^2
    #get the Hubble constant at z=0; this is in km/Mpc/s
    H_0 = cosmo.H(0)
    #use d = cz/H0 to find the distance in cm
    dist = (c*z/H_0).decompose().to('cm')
    print('distance:', dist)
    #multiply by 4*pi*d^2 to get rid of the cm
    h_beta_flux = (h_beta_integral*(4*np.pi*(dist**2))).to('erg/s')

    print(h_beta_flux)

    if cont_subtract == True:
        return h_beta_integral, h_beta_flux.value, s_n_mask, h_beta_spec
    else:
        return h_beta_integral, h_beta_flux.value, h_beta_spec


def calc_hgamma_luminosity(lamdas, spectrum, z, cont_subtract=False, plot=False):
    """
    Calculate the luminosity of the H_gamma line
    The spectrum is in 10^-16 erg/s/cm^2/Ang.  Need to change it to erg/s

    Luminosities should be around 10^40

    Parameters
    ----------
    lamdas : :obj:'~numpy.ndarray'
        the wavelength vector

    spectrum : :obj:'~numpy.ndarray'
        the spectrum or array of spectra.  If in array, needs to be in shape
        [npix, nspec]

    z : float
        redshift of the galaxy

    cont_subtract : boolean
        if True, assumes continuum has not already been subtracted.  Uses the
        median value of the wavelength range 4850-4855A.

    plot : boolean
        if True, plots the area of the spectrum used for the integral.  Default
        is False.

    Returns
    -------
    h_beta_flux : float or :obj:'~numpy.ndarray'
        the flux of the h_beta line

    s_n_mask : :obj:'~numpy.ndarray'
        Returned if cont_subtract==True.  The spectrum is used before it is
        continuum subtracted to find the signal-to-noise value near the Hbeta
        line for each spectrum

    h_beta_spec : :obj:'~numpy.ndarray'
        the region of the spectrum used to calculate the Hbeta flux.  Can be
        plotted to check the whole line is included in the integral.
    """
    #create bounds to integrate over
    #Hgamma is at 4341.68A, allowing 1.5A on either side
    left_limit = 4338.18*(1+z)
    right_limit = 4345.18*(1+z)

    #use the wavelengths to find the values in the spectrum to integrate over
    h_gamma_spec = spectrum[(lamdas>=left_limit)&(lamdas<=right_limit),]
    h_gamma_lam = lamdas[(lamdas>=left_limit) & (lamdas<=right_limit)]

    #create a mask to cut out all spectra with a flux less than 1.0 at its peak
    #flux_mask = np.amax(h_beta_spec, axis=0) < 1.0

    #if the continuum has not already been fit and subtracted, use an approximation to subtract it off
    #also use the continuum to find the S/N and mask things
    #s_n = []
    if cont_subtract == True:
        cont = spectrum[(lamdas>=4315.0*(1+z))&(lamdas<=4320.0*(1+z)),]
        cont_median = np.nanmedian(cont, axis=0)
        h_gamma_spec = h_gamma_spec - cont_median
        #find the standard deviation of the continuum section
        noise = np.std(cont, axis=0)
        s_n = (cont_median/noise)
        #create the S/N mask
        s_n_mask = s_n > 20

    if plot == True:
        plt.figure()
        plt.step(h_gamma_lam, h_gamma_spec.reshape([h_gamma_lam.shape[0], -1]), where='mid')
        plt.show()

    #integrate along the spectrum
    #by integrating, the units are now 10^-16 erg/s/cm^2
    h_gamma_integral = np.trapz(h_gamma_spec, h_gamma_lam, axis=0)
    h_gamma_integral = h_gamma_integral*10**(-16)*units.erg/(units.s*(units.cm*units.cm))

    #now get rid of the cm^2
    #get the Hubble constant at z=0; this is in km/Mpc/s
    H_0 = cosmo.H(0)
    #use d = cz/H0 to find the distance in cm
    dist = (c*z/H_0).decompose().to('cm')
    print('distance:', dist)
    #multiply by 4*pi*d^2 to get rid of the cm
    h_gamma_flux = (h_gamma_integral*(4*np.pi*(dist**2))).to('erg/s')

    print(h_gamma_flux)

    if cont_subtract == True:
        return h_gamma_integral, h_gamma_flux.value, s_n_mask, h_gamma_spec
    else:
        return h_gamma_integral, h_gamma_flux.value, h_gamma_spec


#-------------------------------------------------------------------------------
# SFR CALCULATIONS
#-------------------------------------------------------------------------------

def calc_sfr_integrate(lamdas, spectrum, z, header, cont_subtract=False, include_extinction=True):
    """
    Calculates the star formation rate by integrating over Hbeta
    SFR = C_Halpha (L_Halpha / L_Hbeta)_0 x 10^{-0.4A_Hbeta} x L_Hbeta[erg/s]

    Parameters
    ----------
    lamdas : :obj:'~numpy.ndarray'
        the wavelength vector

    spectrum : :obj:'~numpy.ndarray'
        the spectrum or array of spectra.  If in array, needs to be in shape
        [npix, nspec]

    z : float
        redshift of the galaxy

    header : FITS header object
        the header from the fits file

    cont_subtract : boolean
        if True, assumes continuum has not already been subtracted.  Uses the
        median value of the wavelength range 4850-4855A.

    include_extinction : boolean
        if True, calculates the extinction at halpha and includes this in the
        SFR calculation.  Default is True.


    Returns
    -------
    sfr : float, or :obj:'~numpy.ndarray'
        the star formation rate found using Hbeta in M_sun/yr

    total_sfr : float
        the total SFR of all the spectra input in M_sun/yr

    sfr_surface_density : float, or :obj:'~numpy.ndarray'
        the star formation rate surface density found using Hbeta in M_sun/yr/kpc^2

    h_beta_spec : :obj:'~numpy.ndarray'
        the area of spectrum around Hbeta used to calculate the flux.  Can be
        plotted to check the whole emission line is included.

    s_n_mask : :obj:'~numpy.ndarray'
        Returned if cont_subtract==True.  The spectrum is used before it is
        continuum subtracted to find the signal-to-noise value near the Hbeta
        line for each spectrum
    """
    #first we need to define C_Halpha, using Hao et al. 2011 ApJ 741:124
    #From table 2, uses a Kroupa IMF, solar metallicity and 100Myr
    c_halpha = 10**(-41.257)

    #from Calzetti 2001 PASP 113 we have L_Halpha/L_Hbeta = 2.87
    lum_ratio_alpha_to_beta = 2.87

    if cont_subtract == True:
        hbeta_luminosity, s_n_mask, h_beta_spec = calc_hbeta_luminosity(lamdas, spectrum, z, cont_subtract=cont_subtract)
    if cont_subtract == False:
        hbeta_luminosity, h_beta_spec = calc_hbeta_luminosity(lamdas, spectrum, z, cont_subtract=cont_subtract)

    #calculate the star formation rate
    if include_extinction == True:
        hbeta_extinction = calc_hbeta_extinction(lamdas, z)
        sfr = c_halpha * lum_ratio_alpha_to_beta * 10**(-0.4*hbeta_extinction) * hbeta_luminosity
    elif include_extinction == False:
        sfr = c_halpha * lum_ratio_alpha_to_beta * 10**(-0.4*0.0) * (hbeta_luminosity)

    #sfr = c_halpha * lum_ratio_alpha_to_beta * 10**(-0.4*0.29) * hbeta_luminosity

    total_sfr = np.sum(sfr)

    #get the proper distance per arcsecond
    proper_dist = cosmo.kpc_proper_per_arcmin(z).to(units.kpc/units.arcsec)

    #sfr_surface_density = sfr/((0.7*1.35)*(proper_dist**2))
    sfr_surface_density = sfr/((header['CD1_2']*60*60*header['CD2_1']*60*60)*(proper_dist**2))

    if cont_subtract == True:
        return sfr, total_sfr, sfr_surface_density, s_n_mask, h_beta_spec
    elif cont_subtract == False:
        return sfr, total_sfr, sfr_surface_density, h_beta_spec


def calc_sfr_koffee(outflow_results, outflow_error, no_outflow_results, no_outflow_error, statistical_results, z, header, include_extinction=True, include_outflow=False):
    """
    Calculates the star formation rate using Hbeta
    SFR = C_Halpha (L_Halpha / L_Hbeta)_0 x 10^{-0.4A_Hbeta} x L_Hbeta[erg/s]
    The Hbeta flux is calculated using the results from the KOFFEE fits.

    Parameters
    ----------
    outflow_results : :obj:'~numpy.ndarray'
        Array containing the outflow results found in koffee fits.  This will have
        either shape [6, i, j] or [7, i, j] depending on whether a constant was
        included in the koffee fit. Either way, the flow and galaxy parameters
        are in the same shape.
        [[gal_sigma, gal_mean, gal_amp, flow_sigma, flow_mean, flow_amp], i, j]
        [[gal_sigma, gal_mean, gal_amp, flow_sigma, flow_mean, flow_amp, continuum_const], i, j]

    outflow_error : :obj:'~numpy.ndarray'
        Array containing the outflow errors found in koffee fits.  This will have
        either shape [6, i, j] or [7, i, j] depending on whether a constant was
        included in the koffee fit. Either way, the flow and galaxy parameters
        are in the same shape.
        [[gal_sigma, gal_mean, gal_amp, flow_sigma, flow_mean, flow_amp], i, j]
        [[gal_sigma, gal_mean, gal_amp, flow_sigma, flow_mean, flow_amp, continuum_const], i, j]

    no_outflow_results : :obj:'~numpy.ndarray'
        array of single gaussian results from KOFFEE for Hbeta line. This will
        have either shape [3, i, j] or [4, i, j] depending on whether a constant
        was included in the koffee fit. Either way, the flow and galaxy parameters
        are in the same shape.
        [[gal_sigma, gal_mean, gal_amp], i, j]
        [[gal_sigma, gal_mean, gal_amp, continuum_const], i, j]

    no_outflow_err : :obj:'~numpy.ndarray'
        array of the single gaussian result errors from KOFFEE for Hbeta line.
        This will have either shape [3, i, j] or [4, i, j] depending on whether
        a constant was included in the koffee fit. Either way, the flow and galaxy
        parameters are in the same shape.
        [[gal_sigma, gal_mean, gal_amp], i, j]
        [[gal_sigma, gal_mean, gal_amp, continuum_const], i, j]

    statistical_results : :obj:'~numpy.ndarray'
        Array containing the statistical results from koffee.  This has 0 if no
        flow was found, 1 if a flow was found, 2 if an outflow was found using a
        forced second fit due to the blue chi square test. This should have the
        shape [i, j]

    z : float
        redshift

    header : FITS header object
        the header from the fits file

    include_extinction : boolean
        if True, calculates the extinction at halpha and includes this in the
        SFR calculation.  Default is True.

    include_outflow : boolean
        if True, includes the broad outflow component in the flux calculation.
        If false, uses only the narrow component to calculate flux.  Default is
        False.

    Returns
    -------
    sfr : float, or :obj:'~numpy.ndarray'
        the star formation rate found using Hbeta in M_sun/yr

    sfr_err : float, or :obj:'~numpy.ndarray'
        the error of the star formation rate found using Hbeta in M_sun/yr

    total_sfr : float
        the total SFR of all the spectra input in M_sun/yr

    sfr_surface_density : float, or :obj:'~numpy.ndarray'
        the star formation rate surface density found using Hbeta in M_sun/yr/kpc^2

    sfr_surface_density_err : float, or :obj:'~numpy.ndarray'
        the error of the star formation rate surface density found using Hbeta
        in M_sun/yr/kpc^2
    """
    #first we need to define C_Halpha, using Hao et al. 2011 ApJ 741:124
    #From table 2, uses a Kroupa IMF, solar metallicity and 100Myr
    c_halpha = 10**(-41.257)

    #from Calzetti 2001 PASP 113 we have L_Halpha/L_Hbeta = 2.87
    lum_ratio_alpha_to_beta = 2.87

    #use the statistical results to make an array with systemic line where there are outflows
    #and one gaussian fits where there are no outflows
    #if not including outflow
    if include_outflow == False:
        results = np.full((3, outflow_results.shape[1], outflow_results.shape[2]), np.nan, dtype=np.double)
        error = np.full((3, outflow_results.shape[1], outflow_results.shape[2]), np.nan, dtype=np.double)
    elif include_outflow == True:
        results = np.full((6, outflow_results.shape[1], outflow_results.shape[2]), np.nan, dtype=np.double)
        error = np.full((6, outflow_results.shape[1], outflow_results.shape[2]), np.nan, dtype=np.double)

    #create the mask of where outflows are
    flow_mask = (statistical_results>0)

    #use the flow mask to define the hbeta gaussians
    if include_outflow == False:
        results[:,~flow_mask] = no_outflow_results[:3, ~flow_mask]
        results[:,flow_mask] = outflow_results[:3, flow_mask]

        error[:,~flow_mask] = no_outflow_error[:3, ~flow_mask]
        error[:,flow_mask] = outflow_error[:3, flow_mask]

        #create a new statistical_results for the array, since we also want the
        #spaxels where there is no outflow to be calculated
        stat_res_new = np.full_like(statistical_results, np.nan, dtype=np.double)
        stat_res_new[statistical_results>-1] = 1

    elif include_outflow == True:
        results[:3,~flow_mask] = no_outflow_results[:3, ~flow_mask]
        results[3:,~flow_mask] = np.zeros((3, outflow_results.shape[1], outflow_results.shape[2]))[:,~flow_mask]
        results[:,flow_mask] = outflow_results[:6, flow_mask]

        error[:3,~flow_mask] = no_outflow_error[:3, ~flow_mask]
        error[3:,~flow_mask] = np.zeros((3, outflow_results.shape[1], outflow_results.shape[2]))[:,~flow_mask]
        error[:,flow_mask] = outflow_error[:6, flow_mask]

    #make the flux calculation
    #this does sqrt(2*pi)*amp*sigma
    #so the units are now 10^-16 erg/s/cm^2
    if include_outflow == False:
        h_beta_integral, h_beta_integral_err = calc_flux_from_koffee(results, error, stat_res_new, z, outflow=False)

    elif include_outflow == True:
        systemic_flux, systemic_flux_err, outflow_flux, outflow_flux_err = calc_flux_from_koffee(results, error, statistical_results, z, outflow=True)
        h_beta_integral = np.nansum((systemic_flux, outflow_flux), axis=0)
        h_beta_integral_err = np.sqrt(np.nansum((systemic_flux_err**2, outflow_flux_err**2), axis=0))

    h_beta_integral = h_beta_integral*10**(-16)*units.erg/(units.s*(units.cm*units.cm))
    h_beta_integral_err = h_beta_integral_err*10**(-16)*units.erg/(units.s*(units.cm*units.cm))

    #now get rid of the cm^2
    #get the Hubble constant at z=0; this is in km/Mpc/s
    H_0 = cosmo.H(0)
    #use d = cz/H0 to find the distance in cm
    dist = (c*z/H_0).decompose().to('cm')
    print('distance:', dist)
    #multiply by 4*pi*d^2 to get rid of the cm
    hbeta_luminosity = (h_beta_integral*(4*np.pi*(dist**2))).to('erg/s')
    hbeta_luminosity_err = (h_beta_integral_err*(4*np.pi*(dist**2))).to('erg/s')

    #calculate the star formation rate
    if include_extinction == True:
        hbeta_extinction = calc_hbeta_extinction(lamdas, z)
        sfr = c_halpha * lum_ratio_alpha_to_beta * 10**(-0.4*hbeta_extinction) * hbeta_luminosity
        sfr_err = c_halpha * lum_ratio_alpha_to_beta * 10**(-0.4*hbeta_extinction) * hbeta_luminosity_err

    elif include_extinction == False:
        sfr = c_halpha * lum_ratio_alpha_to_beta * 10**(-0.4*0.0) * (hbeta_luminosity)
        sfr_err = c_halpha * lum_ratio_alpha_to_beta * 10**(-0.4*0.0) * (hbeta_luminosity_err)

    #sfr = c_halpha * lum_ratio_alpha_to_beta * 10**(0.4*0.29) * hbeta_luminosity

    total_sfr = np.nansum(sfr)

    #get the proper distance per arcsecond
    proper_dist = cosmo.kpc_proper_per_arcmin(z).to(units.kpc/units.arcsec)

    #sfr_surface_density = sfr/((0.7*1.35)*(proper_dist**2))
    #sfr_surface_density_err = sfr_err/((0.7*1.35)*(proper_dist**2))

    try:
        x = header['CD1_2']*60*60
        y = header['CD2_1']*60*60

    except:
        x = (-header['CDELT2']*np.sin(header['CROTA2'])) *60*60
        y = (header['CDELT1']*np.sin(header['CROTA2'])) *60*60

    sfr_surface_density = sfr/((x*y)*(proper_dist**2))
    sfr_surface_density_err = sfr_err/((x*y)*(proper_dist**2))

    print(sfr.unit)

    return sfr.value, sfr_err.value, total_sfr.value, sfr_surface_density.value, sfr_surface_density_err.value


def calc_save_as_fits(outflow_results, outflow_error, no_outflow_results, no_outflow_error, statistical_results, z, header, gal_name, output_folder, include_extinction=True, include_outflow=False, shift=None):
    """
    Calculates the outflow velocity and saves the results into a fits file.

    Parameters
    ----------
    outflow_results : :obj:'~numpy.ndarray'
        Array containing the outflow results found in koffee fits.  This will have
        either shape [6, i, j] or [7, i, j] depending on whether a constant was
        included in the koffee fit.  Either way, the flow and galaxy parameters
        are in the same shape.
        [[gal_sigma, gal_mean, gal_amp, flow_sigma, flow_mean, flow_amp], i, j]
        [[gal_sigma, gal_mean, gal_amp, flow_sigma, flow_mean, flow_amp, continuum_const], i, j]

    outflow_error : :obj:'~numpy.ndarray'
        Array containing the outflow errors found in koffee fits.  This will have
        either shape [6, i, j] or [7, i, j] depending on whether a constant was
        included in the koffee fit.  Either way, the flow and galaxy parameters
        are in the same shape.
        [[gal_sigma, gal_mean, gal_amp, flow_sigma, flow_mean, flow_amp], i, j]
        [[gal_sigma, gal_mean, gal_amp, flow_sigma, flow_mean, flow_amp, continuum_const], i, j]

    no_outflow_results : :obj:'~numpy.ndarray'
        array of single gaussian results from KOFFEE for Hbeta line. This will
        have either shape [3, i, j] or [4, i, j] depending on whether a constant
        was included in the koffee fit. Either way, the flow and galaxy parameters
        are in the same shape.
        [[gal_sigma, gal_mean, gal_amp], i, j]
        [[gal_sigma, gal_mean, gal_amp, continuum_const], i, j]

    no_outflow_err : :obj:'~numpy.ndarray'
        array of the single gaussian result errors from KOFFEE for Hbeta line.
        This will have either shape [3, i, j] or [4, i, j] depending on whether
        a constant was included in the koffee fit. Either way, the flow and galaxy
        parameters are in the same shape.
        [[gal_sigma, gal_mean, gal_amp], i, j]
        [[gal_sigma, gal_mean, gal_amp, continuum_const], i, j]

    statistical_results : :obj:'~numpy.ndarray'
        Array containing the statistical results from koffee.  This has 0 if no
        flow was found, 1 if a flow was found, 2 if an outflow was found using a
        forced second fit due to the blue chi square test.

    z : float
        The redshift of the galaxy

    header : FITS file object
        The header of the data fits file, to be used in the new fits file

    gal_name : str
        the name of the galaxy, and whatever descriptor to be used in the saving
        (e.g. IRAS08_binned_2_by_1)

    output_folder : str
        where to save the fits file

    include_extinction : boolean
        if True, calculates the extinction at halpha and includes this in the
        SFR calculation.  Default is True.

    include_outflow : boolean
        if True, includes the broad outflow component in the flux calculation.
        If false, uses only the narrow component to calculate flux.  Default is
        False.

    shift : list or None
        how to alter the header if the wcs is going to be wrong.
        e.g. ['CRPIX2', 32.0] will change the header value of CRPIX2 to 32.0

    Returns
    -------
    A saved fits file
    """
    #calculate the SFR
    sfr, sfr_err, total_sfr, sig_sfr, sig_sfr_err = calc_sfr_koffee(outflow_results, outflow_error, no_outflow_results, no_outflow_error, statistical_results, z, header, include_extinction=include_extinction, include_outflow=include_outflow)

    #copy the header and change the keywords so it's just 2 axes
    new_header = header.copy()

    new_header['NAXIS'] = 2

    del new_header['NAXIS3']
    del new_header['CNAME3']
    del new_header['CRPIX3']
    del new_header['CRVAL3']
    del new_header['CTYPE3']
    del new_header['CUNIT3']

    try:
        del new_header['CD3_3']
    except:
        del new_header['CDELT3']

    if shift:
        new_header[shift[0]] = shift[1]

    #create HDU object for star formation rate
    hdu = fits.PrimaryHDU(sfr, header=new_header)
    #hdu_error = fits.ImageHDU(sfr_err, name='Error')

    #create HDU list
    hdul = fits.HDUList([hdu])

    #write to file
    if shift:
        hdul.writeto(output_folder+gal_name+'_star_formation_rate_shifted.fits')
    else:
        hdul.writeto(output_folder+gal_name+'_star_formation_rate.fits')

    #create HDU object for star formation rate error
    hdu_error = fits.PrimaryHDU(sfr_err, header=new_header)

    #create HDU list
    hdul = fits.HDUList([hdu_error])

    #write to file
    if shift:
        hdul.writeto(output_folder+gal_name+'_star_formation_rate_error_shifted.fits')
    else:
        hdul.writeto(output_folder+gal_name+'_star_formation_rate_error.fits')

    #create HDU object for star formation rate surface density
    hdu = fits.PrimaryHDU(sig_sfr, header=new_header)
    hdu_error = fits.ImageHDU(sig_sfr_err, name='Error')

    #create HDU list
    hdul = fits.HDUList([hdu, hdu_error])

    #write to file
    if shift:
        hdul.writeto(output_folder+gal_name+'_star_formation_rate_surface_density_shifted.fits')
    else:
        hdul.writeto(output_folder+gal_name+'_star_formation_rate_surface_density.fits')



def calc_save_flux_as_fits(outflow_results, outflow_error, statistical_results, z, header, gal_name, emission_line_name, output_folder):
    """
    Calculates the flux and broad-to-narrow flux ratio and saves the results
    into a fits file.

    Parameters
    ----------
    outflow_results : :obj:'~numpy.ndarray'
        Array containing the outflow results found in koffee fits.  This will have
        either shape [6, i, j] or [7, i, j] depending on whether a constant was
        included in the koffee fit.  Either way, the flow and galaxy parameters
        are in the same shape.
        [[gal_sigma, gal_mean, gal_amp, flow_sigma, flow_mean, flow_amp], i, j]
        [[gal_sigma, gal_mean, gal_amp, flow_sigma, flow_mean, flow_amp, continuum_const], i, j]

    outflow_error : :obj:'~numpy.ndarray'
        Array containing the outflow errors found in koffee fits.  This will have
        either shape [6, i, j] or [7, i, j] depending on whether a constant was
        included in the koffee fit.  Either way, the flow and galaxy parameters
        are in the same shape.
        [[gal_sigma, gal_mean, gal_amp, flow_sigma, flow_mean, flow_amp], i, j]
        [[gal_sigma, gal_mean, gal_amp, flow_sigma, flow_mean, flow_amp, continuum_const], i, j]

    statistical_results : :obj:'~numpy.ndarray'
        Array containing the statistical results from koffee.  This has 0 if no
        flow was found, 1 if a flow was found, 2 if an outflow was found using a
        forced second fit due to the blue chi square test.

    z : float
        The redshift of the galaxy

    header : FITS file object
        The header of the data fits file, to be used in the new fits file

    gal_name : str
        the name of the galaxy, and whatever descriptor to be used in the saving
        (e.g. IRAS08_binned_2_by_1)

    emission_line_name : str
        the name of the emission line the results are from
        (e.g. OIII, hbeta, etc.)

    output_folder : str
        where to save the fits file

    Returns
    -------
    A saved fits file
    """
    #calculate the flux
    systemic_flux, systemic_flux_err, outflow_flux, outflow_flux_err = calc_flux_from_koffee(outflow_results, outflow_error, statistical_results, z, outflow=True)

    #calculate the flux ratios
    broad_to_narrow = outflow_flux/systemic_flux
    broad_to_narrow_err = broad_to_narrow * (outflow_flux_err/outflow_flux + systemic_flux_err/systemic_flux)

    #copy the header and change the keywords so it's just 2 axes
    new_header = header.copy()

    new_header['NAXIS'] = 2

    del new_header['NAXIS3']
    del new_header['CNAME3']
    del new_header['CRPIX3']
    del new_header['CRVAL3']
    del new_header['CTYPE3']
    del new_header['CUNIT3']

    try:
        del new_header['CD3_3']
    except:
        del new_header['CDELT3']

    #create HDU object for galaxy flux
    hdu = fits.PrimaryHDU(systemic_flux, header=new_header)
    hdu_error = fits.ImageHDU(systemic_flux_err, name='Error')

    #create HDU list
    hdul = fits.HDUList([hdu, hdu_error])

    #write to file
    hdul.writeto(output_folder+gal_name+'_'+emission_line_name+'_narrow_flux.fits')

    #create HDU object for outflow flux
    hdu = fits.PrimaryHDU(outflow_flux, header=new_header)
    hdu_error = fits.ImageHDU(outflow_flux_err, name='Error')

    #create HDU list
    hdul = fits.HDUList([hdu, hdu_error])

    #write to file
    hdul.writeto(output_folder+gal_name+'_'+emission_line_name+'_broad_flux.fits')

    #create HDU object for broad to narrow flux ratio
    hdu = fits.PrimaryHDU(broad_to_narrow, header=new_header)
    hdu_error = fits.ImageHDU(broad_to_narrow_err, name='Error')

    #create HDU list
    hdul = fits.HDUList([hdu, hdu_error])

    #write to file
    hdul.writeto(output_folder+gal_name+'_'+emission_line_name+'_broad_to_narrow_flux_ratio.fits')
