"""
NAME:
	calculate_equivalent_width.py

AUTHOR:
	Bronwyn Reichardt Chu
	Swinburne
	2020

EMAIL:
	<breichardtchu@swin.edu.au>

PURPOSE:
	To calculate the equivalent width of emission lines
	Written on MacOS Mojave 10.14.5, with Python 3.7

MODIFICATION HISTORY:
		v.1.0 - first created November 2020

"""
import numpy as np

from astropy.cosmology import WMAP9 as cosmo
from astropy.constants import c
from astropy import units as u



def region_around_line(lamda, flux, continuum):
    """
    Cut out the flux around a line and normalise.
    Based on the function from https://python4astronomers.github.io/astropy-UVES/UVES.html

    Parameters
    ----------
    lamda : np.ndarray
        1D array of wavelengths

    flux : np.ndarray of shape (len(lamda), i, j)
        array of flux values for the data (KCWI data cube)

    continuum : list of lists
        Wavelengths for the continuum [[begin1, end1], [begin2, end2]]
        that describe two areas on both sides of the emission line

    Returns
    -------
    wavelength : np.ndarray
        1D array of wavelengths covering the region the polynomial fit

    flux_results : np.ndarray of shape [len(wavelength), i, j]
        array of flux values normalised by the continuum fit
    """
    #create wavelength mask to isolate the areas of the continuum
    #this is where we will fit a polynomial
    cont_mask1 = (lamda > continuum[0][0]) & (lamda < continuum[0][1])
    cont_mask2 = (lamda > continuum[1][0]) & (lamda < continuum[1][1])
    # the "|" means or
    cont_mask_full = cont_mask1 | cont_mask2

    #another wavelength mask to cover the entire range
    lamrange_mask = (lamda > continuum[0][0]) & (lamda < continuum[1][1])

    #create an array for the results
    flux_results = np.full((lamrange_mask.sum(), flux.shape[1], flux.shape[2]), np.nan, dtype=np.double)

    #find the S/N of the data using the first continuum band
    sn_array = np.nanmedian(flux[cont_mask1,:,:], axis=0)/np.nanstd(flux[cont_mask1,:,:], axis=0)

    #iterate through the data cube
    for i in range(flux.shape[1]):
        for j in range(flux.shape[2]):
            #only do the fitting if the S/N is greater than 1
            if sn_array[i,j] > 1:
                #fit a second order polynomial to the continuum region
                line_coeff = np.polyfit(lamda[cont_mask_full], flux[cont_mask_full, i, j], 2)

                #divide the flux by the polynomial and save the result
                flux_results[:,i,j] = flux[lamrange_mask, i, j]/np.polyval(line_coeff, lamda[lamrange_mask])

    return lamda[lamrange_mask], flux_results


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

continuum_bands = {
    "Hbeta1" : [4838.3, 4848.1],
    "Hbeta2" : [4877.6, 4887.4]
}


def calc_ew(lamda, flux, redshift, continuum1=continuum_bands["Hbeta1"], continuum2=continuum_bands["Hbeta2"]):
    """
    Calculates the equivalent width
    Based on the function from https://python4astronomers.github.io/astropy-UVES/UVES.html

    Parameters
    ----------
    lamda : np.ndarray
        1D array of wavelengths

    flux : np.ndarray of shape (len(lamda), i, j)
        array of flux values for the data (KCWI data cube)

    continuum1 : list
        Wavelengths for the first continuum section [begin1, end1]
        that describe the area on the first side of the emission line, default is
        those for Hbeta

    continuum2 : list
        Wavelengths for the second continuum section [begin2, end2]
        that describe the area on the second side of the emission line, default is
        those for Hbeta

    redshift : float
        the redshift of the galaxy

    Returns
    -------
    ew : nd.array
        array of equivalent widths where the continuum is greater than 1
    """
    #multiply the continuum bands by the redshift factor
    continuum1 = [x*(1+redshift) for x in continuum1]
    continuum2 = [x*(1+redshift) for x in continuum2]

    #give the continuum bands units
    #continuum1 = [x*u.AA for x in continuum1]
    #continuum2 = [x*u.AA for x in continuum2]

    #create continuum list of lists
    full_continuum_list = [continuum1, continuum2]

    #give the wavelength vector units
    #lamda = lamda*u.AA

    #put all this into the continuum normalising function
    cont_lams, flux_results = region_around_line(lamda, flux, continuum=full_continuum_list)

    #find the difference between each wavlength
    #delta_lam = np.diff(cont_lams.to(u.AA).value)
    delta_lam = np.diff(cont_lams)

    #calculate the equivalent width
    #mulitply by the delta wavelength and then sum the result
    ew = np.nansum((flux_results - 1.)[:-1,:,:] * delta_lam[:, np.newaxis, np.newaxis], axis=0)

    return ew
