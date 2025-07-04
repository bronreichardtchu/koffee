"""
NAME:
	prepare_cubes.py

AUTHOR:
	Bronwyn Reichardt Chu
	Swinburne
	2019

EMAIL:
	<breichardtchu@swin.edu.au>

PURPOSE:
	To prepare 3D data cubes for ppxf, KOFFEE, and any other routines.
	Written on MacOS Mojave 10.14.5, with Python 3.7

FUNCTIONS INCLUDED:
    read_in_data_fits
    read_in_data_pickle
    bin_data
    save_binned_data
    air_to_vac
    barycentric_corrections
    milky_way_extinction_correction
    hbeta_extinction_correction
    calculate_EBV_from_hbeta_hgamma_ratio
    load_data
    data_cubes_combine_by_pixel
    data_cubes_combine_by_wavelength
    data_coords
    flatten_cube
    sn_cut
    combine_red_blue
    prepare_combine_cubes
    prepare_single_cube

MODIFICATION HISTORY:
		v.1.0 - first created August 2019

"""
import glob
import math
import numpy as np
from datetime import date
import pickle

from astropy.io import fits
from astropy.coordinates import SkyCoord, EarthLocation
from astropy import units
from astropy.time import Time
from astropy import constants as consts

import matplotlib.pyplot as plt
from matplotlib import gridspec

#SpectRes is a spectrum resampling module which can be used to resample the fluxes
# and their uncertainties while preserving the integrated flux more information at
# https://spectres.readthedocs.io/en/latest/ or from the paper at
# https://arxiv.org/pdf/1705.05165.pdf
#from spectres import spectres

#from .display_pixels import cap_display_pixels as cdp
from koffee import brons_display_pixels_kcwi as bdpk
from koffee.calculations import calculate_extinction_checks as calc_ext

import importlib
importlib.reload(calc_ext)

#====================================================================================================
#LOAD DATA
#====================================================================================================
def read_in_data_fits(filename):
    """
    Reads in the data if it is contained in a fits file

    Parameters
    ----------
    filename : str
        points to the file

    Returns
    -------
    lamdas : :obj:'~numpy.ndarray'
        the wavelength vector constructed from the fits header

    data : :obj:'~numpy.ndarray'
        the data cube

    var : :obj:'~numpy.ndarray'
        the variance cube (if contained within the fits file)

    header : FITS header
        the fits header
    """
    #open file and get data and header
    with fits.open(filename) as hdu:
        data = hdu[0].data
        header = hdu[0].header
        #if there is more than one extension in the fits file, assume the second one is the variance
        if len(hdu) > 1:
            var = hdu[1].data
    hdu.close()

    #create the wavelength vector
    try:
        lamdas = np.arange(header['CRVAL3'], header['CRVAL3']+(header['NAXIS3']*header['CD3_3']), header['CD3_3'])
    except KeyError:
        lamdas = np.arange(header['CRVAL3'], header['CRVAL3']+(header['NAXIS3']*header['CDELT3']), header['CDELT3'])

    if 'var' in locals():
        return lamdas, data, var, header
    else:
        return lamdas, data, header


def read_in_data_pickle(filename):
    """
    Reads in the data if it is contained in a pickled file

    Parameters
    ----------
    filename : str
        points to the file location

    Returns
    -------
    lamdas : :obj:'~numpy.ndarray'
        the wavelength vector constructed from the fits header

    data : :obj:'~numpy.ndarray'
        the data cube

    var : :obj:'~numpy.ndarray'
        the variance cube (if contained within the fits file)
    """
    #open file and get data
    with open(filename, 'rb') as f:
        pickled_things = pickle.load(f)
    f.close()

    #use the length of pickled_things to split it into the various parts
    if len(pickled_things) > 2:
        lamdas, data, var = pickled_things
    else:
        lamdas, data = pickled_things

    if 'var' in locals():
        return lamdas, data, var
    else:
        return lamdas, data


#====================================================================================================
#BINNING
#====================================================================================================

def bin_data(lamdas, data, z, bin_size=[3,3], var=None):
    """
    Bins the input data

    Parameters
    ----------
    lamdas : :obj:'~numpy.ndarray'
        wavelength vector

    data : :obj:'~numpy.ndarray'
        the data to be binned

    z : float
        the redshift

    bin_size : int
        the number of spaxels to bin by (Default is [3,3], this will bin 3x3)

    var : :obj:'~numpy.ndarray' or None
        the variance cube which needs to be binned in the exact same way as the
        data, with the same shifts.(Default is None)

    Returns
    -------
    binned_data : :obj:'~numpy.ndarray'
        the binned data

    binned_var : :obj:'~numpy.ndarray'
        the binned variance (if var not None)
    """
    #create empty array to put binned data in
    binned_data = np.empty([data.shape[0], math.ceil(data.shape[1]/bin_size[0]), math.ceil(data.shape[2]/bin_size[1])])

    #create list to add hbeta peak differences to
    hbeta_peak_diff_list = []

    if var is not None:
        binned_var = np.empty_like(binned_data)

    #create lamda mask for hbeta
    hbeta_mask = (lamdas>4862.68*(1+z)-5.0) & (lamdas<4862.68*(1+z)+5.0)

    #calculate the signal-to-noise ratio of hbeta
    if var is not None:
        sn_array = np.nanmedian(data[hbeta_mask,:,:], axis=0)/np.nanmedian(var[hbeta_mask,:,:], axis=0)
    else:
        sn_array = np.nanmedian(data[hbeta_mask,:,:], axis=0)/np.nanstd(data[hbeta_mask,:,:], axis=0)

    #create counter for x direction
    start_xi = 0
    end_xi = bin_size[0]

    #iterate through x direction
    for x in np.arange(data.shape[1]/bin_size[0]):
        #create counter for y direction
        start_yi = 0
        end_yi = bin_size[1]
        #iterate through y direction
        for y in np.arange(data.shape[2]/bin_size[1]):
            #find the central spaxel of the bin
            center_x = int((start_xi+end_xi)/2)
            center_y = int((start_yi+end_yi)/2)

            #only need to shift the hbeta peaks if the S/N is greater than 3
            try:
                if sn_array[center_x, center_y] >= 3:
                    #find where the peak of the hbeta line is
                    try:
                        hbeta_peak = np.argmax(data[:, center_x, center_y][hbeta_mask])
                    except:
                        hbeta_peak = np.argmax(data[:, start_xi, start_yi][hbeta_mask])
                    #print(data[:, center_x, center_y][hbeta_mask].shape)
                    #print('Hbeta peak for central spaxel:', hbeta_peak)

                    #find where the peak of the rest of the spaxels is
                    other_hbeta_peaks = np.argmax(data[:, start_xi:end_xi, start_yi: end_yi][hbeta_mask], axis=0)
                    #print('Other Hbeta peaks', other_hbeta_peaks)

                    #find difference between other_hbeta_peaks and hbeta_peak
                    hbeta_peak_diff = other_hbeta_peaks - hbeta_peak
                    #print('Hbeta peak diff', hbeta_peak_diff, '\n\n')

                    #add to list
                    hbeta_peak_diff_list.append(hbeta_peak_diff)

                    #iterate through the peak differences
                    for (i,j), diff in np.ndenumerate(hbeta_peak_diff):
                        if diff < 0:
                            #shift the spectra to the right
                            data[:, start_xi:end_xi, start_yi: end_yi][abs(diff):, i, j] = data[:, start_xi:end_xi, start_yi: end_yi][:diff, i, j]
                            if var is not None:
                                var[:, start_xi:end_xi, start_yi: end_yi][abs(diff):, i, j] = var[:, start_xi:end_xi, start_yi: end_yi][:diff, i, j]

                        if diff > 0:
                            #shift the spectra to the left
                            data[:, start_xi:end_xi, start_yi: end_yi][:-diff, i, j] = data[:, start_xi:end_xi, start_yi: end_yi][diff:, i, j]
                            if var is not None:
                                var[:, start_xi:end_xi, start_yi: end_yi][:-diff, i, j] = var[:, start_xi:end_xi, start_yi: end_yi][diff:, i, j]

            #if that doesn't work, then it's on the edge of the cube, don't need to shift anything
            except IndexError:
                continue


            """
            #checks
            #find where the peak of the hbeta line is
            try:
                hbeta_peak = np.argmax(data[:, center_x, center_y][hbeta_mask])
            except:
                hbeta_peak = np.argmax(data[:, start_xi, start_yi][hbeta_mask])
            #print(data[:, center_x, center_y][hbeta_mask].shape)
            print('Checking Hbeta peak for central spaxel:', hbeta_peak)

            #find where the peak of the rest of the spaxels is
            other_hbeta_peaks = np.argmax(data[:, start_xi:end_xi, start_yi: end_yi][hbeta_mask], axis=0)
            print('Checking Other Hbeta peaks', other_hbeta_peaks)

            #find difference between other_hbeta_peaks and hbeta_peak
            hbeta_peak_diff = other_hbeta_peaks - hbeta_peak
            print('Checking Hbeta peak diff', hbeta_peak_diff, '\n\n')
            #"""

            #bin the data
            binned_data[:, int(x), int(y)] = np.nansum(data[:, start_xi:end_xi, start_yi:end_yi], axis=(1,2))

            #bin the variance
            if var is not None:
                binned_var[:, int(x), int(y)] = np.nansum(var[:, start_xi:end_xi, start_yi:end_yi], axis=(1,2))

            #increase y counters
            start_yi += bin_size[1]
            end_yi += bin_size[1]

        #increase x counters
        start_xi += bin_size[0]
        end_xi += bin_size[0]

    if var is not None:
        return binned_data, binned_var
    else:
        return binned_data, hbeta_peak_diff_list


def save_binned_data(data_filepath, data_folder, gal_name, z, bin_size=[3,3], var_filepath=None):
    """
    Read in the data from a fits file, bin it by the given binsize, and then
    save the binned data and/or variance cube to a fits file, with the necessary
    changes to the fits file header.
    Parameters
    ----------
    data_filepath : str
        the filepath to the data to be binned

    header : FITS header object
        the old fits header to be changed for the new file

    data_folder : str
        the folder in which to save the new file

    gal_name : str
        the galaxy name and any other descriptors for the filename.
        E.g. 'cgcg453_red_var' for the red variance cube of cgcg453

    z : float
        the redshift

    bin_size : int
        the number of spaxels the cube was binned by, [x,y].  (Default is [3,3])

    var_filepath : str or None
        the filepath to the variance cube to be binned (Default is None)

    Returns
    -------
    Rebins and saves the cube and new header to a fits file.
    """
    #read in the data
    data_stuff = read_in_data_fits(data_filepath)

    if len(data_stuff)>3:
        lamdas, data, var, header = data_stuff
    else:
        lamdas, data, header = data_stuff

    #read in the var
    if var_filepath:
        var_lamdas, var, var_header = read_in_data_fits(var_filepath)


    if 'var' in locals():
        #give it to the binning function
        binned_data, binned_var = bin_data(lamdas, data, z, bin_size=bin_size, var=var)
        print('Binned data shape:', binned_data.shape)
        print('Binned var shape:', binned_var.shape)

    else:
        #give it to the binning function
        binned_data, hbeta_peak_diff_list = bin_data(lamdas, data, z, bin_size=bin_size, var=None)
        print('Binned data shape:', binned_data.shape)


    #copy the header
    new_header = header.copy()

    #change the cards to match the binned data
    #change the size of the spaxels
    #remember that numpy arrays and fits have different directions
    #so a numpy array with shape (lam, x, y) is same as (NAXIS3, NAXIS2, NAXIS1)
    try:
        new_header['CDELT1'] = header['CDELT1']*bin_size[1]
        new_header['CDELT2'] = header['CDELT2']*bin_size[0]
    except:
        new_header['CD1_1'] = header['CD1_1']*bin_size[1]
        new_header['CD1_2'] = header['CD1_2']*bin_size[0]
        new_header['CD2_2'] = header['CD2_2']*bin_size[0]
        new_header['CD2_1'] = header['CD2_1']*bin_size[1]

    #change the number of spaxels
    new_header['NAXIS1'] = math.ceil(header['NAXIS1']/bin_size[1])
    new_header['NAXIS2'] = math.ceil(header['NAXIS2']/bin_size[0])

    #change the reference pixel to the nearest 0.5
    new_header['CRPIX1'] = round((header['CRPIX1']/bin_size[1])*2.0)/2.0
    new_header['CRPIX2'] = round((header['CRPIX2']/bin_size[0])*2.0)/2.0

    #create HDU object
    hdu = fits.PrimaryHDU(binned_data, header=new_header)
    if 'var' in locals():
        hdu_var = fits.ImageHDU(binned_var, header=new_header)
        #create HDU list
        hdul = fits.HDUList([hdu, hdu_var])
    else:
        #create HDU list
        hdul = fits.HDUList([hdu])

    #write to file
    hdul.writeto(data_folder+gal_name+'_binned_'+str(bin_size[0])+'_by_'+str(bin_size[1])+'.fits')

#====================================================================================================
#CORRECTIONS
#====================================================================================================

def air_to_vac(wavelength):
    """
    Implements the air to vacuum wavelength conversion described in eqn 64 and
    65 of Greisen 2006. The error in the index of refraction amounts to 1:10^9,
    which is less than the empirical formula.
    Function slightly altered from specutils.utils.wcs_utils.

    Parameters
    ----------
    wavelength : :obj:'~numpy.ndarray'
        the air wavelength(s) in Angstroms

    Returns
    -------
    wavelength : :obj:'~numpy.ndarray'
        the vacuum wavelength(s) in Angstroms
    """
    #convert wavelength to um from Angstroms
    wlum = wavelength/10000
    #apply the equation from the paper
    return (1+1e-6*(287.6155+1.62887/wlum**2+0.01360/wlum**4)) * wavelength


def barycentric_corrections(lamdas, header):
    """
    Corrects for the earth's rotation... must have the fits header!!! If using
    pickled data, this should already have been applied.

    Parameters
    ----------
    lamdas : :obj:'~numpy.ndarray'
        the wavelength vector

    header : FITS header object
        the header from the fits file

    Returns
    -------
    lamdas : :obj:'~numpy.ndarray'
        the corrected wavelength vector
    """
    keck = EarthLocation.from_geodetic(lat=19.8283*units.deg, lon=-155.4783*units.deg, height=4160*units.m)

    sky_coord = SkyCoord(ra=header['CRVAL1']*units.deg, dec=header['CRVAL2']*units.deg)

    try:
        date = header['DATE-BEG']
    except:
        print("No keyword 'DATE-BEG' in header, using alternate date")
        date = '2018-02-15T08:38:48.054'
        print(date)

    barycentric_correction = sky_coord.radial_velocity_correction(obstime=Time(date), location=keck)

    barycentric_correction = barycentric_correction.to(units.km/units.s)

    c_val = consts.c.to('km/s').value

    lamdas = lamdas*(1.0 + (barycentric_correction.value/c_val))

    return lamdas


def milky_way_extinction_correction(lamdas, data, Av=0.2511):
    """
    Corrects for the extinction caused by light travelling through the dust and
    gas of the Milky Way, as described in Cardelli et al. 1989.

    Parameters
    ----------
    lamdas : :obj:'~numpy.ndarray'
        wavelength vector

    data : :obj:'~numpy.ndarray'
        3D cube of data

    Av : float 
        The extinction from the Milky Way, found using NED which uses the Schlafly
        & Finkbeiner (2011) recalibration of the Schlegel, Finkbeiner & Davis (1998)
        extinction map based on dust emission measured by COBE/DIRBE and IRAS/ISSA.
        Default is 0.2511, which is the value for IRAS 08339+6517.

    Returns
    -------
    data : :obj:'~numpy.ndarray'
        the data corrected for extinction
    """
    #convert lamdas from Angstroms into micrometers
    lamdas = lamdas/10000

    #define the equations from the paper
    y = lamdas**(-1) - 1.82
    a_x = 1.0 + 0.17699*y - 0.50447*(y**2) - 0.02427*(y**3) + 0.72085*(y**4) + 0.01979*(y**5) - 0.77530*(y**6) + 0.32999*(y**7)
    b_x = 1.41338*y + 2.28305*(y**2) + 1.07233*(y**3) - 5.38434*(y**4) - 0.62251*(y**5) + 5.30260*(y**6) - 2.09002*(y**7)

    #define the constants
    Rv = 3.1

    #find A(lambda)
    A_lam = (a_x + b_x/Rv)*Av
    #return A_lam

    #apply to the data
    data = (10**(0.4*A_lam[:, None, None]))*data

    return data


def hbeta_extinction_correction(lamdas, data, var, z, sn_cut=0):
    """
    Corrects for the extinction caused by light travelling through the dust and
    gas of the galaxy, as described in Cardelli et al. 1989. Uses Hbeta/Hgamma
    ratio as in Calzetti et al. 2001

    Parameters
    ----------
    lamdas : :obj:'~numpy.ndarray'
        wavelength vector

    data : :obj:'~numpy.ndarray'
        3D cube of data

    var : :obj:'~numpy.ndarray'
        3D cube of variance

    z : float
        redshift

    sn_cut : float
        the signal-to-noise ratio of the Hgamma line above which the extinction
        correction is calculated.  E.g. if sn_cut=3, then the extinction
        is only calculated for spaxels with Hgamma emission with S/N>=3.  For
        all other spaxels, Av = 0.  Default is 0

    Returns
    -------
    data : :obj:'~numpy.ndarray'
        the data corrected for extinction
    """
    #create the S/N array
    hgamma_mask = (lamdas>(4341.68*(1+z)-5)) & (lamdas<(4341.68*(1+z)+5))
    hgamma_cont_mask = (lamdas>(4600*(1+z))) & (lamdas<(4800*(1+z)))

    #hgamma_sn = np.trapz(data[hgamma_mask,:,:], lamdas[hgamma_mask], axis=0)/np.sqrt(np.sum(var[hgamma_mask,:,:]**2, axis=0))
    hgamma_sn = np.trapz(data[hgamma_mask,:,:], lamdas[hgamma_mask], axis=0)/np.nanstd(data[hgamma_cont_mask,:,:], axis=0)

    #use the hbeta/hgamma ratio to calculate EBV
    ebv = calculate_EBV_from_hbeta_hgamma_ratio(lamdas, data, z)

    #define the constant (using MW expected curve)
    Rv = 3.1

    #use that to calculate Av
    Av = ebv * Rv

    #replace the Av with 0 where the S/N is less than the sn_cut
    Av[hgamma_sn < sn_cut] = 0

    #convert lamdas from Angstroms into micrometers
    lamdas = lamdas/10000

    #define the equations from the paper
    y = lamdas**(-1) - 1.82
    a_x = 1.0 + 0.17699*y - 0.50447*(y**2) - 0.02427*(y**3) + 0.72085*(y**4) + 0.01979*(y**5) - 0.77530*(y**6) + 0.32999*(y**7)
    b_x = 1.41338*y + 2.28305*(y**2) + 1.07233*(y**3) - 5.38434*(y**4) - 0.62251*(y**5) + 5.30260*(y**6) - 2.09002*(y**7)

    #tile a_x and b_x so that they're the right array shape
    a_x = np.tile(a_x, [data.shape[2], data.shape[1], 1]).T
    b_x = np.tile(b_x, [data.shape[2], data.shape[1], 1]).T

    #print('median a_x:', np.nanmedian(a_x))
    #print('a_x shape:', a_x.shape)

    #print('median b_x:', np.nanmedian(b_x))
    #print('b_x shape:', b_x.shape)

    #find A(lambda)
    A_lam = (a_x + b_x/Rv)*Av

    #apply to the data
    data = (10**(0.4*A_lam))*data

    return Av, A_lam, data



def calculate_EBV_from_hbeta_hgamma_ratio(lamdas, data, z):
    """
    Uses Hbeta/Hgamma ratio as in Calzetti et al. 2001

    Parameters
    ----------
    lamdas : :obj:'~numpy.ndarray'
        wavelength vector

    data : :obj:'~numpy.ndarray'
        3D cube of data

    z : float
        redshift

    Returns
    -------
    data : :obj:'~numpy.ndarray'
        the data corrected for extinction
    """
    #calculate the hbeta/hgamma ratio
    hbeta_flux, hgamma_flux, hbeta_hgamma_obs = calc_ext.calc_hbeta_hgamma_amps(lamdas, data, z, cont_subtract=False)
    #hbeta_flux, hgamma_flux, hbeta_hgamma_obs = calc_ext.calc_hbeta_hgamma_integrals(lamdas, data, z, cont_subtract=False, plot=False)

    #set the expected hbeta/hgamma ratio
    hbeta_hgamma_actual = 2.15

    #set the expected differential extinction [k(hgamma)-k(hbeta)]=0.465
    diff_ext = 0.465

    #create an array for the ebv values
    ebv = np.full_like(hbeta_hgamma_obs, np.nan, dtype=np.double)

    #calculate E(B-V) if hbeta_hgamma_obs >= 2.15
    for (i,j), hbeta_hgamma_obs_value in np.ndenumerate(hbeta_hgamma_obs):
        if hbeta_hgamma_obs_value >= 2.15:
            #calculate ebv
            ebv[i,j] = (2.5*np.log10(hbeta_hgamma_obs_value/hbeta_hgamma_actual)) / diff_ext

        else:
            #set ebv to a small value
            ebv[i,j] = 0.01

    return ebv


#====================================================================================================
#SIMPLE DATA READ IN
#====================================================================================================
def load_data(filename, z=0.0, mw_correction=True, Av_mw=0.2511):
    """
    Get the data from the fits file, deredshift (if z is given) and correct to 
    vacuum wavelengths and for the earth's rotation

    Parameters
    ----------
    filename : str
        points to the file

    z : float 
        the redshift of the galaxy, to de-redshift the wavelengths.  Default is 
        0.0, which will not do anything.

    mw_correction : boolean
        whether to apply the milky way extinction correction. Default is True.

    Av_mw : float
        The extinction value from the Milky Way in the direction of the input 
        galaxy.  The default value is 0.2511, which is for IRAS 08339+6517.

    Returns
    -------
    lamdas : :obj:'~numpy.ndarray'
        the wavelength vector constructed from the fits header

    data : :obj:'~numpy.ndarray'
        the data cube

    var : :obj:'~numpy.ndarray'
        the variance cube (if contained within the fits file)

    header : FITS header object
        the fits header
    """
    #open the file and get the data
    fits_stuff = read_in_data_fits(filename)

    if len(fits_stuff) > 3:
        lamdas, data, var, header = fits_stuff
    else:
        lamdas, data, header = fits_stuff

    #correct this from air to vacuum wavelengths
    #Greisen 2006 FITS Paper III (eqn 65)
    lamdas = air_to_vac(lamdas)

    #apply barycentric radial velocity corrections
    lamdas = barycentric_corrections(lamdas, header)

    #apply Milky Way extinction correction
    if mw_correction == True:
        data = milky_way_extinction_correction(lamdas, data, Av=Av_mw)
        if len(fits_stuff) > 3:
            var = milky_way_extinction_correction(lamdas, var, Av=Av_mw)

    # de-redshift the data 
    lamdas = lamdas/(1 + z)

    if len(fits_stuff) > 3:
        return lamdas, data, var, header
    else:
        return lamdas, data, header

#====================================================================================================
#COMBINE CUBES
#====================================================================================================

def data_cubes_combine_by_pixel(filepath, gal_name, z=0.0, Av_mw=0.2511):
    """
    Grabs datacubes and combines them by pixel using addition, finding the mean
    and the median.

    Parameters
    ----------
    filepath : list of str
        the data cubes filepath strings to pass to glob.glob

    gal_name : str
        galaxy name/descriptor

    Av_mw : float
        The extinction value from the Milky Way in the direction of the input 
        galaxy.  The default value is 0.2511, which is for IRAS 08339+6517.

    Returns
    -------
    lamdas : :obj:'~numpy.ndarray'
        the wavelength vector for the cubes

    cube_added : :obj:'~numpy.ndarray'
        all cubes added

    cube_mean : :obj:'~numpy.ndarray'
        the mean of all the cubes

    cube_median : :obj:'~numpy.ndarray'
        the median of all the cubes

    header : FITS header object
        the header from the fits file
    """
    #create list to append datas to
    all_data = []
    all_var = []
    all_lamdas = []

    #iterate through the filenames
    #they should all be from fits files, so we can just use that loading function
    for file in glob.glob(filepath):
        fits_stuff = read_in_data_fits(file)
        if len(fits_stuff) > 3:
            lamdas, data, var, header = fits_stuff
            all_var.append(var)
        else:
            lamdas, data, header = fits_stuff
        #apply corrections to lambdas
        lamdas = air_to_vac(lamdas)
        lamdas = barycentric_corrections(lamdas, header)
        #apply Milky Way extinction correction
        data = milky_way_extinction_correction(lamdas, data, Av=Av_mw)
        #de-redshift 
        lamdas = lamdas/(1+z)
        all_lamdas.append(lamdas)
        #append the data
        all_data.append(data)

    #check if var has the same number of cubes as the data, and if it doesn't, delete it
    if len(all_data) > len(all_var):
        del all_var

    #because the exposures are so close together, the difference in lamda between
    #the first to the last is only around 0.001A.  There's a difference in the
    #total length of about 0.0003A between the longest and shortest wavelength
    #vectors after the corrections.  So I'm taking the median across the whole
    #collection.  This does introduce some error, making the line spread function
    #of the averaged spectra larger.
    lamdas = np.median(all_lamdas, axis=0)

    #adding the data
    cube_added = np.zeros_like(all_data[0])

    for cube in all_data:
        cube_added += cube

    #finding the mean
    cube_mean = np.mean(all_data, axis=0)

    #finding the median
    cube_median = np.median(all_data, axis=0)

    #if all_var in locals():
        #adding the variances


    #pickle the results
    with open(filepath.split('*')[0]+'_'+gal_name+'_combined_by_pixel_'+str(date.today()),'wb') as f:
        pickle.dump([lamdas, cube_added, cube_mean, cube_median], f)
    f.close()

    return lamdas, cube_added, cube_mean, cube_median, header


def data_cubes_combine_by_wavelength(filepath, gal_name, z=0.0, Av_mw=0.2511):
    """
    Grabs datacubes and combines them by interpolating each spectrum in wavelength
    space and making sure to start and end at exactly the same wavelength for
    each spectrum before using addition, finding the mean and the median.

    Parameters
    ----------
    filepath : list of str
        the filepath string to pass to glob.glob

    gal_name : str
        galaxy name/descriptor (string)
    
    z : float 
        redshift (Default is 0.0)

    Av_mw : float
        The extinction value from the Milky Way in the direction of the input 
        galaxy.  The default value is 0.2511, which is for IRAS 08339+6517.

    Returns
    -------
    lamdas : :obj:'~numpy.ndarray'
        the wavelength vector for the cubes

    cube_added : :obj:'~numpy.ndarray'
        all cubes added

    cube_mean : :obj:'~numpy.ndarray'
        the mean of all the cubes

    cube_median : :obj:'~numpy.ndarray'
        the median of all the cubes

    header : FITS header object
        the header from the fits file
    """
    #create list to append datas to
    all_data = []
    all_var = []
    all_lamdas = []
    resampled_data = []

    #iterate through the filenames
    #they should all be from fits files, so we can just use that loading function
    for file in glob.glob(filepath):
        fits_stuff = read_in_data_fits(file)
        if len(fits_stuff) > 3:
            lamdas, data, var, header = fits_stuff
            all_var.append(var)
        else:
            lamdas, data, header = fits_stuff
        #apply corrections to lambdas
        lamdas = air_to_vac(lamdas)
        lamdas = barycentric_corrections(lamdas, header)
        #apply Milky Way extinction correction
        data = milky_way_extinction_correction(lamdas, data, Av=Av_mw)
        #de-redshift 
        lamdas = lamdas/(1+z)
        #append the lambdas
        all_lamdas.append(lamdas)
        #and append the data
        all_data.append(data)

    #check if var has the same number of cubes as the data, and if it doesn't, delete it
    if len(all_data) > len(all_var):
        del all_var

    #because the exposures are so close together, the difference in starting lamda
    #between the first to the last cube is only around 0.001A.  There's a difference
    #in the total length of about 0.0003A between the longest and shortest wavelength
    #vectors after the corrections.  So we interpolate along each spectrum and
    #make sure they all start and end at the same spot.

    #take 50A off the beginning and end of the spectrum, this area tends to be
    #weird anyway and create the new wavelength vector
    new_lamda = np.arange(int(all_lamdas[0][0])+50.0, int(all_lamdas[0][-1])-50.0, 0.5)

    #iterate through each data and lamda:
    for count, data in enumerate(all_data):
        #reshape the data so that it is in 2D and the wavelength axis is last
        data = data.reshape((data.shape[0],-1)).swapaxes(0,1)

        #get the old wavelength vector
        lamda = all_lamdas[count]

        #feed this into SpectRes, which is our resampling module
        new_cube = spectres(new_spec_wavs=new_lamda, old_spec_wavs=lamda, spec_fluxes=data, spec_errs=None)

        resampled_data.append(new_cube)

    #adding the data
    cube_added = np.zeros_like(resampled_data[0])

    for cube in resampled_data:
        cube_added += cube

    #finding the mean
    cube_mean = np.mean(resampled_data, axis=0)

    #finding the median
    cube_median = np.median(resampled_data, axis=0)

    #reshape all of these back into cubes, instead of 2d arrays
    cube_added = cube_added.swapaxes(0,1).reshape((-1,all_data[0].shape[1],all_data[0].shape[2]))

    cube_mean = cube_mean.swapaxes(0,1).reshape((-1,all_data[0].shape[1],all_data[0].shape[2]))

    cube_median = cube_median.swapaxes(0,1).reshape((-1,all_data[0].shape[1],all_data[0].shape[2]))

    #pickle the results
    with open(filepath.split('*')[0]+'_'+gal_name+'_combined_by_wavelength_'+str(date.today()),'wb') as f:
        pickle.dump([lamdas, cube_added, cube_mean, cube_median], f)
    f.close()

    return new_lamda, cube_added, cube_mean, cube_median, header


#====================================================================================================
#COORDINATE ARRAYS
#====================================================================================================
def data_coords(lamdas, data, header, cube_colour, z=0.0, shiftx=None, shifty=None):
    """
    Takes the data cube and creates coordinate arrays that are centred on the
    galaxy.  The arrays can be shifted manually.  If this is not hardcoded in to
    the function inputs, the function finds the centre using the maximum continuum
    value.

    Parameters
    ----------
    lamdas : :obj:'~numpy.ndarray'
        the wavelength vector for the cubes

    data : :obj:'~numpy.ndarray'
        the 3D data cube

    header : FITS header object
        the header from the fits file

    cube_colour : str
        whether it is the 'red' or 'blue' cube

    z : float
        redshift, default is 0.0 (since the cube is usually already de-redshifted)

    shiftx : float or None
        the hardcoded shift in the x direction for the coord arrays (in arcseconds).
        If this is none, it finds the maximum point of the median across a section
        of continuum, and makes this the centre.  Default is None.

    shifty : float or None
        the hardcoded shift in the y direction for the coord arrays (in arcseconds).
        If this is none, it finds the maximum point of the median across a section
        of continuum, and makes this the centre.  Default is None.

    Returns
    -------
    xx : :obj:'~numpy.ndarray'
        2D x coordinate array

    yy : :obj:'~numpy.ndarray'
        2D y coordinate array

    rad : :obj:'~numpy.ndarray'
        2D radius array
    """
    #get the data shape
    s = data[0,:,:].shape

    #NAXIS1 = DEC
    #NAXIS2 = RA
    #NAXIS3 = wavelength

    #create x and y ranges
    x = np.arange(s[0]) #RA
    y = np.arange(s[1]) #DEC

    #CD1_1 = RA degrees per column pixel
    #CD2_1 = DEC degrees per column pixel
    #CD1_2 = RA degrees per row pixel
    #CD2_2 = DEC degrees per row pixel

    #CD1_1 = CDELT1 * cos(CROTA2)
    #CD1_2 = -CDELT2 * sin(CROTA2)
    #CD2_1 = CDELT1 * sin(CROTA2)
    #CD2_2 = CDELT2 * cos(CROTA2)

    #multiply through by header values
    try:
        x = x*header['CD1_2']*60*60
        y = y*header['CD2_1']*60*60
    except:
        x = x * (-header['CDELT2']*np.sin(header['CROTA2'])) *60*60
        y = y * (header['CDELT1']*np.sin(header['CROTA2'])) *60*60

    print("x shape, y shape:", x.shape, y.shape)

    #shift the x and y
    if None not in (shiftx, shifty):
        x = x + shiftx
        y = y + shifty

    else:
        if cube_colour == 'red':
            cont_mask = (lamdas>4600*(1+z))&(lamdas<4800*(1+z))
        elif cube_colour == 'blue':
            cont_mask = (lamdas>3600*(1+z))&(lamdas<3700*(1+z))
        elif cube_colour == 'muse':
            cont_mask = (lamdas>5100*(1+z))&(lamdas<5200*(1+z))
        cont_median = np.median(data[cont_mask,:,:], axis=0)
        i, j = np.unravel_index(cont_median.argmax(), cont_median.shape)
        try:
            shiftx = i*header['CD1_2']*60*60
            shifty = j*header['CD2_1']*60*60
        except:
            shiftx = i * (-header['CDELT2']*np.sin(header['CROTA2'])) *60*60
            shifty = j * (header['CDELT1']*np.sin(header['CROTA2'])) *60*60
        #i, j = np.median(np.where(cont_median==np.nanmax(cont_median)),axis=1)
        print("shiftx, shifty:", shiftx, shifty)
        x = x - shiftx
        y = y - shifty

    #create x and y arrays
    xx, yy = np.meshgrid(x,y, indexing='ij')

    print("xx shape, yy shape", xx.shape, yy.shape)

    #create radius array
    rad = np.sqrt(xx**2+yy**2)
    return xx, yy, rad


#====================================================================================================
#CUT DOWN CUBES
#====================================================================================================

def flatten_cube(xx, yy, rad, data, var=None):
    """
    Takes the cube of data and the coordinate arrays and flattens them

    Parameters
    ----------
    xx : :obj:'~numpy.ndarray'
        2D x coordinate array

    yy : :obj:'~numpy.ndarray'
        2D y coordinate array

    rad : :obj:'~numpy.ndarray'
        2D radius array

    data : :obj:'~numpy.ndarray'
        the 3D data cube

    var : :obj:'~numpy.ndarray' or None
        the 3D variance cube (optional), Default is None

    Returns
    -------
    xx_flat : :obj:'~numpy.ndarray'
        flattened x coordinate array (1D)

    yy_flat : :obj:'~numpy.ndarray'
        flattened y coordinate array (1D)

    rad_flat : :obj:'~numpy.ndarray'
        flattened radius array (1D)

    data_flat : :obj:'~numpy.ndarray'
        flattened data array (2D)

    var_flat : :obj:'~numpy.ndarray'
        flattened variance array (2D) if variance cube inputed
    """
    print('        original shape of arrays:')
    print('xx: '+str(xx.shape)+'; yy: '+str(yy.shape))
    print('radius: '+str(rad.shape)+'; data: '+str(data.shape))
    print('')
    print('        Reshaping arrays')

    #flatten to 1D arrays
    xx_flat = np.reshape(xx, (1,-1))
    yy_flat = np.reshape(yy, (1,-1))
    rad_flat = np.reshape(rad, (1,-1))

    #and squeeze to get rid of extra dimensions
    xx_flat = np.squeeze(xx_flat)
    yy_flat = np.squeeze(yy_flat)
    rad_flat = np.squeeze(rad_flat)

    #flatten data to 2D arrays
    data_flat = data.reshape(data.shape[0], -1)

    if var is not None:
        var_flat = var.reshape(var.shape[0], -1)
        return xx_flat, yy_flat, rad_flat, data_flat, var_flat

    else:
        return xx_flat, yy_flat, rad_flat, data_flat


def sn_cut(lamdas, xx_flat, yy_flat, rad_flat, data_flat, z, sn=3):
    """
    Takes out the spectra where a S/N limit is not met.

    Parameters
    ----------
    lamdas : :obj:'~numpy.ndarray'
        the wavelength vector for the cubes

    xx_flat : :obj:'~numpy.ndarray'
        x coordinate array (1D)

    yy_flat : :obj:'~numpy.ndarray'
        y coordinate array (1D)

    rad_flat : :obj:'~numpy.ndarray'
        radius array (1D)

    data_flat : :obj:'~numpy.ndarray'
        data array (2D)

    z : :obj:'~numpy.ndarray'
        redshift of galaxy

    sn : :obj:'~numpy.ndarray'
        target sn, below which the data is not included (default is 3)

    Returns
    -------
    xx_flat : :obj:'~numpy.ndarray'
        cut x coordinate array (1D)

    yy_flat : :obj:'~numpy.ndarray'
        cut y coordinate array (1D)

    rad_flat : :obj:'~numpy.ndarray'
        cut radius array (1D)

    data_flat : :obj:'~numpy.ndarray'
        cut data array (2D)

    s_n_OIII : :obj:'~numpy.ndarray'
        signal-to-noise array
    """
    #use the root mean square error (standard deviation, assuming gaussian distribution) of the continuum as the error
    #cont_mask = (lamdas>4600*(1+z))&(lamdas<4800*(1+z))
    cont_mask = (lamdas>4866*(1+z))&(lamdas<4900*(1+z))
    OIII_mask = (lamdas>5006*(1+z))&(lamdas<5011*(1+z))
    rms = np.sqrt(np.mean(np.square(data_flat[cont_mask,:]), axis=0))

    #rms is the noise
    #find S/N
    #s_n_cont = np.median(data_flat[cont_mask,:], axis=0)/rms
    s_n_OIII = np.median(data_flat[OIII_mask,:], axis=0)/rms

    #make a mask
    s_n_mask = (s_n_OIII > sn)

    #mask out everything where S/N is less than the input level
    xx_flat = xx_flat[s_n_mask]
    yy_flat = yy_flat[s_n_mask]
    rad_flat = rad_flat[s_n_mask]
    data_flat = data_flat[:,s_n_mask]
    s_n_OIII = s_n_OIII[s_n_mask]

    return xx_flat, yy_flat, rad_flat, data_flat, s_n_OIII


#====================================================================================================
#COMBINE RED AND BLUE CUBES
#====================================================================================================
def combine_red_blue(lam_blue, lam_red, blue_cube, red_cube, blue_noise, red_noise, z, header, cube_shape, results_folder):
    """
    Combines the red and blue cubes into one cube

    Parameters
    ----------
    lam_blue : :obj:'~numpy.ndarray'
        wavelength vector for the blue cube

    lam_red : :obj:'~numpy.ndarray'
        wavelength vector for the red cube

    blue_cube : :obj:'~numpy.ndarray'
        flattened blue data cube

    red_cube : :obj:'~numpy.ndarray'
        flattened red data cube

    blue_noise : :obj:'~numpy.ndarray'
        flattened blue noise cube

    red_noise: :obj:'~numpy.ndarray'
        flattened red noise cube

    z : float
        redshift

    Returns
    -------
    lam_all : :obj:'~numpy.ndarray'
        wavelength vector for the full combined cube

    combined_cube : :obj:'~numpy.ndarray'
        a flattened cube containing blue and red cubes

    combined_noise : :obj:'~numpy.ndarray'
        a flattened cube containing the noise for blue and red cubes
    """
    #make sure they cover the same physical space
    assert blue_cube.shape[1]==red_cube.shape[1], "Cubes must have same spatial dimensions"

    #make sure the wavelength ranges overlap
    assert lam_red[0] < lam_blue[-1], "Wavelength vectors must overlap"

    #need to make the cubes have the same wavelength range
    #combine the wavelength vectors to make one vector
    lam_all = np.append(lam_blue, lam_red[lam_red>lam_blue[-1]])

    #add zeros onto the end of the blue cube
    blue_cube = np.append(blue_cube, np.zeros((lam_all.shape[0]-blue_cube.shape[0], blue_cube.shape[1])), axis=0)
    blue_noise = np.append(blue_noise, np.zeros((lam_all.shape[0]-blue_noise.shape[0], blue_noise.shape[1])), axis=0)

    #add zeros onto the beginning of the red cube
    red_cube = np.append(np.zeros((lam_all.shape[0]-red_cube.shape[0], red_cube.shape[1])), red_cube, axis=0)
    red_noise = np.append(np.zeros((lam_all.shape[0]-red_noise.shape[0], red_noise.shape[1])), red_noise, axis=0)

    #create the correction array which will bring blue cube to same level as red cube
    #we do this because we trust the flux calibration of the red cube more than that of the blue
    #this wavelength range works for IRAS08, will have to check later if it fits other galaxies
    lam_mask = (lam_all>4290*(1+z))&(lam_all<4300*(1+z))
    correction_factor = np.median(red_cube[lam_mask, :], axis=0)-np.median(blue_cube[lam_mask, :], axis=0)

    #add the correction factor to the blue cube (but noise stays the same)
    blue_cube = blue_cube+correction_factor

    #take the difference between the two spectra in the wavelength range before Hgamma
    #so we can use the pixel with the smallest difference as the joining point
    #take the absolute value of the difference
    diff_cube = abs(blue_cube[lam_mask, :]-red_cube[lam_mask, :])

    #find where the minimum values are and add the index of the first value in the diff_cube to the indexes
    min_index = np.argmin(diff_cube, axis=0) + np.where(lam_all==lam_all[lam_mask][0])[0][0]

    #now use this to fill in a new array
    combined_cube = np.zeros_like(blue_cube)
    combined_noise = abs(np.empty_like(blue_cube))

    for i in range(combined_cube.shape[1]):
        combined_cube[:min_index[i],i] = blue_cube[:min_index[i],i]
        combined_cube[min_index[i]:, i] = red_cube[min_index[i]:, i]

        combined_noise[:min_index[i],i] = blue_noise[:min_index[i],i]
        combined_noise[min_index[i]:, i] = red_noise[min_index[i]:, i]

    #make sure the noise vectors don't have NaN values
    combined_noise[np.isnan(combined_noise)] = np.nanmin(combined_noise)

    #make sure all of the noise values are greater than zero by replacing any zeros with the next smallest value in the array
    combined_noise[combined_noise==0] = np.nanmin(combined_noise[combined_noise>0])

    #take the beginning and end pixels off - they tend to be a tad dodgy
    #combined_cube = combined_cube[120:-100,:]
    #lam_all = lam_all[120:-100]
    #combined_noise = combined_noise[120:-100,:]

    #update the fits header
    header['NAXIS3'] = lam_all.shape[0]
    header['CRVAL3'] = lam_all[0]

    #save into a fits file
    hdu_data = fits.PrimaryHDU(combined_cube.reshape(lam_all.shape[0], cube_shape[0], cube_shape[1]), header=header)
    hdu_var = fits.ImageHDU(combined_noise.reshape(lam_all.shape[0], cube_shape[0], cube_shape[1]), header=header)
    hdul = fits.HDUList([hdu_data, hdu_var])
    hdul.writeto(results_folder+'combined_cube.fits')

    return lam_all, combined_cube, combined_noise



#====================================================================================================
#RUN CUBE PREPARATION
#====================================================================================================
def prepare_combine_cubes(data_filepath, var_filepath, gal_name, z, cube_colour, Av_mw=0.2511, spatial_crop=False):
    """
    Runs all the previously defined functions to prepare cubes for KOFFEE, ppxf,
    voronoi binning or whatever else needs to be done.

    Parameters
    ----------
    data_filepath : str
        path to the data cube file

    var_filepath : str
        path to the variance cube file

    gal_name : str
        galaxy name or descriptor

    z : float
        redshift

    cube_colour : str
        'red' or 'blue' cube to use in creating the coordinate arrays

    Av_mw : float
        The extinction value from the Milky Way in the direction of the input 
        galaxy.  The default value is 0.2511, which is for IRAS 08339+6517.

    spatial_crop : boolean
        whether to crop the spatial dimensions - this is needed to match to the
        IRAS08 metacube

    Returns
    -------
    Two saved pickle files combined by pixel and by wavelength with:
    lamdas : :obj:'~numpy.ndarray'
        full wavelength array

    xx_flat : :obj:'~numpy.ndarray'
        x coordinate array (1D)

    yy_flat : :obj:'~numpy.ndarray'
        y coordinate array (1D)

    rad_flat : :obj:'~numpy.ndarray'
        radius array (1D)

    data_flat : :obj:'~numpy.ndarray'
        data array (2D)

    var_flat : :obj:'~numpy.ndarray'
        variance array (2D)
    """
    #combine all the cubes (includes reading them in from fits, air_to_vac and barycentric_corrections, and saves them)
    lamdas_pix, cube_added_pix, cube_mean_pix, cube_median_pix, header_pix = data_cubes_combine_by_pixel(data_filepath, gal_name, z=z, Av_mw=Av_mw)

    _, var_added_pix, var_mean_pix, var_median_pix, _ = data_cubes_combine_by_pixel(var_filepath, gal_name+'_var', z=z, Av_mw=Av_mw)

    lamdas_wav, cube_added_wav, cube_mean_wav, cube_median_wav, header_wav = data_cubes_combine_by_wavelength(data_filepath, gal_name, z=z, Av_mw=Av_mw)

    _, var_added_wav, var_mean_wav, var_median_wav, _ = data_cubes_combine_by_wavelength(var_filepath, gal_name+'_var', z=z, Av_mw=Av_mw)

    #create coordinate arrays
    xx_pix, yy_pix, rad_pix = data_coords(lamdas_pix, cube_median_pix, header_pix, cube_colour=cube_colour, shiftx=None, shifty=None)
    xx_wav, yy_wav, rad_wav = data_coords(lamdas_wav, cube_median_wav, header_wav, cube_colour=cube_colour, shiftx=None, shifty=None)

    #if matching to the IRAS08 metacube, crop the spatial dimensions
    if spatial_crop == True:
        var_added_pix = var_added_pix[:, 14:81, 2:26]
        var_mean_pix = var_mean_pix[:, 14:81, 2:26]
        var_median_pix = var_median_pix[:, 14:81, 2:26]
        var_added_wav = var_added_wav[:, 14:81, 2:26]
        var_mean_wav = var_mean_wav[:, 14:81, 2:26]
        var_median_wav = var_median_wav[:, 14:81, 2:26]

        cube_added_pix = cube_added_pix[:, 14:81, 2:26]
        cube_mean_pix = cube_mean_pix[:, 14:81, 2:26]
        cube_median_pix = cube_median_pix[:, 14:81, 2:26]
        cube_added_wav = cube_added_wav[:, 14:81, 2:26]
        cube_mean_wav = cube_mean_wav[:, 14:81, 2:26]
        cube_median_wav = cube_median_wav[:, 14:81, 2:26]

        xx_pix = xx_pix[14:81, 2:26]
        yy_pix = yy_pix[14:81, 2:26]
        rad_pix = rad_pix[14:81, 2:26]
        xx_wav = xx_wav[14:81, 2:26]
        yy_wav = yy_wav[14:81, 2:26]
        rad_wav = rad_wav[14:81, 2:26]

    #flatten coordinate arrays
    xx_flat_pix, yy_flat_pix, rad_flat_pix, data_flat_pix, var_flat_pix = flatten_cube(xx_pix, yy_pix, rad_pix, cube_median_pix, var_median_pix)
    xx_flat_wav, yy_flat_wav, rad_flat_wav, data_flat_wav , var_flat_wav = flatten_cube(xx_wav, yy_wav, rad_wav, cube_median_wav, var_median_wav)

    with open(data_filepath.split('*')[0]+'_'+gal_name+'_combined_by_pixel_flattened_'+str(date.today()),'wb') as f:
        pickle.dump([lamdas_pix, xx_flat_pix, yy_flat_pix, rad_flat_pix, data_flat_pix, var_flat_pix], f)
    f.close()

    with open(data_filepath.split('*')[0]+'_'+gal_name+'_combined_by_wavelength_flattened_'+str(date.today()),'wb') as f:
        pickle.dump([lamdas_wav, xx_flat_wav, yy_flat_wav, rad_flat_wav, data_flat_wav, var_flat_wav], f)
    f.close()


def prepare_single_cube(data_filepath, gal_name, z, cube_colour, results_folder, data_crop=False, var_filepath=None, var_crop=False, lamda_crop=False, mw_correction=False, Av_mw=0.2511):
    """
    Runs all the previously defined functions when there is only one cube to read
    in and nothing to combine.

    Parameters
    ----------
    data_filepath : str
        path to the data cube file

    gal_name : str
        galaxy name or descriptor

    z : float
        redshift

    cube_colour : str
        'red' or 'blue' cube for kcwi data, 'muse' for muse data. Used in 
        creating the coordinate arrays

    results_folder : str
        where to save the results

    data_crop : boolean
        whether to crop the cube spatially - used to make the blue cube match the
        spatial extent of the red IRAS08 metacube.  Default is False.

    var_filepath : str or None
        path to the variance cube file, or None.  Default is None.

    var_crop : boolean
        whether to crop the variance cube spatially - used to make the blue cube
        match the spatial extent of the red IRAS08 metacube.  Default is False.

    lamda_crop : boolean
        whether or not to crop off the dodgy edges in the wavelength direction.
        Default is False.

    mw_correction : boolean
        whether to apply the milky way extinction correction. Default is False.

    Av_mw : float
        The extinction value from the Milky Way in the direction of the input 
        galaxy.  The default value is 0.2511, which is for IRAS 08339+6517.

    Returns
    -------
    lamdas : :obj:'~numpy.ndarray'
        data wavelength array

    var_lamdas : :obj:'~numpy.ndarray'
        variance wavelength array, if var_filepath given

    xx : :obj:'~numpy.ndarray'
        x coordinate array (2D)

    yy : :obj:'~numpy.ndarray'
        y coordinate array (2D)

    rad : :obj:'~numpy.ndarray'
        radius coordinate array (2D)

    data : :obj:'~numpy.ndarray'
        data cube array (3D)

    var : :obj:'~numpy.ndarray'
        variance cube array (3D), if var_filepath given

    xx_flat : :obj:'~numpy.ndarray'
        flattened x coordinate array (1D)

    yy_flat : :obj:'~numpy.ndarray'
        flattened y coordinate array (1D)

    rad_flat : :obj:'~numpy.ndarray'
        flattened radius coordinate array (1D)

    data_flat : :obj:'~numpy.ndarray'
        flattened data cube array (2D)

    var_flat : :obj:'~numpy.ndarray'
        flattened variance cube array (2D), if var_filepath given

    header : FITS header object
        the header from the data fits file
    """
    #read in the data from the fits file, with all corrections
    fits_stuff = load_data(data_filepath, z=z, mw_correction=mw_correction, Av_mw=Av_mw)

    if len(fits_stuff) > 3:
        lamdas, data, var, header = fits_stuff
        var_lamdas = lamdas.copy()
    else:
        lamdas, data, header = fits_stuff

    #if there is a seperate variance cube, read in the data from the fits file
    if var_filepath:
        var_lamdas, var, var_header = load_data(var_filepath, z=z, mw_correction=mw_correction, Av_mw=Av_mw)

        #need to make variance cube the same size as the metacube
        if var_crop == True:
            var = var[:, 14:81, 2:26]
            #use the wavelength vectors to crop the wavelength
            lam_mask = (var_lamdas>=lamdas[0]) & (var_lamdas<=lamdas[-1]+0.5)
            var_lamdas = var_lamdas[lam_mask]
            var = var[lam_mask, :, :]

    if 'var' in locals():
        #check that the variance is always positive
        if np.all((var>0.0) & (np.isfinite(var))) == False:
            print('The variance is not always positive!!!')
            print('Applying absolute value for now... but this should be checked!!!')
            var = abs(var)
            if np.any(var==0.0):
                print('Replacing 0.0 with 0.00000001 in variance')
                var[np.where(var==0.0)] = 0.00000001

        print('Checked var for negative values')

    #create data coordinates
    xx, yy, rad = data_coords(lamdas, data, header, cube_colour=cube_colour, shiftx=None, shifty=None)

    #used if the blue cube is not the same size as the red cube
    if data_crop == True:
        #crop the data
        data = data[:, 14:81, 2:26]
        #also crop the coordinate arrays
        xx = xx[14:81, 2:26]
        yy = yy[14:81, 2:26]
        rad = rad[14:81, 2:26]

    #used to crop the wavelength to get rid of the dodgy edges of the cubes
    if lamda_crop == True:
        #create the masks
        lamda_crop_mask = (lamdas>lamdas[0]+150)&(lamdas<lamdas[-1]-150)
        var_lamda_crop_mask = (var_lamdas>var_lamdas[0]+150)&(var_lamdas<var_lamdas[-1]-150)

        #mask the data, variance and wavelength vectors
        lamdas = lamdas[lamda_crop_mask]
        var_lamdas = var_lamdas[var_lamda_crop_mask]
        data = data[lamda_crop_mask,:,:]
        var = var[var_lamda_crop_mask,:,:]

    #flatten the cubes
    if 'var' in locals():
        xx_flat, yy_flat, rad_flat, data_flat, var_flat = flatten_cube(xx, yy, rad, data, var=var)

        #pickle the output
        with open(results_folder+'/'+gal_name+'_flattened_'+str(date.today()),'wb') as f:
            pickle.dump([lamdas, xx_flat, yy_flat, rad_flat, data_flat, var_flat], f)
        f.close()

        return lamdas, var_lamdas, xx, yy, rad, data, var, xx_flat, yy_flat, rad_flat, data_flat, var_flat, header

    else:
        xx_flat, yy_flat, rad_flat, data_flat = flatten_cube(xx, yy, rad, data)

        #pickle the output
        with open(results_folder+'/'+gal_name+'_flattened_'+str(date.today()),'wb') as f:
            pickle.dump([lamdas, xx_flat, yy_flat, rad_flat, data_flat], f)
        f.close()

        return lamdas, xx, yy, rad, data, xx_flat, yy_flat, rad_flat, data_flat, header
