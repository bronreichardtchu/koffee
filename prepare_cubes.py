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
    air_to_vac
    barycentric_corrections
    milky_way_extinction_correction
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
from spectres import spectres

#from .display_pixels import cap_display_pixels as cdp
from . import brons_display_pixels_kcwi as bdpk
#import brons_display_pixels_kcwi as bdpk

import importlib
importlib.reload(bdpk)

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
    lamdas = np.arange(header['CRVAL3'], header['CRVAL3']+(header['NAXIS3']*header['CD3_3']), header['CD3_3'])

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


def milky_way_extinction_correction(lamdas, data):
    """
    Corrects for the extinction caused by light travelling through the dust and
    gas of the Milky Way, as described in Cardelli et al. 1989.

    Parameters
    ----------
    lamdas : :obj:'~numpy.ndarray'
        wavelength vector

    data : :obj:'~numpy.ndarray'
        3D cube of data

    Returns
    -------
    data : :obj:'~numpy.ndarray'
        the data corrected for extinction
    """
    #convert lamdas from Angstroms into micrometers
    lamdas = lamdas/10000

    #define the equations from the paper
    y = lamdas - 1.82
    a_x = 1.0 + 0.17699*y - 0.50447*(y**2) - 0.02427*(y**3) + 0.72085*(y**4) + 0.01979*(y**5) - 0.77530*(y**6) + 0.32999*(y**7)
    b_x = 1.41338*y + 2.28305*(y**2) + 1.07233*(y**3) - 5.38434*(y**4) - 0.62251*(y**5) + 5.30260*(y**6) - 2.09002*(y**7)

    #define the constants
    Rv = 3.1
    Av = 0.2511

    #find A(lambda)
    A_lam = (a_x + b_x/Rv)*Av
    #return A_lam

    #apply to the data
    data = (10**(0.4*A_lam[:, None, None]))*data

    return data


#====================================================================================================
#SIMPLE DATA READ IN
#====================================================================================================
def load_data(filename, mw_correction=True):
    """
    Get the data from the fits file and correct to vacuum wavelengths and for
    the earth's rotation

    Parameters
    ----------
    filename : str
        points to the file

    mw_correction : boolean
        whether to apply the milky way extinction correction. Default is True.

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
        data = milky_way_extinction_correction(lamdas, data)
        if len(fits_stuff) > 3:
            var = milky_way_extinction_correction(lamdas, var)

    if len(fits_stuff) > 3:
        return lamdas, data, var, header
    else:
        return lamdas, data, header

#====================================================================================================
#COMBINE CUBES
#====================================================================================================

def data_cubes_combine_by_pixel(filepath, gal_name):
    """
    Grabs datacubes and combines them by pixel using addition, finding the mean
    and the median.

    Parameters
    ----------
    filepath : list of str
        the data cubes filepath strings to pass to glob.glob

    gal_name : str
        galaxy name/descriptor

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
        all_lamdas.append(lamdas)
        #apply Milky Way extinction correction
        data = milky_way_extinction_correction(lamdas, data)
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


def data_cubes_combine_by_wavelength(filepath, gal_name):
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
        #append the lambdas
        all_lamdas.append(lamdas)
        #apply Milky Way extinction correction
        data = milky_way_extinction_correction(lamdas, data)
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
def data_coords(lamdas, data, header, z, cube_colour, shiftx=None, shifty=None):
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

    z : float
        redshift

    cube_colour : str
        whether it is the 'red' or 'blue' cube

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

    #multiply through by header values
    x = x*header['CD1_2']*60*60
    y = y*header['CD2_1']*60*60
    #x = x*header['CDELT1']*60*60
    #y = y*header['CDELT2']*60*60

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
        cont_median = np.median(data[cont_mask,:,:], axis=0)
        i, j = np.unravel_index(cont_median.argmax(), cont_median.shape)
        shiftx = i*header['CD1_2']*60*60
        shifty = j*header['CD2_1']*60*60
        #i, j = np.median(np.where(cont_median==np.nanmax(cont_median)),axis=1)
        #shiftx = i*header['CDELT1']*60*60
        #shifty = j*header['CDELT2']*60*60
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
def combine_red_blue(lam_blue, lam_red, blue_cube, red_cube, blue_noise, red_noise, z):
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

    return lam_all, combined_cube, combined_noise



#====================================================================================================
#RUN CUBE PREPARATION
#====================================================================================================
def prepare_combine_cubes(data_filepath, var_filepath, gal_name, z, cube_colour, spatial_crop=False):
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
    lamdas_pix, cube_added_pix, cube_mean_pix, cube_median_pix, header_pix = data_cubes_combine_by_pixel(data_filepath, gal_name)

    _, var_added_pix, var_mean_pix, var_median_pix, _ = data_cubes_combine_by_pixel(var_filepath, gal_name+'_var')

    lamdas_wav, cube_added_wav, cube_mean_wav, cube_median_wav, header_wav = data_cubes_combine_by_wavelength(data_filepath, gal_name)

    _, var_added_wav, var_mean_wav, var_median_wav, _ = data_cubes_combine_by_wavelength(var_filepath, gal_name+'_var')

    #create coordinate arrays
    xx_pix, yy_pix, rad_pix = data_coords(lamdas_pix, cube_median_pix, header_pix, z, cube_colour=cube_colour, shiftx=None, shifty=None)
    xx_wav, yy_wav, rad_wav = data_coords(lamdas_wav, cube_median_wav, header_wav, z, cube_colour=cube_colour, shiftx=None, shifty=None)

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


def prepare_single_cube(data_filepath, gal_name, z, cube_colour, results_folder, data_corrections=True, data_crop=False, var_filepath=None, var_crop=False, var_corrections=True, lamda_crop=False):
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
        'red' or 'blue' cube to use in creating the coordinate arrays

    results_folder : str
        where to save the results

    data_corrections : boolean
        whether to apply the air_to_vac and barycentric corrections.  Default is
        True.

    data_crop : boolean
        whether to crop the cube spatially - used to make the blue cube match the
        spatial extent of the red IRAS08 metacube.  Default is False.

    var_filepath : str or None
        path to the variance cube file, or None.  Default is None.

    var_crop : boolean
        whether to crop the variance cube spatially - used to make the blue cube
        match the spatial extent of the red IRAS08 metacube.  Default is False.

    var_corrections : boolean
        whether to apply the air_to_vac and barycentric corrections to the variance
        cube.  Default is True.

    lamda_crop : boolean
        whether or not to crop off the dodgy edges in the wavelength direction.
        Default is False.

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
    #read in the data from the fits file
    fits_stuff = read_in_data_fits(data_filepath)

    if len(fits_stuff) > 3:
        lamdas, data, var, header = fits_stuff
    else:
        lamdas, data, header = fits_stuff

    #apply wavelength corrections
    if data_corrections == True:
        lamdas = air_to_vac(lamdas)
        lamdas = barycentric_corrections(lamdas, header)

    #if there is a variance cube, read in the data from the fits file
    if var_filepath:
        var_lamdas, var, var_header = read_in_data_fits(var_filepath)

        #apply wavelength corrections
        if var_corrections == True:
            var_lamdas = air_to_vac(var_lamdas)
            #use same header info as data
            var_lamdas = barycentric_corrections(var_lamdas, header)

        #need to make variance cube the same size as the metacube
        if var_crop == True:
            var = var[:, 14:81, 2:26]
            #use the wavelength vectors to crop the wavelength
            lam_mask = (var_lamdas>=lamdas[0]) & (var_lamdas<=lamdas[-1]+0.5)
            var_lamdas = var_lamdas[lam_mask]
            var = var[lam_mask, :, :]

        #check that the variance is always positive
        if np.all((var>0.0) & (np.isfinite(var))) == False:
            print('The variance is not always positive!!!')
            print('Applying absolute value for now... but this should be checked!!!')
            var = abs(var)
            if np.any(var==0.0):
                print('Replacing 0.0 with 0.00000001 in variance')
                var[np.where(var==0.0)] = 0.00000001

    #apply Milky Way extinction correction
    data = milky_way_extinction_correction(lamdas, data)

    #create data coordinates
    xx, yy, rad = data_coords(lamdas, data, header, z, cube_colour=cube_colour, shiftx=None, shifty=None)

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
    if var_filepath != None:
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
