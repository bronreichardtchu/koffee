"""
NAME:
	convolve_resample_data.py

AUTHOR:
	Bronwyn Reichardt Chu
	Swinburne
	2021

EMAIL:
	<breichardtchu@swin.edu.au>

PURPOSE:
	To convolve and resample the data to match different data sets
	Written on MacOS Mojave 10.14.5, with Python 3.7

FUNCTIONS INCLUDED:


"""

import numpy as np
import matplotlib.pyplot as plt

#from astropy.cosmology import WMAP9 as cosmo
#from astropy.constants import c
#from astropy import units

import prepare_cubes as pc

from astropy.io import fits
from astropy.convolution import Gaussian2DKernel
from astropy.convolution import convolve


def convolve_data(filename, fwhm):
    """
    Convolves the data with a Gaussian of FWHM set by the resolution of the data
    you're trying to match.

    Parameters
    ----------
    filename : str
        The filepath to the data

    fwhm : float
        The full width half maximum of the Gaussian kernel to convolve the data
        with

    Returns
    -------
    convolved_data : :obj:'~numpy.ndarray'
        The convolved data in a numpy array
    """
    #read in the data
    try:
        fits_stuff = pc.load_data(filename, mw_correction=False)
        if len(fits_stuff) > 3:
            lamdas, data, var, header = fits_stuff
        else:
            lamdas, data, header = fits_stuff

    except:
        #not a 3d array, need to read in the data without the lamdas array
        with fits.open(filename) as hdu:
            data = hdu[0].data
            header = hdu[0].header
            #if there is more than one extension in the fits file, assume the second one is the variance
            if len(hdu) > 1:
                var = hdu[1].data
        hdu.close()

    #convert FWHM to standard deviation
    stddev = fwhm/(2*np.sqrt(2*np.log(2)))

    #create the kernel
    gauss_kernel = Gaussian2DKernel(x_stddev=stddev)

    #convolve the data by the kernel
    convolved_data = convolve(data, gauss_kernel, boundary='extend')

    return convolved_data
