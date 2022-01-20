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
from astropy import units as u
from astropy.nddata import Cutout2D
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS

from reproject import reproject_adaptive

import montage_wrapper as montage


def convolve_data(filename, fwhm, output_folder, gal_name):
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

    output_folder : str
        Where to save the new fits file

    gal_name : str
        The galaxy name and any other descriptors for the fits file name

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
            try:
                data.shape
            except AttributeError:
                data = hdu['SCI'].data
                header = hdu['SCI'].header

            #if there is more than one extension in the fits file, assume the second one is the variance
            if len(hdu) > 1:
                if (hdu[1].header['EXTNAME'])=='SCI':
                    print('')
                else:
                    var = hdu[1].data
        hdu.close()

    print(data.shape)

    #convert FWHM to standard deviation
    stddev = fwhm/(2*np.sqrt(2*np.log(2)))

    #create the kernel
    gauss_kernel = Gaussian2DKernel(x_stddev=stddev)

    #convolve the data by the kernel
    convolved_data = convolve(data, gauss_kernel, boundary='extend')

    #if there's a variance cube, should also convolve it
    if 'var' in locals():
        convolved_var = convolve(var, gauss_kernel, boundary='extend')

        #save to fits file
        #create HDU object
        hdu = fits.PrimaryHDU(convolved_data, header=header)
        hdu_error = fits.ImageHDU(convolved_var, name='Error')

        #create HDU list
        hdul = fits.HDUList([hdu, hdu_error])

        #write to file
        hdul.writeto(output_folder+gal_name+'_convolved_to_'+str(fwhm)+'arcsec.fits')

        return convolved_data, convolved_var

    else:
        #save to fits file
        #create HDU object
        hdu = fits.PrimaryHDU(convolved_data, header=header)

        #create HDU list
        hdul = fits.HDUList([hdu])

        #write to file
        hdul.writeto(output_folder+gal_name+'_convolved_to_'+str(fwhm)+'arcsec.fits')

        return convolved_data


def cutout_data(filename, position, size, output_folder, gal_name, plot_cutout=True):
    """
    Cuts out a section of the data around a central position, and saves it in a
    new fits file with appropriate wcs.

    Parameters
    ----------
    filename : str
        The filepath to the data; should be a 2D fits file

    position : str
        The position which will be central to the new cutout data in degrees
        e.g. '129.596583 65.120889' for IRAS08

    size : tuple of floats
        The size of the area to be cutout in arcseconds e.g. (20,20)

    output_folder : str
        Where to save the new fits file

    gal_name : str
        The galaxy name and any other descriptors for the fits file name

    plot_cutout : boolean
        If True, plots where the data was cutout from on the original data, and
        also the new cutout data

    Returns
    -------
    cutout : obj
        a Cutout2D instance, with data and wcs, etc.
    """
    #read in the data
    #not a 3d array
    with fits.open(filename) as hdu:
        data = hdu[0].data
        header = hdu[0].header
        try:
            data.shape
        except AttributeError:
            data = hdu['SCI'].data
            header = hdu['SCI'].header

        #if there is more than one extension in the fits file, assume the second one is the variance
        if len(hdu) > 1:
            if (hdu[1].header['EXTNAME'])=='SCI':
                print('')
            else:
                var = hdu[1].data
    hdu.close()

    print(data.shape)

    #create a WCS for the data
    wcs = WCS(header)

    #turn the position into a sky coordinate
    position = SkyCoord(position, unit='deg')

    #give units to the size of the cutout
    size = u.Quantity(size, u.arcsec)

    #cutout the data
    cutout = Cutout2D(data, position, size, wcs=wcs)

    #save the cutout data to a fits file
    #create HDU object
    hdu = fits.PrimaryHDU(cutout.data, header=cutout.wcs.to_header())

    #create HDU list
    hdul = fits.HDUList([hdu])

    #write to file
    hdul.writeto(output_folder+gal_name+'_cutout.fits')

    #plot where the data was cutout from and the new cutout data
    if plot_cutout == True:
        plt.figure()
        ax1 = plt.subplot(1,2,1, projection=wcs)
        ax1.imshow(np.log10(data), origin='lower')
        cutout.plot_on_original(color='white')
        ax1.coords['ra'].set_axislabel('Right Ascension')
        ax1.coords['dec'].set_axislabel('Declination')
        ax1.set_title(gal_name)

        ax2 = plt.subplot(1,2,2, projection=cutout.wcs)
        ax2.imshow(np.log10(cutout.data), origin='lower')
        ax2.coords['ra'].set_axislabel('Right Ascension')
        ax2.coords['dec'].set_axislabel('Declination')
        ax2.coords['dec'].set_axislabel_position('r')
        ax2.coords['dec'].set_ticklabel_position('r')
        ax2.set_title('Cutout')

        plt.show()

    return cutout


def reproject_data(filename1, filename2, output_folder, gal_name, plot_cutout=True):
    """
    Reprojects the data to a second data set's wcs and resolution

    Parameters
    ----------
    filename1 : str
        The filepath to the data to be reprojected; should be a 2D fits file

    filename2 : str
        The filepath to the data with the WCS to be matched; should be a 2D fits
        file

    output_folder : str
        Where to save the new fits file

    gal_name : str
        The galaxy name and any other descriptors for the fits file name

    Returns
    -------
    cutout : obj
        a Cutout2D instance, with data and wcs, etc.
    """
    #read in the data to be reprojected
    #not a 3d array
    with fits.open(filename1) as hdu:
        data1 = hdu[0].data
        header1 = hdu[0].header
        try:
            data1.shape
        except AttributeError:
            data1 = hdu['SCI'].data
            header1 = hdu['SCI'].header

        #if there is more than one extension in the fits file, assume the second one is the variance
        if len(hdu) > 1:
            if (hdu[1].header['EXTNAME'])=='SCI':
                print('')
            else:
                var1 = hdu[1].data
    hdu.close()

    #read in the file to be matched
    with fits.open(filename2) as hdu:
        data2 = hdu[0].data
        header2 = hdu[0].header
        try:
            data2.shape
        except AttributeError:
            data2 = hdu['SCI'].data
            header2 = hdu['SCI'].header
    hdu.close()

    print(data1.shape)
    print(data2.shape)

    #create a WCS for the data to be reprojected
    wcs1 = WCS(header1)

    #create a WCS for the data to be matched
    wcs2 = WCS(header2)

    #reproject data1 to match data2
    reprojected_data, reproject_footprint = reproject_adaptive((data1, wcs1), header2)

    plt.figure()
    ax1 = plt.subplot(1,2,1, projection=wcs2)
    ax1.imshow(np.log10(reprojected_data), origin='lower')
    ax1.coords['ra'].set_axislabel('Right Ascension')
    ax1.coords['dec'].set_axislabel('Declination')
    ax1.set_title('Reprojected')

    ax2 = plt.subplot(1,2,2, projection=wcs2)
    ax2.imshow(np.log10(data2), origin='lower')
    ax2.coords['ra'].set_axislabel('Right Ascension')
    ax2.coords['dec'].set_axislabel('Declination')
    ax2.coords['dec'].set_axislabel_position('r')
    ax2.coords['dec'].set_ticklabel_position('r')
    ax2.set_title('Matched')

    plt.show()




def resample_data(filename, resolution, output_folder, gal_name):
    """
    Resamples data1 to match the resolution of data2

    Parameters
    ----------
    filename : str
        the filepath for the input fits file

    resolution : float
        the resolution to resample the data to (in Arcseconds)

    output_folder : str
        the filepath for where to put the output fits file
    """
    #From IRAS08 fits file, which is what I'm matching at the moment:
    #CD1_1 is the RA degrees per column pixel
    #CD2_1 is the DEC degrees per column pixel
    #CD1_2 is the RA degrees per row pixel
    #CD2_2 is the DEC degrees per row pixel
    #CNAME1 = 'KCWI RA'
    #CNAME2 = 'KCWI DEC'
    #CTYPE1 = 'RA--TAN'
    #CTYPE2 = 'DEC--TAN'

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

    #copy the header data
    new_header = header.copy()

    #convert resolution from arcseconds to degrees
    resolution = resolution/3600

    #use FITS_TOOLS.downsample.downsamle_axis() to reduce the cube along each axis


    #change the CD things
    #doing this the long way so that the zeros stay zeros
    for key in ['CD1_1', 'CD2_1', 'CD1_2', 'CD2_2']:
        try:
            new_header[key] = (resolution/header[key]) * header[key]
        except:
            new_header[key] = 0.0

    #change the number of spaxels
    try:
        new_header['NAXIS1'] = abs(round(header['NAXIS1']/(resolution/header['CD1_2'])))
        new_header['NAXIS2'] = abs(round(header['NAXIS2']/(resolution/header['CD2_1'])))
    except:
        new_header['NAXIS1'] = abs(round(header['NAXIS1']/(resolution/header['CD1_1'])))
        new_header['NAXIS2'] = abs(round(header['NAXIS2']/(resolution/header['CD2_2'])))

    #change the reference pixel to the nearest 0.5
    try:
        new_header['CRPIX1'] = round((header['CRPIX1']/(resolution/header['CD1_2']))*2.0)/2.0
        new_header['CRPIX2'] = round((header['CRPIX2']/(resolution/header['CD2_1']))*2.0)/2.0
    except:
        new_header['CRPIX1'] = round((header['CRPIX1']/(resolution/header['CD1_1']))*2.0)/2.0
        new_header['CRPIX2'] = round((header['CRPIX2']/(resolution/header['CD2_2']))*2.0)/2.0
