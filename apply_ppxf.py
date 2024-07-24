"""
NAME:
	apply_ppxf.py

AUTHOR:
	Bronwyn Reichardt Chu
	Swinburne
	2019

EMAIL:
	<breichardtchu@swin.edu.au>

PURPOSE:
	To apply ppxf to the data cube.
	Written on MacOS Mojave 10.14.5, with Python 3.7

MODIFICATION HISTORY:
		v.1.0 - first created October 2019

"""

import numpy as np
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import ndimage
from hoki import load

import glob
from os import path
import gzip

import pathlib
import pickle
import datetime
from mpi4py import MPI

from time import perf_counter

#from mpi4py import MPI

from ppxf.ppxf import ppxf, robust_sigma
import ppxf.ppxf_util as util
from astropy.io import fits

#from python_scripts import brons_ppxf_util as bpu
import brons_ppxf_util as bpu

import importlib
importlib.reload(bpu)



#====================================================================================================
#TEMPLATES
#====================================================================================================

def get_SSP_library(filepath):
    """
    Get the SSP library filenames list, the data from the first file,
    and the wavelength range of the SSPs

    Parameters
    ----------
    filepath : str
        the general filepath to all of the ssp files,
        eg. '/models/p_walcher09/*.fits'

    Returns
    -------
    ssp_lib : list of str
        a list of all the file names in the library

    ssp_data : :obj:'~numpy.ndarray'
        the data from the first fits file

	ssp_lamrange : :obj:'~numpy.ndarray'
        the wavelength range of the SSPs
    """
    #load the library
    ssp_lib = glob.glob(filepath)

    #read out the header and data for the first spectrum
    with fits.open(ssp_lib[0]) as hdu:
        ssp_header = hdu[0].header
        ssp_data = hdu[0].data
    hdu.close()

	#find the wavelength range of the models
    ssp_lamrange = ssp_header['CRVAL1'] + np.array([0.0, ssp_header['CDELT1']*(ssp_header['NAXIS1']-1.0)])

    return ssp_lib, ssp_data, ssp_lamrange


def get_SSP_library_new_conroy_models(filepath):
	"""
	Get the SSP library for the new Conroy models - these have a weird file
    structure, so they need their own function.

	Parameters
    ----------
	filepath : (string)
        the general filepath to all of the ssp files,
        eg. '/models/ssp_mist_c3k/SSP*.spec'

	Returns
    -------
    ssp_lib : list of str
        a list of all the file names in the library

    ssp_data : :obj:'~numpy.ndarray'
        the data of the SSPs

    ssp_ages : list
        a list of the ages of the SSPs

    ssp_metals : list
        a list of the metallicities of the SSPs

    ssp_lamrange :  :obj:'~numpy.ndarray'
        the wavelength range of the SSPs
	"""
	#load the library names
	ssp_lib = glob.glob(filepath)

	ssp_data = []
	ssp_ages = []
	ssp_metals = []

	#read in the first file
	for file in ssp_lib:
		with open(file, 'r') as fp:
			index = 0
			for count, line in enumerate(fp):
				#get rid of the comment lines
				if line[0] == "#":
					continue
				#get the size of the arrays and create arrays to fill with population variables and spectra
				if count == 8:
					data_size = np.array([int(s) for s in line[:-1].split()])
					ssp_age = np.zeros((data_size[0]))
					ssp_metal = np.full_like(ssp_age, float(file.split('Z')[1][:6]))
					#spectra needs to have shape [nPixels, nAge] for pPXF
					ssp_spectra = np.zeros((data_size[0], data_size[1]))
				#get the wavelength vector
				if count == 9:
					lamdas = np.array([float(s) for s in line.split()])
					lamdas_mask = (lamdas>3000)&(lamdas<7000)
					lamdas = lamdas[lamdas_mask]
				#get all of the spectra and population variables, and fill the arrays
				if count > 9:
					#make the line into an array
					line_array = np.array([float(s) for s in line.split()])
					#if it has four values, put it in the variables
					if line_array.shape[0] == 4:
						ssp_age[index] = (10**line_array[0])/(10**9)
					#if it has the same length as expected for the spectra, put it in the spectra array
					elif line_array.shape[0] == data_size[1]:
						ssp_spectra[index, :] = line_array
						index += 1
					else:
						print('Line {}: shape {} - line is not the right length!'.format(count, line_array.shape[0]))
			#use the wavelength mask on the spectra
			ssp_spectra = ssp_spectra[:, lamdas_mask]
			#add all of this to the total lists
			ssp_data.append(ssp_spectra)
			ssp_ages.append(ssp_age)
			ssp_metals.append(ssp_metal)
		fp.close()

	#find the wavelength range
	ssp_lamrange = np.array([lamdas[0], lamdas[-1]])

	#turn the data, ages, metals into arrays
	ssp_data = np.array(ssp_data)
	ssp_ages = np.array(ssp_ages)
	ssp_metals = np.array(ssp_metals)

	#reshape the spectra so that they have [nTemplates, nPixels]
	ssp_data = ssp_data.reshape((-1, ssp_data.shape[-1]))
	#reshape the population properties so they have [nPixels]
	ssp_ages = ssp_ages.reshape(-1)
	ssp_metals = ssp_metals.reshape(-1)

	return ssp_lib, ssp_data, ssp_ages, ssp_metals, ssp_lamrange


def get_BPASS_library(filepath, gal_lamrange):
    """
    Loads the first BPASS model into a numpy array, gets the library filenames list,
    the wavelength range of the SSPs and the list of ages

    Parameters
    -----------
    filepath : str
        The filenames of the models to be loaded

    gal_lamrange : :obj:'~numpy.ndarray'
        The first and last wavelengths of the galaxy data going to be fit

    Returns
    -------
    ssp_lamrange : :obj:'~numpy.ndarray'
        Wavelength range of the SSPs

    templates : :obj:'~numpy.ndarray'
        Array of BPASS templates ready to be input into ppxf, needs to be
        TEMPLATES[nPixels, nAge, nMetal]

    ssp_ages : list
        List of ages of the templates

    ssp_metals : list
        List of metals of the templates
    """
    #first we need to make a list of the filenames
    ssp_lib = glob.glob(filepath)

    #open the first file to get the wavelength vector and shape for the rest of the array
    #this creates a pandas dataframe
    ssp_data = load.model_output(ssp_lib[0])

    #get the log ages
    ssp_ages = list(ssp_data.columns)[1:]

    #make into a numpy array
    ssp_data = ssp_data.to_numpy()

    ssp_lamdas = ssp_data[:,0]
    ssp_data = ssp_data[:,1:]

    #mask the wavelengths 3000A-6000A, since the wavelength range of KCWI is 350-560nm
    lamda_mask = (ssp_lamdas>gal_lamrange[0]-200)&(ssp_lamdas<gal_lamrange[1]+200)
    ssp_lamdas = ssp_lamdas[lamda_mask]
    ssp_data = ssp_data[lamda_mask,:]

    #get the lam_range
    ssp_lamrange = np.array([ssp_lamdas[0], ssp_lamdas[-1]])

    #create empty templates array to add templates to in shape [nPix, nAge, nMetal]
    templates = np.empty([ssp_data.shape[0], ssp_data.shape[1], len(ssp_lib)])

    #create empty list to add the metals to
    ssp_metals = []

    for j, file in enumerate(ssp_lib):
        #load the file
        ssp_data = load.model_output(file)
        #convert to numpy array and ignore the wavelength vector
        ssp_data = ssp_data.to_numpy()[:,1:]
        #add to templates array
        templates[:,:,j] = ssp_data[lamda_mask,:]
        #add metals to metals list
        ssp_metals.append(file[-11:-7])

    print('Templates shape: ', templates.shape)

    return ssp_lamrange, templates, ssp_ages, ssp_metals


def get_bc03_library(filepath, gal_lamrange):
    """
    Loads the first Bruzual & Charlot 2003 model into a numpy array, gets the
    library filenames list, the wavelength range of the SSPs and the list of ages

    Parameters
    -----------
    filepath : str
        The filenames of the models to be loaded

    gal_lamrange : :obj:'~numpy.ndarray'
        The first and last wavelengths of the galaxy data going to be fit

    Returns
    -------
    ssp_lib : list
        List of filenames for the ssp library

    ssp_templates : :obj:'~numpy.ndarray'
        The models in a numpy array

    ssp_lamrange : :obj:'~numpy.ndarray'
        Wavelength range of the SSPs

    ssp_ages : list
        List of ages of the templates

    ssp_metals : list
        List of the metallicities of the templates
    """
    #first make a list of the filenames
    ssp_lib = glob.glob(filepath)

    #open the first file to get the shape for the rest of the arrays
    with gzip.GzipFile(ssp_lib[0], 'rb') as f:
        ssp_data = np.loadtxt(f)
    f.close()

    ssp_lamdas = ssp_data[:,0]
    ssp_data = ssp_data[:,1]

    #mask the wavelengths
    lamda_mask = (ssp_lamdas>gal_lamrange[0]-200)&(ssp_lamdas<gal_lamrange[1]+200)
    ssp_lamdas = ssp_lamdas[lamda_mask]
    ssp_data = ssp_data[lamda_mask]

    #get the lam_range
    ssp_lamrange = np.array([ssp_lamdas[0], ssp_lamdas[-1]])

    #make lists to append ages and metals to
    ssp_ages = []
    ssp_metals = []

    #get the ages and metals
    for file in ssp_lib:
        age = file.split('/')[-1].split('_')[1]
        #turn the age into a float
        if age[-3:] == 'Myr':
            age = float(age[:-3])*10**6
        elif age[-3:] == 'Gyr':
            age = float(age[:-3])*10**9
        #append to the ssp_ages
        ssp_ages.append(age)

        #now do the metals
        metal = file.split('/')[-1].split('_')[2].split('.')[0]
        metal = '0.' + metal[1:]
        metal = float(metal)
        #append to the ssp_metals
        ssp_metals.append(metal)

    #read in the rest of the templates
    ssp_templates =  np.empty([ssp_data.size, len(ssp_lib)])

    for j, file in enumerate(ssp_lib):
        with gzip.GzipFile(file, 'rb') as f:
            ssp_data = np.loadtxt(f)
        f.close()

        #add to templates
        lamda_mask = (ssp_data[:,0]>gal_lamrange[0]-200)&(ssp_data[:,0]<gal_lamrange[1]+200)
        ssp_templates[:, j] = ssp_data[lamda_mask,1]

    return ssp_templates, ssp_lamrange, ssp_ages, ssp_metals


#====================================================================================================
#PREP SPECTRA
#====================================================================================================

def wavelength_masking(ssp_lamrange, gal_lamdas, gal_lin, gal_noise):
    """
    Mask the data so that the wavelengths aren't longer than those for the
    template spectrum.

    Parameters
    ----------
    ssp_lamrange : :obj:'~numpy.ndarray'
        the wavelength range for the SSPs

    gal_lamdas : :obj:'~numpy.ndarray'
        the wavelength vector for the galaxy data

    gal_lin : :obj:'~numpy.ndarray'
        the galaxy data

    gal_noise : :obj:'~numpy.ndarray'
        the galaxy noise (square root of the variance)

    Returns
    -------
    gal_lamdas : :obj:'~numpy.ndarray'
        the masked wavelength vector for the galaxy data

    gal_lin : :obj:'~numpy.ndarray'
        the masked galaxy data

    gal_noise : :obj:'~numpy.ndarray'
        the masked galaxy noise
    """
    #create the mask
    lam_mask = (gal_lamdas >= ssp_lamrange[0]) & (gal_lamdas <= (ssp_lamrange[1]+1.0))

    #apply the mask to the data
    gal_lamdas = gal_lamdas[lam_mask]
    gal_lin = gal_lin[lam_mask,]
    gal_noise = gal_noise[lam_mask,]

    return gal_lamdas, gal_lin, gal_noise


def prep_spectra(gal_lamdas, gal_lin, gal_noise):
    """
    Prepare the spectrum for ppxf by running it through the log_rebin function.
    Only works on singular spectra, use in a for loop if running over a cube.

    Parameters
    ----------
    gal_lamdas : :obj:'~numpy.ndarray'
        the wavelength vector for the galaxy data (vector)

    gal_lin : :obj:'~numpy.ndarray'
        the galaxy data (one spectrum)

    gal_noise : :obj:'~numpy.ndarray'
        the galaxy noise (square root of the variance) (one spectrum)

    Returns
    -------
    lamRange_gal : :obj:'~numpy.ndarray'
        the wavelength range of the data

    gal_logspec : :obj:'~numpy.ndarray'
        the log rebinned spectrum

    log_noise : :obj:'~numpy.ndarray'
        the log rebinned noise

    gal_logLam :
        the natural logarithm of the wavelength

    gal_velscale :
        the velocity scale in km/s per pixel
    """
    #find the wavelength range of the new data
    lamrange_gal = np.array([np.min(gal_lamdas), np.max(gal_lamdas)])

    #preserve the smallest velocity step
    velscale = np.nanmin(299792.458 * np.diff(np.log(gal_lamdas)))

    #log_rebin to rebin the spectra to a logarithmic scale
    gal_logspec, gal_logLam, gal_velscale = util.log_rebin(lamrange_gal, gal_lin, velscale=velscale)

    #log_rebin the noise as well
    log_noise, noise_logLam, noise_velscale = util.log_rebin(lamrange_gal, gal_noise, velscale=velscale)

    return lamrange_gal, gal_logspec, log_noise, gal_logLam, gal_velscale


#====================================================================================================
#PREP TEMPLATES
#====================================================================================================

def gauss_emission_line_templates(lamrange_gal, ssp_logLam, fwhm_emlines, tie_balmer=True, limit_doublets=True, vacuum=True, extra_em_lines=False):
    """
    Constructs a set of Gaussian emission lines for the templates.
    See ppxf.ppxf_util.emission_lines for more info.

    Parameters
    ----------
    lamrange_gal : :obj:'~numpy.ndarray'
        the wavelength range of the data

    ssp_logLam : :obj:'~numpy.ndarray'
        the natural logarithm of the wavelength of the ssp templates

    fwhm_emlines : float
        the fwhm of the emission lines to be created

    tie_balmer : bool
        ties the Balmer lines according to a theoretical decrement
        (case B recombination T=1e4 K, n=100 cm^-3) (default=True)

    limit_doublets : bool
        limit the ratio of the [OII] and [SII] doublets to the
        ranges allowed by atomic physics. (default=True)

    vacuum : bool
        whether to make wavelengths in vacuum or air wavelengths.
        (default=True, default in ppxf is False)

    extra_em_lines : bool
        set to True to include extra emission lines often found in
        KCWI data (OII 4317, [OIII]4363, OII4414 and [NeIII]3868).
        (default=False)

    Returns
    -------
    gas_templates : :obj:'~numpy.ndarray'
        the templates for the gas lines

    gas_names : list
        the names of the lines included

    line_wave : list
        the wavelength of the lines included in vacuum wavelengths
    """

    gas_templates, gas_names, line_wave = bpu.emission_lines(ssp_logLam, lamrange_gal, fwhm_emlines, tie_balmer=tie_balmer, limit_doublets=limit_doublets, vacuum=vacuum, extra_em_lines=extra_em_lines)

    return gas_templates, gas_names, line_wave


def prep_templates(ssp_lamrange, ssp_lib, ssp_data, gal_velscale, lamrange_gal, fwhm_gal=1.7, fwhm_temp=1.0, cdelt_temp=1.0, velscale_ratio=1, em_lines=False, fwhm_emlines=2.0, vacuum=True, extra_em_lines=False, tie_balmer=True):
    """
    Prepare the templates by log rebinning and normalising them.

    Parameters
    ----------
    ssp_lamrange : :obj:'~numpy.ndarray'
        the wavelength range of the ssp models

    ssp_lib : list
        the list of filenames included in the ssp library

    ssp_data : :obj:'~numpy.ndarray'
        data from the first ssp file

    gal_velscale : float
        the velocity scale of the data

    lamrange_gal : :obj:'~numpy.ndarray'
        the wavelength range of the data

    fwhm_gal : float
        the fwhm of the data

    fwhm_temp : float
        the fwhm of the templates

    cdelt_temp : float
        the delta wavelength of the templates - Angstroms/pixel

    velscale_ratio : float
        the number by which to divide the galaxy velscale by when log rebinning
        the SSPs.  (default=1, gives the same spectral sampling for the templates
        and the galaxy data.  2 would give templates twice sampled compared to
        galaxy data.)

    em_lines : bool
        whether to include emission lines in the templates and ppxf fit
        (default=False)

    fwhm_emlines : float
        the fwhm of the emission lines to be added to the templates

    vacuum : bool
        whether to make wavelengths in vacuum or air wavelengths.
        (default=True, default in ppxf is False)

    extra_em_lines : bool
        set to True to include extra emission lines often found in
        KCWI data (OII 4317, [OIII]4363, OII4414 and [NeIII]3868).
        (default=False)

    tie_balmer : bool
        ties the Balmer lines according to a theoretical decrement
        (case B recombination T=1e4 K, n=100 cm^-3) (default=True)

    Returns
    -------
    templates : :obj:'~numpy.ndarray'
        the array of logarithmically rebinned template SSPs

    ssp_logLam : :obj:'~numpy.ndarray'
        the logarithmically rebinned wavelength vector for the templates and
        emission lines (if included)

    Notes
    -----
    If emission lines are included in the fit also Returns:
        gas_names : the names of the lines included
        line_wave : the wavelength of the lines included in vacuum wavelengths
        component : assigns which templates belong to which components (stellar,
        emission lines, Balmer lines)
        gas_component : vector, True for gas templates
    """
    #apply log_rebin to the first template, and sample at a velocity scale which defaults to the same as that for the spectra
    log_ssp, ssp_logLam, ssp_velscale = util.log_rebin(ssp_lamrange, ssp_data, velscale=gal_velscale/velscale_ratio)

    #create an array for the templates
    templates = np.empty([log_ssp.size, len(ssp_lib)])

    #find the number of templates, which will be used if em_lines is True
    n_temps = templates.shape[1]

    #add all the models into the template array
    for j, file in enumerate(ssp_lib):
        with fits.open(file) as hdu:
            ssp = hdu[0].data
        hdu.close()
        #convolve with the quadratic difference between galaxy and template resolution
        #smooth the template if the template fwhm is less than the galaxy fwhm
        if fwhm_gal > fwhm_temp:
            ssp = smoothing(fwhm_gal, fwhm_temp, ssp, cdelt_temp)
        else:
            ssp = ssp
        log_ssp = util.log_rebin(ssp_lamrange, ssp, velscale=gal_velscale/velscale_ratio)[0]
        templates[:,j] = log_ssp/np.median(log_ssp)

    if em_lines == True:
        #create the gaussian emission lines
        gas_templates, gas_names, line_wave = gauss_emission_line_templates(lamrange_gal, ssp_logLam, fwhm_emlines, tie_balmer=tie_balmer, limit_doublets=True, vacuum=vacuum, extra_em_lines=extra_em_lines)
        #add them to the templates array
        templates = np.column_stack([templates, gas_templates])

        #set template parameters
        n_forbidden = sum("[" in a for a in gas_names)
        n_balmer = len(gas_names) - n_forbidden

        #assign component=0 to the stellar templates, component=1 to the Balmer lines,
        # and component=2 to the forbidden lines
        component = [0]*n_temps + [1]*n_balmer + [2]*n_forbidden
        #gas_component is True for gas templates
        gas_component = np.array(component) > 0

        return templates, ssp_logLam, gas_names, line_wave, component, gas_component

    else:
        return templates, ssp_logLam


def prep_templates_new_conroy_models(ssp_lamrange, ssp_data, gal_velscale, lamrange_gal, fwhm_gal=1.7, fwhm_temp=1.0, cdelt_temp=1.0, velscale_ratio=1, em_lines=False, fwhm_emlines=2.0, vacuum=True, extra_em_lines=False, tie_balmer=True):
    """
    Prepare the templates by log rebinning and normalising them.

    Parameters
    ----------
    ssp_lamrange : :obj:'~numpy.ndarray'
        the wavelength range of the ssp models

    ssp_data : :obj:'~numpy.ndarray'
        data from the first ssp file

    gal_velscale : float
        the velocity scale of the data

    lamrange_gal : :obj:'~numpy.ndarray'
        the wavelength range of the data

    fwhm_gal : float
        the fwhm of the data

    fwhm_temp : float
        the fwhm of the templates

    cdelt_temp : float
        the delta wavelength of the templates - Angstroms/pixel

    velscale_ratio : float
        the number by which to divide the galaxy velscale by when log rebinning
        the SSPs.  (default=1, gives the same spectral sampling for the templates
        and the galaxy data. 2 would give templates twice sampled compared to
        galaxy data.)

    em_lines : boolean
        whether to include emission lines in the templates and ppxf fit
        (default=False)

    fwhm_emlines : float
        the fwhm of the emission lines to be added to the templates

    vacuum : boolean
        whether to make wavelengths in vacuum or air wavelengths.
        (default=True, default in ppxf is False)

    extra_em_lines : bool
        set to True to include extra emission lines often found in
        KCWI data (OII 4317, [OIII]4363, OII4414 and [NeIII]3868).
        (default=False)

    tie_balmer : bool
        ties the Balmer lines according to a theoretical decrement
        (case B recombination T=1e4 K, n=100 cm^-3) (default=True)

    Returns
    -------
    templates : :obj:'~numpy.ndarray'
        the array of logarithmically rebinned template SSPs

    ssp_logLam : :obj:'~numpy.ndarray'
        the logarithmically rebinned wavelength vector for the templates and
        emission lines (if included)

    Notes
    -----
    If emission lines are included in the fit Returns:
        gas_names : the names of the lines included
        line_wave : the wavelength of the lines included in vacuum wavelengths
        component : assigns which templates belong to which components (stellar,
        emission lines, Balmer lines)
        gas_component : vector, True for gas templates
    """
    #apply log_rebin to the first template, and sample at a velocity scale which defaults to the same as that for the spectra
    log_ssp, ssp_logLam, ssp_velscale = util.log_rebin(ssp_lamrange, ssp_data[0,:], velscale=gal_velscale/velscale_ratio)

    #create an array for the templates
    templates = np.empty([log_ssp.size, ssp_data.shape[0]])

    #find the number of templates, which will be used if em_lines is True
    n_temps = templates.shape[1]

    #add all the models into the template array
	#ssp_data has shape [nTemplates, nPixels] so we can use enumerate to iterate through
	#all of the spectra, before putting them into the templates array with shape
	#[nPixels, nTemplates]
    for j, ssp in enumerate(ssp_data):
        #convolve with the quadratic difference between galaxy and template resolution
        #smooth the template if the template fwhm is less than the galaxy fwhm
        if fwhm_gal > fwhm_temp:
            ssp = smoothing(fwhm_gal, fwhm_temp, ssp, cdelt_temp)
        else:
            ssp = ssp

        log_ssp = util.log_rebin(ssp_lamrange, ssp, velscale=gal_velscale/velscale_ratio)[0]
        templates[:,j] = log_ssp/np.median(log_ssp)

    if em_lines == True:
        #create the gaussian emission lines
        gas_templates, gas_names, line_wave = gauss_emission_line_templates(lamrange_gal, ssp_logLam, fwhm_emlines, tie_balmer=tie_balmer, limit_doublets=True, vacuum=vacuum, extra_em_lines=extra_em_lines)
        #add them to the templates array
        templates = np.column_stack([templates, gas_templates])

        #set template parameters
        n_forbidden = sum("[" in a for a in gas_names)
        n_balmer = len(gas_names) - n_forbidden

        #assign component=0 to the stellar templates, component=1 to the Balmer lines,
        # and component=2 to the forbidden lines
        component = [0]*n_temps + [1]*n_balmer + [2]*n_forbidden
        #gas_component is True for gas templates
        gas_component = np.array(component) > 0

        return templates, ssp_logLam, gas_names, line_wave, component, gas_component

    else:
        return templates, ssp_logLam


def prep_BPASS_models(ssp_templates, ssp_lamrange, gal_velscale, lamrange_gal, fwhm_gal=1.7, fwhm_temp=2.0, cdelt_temp=1.0, velscale_ratio=1, em_lines=False, fwhm_emlines=2.0, vacuum=True, extra_em_lines=False, tie_balmer=True):
    """
    Smooths, Log rebins and normalises the templates

    Parameters
    ----------
    ssp_templates : :obj:'~numpy.ndarray'
        The templates in a numpy array

    ssp_lamrange : :obj:'~numpy.ndarray'
        Wavelength range of the SSPs

    gal_velscale : float
        the velocity scale of the data

    lamrange_gal : :obj:'~numpy.ndarray'
        the wavelength range of the data

    fwhm_gal : float
        the fwhm of the data

    fwhm_temp : float
        the fwhm of the templates

    cdelt_temp : float
        the delta wavelength of the templates in Angstroms/pixel

    velscale_ratio : float
        the number by which to divide the galaxy velscale when log rebinning the
        SSPs (default=1, gives the same spectral sampling for the templates and
        the galaxy data. 2 would give templates twice sampled compared to galaxy
        data.)

    em_lines : boolean
        whether to include emission lines in the templates

    fwhm_emlines : float
        the fwhm of the emission lines added to the templates

    vacuum : boolean
        whether to make the wavelengths in vacuum or air wavelengths
        (default=True, default in ppxf is False.)

    extra_em_lines : bool
        set to True to include extra emission lines often found in
        KCWI data (OII 4317, [OIII]4363, OII4414 and [NeIII]3868).
        (default=False)

    tie_balmer : bool
        ties the Balmer lines according to a theoretical decrement
        (case B recombination T=1e4 K, n=100 cm^-3) (default=True)

    Returns
    -------
    templates : :obj: '~numpy.ndarray'

    ssp_logLam : :obj:'~numpy.ndarray'
        the logarithmically rebinned wavelength vector for the templates and
        emission lines (if included)

    Notes
    -----
    If emission lines are included in the fit also Returns:
        gas_names : the names of the lines included
        line_wave : the wavelength of the lines included in vacuum wavelengths
        component : assigns which templates belong to which components (stellar,
        emission lines, Balmer lines)
        gas_component : vector, True for gas templates
        temp_dim : the original dimensions of the templates without the emission
        lines
    """
    #apply log_rebin to the first template, and sample at a velocity scale which defaults to the same as that for the spectra
    log_ssp, ssp_logLam, ssp_velscale = util.log_rebin(ssp_lamrange, ssp_templates[:,0,0], velscale=gal_velscale/velscale_ratio)

    #create an array for the templates
    templates = np.empty([log_ssp.size, ssp_templates.shape[1], ssp_templates.shape[2]])

    #iterate through all of the templates and add to the empty array
    for i in np.arange(templates.shape[1]):
        for j in np.arange(templates.shape[2]):
            #smooth the template if the template fwhm is less than the galaxy fwhm
            if fwhm_gal > fwhm_temp:
                ssp = smoothing(fwhm_gal, fwhm_temp, ssp_templates[:,i,j], cdelt_temp)
            else:
                ssp = ssp_templates[:,i,j]
            #log-rebin the template
            ssp = util.log_rebin(ssp_lamrange, ssp, velscale=gal_velscale/velscale_ratio)[0]
            #take the median and add to templates
            templates[:,i,j] = ssp

    #divide all the templates by the median
    templates = templates/np.nanmedian(templates)

    if em_lines == True:
        #save the original shape
        temp_dim = templates.shape[1:]
        #flatten the templates array
        templates = templates.reshape([templates.shape[0], -1])
        #get the number of stellar templates
        n_temps = templates.shape[1]

        #create the gaussian emission lines
        gas_templates, gas_names, line_wave = gauss_emission_line_templates(lamrange_gal, ssp_logLam, fwhm_emlines, tie_balmer=tie_balmer, limit_doublets=True, vacuum=vacuum, extra_em_lines=extra_em_lines)
        #add the gas lines to the templates
        templates = np.column_stack([templates, gas_templates])

        #set the template parameters
        n_forbidden = sum("[" in a for a in gas_names)
        n_balmer = len(gas_names) - n_forbidden

        #assign component = 0 to the stellar templates
        #component = 1 to the Balmer lines
        #and component = 2 to the forbidden lines
        component = [0]*n_temps + [1]*n_balmer + [2]*n_forbidden
        #fit the same velocity to all components
        #component = [0]*(n_temps+n_balmer+n_forbidden)
        #gas_component is True for gas templates
        gas_component = np.array(component) > 0
        #gas_component = np.zeros_like(component, dtype=bool)
        #gas_component[-(n_balmer+n_forbidden):] = True

        return templates, ssp_logLam, gas_names, line_wave, component, gas_component, temp_dim

    else:
        #reshape templates
        templates = templates.reshape([templates.shape[0], -1])
        return templates, ssp_logLam


def prep_BC03_models(ssp_lamrange, ssp_templates, gal_velscale, lamrange_gal, fwhm_gal=1.7, fwhm_temp=3.0, cdelt_temp=1.0, velscale_ratio=1, em_lines=False, fwhm_emlines=2.0, vacuum=True, extra_em_lines=False, tie_balmer=True):
    """
    Prepares the templates by log rebinning and normalising them

    Parameters
    ----------
    ssp_lamrange : :obj:'~numpy.ndarray'
        Wavelength range of the SSPs

    ssp_templates : :obj:'~numpy.ndarray'
        The templates in a numpy array

    gal_velscale : float
        the velocity scale of the data

    lamrange_gal : :obj:'~numpy.ndarray'
        the wavelength range of the data

    fwhm_gal : float
        the fwhm of the data

    fwhm_temp : float
        the fwhm of the templates (3A for BC03)

    cdelt_temp : float
        the delta wavelength of the templates in Angstroms/pixel
        (1A/pixel for BC03)

    velscale_ratio : float
        the number by which to divide the galaxy velscale when log rebinning the
        SSPs (default=1, gives the same spectral sampling for the templates and
        the galaxy data. 2 would give templates twice sampled compared to galaxy
        data.)

    em_lines : boolean
        whether to include emission lines in the templates

    fwhm_emlines : float
        the fwhm of the emission lines added to the templates

    vacuum : boolean
        whether to make the wavelengths in vacuum or air wavelengths
        (default=True, default in ppxf is False.)

    extra_em_lines : bool
        set to True to include extra emission lines often found in
        KCWI data (OII 4317, [OIII]4363, OII4414 and [NeIII]3868).
        (default=False)

    tie_balmer : bool
        ties the Balmer lines according to a theoretical decrement
        (case B recombination T=1e4 K, n=100 cm^-3) (default=True)

    Returns
    -------
    templates : :obj: '~numpy.ndarray'

    ssp_logLam : :obj:'~numpy.ndarray'
        the logarithmically rebinned wavelength vector for the templates and
        emission lines (if included)

    Notes
    -----
    If emission lines are included in the fit also Returns:
        gas_names : the names of the lines included
        line_wave : the wavelength of the lines included in vacuum wavelengths
        component : assigns which templates belong to which components (stellar,
        emission lines, Balmer lines)
        gas_component : vector, True for gas templates
    """
    #apply the log_rebin to the first template, and sample at a velocity scale
    #which defaults to the same as that for the spectra
    log_ssp, ssp_logLam, ssp_velscale = util.log_rebin(ssp_lamrange, ssp_templates[:,0], velscale=gal_velscale/velscale_ratio)

    #create an array for the log rebinned templates
    templates = np.empty([log_ssp.size, ssp_templates.shape[1]])

    #find the number of templates, which will be used if em_lines is True
    n_temps = templates.shape[1]

    #add all the models into the template array
    for j in np.arange(ssp_templates.shape[1]):
        #smooth the template if the template fwhm is less than the galaxy fwhm
        if fwhm_gal > fwhm_temp:
            ssp = smoothing(fwhm_gal, fwhm_temp, ssp_templates[:,j], cdelt_temp)
        else:
            ssp = ssp_templates[:,j]
        #log-rebin the template
        ssp = util.log_rebin(ssp_lamrange, ssp, velscale=gal_velscale/velscale_ratio)[0]
        #add to the templates
        templates[:,j] = ssp

    #divide all the templates by the median
    templates = templates/np.nanmedian(templates)

    #add the emission lines to the templates
    if em_lines == True:
        #create the gaussian emission lines
        gas_templates, gas_names, line_wave = gauss_emission_line_templates(lamrange_gal, ssp_logLam, fwhm_emlines, tie_balmer=tie_balmer, limit_doublets=True, vacuum=vacuum, extra_em_lines=extra_em_lines)
        #add the gas lines to the templates
        templates = np.column_stack([templates, gas_templates])

        #set the template parameters
        n_forbidden = sum("[" in a for a in gas_names)
        n_balmer = len(gas_names) - n_forbidden

        #assign component = 0 to the stellar templates
        #component = 1 to the Balmer lines
        #and component = 2 to the forbidden lines
        component = [0]*n_temps + [1]*n_balmer + [2]*n_forbidden
        #fit the same velocity to all components
        #component = [0]*(n_temps+n_balmer+n_forbidden)
        #gas_component is True for gas templates
        gas_component = np.array(component) > 0
        #gas_component = np.zeros_like(component, dtype=bool)
        #gas_component[-(n_balmer+n_forbidden):] = True

        return templates, ssp_logLam, gas_names, line_wave, component, gas_component

    else:
        return templates, ssp_logLam




#====================================================================================================
#ADJUSTMENTS
#====================================================================================================

def velocity_shift(ssp_logLam, gal_logLam, velscale_ratio=1):
    """
    Find the difference in starting wavelength between the templates and the
    data.  An extra velocity shift has to be applied to the template to fit the
    galaxy spectrum.  We remove this artificial shift by using the keyword VSYST
    in the call to PPXF, so that all velocities are measured with respect to dv.
    This assumes that the redshift is negligible.  In the case of a high-redshift
    galaxy, one should de-redshift its wavelength to the rest frame before
    finding dv.

    Parameters
    ----------
    ssp_logLam : :obj: '~numpy.ndarray'
        the logarithmically rebinned wavelength vector for the SSP templates

    gal_logLam : :obj: '~numpy.ndarray'
        the logarithmically rebinned wavelength vector for the galaxy data

    velscale_ratio : float
        the ratio of spectral sampling in km/s per pixel

    Returns
    -------
    dv : float
        the difference in starting velocity between the templates and spectra
    """
    #speed of light in km/s
    c = 299792.458

    #difference in wavelength between the two spectra
    if velscale_ratio > 1:
        dv = (np.mean(ssp_logLam[:velscale_ratio]) - gal_logLam[0])*c  # km/s
    else:
        dv = (ssp_logLam[0] - gal_logLam[0])*c  # km/s

    return dv


def smoothing(fwhm1, fwhm2, input_spectrum, cdelt):
    """
    Smooth either the template to the spectrum, or the spectrum to the template
    FWHM. fwhm1 must always be larger than fwhm2.  Usually want to be smoothing
    template to the spectrum - should have higher resolution templates than data.

    Parameters
    ----------
    fwhm1 : float or :obj: '~numpy.ndarray'
        the first and larger FWHM value, should be the one that you want to
        smooth to (the data)

    fwhm2 : float or :obj: '~numpy.ndarray'
        the second and smaller FWHM value, the one you are smoothing (from the
        templates)

    input_spectrum : :obj: '~numpy.ndarray'
        the spectrum to smooth (usually the template)

    cdelt : float
        the wavelength Angstroms per pixel from the templates (might be CDELT1
        or CD3_3 in a fits header)

    Returns
    -------
    input_spectrum : :obj: '~numpy.ndarray'
        the spectrum smoothed to fwhm1
    """
    #find the difference in the fwhm
    fwhm_diff = np.sqrt(fwhm1**2 - fwhm2**2)
    #calculate the sigma difference in pixels
    sigma = fwhm_diff/2.355/cdelt

    #use the difference to smooth the input
    #if the fwhm1 is the same for all pixels, use the scipy convolution,
    #otherwise use the one from ppxf, which can handle convolving a spectrum
    #by a Gaussian with a different sigma for every pixel
    if np.isscalar(fwhm1):
        input_spectrum = ndimage.gaussian_filter1d(input_spectrum, sigma)
    else:
        input_spectrum = bpu.gaussian_filter1d(input_spectrum, sigma)

    return input_spectrum


def clip_outliers(galaxy, bestfit, mask):
    """
    Repeat the fit after clipping bins deviants more than 3*sigma in relative
    error until the bad bins don't change any more. This function uses eq.(34)
    of Cappellari (2023) https://ui.adsabs.harvard.edu/abs/2023MNRAS.526.3273C
    """
    while True:
        scale = galaxy[mask] @ bestfit[mask]/np.sum(bestfit[mask]**2)
        resid = scale*bestfit[mask] - galaxy[mask]
        err = robust_sigma(resid, zero=1)
        ok_old = mask
        mask = np.abs(bestfit - galaxy) < 3*err
        if np.array_equal(mask, ok_old):
            break
            
    return mask



def median_continuum_convolution(spectrum, window=30, smooth_sigma=10):
    """
    Calculates the median along the spectrum and then smooths it using a gaussian
    convolution as an alternative continuum.

    Parameters
    ----------
    spectrum : :obj: '~numpy.ndarray'
        the spectrum to take the median of

    window : int
        the size of the kernal to be shifted over the data when calculating the
        median

    smooth_sigma : float
        the sigma for the gaussian kernal to smooth the median to make the
        continuum

    Returns
    -------
    continuum : :obj: '~numpy.ndarray'
        the median values for the spectrum
    """
    #use the median filter from scipy to convolve the spectrum
    continuum = ndimage.median_filter(spectrum, window)

    #smooth using the gaussian filter
    continuum = ndimage.gaussian_filter1d(continuum, smooth_sigma)

    return continuum


def subtract_median_continuum(lamdas, data, var, header, z, output_folder, gal_name):
    """
    Finds the median continuum value between Hbeta and OIII and subtracts this
    from the spectrum.

    Parameters
    ----------
    lamdas : :obj: '~numpy.ndarray'
        the linear wavelength array

    data : :obj: '~numpy.ndarray'
        the data cube in linear, non-normalised space

    header : FITS file header
        the data header

    z : float
        the redshift

    Returns
    -------
    Saves the continuum subtracted cube to a new fits file

    subtracted_data : :obj: '~numpy.ndarray'
        the median continuum subtracted data
    """
    #create the mask
    cont_mask = (lamdas>4890*(1+z)) & (lamdas<4930*(1+z))

    #calculate the median
    median_cont = np.nanmedian(data[cont_mask,:,:], axis=0)

    #subtract it from the cube
    subtracted_data = data - median_cont

    #save to fits file
    #create HDU object for galaxy
    hdu = fits.PrimaryHDU(subtracted_data, header=header)
    hdu_error = fits.ImageHDU(var, name='Variance')

    #create HDU list
    hdul = fits.HDUList([hdu, hdu_error])

    #write to file
    hdul.writeto(output_folder+gal_name+'_cont_subtracted_median_fixed.fits')

    return subtracted_data


#====================================================================================================
#RUNNING THE FIT
#====================================================================================================

def run_ppxf(gal_logLam, gal_logspec, gal_velscale, log_noise, templates, ssp_logLam, z, em_lines=False, component=False, reddening=0.13, gas_component=False, gas_names=False, gas_reddening=0.13, goodpixels=None, degree=4, mdegree=0, plot=True, quiet=False):
    """
    And finally, put it all into ppxf and fit the spectrum.

    Input:
        gal_logLam:
        gal_logspec: the logarithmically rebinned wavelength
        gal_velscale: the velocity scale of the data (km/s)
        log_noise: the logarithmically rebinned noise
        templates: the array of templates - stellar, emission lines, and whatever else
        ssp_logLam: the logarithmic wavelength of the SSPs
        dv: Reference velocity in km/s (default 0). The input initial guess and the output velocities are
                measured with respect to this velocity. This keyword is generally used to account for the
                difference in the starting wavelength of the templates and the galaxy spectrum as follows
                vsyst = c*np.log(wave_temp[0]/wave_gal[0])
        z: redshift of the galaxy
        em_lines: whether to include emission lines in the fit (default=False)
        component: When fitting more than one kinematic component, this keyword should contain the component
                number of each input template. In principle every template can belong to a different kinematic
                component.
        gas_component: boolean vector, of the same size as COMPONENT, set to True where the given COMPONENT
                describes a gas emission line. If given, pPXF provides the pp.gas_flux and pp.gas_flux_error
                in output.
        gas_names: string array specifying the names of the emission lines (e.g. gas_names=["Hbeta", "[OIII]",...],
                one per gas line. The length of this vector must match the number of nonzero elements in gas_component.
                This vector is only used by pPXF to print the line names on the console.
        gas_reddening: Set this keyword to an initial estimate of the gas reddening E(B-V) >= 0 to fit a positive
                reddening together with the kinematics and the templates. The fit assumes by default the
                extinction curve of Calzetti et al. (2000, ApJ, 533, 682)
        goodpixels: integer vector containing the indices of the good pixels in the
               GALAXY spectrum (in increasing order). Only these spectral pixels are
               included in the fit.
        degree: degree of the additive polynomial used to correct the continuum in the fit (default=4, set =-1 to
                not inculde an additive polynomial)
        mdegree: degree of the multiplicative polynomial used to correct the continuum in the fit (default=0, degenerate with reddening)
        plot: whether to plot the fit and show it on screen or not (default=True)
        quiet: turns off all the printed stuff in the fit (default=False)

    Returns:
        pp:
    """
    #speed of light in km/s
    c = 299792.458

    #eq (8) of Cappellari (2017)
    if z > 0.0:
        vel = c*np.log(1+z)
    else:
        vel=0

    #starting guess for [Vel, sigma] in km/s
    #starting guess for sigma should be more than 3xvelscale [km/s]
    start = [vel, 100.]

    if em_lines == True:
        #Fit (V, sig, h3, h4) for the stars, but only (V, sig) for the two gas kinematic components
        moments = [2, 2, 2]
        #use the same starting value for the stars and the two gas components
        start = [start, start, start]
        #tie the moments being fitted to the stellar templates
        tied = [['',''], ['p[0]', 'p[1]'], ['p[0]', 'p[1]']]

    else:
        #Fit 4 moments for the stars (V, sig, h3, h4)
        moments = 4

    #set the clock going
    t = perf_counter()

    #do the fit
    if em_lines == True:
        # create the mask where it doesn't mask anything
        mask = np.ones_like(gal_logspec, dtype='bool')

        # run once to estimate the scatter in the spectrum
        pp = ppxf(templates, gal_logspec, log_noise, gal_velscale, start, 
                  moments=moments, plot=plot, #vsyst=dv, 
                  lam=np.exp(gal_logLam), 
                  lam_temp=np.exp(ssp_logLam), 
                  component=component, 
                  gas_component=gas_component, 
                  gas_names=gas_names, 
                  gas_reddening=gas_reddening, 
                  reddening=reddening, 
                  degree=degree, mdegree=mdegree, 
                  quiet=quiet, tied=tied)
        
        # clip the outliers
        clipped_mask = clip_outliers(gal_logspec, pp.bestfit, mask)

        # add clipped pixels to the original mask 
        clipped_mask &= mask 

        # re-run the fit 
        # run once to estimate the scatter in the spectrum
        pp = ppxf(templates, gal_logspec, log_noise, gal_velscale, start, 
                  moments=moments, plot=plot, #vsyst=dv, 
                  lam=np.exp(gal_logLam), 
                  lam_temp=np.exp(ssp_logLam), 
                  component=component, 
                  gas_component=gas_component, 
                  gas_names=gas_names, 
                  gas_reddening=gas_reddening, 
                  reddening=reddening, 
                  degree=degree, mdegree=mdegree,
                  mask=clipped_mask, 
                  quiet=quiet, tied=tied)

        if quiet == False:
            print('Formal errors:')
            print('              dV      dsigma   dh3      dh4')
            print('stars:   ', ''.join("%8.2g" % f for f in pp.error[0]*np.sqrt(pp.chi2)))
            print('balmer:  ', ''.join("%8.2g" % f for f in pp.error[1]*np.sqrt(pp.chi2)))
            print('forbidden', ''.join("%8.2g" % f for f in pp.error[2]*np.sqrt(pp.chi2)))
    else:
        # run once to estimate the scatter in the spectrum
        pp = ppxf(templates, gal_logspec, log_noise, gal_velscale, start, 
                  moments=moments, goodpixels=goodpixels, 
                  plot=plot, #vsyst=dv, 
                  lam=np.exp(gal_logLam), 
                  lam_temp=np.exp(ssp_logLam), 
                  reddening=reddening, 
                  degree=degree, mdegree=mdegree, 
                  quiet=quiet)
        
        # create a mask using goodpixels 
        mask = np.zeros_like(gal_logspec, dtype='bool')
        mask[goodpixels] = True 
        
        # clip the outliers
        new_mask = clip_outliers(gal_logspec, pp.bestfit, mask)

        # add clipped pixels to the original masked emission line regions 
        new_mask &= mask

        # re-run the fit 
        pp = ppxf(templates, gal_logspec, log_noise, gal_velscale, start, 
                  moments=moments, #goodpixels=this_goodpixels, 
                  plot=plot, #vsyst=dv, 
                  lam=np.exp(gal_logLam), 
                  lam_temp=np.exp(ssp_logLam), 
                  reddening=reddening, 
                  degree=degree, mdegree=mdegree,
                  mask=new_mask, 
                  quiet=quiet)

        if quiet == False:
            print('Formal errors:')
            print('     dV      dsigma   dh3      dh4')
            print(''.join("%8.2g" % f for f in pp.error*np.sqrt(pp.chi2)))

    print('Elapsed time in pPXF: %.2f s' % (perf_counter()-t))

    if plot == True:
        plt.show()

    return pp



#====================================================================================================
#FIT CHECKS
#====================================================================================================

def hgamma_hbeta_ratio(lamdas, data, z=0.0):
    """
    Gets the approximate flux ratio between the H_beta and H_gamma emission lines
    """
    #find the maximum flux in the region around H_beta
    Hbeta_mask = (lamdas>4833*(1+z)) & (lamdas<4893*(1+z))
    Hbeta_flux = data[Hbeta_mask,].max(axis=0)

    #find the maximum flux in the region around H_gamma
    try:
        Hgamma_mask = (lamdas>4312*(1+z)) & (lamdas<4372*(1+z))
        Hgamma_flux = data[Hgamma_mask,].max(axis=0)

        #calculate the ratio
        flux_ratio = Hgamma_flux/Hbeta_flux

        return flux_ratio

    except:
        print('Hgamma not in wavelength range')







#====================================================================================================
#COMBINE DATA
#====================================================================================================

def combine_results(lamdas, data_flat, final_shape, results_folder, galaxy_name, header_file, unnormalised=False, em_lines=True):
    """
    Combine the results from the continuum subtraction

    Inputs:
        data_flat: the data to be continuum subtracted
        results_folder: (string) the folder the results are in
        galaxy_name: (string) the name of the galaxy
        header_file: (string) the location and name of the fits file which has the header we will copy

    Outputs:
        A fits file with all of the results put back into the galaxy.
    """
    #create an array for the results to get put into
    cont_subtracted = np.zeros_like(data_flat)

    #iterate through all of the saved results files
    if unnormalised == False:
        for i in range(data_flat.shape[1]):
            try:
                #open all the saved files for the spectrum
                with open(results_folder+galaxy_name+'_{:0>4d}_ppxf_continuum_subtracted'.format(i), 'rb') as f:
                    lam, gal, cont_subtracted_spec = pickle.load(f)
                f.close()
                with open(results_folder+galaxy_name+'_{:0>4d}_ppxf_bestfit'.format(i), 'rb') as f:
                    bestfit = pickle.load(f)
                f.close()
                if em_lines == True:
                    with open(results_folder+galaxy_name+'_{:0>4d}_ppxf_gas_bestfit'.format(i), 'rb') as f:
                        gas_bestfit = pickle.load(f)
                    f.close()
                    #create the continuum
                    continuum = bestfit - gas_bestfit
                elif em_lines == False:
                    #create the continuum
                    continuum = bestfit
                #interpolate the continuum
                continuum = np.interp(lamdas, lam, continuum)
                #save the continuum subtracted spectrum to the array
                cont_subtracted[:,i] = data_flat[:,i] - continuum

            except TypeError:
                #print('TypeError for spectrum '+str(i))

                #open all the saved files for the spectrum
                with open(results_folder+galaxy_name+'_{:0>4d}_ppxf_continuum_subtracted'.format(i), 'rb') as f:
                    lam, gal, cont_subtracted_spec = pickle.load(f)
                f.close()
                with open(results_folder+galaxy_name+'_{:0>4d}_ppxf_bestfit'.format(i), 'rb') as f:
                    bestfit = pickle.load(f)
                f.close()

                #create the continuum
                continuum = bestfit
                #print(continuum)
                #interpolate the continuum
                continuum = np.interp(lamdas, lam, continuum)
                #print(continuum)
                #save the continuum subtracted spectrum to the array
                cont_subtracted[:,i] = data_flat[:,i] - continuum

            except IOError:
                #it might have been a low S/N spectrum, which has a different name
                try:
                    with open(results_folder+galaxy_name+'_{:0>4d}_ppxf_low_sn_normalised_continuum_value'.format(i), 'rb') as f:
                        normalised_continuum, gal_norm = pickle.load(f)
                    f.close()

                    #create the continuum
                    continuum = normalised_continuum

                    #save the continuum subtracted spectrum to the array
                    cont_subtracted[:,i] = data_flat[:,i]/gal_norm - continuum

                except Exception as ex:
                    print('Exception for spectrum '+str(i))
                    print(ex)
                    print('')

    elif unnormalised == True:
        for i in range(data_flat.shape[1]):
            try:
                #open all the saved files for the spectrum
                with open(results_folder+galaxy_name+'_{:0>4d}_ppxf_continuum_subtracted_unnormalised'.format(i), 'rb') as f:
                    lam, gal_norm, gal, cont_subtracted_spec = pickle.load(f)
                f.close()
                with open(results_folder+galaxy_name+'_{:0>4d}_ppxf_bestfit'.format(i), 'rb') as f:
                    bestfit = pickle.load(f)
                f.close()
                if em_lines == True:
                    with open(results_folder+galaxy_name+'_{:0>4d}_ppxf_gas_bestfit'.format(i), 'rb') as f:
                        gas_bestfit = pickle.load(f)
                    f.close()
                    #create the continuum
                    continuum = (bestfit - gas_bestfit)*gal_norm
                elif em_lines == False:
                    #create the continuum
                    continuum = (bestfit)*gal_norm
                #interpolate the continuum
                continuum = np.interp(lamdas, lam, continuum)
                #save the continuum subtracted spectrum to the array
                cont_subtracted[:,i] = data_flat[:,i] - continuum

            except TypeError:
                #print('TypeError for spectrum '+str(i))
                #open all the saved files for the spectrum
                with open(results_folder+galaxy_name+'_{:0>4d}_ppxf_continuum_subtracted'.format(i), 'rb') as f:
                    lam, gal, cont_subtracted_spec = pickle.load(f)
                f.close()
                with open(results_folder+galaxy_name+'_{:0>4d}_ppxf_bestfit'.format(i), 'rb') as f:
                    bestfit = pickle.load(f)
                f.close()

                #create the continuum
                continuum = bestfit
                #print(continuum)
                #interpolate the continuum
                continuum = np.interp(lamdas, lam, continuum)
                #print(continuum)
                #save the continuum subtracted spectrum to the array
                cont_subtracted[:,i] = data_flat[:,i] - continuum

            except IOError:
                #it might have been a low S/N spectrum, which has a different name
                try:
                    with open(results_folder+galaxy_name+'_{:0>4d}_ppxf_low_sn_normalised_continuum_value'.format(i), 'rb') as f:
                        normalised_continuum, gal_norm = pickle.load(f)
                    f.close()

                    #create the continuum
                    continuum = normalised_continuum*gal_norm

                    #save the continuum subtracted spectrum to the array
                    cont_subtracted[:,i] = data_flat[:,i] - continuum

                except Exception as ex:
                    print('Exception for spectrum '+str(i))
                    print(ex)
                    print('')

    #reshape array
    cont_subtracted = cont_subtracted.reshape(final_shape)

    #pickle the results
    if unnormalised == False:
        with open(results_folder+galaxy_name+'_cont_subtracted_cube', 'wb') as f:
            pickle.dump([lam, cont_subtracted], f)
        f.close()

    elif unnormalised == True:
        with open(results_folder+galaxy_name+'_cont_subtracted_unnormalised_cube', 'wb') as f:
            pickle.dump([lam, cont_subtracted], f)
        f.close()

    #create the header
    with fits.open(header_file) as hdu:
        fits_header = hdu[0].header
    hdu.close()

    #change the header to reflect the new wavelength range (if it's not the same)
    #because this usually happens if we've cropped the wavelength in the prepare
    #cubes function, we can guess at the new lam.shape, and if that's still wrong,
    #then use the actual lam vector to set the new header.  This is necessary
    #because if we just use the lam vector, it already has wavelength corrections
    #applied to it.
    if lam.shape[0] != fits_header['NAXIS3']:
        if (fits_header['NAXIS3']-600) == lam.shape[0]:
            fits_header['NAXIS3'] = lam.shape[0]
            fits_header['CRVAL3'] = fits_header['CRVAL3']+150
        else:
            print("Using wavelength corrected lamda vector to set CRVAL3")
            fits_header['NAXIS3'] = lam.shape[0]
            fits_header['CRVAL3'] = lam[0]

    #create the hdu
    hdu = fits.PrimaryHDU(cont_subtracted, header=fits_header)

    #put it into an hdu list
    hdul = fits.HDUList([hdu])

    #write to file
    if unnormalised == False:
        hdul.writeto(results_folder+galaxy_name+'_cont_subtracted_cube.fits')
    elif unnormalised == True:
        hdul.writeto(results_folder+galaxy_name+'_cont_subtracted_unnormalised_cube.fits')



#====================================================================================================
#PLOTTING FUNCTIONS
#====================================================================================================

def plot_fit(pp, galaxy_name, results_folder, i, xx, yy):
	"""
	Plot the pPXF fit

	Inputs:
		pp: the pPXF fit object
		galaxy_name: (string) the galaxy name
		results_folder: (string) the folder where the results are being saved
		i: the spaxel number
		xx: the x coordinate (usually found using xx_flat[i])
		yy: the y coordinate (usually found using yy_flat[i])
	"""
	pp.plot()
	plt.title(galaxy_name+' pPXF fit for pixel {} ({:.2f}, {:.2f})'.format(i, xx, yy))
	plt.savefig(results_folder+galaxy_name+'_{:0>4d}_ppxf_plot.png'.format(i))

	plt.gcf().clear()
	plt.close()


def plot_em_lines_fit(pp, galaxy_name, results_folder, i, xx, yy, z=0.0):
    """
    Plots the fit zoomed in on the Hbeta and Hgamma lines

    Inputs:
        pp: the pPXF fit object
        galaxy_name: (string) the galaxy name
        z: (float) redshift of the galaxy
        results_folder: (string) the folder where the results are being saved
        i: the spaxel number
        xx: the x coordinate (usually found using xx_flat[i])
        yy: the y coordinate (usually found using yy_flat[i])
    """
    fig = plt.figure()
    if pp.lam[0] > 4200:
        ax1 = plt.subplot(221)
        pp.plot()
        #Hgamma has wavelength 4340.471A
        ax1.set_xlim(0.429*(1+z), 0.439*(1+z))
        ax1.set_ylim(0.0, 1.5)
        ax1.set_title(r'H$\gamma$')
        ax1.set_xlabel('')
        ax1.set_ylabel("Relative Flux")

        ax2 = plt.subplot(222)
        pp.plot()
        #Hbeta has wavelength 4861.333A
        ax2.set_xlim(0.481*(1+z), 0.491*(1+z))
        ax2.set_ylim(0.0, 1.5)
        ax2.set_title(r'H$\beta$')
        ax2.set_xlabel('')
        ax2.set_ylabel('')

        ax3 = plt.subplot(223)
        pp.plot()
        #HeI has wavelength 4471.479A
        ax3.set_xlim(0.442*(1+z), 0.452*(1+z))
        ax3.set_ylim(0.0, 1.5)
        ax3.set_title('He I')
        ax3.set_xlabel(r"Wavelength [$\mu$m]")
        ax3.set_ylabel("Relative Flux")

        ax4 = plt.subplot(224)
        pp.plot()
        #[OIII] has wavelengths 4958.92A, 5006.84A
        ax4.set_xlim(0.493*(1+z), 0.503*(1+z))
        ax4.set_ylim(0.0, 1.5)
        ax4.set_title('[O III]')
        ax4.set_xlabel(r"Wavelength [$\mu$m]")
        ax4.set_ylabel('')

    elif pp.lam[0] < 4200:
        ax1 = plt.subplot(221)
        pp.plot()
        #the OII doublet has wavelengths 3726.032, 3728.815
        ax1.set_xlim(0.367*(1+z), 0.377*(1+z))
        ax1.set_ylim(0.0, 1.5)
        ax1.set_title('OII doublet')
        ax1.set_xlabel('')
        ax1.set_ylabel("Relative Flux")

        ax2 = plt.subplot(222)
        pp.plot()
        #[Ne III] has wavelength 3967.47
        ax2.set_xlim(0.391*(1+z), 0.401*(1+z))
        ax2.set_ylim(0.0, 1.5)
        ax2.set_title('[NeIII]')
        ax2.set_xlabel('')
        ax2.set_ylabel('')

        ax3 = plt.subplot(223)
        pp.plot()
        #Hgamma has wavelength 4340.471A
        ax3.set_xlim(0.429*(1+z), 0.439*(1+z))
        ax3.set_ylim(0.0, 1.5)
        ax3.set_title(r'H$\gamma$')
        ax3.set_xlabel(r"Wavelength [$\mu$m]")
        ax3.set_ylabel("Relative Flux")

        ax4 = plt.subplot(224)
        pp.plot()
        #Hdelta has wavelength 4101.76A
        ax4.set_xlim(0.405*(1+z), 0.415*(1+z))
        ax4.set_ylim(0.0, 1.5)
        ax4.set_title(r'H$\delta$')
        ax4.set_xlabel(r"Wavelength [$\mu$m]")
        ax4.set_ylabel('')

    plt.suptitle(galaxy_name+' pPXF fit for pixel {} ({:.2f}, {:.2f})'.format(i, xx, yy))
    plt.savefig(results_folder+galaxy_name+'_{:0>4d}_ppxf_plot_em_lines_zoomedy.png'.format(i))

    plt.gcf().clear()
    plt.close()


def plot_continuum_subtracted(pp, galaxy_name, results_folder, i, xx, yy):
	"""
	Plots the continuum subtracted spectrum

	Inputs:
		pp: the pPXF fit object
		galaxy_name: (string) the galaxy name
		results_folder: (string) the folder where the results are being saved
		i: the spaxel number
		xx: the x coordinate (usually found using xx_flat[i])
		yy: the y coordinate (usually found using yy_flat[i])
	"""
	fig = plt.figure()

	try:
		plt.step(pp.lam, pp.galaxy-(pp.bestfit-pp.gas_bestfit), where='mid')
	except:
		plt.step(pp.lam, pp.galaxy-pp.bestfit, where='mid')

	plt.ylim(-0.25, 1.0)
	plt.xlabel(r"Wavelength [$\AA$]")
	plt.ylabel("Relative Flux")

	plt.suptitle(galaxy_name+' continuum subtracted for pixel {} ({:.2f}, {:.2f})'.format(i, xx, yy))
	plt.savefig(results_folder+galaxy_name+'_{:0>4d}_ppxf_plot_cont_subtracted.png'.format(i))

	plt.gcf().clear()
	plt.close()


def plot_em_lines_cont_subtracted(pp, galaxy_name, results_folder, i, xx, yy, z=0.0):
    """
    Plots the spectrum after the continuum has been subtracted zoomed in on the Hbeta and Hgamma lines

    Inputs:
        pp: the pPXF fit object
        galaxy_name: (string) the galaxy name
        z: (float) redshift of the galaxy
        results_folder: (string) the folder where the results are being saved
        i: the spaxel number
        xx: the x coordinate (usually found using xx_flat[i])
        yy: the y coordinate (usually found using yy_flat[i])
    """
    fig = plt.figure()

    if pp.lam[0] > 4200:
        ax1 = plt.subplot(221)
        try:
            plt.step(pp.lam, pp.galaxy-(pp.bestfit-pp.gas_bestfit), where='mid')
        except:
            plt.step(pp.lam, pp.galaxy-pp.bestfit, where='mid')
        #Hgamma has wavlength 4340.471A
        ax1.set_xlim(4290*(1+z), 4390*(1+z))
        ax1.set_ylim(-0.5, 1.5)
        ax1.set_xlabel('')
        ax1.set_ylabel("Relative Flux")
        ax1.set_title(r'H$\gamma$')

        ax2 = plt.subplot(222)
        try:
            plt.step(pp.lam, pp.galaxy-(pp.bestfit-pp.gas_bestfit), where='mid')
        except:
            plt.step(pp.lam, pp.galaxy-pp.bestfit, where='mid')
        #Hbeta has wavelength 4861.333A
        ax2.set_xlim(4811*(1+z), 4911*(1+z))
        ax2.set_ylim(-0.5, 1.5)
        ax2.set_xlabel('')
        ax2.set_ylabel('')
        ax2.set_title(r'H$\beta$')

        ax3 = plt.subplot(223)
        try:
            plt.step(pp.lam, pp.galaxy-(pp.bestfit-pp.gas_bestfit), where='mid')
        except:
            plt.step(pp.lam, pp.galaxy-pp.bestfit, where='mid')
        #HeI has wavelength 4471.479A
        ax3.set_xlim(4421*(1+z), 4521*(1+z))
        ax3.set_ylim(-0.5, 1.5)
        ax3.set_xlabel(r"Wavelength [$\AA$]")
        ax3.set_ylabel("Relative Flux")
        ax3.set_title('He I')

        ax4 = plt.subplot(224)
        try:
            plt.step(pp.lam, pp.galaxy-(pp.bestfit-pp.gas_bestfit), where='mid')
        except:
            plt.step(pp.lam, pp.galaxy-pp.bestfit, where='mid')
        #[OIII] has wavelengths 4958.92A, 5006.84A
        ax4.set_xlim(4933*(1+z), 5033*(1+z))
        ax4.set_ylim(-0.5, 1.5)
        ax4.set_title('[O III]')
        ax4.set_xlabel(r"Wavelength [$\AA$]")
        ax4.set_ylabel('')

    elif pp.lam[0] < 4200:
        ax1 = plt.subplot(221)
        try:
            plt.step(pp.lam, pp.galaxy-(pp.bestfit-pp.gas_bestfit), where='mid')
        except:
            plt.step(pp.lam, pp.galaxy-pp.bestfit, where='mid')
        #the OII doublet has wavelengths 3726.032, 3728.815
        ax1.set_xlim(3677*(1+z), 3777*(1+z))
        ax1.set_ylim(-0.5, 1.5)
        ax1.set_xlabel('')
        ax1.set_ylabel("Relative Flux")
        ax1.set_title('OII doublet')

        ax2 = plt.subplot(222)
        try:
            plt.step(pp.lam, pp.galaxy-(pp.bestfit-pp.gas_bestfit), where='mid')
        except:
            plt.step(pp.lam, pp.galaxy-pp.bestfit, where='mid')
        #[Ne III] has wavelength 3967.47
        ax2.set_xlim(3917*(1+z), 4017*(1+z))
        ax2.set_ylim(-0.5, 1.5)
        ax2.set_xlabel('')
        ax2.set_ylabel('')
        ax2.set_title('[NeIII]')

        ax3 = plt.subplot(223)
        try:
            plt.step(pp.lam, pp.galaxy-(pp.bestfit-pp.gas_bestfit), where='mid')
        except:
            plt.step(pp.lam, pp.galaxy-pp.bestfit, where='mid')
        #Hgamma has wavelength 4340.471A
        ax3.set_xlim(4290*(1+z), 4390*(1+z))
        ax3.set_ylim(-0.5, 1.5)
        ax3.set_xlabel(r"Wavelength [$\AA$]")
        ax3.set_ylabel("Relative Flux")
        ax3.set_title(r'H$\gamma$')

        ax4 = plt.subplot(224)
        try:
            plt.step(pp.lam, pp.galaxy-(pp.bestfit-pp.gas_bestfit), where='mid')
        except:
            plt.step(pp.lam, pp.galaxy-pp.bestfit, where='mid')
        #Hdelta has wavelength 4101.76A
        ax4.set_xlim(4051*(1+z), 4151*(1+z))
        ax4.set_ylim(-0.5, 1.5)
        ax4.set_xlabel(r"Wavelength [$\AA$]")
        ax4.set_ylabel('')
        ax4.set_title(r'H$\delta$')


    plt.suptitle(galaxy_name+' pPXF fit for pixel {} ({:.2f}, {:.2f})'.format(i, xx, yy))
    plt.savefig(results_folder+galaxy_name+'_{:0>4d}_ppxf_plot_cont_subtracted_em_lines_zoomedy.png'.format(i))

    plt.gcf().clear()
    plt.close()

def plot_compare_continuum_subtractions(lamdas, data_flat, xx_flat, yy_flat, results_folder, galaxy_name, unnormalised=False):
    """
    Compares the interpolated and not interpolated continuum subtractions

    Inputs:
        data_flat: the data to be continuum subtracted
        results_folder: (string) the folder the results are in
        galaxy_name: (string) the name of the galaxy
        header_file: (string) the location and name of the fits file which has the header we will copy

    Outputs:
        A fits file with all of the results put back into the galaxy.

    """
    #iterate through all of the saved results files
    if unnormalised == False:
        for i in range(data_flat.shape[1]):
            try:
                #open all the saved files for the spectrum
                with open(results_folder+galaxy_name+'_{:0>4d}_ppxf_continuum_subtracted'.format(i), 'rb') as f:
                    lam, gal, cont_subtracted_spec = pickle.load(f)
                f.close()
                with open(results_folder+galaxy_name+'_{:0>4d}_ppxf_bestfit'.format(i), 'rb') as f:
                    bestfit = pickle.load(f)
                f.close()
                with open(results_folder+galaxy_name+'_{:0>4d}_ppxf_gas_bestfit'.format(i), 'rb') as f:
                    gas_bestfit = pickle.load(f)
                f.close()
                #create the continuum
                continuum = bestfit - gas_bestfit
                #interpolate the continuum
                continuum = np.interp(lamdas, lam, continuum)
                #save the continuum subtracted spectrum to the array
                cont_subtracted_interp = data_flat[:,i] - continuum

                #plot the things
                fig = plt.figure()
                plt.step(lam, cont_subtracted_spec, where='mid', label='Original')
                plt.step(lamdas, cont_subtracted_interp, where='mid', label='Interpolated')
                plt.legend()

                plt.xlabel(r"Wavelength [$\AA$]")
                plt.ylabel("Relative Flux")

                plt.suptitle(galaxy_name+' continuum subtracted for pixel {} ({:.2f}, {:.2f})'.format(i, xx_flat[i], yy_flat[i]))
                plt.savefig(results_folder+galaxy_name+'_{:0>4d}_ppxf_plot_compare_cont_subtracted.png'.format(i))

                plt.gcf().clear()
                plt.close()

            except IOError:
                print('Could not open file for spectrum '+str(i))

    elif unnormalised == True:
        for i in range(data_flat.shape[1]):
            try:
                #open all the saved files for the spectrum
                with open(results_folder+galaxy_name+'_{:0>4d}_ppxf_continuum_subtracted_unnormalised'.format(i), 'rb') as f:
                    lam, gal_norm, gal, cont_subtracted_spec = pickle.load(f)
                f.close()
                with open(results_folder+galaxy_name+'_{:0>4d}_ppxf_bestfit'.format(i), 'rb') as f:
                    bestfit = pickle.load(f)
                f.close()
                with open(results_folder+galaxy_name+'_{:0>4d}_ppxf_gas_bestfit'.format(i), 'rb') as f:
                    gas_bestfit = pickle.load(f)
                f.close()
                #create the continuum
                continuum = (bestfit - gas_bestfit)*gal_norm
                #interpolate the continuum
                continuum = np.interp(lamdas, lam, continuum)
                #save the continuum subtracted spectrum to the array
                cont_subtracted_interp = data_flat[:,i] - continuum

                #plot the things
                fig = plt.figure()
                plt.step(lam, cont_subtracted_spec, where='mid', label='Original')
                plt.step(lamdas, cont_subtracted_interp, where='mid', label='Interpolated')
                plt.legend()

                plt.xlabel(r"Wavelength [$\AA$]")
                plt.ylabel("Relative Flux")

                plt.suptitle(galaxy_name+' continuum subtracted for pixel {} ({:.2f}, {:.2f})'.format(i, xx_flat[i], yy_flat[i]))
                plt.savefig(results_folder+galaxy_name+'_{:0>4d}_ppxf_plot_compare_cont_subtracted_unnormalised.png'.format(i))

                plt.gcf().clear()
                plt.close()

            except IOError:
                print('Could not open file for spectrum '+str(i))







#====================================================================================================
#MAIN FUNCTION
#====================================================================================================

def main_parallelised(lamdas, data_flat, noise_flat, xx_flat, yy_flat, ssp_filepath, z, results_folder, galaxy_name, cube_colour, sn_cut=3, fwhm_gal=2, fwhm_temp=2.0, cdelt_temp=1.0, em_lines=True, fwhm_emlines=2.0, gas_reddening=0.13, reddening=0.13, degree=4, mdegree=0, vacuum=True, extra_em_lines=False, tie_balmer=True, maskwidth=800, plot=False, quiet=True):
    """
    Parallelising the main() function

    Parameters
    ----------
    cdelt_temp : float
        the sampling of the template.  This is: 0.05 for Conroy models, 1.0 for BPASS,
        1.0 for BC03, 0.9 for Walcher09.  Default=1.0.

    extra_em_lines : bool
        set to True to include extra emission lines often found in
        KCWI data (OII 4317, [OIII]4363, OII4414 and [NeIII]3868).
        (default=False)

    tie_balmer : bool
        ties the Balmer lines according to a theoretical decrement
        (case B recombination T=1e4 K, n=100 cm^-3) (default=True)
    """
    #load the core info
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank() #the nth core
    size = comm.Get_size() #the total number of cores

    print('galaxy wavelength range:', lamdas[0], lamdas[-1])

    #use the filepath to figure out which functions to use
    #load the SSP library
    if 'c3k' in ssp_filepath:
        ssp_lib, ssp_data, ssp_ages, ssp_metals, ssp_lamrange = get_SSP_library_new_conroy_models(ssp_filepath)
    elif 'BPASS' in ssp_filepath:
        ssp_lamrange, ssp_data, ssp_ages, ssp_metals = get_BPASS_library(ssp_filepath, [lamdas[0], lamdas[-1]])
    elif 'bc03' in ssp_filepath:
        ssp_templates, ssp_lamrange, ssp_ages, ssp_metals = get_bc03_library(ssp_filepath, [lamdas[0], lamdas[-1]])
    else:
        ssp_lib, ssp_data, ssp_lamrange = get_SSP_library(ssp_filepath)

    #calculate the S/N array for the data
    if cube_colour == 'red':
        cont_mask = (lamdas>4600*(1+z))&(lamdas<4800*(1+z))
    elif cube_colour == 'blue':
        cont_mask = (lamdas>3600*(1+z))&(lamdas<3700*(1+z))
        #get S/N array for [OII] doublet - 10A mask around 3727A
        OII_mask = (lamdas>3722*(1+z))&(lamdas<3732*(1+z))
        OII_sn_array = abs(np.nanmedian(data_flat[OII_mask,:]/noise_flat[OII_mask,:], axis=0))

    sn_array = abs(np.nanmedian(data_flat[cont_mask,:]/noise_flat[cont_mask,:], axis=0))


    #if the data FWHM is less than the template FWHM, we can choose to smooth the
    #galaxy data (this is not a good thing though)
    if fwhm_gal < fwhm_temp:
        print('WARNING:     SMOOTHING DATA TO LOWER RESOLUTION')
        data_flat = smoothing(fwhm_temp, fwhm_gal, data_flat, cdelt=0.5)
        noise_flat = smoothing(fwhm_temp, fwhm_gal, noise_flat, cdelt=0.5)

    #mask wavelengths so galaxy doesn't cover wavelengths not in the SSPs
    #(works on both single spectra and cubes)
    gal_lamdas, gal_lin, gal_noise = wavelength_masking(ssp_lamrange, lamdas, data_flat, noise_flat)

    #create the arrays to populate with logbinned and normalised spectra
    #gal_logspec = np.empty_like(gal_lin)
    #log_noise = np.empty_like(gal_noise)
    #gal_velscale = np.empty_like(gal_lin[0,:])

    #logbin and normalise the spectra
    #for i, (spec, noise) in enumerate(zip(gal_lin.T, gal_noise.T)):
        #lamrange_gal, gal_logspec[:,i], log_noise[:,i], gal_logLam, gal_velscale[i] = prep_spectra(gal_lamdas, spec, noise)

    lamrange_gal, gal_logspec, log_noise, gal_logLam, gal_velscale = prep_spectra(gal_lamdas, gal_lin, gal_noise)

    #normalise the spectra and noise
    gal_norm = np.nanmedian(gal_logspec, axis=0)
    gal_logspec = gal_logspec/gal_norm
    noise_norm = np.nanmedian(log_noise, axis=0)
    log_noise = log_noise/noise_norm

    #get the flux ratio for the spectra if fitting the red cube
    if cube_colour == 'red':
        gal_flux_ratio = hgamma_hbeta_ratio(lamdas, data_flat, z=0.0)

    #get all of the templates, logbin and normalise them
    if em_lines == True:
        if 'c3k' in ssp_filepath:
            templates, ssp_logLam, gas_names, line_wave, component, gas_component = prep_templates_new_conroy_models(ssp_lamrange, ssp_data, gal_velscale[0], lamrange_gal, z, fwhm_gal=fwhm_gal, fwhm_temp=fwhm_temp, cdelt_temp=cdelt_temp, velscale_ratio=1, em_lines=True, fwhm_emlines=fwhm_emlines, vacuum=vacuum, extra_em_lines=extra_em_lines, tie_balmer=tie_balmer)

        elif 'BPASS' in ssp_filepath:
            templates, ssp_logLam, gas_names, line_wave, component, gas_component, temp_dim = prep_BPASS_models(ssp_data, ssp_lamrange, gal_velscale[0], lamrange_gal, z, fwhm_gal=fwhm_gal, fwhm_temp=fwhm_temp, cdelt_temp=cdelt_temp, velscale_ratio=1, em_lines=True, fwhm_emlines=fwhm_emlines, vacuum=vacuum, extra_em_lines=extra_em_lines, tie_balmer=tie_balmer)

        elif 'bc03' in ssp_filepath:
            templates, ssp_logLam, gas_names, line_wave, component, gas_component = prep_BC03_models(ssp_lamrange, ssp_templates, gal_velscale[0], lamrange_gal, z, fwhm_gal=fwhm_gal, fwhm_temp=fwhm_temp, cdelt_temp=cdelt_temp, velscale_ratio=1, em_lines=True, fwhm_emlines=fwhm_emlines, vacuum=vacuum, extra_em_lines=extra_em_lines, tie_balmer=tie_balmer)

        else:
            templates, ssp_logLam, gas_names, line_wave, component, gas_component = prep_templates(ssp_lamrange, ssp_lib, ssp_data, gal_velscale[0], lamrange_gal, z, fwhm_gal=fwhm_gal, fwhm_temp=fwhm_temp, cdelt_temp=cdelt_temp, velscale_ratio=1, em_lines=True, fwhm_emlines=fwhm_emlines, vacuum=vacuum, extra_em_lines=extra_em_lines, tie_balmer=tie_balmer)

        #set goodpixels as None
        goodpixels = None

    else:
        if 'c3k' in ssp_filepath:
            templates, ssp_logLam = prep_templates_new_conroy_models(ssp_lamrange, ssp_data, gal_velscale, lamrange_gal, z, fwhm_gal=fwhm_gal, fwhm_temp=fwhm_temp, cdelt_temp=cdelt_temp, velscale_ratio=1, em_lines=False, fwhm_emlines=fwhm_emlines, vacuum=vacuum)

        elif 'BPASS' in ssp_filepath:
            templates, ssp_logLam = prep_BPASS_models(ssp_data, ssp_lamrange, gal_velscale, lamrange_gal, z, fwhm_gal=fwhm_gal, fwhm_temp=fwhm_temp, cdelt_temp=cdelt_temp, velscale_ratio=1, em_lines=False, fwhm_emlines=fwhm_emlines, vacuum=vacuum)

        elif 'bc03' in ssp_filepath:
            templates, ssp_logLam = prep_BC03_models(ssp_lamrange, ssp_templates, gal_velscale, lamrange_gal, z, fwhm_gal=fwhm_gal, fwhm_temp=fwhm_temp, cdelt_temp=cdelt_temp, velscale_ratio=1, em_lines=False, fwhm_emlines=fwhm_emlines, vacuum=vacuum)

        else:
            templates, ssp_logLam = prep_templates(ssp_lamrange, ssp_lib, ssp_data, gal_velscale, lamrange_gal, z, fwhm_gal=fwhm_gal, fwhm_temp=fwhm_temp, cdelt_temp=cdelt_temp, velscale_ratio=1, em_lines=False, fwhm_emlines=fwhm_emlines, vacuum=vacuum)

        #create goodpixels vector
        #goodpixels = util.determine_goodpixels(gal_logLam, ssp_lamrange, z)
        goodpixels = bpu.determine_goodpixels(gal_logLam, ssp_lamrange, z, maskwidth=maskwidth)

    #find the difference in starting values for the templates and spectra
    #dv = velocity_shift(ssp_logLam, gal_logLam, velscale_ratio=1)

    num_spectra = gal_logspec[0,:].shape

    #run ppxf
    for i in np.arange(0+rank, num_spectra[0], size):
        print('Fit '+str(i+1)+' of '+str(num_spectra))

        # if the spectrum has a high enough S/N, fit with ppxf
        if sn_array[i] >= sn_cut:
            if em_lines == True:
                if cube_colour == 'blue':
                    #run normally if S/N of [OII] doublet is > 3
                    #if OII_sn_array[i] >= 3:
                    pp = run_ppxf(gal_logLam, gal_logspec[:,i], gal_velscale, 
                                  log_noise[:,i], templates, ssp_logLam, z, 
                                  em_lines=em_lines, 
                                  component=component, 
                                  gas_component=gas_component, 
                                  gas_names=gas_names, 
                                  gas_reddening=gas_reddening, 
                                  reddening=reddening, 
                                  goodpixels=goodpixels, 
                                  degree=degree, 
                                  mdegree=mdegree, 
                                  plot=plot, 
                                  quiet=quiet
                                  )

                    with open(results_folder+galaxy_name+'_{:0>4d}_ppxf_continuum_subtracted'.format(i), 'wb') as f:
                        pickle.dump([pp.lam, pp.galaxy, (pp.galaxy-(pp.bestfit-pp.gas_bestfit))], f)
                    f.close()

                    with open(results_folder+galaxy_name+'_{:0>4d}_ppxf_continuum_subtracted_unnormalised'.format(i), 'wb') as f:
                        pickle.dump([pp.lam, gal_norm[i], pp.galaxy*gal_norm[i], (pp.galaxy-(pp.bestfit-pp.gas_bestfit))*gal_norm[i]], f)
                    f.close()


                elif cube_colour == 'red':
                    #fit the cube
                    pp = run_ppxf(gal_logLam, gal_logspec[:,i], gal_velscale, 
                                  log_noise[:,i], templates, ssp_logLam, z, 
                                  em_lines=em_lines, 
                                  component=component, 
                                  gas_component=gas_component, 
                                  gas_names=gas_names, 
                                  gas_reddening=gas_reddening, 
                                  reddening=reddening, 
                                  goodpixels=goodpixels, 
                                  degree=degree, 
                                  mdegree=mdegree, 
                                  plot=plot, 
                                  quiet=quiet
                                  )

                    #get the flux ratio if fitting the red cube
                    cont_subt_flux_ratio = hgamma_hbeta_ratio(pp.lam, pp.galaxy - (pp.bestfit-pp.gas_bestfit), z=0.0)
                    print('original ratio: ', gal_flux_ratio[i], 'cont subtracted ratio: ', cont_subt_flux_ratio)
                    if abs(cont_subt_flux_ratio-gal_flux_ratio[i]) > 0.5:
                        print('Continuum subtraction overdoing it')
                    else:
                        print('Continuum subtraction within flux ratio bounds')

                    with open(results_folder+galaxy_name+'_{:0>4d}_ppxf_flux_ratio'.format(i), 'wb') as f:
                        pickle.dump([gal_flux_ratio[i], cont_subt_flux_ratio, cont_subt_flux_ratio-gal_flux_ratio[i]], f)
                    f.close()

                    with open(results_folder+galaxy_name+'_{:0>4d}_ppxf_continuum_subtracted'.format(i), 'wb') as f:
                        pickle.dump([pp.lam, pp.galaxy, (pp.galaxy-(pp.bestfit-pp.gas_bestfit))], f)
                    f.close()

                    with open(results_folder+galaxy_name+'_{:0>4d}_ppxf_continuum_subtracted_unnormalised'.format(i), 'wb') as f:
                        pickle.dump([pp.lam, gal_norm[i], pp.galaxy*gal_norm[i], (pp.galaxy-(pp.bestfit-pp.gas_bestfit))*gal_norm[i]], f)
                    f.close()

                #save the results
                with open(results_folder+galaxy_name+'_{:0>4d}_ppxf_solutions_errors'.format(i), 'wb') as f:
                    pickle.dump([pp.sol, pp.error], f)
                f.close()

                with open(results_folder+galaxy_name+'_{:0>4d}_ppxf_weights'.format(i), 'wb') as f:
                    pickle.dump(pp.weights, f)
                f.close()

                with open(results_folder+galaxy_name+'_{:0>4d}_ppxf_bestfit'.format(i), 'wb') as f:
                    pickle.dump(pp.bestfit, f)
                f.close()

                with open(results_folder+galaxy_name+'_{:0>4d}_ppxf_chi2'.format(i), 'wb') as f:
                    pickle.dump(pp.chi2, f)
                f.close()

                with open(results_folder+galaxy_name+'_{:0>4d}_ppxf_gas_bestfit'.format(i), 'wb') as f:
                    pickle.dump(pp.gas_bestfit, f)
                f.close()

                with open(results_folder+galaxy_name+'_{:0>4d}_ppxf_gas_reddening'.format(i), 'wb') as f:
                    pickle.dump(pp.gas_reddening, f)
                f.close()

                with open(results_folder+galaxy_name+'_{:0>4d}_ppxf_reddening'.format(i), 'wb') as f:
                    pickle.dump(pp.reddening, f)
                f.close()



                #save the figures
                plot_fit(pp, galaxy_name, results_folder, i, xx=xx_flat[i], yy=yy_flat[i])

                plot_em_lines_fit(pp, galaxy_name, results_folder, i, xx=xx_flat[i], yy=yy_flat[i])

                plot_continuum_subtracted(pp, galaxy_name, results_folder, i, xx=xx_flat[i], yy=yy_flat[i])

                plot_em_lines_cont_subtracted(pp, galaxy_name, results_folder, i, xx=xx_flat[i], yy=yy_flat[i])

            else:
                pp = run_ppxf(gal_logLam, gal_logspec[:,i], gal_velscale, 
                              log_noise[:,i], templates, ssp_logLam, z, 
                              em_lines=em_lines, 
                              component=False, 
                              gas_component=False, 
                              gas_names=False, 
                              gas_reddening=None, 
                              reddening=reddening, 
                              degree=degree, 
                              goodpixels=goodpixels, 
                              mdegree=mdegree, 
                              plot=plot, 
                              quiet=quiet
                              )

                #get the flux ratio if fitting the red cube
                if cube_colour == 'red':
                    cont_subt_flux_ratio = hgamma_hbeta_ratio(pp.lam, pp.galaxy - (pp.bestfit), z=0.0)
                    print('original ratio: ', gal_flux_ratio[i], 'cont subtracted ratio: ', cont_subt_flux_ratio)
                    if abs(cont_subt_flux_ratio-gal_flux_ratio[i]) > 0.5:
                        print('Continuum subtraction overdoing it')
                    else:
                        print('Continuum subtraction within flux ratio bounds')

                    with open(results_folder+galaxy_name+'_{:0>4d}_ppxf_flux_ratio'.format(i), 'wb') as f:
                        pickle.dump([gal_flux_ratio[i], cont_subt_flux_ratio, cont_subt_flux_ratio-gal_flux_ratio[i]], f)

                #save the results
                with open(results_folder+galaxy_name+'_{:0>4d}_ppxf_solutions_errors'.format(i), 'wb') as f:
                    pickle.dump([pp.sol, pp.error], f)
                f.close()

                with open(results_folder+galaxy_name+'_{:0>4d}_ppxf_weights'.format(i), 'wb') as f:
                    pickle.dump(pp.weights, f)
                f.close()

                with open(results_folder+galaxy_name+'_{:0>4d}_ppxf_bestfit'.format(i), 'wb') as f:
                    pickle.dump(pp.bestfit, f)
                f.close()

                with open(results_folder+galaxy_name+'_{:0>4d}_ppxf_chi2'.format(i), 'wb') as f:
                    pickle.dump(pp.chi2, f)
                f.close()

                with open(results_folder+galaxy_name+'_{:0>4d}_ppxf_reddening'.format(i), 'wb') as f:
                    pickle.dump(pp.reddening, f)
                f.close()

                with open(results_folder+galaxy_name+'_{:0>4d}_ppxf_continuum_subtracted'.format(i), 'wb') as f:
                    pickle.dump([pp.lam, pp.galaxy, (pp.galaxy-pp.bestfit)], f)
                f.close()

                with open(results_folder+galaxy_name+'_{:0>4d}_ppxf_continuum_subtracted_unnormalised'.format(i), 'wb') as f:
                    pickle.dump([pp.lam, gal_norm[i], pp.galaxy*gal_norm[i], (pp.galaxy-pp.bestfit)*gal_norm[i]], f)
                f.close()

                #save the figures
                plot_fit(pp, galaxy_name, results_folder, i, xx=xx_flat[i], yy=yy_flat[i])

                plot_em_lines_fit(pp, galaxy_name, results_folder, i, xx=xx_flat[i], yy=yy_flat[i])

                plot_continuum_subtracted(pp, galaxy_name, results_folder, i, xx=xx_flat[i], yy=yy_flat[i])

                plot_em_lines_cont_subtracted(pp, galaxy_name, results_folder, i, xx=xx_flat[i], yy=yy_flat[i])

        #if the spectrum has a low S/N, then fit the continuum with a straight line
        elif sn_array[i] < sn_cut:
            print('Fitting with median continuum value')

            #normalise the spectrum before fitting the continuum
            #this is to stay in line with the ppxf fitting, which is done to
            #the normalised cube
            normalised_continuum = np.nanmedian(data_flat[cont_mask,i]/gal_norm[i])

            #save the number in a file for later
            with open(results_folder+galaxy_name+'_{:0>4d}_ppxf_low_sn_normalised_continuum_value'.format(i), 'wb') as f:
                pickle.dump([normalised_continuum, gal_norm[i]], f)
            f.close()

    print('==========FINISHED==========')

    #return the last fit object for testing
    #return pp




#======================================================================================================

if __name__ == '__main__':

    #from koffee import prepare_cubes
    import prepare_cubes
    from mpi4py import MPI


    """
    #IRAS08 red cube read-in
    data_filepath = '/Users/breichardtchu/Documents/data/IRAS08_red_cubes/IRAS08339_metacube.fits'
    var_filepath = '/Users/breichardtchu/Documents/data/IRAS08_red_cubes/kb180215_00081_vcubes.fits'
    gal_name = 'IRAS08339'
    z = 0.018950
    ssp_filepath = '/Users/breichardtchu/Documents/models/BPASS_v2.2.1_Tuatara/BPASSv2.2.1_bin-imf135_300/spectra-bin-imf135_300*'
    #results_folder = '/Users/breichardtchu/Documents/code_outputs/IRAS08_ppxf_22Feb2021/'
    results_folder = '/Users/breichardtchu/Documents/code_outputs/IRAS08_ppxf_trial/'

    lamdas, var_lamdas, xx, yy, rad, data, var, xx_flat, yy_flat, rad_flat, data_flat, var_flat, data_header = prepare_cubes.prepare_single_cube(data_filepath=data_filepath, gal_name=gal_name, z=z, cube_colour='red', results_folder=results_folder, data_crop=False, var_filepath=var_filepath, var_crop=True, mw_correction=True)


    #IRAS08 blue cube read-in
    data_filepath = '/Users/breichardtchu/Documents/data/IRAS08_blue_cubes/IRAS08_blue_combined.fits'
    var_filepath = '/Users/breichardtchu/Documents/data/IRAS08_blue_cubes/IRAS08_blue_combined_var.fits'
    gal_name = 'IRAS08339'
    z = 0.018950
    ssp_filepath = '/Users/breichardtchu/Documents/models/BPASS_v2.2.1_Tuatara/BPASSv2.2.1_bin-imf135_300/spectra-bin-imf135_300*'
    results_folder = '/Users/breichardtchu/Documents/code_outputs/IRAS08blue_ppxf_4March2021_combined/'

    lamdas, var_lamdas, xx, yy, rad, data, var, xx_flat, yy_flat, rad_flat, data_flat, var_flat, data_header = prepare_cubes.prepare_single_cube(data_filepath=data_filepath, gal_name=gal_name, z=z, cube_colour='blue', results_folder=results_folder, data_crop=True, var_filepath=var_filepath, var_crop=True, lamda_crop=True, mw_correction=True)


    #CGCG453 red cube read-in
    data_filepath = '/Users/breichardtchu/Documents/data/cgcg453_red_mosaic_binned_by_3.fits'
    var_filepath = '/Users/breichardtchu/Documents/data/cgcg453_red_var_binned_by_3.fits'
    gal_name = 'CGCG453'
    z = 0.0251
    ssp_filepath = '/Users/breichardtchu/Documents/models/BPASS_v2.2.1_Tuatara/BPASSv2.2.1_bin-imf135_300/spectra-bin-imf135_300*'
    results_folder = '/Users/breichardtchu/Documents/code_outputs/cgcg453_red_ppxf_25Jan2021/'

    lamdas, var_lamdas, xx, yy, rad, data, var, xx_flat, yy_flat, rad_flat, data_flat, var_flat, data_header = prepare_cubes.prepare_single_cube(data_filepath=data_filepath, gal_name=gal_name, z=z, cube_colour='red', results_folder=results_folder, data_corrections=True, data_crop=False, var_filepath=var_filepath, var_crop=False, var_corrections=True, lamda_crop=True)
    """

    """
    #J164905 red cube read-in
    data_filepath = '/Users/breichardtchu/Documents/data/J164905/J164905_red_binned_3_by_3.fits'
    var_filepath = '/Users/breichardtchu/Documents/data/J164905/J164905_red_var_binned_3_by_3.fits'
    gal_name = 'J164905_red'
    z = 0.032
    ssp_filepath = '/Users/breichardtchu/Documents/models/p_walcher09/*'
    results_folder = '/Users/breichardtchu/Documents/code_outputs/J164905_red_ppxf_26July2021_walcher09_deg6/'


    lamdas, var_lamdas, xx, yy, rad, data, var, xx_flat, yy_flat, rad_flat, data_flat, var_flat, data_header = prepare_cubes.prepare_single_cube(data_filepath=data_filepath, gal_name=gal_name, z=z, cube_colour='red', results_folder=results_folder, data_crop=False, var_filepath=var_filepath, var_crop=False, lamda_crop=False, mw_correction=True)
    """


    """
    #J164905 blue cube read-in
    data_filepath = '/Users/breichardtchu/Documents/data/J164905/J164905_blue_binned_3_by_3.fits'
    var_filepath = '/Users/breichardtchu/Documents/data/J164905/J164905_blue_var_binned_3_by_3.fits'
    gal_name = 'J164905_blue'
    z = 0.032
    ssp_filepath = '/Users/breichardtchu/Documents/models/bc03/templates/ssp_*'
    results_folder = '/Users/breichardtchu/Documents/code_outputs/J164905_blue_ppxf_14May2021/'


    lamdas, var_lamdas, xx, yy, rad, data, var, xx_flat, yy_flat, rad_flat, data_flat, var_flat, data_header = prepare_cubes.prepare_single_cube(data_filepath=data_filepath, gal_name=gal_name, z=z, cube_colour='blue', results_folder=results_folder, data_crop=False, var_filepath=var_filepath, var_crop=False, lamda_crop=False, mw_correction=True)
    """

    #J155636 red cube read-in
    data_filepath = '/Users/breichardtchu/Documents/data/J155636/J155636_red_binned_3_by_3.fits'
    var_filepath = '/Users/breichardtchu/Documents/data/J155636/J155636_red_var_binned_3_by_3.fits'
    gal_name = 'J155636_red'
    z = 0.035
    ssp_filepath = '/Users/breichardtchu/Documents/models/p_walcher09/*'
    results_folder = '/Users/breichardtchu/Documents/code_outputs/ppxf_J155636/J155636_red_ppxf_26July2021_walcher09_deg6/'


    lamdas, var_lamdas, xx, yy, rad, data, var, xx_flat, yy_flat, rad_flat, data_flat, var_flat, data_header = prepare_cubes.prepare_single_cube(data_filepath=data_filepath, gal_name=gal_name, z=z, cube_colour='red', results_folder=results_folder, data_crop=False, var_filepath=var_filepath, var_crop=False, lamda_crop=False, mw_correction=True)



    #check that the data array is finite everywhere
    assert np.all(np.isfinite(data_flat)), 'GALAXY must be finite'

    #run main without parallelising
    #main(lamdas, data_flat, noise_flat=np.sqrt(abs(var_flat)), xx_flat=xx_flat, yy_flat=yy_flat, ssp_filepath=ssp_filepath, z=z, results_folder=results_folder, galaxy_name=gal_name, cube_colour='red', fwhm_gal=1.7, fwhm_temp=1.0, em_lines=True, fwhm_emlines=3.5, degree=-1, mdegree=0, vacuum=True, plot=False, quiet=True, gas_reddening=None, reddening=0.13)

    #get rid of the spectra with a S/N < 3 so that there aren't any nan errors in the fitting
    #s_n = np.nanmedian(data_flat/np.sqrt(abs(var_flat)), axis=0)
    #sn_mask = s_n > 3
    #data_flat = data_flat[:, sn_mask]
    #var_flat = var_flat[:, sn_mask]
    #xx_flat = xx_flat[sn_mask]
    #yy_flat = yy_flat[sn_mask]
    #s_n = s_n[sn_mask]

    #save the masked data for later just in case
    #with open(results_folder+'/'+gal_name+'_flattened_masked','wb') as f:
    #    pickle.dump([lamdas, xx_flat, yy_flat, data_flat, var_flat, s_n], f)
    #f.close()

    #run main with parallelisation
    #BPASS templates have a fwhm_temp=1.0, cdelt_temp=1.0
    #bc03 templates have a fwhm_temp=3.0, cdelt_temp=1.0; using fwhm_temp=1.0 to make smoothing work
    #Walcher09 templates have a fwhm_temp=1.0, cdelt_temp=0.9
    #reddening=0.13
    main_parallelised(lamdas, data_flat, noise_flat=np.sqrt(abs(var_flat)), xx_flat=xx_flat, yy_flat=yy_flat, ssp_filepath=ssp_filepath, z=z, results_folder=results_folder, galaxy_name=gal_name, cube_colour='red', fwhm_gal=1.7, fwhm_temp=1.0, cdelt_temp=0.9, em_lines=True, fwhm_emlines=3.0, gas_reddening=None, reddening=None, degree=6, mdegree=0, vacuum=True, extra_em_lines=False, tie_balmer=True, plot=False, quiet=True)

    #header_file = '/fred/oz088/Duvet/nnielsen/IRAS08339/Combine/metacube.fits'
    #header_file = '/Users/breichardtchu/Documents/data/IRAS08_red_cubes/IRAS08339_metacube.fits'
    #header_file = '/Users/breichardtchu/Documents/data/IRAS08_blue_cubes/IRAS08_blue_combined.fits'
    #header_file = '/Users/breichardtchu/Documents/data/cgcg453_red_mosaic_binned_by_3.fits'
    #header_file = '/Users/breichardtchu/Documents/data/J164905/J164905_red_binned_3_by_3.fits'
    #header_file = '/Users/breichardtchu/Documents/data/J164905/J164905_blue_binned_3_by_3.fits'
    header_file = '/Users/breichardtchu/Documents/data/J155636/J155636_red_binned_3_by_3.fits'

    #shape=[1756, 67, 24] for red cubes, [2220, 67, 24] for blue cubes for IRAS08
    #[1756, 23, 35] for J164905
    #shape=[1755, 23, 36] for red cubes for CGCG453
    combine_results(lamdas=lamdas, data_flat=data_flat, final_shape=[lamdas.shape[0], 23, 36], results_folder=results_folder, galaxy_name=gal_name, header_file=header_file, unnormalised=True)
