"""
NAME:
	koffee.py
	KOFFEE - Keck Outflow Fitter For Emission linEs

AUTHOR:
	Bronwyn Reichardt Chu
	Swinburne
	2019

EMAIL:
	<breichardtchu@swin.edu.au>

PURPOSE:
	To fit gaussians to emission lines in 3D data cubes.
	Written on MacOS Mojave 10.14.5, with Python 3

MODIFICATION HISTORY:
		v.1.0 - first created May 2019
		v.1.0.1 - thinking about renaming this KOFFEE - Keck Outflow Fitter For Emission linEs
		v.1.0.2 - Added a continuum, being the average of the first 10 pixels in the input spectrum, which is then subtracted from the entire data spectrum so that the Gaussians fit properly, and aren't trying to fit the continuum as well (5th June 2019)
		v.1.0.3 - added a loop over the entire cube, with an exception if the paramaters object comes out weird and a progress bar
		v.1.0.4 - added a continuum to the mock data, and also added a feature so that the user can define what S/N they want the mock data to have
		v.1.0.5 - adding functions to combine data cubes either by pixel, or by regridding the wavelength (ToDo: include variance cubes in this too)

"""
import glob
import pickle
import pathlib
import numpy as np
from datetime import date
from tqdm import tqdm #progress bar module

#make sure matplotlib doesn't create any windows
#import matplotlib
#matplotlib.use('Agg')
import corner
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

#SpectRes is a spectrum resampling module which can be used to resample the fluxes and their uncertainties while preserving the integrated flux
#more information at https://spectres.readthedocs.io/en/latest/ or from the paper at https://arxiv.org/pdf/1705.05165.pdf
#from spectres import spectres

from astropy.modeling import models, fitting
from astropy.io import fits
from astropy.coordinates import SkyCoord, EarthLocation
from astropy import units
from astropy.time import Time
from astropy import constants as consts

from lmfit import Parameters
from lmfit import Model
from lmfit.models import GaussianModel, ConstantModel #, LinearModel, ConstantModel

#===============================================================================
#Test - create mock data
def mock_data(amp, mean, stddev, snr):
	"""
	Creates a mock data set with gaussians.  A list of values for each gaussian property is input; each property must have the same length.

	Args:
		amp: amplitude of the Gaussian (list of floats)
		mean: mean of the Gaussian (list of floats)
		stddev: standard deviation of the Gaussian (list of floats)
		snr: the desired signal-to-noise ratio
	Returns:
		x: the wavelength vector
		y: the flux/intensity
	"""
	np.random.seed(42)
	#create 'wavelengths'
	x = np.linspace(-40.,40.,800)
	#create flux
	gaussians = [0]*len(amp)
	for i in range(len(amp)):
		gaussians[i] = models.Gaussian1D(amp[i], mean[i], stddev[i])

	#add gaussians together
	g = 0
	for i in range(len(gaussians)):
		g += gaussians[i](x)

	#add noise assuming the mean value of the spectrum continuum is 1.0
	noise = 1.0/snr
	y = g + np.random.normal(0.,noise,x.shape)

	return x, y

def air_to_vac(wavelength):
	"""
	Implements the air to vacuum wavelength conversion described in eqn 64 and 65 of Greisen 2006.  The error in the index of refraction amounts to 1:10^9, which is less than the empirical formaula. Function slightly altered from specutils.utils.wcs_utils.

	Args:
		wavelength: the air wavelength(s) in Angstroms
	Returns:
		wavelength: the vacuum wavelength(s) in Angstroms
	"""
	#convert wavelength to um from Angstroms
	wlum = wavelength/10000
	#apply the equation from the paper
	return (1+1e-6*(287.6155+1.62887/wlum**2+0.01360/wlum**4)) * wavelength


def load_data(filename):
	"""
	Get the data from the fits file and correct to vacuum wavelengths and for the earth's rotation

	Args:
		filename: the path to the fits file
		redshift: the redshift of the galaxy
	Returns:
		lamdas: the corrected wavelength vector
		data: 3D array from the fits file
		header: the fits header
	"""
	#open the file and get the data
	with fits.open(filename) as hdu:
		data = hdu[0].data
		header = hdu[0].header
	hdu.close()

	#create the wavelength vector
	lamdas = np.arange(header['CRVAL3'], header['CRVAL3']+(header['NAXIS3']*header['CD3_3']), header['CD3_3'])

	#correct this from air to vacuum wavelengths
	#Greisen 2006 FITS Paper III (eqn 65)
	lamdas = air_to_vac(lamdas)

	#apply barycentric radial velocity corrections

	#keck = EarthLocation.of_site('Keck')
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

	return lamdas, data, header

#===============================================================================
#wavelengths of emission lines at rest in vacuum, taken from http://classic.sdss.org/dr6/algorithms/linestable.html
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

#dictionary of spaxels which are saturated in OIII_4 and need to fit OIII_3 instead
dodgy_spaxels = {
    "IRAS08" : [(27,9), (28,9), (29,9), (30,9), (31,9), (32,9)]
}


#===============================================================================
#combining data cubes using the entire load data routine so that they are corrected before being combined
def data_cubes_combine_by_pixel(filepath):
	"""
	Grabs datacubes and combines them by pixel using addition, finding the mean and the median.

	Args:
		filepath: the filepath string to pass to glob.glob

	Returns:
		lamdas: the wavelength vector for the cubes
		cube_added: all cubes added
		cube_mean: the mean of all the cubes
		cube_median: the median of all the cubes
	"""
	#create list to append datas to
	all_data = []
	all_lamdas = []

	#iterate through the filenames
	for file in glob.glob(filepath):
		lamdas, data, header = load_data(file)
		all_data.append(data)
		all_lamdas.append(lamdas)

	#because the exposures are so close together, the difference in lamda between the first to the last is only around 0.001A.  There's a difference in the total length of about 0.0003A between the longest and shortest wavelength vectors after the corrections.  So I'm averaging across the whole collection.  The difference is so small that it is negligible compared to our resolution
	"""
	start_lam = []
	end_lam = []
	len_lam = []
	for lam in all_lamdas:
		start_lam.append(lam[0])
		end_lam.append(lam[-1])
		len_lam.append(lam[-1]-lam[0])

	print(max(start_lam)-min(start_lam))
	print(max(end_lam)-min(end_lam))
	print(max(len_lam)-min(len_lam))
	"""
	lamdas = np.mean(all_lamdas, axis=0)

	#adding the data
	cube_added = np.zeros_like(all_data[0])

	for cube in all_data:
		cube_added += cube

	#finding the mean
	cube_mean = np.mean(all_data, axis=0)

	#finding the median
	cube_median = np.median(all_data, axis=0)

	#pickle the results
	with open(filepath.split('*')[0]+'combined_by_pixel_'+str(date.today()),'wb') as f:
		pickle.dump([lamdas, cube_added, cube_mean, cube_median], f)
	f.close()

	return lamdas, cube_added, cube_mean, cube_median


def data_cubes_combine_by_wavelength(filepath):
	"""
	Grabs datacubes and combines them by interpolating each spectrum in wavelength space and making sure to start and end at exactly the same wavelength for each spectrum before using addition, finding the mean and the median.

	Args:
		filepath: the filepath string to pass to glob.glob

	Returns:
		lamdas: the wavelength vector for the cubes
		cube_added: all cubes added
		cube_mean: the mean of all the cubes
		cube_median: the median of all the cubes
	"""
	#create list to append datas to
	all_data = []
	all_lamdas = []
	resampled_data = []

	#iterate through the filenames
	for file in glob.glob(filepath):
		lamdas, data, header = load_data(file)
		all_data.append(data)
		all_lamdas.append(lamdas)

	#because the exposures are so close together, the difference in starting lamda between the first to the last cube is only around 0.001A.  There's a difference in the total length of about 0.0003A between the longest and shortest wavelength vectors after the corrections.  So we interpolate along each spectrum and make sure they all start and end at the same spot.

	#take 50A off the beginning and end of the spectrum, this area tends to be weird anyway and create the new wavelength vector
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
	with open(filepath.split('*')[0]+'combined_by_wavelength_'+str(date.today()),'wb') as f:
		pickle.dump([lamdas, cube_added, cube_mean, cube_median], f)
	f.close()

	return new_lamda, cube_added, cube_mean, cube_median


#===============================================================================
#gaussian function
def gaussian_func(x, amp, mean, sigma):
	"""
	Defines the 1D gaussian function.

	Args:
		x: the x-values or wavelength vector
		amp: amplitude or height of the gaussian
		mean: central point of the gaussian
		sigma: the standard deviation of the gaussian

	Returns:
		gauss: the gaussian function
	"""
	gaussian = amp * np.exp(-(x-mean)**2/(2*sigma**2))

	return gaussian


#fitting functions
def gaussian1(wavelength, flux, amp_guess=None, mean_guess=None, sigma_guess=None):
	"""
	Creates a single gaussian to fit to the emission line

	Args:
		wavelength: the wavelength vector
		flux: the flux of the spectrum
		amp_guess: guess for the overall height of the peak (default = None)
		mean_guess: guess for the central position of the peak, usually the observed wavelength of the emission line (default = None)
		sigma_guess: guess for the characteristic width of the peak (default = None)

	Returns:
		g_model: the Gaussian model
		pars: the Parameters object
	"""
	#create models
	#g_model = GaussianModel()
	g_model = Model(gaussian_func, prefix='gauss_')

	#create parameters object
	#pars = g_model.guess(flux, x=wavelength)
	pars = g_model.make_params()

	#update parameters object
	if amp_guess is not None:
		pars['gauss_amp'].set(value=amp_guess[0])
	if amp_guess is None:
		pars['gauss_amp'].set(value=max(flux))

	if mean_guess is not None:
		pars['gauss_mean'].set(value=mean_guess[0])
	if mean_guess is None:
		pars['gauss_mean'].set(value=wavelength[(flux==max(flux))][0])

	if sigma_guess is not None:
		pars['gauss_sigma'].set(value=sigma_guess[0])
	if sigma_guess is None:
		pars['gauss_sigma'].set(value=0.9)

	#set the range of wavelengths the mean can be within to be within 10A of the observed wavelength of the emission line, if this is given:
	if mean_guess is not None:
		pars['gauss_mean'].set(max=mean_guess+10.0, min=mean_guess-10.0)

	#set the sigma to have a minimum of 1.0A and a maximum of 1.7A
	pars['gauss_sigma'].set(min=0.8, max=2.0)


	return g_model, pars



def gaussian2(wavelength, flux, amplitude_guess=None, mean_guess=None, sigma_guess=None, mean_diff=None, sigma_variations=None):
    """
    Creates a combination of 2 gaussians

    Parameters
    ----------
    wavelength :
        the wavelength vector
    flux :
        the flux of the spectrum
    amplitude_guess :
        guesses for the amplitudes of the two gaussians (default = None)
    mean_guess :
        guesses for the central positions of the two gaussians (default = None)
    sigma_guess :
        guesses for the characteristic widths of the two gaussians (default = None)
    mean_diff :
        guess for the difference between the means, and the amount by which
        that can change eg. [2.5Angstroms, 0.5Angstroms] (default = None)
    sigma_variations :
        the amount by which we allow sigma to vary from the guess e.g. 0.5Angstroms (default=None)

    Returns
    -------
    g_model :
        the double Gaussian model
    pars :
        the Parameters object
    """
    #create the first gaussian
    #gauss1 = GaussianModel(prefix='Galaxy_')
    gauss1 = Model(gaussian_func, prefix='Galaxy_')

    #create parameters object
    #pars = gauss1.guess(flux, x=wavelength)
    pars = gauss1.make_params()

    #create the second gaussian
    #gauss2 = GaussianModel(prefix='Flow_')
    gauss2 = Model(gaussian_func, prefix='Flow_')

    #update the Parameters object to include variables from the second gaussian
    pars.update(gauss2.make_params())

    #combine the two gaussians for the model
    g_model = gauss1 + gauss2

    #update the Parameters object so that there are bounds on the gaussians
    #update parameters object
    if amplitude_guess is not None:
        pars['Galaxy_amp'].set(value=amplitude_guess[0], vary=True)
        pars['Flow_amp'].set(value=amplitude_guess[1], vary=True)
    elif amplitude_guess is None:
        pars['Galaxy_amp'].set(value=0.7*max(flux)/0.4)
        pars['Flow_amp'].set(value=0.3*max(flux)/0.4)

    #no negative gaussians
    pars['Flow_amp'].set(min=0.01)
    #we also want the galaxy amplitude to be greater than the flow amplitude, so we define a new parameter
    #amp_diff = Galaxy_amplitude-Flow_amplitude, where amp_diff > 0.05
    pars.add('amp_diff', value=0.1, min=0.05, vary=True)
    pars['Galaxy_amp'].set(expr='amp_diff+Flow_amp')

    #if the peak is obvious, use it as the first guess for the mean... if it is not obvious, use the observed emission line wavelength (if given)
    if mean_guess is not None:
        pars['Galaxy_mean'].set(value=mean_guess[0])
        pars['Flow_mean'].set(value=mean_guess[1])
    if mean_guess is None:
        pars['Galaxy_mean'].set(value=wavelength[np.argmax(flux)])
        pars['Flow_mean'].set(value=wavelength[np.argmax(flux)]-0.1)

    #The center of the galaxy gaussian needs to be within the wavelength range, or near where we expect the emission line to be
    if mean_guess is not None:
        pars['Galaxy_mean'].set(max=mean_guess[0]+10, min=mean_guess[0]-10, vary=True)
    if mean_guess is None:
        pars['Galaxy_mean'].set(max=wavelength[-5], min=wavelength[5], vary=True)

    #The flow gaussian should also be within 5 Angstroms of the galaxy gaussian... so, we define a new parameter, lam_diff =Galaxy_mean-Flow_mean, where -5 < lam_diff < 5
    if mean_diff is not None:
        #pars.add('lam_diff', value=mean_diff[0], max=(mean_diff[0]+mean_diff[0]*mean_diff[1]), min=(mean_diff[0]-mean_diff[0]*mean_diff[1]), vary=True)
        pars.add('lam_diff', value=mean_diff[0], max=(mean_diff[0]+mean_diff[1]), min=(mean_diff[0]-mean_diff[1]), vary=True)
    if mean_diff is None:
        pars.add('lam_diff', value=0.0, max=5.0, min=-5.0, vary=True)
    #pars.add('lam_diff', value=0.0, max=3.0, min=-3.0)
    pars['Flow_mean'].set(expr='Galaxy_mean-lam_diff')

    if sigma_guess is not None:
        pars['Galaxy_sigma'].set(value=sigma_guess[0])
        pars['Flow_sigma'].set(value=sigma_guess[1])
    if sigma_guess is None:
        pars['Galaxy_sigma'].set(value=1.0)
        pars['Flow_sigma'].set(value=3.5)

    if sigma_variations is not None:
        #pars['Galaxy_sigma'].set(max=(sigma_guess[0]+sigma_guess[0]*sigma_variations), min=(sigma_guess[0]-sigma_guess[0]*sigma_variations), vary=True)
        #pars['Flow_sigma'].set(max=(sigma_guess[1]+sigma_guess[1]*sigma_variations), min=(sigma_guess[1]-sigma_guess[1]*sigma_variations), vary=True)
        pars['Galaxy_sigma'].set(max=(sigma_guess[0]+sigma_variations), min=(sigma_guess[0]-sigma_variations), vary=True)
        pars['Flow_sigma'].set(max=(sigma_guess[1]+sigma_variations), min=(sigma_guess[1]-sigma_variations), vary=True)
    if sigma_variations is None:
        #no super duper wide outflows
        pars['Flow_sigma'].set(max=10.0, min=0.8, vary=True)
        #edited this to fit Halpha... remember to change back!!!
        #pars['Flow_sigma'].set(max=3.5, min=0.8, vary=True)
        #also, since each wavelength value is roughly 0.5A apart, the sigma must be more than 0.25A
        pars['Galaxy_sigma'].set(min=0.9, max=2.0, vary=True)#min=2.0 because that's the minimum we can observe with the telescope


    return g_model, pars




def gaussian1_const(wavelength, flux, amp_guess=None, mean_guess=None, sigma_guess=None):
    """
    Creates a single gaussian to fit to the emission line

    Args:
    	wavelength: the wavelength vector
    	flux: the flux of the spectrum
    	amp_guess: guess for the overall height of the peak (default = None)
    	mean_guess: guess for the central position of the peak, usually the observed wavelength of the emission line (default = None)
    	sigma_guess: guess for the characteristic width of the peak (default = None)

    Returns:
    	g_model: the Gaussian model
    	pars: the Parameters object
    """
    #create models
    #g_model = GaussianModel()
    g_model = Model(gaussian_func, prefix='gauss_')

    #create parameters object
    #pars = g_model.guess(flux, x=wavelength)
    pars = g_model.make_params()

    #create the constant
    const = ConstantModel(prefix='Constant_Continuum_')
    #update the parameters object
    pars.update(const.make_params())

    #combine the two gaussians for the model
    g_model = g_model + const

    #give the constant a starting point at zero
    pars['Constant_Continuum_c'].set(value=0.0, min=-0.5*max(flux), max=0.5*max(flux), vary=True)

    #update parameters object for gaussian
    if amp_guess is not None:
        pars['gauss_amp'].set(value=amp_guess[0])
    if amp_guess is None:
        pars['gauss_amp'].set(value=max(flux))

    if mean_guess is not None:
        pars['gauss_mean'].set(value=mean_guess[0])
    if mean_guess is None:
        pars['gauss_mean'].set(value=wavelength[(flux==max(flux))][0])

    if sigma_guess is not None:
        pars['gauss_sigma'].set(value=sigma_guess[0])
    if sigma_guess is None:
        pars['gauss_sigma'].set(value=0.9)

    #set the range of wavelengths the mean can be within to be within 10A of the observed wavelength of the emission line, if this is given:
    if mean_guess is not None:
        pars['gauss_mean'].set(max=mean_guess+10.0, min=mean_guess-10.0)

    #set the sigma to have a minimum of 1.0A and a maximum of 1.7A
    pars['gauss_sigma'].set(min=0.8, max=2.0)


    return g_model, pars



def gaussian2_const(wavelength, flux, amplitude_guess=None, mean_guess=None, sigma_guess=None, mean_diff=None, sigma_variations=None):
    """
    Creates a combination of 2 gaussians

    Parameters
    ----------
    wavelength :
        the wavelength vector
    flux :
        the flux of the spectrum
    amplitude_guess :
        guesses for the amplitudes of the two gaussians (default = None)
    mean_guess :
        guesses for the central positions of the two gaussians (default = None)
    sigma_guess :
        guesses for the characteristic widths of the two gaussians (default = None)
    mean_diff :
        guess for the difference between the means, and the amount by which
        that can change eg. [2.5Angstroms, 0.5Angstroms] (default = None)
    sigma_variations :
        the amount by which we allow sigma to vary from the guess e.g. 0.5Angstroms (default=None)

    Returns
    -------
    g_model :
        the double Gaussian model
    pars :
        the Parameters object
    """
    #create the first gaussian
    #gauss1 = GaussianModel(prefix='Galaxy_')
    gauss1 = Model(gaussian_func, prefix='Galaxy_')

    #create parameters object
    #pars = gauss1.guess(flux, x=wavelength)
    pars = gauss1.make_params()

    #create the second gaussian
    #gauss2 = GaussianModel(prefix='Flow_')
    gauss2 = Model(gaussian_func, prefix='Flow_')

    #update the Parameters object to include variables from the second gaussian
    pars.update(gauss2.make_params())

    #create the constant
    const = ConstantModel(prefix='Constant_Continuum_')
    pars.update(const.make_params())

    #combine the two gaussians for the model
    g_model = gauss1 + gauss2 + const

    #give the constant a starting point at zero
    pars['Constant_Continuum_c'].set(value=0.0, min=-0.5*max(flux), max=0.5*max(flux), vary=True)

    #update the Parameters object so that there are bounds on the gaussians
    #update parameters object
    if amplitude_guess is not None:
        pars['Galaxy_amp'].set(value=amplitude_guess[0], vary=True)
        pars['Flow_amp'].set(value=amplitude_guess[1], vary=True)
    elif amplitude_guess is None:
        pars['Galaxy_amp'].set(value=0.7*max(flux)/0.4)
        pars['Flow_amp'].set(value=0.3*max(flux)/0.4)

    #no negative gaussians
    pars['Flow_amp'].set(min=0.01)
    #we also want the galaxy amplitude to be greater than the flow amplitude, so we define a new parameter
    #amp_diff = Galaxy_amplitude-Flow_amplitude, where amp_diff > 0.05
    pars.add('amp_diff', value=0.1, min=0.05, vary=True)
    pars['Galaxy_amp'].set(expr='amp_diff+Flow_amp')

    #if the peak is obvious, use it as the first guess for the mean... if it is not obvious, use the observed emission line wavelength (if given)
    if mean_guess is not None:
        pars['Galaxy_mean'].set(value=mean_guess[0])
        pars['Flow_mean'].set(value=mean_guess[1])
    if mean_guess is None:
        pars['Galaxy_mean'].set(value=wavelength[np.argmax(flux)])
        pars['Flow_mean'].set(value=wavelength[np.argmax(flux)]-0.1)

    #The center of the galaxy gaussian needs to be within the wavelength range, or near where we expect the emission line to be
    if mean_guess is not None:
        pars['Galaxy_mean'].set(max=mean_guess[0]+10, min=mean_guess[0]-10, vary=True)
    if mean_guess is None:
        pars['Galaxy_mean'].set(max=wavelength[-5], min=wavelength[5], vary=True)

    #The flow gaussian should also be within 5 Angstroms of the galaxy gaussian... so, we define a new parameter, lam_diff =Galaxy_mean-Flow_mean, where -5 < lam_diff < 5
    if mean_diff is not None:
        #pars.add('lam_diff', value=mean_diff[0], max=(mean_diff[0]+mean_diff[0]*mean_diff[1]), min=(mean_diff[0]-mean_diff[0]*mean_diff[1]), vary=True)
        pars.add('lam_diff', value=mean_diff[0], max=(mean_diff[0]+mean_diff[1]), min=(mean_diff[0]-mean_diff[1]), vary=True)
    if mean_diff is None:
        pars.add('lam_diff', value=0.0, max=5.0, min=-5.0, vary=True)
    #pars.add('lam_diff', value=0.0, max=3.0, min=-3.0)
    pars['Flow_mean'].set(expr='Galaxy_mean-lam_diff')

    if sigma_guess is not None:
        pars['Galaxy_sigma'].set(value=sigma_guess[0])
        pars['Flow_sigma'].set(value=sigma_guess[1])
    if sigma_guess is None:
        pars['Galaxy_sigma'].set(value=1.0)
        pars['Flow_sigma'].set(value=3.5)

    if sigma_variations is not None:
        #pars['Galaxy_sigma'].set(max=(sigma_guess[0]+sigma_guess[0]*sigma_variations), min=(sigma_guess[0]-sigma_guess[0]*sigma_variations), vary=True)
        #pars['Flow_sigma'].set(max=(sigma_guess[1]+sigma_guess[1]*sigma_variations), min=(sigma_guess[1]-sigma_guess[1]*sigma_variations), vary=True)
        pars['Galaxy_sigma'].set(max=(sigma_guess[0]+sigma_variations), min=(sigma_guess[0]-sigma_variations), vary=True)
        pars['Flow_sigma'].set(max=(sigma_guess[1]+sigma_variations), min=(sigma_guess[1]-sigma_variations), vary=True)
    if sigma_variations is None:
        #no super duper wide outflows
        pars['Flow_sigma'].set(max=10.0, min=0.8, vary=True)
        #edited this to fit Halpha... remember to change back!!!
        #pars['Flow_sigma'].set(max=3.5, min=0.8, vary=True)
        #also, since each wavelength value is roughly 0.5A apart, the sigma must be more than 0.25A
        pars['Galaxy_sigma'].set(min=0.9, max=2.0, vary=True)#min=2.0 because that's the minimum we can observe with the telescope

    return g_model, pars


def gaussian1_OII_doublet(wavelength, flux, amplitude_guess=None, mean_guess=None, sigma_guess=None, sigma_variations=None):
    """
    Creates a combination of 2 gaussians and a constant to fit the OII doublet

    Parameters
    ----------
    wavelength :
        the wavelength vector
    flux :
        the flux of the spectrum
    amplitude_guess :
        guesses for the amplitudes of the two gaussians (default = None)
    sigma_guess :
        guesses for the characteristic widths of the two gaussians (default = None)
    sigma_variations :
        the amount by which we allow sigma to vary from the guess e.g. 0.5Angstroms (default=None)

    Returns
    -------
    g_model :
        the double Gaussian model
    pars :
        the Parameters object
    """
    #create the first gaussian
    gauss1 = Model(gaussian_func, prefix='Galaxy_red_')

    #create parameters object
    pars = gauss1.make_params()

    #create the second gaussian
    gauss2 = Model(gaussian_func, prefix='Galaxy_blue_')

    #create parameters object
    pars.update(gauss2.make_params())

    #create the constant
    const = ConstantModel(prefix='Constant_Continuum_')
    pars.update(const.make_params())

    #combine the two gaussians for the model
    g_model = gauss1 + gauss2 + const

    #give the constant a starting point at zero
    pars['Constant_Continuum_c'].set(value=0.0, min=-0.5*max(flux), max=0.5*max(flux), vary=True)

    #update the Parameters object so that there are bounds on the gaussians
    #the amplitudes need to be tied such that the ratio of 3729/3726 is between 0.8 and 1.4
    if amplitude_guess is not None:
        pars['Galaxy_red_amp'].set(value=amplitude_guess[0], vary=True)
    elif amplitude_guess is None:
        pars['Galaxy_red_amp'].set(value=0.8*max(flux)/0.4)

    pars.add('gal_amp_ratio', value=1.0, min=0.7, max=1.4, vary=True)
    pars['Galaxy_blue_amp'].set(expr='Galaxy_red_amp/gal_amp_ratio')

    #no negative gaussians
    pars['Galaxy_red_amp'].set(min=0.01)
    pars['Galaxy_blue_amp'].set(min=0.01)

    #use the maximum as the first guess for the mean
    #the second peak is 3Angstroms away from the first
    #The center of the galaxy gaussian needs to be within the wavelength range
    if mean_guess is not None:
        pars['Galaxy_red_mean'].set(value=mean_guess[0], max=wavelength[-5], min=wavelength[5], vary=True)
    if mean_guess is None:
        pars['Galaxy_red_mean'].set(value=wavelength[np.argmax(flux)], max=wavelength[-5], min=wavelength[5], vary=True)


    #gal_lam_diff = Galaxy_red_mean - Galaxy_blue_mean, where this is set by the expected wavelengths
    #since [OII] 3726.032, 3728.815 ... difference has to be 2.783
    pars.add('gal_lam_diff', value=2.783, vary=False)

    pars['Galaxy_blue_mean'].set(expr='Galaxy_red_mean-gal_lam_diff')


    #use the sigma found from previous fits to define the sigma of the outflows
    if sigma_guess is not None:
        pars['Galaxy_red_sigma'].set(value=sigma_guess[0])
    if sigma_guess is None:
        pars['Galaxy_red_sigma'].set(value=1.0)

    #tie the blue gaussians to the red ones for sigma
    pars['Galaxy_blue_sigma'].set(expr='Galaxy_red_sigma')

    if sigma_variations is not None:
        if sigma_guess[0]-sigma_variations > 0.8:
            pars['Galaxy_red_sigma'].set(max=(sigma_guess[0]+sigma_variations), min=(sigma_guess[0]-sigma_variations), vary=True)
        else:
            pars['Galaxy_red_sigma'].set(max=(sigma_guess[0]+sigma_variations), min=0.8, vary=True)

    if sigma_variations is None:
        pars['Galaxy_red_sigma'].set(max=2.0, min=0.8, vary=True)#min=0.8 because that's the minimum we can observe with the telescope

    return g_model, pars


def gaussian2_OII_doublet(wavelength, flux, amplitude_guess=None, mean_guess=None, sigma_guess=None, mean_diff=None, sigma_variations=None):
    """
    Creates a combination of 4 gaussians and a constant to fit the OII doublet

    Parameters
    ----------
    wavelength :
        the wavelength vector
    flux :
        the flux of the spectrum
    amplitude_guess :
        guesses for the amplitudes of the two gaussians (default = None)
    sigma_guess :
        guesses for the characteristic widths of the two gaussians (default = None)
    mean_diff :
        guess for the difference between the means, and the amount by which
        that can change eg. [2.5Angstroms, 0.5Angstroms] (default = None)
    sigma_variations :
        the amount by which we allow sigma to vary from the guess e.g. 0.5Angstroms (default=None)

    Returns
    -------
    g_model :
        the double Gaussian model
    pars :
        the Parameters object
    """
    #create the first gaussian
    gauss1 = Model(gaussian_func, prefix='Galaxy_red_')

    #create parameters object
    pars = gauss1.make_params()

    #create the second gaussian
    gauss2 = Model(gaussian_func, prefix='Flow_red_')

    #update the Parameters object to include variables from the second gaussian
    pars.update(gauss2.make_params())

    #create the third gaussian
    gauss3 = Model(gaussian_func, prefix='Galaxy_blue_')

    #create parameters object
    pars.update(gauss3.make_params())

    #create the second gaussian
    gauss4 = Model(gaussian_func, prefix='Flow_blue_')

    #update the Parameters object to include variables from the second gaussian
    pars.update(gauss4.make_params())

    #create the constant
    const = ConstantModel(prefix='Constant_Continuum_')
    pars.update(const.make_params())

    #combine the two gaussians for the model
    g_model = gauss1 + gauss2 + gauss3 + gauss4 + const

    #give the constant a starting point at zero
    pars['Constant_Continuum_c'].set(value=0.0, min=-0.5*max(flux), max=0.5*max(flux), vary=True)

    #update the Parameters object so that there are bounds on the gaussians
    #the amplitudes need to be tied such that the ratio of 3729/3726 is between 0.8 and 1.4
    if amplitude_guess is not None:
        pars['Galaxy_red_amp'].set(value=amplitude_guess[0], vary=True)
        pars['Flow_red_amp'].set(value=amplitude_guess[1], vary=True)
    elif amplitude_guess is None:
        pars['Galaxy_red_amp'].set(value=0.7*max(flux)/0.4)
        pars['Flow_red_amp'].set(value=0.3*max(flux)/0.4)

    #we also want the galaxy amplitude to be greater than the flow amplitude, so we define a new parameter
    #amp_diff = Galaxy_amplitude-Flow_amplitude, where amp_diff > 0.05
    pars.add('amp_diff_red', value=0.1, min=0.05, vary=True)
    pars.add('amp_diff_blue', value=0.1, min=0.05, vary=True)
    pars['Galaxy_red_amp'].set(expr='amp_diff_red+Flow_red_amp')
    pars['Galaxy_blue_amp'].set(expr='amp_diff_blue+Flow_blue_amp')

    pars.add('gal_amp_ratio', value=1.0, min=0.7, max=1.4, vary=True)
    #pars['Galaxy_blue_amp'].set(expr='Galaxy_red_amp/gal_amp_ratio')
    pars['gal_amp_ratio'].set(expr='Galaxy_red_amp/Galaxy_blue_amp')

    pars.add('flow_amp_ratio', value=1.0, min=0.7, max=1.4, vary=True)
    pars['Flow_blue_amp'].set(expr='Flow_red_amp/flow_amp_ratio')

    #no negative gaussians
    pars['Flow_red_amp'].set(min=0.01)#, max='Galaxy_red_amp')
    pars['Flow_blue_amp'].set(min=0.01)#, max='Galaxy_blue_amp')
    pars['Galaxy_red_amp'].set(min=0.01)
    pars['Galaxy_blue_amp'].set(min=0.01)



    #use the maximum as the first guess for the mean
    #the second peak is 3Angstroms away from the first
    #The center of the galaxy gaussian needs to be within the wavelength range
    if mean_guess is not None:
        pars['Galaxy_red_mean'].set(value=mean_guess[0], max=wavelength[-5], min=wavelength[5], vary=True)
    if mean_guess is None:
        pars['Galaxy_red_mean'].set(value=wavelength[np.argmax(flux)], max=wavelength[-5], min=wavelength[5], vary=True)

    #The flow gaussian mean is defined by the previous fits...
    #so, we define some new parameters:
    #flow_lam_diff = Galaxy_mean-Flow_mean, where lam_diff is set by the previous fits
    if mean_diff is not None:
        pars.add('flow_lam_diff', value=mean_diff[0], max=(mean_diff[0]+mean_diff[1]), min=(mean_diff[0]-mean_diff[1]), vary=True)
    if mean_diff is None:
        pars.add('flow_lam_diff', value=0.0, max=5.0, min=-5.0, vary=True)

    #gal_lam_diff = Galaxy_red_mean - Galaxy_blue_mean, where this is set by the expected wavelengths
    #since [OII] 3726.032, 3728.815 ... difference has to be 2.783
    pars.add('gal_lam_diff', value=2.783, vary=False)

    pars['Galaxy_blue_mean'].set(expr='Galaxy_red_mean-gal_lam_diff')
    pars['Flow_red_mean'].set(expr='Galaxy_red_mean-flow_lam_diff')
    pars['Flow_blue_mean'].set(expr='Galaxy_blue_mean-flow_lam_diff')

    #use the sigma found from previous fits to define the sigma of the outflows
    if sigma_guess is not None:
        pars['Galaxy_red_sigma'].set(value=sigma_guess[0])
        pars['Flow_red_sigma'].set(value=sigma_guess[1])
    if sigma_guess is None:
        pars['Galaxy_red_sigma'].set(value=1.0)
        pars['Flow_red_sigma'].set(value=3.5)

    #tie the blue gaussians to the red ones for sigma
    pars['Galaxy_blue_sigma'].set(expr='Galaxy_red_sigma')
    pars['Flow_blue_sigma'].set(expr='Flow_red_sigma')

    if sigma_variations is not None:
        if sigma_guess[0]-sigma_variations > 0.8:
            pars['Galaxy_red_sigma'].set(max=(sigma_guess[0]+sigma_variations), min=(sigma_guess[0]-sigma_variations), vary=True)
        else:
            pars['Galaxy_red_sigma'].set(max=(sigma_guess[0]+sigma_variations), min=0.8, vary=True)
        if sigma_guess[1]-sigma_variations > 0.8:
            pars['Flow_red_sigma'].set(max=(sigma_guess[1]+sigma_variations), min=(sigma_guess[1]-sigma_variations), vary=True)
        else:
            pars['Flow_red_sigma'].set(max=(sigma_guess[1]+sigma_variations), min=0.8, vary=True)
    if sigma_variations is None:
        #no super duper wide outflows
        pars['Flow_red_sigma'].set(max=10.0, min=0.8, vary=True)
        pars['Galaxy_red_sigma'].set(max=2.0, min=0.8, vary=True)#min=0.8 because that's the minimum we can observe with the telescope

    return g_model, pars




def gaussian_NaD(wavelength, flux, amplitude_guess=None, mean_guess=None, sigma_guess=None):
    """
    Creates a combination of 4 gaussians which fit the NaD absorption line.  Order: galaxy1, galaxy2, flow1, flow2

    Args:
        wavelength: the wavelength vector
        flux: the flux of the spectrum
        amplitude_guess: guesses for the amplitudes of the four gaussians (default = None)
        mean_guess: guesses for the central positions of the four gaussians (default = None)
        sigma_guess: guesses for the characteristic widths of the four gaussians (default = None)

    Returns:
        g_model: the double Gaussian model
        pars: the Parameters object
    """
    #create the first galaxy gaussian
    gauss1 = Model(gaussian_func, prefix='Galaxy1_')

    #create parameters object
    pars = gauss1.make_params()

    #create the second galaxy gaussian
    gauss2 = Model(gaussian_func, prefix='Galaxy2_')

    #create parameters object
    pars.update(gauss2.make_params())

    #create the first flow gaussian
    gauss3 = Model(gaussian_func, prefix='Flow1_')

    #update the Parameters object to include variables from the second gaussian
    pars.update(gauss3.make_params())

    #create the second flow gaussian
    gauss4 = Model(gaussian_func, prefix='Flow2_')

    #update the Parameters object to include variables from the second gaussian
    pars.update(gauss4.make_params())

    #combine the two gaussians for the model
    g_model = gauss1 + gauss2 + gauss3 + gauss4

    #update the Parameters object so that there are bounds on the gaussians
    #update parameters object
    if amplitude_guess is not None:
        pars['Galaxy1_amp'].set(value=amplitude_guess[0], vary=True)
        pars['Galaxy2_amp'].set(value=amplitude_guess[1], vary=True)
        pars['Flow1_amp'].set(value=amplitude_guess[2], vary=True)
        pars['Flow2_amp'].set(value=amplitude_guess[3], vary=True)
    elif amplitude_guess is None:
        pars['Galaxy1_amp'].set(value=0.7*min(flux)/0.4)
        pars['Galaxy2_amp'].set(value=0.7*min(flux)/0.4)
        pars['Flow1_amp'].set(value=0.3*min(flux)/0.4)
        pars['Flow2_amp'].set(value=0.3*min(flux)/0.4)

    #no positive gaussians
    pars['Flow1_amp'].set(max=-0.01)
    pars['Flow2_amp'].set(max=-0.01)
    pars['Galaxy1_amp'].set(max=-0.01, min=min(flux))
    #pars['Galaxy2_amp'].set(max=-0.01, min=min(flux))
    pars['Galaxy2_amp'].set(max=-0.01, min=min(flux)+50)
    #we also want the galaxy amplitude to be less than the flow amplitude (a greater negative value), so we define a new parameter
    #amp_diff = Flow_amplitude-Galaxy_amplitude, where amp_diff > 0.05
    pars.add('amp_diff1', value=10, min=0.05, max=abs(min(flux))-0.05, vary=True)
    pars['Flow1_amp'].set(expr='amp_diff1+Galaxy1_amp')
    pars.add('amp_diff2', value=10, min=0.05, max=abs(min(flux))-0.05, vary=True)
    pars['Flow2_amp'].set(expr='amp_diff2+Galaxy2_amp')

    #if the peak is obvious, use it as the first guess for the mean... if it is not obvious, use the observed emission line wavelength (if given)
    if mean_guess is not None:
        #if the galaxy mean is given, it should be decided by another emission line fit and not change
        pars['Galaxy1_mean'].set(value=mean_guess[0])
        pars['Galaxy2_mean'].set(value=mean_guess[1])
        pars['Flow1_mean'].set(value=mean_guess[2])
        pars['Flow2_mean'].set(value=mean_guess[3])
    if mean_guess is None:
        pars['Galaxy1_mean'].set(value=wavelength[np.argmin(flux)])
        #the first line is usually stronger; and the lines are 6A apart
        pars['Galaxy2_mean'].set(value=wavelength[np.argmin(flux)]+6.0)
        pars['Flow1_mean'].set(value=wavelength[np.argmin(flux)]-0.1)
        pars['Flow2_mean'].set(value=wavelength[np.argmin(flux)]+5.9)

    #The center of the galaxy gaussian needs to be within the wavelength range, or near where we expect the emission line to be
    #if mean_guess is not None:
    #    pars['Galaxy1_mean'].set(max=mean_guess[0]+10, min=mean_guess[0]-10, vary=True)
    #    pars['Galaxy2_mean'].set(max=mean_guess[1]+10, min=mean_guess[1]-10, vary=True)
    if mean_guess is None:
        #pars['Galaxy1_mean'].set(max=wavelength[-5], min=wavelength[5], vary=True)
        #pars['Galaxy2_mean'].set(max=wavelength[-5], min=wavelength[5], vary=True)
        pars['Galaxy1_mean'].set(max=wavelength[np.argmin(flux)]+2.0, min=wavelength[np.argmin(flux)]-2.0, vary=True)
        pars['Galaxy2_mean'].set(max=wavelength[np.argmin(flux)]+8.0, min=wavelength[np.argmin(flux)]+4.0, vary=True)

    #The flow gaussian should also be within 5 Angstroms of the galaxy gaussian... so, we define a new parameter:
    #lam_diff = Galaxy_mean-Flow_mean, where -5 < lam_diff < 5
    #lam_diff has to be the same for both absorption line sets
    pars.add('lam_diff', value=1.0, max=5.0, min=-5.0)
    #pars.add('lam_diff2', value=1.0, max=5.0, min=-5.0)
    #pars.add('lam_diff', value=0.0, max=3.0, min=-3.0)
    pars['Flow1_mean'].set(expr='Galaxy1_mean-lam_diff')
    pars['Flow2_mean'].set(expr='Galaxy2_mean-lam_diff')

    if sigma_guess is not None:
        #if we have an input for sigma, then set the galaxy sigma not to vary
        pars['Galaxy1_sigma'].set(value=sigma_guess[0], vary=False)
        pars['Galaxy2_sigma'].set(value=sigma_guess[1], vary=False)
        pars['Flow1_sigma'].set(value=sigma_guess[2])
        pars['Flow2_sigma'].set(value=sigma_guess[3])
    if sigma_guess is None:
        #since each wavelength value is roughly 1.25A apart, the sigma must be more than 1.0A
        pars['Galaxy1_sigma'].set(value=1.0, min=1.0, max=3.0, vary=True)
        pars['Galaxy2_sigma'].set(value=1.0, min=1.0, max=3.0, vary=True)
        pars['Flow1_sigma'].set(value=3.5)
        pars['Flow2_sigma'].set(value=3.5)

    #no super duper wide outflows
    pars['Flow1_sigma'].set(max=6.0, min=0.8, vary=True)
    pars['Flow2_sigma'].set(expr='Flow1_sigma')
    #pars['Flow2_sigma'].set(max=5.0, min=0.8, vary=True)

    return g_model, pars



def fitter(g_model, pars, wavelength, flux, method='leastsq', verbose=True):
	"""
	Fits the model to the data using a Levenberg-Marquardt least squares method by default (lmfit generally uses the scipy.optimize or scipy.optimize.minimize methods).  Prints the fit report, containing fit statistics and best-fit values with uncertainties and correlations.

	Args:
		g_model: the model to fit to the data
		pars: Parameters object containing the variables and their constraints
		wavelength: wavelength vector
		flux: the flux of the spectrum
		method: the fitting method to use, for example:
			- 'leastsq' - Levenberg-Marquardt least squares method (default)
			- 'emcee' - Maximum likelihood via Monte-Carlo Markov Chain
            - 'dual_annealing'
			- see https://lmfit.github.io/lmfit-py/fitting.html for more options
		verbose: whether to print out results and statistics.  Default is true.
	Returns:
		best_fit: the best fitting model (class)
	"""
	#fit the data
	best_fit = g_model.fit(flux, pars, x=wavelength, method=method)

	#print out the fit report
	if verbose:
		print(best_fit.fit_report())

	return best_fit

#===============================================================================
#making KOFFEE smarter
def check_blue_chi_square(wavelength, flux, best_fit, g_model, OII_doublet_fit=False):
    """
    Checks the chi squared value of the blue side of the fit.  If there's a large residual, KOFFEE will shift the
    starting point for the flow gaussian mean to the blue in fit_cube().

    Parameters
    ----------
    wavelength :
        the wavelength vector
    flux :
        the data vector
    best_fit :
        the best_fit object
    g_model :
        the gaussian model object
    OII_doublet_fit : boolean
        whether it was a fit to the OII doublet or not (default = False)

    Returns
    -------
    chi_square : int
        the chi squared residual of the blue side of the fit

    """
    #get the residuals
    residual = best_fit.residual

    #we only want them for the area which is between 1 sigma to the blue of the systemic mean, and 5A blue of that.
    #create the wavelength mask
    if str(type(g_model)) == "<class 'lmfit.model.Model'>":
        one_sigma_blue = (best_fit.best_values['gauss_mean'] - best_fit.best_values['gauss_sigma'])-1.0

    if str(type(g_model)) == "<class 'lmfit.model.CompositeModel'>":
        if OII_doublet_fit == True:
            one_sigma_blue = (best_fit.best_values['Galaxy_red_mean'] - best_fit.best_values['Galaxy_red_sigma'])-1.0
        else:
            one_sigma_blue = (best_fit.best_values['Galaxy_mean'] - best_fit.best_values['Galaxy_sigma'])-1.0

    blue_left_bound = one_sigma_blue - 4.0
    lam_mask = (wavelength > blue_left_bound) & (wavelength < one_sigma_blue)
    #print(blue_left_bound, one_sigma_blue)

    #calculate the chi squared
    chi_square = np.sum(residual[lam_mask]**2)

    return chi_square



#===============================================================================
#plotting functions
def plot_fit(wavelength, flux, g_model, pars, best_fit, plot_initial=False, include_const=False):
    """
    Plots the fit to the data with residuals

    Parameters
    ----------
    wavelength :
        wavelength vector
    flux :
        the flux of the spectrum
    initial_fit :
        the initial model before fitting
    best_fit : class
        the best fitting model
    plot_initial : bool
        Default is False. Whether to plot the initial guess or not.
    """
    #create a more finely sampled wavelength space
    fine_sampling = np.linspace(min(wavelength), max(wavelength), 1000)

    #get parameters from the best_fit
    best_fit_pars = best_fit.params

    #plot the stuff
    fig1 = plt.figure(figsize=(9,5))
    #add_axes has (xstart, ystart, xend, yend)
    frame1 = fig1.add_axes((.1,.3,.8,.6))
    plt.step(wavelength, flux, where='mid', label='Data')

    #get initial guess for fit
    if plot_initial:
        initial_fit = g_model.eval(pars, x=wavelength)
        plt.step(wavelength, initial_fit, where='mid', label='Initial Guess')

    #if the model was a 1-component Gaussian, plot the best fit gaussian
    if str(type(g_model)) == "<class 'lmfit.model.Model'>":
        label = "Best Fit (Amp: {:.2f}, Mean: {:.2f}, \n Sigma: {:.2f})".format(best_fit.best_values['gauss_amp'], best_fit.best_values['gauss_mean'], best_fit.best_values['gauss_sigma'])
        plt.step(fine_sampling, best_fit.eval(x=fine_sampling), where='mid', label=label)

    #if the original model was a multiple-component Gaussian fit, plot the two gaussians and constant separately
    if str(type(g_model)) == "<class 'lmfit.model.CompositeModel'>":
        plt.step(fine_sampling, best_fit.eval(x=fine_sampling), where='mid', label='Best Fit')
        for i in range(len(best_fit.components)):
            try:
                #get the parameters for the amp, center and sigma
                amp_par = best_fit.params[best_fit.components[i].prefix+'amp']
                mean_par = best_fit.params[best_fit.components[i].prefix+'mean']
                sigma_par = best_fit.params[best_fit.components[i].prefix+'sigma']
                #put the parameter values into a string for the graph label
                label = best_fit.components[i].prefix[:-1]+" (Amp: {:.2f}, Mean: {:.2f}, \n Sigma: {:.2f})".format(amp_par.value, mean_par.value, sigma_par.value)
                #plot the curves
                plt.step(fine_sampling, best_fit.components[i].eval(best_fit_pars, x=fine_sampling), where='mid', label=label)
            except:
                #if the try doesn't work, it should be the constant, so use this line instead
                #make the label for the constant component
                label = best_fit.components[i].prefix[:-1]+": {:.2f}".format(best_fit.best_values['Constant_Continuum_c'])
                #plot the constant line
                plt.step(fine_sampling, np.full_like(fine_sampling, best_fit.components[i].eval(best_fit_pars, x=fine_sampling)), where='mid', label=label)


    #plt.xlim(best_fit.params[best_fit.components[0].prefix+'mean'].value-8.0, best_fit.params[best_fit.components[0].prefix+'mean'].value+8.0)
    plt.ylabel('Flux ($10^{-16}$ erg s$^{-1}$ cm$^{-2}$ $\AA^{-1}$)')
    frame1.axes.get_xaxis().set_ticks([])
    plt.legend(loc='upper right', fontsize=8, frameon=False)


    #create frame for residual plot
    frame2 = fig1.add_axes((.1,.1,.8,.2))
    difference = best_fit.best_fit - flux
    plt.scatter(wavelength, difference, c='r', s=2)
    #plt.xlim(best_fit.params[best_fit.components[0].prefix+'mean'].value-8.0, best_fit.params[best_fit.components[0].prefix+'mean'].value+8.0)
    plt.ylabel('Residuals')
    plt.xlabel('Wavelength ($\AA$)')

    #plt.show()
    return fig1


#===============================================================================
#Applying to entire cube
def fit_cube(galaxy_name, redshift, emission_line, output_folder_loc, emission_line2=None, OII_doublet=False, filename=None, filename2=None, data_cube_stuff=None, emission_dict=all_the_lines, cont_subtract=False, include_const=False, plotting=True, method='leastsq', correct_bad_spaxels=False):
    """
    Fits the entire cube, and checks whether one or two gaussians fit the emission line best.  Must have either the filename to the fits file, or the data_cube_stuff.

    Parameters
    ----------
    galaxy_name : str
        name of the galaxy
    redshift : int
        redshift of the galaxy
    emission_line : str
        the emission line to be fit. Options:
        "Hdelta", "Hgamma", "Hbeta", "Halpha", "OII_1", "OII_2", "HeI", "SII", "OIII_1", "OIII_2", "OIII_3", "OIII_4"
    output_folder_loc : str
        file path to where to put the results and plots output folder
    emission_line2 : str
        the second emission line to be fit using the results from the first.  Default is None. Options:
        "Hdelta", "Hgamma", "Hbeta", "Halpha", "OII_1", "OII_2", "HeI", "SII", "OIII_1", "OIII_2", "OIII_3", "OIII_4"
    OII_doublet : boolean
        whether to fit the OII doublet using the results from the first fit.  Default is False. Uses "OII_1", from the dictionary
    filename : str
        the file path to the data cube - if data_cube_stuff is not given
    filename2 : str
        the file path to the second data cube - generally the continuum subtracted cube.
        If this is not None, the first cube is used to create the S/N mask, and
        this cube is used for fitting.
    data_cube_stuff :
        [lamdas, data] if the filename is not given
    emission_dict : dict
        dictionary of emission lines
    cont_subtract : bool
        when True, use the first 10 pixels in the spectrum to define the continuum
        and subtracts it.  Use False when continuum has already been fit and subtracted.
    plotting : bool
        when True, each best_fit is plotted with its residuals and saved
    method : str
        the fitting method (see )
    correct_bad_spaxels : bool
        Default is False. Takes spaxels (28, 9), (29, 9) and (30, 9), which are saturated in IRAS08, and uses [OIII]4960 rather than [OIII]5007

    Returns
    -------
    outflow_results : :obj:'~numpy.ndarray' object
        array with galaxy sigma, center, amplitude and outflow sigma, center, amplitude values in the same spatial shape as the input data array
    outflow_error : :obj:'~numpy.ndarray'
        array with galaxy sigma, center, amplitude and outflow sigma, center, amplitude errors in the same spatial shape as the input data array
    no_outflow_results : :obj:'~numpy.ndarray'
        array with single gaussian sigma, center, amplitude values in the same spatial shape as the input data array
    no_outflow_error : :obj:'~numpy.ndarray'
        array with single gaussian sigma, center, amplitude errors in the same spatial shape as the input data array
    statistical results : :obj:'~numpy.ndarray'
        array with 0 where one gaussian gave a better BIC value, and 1 where two gaussians gave a better BIC value.

    """
    #get the original data to create the S/N mask
    if filename:
        lamdas, data, header = load_data(filename)
    elif data_cube_stuff:
        lamdas, data = data_cube_stuff
    else:
        print('Data input not understood.  Need either a fits file path or the [lamdas, data] in data_cube_stuff.')
        return

    #create filepath to output folder
    output_folder_loc = output_folder_loc+galaxy_name+'koffee_results_'+emission_line+'_'+str(date.today())+'/'
    pathlib.Path(output_folder_loc).mkdir(exist_ok=True)

    #output_folder_chosen_graphs_loc = output_folder_loc+'/code_preferred_fits/'
    #pathlib.Path(output_folder_chosen_graphs_loc).mkdir(exist_ok=True)

    #create arrays to store results in, with columns for amplitude, mean and standard deviation of gaussians
    #emission line and outflow
    if include_const == True:
        outflow_results = np.empty_like(data[:7,:,:])
        outflow_error = np.empty_like(data[:7,:,:])

        no_outflow_results = np.empty_like(data[:4,:,:])
        no_outflow_error = np.empty_like(data[:4,:,:])

    elif include_const == False:
        outflow_results = np.empty_like(data[:6,:,:])
        outflow_error = np.empty_like(data[:6,:,:])

        no_outflow_results = np.empty_like(data[:3,:,:])
        no_outflow_error = np.empty_like(data[:3,:,:])

    #1 for outflow, 0 for no outflow
    statistical_results = np.empty_like(data[0,:,:])

    #save the blue chi squared values check
    blue_chi_square = np.empty_like(data[0,:,:])

    #save the chi squared values for single and double gaussian fits
    chi_square = np.empty_like(data[:2,:,:])

    #use redshift to shift emission line to observed wavelength
    em_rest = emission_dict[emission_line]
    em_observed = em_rest*(1+redshift)

    #create mask to choose emission line wavelength range
    mask_lam = (lamdas > em_observed-20.) & (lamdas < em_observed+20.)

    #create the S/N array
    sn_array = np.median(data[mask_lam,:,:][:10,:,:], axis=0)/np.std(data[mask_lam,:,:][:10,:,:], axis=0)

    if filename2:
        lamdas, data, header = load_data(filename2)
        print('Second filename being used for koffee fits')
        #recreate mask to choose emission line wavelength range with new lamdas
        mask_lam = (lamdas > em_observed-20.) & (lamdas < em_observed+20.)

    #if fitting a second emission line, create the wavelength mask
    if emission_line2:
        print('Emission Line '+emission_line2+' also being fit')
        #use redshift to shift emission line to observed wavelength
        em_rest2 = emission_dict[emission_line2]
        em_observed = em_rest2*(1+redshift)

        #create mask to choose emission line wavelength range
        mask_lam2 = (lamdas > em_observed-20.) & (lamdas < em_observed+20.)

        #create arrays to save results in
        if include_const == True:
            outflow_results2 = np.empty_like(data[:7,:,:])
            outflow_error2 = np.empty_like(data[:7,:,:])
            no_outflow_results2 = np.empty_like(data[:4,:,:])
            no_outflow_error2 = np.empty_like(data[:4,:,:])
        elif include_const == False:
            outflow_results2 = np.empty_like(data[:6,:,:])
            outflow_error2 = np.empty_like(data[:6,:,:])
            no_outflow_results2 = np.empty_like(data[:3,:,:])
            no_outflow_error2 = np.empty_like(data[:3,:,:])

        chi_square2 = np.empty_like(data[:2,:,:])

    #if fitting the OII doublet, create the wavelength mask
    if OII_doublet == True:
        print('OII doublet also being fit')
        #use redshift to shift emission line to observed wavelength
        em_rest3 = emission_dict['OII_1']
        em_observed = em_rest3*(1+redshift)

        #create mask to choose emission line wavelength range
        mask_lam3 = (lamdas > em_observed-20.) & (lamdas < em_observed+20.)

        #create arrays to save results in - always use a constant when fitting OII doublet
        outflow_results3 = np.empty_like(data[:13,:,:])
        outflow_error3 = np.empty_like(data[:13,:,:])

        no_outflow_results3 = np.empty_like(data[:7,:,:])
        no_outflow_error3 = np.empty_like(data[:7,:,:])

        chi_square3 = np.empty_like(data[:2,:,:])

    if correct_bad_spaxels == True and emission_line == 'OIII_4':
        #get the OIII_3 emission line and redshift it
        em_rest_OIII_3 = emission_dict['OIII_3']
        em_obs_OIII_3 = em_rest_OIII_3*(1+redshift)

        #use OIII_3 to make the mask
        mask_lam_OIII_3 = (lamdas > em_obs_OIII_3-20.) & (lamdas < em_obs_OIII_3+20.)

    #loop through cube
    with tqdm(total=data.shape[1]*data.shape[2]) as pbar:
        for i in range(data.shape[1]):
            for j in range(data.shape[2]):
                try:
                    #apply mask to data
                    if correct_bad_spaxels == True and emission_line == 'OIII_4':
                        if (i, j) in dodgy_spaxels[galaxy_name]:
                            #apply the alternative mask to the wavelength
                            masked_lamdas = lamdas[mask_lam_OIII_3]
                            #apply the mask to the data
                            flux = data[:,i,j][mask_lam_OIII_3]
                            print('Using [OIII] '+str(em_rest_OIII_3)+' for spaxel '+str(i)+','+str(j))

                        else:
                            flux = data[:,i,j][mask_lam]
                            #apply mask to lamdas
                            masked_lamdas = lamdas[mask_lam]

                    elif correct_bad_spaxels == False:
                        flux = data[:,i,j][mask_lam]
                        #apply mask to lamdas
                        masked_lamdas = lamdas[mask_lam]

                    #take out the continuum by finding the average value of the first 10 pixels in the data and minusing them off
                    if cont_subtract==True:
                        continuum = np.median(flux[:10])
                        flux = flux-continuum

                    #only fit if the S/N is greater than 20
                    if sn_array[i,j] >= 20:
                        #create model for 1 Gaussian fit
                        if include_const == True:
                            g_model1, pars1 = gaussian1_const(masked_lamdas, flux)
                        elif include_const == False:
                            g_model1, pars1 = gaussian1(masked_lamdas, flux)
                        #fit model for 1 Gaussian fit
                        best_fit1 = fitter(g_model1, pars1, masked_lamdas, flux, method=method, verbose=False)

                        #create and fit model for 2 Gaussian fit, using 1 Gaussian fit as first guess for the mean
                        if include_const == True:
                            g_model2, pars2 = gaussian2_const(masked_lamdas, flux, amplitude_guess=None, mean_guess=None, sigma_guess=None)
                        elif include_const == False:
                            g_model2, pars2 = gaussian2(masked_lamdas, flux, amplitude_guess=None, mean_guess=None, sigma_guess=None)
                        best_fit2 = fitter(g_model2, pars2, masked_lamdas, flux, method=method, verbose=False)

                        if best_fit2.bic < (best_fit1.bic-10):
                            stat_res = 1
                            #save blue chi square
                            blue_chi_square[i,j] = check_blue_chi_square(masked_lamdas, flux, best_fit2, g_model2)
                        else:
                            stat_res = 0
                            #save blue chi square
                            blue_chi_square[i,j] = check_blue_chi_square(masked_lamdas, flux, best_fit1, g_model1)

                        """if blue_chi_square[i,j] > 0.1:
                            if include_const == True:
                                g_model2_refit, pars2_refit = gaussian2_const(masked_lamdas, flux, amplitude_guess=None, mean_guess=[masked_lamdas[flux.argmax()], masked_lamdas[flux.argmax()]-4.0], sigma_guess=[1.0,8.0])
                            elif include_const == False:
                                g_model2_refit, pars2_refit = gaussian2(masked_lamdas, flux, amplitude_guess=None, mean_guess=[masked_lamdas[flux.argmax()], masked_lamdas[flux.argmax()]-4.0], sigma_guess=[1.0,8.0])
                            best_fit2_refit = fitter(g_model2_refit, pars2_refit, masked_lamdas, flux, method=method, verbose=False)

                            #force it to take the new fit
                            stat_res = 2"""

                        #-------------------
                        #FIT THE SECOND LINE
                        #-------------------
                        if emission_line2:
                            #only want to fit hbeta where there is an outflow
                            if stat_res > 0:
                                #mask the lamdas and data
                                masked_lamdas2 = lamdas[mask_lam2]
                                flux2 = data[:,i,j][mask_lam2]
                                #get the guess values from the previous fits
                                if stat_res == 1:
                                    mean_diff = best_fit2.params['Galaxy_mean'].value - best_fit2.params['Flow_mean'].value
                                    sigma_guess = [best_fit2.params['Galaxy_sigma'].value, best_fit2.params['Flow_sigma'].value]
                                elif stat_res == 2:
                                    mean_diff = best_fit2_refit.params['Galaxy_mean'].value - best_fit2_refit.params['Flow_mean'].value
                                    sigma_guess = [best_fit2_refit.params['Galaxy_sigma'].value, best_fit2_refit.params['Flow_sigma'].value]
                                #create the fitting objects
                                if include_const == True:
                                    #for the one gaussian fit
                                    g_model1_second, pars1_second = gaussian1_const(masked_lamdas2, flux2, amp_guess=None, mean_guess=None, sigma_guess=None)

                                    #for the two gaussian fit
                                    g_model2_second, pars2_second = gaussian2_const(masked_lamdas2, flux2, amplitude_guess=None, mean_guess=[masked_lamdas2[flux2.argmax()], masked_lamdas2[flux2.argmax()]-mean_diff], sigma_guess=sigma_guess, mean_diff=[mean_diff, 1.5], sigma_variations=1.5)
                                elif include_const == False:
                                    #for the one gaussian fit
                                    g_model1_second, pars1_second = gaussian1(masked_lamdas2, flux2, amp_guess=None, mean_guess=None, sigma_guess=None)
                                    #for the two gaussian fit
                                    g_model2_second, pars2_second = gaussian2(masked_lamdas2, flux2, amplitude_guess=None, mean_guess=[masked_lamdas2[flux2.argmax()], masked_lamdas2[flux2.argmax()]-mean_diff], sigma_guess=sigma_guess, mean_diff=[mean_diff, 1.5], sigma_variations=1.5)

                                #do the fit
                                #for the one gaussian fit
                                best_fit1_second = fitter(g_model1_second, pars1_second, masked_lamdas2, flux2, method=method, verbose=False)

                                #for the two gaussian fit
                                best_fit2_second = fitter(g_model2_second, pars2_second, masked_lamdas2, flux2, method=method, verbose=False)

                                #fit the one gaussian fit of the emission line
                                fig3 = plot_fit(masked_lamdas2, flux2, g_model1_second, pars1_second, best_fit1_second, plot_initial=False, include_const=include_const)
                                fig3.suptitle(emission_line2+' ['+str(em_rest2)+'] fit without outflow')
                                fig3.savefig(output_folder_loc+galaxy_name+'_best_fit_'+emission_line2+'_no_outflow_second_fit_'+str(i)+'_'+str(j))
                                plt.close(fig3)

                                #fit the first fit of the second emission line
                                fig4 = plot_fit(masked_lamdas2, flux2, g_model2_second, pars2_second, best_fit2_second, plot_initial=False, include_const=include_const)
                                fig4.suptitle(emission_line2+' ['+str(em_rest2)+'] first fit')
                                fig4.savefig(output_folder_loc+galaxy_name+'_best_fit_'+emission_line2+'_outflow_second_fit_'+str(i)+'_'+str(j)+'_first_try')
                                plt.close(fig4)

                                #check the fit using the blue-side-residual test
                                blue_chi_square_check = check_blue_chi_square(masked_lamdas2, flux2, best_fit2_second, g_model2_second)

                                if blue_chi_square_check > 0.2:
                                    #refit using...
                                    print('Refitting', emission_line2, 'fit for spaxel ', str(i), str(j))
                                    print('This spaxel had blue_chi_square_check ', str(blue_chi_square_check))
                                    #create the fitting objects
                                    if include_const == True:
                                        g_model2_refit_second, pars2_refit_second = gaussian2_const(masked_lamdas2, flux2, amplitude_guess=None, mean_guess=[masked_lamdas2[flux2.argmax()], masked_lamdas2[flux2.argmax()]-mean_diff], sigma_guess=sigma_guess, mean_diff=[mean_diff, 0.5], sigma_variations=0.5)
                                    elif include_const == False:
                                        g_model2_refit_second, pars2_refit_second = gaussian2(masked_lamdas2, flux2, amplitude_guess=None, mean_guess=[masked_lamdas2[flux2.argmax()], masked_lamdas2[flux2.argmax()]-mean_diff], sigma_guess=sigma_guess, mean_diff=[mean_diff, 0.5], sigma_variations=0.5)
                                    #do the fit
                                    best_fit2_refit_second = fitter(g_model2_refit_second, pars2_refit_second, masked_lamdas2, flux2, method=method, verbose=False)

                                    print(emission_line2, 'fit blue-chi-squared-fit chi squared value: ', best_fit2_refit_second.bic)
                                    print(emission_line2, 'fit original chi squared value: ', best_fit2_second.bic)

                                    #plot the refit
                                    fig4 = plot_fit(masked_lamdas2, flux2, g_model2_refit_second, pars2_refit_second, best_fit2_refit_second, plot_initial=False, include_const=include_const)
                                    fig4.suptitle(emission_line2+' ['+str(em_rest2)+'] Chi Square Refit')
                                    fig4.savefig(output_folder_loc+galaxy_name+'_best_fit_'+emission_line2+'_outflow_second_fit_'+str(i)+'_'+str(j)+'_chi_square_refit')
                                    plt.close(fig4)

                                    g_model2_second = g_model2_refit_second
                                    pars2_second = pars2_refit_second
                                    best_fit2_second = best_fit2_refit_second

                                #check that a single gaussian fit wouldn't be better if the flow amplitude
                                #is greater than 80% of the galaxy amplitude
                                if best_fit2_second.params['Flow_amp'].value > 0.9*best_fit2_second.params['Galaxy_amp'].value:
                                    print('Doing one Gaussian fit for spaxel ', str(i), str(j))
                                    #create the fitting objects
                                    if include_const == True:
                                        g_model1_refit_second, pars1_refit_second = gaussian1_const(masked_lamdas2, flux2)#, amp_guess=None, mean_guess=masked_lamdas2[flux2.argmax()])#, sigma_guess=sigma_guess[0])
                                    elif include_const == False:
                                        g_model1_refit_second, pars1_refit_second = gaussian1(masked_lamdas2, flux2)#, amp_guess=None, mean_guess=masked_lamdas2[flux2.argmax()], sigma_guess=sigma_guess[0])

                                    #do the fit
                                    best_fit1_refit_second = fitter(g_model1_refit_second, pars1_refit_second, masked_lamdas2, flux2, method=method, verbose=False)

                                    #do the BIC test
                                    #if the 2 gaussian fit has a lower BIC by 10 or more, it's the better fit
                                    print('One Gaussian fit BIC: ', str(best_fit1_refit_second.bic))
                                    print('Two Gaussian fit BIC: ', str(best_fit2_second.bic))
                                    print('Difference: ', str(best_fit2_second.bic - best_fit1_refit_second.bic))

                                    #plot the refit
                                    fig4 = plot_fit(masked_lamdas2, flux2, g_model1_refit_second, pars1_refit_second, best_fit1_refit_second, plot_initial=False, include_const=include_const)
                                    fig4.suptitle(emission_line2+' ['+str(em_rest2)+'] BIC test Refit')
                                    fig4.savefig(output_folder_loc+galaxy_name+'_best_fit_'+emission_line2+'_outflow_second_fit_'+str(i)+'_'+str(j)+'_BIC_test_refit')
                                    plt.close(fig4)

                                    if best_fit2_second.params['Flow_amp'].value > 0.98*best_fit2_second.params['Galaxy_amp'].value:
                                        print('Using one Gaussian fit for spaxel ', str(i), str(j))

                                        g_model2_second = g_model1_refit_second
                                        pars2_second = pars1_refit_second
                                        best_fit2_second = best_fit1_refit_second

                                    else:
                                        if best_fit1_refit_second.bic < best_fit2_second.bic+50:
                                            print('One Gaussian fit better for spaxel ', str(i), str(j))

                                            g_model2_second = g_model1_refit_second
                                            pars2_second = pars1_refit_second
                                            best_fit2_second = best_fit1_refit_second


                            #put the results into the array to be saved
                            chi_square2[:,i,j] = (best_fit1_second.bic, best_fit2_second.bic)
                            try:
                                #try to save the results
                                if include_const == True:
                                    outflow_results2[:,i,j] = (best_fit2_second.params['Galaxy_sigma'].value, best_fit2_second.params['Galaxy_mean'].value, best_fit2_second.params['Galaxy_amp'].value, best_fit2_second.params['Flow_sigma'].value, best_fit2_second.params['Flow_mean'].value, best_fit2_second.params['Flow_amp'].value, best_fit2_second.params['Constant_Continuum_c'].value)
                                    outflow_error2[:,i,j] = (best_fit2_second.params['Galaxy_sigma'].stderr, best_fit2_second.params['Galaxy_mean'].stderr, best_fit2_second.params['Galaxy_amp'].stderr, best_fit2_second.params['Flow_sigma'].stderr, best_fit2_second.params['Flow_mean'].stderr, best_fit2_second.params['Flow_amp'].stderr, best_fit2_second.params['Constant_Continuum_c'].stderr)
                                    no_outflow_results2[:,i,j] = (best_fit1_second.params['gauss_sigma'].value, best_fit1_second.params['gauss_mean'].value, best_fit1_second.params['gauss_amp'].value, best_fit1_second.params['Constant_Continuum_c'].value)
                                    no_outflow_error2[:,i,j] = (best_fit1_second.params['gauss_sigma'].stderr, best_fit1_second.params['gauss_mean'].stderr, best_fit1_second.params['gauss_amp'].stderr, best_fit1_second.params['Constant_Continuum_c'].stderr)
                                elif include_const == False:
                                    outflow_results2[:,i,j] = (best_fit2_second.params['Galaxy_sigma'].value, best_fit2_second.params['Galaxy_mean'].value, best_fit2_second.params['Galaxy_amp'].value, best_fit2_second.params['Flow_sigma'].value, best_fit2_second.params['Flow_mean'].value, best_fit2_second.params['Flow_amp'].value)
                                    outflow_error2[:,i,j] = (best_fit2_second.params['Galaxy_sigma'].stderr, best_fit2_second.params['Galaxy_mean'].stderr, best_fit2_second.params['Galaxy_amp'].stderr, best_fit2_second.params['Flow_sigma'].stderr, best_fit2_second.params['Flow_mean'].stderr, best_fit2_second.params['Flow_amp'].stderr)
                                    no_outflow_results2[:,i,j] = (best_fit1_second.params['gauss_sigma'].value, best_fit1_second.params['gauss_mean'].value, best_fit1_second.params['gauss_amp'].value)
                                    no_outflow_error2[:,i,j] = (best_fit2_second.params['gauss_sigma'].stderr, best_fit1_second.params['gauss_mean'].stderr, best_fit1_second.params['gauss_amp'].stderr)
                            except:
                                #if that doesn't work, then the one gaussian fit was better
                                if include_const == True:
                                    outflow_results2[:,i,j] = (best_fit2_second.params['gauss_sigma'].value, best_fit2_second.params['gauss_mean'].value, best_fit2_second.params['gauss_amp'].value, np.nan, np.nan, np.nan, best_fit2_second.params['Constant_Continuum_c'].value)
                                    outflow_error2[:,i,j] = (best_fit2_second.params['gauss_sigma'].stderr, best_fit2_second.params['gauss_mean'].stderr, best_fit2_second.params['gauss_amp'].stderr, np.nan, np.nan, np.nan, best_fit2_second.params['Constant_Continuum_c'].stderr)
                                    no_outflow_results2[:,i,j] = (best_fit1_second.params['gauss_sigma'].value, best_fit1_second.params['gauss_mean'].value, best_fit1_second.params['gauss_amp'].value, best_fit1_second.params['Constant_Continuum_c'].value)
                                    no_outflow_error2[:,i,j] = (best_fit1_second.params['gauss_sigma'].stderr, best_fit1_second.params['gauss_mean'].stderr, best_fit1_second.params['gauss_amp'].stderr, best_fit1_second.params['Constant_Continuum_c'].stderr)
                                elif include_const == False:
                                    outflow_results2[:,i,j] = (best_fit2_second.params['gauss_sigma'].value, best_fit2_second.params['gauss_mean'].value, best_fit2_second.params['gauss_amp'].value, np.nan, np.nan, np.nan)
                                    outflow_error2[:,i,j] = (best_fit2_second.params['gauss_sigma'].stderr, best_fit2_second.params['gauss_mean'].stderr, best_fit2_second.params['gauss_amp'].stderr, np.nan, np.nan, np.nan)
                                    no_outflow_results2[:,i,j] = (best_fit1_second.params['gauss_sigma'].value, best_fit1_second.params['gauss_mean'].value, best_fit1_second.params['gauss_amp'].value)
                                    no_outflow_error2[:,i,j] = (best_fit1_second.params['gauss_sigma'].stderr, best_fit1_second.params['gauss_mean'].stderr, best_fit1_second.params['gauss_amp'].stderr)

                        #-------------------
                        #FIT THE OII DOUBLET
                        #-------------------
                        if OII_doublet == True:
                            #only want to fit OII doublet where there is an outflow
                            if stat_res > 0:
                                #mask the lamdas and data
                                masked_lamdas3 = lamdas[mask_lam3]
                                flux3 = data[:,i,j][mask_lam3]
                                #get the guess values from the previous fits
                                if stat_res == 1:
                                    mean_diff = best_fit2.params['Galaxy_mean'].value - best_fit2.params['Flow_mean'].value
                                    sigma_guess = [best_fit2.params['Galaxy_sigma'].value, best_fit2.params['Flow_sigma'].value]
                                elif stat_res == 2:
                                    mean_diff = best_fit2_refit.params['Galaxy_mean'].value - best_fit2_refit.params['Flow_mean'].value
                                    sigma_guess = [best_fit2_refit.params['Galaxy_sigma'].value, best_fit2_refit.params['Flow_sigma'].value]

                                #create the fitting objects for the 2 gaussian fit
                                g_model1_third, pars1_third = gaussian1_OII_doublet(masked_lamdas3, flux3, amplitude_guess=None, mean_guess=[masked_lamdas3[np.argmax(flux3)]+1.0], sigma_guess=sigma_guess, sigma_variations=1.5)

                                #do the fit
                                best_fit1_third = fitter(g_model1_third, pars1_third, masked_lamdas3, flux3, method=method, verbose=False)

                                #plot the fit
                                fig4 = plot_fit(masked_lamdas3, flux3, g_model1_third, pars1_third, best_fit1_third, plot_initial=False, include_const=True)
                                fig4.suptitle('OII doublet ['+str(em_rest3)+', '+str(emission_dict['OII_2']*(1+redshift))+'] not including outflows')
                                fig4.savefig(output_folder_loc+galaxy_name+'_best_fit_OII_doublet_no_outflow_'+str(i)+'_'+str(j))
                                plt.close(fig4)

                                #create the fitting objects for the four gaussian fit
                                g_model2_third, pars2_third = gaussian2_OII_doublet(masked_lamdas3, flux3, amplitude_guess=None, mean_guess=None, sigma_guess=sigma_guess, mean_diff=[mean_diff, 1.5], sigma_variations=1.5)

                                #do the fit
                                best_fit2_third = fitter(g_model2_third, pars2_third, masked_lamdas3, flux3, method=method, verbose=False)

                                #plot the fit
                                fig4 = plot_fit(masked_lamdas3, flux3, g_model2_third, pars2_third, best_fit2_third, plot_initial=False, include_const=True)
                                fig4.suptitle('OII doublet ['+str(em_rest3)+', '+str(emission_dict['OII_2']*(1+redshift))+']')
                                fig4.savefig(output_folder_loc+galaxy_name+'_best_fit_OII_doublet_outflow_'+str(i)+'_'+str(j))
                                plt.close(fig4)

                                #check the fit using the blue-side-residual test
                                blue_chi_square_check = check_blue_chi_square(masked_lamdas3, flux3, best_fit2_third, g_model2_third, OII_doublet_fit=True)
                                print('OII doublet blue chi square check for spaxel ', str(i), str(j), ' is ', str(blue_chi_square_check))

                                if blue_chi_square_check > 100.0:
                                    #refit using...
                                    print('Refitting OII doublet fit for spaxel ', str(i), str(j))
                                    print('This spaxel had blue_chi_square_check ', str(blue_chi_square_check))
                                    #create the fitting objects
                                    g_model2_refit_third, pars2_refit_third = gaussian2_OII_doublet(masked_lamdas3, flux3, amplitude_guess=None, mean_guess=[masked_lamdas3[np.argmax(flux3)]+1.0], sigma_guess=sigma_guess, mean_diff=[mean_diff, 1.0], sigma_variations=0.25)

                                    #do the fit
                                    best_fit2_refit_third = fitter(g_model2_refit_third, pars2_refit_third, masked_lamdas3, flux3, method=method, verbose=False)

                                    print('OII doublet fit blue-chi-squared-fit chi squared value: ', best_fit2_refit_third.bic)
                                    print('OII doublet fit original chi squared value: ', best_fit2_third.bic)

                                    #plot the refit
                                    fig5 = plot_fit(masked_lamdas3, flux3, g_model2_refit_third, pars2_refit_third, best_fit2_refit_third, plot_initial=False, include_const=True)
                                    fig5.suptitle('OII doublet Chi Square Refit')
                                    fig5.savefig(output_folder_loc+galaxy_name+'_best_fit_OII_doublet_outflow_'+str(i)+'_'+str(j)+'_chi_square_refit')
                                    plt.close(fig5)

                                    g_model2_third = g_model2_refit_third
                                    pars2_third = pars2_refit_third
                                    best_fit2_third = best_fit2_refit_third
                                    print('Replaced OII doublet values with refit values')


                            #put the results into the array to be saved
                            chi_square3[:,i,j] = (best_fit1_third.bic, best_fit2_third.bic)

                            no_outflow_results3[:,i,j] = (best_fit1_third.params['Galaxy_blue_sigma'].value, best_fit1_third.params['Galaxy_blue_mean'].value, best_fit1_third.params['Galaxy_blue_amp'].value, best_fit1_third.params['Galaxy_red_sigma'].value, best_fit1_third.params['Galaxy_red_mean'].value, best_fit1_third.params['Galaxy_red_amp'].value, best_fit1_third.params['Constant_Continuum_c'].value)
                            no_outflow_error3[:,i,j] = (best_fit1_third.params['Galaxy_blue_sigma'].stderr, best_fit1_third.params['Galaxy_blue_mean'].stderr, best_fit1_third.params['Galaxy_blue_amp'].stderr, best_fit1_third.params['Galaxy_red_sigma'].stderr, best_fit1_third.params['Galaxy_red_mean'].stderr, best_fit1_third.params['Galaxy_red_amp'].stderr, best_fit1_third.params['Constant_Continuum_c'].stderr)

                            outflow_results3[:,i,j] = (best_fit2_third.params['Galaxy_blue_sigma'].value, best_fit2_third.params['Galaxy_blue_mean'].value, best_fit2_third.params['Galaxy_blue_amp'].value, best_fit2_third.params['Galaxy_red_sigma'].value, best_fit2_third.params['Galaxy_red_mean'].value, best_fit2_third.params['Galaxy_red_amp'].value, best_fit2_third.params['Flow_blue_sigma'].value, best_fit2_third.params['Flow_blue_mean'].value, best_fit2_third.params['Flow_blue_amp'].value, best_fit2_third.params['Flow_red_sigma'].value, best_fit2_third.params['Flow_red_mean'].value, best_fit2_third.params['Flow_red_amp'].value, best_fit2_third.params['Constant_Continuum_c'].value)
                            outflow_error3[:,i,j] = (best_fit2_third.params['Galaxy_blue_sigma'].stderr, best_fit2_third.params['Galaxy_blue_mean'].stderr, best_fit2_third.params['Galaxy_blue_amp'].stderr, best_fit2_third.params['Galaxy_red_sigma'].stderr, best_fit2_third.params['Galaxy_red_mean'].stderr, best_fit2_third.params['Galaxy_red_amp'].stderr, best_fit2_third.params['Flow_blue_sigma'].stderr, best_fit2_third.params['Flow_blue_mean'].stderr, best_fit2_third.params['Flow_blue_amp'].stderr, best_fit2_third.params['Flow_red_sigma'].stderr, best_fit2_third.params['Flow_red_mean'].stderr, best_fit2_third.params['Flow_red_amp'].stderr, best_fit2_third.params['Constant_Continuum_c'].stderr)

                        #put stat_res into the array
                        statistical_results[i,j] = stat_res

                        #emission and outflow
                        if statistical_results[i,j] == 2:
                            chi_square[:,i,j] = (best_fit1.bic, best_fit2_refit.bic)
                            if include_const == True:
                                outflow_results[:,i,j] = (best_fit2_refit.params['Galaxy_sigma'].value, best_fit2_refit.params['Galaxy_mean'].value, best_fit2_refit.params['Galaxy_amp'].value, best_fit2_refit.params['Flow_sigma'].value, best_fit2_refit.params['Flow_mean'].value, best_fit2_refit.params['Flow_amp'].value, best_fit2_refit.params['Constant_Continuum_c'].value)
                                outflow_error[:,i,j] = (best_fit2_refit.params['Galaxy_sigma'].stderr, best_fit2_refit.params['Galaxy_mean'].stderr, best_fit2_refit.params['Galaxy_amp'].stderr, best_fit2_refit.params['Flow_sigma'].stderr, best_fit2_refit.params['Flow_mean'].stderr, best_fit2_refit.params['Flow_amp'].stderr, best_fit2_refit.params['Constant_Continuum_c'].stderr)
                            elif include_const == False:
                                outflow_results[:,i,j] = (best_fit2_refit.params['Galaxy_sigma'].value, best_fit2_refit.params['Galaxy_mean'].value, best_fit2_refit.params['Galaxy_amp'].value, best_fit2_refit.params['Flow_sigma'].value, best_fit2_refit.params['Flow_mean'].value, best_fit2_refit.params['Flow_amp'].value)
                                outflow_error[:,i,j] = (best_fit2_refit.params['Galaxy_sigma'].stderr, best_fit2_refit.params['Galaxy_mean'].stderr, best_fit2_refit.params['Galaxy_amp'].stderr, best_fit2_refit.params['Flow_sigma'].stderr, best_fit2_refit.params['Flow_mean'].stderr, best_fit2_refit.params['Flow_amp'].stderr)

                        else:
                            chi_square[:,i,j] = (best_fit1.bic, best_fit2.bic)
                            if include_const == True:
                                outflow_results[:,i,j] = (best_fit2.params['Galaxy_sigma'].value, best_fit2.params['Galaxy_mean'].value, best_fit2.params['Galaxy_amp'].value, best_fit2.params['Flow_sigma'].value, best_fit2.params['Flow_mean'].value, best_fit2.params['Flow_amp'].value, best_fit2.params['Constant_Continuum_c'].value)
                                outflow_error[:,i,j] = (best_fit2.params['Galaxy_sigma'].stderr, best_fit2.params['Galaxy_mean'].stderr, best_fit2.params['Galaxy_amp'].stderr, best_fit2.params['Flow_sigma'].stderr, best_fit2.params['Flow_mean'].stderr, best_fit2.params['Flow_amp'].stderr, best_fit2.params['Constant_Continuum_c'].stderr)
                            elif include_const == False:
                                outflow_results[:,i,j] = (best_fit2.params['Galaxy_sigma'].value, best_fit2.params['Galaxy_mean'].value, best_fit2.params['Galaxy_amp'].value, best_fit2.params['Flow_sigma'].value, best_fit2.params['Flow_mean'].value, best_fit2.params['Flow_amp'].value)
                                outflow_error[:,i,j] = (best_fit2.params['Galaxy_sigma'].stderr, best_fit2.params['Galaxy_mean'].stderr, best_fit2.params['Galaxy_amp'].stderr, best_fit2.params['Flow_sigma'].stderr, best_fit2.params['Flow_mean'].stderr, best_fit2.params['Flow_amp'].stderr)

                        #just emission
                        if include_const == True:
                            no_outflow_results[:,i,j] = (best_fit1.params['gauss_sigma'].value, best_fit1.params['gauss_mean'].value, best_fit1.params['gauss_amp'].value, best_fit1.params['Constant_Continuum_c'].value)
                            no_outflow_error[:,i,j] = (best_fit1.params['gauss_sigma'].stderr, best_fit1.params['gauss_mean'].stderr, best_fit1.params['gauss_amp'].stderr, best_fit1.params['Constant_Continuum_c'].stderr)
                        elif include_const == False:
                            no_outflow_results[:,i,j] = (best_fit1.params['gauss_sigma'].value, best_fit1.params['gauss_mean'].value, best_fit1.params['gauss_amp'].value)
                            no_outflow_error[:,i,j] = (best_fit1.params['gauss_sigma'].stderr, best_fit1.params['gauss_mean'].stderr, best_fit1.params['gauss_amp'].stderr)

                        #plot fit results
                        if plotting == True:
                            fig1 = plot_fit(masked_lamdas, flux, g_model1, pars1, best_fit1, plot_initial=False, include_const=include_const)
                            if correct_bad_spaxels == True and (i, j) in dodgy_spaxels[galaxy_name]:
                                fig1.suptitle('[OIII] '+str(em_rest_OIII_3))
                                fig1.savefig(output_folder_loc+galaxy_name+'_best_fit_OIII_3_no_outflow_'+str(i)+'_'+str(j))
                            else:
                                fig1.suptitle(emission_line+' ['+str(em_rest)+']')
                                fig1.savefig(output_folder_loc+galaxy_name+'_best_fit_'+emission_line+'_no_outflow_'+str(i)+'_'+str(j))
                            plt.close(fig1)

                            if statistical_results[i,j] == 2:
                                fig2 = plot_fit(masked_lamdas, flux, g_model2_refit, pars2_refit, best_fit2_refit, plot_initial=False, include_const=include_const)
                            else:
                                fig2 = plot_fit(masked_lamdas, flux, g_model2, pars2, best_fit2, plot_initial=False)

                            if correct_bad_spaxels == True and (i, j) in dodgy_spaxels[galaxy_name]:
                                fig2.suptitle('[OIII] '+str(em_rest_OIII_3))
                                fig2.savefig(output_folder_loc+galaxy_name+'_best_fit_OIII_3_outflow_'+str(i)+'_'+str(j))
                            else:
                                fig2.suptitle(emission_line+' ['+str(em_rest)+']')
                                fig2.savefig(output_folder_loc+galaxy_name+'_best_fit_'+emission_line+'_outflow_'+str(i)+'_'+str(j))
                            plt.close(fig2)

                            if emission_line2:
                                fig3 = plot_fit(masked_lamdas2, flux2, g_model2_second, pars2_second, best_fit2_second, plot_initial=False, include_const=include_const)
                                fig3.suptitle(emission_line2+' ['+str(em_rest2)+'] final fit')
                                fig3.savefig(output_folder_loc+galaxy_name+'_best_fit_'+emission_line2+'_outflow_second_fit_'+str(i)+'_'+str(j)+'_final_fit')
                                plt.close(fig3)

                            #if OII_doublet == True:
                                #fig4 = plot_fit(masked_lamdas3, flux3, g_model2_third, pars2_third, best_fit2_third, plot_initial=False, include_const=True)
                                #fig4.suptitle('OII doublet ['+str(em_rest3)+', '+str(emission_dict['OII_2']*(1+redshift))+']')
                                #fig4.savefig(output_folder_loc+galaxy_name+'_best_fit_OII_doublet_outflow_'+str(i)+'_'+str(j))
                                #plt.close(fig4)


                    #if the S/N is less than 20:
                    else:
                        print('S/N for '+str(i)+', '+str(j)+' is '+str(s_n))
                        #statistical results have no outflow
                        statistical_results[i,j] = 0

                        #chi squared for the fits
                        chi_square[:,i,j] = (np.nan, np.nan)

                        #blue chi square
                        blue_chi_square[i,j] = np.nan


                        if include_const == True:
                            #emission and outflow
                            outflow_results[:,i,j] = (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)
                            outflow_error[:,i,j] = (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)

                            #just emission
                            no_outflow_results[:,i,j] = (np.nan, np.nan, np.nan, np.nan)
                            no_outflow_error[:,i,j] = (np.nan, np.nan, np.nan, np.nan)

                        elif include_const == False:
                            #emission and outflow
                            outflow_results[:,i,j] = (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)
                            outflow_error[:,i,j] = (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)

                            #just emission
                            no_outflow_results[:,i,j] = (np.nan, np.nan, np.nan)
                            no_outflow_error[:,i,j] = (np.nan, np.nan, np.nan)

                        if emission_line2:
                            chi_square2[:,i,j] = (np.nan, np.nan)
                            #emission and outflow
                            if include_const == True:
                                outflow_results2[:,i,j] = (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)
                                outflow_error2[:,i,j] = (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)
                                no_outflow_results2[:,i,j] = (np.nan, np.nan, np.nan, np.nan)
                                no_outflow_error2[:,i,j] = (np.nan, np.nan, np.nan, np.nan)
                            elif include_const == False:
                                outflow_results2[:,i,j] = (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)
                                outflow_error2[:,i,j] = (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)
                                no_outflow_results2[:,i,j] = (np.nan, np.nan, np.nan)
                                no_outflow_error2[:,i,j] = (np.nan, np.nan, np.nan)

                        if OII_doublet == True:
                            chi_square3[:,i,j] = (np.nan, np.nan)
                            #emission and outflow
                            outflow_results3[:,i,j] = (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)
                            outflow_error3[:,i,j] = (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)

                            #just emission
                            no_outflow_results3[:,i,j] = (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)
                            no_outflow_error3[:,i,j] = (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)


                except:
                    #statistical results obviously no outflow
                    statistical_results[i,j] = 0

                    #chi squared for the fits
                    chi_square[:,i,j] = (np.nan, np.nan)

                    #blue chi square
                    blue_chi_square[i,j] = np.nan


                    if include_const == True:
                        #emission and outflow
                        outflow_results[:,i,j] = (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)
                        outflow_error[:,i,j] = (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)

                        #just emission
                        no_outflow_results[:,i,j] = (np.nan, np.nan, np.nan, np.nan)
                        no_outflow_error[:,i,j] = (np.nan, np.nan, np.nan, np.nan)

                    elif include_const == False:
                        #emission and outflow
                        outflow_results[:,i,j] = (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)
                        outflow_error[:,i,j] = (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)

                        #just emission
                        no_outflow_results[:,i,j] = (np.nan, np.nan, np.nan)
                        no_outflow_error[:,i,j] = (np.nan, np.nan, np.nan)

                    if emission_line2:
                        chi_square2[:,i,j] = (np.nan, np.nan)
                        #emission and outflow
                        if include_const == True:
                            outflow_results2[:,i,j] = (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)
                            outflow_error2[:,i,j] = (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)
                            no_outflow_results2[:,i,j] = (np.nan, np.nan, np.nan, np.nan)
                            no_outflow_error2[:,i,j] = (np.nan, np.nan, np.nan, np.nan)
                        elif include_const == False:
                            outflow_results2[:,i,j] = (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)
                            outflow_error2[:,i,j] = (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)
                            no_outflow_results2[:,i,j] = (np.nan, np.nan, np.nan)
                            no_outflow_error2[:,i,j] = (np.nan, np.nan, np.nan)

                    if OII_doublet == True:
                        chi_square3[:,i,j] = (np.nan, np.nan)
                        #emission and outflow
                        outflow_results3[:,i,j] = (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)
                        outflow_error3[:,i,j] = (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)

                        #just emission
                        no_outflow_results3[:,i,j] = (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)
                        no_outflow_error3[:,i,j] = (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)

                #update progress bar
                pbar.update(1)


    if include_const == True:
        np.savetxt(output_folder_loc+galaxy_name+'_outflow_results_'+emission_line+'.txt', np.reshape(outflow_results, (7, -1)))
        np.savetxt(output_folder_loc+galaxy_name+'_outflow_error_'+emission_line+'.txt', np.reshape(outflow_error, (7, -1)))
        np.savetxt(output_folder_loc+galaxy_name+'_no_outflow_results_'+emission_line+'.txt', np.reshape(no_outflow_results, (4, -1)))
        np.savetxt(output_folder_loc+galaxy_name+'_no_outflow_error_'+emission_line+'.txt', np.reshape(no_outflow_error, (4, -1)))

    elif include_const == False:
        np.savetxt(output_folder_loc+galaxy_name+'_outflow_results_'+emission_line+'.txt', np.reshape(outflow_results, (6, -1)))
        np.savetxt(output_folder_loc+galaxy_name+'_outflow_error_'+emission_line+'.txt', np.reshape(outflow_error, (6, -1)))
        np.savetxt(output_folder_loc+galaxy_name+'_no_outflow_results_'+emission_line+'.txt', np.reshape(no_outflow_results, (3, -1)))
        np.savetxt(output_folder_loc+galaxy_name+'_no_outflow_error_'+emission_line+'.txt', np.reshape(no_outflow_error, (3, -1)))

    np.savetxt(output_folder_loc+galaxy_name+'_stat_results_'+emission_line+'.txt', np.reshape(statistical_results, (1, -1)))
    np.savetxt(output_folder_loc+galaxy_name+'_chi_squared_'+emission_line+'.txt', np.reshape(chi_square, (2,-1)))

    if emission_line2:
        if include_const == True:
            np.savetxt(output_folder_loc+galaxy_name+'_outflow_results_'+emission_line2+'.txt', np.reshape(outflow_results2, (7, -1)))
            np.savetxt(output_folder_loc+galaxy_name+'_outflow_error_'+emission_line2+'.txt', np.reshape(outflow_error2, (7, -1)))
            np.savetxt(output_folder_loc+galaxy_name+'_no_outflow_results_'+emission_line2+'.txt', np.reshape(no_outflow_results2, (4, -1)))
            np.savetxt(output_folder_loc+galaxy_name+'_no_outflow_error_'+emission_line2+'.txt', np.reshape(no_outflow_error2, (4, -1)))
        elif include_const == False:
            np.savetxt(output_folder_loc+galaxy_name+'_outflow_results_'+emission_line2+'.txt', np.reshape(outflow_results2, (6, -1)))
            np.savetxt(output_folder_loc+galaxy_name+'_outflow_error_'+emission_line2+'.txt', np.reshape(outflow_error2, (6, -1)))
            np.savetxt(output_folder_loc+galaxy_name+'_no_outflow_results_'+emission_line2+'.txt', np.reshape(outflow_results2, (3, -1)))
            np.savetxt(output_folder_loc+galaxy_name+'_no_outflow_error_'+emission_line2+'.txt', np.reshape(outflow_error2, (3, -1)))
        np.savetxt(output_folder_loc+galaxy_name+'_chi_squared_'+emission_line2+'.txt', np.reshape(chi_square2, (2,-1)))

    if OII_doublet == True:
        np.savetxt(output_folder_loc+galaxy_name+'_outflow_results_OII_doublet.txt', np.reshape(outflow_results3, (13, -1)))
        np.savetxt(output_folder_loc+galaxy_name+'_outflow_error_OII_doublet.txt', np.reshape(outflow_error3, (13, -1)))

        np.savetxt(output_folder_loc+galaxy_name+'_no_outflow_results_OII_doublet.txt', np.reshape(no_outflow_results3, (7, -1)))
        np.savetxt(output_folder_loc+galaxy_name+'_no_outflow_error_OII_doublet.txt', np.reshape(no_outflow_error3, (7, -1)))
        np.savetxt(output_folder_loc+galaxy_name+'_chi_squared_OII_doublet.txt', np.reshape(chi_square3, (2, -1)))




    if OII_doublet==False and emission_line2:
        return outflow_results, outflow_error, no_outflow_results, no_outflow_error, statistical_results, chi_square, blue_chi_square, outflow_results2, outflow_error2, no_outflow_results2, no_outflow_error2, chi_square2

    if OII_doublet==True and emission_line2:
        return outflow_results, outflow_error, no_outflow_results, no_outflow_error, statistical_results, chi_square, blue_chi_square, outflow_results2, outflow_error2, no_outflow_results2, no_outflow_error2, chi_square2, outflow_results3, outflow_error3, no_outflow_results3, no_outflow_error3, chi_square3

    else:
        return outflow_results, outflow_error, no_outflow_results, no_outflow_error, statistical_results, chi_square, blue_chi_square







#===============================================================================
#General Plotting of cube data

def plot_vel_diff(outflow_results, statistical_results, lamdas, xx, yy, data, z):
    """
    Plots the velocity difference between the main galaxy line and the outflow where there is an outflow, and 0km/s where there is no outflow (v_broad-v_narrow)
    """
    #create array to keep velocity difference in
    vel_out = np.empty_like(statistical_results)

    #create outflow mask
    flow_mask = (statistical_results>0)

    #de-redshift the data first!!!
    systemic_mean = outflow_results[1,:,:][flow_mask]/(1+z)
    flow_mean = outflow_results[4,:,:][flow_mask]/(1+z)
    flow_sigma = outflow_results[3,:,:][flow_mask]/(1+z)

    #find the velocity difference
    #doing c*(lam_gal-lam_out)/lam_gal
    vel_diff = 299792.458*(systemic_mean-flow_mean)/systemic_mean

    v_out = 2*flow_sigma*299792.458/systemic_mean + vel_diff

    vel_out[~flow_mask] = np.nan
    vel_out[flow_mask] = v_out

    #create an array with zero where there are no outflows, but there is high enough S/N
    no_vel_out = np.empty_like(vel_out)
    no_vel_out[np.isnan(outflow_results[1,:,:])] = np.nan
    no_vel_out[flow_mask] = np.nan
    no_vel_out[np.isnan(outflow_results[1,:,:])][statistical_results[np.isnan(outflow_results[1,:,:])]==0] = 0.0

    #find the median of the continuum for the contours
    cont_mask = (lamdas>4600*(1+z))&(lamdas<4800*(1+z))
    cont_median = np.median(data[cont_mask,:,:], axis=0)

    xmin, xmax = xx.min(), xx.max()
    ymin, ymax = yy.min(), yy.max()

    plt.figure()
    plt.rcParams['axes.facecolor']='black'
    im = plt.pcolormesh(xx, yy, vel_out, cmap='viridis_r')
    plt.contour(yy, xx, cont_median, colors='white', linewidths=0.7, alpha=0.7, levels=(0.2,0.3,0.4,0.7,1.0,2.0,4.0))
    plt.gca().invert_xaxis()
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.colorbar(im, label=r'$\Delta v_{\rm broad-narrow}$ (km s$^{-1}$)', pad=0.01)
    plt.xlabel('Arcseconds')
    plt.ylabel('Arcseconds')
    plt.show()

    plt.figure()
    plt.rcParams['axes.facecolor']='black'
    #im = plt.pcolormesh(yy, xx, vel_diff.T, cmap='viridis_r', vmin=-150, vmax=60)
    im1 = plt.pcolormesh(xx, yy, no_vel_out, cmap='binary', vmin=0.0, vmax=50.0)
    im2 = plt.pcolormesh(xx, yy, vel_out, cmap='viridis_r')
    #plt.gca().invert_xaxis()
    plt.contour(yy, xx, cont_median, colors='white', linewidths=0.7, alpha=0.7, levels=(0.2,0.3,0.4,0.7,1.0,2.0,4.0))
    plt.colorbar(im2, label=r'$\Delta v_{\rm broad-narrow}$ (km s$^{-1}$)', pad=0.01)
    plt.xlabel('Arcseconds')
    plt.ylabel('Arcseconds')
    plt.show()

    return vel_out





#===============================================================================

#===============================================================================


if __name__ == '__main__':

	#with open('/fred/oz088/Duvet/Bron/ppxf_fitting_results/IRAS08_metacube_22Jan2020/IRAS08339_cont_subtracted_unnormalised_cube', 'rb') as f:
	#	lamdas, data = pickle.load(f)
	#f.close()

	#outflow_results, outflow_error, no_outflow_results, no_outflow_error, statistical_results = fit_cube(galaxy_name='IRAS08', redshift=0.018950, emission_line='OIII_4', output_folder_loc='/fred/oz088/Duvet/Bron/koffee_results/', filename=None, data_cube_stuff=[lamdas, data], emission_dict=all_the_lines, cont_subtract=False, plotting=True)

    #filename = '/fred/oz088/Duvet/nnielsen/IRAS08339/Combine/metacube.fits'
    filename1 = '../../data/IRAS08_red_cubes/IRAS08339_metacube.fits'
    filename2 = '../../code_outputs/IRAS08_ppxf_25June2020/IRAS08339_cont_subtracted_unnormalised_cube.fits'

    outflow_results, outflow_error, no_outflow_results, no_outflow_error, statistical_results, blue_chi_square, outflow_results2, outflow_error2 = fit_cube(galaxy_name='IRAS08', redshift=0.018950, emission_line='OIII_4', output_folder_loc='../../code_outputs/koffee_results_IRAS08/', emission_line2='Hbeta', filename=filename1, filename2=filename2, data_cube_stuff=None, emission_dict=all_the_lines, cont_subtract=False, include_const=True, plotting=True, method='leastsq', correct_bad_spaxels=True)

    #filename = '../data/Apr29_final_cubes/final_mosaic/kbmosaic_full_fixed.fits'

    #outflow_results, outflow_error, no_outflow_results, no_outflow_error, statistical_results, blue_chi_square = fit_cube(galaxy_name='J125', redshift=0.03821, emission_line='OIII_4', output_folder_loc='../code_outputs/', filename=filename, data_cube_stuff=None, emission_dict=all_the_lines, cont_subtract=True, plotting=True, method='leastsq', correct_bad_spaxels=False)
