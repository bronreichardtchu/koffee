"""
NAME:
    koffee_fitting_functions.py

AUTHOR:
    Bronwyn Reichardt Chu
    Swinburne
    2021

EMAIL:
    <breichardtchu@swin.edu.au>

PURPOSE:
    All the fitting functions used in koffee
    Written on MacOS Mojave 10.14.5, with Python 3.7

FUNCTIONS INLCUDED:
    fitter
    gaussian_func
    gaussian1
    gaussian2
    gaussian1_const
    gaussian2_const
    gaussian1_OII_doublet
    gaussian2_OII_doublet
    gaussian_NaD

MODIFICATION HISTORY:
    v.1.0 - first created January 2021, copying functions across from the
    original koffee.py file.

"""

import numpy as np

#make sure matplotlib doesn't create any windows
#import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt

from astropy.io import fits

from astropy import units

from lmfit import Parameters
from lmfit import Model
from lmfit.models import GaussianModel, ConstantModel #, LinearModel, ConstantModel



#===============================================================================
#FITTING FUNCTION
#===============================================================================
def fitter(g_model, pars, wavelength, flux, weights=None, method='leastsq', verbose=True):
    """
    Fits the model to the data using a Levenberg-Marquardt least squares method
    by default (lmfit generally uses the scipy.optimize or scipy.optimize.minimize
    methods).  Prints the fit report, containing fit statistics and best-fit
    values with uncertainties and correlations.

    Parameters
    ---------
    g_model :
        the model to fit to the data

    pars :
        Parameters object containing the variables and their constraints

    wavelength : :obj:'~numpy.ndarray'
        wavelength vector

    flux : :obj:'~numpy.ndarray'
        the flux of the spectrum

    weights : :obj:'~numpy.ndarray' or None
        the inverse of the variance, or None.  Default is None.

    method : str
        the fitting method to use, for example:
        - 'leastsq' - Levenberg-Marquardt least squares method (default)
        - 'emcee' - Maximum likelihood via Monte-Carlo Markov Chain
        - 'dual_annealing'
        - see https://lmfit.github.io/lmfit-py/fitting.html for more options

    verbose : boolean
        whether to print out results and statistics.  Default is true.

    Returns
    -------
    best_fit : class
        the best fitting model
    """
    #fit the data
    best_fit = g_model.fit(flux, pars, x=wavelength, weights=weights, method=method)

    #print out the fit report
    if verbose:
        print(best_fit.fit_report())

    return best_fit


#===============================================================================
#GAUSSIAN FUNCTION
#===============================================================================
def gaussian_func(x, amp, mean, sigma):
    """
    Defines the 1D gaussian function.

    Parameters
    ----------
    x : :obj:'~numpy.ndarray'
        the x-values or wavelength vector

    amp : float
        amplitude or height of the gaussian

    mean : float
        central point of the gaussian

    sigma : float
        the standard deviation of the gaussian

    Returns
    -------
    gaussian : :obj:'~numpy.ndarray'
        the gaussian function
    """
    gaussian = amp * np.exp(-(x-mean)**2/(2*sigma**2))

    return gaussian


#===============================================================================
#SIMPLE FITTING FUNCTIONS
#===============================================================================
def gaussian1(wavelength, flux, amp_guess=None, mean_guess=None, sigma_guess=None):
    """
    Creates a single gaussian to fit to the emission line

    Parameters
    ----------
    wavelength : :obj:'~numpy.ndarray'
        the wavelength vector

    flux : :obj:'~numpy.ndarray'
        the flux of the spectrum

    amp_guess : [float]
        guess for the overall height of the peak. If None, uses maximum value of
        the flux array. (default = None)

    mean_guess : [float]
        guess for the central position of the peak, usually the observed
        wavelength of the emission line.  If None, uses the wavelength position
        of the maximum value from the flux array. (default = None)

    sigma_guess : [float]
        guess for the characteristic width of the peak.  If None, uses a value
        of 0.9 Angstroms. (default = None)

    Returns
    -------
    g_model : the Gaussian model

    pars : the Parameters object
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
    wavelength : :obj:'~numpy.ndarray'
        the wavelength vector

    flux : :obj:'~numpy.ndarray'
        the flux of the spectrum

    amplitude_guess : list of floats
        guesses for the amplitudes of the two gaussians.  If None, uses
        0.7*max(flux)/0.4 for the narrow gaussian, and 0.3*max(flux)/0.4 for the
        broad gaussian. (default = None)

    mean_guess : list of floats
        guesses for the central positions of the two gaussians.  If None, uses
        the wavelength position of the maximum value from the flux array for the
        narrow gaussian, and 0.1 Angstroms less for the broad gaussian.
        (default = None)

    sigma_guess : list of floats
        guesses for the characteristic widths of the two gaussians.  If None, uses
        1.0A for the narrow gaussian and 3.5A for the broad gaussian.
        (default = None)

    mean_diff : list of floats
        guess for the difference between the means, and the amount by which
        that can change eg. [2.5Angstroms, 0.5Angstroms]. If None, assumed to be
        [0.0Angstroms, 5.0Angstroms]. (default = None)

    sigma_variations : list of floats
        the amount by which we allow sigma to vary from the guess e.g. 0.5Angstroms.
        If None, assumes a sigma min of 0.8A and max of 10A for the broad gaussian,
        and min of 0.9A and max of 2.0A for the narrow gaussian. (default=None)

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
        pars['Galaxy_sigma'].set(max=(sigma_guess[0]+sigma_variations), min=(max((0.8, sigma_guess[0]-sigma_variations))), vary=True)
        pars['Flow_sigma'].set(max=(sigma_guess[1]+sigma_variations), min=(max((0.8, sigma_guess[1]-sigma_variations))), vary=True)
    if sigma_variations is None:
        #no super duper wide outflows
        pars['Flow_sigma'].set(max=10.0, min=0.8, vary=True)
        #edited this to fit Halpha... remember to change back!!!
        #pars['Flow_sigma'].set(max=3.5, min=0.8, vary=True)
        #also, since each wavelength value is roughly 0.5A apart, the sigma must be more than 0.25A
        pars['Galaxy_sigma'].set(min=0.8, max=2.0, vary=True)#min=0.8 because that's the minimum we can observe with the telescope


    return g_model, pars


#===============================================================================
#SIMPLE FITTING FUNCTIONS WITH CONSTANT
#===============================================================================
def gaussian1_const(wavelength, flux, amp_guess=None, mean_guess=None, sigma_guess=None):
    """
    Creates a single gaussian to fit to the emission line with a constant for
    the continuum level

    Parameters
    ----------
    wavelength : :obj:'~numpy.ndarray'
        the wavelength vector

    flux : :obj:'~numpy.ndarray'
        the flux of the spectrum

    amp_guess : [float]
        guess for the overall height of the peak. If None, uses maximum value of
        the flux array. (default = None)

    mean_guess : [float]
        guess for the central position of the peak, usually the observed
        wavelength of the emission line.  If None, uses the wavelength position
        of the maximum value from the flux array. (default = None)

    sigma_guess : [float]
        guess for the characteristic width of the peak.  If None, uses a value
        of 0.9 Angstroms. (default = None)

    Returns
    -------
    g_model :
        the Gaussian model

    pars :
        the Parameters object
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
    Creates a combination of 2 gaussians with a constant for the continuum level

    Parameters
    ----------
    wavelength : :obj:'~numpy.ndarray'
        the wavelength vector

    flux : :obj:'~numpy.ndarray'
        the flux of the spectrum

    amplitude_guess : list of floats
        guesses for the amplitudes of the two gaussians.  If None, uses
        0.7*max(flux)/0.4 for the narrow gaussian, and 0.3*max(flux)/0.4 for the
        broad gaussian. (default = None)

    mean_guess : list of floats
        guesses for the central positions of the two gaussians.  If None, uses
        the wavelength position of the maximum value from the flux array for the
        narrow gaussian, and 0.1 Angstroms less for the broad gaussian.
        (default = None)

    sigma_guess : list of floats
        guesses for the characteristic widths of the two gaussians.  If None, uses
        1.0A for the narrow gaussian and 3.5A for the broad gaussian.
        (default = None)

    mean_diff : list of floats
        guess for the difference between the means, and the amount by which
        that can change eg. [2.5Angstroms, 0.5Angstroms]. If None, assumed to be
        [0.0Angstroms, 5.0Angstroms]. (default = None)

    sigma_variations : list of floats
        the amount by which we allow sigma to vary from the guess e.g. 0.5Angstroms.
        If None, assumes a sigma min of 0.8A and max of 10A for the broad gaussian,
        and min of 0.9A and max of 2.0A for the narrow gaussian. (default=None)

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
        pars['Galaxy_sigma'].set(max=(sigma_guess[0]+sigma_variations), min=(max((0.8, sigma_guess[0]-sigma_variations))), vary=True)
        pars['Flow_sigma'].set(max=(sigma_guess[1]+sigma_variations), min=(max((0.8, sigma_guess[1]-sigma_variations))), vary=True)
    if sigma_variations is None:
        #no super duper wide outflows
        pars['Flow_sigma'].set(max=10.0, min=0.8, vary=True)
        #edited this to fit Halpha... remember to change back!!!
        #pars['Flow_sigma'].set(max=3.5, min=0.8, vary=True)
        #also, since each wavelength value is roughly 0.5A apart, the sigma must be more than 0.25A
        pars['Galaxy_sigma'].set(min=0.8, max=2.0, vary=True)#min=2.0 because that's the minimum we can observe with the telescope

    return g_model, pars


#===============================================================================
#OII DOUBLET FITTING FUNCTIONS
#===============================================================================
def gaussian1_OII_doublet(wavelength, flux, amplitude_guess=None, mean_guess=None, sigma_guess=None, sigma_variations=None):
    """
    Creates a combination of 2 gaussians and a constant to fit the OII doublet

    Parameters
    ----------
    wavelength : :obj:'~numpy.ndarray'
        the wavelength vector

    flux : :obj:'~numpy.ndarray'
        the flux of the spectrum

    amplitude_guess : list of floats
        guesses for the amplitudes of the two gaussians.  If None, uses
        0.8*max(flux)/0.4 for the red gaussian, and restricts the blue gaussian
        to have an amplitude such that amp_red/amp_blue is between 0.7 and 1.4.
        (default = None)

    mean_guess : list of floats
        guesses for the central positions of the two gaussians.  If None, uses
        the wavelength position of the maximum value from the flux array for the
        red gaussian, and 2.783 Angstroms less for the blue gaussian.
        (default = None)

    sigma_guess : list of floats
        guesses for the characteristic widths of the two gaussians.  If None, uses
        1.0A for the red gaussian and the blue gaussian is set to always have the
        same sigma as the red.  (default = None)

    sigma_variations : list of floats
        the amount by which we allow sigma to vary from the guess e.g. 0.5Angstroms.
        If None, assumes a sigma min of 0.8A and max of 2.0A. (default=None)

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
    #the amplitudes need to be tied such that the ratio of 3729/3726 is between 0.7 and 1.4
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
    wavelength : :obj:'~numpy.ndarray'
        the wavelength vector

    flux : :obj:'~numpy.ndarray'
        the flux of the spectrum

    amplitude_guess : list of floats
        guesses for the amplitudes of the two gaussians.  If None, uses
        0.7*max(flux)/0.4 for the red narrow gaussian, and restricts the blue
        narrow gaussian to have an amplitude such that amp_red/amp_blue is between
        0.7 and 1.4.  Similarly, uses 0.3*max(flux)/0.4 for the red broad gaussian,
        and restricts the blue broad gaussian to have an amplitude such that
        amp_red/amp_blue is between 0.7 and 1.4.  (default = None)

    mean_guess : list of floats
        guesses for the central positions of the red gaussians.  If None, uses
        the wavelength position of the maximum value from the flux array for the
        red narrow and broad gaussians, and 2.783 Angstroms less for the blue narrow
        and broad gaussians.  (default = None)

    sigma_guess : list of floats
        guesses for the characteristic widths of the two gaussians.  If None, uses
        1.0A for the red gaussian and the blue gaussian is set to always have the
        same sigma as the red.  (default = None)

    mean_diff : list of floats
        guess for the difference between the means for the narrow and broad gaussians,
        and the amount by which that can change eg. [2.5Angstroms, 0.5Angstroms].
        The mean_diff between narrow and broad is always the same for blue as it
        is for the red pair.  If None, assumed to be [0.0Angstroms, 5.0Angstroms].
        (default = None)

    sigma_variations : list of floats
        the amount by which we allow sigma to vary from the guess e.g. 0.5Angstroms.
        If None, assumes a sigma min of 0.8A and max of 2.0A for the narrow
        gaussians, and a sigma min of 0.8A and max of 10A for the broad gaussians.
        (default=None)

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


#===============================================================================
#NaD ABSORPTION DOUBLET FITTING FUNCTION
#===============================================================================
def gaussian_NaD(wavelength, flux, amplitude_guess=None, mean_guess=None, sigma_guess=None):
    """
    Creates a combination of 4 gaussians which fit the NaD absorption line.
    Order: galaxy1, galaxy2, flow1, flow2

    Parameters
    ----------
    wavelength : :obj:'~numpy.ndarray'
        the wavelength vector

    flux : :obj:'~numpy.ndarray'
        the flux of the spectrum

    amplitude_guess : list of floats
        guesses for the amplitudes of the two gaussians.  If None, uses
        0.7*min(flux)/0.4 for the narrow gaussians, and uses 0.3*min(flux)/0.4
        for the broad gaussians.  (default = None)

    mean_guess : list of floats
        guesses for the central positions of the gaussians.  If None, uses the
        wavelength position of the minimum value from the flux array for the blue
        narrow gaussian, with the red narrow gaussian set 6A away.  The broad
        gaussians are then set to be 0.1A to the blue of their associated narrow
        gaussian.  (default = None)

    sigma_guess : list of floats
        guesses for the characteristic widths of the two gaussians.  If None, uses
        1.0A for the narrow gaussians, and 3.5A for the broad gaussians.
        (default = None)

    Returns
    -------
    g_model :
        the double Gaussian model

    pars :
        the Parameters object
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
