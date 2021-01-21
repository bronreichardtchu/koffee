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

FUNCTIONS INCLUDED:
    mock_data
    check_blue_chi_square
    plot_fit
    fit_cube
    read_output_files

DICTIONARIES INCLUDED:
    all_the_lines       -   holds the wavelengths of emission lines to fit
    dodgy_spaxels       -   holds location of spaxels saturated in OIII 5007

MODIFICATION HISTORY:
    v.1.0 - first created May 2019
    v.1.0.1 - thinking about renaming this KOFFEE - Keck Outflow Fitter For Emission linEs
    v.1.0.2 - Added a continuum, being the average of the first 10 pixels in the input spectrum, which is then subtracted from the entire data spectrum so that the Gaussians fit properly, and aren't trying to fit the continuum as well (5th June 2019)
    v.1.0.3 - added a loop over the entire cube, with an exception if the paramaters object comes out weird and a progress bar
    v.1.0.4 - added a continuum to the mock data, and also added a feature so that the user can define what S/N they want the mock data to have
    v.1.0.5 - adding functions to combine data cubes either by pixel, or by regridding the wavelength (ToDo: include variance cubes in this too)

"""

import pickle
import pathlib
import numpy as np
from datetime import date
from tqdm import tqdm #progress bar module

#make sure matplotlib doesn't create any windows
#import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt

from . import prepare_cubes as pc
from . import koffee_fitting_functions as kff

from astropy.modeling import models

from lmfit import Parameters
from lmfit import Model

import importlib
importlib.reload(kff)

#===============================================================================
#MOCK DATA
#===============================================================================
def mock_data(amp, mean, stddev, snr):
    """
    Creates a mock data set with gaussians.  A list of values for each gaussian
    property is input; each property must have the same length.

    Parameters
    ----------
    amp : list of floats
        amplitudes of the Gaussians

    mean : list of floats
        means of the Gaussians

    stddev : list of floats
        standard deviations of the Gaussians

    snr : float
        the desired signal-to-noise ratio

    Returns
    -------
    x : :obj:'~numpy.ndarray'
        the wavelength vector

    y : :obj:'~numpy.ndarray'
        the flux/intensity
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



#===============================================================================
#USEFUL DICTIONARIES
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

#dictionary of spaxels which are saturated in OIII_4 and need to fit OIII_3 instead
dodgy_spaxels = {
    "IRAS08" : [(27,9), (28,9), (29,9), (30,9), (31,9), (32,9)]
}




#===============================================================================
#MAKING KOFFEE SMARTER
#===============================================================================
def check_blue_chi_square(wavelength, flux, best_fit, g_model, OII_doublet_fit=False):
    """
    Checks the chi squared value of the blue side of the fit.  If there's a large
    residual, KOFFEE will shift the starting point for the flow gaussian mean to
    the blue in fit_cube().

    Parameters
    ----------
    wavelength : :obj:'~numpy.ndarray'
        the wavelength vector

    flux : :obj:'~numpy.ndarray'
        the data vector

    best_fit : class
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
            try:
                one_sigma_blue = (best_fit.best_values['Galaxy_mean'] - best_fit.best_values['Galaxy_sigma'])-1.0
            except:
                one_sigma_blue = (best_fit.best_values['gauss_mean'] - best_fit.best_values['gauss_sigma'])-1.0

    blue_left_bound = one_sigma_blue - 4.0
    lam_mask = (wavelength > blue_left_bound) & (wavelength < one_sigma_blue)
    #print(blue_left_bound, one_sigma_blue)

    #calculate the chi squared
    chi_square = np.sum(residual[lam_mask]**2)

    return chi_square



#===============================================================================
#PLOTTING
#===============================================================================
def plot_fit(wavelength, flux, g_model, pars, best_fit, plot_initial=False, include_const=False):
    """
    Plots the fit to the data with residuals

    Parameters
    ----------
    wavelength : :obj:'~numpy.ndarray'
        wavelength vector

    flux : :obj:'~numpy.ndarray'
        the flux of the spectrum

    initial_fit :
        the initial model before fitting

    best_fit : class
        the best fitting model

    plot_initial : boolean
        Default is False. Whether to plot the initial guess or not.

    Returns
    -------
    A figure of the fit to the data, with a panel below showing the residuals
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
#MAIN FUNCTION _ APPLY TO WHOLE CUBE
def fit_cube(galaxy_name, redshift, emission_line, output_folder_loc, emission_line2=None, OII_doublet=False, filename=None, filename2=None, data_cube_stuff=None, emission_dict=all_the_lines, cont_subtract=False, include_const=False, plotting=True, method='leastsq', correct_bad_spaxels=False, koffee_checks=True):
    """
    Fits the entire cube, and checks whether one or two gaussians fit the
    emission line best.  Must have either the filename to the fits file, or the
    data_cube_stuff.

    Parameters
    ----------
    galaxy_name : str
        name of the galaxy

    redshift : int
        redshift of the galaxy

    emission_line : str
        the emission line to be fit. Options:
        "Hdelta", "Hgamma", "Hbeta", "Halpha", "OII_1", "OII_2", "HeI", "SII",
        "OIII_1", "OIII_2", "OIII_3", "OIII_4"

    output_folder_loc : str
        file path to where to put the results and plots output folder

    emission_line2 : str
        the second emission line to be fit using the results from the first.
        Default is None. Options:
        "Hdelta", "Hgamma", "Hbeta", "Halpha", "OII_1", "OII_2", "HeI", "SII",
        "OIII_1", "OIII_2", "OIII_3", "OIII_4"

    OII_doublet : boolean
        whether to fit the OII doublet using the results from the first fit.
        Default is False. Uses "OII_1", from the dictionary

    filename : str
        the file path to the data cube - if data_cube_stuff is not given

    filename2 : str
        the file path to the second data cube - generally the continuum subtracted
        cube. If this is not None, the first cube is used to create the S/N mask,
        and this cube is used for fitting.

    data_cube_stuff : list of :obj:'~numpy.ndarray'
        [lamdas, data] if the filename is not given

    emission_dict : dict
        dictionary of emission lines

    cont_subtract : bool
        when True, use the first 10 pixels in the spectrum to define the continuum
        and subtracts it.  Use False when continuum has already been fit and
        subtracted.

    include_const : bool
        when True, a constant is included in the gaussian fit.  Default is False.

    plotting : bool
        when True, each best_fit is plotted with its residuals and saved

    method : str
        the fitting method (see )

    correct_bad_spaxels : bool
        Default is False. Takes spaxels (28, 9), (29, 9) and (30, 9), which are
        saturated in IRAS08, and uses [OIII]4960 rather than [OIII]5007

    koffee_checks : bool
        when True, KOFFEE's extra tests, like the blue-side chi squared residual
        test, are applied and the spaxels which fail are refit with different
        prior guesses.  Default is True.

    Returns
    -------
    outflow_results : :obj:'~numpy.ndarray' object
        array with galaxy sigma, center, amplitude and outflow sigma, center,
        amplitude values in the same spatial shape as the input data array

    outflow_error : :obj:'~numpy.ndarray'
        array with galaxy sigma, center, amplitude and outflow sigma, center,
        amplitude errors in the same spatial shape as the input data array

    no_outflow_results : :obj:'~numpy.ndarray'
        array with single gaussian sigma, center, amplitude values in the same
        spatial shape as the input data array

    no_outflow_error : :obj:'~numpy.ndarray'
        array with single gaussian sigma, center, amplitude errors in the same
        spatial shape as the input data array

    statistical results : :obj:'~numpy.ndarray'
        array with 0 where one gaussian gave a better BIC value, and 1 where two
        gaussians gave a better BIC value.

    chi_square :obj:'~numpy.ndarray' object
        array with the chi square values for the [single gaussian, double gaussian]
        fits in the same spatial shape as the input data array

    outflow_results2 : :obj:'~numpy.ndarray' object
        array with galaxy sigma, center, amplitude and outflow sigma, center,
        amplitude values in the same spatial shape as the input data array, for
        the second emission line fit, if emission_line2 is not None.

    outflow_error2 : :obj:'~numpy.ndarray'
        array with galaxy sigma, center, amplitude and outflow sigma, center,
        amplitude errors in the same spatial shape as the input data array for
        the second emission line fit, if emission_line2 is not None.

    no_outflow_results2 : :obj:'~numpy.ndarray'
        array with single gaussian sigma, center, amplitude values in the same
        spatial shape as the input data array for the second emission line fit,
        if emission_line2 is not None.

    no_outflow_error2 : :obj:'~numpy.ndarray'
        array with single gaussian sigma, center, amplitude errors in the same
        spatial shape as the input data array for the second emission line fit,
        if emission_line2 is not None.

    chi_square2 :obj:'~numpy.ndarray' object
        array with the chi square values for the [single gaussian, double gaussian]
        fits in the same spatial shape as the input data array for the second
        emission line fit, if emission_line2 is not None.

    outflow_results3 : :obj:'~numpy.ndarray' object
        array with galaxy sigma, center, amplitude and outflow sigma, center,
        amplitude values in the same spatial shape as the input data array for
        the OII doublet line fit, if OII_doublet is True.

    outflow_error3 : :obj:'~numpy.ndarray'
        array with galaxy sigma, center, amplitude and outflow sigma, center,
        amplitude errors in the same spatial shape as the input data array for
        the OII doublet line fit, if OII_doublet is True.

    no_outflow_results3 : :obj:'~numpy.ndarray'
        array with single gaussian sigma, center, amplitude values in the same
        spatial shape as the input data array for the OII doublet line fit, if
        OII_doublet is True.

    no_outflow_error3 : :obj:'~numpy.ndarray'
        array with single gaussian sigma, center, amplitude errors in the same
        spatial shape as the input data array for the OII doublet line fit, if
        OII_doublet is True.

    chi_square3 :obj:'~numpy.ndarray' object
        array with the chi square values for the [single gaussian, double gaussian]
        fits in the same spatial shape as the input data array for the OII doublet
        line fit, if OII_doublet is True.
    """
    #get the original data to create the S/N mask
    if filename:
        fits_stuff = pc.load_data(filename, mw_correction=False)
        if len(fits_stuff) > 3:
            lamdas, data, var, header = fits_stuff
        else:
            lamdas, data, header = fits_stuff
    elif data_cube_stuff:
        lamdas, data = data_cube_stuff
    else:
        print('Data input not understood.  Need either a fits file path or the [lamdas, data] in data_cube_stuff.')
        return

    #create filepath to output folder
    output_folder_loc = output_folder_loc+galaxy_name+'koffee_results_'+emission_line+'_'+str(date.today())+'/'
    pathlib.Path(output_folder_loc).mkdir(exist_ok=True)

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
        fits_stuff = pc.load_data(filename2, mw_correction=False)
        if len(fits_stuff) > 3:
            lamdas, data, var, header = fits_stuff
        else:
            lamdas, data, header = fits_stuff
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
                        #print('Spaxel '+str(i)+','+str(j)+' has S/N > 20')
                        #create model for 1 Gaussian fit
                        if include_const == True:
                            g_model1, pars1 = kff.gaussian1_const(masked_lamdas, flux)
                        elif include_const == False:
                            g_model1, pars1 = kff.gaussian1(masked_lamdas, flux)

                        #fit model for 1 Gaussian fit
                        best_fit1 = kff.fitter(g_model1, pars1, masked_lamdas, flux, method=method, verbose=False)

                        #create and fit model for 2 Gaussian fit, using 1 Gaussian fit as first guess for the mean
                        if include_const == True:
                            g_model2, pars2 = kff.gaussian2_const(masked_lamdas, flux, amplitude_guess=None, mean_guess=None, sigma_guess=None)
                        elif include_const == False:
                            g_model2, pars2 = kff.gaussian2(masked_lamdas, flux, amplitude_guess=None, mean_guess=None, sigma_guess=None)
                        best_fit2 = kff.fitter(g_model2, pars2, masked_lamdas, flux, method=method, verbose=False)


                        if best_fit2.bic < (best_fit1.bic-10):
                            stat_res = 1
                            #save blue chi square
                            blue_chi_square[i,j] = check_blue_chi_square(masked_lamdas, flux, best_fit2, g_model2)
                        else:
                            stat_res = 0
                            #save blue chi square
                            blue_chi_square[i,j] = check_blue_chi_square(masked_lamdas, flux, best_fit1, g_model1)

                        if koffee_checks == True:
                            if blue_chi_square[i,j] > 0.1:
                                if include_const == True:
                                    g_model2_refit, pars2_refit = kff.gaussian2_const(masked_lamdas, flux, amplitude_guess=None, mean_guess=[masked_lamdas[flux.argmax()], masked_lamdas[flux.argmax()]-4.0], sigma_guess=[1.0,8.0])
                                elif include_const == False:
                                    g_model2_refit, pars2_refit = kff.gaussian2(masked_lamdas, flux, amplitude_guess=None, mean_guess=[masked_lamdas[flux.argmax()], masked_lamdas[flux.argmax()]-4.0], sigma_guess=[1.0,8.0])
                                best_fit2_refit = kff.fitter(g_model2_refit, pars2_refit, masked_lamdas, flux, method=method, verbose=False)

                                #force it to take the new fit
                                stat_res = 2



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
                                    g_model1_second, pars1_second = kff.gaussian1_const(masked_lamdas2, flux2, amp_guess=None, mean_guess=None, sigma_guess=None)

                                    #for the two gaussian fit
                                    g_model2_second, pars2_second = kff.gaussian2_const(masked_lamdas2, flux2, amplitude_guess=None, mean_guess=[masked_lamdas2[flux2.argmax()], masked_lamdas2[flux2.argmax()]-mean_diff], sigma_guess=sigma_guess, mean_diff=[mean_diff, 1.5], sigma_variations=1.5)
                                elif include_const == False:
                                    #for the one gaussian fit
                                    g_model1_second, pars1_second = kff.gaussian1(masked_lamdas2, flux2, amp_guess=None, mean_guess=None, sigma_guess=None)
                                    #for the two gaussian fit
                                    g_model2_second, pars2_second = kff.gaussian2(masked_lamdas2, flux2, amplitude_guess=None, mean_guess=[masked_lamdas2[flux2.argmax()], masked_lamdas2[flux2.argmax()]-mean_diff], sigma_guess=sigma_guess, mean_diff=[mean_diff, 1.5], sigma_variations=1.5)

                                #do the fit
                                #for the one gaussian fit
                                best_fit1_second = kff.fitter(g_model1_second, pars1_second, masked_lamdas2, flux2, method=method, verbose=False)

                                #for the two gaussian fit
                                best_fit2_second = kff.fitter(g_model2_second, pars2_second, masked_lamdas2, flux2, method=method, verbose=False)

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


                                if koffee_checks == True:
                                    #check the fit using the blue-side-residual test
                                    blue_chi_square_check = check_blue_chi_square(masked_lamdas2, flux2, best_fit2_second, g_model2_second)

                                    if blue_chi_square_check > 0.2:
                                        #refit using...
                                        print('Refitting', emission_line2, 'fit for spaxel ', str(i), str(j))
                                        print('This spaxel had blue_chi_square_check ', str(blue_chi_square_check))
                                        #create the fitting objects
                                        if include_const == True:
                                            g_model2_refit_second, pars2_refit_second = kff.gaussian2_const(masked_lamdas2, flux2, amplitude_guess=None, mean_guess=[masked_lamdas2[flux2.argmax()], masked_lamdas2[flux2.argmax()]-mean_diff], sigma_guess=sigma_guess, mean_diff=[mean_diff, 0.5], sigma_variations=0.5)
                                        elif include_const == False:
                                            g_model2_refit_second, pars2_refit_second = kff.gaussian2(masked_lamdas2, flux2, amplitude_guess=None, mean_guess=[masked_lamdas2[flux2.argmax()], masked_lamdas2[flux2.argmax()]-mean_diff], sigma_guess=sigma_guess, mean_diff=[mean_diff, 0.5], sigma_variations=0.5)
                                        #do the fit
                                        best_fit2_refit_second = kff.fitter(g_model2_refit_second, pars2_refit_second, masked_lamdas2, flux2, method=method, verbose=False)

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
                                    #is greater than 90% of the galaxy amplitude
                                    if best_fit2_second.params['Flow_amp'].value > 0.9*best_fit2_second.params['Galaxy_amp'].value:
                                        print('Doing one Gaussian fit for spaxel ', str(i), str(j))
                                        #create the fitting objects
                                        if include_const == True:
                                            g_model1_refit_second, pars1_refit_second = kff.gaussian1_const(masked_lamdas2, flux2)#, amp_guess=None, mean_guess=masked_lamdas2[flux2.argmax()])#, sigma_guess=sigma_guess[0])
                                        elif include_const == False:
                                            g_model1_refit_second, pars1_refit_second = kff.gaussian1(masked_lamdas2, flux2)#, amp_guess=None, mean_guess=masked_lamdas2[flux2.argmax()], sigma_guess=sigma_guess[0])

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
                                g_model1_third, pars1_third = kff.gaussian1_OII_doublet(masked_lamdas3, flux3, amplitude_guess=None, mean_guess=[masked_lamdas3[np.argmax(flux3)]+1.0], sigma_guess=sigma_guess, sigma_variations=1.5)

                                #do the fit
                                best_fit1_third = kff.fitter(g_model1_third, pars1_third, masked_lamdas3, flux3, method=method, verbose=False)

                                #plot the fit
                                fig4 = plot_fit(masked_lamdas3, flux3, g_model1_third, pars1_third, best_fit1_third, plot_initial=False, include_const=True)
                                fig4.suptitle('OII doublet ['+str(em_rest3)+', '+str(emission_dict['OII_2']*(1+redshift))+'] not including outflows')
                                fig4.savefig(output_folder_loc+galaxy_name+'_best_fit_OII_doublet_no_outflow_'+str(i)+'_'+str(j))
                                plt.close(fig4)

                                #create the fitting objects for the four gaussian fit
                                g_model2_third, pars2_third = kff.gaussian2_OII_doublet(masked_lamdas3, flux3, amplitude_guess=None, mean_guess=None, sigma_guess=sigma_guess, mean_diff=[mean_diff, 1.5], sigma_variations=1.5)

                                #do the fit
                                best_fit2_third = kff.fitter(g_model2_third, pars2_third, masked_lamdas3, flux3, method=method, verbose=False)

                                #plot the fit
                                fig4 = plot_fit(masked_lamdas3, flux3, g_model2_third, pars2_third, best_fit2_third, plot_initial=False, include_const=True)
                                fig4.suptitle('OII doublet ['+str(em_rest3)+', '+str(emission_dict['OII_2']*(1+redshift))+']')
                                fig4.savefig(output_folder_loc+galaxy_name+'_best_fit_OII_doublet_outflow_'+str(i)+'_'+str(j))
                                plt.close(fig4)

                                #check the fit using the blue-side-residual test
                                if koffee_checks == True:
                                    blue_chi_square_check = check_blue_chi_square(masked_lamdas3, flux3, best_fit2_third, g_model2_third, OII_doublet_fit=True)
                                    print('OII doublet blue chi square check for spaxel ', str(i), str(j), ' is ', str(blue_chi_square_check))

                                    if blue_chi_square_check > 100.0:
                                        #refit using...
                                        print('Refitting OII doublet fit for spaxel ', str(i), str(j))
                                        print('This spaxel had blue_chi_square_check ', str(blue_chi_square_check))
                                        #create the fitting objects
                                        g_model2_refit_third, pars2_refit_third = kff.gaussian2_OII_doublet(masked_lamdas3, flux3, amplitude_guess=None, mean_guess=[masked_lamdas3[np.argmax(flux3)]+1.0], sigma_guess=sigma_guess, mean_diff=[mean_diff, 1.0], sigma_variations=0.25)

                                        #do the fit
                                        best_fit2_refit_third = kff.fitter(g_model2_refit_third, pars2_refit_third, masked_lamdas3, flux3, method=method, verbose=False)

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
                                if stat_res > 0:
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
                        #print('S/N for '+str(i)+', '+str(j)+' is '+str(sn_array[i,j]))
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
# READ IN OUTFPUT FILES
#===============================================================================

def read_output_files(output_folder, galaxy_name, spatial_shape, include_const=True, emission_line1='OIII_4', emission_line2=None, OII_doublet=False):
    """
    Read in the output files and return numpy arrays (useful when working in
    ipython terminal)

    Parameters
    ----------
    output_folder : str
        location of the results folder

    galaxy_name : str
        the galaxy name or descriptor used in the results files

    spatial_shape : list of int
        the spatial shape of the array.  e.g. for IRAS08 [67, 24]

    include_const : boolean
        whether a constant was included in the gaussian fitting.  This is important
        for reading the arrays into the correct shape. Default is True.

    emission_line1 : str
        the name of the first emission line fit, used in the filenames.  Default
        is 'OIII_4'.  Options:
        "Hdelta", "Hgamma", "Hbeta", "Halpha", "OII_1", "OII_2", "HeI", "SII",
        "OIII_1", "OIII_2", "OIII_3", "OIII_4"

    emission_line2 : str
        the name of the second emission line fit, used in the filenames, or None
        if no second line was fit.  Default is None.  Options:
        "Hdelta", "Hgamma", "Hbeta", "Halpha", "OII_1", "OII_2", "HeI", "SII",
        "OIII_1", "OIII_2", "OIII_3", "OIII_4"

    OII_doublet : boolean
        whether the OII doublet was fit.  If False, the function won't search for
        those output files.  Default is False.

    Returns
    -------
    outflow_results : :obj:'~numpy.ndarray' object
        array with galaxy sigma, center, amplitude and outflow sigma, center,
        amplitude values in the same spatial shape as the input data array

    outflow_error : :obj:'~numpy.ndarray'
        array with galaxy sigma, center, amplitude and outflow sigma, center,
        amplitude errors in the same spatial shape as the input data array

    no_outflow_results : :obj:'~numpy.ndarray'
        array with single gaussian sigma, center, amplitude values in the same
        spatial shape as the input data array

    no_outflow_error : :obj:'~numpy.ndarray'
        array with single gaussian sigma, center, amplitude errors in the same
        spatial shape as the input data array

    statistical results : :obj:'~numpy.ndarray'
        array with 0 where one gaussian gave a better BIC value, and 1 where two
        gaussians gave a better BIC value.

    chi_square :obj:'~numpy.ndarray' object
        array with the chi square values for the [single gaussian, double gaussian]
        fits in the same spatial shape as the input data array

    outflow_results2 : :obj:'~numpy.ndarray' object
        array with galaxy sigma, center, amplitude and outflow sigma, center,
        amplitude values in the same spatial shape as the input data array, for
        the second emission line fit, if emission_line2 is not None.

    outflow_error2 : :obj:'~numpy.ndarray'
        array with galaxy sigma, center, amplitude and outflow sigma, center,
        amplitude errors in the same spatial shape as the input data array for
        the second emission line fit, if emission_line2 is not None.

    no_outflow_results2 : :obj:'~numpy.ndarray'
        array with single gaussian sigma, center, amplitude values in the same
        spatial shape as the input data array for the second emission line fit,
        if emission_line2 is not None.

    no_outflow_error2 : :obj:'~numpy.ndarray'
        array with single gaussian sigma, center, amplitude errors in the same
        spatial shape as the input data array for the second emission line fit,
        if emission_line2 is not None.

    chi_square2 :obj:'~numpy.ndarray' object
        array with the chi square values for the [single gaussian, double gaussian]
        fits in the same spatial shape as the input data array for the second
        emission line fit, if emission_line2 is not None.

    outflow_results3 : :obj:'~numpy.ndarray' object
        array with galaxy sigma, center, amplitude and outflow sigma, center,
        amplitude values in the same spatial shape as the input data array for
        the OII doublet line fit, if OII_doublet is True.

    outflow_error3 : :obj:'~numpy.ndarray'
        array with galaxy sigma, center, amplitude and outflow sigma, center,
        amplitude errors in the same spatial shape as the input data array for
        the OII doublet line fit, if OII_doublet is True.

    no_outflow_results3 : :obj:'~numpy.ndarray'
        array with single gaussian sigma, center, amplitude values in the same
        spatial shape as the input data array for the OII doublet line fit, if
        OII_doublet is True.

    no_outflow_error3 : :obj:'~numpy.ndarray'
        array with single gaussian sigma, center, amplitude errors in the same
        spatial shape as the input data array for the OII doublet line fit, if
        OII_doublet is True.

    chi_square3 :obj:'~numpy.ndarray' object
        array with the chi square values for the [single gaussian, double gaussian]
        fits in the same spatial shape as the input data array for the OII doublet
        line fit, if OII_doublet is True.
    """
    #first emission line files - usually [OIII]5007
    outflow_results = np.loadtxt(output_folder+galaxy_name+'_outflow_results_'+emission_line1+'.txt')
    outflow_error = np.loadtxt(output_folder+galaxy_name+'_outflow_error_'+emission_line1+'.txt')

    no_outflow_results = np.loadtxt(output_folder+galaxy_name+'_no_outflow_results_'+emission_line1+'.txt')
    no_outflow_error = np.loadtxt(output_folder+galaxy_name+'_no_outflow_error_'+emission_line1+'.txt')

    chi_square = np.loadtxt(output_folder+galaxy_name+'_chi_squared_'+emission_line1+'.txt')
    statistical_results = np.loadtxt(output_folder+galaxy_name+'_stat_results_'+emission_line1+'.txt')

    #reshape arrays
    if include_const == True:
        outflow_results = outflow_results.reshape(7, spatial_shape[0], spatial_shape[1])
        outflow_error = outflow_error.reshape(7, spatial_shape[0], spatial_shape[1])
        no_outflow_results = no_outflow_results.reshape(4, spatial_shape[0], spatial_shape[1])
        no_outflow_error = no_outflow_error.reshape(4, spatial_shape[0], spatial_shape[1])
        chi_square = chi_square.reshape(2, spatial_shape[0], spatial_shape[1])
        statistical_results = statistical_results.reshape(spatial_shape[0], spatial_shape[1])

    elif include_const == False:
        outflow_results = outflow_results.reshape(6, spatial_shape[0], spatial_shape[1])
        outflow_error = outflow_error.reshape(6, spatial_shape[0], spatial_shape[1])
        no_outflow_results = no_outflow_results.reshape(3, spatial_shape[0], spatial_shape[1])
        no_outflow_error = no_outflow_error.reshape(3, spatial_shape[0], spatial_shape[1])
        chi_square = chi_square.reshape(2, spatial_shape[0], spatial_shape[1])
        statistical_results = statistical_results.reshape(spatial_shape[0], spatial_shape[1])

    #second emission line files = Hbeta
    if emission_line2:
        outflow_results2 = np.loadtxt(output_folder+galaxy_name+'_outflow_results_'+emission_line2+'.txt')
        outflow_error2 = np.loadtxt(output_folder+galaxy_name+'_outflow_error_'+emission_line2+'.txt')

        no_outflow_results2 = np.loadtxt(output_folder+galaxy_name+'_no_outflow_results_'+emission_line2+'.txt')
        no_outflow_error2 = np.loadtxt(output_folder+galaxy_name+'_no_outflow_error_'+emission_line2+'.txt')

        chi_square2 = np.loadtxt(output_folder+galaxy_name+'_chi_squared_'+emission_line2+'.txt')

        #reshape arrays
        if include_const == True:
            outflow_results2 = outflow_results2.reshape(7, spatial_shape[0], spatial_shape[1])
            outflow_error2 = outflow_error2.reshape(7, spatial_shape[0], spatial_shape[1])
            no_outflow_results2 = no_outflow_results2.reshape(4, spatial_shape[0], spatial_shape[1])
            no_outflow_error2 = no_outflow_error2.reshape(4, spatial_shape[0], spatial_shape[1])
            chi_square2 = chi_square2.reshape(2, spatial_shape[0], spatial_shape[1])

        elif include_const == False:
            outflow_results2 = outflow_results2.reshape(6, spatial_shape[0], spatial_shape[1])
            outflow_error2 = outflow_error2.reshape(6, spatial_shape[0], spatial_shape[1])
            no_outflow_results2 = no_outflow_results2.reshape(3, spatial_shape[0], spatial_shape[1])
            no_outflow_error2 = no_outflow_error2.reshape(3, spatial_shape[0], spatial_shape[1])
            chi_square2 = chi_square2.reshape(2, spatial_shape[0], spatial_shape[1])

    #second emission line files = Hbeta
    if OII_doublet == True:
        outflow_results3 = np.loadtxt(output_folder+galaxy_name+'_outflow_results_OII_doublet.txt')
        outflow_error3 = np.loadtxt(output_folder+galaxy_name+'_outflow_error_OII_doublet.txt')

        no_outflow_results3 = np.loadtxt(output_folder+galaxy_name+'_no_outflow_results_OII_doublet.txt')
        no_outflow_error3 = np.loadtxt(output_folder+galaxy_name+'_no_outflow_error_OII_doublet.txt')

        chi_square3 = np.loadtxt(output_folder+galaxy_name+'_chi_squared_OII_doublet.txt')

        #reshape arrays
        outflow_results3 = outflow_results3.reshape(13, spatial_shape[0], spatial_shape[1])
        outflow_error3 = outflow_error3.reshape(13, spatial_shape[0], spatial_shape[1])
        no_outflow_results3 = no_outflow_results3.reshape(7, spatial_shape[0], spatial_shape[1])
        no_outflow_error3 = no_outflow_error3.reshape(7, spatial_shape[0], spatial_shape[1])
        chi_square3 = chi_square3.reshape(2, spatial_shape[0], spatial_shape[1])

    if OII_doublet==False and emission_line2:
        return outflow_results, outflow_error, no_outflow_results, no_outflow_error, statistical_results, chi_square, outflow_results2, outflow_error2, no_outflow_results2, no_outflow_error2, chi_square2

    if OII_doublet==True and emission_line2:
        return outflow_results, outflow_error, no_outflow_results, no_outflow_error, statistical_results, chi_square, outflow_results2, outflow_error2, no_outflow_results2, no_outflow_error2, chi_square2, outflow_results3, outflow_error3, no_outflow_results3, no_outflow_error3, chi_square3

    else:
        return outflow_results, outflow_error, no_outflow_results, no_outflow_error, statistical_results, chi_square



#===============================================================================
# MAIN
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
