"""
NAME:
	fit_hbeta_flux.py

AUTHOR:
	Bronwyn Reichardt Chu
	Swinburne
	2019

EMAIL:
	<breichardtchu@swin.edu.au>

PURPOSE:
	To fit gaussians to the Hbeta emission line in 3D data cubes.
	Written on MacOS Mojave 10.14.5, with Python 3

MODIFICATION HISTORY:
		v.1.0 - first created November 2020

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
import matplotlib.pyplot as plt

from astropy.io import fits

from . import koffee


#===============================================================================
#HBETA FLUX FUNCTION
#===============================================================================

def calculate_hbeta_flux(hbeta_results, hbeta_error):
    """
    Calculates the Hbeta flux from the fitting results

    Parameters
    ----------
    hbeta_results : array
        Array of results from fitting
    hbeta_error : array
        Array of errors from fitting

    Returns
    -------
    hbeta_flux : array
        Array of hbeta fluxes
    hbeta_error : array
        Array of hbeta flux errors
    """
    #get the sigma
    sigma = hbeta_results[0,:,:]

    sigma_err = hbeta_error[0,:,:]

    #get the amplitude
    amp = hbeta_results[2,:,:]

    amp_err = hbeta_error[2,:,:]

    #calculate the flux
    hbeta_flux = np.sqrt(2*np.pi) * sigma * amp

    #calculate the error
    hbeta_flux_error = hbeta_flux * np.sqrt(2*np.pi) * np.sqrt((sigma_err/sigma)**2 + (amp_err/amp)**2)

    return hbeta_flux, hbeta_flux_error




#===============================================================================
#FIT ENTIRE CUBE
#===============================================================================

def fit_cube(galaxy_name, redshift, output_folder_loc, filename=None, filename2=None, data_cube_stuff=None, cont_subtract=False, include_const=False, plotting=True, method='leastsq'):
    """
    Fits the entire cube, and checks whether one or two gaussians fit the emission line best.  Must have either the filename to the fits file, or the data_cube_stuff.

    Parameters
    ----------
    galaxy_name : str
        name of the galaxy
    redshift : int
        redshift of the galaxy
    output_folder_loc : str
        file path to where to put the results and plots output folder
    filename : str
        the file path to the data cube - if data_cube_stuff is not given
    filename2 : str
        the file path to the second data cube - generally the continuum subtracted cube.
        If this is not None, the first cube is used to create the S/N mask, and
        this cube is used for fitting.
    data_cube_stuff :
        [lamdas, data] if the filename is not given
    cont_subtract : bool
        when True, use the first 10 pixels in the spectrum to define the continuum
        and subtracts it.  Use False when continuum has already been fit and subtracted.
    plotting : bool
        when True, each best_fit is plotted with its residuals and saved
    method : str
        the fitting method (see )

    Returns
    -------
    hbeta_results : :obj:'~numpy.ndarray' object
        array with galaxy sigma, center, amplitude and outflow sigma, center, amplitude values in the same spatial shape as the input data array
    hbeta_error : :obj:'~numpy.ndarray'
        array with galaxy sigma, center, amplitude and outflow sigma, center, amplitude errors in the same spatial shape as the input data array
    """
    #get the original data to create the S/N mask
    if filename:
        lamdas, data, header = koffee.load_data(filename)
    elif data_cube_stuff:
        lamdas, data = data_cube_stuff
    else:
        print('Data input not understood.  Need either a fits file path or the [lamdas, data] in data_cube_stuff.')
        return

    #create filepath to output folder
    output_folder_loc = output_folder_loc+galaxy_name+'koffee_hbeta_fits_'+str(date.today())+'/'
    pathlib.Path(output_folder_loc).mkdir(exist_ok=True)

    #create arrays to store results in, with columns for amplitude, mean and standard deviation of gaussians
    #emission line and outflow
    if include_const == True:
        hbeta_results = np.empty_like(data[:4,:,:])
        hbeta_error = np.empty_like(data[:4,:,:])

    elif include_const == False:
        hbeta_results = np.empty_like(data[:3,:,:])
        hbeta_error = np.empty_like(data[:3,:,:])

    #save the chi squared values
    chi_square = np.empty_like(data[:1,:,:])

    #use redshift to shift emission line to observed wavelength
    em_rest = 4862.68
    em_observed = em_rest*(1+redshift)

    #create mask to choose emission line wavelength range
    mask_lam = (lamdas > em_observed-20.) & (lamdas < em_observed+20.)

    #create the S/N array
    sn_array = np.median(data[mask_lam,:,:][:10,:,:], axis=0)/np.std(data[mask_lam,:,:][:10,:,:], axis=0)

    if filename2:
        lamdas, data, header = koffee.load_data(filename2)
        print('Second filename being used for halpha fits')
        #recreate mask to choose emission line wavelength range with new lamdas
        mask_lam = (lamdas > em_observed-20.) & (lamdas < em_observed+20.)

    #loop through cube
    with tqdm(total=data.shape[1]*data.shape[2]) as pbar:
        for i in range(data.shape[1]):
            for j in range(data.shape[2]):
                #apply mask to data
                flux = data[:,i,j][mask_lam]
                #apply mask to lamdas
                masked_lamdas = lamdas[mask_lam]

                #take out the continuum by finding the average value of the first 10 pixels in the data and minusing them off
                if cont_subtract==True:
                    continuum = np.median(flux[:10])
                    flux = flux-continuum

                #only fit if the S/N is greater than 1
                if sn_array[i,j] >= 1:
                    #create model for 1 Gaussian fit
                    if include_const == True:
                        g_model1, pars1 = koffee.gaussian1_const(masked_lamdas, flux)
                    elif include_const == False:
                        g_model1, pars1 = koffee.gaussian1(masked_lamdas, flux)
                    #fit model for 1 Gaussian fit
                    best_fit1 = koffee.fitter(g_model1, pars1, masked_lamdas, flux, method=method, verbose=False)


                    #save results
                    if include_const == True:
                        hbeta_results[:,i,j] = (best_fit1.params['gauss_sigma'].value, best_fit1.params['gauss_mean'].value, best_fit1.params['gauss_amp'].value, best_fit1.params['Constant_Continuum_c'].value)
                        hbeta_error[:,i,j] = (best_fit1.params['gauss_sigma'].stderr, best_fit1.params['gauss_mean'].stderr, best_fit1.params['gauss_amp'].stderr, best_fit1.params['Constant_Continuum_c'].stderr)
                    elif include_const == False:
                        hbeta_results[:,i,j] = (best_fit1.params['gauss_sigma'].value, best_fit1.params['gauss_mean'].value, best_fit1.params['gauss_amp'].value)
                        hbeta_error[:,i,j] = (best_fit1.params['gauss_sigma'].stderr, best_fit1.params['gauss_mean'].stderr, best_fit1.params['gauss_amp'].stderr)

                    chi_square[:,i,j] = best_fit1.bic

                    #plot fit results
                    if plotting == True:
                        fig1 = koffee.plot_fit(masked_lamdas, flux, g_model1, pars1, best_fit1, plot_initial=False, include_const=include_const)
                        fig1.suptitle(r'H$\beta$')
                        fig1.savefig(output_folder_loc+galaxy_name+'_best_fit_hbeta_no_outflow_'+str(i)+'_'+str(j))
                        plt.close(fig1)


                #if the S/N is less than 20:
                else:
                    print('S/N for '+str(i)+', '+str(j)+' is '+str(sn_array[i,j]))

                    #chi squared for the fits
                    chi_square[:,i,j] = np.nan


                    if include_const == True:
                        hbeta_results[:,i,j] = (np.nan, np.nan, np.nan, np.nan)
                        hbeta_error[:,i,j] = (np.nan, np.nan, np.nan, np.nan)

                    elif include_const == False:
                        hbeta_results[:,i,j] = (np.nan, np.nan, np.nan)
                        hbeta_error[:,i,j] = (np.nan, np.nan, np.nan)

                #update progress bar
                pbar.update(1)

    #calculate the hbeta flux
    hbeta_flux, hbeta_flux_error = calculate_hbeta_flux(hbeta_results, hbeta_error)

    #save all of the results to text files
    if include_const == True:
        np.savetxt(output_folder_loc+galaxy_name+'_hbeta_results.txt', np.reshape(hbeta_results, (4, -1)))
        np.savetxt(output_folder_loc+galaxy_name+'_hbeta_error.txt', np.reshape(hbeta_error, (4, -1)))

    elif include_const == False:
        np.savetxt(output_folder_loc+galaxy_name+'_hbeta_results.txt', np.reshape(hbeta_results, (3, -1)))
        np.savetxt(output_folder_loc+galaxy_name+'_hbeta_error.txt', np.reshape(hbeta_error, (3, -1)))

    np.savetxt(output_folder_loc+galaxy_name+'_hbeta_chi_squared.txt', np.reshape(chi_square, (2,-1)))

    np.savetxt(output_folder_loc+galaxy_name+'_hbeta_flux.txt', hbeta_flux)
    np.savetxt(output_folder_loc+galaxy_name+'_hbeta_flux_error.txt', hbeta_flux_error)

    #save the hbeta flux results as a fits file
    hdu1 = fits.PrimaryHDU(hbeta_flux, header=header)
    hdu2 = fits.ImageHDU(hbeta_flux_error)
    hdul = fits.HDUList([hdu1, hdu2])
    hdul.writeto(output_folder_loc+galaxy_name+'_hbeta_flux.fits')



    return hbeta_results, hbeta_error, chi_square, hbeta_flux, hbeta_flux_error
