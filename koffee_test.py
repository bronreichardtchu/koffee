"""
NAME:
	koffee_test.py
	KOFFEE - Keck Outflow Fitter For Emission linEs

AUTHOR:
	Bronwyn Reichardt Chu
	Swinburne
	2019

EMAIL:
	<breichardtchu@swin.edu.au>

PURPOSE:
	To test the KOFEE code.
	Written on Windows 7, with Python 3

MODIFICATION HISTORY:
		v.1.0 - first created July 2019

"""

from . import koffee

import pickle
import numpy as np
from datetime import date
from tqdm import tqdm #progress bar module

import corner
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from astropy.modeling import models, fitting
from astropy.io import fits
from astropy.coordinates import SkyCoord, EarthLocation
from astropy import units
from astropy.time import Time
from astropy import constants as consts

from lmfit import Parameters
from lmfit.models import GaussianModel #, LinearModel, ConstantModel


#Mock data testing
def mock_test(amp, mean, stddev, snr, changing):
	"""
	Runs through changing the mock data gaussian outflow shape until it reaches a greater than 10 difference in the bayesian information criterion between the one and two gaussian fits.

	Args:
		amp: amplitude of the Gaussian (list of floats)
		mean: mean of the Gaussian (list of floats)
		stddev: standard deviation of the Gaussian (list of floats)
		snr: the S/N ratio of the mock data
		changing: whether the amp, mean, stddev or snr is the value of the outflow gaussian which is changing
	Returns:
		A printed statement about the point where the difference between a one and two gaussian fit becomes statistically significant
	"""
	for i in np.arange(100):
		mock_x, mock_y = mock_data(amp, mean, stddev, snr)
		gauss1, pars1 = gaussian1(mock_x, mock_y)
		gauss2, pars2 = gaussian2(mock_x, mock_y)
		best_fit1 = fitter(gauss1, pars1, mock_x, mock_y, verbose=False)
		best_fit2 = fitter(gauss2, pars2, mock_x, mock_y, verbose=False)

		#if the BIC for the 2 gaussian is greater than 10 less than the 1 gaussian BIC, then keep fitting
		if best_fit2.bic > best_fit1.bic-10:
			if changing == 'amp':
				amp[1] = amp[1]+0.05

			elif changing == 'mean':
				mean[1] = mean[1]-0.01

			elif changing == 'stddev':
				stddev[1] = stddev[1]+0.01

			elif changing == 'snr':
				snr = snr+2

		else:
			print('best_fit2 BIC is statistically significant on '+str(i)+'th run')
			print('best_fit1 BIC:'+str(best_fit1.bic))
			print('best_fit2 BIC:'+str(best_fit2.bic))
			print('amp:', amp, 'mean:', mean, 'stddev:', stddev, 'S/N:', snr)
			print(best_fit2.params)
			fig = plot_fit(mock_x, mock_y, gauss2, pars2, best_fit2)
			fig.suptitle('Changing '+changing+'\nGal:'+str((amp[0],mean[0],stddev[0]))+', Out:'+str((amp[1],mean[1],stddev[1]))+', S/N:'+str(snr))
			return fig
			break



def mock_test_all_changing(gal_amp, gal_mean, gal_stddev):
	"""
	Iterates through amplitude, mean and standard deviation changing the mock data gaussian outflow shape as well as the S/N and records the bayesian information criterion of the one and two gaussian fits.

	Args:
		gal_amp: amplitude of the galaxy Gaussian
		gal_mean: mean of the galaxy Gaussian
		gal_stddev: standard deviation of the galaxy Gaussian
	Returns:
		An array of values describing where the BIC has become statistically significant
	"""
	#create results array (snr,amp,mean,stddev)
	results_array = np.empty((10,25,21,20))
	amp_array = np.empty_like(results_array)
	mean_array = np.empty_like(results_array)
	stddev_array = np.empty_like(results_array)

	#create arrays to save uncertainties (standard error) into
	amp_err_array = np.empty_like(results_array)
	mean_err_array = np.empty_like(results_array)
	stddev_err_array = np.empty_like(results_array)

	#create arrays to loop through
	snr_range = np.arange(2.0,21.0,2.0)
	amp_range = np.arange(0.2,5.2,0.2)
	mean_range = np.arange(-2.0,0.1,0.1)
	stddev_range = np.arange(0.1,2.1,0.1)

	#create progressbar
	pbar = tqdm(total=105000)

	#use enumerate to create counts for indexing
	for snr_count, snr in enumerate(snr_range):
		for amp_count, amp in enumerate(amp_range):
			for mean_count, mean in enumerate(mean_range):
				for stddev_count, stddev in enumerate(stddev_range):
					#create data
					mock_x, mock_y = mock_data([gal_amp, amp], [gal_mean, mean], [gal_stddev, stddev],snr)
					#create models
					gauss1, pars1 = gaussian1(mock_x, mock_y)
					gauss2, pars2 = gaussian2(mock_x, mock_y)
					#fit the models
					best_fit1 = fitter(gauss1, pars1, mock_x, mock_y, verbose=False)
					best_fit2 = fitter(gauss2, pars2, mock_x, mock_y, verbose=False)

					#update the progress bar
					pbar.update(1)

					#save results
					amp_array[snr_count, amp_count, mean_count, stddev_count] = best_fit2.params['Outflow_amplitude'].value
					mean_array[snr_count, amp_count, mean_count, stddev_count] = best_fit2.params['Outflow_center'].value
					stddev_array[snr_count, amp_count, mean_count, stddev_count] = best_fit2.params['Outflow_sigma'].value

					#save uncertainties
					amp_err_array[snr_count, amp_count, mean_count, stddev_count] = best_fit2.params['Outflow_amplitude'].stderr
					mean_err_array[snr_count, amp_count, mean_count, stddev_count] = best_fit2.params['Outflow_center'].stderr
					stddev_err_array[snr_count, amp_count, mean_count, stddev_count] = best_fit2.params['Outflow_sigma'].stderr

					#create array with 1 for one gaussian, 2 for 2 gaussians fit and 1.5 for not statistically significant
					if best_fit1.bic < best_fit2.bic-10:
						results_array[snr_count, amp_count, mean_count, stddev_count] = 1
					elif best_fit2.bic < best_fit1.bic-10:
						results_array[snr_count, amp_count, mean_count, stddev_count] = 2
					elif (best_fit1.bic-10 < best_fit2.bic < best_fit1.bic) or (best_fit2.bic-10 < best_fit1.bic < best_fit2.bic):
						results_array[snr_count, amp_count, mean_count, stddev_count] = 1.5

	#close progressbar
	pbar.close()

	#pickle results
	with open('mock_results_sn_amp_mean_std'+str(date.today()),'wb') as f:
		pickle.dump([results_array, amp_array, mean_array, stddev_array],f)
	f.close()

	with open('mock_errors_amp_mean_std'+str(date.today()),'wb') as f:
		pickle.dump([amp_err_array, mean_err_array, stddev_err_array],f)
	f.close()

	#return results
	return results_array, amp_array, mean_array, stddev_array, amp_err_array, mean_err_array, stddev_err_array



#========================================================================
#Plotting test results

def recreate_original_arrays():
	"""
	Recreates the original input arrays
	"""
	#create original value arrays (should have shape [10,25,21,20])
	sn_array = np.repeat(np.repeat(np.repeat(np.arange(2.0,21.0,2.0), 25), 21), 20).reshape((10,25,21,20))
	amp_orig = np.repeat(np.repeat(np.array([np.arange(0.2,5.2,0.2)]*10), 21), 20).reshape((10,25,21,20))
	mean_orig = np.repeat(np.array([[np.arange(-2.0,0.1,0.1)]*25]*10), 20).reshape((10,25,21,20))
	stddev_orig = np.array([[[np.arange(0.1,2.1,0.1)]*21]*25]*10)

	return sn_array, amp_orig, mean_orig, stddev_orig


def corner_plotting_tests(results_array, amp_array, mean_array, stddev_array):
	"""
	Creates corner plots for each of the 1 gauss fit, 2 gauss fit and statistically insignificant fits for both the input and fitted values.  The fitted outflow values are always from the 2 gaussian fit, however.
	"""
	#create masks for getting out 1 gauss, 2 gauss and insignificant fits
	mask_1gauss = (results_array == 1)
	mask_2gauss = (results_array == 2)
	mask_insig = (results_array == 1.5)

	#get original arrays
	sn_array, amp_orig, mean_orig, stddev_orig = recreate_original_arrays()

	#create names for figures
	fig_titles1 = ['Fitted', 'Input']
	fig_titles2 = ['a 1 Gaussian fit was returned', 'a 2 Gaussian fit was returned', 'the difference between a 1 or 2 Gaussian fit \nwas statistically insignificant']

	#loop through masks
	for count, mask in enumerate([mask_1gauss, mask_2gauss, mask_insig]):
		#apply mask to results
		amp_masked = amp_array[mask]
		mean_masked = mean_array[mask]
		stddev_masked = stddev_array[mask]
		sn_masked = sn_array[mask]

		#apply mask to original values
		amp_orig_masked = amp_orig[mask]
		mean_orig_masked = mean_orig[mask]
		stddev_orig_masked = stddev_orig[mask]

		#find the mean of the fitted values
		amp_mean = amp_masked.mean()
		mean_mean = mean_masked.mean()
		stddev_mean = stddev_masked.mean()

		#find the mean of the input values
		amp_orig_mean = amp_orig_masked.mean()
		mean_orig_mean = mean_orig_masked.mean()
		stddev_orig_mean = stddev_orig_masked.mean()
		sn_mean = sn_masked.mean()

		#create lists to loop over later
		orig_means = [sn_mean, amp_orig_mean, mean_orig_mean, stddev_orig_mean]
		fitted_means = [sn_mean, amp_mean, mean_mean, stddev_mean]

		print(orig_means)
		print(fitted_means)

		#flatten the arrays
		amp_flat = np.reshape(amp_masked, -1)
		mean_flat = np.reshape(mean_masked, -1)
		stddev_flat = np.reshape(stddev_masked, -1)
		sn_flat = np.reshape(sn_masked, -1)

		amp_orig_flat = np.reshape(amp_orig_masked, -1)
		mean_orig_flat = np.reshape(mean_orig_masked, -1)
		stddev_orig_flat = np.reshape(stddev_orig_masked, -1)

		#combine arrays
		results = np.vstack([sn_flat, amp_flat, mean_flat, stddev_flat]).T
		input_values = np.vstack([sn_flat, amp_orig_flat, mean_orig_flat, stddev_orig_flat]).T

		for plot_count, plotting_values in enumerate([results, input_values]):
			#create corner plot
			figure = corner.corner(plotting_values,
				labels=['S/N','Amplitude','Mean','Standard Deviation'],
				show_titles=True,
				title_kwargs={"fontsize":8})

			#get the axes
			axes = np.array(figure.axes).reshape((4, 4))

			# Loop over the diagonal
			for i in range(4):
				ax = axes[i,i]
				ax.axvline(fitted_means[i], color='r')
				ax.axvline(orig_means[i], color='g')

			#loop over the histograms
			for i in range(4):
				for j in range(i):
					ax = axes[i,j]
					ax.axvline(fitted_means[j],color='r')
					ax.axhline(fitted_means[i], color='r')
					ax.axvline(orig_means[j], color='g')
					ax.axhline(orig_means[i], color='g')
					ax.plot(fitted_means[j], fitted_means[i], 'sr')
					ax.plot(orig_means[j], orig_means[i], 'sg')

			figure.suptitle(fig_titles1[plot_count]+' Outflow Gaussian values for which \n'+fig_titles2[count])
			green_line = Line2D([], [], color='green', marker='s',
	                          markersize=5, label='Input Mean')
			red_line = Line2D([], [], color='red', marker='s',
	                          markersize=5, label='Fitted Mean')
			figure.legend(handles=[green_line, red_line], bbox_to_anchor=(0.95,0.9))

			plt.savefig('breaking_koffee/'+fig_titles1[plot_count]+' Outflow Gaussian values for which \n'+fig_titles2[count])

def plot_the_difference(results_array, amp_array, mean_array, stddev_array):
	"""
	Plots the difference between the original input and the fitted values for the outflow gaussian against the S/N

	Input:
		results_array: the results array holding whether it was a 1, 2 gaussian or insignificant fit
	"""
	#get the original values arrays
	sn_array, amp_orig, mean_orig, stddev_orig = recreate_original_arrays()

	#find the difference between the input and fitted arrays
	amp_diff = amp_orig - amp_array
	mean_diff = mean_orig - mean_array
	stddev_diff = stddev_orig - stddev_array

	#mask to get 1 gaussian, 2 gaussian and insignificant fit values
	mask_1gauss = (results_array == 1)
	mask_2gauss = (results_array == 2)
	mask_insig = (results_array == 1.5)

	#plot against S/N
	fig, ax = plt.subplots(3,3, sharex=True, sharey='row')
	for count, mask in enumerate([mask_1gauss, mask_2gauss, mask_insig]):
		ax[0, count].scatter(sn_array[mask], amp_diff[mask], s=2)
		ax[1, count].scatter(sn_array[mask], mean_diff[mask], s=2)
		ax[2, count].scatter(sn_array[mask], stddev_diff[mask], s=2)

	ax[0,0].set_title('1 Gaussian Fit')
	ax[0,1].set_title('2 Gaussian Fit')
	ax[0,2].set_title('Statistically Insignificant')

	ax[0,0].set_ylabel('Input - Fitted Amp')
	ax[1,0].set_ylabel('Input - Fitted Mean')
	ax[2,0].set_ylabel('Input - Fitted Stddev')

	ax[2,0].set_xlabel('Input Signal/Noise')
	ax[2,1].set_xlabel('Input Signal/Noise')
	ax[2,2].set_xlabel('Input Signal/Noise')

	plt.show()

def plot_the_difference_scatter(results_array, amp_array, mean_array, stddev_array):
	"""
	Plots the difference between the original input and the fitted values for the outflow gaussian against the amplitude, coloured by S/N.

	Input:
		results_array: the results array holding whether it was a 1, 2 gaussian or insignificant fit
	"""
	#get the original values arrays
	sn_array, amp_orig, mean_orig, stddev_orig = recreate_original_arrays()

	#find the difference between the input and fitted arrays
	amp_diff = amp_orig - amp_array
	mean_diff = mean_orig - mean_array
	stddev_diff = stddev_orig - stddev_array

	#mask to get 1 gaussian, 2 gaussian and insignificant fit values
	mask_1gauss = (results_array == 1)
	mask_2gauss = (results_array == 2)
	mask_insig = (results_array == 1.5)

	#define the x and y variables
	var_x = stddev_diff
	var_y1 = amp_diff
	var_y2 = mean_diff

	var_x_label = 'Input - Fitted Stddev'
	var_y1_label = 'Input - Fitted Amp'
	var_y2_label = 'Input - Fitted Mean'

	#plot against amp, coloured by S/N
	fig, ax = plt.subplots(2,3, sharex=True, sharey='row', figsize=(12,6))
	for count, mask in enumerate([mask_1gauss, mask_2gauss, mask_insig]):
		f = ax[0, count].scatter(var_x[mask], var_y1[mask], s=2, c=sn_array[mask], cmap='viridis')
		f = ax[1, count].scatter(var_x[mask], var_y2[mask], s=2, c=sn_array[mask], cmap='viridis')

	ax[0,0].set_title('1 Gaussian Fit')
	ax[0,1].set_title('2 Gaussian Fit')
	ax[0,2].set_title('Statistically Insignificant')

	ax[0,0].set_ylabel(var_y1_label)
	ax[1,0].set_ylabel(var_y2_label)

	ax[1,0].set_xlabel(var_x_label)
	ax[1,1].set_xlabel(var_x_label)
	ax[1,2].set_xlabel(var_x_label)

	plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
	cax = plt.axes([0.85, 0.1, 0.03, 0.8])
	cbar = fig.colorbar(f, cax=cax, orientation='vertical', fraction=0.1)
	cbar.ax.set_ylabel('S/N', rotation='horizontal')

	plt.show()


def plot_sn_cuts(results_array, amp_array, mean_array, stddev_array, amp_err, mean_err, stddev_err):
	"""
	Cuts the S/N into bins
	"""
	import matplotlib.colors as mcolors

	#get the original values arrays
	sn_array, amp_orig, mean_orig, stddev_orig = recreate_original_arrays()

	#make masks for the number of models
	mask_1gauss = (results_array == 1)
	mask_2gauss = (results_array == 2)
	mask_insig = (results_array == 1.5)

	titles = ['1 Gaussian Fit', 'Insignificant Fits', '2 Gaussian Fits']

	for num, mod_mask in enumerate([mask_1gauss, mask_insig, mask_2gauss]):

		#make masks for the S/N
		sn_5 = (sn_array[mod_mask] < 5)
		sn_10 = (sn_array[mod_mask] >= 5) & (sn_array[mod_mask] < 10)
		sn_20 = (sn_array[mod_mask] >= 10) & (sn_array[mod_mask] <= 20)

		#create figure
		fig, ax = plt.subplots(3,3, figsize=(12,6))

		#iterate through the mask sections
		for count, mask in enumerate([sn_5, sn_10, sn_20]):
			#calculate percentage accuracy
			amp_perc = abs((amp_orig[mod_mask][mask]-amp_array[mod_mask][mask]))/amp_orig[mod_mask][mask] *100
			mean_perc = abs((mean_orig[mod_mask][mask]-mean_array[mod_mask][mask]))/mean_orig[mod_mask][mask] *100
			stddev_perc = abs((stddev_orig[mod_mask][mask]-stddev_array[mod_mask][mask]))/stddev_orig[mod_mask][mask] *100


			#ax[0,count].contour(X,Y,Z) #label='Avg % Error: {:.2f}\nAvg Error: {:.2f}'.format(np.nanmean(amp_perc), np.nanmean(amp_err[mod_mask][mask])))
			#ax[0,count].legend()
			ax[0,count].hist2d(amp_perc, amp_err[mod_mask][mask], bins=50, range=[[0,max(amp_perc)],[0,max(amp_err[mod_mask][mask])]], norm=mcolors.PowerNorm(0.3))

			#ax[1,count].scatter(mean_perc, mean_err[mod_mask][mask], s=2, label='Avg % Error: {:.2f}\nAvg Error: {:.2f}'.format(np.nanmean(mean_perc), np.nanmean(mean_err[mod_mask][mask])))
			#ax[1,count].legend()
			ax[1,count].hist2d(mean_perc, mean_err[mod_mask][mask], bins=50, range=[[0,max(mean_perc)],[0,max(mean_err[mod_mask][mask])]], norm=mcolors.PowerNorm(0.3))

			#ax[2,count].scatter(stddev_perc, stddev_err[mod_mask][mask], s=2, label='Avg % Error: {:.2f}\nAvg Error: {:.2f}'.format(np.nanmean(stddev_perc), np.nanmean(stddev_err[mod_mask][mask])))
			#ax[2,count].legend()
			ax[2,count].hist2d(stddev_perc, stddev_err[mod_mask][mask], bins=50, range=[[0, max(stddev_perc)], [0,max(stddev_err[mod_mask][mask])]], norm=mcolors.PowerNorm(0.3))

			ax[0,count].set_xlabel('Amplitude [[In-Out]/In]*100')
			ax[1,count].set_xlabel('Mean [[In-Out]/In]*100')
			ax[2,count].set_xlabel('Stddev [[In-Out]/In]*100')

		ax[0,0].set_ylabel('Amplitude LmFit Uncertainty')
		ax[1,0].set_ylabel('Mean LmFit Uncertainty')
		ax[2,0].set_ylabel('Stddev LmFit Uncertainty')

		ax[0,0].set_title('S/N < 5')
		ax[0,1].set_title('5 < S/N < 10')
		ax[0,2].set_title('10 < S/N < 20')

		plt.suptitle(titles[num])

	plt.show()



def plot_input_shapes(results_array):
	"""
	Plotting the input shapes of the mock emission lines coloured by whether they were 1 Gaussian, 2 Gaussian or insignificant fits; and split into panels of S/N < 5, 5-10, 10-15 and 15-20
	"""
	#get the original values arrays
	sn_array, amp_orig, mean_orig, stddev_orig = recreate_original_arrays()

	#make masks for the S/N
	sn_5 = (sn_array <= 5)
	sn_10 = (sn_array > 5) & (sn_array <= 10)
	sn_15 = (sn_array > 10) & (sn_array <= 15)
	sn_20 = (sn_array > 15) & (sn_array <= 20)

	titles = ['S/N <= 5', '5 < S/N <= 10', '10 < S/N <= 15', '15 < S/N <= 20']
	colours = ['r', 'b', 'gray']

	#create figure
	fig, ax = plt.subplots(1,4, figsize=(12,6), sharey=True)

	for count, sn_mask in enumerate([sn_5, sn_10, sn_15, sn_20]):
		#mask the original arrays and results array
		amp_orig_masked = amp_orig[sn_mask]
		mean_orig_masked = mean_orig[sn_mask]
		stddev_orig_masked = stddev_orig[sn_mask]
		sn_array_masked = sn_array[sn_mask]
		res_array_masked = results_array[sn_mask]

		#create results masks
		mask_1gauss = (res_array_masked == 1)
		mask_2gauss = (res_array_masked == 2)
		mask_insig = (res_array_masked == 1.5)

		for num, res_mask in enumerate([mask_2gauss, mask_1gauss, mask_insig]):

			for x in np.arange(amp_orig_masked[res_mask].shape[0]):
				#create the mock data
				mock_x, mock_y = mock_data([5.0, amp_orig_masked[res_mask][x]], [0.0, mean_orig_masked[res_mask][x]], [0.5, stddev_orig_masked[res_mask][x]], sn_array_masked[res_mask][x])

				#plot the data onto the figure
				ax[count].step(mock_x, mock_y, c=colours[num], alpha=0.02, lw=0.5)

		ax[count].set_title(titles[count])
		ax[count].set_xlabel('Mock Wavelength')

	ax[0].set_ylabel('Flux')

	#create legend
	red_line = Line2D([], [], color='r',label='One Gaussian Fits')
	blue_line = Line2D([], [], color='b',label='Two Gaussian Fits')
	grey_line = Line2D([], [], color='gray',label='Stat. insignificant Fits')
	ax[3].legend(handles=[red_line, blue_line, grey_line])

	plt.subplots_adjust(left=0.08, right=0.98, top=0.94, wspace=0.15)

	plt.show()



def plot_input_flux(results_array, amp_array, mean_array, stddev_array):
	"""
	Plotting the integrated flux of the input mock emission lines against the absolute value of the input-output values coloured by whether they were 1 Gaussian, 2 Gaussian or insignificant fits; and split into panels of S/N < 5, 5-10, 10-15 and 15-20
	"""
	#get the original values arrays
	sn_array, amp_orig, mean_orig, stddev_orig = recreate_original_arrays()

	#make masks for the S/N
	sn_5 = (sn_array <= 5)
	sn_10 = (sn_array > 5) & (sn_array <= 10)
	sn_15 = (sn_array > 10) & (sn_array <= 15)
	sn_20 = (sn_array > 15) & (sn_array <= 20)

	titles = ['S/N <= 5', '5 < S/N <= 10', '10 < S/N <= 15', '15 < S/N <= 20']
	colours = ['r', 'b', 'gray']

	#create figure
	fig, ax = plt.subplots(3,4, figsize=(12,6), sharey='row')

	for count, sn_mask in enumerate([sn_5, sn_10, sn_15, sn_20]):
		#mask the results array
		res_array_masked = results_array[sn_mask]

		#create results masks
		mask_1gauss = (res_array_masked == 1)
		mask_2gauss = (res_array_masked == 2)
		mask_insig = (res_array_masked == 1.5)

		for num, res_mask in enumerate([mask_2gauss, mask_1gauss, mask_insig]):
			#create arrays to fill
			int_array = np.empty_like(amp_array[sn_mask][res_mask])

			for x in np.arange(amp_array[sn_mask][res_mask].shape[0]):
				#create the mock data
				mock_x, mock_y = mock_data([5.0, amp_orig[sn_mask][res_mask][x]], [0.0, mean_orig[sn_mask][res_mask][x]], [0.5, stddev_orig[sn_mask][res_mask][x]], sn_array[sn_mask][res_mask][x])

				#integrate it
				int_array[x] = np.trapz(y=mock_y, x=mock_x)

			#plot the data onto the figure
			ax[0,count].scatter(int_array, abs(amp_orig[sn_mask][res_mask]-amp_array[sn_mask][res_mask]), c=colours[num], s=2, alpha=0.05)
			ax[1,count].scatter(int_array, abs(mean_orig[sn_mask][res_mask]-mean_array[sn_mask][res_mask]), c=colours[num], s=2, alpha=0.05)
			ax[2,count].scatter(int_array, abs(stddev_orig[sn_mask][res_mask]-stddev_array[sn_mask][res_mask]), c=colours[num], s=2, alpha=0.05)


		ax[0,count].set_title(titles[count])
		ax[2,count].set_xlabel('Integrated Flux')

	ax[0,0].set_ylabel('Abs(Input-Output)\n Amplitude')
	ax[1,0].set_ylabel('Abs(Input-Output)\n Mean')
	ax[2,0].set_ylabel('Abs(Input-Output)\n Stddev')


	#create legend
	red_line = Line2D([], [], color='r',label='One Gaussian Fits')
	blue_line = Line2D([], [], color='b',label='Two Gaussian Fits')
	grey_line = Line2D([], [], color='gray',label='Stat. Insig. Fits')
	ax[2,3].legend(handles=[red_line, blue_line, grey_line], fontsize=6)

	plt.subplots_adjust(left=0.08, right=0.98, top=0.94, bottom=0.1, wspace=0.15)

	plt.show()




def plot_change_in_abs_error_with_sn(fit_type, results_array, amp_array, mean_array, stddev_array):
	"""
	Plotting the standard deviation and median of the absolute value of the input-output values in bins of S/N < 5, 5-7.5, 7.5-10, 10-12.5, 12.5-15, 15-17.5 and 17.5-20

	Args:
		fit_type: 1, 1.5 or 2 (1 Gaussian, statistically insignificant or 2 Gaussian fit)
	"""
	#get the original values arrays
	sn_array, amp_orig, mean_orig, stddev_orig = recreate_original_arrays()

	#make mask for fit_type
	mask_fit = (results_array == fit_type)

	#create absolute error arrays
	amp_abs = abs(amp_orig[mask_fit]-amp_array[mask_fit])
	mean_abs = abs(mean_orig[mask_fit]-mean_array[mask_fit])
	stddev_abs = abs(stddev_orig[mask_fit]-stddev_array[mask_fit])

	sn_array = sn_array[mask_fit]

	#make masks for the S/N
	sn_5 = (sn_array <= 5)
	#sn_75 = (sn_array > 5) & (sn_array <= 7.5)
	sn_10 = (sn_array > 5) & (sn_array <= 10)
	#sn_125 = (sn_array > 10) & (sn_array <= 12.5)
	sn_15 = (sn_array > 10) & (sn_array <= 15)
	#sn_175 = (sn_array > 15) & (sn_array <= 17.5)
	sn_20 = (sn_array > 15) & (sn_array <= 20)

	titles = ['Parameter: Amplitude', 'Parameter: Mean', 'Parameter: Standard Deviation']

	#create figure
	fig, ax = plt.subplots(1,3, figsize=(12,4))#, sharey='row')

	for count, sn_mask in enumerate([sn_5, sn_10, sn_15, sn_20]):#sn_75, sn_125, sn_175]):
		#plot the data onto the figure
		ax[0].errorbar(np.median(sn_array[sn_mask]), np.median(amp_abs[sn_mask]), xerr=np.std(sn_array[sn_mask]), yerr=np.std(amp_abs[sn_mask]), c='b', fmt='o')
		ax[1].errorbar(np.median(sn_array[sn_mask]), np.median(mean_abs[sn_mask]), xerr=np.std(sn_array[sn_mask]), yerr=np.std(mean_abs[sn_mask]), c='b', fmt='o')
		ax[2].errorbar(np.median(sn_array[sn_mask]), np.median(stddev_abs[sn_mask]), xerr=np.std(sn_array[sn_mask]), yerr=np.std(stddev_abs[sn_mask]), c='b', fmt='o')

	for x in np.arange(3):
		ax[x].set_title(titles[x])
		ax[x].set_xlabel('Signal/Noise')

	ax[0].set_ylabel('Median of Abs(Input-Output)')

	plt.subplots_adjust(left=0.06, right=0.98, top=0.92, bottom=0.12, wspace=0.18)

	plt.show()
