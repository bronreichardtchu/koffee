"""
NAME:
	calculate_star_formation_rate.py

AUTHOR:
	Bronwyn Reichardt Chu
	Swinburne
	2020

EMAIL:
	<breichardtchu@swin.edu.au>

PURPOSE:
	To apply ppxf to the data cube.
	Written on MacOS Mojave 10.14.5, with Python 3.7

MODIFICATION HISTORY:
		v.1.0 - first created January 2020

"""
import numpy as np
import pickle
import matplotlib as mpl
import matplotlib.pyplot as plt

from astropy.cosmology import WMAP9 as cosmo
from astropy.constants import c
from astropy import units

from math import pi
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d


def calc_hbeta_extinction(lamdas, z):
    """
    Calculates the H_beta extinction - corrects for the extinction caused by light travelling through the dust and gas of the original galaxy, using the Cardelli et al. 1989 curves and Av = E(B-V)*Rv.  The value for Av ~ 2.11 x C(Hbeta) where C(Hbeta) = 0.24 from Lopez-Sanchez et al. 2006 A&A 449.

    Inputs:
        lamdas: the wavelength vector
        z: redshift

    Returns:
        A_hbeta: the extinction correction factor at the Hbeta line
    """
    #convert lamdas from Angstroms into micrometers
    lamdas = lamdas/10000

    #define the equations from the paper
    y = lamdas - 1.82
    a_x = 1.0 + 0.17699*y - 0.50447*(y**2) - 0.02427*(y**3) + 0.72085*(y**4) + 0.01979*(y**5) - 0.77530*(y**6) + 0.32999*(y**7)
    b_x = 1.41338*y + 2.28305*(y**2) + 1.07233*(y**3) - 5.38434*(y**4) - 0.62251*(y**5) + 5.30260*(y**6) - 2.09002*(y**7)

    #define the constants
    Rv = 3.1
    Av = 2.11*0.24

    #find A(lambda)
    A_lam = (a_x + b_x/Rv)*Av

    #find A_hbeta
    #first redshift the hbeta wavelength and convert to micrometers
    hbeta = (4861.333*(1+z))/10000
    #then find in lamdas array
    index = (np.abs(lamdas - hbeta)).argmin()
    #use the index to find A_hbeta
    A_hbeta = A_lam[index]

    return A_hbeta


def calc_hbeta_luminosity(lamdas, spectrum, z, cont_subtract=False):
    """
    Calculate the luminosity of the H_beta line
    The spectrum is in 10^-16 erg/s/cm^2/Ang.  Need to change it to erg/s

    Luminosities should be around 10^40

    Inputs:
        lamdas: the wavelength vector
        spectrum: the spectrum or array of spectra.  If in array, needs to be in shape [npix, nspec]
        z: redshift of the galaxy
        cont_subtract: if True, assumes continuum has not already been subtracted.  Uses the median value of the wavelength range 4850-4855A.

    Returns:
        h_beta_flux: the flux of the h_beta line
    """
    #create bounds to integrate over
    #Hbeta is at 4861.33A, allowing 1.5A on either side
    left_limit = 4855.83*(1+z)
    right_limit = 4866.83*(1+z)

    #use the wavelengths to find the values in the spectrum to integrate over
    h_beta_spec = spectrum[(lamdas>=left_limit)&(lamdas<=right_limit),]
    h_beta_lam = lamdas[(lamdas>=left_limit) & (lamdas<=right_limit)]

    #create a mask to cut out all spectra with a flux less than 1.0 at its peak
    #flux_mask = np.amax(h_beta_spec, axis=0) < 1.0

    #if the continuum has not already been fit and subtracted, use an approximation to subtract it off
    #also use the continuum to find the S/N and mask things
    #s_n = []
    if cont_subtract == True:
        cont = spectrum[(lamdas>=4850.0*(1+z))&(lamdas<=4855.0*(1+z)),]
        cont_median = np.nanmedian(cont, axis=0)
        h_beta_spec = h_beta_spec - cont_median
        #find the standard deviation of the continuum section
        noise = np.std(cont, axis=0)
        s_n = (cont_median/noise)
        #create the S/N mask
        s_n_mask = s_n > 20



    plt.figure()
    plt.step(h_beta_lam, h_beta_spec)
    plt.show()

    #integrate along the spectrum
    #by integrating, the units are now 10^-16 erg/s/cm^2
    h_beta_integral = np.trapz(h_beta_spec, h_beta_lam, axis=0)
    h_beta_integral = h_beta_integral*10**(-16)*units.erg/(units.s*(units.cm*units.cm))

    #now get rid of the cm^2
    #get the Hubble constant at z=0; this is in km/Mpc/s
    H_0 = cosmo.H(0)
    #use d = cz/H0 to find the distance in cm
    dist = (c*z/H_0).decompose().to('cm')
    print('distance:', dist)
    #multiply by 4*pi*d^2 to get rid of the cm
    h_beta_flux = (h_beta_integral*(4*pi*(dist**2))).to('erg/s')

    print(h_beta_flux)

    return h_beta_flux.value, s_n_mask, h_beta_spec


def calc_sfr(lamdas, spectrum, z, cont_subtract=False):
    """
    Calculates the star formation rate using Halpha
    SFR = C_Halpha (L_Halpha / L_Hbeta)_0 x 10^{-0.4A_Hbeta} x L_Hbeta[erg/s]

    Inputs:
        lamdas: array of wavelength
        spectrum: vector or array of spectra (shape: [npix, nspec])
        z: (float) redshift
        cont_subtract: if True, assumes continuum has not already been subtracted.  Uses the median value of the wavelength range 4850-4855A.

    Returns:
        sfr: (float, or array of floats) the star formation rate found using hbeta
    """
    #first we need to define C_Halpha, using Hao et al. 2011 ApJ 741:124
    #From table 2, uses a Kroupa IMF, solar metallicity and 100Myr
    c_halpha = 10**(-41.257)

    #from Calzetti 2001 PASP 113 we have L_Halpha/L_Hbeta = 2.87
    lum_ratio_alpha_to_beta = 2.87

    hbeta_extinction = calc_hbeta_extinction(lamdas, z)

    hbeta_luminosity, s_n_mask, h_beta_spec = calc_hbeta_luminosity(lamdas, spectrum, z, cont_subtract=cont_subtract)

    #calculate the star formation rate
    sfr = c_halpha * lum_ratio_alpha_to_beta * 10**(0.4*hbeta_extinction) * hbeta_luminosity
    #sfr = c_halpha * lum_ratio_alpha_to_beta * 10**(0.4*1.0) * (hbeta_luminosity)
    #sfr = c_halpha * lum_ratio_alpha_to_beta * 10**(0.4*0.29) * hbeta_luminosity

    total_sfr = np.sum(sfr)

    sfr_surface_density = sfr/((0.7*1.35)*(0.388**2))

    return sfr, total_sfr, sfr_surface_density, s_n_mask, h_beta_spec


def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)



def chen_et_al_2010(sfr_surface_density_min, sfr_surface_density_max, scale_factor=1):
    """
    The trendline from Chen et al. (2010) where v_out is proportional to (SFR surface density)^0.1
    """
    #create a vector for sfr surface density
    sfr_surface_density = np.linspace(sfr_surface_density_min, sfr_surface_density_max+4, num=1000)

    #use the relationship to predict the v_out
    v_out = scale_factor*sfr_surface_density**0.1

    return sfr_surface_density, v_out


def murray_et_al_2011(sfr_surface_density_min, sfr_surface_density_max, scale_factor=1):
    """
    The trendline from Murray et al. (2011) where v_out is proportional to (SFR surface density)^2
    """
    #create a vector for sfr surface density
    sfr_surface_density = np.linspace(sfr_surface_density_min, sfr_surface_density_max+4, num=1000)

    #use the relationship to predict the v_out
    v_out = scale_factor*sfr_surface_density**2

    return sfr_surface_density, v_out


def fitting_function(x, a, b):
    """
    My fitting function to be fit to the v_out to sfr surface density data

    Inputs:
        x: (vector) the star formation surface density
        a, b: (int) constants to be fit

    Returns:
        y: (vector) the v_out
    """
    return a*(x**b)




def plot_sfr_vout(sfr, outflow_results, outflow_error, stat_results, z, inflow_eq=False):
    """
    Plots the SFR against the outflow velocity.

    Inputs:
        sfr: (array) the star formation rate
        outflow_results: (array) array of outflow results from KOFFEE.  Should be (6, sfr.shape)
        stat_results: (array) array of statistical results from KOFFEE.  Should be same shape as sfr.
    """
    #create outflow mask
    flow_mask = (stat_results>0)

    #de-redshift the data first!!!
    systemic_mean = outflow_results[1,:][flow_mask]/(1+z)
    flow_mean = outflow_results[4,:][flow_mask]/(1+z)
    flow_sigma = outflow_results[3,:][flow_mask]/(1+z)

    #find the velocity difference
    #doing c*(lam_gal-lam_out)/lam_gal
    vel_diff = 299792.458*(systemic_mean-flow_mean)/systemic_mean

    v_out = 2*flow_sigma*299792.458/systemic_mean + vel_diff
    #mask out the points above 500km/s since they are all bad fits
    mask_500 = v_out<500
    v_out = v_out[mask_500]

    #add the errors of the flow and galaxy mean values
    #vel_err = (299792.458*((outflow_error[4,:][flow_mask]+outflow_error[1,:][flow_mask])/(outflow_results[4,:][flow_mask]-outflow_results[1,:][flow_mask]) + outflow_error[1,:][flow_mask]/outflow_results[1,:][flow_mask]))*vel_diff

    #vel_diff_out = vel_diff[vel_diff > -20]
    #vel_err_out = vel_err[vel_diff < 0]

    #vel_diff_in = vel_diff[vel_diff <= -20]
    #vel_err_in = vel_err[vel_diff >= 0]

    #vel_diff_mid = vel_diff[(vel_diff > -20)&(vel_diff<=20)]

    #v_out = 2*flow_sigma[vel_diff > -20]*299792.458/systemic_mean[vel_diff > -20] + vel_diff_out
    #v_in = 2*flow_sigma[vel_diff <= -20]*299792.458/systemic_mean[vel_diff <= -20] + vel_diff_in
    #v_mid = 2*flow_sigma[(vel_diff > -20)&(vel_diff<=20)]*299792.458/systemic_mean[(vel_diff > -20)&(vel_diff<=20)] + vel_diff_mid

    if inflow_eq == True:
        v_in = -2*flow_sigma[vel_diff <= -20]*299792.458/systemic_mean[vel_diff <= -20] + vel_diff_in

    #get the sfr for the outflow spaxels
    sfr_out = sfr[flow_mask]
    sfr_out = sfr_out[mask_500]
    #sfr_out = sfr[flow_mask][vel_diff > -20]
    #sfr_in = sfr[flow_mask][vel_diff <= -20]
    #sfr_mid = sfr[flow_mask][(vel_diff > -20)&(vel_diff<=20)]

    #get the sfr for the no outflow spaxels
    sfr_no_out = sfr[~flow_mask]
    vel_no_out = np.zeros_like(sfr_no_out)

    #take the log of sfr_out
    log_sfr_out = np.log10(sfr_out)

    #create the median points for the outflowing sfr
    #first_bin, last_bin = log_sfr_out.min(), log_sfr_out.max()
    first_bin, last_bin = -1., log_sfr_out.max()
    bin_width = (last_bin-first_bin)/8
    #loop through all the bins
    bin_edges = [first_bin, first_bin+bin_width]
    sfr_bin_medians = []
    v_out_bin_medians = []
    sfr_bin_stdev = []
    v_out_bin_stdev = []
    while bin_edges[1] <= last_bin+bin_width-bin_width/6:
        #create the bin
        sfr_bin = log_sfr_out[(log_sfr_out>=bin_edges[0])&(log_sfr_out<bin_edges[1])]
        v_out_bin = v_out[(log_sfr_out>=bin_edges[0])&(log_sfr_out<bin_edges[1])]

        #find the median in the bin
        sfr_median = np.nanmedian(sfr_bin)
        v_out_median = np.nanmedian(v_out_bin)

        #find the standard deviation in the bin
        sfr_stdev = np.nanstd(sfr_bin)
        v_out_stdev = np.nanstd(v_out_bin)

        #use the stdev to cut out any points greater than 2 sigma away from the median
        if np.any(v_out_bin >= v_out_median+2*v_out_stdev) or np.any(v_out_bin <= v_out_median-2*v_out_stdev):
            v_out_median = np.nanmedian(v_out_bin[(v_out_bin>v_out_median-2*v_out_stdev)&(v_out_bin<v_out_median+2*v_out_stdev)])
            v_out_stdev = np.nanstd(v_out_bin[(v_out_bin>v_out_median-2*v_out_stdev)&(v_out_bin<v_out_median+2*v_out_stdev)])

        sfr_bin_medians.append(sfr_median)
        v_out_bin_medians.append(v_out_median)
        sfr_bin_stdev.append(sfr_stdev)
        v_out_bin_stdev.append(v_out_stdev)

        #change bin_edges
        bin_edges = [bin_edges[0]+0.15, bin_edges[1]+0.15]

    sfr_bin_medians = np.array(sfr_bin_medians)
    v_out_bin_medians = np.array(v_out_bin_medians)
    sfr_bin_stdev = np.array(sfr_bin_stdev)
    v_out_bin_stdev = np.array(v_out_bin_stdev)
    print('sfr medians', sfr_bin_medians)
    print('v_out medians', v_out_bin_medians)

    #fit our own trend
    popt, pcov = curve_fit(fitting_function, 10**(sfr_bin_medians), v_out_bin_medians)
    print(popt)

    #create a vector for sfr_out
    sfr_surface_density = np.linspace(sfr_out.min(), sfr_out.max()+4.0, num=1000)

    #create vectors to plot the literature trends
    sfr_surface_density_chen, v_out_chen = chen_et_al_2010(sfr_out.min(), sfr_out.max(), scale_factor=popt[0])
    #sfr_surface_density_murray, v_out_murray = murray_et_al_2011(sfr_no_out.min(), sfr_out.max(), scale_factor=popt[0])
    sfr_surface_density_murray, v_out_murray = murray_et_al_2011(sfr_out.min(), sfr_out.max(), scale_factor=50)

    #plot it
    plt.figure()
    plt.rcParams['axes.facecolor']='white'
    #plt.errorbar(sfr_out, v_out, yerr=vel_err_out, marker='o', lw=0, label='Outflow', alpha=0.2)
    #plt.errorbar(sfr_in, abs(v_in), yerr=vel_err_in, marker='o', lw=0, label='Inflow', alpha=0.2)
    plt.plot(10**log_sfr_out, v_out, marker='o', lw=0, label='Flow spaxels', alpha=0.4)
    #plt.plot(sfr_in, v_in, marker='o', lw=0, label='Redshifted flow', alpha=0.4)
    #plt.plot(sfr_mid, v_mid, marker='o', lw=0, label='Centred flows', alpha=0.4)
    #plt.plot(sfr_no_out, vel_no_out, marker='o', lw=0, label='No flow spaxels', alpha=0.4)

    plt.plot(10**sfr_bin_medians, v_out_bin_medians, marker='', color='tab:blue', lw=3.0)
    #plt.errorbar(10**sfr_bin_medians, v_out_bin_medians, xerr=sfr_bin_stdev, yerr=v_out_bin_stdev, marker='', c='k', lw=2.0)
    #plt.fill_between(x=10**sfr_bin_medians, y1=v_out_bin_medians-v_out_bin_stdev, y2=v_out_bin_medians+v_out_bin_stdev, color='grey', alpha=0.2)

    plt.plot(sfr_surface_density, fitting_function(sfr_surface_density, *popt), 'r-', label='Fit: $v_{out}=%5.0f$ $\Sigma_{SFR}^{%5.2f}$' % tuple(popt))

    plt.plot(sfr_surface_density_chen, v_out_chen, ':k', label='Energy driven, $v_{out} \propto \Sigma_{SFR}^{0.1}$')
    plt.plot(sfr_surface_density_murray, v_out_murray, '--k', label='Momentum driven, $v_{out} \propto \Sigma_{SFR}^{2}$')
    plt.xscale('log')
    plt.ylim(100, 500)
    plt.legend(frameon=False, fontsize='x-small', loc='lower left')
    plt.ylabel('Maximum Flow Velocity [km s$^{-1}$]')
    #plt.xlabel('Log Star Formation Rate Surface Density (M$_\odot$ yr$^{-1}$ kpc$^{-2}$)')
    plt.xlabel('Log $\Sigma_{SFR}$ [M$_\odot$ yr$^{-1}$ kpc$^{-2}$]')
    plt.show()

    return flow_mask, vel_diff, v_out, v_out_bin_stdev

def plot_sfr_surface_density_radius(sig_sfr, rad_flat, stat_results):
    """
    Plots the SFR surface density against galaxy radius

    Inputs:
        sig_sfr: the SFR surface density
        rad: the flattend array of radius
        stat_results: the statistical results from KOFFEE

    """
    #create the flow mask
    flow_mask = (stat_results > 0)

    #create the no-flow mask
    no_flow_mask = (stat_results == 0)

    #apply to the rad array
    rad_flow = rad_flat[flow_mask]
    rad_no_flow = rad_flat[no_flow_mask]

    #apply to the SFR surface density array
    sig_sfr_flow = sig_sfr[flow_mask]
    sig_sfr_no_flow = sig_sfr[no_flow_mask]

    #plot
    plt.figure()
    plt.scatter(rad_flow, sig_sfr_flow, label='Outflow spaxels')
    plt.scatter(rad_no_flow, sig_sfr_no_flow, label='No outflow spaxels')
    plt.yscale('log')
    plt.xlabel('Radius [Arcseconds]')
    plt.ylabel('Log $\Sigma_{SFR}$ [M$_\odot$ yr$^{-1}$]')
    plt.legend()
    plt.show()

def plot_vel_out_radius(rad, outflow_results, stat_results, z):
    """
    Plots the SFR surface density against galaxy radius

    Inputs:
        sig_sfr: the SFR surface density
        rad: the array of radius
        stat_results: the statistical results from KOFFEE

    """
    #create the flow mask
    flow_mask = (stat_results > 0)

    #de-redshift the data first!!!
    systemic_mean = outflow_results[1,:]/(1+z)
    flow_mean = outflow_results[4,:]/(1+z)
    flow_sigma = outflow_results[3,:]/(1+z)

    #find the velocity difference
    #doing c*(lam_gal-lam_out)/lam_gal
    vel_diff = 299792.458*(systemic_mean-flow_mean)/systemic_mean

    v_out = 2*flow_sigma*299792.458/systemic_mean + vel_diff

    vel_out = v_out[flow_mask]
    #mask out the spaxels above 500km/s (they're bad fits)
    mask_500 = vel_out<500
    vel_out = vel_out[mask_500]

    #apply to the rad array
    rad_flow = rad[flow_mask]
    rad_flow = rad_flow[mask_500]

    #create the median points for the outflowing vel
    first_bin, last_bin = rad_flow.min(), 6.4
    bin_width = (last_bin-first_bin)/8
    #loop through all the bins
    bin_edges = [first_bin, first_bin+bin_width]
    rad_bin_medians = []
    v_out_bin_medians = []
    rad_bin_stdev = []
    v_out_bin_stdev = []
    while bin_edges[1] <= last_bin+bin_width-bin_width/6:
        #create the bin
        rad_bin = rad_flow[(rad_flow>=bin_edges[0])&(rad_flow<bin_edges[1])]
        v_out_bin = vel_out[(rad_flow>=bin_edges[0])&(rad_flow<bin_edges[1])]

        #find the median in the bin
        rad_median = np.nanmedian(rad_bin)
        v_out_median = np.nanmedian(v_out_bin)

        #find the standard deviation in the bin
        rad_stdev = np.nanstd(rad_bin)
        v_out_stdev = np.nanstd(v_out_bin)

        #use the stdev to cut out any points greater than 2 sigma away from the median
        if np.any(v_out_bin >= v_out_median+2*v_out_stdev) or np.any(v_out_bin <= v_out_median-2*v_out_stdev):
            v_out_median = np.nanmedian(v_out_bin[(v_out_bin>v_out_median-2*v_out_stdev)&(v_out_bin<v_out_median+2*v_out_stdev)])
            v_out_stdev = np.nanstd(v_out_bin[(v_out_bin>v_out_median-2*v_out_stdev)&(v_out_bin<v_out_median+2*v_out_stdev)])

        rad_bin_medians.append(rad_median)
        v_out_bin_medians.append(v_out_median)
        rad_bin_stdev.append(rad_stdev)
        v_out_bin_stdev.append(v_out_stdev)

        #change bin_edges
        bin_edges = [bin_edges[0]+0.5, bin_edges[1]+0.5]

    rad_bin_medians = np.array(rad_bin_medians)
    v_out_bin_medians = np.array(v_out_bin_medians)
    rad_bin_stdev = np.array(rad_bin_stdev)
    v_out_bin_stdev = np.array(v_out_bin_stdev)
    print('radius medians', rad_bin_medians)
    print('v_out medians', v_out_bin_medians)

    #plot
    plt.figure()
    plt.scatter(rad_flow, vel_out, alpha=0.4)
    plt.plot(rad_bin_medians, v_out_bin_medians, marker='', color='tab:blue', lw=3.0)
    plt.axvline(2.5, ls='--', c='k', lw=3)
    plt.axvline(6.4, ls='--', c='k', lw=3)
    plt.ylim(100,500)
    plt.xlim(-0.4, 7.8)
    plt.xlabel('Radius [Arcseconds]')
    plt.ylabel('Maximum Outflow Velocity [km s$^{-1}$]')
    plt.show()

    return v_out

def plot_outflow_frequency_radius(rad_flat, stat_results):
    """
    Plots the frequency of outflow spaxels, and non-outflow spaxels against radius, using the flattened arrays.
    """
    #create flow mask
    flow_mask = (stat_results > 0)

    #create no_flow mask
    no_flow_mask = (stat_results == 0)

    #create low S/N mask
    low_sn_mask = np.isnan(stat_results)

    #get the radius array for each type of spaxel
    rad_flow = rad_flat[flow_mask]
    rad_no_flow = rad_flat[no_flow_mask]
    rad_low_sn = rad_flat[low_sn_mask]

    #get total number of spaxels
    total_spaxels = stat_results.shape[0]
    print(total_spaxels)
    total_outflows = rad_flow.shape[0]
    total_no_flows = rad_no_flow.shape[0]

    #iterate through the radii and get the number of spaxels in that range
    num_flow = []
    num_no_flow = []
    num_low_sn = []
    num_spax_rad = []
    num_spax_flow_rad = []

    cum_spax = []
    cumulative_spaxels = 0

    step_range = 0.5

    for i in np.arange(0,21, step_range):
        flow_bin = rad_flow[(rad_flow>=i)&(rad_flow<(i+step_range))].shape[0]
        no_flow_bin = rad_no_flow[(rad_no_flow>=i)&(rad_no_flow<(i+step_range))].shape[0]
        low_sn_bin = rad_low_sn[(rad_low_sn>=i)&(rad_low_sn<(i+step_range))].shape[0]

        rad_bin = flow_bin+no_flow_bin+low_sn_bin
        rad_flow_bin = flow_bin + no_flow_bin

        cumulative_spaxels = cumulative_spaxels + rad_flow_bin
        cum_spax.append(cumulative_spaxels)

        num_flow.append(flow_bin)
        num_no_flow.append(no_flow_bin)
        num_low_sn.append(low_sn_bin)
        num_spax_rad.append(rad_bin)
        num_spax_flow_rad.append(rad_flow_bin)

    num_flow = np.array(num_flow)
    num_no_flow = np.array(num_no_flow)
    num_low_sn = np.array(num_low_sn)
    num_spax_rad = np.array(num_spax_rad)
    num_spax_flow_rad = np.array(num_spax_flow_rad)
    cum_spax = np.array(cum_spax)
    cum_spax = cum_spax/cum_spax[-1]

    #create the plot
    """
    plt.figure()
    plt.fill_between(np.arange(0,21, step_range), 0.0, cum_spax, color='grey', alpha=0.4)
    plt.plot(np.arange(0,21, step_range), num_low_sn/num_spax_rad, label='Low S/N', alpha=0.5)
    plt.plot(np.arange(0,21, step_range), num_flow/num_spax_rad, label='Outflows', alpha=0.5)
    plt.plot(np.arange(0,21, step_range), num_no_flow/num_spax_rad, label='No Outflows', alpha=0.5)
    plt.xlabel('Radius [Arcseconds]')
    plt.ylabel('Fraction of spaxels')
    plt.xlim(0,20)
    plt.legend()
    plt.show()
    """

    plt.figure()
    plt.fill_between(np.arange(0,21, step_range), 0.0, cum_spax, color='grey', alpha=0.4)
    plt.plot(np.arange(0,21, step_range), num_flow/num_spax_flow_rad, label="Outflow Spaxels", lw=3)
    plt.text(0.2, 0.91, 'Outflow Spaxels', fontsize=14, color='tab:blue')
    plt.plot(np.arange(0,21, step_range), num_no_flow/num_spax_flow_rad, label="No Flow Spaxels", lw=3)
    plt.text(0.2, 0.05, 'No Flow Spaxels', fontsize=14, color='tab:orange')
    plt.axvline(6.4, ls='--', c='k', lw=3)
    plt.axvline(2.5, ls='--', c='k', lw=3)
    plt.xlabel('Radius [Arcseconds]')
    plt.ylabel('Spaxel PDF')
    plt.xlim(-0.4, 7.8)
    #plt.legend(loc='upper right')
    plt.show()


def plot_outflow_frequency_sfr_surface_density(sig_sfr, stat_results):
    """
    Plots the frequency of outflow spaxels, and non-outflow spaxels against radius, using the flattened arrays.
    """
    #create flow mask
    flow_mask = (stat_results > 0)

    #create no_flow mask
    no_flow_mask = (stat_results == 0)

    #create low S/N mask
    low_sn_mask = np.isnan(stat_results)

    #get the radius array for each type of spaxel
    sig_sfr_flow = sig_sfr[flow_mask]
    sig_sfr_no_flow = sig_sfr[no_flow_mask]
    sig_sfr_low_sn = sig_sfr[low_sn_mask]

    #get total number of spaxels
    total_spaxels = stat_results.shape[0]
    print(total_spaxels)
    total_outflows = sig_sfr_flow.shape[0]
    total_no_flows = sig_sfr_no_flow.shape[0]

    #iterate through the sfr surface densities and get the number of spaxels in that range
    num_flow = []
    num_no_flow = []
    num_low_sn = []
    num_spax_sig_sfr = []
    num_spax_flow_sig_sfr = []

    cum_spax = []
    cumulative_spaxels = 0

    step_range = 0.1

    for i in np.arange(sig_sfr.min(),sig_sfr.max(),step_range):
        flow_bin = sig_sfr_flow[(sig_sfr_flow>=i)&(sig_sfr_flow<(i+step_range))].shape[0]
        no_flow_bin = sig_sfr_no_flow[(sig_sfr_no_flow>=i)&(sig_sfr_no_flow<(i+step_range))].shape[0]
        low_sn_bin = sig_sfr_low_sn[(sig_sfr_low_sn>=i)&(sig_sfr_low_sn<(i+step_range))].shape[0]

        sig_sfr_bin = flow_bin+no_flow_bin+low_sn_bin
        sig_sfr_flow_bin = flow_bin + no_flow_bin

        cumulative_spaxels = cumulative_spaxels + sig_sfr_flow_bin
        cum_spax.append(cumulative_spaxels)

        num_flow.append(flow_bin)
        num_no_flow.append(no_flow_bin)
        num_low_sn.append(low_sn_bin)
        num_spax_sig_sfr.append(sig_sfr_bin)
        num_spax_flow_sig_sfr.append(sig_sfr_flow_bin)

    num_flow = np.array(num_flow)
    num_no_flow = np.array(num_no_flow)
    num_low_sn = np.array(num_low_sn)
    num_spax_sig_sfr = np.array(num_spax_sig_sfr)
    num_spax_flow_sig_sfr = np.array(num_spax_flow_sig_sfr)
    cum_spax = np.array(cum_spax)
    cum_spax = cum_spax/cum_spax[-1]

    #create the fractions
    flow_fraction = num_flow/num_spax_flow_sig_sfr
    no_flow_fraction = num_no_flow/num_spax_flow_sig_sfr

    #replace nan values with the mean of the two values in front and behind.
    for index, val in np.ndenumerate(flow_fraction):
        if np.isnan(val):
            flow_fraction[index] = np.nanmean([flow_fraction[index[0]-1],flow_fraction[index[0]+1]])

    for index, val in np.ndenumerate(no_flow_fraction):
        if np.isnan(val):
            no_flow_fraction[index] = np.nanmean([no_flow_fraction[index[0]-1],no_flow_fraction[index[0]+1]])

    #fit the frequencies using a spline fit
    sig_sfr_range = np.log10(np.arange(sig_sfr.min(),sig_sfr.max(),step_range))
    fit_flow = interp1d(sig_sfr_range, flow_fraction)
    fit_no_flow = interp1d(sig_sfr_range, no_flow_fraction)
    #new_sig_sfr_range = np.log10(np.arange(sig_sfr.min(),sig_sfr.max(),step_range/2))

    #create the plot
    fig, ax = plt.subplots()
    ax.fill_between(np.arange(sig_sfr.min(),sig_sfr.max(),step_range),0.0, cum_spax, color='grey', alpha=0.4)
    ax.plot(np.arange(sig_sfr.min(),sig_sfr.max(),step_range), flow_fraction, label="Outflow Spaxels", ls='-', lw=3, color='tab:blue')
    #ax.plot(sig_sfr_range, num_flow/num_spax_flow_sig_sfr, label="Outflow Spaxels", ls='-', lw=3)
    #ax.plot(sig_sfr_range, fit_flow(sig_sfr_range), label='Fit to Outflow Spaxels')
    ax.text(1.2, 0.91, 'Outflow Spaxels', fontsize=14, color='tab:blue')
    ax.plot(np.arange(sig_sfr.min(),sig_sfr.max(),step_range), no_flow_fraction, label="No Flow Spaxels", ls='-', lw=3, color='tab:orange')
    #ax.plot(sig_sfr_range, fit_no_flow(sig_sfr_range), label='Fit to No flow Spaxels')
    ax.text(1.1, 0.05, 'No Flow Spaxels', fontsize=14, color='tab:orange')
    ax.axvline(1.0, ls='--', c='k', lw=2, label='Genzel+2011, Newman+2012')
    ax.text(0.83, 0.2, 'Genzel+2011, Newman+2012', rotation='vertical')
    ax.axvline(0.1, ls=':', c='k', lw=2, label='Heckman 2000')
    ax.text(0.083, 0.65, 'Heckman 2000', rotation='vertical')
    ax.set_xscale('log')
    #ax.set_xticks([0,0.5,1,5,10])
    #ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    ax.set_xlabel('Log $\Sigma_{SFR}$ [M$_\odot$ yr$^{-1}$ kpc$^{-2}$]')
    ax.set_ylabel('Spaxel PDF')
    ax.set_xlim(0.02,11)
    #plt.legend(frameon=False, fontsize='x-small', loc=(0.12,0.87))
    plt.show()




def plot_sfr_surface_density_vout_loops(sfr, outflow_results, outflow_error, stat_results, z, h_beta_spec):
    """
    Try the same thing as above but with for loops
    """
    #create arrays to put the flow velocity results in
    vel_res = np.empty_like(sfr)
    vel_diffs = np.empty_like(sfr)
    vel_res_type = np.empty_like(sfr)

    for i in np.arange(sfr.shape[0]):
        #if the maximum flux in the hbeta area is greater than 1.0, we do our maths on it
        if h_beta_spec[:,i].max() > 1.0:
            #find whether stat_results decided if there was an outflow or not
            if stat_results[i] == 1:
                #de-redshift the data
                systemic_mean = outflow_results[1,i]/(1+z)
                flow_mean = outflow_results[4,i]/(1+z)
                flow_sigma = outflow_results[3,i]/(1+z)

                #calculate the velocity difference
                vel_diff = 299792.458*(systemic_mean-flow_mean)/systemic_mean

                #use the velocity difference to decide if it is an in or an outflow
                # if it is positive, it's an outflow
                if vel_diff > 0:
                    vel_res_type[i] = 1.0
                    #calculate the v_out
                    vel_res[i] = 2*flow_sigma*299792.458/systemic_mean + vel_diff
                    vel_diffs[i] = vel_diff

                elif vel_diff <= 0:
                    vel_res_type[i] = -1.0
                    #calculate the v_out
                    vel_res[i] = 2*flow_sigma*299792.458/systemic_mean + vel_diff
                    vel_diffs[i] = vel_diff


            #if stat_results is zero, KOFFEE didn't find an outflow
            elif stat_results[i] == 0:
                vel_res[i] = 0.0
                vel_res_type[i] = 0.0
                vel_diffs[i] = 0.0

        #otherwise, we cannot calculate the sfr reliably
        else:
            vel_res[i] = 0.0
            vel_res_type[i] = 0.0
            vel_diffs[i] = 0.0

    #now do the plotting
    plt.figure()
    plt.plot(sfr[vel_res_type==1], vel_res[vel_res_type==1], marker='o', lw=0, label='Outflow', alpha=0.2)
    plt.plot(sfr[vel_res_type==-1], vel_res[vel_res_type==-1], marker='o', lw=0, label='Inflow', alpha=0.2)
    plt.xscale('log')
    plt.legend(frameon=False, fontsize='small', loc='lower left')
    plt.ylabel('Flow velocity (km s$^{-1}$)')
    plt.xlabel('Log Star Formation Rate Surface Density (M$_\odot$ yr$^{-1}$ kpc$^{-2}$)')
    plt.show()

    return vel_res, vel_res_type, vel_diffs


def plot_sigma_vel_diff(outflow_results, outflow_error, stat_results, z):
    """
    Plots the velocity difference between systemic and flow components against the flow sigma
    """
    #create outflow mask
    flow_mask = (stat_results==1)

    #de-redshift the data first!!!
    systemic_mean = outflow_results[1,:][flow_mask]/(1+z)
    flow_mean = outflow_results[4,:][flow_mask]/(1+z)
    flow_sigma = outflow_results[3,:][flow_mask]/(1+z)

    #find the velocity difference
    #doing c*(lam_gal-lam_out)/lam_gal
    vel_diff = 299792.458*(systemic_mean-flow_mean)/systemic_mean

    #seperate them into out and in flows
    vel_diff_out = vel_diff[vel_diff > 20]
    vel_diff_mid = vel_diff[(vel_diff<=20)&(vel_diff>-20)]
    vel_diff_in = vel_diff[vel_diff <= -20]

    sigma_out = flow_sigma[vel_diff > 20]*299792.458/systemic_mean[vel_diff > 20]
    sigma_mid = flow_sigma[(vel_diff<=20)&(vel_diff>-20)]*299792.458/systemic_mean[(vel_diff<=20)&(vel_diff>-20)]
    sigma_in = flow_sigma[vel_diff <= -20]*299792.458/systemic_mean[vel_diff <= -20]

    #plot it
    plt.figure()
    #plt.errorbar(sfr_out, v_out, yerr=vel_err_out, marker='o', lw=0, label='Outflow', alpha=0.2)
    #plt.errorbar(sfr_in, abs(v_in), yerr=vel_err_in, marker='o', lw=0, label='Inflow', alpha=0.2)
    plt.plot(vel_diff_out, sigma_out, marker='o', lw=0, label='Blue shifted flow', alpha=0.4)
    plt.plot(vel_diff_in, sigma_in, marker='o', lw=0, label='Red shifted flow', alpha=0.4)
    plt.plot(vel_diff_mid, sigma_mid, marker='o', lw=0, label='Centred flow', alpha=0.4)
    #plt.xscale('log')
    plt.legend(frameon=False, fontsize='small', loc='lower left')
    plt.ylabel('Flow Dispersion ($\sigma_{out}$) [km s$^{-1}$]')
    plt.xlabel('Velocity Difference (v$_{sys}$-v$_{flow}$) [km s$^{-1}$]')
    plt.show()


def plot_sigma_sfr(sfr, outflow_results, outflow_error, stat_results, z):
    """
    Plots the velocity difference between systemic and flow components against the flow sigma
    """
    #create outflow mask
    flow_mask = (stat_results==1)

    #de-redshift the data first!!!
    systemic_mean = outflow_results[1,:][flow_mask]/(1+z)
    flow_mean = outflow_results[4,:][flow_mask]/(1+z)
    flow_sigma = outflow_results[3,:][flow_mask]/(1+z)

    #find the velocity difference
    #doing c*(lam_gal-lam_out)/lam_gal
    vel_diff = 299792.458*(systemic_mean-flow_mean)/systemic_mean

    #seperate them into out and in flows
    vel_diff_out = vel_diff[vel_diff > 20]
    vel_diff_mid = vel_diff[(vel_diff<=20)&(vel_diff>-20)]
    vel_diff_in = vel_diff[vel_diff <= -20]

    sigma_out = flow_sigma[vel_diff > 20]*299792.458/systemic_mean[vel_diff > 20]
    sigma_mid = flow_sigma[(vel_diff<=20)&(vel_diff>-20)]*299792.458/systemic_mean[(vel_diff<=20)&(vel_diff>-20)]
    sigma_in = flow_sigma[vel_diff <= -20]*299792.458/systemic_mean[vel_diff <= -20]

    sfr_out = sfr[flow_mask][vel_diff > 20]
    sfr_mid = sfr[flow_mask][(vel_diff<=20)&(vel_diff>-20)]
    sfr_in = sfr[flow_mask][vel_diff <= -20]

    #plot it
    plt.figure()
    #plt.errorbar(sfr_out, v_out, yerr=vel_err_out, marker='o', lw=0, label='Outflow', alpha=0.2)
    #plt.errorbar(sfr_in, abs(v_in), yerr=vel_err_in, marker='o', lw=0, label='Inflow', alpha=0.2)
    plt.plot(sfr_out, sigma_out, marker='o', lw=0, label='Blue shifted flow', alpha=0.4)
    plt.plot(sfr_in, sigma_in, marker='o', lw=0, label='Red shifted flow', alpha=0.4)
    plt.plot(sfr_mid, sigma_mid, marker='o', lw=0, label='Centred flow', alpha=0.4)
    plt.xscale('log')
    plt.legend(frameon=False, fontsize='small', loc='lower left')
    plt.ylabel('Flow Dispersion ($\sigma_{out}$) [km s$^{-1}$]')
    plt.xlabel('Log Star Formation Rate Surface Density (M$_\odot$ yr$^{-1}$ kpc$^{-2}$)')
    plt.show()


def plot_vel_diff_sfr(sfr, outflow_results, outflow_error, stat_results, z):
    """
    Plots the velocity difference between systemic and flow components against the flow sigma
    """
    #create outflow mask
    flow_mask = (stat_results==1)

    #de-redshift the data first!!!
    systemic_mean = outflow_results[1,:][flow_mask]/(1+z)
    flow_mean = outflow_results[4,:][flow_mask]/(1+z)

    #find the velocity difference
    #doing c*(lam_gal-lam_out)/lam_gal
    vel_diff = 299792.458*(systemic_mean-flow_mean)/systemic_mean

    #seperate them into out and in flows
    vel_diff_out = vel_diff[vel_diff > 20]
    vel_diff_mid = vel_diff[(vel_diff<=20)&(vel_diff>-20)]
    vel_diff_in = vel_diff[vel_diff <= -20]

    sfr_out = sfr[flow_mask][vel_diff > 20]
    sfr_mid = sfr[flow_mask][(vel_diff<=20)&(vel_diff>-20)]
    sfr_in = sfr[flow_mask][vel_diff <= -20]

    #plot it
    plt.figure()
    #plt.errorbar(sfr_out, v_out, yerr=vel_err_out, marker='o', lw=0, label='Outflow', alpha=0.2)
    #plt.errorbar(sfr_in, abs(v_in), yerr=vel_err_in, marker='o', lw=0, label='Inflow', alpha=0.2)
    plt.plot(sfr_out, vel_diff_out, marker='o', lw=0, label='Blue shifted flow', alpha=0.4)
    plt.plot(sfr_in, vel_diff_in, marker='o', lw=0, label='Red shifted flow', alpha=0.4)
    plt.plot(sfr_mid, vel_diff_mid, marker='o', lw=0, label='Centred flow', alpha=0.4)
    plt.xscale('log')
    plt.legend(frameon=False, fontsize='small', loc='lower left')
    plt.ylabel('Velocity Difference ($v_{sys}-v_{broad}$) [km s$^{-1}$]')
    plt.xlabel('Log Star Formation Rate Surface Density (M$_\odot$ yr$^{-1}$ kpc$^{-2}$)')
    plt.show()


def plot_vdiff_amp_ratio(outflow_results, outflow_error, stat_results, z):
    """
    Plots the velocity difference between systemic and flow components against the flow sigma
    """
    #create outflow mask
    flow_mask = (stat_results==1)

    #de-redshift the data first!!!
    systemic_amp = outflow_results[2,:][flow_mask]/(1+z)
    flow_amp = outflow_results[5,:][flow_mask]/(1+z)
    systemic_mean = outflow_results[1,:][flow_mask]/(1+z)
    flow_mean = outflow_results[4,:][flow_mask]/(1+z)

    #find the velocity difference
    #doing c*(lam_gal-lam_out)/lam_gal
    vel_diff = 299792.458*(systemic_mean-flow_mean)/systemic_mean

    #find the amplitude ratio
    amp_ratio = flow_amp/systemic_amp

    #seperate them into out and in flows
    amp_ratio_out = amp_ratio[vel_diff > 20]
    amp_ratio_mid = amp_ratio[(vel_diff<=20)&(vel_diff>-20)]
    amp_ratio_in = amp_ratio[vel_diff <= -20]

    vel_diff_out = vel_diff[vel_diff > 20]
    vel_diff_mid = vel_diff[(vel_diff<=20)&(vel_diff>-20)]
    vel_diff_in = vel_diff[vel_diff <= -20]



    #plot it
    plt.figure()
    #plt.errorbar(sfr_out, v_out, yerr=vel_err_out, marker='o', lw=0, label='Outflow', alpha=0.2)
    #plt.errorbar(sfr_in, abs(v_in), yerr=vel_err_in, marker='o', lw=0, label='Inflow', alpha=0.2)
    plt.plot(vel_diff_out, amp_ratio_out, marker='o', lw=0, label='Blue shifted flow', alpha=0.4)
    plt.plot(vel_diff_in, amp_ratio_in, marker='o', lw=0, label='Red shifted flow', alpha=0.4)
    plt.plot(vel_diff_mid, amp_ratio_mid, marker='o', lw=0, label='Centred flow', alpha=0.4)
    #plt.xscale('log')
    plt.legend(frameon=False, fontsize='small', loc='lower left')
    plt.ylabel('Amplitude Ratio (broad/systemic)')
    plt.xlabel('Velocity Difference (km/s)')
    plt.show()
