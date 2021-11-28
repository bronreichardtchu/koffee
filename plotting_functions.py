"""
NAME:
	plotting_functions.py

AUTHOR:
	Bronwyn Reichardt Chu
	Swinburne
	2021

EMAIL:
	<breichardtchu@swin.edu.au>

PURPOSE:
	To hold useful plotting functions
	Written on MacOS Mojave 10.14.5, with Python 3.7

FUNCTIONS INLCUDED:
    get_rc_params
    chen_et_al_2010
    murray_et_al_2011
    davies_et_al_2019
    kim_et_al_2020
    fitting_function
    running_mean
    lower_quantile
    upper_quantile
    binned_median_quantile_log
    binned_median_quantile_lin
    pearson_correlation
    read_in_create_wcs
    plot_continuum_contours

MODIFICATION HISTORY:
		v.1.0 - first created January 2021

"""
import numpy as np
import scipy.stats as stats

from astropy.io import fits
from astropy.wcs import WCS


#===============================================================================
# DEFINE PLOTTING PARAMETERS
#===============================================================================


def get_rc_params():
    """
    Define the rcParams that will be used in all the plots.

    Returns
    -------
    rc_params dictionary object
    """

    rc_params = {
        "text.usetex": False,
        "axes.facecolor": 'white',

        #"figure.dpi": 125,
        #"legend.fontsize": 12,
        "legend.frameon": False,
        #"legend.markerscale": 1.0,

        "axes.labelsize": 'large',

        "xtick.direction": 'in',
        "xtick.labelsize": 'medium',
        "xtick.minor.visible": True,
        "xtick.top": True,
        "xtick.major.width": 1,

        "ytick.direction": 'in',
        "ytick.labelsize": 'medium',
        "ytick.minor.visible": True,
        "ytick.right": True,
        "ytick.major.width": 1,
    }

    return rc_params


#===============================================================================
# RELATIONS FROM OTHER PAPERS
#===============================================================================


def chen_et_al_2010(sfr_surface_density_min, sfr_surface_density_max, scale_factor=1):
    """
    The trendline from Chen et al. (2010) where v_out is proportional to (SFR surface density)^0.1
    (Energy driven winds - SNe feedback)

    Parameters
    ----------
    sfr_surface_density_min : float
        The minimum value of the SFR surface density

    sfr_surface_density_max : float
        The maximum value of the SFR surface density

    scale_factor : float
        The number by which to scale the trend, can be used to bring the trend
        into the range of the data on the plot (Default = 1)

    Returns
    -------
    sfr_surface_density : :obj:'~numpy.ndarray'
        vector of SFR surface densities

    v_out : :obj:'~numpy.ndarray'
        vector of outflow velocities following the trend
    """
    #create a vector for sfr surface density
    sfr_surface_density = np.linspace(sfr_surface_density_min, sfr_surface_density_max+4, num=1000)

    #use the relationship to predict the v_out
    v_out = scale_factor*sfr_surface_density**0.1

    return sfr_surface_density, v_out


def murray_et_al_2011(sfr_surface_density_min, sfr_surface_density_max, scale_factor=1):
    """
    The trendline from Murray et al. (2011) where v_out is proportional to (SFR surface density)^2
    (Momentum driven winds - radiative feedback from young stars)

    Parameters
    ----------
    sfr_surface_density_min : float
        The minimum value of the SFR surface density

    sfr_surface_density_max : float
        The maximum value of the SFR surface density

    scale_factor : float
        The number by which to scale the trend, can be used to bring the trend
        into the range of the data on the plot (Default = 1)

    Returns
    -------
    sfr_surface_density : :obj:'~numpy.ndarray'
        vector of SFR surface densities

    v_out : :obj:'~numpy.ndarray'
        vector of outflow velocities following the trend
    """
    #create a vector for sfr surface density
    sfr_surface_density = np.linspace(sfr_surface_density_min, sfr_surface_density_max+4, num=1000)

    #use the relationship to predict the v_out
    v_out = scale_factor*sfr_surface_density**2

    return sfr_surface_density, v_out

def davies_et_al_2019(sfr_surface_density_min, sfr_surface_density_max):
    """
    The trendline from Davies et al. (2019) where the flow velocity dispersion
    is proportional to SFR surface density.

    Parameters
    ----------
    sfr_surface_density_min : float
        The minimum value of the SFR surface density

    sfr_surface_density_max : float
        The maximum value of the SFR surface density

    Returns
    -------
    sfr_surface_density : :obj:'~numpy.ndarray'
        vector of SFR surface densities

    vel_disp : :obj:'~numpy.ndarray'
        vector of outflow velocity dispersions following the trend
    """
    #create a vector for sfr surface density
    sfr_surface_density = np.linspace(sfr_surface_density_min, sfr_surface_density_max+4, num=1000)

    #use the relationship to predict the v_out
    vel_disp = 241*sfr_surface_density**0.3

    return sfr_surface_density, vel_disp

def kim_et_al_2020(sfr_surface_density_min, sfr_surface_density_max, scale_factor=1):
    """
    The trendline from Kim et al. (2020) where mass the loading factor is proportional
    to (SFR surface density)^-0.44

    Parameters
    ----------
    sfr_surface_density_min : float
        The minimum value of the SFR surface density

    sfr_surface_density_max : float
        The maximum value of the SFR surface density

    scale_factor : float
        The number by which to scale the trend, can be used to bring the trend
        into the range of the data on the plot (Default = 1)

    Returns
    -------
    sfr_surface_density : :obj:'~numpy.ndarray'
        vector of SFR surface densities

    mlf : :obj:'~numpy.ndarray'
        vector of mass loading factors following the trend
    """
    #create a vector for sfr surface density
    sfr_surface_density = np.linspace(sfr_surface_density_min, sfr_surface_density_max+4, num=1000)

    #use the relationship to predict the v_out
    mlf = scale_factor*sfr_surface_density**-0.44

    return sfr_surface_density, mlf


#===============================================================================
# USEFUL LITTLE FUNCTIONS
#===============================================================================

def fitting_function(x, a, b):
    """
    My fitting function to be fit to the v_out to sfr surface density data

    Parameters
    ----------
    x : (vector)
        the SFR surface density

    a, b : (int)
        constants to be fit

    Returns
    -------
    y : (vector)
        the outflow velocity
    """
    return a*(x**b)

def running_mean(x, N):
    """
    Calculates the running mean

    Parameters
    ----------
    x : :obj:'~numpy.ndarray'
        data
    N : integer
        bin size
    """

    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)

def lower_quantile(x):
    """
    Calculate the lower quantile of x (data :obj:'~numpy.nd:obj:'~numpy.ndarray'')
    """
    return np.nanquantile(x, 0.33)

def upper_quantile(x):
    """
    Calculate the upper quantile of x (data :obj:'~numpy.ndarray')
    """
    return np.nanquantile(x, 0.66)


def binned_median_quantile_log(x, y, num_bins, weights=None, min_bin=None, max_bin=None):
    """
    Calculate the median, upper and lower quantile for an array of data in
    logarithmically increasing bins

    Parameters
    ----------
    x : :obj:'~numpy.ndarray'
        x-axis logarithmic data

    y : :obj:'~numpy.ndarray'
        y-axis data

    num_bins : integer
        the number of bins to divide the data into

    weights : :obj:'~numpy.ndarray'
        array to multiply x by, usually the error (Default = None)

    min_bin : float
        starting value of the first bin (Default = None)

    max_bin : float
        ending value of the last bin (Default = None)

    Returns
    -------
    logspace : :obj:'~numpy.ndarray'
        the logarithmic array of bin edges in x

    bin_center : :obj:'~numpy.ndarray'
        values indicating the centres of the bins in x

    bin_avg : :obj:'~numpy.ndarray'
        values of the median of the bins in y

    lower_quantile : :obj:'~numpy.ndarray'
        values for the lower quantile of each bin in y

    upper_quantile : :obj:'~numpy.ndarray'
        values for the upper quantile of each bin in y

    bin_stdev : :obj:'~numpy.ndarray'
        values for the standard deviation of each bin in y
    """
    if min_bin == None:
        min_bin = np.nanmin(x)

    if max_bin == None:
        max_bin = np.nanmax(x)

    #create the logspace - these are the bin edges
    logspace = np.logspace(np.log10(min_bin), np.log10(max_bin), num=num_bins+1)

    #calculate the average
    bin_avg = np.zeros(len(logspace)-1)
    upper_quantile = np.zeros(len(logspace)-1)
    lower_quantile = np.zeros(len(logspace)-1)
    bin_stdev = np.zeros(len(logspace)-1)

    for i in range(0, len(logspace)-1):
        left_bound = logspace[i]
        right_bound = logspace[i+1]
        items_in_bin = y[(x>left_bound)&(x<=right_bound)]
        print('Number of items in bin '+str(i)+': '+str(items_in_bin.shape))
        #calculate the median of the bin
        if weights == None:
            bin_avg[i] = np.nanmedian(items_in_bin)
        else:
            weights_in_bin = weights[0][(x>left_bound)&(x<=right_bound)]
            weights_in_bin = 1.0 - weights_in_bin/items_in_bin
            bin_avg[i] = np.average(items_in_bin, weights=weights_in_bin)

        #calculate the quartiles of the bin
        if items_in_bin.shape[0] < 10:
            upper_quantile[i] = np.nanquantile(items_in_bin, 0.80)
            lower_quantile[i] = np.nanquantile(items_in_bin, 0.20)
        else:
            upper_quantile[i] = np.nanquantile(items_in_bin, 0.66)
            lower_quantile[i] = np.nanquantile(items_in_bin, 0.33)

        #calculate the standard deviation of the bin
        bin_stdev[i] = np.nanstd(items_in_bin)

    #calculate the bin center for plotting
    bin_center = np.zeros(len(logspace)-1)
    for i in range(0, len(logspace)-1):
        bin_center[i] = np.nanmean([logspace[i],logspace[i+1]])

    return logspace, bin_center, bin_avg, lower_quantile, upper_quantile, bin_stdev


def binned_median_quantile_lin(x, y, num_bins, weights=None, min_bin=None, max_bin=None):
    """
    Calculate the median, upper and lower quantile for an array of data in
    linearly increasing bins

    Parameters
    ----------
    x : :obj:'~numpy.ndarray'
        x-axis linear data

    y : :obj:'~numpy.ndarray'
        y-axis data

    num_bins : integer
        the number of bins to divide the data into

    weights : :obj:'~numpy.ndarray'
        array to multiply x by, usually the error (Default = None)

    min_bin : float
        starting value of the first bin (Default = None)

    max_bin : float
        ending value of the last bin (Default = None)

    Returns
    -------
    linspace : :obj:'~numpy.ndarray'
        the array of linear bin edges in x

    bin_center : :obj:'~numpy.ndarray'
        values indicating the centres of the bins in x

    bin_avg : :obj:'~numpy.ndarray'
        values of the median of the bins in y

    lower_quantile : :obj:'~numpy.ndarray'
        values for the lower quantile of each bin in y

    upper_quantile : :obj:'~numpy.ndarray'
        values for the upper quantile of each bin in y

    bin_stdev : :obj:'~numpy.ndarray'
        values for the standard deviation of each bin in y
    """
    if min_bin == None:
        min_bin = np.nanmin(x)
    if max_bin == None:
        max_bin = np.nanmax(x)

    #create the logspace - these are the bin edges
    linspace = np.linspace(min_bin, max_bin, num=num_bins+1)

    #calculate the average
    bin_avg = np.zeros(len(linspace)-1)
    upper_quantile = np.zeros(len(linspace)-1)
    lower_quantile = np.zeros(len(linspace)-1)
    bin_stdev = np.zeros(len(linspace)-1)

    for i in range(0, len(linspace)-1):
        left_bound = linspace[i]
        right_bound = linspace[i+1]
        items_in_bin = y[(x>left_bound)&(x<=right_bound)]
        print('Number of items in bin '+str(i)+': '+str(items_in_bin.shape))
        if weights == None:
            bin_avg[i] = np.nanmedian(items_in_bin)
        else:
            weights_in_bin = weights[0][(x>left_bound)&(x<=right_bound)]
            weights_in_bin = 1.0 - weights_in_bin/items_in_bin
            bin_avg[i] = np.average(items_in_bin, weights=weights_in_bin)

        if items_in_bin.shape[0] < 10:
            upper_quantile[i] = np.nanquantile(items_in_bin, 0.80)
            lower_quantile[i] = np.nanquantile(items_in_bin, 0.20)
        else:
            upper_quantile[i] = np.nanquantile(items_in_bin, 0.66)
            lower_quantile[i] = np.nanquantile(items_in_bin, 0.33)

        #calculate the standard deviation of the bin
        bin_stdev[i] = np.nanstd(items_in_bin)

    #calculate the bin center for plotting
    bin_center = np.zeros(len(linspace)-1)
    for i in range(0, len(linspace)-1):
        bin_center[i] = np.nanmean([linspace[i],linspace[i+1]])

    return linspace, bin_center, bin_avg, lower_quantile, upper_quantile, bin_stdev


def pearson_correlation(x, y):
    """
    Calculate the Pearson correlation coefficient and p-value

    Parameters
    ----------
    x : :obj:'~numpy.ndarray'
        Input array - x values

    y : :obj:'~numpy.ndarray'
        Input array - y values

    Returns
    -------
    r : float
        Pearson's correlation coefficient

    p_value : float
        Two-tailed p-value
    """
    r, p_value = stats.pearsonr(x, y)

    return r, p_value


def spearman_coefficient(x, y):
    """
    Calculate the Spearman correlation coefficient and p-value

    Parameters
    ----------
    x : :obj:'~numpy.ndarray'
        Input array - x values

    y : :obj:'~numpy.ndarray'
        Input array - y values

    Returns
    -------
    r : float
        Spearman correlation coefficient

    p_value : float
        Two-tailed p-value
    """
    #make sure the arrays are 1D, not 2D
    x = np.ravel(x)
    y = np.ravel(y)

    r, p_value = stats.spearmanr(x, y, nan_policy='omit')

    return r, p_value


def read_in_create_wcs(fits_file, index=0, shift=None):
    """
    Reads in the fits file and creates the wcs

    Parameters
    ----------
    fits_file : string
        the filepath for the fits file to read in

    index : int
        the index of the extension to be loaded (default is 0)

    shift : list or None
        how to alter the header if the wcs is going to be wrong.
        e.g. ['CRPIX2', 32.0] will change the header value of CRPIX2 to 32.0

    Returns
    -------
    fits_data : :obj:'~numpy.ndarray'
        the fits data as a numpy array

    fits_wcs : astropy WCS object
        the world coordinate system for the fits file
    """
    #read the data in from fits
    with fits.open(fits_file) as hdu:
        hdu.info()
        fits_data = hdu[index].data
        fits_header = hdu[index].header
    hdu.close()

    #shift the header
    if shift:
        fits_header[shift[0]] = shift[1]

    #create the WCS
    fits_wcs = WCS(fits_header)

    return fits_data, fits_header, fits_wcs


def plot_continuum_contours(lamdas, xx, yy, data, z, ax):
    """
    Plots the continuum contours, using the rest wavelengths between 4600 and 4800 to define the continuum.

    Parameters
    ----------
    lamdas : :obj:'~numpy.ndarray'
        wavelength vector (1D)

    xx : :obj:'~numpy.ndarray'
        x coordinate array (2D)

    yy : :obj:'~numpy.ndarray'
        y coordinate array (2D)

    data : :obj:'~numpy.ndarray'
        data array (3D)

    z : float
        redshift of the galaxy

    ax : matplotlib axis instance
        axis for matplotlib to draw on

    Returns
    -------
    cont_contours : matplotlib.contour.QuadContourSet instance

    """
    #create a mask for the continuum
    cont_mask = (lamdas>4600*(1+z))&(lamdas<4800*(1+z))

    #find the median of the continuum
    cont_median = np.median(data[cont_mask,:,:], axis=0)

    #create the contours
    cont_contours = ax.contour(xx, yy, cont_median, colors='black', linewidths=0.7, alpha=0.7, levels=(0.2,0.3,0.4,0.7,1.0,2.0,4.0))

    return cont_contours
